//! Matrix-core (MFMA) GEMM for CDNA4 / gfx950 — the inference compute brick.
//!
//! Built bottom-up on the raw KFD/AQL path (no ROCm/hipBLASLt/CK at runtime).
//! This module starts with a single-tile validation of the FP16 matrix core and
//! its per-lane fragment layout, then grows into a tiled GEMM (and the FP8 /
//! MXFP4 scaled-MFMA path) designed to be fused into device-resident decode
//! megakernels.

use anyhow::{anyhow, Result};
use std::time::{Duration, Instant};

use crate::gpu::GpuDevice;

/// Dispatch the best available FP16 GEMM for the shape: the LDS-staged,
/// register-blocked kernel when M,N are multiples of 64, else the simple tiled
/// kernel.
fn arm_gemm(dev: &mut GpuDevice, av: u64, bv: u64, cv: u64, m: u32, n: u32, k: u32) -> Result<()> {
    // lds2 (16.6 TF) currently beats the OpenCL double-buffered variant (14.9 TF)
    // — naive prefetch inflates VGPRs to 240 and crushes occupancy; near-peak
    // needs async global→LDS copies + scheduling below OpenCL (next chapter).
    if m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(32) {
        dev.arm_gemm_f16_lds2(av, bv, cv, m, n, k)
    } else if m.is_multiple_of(64) && n.is_multiple_of(64) {
        dev.arm_gemm_f16_lds(av, bv, cv, m, n, k)
    } else {
        dev.arm_gemm_f16(av, bv, cv, m, n, k)
    }
}

/// Round-to-nearest-even f32 → f16 (IEEE half) bit pattern. No external deps.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // Inf / NaN
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        // subnormal / underflow
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        // round to nearest even
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half_mant = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | half_mant
}

/// f16 (IEEE half) bit pattern → f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let e32 = (e + 127 - 15) as u32;
            (sign << 31) | (e32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        let e32 = exp + 127 - 15;
        (sign << 31) | (e32 << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Tiled FP16 GEMM: C = A·B on the matrix core, verified against an f64
/// reference computed from the same f16-rounded inputs (so only matrix-core /
/// accumulation-order error remains). M, N, K must be multiples of 16.
pub fn check_gemm_f16(node_id: u32, m: u32, n: u32, k: u32) -> Result<()> {
    let mut dev = GpuDevice::open(node_id)?;
    check_gemm_f16_on(&mut dev, m, n, k)
}

/// Like [`check_gemm_f16`] but on an already-open device (reuse across shapes).
pub fn check_gemm_f16_on(dev: &mut GpuDevice, m: u32, n: u32, k: u32) -> Result<()> {
    if !m.is_multiple_of(16) || !n.is_multiple_of(16) || !k.is_multiple_of(16) {
        return Err(anyhow!("M,N,K must be multiples of 16 (got {m},{n},{k})"));
    }
    let node_id = dev.node_id();
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);

    // Deterministic pseudo-random small values.
    let gen =
        |i: usize, s: usize| (((i * 1103515245 + s * 12345) >> 8) & 0xff) as f32 / 256.0 - 0.5;
    let a16: Vec<u16> = (0..mu * ku).map(|i| f32_to_f16(gen(i, 1))).collect();
    let b16: Vec<u16> = (0..ku * nu).map(|i| f32_to_f16(gen(i, 2))).collect();

    let mut a_buf = dev.alloc(mu * ku * 2)?;
    let mut b_buf = dev.alloc(ku * nu * 2)?;
    let mut c_buf = dev.alloc(mu * nu * 4)?;
    unsafe {
        a_buf.as_mut_slice_of::<u16>()[..mu * ku].copy_from_slice(&a16);
        b_buf.as_mut_slice_of::<u16>()[..ku * nu].copy_from_slice(&b16);
        for v in c_buf.as_mut_slice_of::<f32>() {
            *v = 0.0;
        }
    }
    let (av, bv, cv) = (a_buf.va(), b_buf.va(), c_buf.va());
    arm_gemm(dev, av, bv, cv, m, n, k)?;
    dev.wait(Duration::from_secs(10))?;

    // f64 reference from f16-rounded inputs.
    let af: Vec<f32> = a16.iter().map(|&h| f16_to_f32(h)).collect();
    let bf: Vec<f32> = b16.iter().map(|&h| f16_to_f32(h)).collect();
    let got = unsafe { c_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for mi in 0..mu {
        for ni in 0..nu {
            let mut acc = 0f64;
            for ki in 0..ku {
                acc += af[mi * ku + ki] as f64 * bf[ki * nu + ni] as f64;
            }
            let g = got[mi * nu + ni] as f64;
            let rel = (g - acc).abs() / (acc.abs() + 1.0);
            max_rel = max_rel.max(rel);
        }
    }
    if max_rel > 5e-2 {
        return Err(anyhow!(
            "GEMM {m}x{n}x{k} mismatch: max relative error {max_rel:.4} > 5e-2"
        ));
    }
    println!("  gemm: {m}x{n}x{k} FP16 matrix core on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Benchmark the tiled FP16 GEMM and report TFLOPS.
pub fn bench_gemm_f16(node_id: u32, m: u32, n: u32, k: u32, iters: usize) -> Result<f64> {
    let mut dev = GpuDevice::open(node_id)?;
    bench_gemm_f16_on(&mut dev, m, n, k, iters)
}

/// Like [`bench_gemm_f16`] but on an already-open device.
pub fn bench_gemm_f16_on(dev: &mut GpuDevice, m: u32, n: u32, k: u32, iters: usize) -> Result<f64> {
    if !m.is_multiple_of(16) || !n.is_multiple_of(16) || !k.is_multiple_of(16) {
        return Err(anyhow!("M,N,K must be multiples of 16"));
    }
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let mut a_buf = dev.alloc(mu * ku * 2)?;
    let mut b_buf = dev.alloc(ku * nu * 2)?;
    let c_buf = dev.alloc(mu * nu * 4)?;
    unsafe {
        for v in a_buf.as_mut_slice_of::<u16>() {
            *v = 0x3c00; // 1.0 in f16
        }
        for v in b_buf.as_mut_slice_of::<u16>() {
            *v = 0x3c00;
        }
    }
    let (av, bv, cv) = (a_buf.va(), b_buf.va(), c_buf.va());
    for _ in 0..3 {
        arm_gemm(dev, av, bv, cv, m, n, k)?;
        dev.wait(Duration::from_secs(10))?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        arm_gemm(dev, av, bv, cv, m, n, k)?;
        dev.wait(Duration::from_secs(10))?;
    }
    let t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let flops = 2.0 * mu as f64 * nu as f64 * ku as f64;
    Ok(flops / t / 1e12)
}

/// Validate the FP16 matrix core: compute D = A·B for one 16×16×16 tile on the
/// GPU and compare against an f32 CPU reference. Confirms MFMA emission on
/// gfx950 and that our per-lane A/B/D fragment layout is correct.
///
/// Inputs are small integers (exactly representable in f16) so the GPU's f16
/// multiply-accumulate matches the f32 reference exactly.
pub fn check_mfma_tile_16(node_id: u32) -> Result<()> {
    let mut dev = GpuDevice::open(node_id)?;
    check_mfma_tile_16_on(&mut dev)
}

/// Like [`check_mfma_tile_16`] but on an already-open device.
pub fn check_mfma_tile_16_on(dev: &mut GpuDevice) -> Result<()> {
    const N: usize = 16;
    let node_id = dev.node_id();

    // Deterministic small-integer inputs (exact in f16).
    let mut a = vec![0f32; N * N];
    let mut b = vec![0f32; N * N];
    for r in 0..N {
        for c in 0..N {
            a[r * N + c] = ((r + c) % 7) as f32 - 3.0; // -3..3
            b[r * N + c] = ((r * 2 + c) % 5) as f32 - 2.0; // -2..2
        }
    }

    let mut a_buf = dev.alloc(N * N * 4)?;
    let mut b_buf = dev.alloc(N * N * 4)?;
    let mut d_buf = dev.alloc(N * N * 4)?;
    unsafe {
        a_buf.as_mut_slice_of::<f32>()[..N * N].copy_from_slice(&a);
        b_buf.as_mut_slice_of::<f32>()[..N * N].copy_from_slice(&b);
        for v in d_buf.as_mut_slice_of::<f32>() {
            *v = 0.0;
        }
    }
    let (av, bv, dv) = (a_buf.va(), b_buf.va(), d_buf.va());

    dev.arm_mfma_tile_16(av, bv, dv)?;
    dev.wait(Duration::from_secs(5))?;

    // CPU reference: D = A·B (A row-major M×K, B row-major K×N).
    let mut expect = vec![0f32; N * N];
    for m in 0..N {
        for n in 0..N {
            let mut acc = 0f32;
            for k in 0..N {
                acc += a[m * N + k] * b[k * N + n];
            }
            expect[m * N + n] = acc;
        }
    }

    let got = unsafe { d_buf.as_mut_slice_of::<f32>() };
    let mut bad = 0usize;
    let mut first = None;
    for i in 0..N * N {
        if (got[i] - expect[i]).abs() > 1e-3 {
            bad += 1;
            if first.is_none() {
                first = Some((i, got[i], expect[i]));
            }
        }
    }
    if bad != 0 {
        let (i, g, e) = first.unwrap();
        return Err(anyhow!(
            "MFMA tile mismatch: {bad}/{} elements differ; first at [{}][{}] gpu={g} cpu={e}",
            N * N,
            i / N,
            i % N
        ));
    }
    println!(
        "  gemm: 16x16x16 FP16 matrix core on node {node_id}: bit-exact vs CPU reference ({} elements)",
        N * N
    );
    Ok(())
}

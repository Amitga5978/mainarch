//! Transformer decode-layer primitives for gfx950 (raw KFD/AQL, no ROCm):
//! RMSNorm and RoPE. The elementwise/normalization companions to the
//! FlashDecoding attention core — the rest of a decode step's non-GEMM ops.

use anyhow::{anyhow, Result};
use std::fs;
use std::time::{Duration, Instant};

use crate::attn::{e4m3_to_f32, f32_to_e4m3};
use crate::gemm::{f16_to_f32, f32_to_f16};
use crate::gpu::GpuDevice;

fn f32_to_e8m0_ru(x: f32) -> u8 {
    if !x.is_finite() || x <= 0.0 {
        return 127;
    }
    (x.log2().ceil() as i32 + 127).clamp(0, 255) as u8
}

fn e8m0_to_f32(x: u8) -> f32 {
    2.0f32.powi(x as i32 - 127)
}

fn host_kernel_release() -> String {
    fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn host_noretry_flag() -> String {
    match fs::read_to_string("/proc/cmdline") {
        Ok(cmdline) if cmdline.contains("amdgpu.noretry=1") => "amdgpu.noretry=1".to_string(),
        Ok(cmdline) if cmdline.contains("amdgpu.noretry=0") => "amdgpu.noretry=0".to_string(),
        Ok(_) => "amdgpu.noretry not present".to_string(),
        Err(_) => "cmdline unavailable".to_string(),
    }
}

fn host_kernel_version(release: &str) -> Option<(u32, u32)> {
    let base = release.split('-').next().unwrap_or(release);
    let mut parts = base.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn gfx950_f8_mode_hint(release: &str) -> &'static str {
    match host_kernel_version(release) {
        Some((major, minor)) if major < 6 || (major == 6 && minor < 12) => {
            "predates public gfx950 KFD F8_MODE fixes; scaled-FP8 numerics are expected to be unsafe until amdkfd is patched/backported"
        }
        Some((6, 12 | 13)) => {
            "requires confirmation that gfx950 F8_MODE fixes are backported; run this numeric gate before scaled-FP8 tile work"
        }
        Some((major, minor)) if major > 6 || (major == 6 && minor >= 14) => {
            "new enough to plausibly include gfx950 F8_MODE fixes, but this numeric gate is still authoritative"
        }
        Some(_) => "unknown kernel support window for gfx950 F8_MODE; this numeric gate is authoritative",
        None => "unable to parse kernel release; this numeric gate is authoritative",
    }
}

pub fn check_mfma_scale_probe_on(dev: &mut GpuDevice, iters: usize) -> Result<()> {
    let node_id = dev.node_id();
    let mut a_buf = dev.alloc_device(8 * 4)?;
    let mut b_buf = dev.alloc_device(8 * 4)?;
    let mut scale_buf = dev.alloc_device(2 * 4)?;
    let mut out_buf = dev.alloc(4 * 4)?;
    unsafe {
        let a = a_buf.as_mut_slice_of::<u32>();
        let b = b_buf.as_mut_slice_of::<u32>();
        for v in a.iter_mut().chain(b.iter_mut()) {
            *v = 0; // four OCP E4M3 zero values per dword
        }
        let s = scale_buf.as_mut_slice_of::<u32>();
        s[0] = 0;
        s[1] = 0;
        for v in out_buf.as_mut_slice_of::<f32>() {
            *v = 0.0;
        }
    }
    let (av, bv, sv, ov) = (a_buf.va(), b_buf.va(), scale_buf.va(), out_buf.va());
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_mfma_scale_f8f6f4_probe(av, bv, sv, ov)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    run(dev)?;
    let got = unsafe { out_buf.as_mut_slice_of::<f32>()[..4].to_vec() };
    for _ in 0..3 {
        run(dev)?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run(dev)?;
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64;
    println!(
        "  mfma-scale-probe: node {node_id} - V_MFMA_SCALE_F32_16X16X128_F8F6F4 retired (retirement probe, not numeric GEMM); out [{:.6e}, {:.6e}, {:.6e}, {:.6e}], avg {:.2} us",
        got[0], got[1], got[2], got[3], us
    );
    Ok(())
}

pub fn check_mfma_scale_calibrate_on(
    dev: &mut GpuDevice,
    iters: usize,
    report_only: bool,
) -> Result<()> {
    let node_id = dev.node_id();
    let kernel_release = host_kernel_release();
    let noretry = host_noretry_flag();
    let f8_hint = gfx950_f8_mode_hint(&kernel_release);
    println!("  mfma-scale-calibrate host: kernel {kernel_release}, {noretry}; {f8_hint}");
    let mut a_buf = dev.alloc_device(8 * 4)?;
    let mut b_buf = dev.alloc_device(8 * 4)?;
    let mut scale_buf = dev.alloc_device(2 * 4)?;
    let mut out_buf = dev.alloc(16 * 4)?;
    unsafe {
        let a = a_buf.as_mut_slice_of::<u32>();
        let b = b_buf.as_mut_slice_of::<u32>();
        for v in a.iter_mut().chain(b.iter_mut()) {
            *v = 0; // four OCP E4M3 zero values per dword
        }
        let s = scale_buf.as_mut_slice_of::<u32>();
        s[0] = 0;
        s[1] = 0x7f7f_7f7f; // four neutral E8M0 exponent bytes
        for v in out_buf.as_mut_slice_of::<f32>() {
            *v = 0.0;
        }
    }
    let (av, bv, sv, ov) = (a_buf.va(), b_buf.va(), scale_buf.va(), out_buf.va());
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_mfma_scale_f8f6f4_calibrate(av, bv, sv, ov)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    run(dev)?;

    let got = unsafe { out_buf.as_mut_slice_of::<f32>()[..16].to_vec() };
    let cases = [
        "ck-zero-scale",
        "ck-neutral-scale",
        "llvm-zero-scale",
        "llvm-neutral-scale",
    ];
    let expect = [1.0f32, 2.0, 3.0, 4.0];
    let mut best_err = f64::INFINITY;
    let mut best_case = "none";
    let mut finite_cases = 0usize;
    let mut nonfinite_cases = 0usize;
    for (ci, name) in cases.iter().enumerate() {
        let base = ci * 4;
        let mut case_err = 0.0f64;
        let mut case_finite = true;
        for j in 0..4 {
            let v = got[base + j];
            let e = if v.is_finite() {
                (v as f64 - expect[j] as f64).abs()
            } else {
                case_finite = false;
                f64::INFINITY
            };
            case_err = case_err.max(e);
        }
        if case_finite {
            finite_cases += 1;
        } else {
            nonfinite_cases += 1;
        }
        if case_err < best_err {
            best_err = case_err;
            best_case = name;
        }
        let bits = [
            got[base].to_bits(),
            got[base + 1].to_bits(),
            got[base + 2].to_bits(),
            got[base + 3].to_bits(),
        ];
        println!(
            "  mfma-scale-calibrate: {name:18} out [{:.6e}, {:.6e}, {:.6e}, {:.6e}] bits [0x{:08x},0x{:08x},0x{:08x},0x{:08x}] finite={} max_abs_vs_acc {:.3e}",
            got[base],
            got[base + 1],
            got[base + 2],
            got[base + 3],
            bits[0],
            bits[1],
            bits[2],
            bits[3],
            case_finite,
            case_err
        );
    }
    println!(
        "  mfma-scale-calibrate verdict: ready={} best_case={best_case} finite_cases={} nonfinite_cases={} report_only={}",
        best_err <= 1e-5,
        finite_cases,
        nonfinite_cases,
        report_only
    );
    if best_err > 1e-5 {
        let msg = format!(
            "mfma-scale-calibrate: no scaled-MFMA immediate/scale form preserved a zero-input accumulator on node {node_id}; best_case={best_case} max_abs={best_err:.3e} finite_cases={finite_cases} nonfinite_cases={nonfinite_cases}. Host kernel {kernel_release}, {noretry}: {f8_hint}. Treat raw KFD scaled-FP8 numerics as not ready, likely F8_MODE/format state, and do not build numeric tile kernels on this queue yet"
        );
        if report_only {
            println!("  mfma-scale-calibrate report-only: {msg}");
            return Ok(());
        }
        return Err(anyhow!(msg));
    }

    for _ in 0..3 {
        run(dev)?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run(dev)?;
    }
    let us = start.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64;
    println!(
        "  mfma-scale-calibrate: node {node_id} numeric zero-input gate passed via {best_case}, avg {:.2} us",
        us
    );
    Ok(())
}

/// Verify FP8-weight GEMV (per-row E4M3 scale) vs an f64 ref over decoded W;
/// report read bandwidth (half the FP16 weight bytes).
pub fn check_gemv_fp8_on(dev: &mut GpuDevice, n: u32, k: u32, iters: usize) -> Result<()> {
    let node_id = dev.node_id();
    let (nu, ku) = (n as usize, k as usize);
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let wf: Vec<f32> = (0..nu * ku).map(|i| gen(i, 1)).collect();
    let x16: Vec<u16> = (0..ku).map(|i| f32_to_f16(gen(i, 4))).collect();
    // Per-row E4M3 quantization: scale = max|row| / 448.
    let mut wq = vec![0u8; nu * ku];
    let mut sc = vec![1.0f32; nu];
    for nn in 0..nu {
        let row = &wf[nn * ku..nn * ku + ku];
        let maxabs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if maxabs > 0.0 { maxabs / 448.0 } else { 1.0 };
        sc[nn] = scale;
        let inv = 1.0 / scale;
        for kk in 0..ku {
            wq[nn * ku + kk] = f32_to_e4m3(row[kk] * inv);
        }
    }
    let mut w_buf = dev.alloc_device(nu * ku)?;
    let mut sc_buf = dev.alloc_device(nu * 4)?;
    let mut x_buf = dev.alloc_device(ku * 2)?;
    let mut y_buf = dev.alloc(nu * 4)?;
    unsafe {
        w_buf.as_mut_slice_of::<u8>()[..nu * ku].copy_from_slice(&wq);
        sc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&sc);
        x_buf.as_mut_slice_of::<u16>()[..ku].copy_from_slice(&x16);
    }
    let (wv, scv, xv, yv) = (w_buf.va(), sc_buf.va(), x_buf.va(), y_buf.va());
    dev.arm_gemv_fp8(wv, scv, xv, yv, n, k)?;
    dev.wait(Duration::from_secs(10))?;

    let xf: Vec<f64> = x16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let got = unsafe { y_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for nn in 0..nu {
        let mut acc = 0f64;
        for kk in 0..ku {
            acc += e4m3_to_f32(wq[nn * ku + kk]) as f64 * sc[nn] as f64 * xf[kk];
        }
        max_rel = max_rel.max((got[nn] as f64 - acc).abs() / (acc.abs() + 1e-2));
    }
    if max_rel > 1e-2 {
        return Err(anyhow!(
            "gemv-fp8 N={n} K={k} mismatch: {max_rel:.4} > 1e-2"
        ));
    }
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv_fp8(wv, scv, xv, yv, n, k)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    for _ in 0..3 {
        run(dev)?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run(dev)?;
    }
    let t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let gbps = (nu * ku + nu * 4) as f64 / t / 1e9;
    println!("  gemv-fp8: N={n} K={k} on node {node_id}: correct (rel err {max_rel:.2e}); {gbps:.0} GB/s (½ the weight bytes)");
    Ok(())
}

/// Verify FP8-weight GEMV with serving-shaped 128x128 E8M0 block scales. This
/// is the raw KFD/AQL W8A16 consumer gate before a full matrix-core A8W8 path.
pub fn check_gemv_fp8_wblock_on(dev: &mut GpuDevice, n: u32, k: u32, iters: usize) -> Result<()> {
    let node_id = dev.node_id();
    let (nu, ku) = (n as usize, k as usize);
    if ku % 512 != 0 {
        return Err(anyhow!("gemv-fp8-wblock requires K % 512 == 0"));
    }
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let wf: Vec<f32> = (0..nu * ku).map(|i| gen(i, 7)).collect();
    let x16: Vec<u16> = (0..ku).map(|i| f32_to_f16(gen(i, 13))).collect();
    let act_group = 64usize;
    let act_groups = ku.div_ceil(act_group);
    let mut xq = vec![0u8; ku];
    let mut x_act16 = vec![0u16; ku];
    let mut sx_packed = vec![0u32; act_groups.div_ceil(4)];
    for group in 0..act_groups {
        let start = group * act_group;
        let end = (start + act_group).min(ku);
        let mut maxabs = 0.0f32;
        for &h in &x16[start..end] {
            maxabs = maxabs.max(f16_to_f32(h).abs());
        }
        let raw_scale = if maxabs > 0.0 { maxabs / 448.0 } else { 1.0 };
        let code = f32_to_e8m0_ru(raw_scale);
        let scale = e8m0_to_f32(code);
        sx_packed[group / 4] |= (code as u32) << ((group & 3) * 8);
        let inv = 1.0 / scale;
        for kk in start..end {
            xq[kk] = f32_to_e4m3(f16_to_f32(x16[kk]) * inv);
            x_act16[kk] = f32_to_f16(e4m3_to_f32(xq[kk]) * scale);
        }
    }
    let nblocks = nu.div_ceil(128);
    let kblocks = ku.div_ceil(128);
    let packed_kblocks = kblocks.div_ceil(4);
    let mut wq = vec![0u8; nu * ku];
    let mut sc_e8m0 = vec![0u8; nblocks * kblocks];
    let mut sc_f32 = vec![1.0f32; nblocks * kblocks];
    let mut sc_packed = vec![0u32; nblocks * packed_kblocks];
    for nb in 0..nblocks {
        let row0 = nb * 128;
        let row1 = (row0 + 128).min(nu);
        for kb in 0..kblocks {
            let col0 = kb * 128;
            let col1 = (col0 + 128).min(ku);
            let mut maxabs = 0.0f32;
            for nn in row0..row1 {
                for kk in col0..col1 {
                    maxabs = maxabs.max(wf[nn * ku + kk].abs());
                }
            }
            let raw_scale = if maxabs > 0.0 { maxabs / 448.0 } else { 1.0 };
            let code = f32_to_e8m0_ru(raw_scale);
            let scale = e8m0_to_f32(code);
            sc_e8m0[nb * kblocks + kb] = code;
            sc_f32[nb * kblocks + kb] = scale;
            sc_packed[nb * packed_kblocks + kb / 4] |= (code as u32) << ((kb & 3) * 8);
            let inv = 1.0 / scale;
            for nn in row0..row1 {
                for kk in col0..col1 {
                    wq[nn * ku + kk] = f32_to_e4m3(wf[nn * ku + kk] * inv);
                }
            }
        }
    }
    let tile_stride = 16 + 128 * 128;
    let mut w_tiled = vec![0u8; nblocks * kblocks * tile_stride];
    for nb in 0..nblocks {
        for kb in 0..kblocks {
            let tile_base = (nb * kblocks + kb) * tile_stride;
            let sc = sc_e8m0[nb * kblocks + kb];
            for slot in 0..4 {
                w_tiled[tile_base + slot] = sc;
            }
            let row0 = nb * 128;
            let col0 = kb * 128;
            for rr in 0..128 {
                let nn = row0 + rr;
                if nn >= nu {
                    break;
                }
                let src0 = nn * ku + col0;
                let dst0 = tile_base + 16 + rr * 128;
                w_tiled[dst0..dst0 + 128].copy_from_slice(&wq[src0..src0 + 128]);
            }
        }
    }

    let mut w_buf = dev.alloc_device(nu * ku)?;
    let mut w_tiled_buf = dev.alloc_device(w_tiled.len())?;
    let mut sc_pack_buf = dev.alloc_device(sc_packed.len() * 4)?;
    let mut sc_f32_buf = dev.alloc_device(sc_f32.len() * 4)?;
    let mut x_buf = dev.alloc_device(ku * 2)?;
    let mut xq_buf = dev.alloc_device(ku)?;
    let mut sx_pack_buf = dev.alloc_device(sx_packed.len() * 4)?;
    let mut y_pack_buf = dev.alloc(nu * 4)?;
    let mut y_act_buf = dev.alloc(nu * 4)?;
    let mut y_tiled_buf = dev.alloc(nu * 4)?;
    let mut y_f32_buf = dev.alloc(nu * 4)?;
    unsafe {
        w_buf.as_mut_slice_of::<u8>()[..nu * ku].copy_from_slice(&wq);
        w_tiled_buf.as_mut_slice_of::<u8>()[..w_tiled.len()].copy_from_slice(&w_tiled);
        sc_pack_buf.as_mut_slice_of::<u32>()[..sc_packed.len()].copy_from_slice(&sc_packed);
        sc_f32_buf.as_mut_slice_of::<f32>()[..sc_f32.len()].copy_from_slice(&sc_f32);
        x_buf.as_mut_slice_of::<u16>()[..ku].copy_from_slice(&x16);
        xq_buf.as_mut_slice_of::<u8>()[..ku].copy_from_slice(&xq);
        sx_pack_buf.as_mut_slice_of::<u32>()[..sx_packed.len()].copy_from_slice(&sx_packed);
    }
    let (wv, wtv, spv, sfv, xv, xqv, sxv, ypv, yav, ytv, yfv) = (
        w_buf.va(),
        w_tiled_buf.va(),
        sc_pack_buf.va(),
        sc_f32_buf.va(),
        x_buf.va(),
        xq_buf.va(),
        sx_pack_buf.va(),
        y_pack_buf.va(),
        y_act_buf.va(),
        y_tiled_buf.va(),
        y_f32_buf.va(),
    );
    dev.arm_gemv_fp8_wblock_e8m0(wv, spv, xv, ypv, n, k)?;
    dev.wait(Duration::from_secs(10))?;
    dev.arm_gemv_fp8_wblock_act_e8m0(wv, spv, xqv, sxv, yav, n, k, act_group as u32)?;
    dev.wait(Duration::from_secs(10))?;
    dev.arm_gemv_fp8_wblock_tiled_e8m0(wtv, xv, ytv, n, k)?;
    dev.wait(Duration::from_secs(10))?;
    dev.arm_gemv_fp8_wblock_f32(wv, sfv, xv, yfv, n, k)?;
    dev.wait(Duration::from_secs(10))?;

    let xf: Vec<f64> = x16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let xaf: Vec<f64> = x_act16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let got_pack = unsafe { y_pack_buf.as_mut_slice_of::<f32>() };
    let got_act = unsafe { y_act_buf.as_mut_slice_of::<f32>() };
    let got_tiled = unsafe { y_tiled_buf.as_mut_slice_of::<f32>() };
    let got_f32 = unsafe { y_f32_buf.as_mut_slice_of::<f32>() };
    let mut max_pack_rel = 0f64;
    let mut max_act_rel = 0f64;
    let mut max_tiled_rel = 0f64;
    let mut max_f32_rel = 0f64;
    let mut tiled_bit_mismatches = 0usize;
    for nn in 0..nu {
        let nb = nn / 128;
        let mut acc = 0f64;
        let mut acc_act = 0f64;
        for kk in 0..ku {
            let kb = kk / 128;
            let scale = sc_f32[nb * kblocks + kb] as f64;
            let wv = e4m3_to_f32(wq[nn * ku + kk]) as f64 * scale;
            acc += wv * xf[kk];
            acc_act += wv * xaf[kk];
        }
        let denom = acc.abs() + 1e-2;
        let act_denom = acc_act.abs() + 1e-2;
        max_pack_rel = max_pack_rel.max((got_pack[nn] as f64 - acc).abs() / denom);
        max_act_rel = max_act_rel.max((got_act[nn] as f64 - acc_act).abs() / act_denom);
        max_tiled_rel = max_tiled_rel.max((got_tiled[nn] as f64 - acc).abs() / denom);
        max_f32_rel = max_f32_rel.max((got_f32[nn] as f64 - acc).abs() / denom);
        if got_tiled[nn].to_bits() != got_pack[nn].to_bits() {
            tiled_bit_mismatches += 1;
        }
    }
    if max_pack_rel > 1e-2
        || max_act_rel > 1e-2
        || max_tiled_rel > 1e-2
        || max_f32_rel > 1e-2
        || tiled_bit_mismatches != 0
    {
        return Err(anyhow!(
            "gemv-fp8-wblock N={n} K={k} mismatch: packed {max_pack_rel:.4}, act {max_act_rel:.4}, tiled {max_tiled_rel:.4}, f32 {max_f32_rel:.4}, tiled bit mismatches {tiled_bit_mismatches}"
        ));
    }

    let run_pack = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv_fp8_wblock_e8m0(wv, spv, xv, ypv, n, k)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    let run_act = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv_fp8_wblock_act_e8m0(wv, spv, xqv, sxv, yav, n, k, act_group as u32)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    let run_tiled = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv_fp8_wblock_tiled_e8m0(wtv, xv, ytv, n, k)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    let run_f32 = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv_fp8_wblock_f32(wv, sfv, xv, yfv, n, k)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    for _ in 0..3 {
        run_pack(dev)?;
        run_act(dev)?;
        run_tiled(dev)?;
        run_f32(dev)?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run_pack(dev)?;
    }
    let packed_t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run_act(dev)?;
    }
    let act_t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run_tiled(dev)?;
    }
    let tiled_t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run_f32(dev)?;
    }
    let f32_t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let packed_bytes = (nu * ku + sc_packed.len() * 4) as f64;
    let act_bytes = (nu * ku + sc_packed.len() * 4 + xq.len() + sx_packed.len() * 4) as f64;
    let tiled_bytes = w_tiled.len() as f64;
    let f32_bytes = (nu * ku + sc_f32.len() * 4) as f64;
    let packed_gbps = packed_bytes / packed_t / 1e9;
    let act_gbps = act_bytes / act_t / 1e9;
    let tiled_gbps = tiled_bytes / tiled_t / 1e9;
    let f32_gbps = f32_bytes / f32_t / 1e9;
    println!(
        "  gemv-fp8-wblock: N={n} K={k} on node {node_id} - packed rel {max_pack_rel:.2e}, act rel {max_act_rel:.2e}, tiled rel {max_tiled_rel:.2e}, f32 rel {max_f32_rel:.2e}, tiled bitdiff {tiled_bit_mismatches}; packed {packed_gbps:.0} GB/s, act-packed {act_gbps:.0} GB/s, tiled {tiled_gbps:.0} GB/s, f32-scale {f32_gbps:.0} GB/s"
    );
    Ok(())
}

/// Verify decode GEMV (y = W·x) vs an f64 reference, and report read bandwidth.
pub fn check_gemv_on(dev: &mut GpuDevice, n: u32, k: u32, iters: usize) -> Result<()> {
    let node_id = dev.node_id();
    let (nu, ku) = (n as usize, k as usize);
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let w16: Vec<u16> = (0..nu * ku).map(|i| f32_to_f16(gen(i, 1))).collect();
    let x16: Vec<u16> = (0..ku).map(|i| f32_to_f16(gen(i, 4))).collect();

    let mut w_buf = dev.alloc_device(nu * ku * 2)?;
    let mut x_buf = dev.alloc_device(ku * 2)?;
    let mut y_buf = dev.alloc(nu * 4)?;
    unsafe {
        w_buf.as_mut_slice_of::<u16>()[..nu * ku].copy_from_slice(&w16);
        x_buf.as_mut_slice_of::<u16>()[..ku].copy_from_slice(&x16);
    }
    let (wv, xv, yv) = (w_buf.va(), x_buf.va(), y_buf.va());
    dev.arm_gemv(wv, xv, yv, n, k)?;
    dev.wait(Duration::from_secs(10))?;

    let xf: Vec<f64> = x16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let got = unsafe { y_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for nn in 0..nu {
        let mut acc = 0f64;
        for kk in 0..ku {
            acc += f16_to_f32(w16[nn * ku + kk]) as f64 * xf[kk];
        }
        max_rel = max_rel.max((got[nn] as f64 - acc).abs() / (acc.abs() + 1e-2));
    }
    if max_rel > 1e-2 {
        return Err(anyhow!(
            "gemv N={n} K={k} mismatch: max rel err {max_rel:.4} > 1e-2"
        ));
    }
    // Bandwidth: read W (N×K f16) once.
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv(wv, xv, yv, n, k)?;
        dev.wait(Duration::from_secs(10))?;
        Ok(())
    };
    for _ in 0..3 {
        run(dev)?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run(dev)?;
    }
    let t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let gbps = (nu * ku * 2) as f64 / t / 1e9;
    println!("  gemv: N={n} K={k} on node {node_id}: correct (rel err {max_rel:.2e}); {gbps:.0} GB/s ({:.0}% HBM)", gbps / 8000.0 * 100.0);
    Ok(())
}

/// Verify RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight, vs an f64 reference.
pub fn check_rmsnorm_on(dev: &mut GpuDevice, h: u32) -> Result<()> {
    let node_id = dev.node_id();
    let hu = h as usize;
    let eps = 1e-6f32;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let x16: Vec<u16> = (0..hu).map(|i| f32_to_f16(gen(i, 3))).collect();
    let w16: Vec<u16> = (0..hu).map(|i| f32_to_f16(gen(i, 9) + 1.0)).collect();

    let mut x_buf = dev.alloc_device(hu * 2)?;
    let mut w_buf = dev.alloc_device(hu * 2)?;
    let mut y_buf = dev.alloc(hu * 2)?;
    unsafe {
        x_buf.as_mut_slice_of::<u16>()[..hu].copy_from_slice(&x16);
        w_buf.as_mut_slice_of::<u16>()[..hu].copy_from_slice(&w16);
    }
    dev.arm_rmsnorm(x_buf.va(), w_buf.va(), y_buf.va(), h, eps)?;
    dev.wait(Duration::from_secs(5))?;

    let ss: f64 = (0..hu).map(|i| (f16_to_f32(x16[i]) as f64).powi(2)).sum();
    let rms = 1.0 / (ss / hu as f64 + eps as f64).sqrt();
    let got = unsafe { y_buf.as_mut_slice_of::<u16>() };
    let mut max_rel = 0f64;
    for i in 0..hu {
        let exp = f16_to_f32(x16[i]) as f64 * rms * f16_to_f32(w16[i]) as f64;
        let g = f16_to_f32(got[i]) as f64;
        max_rel = max_rel.max((g - exp).abs() / (exp.abs() + 1e-3));
    }
    if max_rel > 5e-3 {
        return Err(anyhow!(
            "rmsnorm H={h} mismatch: max rel err {max_rel:.4} > 5e-3"
        ));
    }
    println!("  rmsnorm: H={h} on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Verify SwiGLU: y = silu(gate) * up, vs an f64 reference.
pub fn check_swiglu_on(dev: &mut GpuDevice, n: u32) -> Result<()> {
    let node_id = dev.node_id();
    let nu = n as usize;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let gate: Vec<u16> = (0..nu).map(|i| f32_to_f16(gen(i, 2) * 4.0)).collect();
    let up: Vec<u16> = (0..nu).map(|i| f32_to_f16(gen(i, 6))).collect();

    let mut g_buf = dev.alloc_device(nu * 2)?;
    let mut u_buf = dev.alloc_device(nu * 2)?;
    let mut y_buf = dev.alloc(nu * 2)?;
    unsafe {
        g_buf.as_mut_slice_of::<u16>()[..nu].copy_from_slice(&gate);
        u_buf.as_mut_slice_of::<u16>()[..nu].copy_from_slice(&up);
    }
    dev.arm_swiglu(g_buf.va(), u_buf.va(), y_buf.va(), n)?;
    dev.wait(Duration::from_secs(5))?;

    let got = unsafe { y_buf.as_mut_slice_of::<u16>() };
    let mut max_rel = 0f64;
    for i in 0..nu {
        let g = f16_to_f32(gate[i]) as f64;
        let silu = g / (1.0 + (-g).exp());
        let exp = silu * f16_to_f32(up[i]) as f64;
        let got_v = f16_to_f32(got[i]) as f64;
        max_rel = max_rel.max((got_v - exp).abs() / (exp.abs() + 1e-2));
    }
    if max_rel > 5e-3 {
        return Err(anyhow!(
            "swiglu n={n} mismatch: max rel err {max_rel:.4} > 5e-3"
        ));
    }
    println!("  swiglu: n={n} on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Verify RoPE (half-rotation) at a given position vs an f64 reference.
pub fn check_rope_on(dev: &mut GpuDevice, h: u32) -> Result<()> {
    let node_id = dev.node_id();
    let hu = h as usize;
    let half = hu / 2;
    let pos = 12345u32;
    let theta = 10000.0f32;
    let gen = |i: usize| (((i * 2654435761 + 5 * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let x16: Vec<u16> = (0..hu).map(|i| f32_to_f16(gen(i))).collect();

    let mut x_buf = dev.alloc_device(hu * 2)?;
    unsafe {
        x_buf.as_mut_slice_of::<u16>()[..hu].copy_from_slice(&x16);
    }
    dev.arm_rope(x_buf.va(), h, pos, theta)?;
    dev.wait(Duration::from_secs(5))?;

    let got = unsafe { x_buf.as_mut_slice_of::<u16>() };
    let mut max_abs = 0f64;
    for i in 0..half {
        let freq = (theta as f64).powf(-2.0 * i as f64 / hu as f64);
        let ang = pos as f64 * freq;
        let (c, s) = (ang.cos(), ang.sin());
        let a = f16_to_f32(x16[i]) as f64;
        let b = f16_to_f32(x16[i + half]) as f64;
        let e0 = a * c - b * s;
        let e1 = b * c + a * s;
        max_abs = max_abs
            .max((f16_to_f32(got[i]) as f64 - e0).abs())
            .max((f16_to_f32(got[i + half]) as f64 - e1).abs());
    }
    if max_abs > 5e-3 {
        return Err(anyhow!(
            "rope H={h} mismatch: max abs err {max_abs:.4} > 5e-3"
        ));
    }
    println!("  rope: H={h} pos={pos} on node {node_id}: correct (max abs err {max_abs:.2e})");
    Ok(())
}

/// Verify the MoE FFN decode primitive (Qwen3-MoE: router top-K → softmax over
/// the K selected, no shared expert → router-weighted sum of expert SwiGLUs) vs
/// an f64 reference, and report expert-weight read bandwidth. Weight tensors:
/// gate/up [E][I][H], down [E][H][I]; router gate [E][H]. The expert index is
/// resolved on-device (no host round-trip) — the host reads the GPU's selected
/// ids and validates the router softmax and the FFN math against them.
#[allow(clippy::too_many_arguments)]
pub fn check_moe_ffn_on(
    dev: &mut GpuDevice,
    h: u32,
    i_dim: u32,
    e: u32,
    topk: u32,
    iters: usize,
) -> Result<()> {
    let node_id = dev.node_id();
    let (hu, iu, eu, ku) = (h as usize, i_dim as usize, e as usize, topk as usize);
    let gen = |a: usize, s: usize| {
        ((a.wrapping_mul(2654435761)
            .wrapping_add(s.wrapping_mul(40503))
            >> 11)
            & 0xff) as f32
            / 256.0
            - 0.5
    };
    let x16: Vec<u16> = (0..hu).map(|a| f32_to_f16(gen(a, 7))).collect();
    let wgate16: Vec<u16> = (0..eu * hu).map(|a| f32_to_f16(gen(a, 11))).collect();
    let gate16: Vec<u16> = (0..eu * iu * hu)
        .map(|a| f32_to_f16(gen(a, 13) * 0.1))
        .collect();
    let up16: Vec<u16> = (0..eu * iu * hu)
        .map(|a| f32_to_f16(gen(a, 17) * 0.1))
        .collect();
    let down16: Vec<u16> = (0..eu * hu * iu)
        .map(|a| f32_to_f16(gen(a, 19) * 0.1))
        .collect();

    let mut x_buf = dev.alloc_device(hu * 2)?;
    let mut wgate_buf = dev.alloc_device(eu * hu * 2)?;
    let mut gate_buf = dev.alloc_device(eu * iu * hu * 2)?;
    let mut up_buf = dev.alloc_device(eu * iu * hu * 2)?;
    let mut down_buf = dev.alloc_device(eu * hu * iu * 2)?;
    let logits_buf = dev.alloc_device(eu * 4)?;
    let mut ids_buf = dev.alloc_device(ku * 4)?;
    let mut w_buf = dev.alloc_device(ku * 4)?;
    let h_buf = dev.alloc_device(iu * 2)?;
    let mut out_buf = dev.alloc_device(hu * 4)?;
    unsafe {
        x_buf.as_mut_slice_of::<u16>()[..hu].copy_from_slice(&x16);
        wgate_buf.as_mut_slice_of::<u16>()[..eu * hu].copy_from_slice(&wgate16);
        gate_buf.as_mut_slice_of::<u16>()[..eu * iu * hu].copy_from_slice(&gate16);
        up_buf.as_mut_slice_of::<u16>()[..eu * iu * hu].copy_from_slice(&up16);
        down_buf.as_mut_slice_of::<u16>()[..eu * hu * iu].copy_from_slice(&down16);
    }
    let (xv, wgv, gv, uv, dv) = (
        x_buf.va(),
        wgate_buf.va(),
        gate_buf.va(),
        up_buf.va(),
        down_buf.va(),
    );
    let (lv, idv, wv, hv, ov) = (
        logits_buf.va(),
        ids_buf.va(),
        w_buf.va(),
        h_buf.va(),
        out_buf.va(),
    );

    // One correctness execution with a clean (zeroed) accumulator.
    unsafe {
        for v in out_buf.as_mut_slice_of::<f32>()[..hu].iter_mut() {
            *v = 0.0;
        }
    }
    dev.arm_gemv(wgv, xv, lv, e, h)?;
    dev.wait(Duration::from_secs(10))?;
    dev.arm_moe_router_topk(lv, idv, wv, e, topk)?;
    dev.wait(Duration::from_secs(10))?;
    for slot in 0..topk {
        dev.arm_moe_gate_up_swiglu(gv, uv, xv, idv, hv, slot, e, i_dim, h)?;
        dev.wait(Duration::from_secs(10))?;
        dev.arm_moe_down_accum(dv, hv, idv, wv, ov, slot, e, h, i_dim)?;
        dev.wait(Duration::from_secs(10))?;
    }
    let ids: Vec<u32> = unsafe { ids_buf.as_mut_slice_of::<u32>()[..ku].to_vec() };
    let wts: Vec<f32> = unsafe { w_buf.as_mut_slice_of::<f32>()[..ku].to_vec() };
    let got: Vec<f32> = unsafe { out_buf.as_mut_slice_of::<f32>()[..hu].to_vec() };

    let xf: Vec<f64> = x16.iter().map(|&v| f16_to_f32(v) as f64).collect();
    for &id in &ids {
        if id >= e {
            return Err(anyhow!("moe-ffn: expert id {id} out of range E={e}"));
        }
    }
    // Router softmax over the GPU-selected ids' logits (f64), vs the GPU weights.
    let mut hl = vec![0f64; ku];
    for (j, &id) in ids.iter().enumerate() {
        let mut acc = 0f64;
        for k in 0..hu {
            acc += f16_to_f32(wgate16[id as usize * hu + k]) as f64 * xf[k];
        }
        hl[j] = acc;
    }
    let mx = hl.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (mut s, mut hw) = (0f64, vec![0f64; ku]);
    for j in 0..ku {
        hw[j] = (hl[j] - mx).exp();
        s += hw[j];
    }
    let mut w_rel = 0f64;
    for j in 0..ku {
        hw[j] /= s;
        w_rel = w_rel.max((wts[j] as f64 - hw[j]).abs() / (hw[j].abs() + 1e-6));
    }
    // FFN reference over the GPU-selected ids and weights.
    let mut out_ref = vec![0f64; hu];
    for (slot, &id) in ids.iter().enumerate() {
        let (off_gu, off_dn) = (id as usize * iu * hu, id as usize * hu * iu);
        let mut hh = vec![0f64; iu];
        for r in 0..iu {
            let (mut g, mut u) = (0f64, 0f64);
            for k in 0..hu {
                g += f16_to_f32(gate16[off_gu + r * hu + k]) as f64 * xf[k];
                u += f16_to_f32(up16[off_gu + r * hu + k]) as f64 * xf[k];
            }
            hh[r] = (g / (1.0 + (-g).exp())) * u;
        }
        let wj = wts[slot] as f64;
        for n in 0..hu {
            let mut d = 0f64;
            for r in 0..iu {
                d += f16_to_f32(down16[off_dn + n * iu + r]) as f64 * hh[r];
            }
            out_ref[n] += wj * d;
        }
    }
    let (mut num, mut den) = (0f64, 0f64);
    for n in 0..hu {
        let d = got[n] as f64 - out_ref[n];
        num += d * d;
        den += out_ref[n] * out_ref[n];
    }
    let rel_l2 = (num / den.max(1e-30)).sqrt();
    if w_rel > 1e-2 {
        return Err(anyhow!(
            "moe-ffn router weights mismatch: {w_rel:.4} > 1e-2"
        ));
    }
    if rel_l2 > 1e-2 {
        return Err(anyhow!(
            "moe-ffn H={h} I={i_dim} E={e} mismatch: rel-L2 {rel_l2:.4} > 1e-2"
        ));
    }

    // Bandwidth: read topk experts' gate+up+down (f16) per token. Each dispatch
    // waits (dispatch-chaining, as in the attention path, is a later perf step).
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_gemv(wgv, xv, lv, e, h)?;
        dev.wait(Duration::from_secs(10))?;
        dev.arm_moe_router_topk(lv, idv, wv, e, topk)?;
        dev.wait(Duration::from_secs(10))?;
        for slot in 0..topk {
            dev.arm_moe_gate_up_swiglu(gv, uv, xv, idv, hv, slot, e, i_dim, h)?;
            dev.wait(Duration::from_secs(10))?;
            dev.arm_moe_down_accum(dv, hv, idv, wv, ov, slot, e, h, i_dim)?;
            dev.wait(Duration::from_secs(10))?;
        }
        Ok(())
    };
    for _ in 0..2 {
        run(dev)?;
    }
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        run(dev)?;
    }
    let t = start.elapsed().as_secs_f64() / iters.max(1) as f64;
    let bytes = ku as f64 * 3.0 * iu as f64 * hu as f64 * 2.0;
    let gbps = bytes / t / 1e9;
    println!(
        "  moe-ffn: H={h} I={i_dim} E={e} top{topk} on node {node_id}: correct (out rel-L2 {rel_l2:.2e}, w rel {w_rel:.2e}); {gbps:.0} GB/s weights, {:.1} µs",
        t * 1e6
    );
    Ok(())
}

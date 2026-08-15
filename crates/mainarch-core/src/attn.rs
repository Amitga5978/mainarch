//! FlashDecoding attention (decode phase) for gfx950 — the 1M-context decode
//! core, built on the raw KFD/AQL path (no ROCm/CK at runtime).
//!
//! Decode attention is memory-bound: one query reads the entire KV cache once,
//! so SOTA = saturate HBM bandwidth. Split-KV parallelizes the long KV across
//! many workgroups (online softmax per split), then a combine kernel merges the
//! partial softmax states. FP16 KV first; FP8/MXFP4 KV (the 2-4x decode
//! multiplier) is the next chapter.

// Dense decode-attention math indexes parallel q/k/v/score arrays by position;
// range-indexed loops read more clearly here than iterator adapters.
#![allow(clippy::needless_range_loop)]

use anyhow::{anyhow, Result};
use std::time::{Duration, Instant};

use crate::gemm::{f16_to_f32, f32_to_f16};
use crate::gpu::GpuDevice;

const HEAD_DIM_MAX: u32 = 128;
/// Query heads per KV head in the GQA path — must match `#define GQA_G` in the kernel.
const GQA_G: usize = 8;

pub(crate) fn split_count(n: u32) -> u32 {
    if n <= 4096 {
        // NB (2026-06-19): after the grouped-GQA/batched-combine path landed,
        // the old 512-split short-context floor became overkill for the
        // model-decode serving gate. On MI355X, N=512 capstone chained_full was
        // 39.5us at 32 splits vs 47.3us at 512 splits; full Qwen model-decode
        // improved 1171us/token -> 947-967us/token with this short bin.
        return 32;
    }
    if n <= 8192 {
        // 8K decode: 192 splits beat the old 512 floor on MI355X
        // (47.0us vs 51.5us chained_full, 2026-06-19).
        return 192;
    }
    if n <= 32768 {
        // 16K-32K decode: 256 splits beat the 512-split floor on MI355X after
        // the grouped-GQA/chained path and later kernel tuning. Measured
        // chained_full: 16K 49.7us at 256 splits, 24K 51.1-51.3us vs 55.2us,
        // 32K 56.3us vs 56.5us (2026-06-20). Larger contexts stay on the
        // long-context floor below.
        return 256;
    }
    // Target enough splits to fill the GPU (occupancy), capped. Short contexts
    // were occupancy-starved at 512 tokens/split; with the two-pass tree combine
    // + chained dispatch the combine cost grows only ~√splits, so we can afford
    // more splits (~128 tokens each) before the combine outweighs the gain.
    //
    // NB (2026-06-14): a too-aggressive relaxation to ~5 splits regressed
    // model-decode badly. The <=1024-token bin above is the measured middle
    // ground after grouped-GQA/batched-combine: enough occupancy without paying
    // the 512-split combine cost.
    //
    // NB (2026-06-18): MI355X capstone stage profiling showed the 1M
    // paged+GQA+FP4 path is split/load dominated and over-split at 512
    // tokens/chunk: 2048 splits gave 281.0 µs chained full, while 512 splits
    // (2048 tokens/chunk) gave 250.1 µs. Keep the short-context floor, but use
    // larger long-context chunks once the floor no longer controls occupancy.
    let tokens_per_split = if n >= 524_288 { 2048 } else { 512 };
    (n / tokens_per_split).clamp(512, 4096)
}

/// Batched GQA combine: merge head-major partials for all `num_heads` heads into
/// O[head*D..] in 2 dispatches total (not 2 per head). `inter_va` must hold
/// num_heads * combine_groups(num_splits) partials.
pub(crate) fn combine_decode_gqa(
    dev: &mut GpuDevice,
    pv: u64,
    inter_va: u64,
    ov: u64,
    d: u32,
    num_splits: u32,
    num_heads: u32,
) -> Result<()> {
    let ng = combine_groups(num_splits);
    if ng == 0 {
        dev.arm_attn_decode_combine_gqa(pv, ov, d, num_splits, num_heads)?;
        dev.wait(Duration::from_secs(10))?;
    } else {
        let gs = num_splits.div_ceil(ng);
        dev.chain_next();
        dev.arm_attn_decode_reduce_partials_gqa(pv, inter_va, num_splits, d, gs, ng, num_heads)?;
        dev.arm_attn_decode_combine_gqa(inter_va, ov, d, ng, num_heads)?;
        dev.wait(Duration::from_secs(10))?;
    }
    Ok(())
}

/// Batched GQA combine that writes the final normalized attention output as f16.
/// The partials and intermediate tree reduction remain f32 for numerical stability.
pub(crate) fn enqueue_combine_decode_gqa_f16(
    dev: &mut GpuDevice,
    pv: u64,
    inter_va: u64,
    ov: u64,
    d: u32,
    num_splits: u32,
    num_heads: u32,
) -> Result<()> {
    let ng = combine_groups(num_splits);
    if ng == 0 {
        dev.chain_next();
        dev.arm_attn_decode_combine_gqa_f16(pv, ov, d, num_splits, num_heads)?;
    } else {
        let gs = num_splits.div_ceil(ng);
        dev.chain_next();
        dev.arm_attn_decode_reduce_partials_gqa(pv, inter_va, num_splits, d, gs, ng, num_heads)?;
        dev.chain_next();
        dev.arm_attn_decode_combine_gqa_f16(inter_va, ov, d, ng, num_heads)?;
    }
    Ok(())
}

/// Number of intermediate partials the two-pass combine produces (≈√num_splits),
/// or 0 if a single-pass combine is used.
pub(crate) fn combine_groups(num_splits: u32) -> u32 {
    if num_splits <= 128 {
        return 0;
    }
    if num_splits == 192 {
        // 8K GQA/FP4 decode bin: 16 groups (12 splits/group) beats the sqrt
        // default of 14 groups on MI355X (44.1us vs 46.9us chained_full).
        return 16;
    }
    let gs = (num_splits as f64).sqrt().ceil() as u32;
    num_splits.div_ceil(gs)
}

/// Merge `num_splits` softmax partials into the final output O[D]. For large
/// num_splits this is a two-pass tree reduction (pass A across CUs, pass B a
/// short final combine) using `inter_va` as scratch; otherwise a direct combine.
pub(crate) fn combine_decode(
    dev: &mut GpuDevice,
    pv: u64,
    inter_va: u64,
    ov: u64,
    d: u32,
    num_splits: u32,
) -> Result<()> {
    // Chained onto the (already-enqueued, unsignaled) split: the AQL barrier bit
    // orders each dispatch after the prior, so we wait ONCE at the end instead of
    // a host round-trip (~11us) per dispatch.
    let ng = combine_groups(num_splits);
    if ng == 0 {
        dev.arm_attn_decode_combine(pv, ov, d, num_splits)?;
        dev.wait(Duration::from_secs(10))?;
    } else {
        let gs = num_splits.div_ceil(ng);
        dev.chain_next();
        dev.arm_attn_decode_reduce_partials(pv, inter_va, num_splits, d, gs, ng)?;
        dev.arm_attn_decode_combine(inter_va, ov, d, ng)?;
        dev.wait(Duration::from_secs(10))?;
    }
    Ok(())
}

/// Verify decode attention: O = softmax(scale·q·Kᵀ)·V, bit-close to an f64
/// reference. q,K,V are FP16; head dim D ≤ 128, KV length N.
pub fn check_attn_decode(node_id: u32, n: u32, d: u32) -> Result<()> {
    let mut dev = GpuDevice::open(node_id)?;
    check_attn_decode_on(&mut dev, n, d)
}

/// Verify FlashInfer-style paged KV append: source K/V rows are addressed by
/// `[append_token][kv_head][D]`, then mapped through batch_indices + positions +
/// paged metadata into NHD pages `[page][offset][kv_head][D]`.
pub fn check_kv_append_paged_on(dev: &mut GpuDevice, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("paged KV append selftest requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let du = d as usize;
    let nkv = 4usize;
    let block_size = 16usize;
    let physical_blocks = 12usize;
    let batch = 3usize;
    let indices: Vec<u32> = vec![5, 2, 9, 1, 7, 3, 10, 0, 11, 4, 8, 6];
    let indptr: Vec<u32> = vec![0, 4, 8, 12];
    let last_page_len: Vec<u32> = vec![16, 7, 16];
    let batch_indices: Vec<u32> = vec![0, 1, 2, 1, 0, 2];
    let positions: Vec<u32> = vec![0, 54, 63, 16, 37, 3];
    let append_count = batch_indices.len();
    let total_elems = append_count * nkv * du;
    let cache_elems = physical_blocks * block_size * nkv * du;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let k16: Vec<u16> = (0..total_elems).map(|i| f32_to_f16(gen(i, 31))).collect();
    let v16: Vec<u16> = (0..total_elems).map(|i| f32_to_f16(gen(i, 32))).collect();
    let sentinel = f32_to_f16(-13.0);

    let mut kcache_buf = dev.alloc_device(cache_elems * 2)?;
    let mut vcache_buf = dev.alloc_device(cache_elems * 2)?;
    let mut k_buf = dev.alloc_device(total_elems * 2)?;
    let mut v_buf = dev.alloc_device(total_elems * 2)?;
    let mut indices_buf = dev.alloc_device(indices.len() * 4)?;
    let mut indptr_buf = dev.alloc_device(indptr.len() * 4)?;
    let mut last_buf = dev.alloc_device(last_page_len.len() * 4)?;
    let mut batch_buf = dev.alloc_device(batch_indices.len() * 4)?;
    let mut pos_buf = dev.alloc_device(positions.len() * 4)?;
    unsafe {
        kcache_buf.as_mut_slice_of::<u16>()[..cache_elems].fill(sentinel);
        vcache_buf.as_mut_slice_of::<u16>()[..cache_elems].fill(sentinel);
        k_buf.as_mut_slice_of::<u16>()[..total_elems].copy_from_slice(&k16);
        v_buf.as_mut_slice_of::<u16>()[..total_elems].copy_from_slice(&v16);
        indices_buf.as_mut_slice_of::<u32>()[..indices.len()].copy_from_slice(&indices);
        indptr_buf.as_mut_slice_of::<u32>()[..indptr.len()].copy_from_slice(&indptr);
        last_buf.as_mut_slice_of::<u32>()[..last_page_len.len()].copy_from_slice(&last_page_len);
        batch_buf.as_mut_slice_of::<u32>()[..batch_indices.len()].copy_from_slice(&batch_indices);
        pos_buf.as_mut_slice_of::<u32>()[..positions.len()].copy_from_slice(&positions);
    }
    dev.arm_kv_append_paged(
        kcache_buf.va(),
        vcache_buf.va(),
        k_buf.va(),
        v_buf.va(),
        indices_buf.va(),
        indptr_buf.va(),
        last_buf.va(),
        batch_buf.va(),
        pos_buf.va(),
        append_count as u32,
        batch as u32,
        indices.len() as u32,
        physical_blocks as u32,
        nkv as u32,
        block_size as u32,
        d,
    )?;
    dev.wait(Duration::from_secs(5))?;

    let kc = unsafe { kcache_buf.as_mut_slice_of::<u16>() };
    let vc = unsafe { vcache_buf.as_mut_slice_of::<u16>() };
    for token in 0..append_count {
        let b = batch_indices[token] as usize;
        let pos = positions[token] as usize;
        let lo = indptr[b] as usize;
        let page = indices[lo + pos / block_size] as usize;
        let offset = pos % block_size;
        for h in 0..nkv {
            for dd in 0..du {
                let src = (token * nkv + h) * du + dd;
                let dst = (((page * block_size + offset) * nkv + h) * du) + dd;
                if kc[dst] != k16[src] || vc[dst] != v16[src] {
                    return Err(anyhow!(
                        "paged KV append mismatch token {token} head {h} d {dd}: dst {dst}"
                    ));
                }
            }
        }
    }

    let untouched_page = 2usize;
    let untouched_dst = (((untouched_page * block_size + 15) * nkv + 3) * du) + 127;
    if kc[untouched_dst] != sentinel || vc[untouched_dst] != sentinel {
        return Err(anyhow!("paged KV append overwrote an untouched cache slot"));
    }

    let bad_batch_indices: Vec<u32> = vec![99];
    let bad_positions: Vec<u32> = vec![u32::MAX];
    let bad_k16: Vec<u16> = (0..nkv * du)
        .map(|i| f32_to_f16(100.0 + i as f32))
        .collect();
    let bad_v16: Vec<u16> = (0..nkv * du)
        .map(|i| f32_to_f16(200.0 + i as f32))
        .collect();
    unsafe {
        k_buf.as_mut_slice_of::<u16>()[..nkv * du].copy_from_slice(&bad_k16);
        v_buf.as_mut_slice_of::<u16>()[..nkv * du].copy_from_slice(&bad_v16);
        batch_buf.as_mut_slice_of::<u32>()[0] = bad_batch_indices[0];
        pos_buf.as_mut_slice_of::<u32>()[0] = bad_positions[0];
    }
    dev.arm_kv_append_paged(
        kcache_buf.va(),
        vcache_buf.va(),
        k_buf.va(),
        v_buf.va(),
        indices_buf.va(),
        indptr_buf.va(),
        last_buf.va(),
        batch_buf.va(),
        pos_buf.va(),
        1,
        batch as u32,
        indices.len() as u32,
        physical_blocks as u32,
        nkv as u32,
        block_size as u32,
        d,
    )?;
    dev.wait(Duration::from_secs(5))?;
    let kc = unsafe { kcache_buf.as_mut_slice_of::<u16>() };
    let vc = unsafe { vcache_buf.as_mut_slice_of::<u16>() };
    if kc[untouched_dst] != sentinel || vc[untouched_dst] != sentinel {
        return Err(anyhow!("invalid paged KV append wrote into the cache"));
    }

    println!(
        "  paged-kv-append: {} tokens x {} KV heads D={} into {} pages (NHD, shuffled) on node {} - correct",
        append_count, nkv, d, physical_blocks, node_id
    );
    Ok(())
}

/// Verify the full paged KV write/read loop: append every logical K/V row through
/// FlashInfer-style paged metadata, then decode from the same physical pages.
pub fn check_attn_decode_paged_after_append_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("append-fed paged decode requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 37))).collect();
    let k16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 38))).collect();
    let v16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 39))).collect();
    let table: Vec<u32> = (0..nblocks)
        .map(|i| ((i * 17 + 11) % nblocks) as u32)
        .collect();
    let indptr = vec![0u32, nblocks as u32];
    let last_page_len = vec![bs as u32];
    let batch_indices = vec![0u32; nu];
    let positions: Vec<u32> = (0..nu).map(|i| i as u32).collect();

    let mut q_buf = dev.alloc(du * 2)?;
    let mut kcache_buf = dev.alloc_device(nu * du * 2)?;
    let mut vcache_buf = dev.alloc_device(nu * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du * 2)?;
    let mut v_buf = dev.alloc_device(nu * du * 2)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let mut indptr_buf = dev.alloc_device(indptr.len() * 4)?;
    let mut last_buf = dev.alloc_device(last_page_len.len() * 4)?;
    let mut batch_buf = dev.alloc_device(batch_indices.len() * 4)?;
    let mut pos_buf = dev.alloc_device(positions.len() * 4)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let mut o_buf = dev.alloc(du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        kcache_buf.as_mut_slice_of::<u16>()[..nu * du].fill(f32_to_f16(-17.0));
        vcache_buf.as_mut_slice_of::<u16>()[..nu * du].fill(f32_to_f16(-19.0));
        k_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&k16);
        v_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&v16);
        tbl_buf.as_mut_slice_of::<u32>()[..nblocks].copy_from_slice(&table);
        indptr_buf.as_mut_slice_of::<u32>()[..indptr.len()].copy_from_slice(&indptr);
        last_buf.as_mut_slice_of::<u32>()[..last_page_len.len()].copy_from_slice(&last_page_len);
        batch_buf.as_mut_slice_of::<u32>()[..batch_indices.len()].copy_from_slice(&batch_indices);
        pos_buf.as_mut_slice_of::<u32>()[..positions.len()].copy_from_slice(&positions);
    }
    dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;
    dev.check_paged_kv_metadata(
        indptr_buf.va(),
        tbl_buf.va(),
        last_buf.va(),
        1,
        nblocks as u32,
        nblocks as u32,
        bs as u32,
    )?;
    dev.arm_kv_append_paged(
        kcache_buf.va(),
        vcache_buf.va(),
        k_buf.va(),
        v_buf.va(),
        tbl_buf.va(),
        indptr_buf.va(),
        last_buf.va(),
        batch_buf.va(),
        pos_buf.va(),
        n,
        1,
        nblocks as u32,
        nblocks as u32,
        1,
        bs as u32,
        d,
    )?;
    dev.wait(Duration::from_secs(10))?;
    dev.chain_next();
    dev.arm_attn_decode_split_paged(
        q_buf.va(),
        kcache_buf.va(),
        vcache_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
    )?;

    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut scores = vec![0f64; nu];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..nu {
        let mut s = 0f64;
        for dd in 0..du {
            s += qf[dd] * f16_to_f32(k16[i * du + dd]) as f64;
        }
        scores[i] = s * scale as f64;
        mx = mx.max(scores[i]);
    }
    let mut z = 0f64;
    for s in &mut scores {
        *s = (*s - mx).exp();
        z += *s;
    }
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for dd in 0..du {
        let mut e = 0f64;
        for i in 0..nu {
            e += scores[i] / z * f16_to_f32(v16[i * du + dd]) as f64;
        }
        max_rel = max_rel.max((got[dd] as f64 - e).abs() / (e.abs() + 1e-3));
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "append-fed paged decode N={n} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!(
        "  attn-paged+append: decode N={n} D={d} block={bs} ({nblocks} appended pages) on node {node_id} - correct (max rel err {max_rel:.2e})"
    );
    Ok(())
}

pub fn check_attn_decode_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d > HEAD_DIM_MAX {
        return Err(anyhow!("head dim {d} > {HEAD_DIM_MAX}"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 7))).collect();
    let k16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 1))).collect();
    let v16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 2))).collect();

    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du * 2)?;
    let mut v_buf = dev.alloc_device(nu * du * 2)?;
    // partials live in VRAM: the split writes them and the combine reads them
    // scattered across all splits — host memory would force PCIe round-trips.
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let mut o_buf = dev.alloc(du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&k16);
        v_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&v16);
    }
    let (qv, kv, vv, pv, iv, ov) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );

    dev.chain_next();
    dev.arm_attn_decode_split(qv, kv, vv, pv, n, d, scale, num_splits)?;
    combine_decode(dev, pv, iv, ov, d, num_splits)?;

    // f64 reference (direct softmax).
    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut scores = vec![0f64; nu];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..nu {
        let mut s = 0f64;
        for dd in 0..du {
            s += qf[dd] * f16_to_f32(k16[i * du + dd]) as f64;
        }
        scores[i] = s * scale as f64;
        mx = mx.max(scores[i]);
    }
    let mut z = 0f64;
    for s in &mut scores {
        *s = (*s - mx).exp();
        z += *s;
    }
    let mut expect = vec![0f64; du];
    for i in 0..nu {
        let w = scores[i] / z;
        for dd in 0..du {
            expect[dd] += w * f16_to_f32(v16[i * du + dd]) as f64;
        }
    }
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for dd in 0..du {
        let rel = (got[dd] as f64 - expect[dd]).abs() / (expect[dd].abs() + 1e-3);
        max_rel = max_rel.max(rel);
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode N={n} D={d} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn: decode N={n} D={d} ({num_splits} splits) on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Verify paged FP16 decode: KV stored in physical blocks behind a SHUFFLED
/// block table (physical layout != logical). Proves the block-table indirection
/// (vLLM/SGLang paged attention) works on the raw KFD/AQL path.
pub fn check_attn_decode_paged_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("paged path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let bs = 16usize; // tokens per block
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 7))).collect();
    let k16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 1))).collect();
    let v16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 2))).collect();
    // Shuffled block table: logical block i -> physical block (nblocks-1-i).
    let table: Vec<u32> = (0..nblocks).map(|i| (nblocks - 1 - i) as u32).collect();
    // Scatter logical blocks into physical positions per the table.
    let mut kphys = vec![0u16; nu * du];
    let mut vphys = vec![0u16; nu * du];
    for lb in 0..nblocks {
        let pb = table[lb] as usize;
        for j in 0..bs * du {
            kphys[pb * bs * du + j] = k16[lb * bs * du + j];
            vphys[pb * bs * du + j] = v16[lb * bs * du + j];
        }
    }

    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du * 2)?;
    let mut v_buf = dev.alloc_device(nu * du * 2)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let mut o_buf = dev.alloc(du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&kphys);
        v_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&vphys);
        tbl_buf.as_mut_slice_of::<u32>()[..nblocks].copy_from_slice(&table);
    }
    let table_check = dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;
    let meta_check = check_flashinfer_paged_metadata_guard(dev, &table, nblocks, bs)?;
    let mut bad_tbl_buf = dev.alloc_device(nblocks * 4)?;
    unsafe {
        let bad = bad_tbl_buf.as_mut_slice_of::<u32>();
        bad[..nblocks].copy_from_slice(&table);
        bad[0] = nblocks as u32;
    }
    let bad_err = dev
        .check_paged_block_table(bad_tbl_buf.va(), nblocks as u32, nblocks as u32)
        .expect_err("paged block-table guard must reject out-of-range physical blocks");
    if !bad_err.to_string().contains("out-of-range") {
        return Err(anyhow!(
            "unexpected paged block-table guard error: {bad_err:#}"
        ));
    }
    dev.chain_next();
    dev.arm_attn_decode_split_paged(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        bad_tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
    )?;
    dev.chain_next();
    dev.arm_attn_decode_split_paged(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
    )?;

    // Reference over the LOGICAL K/V (the kernel must reconstruct this order).
    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut scores = vec![0f64; nu];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..nu {
        let mut s = 0f64;
        for dd in 0..du {
            s += qf[dd] * f16_to_f32(k16[i * du + dd]) as f64;
        }
        scores[i] = s * scale as f64;
        mx = mx.max(scores[i]);
    }
    let mut z = 0f64;
    for s in &mut scores {
        *s = (*s - mx).exp();
        z += *s;
    }
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for dd in 0..du {
        let mut e = 0f64;
        for i in 0..nu {
            e += scores[i] / z * f16_to_f32(v16[i * du + dd]) as f64;
        }
        max_rel = max_rel.max((got[dd] as f64 - e).abs() / (e.abs() + 1e-3));
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode-paged N={n} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-paged: decode N={n} D={d} block={bs} (shuffled table, {nblocks} blocks, max page {}, batched max page {}) on node {node_id}: correct (max rel err {max_rel:.2e})", table_check.max_entry, meta_check.max_index);
    Ok(())
}

fn check_flashinfer_paged_metadata_guard(
    dev: &mut GpuDevice,
    indices: &[u32],
    nblocks: usize,
    page_size: usize,
) -> Result<crate::gpu::PagedKvMetadataCheck> {
    if nblocks < 8 {
        return Err(anyhow!(
            "batched paged KV metadata guard requires at least 8 blocks"
        ));
    }
    let batch = 3usize;
    let split0 = nblocks / 4;
    let split1 = nblocks / 2;
    let indptr = vec![0u32, split0 as u32, split1 as u32, nblocks as u32];
    let last = vec![
        page_size as u32,
        (page_size as u32).min(7).max(1),
        page_size as u32,
    ];

    let mut indptr_buf = dev.alloc_device(indptr.len() * 4)?;
    let mut indices_buf = dev.alloc_device(indices.len() * 4)?;
    let mut last_buf = dev.alloc_device(last.len() * 4)?;
    unsafe {
        indptr_buf.as_mut_slice_of::<u32>()[..indptr.len()].copy_from_slice(&indptr);
        indices_buf.as_mut_slice_of::<u32>()[..indices.len()].copy_from_slice(indices);
        last_buf.as_mut_slice_of::<u32>()[..last.len()].copy_from_slice(&last);
    }
    let ok = dev.check_paged_kv_metadata(
        indptr_buf.va(),
        indices_buf.va(),
        last_buf.va(),
        batch as u32,
        nblocks as u32,
        nblocks as u32,
        page_size as u32,
    )?;

    let mut bad_last_buf = dev.alloc_device(last.len() * 4)?;
    unsafe {
        let bad = bad_last_buf.as_mut_slice_of::<u32>();
        bad[..last.len()].copy_from_slice(&last);
        bad[1] = 0;
    }
    let bad_last_err = dev
        .check_paged_kv_metadata(
            indptr_buf.va(),
            indices_buf.va(),
            bad_last_buf.va(),
            batch as u32,
            nblocks as u32,
            nblocks as u32,
            page_size as u32,
        )
        .expect_err("paged KV metadata guard must reject last_page_len=0");
    if !bad_last_err.to_string().contains("last_page_len") {
        return Err(anyhow!(
            "unexpected paged KV last-page guard error: {bad_last_err:#}"
        ));
    }

    let mut bad_indptr_buf = dev.alloc_device(indptr.len() * 4)?;
    unsafe {
        let bad = bad_indptr_buf.as_mut_slice_of::<u32>();
        bad[..indptr.len()].copy_from_slice(&indptr);
        bad[batch] = nblocks as u32 + 1;
    }
    let bad_indptr_err = dev
        .check_paged_kv_metadata(
            bad_indptr_buf.va(),
            indices_buf.va(),
            last_buf.va(),
            batch as u32,
            nblocks as u32,
            nblocks as u32,
            page_size as u32,
        )
        .expect_err("paged KV metadata guard must reject indptr total mismatch");
    if !bad_indptr_err.to_string().contains("indptr") {
        return Err(anyhow!(
            "unexpected paged KV indptr guard error: {bad_indptr_err:#}"
        ));
    }

    let mut bad_indices_buf = dev.alloc_device(indices.len() * 4)?;
    unsafe {
        let bad = bad_indices_buf.as_mut_slice_of::<u32>();
        bad[..indices.len()].copy_from_slice(indices);
        bad[0] = nblocks as u32;
    }
    let bad_index_err = dev
        .check_paged_kv_metadata(
            indptr_buf.va(),
            bad_indices_buf.va(),
            last_buf.va(),
            batch as u32,
            nblocks as u32,
            nblocks as u32,
            page_size as u32,
        )
        .expect_err("paged KV metadata guard must reject out-of-range page indices");
    if !bad_index_err.to_string().contains("page index") {
        return Err(anyhow!(
            "unexpected paged KV page-index guard error: {bad_index_err:#}"
        ));
    }

    Ok(ok)
}

/// Verify the capstone: paged + GQA + FP4. G query heads share one FP4 KV head
/// stored in shuffled physical blocks behind a block table. The real Qwen 1M
/// decode primitive, end to end on the raw KFD/AQL path.
pub fn check_attn_decode_fp4_gqa_paged_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("capstone requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let mut q16 = vec![0u16; GQA_G * du];
    for h in 0..GQA_G {
        for i in 0..du {
            q16[h * du + i] = f32_to_f16(gen(i, 7 + h));
        }
    }
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k4, ksc) = quantize_fp4_blocks(&kf, nu); // logical
    let (v4, vsc) = quantize_fp4_blocks(&vf, nu);
    // Shuffled block table + scatter logical -> physical (KV bytes AND scales).
    let table: Vec<u32> = (0..nblocks).map(|i| (nblocks - 1 - i) as u32).collect();
    let mut k4p = vec![0u8; nu * 64];
    let mut v4p = vec![0u8; nu * 64];
    let mut kscp = vec![127u8; nu * 4];
    let mut vscp = vec![127u8; nu * 4];
    for i in 0..nu {
        let lb = i / bs;
        let row = table[lb] as usize * bs + (i - lb * bs);
        k4p[row * 64..row * 64 + 64].copy_from_slice(&k4[i * 64..i * 64 + 64]);
        v4p[row * 64..row * 64 + 64].copy_from_slice(&v4[i * 64..i * 64 + 64]);
        kscp[row * 4..row * 4 + 4].copy_from_slice(&ksc[i * 4..i * 4 + 4]);
        vscp[row * 4..row * 4 + 4].copy_from_slice(&vsc[i * 4..i * 4 + 4]);
    }

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut v_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let mut o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&k4p);
        v_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&v4p);
        ksc_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&kscp);
        vsc_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&vscp);
        tbl_buf.as_mut_slice_of::<u32>()[..nblocks].copy_from_slice(&table);
    }
    let table_check = dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_gqa_paged(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode_gqa(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
        GQA_G as u32,
    )?;

    // Reference per head over the LOGICAL decoded FP4.
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for h in 0..GQA_G {
        let qf: Vec<f64> = (0..du)
            .map(|i| f16_to_f32(q16[h * du + i]) as f64)
            .collect();
        let mut sc = vec![0f64; nu];
        let mut mx = f64::NEG_INFINITY;
        for i in 0..nu {
            let mut s = 0f64;
            for dd in 0..du {
                s += qf[dd] * fp4_decode(&k4, &ksc, i, dd) as f64;
            }
            sc[i] = s * scale as f64;
            mx = mx.max(sc[i]);
        }
        let mut z = 0f64;
        for i in 0..nu {
            sc[i] = (sc[i] - mx).exp();
            z += sc[i];
        }
        for dd in 0..du {
            let mut e = 0f64;
            for i in 0..nu {
                e += sc[i] / z * fp4_decode(&v4, &vsc, i, dd) as f64;
            }
            max_rel = max_rel.max((got[h * du + dd] as f64 - e).abs() / (e.abs() + 1e-3));
        }
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "capstone N={n} G={GQA_G} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-CAPSTONE (paged+GQA+FP4): N={n} G={GQA_G} block={bs} (shuffled, {nblocks} blk, max page {}) on node {node_id}: correct (max rel err {max_rel:.2e})", table_check.max_entry);
    Ok(())
}

/// Verify the compressed serving loop: append already-quantized FP4 K/V rows
/// into physical pages, then run the paged+GQA+FP4 capstone decoder.
pub fn check_attn_decode_fp4_gqa_paged_after_append_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("append-fed FP4 capstone requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let mut q16 = vec![0u16; GQA_G * du];
    for h in 0..GQA_G {
        for i in 0..du {
            q16[h * du + i] = f32_to_f16(gen(i, 47 + h));
        }
    }
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 48)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 49)).collect();
    let (k4, ksc) = quantize_fp4_blocks(&kf, nu);
    let (v4, vsc) = quantize_fp4_blocks(&vf, nu);
    let table: Vec<u32> = (0..nblocks)
        .map(|i| ((i * 17 + 11) % nblocks) as u32)
        .collect();
    let indptr = vec![0u32, nblocks as u32];
    let last_page_len = vec![bs as u32];
    let batch_indices = vec![0u32; nu];
    let positions: Vec<u32> = (0..nu).map(|i| i as u32).collect();

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut v_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let mut ksrc_buf = dev.alloc_device(nu * 64)?;
    let mut vsrc_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_src_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_src_buf = dev.alloc_device(nu * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let mut indptr_buf = dev.alloc_device(indptr.len() * 4)?;
    let mut last_buf = dev.alloc_device(last_page_len.len() * 4)?;
    let mut batch_buf = dev.alloc_device(batch_indices.len() * 4)?;
    let mut pos_buf = dev.alloc_device(positions.len() * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let mut o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * 64].fill(0xa5);
        v_buf.as_mut_slice_of::<u8>()[..nu * 64].fill(0x5a);
        ksc_buf.as_mut_slice_of::<u8>()[..nu * 4].fill(0);
        vsc_buf.as_mut_slice_of::<u8>()[..nu * 4].fill(0);
        ksrc_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&k4);
        vsrc_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&v4);
        ksc_src_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&ksc);
        vsc_src_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&vsc);
        tbl_buf.as_mut_slice_of::<u32>()[..nblocks].copy_from_slice(&table);
        indptr_buf.as_mut_slice_of::<u32>()[..indptr.len()].copy_from_slice(&indptr);
        last_buf.as_mut_slice_of::<u32>()[..last_page_len.len()].copy_from_slice(&last_page_len);
        batch_buf.as_mut_slice_of::<u32>()[..batch_indices.len()].copy_from_slice(&batch_indices);
        pos_buf.as_mut_slice_of::<u32>()[..positions.len()].copy_from_slice(&positions);
    }
    dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;
    dev.check_paged_kv_metadata(
        indptr_buf.va(),
        tbl_buf.va(),
        last_buf.va(),
        1,
        nblocks as u32,
        nblocks as u32,
        bs as u32,
    )?;
    dev.arm_kv_append_paged_fp4(
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        ksrc_buf.va(),
        vsrc_buf.va(),
        ksc_src_buf.va(),
        vsc_src_buf.va(),
        tbl_buf.va(),
        indptr_buf.va(),
        last_buf.va(),
        batch_buf.va(),
        pos_buf.va(),
        n,
        1,
        nblocks as u32,
        nblocks as u32,
        bs as u32,
    )?;
    dev.wait(Duration::from_secs(10))?;
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_gqa_paged(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode_gqa(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
        GQA_G as u32,
    )?;

    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for h in 0..GQA_G {
        let qf: Vec<f64> = (0..du)
            .map(|i| f16_to_f32(q16[h * du + i]) as f64)
            .collect();
        let mut sc = vec![0f64; nu];
        let mut mx = f64::NEG_INFINITY;
        for i in 0..nu {
            let mut s = 0f64;
            for dd in 0..du {
                s += qf[dd] * fp4_decode(&k4, &ksc, i, dd) as f64;
            }
            sc[i] = s * scale as f64;
            mx = mx.max(sc[i]);
        }
        let mut z = 0f64;
        for i in 0..nu {
            sc[i] = (sc[i] - mx).exp();
            z += sc[i];
        }
        for dd in 0..du {
            let mut e = 0f64;
            for i in 0..nu {
                e += sc[i] / z * fp4_decode(&v4, &vsc, i, dd) as f64;
            }
            max_rel = max_rel.max((got[h * du + dd] as f64 - e).abs() / (e.abs() + 1e-3));
        }
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "append-fed FP4 capstone N={n} G={GQA_G} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-CAPSTONE+append (paged+GQA+FP4): N={n} G={GQA_G} block={bs} ({nblocks} appended FP4 rows) on node {node_id} - correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Benchmark the capstone (paged + GQA + FP4). Identity block table (perf only).
/// Returns (µs per head-group decode, µs per head).
pub fn bench_attn_decode_fp4_gqa_paged_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let bs = 16usize;
    let nblocks = nu / bs;
    let q_buf = dev.alloc(GQA_G * du * 2)?;
    let k_buf = dev.alloc_device(nu * 64)?;
    let v_buf = dev.alloc_device(nu * 64)?;
    let ksc_buf = dev.alloc_device(nu * 4)?;
    let vsc_buf = dev.alloc_device(nu * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        // Identity block table (we measure compute, not a particular mapping).
        let t = tbl_buf.as_mut_slice_of::<u32>();
        for i in 0..nblocks {
            t[i] = i as u32;
        }
    }
    let (qv, kv, vv, kscv, vscv, tblv, pv0, iv, ov0) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );
    dev.check_paged_block_table(tblv, nblocks as u32, nblocks as u32)?;
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.chain_next();
        dev.arm_attn_decode_split_fp4_gqa_paged(
            qv,
            kv,
            vv,
            kscv,
            vscv,
            tblv,
            bs as u32,
            nblocks as u32,
            pv0,
            n,
            d,
            scale,
            num_splits,
        )?;
        combine_decode_gqa(dev, pv0, iv, ov0, d, num_splits, GQA_G as u32)?;
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
    Ok((t * 1e6, t * 1e6 / GQA_G as f64))
}

/// Gate the production-shaped paged+GQA+FP4 decode path against batch-dependent
/// reduction drift. This uses the graph-friendly grouped-meta split kernel with
/// a fixed token split size: request 0 must produce bitwise-identical f32 output
/// when run alone and when run in a larger batch of duplicate logical requests.
pub fn check_attn_decode_fp4_gqa_paged_batch_invariance_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    batch_groups: u32,
    fixed_split_tokens: u32,
    iters: usize,
) -> Result<()> {
    if d != 128 {
        return Err(anyhow!(
            "deterministic capstone gate requires D=128 (got {d})"
        ));
    }
    if n == 0 {
        return Err(anyhow!("deterministic capstone gate requires N > 0"));
    }
    if batch_groups < 2 {
        return Err(anyhow!(
            "deterministic capstone gate requires batch_groups >= 2 (got {batch_groups})"
        ));
    }
    if fixed_split_tokens == 0 {
        return Err(anyhow!(
            "deterministic capstone gate requires fixed_split_tokens > 0"
        ));
    }

    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;
    let groups = batch_groups as usize;
    let heads = groups
        .checked_mul(GQA_G)
        .ok_or_else(|| anyhow!("deterministic capstone gate head count overflow"))?;
    let kv_heads = groups;
    let rows_per_head = nu;
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = n.div_ceil(fixed_split_tokens).max(1);
    let ps = du + 2;
    let inter_splits = combine_groups(num_splits).max(1) as usize;

    let mut q_buf = dev.alloc(heads * du * 2)?;
    let mut k_buf = dev.alloc_device(kv_heads * rows_per_head * 64)?;
    let mut v_buf = dev.alloc_device(kv_heads * rows_per_head * 64)?;
    let mut ksc_buf = dev.alloc_device(kv_heads * rows_per_head * 4)?;
    let mut vsc_buf = dev.alloc_device(kv_heads * rows_per_head * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let mut seq_lens_buf = dev.alloc_device(4)?;
    let single_part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let single_inter_buf = dev.alloc_device(GQA_G * inter_splits * ps * 4)?;
    let mut single_o_buf = dev.alloc(GQA_G * du * 4)?;
    let batch_part_buf = dev.alloc_device(heads * num_splits as usize * ps * 4)?;
    let batch_inter_buf = dev.alloc_device(heads * inter_splits * ps * 4)?;
    let mut batch_o_buf = dev.alloc(heads * du * 4)?;

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let fp4_byte = |row: usize, byte: usize, seed: usize| -> u8 {
        let x = row
            .wrapping_mul(131)
            .wrapping_add(byte.wrapping_mul(17))
            .wrapping_add(seed.wrapping_mul(29));
        ((x ^ (x >> 7) ^ (x >> 13)) & 0xff) as u8
    };
    let scale_byte =
        |row: usize, lane: usize, seed: usize| -> u8 { 126 + ((row + lane * 3 + seed) % 3) as u8 };

    unsafe {
        let q = q_buf.as_mut_slice_of::<u16>();
        for group in 0..groups {
            for h in 0..GQA_G {
                for i in 0..du {
                    q[(group * GQA_G + h) * du + i] = f32_to_f16(gen(i, 71 + h));
                }
            }
        }

        let table = tbl_buf.as_mut_slice_of::<u32>();
        for logical_block in 0..nblocks {
            table[logical_block] = ((logical_block * 17 + 11) % nblocks) as u32;
        }

        let k = k_buf.as_mut_slice_of::<u8>();
        let v = v_buf.as_mut_slice_of::<u8>();
        let ks = ksc_buf.as_mut_slice_of::<u8>();
        let vs = vsc_buf.as_mut_slice_of::<u8>();
        for kvh in 0..kv_heads {
            for logical_block in 0..nblocks {
                let physical_block = table[logical_block] as usize;
                for offset in 0..bs {
                    let logical_row = logical_block * bs + offset;
                    let physical_row = physical_block * bs + offset;
                    let row_base = (kvh * rows_per_head + physical_row) * 64;
                    for byte in 0..64 {
                        k[row_base + byte] = fp4_byte(logical_row, byte, 3);
                        v[row_base + byte] = fp4_byte(logical_row, byte, 5);
                    }
                    let scale_base = (kvh * rows_per_head + physical_row) * 4;
                    for lane in 0..4 {
                        ks[scale_base + lane] = scale_byte(logical_row, lane, 7);
                        vs[scale_base + lane] = scale_byte(logical_row, lane, 11);
                    }
                }
            }
        }
        seq_lens_buf.as_mut_slice_of::<u32>()[0] = n;
    }

    dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;

    let qv = q_buf.va();
    let kv = k_buf.va();
    let vv = v_buf.va();
    let ksv = ksc_buf.va();
    let vsv = vsc_buf.va();
    let tblv = tbl_buf.va();
    let seqv = seq_lens_buf.va();
    let single_pv = single_part_buf.va();
    let single_iv = single_inter_buf.va();
    let single_ov = single_o_buf.va();
    let batch_pv = batch_part_buf.va();
    let batch_iv = batch_inter_buf.va();
    let batch_ov = batch_o_buf.va();
    let run = |dev: &mut GpuDevice,
               group_count: u32,
               partials_va: u64,
               inter_va: u64,
               output_va: u64|
     -> Result<Duration> {
        let started = Instant::now();
        dev.chain_next();
        dev.arm_attn_decode_split_fp4_gqa_paged_groups_meta(
            qv,
            kv,
            vv,
            ksv,
            vsv,
            tblv,
            bs as u32,
            nblocks as u32,
            partials_va,
            seqv,
            n,
            d,
            scale,
            num_splits,
            group_count,
            GQA_G as u32,
            rows_per_head as u32,
        )?;
        combine_decode_gqa(
            dev,
            partials_va,
            inter_va,
            output_va,
            d,
            num_splits,
            group_count * GQA_G as u32,
        )?;
        Ok(started.elapsed())
    };

    let rounds = iters.max(1);
    let mut single_total = Duration::ZERO;
    let mut batch_total = Duration::ZERO;
    let mut reference_bits: Option<Vec<u32>> = None;
    for round in 0..rounds {
        unsafe {
            single_o_buf.as_mut_slice_of::<f32>()[..GQA_G * du].fill(-777.0);
            batch_o_buf.as_mut_slice_of::<f32>()[..heads * du].fill(-777.0);
        }
        single_total += run(dev, 1, single_pv, single_iv, single_ov)?;
        batch_total += run(dev, batch_groups, batch_pv, batch_iv, batch_ov)?;

        let single_bits: Vec<u32> = unsafe {
            single_o_buf.as_mut_slice_of::<f32>()[..GQA_G * du]
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let batch_bits: Vec<u32> = unsafe {
            batch_o_buf.as_mut_slice_of::<f32>()[..heads * du]
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for (idx, (&solo, &batched)) in single_bits
            .iter()
            .zip(batch_bits[..GQA_G * du].iter())
            .enumerate()
        {
            if solo != batched {
                return Err(anyhow!(
                    "deterministic capstone mismatch round {round} request0 head {} dim {}: single=0x{solo:08x} batched=0x{batched:08x}",
                    idx / du,
                    idx % du
                ));
            }
        }
        for group in 1..groups {
            let base = group * GQA_G * du;
            for idx in 0..GQA_G * du {
                let ref_bits = batch_bits[idx];
                let got = batch_bits[base + idx];
                if got != ref_bits {
                    return Err(anyhow!(
                        "deterministic capstone duplicate-group mismatch round {round} group {group} head {} dim {}: group0=0x{ref_bits:08x} group{group}=0x{got:08x}",
                        idx / du,
                        idx % du
                    ));
                }
            }
        }
        if let Some(reference) = &reference_bits {
            for (idx, (&want, &got)) in reference.iter().zip(single_bits.iter()).enumerate() {
                if want != got {
                    return Err(anyhow!(
                        "deterministic capstone run-to-run mismatch round {round} head {} dim {}: round0=0x{want:08x} round{round}=0x{got:08x}",
                        idx / du,
                        idx % du
                    ));
                }
            }
        } else {
            reference_bits = Some(single_bits);
        }
    }

    let single_avg_us = single_total.as_secs_f64() * 1.0e6 / rounds as f64;
    let batch_avg_us = batch_total.as_secs_f64() * 1.0e6 / rounds as f64;
    println!(
        "  attn-CAPSTONE deterministic fixed-split: N={n} D={d} G={GQA_G} groups 1->{batch_groups} fixed_split_tokens={fixed_split_tokens} splits={num_splits} on node {node_id} - bitwise batch-invariant over {rounds} rounds, single avg {single_avg_us:.1} us, batch avg {batch_avg_us:.1} us ({:.1} us/group)",
        batch_avg_us / batch_groups as f64
    );
    Ok(())
}

/// Profile the capstone decode pipeline by phase. This keeps the same
/// paged+GQA+FP4 split and GQA merge kernels as the production benchmark, but
/// times the split, first-stage reduce, final combine, and fully chained path
/// separately so merge/fusion work is driven by measured cost.
pub fn bench_attn_decode_fp4_gqa_paged_stages_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    splits: Option<u32>,
    fixed_split_tokens: Option<u32>,
    warmup_iters: usize,
    iters: usize,
) -> Result<(f64, f64, f64, f64)> {
    if d != 128 {
        return Err(anyhow!("capstone stage profile requires D=128 (got {d})"));
    }
    if n == 0 {
        return Err(anyhow!("capstone stage profile requires N > 0"));
    }
    if splits.is_some() && fixed_split_tokens.is_some() {
        return Err(anyhow!(
            "capstone stage profile accepts either splits or fixed_split_tokens, not both"
        ));
    }

    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = if let Some(split_count) = splits {
        split_count
    } else if let Some(split_tokens) = fixed_split_tokens {
        if split_tokens == 0 {
            return Err(anyhow!(
                "capstone stage profile requires fixed_split_tokens > 0"
            ));
        }
        n.div_ceil(split_tokens).max(1)
    } else {
        split_count(n)
    };
    if num_splits == 0 {
        return Err(anyhow!("capstone stage profile requires splits > 0"));
    }
    let split_source = if splits.is_some() {
        "override"
    } else if fixed_split_tokens.is_some() {
        "fixed_split_tokens"
    } else {
        "heuristic"
    };
    let split_token_note = fixed_split_tokens
        .map(|value| format!(" fixed_split_tokens={value}"))
        .unwrap_or_default();
    let reduce_groups = combine_groups(num_splits);
    let reduce_group_size = if reduce_groups == 0 {
        0
    } else {
        num_splits.div_ceil(reduce_groups)
    };
    let ps = du + 2;
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut v_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf = dev.alloc_device(GQA_G * (reduce_groups.max(1) as usize) * ps * 4)?;
    let o_buf = dev.alloc(GQA_G * du * 4)?;

    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].fill(f32_to_f16(0.03125));
        k_buf.as_mut_slice_of::<u8>()[..nu * 64].fill(0x11);
        v_buf.as_mut_slice_of::<u8>()[..nu * 64].fill(0x22);
        ksc_buf.as_mut_slice_of::<u8>()[..nu * 4].fill(127);
        vsc_buf.as_mut_slice_of::<u8>()[..nu * 4].fill(127);
        let table = tbl_buf.as_mut_slice_of::<u32>();
        for (i, slot) in table.iter_mut().enumerate().take(nblocks) {
            *slot = i as u32;
        }
    }
    dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;

    let qv = q_buf.va();
    let kv = k_buf.va();
    let vv = v_buf.va();
    let ksv = ksc_buf.va();
    let vsv = vsc_buf.va();
    let tblv = tbl_buf.va();
    let pv = part_buf.va();
    let iv = inter_buf.va();
    let ov = o_buf.va();
    let heads = GQA_G as u32;

    let run_split = |dev: &mut GpuDevice| -> Result<Duration> {
        let started = Instant::now();
        dev.arm_attn_decode_split_fp4_gqa_paged(
            qv,
            kv,
            vv,
            ksv,
            vsv,
            tblv,
            bs as u32,
            nblocks as u32,
            pv,
            n,
            d,
            scale,
            num_splits,
        )?;
        dev.wait(Duration::from_secs(10))?;
        Ok(started.elapsed())
    };
    let run_reduce = |dev: &mut GpuDevice| -> Result<Duration> {
        if reduce_groups == 0 {
            return Ok(Duration::ZERO);
        }
        let started = Instant::now();
        dev.arm_attn_decode_reduce_partials_gqa(
            pv,
            iv,
            num_splits,
            d,
            reduce_group_size,
            reduce_groups,
            heads,
        )?;
        dev.wait(Duration::from_secs(10))?;
        Ok(started.elapsed())
    };
    let run_final = |dev: &mut GpuDevice| -> Result<Duration> {
        let started = Instant::now();
        if reduce_groups == 0 {
            dev.arm_attn_decode_combine_gqa(pv, ov, d, num_splits, heads)?;
        } else {
            dev.arm_attn_decode_combine_gqa(iv, ov, d, reduce_groups, heads)?;
        }
        dev.wait(Duration::from_secs(10))?;
        Ok(started.elapsed())
    };
    let run_chained = |dev: &mut GpuDevice| -> Result<Duration> {
        let started = Instant::now();
        dev.chain_next();
        dev.arm_attn_decode_split_fp4_gqa_paged(
            qv,
            kv,
            vv,
            ksv,
            vsv,
            tblv,
            bs as u32,
            nblocks as u32,
            pv,
            n,
            d,
            scale,
            num_splits,
        )?;
        combine_decode_gqa(dev, pv, iv, ov, d, num_splits, heads)?;
        Ok(started.elapsed())
    };

    for _ in 0..warmup_iters {
        run_chained(dev)?;
    }

    let rounds = iters.max(1);
    let mut split_total = Duration::ZERO;
    let mut reduce_total = Duration::ZERO;
    let mut final_total = Duration::ZERO;
    let mut chained_total = Duration::ZERO;

    for _ in 0..rounds {
        split_total += run_split(dev)?;
    }
    for _ in 0..rounds {
        reduce_total += run_reduce(dev)?;
    }
    for _ in 0..rounds {
        final_total += run_final(dev)?;
    }
    for _ in 0..rounds {
        chained_total += run_chained(dev)?;
    }

    let split_us = split_total.as_secs_f64() * 1.0e6 / rounds as f64;
    let reduce_us = reduce_total.as_secs_f64() * 1.0e6 / rounds as f64;
    let final_us = final_total.as_secs_f64() * 1.0e6 / rounds as f64;
    let chained_us = chained_total.as_secs_f64() * 1.0e6 / rounds as f64;
    let staged_sum = split_us + reduce_us + final_us;
    println!(
        "  attn-CAPSTONE-stage-profile: N={n} D={d} G={GQA_G} block={bs} splits={num_splits} split_source={split_source} reduce_groups={reduce_groups} reduce_group_size={reduce_group_size}{split_token_note} warmup_iters={warmup_iters} on node {node_id} over {rounds} rounds - split {split_us:.1} us, reduce {reduce_us:.1} us, final {final_us:.1} us, staged_sum {staged_sum:.1} us, chained_full {chained_us:.1} us"
    );
    Ok((split_us, reduce_us, final_us, chained_us))
}

// ---- OCP E4M3 (e4m3fn) codec — must match the kernel's e4m3_to_f32 exactly. ----

/// Decode an E4M3 byte to f32 (bit-identical to the kernel decoder).
pub(crate) fn e4m3_to_f32(b: u8) -> f32 {
    let e = ((b >> 3) & 0xF) as u32;
    let m = (b & 0x7) as u32;
    let v = if e == 0 {
        m as f32 * (1.0 / 512.0) // subnormal: m * 2^-9
    } else {
        f32::from_bits(((e + 120) << 23) | (m << 20))
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// Relative L2 error of round-tripping `x` through E2M1 with a per-`block` E8M0
/// (power-of-two) scale — the FP4 KV format.
fn e2m1_block_rel_l2(x: &[f32], block: usize) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for chunk in x.chunks(block) {
        let maxabs = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let e = if maxabs > 0.0 {
            ((maxabs / 6.0).log2().ceil() as i32).clamp(-127, 127)
        } else {
            0
        };
        let scale = f32::from_bits(((e + 127) as u32) << 23);
        for &v in chunk {
            let q = e2m1_to_f32(f32_to_e2m1(v / scale)) * scale;
            num += (q as f64 - v as f64).powi(2);
            den += (v as f64).powi(2);
        }
    }
    (num / den.max(1e-30)).sqrt()
}

/// Relative L2 error of round-tripping `x` through E2M1 with a per-`block` E4M3
/// scale (max/6 quantized to E4M3, not power-of-two) — the NVFP4 format. The
/// finer scale preserves the block max better than E8M0.
fn nvfp4_block_rel_l2(x: &[f32], block: usize) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for chunk in x.chunks(block) {
        let maxabs = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        // NVFP4 scale: max/6 quantized to E4M3 (round-nearest, the standard).
        let raw = if maxabs > 0.0 { maxabs / 6.0 } else { 1.0 };
        let scale = e4m3_to_f32(f32_to_e4m3(raw)).max(1e-30);
        for &v in chunk {
            let q = e2m1_to_f32(f32_to_e2m1(v / scale)) * scale;
            num += (q as f64 - v as f64).powi(2);
            den += (v as f64).powi(2);
        }
    }
    (num / den.max(1e-30)).sqrt()
}

/// Relative L2 error of round-tripping `x` through E4M3 with a per-`block`
/// (max/448) scale — the FP8 KV format.
fn e4m3_block_rel_l2(x: &[f32], block: usize) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for chunk in x.chunks(block) {
        let maxabs = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if maxabs > 0.0 { maxabs / 448.0 } else { 1.0 };
        for &v in chunk {
            let q = e4m3_to_f32(f32_to_e4m3(v / scale)) * scale;
            num += (q as f64 - v as f64).powi(2);
            den += (v as f64).powi(2);
        }
    }
    (num / den.max(1e-30)).sqrt()
}

/// Characterize KV quantization accuracy on realistic, outlier-bearing data
/// (the real production question for 4-bit KV). Generates head_dim vectors with
/// a Gaussian-ish bulk plus heavy-tailed outliers (as real K/V have) and reports
/// the relative-L2 error of each format. GPU-free (host numerics only).
pub fn characterize_kv_quant(rows: usize, head_dim: usize) {
    // Deterministic pseudo-Gaussian (sum of hashed uniforms) + ~2% outliers ×8.
    let mut x = vec![0f32; rows * head_dim];
    for (i, xi) in x.iter_mut().enumerate() {
        let mut acc = 0f32;
        for k in 0..6 {
            let h = ((i * 2654435761 + k * 40503 + 7) >> 9) & 0xffff;
            acc += h as f32 / 65536.0 - 0.5;
        }
        let mut v = acc / 6f32.sqrt() * 0.5; // ~N(0, small)
        if ((i * 2246822519usize) >> 13).is_multiple_of(50) {
            v *= 8.0; // heavy-tailed outlier
        }
        *xi = v;
    }
    let fp4_b32 = e2m1_block_rel_l2(&x, 32);
    let fp4_b16 = e2m1_block_rel_l2(&x, 16);
    let nvfp4_b16 = nvfp4_block_rel_l2(&x, 16);
    let fp8_tok = e4m3_block_rel_l2(&x, head_dim);
    let fp8_b32 = e4m3_block_rel_l2(&x, 32);
    println!("  KV-quant accuracy (rel L2, realistic Gaussian+2%-outlier data, D={head_dim}):");
    println!(
        "    FP8   E4M3 per-token       : {fp8_tok:.4}   ({:.2} B/elem)",
        1.0 + 4.0 / head_dim as f64
    );
    println!("    FP8   E4M3 block-32        : {fp8_b32:.4}   (1.03 B/elem)");
    println!("    FP4   E2M1 block-32 (E8M0) : {fp4_b32:.4}   (0.53 B/elem)");
    println!("    FP4   E2M1 block-16 (E8M0) : {fp4_b16:.4}   (0.56 B/elem)");
    println!("    NVFP4 E2M1 block-16 (E4M3) : {nvfp4_b16:.4}   (0.56 B/elem)");
}

/// Quantize f32 to E4M3 (round-to-nearest-even, saturate to ±448). 0x7e is the
/// largest finite magnitude; 0x7f is NaN (never produced for finite inputs).
pub(crate) fn f32_to_e4m3(x: f32) -> u8 {
    if !x.is_finite() {
        return 0x7f;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let ax = x.abs();
    if ax == 0.0 {
        return sign;
    }
    if ax >= 448.0 {
        return sign | 0x7e;
    }
    if ax < f32::from_bits((127 - 6) << 23) {
        // subnormal: value = m * 2^-9
        let m = (ax * 512.0).round() as i32;
        if m >= 8 {
            return sign | (1 << 3);
        }
        return sign | (m as u8);
    }
    let bits = ax.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mant = bits & 0x7F_FFFF;
    let m3 = mant >> 20;
    let rbit = (mant >> 19) & 1;
    let sticky = (mant & 0x7_FFFF) != 0;
    let mut m = m3;
    let mut e = exp + 7;
    if rbit == 1 && (sticky || (m3 & 1) == 1) {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }
    if e > 15 || (e == 15 && m > 6) {
        return sign | 0x7e;
    }
    sign | ((e as u8) << 3) | (m as u8)
}

/// Quantize an N×D f32 matrix to E4M3 with one scale per row (token):
/// scale[t] = max_d|X[t][d]| / 448. Returns (bytes, scales).
fn quantize_e4m3_rows(x: &[f32], n: usize, d: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = vec![0u8; n * d];
    let mut scales = vec![1.0f32; n];
    for t in 0..n {
        let row = &x[t * d..t * d + d];
        let maxabs = row.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if maxabs > 0.0 { maxabs / 448.0 } else { 1.0 };
        scales[t] = scale;
        let inv = 1.0 / scale;
        for j in 0..d {
            bytes[t * d + j] = f32_to_e4m3(row[j] * inv);
        }
    }
    (bytes, scales)
}

/// Verify FP8-E4M3 KV decode attention. K/V are quantized per-token to E4M3; the
/// reference is computed from the *decoded* values (so this validates the kernel,
/// not FP8-vs-FP16 — that error is reported separately).
pub fn check_attn_decode_fp8_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("fp8 split path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 7))).collect();
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k8, ksc) = quantize_e4m3_rows(&kf, nu, du);
    let (v8, vsc) = quantize_e4m3_rows(&vf, nu, du);

    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du)?;
    let mut v_buf = dev.alloc_device(nu * du)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let mut o_buf = dev.alloc(du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * du].copy_from_slice(&k8);
        v_buf.as_mut_slice_of::<u8>()[..nu * du].copy_from_slice(&v8);
        ksc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&ksc);
        vsc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&vsc);
    }

    dev.chain_next();
    dev.arm_attn_decode_split_fp8(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
    )?;

    // f64 reference over the DECODED (dequantized) K/V.
    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut scores = vec![0f64; nu];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..nu {
        let mut s = 0f64;
        for dd in 0..du {
            let kdq = e4m3_to_f32(k8[i * du + dd]) as f64 * ksc[i] as f64;
            s += qf[dd] * kdq;
        }
        scores[i] = s * scale as f64;
        mx = mx.max(scores[i]);
    }
    let mut z = 0f64;
    for s in &mut scores {
        *s = (*s - mx).exp();
        z += *s;
    }
    let mut expect = vec![0f64; du];
    for i in 0..nu {
        let w = scores[i] / z;
        for dd in 0..du {
            let vdq = e4m3_to_f32(v8[i * du + dd]) as f64 * vsc[i] as f64;
            expect[dd] += w * vdq;
        }
    }
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for dd in 0..du {
        let rel = (got[dd] as f64 - expect[dd]).abs() / (expect[dd].abs() + 1e-3);
        max_rel = max_rel.max(rel);
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode-fp8 N={n} D={d} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-fp8: decode N={n} D={d} ({num_splits} splits) on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Benchmark FP8-E4M3 KV decode attention; returns (GB/s on FP8 KV bytes, us).
pub fn bench_attn_decode_fp8_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let q_buf = dev.alloc(du * 2)?;
    let k_buf = dev.alloc_device(nu * du)?;
    let v_buf = dev.alloc_device(nu * du)?;
    let ksc_buf = dev.alloc_device(nu * 4)?;
    let vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let o_buf = dev.alloc(du * 4)?;
    let (qv, kv, vv, kscv, vscv, pv, iv, ov) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.chain_next();
        dev.arm_attn_decode_split_fp8(qv, kv, vv, kscv, vscv, pv, n, d, scale, num_splits)?;
        combine_decode(dev, pv, iv, ov, d, num_splits)?;
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
    // FP8 KV bytes read: K and V, each N×D one byte.
    let bytes = 2.0 * nu as f64 * du as f64;
    Ok((bytes / t / 1e9, t * 1e6))
}

// ---- E2M1 (FP4) codec — magnitudes match the hardware cvt (probe-verified). ----
const E2M1_MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e2m1_to_f32(nibble: u8) -> f32 {
    let mag = E2M1_MAG[(nibble & 7) as usize];
    if nibble & 8 != 0 {
        -mag
    } else {
        mag
    }
}

/// Quantize f32 to an E2M1 nibble (sign + nearest magnitude index).
fn f32_to_e2m1(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 8u8 } else { 0 };
    let a = x.abs();
    // round to nearest of {0,.5,1,1.5,2,3,4,6} via midpoints
    let idx = if a < 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a < 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a < 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a < 5.0 {
        6
    } else {
        7
    };
    sign | idx
}

/// Quantize an N×128 f32 matrix to FP4 (E2M1) with a per-block-32 f32 scale
/// (4 blocks/row). Returns (packed bytes [N*64], scales [N*4]). Byte d/2 holds
/// dim d in the low nibble if d even, high nibble if odd (matches the hardware
/// cvt: sel k → dims 2k(low),2k+1(high)).
pub(crate) fn quantize_fp4_blocks(x: &[f32], n: usize) -> (Vec<u8>, Vec<u8>) {
    let mut bytes = vec![0u8; n * 64];
    let mut scales = vec![127u8; n * 4]; // E8M0: byte = e + 127 (so 2^0 = 1)
    for t in 0..n {
        for bl in 0..4 {
            let base = t * 128 + bl * 32;
            let maxabs = (0..32).fold(0.0f32, |a, j| a.max(x[base + j].abs()));
            // E8M0 scale (the cvt uses only the exponent ⇒ power of two). e =
            // ceil(log2(maxabs/6)) guarantees maxabs/2^e <= 6 (E2M1 max). Stored
            // as the biased exponent byte e+127.
            let e = if maxabs > 0.0 {
                ((maxabs / 6.0).log2().ceil() as i32).clamp(-127, 127)
            } else {
                0
            };
            scales[t * 4 + bl] = (e + 127) as u8;
            let inv = f32::from_bits(((127 - e) as u32) << 23); // 1 / 2^e = 2^-e
            for j in 0..32 {
                let d = bl * 32 + j;
                let nib = f32_to_e2m1(x[t * 128 + d] * inv);
                bytes[t * 64 + d / 2] |= nib << (4 * (d & 1));
            }
        }
    }
    (bytes, scales)
}

/// Decode FP4 element `d` of token `t` from packed bytes + E8M0 block scales.
pub(crate) fn fp4_decode(bytes: &[u8], scales: &[u8], t: usize, d: usize) -> f32 {
    let nib = (bytes[t * 64 + d / 2] >> (4 * (d & 1))) & 0xF;
    let scale = f32::from_bits((scales[t * 4 + d / 32] as u32) << 23); // 2^(byte-127)
    e2m1_to_f32(nib) * scale
}

/// Verify the standalone FP4 dequant + dot-product score primitive. This probes
/// the exact serving-critical boundary: packed FP4 KV rows are dequantized in
/// registers and immediately consumed by the QK dot, with no global BF16 staging.
pub fn check_fp4_dot_probe_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("FP4 dot probe requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 57))).collect();
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 58)).collect();
    let (k4, ksc) = quantize_fp4_blocks(&kf, nu);
    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut out_buf = dev.alloc_device(nu * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice()[..nu * 64].copy_from_slice(&k4);
        ksc_buf.as_mut_slice()[..nu * 4].copy_from_slice(&ksc);
    }
    let num_wg = n.clamp(1, 4096);
    dev.arm_fp4_dot_probe(
        q_buf.va(),
        k_buf.va(),
        ksc_buf.va(),
        out_buf.va(),
        n,
        d,
        num_wg,
    )?;
    dev.wait(Duration::from_secs(10))?;

    let got = unsafe { out_buf.as_mut_slice_of::<f32>() };
    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut max_rel = 0f64;
    for t in 0..nu {
        let mut exp = 0f64;
        for dd in 0..du {
            exp += qf[dd] * fp4_decode(&k4, &ksc, t, dd) as f64;
        }
        max_rel = max_rel.max((got[t] as f64 - exp).abs() / (exp.abs() + 1e-3));
    }
    if max_rel > 2e-4 {
        return Err(anyhow!(
            "FP4 dot probe N={n} D={d} mismatch: max rel err {max_rel:.4} > 2e-4"
        ));
    }
    println!(
        "  fp4-dot-probe: N={} D={} on node {} - correct (max rel err {:.2e})",
        n, d, node_id, max_rel
    );
    Ok(())
}

/// Benchmark the standalone FP4 dequant + dot primitive at long-context scale.
/// Effective bandwidth counts the compressed row bytes, scale bytes, and f32
/// output bytes; q is reused per workgroup and intentionally not counted.
pub fn bench_fp4_dot_probe_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    bench_fp4_dot_probe_on_with_wg(dev, n, d, iters, n.clamp(1, 8192))
}

pub fn bench_fp4_dot_probe_on_with_wg(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
    num_wg: u32,
) -> Result<(f64, f64)> {
    if d != 128 {
        return Err(anyhow!("FP4 dot probe requires D=128 (got {d})"));
    }
    if n == 0 {
        return Err(anyhow!("FP4 dot probe requires N > 0"));
    }
    if num_wg == 0 {
        return Err(anyhow!("FP4 dot probe requires num_wg > 0"));
    }
    let (nu, du) = (n as usize, d as usize);
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 59))).collect();
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 60)).collect();
    let (k4, ksc) = quantize_fp4_blocks(&kf, nu);
    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let out_buf = dev.alloc_device(nu * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice()[..nu * 64].copy_from_slice(&k4);
        ksc_buf.as_mut_slice()[..nu * 4].copy_from_slice(&ksc);
    }
    dev.arm_fp4_dot_probe(
        q_buf.va(),
        k_buf.va(),
        ksc_buf.va(),
        out_buf.va(),
        n,
        d,
        num_wg,
    )?;
    dev.wait(Duration::from_secs(10))?;
    let reps = iters.max(1);
    let mut total = Duration::ZERO;
    for _ in 0..reps {
        let t0 = Instant::now();
        dev.arm_fp4_dot_probe(
            q_buf.va(),
            k_buf.va(),
            ksc_buf.va(),
            out_buf.va(),
            n,
            d,
            num_wg,
        )?;
        total += dev.wait(Duration::from_secs(10))?.max(t0.elapsed());
    }
    let us = total.as_secs_f64() * 1e6 / reps as f64;
    let bytes = n as f64 * (64.0 + 4.0 + 4.0);
    let gbps = bytes / (us * 1e-6) / 1e9;
    Ok((us, gbps))
}

pub fn bench_fp4_dot_probe_wg_sweep_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
    wgs: &[u32],
) -> Result<()> {
    if wgs.is_empty() {
        return Err(anyhow!(
            "FP4 dot probe workgroup sweep requires at least one workgroup count"
        ));
    }
    let node_id = dev.node_id();
    println!(
        "  fp4-dot-sweep: N={n} D={d} node={node_id} iters={} (compressed bytes count K+scale+out)",
        iters.max(1)
    );
    println!("  {:>10}   {:>12}   {:>12}", "num_wg", "us", "GB/s");
    println!("  {}", "-".repeat(42));
    let mut best = (0u32, f64::INFINITY, 0.0f64);
    for &num_wg in wgs {
        let (us, gbps) = bench_fp4_dot_probe_on_with_wg(dev, n, d, iters, num_wg)?;
        if us < best.1 {
            best = (num_wg, us, gbps);
        }
        println!("  {num_wg:>10}   {us:>12.1}   {gbps:>12.1}");
    }
    println!(
        "  fp4-dot-sweep-best: num_wg={} us={:.1} GB/s={:.1}",
        best.0, best.1, best.2
    );
    Ok(())
}

/// Quantize N×128 f32 to NVFP4: E2M1 values (64 B/token) + a per-block-16 E4M3
/// scale (8 blocks/row). Returns (packed bytes, E4M3 scale bytes [N*8]).
fn quantize_nvfp4(x: &[f32], n: usize) -> (Vec<u8>, Vec<u8>) {
    let mut bytes = vec![0u8; n * 64];
    let mut scales = vec![0u8; n * 8];
    for t in 0..n {
        for bl in 0..8 {
            let base = t * 128 + bl * 16;
            let maxabs = (0..16).fold(0.0f32, |a, j| a.max(x[base + j].abs()));
            let raw = if maxabs > 0.0 { maxabs / 6.0 } else { 1.0 };
            let sbyte = f32_to_e4m3(raw);
            let scale = e4m3_to_f32(sbyte).max(1e-30);
            scales[t * 8 + bl] = sbyte;
            let inv = 1.0 / scale;
            for j in 0..16 {
                let d = bl * 16 + j;
                let nib = f32_to_e2m1(x[t * 128 + d] * inv);
                bytes[t * 64 + d / 2] |= nib << (4 * (d & 1));
            }
        }
    }
    (bytes, scales)
}

/// Decode NVFP4 element `d` of token `t` (E2M1 value × E4M3 block-16 scale).
fn nvfp4_decode(bytes: &[u8], scales: &[u8], t: usize, d: usize) -> f32 {
    let nib = (bytes[t * 64 + d / 2] >> (4 * (d & 1))) & 0xF;
    e2m1_to_f32(nib) * e4m3_to_f32(scales[t * 8 + d / 16])
}

/// Verify FP4 (E2M1, per-block-32) KV decode. Reference is over the decoded FP4
/// values (validates the kernel); also reports FP4-vs-true relative error.
pub fn check_attn_decode_fp4_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("fp4 path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 7))).collect();
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k4, ksc) = quantize_fp4_blocks(&kf, nu);
    let (v4, vsc) = quantize_fp4_blocks(&vf, nu);

    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut v_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let mut o_buf = dev.alloc(du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&k4);
        v_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&v4);
        ksc_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&ksc);
        vsc_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&vsc);
    }
    dev.chain_next();
    dev.arm_attn_decode_split_fp4(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
    )?;

    // f64 reference over decoded FP4; also track FP4-vs-true error.
    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut scores = vec![0f64; nu];
    let mut scores_true = vec![0f64; nu];
    let mut mx = f64::NEG_INFINITY;
    let mut mx_t = f64::NEG_INFINITY;
    for i in 0..nu {
        let mut s = 0f64;
        let mut st = 0f64;
        for dd in 0..du {
            s += qf[dd] * fp4_decode(&k4, &ksc, i, dd) as f64;
            st += qf[dd] * kf[i * du + dd] as f64;
        }
        scores[i] = s * scale as f64;
        scores_true[i] = st * scale as f64;
        mx = mx.max(scores[i]);
        mx_t = mx_t.max(scores_true[i]);
    }
    let (mut z, mut zt) = (0f64, 0f64);
    for i in 0..nu {
        scores[i] = (scores[i] - mx).exp();
        scores_true[i] = (scores_true[i] - mx_t).exp();
        z += scores[i];
        zt += scores_true[i];
    }
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let (mut max_rel, mut max_rel_true) = (0f64, 0f64);
    for dd in 0..du {
        let mut exp_dec = 0f64;
        let mut exp_true = 0f64;
        for i in 0..nu {
            exp_dec += scores[i] / z * fp4_decode(&v4, &vsc, i, dd) as f64;
            exp_true += scores_true[i] / zt * vf[i * du + dd] as f64;
        }
        max_rel = max_rel.max((got[dd] as f64 - exp_dec).abs() / (exp_dec.abs() + 1e-3));
        max_rel_true =
            max_rel_true.max((got[dd] as f64 - exp_true).abs() / (exp_true.abs() + 1e-3));
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode-fp4 N={n} D={d} kernel mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-fp4: decode N={n} D={d} ({num_splits} splits) on node {node_id}: kernel correct (rel err {max_rel:.2e}); FP4-vs-true rel err {max_rel_true:.2e}");
    Ok(())
}

/// Benchmark FP4 KV decode; returns (GB/s on FP4 KV bytes incl. scales, us).
pub fn bench_attn_decode_fp4_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let q_buf = dev.alloc(du * 2)?;
    let k_buf = dev.alloc_device(nu * 64)?;
    let v_buf = dev.alloc_device(nu * 64)?;
    let ksc_buf = dev.alloc_device(nu * 4)?;
    let vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let o_buf = dev.alloc(du * 4)?;
    let (qv, kv, vv, kscv, vscv, pv, iv, ov) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.chain_next();
        dev.arm_attn_decode_split_fp4(qv, kv, vv, kscv, vscv, pv, n, d, scale, num_splits)?;
        combine_decode(dev, pv, iv, ov, d, num_splits)?;
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
    // FP4 KV bytes read: K+V packed (N*64 each) + E8M0 scales (N*4 each).
    let bytes = 2.0 * (nu as f64 * 64.0 + nu as f64 * 4.0);
    Ok((bytes / t / 1e9, t * 1e6))
}

/// Verify NVFP4 decode (E2M1 + per-block-16 E4M3 scale). Reference over decoded
/// values (validates the kernel); reports NVFP4-vs-true (expect ~9% on real data).
pub fn check_attn_decode_nvfp4_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("nvfp4 path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let q16: Vec<u16> = (0..du).map(|i| f32_to_f16(gen(i, 7))).collect();
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k4, ksc) = quantize_nvfp4(&kf, nu);
    let (v4, vsc) = quantize_nvfp4(&vf, nu);

    let mut q_buf = dev.alloc(du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut v_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 8)?;
    let mut vsc_buf = dev.alloc_device(nu * 8)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let mut o_buf = dev.alloc(du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&k4);
        v_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&v4);
        ksc_buf.as_mut_slice_of::<u8>()[..nu * 8].copy_from_slice(&ksc);
        vsc_buf.as_mut_slice_of::<u8>()[..nu * 8].copy_from_slice(&vsc);
    }
    dev.chain_next();
    dev.arm_attn_decode_split_nvfp4(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
    )?;

    // Reference over decoded NVFP4 (validates the kernel; accuracy-vs-true is
    // measured properly on realistic data by characterize_kv_quant, since the
    // synthetic gen() data aligns artificially with power-of-two scales).
    let qf: Vec<f64> = q16.iter().map(|&h| f16_to_f32(h) as f64).collect();
    let mut scores = vec![0f64; nu];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..nu {
        let mut s = 0f64;
        for dd in 0..du {
            s += qf[dd] * nvfp4_decode(&k4, &ksc, i, dd) as f64;
        }
        scores[i] = s * scale as f64;
        mx = mx.max(scores[i]);
    }
    let mut z = 0f64;
    for i in 0..nu {
        scores[i] = (scores[i] - mx).exp();
        z += scores[i];
    }
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for dd in 0..du {
        let mut ed = 0f64;
        for i in 0..nu {
            ed += scores[i] / z * nvfp4_decode(&v4, &vsc, i, dd) as f64;
        }
        max_rel = max_rel.max((got[dd] as f64 - ed).abs() / (ed.abs() + 1e-3));
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-nvfp4 N={n} kernel mismatch: {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-nvfp4: decode N={n} D={d} ({num_splits} splits) on node {node_id}: kernel correct (rel err {max_rel:.2e})");
    Ok(())
}

/// Verify GQA FP8 decode: GQA_G query heads share one FP8 KV head. Reference is
/// computed per head over the decoded KV. Validates the KV-read-once kernel.
pub fn check_attn_decode_fp8_gqa_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("gqa fp8 path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2; // partial stride

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    // GQA_G query heads (distinct seeds), one shared KV head.
    let mut q16 = vec![0u16; GQA_G * du];
    for h in 0..GQA_G {
        for i in 0..du {
            q16[h * du + i] = f32_to_f16(gen(i, 7 + h));
        }
    }
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k8, ksc) = quantize_e4m3_rows(&kf, nu, du);
    let (v8, vsc) = quantize_e4m3_rows(&vf, nu, du);

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du)?;
    let mut v_buf = dev.alloc_device(nu * du)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let mut o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * du].copy_from_slice(&k8);
        v_buf.as_mut_slice_of::<u8>()[..nu * du].copy_from_slice(&v8);
        ksc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&ksc);
        vsc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&vsc);
    }

    dev.chain_next();
    dev.arm_attn_decode_split_fp8_gqa(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode_gqa(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
        GQA_G as u32,
    )?;

    // Per-head f64 reference over decoded KV.
    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for h in 0..GQA_G {
        let qf: Vec<f64> = (0..du)
            .map(|i| f16_to_f32(q16[h * du + i]) as f64)
            .collect();
        let mut scores = vec![0f64; nu];
        let mut mx = f64::NEG_INFINITY;
        for i in 0..nu {
            let mut s = 0f64;
            for dd in 0..du {
                let kdq = e4m3_to_f32(k8[i * du + dd]) as f64 * ksc[i] as f64;
                s += qf[dd] * kdq;
            }
            scores[i] = s * scale as f64;
            mx = mx.max(scores[i]);
        }
        let mut z = 0f64;
        for s in &mut scores {
            *s = (*s - mx).exp();
            z += *s;
        }
        for dd in 0..du {
            let mut acc = 0f64;
            for i in 0..nu {
                acc += scores[i] / z * (e4m3_to_f32(v8[i * du + dd]) as f64 * vsc[i] as f64);
            }
            let rel = (got[h * du + dd] as f64 - acc).abs() / (acc.abs() + 1e-3);
            max_rel = max_rel.max(rel);
        }
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode-fp8-gqa N={n} G={GQA_G} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-fp8-gqa: decode N={n} G={GQA_G} heads/KV ({num_splits} splits) on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Verify paged GQA FP8 decode: one FP8 KV head is scattered into physical
/// pages behind a block table and shared by GQA_G query heads.
pub fn check_attn_decode_fp8_gqa_paged_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("paged gqa fp8 path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let mut q16 = vec![0u16; GQA_G * du];
    for h in 0..GQA_G {
        for i in 0..du {
            q16[h * du + i] = f32_to_f16(gen(i, 7 + h));
        }
    }
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k8, ksc) = quantize_e4m3_rows(&kf, nu, du);
    let (v8, vsc) = quantize_e4m3_rows(&vf, nu, du);

    let table: Vec<u32> = (0..nblocks).map(|i| (nblocks - 1 - i) as u32).collect();
    let mut k8p = vec![0u8; nu * du];
    let mut v8p = vec![0u8; nu * du];
    let mut kscp = vec![0.0f32; nu];
    let mut vscp = vec![0.0f32; nu];
    for i in 0..nu {
        let lb = i / bs;
        let row = table[lb] as usize * bs + (i - lb * bs);
        k8p[row * du..row * du + du].copy_from_slice(&k8[i * du..i * du + du]);
        v8p[row * du..row * du + du].copy_from_slice(&v8[i * du..i * du + du]);
        kscp[row] = ksc[i];
        vscp[row] = vsc[i];
    }

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du)?;
    let mut v_buf = dev.alloc_device(nu * du)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let mut o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * du].copy_from_slice(&k8p);
        v_buf.as_mut_slice_of::<u8>()[..nu * du].copy_from_slice(&v8p);
        ksc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&kscp);
        vsc_buf.as_mut_slice_of::<f32>()[..nu].copy_from_slice(&vscp);
        tbl_buf.as_mut_slice_of::<u32>()[..nblocks].copy_from_slice(&table);
    }

    let table_check = dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;
    dev.chain_next();
    dev.arm_attn_decode_split_fp8_gqa_paged(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode_gqa(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
        GQA_G as u32,
    )?;

    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for h in 0..GQA_G {
        let qf: Vec<f64> = (0..du)
            .map(|i| f16_to_f32(q16[h * du + i]) as f64)
            .collect();
        let mut scores = vec![0f64; nu];
        let mut mx = f64::NEG_INFINITY;
        for i in 0..nu {
            let mut s = 0f64;
            for dd in 0..du {
                let kdq = e4m3_to_f32(k8[i * du + dd]) as f64 * ksc[i] as f64;
                s += qf[dd] * kdq;
            }
            scores[i] = s * scale as f64;
            mx = mx.max(scores[i]);
        }
        let mut z = 0f64;
        for s in &mut scores {
            *s = (*s - mx).exp();
            z += *s;
        }
        for dd in 0..du {
            let mut acc = 0f64;
            for i in 0..nu {
                acc += scores[i] / z * (e4m3_to_f32(v8[i * du + dd]) as f64 * vsc[i] as f64);
            }
            let rel = (got[h * du + dd] as f64 - acc).abs() / (acc.abs() + 1e-3);
            max_rel = max_rel.max(rel);
        }
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode-fp8-gqa-paged N={n} G={GQA_G} mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-fp8-gqa-paged: decode N={n} G={GQA_G} block={bs} (shuffled, {nblocks} blk, max page {}) on node {node_id} - correct (max rel err {max_rel:.2e})", table_check.max_entry);
    Ok(())
}

/// Benchmark GQA FP8 decode: one KV-read serves GQA_G heads. Returns (us per
/// head-group decode, us per head).
pub fn bench_attn_decode_fp8_gqa_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let q_buf = dev.alloc(GQA_G * du * 2)?;
    let k_buf = dev.alloc_device(nu * du)?;
    let v_buf = dev.alloc_device(nu * du)?;
    let ksc_buf = dev.alloc_device(nu * 4)?;
    let vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let o_buf = dev.alloc(GQA_G * du * 4)?;
    let (qv, kv, vv, kscv, vscv, pv0, iv, ov0) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.chain_next();
        dev.arm_attn_decode_split_fp8_gqa(qv, kv, vv, kscv, vscv, pv0, n, d, scale, num_splits)?;
        combine_decode_gqa(dev, pv0, iv, ov0, d, num_splits, GQA_G as u32)?;
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
    Ok((t * 1e6, t * 1e6 / GQA_G as f64))
}

/// Verify FP16 GQA decode: GQA_G query heads share one FP16 KV head (the model
/// decode loop's attention), bit-close to an f64 reference over the f16 KV. This
/// is the same kernel the decode loop uses; the win is reading the shared KV once
/// per group instead of once per query head.
pub fn check_attn_decode_gqa_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("gqa path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let mut q16 = vec![0u16; GQA_G * du];
    for h in 0..GQA_G {
        for i in 0..du {
            q16[h * du + i] = f32_to_f16(gen(i, 7 + h));
        }
    }
    let k16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 1))).collect();
    let v16: Vec<u16> = (0..nu * du).map(|i| f32_to_f16(gen(i, 2))).collect();

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * du * 2)?;
    let mut v_buf = dev.alloc_device(nu * du * 2)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let mut o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&k16);
        v_buf.as_mut_slice_of::<u16>()[..nu * du].copy_from_slice(&v16);
    }

    dev.chain_next();
    dev.arm_attn_decode_split_gqa(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode_gqa(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
        GQA_G as u32,
    )?;

    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let mut max_rel = 0f64;
    for h in 0..GQA_G {
        let qf: Vec<f64> = (0..du)
            .map(|i| f16_to_f32(q16[h * du + i]) as f64)
            .collect();
        let mut scores = vec![0f64; nu];
        let mut mx = f64::NEG_INFINITY;
        for i in 0..nu {
            let mut s = 0f64;
            for dd in 0..du {
                s += qf[dd] * f16_to_f32(k16[i * du + dd]) as f64;
            }
            scores[i] = s * scale as f64;
            mx = mx.max(scores[i]);
        }
        let mut z = 0f64;
        for s in &mut scores {
            *s = (*s - mx).exp();
            z += *s;
        }
        for dd in 0..du {
            let mut acc = 0f64;
            for i in 0..nu {
                acc += scores[i] / z * f16_to_f32(v16[i * du + dd]) as f64;
            }
            let rel = (got[h * du + dd] as f64 - acc).abs() / (acc.abs() + 1e-3);
            max_rel = max_rel.max(rel);
        }
    }
    if max_rel > 5e-3 {
        return Err(anyhow!(
            "attn-decode-gqa N={n} G={GQA_G} mismatch: max rel err {max_rel:.4} > 5e-3"
        ));
    }
    println!("  attn-gqa(fp16): decode N={n} G={GQA_G} heads/KV ({num_splits} splits) on node {node_id}: correct (max rel err {max_rel:.2e})");
    Ok(())
}

/// Benchmark FP16 GQA decode: one KV-read serves GQA_G heads. Returns (us per
/// head-group decode, us per head).
pub fn bench_attn_decode_gqa_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let q_buf = dev.alloc(GQA_G * du * 2)?;
    let k_buf = dev.alloc_device(nu * du * 2)?;
    let v_buf = dev.alloc_device(nu * du * 2)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let o_buf = dev.alloc(GQA_G * du * 4)?;
    let (qv, kv, vv, pv0, iv, ov0) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.chain_next();
        dev.arm_attn_decode_split_gqa(qv, kv, vv, pv0, n, d, scale, num_splits)?;
        combine_decode_gqa(dev, pv0, iv, ov0, d, num_splits, GQA_G as u32)?;
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
    Ok((t * 1e6, t * 1e6 / GQA_G as f64))
}

/// Verify GQA+FP4 decode: GQA_G query heads share one FP4 KV head (the headline
/// 1M config). Reference per head over decoded FP4; reports FP4-vs-true too.
pub fn check_attn_decode_fp4_gqa_on(dev: &mut GpuDevice, n: u32, d: u32) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("gqa fp4 path requires D=128 (got {d})"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let mut q16 = vec![0u16; GQA_G * du];
    for h in 0..GQA_G {
        for i in 0..du {
            q16[h * du + i] = f32_to_f16(gen(i, 7 + h));
        }
    }
    let kf: Vec<f32> = (0..nu * du).map(|i| gen(i, 1)).collect();
    let vf: Vec<f32> = (0..nu * du).map(|i| gen(i, 2)).collect();
    let (k4, ksc) = quantize_fp4_blocks(&kf, nu);
    let (v4, vsc) = quantize_fp4_blocks(&vf, nu);

    let mut q_buf = dev.alloc(GQA_G * du * 2)?;
    let mut k_buf = dev.alloc_device(nu * 64)?;
    let mut v_buf = dev.alloc_device(nu * 64)?;
    let mut ksc_buf = dev.alloc_device(nu * 4)?;
    let mut vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let mut o_buf = dev.alloc(GQA_G * du * 4)?;
    unsafe {
        q_buf.as_mut_slice_of::<u16>()[..GQA_G * du].copy_from_slice(&q16);
        k_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&k4);
        v_buf.as_mut_slice_of::<u8>()[..nu * 64].copy_from_slice(&v4);
        ksc_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&ksc);
        vsc_buf.as_mut_slice_of::<u8>()[..nu * 4].copy_from_slice(&vsc);
    }
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_gqa(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        n,
        d,
        scale,
        num_splits,
    )?;
    combine_decode_gqa(
        dev,
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
        d,
        num_splits,
        GQA_G as u32,
    )?;

    let got = unsafe { o_buf.as_mut_slice_of::<f32>() };
    let (mut max_rel, mut max_rel_true) = (0f64, 0f64);
    for h in 0..GQA_G {
        let qf: Vec<f64> = (0..du)
            .map(|i| f16_to_f32(q16[h * du + i]) as f64)
            .collect();
        let mut sc = vec![0f64; nu];
        let mut sct = vec![0f64; nu];
        let (mut mx, mut mxt) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for i in 0..nu {
            let (mut s, mut st) = (0f64, 0f64);
            for dd in 0..du {
                s += qf[dd] * fp4_decode(&k4, &ksc, i, dd) as f64;
                st += qf[dd] * kf[i * du + dd] as f64;
            }
            sc[i] = s * scale as f64;
            sct[i] = st * scale as f64;
            mx = mx.max(sc[i]);
            mxt = mxt.max(sct[i]);
        }
        let (mut z, mut zt) = (0f64, 0f64);
        for i in 0..nu {
            sc[i] = (sc[i] - mx).exp();
            sct[i] = (sct[i] - mxt).exp();
            z += sc[i];
            zt += sct[i];
        }
        for dd in 0..du {
            let (mut ed, mut et) = (0f64, 0f64);
            for i in 0..nu {
                ed += sc[i] / z * fp4_decode(&v4, &vsc, i, dd) as f64;
                et += sct[i] / zt * vf[i * du + dd] as f64;
            }
            max_rel = max_rel.max((got[h * du + dd] as f64 - ed).abs() / (ed.abs() + 1e-3));
            max_rel_true =
                max_rel_true.max((got[h * du + dd] as f64 - et).abs() / (et.abs() + 1e-3));
        }
    }
    if max_rel > 2e-2 {
        return Err(anyhow!(
            "attn-decode-fp4-gqa N={n} G={GQA_G} kernel mismatch: max rel err {max_rel:.4} > 2e-2"
        ));
    }
    println!("  attn-fp4-gqa: decode N={n} G={GQA_G} heads/KV ({num_splits} splits) on node {node_id}: kernel correct (rel err {max_rel:.2e}); FP4-vs-true {max_rel_true:.2e}");
    Ok(())
}

/// Benchmark GQA+FP4 decode: one 4-bit KV-read serves GQA_G heads. Returns
/// (us per head-group decode, us per head).
pub fn bench_attn_decode_fp4_gqa_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let ps = du + 2;
    let q_buf = dev.alloc(GQA_G * du * 2)?;
    let k_buf = dev.alloc_device(nu * 64)?;
    let v_buf = dev.alloc_device(nu * 64)?;
    let ksc_buf = dev.alloc_device(nu * 4)?;
    let vsc_buf = dev.alloc_device(nu * 4)?;
    let part_buf = dev.alloc_device(GQA_G * num_splits as usize * ps * 4)?;
    let inter_buf =
        dev.alloc_device(GQA_G * (combine_groups(num_splits).max(1) as usize) * ps * 4)?;
    let o_buf = dev.alloc(GQA_G * du * 4)?;
    let (qv, kv, vv, kscv, vscv, pv0, iv, ov0) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.chain_next();
        dev.arm_attn_decode_split_fp4_gqa(qv, kv, vv, kscv, vscv, pv0, n, d, scale, num_splits)?;
        combine_decode_gqa(dev, pv0, iv, ov0, d, num_splits, GQA_G as u32)?;
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
    Ok((t * 1e6, t * 1e6 / GQA_G as f64))
}

/// Pure VRAM streaming-read bandwidth (float4 loads, grid-stride, full
/// occupancy) — the real achievable HBM read ceiling in this environment.
pub fn bench_vram_read_on(dev: &mut GpuDevice, bytes: usize, iters: usize) -> Result<f64> {
    let n4 = bytes / 16; // float4 elements
    let in_buf = dev.alloc_device(n4 * 16)?;
    let out_buf = dev.alloc(256 * 4)?;
    let (iv, ov) = (in_buf.va(), out_buf.va());
    let num_wg = 4096u32; // plenty for grid-stride coverage at full occupancy
    let run = |dev: &mut GpuDevice| -> Result<()> {
        dev.arm_mem_stream(iv, ov, n4 as u32, num_wg)?;
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
    Ok((n4 * 16) as f64 / t / 1e9)
}

/// Benchmark decode attention; report effective KV-read bandwidth vs HBM peak.
pub fn bench_attn_decode(node_id: u32, n: u32, d: u32, iters: usize) -> Result<(f64, f64)> {
    let mut dev = GpuDevice::open(node_id)?;
    bench_attn_decode_on(&mut dev, n, d, iters)
}

pub fn bench_attn_decode_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
    iters: usize,
) -> Result<(f64, f64)> {
    let (nu, du) = (n as usize, d as usize);
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = split_count(n);
    let q_buf = dev.alloc(du * 2)?;
    let k_buf = dev.alloc_device(nu * du * 2)?;
    let v_buf = dev.alloc_device(nu * du * 2)?;
    let part_buf = dev.alloc_device(num_splits as usize * (du + 2) * 4)?;
    let inter_buf =
        dev.alloc_device((combine_groups(num_splits).max(1) as usize) * (du + 2) * 4)?;
    let o_buf = dev.alloc(du * 4)?;
    let (qv, kv, vv, pv, iv, ov) = (
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        part_buf.va(),
        inter_buf.va(),
        o_buf.va(),
    );

    let skip_combine = std::env::var("SKIP_COMBINE").is_ok();
    let run = |dev: &mut GpuDevice| -> Result<()> {
        if skip_combine {
            dev.arm_attn_decode_split(qv, kv, vv, pv, n, d, scale, num_splits)?;
            dev.wait(Duration::from_secs(10))?;
        } else {
            dev.chain_next();
            dev.arm_attn_decode_split(qv, kv, vv, pv, n, d, scale, num_splits)?;
            combine_decode(dev, pv, iv, ov, d, num_splits)?;
        }
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
    // KV bytes read: K and V, each N×D FP16.
    let bytes = 2.0 * nu as f64 * du as f64 * 2.0;
    let gbps = bytes / t / 1e9;
    Ok((gbps, t * 1e6))
}

/// Verify one full Qwen3-235B-A22B decode layer assembled from the raw-ABI
/// primitives: RMSNorm → QKV GEMV → per-head QK-norm → RoPE(θ=5e6) → GQA
/// attention (64 Q / 4 KV heads, group 16) → O GEMV → +residual → RMSNorm →
/// MoE FFN (top-8) → +residual. Validated vs an f64 reference that mirrors the
/// f16 datapath (round-trips at every f16 storage point). MoE expert count E is
/// reduced here to bound host RAM (the kernels are E-agnostic); per-expert dims
/// (H=4096, I=1536) and head dims are the real model's.
pub fn check_decode_layer_on(dev: &mut GpuDevice) -> Result<()> {
    let node_id = dev.node_id();
    const H: usize = 4096;
    const NH: usize = 64;
    const NKV: usize = 4;
    const D: usize = 128;
    const GRP: usize = NH / NKV; // 16
    const QD: usize = NH * D; // 8192
    const KVD: usize = NKV * D; // 512
    const II: usize = 1536; // MoE intermediate
    const E: usize = 16;
    const TOPK: usize = 8;
    const L: usize = 2048; // context length (current token at L-1)
    let pos = (L - 1) as u32;
    let theta = 5_000_000.0f32;
    let eps = 1e-6f32;
    let scale = 1.0f32 / (D as f32).sqrt();
    let r16 = |x: f64| f16_to_f32(f32_to_f16(x as f32)) as f64;
    let gen = |a: usize, s: usize| {
        ((a.wrapping_mul(2654435761)
            .wrapping_add(s.wrapping_mul(40503))
            >> 11)
            & 0xff) as f32
            / 256.0
            - 0.5
    };

    let h16: Vec<u16> = (0..H).map(|i| f32_to_f16(gen(i, 1))).collect();
    let innorm16: Vec<u16> = (0..H).map(|i| f32_to_f16(gen(i, 2) + 1.0)).collect();
    let postnorm16: Vec<u16> = (0..H).map(|i| f32_to_f16(gen(i, 3) + 1.0)).collect();
    let qnorm16: Vec<u16> = (0..D).map(|i| f32_to_f16(gen(i, 4) + 1.0)).collect();
    let knorm16: Vec<u16> = (0..D).map(|i| f32_to_f16(gen(i, 5) + 1.0)).collect();
    let wq16: Vec<u16> = (0..QD * H).map(|i| f32_to_f16(gen(i, 6) * 0.02)).collect();
    let wk16: Vec<u16> = (0..KVD * H).map(|i| f32_to_f16(gen(i, 7) * 0.02)).collect();
    let wv16: Vec<u16> = (0..KVD * H).map(|i| f32_to_f16(gen(i, 8) * 0.02)).collect();
    let wo16: Vec<u16> = (0..H * QD).map(|i| f32_to_f16(gen(i, 9) * 0.02)).collect();
    let wgate16: Vec<u16> = (0..E * H).map(|i| f32_to_f16(gen(i, 10) * 0.02)).collect();
    let gate16: Vec<u16> = (0..E * II * H)
        .map(|i| f32_to_f16(gen(i, 11) * 0.05))
        .collect();
    let up16: Vec<u16> = (0..E * II * H)
        .map(|i| f32_to_f16(gen(i, 12) * 0.05))
        .collect();
    let down16: Vec<u16> = (0..E * H * II)
        .map(|i| f32_to_f16(gen(i, 13) * 0.05))
        .collect();
    let kcache16: Vec<u16> = (0..NKV * L * D).map(|i| f32_to_f16(gen(i, 14))).collect();
    let vcache16: Vec<u16> = (0..NKV * L * D).map(|i| f32_to_f16(gen(i, 15))).collect();

    // Device buffers.
    let mut h_buf = dev.alloc_device(H * 2)?;
    let mut innorm_buf = dev.alloc_device(H * 2)?;
    let mut postnorm_buf = dev.alloc_device(H * 2)?;
    let mut qnorm_buf = dev.alloc_device(D * 2)?;
    let mut knorm_buf = dev.alloc_device(D * 2)?;
    let mut wq_buf = dev.alloc_device(QD * H * 2)?;
    let mut wk_buf = dev.alloc_device(KVD * H * 2)?;
    let mut wv_buf = dev.alloc_device(KVD * H * 2)?;
    let mut wo_buf = dev.alloc_device(H * QD * 2)?;
    let mut wgate_buf = dev.alloc_device(E * H * 2)?;
    let mut gate_buf = dev.alloc_device(E * II * H * 2)?;
    let mut up_buf = dev.alloc_device(E * II * H * 2)?;
    let mut down_buf = dev.alloc_device(E * H * II * 2)?;
    let mut kcache_buf = dev.alloc_device(NKV * L * D * 2)?;
    let mut vcache_buf = dev.alloc_device(NKV * L * D * 2)?;
    let hn_buf = dev.alloc_device(H * 2)?;
    let h2n_buf = dev.alloc_device(H * 2)?;
    let qf32_buf = dev.alloc_device(QD * 4)?;
    let kf32_buf = dev.alloc_device(KVD * 4)?;
    let vf32_buf = dev.alloc_device(KVD * 4)?;
    let qf16_buf = dev.alloc_device(QD * 2)?;
    let mut kf16_buf = dev.alloc_device(KVD * 2)?;
    let mut vf16_buf = dev.alloc_device(KVD * 2)?;
    let attn32_buf = dev.alloc_device(QD * 4)?;
    let attn16_buf = dev.alloc_device(QD * 2)?;
    let oproj_buf = dev.alloc_device(H * 4)?;
    let logits_buf = dev.alloc_device(E * 4)?;
    let mut ids_buf = dev.alloc_device(TOPK * 4)?;
    let mut rw_buf = dev.alloc_device(TOPK * 4)?;
    let hi_buf = dev.alloc_device(II * 2)?;
    let mut moe_buf = dev.alloc_device(H * 4)?;
    let ns = split_count(L as u32);
    let part_buf = dev.alloc_device(ns as usize * (D + 2) * 4)?;
    let inter_buf = dev.alloc_device((combine_groups(ns).max(1) as usize) * (D + 2) * 4)?;
    unsafe {
        h_buf.as_mut_slice_of::<u16>()[..H].copy_from_slice(&h16);
        innorm_buf.as_mut_slice_of::<u16>()[..H].copy_from_slice(&innorm16);
        postnorm_buf.as_mut_slice_of::<u16>()[..H].copy_from_slice(&postnorm16);
        qnorm_buf.as_mut_slice_of::<u16>()[..D].copy_from_slice(&qnorm16);
        knorm_buf.as_mut_slice_of::<u16>()[..D].copy_from_slice(&knorm16);
        wq_buf.as_mut_slice_of::<u16>()[..QD * H].copy_from_slice(&wq16);
        wk_buf.as_mut_slice_of::<u16>()[..KVD * H].copy_from_slice(&wk16);
        wv_buf.as_mut_slice_of::<u16>()[..KVD * H].copy_from_slice(&wv16);
        wo_buf.as_mut_slice_of::<u16>()[..H * QD].copy_from_slice(&wo16);
        wgate_buf.as_mut_slice_of::<u16>()[..E * H].copy_from_slice(&wgate16);
        gate_buf.as_mut_slice_of::<u16>()[..E * II * H].copy_from_slice(&gate16);
        up_buf.as_mut_slice_of::<u16>()[..E * II * H].copy_from_slice(&up16);
        down_buf.as_mut_slice_of::<u16>()[..E * H * II].copy_from_slice(&down16);
        kcache_buf.as_mut_slice_of::<u16>()[..NKV * L * D].copy_from_slice(&kcache16);
        vcache_buf.as_mut_slice_of::<u16>()[..NKV * L * D].copy_from_slice(&vcache16);
    }
    let b = |va: u64, off: usize| va + off as u64;
    let (hv, inv, pov, qnv, knv) = (
        h_buf.va(),
        innorm_buf.va(),
        postnorm_buf.va(),
        qnorm_buf.va(),
        knorm_buf.va(),
    );
    let (wqv, wkv, wvv, wov) = (wq_buf.va(), wk_buf.va(), wv_buf.va(), wo_buf.va());
    let (wgv, gv, uv, dv) = (wgate_buf.va(), gate_buf.va(), up_buf.va(), down_buf.va());
    let (kcv, vcv) = (kcache_buf.va(), vcache_buf.va());
    let (hnv, h2nv) = (hn_buf.va(), h2n_buf.va());
    let (qf32v, kf32v, vf32v) = (qf32_buf.va(), kf32_buf.va(), vf32_buf.va());
    let (qf16v, kf16v, vf16v) = (qf16_buf.va(), kf16_buf.va(), vf16_buf.va());
    let (a32v, a16v, opv) = (attn32_buf.va(), attn16_buf.va(), oproj_buf.va());
    let (lv, idv, rwv, hiv, mov) = (
        logits_buf.va(),
        ids_buf.va(),
        rw_buf.va(),
        hi_buf.va(),
        moe_buf.va(),
    );
    let (pv, iv) = (part_buf.va(), inter_buf.va());
    let to = Duration::from_secs(20);

    // ---- forward ----
    let fwd_start = Instant::now();
    dev.arm_rmsnorm(hv, inv, hnv, H as u32, eps)?;
    dev.wait(to)?;
    dev.arm_gemv(wqv, hnv, qf32v, QD as u32, H as u32)?;
    dev.wait(to)?;
    dev.arm_gemv(wkv, hnv, kf32v, KVD as u32, H as u32)?;
    dev.wait(to)?;
    dev.arm_gemv(wvv, hnv, vf32v, KVD as u32, H as u32)?;
    dev.wait(to)?;
    dev.arm_cast_f32_f16(qf32v, qf16v, QD as u32)?;
    dev.wait(to)?;
    dev.arm_cast_f32_f16(kf32v, kf16v, KVD as u32)?;
    dev.wait(to)?;
    dev.arm_cast_f32_f16(vf32v, vf16v, KVD as u32)?;
    dev.wait(to)?;
    for hh in 0..NH {
        dev.arm_rmsnorm(
            b(qf16v, hh * D * 2),
            qnv,
            b(qf16v, hh * D * 2),
            D as u32,
            eps,
        )?;
        dev.wait(to)?;
    }
    for hh in 0..NKV {
        dev.arm_rmsnorm(
            b(kf16v, hh * D * 2),
            knv,
            b(kf16v, hh * D * 2),
            D as u32,
            eps,
        )?;
        dev.wait(to)?;
    }
    for hh in 0..NH {
        dev.arm_rope(b(qf16v, hh * D * 2), D as u32, pos, theta)?;
        dev.wait(to)?;
    }
    for hh in 0..NKV {
        dev.arm_rope(b(kf16v, hh * D * 2), D as u32, pos, theta)?;
        dev.wait(to)?;
    }
    // Place the current token's k,v into the cache slot at position `pos`.
    {
        let kf = unsafe { kf16_buf.as_mut_slice_of::<u16>()[..KVD].to_vec() };
        let vf = unsafe { vf16_buf.as_mut_slice_of::<u16>()[..KVD].to_vec() };
        let kc = unsafe { kcache_buf.as_mut_slice_of::<u16>() };
        let vc = unsafe { vcache_buf.as_mut_slice_of::<u16>() };
        for hh in 0..NKV {
            for d in 0..D {
                kc[(hh * L + (L - 1)) * D + d] = kf[hh * D + d];
                vc[(hh * L + (L - 1)) * D + d] = vf[hh * D + d];
            }
        }
    }
    for hh in 0..NH {
        let kvh = hh / GRP;
        dev.chain_next();
        dev.arm_attn_decode_split(
            b(qf16v, hh * D * 2),
            b(kcv, kvh * L * D * 2),
            b(vcv, kvh * L * D * 2),
            pv,
            L as u32,
            D as u32,
            scale,
            ns,
        )?;
        combine_decode(dev, pv, iv, b(a32v, hh * D * 4), D as u32, ns)?;
    }
    dev.arm_cast_f32_f16(a32v, a16v, QD as u32)?;
    dev.wait(to)?;
    dev.arm_gemv(wov, a16v, opv, H as u32, QD as u32)?;
    dev.wait(to)?;
    dev.arm_add_into_f16(hv, opv, H as u32)?;
    dev.wait(to)?;
    dev.arm_rmsnorm(hv, pov, h2nv, H as u32, eps)?;
    dev.wait(to)?;
    dev.arm_gemv(wgv, h2nv, lv, E as u32, H as u32)?;
    dev.wait(to)?;
    dev.arm_moe_router_topk(lv, idv, rwv, E as u32, TOPK as u32)?;
    dev.wait(to)?;
    unsafe {
        for v in moe_buf.as_mut_slice_of::<f32>()[..H].iter_mut() {
            *v = 0.0;
        }
    }
    for slot in 0..TOPK as u32 {
        dev.arm_moe_gate_up_swiglu(gv, uv, h2nv, idv, hiv, slot, E as u32, II as u32, H as u32)?;
        dev.wait(to)?;
        dev.arm_moe_down_accum(dv, hiv, idv, rwv, mov, slot, E as u32, H as u32, II as u32)?;
        dev.wait(to)?;
    }
    dev.arm_add_into_f16(hv, mov, H as u32)?;
    dev.wait(to)?;
    let fwd_us = fwd_start.elapsed().as_secs_f64() * 1e6;
    let got = unsafe { h_buf.as_mut_slice_of::<u16>()[..H].to_vec() };
    let ids = unsafe { ids_buf.as_mut_slice_of::<u32>()[..TOPK].to_vec() };
    let rw = unsafe { rw_buf.as_mut_slice_of::<f32>()[..TOPK].to_vec() };

    // ---- f64 reference (mirrors the f16 datapath) ----
    let d16 = |v: u16| f16_to_f32(v) as f64;
    let mut hh: Vec<f64> = (0..H).map(|j| d16(h16[j])).collect();
    let rms = |v: &[f64], w: &dyn Fn(usize) -> f64, n: usize| {
        let ss: f64 = v.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let r = 1.0 / (ss + eps as f64).sqrt();
        (0..n).map(|j| r16(v[j] * r * w(j))).collect::<Vec<f64>>()
    };
    let hn = rms(&hh, &|j| d16(innorm16[j]), H);
    let proj = |w: &[u16], rows: usize| -> Vec<f64> {
        (0..rows)
            .map(|a| r16((0..H).map(|j| d16(w[a * H + j]) * hn[j]).sum::<f64>()))
            .collect()
    };
    let mut q = proj(&wq16, QD);
    let mut k = proj(&wk16, KVD);
    let v = proj(&wv16, KVD);
    // per-head QK-norm
    for head in 0..NH {
        let s: f64 = (0..D)
            .map(|d| q[head * D + d] * q[head * D + d])
            .sum::<f64>()
            / D as f64;
        let r = 1.0 / (s + eps as f64).sqrt();
        for d in 0..D {
            q[head * D + d] = r16(q[head * D + d] * r * d16(qnorm16[d]));
        }
    }
    for head in 0..NKV {
        let s: f64 = (0..D)
            .map(|d| k[head * D + d] * k[head * D + d])
            .sum::<f64>()
            / D as f64;
        let r = 1.0 / (s + eps as f64).sqrt();
        for d in 0..D {
            k[head * D + d] = r16(k[head * D + d] * r * d16(knorm16[d]));
        }
    }
    // RoPE (half-rotation) at `pos`
    let rope = |x: &mut [f64], head: usize| {
        for i in 0..D / 2 {
            let freq = (theta as f64).powf(-2.0 * i as f64 / D as f64);
            let ang = pos as f64 * freq;
            let (c, s) = (ang.cos(), ang.sin());
            let (a, bb) = (x[head * D + i], x[head * D + i + D / 2]);
            x[head * D + i] = r16(a * c - bb * s);
            x[head * D + i + D / 2] = r16(bb * c + a * s);
        }
    };
    for head in 0..NH {
        rope(&mut q, head);
    }
    for head in 0..NKV {
        rope(&mut k, head);
    }
    // attention per q-head over the cache (current k,v at L-1)
    let kget = |kvh: usize, t: usize, d: usize| -> f64 {
        if t == L - 1 {
            k[kvh * D + d]
        } else {
            d16(kcache16[(kvh * L + t) * D + d])
        }
    };
    let vget = |kvh: usize, t: usize, d: usize| -> f64 {
        if t == L - 1 {
            v[kvh * D + d]
        } else {
            d16(vcache16[(kvh * L + t) * D + d])
        }
    };
    let mut attn = vec![0f64; QD];
    for head in 0..NH {
        let kvh = head / GRP;
        let mut sc = vec![0f64; L];
        let mut mx = f64::NEG_INFINITY;
        for t in 0..L {
            let s: f64 = (0..D)
                .map(|d| q[head * D + d] * kget(kvh, t, d))
                .sum::<f64>()
                * scale as f64;
            sc[t] = s;
            mx = mx.max(s);
        }
        let mut z = 0f64;
        for s in &mut sc {
            *s = (*s - mx).exp();
            z += *s;
        }
        for d in 0..D {
            let mut acc = 0f64;
            for t in 0..L {
                acc += sc[t] / z * vget(kvh, t, d);
            }
            attn[head * D + d] = r16(acc);
        }
    }
    // O projection + residual 1
    for j in 0..H {
        let o: f64 = (0..QD).map(|a| d16(wo16[j * QD + a]) * attn[a]).sum();
        hh[j] = r16(hh[j] + o);
    }
    // post-attention RMSNorm
    let h2n = rms(&hh, &|j| d16(postnorm16[j]), H);
    // MoE (use the GPU's selected ids + router weights, as in check_moe_ffn)
    let mut moe = vec![0f64; H];
    for (slot, &id) in ids.iter().enumerate() {
        let (go, dofs) = (id as usize * II * H, id as usize * H * II);
        let mut hi = vec![0f64; II];
        for r in 0..II {
            let mut g = 0f64;
            let mut u = 0f64;
            for j in 0..H {
                g += d16(gate16[go + r * H + j]) * h2n[j];
                u += d16(up16[go + r * H + j]) * h2n[j];
            }
            hi[r] = r16((g / (1.0 + (-g).exp())) * u);
        }
        let wj = rw[slot] as f64;
        for n in 0..H {
            let mut d = 0f64;
            for r in 0..II {
                d += d16(down16[dofs + n * II + r]) * hi[r];
            }
            moe[n] += wj * d;
        }
    }
    // residual 2
    for j in 0..H {
        hh[j] = r16(hh[j] + moe[j]);
    }
    let (mut num, mut den) = (0f64, 0f64);
    for j in 0..H {
        let diff = d16(got[j]) - hh[j];
        num += diff * diff;
        den += hh[j] * hh[j];
    }
    let rel_l2 = (num / den.max(1e-30)).sqrt();
    if rel_l2 > 2e-2 {
        return Err(anyhow!(
            "decode-layer mismatch: rel-L2 {rel_l2:.4} > 2e-2 (H={H}, L={L})"
        ));
    }
    println!(
        "  decode-layer: Qwen3-235B-A22B block (H={H}, {NH}Q/{NKV}KV d{D}, MoE {E}×top{TOPK}, L={L}) on node {node_id}: correct (rel-L2 {rel_l2:.2e}); {fwd_us:.0} µs/layer (per-dispatch; chaining is the next perf step)"
    );
    Ok(())
}

/// Differential gate for the multi-head decode attention kernel.
///
/// The grouped-query kernel this variant was derived from is already validated
/// against an f64 reference, so rather than re-deriving that reference this gate
/// proves the two agree. It builds the one configuration where they must:
///
/// ```text
///   GQA: num_groups=1, q_heads_per_kv=G, group width G  -> G query heads share KV head 0
///   MHA: num_groups=G, q_heads_per_kv=1, group width 1  -> query head h reads KV head h
/// ```
///
/// With every KV head holding identical bytes, query head `h` sees the same
/// cache either way, so the two runs have to agree *bitwise*. That makes the
/// test sharp: a wrong `head_base`, a wrong `kvh`, or a wrong partials stride in
/// the new kernel all break the equality, and none of them would be caught by
/// merely checking the output looks reasonable.
pub fn check_attn_decode_fp4_mha_paged_matches_gqa_on(
    dev: &mut GpuDevice,
    n: u32,
    d: u32,
) -> Result<()> {
    if d != 128 {
        return Err(anyhow!("mha equivalence gate requires D=128 (got {d})"));
    }
    if n == 0 {
        return Err(anyhow!("mha equivalence gate requires N > 0"));
    }
    let node_id = dev.node_id();
    let (nu, du) = (n as usize, d as usize);
    let bs = 16usize;
    if nu % bs != 0 {
        return Err(anyhow!("N must be a multiple of block size {bs}"));
    }
    let nblocks = nu / bs;
    let heads = GQA_G;
    let kv_heads = GQA_G; // identical copies, so either mapping sees the same bytes
    let rows_per_head = nu;
    let scale = 1.0f32 / (d as f32).sqrt();
    let num_splits = 2u32;
    let ps = du + 2;
    let inter_splits = combine_groups(num_splits).max(1) as usize;

    let mut q_buf = dev.alloc(heads * du * 2)?;
    let mut k_buf = dev.alloc_device(kv_heads * rows_per_head * 64)?;
    let mut v_buf = dev.alloc_device(kv_heads * rows_per_head * 64)?;
    let mut ksc_buf = dev.alloc_device(kv_heads * rows_per_head * 4)?;
    let mut vsc_buf = dev.alloc_device(kv_heads * rows_per_head * 4)?;
    let mut tbl_buf = dev.alloc_device(nblocks * 4)?;
    let mut seq_lens_buf = dev.alloc_device(4)?;
    let gqa_part_buf = dev.alloc_device(heads * num_splits as usize * ps * 4)?;
    let gqa_inter_buf = dev.alloc_device(heads * inter_splits * ps * 4)?;
    let mut gqa_o_buf = dev.alloc(heads * du * 4)?;
    let mha_part_buf = dev.alloc_device(heads * num_splits as usize * ps * 4)?;
    let mha_inter_buf = dev.alloc_device(heads * inter_splits * ps * 4)?;
    let mut mha_o_buf = dev.alloc(heads * du * 4)?;

    let gen =
        |i: usize, s: usize| (((i * 2654435761 + s * 40503) >> 11) & 0xff) as f32 / 256.0 - 0.5;
    let fp4_byte = |row: usize, byte: usize, seed: usize| -> u8 {
        let x = row
            .wrapping_mul(131)
            .wrapping_add(byte.wrapping_mul(17))
            .wrapping_add(seed.wrapping_mul(29));
        ((x ^ (x >> 7) ^ (x >> 13)) & 0xff) as u8
    };
    let scale_byte =
        |row: usize, lane: usize, seed: usize| -> u8 { 126 + ((row + lane * 3 + seed) % 3) as u8 };

    unsafe {
        let q = q_buf.as_mut_slice_of::<u16>();
        for h in 0..heads {
            for i in 0..du {
                q[h * du + i] = f32_to_f16(gen(i, 71 + h));
            }
        }
        let table = tbl_buf.as_mut_slice_of::<u32>();
        for logical_block in 0..nblocks {
            table[logical_block] = ((logical_block * 17 + 11) % nblocks) as u32;
        }
        let k = k_buf.as_mut_slice_of::<u8>();
        let v = v_buf.as_mut_slice_of::<u8>();
        let ks = ksc_buf.as_mut_slice_of::<u8>();
        let vs = vsc_buf.as_mut_slice_of::<u8>();
        // Every KV head gets the SAME bytes. That is what makes the two
        // head->KV mappings interchangeable.
        for kvh in 0..kv_heads {
            for logical_block in 0..nblocks {
                let physical_block = table[logical_block] as usize;
                for offset in 0..bs {
                    let logical_row = logical_block * bs + offset;
                    let physical_row = physical_block * bs + offset;
                    let row_base = (kvh * rows_per_head + physical_row) * 64;
                    for byte in 0..64 {
                        k[row_base + byte] = fp4_byte(logical_row, byte, 3);
                        v[row_base + byte] = fp4_byte(logical_row, byte, 5);
                    }
                    let scale_base = (kvh * rows_per_head + physical_row) * 4;
                    for lane in 0..4 {
                        ks[scale_base + lane] = scale_byte(logical_row, lane, 7);
                        vs[scale_base + lane] = scale_byte(logical_row, lane, 11);
                    }
                }
            }
        }
        seq_lens_buf.as_mut_slice_of::<u32>()[0] = n;
    }

    dev.check_paged_block_table(tbl_buf.va(), nblocks as u32, nblocks as u32)?;

    let to = Duration::from_secs(20);

    // Grouped-query: one group of GQA_G heads, all sharing KV head 0.
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_gqa_paged_groups_meta(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        gqa_part_buf.va(),
        seq_lens_buf.va(),
        n,
        d,
        scale,
        num_splits,
        1,
        GQA_G as u32,
        rows_per_head as u32,
    )?;
    combine_decode_gqa(
        dev,
        gqa_part_buf.va(),
        gqa_inter_buf.va(),
        gqa_o_buf.va(),
        d,
        num_splits,
        heads as u32,
    )?;
    dev.wait(to)?;

    // Multi-head: GQA_G groups of one head, head h reading KV head h.
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_mha_paged_groups_meta(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        mha_part_buf.va(),
        seq_lens_buf.va(),
        n,
        d,
        scale,
        num_splits,
        heads as u32,
        1,
        rows_per_head as u32,
    )?;
    combine_decode_gqa(
        dev,
        mha_part_buf.va(),
        mha_inter_buf.va(),
        mha_o_buf.va(),
        d,
        num_splits,
        heads as u32,
    )?;
    dev.wait(to)?;

    let (gqa_bits, mha_bits) = unsafe {
        (
            gqa_o_buf.as_mut_slice_of::<u32>()[..heads * du].to_vec(),
            mha_o_buf.as_mut_slice_of::<u32>()[..heads * du].to_vec(),
        )
    };
    let mut mismatches = 0usize;
    let mut first: Option<(usize, u32, u32)> = None;
    for (i, (a, b)) in gqa_bits.iter().zip(mha_bits.iter()).enumerate() {
        if a != b {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, *a, *b));
            }
        }
    }
    if mismatches != 0 {
        let (i, a, b) = first.unwrap();
        return Err(anyhow!(
            "mha kernel disagrees with the validated gqa kernel: {mismatches}/{} lanes differ, \
             first at head {} dim {} (gqa {:08x} = {}, mha {:08x} = {})",
            gqa_bits.len(),
            i / du,
            i % du,
            a,
            f32::from_bits(a),
            b,
            f32::from_bits(b),
        ));
    }
    let nonzero = gqa_bits
        .iter()
        .filter(|w| f32::from_bits(**w) != 0.0)
        .count();
    if nonzero == 0 {
        return Err(anyhow!(
            "both kernels produced all-zero output, so the comparison proves nothing"
        ));
    }

    // The equality above has a hole worth closing: a kernel that ignored the
    // head index entirely and always read KV head 0 would also pass it, because
    // every KV head currently holds the same bytes. So break that symmetry and
    // require the answer to change. Perturbing KV heads 1.. must move the MHA
    // output away from the GQA output, which is only true if head h genuinely
    // reads KV head h.
    unsafe {
        let k = k_buf.as_mut_slice_of::<u8>();
        for kvh in 1..kv_heads {
            for row in 0..rows_per_head {
                let row_base = (kvh * rows_per_head + row) * 64;
                for byte in 0..64 {
                    k[row_base + byte] ^= 0x5a;
                }
            }
        }
    }
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_mha_paged_groups_meta(
        q_buf.va(),
        k_buf.va(),
        v_buf.va(),
        ksc_buf.va(),
        vsc_buf.va(),
        tbl_buf.va(),
        bs as u32,
        nblocks as u32,
        mha_part_buf.va(),
        seq_lens_buf.va(),
        n,
        d,
        scale,
        num_splits,
        heads as u32,
        1,
        rows_per_head as u32,
    )?;
    combine_decode_gqa(
        dev,
        mha_part_buf.va(),
        mha_inter_buf.va(),
        mha_o_buf.va(),
        d,
        num_splits,
        heads as u32,
    )?;
    dev.wait(to)?;
    let perturbed = unsafe { mha_o_buf.as_mut_slice_of::<u32>()[..heads * du].to_vec() };

    // Head 0's KV was left untouched, so head 0 must be unchanged.
    if perturbed[..du] != gqa_bits[..du] {
        return Err(anyhow!(
            "perturbing KV heads 1.. changed head 0's output, so head 0 is reading a KV head it              does not own"
        ));
    }
    // Heads 1.. owned the perturbed KV, so every one of them must have moved.
    let mut unchanged_heads = Vec::new();
    for h in 1..heads {
        let lo = h * du;
        let hi = lo + du;
        if perturbed[lo..hi] == gqa_bits[lo..hi] {
            unchanged_heads.push(h);
        }
    }
    if !unchanged_heads.is_empty() {
        return Err(anyhow!(
            "head(s) {:?} did not change when their own KV head was perturbed, so the kernel is              not honouring the head->KV mapping and the equality above was vacuous",
            unchanged_heads
        ));
    }
    println!(
        "  mha-equivalence: node {node_id}, N={n} D={d} heads={heads} splits={num_splits} \
blocks={nblocks}: {} f32 lanes bitwise identical to the validated GQA kernel ({nonzero} non-zero)",
        gqa_bits.len()
    );
    println!(
        "  mha-head-mapping: perturbing KV heads 1..{} moved all {} of them and left head 0 \
untouched, so head h reads KV head h",
        heads,
        heads - 1
    );
    Ok(())
}

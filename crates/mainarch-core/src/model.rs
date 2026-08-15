//! Multi-layer Qwen3-235B-A22B decode model assembled from the raw-ABI
//! primitives — the end-to-end decode loop, no ROCm:
//!
//!   embedding → N × decoder layer (per-layer growing KV cache) → final RMSNorm
//!   → LM head → greedy argmax → next token.
//!
//! Each decoder layer is the validated block (RMSNorm → QKV → per-head QK-norm →
//! RoPE θ=5e6 → GQA attention → O-proj → residual → RMSNorm → MoE FFN → residual)
//! driven through `layer_forward`. Depth / vocab / expert-count are reduced here
//! to bound host RAM and the f64 oracle; the head dims and the per-op math are the
//! real model's. Correctness is checked per decode step: the full-model logits are
//! compared (rel-L2) to an f64 reference that follows the GPU's generated token
//! sequence AND its per-layer expert selections (so MoE tie-breaks never confound
//! the math check — the router itself is validated separately in `layer::check_moe_ffn`).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use anyhow::{anyhow, Result};
use std::time::{Duration, Instant};

use crate::attn::{enqueue_combine_decode_gqa_f16, fp4_decode, quantize_fp4_blocks, split_count};
use crate::gemm::{f16_to_f32, f32_to_f16};
use crate::gpu::GpuDevice;

#[derive(Clone, Copy)]
struct Dims {
    h: u32,
    nh: u32,
    nkv: u32,
    d: u32,
    i: u32,
    e: u32,
    topk: u32,
    lmax: u32,
    vocab: u32,
    theta: f32,
    eps: f32,
}
impl Dims {
    fn qd(&self) -> u32 {
        self.nh * self.d
    }
    fn kvd(&self) -> u32 {
        self.nkv * self.d
    }
    fn grp(&self) -> u32 {
        self.nh / self.nkv
    }
}

#[derive(Clone, Copy)]
struct LayerVas {
    in_norm: u64,
    wq: u64,
    wk: u64,
    wv: u64,
    q_norm: u64,
    k_norm: u64,
    wo: u64,
    post_norm: u64,
    wgate: u64,
    gate: u64,
    up: u64,
    down: u64,
}

#[derive(Clone, Copy)]
struct Scratch {
    hn: u64,
    qf32: u64,
    kf32: u64,
    vf32: u64,
    qf16: u64,
    a16: u64,
    oproj: u64,
    hi: u64,
    moe: u64,
    ids_cur: u64,
    rw_cur: u64,
    part: u64,
    inter: u64,
}

#[derive(Clone, Copy)]
struct PagedMeta {
    block_table: u64,
    indices: u64,
    indptr: u64,
    last_page_len: u64,
    batch_indices: u64,
    positions: u64,
    seq_lens: u64,
    step: u64,
    block_size: u32,
    logical_blocks: u32,
    physical_blocks: u32,
    rows_per_head: u32,
}

#[derive(Clone, Copy)]
struct LayerFp4Kv {
    k: u64,
    v: u64,
    sk: u64,
    sv: u64,
}

const PROOF_NLAYERS: usize = 4;
const PROOF_PREFILL: usize = 512;
const PROOF_TGEN: usize = 4;
const PROOF_BLOCK_SIZE: usize = 16;
const PROOF_PHYSICAL_BLOCKS: usize = 132;
const PROOF_SEED_TOKEN: u32 = 7;
const PROOF_EXPECTED_TOKENS: [u32; PROOF_TGEN + 1] = [7, 46, 51, 90, 1];

fn proof_dims() -> Dims {
    Dims {
        h: 4096,
        nh: 64,
        nkv: 4,
        d: 128,
        i: 512,
        e: 16,
        topk: 8,
        lmax: 516,
        vocab: 4096,
        theta: 5_000_000.0,
        eps: 1e-6,
    }
}

struct CachedLayerFp4Kv {
    k: crate::DeviceBuffer,
    v: crate::DeviceBuffer,
    sk: crate::DeviceBuffer,
    sv: crate::DeviceBuffer,
}

impl CachedLayerFp4Kv {
    fn vas(&self) -> LayerFp4Kv {
        LayerFp4Kv {
            k: self.k.va(),
            v: self.v.va(),
            sk: self.sk.va(),
            sv: self.sv.va(),
        }
    }
}

struct CachedLayerFp4KvInitial {
    k: Vec<u8>,
    v: Vec<u8>,
    sk: Vec<u8>,
    sv: Vec<u8>,
}

pub struct CachedModelDecodeProofState {
    block_table: Vec<u32>,
    last: u32,
    pos: u32,
    seq_len: u32,
    step: u32,
    tokens: Vec<u32>,
}

impl CachedModelDecodeProofState {
    pub fn byte_len(&self) -> usize {
        4 * std::mem::size_of::<u32>()
            + self.block_table.len() * std::mem::size_of::<u32>()
            + self.tokens.len() * std::mem::size_of::<u32>()
    }

    fn validate_resume_metadata(
        &self,
        block_size: u32,
        logical_blocks: u32,
        lmax: u32,
    ) -> Result<()> {
        if self.block_table.len() != logical_blocks as usize {
            return Err(anyhow!(
                "cached model-decode state block-table length mismatch: got {}, expected {}",
                self.block_table.len(),
                logical_blocks
            ));
        }
        if self.tokens.len() != PROOF_TGEN + 1 {
            return Err(anyhow!(
                "cached model-decode state token history length mismatch"
            ));
        }
        if self.tokens[0] != PROOF_SEED_TOKEN {
            return Err(anyhow!(
                "cached model-decode state seed token mismatch: got {}, expected {}",
                self.tokens[0],
                PROOF_SEED_TOKEN
            ));
        }
        if self.last == 0 || self.last > block_size {
            return Err(anyhow!(
                "cached model-decode state last_page_len {} outside 1..={}",
                self.last,
                block_size
            ));
        }
        if self.seq_len == 0 || self.seq_len > lmax {
            return Err(anyhow!(
                "cached model-decode state seq_len {} outside 1..={}",
                self.seq_len,
                lmax
            ));
        }
        if self.pos >= lmax {
            return Err(anyhow!(
                "cached model-decode state position {} exceeds max position {}",
                self.pos,
                lmax.saturating_sub(1)
            ));
        }
        let expected_last = {
            let rem = self.seq_len % block_size;
            if rem == 0 {
                block_size
            } else {
                rem
            }
        };
        if self.last != expected_last {
            return Err(anyhow!(
                "cached model-decode state last_page_len mismatch: got {}, expected {} for seq_len {} and block_size {}",
                self.last,
                expected_last,
                self.seq_len,
                block_size
            ));
        }
        let needed_blocks = self.seq_len.div_ceil(block_size) as usize;
        if needed_blocks > self.block_table.len() {
            return Err(anyhow!(
                "cached model-decode state seq_len {} needs {} blocks but block table has {}",
                self.seq_len,
                needed_blocks,
                self.block_table.len()
            ));
        }
        if self.step as usize > PROOF_TGEN {
            return Err(anyhow!(
                "cached model-decode state step {} exceeds proof generation length {}",
                self.step,
                PROOF_TGEN
            ));
        }
        Ok(())
    }
}

pub struct CachedModelDecodeProof {
    node_id: u32,
    dm: Dims,
    lvs: Vec<LayerVas>,
    kv4s: Vec<CachedLayerFp4Kv>,
    kv_initial: Vec<CachedLayerFp4KvInitial>,
    _keep: Vec<crate::DeviceBuffer>,
    _block_table_buf: crate::DeviceBuffer,
    _indptr_buf: crate::DeviceBuffer,
    last_buf: crate::DeviceBuffer,
    _batch_buf: crate::DeviceBuffer,
    pos_buf: crate::DeviceBuffer,
    seq_len_buf: crate::DeviceBuffer,
    step_buf: crate::DeviceBuffer,
    pm: PagedMeta,
    embed_buf: crate::DeviceBuffer,
    lmhead_va: u64,
    finalnorm_va: u64,
    h_buf: crate::DeviceBuffer,
    hf_va: u64,
    logits_buf: crate::DeviceBuffer,
    tokens_buf: crate::DeviceBuffer,
    ids_buf: crate::DeviceBuffer,
    rw_buf: crate::DeviceBuffer,
    sc: Scratch,
}

pub struct CachedModelDecodeProofReport {
    pub gpu_us: f64,
    pub gpu_us_per_token: f64,
    pub tokens: Vec<u32>,
    pub generated_steps: usize,
    pub stopped_early: bool,
    pub stop_reason: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct CachedModelDecodeProofStep {
    pub step_index: usize,
    pub token: u32,
    pub gpu_us: f64,
    pub gpu_us_per_layer: f64,
}

pub enum CachedModelDecodeProofStepDecision {
    Continue,
    Stop { reason: String },
}

impl CachedModelDecodeProof {
    pub fn new(dev: &mut GpuDevice) -> Result<Self> {
        let node_id = dev.node_id();
        let dm = proof_dims();
        let (h, nh, nkv, d) = (
            dm.h as usize,
            dm.nh as usize,
            dm.nkv as usize,
            dm.d as usize,
        );
        let (ii, e, topk, vocab, lmax) = (
            dm.i as usize,
            dm.e as usize,
            dm.topk as usize,
            dm.vocab as usize,
            dm.lmax as usize,
        );
        let qd = nh * d;
        let kvd = nkv * d;
        let logical_blocks = lmax.div_ceil(PROOF_BLOCK_SIZE);
        let physical_blocks = PROOF_PHYSICAL_BLOCKS.max(logical_blocks + 1);
        let rows_per_head = physical_blocks * PROOF_BLOCK_SIZE;
        let gen = |a: usize, s: usize| {
            ((a.wrapping_mul(2654435761)
                .wrapping_add(s.wrapping_mul(40503))
                >> 11)
                & 0xff) as f32
                / 256.0
                - 0.5
        };
        let mk = |n: usize, seed: usize, sc: f32, bias: f32| -> Vec<u16> {
            (0..n)
                .map(|x| f32_to_f16(gen(x, seed) * sc + bias))
                .collect()
        };
        let up_dev = |data: &[u16],
                      dev: &mut GpuDevice,
                      keep: &mut Vec<crate::DeviceBuffer>|
         -> Result<u64> {
            let mut b = dev.alloc_device(data.len() * 2)?;
            unsafe { b.as_mut_slice_of::<u16>()[..data.len()].copy_from_slice(data) };
            let va = b.va();
            keep.push(b);
            Ok(va)
        };
        let up_bytes_buf = |data: &[u8], dev: &mut GpuDevice| -> Result<crate::DeviceBuffer> {
            let mut b = dev.alloc_device(data.len())?;
            unsafe { b.as_mut_slice()[..data.len()].copy_from_slice(data) };
            Ok(b)
        };

        let mut keep = Vec::new();
        let mut lvs = Vec::with_capacity(PROOF_NLAYERS);
        let mut kv4s = Vec::with_capacity(PROOF_NLAYERS);
        let mut kv_initial = Vec::with_capacity(PROOF_NLAYERS);
        for layer in 0..PROOF_NLAYERS {
            let s0 = layer * 100;
            let lv = LayerVas {
                in_norm: up_dev(&mk(h, s0 + 1, 0.0, 1.0), dev, &mut keep)?,
                wq: up_dev(&mk(qd * h, s0 + 2, 0.02, 0.0), dev, &mut keep)?,
                wk: up_dev(&mk(kvd * h, s0 + 3, 0.02, 0.0), dev, &mut keep)?,
                wv: up_dev(&mk(kvd * h, s0 + 4, 0.02, 0.0), dev, &mut keep)?,
                q_norm: up_dev(&mk(d, s0 + 5, 0.0, 1.0), dev, &mut keep)?,
                k_norm: up_dev(&mk(d, s0 + 6, 0.0, 1.0), dev, &mut keep)?,
                wo: up_dev(&mk(h * qd, s0 + 7, 0.02, 0.0), dev, &mut keep)?,
                post_norm: up_dev(&mk(h, s0 + 8, 0.0, 1.0), dev, &mut keep)?,
                wgate: up_dev(&mk(e * h, s0 + 9, 0.02, 0.0), dev, &mut keep)?,
                gate: up_dev(&mk(e * ii * h, s0 + 10, 0.05, 0.0), dev, &mut keep)?,
                up: up_dev(&mk(e * ii * h, s0 + 11, 0.05, 0.0), dev, &mut keep)?,
                down: up_dev(&mk(e * h * ii, s0 + 12, 0.05, 0.0), dev, &mut keep)?,
            };

            let mut kc16 = vec![0u16; nkv * lmax * d];
            let mut vc16 = vec![0u16; nkv * lmax * d];
            for head in 0..nkv {
                for t in 0..PROOF_PREFILL {
                    for x in 0..d {
                        kc16[(head * lmax + t) * d + x] =
                            f32_to_f16(gen((head * lmax + t) * d + x, s0 + 21));
                        vc16[(head * lmax + t) * d + x] =
                            f32_to_f16(gen((head * lmax + t) * d + x, s0 + 22));
                    }
                }
            }
            let mut k4_all = vec![0u8; nkv * rows_per_head * 64];
            let mut v4_all = vec![0u8; nkv * rows_per_head * 64];
            let mut ksc_all = vec![127u8; nkv * rows_per_head * 4];
            let mut vsc_all = vec![127u8; nkv * rows_per_head * 4];
            for head in 0..nkv {
                let mut kf = vec![0f32; rows_per_head * d];
                let mut vf = vec![0f32; rows_per_head * d];
                for t in 0..lmax {
                    let phys_t =
                        (t / PROOF_BLOCK_SIZE + 1) * PROOF_BLOCK_SIZE + (t % PROOF_BLOCK_SIZE);
                    for x in 0..d {
                        let src = (head * lmax + t) * d + x;
                        kf[phys_t * d + x] = f16_to_f32(kc16[src]);
                        vf[phys_t * d + x] = f16_to_f32(vc16[src]);
                    }
                }
                let (k4, ksc) = quantize_fp4_blocks(&kf, rows_per_head);
                let (v4, vsc) = quantize_fp4_blocks(&vf, rows_per_head);
                let row0 = head * rows_per_head;
                k4_all[row0 * 64..(row0 + rows_per_head) * 64].copy_from_slice(&k4);
                v4_all[row0 * 64..(row0 + rows_per_head) * 64].copy_from_slice(&v4);
                ksc_all[row0 * 4..(row0 + rows_per_head) * 4].copy_from_slice(&ksc);
                vsc_all[row0 * 4..(row0 + rows_per_head) * 4].copy_from_slice(&vsc);
            }
            let kv = CachedLayerFp4Kv {
                k: up_bytes_buf(&k4_all, dev)?,
                v: up_bytes_buf(&v4_all, dev)?,
                sk: up_bytes_buf(&ksc_all, dev)?,
                sv: up_bytes_buf(&vsc_all, dev)?,
            };
            kv_initial.push(CachedLayerFp4KvInitial {
                k: k4_all,
                v: v4_all,
                sk: ksc_all,
                sv: vsc_all,
            });
            lvs.push(lv);
            kv4s.push(kv);
        }

        let block_table: Vec<u32> = (0..logical_blocks).map(|i| (i + 1) as u32).collect();
        let indptr: Vec<u32> = vec![0, logical_blocks as u32];
        let last_len = if lmax % PROOF_BLOCK_SIZE == 0 {
            PROOF_BLOCK_SIZE
        } else {
            lmax % PROOF_BLOCK_SIZE
        };
        let last_page_len: Vec<u32> = vec![last_len as u32];
        let batch_indices: Vec<u32> = vec![0];
        let mut block_table_buf = dev.alloc_device(block_table.len() * 4)?;
        let mut indptr_buf = dev.alloc_device(indptr.len() * 4)?;
        let mut last_buf = dev.alloc_device(last_page_len.len() * 4)?;
        let mut batch_buf = dev.alloc_device(batch_indices.len() * 4)?;
        let mut pos_buf = dev.alloc_device(4)?;
        let mut seq_len_buf = dev.alloc_device(4)?;
        let mut step_buf = dev.alloc_device(4)?;
        unsafe {
            block_table_buf.as_mut_slice_of::<u32>()[..block_table.len()]
                .copy_from_slice(&block_table);
            indptr_buf.as_mut_slice_of::<u32>()[..indptr.len()].copy_from_slice(&indptr);
            last_buf.as_mut_slice_of::<u32>()[..last_page_len.len()]
                .copy_from_slice(&last_page_len);
            batch_buf.as_mut_slice_of::<u32>()[..batch_indices.len()]
                .copy_from_slice(&batch_indices);
            pos_buf.as_mut_slice_of::<u32>()[0] = PROOF_PREFILL as u32;
            seq_len_buf.as_mut_slice_of::<u32>()[0] = PROOF_PREFILL as u32;
            step_buf.as_mut_slice_of::<u32>()[0] = 0;
        }
        let pm = PagedMeta {
            block_table: block_table_buf.va(),
            indices: block_table_buf.va(),
            indptr: indptr_buf.va(),
            last_page_len: last_buf.va(),
            batch_indices: batch_buf.va(),
            positions: pos_buf.va(),
            seq_lens: seq_len_buf.va(),
            step: step_buf.va(),
            block_size: PROOF_BLOCK_SIZE as u32,
            logical_blocks: logical_blocks as u32,
            physical_blocks: physical_blocks as u32,
            rows_per_head: rows_per_head as u32,
        };
        dev.check_paged_block_table(pm.block_table, pm.logical_blocks, pm.physical_blocks)?;
        dev.check_paged_kv_metadata(
            pm.indptr,
            pm.indices,
            pm.last_page_len,
            1,
            pm.logical_blocks,
            pm.physical_blocks,
            pm.block_size,
        )?;

        let embed16 = mk(vocab * h, 7001, 0.05, 0.0);
        let lmhead16 = mk(vocab * h, 7002, 0.05, 0.0);
        let finalnorm16 = mk(h, 7003, 0.0, 1.0);
        let mut embed_buf = dev.alloc_device(vocab * h * 2)?;
        let lmhead_va = up_dev(&lmhead16, dev, &mut keep)?;
        let finalnorm_va = up_dev(&finalnorm16, dev, &mut keep)?;
        unsafe { embed_buf.as_mut_slice_of::<u16>()[..vocab * h].copy_from_slice(&embed16) };

        let h_buf = dev.alloc_device(h * 2)?;
        let hf_va = up_dev(&vec![0u16; h], dev, &mut keep)?;
        let logits_buf = dev.alloc_device(PROOF_TGEN * vocab * 4)?;
        let tokens_buf = dev.alloc_device((PROOF_TGEN + 1) * 4)?;
        let ids_buf = dev.alloc_device(PROOF_TGEN * PROOF_NLAYERS * topk * 4)?;
        let rw_buf = dev.alloc_device(PROOF_TGEN * PROOF_NLAYERS * topk * 4)?;
        let ns = split_count(dm.lmax);
        let sc = Scratch {
            hn: up_dev(&vec![0u16; h], dev, &mut keep)?,
            qf32: up_dev(&vec![0u16; qd * 2], dev, &mut keep)?,
            kf32: up_dev(&vec![0u16; kvd * 2], dev, &mut keep)?,
            vf32: up_dev(&vec![0u16; kvd * 2], dev, &mut keep)?,
            qf16: up_dev(&vec![0u16; qd], dev, &mut keep)?,
            a16: up_dev(&vec![0u16; qd], dev, &mut keep)?,
            oproj: up_dev(&vec![0u16; h * 2], dev, &mut keep)?,
            hi: up_dev(&vec![0u16; topk * ii], dev, &mut keep)?,
            moe: up_dev(&vec![0u16; h * 2], dev, &mut keep)?,
            ids_cur: up_dev(&vec![0u16; topk * 2], dev, &mut keep)?,
            rw_cur: up_dev(&vec![0u16; topk * 2], dev, &mut keep)?,
            part: up_dev(&vec![0u16; nh * ns as usize * (d + 2) * 2], dev, &mut keep)?,
            inter: up_dev(
                &vec![0u16; nh * crate::attn::combine_groups(ns).max(1) as usize * (d + 2) * 2],
                dev,
                &mut keep,
            )?,
        };

        Ok(Self {
            node_id,
            dm,
            lvs,
            kv4s,
            kv_initial,
            _keep: keep,
            _block_table_buf: block_table_buf,
            _indptr_buf: indptr_buf,
            last_buf,
            _batch_buf: batch_buf,
            pos_buf,
            seq_len_buf,
            step_buf,
            pm,
            embed_buf,
            lmhead_va,
            finalnorm_va,
            h_buf,
            hf_va,
            logits_buf,
            tokens_buf,
            ids_buf,
            rw_buf,
            sc,
        })
    }

    fn reset_request_metadata_state(&mut self) {
        unsafe {
            self.last_buf.as_mut_slice_of::<u32>()[0] = PROOF_BLOCK_SIZE as u32;
            self.pos_buf.as_mut_slice_of::<u32>()[0] = PROOF_PREFILL as u32;
            self.seq_len_buf.as_mut_slice_of::<u32>()[0] = PROOF_PREFILL as u32;
            self.step_buf.as_mut_slice_of::<u32>()[0] = 0;
            self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1].fill(0);
            self.tokens_buf.as_mut_slice_of::<u32>()[0] = PROOF_SEED_TOKEN;
            self.logits_buf.as_mut_slice_of::<f32>()[..PROOF_TGEN * self.dm.vocab as usize]
                .fill(0.0);
            self.ids_buf.as_mut_slice_of::<u32>()
                [..PROOF_TGEN * PROOF_NLAYERS * self.dm.topk as usize]
                .fill(0);
            self.rw_buf.as_mut_slice_of::<f32>()
                [..PROOF_TGEN * PROOF_NLAYERS * self.dm.topk as usize]
                .fill(0.0);
            for (idx, slot) in self._block_table_buf.as_mut_slice_of::<u32>()
                [..self.pm.logical_blocks as usize]
                .iter_mut()
                .enumerate()
            {
                *slot = (idx + 1) as u32;
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    fn reset_request_state(&mut self) {
        unsafe {
            for (kv, init) in self.kv4s.iter_mut().zip(&self.kv_initial) {
                kv.k.as_mut_slice()[..init.k.len()].copy_from_slice(&init.k);
                kv.v.as_mut_slice()[..init.v.len()].copy_from_slice(&init.v);
                kv.sk.as_mut_slice()[..init.sk.len()].copy_from_slice(&init.sk);
                kv.sv.as_mut_slice()[..init.sv.len()].copy_from_slice(&init.sv);
            }
        }
        self.reset_request_metadata_state();
    }

    pub fn run(&mut self, dev: &mut GpuDevice) -> Result<CachedModelDecodeProofReport> {
        self.reset_request_state();
        dev.check_paged_block_table(
            self.pm.block_table,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
        )?;
        dev.check_paged_kv_metadata(
            self.pm.indptr,
            self.pm.indices,
            self.pm.last_page_len,
            1,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
            self.pm.block_size,
        )?;
        let metadata_guard = std::env::var("MAINARCH_DECODE_METADATA_GUARD")
            .map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("0")
                    || v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false);

        let to = Duration::from_secs(20);
        let gpu_start = Instant::now();
        for t in 0..PROOF_TGEN {
            if !metadata_guard {
                dev.chain_next();
            }
            dev.arm_decode_step_embed_rmsnorm_token(
                self.embed_buf.va(),
                self.tokens_buf.va(),
                self.lvs[0].in_norm,
                self.h_buf.va(),
                self.sc.hn,
                self.pm.step,
                self.pm.positions,
                self.pm.seq_lens,
                self.pm.last_page_len,
                PROOF_PREFILL as u32,
                self.pm.block_size,
                (PROOF_TGEN + 1) as u32,
                self.dm.vocab,
                self.dm.h,
                self.dm.eps,
            )?;
            if metadata_guard {
                dev.wait(to)?;
                dev.check_paged_kv_metadata(
                    self.pm.indptr,
                    self.pm.indices,
                    self.pm.last_page_len,
                    1,
                    self.pm.logical_blocks,
                    self.pm.physical_blocks,
                    self.pm.block_size,
                )?;
            }
            for layer in 0..PROOF_NLAYERS {
                let (tail_norm_w, tail_norm_out) = if layer + 1 < PROOF_NLAYERS {
                    (self.lvs[layer + 1].in_norm, self.sc.hn)
                } else {
                    (self.finalnorm_va, self.hf_va)
                };
                layer_forward_normed(
                    dev,
                    &self.lvs[layer],
                    &self.kv4s[layer].vas(),
                    self.pm,
                    &self.sc,
                    self.h_buf.va(),
                    self.ids_buf.va(),
                    self.rw_buf.va(),
                    layer as u32,
                    PROOF_NLAYERS as u32,
                    PROOF_TGEN as u32,
                    tail_norm_w,
                    tail_norm_out,
                    self.dm,
                )?;
            }
            dev.chain_next();
            dev.arm_gemv_step(
                self.lmhead_va,
                self.hf_va,
                self.logits_buf.va(),
                self.pm.step,
                PROOF_TGEN as u32,
                self.dm.vocab,
                self.dm.h,
            )?;
            if !metadata_guard && t + 1 < PROOF_TGEN {
                dev.chain_next();
            }
            dev.arm_argmax_f32_step(
                self.logits_buf.va(),
                self.tokens_buf.va(),
                self.pm.step,
                (PROOF_TGEN + 1) as u32,
                self.dm.vocab,
            )?;
            if metadata_guard {
                dev.wait(to)?;
            }
        }
        if !metadata_guard {
            dev.wait(to)?;
        }
        let gpu_us = gpu_start.elapsed().as_secs_f64() * 1e6;
        let token_hist =
            unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1].to_vec() };
        if token_hist.as_slice() != PROOF_EXPECTED_TOKENS {
            return Err(anyhow!(
                "cached model-decode token trace mismatch: got {:?}, expected {:?}",
                token_hist,
                PROOF_EXPECTED_TOKENS
            ));
        }
        println!(
            "  model-decode cached-runner: Qwen3-235B-A22B arch (H={}, {}L, {}Q/{}KV d{}, paged FP4 KV, MoE {}×top{}, vocab {}, prefill {}, gen {}) on node {}",
            self.dm.h,
            PROOF_NLAYERS,
            self.dm.nh,
            self.dm.nkv,
            self.dm.d,
            self.dm.e,
            self.dm.topk,
            self.dm.vocab,
            PROOF_PREFILL,
            PROOF_TGEN,
            self.node_id
        );
        println!(
            "    - correct: cached greedy token trace {:?}",
            &token_hist[1..]
        );
        println!(
            "    - GPU decode: {:.0} µs/token ({} layers, {:.0} µs/layer)",
            gpu_us / PROOF_TGEN as f64,
            PROOF_NLAYERS,
            gpu_us / (PROOF_TGEN * PROOF_NLAYERS) as f64
        );
        Ok(CachedModelDecodeProofReport {
            gpu_us,
            gpu_us_per_token: gpu_us / PROOF_TGEN as f64,
            tokens: token_hist,
            generated_steps: PROOF_TGEN,
            stopped_early: false,
            stop_reason: None,
        })
    }

    pub fn run_stepwise<F>(
        &mut self,
        dev: &mut GpuDevice,
        mut on_step: F,
    ) -> Result<CachedModelDecodeProofReport>
    where
        F: FnMut(CachedModelDecodeProofStep) -> Result<()>,
    {
        self.run_stepwise_controlled(
            dev,
            |_| Ok(CachedModelDecodeProofStepDecision::Continue),
            |step| {
                on_step(step)?;
                Ok(CachedModelDecodeProofStepDecision::Continue)
            },
        )
    }

    pub fn begin_stepwise_request(&mut self, dev: &mut GpuDevice) -> Result<()> {
        self.reset_request_state();
        dev.check_paged_block_table(
            self.pm.block_table,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
        )?;
        dev.check_paged_kv_metadata(
            self.pm.indptr,
            self.pm.indices,
            self.pm.last_page_len,
            1,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
            self.pm.block_size,
        )?;
        Ok(())
    }

    pub fn begin_stepwise_request_with_block_table(
        &mut self,
        dev: &mut GpuDevice,
        block_table: &[u32],
    ) -> Result<usize> {
        self.reset_request_metadata_state();
        let initialized_pages = self.install_stepwise_block_table(block_table)?;
        dev.check_paged_block_table(
            self.pm.block_table,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
        )?;
        dev.check_paged_kv_metadata(
            self.pm.indptr,
            self.pm.indices,
            self.pm.last_page_len,
            1,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
            self.pm.block_size,
        )?;
        Ok(initialized_pages)
    }

    fn install_stepwise_block_table(&mut self, block_table: &[u32]) -> Result<usize> {
        let logical_blocks = self.pm.logical_blocks as usize;
        let physical_blocks = self.pm.physical_blocks as usize;
        if block_table.len() != logical_blocks {
            return Err(anyhow!(
                "cached model-decode block-table length mismatch: got {}, expected {}",
                block_table.len(),
                logical_blocks
            ));
        }
        let mut seen = vec![false; physical_blocks];
        for (idx, &block_id) in block_table.iter().enumerate() {
            let block_id = block_id as usize;
            if block_id >= physical_blocks {
                return Err(anyhow!(
                    "cached model-decode block-table entry out of range: logical block {} -> physical block {} (capacity {})",
                    idx,
                    block_id,
                    physical_blocks
                ));
            }
            if seen[block_id] {
                return Err(anyhow!(
                    "cached model-decode block-table reuses physical block {} before copy-on-write support",
                    block_id
                ));
            }
            seen[block_id] = true;
        }
        unsafe {
            self._block_table_buf.as_mut_slice_of::<u32>()[..logical_blocks]
                .copy_from_slice(block_table);
            let rows_per_head = self.pm.rows_per_head as usize;
            let block_size = self.pm.block_size as usize;
            let nkv = self.dm.nkv as usize;
            // Scheduler-owned admission writes the synthetic prefill image
            // straight into the assigned physical pages. We intentionally do
            // not reset the whole canonical KV pool first; stale pages are
            // overwritten only for the blocks owned by this request.
            for (logical_block, &physical_block) in block_table.iter().enumerate() {
                let canonical_block = logical_block + 1;
                let physical_block = physical_block as usize;
                for (kv, init) in self.kv4s.iter_mut().zip(&self.kv_initial) {
                    write_initial_kv_block_to_physical(
                        &init.k,
                        kv.k.as_mut_slice(),
                        nkv,
                        rows_per_head,
                        block_size,
                        canonical_block,
                        physical_block,
                        64,
                    );
                    write_initial_kv_block_to_physical(
                        &init.v,
                        kv.v.as_mut_slice(),
                        nkv,
                        rows_per_head,
                        block_size,
                        canonical_block,
                        physical_block,
                        64,
                    );
                    write_initial_kv_block_to_physical(
                        &init.sk,
                        kv.sk.as_mut_slice(),
                        nkv,
                        rows_per_head,
                        block_size,
                        canonical_block,
                        physical_block,
                        4,
                    );
                    write_initial_kv_block_to_physical(
                        &init.sv,
                        kv.sv.as_mut_slice(),
                        nkv,
                        rows_per_head,
                        block_size,
                        canonical_block,
                        physical_block,
                        4,
                    );
                }
            }
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(logical_blocks)
    }

    pub fn capture_stepwise_state(&mut self) -> CachedModelDecodeProofState {
        unsafe {
            CachedModelDecodeProofState {
                block_table: self._block_table_buf.as_mut_slice_of::<u32>()
                    [..self.pm.logical_blocks as usize]
                    .to_vec(),
                last: self.last_buf.as_mut_slice_of::<u32>()[0],
                pos: self.pos_buf.as_mut_slice_of::<u32>()[0],
                seq_len: self.seq_len_buf.as_mut_slice_of::<u32>()[0],
                step: self.step_buf.as_mut_slice_of::<u32>()[0],
                tokens: self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1].to_vec(),
            }
        }
    }

    pub fn capture_stepwise_state_into(
        &mut self,
        state: &mut CachedModelDecodeProofState,
    ) -> Result<()> {
        unsafe {
            if self._block_table_buf.as_mut_slice_of::<u32>().len() < state.block_table.len()
                || state.block_table.len() != self.pm.logical_blocks as usize
            {
                return Err(anyhow!(
                    "cached model-decode state block-table length mismatch"
                ));
            }
            state.block_table.copy_from_slice(
                &self._block_table_buf.as_mut_slice_of::<u32>()[..self.pm.logical_blocks as usize],
            );
            state.last = self.last_buf.as_mut_slice_of::<u32>()[0];
            state.pos = self.pos_buf.as_mut_slice_of::<u32>()[0];
            state.seq_len = self.seq_len_buf.as_mut_slice_of::<u32>()[0];
            state.step = self.step_buf.as_mut_slice_of::<u32>()[0];
            if state.tokens.len() != PROOF_TGEN + 1 {
                return Err(anyhow!(
                    "cached model-decode state trace buffer length mismatch"
                ));
            }
            state
                .tokens
                .copy_from_slice(&self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1]);
        }
        Ok(())
    }

    pub fn restore_stepwise_state(&mut self, state: &CachedModelDecodeProofState) -> Result<()> {
        state.validate_resume_metadata(self.pm.block_size, self.pm.logical_blocks, self.dm.lmax)?;
        unsafe {
            self._block_table_buf.as_mut_slice_of::<u32>()[..state.block_table.len()]
                .copy_from_slice(&state.block_table);
            self.last_buf.as_mut_slice_of::<u32>()[0] = state.last;
            self.pos_buf.as_mut_slice_of::<u32>()[0] = state.pos;
            self.seq_len_buf.as_mut_slice_of::<u32>()[0] = state.seq_len;
            self.step_buf.as_mut_slice_of::<u32>()[0] = state.step;
            self.tokens_buf.as_mut_slice_of::<u32>()[..state.tokens.len()]
                .copy_from_slice(&state.tokens);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(())
    }

    pub fn run_stepwise_next(
        &mut self,
        dev: &mut GpuDevice,
        step_index: usize,
    ) -> Result<CachedModelDecodeProofStep> {
        if step_index >= PROOF_TGEN {
            return Err(anyhow!(
                "cached model-decode step {} exceeds proof generation length {}",
                step_index,
                PROOF_TGEN
            ));
        }
        let metadata_guard = std::env::var("MAINARCH_DECODE_METADATA_GUARD")
            .map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("0")
                    || v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false);

        let to = Duration::from_secs(20);
        let step_start = Instant::now();
        if !metadata_guard {
            dev.chain_next();
        }
        dev.arm_decode_step_embed_rmsnorm_token(
            self.embed_buf.va(),
            self.tokens_buf.va(),
            self.lvs[0].in_norm,
            self.h_buf.va(),
            self.sc.hn,
            self.pm.step,
            self.pm.positions,
            self.pm.seq_lens,
            self.pm.last_page_len,
            PROOF_PREFILL as u32,
            self.pm.block_size,
            (PROOF_TGEN + 1) as u32,
            self.dm.vocab,
            self.dm.h,
            self.dm.eps,
        )?;
        if metadata_guard {
            dev.wait(to)?;
            dev.check_paged_kv_metadata(
                self.pm.indptr,
                self.pm.indices,
                self.pm.last_page_len,
                1,
                self.pm.logical_blocks,
                self.pm.physical_blocks,
                self.pm.block_size,
            )?;
        }
        for layer in 0..PROOF_NLAYERS {
            let (tail_norm_w, tail_norm_out) = if layer + 1 < PROOF_NLAYERS {
                (self.lvs[layer + 1].in_norm, self.sc.hn)
            } else {
                (self.finalnorm_va, self.hf_va)
            };
            layer_forward_normed(
                dev,
                &self.lvs[layer],
                &self.kv4s[layer].vas(),
                self.pm,
                &self.sc,
                self.h_buf.va(),
                self.ids_buf.va(),
                self.rw_buf.va(),
                layer as u32,
                PROOF_NLAYERS as u32,
                PROOF_TGEN as u32,
                tail_norm_w,
                tail_norm_out,
                self.dm,
            )?;
        }
        dev.chain_next();
        dev.arm_gemv_step(
            self.lmhead_va,
            self.hf_va,
            self.logits_buf.va(),
            self.pm.step,
            PROOF_TGEN as u32,
            self.dm.vocab,
            self.dm.h,
        )?;
        dev.arm_argmax_f32_step(
            self.logits_buf.va(),
            self.tokens_buf.va(),
            self.pm.step,
            (PROOF_TGEN + 1) as u32,
            self.dm.vocab,
        )?;
        dev.wait(to)?;
        let step_us = step_start.elapsed().as_secs_f64() * 1e6;
        let token = unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[step_index + 1] };
        if token != PROOF_EXPECTED_TOKENS[step_index + 1] {
            let token_hist =
                unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1].to_vec() };
            return Err(anyhow!(
                "cached model-decode step {} token mismatch: got {}, expected {}; trace {:?}",
                step_index,
                token,
                PROOF_EXPECTED_TOKENS[step_index + 1],
                token_hist
            ));
        }
        Ok(CachedModelDecodeProofStep {
            step_index,
            token,
            gpu_us: step_us,
            gpu_us_per_layer: step_us / PROOF_NLAYERS as f64,
        })
    }

    pub fn run_stepwise_controlled<B, F>(
        &mut self,
        dev: &mut GpuDevice,
        mut before_step: B,
        mut on_step: F,
    ) -> Result<CachedModelDecodeProofReport>
    where
        B: FnMut(usize) -> Result<CachedModelDecodeProofStepDecision>,
        F: FnMut(CachedModelDecodeProofStep) -> Result<CachedModelDecodeProofStepDecision>,
    {
        self.reset_request_state();
        dev.check_paged_block_table(
            self.pm.block_table,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
        )?;
        dev.check_paged_kv_metadata(
            self.pm.indptr,
            self.pm.indices,
            self.pm.last_page_len,
            1,
            self.pm.logical_blocks,
            self.pm.physical_blocks,
            self.pm.block_size,
        )?;
        let metadata_guard = std::env::var("MAINARCH_DECODE_METADATA_GUARD")
            .map(|v| {
                let v = v.trim();
                !(v.eq_ignore_ascii_case("0")
                    || v.eq_ignore_ascii_case("false")
                    || v.eq_ignore_ascii_case("off")
                    || v.eq_ignore_ascii_case("no"))
            })
            .unwrap_or(false);

        let to = Duration::from_secs(20);
        let mut gpu_us = 0.0;
        for t in 0..PROOF_TGEN {
            if let CachedModelDecodeProofStepDecision::Stop { reason } = before_step(t)? {
                let token_hist =
                    unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[..t + 1].to_vec() };
                return Ok(CachedModelDecodeProofReport {
                    gpu_us,
                    gpu_us_per_token: if t == 0 { 0.0 } else { gpu_us / t as f64 },
                    tokens: token_hist,
                    generated_steps: t,
                    stopped_early: true,
                    stop_reason: Some(reason),
                });
            }
            let step_start = Instant::now();
            if !metadata_guard {
                dev.chain_next();
            }
            dev.arm_decode_step_embed_rmsnorm_token(
                self.embed_buf.va(),
                self.tokens_buf.va(),
                self.lvs[0].in_norm,
                self.h_buf.va(),
                self.sc.hn,
                self.pm.step,
                self.pm.positions,
                self.pm.seq_lens,
                self.pm.last_page_len,
                PROOF_PREFILL as u32,
                self.pm.block_size,
                (PROOF_TGEN + 1) as u32,
                self.dm.vocab,
                self.dm.h,
                self.dm.eps,
            )?;
            if metadata_guard {
                dev.wait(to)?;
                dev.check_paged_kv_metadata(
                    self.pm.indptr,
                    self.pm.indices,
                    self.pm.last_page_len,
                    1,
                    self.pm.logical_blocks,
                    self.pm.physical_blocks,
                    self.pm.block_size,
                )?;
            }
            for layer in 0..PROOF_NLAYERS {
                let (tail_norm_w, tail_norm_out) = if layer + 1 < PROOF_NLAYERS {
                    (self.lvs[layer + 1].in_norm, self.sc.hn)
                } else {
                    (self.finalnorm_va, self.hf_va)
                };
                layer_forward_normed(
                    dev,
                    &self.lvs[layer],
                    &self.kv4s[layer].vas(),
                    self.pm,
                    &self.sc,
                    self.h_buf.va(),
                    self.ids_buf.va(),
                    self.rw_buf.va(),
                    layer as u32,
                    PROOF_NLAYERS as u32,
                    PROOF_TGEN as u32,
                    tail_norm_w,
                    tail_norm_out,
                    self.dm,
                )?;
            }
            dev.chain_next();
            dev.arm_gemv_step(
                self.lmhead_va,
                self.hf_va,
                self.logits_buf.va(),
                self.pm.step,
                PROOF_TGEN as u32,
                self.dm.vocab,
                self.dm.h,
            )?;
            dev.arm_argmax_f32_step(
                self.logits_buf.va(),
                self.tokens_buf.va(),
                self.pm.step,
                (PROOF_TGEN + 1) as u32,
                self.dm.vocab,
            )?;
            dev.wait(to)?;
            let step_us = step_start.elapsed().as_secs_f64() * 1e6;
            gpu_us += step_us;
            let token = unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[t + 1] };
            if token != PROOF_EXPECTED_TOKENS[t + 1] {
                let token_hist =
                    unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1].to_vec() };
                return Err(anyhow!(
                    "cached model-decode step {} token mismatch: got {}, expected {}; trace {:?}",
                    t,
                    token,
                    PROOF_EXPECTED_TOKENS[t + 1],
                    token_hist
                ));
            }
            let decision = on_step(CachedModelDecodeProofStep {
                step_index: t,
                token,
                gpu_us: step_us,
                gpu_us_per_layer: step_us / PROOF_NLAYERS as f64,
            })?;
            if let CachedModelDecodeProofStepDecision::Stop { reason } = decision {
                let generated_steps = t + 1;
                let token_hist = unsafe {
                    self.tokens_buf.as_mut_slice_of::<u32>()[..generated_steps + 1].to_vec()
                };
                return Ok(CachedModelDecodeProofReport {
                    gpu_us,
                    gpu_us_per_token: gpu_us / generated_steps as f64,
                    tokens: token_hist,
                    generated_steps,
                    stopped_early: true,
                    stop_reason: Some(reason),
                });
            }
        }
        let token_hist =
            unsafe { self.tokens_buf.as_mut_slice_of::<u32>()[..PROOF_TGEN + 1].to_vec() };
        if token_hist.as_slice() != PROOF_EXPECTED_TOKENS {
            return Err(anyhow!(
                "cached model-decode token trace mismatch: got {:?}, expected {:?}",
                token_hist,
                PROOF_EXPECTED_TOKENS
            ));
        }
        println!(
            "  model-decode cached-runner stepwise: Qwen3-235B-A22B arch (H={}, {}L, {}Q/{}KV d{}, paged FP4 KV, MoE {}×top{}, vocab {}, prefill {}, gen {}) on node {}",
            self.dm.h,
            PROOF_NLAYERS,
            self.dm.nh,
            self.dm.nkv,
            self.dm.d,
            self.dm.e,
            self.dm.topk,
            self.dm.vocab,
            PROOF_PREFILL,
            PROOF_TGEN,
            self.node_id
        );
        println!(
            "    - correct: cached stepwise greedy token trace {:?}",
            &token_hist[1..]
        );
        println!(
            "    - GPU decode stepwise: {:.0} µs/token ({} layers, {:.0} µs/layer)",
            gpu_us / PROOF_TGEN as f64,
            PROOF_NLAYERS,
            gpu_us / (PROOF_TGEN * PROOF_NLAYERS) as f64
        );
        Ok(CachedModelDecodeProofReport {
            gpu_us,
            gpu_us_per_token: gpu_us / PROOF_TGEN as f64,
            tokens: token_hist,
            generated_steps: PROOF_TGEN,
            stopped_early: false,
            stop_reason: None,
        })
    }
}

fn write_initial_kv_block_to_physical(
    src: &[u8],
    dst: &mut [u8],
    heads: usize,
    rows_per_head: usize,
    block_size: usize,
    src_block: usize,
    dst_block: usize,
    row_bytes: usize,
) {
    for head in 0..heads {
        let src_start = (head * rows_per_head + src_block * block_size) * row_bytes;
        let src_end = src_start + block_size * row_bytes;
        let dst_start = (head * rows_per_head + dst_block * block_size) * row_bytes;
        let dst_end = dst_start + block_size * row_bytes;
        dst[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
    }
}

/// One decoder layer, in place on the f16 residual stream `hva`, attending over
/// the current device-side sequence metadata. The current token's K,V are
/// appended at `positions[0]`, and attention reads `seq_lens[0]`.
/// `s.hn` must already hold this layer's input RMSNorm. The final MoE residual
/// add is fused with the next RMSNorm/final norm into `tail_norm_out`, matching
/// the existing add-then-f16-round-then-RMSNorm numerics while avoiding the
/// standalone `add_into_f16` dispatch.
fn layer_forward_normed(
    dev: &mut GpuDevice,
    w: &LayerVas,
    kv: &LayerFp4Kv,
    pm: PagedMeta,
    s: &Scratch,
    hva: u64,
    ids_hist_va: u64,
    rw_hist_va: u64,
    layer_index: u32,
    num_layers: u32,
    history_steps: u32,
    tail_norm_w: u64,
    tail_norm_out: u64,
    dm: Dims,
) -> Result<()> {
    let (h, qd, kvd, d, ii, e, topk) = (dm.h, dm.qd(), dm.kvd(), dm.d, dm.i, dm.e, dm.topk);
    let ns = split_count(dm.lmax);
    let scale = 1.0f32 / (d as f32).sqrt();

    // Each AQL packet carries the barrier bit, so chained (unsignaled) dispatches
    // execute strictly in order — we wait once per BATCH instead of per dispatch.
    // Batches stay under the kernarg/AQL-ring depth (both 64); reused part/inter
    // scratch is safe because the barrier serializes the chain.
    dev.chain_next();
    dev.arm_gemv_qkv(w.wq, w.wk, w.wv, s.hn, s.qf32, s.kf32, s.vf32, qd, kvd, h)?;
    dev.chain_next();
    dev.arm_cast_qk_rope_append_paged_fp4_vf32_q64_k4_d128_meta(
        s.qf32,
        w.q_norm,
        s.qf16,
        s.kf32,
        w.k_norm,
        s.vf32,
        kv.k,
        kv.v,
        kv.sk,
        kv.sv,
        pm.indices,
        pm.indptr,
        pm.last_page_len,
        pm.batch_indices,
        pm.positions,
        pm.logical_blocks,
        pm.physical_blocks,
        pm.block_size,
        dm.theta,
        dm.eps,
    )?;
    // Attention (fused GQA): each KV head's cache is read ONCE per group of
    // GQA_SPLIT_G query heads, not once per query head. Qwen is 64Q/4KV
    // (group 16); the old per-head split re-read the shared KV 16x and ran one
    // split+combine per head (64 heads × 3 dispatches). The batched split grid
    // below launches all GQA_SPLIT_G subgroups in one AQL packet:
    // grid_x=(nh/GQA_SPLIT_G)*num_splits, with each workgroup computing one
    // (subgroup, split) partial. The batched two-pass combine still reduces all
    // heads with a single wait.
    const GQA_SPLIT_G: u32 = 8; // must equal #define GQA_G in the kernel
    let grp = dm.grp();
    debug_assert!(
        dm.nh.is_multiple_of(GQA_SPLIT_G) && grp.is_multiple_of(GQA_SPLIT_G),
        "GQA fusion requires nh and group size to be multiples of GQA_SPLIT_G"
    );
    dev.chain_next();
    dev.arm_attn_decode_split_fp4_gqa_paged_groups_meta(
        s.qf16,
        kv.k,
        kv.v,
        kv.sk,
        kv.sv,
        pm.block_table,
        pm.block_size,
        pm.physical_blocks,
        s.part,
        pm.seq_lens,
        dm.lmax,
        d,
        scale,
        ns,
        dm.nh / GQA_SPLIT_G,
        grp,
        pm.rows_per_head,
    )?;
    // Batched two-pass combine over ALL heads, keeping f32 max/LSE reduction but
    // storing final normalized attention directly as f16 for O-proj input. This
    // stays on the AQL queue; the layer-end wait below covers combine + O-proj +
    // MoE instead of round-tripping through the host between attention and O-proj.
    enqueue_combine_decode_gqa_f16(dev, s.part, s.inter, s.a16, d, ns, dm.nh)?;
    // Post-attention + MoE, chained to the combine and then one layer-end wait.
    dev.chain_next();
    dev.arm_gemv(w.wo, s.a16, s.oproj, h, qd)?;
    dev.chain_next();
    dev.arm_add_rmsnorm(hva, s.oproj, w.post_norm, s.hn, h, dm.eps)?; // hn now holds post-attn norm
    dev.chain_next();
    dev.arm_moe_router_gemv_topk_log_step(
        w.wgate,
        s.hn,
        s.ids_cur,
        s.rw_cur,
        ids_hist_va,
        rw_hist_va,
        pm.step,
        e,
        h,
        topk,
        history_steps,
        layer_index,
        num_layers,
    )?;
    dev.chain_next();
    dev.arm_moe_gate_up_swiglu_slots(w.gate, w.up, s.hn, s.ids_cur, s.hi, topk, e, ii, h)?;
    dev.chain_next();
    dev.arm_moe_down_accum_slots(w.down, s.hi, s.ids_cur, s.rw_cur, s.moe, topk, e, h, ii)?;
    dev.chain_next();
    dev.arm_add_rmsnorm(hva, s.moe, tail_norm_w, tail_norm_out, h, dm.eps)?;
    Ok(())
}

struct LayerHost {
    in_norm: Vec<u16>,
    wq: Vec<u16>,
    wk: Vec<u16>,
    wv: Vec<u16>,
    q_norm: Vec<u16>,
    k_norm: Vec<u16>,
    wo: Vec<u16>,
    post_norm: Vec<u16>,
    wgate: Vec<u16>,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
}

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

/// f64 reference for one layer, mirroring the f16 datapath, using the GPU's
/// selected expert ids/weights. Appends the current K,V into the f64 caches at
/// `pos`, returns the updated residual.
fn layer_ref(
    lh: &LayerHost,
    h_in: &[f64],
    pos: usize,
    kc: &mut [f64],
    vc: &mut [f64],
    ids: &[u32],
    rw: &[f32],
    dm: Dims,
) -> Vec<f64> {
    let (h, nh, nkv, d) = (
        dm.h as usize,
        dm.nh as usize,
        dm.nkv as usize,
        dm.d as usize,
    );
    let (ii, lmax, grp) = (dm.i as usize, dm.lmax as usize, dm.grp() as usize);
    let qd = nh * d;
    let kvd = nkv * d;
    let eps = dm.eps as f64;
    let d16 = |v: u16| f16_to_f32(v) as f64;
    let r16 = |x: f64| f16_to_f32(f32_to_f16(x as f32)) as f64;
    let rms = |v: &[f64], w: &[u16], n: usize| {
        let ss: f64 = v.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let r = 1.0 / (ss + eps).sqrt();
        (0..n)
            .map(|j| r16(v[j] * r * d16(w[j])))
            .collect::<Vec<f64>>()
    };
    let proj = |w: &[u16], x: &[f64], rows: usize, k: usize| -> Vec<f64> {
        (0..rows)
            .map(|a| r16((0..k).map(|j| d16(w[a * k + j]) * x[j]).sum::<f64>()))
            .collect()
    };
    let mut hh = h_in.to_vec();
    let hn = rms(&hh, &lh.in_norm, h);
    let mut q = proj(&lh.wq, &hn, qd, h);
    let mut k = proj(&lh.wk, &hn, kvd, h);
    let v = proj(&lh.wv, &hn, kvd, h);
    // per-head QK-norm
    for head in 0..nh {
        let s: f64 = (0..d)
            .map(|x| q[head * d + x] * q[head * d + x])
            .sum::<f64>()
            / d as f64;
        let r = 1.0 / (s + eps).sqrt();
        for x in 0..d {
            q[head * d + x] = r16(q[head * d + x] * r * d16(lh.q_norm[x]));
        }
    }
    for head in 0..nkv {
        let s: f64 = (0..d)
            .map(|x| k[head * d + x] * k[head * d + x])
            .sum::<f64>()
            / d as f64;
        let r = 1.0 / (s + eps).sqrt();
        for x in 0..d {
            k[head * d + x] = r16(k[head * d + x] * r * d16(lh.k_norm[x]));
        }
    }
    // RoPE at pos
    let rope = |x: &mut [f64], head: usize| {
        for i in 0..d / 2 {
            let freq = (dm.theta as f64).powf(-2.0 * i as f64 / d as f64);
            let ang = pos as f64 * freq;
            let (c, s) = (ang.cos(), ang.sin());
            let (a, b) = (x[head * d + i], x[head * d + i + d / 2]);
            x[head * d + i] = r16(a * c - b * s);
            x[head * d + i + d / 2] = r16(b * c + a * s);
        }
    };
    for head in 0..nh {
        rope(&mut q, head);
    }
    for head in 0..nkv {
        rope(&mut k, head);
    }
    // Append current K,V into the reference caches after the same row-local FP4
    // quantization used by the GPU paged append path.
    for head in 0..nkv {
        let krow: Vec<f32> = (0..d).map(|x| k[head * d + x] as f32).collect();
        let vrow: Vec<f32> = (0..d).map(|x| v[head * d + x] as f32).collect();
        let (k4, ksc) = quantize_fp4_blocks(&krow, 1);
        let (v4, vsc) = quantize_fp4_blocks(&vrow, 1);
        for x in 0..d {
            kc[(head * lmax + pos) * d + x] = fp4_decode(&k4, &ksc, 0, x) as f64;
            vc[(head * lmax + pos) * d + x] = fp4_decode(&v4, &vsc, 0, x) as f64;
        }
    }
    // attention per head over 0..=pos
    let scale = 1.0 / (d as f64).sqrt();
    let mut attn = vec![0f64; qd];
    for head in 0..nh {
        let kvh = head / grp;
        let mut sc = vec![0f64; pos + 1];
        let mut mx = f64::NEG_INFINITY;
        for t in 0..=pos {
            let s: f64 = (0..d)
                .map(|x| q[head * d + x] * kc[(kvh * lmax + t) * d + x])
                .sum::<f64>()
                * scale;
            sc[t] = s;
            mx = mx.max(s);
        }
        let mut z = 0f64;
        for s in &mut sc {
            *s = (*s - mx).exp();
            z += *s;
        }
        for x in 0..d {
            let mut acc = 0f64;
            for t in 0..=pos {
                acc += sc[t] / z * vc[(kvh * lmax + t) * d + x];
            }
            attn[head * d + x] = r16(acc);
        }
    }
    // O proj + residual 1
    let o = proj(&lh.wo, &attn, h, qd);
    for j in 0..h {
        hh[j] = r16(hh[j] + o[j]);
    }
    // post-attn norm + MoE (GPU's expert selection)
    let h2n = rms(&hh, &lh.post_norm, h);
    let mut moe = vec![0f64; h];
    for (slot, &id) in ids.iter().enumerate() {
        let (go, dofs) = (id as usize * ii * h, id as usize * h * ii);
        let mut hi = vec![0f64; ii];
        for r in 0..ii {
            let mut g = 0f64;
            let mut u = 0f64;
            for j in 0..h {
                g += d16(lh.gate[go + r * h + j]) * h2n[j];
                u += d16(lh.up[go + r * h + j]) * h2n[j];
            }
            hi[r] = r16((g / (1.0 + (-g).exp())) * u);
        }
        let wj = rw[slot] as f64;
        for n in 0..h {
            let mut acc = 0f64;
            for r in 0..ii {
                acc += d16(lh.down[dofs + n * ii + r]) * hi[r];
            }
            moe[n] += wj * acc;
        }
    }
    for j in 0..h {
        hh[j] = r16(hh[j] + moe[j]);
    }
    hh
}

/// Build a reduced-scale Qwen3-235B-A22B-architecture decode model with random
/// weights, generate T tokens greedily on the GPU, and validate every step's
/// full-model logits against an f64 reference. No ROCm.
pub fn check_model_decode_on(dev: &mut GpuDevice) -> Result<()> {
    let node_id = dev.node_id();
    let dm = Dims {
        h: 4096,
        nh: 64,
        nkv: 4,
        d: 128,
        i: 512,
        e: 16,
        topk: 8,
        lmax: 516,
        vocab: 4096,
        theta: 5_000_000.0,
        eps: 1e-6,
    };
    const NLAYERS: usize = 4;
    const PREFILL: usize = 512;
    const TGEN: usize = 4;
    const BLOCK_SIZE: usize = 16;
    let (h, nh, nkv, d) = (
        dm.h as usize,
        dm.nh as usize,
        dm.nkv as usize,
        dm.d as usize,
    );
    let (ii, e, topk, vocab, lmax) = (
        dm.i as usize,
        dm.e as usize,
        dm.topk as usize,
        dm.vocab as usize,
        dm.lmax as usize,
    );
    let qd = nh * d;
    let kvd = nkv * d;
    let logical_blocks = lmax.div_ceil(BLOCK_SIZE);
    // Physical block 0 is a permanent null/sink block for clamped or padded
    // paged-KV entries. Real logical context blocks map to physical 1..N.
    let physical_blocks = logical_blocks + 1;
    let rows_per_head = physical_blocks * BLOCK_SIZE;
    let gen = |a: usize, s: usize| {
        ((a.wrapping_mul(2654435761)
            .wrapping_add(s.wrapping_mul(40503))
            >> 11)
            & 0xff) as f32
            / 256.0
            - 0.5
    };
    let mk = |n: usize, seed: usize, sc: f32, bias: f32| -> Vec<u16> {
        (0..n)
            .map(|x| f32_to_f16(gen(x, seed) * sc + bias))
            .collect()
    };

    // Per-layer host weights (kept for the f64 reference) + device uploads.
    let mut hosts: Vec<LayerHost> = Vec::with_capacity(NLAYERS);
    let mut keep: Vec<crate::DeviceBuffer> = Vec::new();
    let mut lvs: Vec<LayerVas> = Vec::with_capacity(NLAYERS);
    let mut kv4s: Vec<LayerFp4Kv> = Vec::with_capacity(NLAYERS);
    let mut host_kc: Vec<Vec<f64>> = Vec::new();
    let mut host_vc: Vec<Vec<f64>> = Vec::new();
    let up_dev =
        |data: &[u16], dev: &mut GpuDevice, keep: &mut Vec<crate::DeviceBuffer>| -> Result<u64> {
            let mut b = dev.alloc_device(data.len() * 2)?;
            unsafe { b.as_mut_slice_of::<u16>()[..data.len()].copy_from_slice(data) };
            let va = b.va();
            keep.push(b);
            Ok(va)
        };
    let up_bytes =
        |data: &[u8], dev: &mut GpuDevice, keep: &mut Vec<crate::DeviceBuffer>| -> Result<u64> {
            let mut b = dev.alloc_device(data.len())?;
            unsafe { b.as_mut_slice()[..data.len()].copy_from_slice(data) };
            let va = b.va();
            keep.push(b);
            Ok(va)
        };
    for layer in 0..NLAYERS {
        let s0 = layer * 100;
        let lh = LayerHost {
            in_norm: mk(h, s0 + 1, 0.0, 1.0),
            wq: mk(qd * h, s0 + 2, 0.02, 0.0),
            wk: mk(kvd * h, s0 + 3, 0.02, 0.0),
            wv: mk(kvd * h, s0 + 4, 0.02, 0.0),
            q_norm: mk(d, s0 + 5, 0.0, 1.0),
            k_norm: mk(d, s0 + 6, 0.0, 1.0),
            wo: mk(h * qd, s0 + 7, 0.02, 0.0),
            post_norm: mk(h, s0 + 8, 0.0, 1.0),
            wgate: mk(e * h, s0 + 9, 0.02, 0.0),
            gate: mk(e * ii * h, s0 + 10, 0.05, 0.0),
            up: mk(e * ii * h, s0 + 11, 0.05, 0.0),
            down: mk(e * h * ii, s0 + 12, 0.05, 0.0),
        };
        let lv = LayerVas {
            in_norm: up_dev(&lh.in_norm, dev, &mut keep)?,
            wq: up_dev(&lh.wq, dev, &mut keep)?,
            wk: up_dev(&lh.wk, dev, &mut keep)?,
            wv: up_dev(&lh.wv, dev, &mut keep)?,
            q_norm: up_dev(&lh.q_norm, dev, &mut keep)?,
            k_norm: up_dev(&lh.k_norm, dev, &mut keep)?,
            wo: up_dev(&lh.wo, dev, &mut keep)?,
            post_norm: up_dev(&lh.post_norm, dev, &mut keep)?,
            wgate: up_dev(&lh.wgate, dev, &mut keep)?,
            gate: up_dev(&lh.gate, dev, &mut keep)?,
            up: up_dev(&lh.up, dev, &mut keep)?,
            down: up_dev(&lh.down, dev, &mut keep)?,
        };
        // KV caches: prefill positions 0..PREFILL with random (already-processed
        // context); the rest is filled by kv_append during decode.
        let mut kc16 = vec![0u16; nkv * lmax * d];
        let mut vc16 = vec![0u16; nkv * lmax * d];
        for head in 0..nkv {
            for t in 0..PREFILL {
                for x in 0..d {
                    kc16[(head * lmax + t) * d + x] =
                        f32_to_f16(gen((head * lmax + t) * d + x, s0 + 21));
                    vc16[(head * lmax + t) * d + x] =
                        f32_to_f16(gen((head * lmax + t) * d + x, s0 + 22));
                }
            }
        }
        let mut k4_all = vec![0u8; nkv * rows_per_head * 64];
        let mut v4_all = vec![0u8; nkv * rows_per_head * 64];
        let mut ksc_all = vec![127u8; nkv * rows_per_head * 4];
        let mut vsc_all = vec![127u8; nkv * rows_per_head * 4];
        let mut kc_ref = vec![0f64; nkv * lmax * d];
        let mut vc_ref = vec![0f64; nkv * lmax * d];
        for head in 0..nkv {
            let mut kf = vec![0f32; rows_per_head * d];
            let mut vf = vec![0f32; rows_per_head * d];
            for t in 0..lmax {
                let phys_t = (t / BLOCK_SIZE + 1) * BLOCK_SIZE + (t % BLOCK_SIZE);
                for x in 0..d {
                    let src = (head * lmax + t) * d + x;
                    kf[phys_t * d + x] = f16_to_f32(kc16[src]);
                    vf[phys_t * d + x] = f16_to_f32(vc16[src]);
                }
            }
            let (k4, ksc) = quantize_fp4_blocks(&kf, rows_per_head);
            let (v4, vsc) = quantize_fp4_blocks(&vf, rows_per_head);
            let row0 = head * rows_per_head;
            k4_all[row0 * 64..(row0 + rows_per_head) * 64].copy_from_slice(&k4);
            v4_all[row0 * 64..(row0 + rows_per_head) * 64].copy_from_slice(&v4);
            ksc_all[row0 * 4..(row0 + rows_per_head) * 4].copy_from_slice(&ksc);
            vsc_all[row0 * 4..(row0 + rows_per_head) * 4].copy_from_slice(&vsc);
            for t in 0..lmax {
                let phys_t = (t / BLOCK_SIZE + 1) * BLOCK_SIZE + (t % BLOCK_SIZE);
                for x in 0..d {
                    kc_ref[(head * lmax + t) * d + x] = fp4_decode(&k4, &ksc, phys_t, x) as f64;
                    vc_ref[(head * lmax + t) * d + x] = fp4_decode(&v4, &vsc, phys_t, x) as f64;
                }
            }
        }
        let kv4 = LayerFp4Kv {
            k: up_bytes(&k4_all, dev, &mut keep)?,
            v: up_bytes(&v4_all, dev, &mut keep)?,
            sk: up_bytes(&ksc_all, dev, &mut keep)?,
            sv: up_bytes(&vsc_all, dev, &mut keep)?,
        };
        kv4s.push(kv4);
        // keep the prefilled host caches in the LayerHost? store separately:
        hosts.push(lh);
        // stash prefill host caches for the ref via the layer index (parallel vecs)
        host_kc.push(kc_ref);
        host_vc.push(vc_ref);
        lvs.push(lv);
    }

    let block_table: Vec<u32> = (0..logical_blocks).map(|i| (i + 1) as u32).collect();
    let indptr: Vec<u32> = vec![0, logical_blocks as u32];
    let last_len = if lmax % BLOCK_SIZE == 0 {
        BLOCK_SIZE
    } else {
        lmax % BLOCK_SIZE
    };
    let last_page_len: Vec<u32> = vec![last_len as u32];
    let batch_indices: Vec<u32> = vec![0];
    let mut block_table_buf = dev.alloc_device(block_table.len() * 4)?;
    let mut indptr_buf = dev.alloc_device(indptr.len() * 4)?;
    let mut last_buf = dev.alloc_device(last_page_len.len() * 4)?;
    let mut batch_buf = dev.alloc_device(batch_indices.len() * 4)?;
    let mut pos_buf = dev.alloc_device(4)?;
    let mut seq_len_buf = dev.alloc_device(4)?;
    let mut step_buf = dev.alloc_device(4)?;
    unsafe {
        block_table_buf.as_mut_slice_of::<u32>()[..block_table.len()].copy_from_slice(&block_table);
        indptr_buf.as_mut_slice_of::<u32>()[..indptr.len()].copy_from_slice(&indptr);
        last_buf.as_mut_slice_of::<u32>()[..last_page_len.len()].copy_from_slice(&last_page_len);
        batch_buf.as_mut_slice_of::<u32>()[..batch_indices.len()].copy_from_slice(&batch_indices);
        pos_buf.as_mut_slice_of::<u32>()[0] = PREFILL as u32;
        seq_len_buf.as_mut_slice_of::<u32>()[0] = PREFILL as u32;
        step_buf.as_mut_slice_of::<u32>()[0] = 0;
    }
    let pm = PagedMeta {
        block_table: block_table_buf.va(),
        indices: block_table_buf.va(),
        indptr: indptr_buf.va(),
        last_page_len: last_buf.va(),
        batch_indices: batch_buf.va(),
        positions: pos_buf.va(),
        seq_lens: seq_len_buf.va(),
        step: step_buf.va(),
        block_size: BLOCK_SIZE as u32,
        logical_blocks: logical_blocks as u32,
        physical_blocks: physical_blocks as u32,
        rows_per_head: rows_per_head as u32,
    };
    // Preflight the exact paged metadata consumed by FP4 append + paged GQA
    // attention. This keeps bad physical page ids / malformed FlashInfer-style
    // metadata on the tiny bounds-check kernels instead of letting append or
    // attention discover them as a fatal MI355 no-retry GPUVM fault.
    dev.check_paged_block_table(pm.block_table, pm.logical_blocks, pm.physical_blocks)?;
    dev.check_paged_kv_metadata(
        pm.indptr,
        pm.indices,
        pm.last_page_len,
        1,
        pm.logical_blocks,
        pm.physical_blocks,
        pm.block_size,
    )?;
    let metadata_guard = std::env::var("MAINARCH_DECODE_METADATA_GUARD")
        .map(|v| {
            let v = v.trim();
            !(v.eq_ignore_ascii_case("0")
                || v.eq_ignore_ascii_case("false")
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("no"))
        })
        .unwrap_or(false);
    if metadata_guard {
        println!(
            "  model-decode metadata guard enabled: per-step paged KV metadata and position checks"
        );
    }

    // Embedding, LM head, final norm.
    let embed16 = mk(vocab * h, 7001, 0.05, 0.0);
    let lmhead16 = mk(vocab * h, 7002, 0.05, 0.0);
    let finalnorm16 = mk(h, 7003, 0.0, 1.0);
    let mut embed_buf = dev.alloc_device(vocab * h * 2)?;
    let lmhead_va = up_dev(&lmhead16, dev, &mut keep)?;
    let finalnorm_va = up_dev(&finalnorm16, dev, &mut keep)?;
    unsafe { embed_buf.as_mut_slice_of::<u16>()[..vocab * h].copy_from_slice(&embed16) };

    // Residual stream + scratch buffers.
    let h_buf = dev.alloc_device(h * 2)?;
    let hf_va = up_dev(&vec![0u16; h], dev, &mut keep)?;
    let mut logits_buf = dev.alloc_device(TGEN * vocab * 4)?;
    let mut tokens_buf = dev.alloc_device((TGEN + 1) * 4)?;
    let mut ids_buf = dev.alloc_device(TGEN * NLAYERS * topk * 4)?;
    let mut rw_buf = dev.alloc_device(TGEN * NLAYERS * topk * 4)?;
    let ns = split_count(dm.lmax);
    let sc = Scratch {
        hn: up_dev(&vec![0u16; h], dev, &mut keep)?,
        qf32: up_dev(&vec![0u16; qd * 2], dev, &mut keep)?,
        kf32: up_dev(&vec![0u16; kvd * 2], dev, &mut keep)?,
        vf32: up_dev(&vec![0u16; kvd * 2], dev, &mut keep)?,
        qf16: up_dev(&vec![0u16; qd], dev, &mut keep)?,
        a16: up_dev(&vec![0u16; qd], dev, &mut keep)?,
        oproj: up_dev(&vec![0u16; h * 2], dev, &mut keep)?,
        hi: up_dev(&vec![0u16; topk * ii], dev, &mut keep)?,
        moe: up_dev(&vec![0u16; h * 2], dev, &mut keep)?,
        ids_cur: up_dev(&vec![0u16; topk * 2], dev, &mut keep)?,
        rw_cur: up_dev(&vec![0u16; topk * 2], dev, &mut keep)?,
        // Head-major partials/intermediates for the fused GQA attention: all nh
        // heads' partials live at once (one batched combine over all heads), so
        // these are nh× the single-head size. (f32 partials → *2 over the u16 vec.)
        part: up_dev(&vec![0u16; nh * ns as usize * (d + 2) * 2], dev, &mut keep)?,
        inter: up_dev(
            &vec![0u16; nh * crate::attn::combine_groups(ns).max(1) as usize * (d + 2) * 2],
            dev,
            &mut keep,
        )?,
    };
    let hva = h_buf.va();

    // ---- GPU decode loop ----
    let seed_tok = 7usize;
    unsafe {
        let toks = tokens_buf.as_mut_slice_of::<u32>();
        toks[..TGEN + 1].fill(0);
        toks[0] = seed_tok as u32;
    }
    let to = Duration::from_secs(20);
    let gpu_start = Instant::now();
    for t in 0..TGEN {
        if !metadata_guard {
            dev.chain_next();
        }
        dev.arm_decode_step_embed_rmsnorm_token(
            embed_buf.va(),
            tokens_buf.va(),
            lvs[0].in_norm,
            hva,
            sc.hn,
            pm.step,
            pm.positions,
            pm.seq_lens,
            pm.last_page_len,
            PREFILL as u32,
            pm.block_size,
            (TGEN + 1) as u32,
            dm.vocab,
            dm.h,
            dm.eps,
        )?;
        if metadata_guard {
            dev.wait(to)?;
            dev.check_paged_kv_metadata(
                pm.indptr,
                pm.indices,
                pm.last_page_len,
                1,
                pm.logical_blocks,
                pm.physical_blocks,
                pm.block_size,
            )?;
            let observed_pos = unsafe { pos_buf.as_mut_slice_of::<u32>()[0] };
            let observed_seq_len = unsafe { seq_len_buf.as_mut_slice_of::<u32>()[0] };
            let observed_last_page_len = unsafe { last_buf.as_mut_slice_of::<u32>()[0] };
            let expected_pos = PREFILL as u32 + t as u32;
            let expected_seq_len = expected_pos + 1;
            let rem = expected_seq_len % pm.block_size;
            let expected_last_page_len = if rem == 0 { pm.block_size } else { rem };
            if observed_pos != expected_pos
                || observed_seq_len != expected_seq_len
                || observed_last_page_len != expected_last_page_len
            {
                return Err(anyhow!(
                    "model-decode metadata guard failed at step {t}: pos={observed_pos}/{expected_pos} seq_len={observed_seq_len}/{expected_seq_len} last_page_len={observed_last_page_len}/{expected_last_page_len}"
                ));
            }
        }
        for layer in 0..NLAYERS {
            let (tail_norm_w, tail_norm_out) = if layer + 1 < NLAYERS {
                (lvs[layer + 1].in_norm, sc.hn)
            } else {
                (finalnorm_va, hf_va)
            };
            layer_forward_normed(
                dev,
                &lvs[layer],
                &kv4s[layer],
                pm,
                &sc,
                hva,
                ids_buf.va(),
                rw_buf.va(),
                layer as u32,
                NLAYERS as u32,
                TGEN as u32,
                tail_norm_w,
                tail_norm_out,
                dm,
            )?;
        }
        dev.chain_next();
        dev.arm_gemv_step(
            lmhead_va,
            hf_va,
            logits_buf.va(),
            pm.step,
            TGEN as u32,
            dm.vocab,
            dm.h,
        )?;
        if !metadata_guard && t + 1 < TGEN {
            dev.chain_next();
        }
        dev.arm_argmax_f32_step(
            logits_buf.va(),
            tokens_buf.va(),
            pm.step,
            (TGEN + 1) as u32,
            dm.vocab,
        )?;
        if metadata_guard {
            dev.wait(to)?;
        }
    }
    if !metadata_guard {
        dev.wait(to)?;
    }

    let gpu_us = gpu_start.elapsed().as_secs_f64() * 1e6;
    let token_hist = unsafe { tokens_buf.as_mut_slice_of::<u32>()[..TGEN + 1].to_vec() };
    let logits_all = unsafe { logits_buf.as_mut_slice_of::<f32>()[..TGEN * vocab].to_vec() };
    let ids_all = unsafe { ids_buf.as_mut_slice_of::<u32>()[..TGEN * NLAYERS * topk].to_vec() };
    let rw_all = unsafe { rw_buf.as_mut_slice_of::<f32>()[..TGEN * NLAYERS * topk].to_vec() };
    let inputs: Vec<usize> = token_hist[..TGEN].iter().map(|&v| v as usize).collect();
    for (i, &tok) in inputs.iter().enumerate() {
        if tok >= vocab {
            return Err(anyhow!(
                "model-decode GPU token history[{i}]={tok} exceeds vocab {vocab}"
            ));
        }
    }
    let gpu_logits: Vec<Vec<f32>> = (0..TGEN)
        .map(|t| logits_all[t * vocab..(t + 1) * vocab].to_vec())
        .collect();
    let ids_log: Vec<Vec<[u32; 8]>> = (0..TGEN)
        .map(|t| {
            (0..NLAYERS)
                .map(|layer| {
                    let mut ids = [0u32; 8];
                    let off = (t * NLAYERS + layer) * topk;
                    ids[..topk].copy_from_slice(&ids_all[off..off + topk]);
                    ids
                })
                .collect()
        })
        .collect();
    let rw_log: Vec<Vec<[f32; 8]>> = (0..TGEN)
        .map(|t| {
            (0..NLAYERS)
                .map(|layer| {
                    let mut rw = [0f32; 8];
                    let off = (t * NLAYERS + layer) * topk;
                    rw[..topk].copy_from_slice(&rw_all[off..off + topk]);
                    rw
                })
                .collect()
        })
        .collect();
    // ---- f64 reference following the GPU's tokens + expert selections ----
    let d16 = |v: u16| f16_to_f32(v) as f64;
    let r16 = |x: f64| f16_to_f32(f32_to_f16(x as f32)) as f64;
    let mut ref_kc = host_kc.clone();
    let mut ref_vc = host_vc.clone();
    let mut max_rel = 0f64;
    let mut tokens_str = String::new();
    for t in 0..TGEN {
        let pos = PREFILL + t;
        let tok_in = inputs[t];
        let mut hh: Vec<f64> = (0..h).map(|j| d16(embed16[tok_in * h + j])).collect();
        for layer in 0..NLAYERS {
            hh = layer_ref(
                &hosts[layer],
                &hh,
                pos,
                &mut ref_kc[layer],
                &mut ref_vc[layer],
                &ids_log[t][layer],
                &rw_log[t][layer],
                dm,
            );
        }
        // final norm + LM head (logits are f32 on GPU, no f16 round at the very end)
        let ss: f64 = hh.iter().map(|x| x * x).sum::<f64>() / h as f64;
        let r = 1.0 / (ss + dm.eps as f64).sqrt();
        let hf: Vec<f64> = (0..h)
            .map(|j| r16(hh[j] * r * d16(finalnorm16[j])))
            .collect();
        let (mut num, mut den) = (0f64, 0f64);
        for a in 0..vocab {
            let lg: f64 = (0..h).map(|j| d16(lmhead16[a * h + j]) * hf[j]).sum();
            let diff = gpu_logits[t][a] as f64 - lg;
            num += diff * diff;
            den += lg * lg;
        }
        let rel = (num / den.max(1e-30)).sqrt();
        max_rel = max_rel.max(rel);
        let cpu_tok = argmax(&gpu_logits[t]);
        let gpu_tok = token_hist[t + 1] as usize;
        if gpu_tok != cpu_tok {
            return Err(anyhow!(
                "model-decode GPU argmax mismatch at step {t}: gpu token {gpu_tok}, CPU argmax {cpu_tok}"
            ));
        }
        tokens_str.push_str(&format!("{} ", gpu_tok));
    }
    if max_rel > 3e-2 {
        return Err(anyhow!(
            "model-decode mismatch: per-step logit rel-L2 {max_rel:.4} > 3e-2"
        ));
    }
    println!(
        "  model-decode: Qwen3-235B-A22B arch (H={}, {}L, {}Q/{}KV d{}, paged FP4 KV, MoE {}×top{}, vocab {}, prefill {}, gen {}) on node {node_id}",
        dm.h, NLAYERS, dm.nh, dm.nkv, dm.d, dm.e, dm.topk, dm.vocab, PREFILL, TGEN
    );
    println!(
        "    - correct: max per-step logit rel-L2 {max_rel:.2e}; greedy tokens [ {}]",
        tokens_str
    );
    println!(
        "    - GPU decode: {:.0} µs/token ({} layers, {:.0} µs/layer)",
        gpu_us / TGEN as f64,
        NLAYERS,
        gpu_us / (TGEN * NLAYERS) as f64
    );
    Ok(())
}

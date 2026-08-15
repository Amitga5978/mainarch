//! Multi-GPU all-reduce over XGMI, through the raw KFD/AQL path.
//!
//! All devices share one [`Kfd`] (one process VM), so a buffer allocated on
//! one device can be mapped into a peer's VM and accessed directly over XGMI —
//! the kernel reads the peer pointer like local memory. Data buffers live in
//! device-local VRAM (HBM), so peer access is real GPU-to-GPU XGMI traffic.
//!
//! The all-reduce is direct: rank 0 sums all R peer buffers in one
//! `reduce_peers` kernel (peer reads fan out across XGMI links), then scatters
//! the result back to every rank in one `broadcast_peers` kernel — two host
//! round-trips regardless of R. This minimizes launch latency, which dominates
//! small-message all-reduce.

use anyhow::{anyhow, Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::attn::e4m3_to_f32;
use crate::{DeviceBuffer, GpuDevice, Kfd};

const OP_TIMEOUT: Duration = Duration::from_secs(20);
const DIRECT_READY_WATCHDOG_TICKS: u64 = 10_000_000;
const DIRECT_READY_WATCHDOG_SPINS: u64 = 1_000_000;

/// A resident multi-GPU all-reduce context: one buffer per device, all
/// cross-mapped so any device can read any other's buffer over XGMI.
pub struct AllReduce {
    devices: Vec<GpuDevice>,
    bufs: Vec<DeviceBuffer>,
    /// Per-device f32 residual buffers for the fused all-reduce + residual +
    /// RMSNorm path. These are allocated and peer-mapped once so decode can reuse
    /// the f32-stable boundary without late KFD map calls.
    residuals: Vec<DeviceBuffer>,
    /// Per-device peer-pointer tables for `residuals`.
    residual_ptrs: Vec<DeviceBuffer>,
    /// Per-device peer-pointer tables (each holds the R peer-buffer VAs,
    /// resident in that device's VM) for the pointer-array kernels.
    ptrs: Vec<DeviceBuffer>,
    /// Per-device staging buffers (R contiguous chunk slots) for the
    /// write-based reduce-scatter, cross-mapped to all peers.
    stage: Vec<DeviceBuffer>,
    /// Per-device tables of all peers' staging VAs.
    stage_ptrs: Vec<DeviceBuffer>,
    /// Per-device arrival-flag buffers (R u32) for the in-kernel cross-GPU
    /// barrier, cross-mapped so peers can signal.
    flags: Vec<DeviceBuffer>,
    /// Per-device tables of all peers' flag VAs.
    flag_ptrs: Vec<DeviceBuffer>,
    /// Per-device intra-GPU grid-barrier counters (2 u32: arrival, sense).
    gbar: Vec<DeviceBuffer>,
    /// Rank-0 resident readiness flags for decode-sized direct all-reduce
    /// chaining. Producer queues write slots [0..R); reduce_peers_wait_ready_flags
    /// polls them and writes slot R on watchdog timeout.
    direct_ready_flags: DeviceBuffer,
    /// Rank-0 resident per-rank terminal producer counters for fused producer
    /// publication. Producer workgroups increment slots [0..R); the terminal
    /// producer workgroup publishes the matching direct_ready_flags slot.
    direct_ready_counts: DeviceBuffer,
    /// Device-side timeout, in s_memrealtime ticks, for the direct ready-flag
    /// wait loop. A tripped timeout writes direct_ready_flags[R].
    direct_ready_watchdog_ticks: u64,
    /// Monotonic barrier sequence base for the GPU-driven kernel (2 per call).
    seq: u32,
    /// Fixed workgroup count for the one-shot kernel (must be co-resident).
    oneshot_wg: u32,
    /// Byte threshold at/above which the GPU-driven one-shot kernel is used.
    oneshot_min_bytes: usize,
    /// Allocated capacity (elements per rank).
    cap: u32,
    /// Active element count for the current operation (<= cap).
    n: u32,
    /// Byte threshold at/below which the latency-optimal direct scheme is used;
    /// above it, the bandwidth-optimal reduce-scatter/all-gather is used.
    direct_max_bytes: usize,
    /// Byte threshold at/above which reduce-scatter switches from read-pull to
    /// write-push (writes win at scale; the extra barrier amortizes).
    write_rs_min_bytes: usize,
    /// Per-device SDMA copy-engine queues (R per device), created lazily for the
    /// SDMA all-reduce path.
    sdma: Vec<Vec<crate::SdmaQueue>>,
    /// Per-device CU-store destination tables for the hybrid all-reduce
    /// (reduce-scatter staging-slot VAs / all-gather peer-buffer VAs), filled
    /// per call. Created lazily.
    hyb_stbl: Vec<DeviceBuffer>,
    hyb_gtbl: Vec<DeviceBuffer>,
    /// Shared KFD handle (one process VM), kept so the concurrent path can open
    /// a second device queue per node.
    kfd: Arc<Kfd>,
    /// Second per-node device handles (independent AQL queue) so the concurrent
    /// all-reduce can run the SDMA partition's reduce alongside the one-shot
    /// kernel on the primary queue. Created lazily.
    devices2: Vec<GpuDevice>,
    /// Second staging set for the concurrent path's SDMA partition (so it does
    /// not collide with the one-shot's staging). Created lazily, peer-mapped.
    stage_b: Vec<DeviceBuffer>,
    /// Per-device coordination semaphores (2R+1 u32) for the device-side
    /// dual-path all-reduce: SDMA scatter-done[R], gather release, gather-done[R].
    /// Device-local (kernel and SDMA on the same GPU). Created lazily.
    dev_sem: Vec<DeviceBuffer>,
    /// Rank-0 resident partial workspace for the decode-side fused TP MLP
    /// all-reduce + f16 residual + f16 RMSNorm handoff.
    resident_f16_partial: Option<DeviceBuffer>,
}

impl AllReduce {
    /// Bring up `nodes` as a single all-reduce group over `n` f32 elements
    /// each. Buffers are allocated per device and peer-mapped to every other
    /// device in the group.
    pub fn new(nodes: &[u32], n: u32) -> Result<Self> {
        Self::new_shared(Arc::new(Kfd::open()?), nodes, n)
    }

    /// Bring up `nodes` on an existing process-level KFD handle. This is used
    /// by serving gates that already acquired one rank's VM before adding the
    /// TP collective leg; `Kfd::ensure_vm` remains idempotent on the shared
    /// handle, avoiding a second `AMDKFD_IOC_ACQUIRE_VM` for the same GPU.
    pub fn new_shared(kfd: Arc<Kfd>, nodes: &[u32], n: u32) -> Result<Self> {
        if nodes.is_empty() {
            return Err(anyhow!("all-reduce needs at least one GPU node"));
        }
        let mut devices = Vec::with_capacity(nodes.len());
        for &node in nodes {
            devices.push(GpuDevice::open_shared(kfd.clone(), node)?);
        }

        let bytes = (n as usize) * 4;
        let mut bufs = Vec::with_capacity(nodes.len());
        for d in &devices {
            // Device-local VRAM (HBM) so peer access is real GPU-to-GPU XGMI
            // traffic, not host-RAM reads. Large-BAR keeps it CPU-mappable for
            // fills/verification.
            bufs.push(d.kfd().alloc_vram(d.node_id(), bytes)?);
        }

        // Cross-map every buffer into every device's VM so peer reads/writes
        // over XGMI are legal from any device in the group.
        for (bi, buf) in bufs.iter().enumerate() {
            for (di, dev) in devices.iter().enumerate() {
                if bi != di {
                    kfd.map_buffer_to_peer(buf, dev.node_id())
                        .with_context(|| {
                            format!("mapping buffer {bi} into peer node {}", dev.node_id())
                        })?;
                }
            }
        }

        let mut residuals = Vec::with_capacity(nodes.len());
        for d in &devices {
            residuals.push(d.kfd().alloc_vram(d.node_id(), bytes)?);
        }
        for (bi, buf) in residuals.iter().enumerate() {
            for (di, dev) in devices.iter().enumerate() {
                if bi != di {
                    kfd.map_buffer_to_peer(buf, dev.node_id())
                        .with_context(|| {
                            format!("mapping residual {bi} into peer node {}", dev.node_id())
                        })?;
                }
            }
        }
        let mut residual_ptrs = Vec::with_capacity(devices.len());
        for d in &devices {
            let mut t = d.alloc(residuals.len() * 8)?;
            {
                let slots = unsafe { t.as_mut_slice_of::<u64>() };
                for (i, b) in residuals.iter().enumerate() {
                    slots[i] = b.va();
                }
            }
            residual_ptrs.push(t);
        }

        // Per-device peer-pointer tables (host-visible on each device) holding
        // the R peer buffer VAs. The VAs are process-global (one VM space), so
        // every table holds identical values; each device reads its own copy.
        let mut ptrs = Vec::with_capacity(devices.len());
        for d in &devices {
            let mut t = d.alloc(bufs.len() * 8)?;
            {
                let slots = unsafe { t.as_mut_slice_of::<u64>() };
                for (i, b) in bufs.iter().enumerate() {
                    slots[i] = b.va();
                }
            }
            ptrs.push(t);
        }

        // Staging buffers for the write-based reduce-scatter: R chunk-slots of
        // the (mult-of-4, ceil) chunk length, so R*cl_cap >= cap. Cross-mapped
        // to every peer so each GPU can push into others' staging.
        let r = devices.len();
        let cl_cap = round_up4(n.div_ceil(r as u32));
        let stage_elems = (r as u32 * cl_cap) as usize;
        let mut stage = Vec::with_capacity(r);
        for d in &devices {
            stage.push(d.kfd().alloc_vram(d.node_id(), stage_elems * 4)?);
        }
        for (bi, s) in stage.iter().enumerate() {
            for (di, dev) in devices.iter().enumerate() {
                if bi != di {
                    kfd.map_buffer_to_peer(s, dev.node_id()).with_context(|| {
                        format!("mapping staging {bi} into node {}", dev.node_id())
                    })?;
                }
            }
        }
        let mut stage_ptrs = Vec::with_capacity(r);
        for d in &devices {
            let mut t = d.alloc(r * 8)?;
            {
                let slots = unsafe { t.as_mut_slice_of::<u64>() };
                for (i, s) in stage.iter().enumerate() {
                    slots[i] = s.va();
                }
            }
            stage_ptrs.push(t);
        }

        // Decode-sized all-reduces are latency-dominated. On MI355X/gfx950 the
        // direct path wins through 512 KiB; 1 MiB crosses back to RSAG. Keep
        // this env-overridable for per-node tuning.
        let direct_max_bytes = std::env::var("MAINARCH_DIRECT_MAX_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512 * 1024);
        let write_rs_min_bytes = std::env::var("MAINARCH_WRITE_RS_MIN_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12 * 1024 * 1024);

        // Cross-GPU arrival flags (R u32) per device, cross-mapped + zeroed.
        let mut flags = Vec::with_capacity(r);
        for d in &devices {
            let mut f = d.kfd().alloc_vram(d.node_id(), r * 4)?;
            unsafe {
                for v in f.as_mut_slice_of::<u32>() {
                    *v = 0;
                }
            }
            flags.push(f);
        }
        for (bi, f) in flags.iter().enumerate() {
            for (di, dev) in devices.iter().enumerate() {
                if bi != di {
                    kfd.map_buffer_to_peer(f, dev.node_id()).with_context(|| {
                        format!("mapping flags {bi} into node {}", dev.node_id())
                    })?;
                }
            }
        }
        let mut flag_ptrs = Vec::with_capacity(r);
        for d in &devices {
            let mut t = d.alloc(r * 8)?;
            {
                let slots = unsafe { t.as_mut_slice_of::<u64>() };
                for (i, f) in flags.iter().enumerate() {
                    slots[i] = f.va();
                }
            }
            flag_ptrs.push(t);
        }
        // Intra-GPU grid barrier counters (device-local), zeroed.
        let mut gbar = Vec::with_capacity(r);
        for d in &devices {
            let mut g = d.kfd().alloc_vram(d.node_id(), 64)?;
            unsafe {
                for v in g.as_mut_slice_of::<u32>() {
                    *v = 0;
                }
            }
            gbar.push(g);
        }
        let mut direct_ready_flags = devices[0]
            .kfd()
            .alloc_public_coherent_vram(devices[0].node_id(), (r + 1) * 4)?;
        unsafe {
            direct_ready_flags.as_mut_slice_of::<u32>()[..(r + 1)].fill(0);
        }
        for dev in devices.iter().skip(1) {
            kfd.map_buffer_to_peer(&direct_ready_flags, dev.node_id())
                .with_context(|| {
                    format!(
                        "mapping rank0 direct ready flags into peer node {}",
                        dev.node_id()
                    )
                })?;
        }
        let mut direct_ready_counts = devices[0]
            .kfd()
            .alloc_public_coherent_vram(devices[0].node_id(), r * 4)?;
        unsafe {
            direct_ready_counts.as_mut_slice_of::<u32>()[..r].fill(0);
        }
        for dev in devices.iter().skip(1) {
            kfd.map_buffer_to_peer(&direct_ready_counts, dev.node_id())
                .with_context(|| {
                    format!(
                        "mapping rank0 direct ready counts into peer node {}",
                        dev.node_id()
                    )
                })?;
        }
        let direct_ready_watchdog_ticks = DIRECT_READY_WATCHDOG_TICKS;
        let oneshot_wg = std::env::var("MAINARCH_ONESHOT_WG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        // The GPU-driven one-shot kernel streams the whole op in one launch for
        // the highest large-message bandwidth. Its in-kernel barriers carry
        // watchdog spin-caps (they give up rather than wedge the CP), and the
        // co-resident `oneshot_wg` keeps the grid barrier completable, so it is
        // safe to route large messages here by default.
        let oneshot_min_bytes = std::env::var("MAINARCH_ONESHOT_MIN_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16 * 1024 * 1024);

        Ok(Self {
            devices,
            bufs,
            residuals,
            residual_ptrs,
            ptrs,
            stage,
            stage_ptrs,
            flags,
            flag_ptrs,
            gbar,
            direct_ready_flags,
            direct_ready_counts,
            direct_ready_watchdog_ticks,
            seq: 0,
            oneshot_wg,
            oneshot_min_bytes,
            cap: n,
            n,
            direct_max_bytes,
            write_rs_min_bytes,
            sdma: Vec::new(),
            hyb_stbl: Vec::new(),
            hyb_gtbl: Vec::new(),
            kfd,
            devices2: Vec::new(),
            stage_b: Vec::new(),
            dev_sem: Vec::new(),
            resident_f16_partial: None,
        })
    }

    /// SDMA-engine all-reduce: write-based reduce-scatter + all-gather where every
    /// XGMI copy runs on the dedicated SDMA copy engines (one queue per
    /// destination per GPU) instead of compute-kernel stores; the reduce stays on
    /// the CUs. Phases sync host-side. n a multiple of 4.
    pub fn all_reduce_sum_sdma(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        let cl = round_up4(n.div_ceil(r as u32));
        if r < 2 || cl == 0 || (n & 3) != 0 {
            return self.all_reduce_sum_rsag(true);
        }
        if self.sdma.is_empty() {
            let mut all = Vec::with_capacity(r);
            for d in &self.devices {
                let (kfd, node) = (d.kfd(), d.node_id());
                let mut qv = Vec::with_capacity(2 * r);
                for _ in 0..(2 * r) {
                    qv.push(crate::SdmaQueue::new_xgmi(kfd, node)?);
                }
                all.push(qv);
            }
            self.sdma = all;
        }
        let chunk_len = |c: usize| -> u32 {
            let off = c as u32 * cl;
            if off >= n {
                0
            } else {
                cl.min(n - off)
            }
        };
        let to = OP_TIMEOUT;
        let start = Instant::now();
        // Scatter: GPU g pushes its chunk c to owner c's staging slot g (SDMA).
        for g in 0..r {
            for c in 0..r {
                let len = chunk_len(c);
                if len == 0 {
                    continue;
                }
                let src = self.bufs[g].va() + (c as u32 * cl) as u64 * 4;
                let dst = self.stage[c].va() + (g as u32 * cl) as u64 * 4;
                self.sdma[g][c].copy_async(src, dst, len as usize * 4);
            }
        }
        for g in 0..r {
            for c in 0..r {
                if chunk_len(c) > 0 {
                    self.sdma[g][c].wait(to)?;
                }
            }
        }
        let t_scatter = start.elapsed();
        // Reduce: each GPU sums its R staging slots into its own chunk (CUs).
        for g in 0..r {
            let len = chunk_len(g);
            if len == 0 {
                continue;
            }
            let (out, st) = (self.bufs[g].va(), self.stage[g].va());
            self.devices[g].arm_gather_reduce_local(out, st, r as u32, g as u32 * cl, cl, len)?;
        }
        for g in 0..r {
            if chunk_len(g) > 0 {
                self.devices[g].wait(to)?;
            }
        }
        let t_reduce = start.elapsed();
        // All-gather: GPU g pushes its reduced chunk g to every peer's buffer (SDMA).
        for g in 0..r {
            let len = chunk_len(g);
            if len == 0 {
                continue;
            }
            let src = self.bufs[g].va() + (g as u32 * cl) as u64 * 4;
            for h in 0..r {
                let dst = self.bufs[h].va() + (g as u32 * cl) as u64 * 4;
                self.sdma[g][h].copy_async(src, dst, len as usize * 4);
            }
        }
        for g in 0..r {
            if chunk_len(g) > 0 {
                for h in 0..r {
                    self.sdma[g][h].wait(to)?;
                }
            }
        }
        let total = start.elapsed();
        if std::env::var_os("MAINARCH_SDMA_TIMING").is_some() {
            let us = |d: Duration| d.as_secs_f64() * 1e6;
            eprintln!(
                "  sdma-allreduce phases: scatter {:.0}us  reduce {:.0}us  gather {:.0}us  total {:.0}us",
                us(t_scatter),
                us(t_reduce) - us(t_scatter),
                us(total) - us(t_reduce),
                us(total),
            );
        }
        Ok(total)
    }

    /// Hybrid all-reduce: each XGMI write phase drives the SDMA copy engines AND
    /// the CU-store kernels concurrently — two independent write paths — so the
    /// aggregate egress exceeds what either reaches alone (measured P2P: SDMA 332,
    /// CU 264, hybrid 378 at 256 MiB, past RCCL's 363). Reduce-scatter splits the
    /// R chunks between SDMA (first `nc_sdma`) and the CU kernel (the rest);
    /// all-gather splits the R-1 peer writes the same way. The reduce stays on the
    /// CUs and phases sync host-side. Requires uniform chunks (n % (4R) == 0),
    /// else falls back to the device-side one-shot.
    pub fn all_reduce_sum_hybrid(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        if r < 2 || n == 0 || !(n as usize).is_multiple_of(4 * r) {
            return self.all_reduce_sum_oneshot();
        }
        let cl = n / r as u32; // uniform chunk length (elements)
        let cl4 = cl / 4; // float4 per chunk
                          // Percent of write volume routed to the CU-store path (rest to SDMA).
        let kfrac = std::env::var("MAINARCH_HYBRID_KFRAC")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(45)
            .min(95);
        let nc_cu = (((r * kfrac) + 50) / 100).clamp(1, r - 1); // CU chunks (scatter)
        let nc_sdma = r - nc_cu;
        let cu_g = ((((r - 1) * kfrac) + 50) / 100).clamp(1, r - 1); // CU peers (gather)
        let sd_g = (r - 1) - cu_g;
        // Lazy resources: R SDMA XGMI queues per device + CU dest tables.
        if self.sdma.is_empty() {
            let mut all = Vec::with_capacity(r);
            for d in &self.devices {
                let (kfd, node) = (d.kfd(), d.node_id());
                let mut qv = Vec::with_capacity(2 * r);
                for _ in 0..(2 * r) {
                    qv.push(crate::SdmaQueue::new_xgmi(kfd, node)?);
                }
                all.push(qv);
            }
            self.sdma = all;
        }
        if self.hyb_stbl.is_empty() {
            for d in &self.devices {
                self.hyb_stbl.push(d.alloc(r * 8)?);
                self.hyb_gtbl.push(d.alloc(r * 8)?);
            }
        }
        let to = OP_TIMEOUT;
        let start = Instant::now();

        // ---- Reduce-scatter: device g writes chunk c -> stage[c] slot g. ----
        // SDMA path: chunks [0, nc_sdma).  CU path: chunks [nc_sdma, r).
        for g in 0..r {
            // CU scatter table: CU chunk p targets stage[nc_sdma+p] slot g.
            let dst_vas: Vec<u64> = (0..nc_cu)
                .map(|p| self.stage[nc_sdma + p].va() + (g as u32 * cl) as u64 * 4)
                .collect();
            {
                let slots = unsafe { self.hyb_stbl[g].as_mut_slice_of::<u64>() };
                slots[..nc_cu].copy_from_slice(&dst_vas);
            }
            for c in 0..nc_sdma {
                let src = self.bufs[g].va() + (c as u32 * cl) as u64 * 4;
                let dst = self.stage[c].va() + (g as u32 * cl) as u64 * 4;
                self.sdma[g][c].copy_async(src, dst, cl as usize * 4);
            }
            let src_cu = self.bufs[g].va() + (nc_sdma as u32 * cl) as u64 * 4;
            let tbl = self.hyb_stbl[g].va();
            let kwg = (nc_cu as u32) * 32;
            self.devices[g].arm_p2p_write(src_cu, tbl, nc_cu as u32, cl4, 0, cl4, kwg)?;
        }
        for g in 0..r {
            for c in 0..nc_sdma {
                self.sdma[g][c].wait(to)?;
            }
            self.devices[g].wait(to)?;
        }

        // ---- Reduce: device g sums its R staging slots into buf[g] chunk g. ----
        for g in 0..r {
            let (out, st) = (self.bufs[g].va(), self.stage[g].va());
            self.devices[g].arm_gather_reduce_local(out, st, r as u32, g as u32 * cl, cl, cl)?;
        }
        for g in 0..r {
            self.devices[g].wait(to)?;
        }

        // ---- All-gather: device g writes reduced chunk g -> buf[h] chunk g. ----
        // Peers (h != g): SDMA path first `sd_g`, CU path the remaining `cu_g`.
        for g in 0..r {
            let peers: Vec<usize> = (0..r).filter(|&h| h != g).collect();
            let dst_vas: Vec<u64> = (0..cu_g)
                .map(|p| self.bufs[peers[sd_g + p]].va() + (g as u32 * cl) as u64 * 4)
                .collect();
            {
                let slots = unsafe { self.hyb_gtbl[g].as_mut_slice_of::<u64>() };
                slots[..cu_g].copy_from_slice(&dst_vas);
            }
            let src_g = self.bufs[g].va() + (g as u32 * cl) as u64 * 4;
            for (p, &peer) in peers.iter().enumerate().take(sd_g) {
                let dst = self.bufs[peer].va() + (g as u32 * cl) as u64 * 4;
                self.sdma[g][p].copy_async(src_g, dst, cl as usize * 4);
            }
            let tbl = self.hyb_gtbl[g].va();
            let kwg = (cu_g as u32) * 32;
            self.devices[g].arm_p2p_broadcast(src_g, tbl, cu_g as u32, 0, cl4, kwg)?;
        }
        for g in 0..r {
            for p in 0..sd_g {
                self.sdma[g][p].wait(to)?;
            }
            self.devices[g].wait(to)?;
        }

        Ok(start.elapsed())
    }

    /// Reduce-scatter + all-gather over the sub-range [base, base+n) of `bufs`,
    /// every XGMI copy on the SDMA engines and the reduce on `devs`' CUs;
    /// `stage` is a private R-slot staging set. Phases sync host-side. Used both
    /// standalone and as the SDMA partition of the concurrent path.
    #[allow(clippy::too_many_arguments)]
    fn sdma_allreduce_range(
        devs: &mut [GpuDevice],
        sdma: &mut [Vec<crate::SdmaQueue>],
        bufs: &[DeviceBuffer],
        stage: &[DeviceBuffer],
        base: u32,
        n: u32,
        to: Duration,
    ) -> Result<()> {
        let r = devs.len();
        let cl = round_up4(n.div_ceil(r as u32));
        if r < 2 || cl == 0 || (n & 3) != 0 {
            return Err(anyhow!(
                "sdma range needs r>=2 and n a positive multiple of 4"
            ));
        }
        let chunk_len = |c: usize| -> u32 {
            let off = c as u32 * cl;
            if off >= n {
                0
            } else {
                cl.min(n - off)
            }
        };
        // Scatter: device g pushes chunk c -> stage[c] slot g (SDMA).
        for g in 0..r {
            for c in 0..r {
                let len = chunk_len(c);
                if len == 0 {
                    continue;
                }
                let src = bufs[g].va() + ((base + c as u32 * cl) as u64) * 4;
                let dst = stage[c].va() + (g as u32 * cl) as u64 * 4;
                sdma[g][c].copy_async(src, dst, len as usize * 4);
            }
        }
        for row in sdma.iter() {
            for (c, q) in row.iter().enumerate() {
                if chunk_len(c) > 0 {
                    q.wait(to)?;
                }
            }
        }
        // Reduce: device g sums its R staging slots into buf[g]'s chunk g (CUs).
        for g in 0..r {
            let len = chunk_len(g);
            if len == 0 {
                continue;
            }
            devs[g].arm_gather_reduce_local(
                bufs[g].va(),
                stage[g].va(),
                r as u32,
                base + g as u32 * cl,
                cl,
                len,
            )?;
        }
        for (g, dev) in devs.iter().enumerate() {
            if chunk_len(g) > 0 {
                dev.wait(to)?;
            }
        }
        // All-gather: device g pushes its reduced chunk g to every peer (SDMA).
        for g in 0..r {
            let len = chunk_len(g);
            if len == 0 {
                continue;
            }
            let src = bufs[g].va() + ((base + g as u32 * cl) as u64) * 4;
            for h in 0..r {
                let dst = bufs[h].va() + ((base + g as u32 * cl) as u64) * 4;
                sdma[g][h].copy_async(src, dst, len as usize * 4);
            }
        }
        for (g, row) in sdma.iter().enumerate() {
            if chunk_len(g) > 0 {
                for q in row.iter() {
                    q.wait(to)?;
                }
            }
        }
        Ok(())
    }

    /// Concurrent dual-engine all-reduce: the device-side one-shot kernel runs
    /// partition A [0, An) on the primary queues while a host-orchestrated SDMA
    /// all-reduce runs partition B [An, n) on the secondary queues + SDMA
    /// engines. The one-shot streams XGMI continuously, so partition B's
    /// reduce/host-sync gaps overlap it — the link stays full and aggregate
    /// egress approaches the dual-path ceiling (~378 GB/s, past RCCL). Requires
    /// n%4==0 with both partitions >= 4R; else falls back to the one-shot.
    pub fn all_reduce_sum_concurrent(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        if r < 2 || (n & 3) != 0 {
            return self.all_reduce_sum_oneshot();
        }
        let fa = std::env::var("MAINARCH_CONCURRENT_FA")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(50)
            .clamp(10, 90);
        let an = ((n as usize * fa / 100) as u32) & !3u32; // 4-aligned A partition
        let bn = n - an;
        if an < 4 * r as u32 || bn < 4 * r as u32 {
            return self.all_reduce_sum_oneshot();
        }
        // Lazy: second device handles (independent queue) for partition B's reduce.
        if self.devices2.is_empty() {
            let nodes: Vec<u32> = self.devices.iter().map(|d| d.node_id()).collect();
            let mut d2 = Vec::with_capacity(r);
            for &node in &nodes {
                d2.push(GpuDevice::open_shared(self.kfd.clone(), node)?);
            }
            self.devices2 = d2;
        }
        if self.sdma.is_empty() {
            let mut all = Vec::with_capacity(r);
            for d in &self.devices {
                let (kfd, node) = (d.kfd(), d.node_id());
                let mut qv = Vec::with_capacity(2 * r);
                for _ in 0..(2 * r) {
                    qv.push(crate::SdmaQueue::new_xgmi(kfd, node)?);
                }
                all.push(qv);
            }
            self.sdma = all;
        }
        if self.stage_b.is_empty() {
            let cl_cap = round_up4(self.cap.div_ceil(r as u32));
            let stage_elems = (r as u32 * cl_cap) as usize;
            let mut sb = Vec::with_capacity(r);
            for d in &self.devices {
                sb.push(d.kfd().alloc_vram(d.node_id(), stage_elems * 4)?);
            }
            for (bi, s) in sb.iter().enumerate() {
                for (di, dev) in self.devices.iter().enumerate() {
                    if bi != di {
                        self.kfd.map_buffer_to_peer(s, dev.node_id())?;
                    }
                }
            }
            self.stage_b = sb;
        }

        // One-shot parameters for partition A.
        let cl = round_up4(an.div_ceil(r as u32));
        let tiles = std::env::var("MAINARCH_ONESHOT_TILES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4u32)
            .max(1);
        let seq_base = self.seq;
        self.seq = self.seq.wrapping_add(tiles + 2);
        let wg = (self.oneshot_wg / r as u32).max(1) * r as u32;

        let start = Instant::now();
        // Launch A: device-side one-shot on [0, An), primary queues — no wait yet.
        for g in 0..r {
            let own = self.bufs[g].va();
            let peer_bufs = self.ptrs[g].va();
            let stage_ptrs = self.stage_ptrs[g].va();
            let my_stage = self.stage[g].va();
            let my_flags = self.flags[g].va();
            let peer_flag_ptrs = self.flag_ptrs[g].va();
            let gbar = self.gbar[g].va();
            self.devices[g].arm_allreduce_oneshot(
                own,
                peer_bufs,
                stage_ptrs,
                my_stage,
                my_flags,
                peer_flag_ptrs,
                gbar,
                r as u32,
                g as u32,
                cl,
                an,
                wg,
                seq_base,
                tiles,
            )?;
        }
        // Run B concurrently: SDMA all-reduce on [An, n), secondary queues.
        let b_res = Self::sdma_allreduce_range(
            &mut self.devices2,
            &mut self.sdma,
            &self.bufs,
            &self.stage_b,
            an,
            bn,
            OP_TIMEOUT,
        );
        // Wait for A (the one-shot), then surface any B error.
        for g in 0..r {
            self.devices[g].wait(OP_TIMEOUT)?;
        }
        b_res?;
        Ok(start.elapsed())
    }

    /// Device-side dual-path all-reduce: per device, the CU kernel writes the
    /// first `kfrac`% of each chunk while a host-pre-armed SDMA program writes
    /// the rest, the two coordinated entirely through memory semaphores
    /// (FENCE/POLL_REGMEM) — NO host round-trips between phases. Both write paths
    /// stream simultaneously and the device-side barriers keep XGMI busy across
    /// the reduce, so aggregate egress approaches the dual-path ceiling
    /// (~378 GB/s, past RCCL). Requires uniform chunks (n % (4R) == 0); else
    /// falls back to the one-shot.
    pub fn all_reduce_sum_devhybrid(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        if r < 2 || !(n as usize).is_multiple_of(4 * r) {
            return self.all_reduce_sum_oneshot();
        }
        let cl = n / r as u32; // uniform chunk (floats)
        let clv = cl / 4; // float4 per chunk
        let kfrac = std::env::var("MAINARCH_HYBRID_KFRAC")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(45)
            .min(100);
        let cu_clv = (clv * kfrac / 100).min(clv); // CU float4 per chunk (0 = pure SDMA)
        let cu_floats = cu_clv * 4; // CU floats per chunk, from start
        let sdma_floats = cl - cu_floats; // SDMA floats per chunk, from cu_floats
                                          // Lazy: SDMA queues (>=R per device) + per-device semaphores (2R+1 u32).
        if self.sdma.is_empty() {
            let mut all = Vec::with_capacity(r);
            for d in &self.devices {
                let (kfd, node) = (d.kfd(), d.node_id());
                let mut qv = Vec::with_capacity(2 * r);
                for _ in 0..(2 * r) {
                    qv.push(crate::SdmaQueue::new_xgmi(kfd, node)?);
                }
                all.push(qv);
            }
            self.sdma = all;
        }
        if self.dev_sem.is_empty() {
            for d in &self.devices {
                let mut s = d
                    .kfd()
                    .alloc_vram(d.node_id(), ((2 * r + 1) * 4).max(256))?;
                unsafe {
                    for v in s.as_mut_slice_of::<u32>() {
                        *v = 0;
                    }
                }
                self.dev_sem.push(s);
            }
        }
        let seq_base = self.seq;
        self.seq = self.seq.wrapping_add(2);
        let seq = seq_base + 1;
        let wg = (self.oneshot_wg / r as u32).max(1) * r as u32;
        let to = OP_TIMEOUT;
        let start = Instant::now();

        // 1. Pre-arm + commit each device's SDMA program (one queue per part):
        //    scatter SDMA-fraction -> FENCE scatter-done[q] -> POLL release ->
        //    gather SDMA-fraction -> FENCE gather-done[q]. Runs concurrently with
        //    the kernel; the kernel releases the gather after its reduce.
        for g in 0..r {
            let sem_va = self.dev_sem[g].va();
            let release_va = sem_va + (r as u64) * 4;
            for q in 0..r {
                if q == g {
                    continue; // local chunk handled fully by the CU kernel
                }
                if sdma_floats > 0 {
                    let src = self.bufs[g].va() + ((q as u32 * cl + cu_floats) as u64) * 4;
                    let dst = self.stage[q].va() + ((g as u32 * cl + cu_floats) as u64) * 4;
                    self.sdma[g][q].push_copy(src, dst, sdma_floats as usize * 4);
                }
                self.sdma[g][q].push_fence(sem_va + (q as u64) * 4, seq);
                self.sdma[g][q].push_poll_eq(release_va, seq, 0xffff_ffff);
                if sdma_floats > 0 {
                    let off = (g as u32 * cl + cu_floats) as u64;
                    let src = self.bufs[g].va() + off * 4;
                    let dst = self.bufs[q].va() + off * 4;
                    self.sdma[g][q].push_copy(src, dst, sdma_floats as usize * 4);
                }
                self.sdma[g][q].push_fence(sem_va + ((r + 1 + q) as u64) * 4, seq);
                self.sdma[g][q].commit();
            }
        }
        // 2. Launch the dual-path kernels (coordinate with the SDMA via sem).
        for g in 0..r {
            let own = self.bufs[g].va();
            let peer_bufs = self.ptrs[g].va();
            let stage_ptrs = self.stage_ptrs[g].va();
            let my_stage = self.stage[g].va();
            let my_flags = self.flags[g].va();
            let peer_flag_ptrs = self.flag_ptrs[g].va();
            let gbar = self.gbar[g].va();
            let sem_va = self.dev_sem[g].va();
            self.devices[g].arm_allreduce_dualpath(
                own,
                peer_bufs,
                stage_ptrs,
                my_stage,
                my_flags,
                peer_flag_ptrs,
                gbar,
                sem_va,
                r as u32,
                g as u32,
                cl,
                n,
                wg,
                seq_base,
                cu_clv,
                cu_clv,
            )?;
        }
        // 3. Wait the kernels, then drain the SDMA queues.
        for g in 0..r {
            self.devices[g].wait(to)?;
        }
        for g in 0..r {
            for q in 0..r {
                if q == g {
                    continue;
                }
                self.sdma[g][q].wait(to)?;
            }
        }
        Ok(start.elapsed())
    }

    /// Bidirectional pipelined all-reduce: reduce-scatter by READ (each GPU pulls
    /// its chunk from peers — INCOMING traffic) pipelined against all-gather by
    /// WRITE (push the reduced chunk to peers — OUTGOING traffic), tile by tile,
    /// so both XGMI link directions run simultaneously. The diagnostic showed our
    /// push-only scatter+gather run sequentially (988+988µs, one direction at a
    /// time); overlapping read(tile t) with write(tile t-1) uses the idle return
    /// path. No cross-device barriers needed: each peer slot is written by exactly
    /// one GPU and read only by that same GPU before it overwrites it. Requires
    /// n % (4*R*tiles) == 0; else falls back to the one-shot.
    pub fn all_reduce_sum_bidir(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        let t = std::env::var("MAINARCH_BIDIR_TILES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
            .max(1);
        if r < 2 || !(n as usize).is_multiple_of(4 * r * t as usize) {
            return self.all_reduce_sum_oneshot();
        }
        let cl = n / r as u32; // chunk floats (uniform)
        let tl = cl / t; // tile floats
        if self.sdma.is_empty() {
            let mut all = Vec::with_capacity(r);
            for d in &self.devices {
                let (kfd, node) = (d.kfd(), d.node_id());
                let mut qv = Vec::with_capacity(2 * r);
                for _ in 0..(2 * r) {
                    qv.push(crate::SdmaQueue::new_xgmi(kfd, node)?);
                }
                all.push(qv);
            }
            self.sdma = all;
        }
        if self.hyb_gtbl.is_empty() {
            for d in &self.devices {
                self.hyb_gtbl.push(d.alloc(r * 8)?);
            }
        }
        let to = OP_TIMEOUT;
        // Launch reduce-scatter READ of tile `tt` on every device: GPU g pulls
        // peer p's chunk-g tile into its own staging slot p (sdma[g][0..r]).
        let read_tile = |s: &mut Self, tt: u32| {
            for g in 0..r {
                for p in 0..r {
                    let src = s.bufs[p].va() + ((g as u32 * cl + tt * tl) as u64) * 4;
                    let dst = s.stage[g].va() + ((p as u32 * cl + tt * tl) as u64) * 4;
                    s.sdma[g][p].copy_async(src, dst, tl as usize * 4);
                }
            }
        };
        let wait_reads = |s: &Self| -> Result<()> {
            for g in 0..r {
                for p in 0..r {
                    s.sdma[g][p].wait(to)?;
                }
            }
            Ok(())
        };
        let reduce_tile = |s: &mut Self, tt: u32| -> Result<()> {
            for g in 0..r {
                let out = s.bufs[g].va();
                let st = s.stage[g].va() + (tt * tl) as u64 * 4;
                s.devices[g].arm_gather_reduce_local(
                    out,
                    st,
                    r as u32,
                    g as u32 * cl + tt * tl,
                    cl,
                    tl,
                )?;
            }
            for g in 0..r {
                s.devices[g].wait(to)?;
            }
            Ok(())
        };
        // Launch all-gather WRITE of tile `tt` via the CU stores (p2p_broadcast):
        // GPU g pushes its reduced chunk-g tile to every peer. Using the CUs (not
        // SDMA) means the OUTGOING write runs on different hardware than the
        // INCOMING SDMA read — so the two truly overlap on the bidirectional link.
        let write_tile = |s: &mut Self, tt: u32| -> Result<()> {
            for g in 0..r {
                let base = (g as u32 * cl + tt * tl) as u64 * 4;
                let vas: Vec<u64> = (0..r)
                    .filter(|&p| p != g)
                    .map(|p| s.bufs[p].va() + base)
                    .collect();
                {
                    let slots = unsafe { s.hyb_gtbl[g].as_mut_slice_of::<u64>() };
                    slots[..vas.len()].copy_from_slice(&vas);
                }
                let src = s.bufs[g].va() + base;
                let tbl = s.hyb_gtbl[g].va();
                let kwg = ((r - 1) as u32) * 32;
                s.devices[g].arm_p2p_broadcast(src, tbl, (r - 1) as u32, 0, tl / 4, kwg)?;
            }
            Ok(())
        };
        let wait_cu = |s: &Self| -> Result<()> {
            for g in 0..r {
                s.devices[g].wait(to)?;
            }
            Ok(())
        };

        let start = Instant::now();
        // Prologue: read + reduce tile 0.
        read_tile(self, 0);
        wait_reads(self)?;
        reduce_tile(self, 0)?;
        // Steady state: SDMA read(tt) INCOMING overlaps CU write(tt-1) OUTGOING —
        // different engines, opposite link directions, so both run at once.
        for tt in 1..t {
            read_tile(self, tt); // SDMA, incoming
            write_tile(self, tt - 1)?; // CU, outgoing: overlaps the read
            wait_cu(self)?; // CU write done
            wait_reads(self)?; // SDMA read done
            reduce_tile(self, tt)?; // CU reduce (queue now free)
        }
        // Epilogue: write the final tile.
        write_tile(self, t - 1)?;
        wait_cu(self)?;
        Ok(start.elapsed())
    }

    /// Bidirectional v2: reduce-scatter by CU READ (reduce_peers pulls each peer's
    /// chunk directly and sums in one pass — INCOMING, on the CUs, no staging)
    /// pipelined against all-gather by SDMA WRITE (push reduced chunk — OUTGOING,
    /// on the SDMA engines). CU-read and SDMA-write are different hardware on
    /// opposite link directions, so read(tile t) overlaps write(tile t-1). No
    /// cross-device barriers. Requires n % (4*R*tiles) == 0.
    pub fn all_reduce_sum_bidir2(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        let t = std::env::var("MAINARCH_BIDIR_TILES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(4)
            .max(1);
        if r < 2 || !(n as usize).is_multiple_of(4 * r * t as usize) {
            return self.all_reduce_sum_oneshot();
        }
        let cl = n / r as u32;
        let tl = cl / t;
        if self.sdma.is_empty() {
            let mut all = Vec::with_capacity(r);
            for d in &self.devices {
                let (kfd, node) = (d.kfd(), d.node_id());
                let mut qv = Vec::with_capacity(2 * r);
                for _ in 0..(2 * r) {
                    qv.push(crate::SdmaQueue::new_xgmi(kfd, node)?);
                }
                all.push(qv);
            }
            self.sdma = all;
        }
        if self.hyb_stbl.is_empty() {
            for d in &self.devices {
                self.hyb_stbl.push(d.alloc(r * 8)?);
                self.hyb_gtbl.push(d.alloc(r * 8)?);
            }
        }
        let to = OP_TIMEOUT;
        // CU read-reduce of tile tt: GPU g sums peer p's chunk-g tile directly
        // into own (reduce_peers reads all peers over XGMI — incoming).
        let read_reduce = |s: &mut Self, tt: u32| -> Result<()> {
            for g in 0..r {
                let base = (g as u32 * cl + tt * tl) as u64 * 4;
                let vas: Vec<u64> = (0..r).map(|p| s.bufs[p].va() + base).collect();
                {
                    let slots = unsafe { s.hyb_stbl[g].as_mut_slice_of::<u64>() };
                    slots[..r].copy_from_slice(&vas);
                }
                let dst = s.bufs[g].va() + base;
                let tbl = s.hyb_stbl[g].va();
                s.devices[g].arm_reduce_peers(dst, tbl, r as u32, tl)?;
            }
            Ok(())
        };
        let wait_cu = |s: &Self| -> Result<()> {
            for g in 0..r {
                s.devices[g].wait(to)?;
            }
            Ok(())
        };
        // SDMA write all-gather of tile tt: push reduced chunk-g tile to peers.
        let write_tile = |s: &mut Self, tt: u32| {
            for g in 0..r {
                let src = s.bufs[g].va() + ((g as u32 * cl + tt * tl) as u64) * 4;
                for p in 0..r {
                    if p == g {
                        continue;
                    }
                    let dst = s.bufs[p].va() + ((g as u32 * cl + tt * tl) as u64) * 4;
                    s.sdma[g][p].copy_async(src, dst, tl as usize * 4);
                }
            }
        };
        let wait_writes = |s: &Self| -> Result<()> {
            for g in 0..r {
                for p in 0..r {
                    if p != g {
                        s.sdma[g][p].wait(to)?;
                    }
                }
            }
            Ok(())
        };

        let start = Instant::now();
        read_reduce(self, 0)?;
        wait_cu(self)?;
        for tt in 1..t {
            read_reduce(self, tt)?; // CU, incoming
            write_tile(self, tt - 1); // SDMA, outgoing: overlaps the CU read-reduce
            wait_writes(self)?;
            wait_cu(self)?;
        }
        write_tile(self, t - 1);
        wait_writes(self)?;
        Ok(start.elapsed())
    }

    pub fn ranks(&self) -> usize {
        self.devices.len()
    }

    pub fn n(&self) -> u32 {
        self.n
    }

    /// Set the active element count for subsequent operations (<= capacity).
    pub fn set_n(&mut self, n: u32) -> Result<()> {
        if n > self.cap {
            return Err(anyhow!("active n {n} exceeds capacity {}", self.cap));
        }
        self.n = n;
        Ok(())
    }

    /// Load rank `r`'s input vector (host-visible write).
    pub fn set_input(&mut self, r: usize, data: &[f32]) -> Result<()> {
        let f = unsafe { self.bufs[r].as_mut_slice_of::<f32>() };
        if data.len() > f.len() {
            return Err(anyhow!("input longer than buffer"));
        }
        f[..data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Return the GPU VA of rank `r`'s resident all-reduce input/output buffer.
    /// Serving proofs use this to let an upstream GPU kernel write directly into
    /// the collective input buffer instead of writing to a temporary and
    /// host-staging the handoff.
    pub fn input_va(&self, r: usize) -> Result<u64> {
        self.bufs.get(r).map(DeviceBuffer::va).ok_or_else(|| {
            anyhow!(
                "rank {r} out of range for {} all-reduce buffers",
                self.bufs.len()
            )
        })
    }

    pub fn direct_ready_flags_va(&self) -> u64 {
        self.direct_ready_flags.va()
    }

    pub fn direct_ready_counts_va(&self) -> u64 {
        self.direct_ready_counts.va()
    }

    pub fn direct_ready_flag_value(&mut self, rank: usize) -> Result<u32> {
        let r = self.devices.len();
        if rank > r {
            return Err(anyhow!(
                "direct ready flag rank {rank} out of range for {} producer slots plus error slot",
                r
            ));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        Ok(unsafe { self.direct_ready_flags.as_mut_slice_of::<u32>()[rank] })
    }

    pub fn direct_ready_count_value(&mut self, rank: usize) -> Result<u32> {
        let r = self.devices.len();
        if rank >= r {
            return Err(anyhow!(
                "direct ready count rank {rank} out of range for {r} producer slots"
            ));
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        Ok(unsafe { self.direct_ready_counts.as_mut_slice_of::<u32>()[rank] })
    }

    pub fn direct_ready_flags_allocation_flags(&self) -> u32 {
        self.direct_ready_flags.allocation_flags()
    }

    pub fn direct_ready_counts_allocation_flags(&self) -> u32 {
        self.direct_ready_counts.allocation_flags()
    }

    pub fn direct_ready_flags_public_coherent_vram(&self) -> bool {
        self.direct_ready_flags.is_public_coherent_vram()
            && self.direct_ready_counts.is_public_coherent_vram()
    }

    pub fn reset_direct_ready_flags(&mut self) {
        let r = self.devices.len();
        unsafe {
            self.direct_ready_flags.as_mut_slice_of::<u32>()[..(r + 1)].fill(0);
            self.direct_ready_counts.as_mut_slice_of::<u32>()[..r].fill(0);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
    }

    pub fn direct_ready_flag_error(&mut self) -> u32 {
        let r = self.devices.len();
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        unsafe { self.direct_ready_flags.as_mut_slice_of::<u32>()[r] }
    }

    pub fn direct_ready_watchdog_ticks(&self) -> u64 {
        self.direct_ready_watchdog_ticks
    }

    pub fn direct_ready_watchdog_spins(&self) -> u64 {
        DIRECT_READY_WATCHDOG_SPINS
    }

    pub fn poison_direct_ready_producer_for_fault_injection(&mut self, rank: usize) -> Result<()> {
        let r = self.devices.len();
        if rank >= r {
            return Err(anyhow!(
                "direct ready producer fault injection rank {rank} out of range for {r} ranks"
            ));
        }
        unsafe {
            self.direct_ready_flags.as_mut_slice_of::<u32>()[rank] = 0;
            self.direct_ready_counts.as_mut_slice_of::<u32>()[rank] = u32::MAX;
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Read rank `r`'s currently resident input buffer before the all-reduce
    /// overwrites it with the broadcast output. This is a proof/readback helper,
    /// not part of the serving fast path.
    pub fn input_snapshot(&mut self, r: usize) -> Result<Vec<f32>> {
        let n = self.n as usize;
        let len = self.bufs.len();
        let buf = self
            .bufs
            .get_mut(r)
            .ok_or_else(|| anyhow!("rank {r} out of range for {len} all-reduce buffers"))?;
        let f = unsafe { buf.as_mut_slice_of::<f32>() };
        Ok(f[..n].to_vec())
    }

    /// Read rank `r`'s buffer (its post-all-reduce result).
    pub fn output(&mut self, r: usize) -> Vec<f32> {
        let n = self.n as usize;
        let f = unsafe { self.bufs[r].as_mut_slice_of::<f32>() };
        f[..n].to_vec()
    }

    /// Sum all-reduce: every rank's buffer becomes the elementwise sum across
    /// all ranks. Picks the latency-optimal direct scheme for small messages
    /// and the bandwidth-optimal reduce-scatter/all-gather for large ones.
    pub fn all_reduce_sum(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let bytes = self.n as usize * 4;
        let force = std::env::var("MAINARCH_ALLREDUCE_ALGO").ok();
        // Three regimes, best algorithm per size:
        //  - small: direct (2 dispatches, latency-optimal)
        //  - mid:   read-based reduce-scatter + push all-gather (fewer barriers)
        //  - large: write-based reduce-scatter + push all-gather (writes win at
        //           scale; the extra barrier amortizes)
        match force.as_deref() {
            Some("direct") => return self.all_reduce_sum_direct(),
            Some("rsag") | Some("rsag-read") => return self.all_reduce_sum_rsag(false),
            Some("rsag-write") => return self.all_reduce_sum_rsag(true),
            Some("oneshot") => return self.all_reduce_sum_oneshot(),
            Some("sdma") => return self.all_reduce_sum_sdma(),
            Some("hybrid") => return self.all_reduce_sum_hybrid(),
            Some("concurrent") => return self.all_reduce_sum_concurrent(),
            Some("devhybrid") => return self.all_reduce_sum_devhybrid(),
            Some("bidir") => return self.all_reduce_sum_bidir(),
            Some("bidir2") => return self.all_reduce_sum_bidir2(),
            _ => {}
        }
        if r < 2 || (self.n as usize) < 4 * r || bytes <= self.direct_max_bytes {
            self.all_reduce_sum_direct()
        } else if bytes >= self.oneshot_min_bytes && (self.n & 3) == 0 {
            // GPU-driven single-kernel: streams the whole op continuously with
            // device-side barriers — highest large-message bandwidth.
            self.all_reduce_sum_oneshot()
        } else if bytes < self.write_rs_min_bytes {
            self.all_reduce_sum_rsag(false)
        } else {
            self.all_reduce_sum_rsag(true)
        }
    }

    /// GPU-driven one-shot all-reduce: a single persistent kernel per GPU runs
    /// the whole op (scatter, reduce, all-gather) with device-side barriers, so
    /// one fixed co-resident grid streams the entire ~2(R-1)/R·n of XGMI traffic
    /// in one launch and reaches peak link bandwidth. Requires n a multiple of 4
    /// and at least one element per rank; otherwise falls back to the staged
    /// path.
    pub fn all_reduce_sum_oneshot(&mut self) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        let cl = round_up4(n.div_ceil(r as u32));
        if r < 2 || cl == 0 || (n & 3) != 0 {
            return self.all_reduce_sum_rsag(true);
        }
        let tiles = std::env::var("MAINARCH_ONESHOT_TILES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4u32)
            .max(1);
        let seq_base = self.seq;
        // Reserve `tiles` barrier sequence numbers (seq_base+1..=seq_base+tiles)
        // plus margin so the next op never collides.
        self.seq = self.seq.wrapping_add(tiles + 2);
        // Big-burst scatter/gather assign workgroups to peers round-robin, so the
        // grid must be a multiple of R for balanced per-link coverage. The grid
        // size is size-adaptive: large messages want a bigger co-resident grid
        // (more in-flight XGMI stores; MI355 1 GiB busbw peaks at wg=256), small
        // ones want fewer (the grid-barrier cost dominates). Env overrides.
        let bytes = self.n as usize * 4;
        let wg_target = std::env::var("MAINARCH_ONESHOT_WG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(if bytes >= 128 * 1024 * 1024 { 256 } else { 96 });
        let wg = (wg_target / r as u32).max(1) * r as u32;
        let start = Instant::now();
        for g in 0..r {
            let own = self.bufs[g].va();
            let peer_bufs = self.ptrs[g].va();
            let stage_ptrs = self.stage_ptrs[g].va();
            let my_stage = self.stage[g].va();
            let my_flags = self.flags[g].va();
            let peer_flag_ptrs = self.flag_ptrs[g].va();
            let gbar = self.gbar[g].va();
            self.devices[g].arm_allreduce_oneshot(
                own,
                peer_bufs,
                stage_ptrs,
                my_stage,
                my_flags,
                peer_flag_ptrs,
                gbar,
                r as u32,
                g as u32,
                cl,
                n,
                wg,
                seq_base,
                tiles,
            )?;
        }
        for g in 0..r {
            self.devices[g].wait(OP_TIMEOUT)?;
        }
        Ok(start.elapsed())
    }

    /// Latency-optimal: rank 0 sums all R peers in one `reduce_peers` kernel,
    /// then scatters back in one `broadcast_peers` kernel — two host
    /// round-trips, all traffic on rank 0's links. Best for small messages.
    pub fn all_reduce_sum_direct(&mut self) -> Result<Duration> {
        let r = self.devices.len() as u32;
        let n = self.n;
        let ptrs_va = self.ptrs[0].va();
        let dst_va = self.bufs[0].va();
        let start = Instant::now();
        if r == 1 {
            return Ok(start.elapsed());
        }
        self.devices[0].arm_reduce_peers(dst_va, ptrs_va, r, n)?;
        self.devices[0].wait(OP_TIMEOUT)?;
        if n >= 131_072 {
            self.devices[0].arm_broadcast_peers_skip0(dst_va, ptrs_va, r, n)?;
        } else {
            self.devices[0].arm_broadcast_peers(dst_va, ptrs_va, r, n)?;
        }
        self.devices[0].wait(OP_TIMEOUT)?;
        Ok(start.elapsed())
    }

    /// Dispatch the latency-optimal direct all-reduce after upstream queues
    /// publish device-side ready flags. The rank-0 reduce kernel performs a
    /// bounded CU-side acquire wait before reading peer outputs; broadcast is
    /// queued behind it and carries the final completion signal.
    pub fn dispatch_all_reduce_sum_direct_wait_ready_flags(&mut self) -> Result<()> {
        let r = self.devices.len() as u32;
        let n = self.n;
        if r == 1 {
            return Ok(());
        }
        let bytes = n as usize * 4;
        if bytes > self.direct_max_bytes {
            return Err(anyhow!(
                "direct ready-flag all-reduce is only enabled for latency-sized payloads; bytes={bytes} direct_max_bytes={}",
                self.direct_max_bytes
            ));
        }
        let ptrs_va = self.ptrs[0].va();
        let dst_va = self.bufs[0].va();
        self.devices[0].chain_next();
        self.devices[0].arm_reduce_peers_wait_ready_flags(
            dst_va,
            ptrs_va,
            self.direct_ready_flags.va(),
            r,
            n,
        )?;
        if n >= 131_072 {
            self.devices[0].arm_broadcast_peers_skip0(dst_va, ptrs_va, r, n)?;
        } else {
            self.devices[0].arm_broadcast_peers(dst_va, ptrs_va, r, n)?;
        }
        Ok(())
    }

    /// Wait for the final rank-0 direct all-reduce packet queued by
    /// `dispatch_all_reduce_sum_direct_wait_ready_flags`.
    pub fn wait_direct_ready_allreduce_completion(&mut self) -> Result<Duration> {
        let elapsed = self.devices[0].wait(OP_TIMEOUT)?;
        let err = self.direct_ready_flag_error();
        if err != 0 {
            return Err(anyhow!(
                "direct ready-flag all-reduce watchdog tripped: flags[parts]=0x{err:08x}"
            ));
        }
        Ok(elapsed)
    }

    /// Bandwidth-optimal: parallel reduce-scatter then push all-gather. Every
    /// GPU owns one chunk and drives its own XGMI links concurrently, so the
    /// full fabric is used (vs only rank 0's links in the direct scheme).
    ///
    /// `write_based` selects the reduce-scatter style: read-pull (each owner
    /// reads its chunk from peers; 2 barriers) for mid sizes, or write-push
    /// (each GPU scatters into peers' staging, then reduces locally; 3 barriers
    /// but all cross-GPU traffic is writes) for large sizes.
    pub fn all_reduce_sum_rsag(&mut self, write_based: bool) -> Result<Duration> {
        let r = self.devices.len();
        let n = self.n;
        // Uniform ceil chunk length, multiple of 4 so every chunk offset and
        // staging slot stays 16-byte aligned. Chunk g covers
        // [g*cl, min((g+1)*cl, n)).
        let cl = round_up4(n.div_ceil(r as u32));
        let start = Instant::now();
        if r < 2 || cl == 0 {
            return self.all_reduce_sum_direct();
        }
        let chunk_off = |g: usize| g as u32 * cl;
        let chunk_len = |g: usize| {
            let off = chunk_off(g);
            if off >= n {
                0
            } else {
                cl.min(n - off)
            }
        };

        // All-gather mode: push (each owner WRITES its chunk to all peers;
        // remote writes stream at near link BW — default) or pull (each GPU
        // reads the chunks it lacks — comparison only).
        let push = !matches!(std::env::var("MAINARCH_AG_MODE").as_deref(), Ok("pull"));

        // Write-based reduce-scatter needs an extra scatter phase (push into
        // peers' staging) before the local reduce.
        if write_based {
            for g in 0..r {
                let own = self.bufs[g].va();
                let sp = self.stage_ptrs[g].va();
                self.devices[g].arm_scatter_to_staging(own, sp, r as u32, cl, g as u32, n)?;
            }
            for g in 0..r {
                self.devices[g].wait(OP_TIMEOUT)?;
            }
        }

        // Reduce step: each GPU produces its own final chunk (read-pull
        // reduce-scatter, or local reduce of its staging slots). With push
        // all-gather the reduce and broadcast fuse on each device's queue — the
        // AQL barrier bit orders broadcast after reduce, and we sync only once
        // at the very end. So GPU g can broadcast its chunk while GPU h is still
        // reducing — the XGMI links never idle on a global barrier. This is
        // safe because each GPU exclusively owns its chunk's data flow.
        for g in 0..r {
            let len = chunk_len(g);
            if len == 0 {
                continue;
            }
            if push {
                self.devices[g].chain_next();
            }
            if write_based {
                let out = self.bufs[g].va();
                let st = self.stage[g].va();
                self.devices[g].arm_gather_reduce_local(
                    out,
                    st,
                    r as u32,
                    chunk_off(g),
                    cl,
                    len,
                )?;
            } else {
                let dst = self.bufs[g].va();
                let ptrs = self.ptrs[g].va();
                self.devices[g].arm_reduce_scatter(dst, ptrs, r as u32, chunk_off(g), len)?;
            }
        }

        if push {
            // Fused broadcast (no global barrier before it).
            for g in 0..r {
                let len = chunk_len(g);
                if len == 0 {
                    continue;
                }
                let src = self.bufs[g].va();
                let ptrs = self.ptrs[g].va();
                self.devices[g].arm_broadcast_chunk_skip_owner(
                    src,
                    ptrs,
                    r as u32,
                    chunk_off(g),
                    len,
                    g as u32,
                )?;
            }
            for g in 0..r {
                if chunk_len(g) != 0 {
                    self.devices[g].wait(OP_TIMEOUT)?;
                }
            }
        } else {
            // Pull all-gather reads peers' reduced chunks, so it needs a global
            // barrier after the reduce step.
            for g in 0..r {
                if chunk_len(g) != 0 {
                    self.devices[g].wait(OP_TIMEOUT)?;
                }
            }
            for g in 0..r {
                let dst = self.bufs[g].va();
                let ptrs = self.ptrs[g].va();
                self.devices[g].arm_all_gather(dst, ptrs, cl, r as u32, n, g as u32)?;
            }
            for g in 0..r {
                self.devices[g].wait(OP_TIMEOUT)?;
            }
        }

        Ok(start.elapsed())
    }

    /// Diagnostic: time each phase of the write-based path in isolation
    /// (scatter, local-reduce, push all-gather) with a barrier around each, to
    /// see which dominates and the effective per-GPU XGMI write bandwidth.
    /// Returns (scatter, local_reduce, all_gather) average durations.
    pub fn probe_phases(&mut self, iters: usize) -> Result<(Duration, Duration, Duration)> {
        let r = self.devices.len();
        let n = self.n;
        let cl = round_up4(n.div_ceil(r as u32));
        let chunk_off = |g: usize| g as u32 * cl;
        let chunk_len = |g: usize| {
            let off = chunk_off(g);
            if off >= n {
                0
            } else {
                cl.min(n - off)
            }
        };
        let (mut t_sc, mut t_lr, mut t_ag) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        for _ in 0..iters.max(1) {
            // scatter
            let s = Instant::now();
            for g in 0..r {
                let own = self.bufs[g].va();
                let sp = self.stage_ptrs[g].va();
                self.devices[g].arm_scatter_to_staging(own, sp, r as u32, cl, g as u32, n)?;
            }
            for g in 0..r {
                self.devices[g].wait(OP_TIMEOUT)?;
            }
            t_sc += s.elapsed();
            // local reduce
            let s = Instant::now();
            for g in 0..r {
                let len = chunk_len(g);
                if len == 0 {
                    continue;
                }
                let out = self.bufs[g].va();
                let st = self.stage[g].va();
                self.devices[g].arm_gather_reduce_local(
                    out,
                    st,
                    r as u32,
                    chunk_off(g),
                    cl,
                    len,
                )?;
            }
            for g in 0..r {
                if chunk_len(g) != 0 {
                    self.devices[g].wait(OP_TIMEOUT)?;
                }
            }
            t_lr += s.elapsed();
            // push all-gather
            let s = Instant::now();
            for g in 0..r {
                let len = chunk_len(g);
                if len == 0 {
                    continue;
                }
                let src = self.bufs[g].va();
                let ptrs = self.ptrs[g].va();
                self.devices[g].arm_broadcast_chunk(src, ptrs, r as u32, chunk_off(g), len)?;
            }
            for g in 0..r {
                if chunk_len(g) != 0 {
                    self.devices[g].wait(OP_TIMEOUT)?;
                }
            }
            t_ag += s.elapsed();
        }
        let k = iters.max(1) as u32;
        Ok((t_sc / k, t_lr / k, t_ag / k))
    }

    /// Time `iters` all-reduce calls (after `warmup`) and return the average
    /// GPU-phase duration. Inputs are reset between iterations so values stay
    /// bounded; only the all-reduce itself is timed.
    pub fn benchmark(&mut self, iters: usize, warmup: usize) -> Result<Duration> {
        let r = self.ranks();
        let n = self.n as usize;
        let reset = |ar: &mut Self| -> Result<()> {
            for rank in 0..r {
                let f = unsafe { ar.bufs[rank].as_mut_slice_of::<f32>() };
                for (i, v) in f.iter_mut().enumerate().take(n) {
                    *v = rank as f32 + i as f32 * 1e-3;
                }
            }
            Ok(())
        };
        for _ in 0..warmup {
            reset(self)?;
            self.all_reduce_sum()?;
        }
        let mut total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset(self)?;
            total += self.all_reduce_sum()?;
        }
        Ok(total / iters.max(1) as u32)
    }

    /// Benchmark the opt-in persistent rank-0 direct prototype. The kernel is
    /// launched once and each measured operation is triggered through a
    /// host-visible control block, so this measures the direct algorithm without
    /// per-op AQL dispatch overhead. It is intentionally scoped to the current
    /// active `n` and does not affect the production selector.
    pub fn benchmark_persistent_direct(&mut self, iters: usize, warmup: usize) -> Result<Duration> {
        let r = self.ranks();
        if r < 2 {
            return self.benchmark(iters, warmup);
        }
        let reset_n = self.cap as usize;
        let total_ops = warmup
            .checked_add(iters.max(1))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| anyhow!("persistent direct op count overflows u32"))?;
        let persistent_wg = std::env::var("MAINARCH_PERSIST_DIRECT_WG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16);
        let mut ctrl = self.devices[0].alloc(64)?;
        unsafe {
            for v in ctrl.as_mut_slice_of::<u32>() {
                *v = 0;
            }
            for v in self.gbar[0].as_mut_slice_of::<u32>() {
                *v = 0;
            }
        }
        let ctrl_ptr = unsafe { ctrl.as_mut_slice_of::<u32>().as_mut_ptr() };

        let reset = |ar: &mut Self| -> Result<()> {
            for rank in 0..r {
                let f = unsafe { ar.bufs[rank].as_mut_slice_of::<f32>() };
                for (i, v) in f.iter_mut().enumerate().take(reset_n) {
                    *v = rank as f32 + i as f32 * 1e-3;
                }
            }
            Ok(())
        };

        self.devices[0].arm_allreduce_direct_persistent(
            self.bufs[0].va(),
            self.ptrs[0].va(),
            ctrl.va(),
            self.gbar[0].va(),
            r as u32,
            self.n,
            total_ops,
            persistent_wg,
        )?;

        let mut total = Duration::ZERO;
        for seq in 1..=total_ops {
            reset(self)?;
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            let start = Instant::now();
            unsafe {
                std::ptr::write_volatile(ctrl_ptr, seq);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            crate::gpu::poll_until(OP_TIMEOUT, || unsafe {
                std::ptr::read_volatile(ctrl_ptr.add(1)) >= seq
                    || std::ptr::read_volatile(ctrl_ptr.add(2)) != 0
            })
            .with_context(|| format!("persistent direct all-reduce op {seq} did not complete"))?;
            let err = unsafe { std::ptr::read_volatile(ctrl_ptr.add(2)) };
            if err != 0 {
                return Err(anyhow!(
                    "persistent direct all-reduce error marker 0x{err:08x}"
                ));
            }
            if seq > warmup as u32 {
                total += start.elapsed();
            }
        }
        self.devices[0].wait(OP_TIMEOUT)?;

        let inputs: Vec<Vec<f32>> = (0..r)
            .map(|rank| {
                (0..self.n as usize)
                    .map(|i| rank as f32 + i as f32 * 1e-3)
                    .collect()
            })
            .collect();
        let expect = sequential_sum_reference(&inputs, self.n);
        for rank in 0..r {
            let got = self.output(rank);
            for i in 0..self.n as usize {
                if got[i].to_bits() != expect[i].to_bits() {
                    return Err(anyhow!(
                        "persistent direct mismatch on rank {rank}[{i}]: gpu={} cpu={}",
                        got[i],
                        expect[i]
                    ));
                }
            }
        }

        Ok(total / iters.max(1) as u32)
    }

    /// Benchmark the opt-in persistent all-rank DDA-flat prototype. One
    /// resident kernel per rank reads all peers' input buffers directly over
    /// XGMI and writes that rank's reduced vector into a separate output buffer.
    /// The separate output keeps peer inputs immutable during the op, avoiding
    /// the in-place read/write race that an all-rank direct variant would have.
    pub fn benchmark_persistent_dda_flat(
        &mut self,
        iters: usize,
        warmup: usize,
    ) -> Result<Duration> {
        let r = self.ranks();
        if r < 2 {
            return self.benchmark(iters, warmup);
        }
        let n = self.n as usize;
        let reset_n = self.cap as usize;
        let total_ops = warmup
            .checked_add(iters.max(1))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| anyhow!("persistent DDA op count overflows u32"))?;
        let persistent_wg = persistent_dda_wg(self.n as usize * 4);

        let mut ctrls = Vec::with_capacity(r);
        let mut outs = Vec::with_capacity(r);
        for rank in 0..r {
            let mut ctrl = self.devices[rank].alloc(64)?;
            let mut out = self.devices[rank].alloc_device(reset_n * 4)?;
            unsafe {
                for v in ctrl.as_mut_slice_of::<u32>() {
                    *v = 0;
                }
                for v in out.as_mut_slice_of::<f32>().iter_mut().take(reset_n) {
                    *v = 0.0;
                }
                for v in self.gbar[rank].as_mut_slice_of::<u32>() {
                    *v = 0;
                }
            }
            ctrls.push(ctrl);
            outs.push(out);
        }
        let ctrl_ptrs: Vec<*mut u32> = ctrls
            .iter_mut()
            .map(|ctrl| unsafe { ctrl.as_mut_slice_of::<u32>().as_mut_ptr() })
            .collect();

        for rank in 0..r {
            self.devices[rank].arm_allreduce_dda_persistent(
                outs[rank].va(),
                self.ptrs[rank].va(),
                ctrls[rank].va(),
                self.gbar[rank].va(),
                r as u32,
                self.n,
                total_ops,
                persistent_wg,
            )?;
        }

        let mut total = Duration::ZERO;
        for seq in 1..=total_ops {
            for rank in 0..r {
                let f = unsafe { self.bufs[rank].as_mut_slice_of::<f32>() };
                for (i, v) in f.iter_mut().enumerate().take(reset_n) {
                    *v = rank as f32 + i as f32 * 1e-3;
                }
                let out = unsafe { outs[rank].as_mut_slice_of::<f32>() };
                for v in out.iter_mut().take(n) {
                    *v = 0.0;
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            let start = Instant::now();
            for &ctrl_ptr in &ctrl_ptrs {
                unsafe {
                    std::ptr::write_volatile(ctrl_ptr, seq);
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            crate::gpu::poll_until(OP_TIMEOUT, || unsafe {
                let mut all_done = true;
                for &ctrl_ptr in &ctrl_ptrs {
                    if std::ptr::read_volatile(ctrl_ptr.add(2)) != 0 {
                        return true;
                    }
                    if std::ptr::read_volatile(ctrl_ptr.add(1)) < seq {
                        all_done = false;
                    }
                }
                all_done
            })
            .with_context(|| format!("persistent DDA all-reduce op {seq} did not complete"))?;
            for (rank, &ctrl_ptr) in ctrl_ptrs.iter().enumerate() {
                let err = unsafe { std::ptr::read_volatile(ctrl_ptr.add(2)) };
                if err != 0 {
                    return Err(anyhow!(
                        "persistent DDA all-reduce rank {rank} error marker 0x{err:08x}"
                    ));
                }
            }
            if seq > warmup as u32 {
                total += start.elapsed();
            }
        }
        for rank in 0..r {
            self.devices[rank].wait(OP_TIMEOUT)?;
        }

        let inputs: Vec<Vec<f32>> = (0..r)
            .map(|rank| {
                (0..self.n as usize)
                    .map(|i| rank as f32 + i as f32 * 1e-3)
                    .collect()
            })
            .collect();
        let expect = sequential_sum_reference(&inputs, self.n);
        for rank in 0..r {
            let got = unsafe { outs[rank].as_mut_slice_of::<f32>() };
            for i in 0..self.n as usize {
                if got[i].to_bits() != expect[i].to_bits() {
                    return Err(anyhow!(
                        "persistent DDA mismatch on rank {rank}[{i}]: gpu={} cpu={}",
                        got[i],
                        expect[i]
                    ));
                }
            }
        }

        Ok(total / iters.max(1) as u32)
    }

    /// Benchmark the persistent all-rank DDA-flat prototype with device peer
    /// flags for start, ready, and consumed synchronization. The host writes
    /// only rank 0's go flag and observes rank 0's done flag; all cross-rank
    /// start/exit ordering is carried by pre-mapped peer flag slots.
    pub fn benchmark_persistent_dda_peer_flat(
        &mut self,
        iters: usize,
        warmup: usize,
    ) -> Result<Duration> {
        let r = self.ranks();
        if r < 2 {
            return self.benchmark(iters, warmup);
        }
        let n = self.n as usize;
        let reset_n = self.cap as usize;
        let total_ops = warmup
            .checked_add(iters.max(1))
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| anyhow!("persistent peer-flag DDA op count overflows u32"))?;
        let persistent_wg = persistent_dda_peer_wg(self.n as usize * 4);

        let mut ctrls = Vec::with_capacity(r);
        let mut outs = Vec::with_capacity(r);
        for rank in 0..r {
            let mut ctrl = self.devices[rank].alloc(64)?;
            let mut out = self.devices[rank].alloc_device(reset_n * 4)?;
            unsafe {
                for v in ctrl.as_mut_slice_of::<u32>() {
                    *v = 0;
                }
                for v in out.as_mut_slice_of::<f32>().iter_mut().take(reset_n) {
                    *v = 0.0;
                }
                for v in self.flags[rank].as_mut_slice_of::<u32>() {
                    *v = 0;
                }
                for v in self.gbar[rank].as_mut_slice_of::<u32>() {
                    *v = 0;
                }
            }
            ctrls.push(ctrl);
            outs.push(out);
        }
        let ctrl_ptrs: Vec<*mut u32> = ctrls
            .iter_mut()
            .map(|ctrl| unsafe { ctrl.as_mut_slice_of::<u32>().as_mut_ptr() })
            .collect();

        for rank in 0..r {
            self.devices[rank].arm_allreduce_dda_peer_persistent(
                outs[rank].va(),
                self.ptrs[rank].va(),
                ctrls[rank].va(),
                self.flags[rank].va(),
                self.flag_ptrs[rank].va(),
                self.gbar[rank].va(),
                r as u32,
                rank as u32,
                self.n,
                total_ops,
                persistent_wg,
            )?;
        }

        let mut total = Duration::ZERO;
        for seq in 1..=total_ops {
            for rank in 0..r {
                let f = unsafe { self.bufs[rank].as_mut_slice_of::<f32>() };
                for (i, v) in f.iter_mut().enumerate().take(reset_n) {
                    *v = rank as f32 + i as f32 * 1e-3;
                }
                let out = unsafe { outs[rank].as_mut_slice_of::<f32>() };
                for v in out.iter_mut().take(n) {
                    *v = 0.0;
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            let start = Instant::now();
            unsafe {
                std::ptr::write_volatile(ctrl_ptrs[0], seq);
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            crate::gpu::poll_until(OP_TIMEOUT, || unsafe {
                if std::ptr::read_volatile(ctrl_ptrs[0].add(1)) >= seq {
                    return true;
                }
                for &ctrl_ptr in &ctrl_ptrs {
                    if std::ptr::read_volatile(ctrl_ptr.add(2)) != 0 {
                        return true;
                    }
                }
                false
            })
            .with_context(|| {
                format!("persistent peer-flag DDA all-reduce op {seq} did not complete")
            })?;
            for (rank, &ctrl_ptr) in ctrl_ptrs.iter().enumerate() {
                let err = unsafe { std::ptr::read_volatile(ctrl_ptr.add(2)) };
                if err != 0 {
                    return Err(anyhow!(
                        "persistent peer-flag DDA all-reduce rank {rank} error marker 0x{err:08x}"
                    ));
                }
            }
            if seq > warmup as u32 {
                total += start.elapsed();
            }
        }
        for rank in 0..r {
            self.devices[rank].wait(OP_TIMEOUT)?;
        }

        let inputs: Vec<Vec<f32>> = (0..r)
            .map(|rank| {
                (0..self.n as usize)
                    .map(|i| rank as f32 + i as f32 * 1e-3)
                    .collect()
            })
            .collect();
        let expect = sequential_sum_reference(&inputs, self.n);
        for rank in 0..r {
            let got = unsafe { outs[rank].as_mut_slice_of::<f32>() };
            for i in 0..self.n as usize {
                if got[i].to_bits() != expect[i].to_bits() {
                    return Err(anyhow!(
                        "persistent peer-flag DDA mismatch on rank {rank}[{i}]: gpu={} cpu={}",
                        got[i],
                        expect[i]
                    ));
                }
            }
        }

        Ok(total / iters.max(1) as u32)
    }

    /// Benchmark and validate the grid-strided fused direct all-reduce +
    /// RMSNorm substrate. Returns (average duration, max relative error).
    pub fn benchmark_fused_direct_rmsnorm(
        &mut self,
        iters: usize,
        warmup: usize,
        eps: f32,
    ) -> Result<(Duration, f64)> {
        let r = self.ranks();
        let n = self.n as usize;
        if n == 0 {
            return Err(anyhow!("fused allreduce+rmsnorm requires n > 0"));
        }
        let num_wg = fused_rmsnorm_grid_wg(n);
        let mut weight = self.devices[0].alloc_device(n * 4)?;
        let mut partial = self.devices[0].alloc_device(num_wg as usize * 4)?;
        unsafe {
            let w = weight.as_mut_slice_of::<f32>();
            for (i, v) in w.iter_mut().enumerate().take(n) {
                *v = 1.0 + (i % 7) as f32 * 1e-3;
            }
        }
        let reset = |ar: &mut Self| -> Result<()> {
            for rank in 0..r {
                let f = unsafe { ar.bufs[rank].as_mut_slice_of::<f32>() };
                for (i, v) in f.iter_mut().enumerate().take(n) {
                    *v = rank as f32 + i as f32 * 1e-3;
                }
            }
            Ok(())
        };
        for _ in 0..warmup {
            reset(self)?;
            reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
            self.devices[0].arm_allreduce_direct_rmsnorm_grid(
                self.bufs[0].va(),
                self.ptrs[0].va(),
                weight.va(),
                partial.va(),
                self.gbar[0].va(),
                r as u32,
                self.n,
                eps,
                num_wg,
            )?;
            self.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset(self)?;
            reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
            let start = Instant::now();
            self.devices[0].arm_allreduce_direct_rmsnorm_grid(
                self.bufs[0].va(),
                self.ptrs[0].va(),
                weight.va(),
                partial.va(),
                self.gbar[0].va(),
                r as u32,
                self.n,
                eps,
                num_wg,
            )?;
            self.devices[0].wait(OP_TIMEOUT)?;
            total += start.elapsed();
        }

        let weights = unsafe { weight.as_mut_slice_of::<f32>() };
        let mut reduced = vec![0.0f64; n];
        for rank in 0..r {
            for (i, v) in reduced.iter_mut().enumerate() {
                *v += rank as f64 + i as f64 * 1e-3;
            }
        }
        let ss: f64 = reduced.iter().map(|v| v * v).sum();
        let inv = 1.0f64 / (ss / n as f64 + eps as f64).sqrt();
        let expect: Vec<f64> = reduced
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * inv * *w as f64)
            .collect();

        let mut max_rel = 0.0f64;
        for rank in 0..r {
            let got = self.output(rank);
            for i in 0..n {
                let e = expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_rel = max_rel.max(rel);
            }
        }
        if max_rel > 2e-5 {
            return Err(anyhow!(
                "fused allreduce+rmsnorm mismatch: max rel err {max_rel:.3e} > 2e-5"
            ));
        }

        Ok((total / iters.max(1) as u32, max_rel))
    }

    /// Benchmark and validate the grid-strided fused direct all-reduce +
    /// residual add/writeback + RMSNorm substrate. Returns
    /// (average duration, output max relative error, residual max relative error).
    pub fn benchmark_fused_direct_residual_rmsnorm(
        &mut self,
        iters: usize,
        warmup: usize,
        eps: f32,
    ) -> Result<(Duration, f64, f64)> {
        let r = self.ranks();
        let n = self.n as usize;
        if n == 0 {
            return Err(anyhow!("fused allreduce+residual+rmsnorm requires n > 0"));
        }
        let num_wg = fused_rmsnorm_grid_wg(n);
        let mut weight = self.devices[0].alloc_device(n * 4)?;
        let mut partial = self.devices[0].alloc_device(num_wg as usize * 4)?;
        let mut residuals = Vec::with_capacity(r);
        for rank in 0..r {
            residuals.push(self.devices[rank].alloc_device(n * 4)?);
        }
        for (bi, residual) in residuals.iter().enumerate() {
            for (di, dev) in self.devices.iter().enumerate() {
                if bi != di {
                    self.kfd
                        .map_buffer_to_peer(residual, dev.node_id())
                        .with_context(|| {
                            format!("mapping residual {bi} into peer node {}", dev.node_id())
                        })?;
                }
            }
        }
        let mut residual_ptrs = self.devices[0].alloc_device(r * 8)?;
        unsafe {
            let ptrs = residual_ptrs.as_mut_slice_of::<u64>();
            for rank in 0..r {
                ptrs[rank] = residuals[rank].va();
            }
            let w = weight.as_mut_slice_of::<f32>();
            for (i, v) in w.iter_mut().enumerate().take(n) {
                *v = 1.0 + (i % 7) as f32 * 1e-3;
            }
        }
        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(self, &mut residuals)?;
            reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
            self.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
                self.bufs[0].va(),
                self.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                partial.va(),
                self.gbar[0].va(),
                r as u32,
                self.n,
                eps,
                num_wg,
            )?;
            self.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(self, &mut residuals)?;
            reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
            let start = Instant::now();
            self.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
                self.bufs[0].va(),
                self.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                partial.va(),
                self.gbar[0].va(),
                r as u32,
                self.n,
                eps,
                num_wg,
            )?;
            self.devices[0].wait(OP_TIMEOUT)?;
            total += start.elapsed();
        }

        let weights = unsafe { weight.as_mut_slice_of::<f32>() };
        let mut reduced = vec![0.0f64; n];
        for rank in 0..r {
            for (i, v) in reduced.iter_mut().enumerate() {
                *v += rank as f64 + i as f64 * 1e-3;
            }
        }
        let residual_expect: Vec<f64> = reduced
            .iter()
            .enumerate()
            .map(|(i, v)| *v + residual_seed(i))
            .collect();
        let ss: f64 = residual_expect.iter().map(|v| v * v).sum();
        let inv = 1.0f64 / (ss / n as f64 + eps as f64).sqrt();
        let output_expect: Vec<f64> = residual_expect
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * inv * *w as f64)
            .collect();

        let mut max_out_rel = 0.0f64;
        for rank in 0..r {
            let got = self.output(rank);
            for i in 0..n {
                let e = output_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_out_rel = max_out_rel.max(rel);
            }
        }

        let mut max_res_rel = 0.0f64;
        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }
        if max_out_rel > 2e-5 || max_res_rel > 2e-5 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm mismatch: max_out_rel {max_out_rel:.3e}, max_res_rel {max_res_rel:.3e}"
            ));
        }

        Ok((total / iters.max(1) as u32, max_out_rel, max_res_rel))
    }

    /// Run one supplied fused direct all-reduce + residual + RMSNorm operation.
    /// Inputs are f32 contribution vectors, one per rank. The residual stream is
    /// replicated to every rank, matching the decode-side TP boundary after a
    /// row-parallel projection. Rank 0 launches the direct XGMI kernel, which
    /// sums `inputs`, writes the reduced residual to every residual buffer, and
    /// broadcasts the normalized f32 output to every rank input buffer.
    pub fn fused_direct_residual_rmsnorm_once(
        &mut self,
        inputs: &[Vec<f32>],
        residual: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<FusedResidualRmsnormProof> {
        let r = self.ranks();
        let n = self.n as usize;
        if inputs.len() != r {
            return Err(anyhow!(
                "fused supplied proof expected {r} rank inputs, got {}",
                inputs.len()
            ));
        }
        if residual.len() != n || weight.len() != n {
            return Err(anyhow!(
                "fused supplied proof length mismatch: n={n} residual={} weight={}",
                residual.len(),
                weight.len()
            ));
        }
        if n == 0 {
            return Err(anyhow!("fused supplied proof requires n > 0"));
        }
        for (rank, input) in inputs.iter().enumerate() {
            if input.len() != n {
                return Err(anyhow!(
                    "fused supplied proof rank {rank} input length {} != n {n}",
                    input.len()
                ));
            }
            self.set_input(rank, input)?;
        }
        self.fused_direct_residual_rmsnorm_resident_once(residual, weight, eps)
    }

    /// Run one fused direct all-reduce + residual + RMSNorm operation using the
    /// already-populated resident rank input buffers. Upstream decode kernels can
    /// write directly into `input_va(rank)` and then call this without an eager
    /// host-staged input copy.
    pub fn fused_direct_residual_rmsnorm_resident_once(
        &mut self,
        residual: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<FusedResidualRmsnormProof> {
        self.fused_direct_residual_rmsnorm_resident_impl(residual, weight, eps, None)
    }

    /// Run the resident fused all-reduce + residual + RMSNorm operation and
    /// additionally cast rank 0's reduced residual stream into a caller-owned
    /// f16 buffer. This keeps the post-attention residual handoff resident for
    /// downstream decode kernels while preserving the proof readback path.
    pub fn fused_direct_residual_rmsnorm_resident_with_rank0_residual_f16_once(
        &mut self,
        residual: &[f32],
        weight: &[f32],
        eps: f32,
        rank0_residual_f16_va: u64,
    ) -> Result<FusedResidualRmsnormProof> {
        self.fused_direct_residual_rmsnorm_resident_impl(
            residual,
            weight,
            eps,
            Some(Rank0ResidualCast::F16(rank0_residual_f16_va)),
        )
    }

    /// Run the resident fused all-reduce + residual + RMSNorm operation and
    /// additionally cast rank 0's reduced residual stream into a caller-owned
    /// bf16 buffer. This preserves Qwen BF16 residual range while keeping the
    /// proof readback path in f32.
    pub fn fused_direct_residual_rmsnorm_resident_with_rank0_residual_bf16_once(
        &mut self,
        residual: &[f32],
        weight: &[f32],
        eps: f32,
        rank0_residual_bf16_va: u64,
    ) -> Result<FusedResidualRmsnormProof> {
        self.fused_direct_residual_rmsnorm_resident_impl(
            residual,
            weight,
            eps,
            Some(Rank0ResidualCast::Bf16(rank0_residual_bf16_va)),
        )
    }

    fn fused_direct_residual_rmsnorm_resident_impl(
        &mut self,
        residual: &[f32],
        weight: &[f32],
        eps: f32,
        rank0_residual_cast: Option<Rank0ResidualCast>,
    ) -> Result<FusedResidualRmsnormProof> {
        let r = self.ranks();
        let n = self.n as usize;
        if residual.len() != n || weight.len() != n {
            return Err(anyhow!(
                "resident fused proof length mismatch: n={n} residual={} weight={}",
                residual.len(),
                weight.len()
            ));
        }
        if n == 0 {
            return Err(anyhow!("resident fused proof requires n > 0"));
        }
        let num_wg = fused_rmsnorm_grid_wg(n);
        let mut weight_dev = self.devices[0].alloc_device(n * 4)?;
        let mut partial = self.devices[0].alloc_device(num_wg as usize * 4)?;
        unsafe {
            weight_dev.as_mut_slice_of::<f32>()[..n].copy_from_slice(weight);
            for rank in 0..r {
                self.residuals[rank].as_mut_slice_of::<f32>()[..n].copy_from_slice(residual);
            }
        }
        reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
        let start = Instant::now();
        self.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
            self.bufs[0].va(),
            self.ptrs[0].va(),
            self.residual_ptrs[0].va(),
            weight_dev.va(),
            partial.va(),
            self.gbar[0].va(),
            r as u32,
            self.n,
            eps,
            num_wg,
        )?;
        self.devices[0].wait(OP_TIMEOUT)?;
        if let Some(cast) = rank0_residual_cast {
            match cast {
                Rank0ResidualCast::F16(dst_va) => {
                    self.devices[0].arm_cast_f32_f16(self.residuals[0].va(), dst_va, self.n)?;
                    self.devices[0].wait(OP_TIMEOUT)?;
                }
                Rank0ResidualCast::Bf16(dst_va) => {
                    self.devices[0].arm_cast_f32_bf16(self.residuals[0].va(), dst_va, self.n)?;
                    self.devices[0].wait(OP_TIMEOUT)?;
                }
            }
        }
        let time = start.elapsed();
        let outputs = (0..r).map(|rank| self.output(rank)).collect::<Vec<_>>();
        let residuals_out = self
            .residuals
            .iter_mut()
            .map(|buf| unsafe { buf.as_mut_slice_of::<f32>()[..n].to_vec() })
            .collect::<Vec<_>>();
        Ok(FusedResidualRmsnormProof {
            time,
            outputs,
            residuals: residuals_out,
        })
    }

    /// Run one fused direct all-reduce + f16 residual update + f16 RMSNorm
    /// operation using already-populated resident rank input buffers. This is
    /// the decode-side MLP boundary form: peer MLP down-projection outputs are
    /// f32 in `input_va(rank)`, the residual stream is f16 on rank 0, and the
    /// next-layer normalized input is written as f16.
    pub fn fused_direct_residual_f16_rmsnorm_f16_resident_once(
        &mut self,
        rank0_residual_f16_va: u64,
        rank0_weight_f16_va: u64,
        rank0_output_f16_va: u64,
        eps: f32,
    ) -> Result<Duration> {
        let r = self.ranks();
        let n = self.n as usize;
        if n == 0 {
            return Err(anyhow!("resident fused f16 proof requires n > 0"));
        }
        let num_wg = fused_rmsnorm_grid_wg(n);
        let partial_bytes = num_wg as usize * 4;
        let needs_partial = self
            .resident_f16_partial
            .as_ref()
            .is_none_or(|partial| partial.len() < partial_bytes);
        if needs_partial {
            self.resident_f16_partial = Some(self.devices[0].alloc_device(partial_bytes)?);
        }
        let partial_va = {
            let partial = self
                .resident_f16_partial
                .as_mut()
                .ok_or_else(|| anyhow!("resident fused f16 partial workspace missing"))?;
            reset_fused_rmsnorm_sync_buffers(&mut self.gbar[0], partial, num_wg)?;
            partial.va()
        };
        let start = Instant::now();
        unsafe {
            self.devices[0].arm_allreduce_direct_residual_f16_rmsnorm_f16_grid_trusted(
                self.bufs[0].va(),
                self.ptrs[0].va(),
                rank0_residual_f16_va,
                rank0_weight_f16_va,
                rank0_output_f16_va,
                partial_va,
                self.gbar[0].va(),
                r as u32,
                self.n,
                eps,
                num_wg,
            )?;
        }
        self.devices[0].wait(OP_TIMEOUT)?;
        Ok(start.elapsed())
    }

    /// Benchmark and validate fused all-reduce + residual + RMSNorm + per-group
    /// OCP E4M3 quantization. Returns duration and max relative errors for f32
    /// output, residual writeback, scale, and dequantized FP8 output.
    pub fn benchmark_fused_direct_residual_rmsnorm_fp8_group(
        &mut self,
        iters: usize,
        warmup: usize,
        eps: f32,
        group_size: usize,
        packed_scales: bool,
        quant_only: bool,
    ) -> Result<(Duration, f64, f64, f64, f64)> {
        let r = self.ranks();
        let n = self.n as usize;
        if n == 0 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 requires n > 0"
            ));
        }
        let group_size = group_size.max(1);
        let groups = n.div_ceil(group_size);
        let packed_words = groups.div_ceil(4);
        let scale_bytes = if packed_scales {
            packed_words * 4
        } else {
            groups * 4
        };
        let num_wg = fused_rmsnorm_grid_wg(n);
        let inline_group_max = packed_scales
            && quant_only
            && r == 8
            && group_size == 64
            && n >= 8192
            && n <= (num_wg as usize * 256);
        let mut weight = self.devices[0].alloc_device(n * 4)?;
        let partial_floats = num_wg as usize + if inline_group_max { groups } else { 0 };
        let mut partial = self.devices[0].alloc_device(partial_floats * 4)?;
        let mut residuals = Vec::with_capacity(r);
        let mut quant = Vec::with_capacity(r);
        let mut scales = Vec::with_capacity(r);
        for rank in 0..r {
            residuals.push(self.devices[rank].alloc_device(n * 4)?);
            quant.push(self.devices[rank].alloc_device(n)?);
            scales.push(self.devices[rank].alloc_device(scale_bytes)?);
        }
        map_peer_buffers(&self.kfd, &self.devices, &residuals, "residual")?;
        map_peer_buffers(&self.kfd, &self.devices, &quant, "quant")?;
        map_peer_buffers(&self.kfd, &self.devices, &scales, "scale")?;
        let mut residual_ptrs = self.devices[0].alloc_device(r * 8)?;
        let mut quant_ptrs = self.devices[0].alloc_device(r * 8)?;
        let mut scale_ptrs = self.devices[0].alloc_device(r * 8)?;
        unsafe {
            let ptrs = residual_ptrs.as_mut_slice_of::<u64>();
            let qptrs = quant_ptrs.as_mut_slice_of::<u64>();
            let sptrs = scale_ptrs.as_mut_slice_of::<u64>();
            for rank in 0..r {
                ptrs[rank] = residuals[rank].va();
                qptrs[rank] = quant[rank].va();
                sptrs[rank] = scales[rank].va();
            }
            let w = weight.as_mut_slice_of::<f32>();
            for (i, v) in w.iter_mut().enumerate().take(n) {
                *v = 1.0 + (i % 7) as f32 * 1e-3;
            }
        }
        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(self, &mut residuals)?;
            reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
            if packed_scales {
                self.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
                    self.bufs[0].va(),
                    self.ptrs[0].va(),
                    residual_ptrs.va(),
                    weight.va(),
                    quant_ptrs.va(),
                    scale_ptrs.va(),
                    partial.va(),
                    self.gbar[0].va(),
                    r as u32,
                    self.n,
                    eps,
                    num_wg,
                    group_size as u32,
                    !quant_only,
                )?;
            } else {
                self.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_grid(
                    self.bufs[0].va(),
                    self.ptrs[0].va(),
                    residual_ptrs.va(),
                    weight.va(),
                    quant_ptrs.va(),
                    scale_ptrs.va(),
                    partial.va(),
                    self.gbar[0].va(),
                    r as u32,
                    self.n,
                    eps,
                    num_wg,
                    group_size as u32,
                )?;
            }
            self.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(self, &mut residuals)?;
            reset_fused_rmsnorm_sync(self, &mut partial, num_wg)?;
            let start = Instant::now();
            if packed_scales {
                self.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
                    self.bufs[0].va(),
                    self.ptrs[0].va(),
                    residual_ptrs.va(),
                    weight.va(),
                    quant_ptrs.va(),
                    scale_ptrs.va(),
                    partial.va(),
                    self.gbar[0].va(),
                    r as u32,
                    self.n,
                    eps,
                    num_wg,
                    group_size as u32,
                    !quant_only,
                )?;
            } else {
                self.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_grid(
                    self.bufs[0].va(),
                    self.ptrs[0].va(),
                    residual_ptrs.va(),
                    weight.va(),
                    quant_ptrs.va(),
                    scale_ptrs.va(),
                    partial.va(),
                    self.gbar[0].va(),
                    r as u32,
                    self.n,
                    eps,
                    num_wg,
                    group_size as u32,
                )?;
            }
            self.devices[0].wait(OP_TIMEOUT)?;
            total += start.elapsed();
        }

        let weights = unsafe { weight.as_mut_slice_of::<f32>() };
        let mut reduced = vec![0.0f64; n];
        for rank in 0..r {
            for (i, v) in reduced.iter_mut().enumerate() {
                *v += rank as f64 + i as f64 * 1e-3;
            }
        }
        let residual_expect: Vec<f64> = reduced
            .iter()
            .enumerate()
            .map(|(i, v)| *v + residual_seed(i))
            .collect();
        let ss: f64 = residual_expect.iter().map(|v| v * v).sum();
        let inv = 1.0f64 / (ss / n as f64 + eps as f64).sqrt();
        let output_expect: Vec<f64> = residual_expect
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * inv * *w as f64)
            .collect();
        let mut scale_expect = vec![1.0f32; groups];
        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(n);
            let maxabs = output_expect[start..end]
                .iter()
                .fold(0.0f64, |a, &v| a.max(v.abs()));
            scale_expect[group] = if maxabs > 0.0 {
                (maxabs as f32) / 448.0
            } else {
                1.0
            };
        }

        let mut max_out_rel = 0.0f64;
        if !quant_only {
            for rank in 0..r {
                let got = self.output(rank);
                for i in 0..n {
                    let e = output_expect[i];
                    let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                    max_out_rel = max_out_rel.max(rel);
                }
            }
        }

        let mut max_res_rel = 0.0f64;
        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }

        let mut max_scale_rel = 0.0f64;
        let mut max_deq_rel = 0.0f64;
        for rank in 0..r {
            let got_q = unsafe { quant[rank].as_mut_slice_of::<u8>() };
            if packed_scales {
                let got_s = unsafe { scales[rank].as_mut_slice_of::<u32>() };
                for group in 0..groups {
                    let word = got_s[group / 4];
                    let shift = ((group & 3) * 8) as u32;
                    let got_code = ((word >> shift) & 0xff) as u8;
                    let expect_code = f32_to_e8m0_ru(scale_expect[group]);
                    max_scale_rel =
                        max_scale_rel.max((got_code as i32 - expect_code as i32).abs() as f64);
                    let scale = e8m0_to_f32(got_code) as f64;
                    let start = group * group_size;
                    let end = (start + group_size).min(n);
                    for i in start..end {
                        let deq = e4m3_to_f32(got_q[i]) as f64 * scale;
                        let rel =
                            (deq - output_expect[i]).abs() / output_expect[i].abs().max(1e-12);
                        max_deq_rel = max_deq_rel.max(rel);
                    }
                }
            } else {
                let got_s = unsafe { scales[rank].as_mut_slice_of::<f32>() };
                for group in 0..groups {
                    let e = scale_expect[group] as f64;
                    let rel = ((got_s[group] as f64 - e).abs()) / e.abs().max(1e-30);
                    max_scale_rel = max_scale_rel.max(rel);
                    let start = group * group_size;
                    let end = (start + group_size).min(n);
                    for i in start..end {
                        let deq = e4m3_to_f32(got_q[i]) as f64 * got_s[group] as f64;
                        let rel =
                            (deq - output_expect[i]).abs() / output_expect[i].abs().max(1e-12);
                        max_deq_rel = max_deq_rel.max(rel);
                    }
                }
            }
        }
        if packed_scales && max_scale_rel > 0.0 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 packed scale mismatch: max_scale_code_delta {max_scale_rel:.0}"
            ));
        }
        if !packed_scales && (max_out_rel > 2e-5 || max_res_rel > 2e-5 || max_scale_rel > 2e-5) {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 mismatch: max_out_rel {max_out_rel:.3e}, max_res_rel {max_res_rel:.3e}, max_scale_rel {max_scale_rel:.3e}"
            ));
        }
        if packed_scales && !quant_only && (max_out_rel > 2e-5 || max_res_rel > 2e-5) {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 packed mismatch: max_out_rel {max_out_rel:.3e}, max_res_rel {max_res_rel:.3e}"
            ));
        }
        if packed_scales && quant_only && max_res_rel > 2e-5 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 quant-only mismatch: max_res_rel {max_res_rel:.3e}"
            ));
        }
        if max_deq_rel > 0.15 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 dequant mismatch: max_deq_rel {max_deq_rel:.3e} > 0.15"
            ));
        }

        Ok((
            total / iters.max(1) as u32,
            max_out_rel,
            max_res_rel,
            max_scale_rel,
            max_deq_rel,
        ))
    }
}

fn fused_rmsnorm_grid_wg(n: usize) -> u32 {
    std::env::var("MAINARCH_FUSED_RMS_WG")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(|| {
            if n <= 2048 {
                8
            } else if n <= 4096 {
                16
            } else if (4_608..=6_144).contains(&n) || n == 11_264 {
                24
            } else if n == 10_240 || n == 12_288 {
                48
            } else if n <= 16_384 {
                32
            } else if n <= 24_576 {
                48
            } else {
                64
            }
        })
        .min(64)
}

fn persistent_dda_wg(bytes: usize) -> u32 {
    if let Some(v) = std::env::var("MAINARCH_PERSIST_DDA_WG")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
    {
        return v;
    }
    match bytes {
        0..=16_384 => 1,
        16_385..=65_536 => 4,
        65_537..=262_144 => 8,
        262_145..=4_194_304 => 16,
        4_194_305..=67_108_864 => 32,
        _ => 64,
    }
}

fn persistent_dda_peer_wg(bytes: usize) -> u32 {
    if let Some(v) = std::env::var("MAINARCH_PERSIST_DDA_PEER_WG")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
    {
        return v;
    }
    if let Some(v) = std::env::var("MAINARCH_PERSIST_DDA_WG")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v > 0)
    {
        return v;
    }
    match bytes {
        // TP8 decode-payload tuning on MI355X showed the smallest payloads want
        // low resident pressure, while 64 KiB specifically is faster at 6 WGs
        // than 4/8/16 on the current peer-flag DDA kernel.
        0..=32_768 => 8,
        32_769..=65_536 => 6,
        65_537..=262_144 => 8,
        262_145..=524_288 => 32,
        524_289..=1_048_576 => 64,
        _ => 96,
    }
}

fn f32_to_e8m0_ru(x: f32) -> u8 {
    if !x.is_finite() || x <= 0.0 {
        return 127;
    }
    (x.log2().ceil() as i32 + 127).clamp(0, 255) as u8
}

fn e8m0_to_f32(x: u8) -> f32 {
    2.0f32.powi(x as i32 - 127)
}

fn reset_fused_rmsnorm_sync(
    ar: &mut AllReduce,
    partial: &mut DeviceBuffer,
    num_wg: u32,
) -> Result<()> {
    reset_fused_rmsnorm_sync_buffers(&mut ar.gbar[0], partial, num_wg)
}

fn reset_fused_rmsnorm_sync_buffers(
    gbar: &mut DeviceBuffer,
    partial: &mut DeviceBuffer,
    num_wg: u32,
) -> Result<()> {
    unsafe {
        for v in gbar.as_mut_slice_of::<u32>().iter_mut().take(2) {
            *v = 0;
        }
        for v in partial
            .as_mut_slice_of::<f32>()
            .iter_mut()
            .take(num_wg as usize)
        {
            *v = 0.0;
        }
    }
    Ok(())
}

fn map_peer_buffers(
    kfd: &Kfd,
    devices: &[GpuDevice],
    buffers: &[DeviceBuffer],
    label: &str,
) -> Result<()> {
    for (bi, buffer) in buffers.iter().enumerate() {
        for (di, dev) in devices.iter().enumerate() {
            if bi != di {
                kfd.map_buffer_to_peer(buffer, dev.node_id())
                    .with_context(|| {
                        format!("mapping {label} {bi} into peer node {}", dev.node_id())
                    })?;
            }
        }
    }
    Ok(())
}

fn residual_seed(i: usize) -> f64 {
    0.25 + i as f64 * 2e-4
}

fn reset_residual_rmsnorm_inputs(ar: &mut AllReduce, residuals: &mut [DeviceBuffer]) -> Result<()> {
    let r = ar.ranks();
    let n = ar.n as usize;
    for rank in 0..r {
        let f = unsafe { ar.bufs[rank].as_mut_slice_of::<f32>() };
        for (i, v) in f.iter_mut().enumerate().take(n) {
            *v = rank as f32 + i as f32 * 1e-3;
        }
    }
    for residual in residuals {
        let f = unsafe { residual.as_mut_slice_of::<f32>() };
        for (i, v) in f.iter_mut().enumerate().take(n) {
            *v = residual_seed(i) as f32;
        }
    }
    Ok(())
}

/// One row of an all-reduce bandwidth sweep, mirroring rccl-tests.
#[derive(Debug, Clone, Copy)]
pub struct BenchRow {
    pub bytes: usize,
    pub count: u32,
    pub ranks: usize,
    pub time: Duration,
    pub algbw_gbps: f64,
    pub busbw_gbps: f64,
}

#[derive(Debug, Clone)]
pub struct FusedResidualRmsnormProof {
    pub time: Duration,
    pub outputs: Vec<Vec<f32>>,
    pub residuals: Vec<Vec<f32>>,
}

enum Rank0ResidualCast {
    F16(u64),
    Bf16(u64),
}

/// One row for the packed FP8 quant-only TP handoff sweep.
#[derive(Debug, Clone, Copy)]
pub struct LowBitHandoffRow {
    pub bytes: usize,
    pub count: u32,
    pub time: Duration,
    pub f32eq_algbw_gbps: f64,
    pub f32eq_busbw_gbps: f64,
    pub wire_bytes: usize,
    pub wire_algbw_gbps: f64,
    pub wire_busbw_gbps: f64,
    pub max_res_rel: f64,
    pub max_scale_rel: f64,
    pub max_deq_rel: f64,
}

/// One row for the fused direct all-reduce + residual + RMSNorm baseline sweep.
#[derive(Debug, Clone, Copy)]
pub struct ResidualRmsnormRow {
    pub bytes: usize,
    pub count: u32,
    pub time: Duration,
    pub algbw_gbps: f64,
    pub busbw_gbps: f64,
    pub max_out_rel: f64,
    pub max_res_rel: f64,
}

/// One row for a matched f32 vs packed FP8 quant-only TP handoff comparison.
#[derive(Debug, Clone, Copy)]
pub struct ResidualRmsnormCompareRow {
    pub bytes: usize,
    pub count: u32,
    pub f32_time: Duration,
    pub fp8_time: Duration,
    pub int4_time: Duration,
    pub f32_busbw_gbps: f64,
    pub fp8_f32eq_busbw_gbps: f64,
    pub int4_f32eq_busbw_gbps: f64,
    pub wire_bytes: usize,
    pub wire_busbw_gbps: f64,
    pub max_out_rel: f64,
    pub max_res_rel: f64,
    pub max_scale_rel: f64,
    pub max_deq_rel: f64,
    pub int4_wire_bytes: usize,
    pub int4_deq_rel_l2: f64,
}

/// Measured INT4 handoff group-size policy for decode-sized TP payloads.
///
/// The TP8 first-pass unroll made the register-packed group-size-32 path faster
/// at both 32 KiB and 64 KiB decode gates while also matching the OCP/MX block
/// size. Keep this policy explicit so serving and benchmark callers do not
/// silently inherit a stale fixed group size.
pub fn int4_handoff_auto_group_size(bytes: usize) -> usize {
    let _ = bytes;
    32
}

pub const SERVING_TP8_DECODE_RANKS: usize = 8;
pub const SERVING_TP8_DECODE_DDA_MIN_BYTES: usize = 32 * 1024;
pub const SERVING_TP8_DECODE_DDA_MAX_BYTES: usize = 256 * 1024;
pub const SERVING_TP8_DECODE_CROSSOVER_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingAllReducePhase {
    Decode,
    Prefill,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServingAllReducePath {
    PeerFlagDdaFlatDecode,
    GenericRawXgmi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingAllReducePolicy {
    pub phase: ServingAllReducePhase,
    pub path: ServingAllReducePath,
    pub ranks: usize,
    pub min_bytes: usize,
    pub max_bytes: usize,
    pub dda_min_bytes: usize,
    pub dda_max_bytes: usize,
    pub crossover_bytes: usize,
}

impl ServingAllReducePolicy {
    pub fn uses_peer_flag_dda_decode(self) -> bool {
        self.path == ServingAllReducePath::PeerFlagDdaFlatDecode
    }

    pub fn selected_policy_label(self) -> Option<&'static str> {
        if self.uses_peer_flag_dda_decode() {
            Some(
                "serving decode window selected persistent peer-flag DDA-flat all-reduce (rank0 trigger, device peer flags)",
            )
        } else {
            None
        }
    }

    pub fn serving_note(self) -> String {
        format!(
            "serving policy selects peer-flag DDA for TP8 decode-sized payloads from {} through {} B; treat {} B as crossover and prefer RCCL/QuickReduce-style paths for larger unfused transfers",
            self.dda_min_bytes, self.dda_max_bytes, self.crossover_bytes
        )
    }

    pub fn backend_name(self) -> String {
        match self.path {
            ServingAllReducePath::PeerFlagDdaFlatDecode => {
                format!("raw-kfd-xgmi-peer-dda-decode({} ranks)", self.ranks)
            }
            ServingAllReducePath::GenericRawXgmi => format!("raw-kfd-xgmi({} ranks)", self.ranks),
        }
    }
}

pub fn serving_tp8_decode_dda_max_bytes() -> usize {
    std::env::var("MAINARCH_RCCL_TEST_DDA_MAX_BYTES")
        .or_else(|_| std::env::var("MAINARCH_GPU_XGMI_DDA_MAX_BYTES"))
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(SERVING_TP8_DECODE_DDA_MAX_BYTES)
}

pub fn select_serving_allreduce_policy(
    phase: ServingAllReducePhase,
    ranks: usize,
    min_bytes: usize,
    max_bytes: usize,
) -> ServingAllReducePolicy {
    let dda_max_bytes = serving_tp8_decode_dda_max_bytes();
    let use_peer_dda = phase == ServingAllReducePhase::Decode
        && ranks == SERVING_TP8_DECODE_RANKS
        && min_bytes >= SERVING_TP8_DECODE_DDA_MIN_BYTES
        && max_bytes <= dda_max_bytes
        && min_bytes <= max_bytes;
    ServingAllReducePolicy {
        phase,
        path: if use_peer_dda {
            ServingAllReducePath::PeerFlagDdaFlatDecode
        } else {
            ServingAllReducePath::GenericRawXgmi
        },
        ranks,
        min_bytes,
        max_bytes,
        dda_min_bytes: SERVING_TP8_DECODE_DDA_MIN_BYTES,
        dda_max_bytes,
        crossover_bytes: SERVING_TP8_DECODE_CROSSOVER_BYTES,
    }
}

#[derive(Debug, Clone)]
pub struct ServingAllReduceSweep {
    pub policy: ServingAllReducePolicy,
    pub backend_name: String,
    pub rows: Vec<BenchRow>,
}

#[derive(Debug, Clone)]
pub struct ServingAllReduceHotBench {
    pub policy: ServingAllReducePolicy,
    pub backend_name: String,
    pub row: BenchRow,
    pub setup_time: Duration,
    pub hot_wall_time: Duration,
}

/// Execute the serving all-reduce policy selected for a decode/prefill byte
/// window. Serving callers should use this instead of branching on individual
/// benchmark kernels so the CR/QuickReduce/RCCL-style routing seam stays in
/// core, next to the payload thresholds it depends on.
pub fn benchmark_serving_allreduce_sweep(
    phase: ServingAllReducePhase,
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<ServingAllReduceSweep> {
    let policy = select_serving_allreduce_policy(phase, nodes.len(), min_bytes, max_bytes);
    let rows = if policy.uses_peer_flag_dda_decode() {
        benchmark_persistent_dda_peer_flat_sweep(nodes, min_bytes, max_bytes, iters, warmup)?
    } else {
        benchmark_sweep(nodes, min_bytes, max_bytes, iters, warmup)?
    };
    let backend_name = policy.backend_name();
    Ok(ServingAllReduceSweep {
        policy,
        backend_name,
        rows,
    })
}

/// Construct the serving collective context once, then report hot per-token
/// collective timing separately from one-time process/VM/peer-map setup.
pub fn benchmark_serving_allreduce_hot_shared_kfd(
    kfd: Arc<Kfd>,
    phase: ServingAllReducePhase,
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<ServingAllReduceHotBench> {
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let policy = select_serving_allreduce_policy(phase, nodes.len(), bytes, bytes);
    let setup_started = Instant::now();
    let mut ar = AllReduce::new_shared(kfd, nodes, count)?;
    let setup_time = setup_started.elapsed();

    ar.set_n(count)?;
    let hot_started = Instant::now();
    let time = if policy.uses_peer_flag_dda_decode() {
        ar.benchmark_persistent_dda_peer_flat(iters, warmup)?
    } else {
        ar.benchmark(iters, warmup)?
    };
    let hot_wall_time = hot_started.elapsed();
    let r = nodes.len().max(1);
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    let row = BenchRow {
        bytes,
        count,
        ranks: r,
        time,
        algbw_gbps: algbw,
        busbw_gbps: busbw,
    };
    let backend_name = policy.backend_name();
    Ok(ServingAllReduceHotBench {
        policy,
        backend_name,
        row,
        setup_time,
        hot_wall_time,
    })
}

pub fn benchmark_serving_allreduce_sweep_shared_kfd(
    kfd: Arc<Kfd>,
    phase: ServingAllReducePhase,
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<ServingAllReduceSweep> {
    let policy = select_serving_allreduce_policy(phase, nodes.len(), min_bytes, max_bytes);
    let r = nodes.len().max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let mut ar = AllReduce::new_shared(kfd, nodes, max_count)?;

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let time = if policy.uses_peer_flag_dda_decode() {
            ar.benchmark_persistent_dda_peer_flat(iters, warmup)?
        } else {
            ar.benchmark(iters, warmup)?
        };
        let secs = time.as_secs_f64();
        let algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        });
        bytes *= 2;
    }
    let backend_name = policy.backend_name();
    Ok(ServingAllReduceSweep {
        policy,
        backend_name,
        rows,
    })
}

/// Run an all-reduce bandwidth sweep over a power-of-two byte ladder on the
/// given nodes, returning one row per size. busbw uses the rccl-tests
/// convention: `algbw * 2*(R-1)/R`.
///
/// The device group + buffers are allocated once at the maximum size and the
/// active element count is varied per measurement, so the VM/queues are set up
/// exactly once.
pub fn benchmark_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<Vec<BenchRow>> {
    let r = nodes.len().max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let mut ar = AllReduce::new(nodes, max_count)?;

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let time = ar.benchmark(iters, warmup)?;
        let secs = time.as_secs_f64();
        let algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        });
        bytes *= 2;
    }
    Ok(rows)
}

/// Benchmark the opt-in persistent direct all-reduce prototype for one size.
pub fn benchmark_persistent_direct(
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<BenchRow> {
    let r = nodes.len().max(1);
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let mut ar = AllReduce::new(nodes, count)?;
    let time = ar.benchmark_persistent_direct(iters, warmup)?;
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    Ok(BenchRow {
        bytes,
        count,
        ranks: r,
        time,
        algbw_gbps: algbw,
        busbw_gbps: busbw,
    })
}

/// Benchmark the opt-in persistent all-rank DDA-flat all-reduce prototype for one size.
pub fn benchmark_persistent_dda_flat(
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<BenchRow> {
    let r = nodes.len().max(1);
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let mut ar = AllReduce::new(nodes, count)?;
    let time = ar.benchmark_persistent_dda_flat(iters, warmup)?;
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    Ok(BenchRow {
        bytes,
        count,
        ranks: r,
        time,
        algbw_gbps: algbw,
        busbw_gbps: busbw,
    })
}

/// Benchmark the opt-in persistent peer-flag all-rank DDA-flat prototype for one size.
pub fn benchmark_persistent_dda_peer_flat(
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<BenchRow> {
    let r = nodes.len().max(1);
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let mut ar = AllReduce::new(nodes, count)?;
    let time = ar.benchmark_persistent_dda_peer_flat(iters, warmup)?;
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    Ok(BenchRow {
        bytes,
        count,
        ranks: r,
        time,
        algbw_gbps: algbw,
        busbw_gbps: busbw,
    })
}

/// Run a power-of-two sweep of the resident all-rank DDA-flat all-reduce prototype.
/// The device group and peer mappings are allocated once at the maximum size.
pub fn benchmark_persistent_dda_flat_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<Vec<BenchRow>> {
    let r = nodes.len().max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let mut ar = AllReduce::new(nodes, max_count)?;

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let time = ar.benchmark_persistent_dda_flat(iters, warmup)?;
        let secs = time.as_secs_f64();
        let algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        });
        bytes *= 2;
    }
    Ok(rows)
}

/// Run a power-of-two sweep of the resident peer-flag all-rank DDA-flat prototype.
/// The device group and peer mappings are allocated once at the maximum size.
pub fn benchmark_persistent_dda_peer_flat_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<Vec<BenchRow>> {
    let r = nodes.len().max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let mut ar = AllReduce::new(nodes, max_count)?;

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let time = ar.benchmark_persistent_dda_peer_flat(iters, warmup)?;
        let secs = time.as_secs_f64();
        let algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        });
        bytes *= 2;
    }
    Ok(rows)
}

/// Run a power-of-two sweep of the resident rank-0 direct all-reduce prototype.
/// The device group and peer mappings are allocated once at the maximum size;
/// persistent measurements reset the full allocation before each op so larger
/// active sizes cannot observe stale capacity-tail state.
pub fn benchmark_persistent_direct_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
) -> Result<Vec<BenchRow>> {
    let r = nodes.len().max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let mut ar = AllReduce::new(nodes, max_count)?;

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let time = ar.benchmark_persistent_direct(iters, warmup)?;
        let secs = time.as_secs_f64();
        let algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        });
        bytes *= 2;
    }
    Ok(rows)
}

/// Benchmark and validate the fused direct all-reduce + RMSNorm substrate.
pub fn benchmark_fused_direct_rmsnorm(
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
    eps: f32,
) -> Result<(BenchRow, f64)> {
    let r = nodes.len().max(1);
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let mut ar = AllReduce::new(nodes, count)?;
    let (time, max_rel) = ar.benchmark_fused_direct_rmsnorm(iters, warmup, eps)?;
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    Ok((
        BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        },
        max_rel,
    ))
}

/// Benchmark and validate the fused direct all-reduce + residual + RMSNorm substrate.
pub fn benchmark_fused_direct_residual_rmsnorm(
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
    eps: f32,
) -> Result<(BenchRow, f64, f64)> {
    let r = nodes.len().max(1);
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let mut ar = AllReduce::new(nodes, count)?;
    let (time, max_out_rel, max_res_rel) =
        ar.benchmark_fused_direct_residual_rmsnorm(iters, warmup, eps)?;
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    Ok((
        BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        },
        max_out_rel,
        max_res_rel,
    ))
}

/// Sweep the fused direct all-reduce + residual + RMSNorm baseline while
/// allocating the multi-GPU group once at maximum capacity.
pub fn benchmark_fused_direct_residual_rmsnorm_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
    eps: f32,
) -> Result<Vec<ResidualRmsnormRow>> {
    let r = nodes.len().max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let cap = max_count as usize;
    let mut ar = AllReduce::new(nodes, max_count)?;
    let num_wg_cap = fused_rmsnorm_grid_wg(cap);
    let mut weight = ar.devices[0].alloc_device(cap * 4)?;
    let mut partial = ar.devices[0].alloc_device(num_wg_cap as usize * 4)?;
    let mut residuals = Vec::with_capacity(r);
    for rank in 0..r {
        residuals.push(ar.devices[rank].alloc_device(cap * 4)?);
    }
    map_peer_buffers(&ar.kfd, &ar.devices, &residuals, "residual")?;
    let mut residual_ptrs = ar.devices[0].alloc_device(r * 8)?;
    unsafe {
        let ptrs = residual_ptrs.as_mut_slice_of::<u64>();
        for rank in 0..r {
            ptrs[rank] = residuals[rank].va();
        }
        let w = weight.as_mut_slice_of::<f32>();
        for (i, v) in w.iter_mut().enumerate().take(cap) {
            *v = 1.0 + (i % 7) as f32 * 1e-3;
        }
    }

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let n = count as usize;
        let num_wg = fused_rmsnorm_grid_wg(n);
        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            let start = Instant::now();
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
            total += start.elapsed();
        }
        let time = total / iters.max(1) as u32;

        let weights = unsafe { weight.as_mut_slice_of::<f32>() };
        let mut reduced = vec![0.0f64; n];
        for rank in 0..r {
            for (i, v) in reduced.iter_mut().enumerate() {
                *v += rank as f64 + i as f64 * 1e-3;
            }
        }
        let residual_expect: Vec<f64> = reduced
            .iter()
            .enumerate()
            .map(|(i, v)| *v + residual_seed(i))
            .collect();
        let ss: f64 = residual_expect.iter().map(|v| v * v).sum();
        let inv = 1.0f64 / (ss / n as f64 + eps as f64).sqrt();
        let output_expect: Vec<f64> = residual_expect
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * inv * *w as f64)
            .collect();
        let mut max_out_rel = 0.0f64;
        for rank in 0..r {
            let got = ar.output(rank);
            for i in 0..n {
                let e = output_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_out_rel = max_out_rel.max(rel);
            }
        }
        let mut max_res_rel = 0.0f64;
        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }
        if max_out_rel > 2e-5 || max_res_rel > 2e-5 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm sweep mismatch at {bytes}B: max_out_rel {max_out_rel:.3e}, max_res_rel {max_res_rel:.3e}"
            ));
        }
        let secs = time.as_secs_f64();
        let algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(ResidualRmsnormRow {
            bytes,
            count,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
            max_out_rel,
            max_res_rel,
        });
        match bytes.checked_mul(2) {
            Some(next) if next <= max_bytes => bytes = next,
            _ => break,
        }
    }
    Ok(rows)
}

/// Benchmark and validate fused all-reduce + residual + RMSNorm + FP8 group quant.
pub fn benchmark_fused_direct_residual_rmsnorm_fp8_group(
    nodes: &[u32],
    bytes: usize,
    iters: usize,
    warmup: usize,
    eps: f32,
    group_size: usize,
    packed_scales: bool,
    quant_only: bool,
) -> Result<(BenchRow, f64, f64, f64, f64)> {
    let r = nodes.len().max(1);
    let bytes = bytes.max(4);
    let count = (bytes / 4) as u32;
    let mut ar = AllReduce::new(nodes, count)?;
    let (time, max_out_rel, max_res_rel, max_scale_rel, max_deq_rel) = ar
        .benchmark_fused_direct_residual_rmsnorm_fp8_group(
            iters,
            warmup,
            eps,
            group_size,
            packed_scales,
            quant_only,
        )?;
    let secs = time.as_secs_f64();
    let algbw = if secs > 0.0 {
        bytes as f64 / secs / 1e9
    } else {
        0.0
    };
    let busbw = algbw * 2.0 * (r as f64 - 1.0) / r as f64;
    Ok((
        BenchRow {
            bytes,
            count,
            ranks: r,
            time,
            algbw_gbps: algbw,
            busbw_gbps: busbw,
        },
        max_out_rel,
        max_res_rel,
        max_scale_rel,
        max_deq_rel,
    ))
}

/// Sweep the packed FP8 quant-only TP handoff while allocating the multi-GPU
/// group and per-rank quant/residual workspaces once at maximum capacity.
pub fn benchmark_fused_direct_residual_rmsnorm_fp8_quant_only_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
    eps: f32,
    group_size: usize,
) -> Result<Vec<LowBitHandoffRow>> {
    let r = nodes.len().max(1);
    let group_size = group_size.max(1);
    let max_count = (max_bytes.max(4) / 4) as u32;
    let cap = max_count as usize;
    let mut ar = AllReduce::new(nodes, max_count)?;
    let groups_cap = cap.div_ceil(group_size);
    let scale_bytes_cap = groups_cap.div_ceil(4) * 4;
    let num_wg_cap = fused_rmsnorm_grid_wg(cap);
    let partial_floats = num_wg_cap as usize + groups_cap;

    let mut weight = ar.devices[0].alloc_device(cap * 4)?;
    let mut partial = ar.devices[0].alloc_device(partial_floats * 4)?;
    let mut residuals = Vec::with_capacity(r);
    let mut quant = Vec::with_capacity(r);
    let mut scales = Vec::with_capacity(r);
    for rank in 0..r {
        residuals.push(ar.devices[rank].alloc_device(cap * 4)?);
        quant.push(ar.devices[rank].alloc_device(cap)?);
        scales.push(ar.devices[rank].alloc_device(scale_bytes_cap)?);
    }
    map_peer_buffers(&ar.kfd, &ar.devices, &residuals, "residual")?;
    map_peer_buffers(&ar.kfd, &ar.devices, &quant, "quant")?;
    map_peer_buffers(&ar.kfd, &ar.devices, &scales, "scale")?;

    let mut residual_ptrs = ar.devices[0].alloc_device(r * 8)?;
    let mut quant_ptrs = ar.devices[0].alloc_device(r * 8)?;
    let mut scale_ptrs = ar.devices[0].alloc_device(r * 8)?;
    unsafe {
        let ptrs = residual_ptrs.as_mut_slice_of::<u64>();
        let qptrs = quant_ptrs.as_mut_slice_of::<u64>();
        let sptrs = scale_ptrs.as_mut_slice_of::<u64>();
        for rank in 0..r {
            ptrs[rank] = residuals[rank].va();
            qptrs[rank] = quant[rank].va();
            sptrs[rank] = scales[rank].va();
        }
        let w = weight.as_mut_slice_of::<f32>();
        for (i, v) in w.iter_mut().enumerate().take(cap) {
            *v = 1.0 + (i % 7) as f32 * 1e-3;
        }
    }

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let n = count as usize;
        let groups = n.div_ceil(group_size);
        let scale_bytes = groups.div_ceil(4) * 4;
        let num_wg = fused_rmsnorm_grid_wg(n);

        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                quant_ptrs.va(),
                scale_ptrs.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
                group_size as u32,
                false,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
        }

        let mut total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            let start = Instant::now();
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                quant_ptrs.va(),
                scale_ptrs.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
                group_size as u32,
                false,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
            total += start.elapsed();
        }
        let time = total / iters.max(1) as u32;

        let weights = unsafe { weight.as_mut_slice_of::<f32>() };
        let mut reduced = vec![0.0f64; n];
        for rank in 0..r {
            for (i, v) in reduced.iter_mut().enumerate() {
                *v += rank as f64 + i as f64 * 1e-3;
            }
        }
        let residual_expect: Vec<f64> = reduced
            .iter()
            .enumerate()
            .map(|(i, v)| *v + residual_seed(i))
            .collect();
        let ss: f64 = residual_expect.iter().map(|v| v * v).sum();
        let inv = 1.0f64 / (ss / n as f64 + eps as f64).sqrt();
        let output_expect: Vec<f64> = residual_expect
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * inv * *w as f64)
            .collect();
        let mut scale_expect = vec![1.0f32; groups];
        for (group, slot) in scale_expect.iter_mut().enumerate() {
            let start = group * group_size;
            let end = (start + group_size).min(n);
            let maxabs = output_expect[start..end]
                .iter()
                .fold(0.0f64, |a, &v| a.max(v.abs()));
            *slot = if maxabs > 0.0 {
                (maxabs as f32) / 448.0
            } else {
                1.0
            };
        }

        let mut max_res_rel = 0.0f64;
        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }

        let mut max_scale_rel = 0.0f64;
        let mut max_deq_rel = 0.0f64;
        for rank in 0..r {
            let got_q = unsafe { quant[rank].as_mut_slice_of::<u8>() };
            let got_s = unsafe { scales[rank].as_mut_slice_of::<u32>() };
            for group in 0..groups {
                let word = got_s[group / 4];
                let shift = ((group & 3) * 8) as u32;
                let got_code = ((word >> shift) & 0xff) as u8;
                let expect_code = f32_to_e8m0_ru(scale_expect[group]);
                max_scale_rel =
                    max_scale_rel.max((got_code as i32 - expect_code as i32).abs() as f64);
                let scale = e8m0_to_f32(got_code) as f64;
                let start = group * group_size;
                let end = (start + group_size).min(n);
                for i in start..end {
                    let deq = e4m3_to_f32(got_q[i]) as f64 * scale;
                    let rel = (deq - output_expect[i]).abs() / output_expect[i].abs().max(1e-12);
                    max_deq_rel = max_deq_rel.max(rel);
                }
            }
        }
        if max_res_rel > 2e-5 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 quant-only sweep residual mismatch at {bytes}B: max_res_rel {max_res_rel:.3e}"
            ));
        }
        if max_scale_rel > 0.0 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 quant-only sweep packed scale mismatch at {bytes}B: max_scale_code_delta {max_scale_rel:.0}"
            ));
        }
        if max_deq_rel > 0.15 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm+fp8 quant-only sweep dequant mismatch at {bytes}B: max_deq_rel {max_deq_rel:.3e} > 0.15"
            ));
        }

        let secs = time.as_secs_f64();
        let f32eq_algbw = if secs > 0.0 {
            bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let f32eq_busbw = f32eq_algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        let wire_bytes = n + scale_bytes;
        let wire_algbw = if secs > 0.0 {
            wire_bytes as f64 / secs / 1e9
        } else {
            0.0
        };
        let wire_busbw = wire_algbw * 2.0 * (r as f64 - 1.0) / r as f64;
        rows.push(LowBitHandoffRow {
            bytes,
            count,
            time,
            f32eq_algbw_gbps: f32eq_algbw,
            f32eq_busbw_gbps: f32eq_busbw,
            wire_bytes,
            wire_algbw_gbps: wire_algbw,
            wire_busbw_gbps: wire_busbw,
            max_res_rel,
            max_scale_rel,
            max_deq_rel,
        });

        match bytes.checked_mul(2) {
            Some(next) if next <= max_bytes => bytes = next,
            _ => break,
        }
    }
    Ok(rows)
}

/// Compare fused f32 residual+RMSNorm against packed FP8 quant-only handoff
/// with one shared multi-GPU VM/workspace setup.
pub fn benchmark_fused_direct_residual_rmsnorm_compare_sweep(
    nodes: &[u32],
    min_bytes: usize,
    max_bytes: usize,
    iters: usize,
    warmup: usize,
    eps: f32,
    group_size: usize,
) -> Result<Vec<ResidualRmsnormCompareRow>> {
    let r = nodes.len().max(1);
    let auto_group = group_size == 0;
    let alloc_group_size = if auto_group { 32 } else { group_size.max(1) };
    let max_count = (max_bytes.max(4) / 4) as u32;
    let cap = max_count as usize;
    let groups_cap = cap.div_ceil(alloc_group_size);
    let scale_bytes_cap = groups_cap.div_ceil(4) * 4;
    let int4_bytes_cap = cap.div_ceil(2);
    let num_wg_cap = fused_rmsnorm_grid_wg(cap);
    let partial_floats = num_wg_cap as usize + groups_cap;
    let mut ar = AllReduce::new(nodes, max_count)?;

    let mut weight = ar.devices[0].alloc_device(cap * 4)?;
    let mut partial = ar.devices[0].alloc_device(partial_floats * 4)?;
    let mut residuals = Vec::with_capacity(r);
    let mut quant = Vec::with_capacity(r);
    let mut scales = Vec::with_capacity(r);
    let mut int4_quant = Vec::with_capacity(r);
    let mut int4_scales = Vec::with_capacity(r);
    for rank in 0..r {
        residuals.push(ar.devices[rank].alloc_device(cap * 4)?);
        quant.push(ar.devices[rank].alloc_device(cap)?);
        scales.push(ar.devices[rank].alloc_device(scale_bytes_cap)?);
        int4_quant.push(ar.devices[rank].alloc_device(int4_bytes_cap)?);
        int4_scales.push(ar.devices[rank].alloc_device(scale_bytes_cap)?);
    }
    map_peer_buffers(&ar.kfd, &ar.devices, &residuals, "residual")?;
    map_peer_buffers(&ar.kfd, &ar.devices, &quant, "quant")?;
    map_peer_buffers(&ar.kfd, &ar.devices, &scales, "scale")?;
    map_peer_buffers(&ar.kfd, &ar.devices, &int4_quant, "int4_quant")?;
    map_peer_buffers(&ar.kfd, &ar.devices, &int4_scales, "int4_scale")?;

    let mut residual_ptrs = ar.devices[0].alloc_device(r * 8)?;
    let mut quant_ptrs = ar.devices[0].alloc_device(r * 8)?;
    let mut scale_ptrs = ar.devices[0].alloc_device(r * 8)?;
    let mut int4_quant_ptrs = ar.devices[0].alloc_device(r * 8)?;
    let mut int4_scale_ptrs = ar.devices[0].alloc_device(r * 8)?;
    unsafe {
        let ptrs = residual_ptrs.as_mut_slice_of::<u64>();
        let qptrs = quant_ptrs.as_mut_slice_of::<u64>();
        let sptrs = scale_ptrs.as_mut_slice_of::<u64>();
        let i4qptrs = int4_quant_ptrs.as_mut_slice_of::<u64>();
        let i4sptrs = int4_scale_ptrs.as_mut_slice_of::<u64>();
        for rank in 0..r {
            ptrs[rank] = residuals[rank].va();
            qptrs[rank] = quant[rank].va();
            sptrs[rank] = scales[rank].va();
            i4qptrs[rank] = int4_quant[rank].va();
            i4sptrs[rank] = int4_scales[rank].va();
        }
        let w = weight.as_mut_slice_of::<f32>();
        for (i, v) in w.iter_mut().enumerate().take(cap) {
            *v = 1.0 + (i % 7) as f32 * 1e-3;
        }
    }

    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(4);
    while bytes <= max_bytes {
        let count = (bytes / 4) as u32;
        ar.set_n(count)?;
        let n = count as usize;
        let row_group_size = if auto_group {
            int4_handoff_auto_group_size(bytes)
        } else {
            alloc_group_size
        };
        let groups = n.div_ceil(row_group_size);
        let scale_bytes = groups.div_ceil(4) * 4;
        let num_wg = fused_rmsnorm_grid_wg(n);

        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut f32_total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            let start = Instant::now();
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
            f32_total += start.elapsed();
        }
        let f32_time = f32_total / iters.max(1) as u32;

        let weights = unsafe { weight.as_mut_slice_of::<f32>() };
        let mut reduced = vec![0.0f64; n];
        for rank in 0..r {
            for (i, v) in reduced.iter_mut().enumerate() {
                *v += rank as f64 + i as f64 * 1e-3;
            }
        }
        let residual_expect: Vec<f64> = reduced
            .iter()
            .enumerate()
            .map(|(i, v)| *v + residual_seed(i))
            .collect();
        let ss: f64 = residual_expect.iter().map(|v| v * v).sum();
        let inv = 1.0f64 / (ss / n as f64 + eps as f64).sqrt();
        let output_expect: Vec<f64> = residual_expect
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * inv * *w as f64)
            .collect();

        let mut max_out_rel = 0.0f64;
        for rank in 0..r {
            let got = ar.output(rank);
            for i in 0..n {
                let e = output_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_out_rel = max_out_rel.max(rel);
            }
        }
        let mut max_res_rel = 0.0f64;
        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }

        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                quant_ptrs.va(),
                scale_ptrs.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
                row_group_size as u32,
                false,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut fp8_total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            let start = Instant::now();
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_fp8_group_packed_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                quant_ptrs.va(),
                scale_ptrs.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
                row_group_size as u32,
                false,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
            fp8_total += start.elapsed();
        }
        let fp8_time = fp8_total / iters.max(1) as u32;

        let mut scale_expect = vec![1.0f32; groups];
        for (group, slot) in scale_expect.iter_mut().enumerate() {
            let start = group * row_group_size;
            let end = (start + row_group_size).min(n);
            let maxabs = output_expect[start..end]
                .iter()
                .fold(0.0f64, |a, &v| a.max(v.abs()));
            *slot = if maxabs > 0.0 {
                (maxabs as f32) / 448.0
            } else {
                1.0
            };
        }
        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }
        let mut max_scale_rel = 0.0f64;
        let mut max_deq_rel = 0.0f64;
        for rank in 0..r {
            let got_q = unsafe { quant[rank].as_mut_slice_of::<u8>() };
            let got_s = unsafe { scales[rank].as_mut_slice_of::<u32>() };
            for group in 0..groups {
                let word = got_s[group / 4];
                let shift = ((group & 3) * 8) as u32;
                let got_code = ((word >> shift) & 0xff) as u8;
                let expect_code = f32_to_e8m0_ru(scale_expect[group]);
                max_scale_rel =
                    max_scale_rel.max((got_code as i32 - expect_code as i32).abs() as f64);
                let scale = e8m0_to_f32(got_code) as f64;
                let start = group * row_group_size;
                let end = (start + row_group_size).min(n);
                for i in start..end {
                    let deq = e4m3_to_f32(got_q[i]) as f64 * scale;
                    let rel = (deq - output_expect[i]).abs() / output_expect[i].abs().max(1e-12);
                    max_deq_rel = max_deq_rel.max(rel);
                }
            }
        }
        for _ in 0..warmup {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_int4_group_packed_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                int4_quant_ptrs.va(),
                int4_scale_ptrs.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
                row_group_size as u32,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
        }
        let mut int4_total = Duration::ZERO;
        for _ in 0..iters.max(1) {
            reset_residual_rmsnorm_inputs(&mut ar, &mut residuals)?;
            reset_fused_rmsnorm_sync(&mut ar, &mut partial, num_wg)?;
            let start = Instant::now();
            ar.devices[0].arm_allreduce_direct_residual_rmsnorm_int4_group_packed_grid(
                ar.bufs[0].va(),
                ar.ptrs[0].va(),
                residual_ptrs.va(),
                weight.va(),
                int4_quant_ptrs.va(),
                int4_scale_ptrs.va(),
                partial.va(),
                ar.gbar[0].va(),
                r as u32,
                ar.n,
                eps,
                num_wg,
                row_group_size as u32,
            )?;
            ar.devices[0].wait(OP_TIMEOUT)?;
            int4_total += start.elapsed();
        }
        let int4_time = int4_total / iters.max(1) as u32;

        for residual in &mut residuals {
            let got = unsafe { residual.as_mut_slice_of::<f32>() };
            for i in 0..n {
                let e = residual_expect[i];
                let rel = ((got[i] as f64 - e).abs()) / e.abs().max(1e-12);
                max_res_rel = max_res_rel.max(rel);
            }
        }
        let mut int4_scale_expect = vec![1.0f32; groups];
        for (group, slot) in int4_scale_expect.iter_mut().enumerate() {
            let start = group * row_group_size;
            let end = (start + row_group_size).min(n);
            let maxabs = output_expect[start..end]
                .iter()
                .fold(0.0f64, |a, &v| a.max(v.abs()));
            *slot = if maxabs > 0.0 {
                (maxabs as f32) / 7.0
            } else {
                1.0
            };
        }
        let mut int4_err2 = 0.0f64;
        let mut int4_ref2 = 0.0f64;
        for rank in 0..r {
            let got_q = unsafe { int4_quant[rank].as_mut_slice_of::<u8>() };
            let got_s = unsafe { int4_scales[rank].as_mut_slice_of::<u32>() };
            for group in 0..groups {
                let word = got_s[group / 4];
                let shift = ((group & 3) * 8) as u32;
                let got_code = ((word >> shift) & 0xff) as u8;
                let expect_code = f32_to_e8m0_ru(int4_scale_expect[group]);
                max_scale_rel =
                    max_scale_rel.max((got_code as i32 - expect_code as i32).abs() as f64);
                let scale = e8m0_to_f32(got_code) as f64;
                let start = group * row_group_size;
                let end = (start + row_group_size).min(n);
                for i in start..end {
                    let byte = got_q[i >> 1];
                    let code = if (i & 1) == 0 {
                        byte & 0x0f
                    } else {
                        (byte >> 4) & 0x0f
                    };
                    let signed = if code >= 8 {
                        code as i32 - 16
                    } else {
                        code as i32
                    };
                    let deq = signed as f64 * scale;
                    let want = output_expect[i];
                    let err = deq - want;
                    int4_err2 += err * err;
                    int4_ref2 += want * want;
                }
            }
        }
        let int4_deq_rel_l2 = (int4_err2 / int4_ref2.max(1e-30)).sqrt();
        if max_out_rel > 2e-5 || max_res_rel > 2e-5 || max_scale_rel > 0.0 || max_deq_rel > 0.15 {
            return Err(anyhow!(
                "fused allreduce+residual+rmsnorm compare mismatch at {bytes}B: max_out_rel {max_out_rel:.3e}, max_res_rel {max_res_rel:.3e}, max_scale_delta {max_scale_rel:.0}, max_deq_rel {max_deq_rel:.3e}"
            ));
        }

        let bus_factor = 2.0 * (r as f64 - 1.0) / r as f64;
        let f32_secs = f32_time.as_secs_f64();
        let fp8_secs = fp8_time.as_secs_f64();
        let int4_secs = int4_time.as_secs_f64();
        let f32_busbw = if f32_secs > 0.0 {
            bytes as f64 / f32_secs / 1e9 * bus_factor
        } else {
            0.0
        };
        let fp8_f32eq_busbw = if fp8_secs > 0.0 {
            bytes as f64 / fp8_secs / 1e9 * bus_factor
        } else {
            0.0
        };
        let int4_f32eq_busbw = if int4_secs > 0.0 {
            bytes as f64 / int4_secs / 1e9 * bus_factor
        } else {
            0.0
        };
        let wire_bytes = n + scale_bytes;
        let int4_wire_bytes = n.div_ceil(2) + scale_bytes;
        let wire_busbw = if fp8_secs > 0.0 {
            wire_bytes as f64 / fp8_secs / 1e9 * bus_factor
        } else {
            0.0
        };
        rows.push(ResidualRmsnormCompareRow {
            bytes,
            count,
            f32_time,
            fp8_time,
            int4_time,
            f32_busbw_gbps: f32_busbw,
            fp8_f32eq_busbw_gbps: fp8_f32eq_busbw,
            int4_f32eq_busbw_gbps: int4_f32eq_busbw,
            wire_bytes,
            wire_busbw_gbps: wire_busbw,
            max_out_rel,
            max_res_rel,
            max_scale_rel,
            max_deq_rel,
            int4_wire_bytes,
            int4_deq_rel_l2,
        });

        match bytes.checked_mul(2) {
            Some(next) if next <= max_bytes => bytes = next,
            _ => break,
        }
    }
    Ok(rows)
}

/// Correctness check: R ranks of deterministic data, GPU all-reduce, compared
/// bit-exact to a CPU reference summing in tree order.
pub fn check(nodes: &[u32], n: u32) -> Result<Duration> {
    let r = nodes.len();
    let mut ar = AllReduce::new(nodes, n)?;

    // Rank r, element i -> r as f32 + i*1e-3.
    let inputs: Vec<Vec<f32>> = (0..r)
        .map(|rank| {
            (0..n as usize)
                .map(|i| rank as f32 + i as f32 * 1e-3)
                .collect()
        })
        .collect();
    for (rank, v) in inputs.iter().enumerate() {
        ar.set_input(rank, v)?;
    }

    let elapsed = ar.all_reduce_sum()?;

    // CPU reference in the same sequential order (p = 0..R) the direct
    // reduce_peers kernel accumulates, so results match bit-for-bit.
    let expect = sequential_sum_reference(&inputs, n);

    for rank in 0..r {
        let got = ar.output(rank);
        for i in 0..n as usize {
            if got[i].to_bits() != expect[i].to_bits() {
                return Err(anyhow!(
                    "all-reduce mismatch on rank {rank}[{i}]: gpu={} cpu={}",
                    got[i],
                    expect[i]
                ));
            }
        }
    }
    Ok(elapsed)
}

/// Reproduce the GPU's sequential (p = 0..R) reduction order on the CPU.
/// Round up to the next multiple of 4 (keeps chunk offsets 16-byte aligned).
fn round_up4(x: u32) -> u32 {
    (x + 3) & !3
}

fn sequential_sum_reference(inputs: &[Vec<f32>], n: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; n as usize];
    for v in inputs {
        for (o, x) in out.iter_mut().zip(v.iter()).take(n as usize) {
            *o += *x;
        }
    }
    out
}

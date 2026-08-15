#![allow(clippy::needless_range_loop)]
// Throwaway diagnostic: isolate XGMI write-bandwidth regimes to find the
// large-message all-reduce ceiling.
//
//   A single  : 1 GPU writes to 1 peer (one link)
//   B fanout7  : 1 GPU writes to all 7 peers (7 outbound links, 1 sender)
//   C ring     : all 8 GPUs each write to their +1 neighbor (8 links, each GPU
//                receives from exactly 1 — no receiver contention)
//   D all2all  : all 8 GPUs each write to all 7 peers (full all-to-all)
//
// Reports GB/s per sending GPU and the implied per-link GB/s.
use std::sync::Arc;
use std::time::{Duration, Instant};

use mainarch_core::{DeviceBuffer, GpuDevice, Kfd};

fn build_ptrs(dev: &GpuDevice, targets: &[u64]) -> anyhow::Result<DeviceBuffer> {
    let mut t = dev.alloc(targets.len().max(1) * 8)?;
    let slots = unsafe { t.as_mut_slice_of::<u64>() };
    for (i, v) in targets.iter().enumerate() {
        slots[i] = *v;
    }
    Ok(t)
}

fn main() -> anyhow::Result<()> {
    let nodes: Vec<u32> = mainarch_core::enumerate_gpus()?
        .iter()
        .map(|g| g.node_id)
        .collect();
    let r = nodes.len();
    let kfd = Arc::new(Kfd::open()?);
    let devs: Vec<GpuDevice> = nodes
        .iter()
        .map(|&n| GpuDevice::open_shared(kfd.clone(), n))
        .collect::<anyhow::Result<_>>()?;

    let mb: u32 = std::env::var("MAINARCH_PROBE_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let n: u32 = mb * 1024 * 1024 / 4; // mb MiB per buffer
    let bytes = n as usize * 4;
    let bufs: Vec<DeviceBuffer> = devs
        .iter()
        .map(|d| kfd.alloc_vram(d.node_id(), bytes))
        .collect::<anyhow::Result<_>>()?;
    for b in &bufs {
        for d in &devs {
            let _ = kfd.map_buffer_to_peer(b, d.node_id());
        }
    }
    let va: Vec<u64> = bufs.iter().map(|b| b.va()).collect();
    let to = Duration::from_secs(20);
    let iters = 30u32;
    let mut devs = devs;
    let gbps = |out_bytes: f64, secs: f64| out_bytes / secs / 1e9;

    // A: single link 0 -> 1
    {
        let ptrs = build_ptrs(&devs[0], &[va[1]])?;
        let pv = ptrs.va();
        for _ in 0..3 {
            devs[0].arm_broadcast_chunk(va[0], pv, 1, 0, n)?;
            devs[0].wait(to)?;
        }
        let s = Instant::now();
        for _ in 0..iters {
            devs[0].arm_broadcast_chunk(va[0], pv, 1, 0, n)?;
            devs[0].wait(to)?;
        }
        let t = s.elapsed().as_secs_f64() / iters as f64;
        println!(
            "A single   : {:7.1} GB/s (1 link)  [{:.1} us]",
            gbps(bytes as f64, t),
            t * 1e6
        );
    }

    // B: fanout 0 -> all peers
    {
        let targets: Vec<u64> = (1..r).map(|i| va[i]).collect();
        let ptrs = build_ptrs(&devs[0], &targets)?;
        let pv = ptrs.va();
        let parts = targets.len() as u32;
        for _ in 0..3 {
            devs[0].arm_broadcast_chunk(va[0], pv, parts, 0, n)?;
            devs[0].wait(to)?;
        }
        let s = Instant::now();
        for _ in 0..iters {
            devs[0].arm_broadcast_chunk(va[0], pv, parts, 0, n)?;
            devs[0].wait(to)?;
        }
        let t = s.elapsed().as_secs_f64() / iters as f64;
        let out = bytes as f64 * parts as f64;
        println!(
            "B fanout{}  : {:7.1} GB/s/gpu out ({:.1}/link) [{:.1} us]",
            parts,
            gbps(out, t),
            gbps(bytes as f64, t),
            t * 1e6
        );
    }

    // C: ring — each GPU writes to +1 neighbor concurrently
    {
        let ptrs: Vec<DeviceBuffer> = (0..r)
            .map(|g| build_ptrs(&devs[g], &[va[(g + 1) % r]]))
            .collect::<anyhow::Result<_>>()?;
        let pv: Vec<u64> = ptrs.iter().map(|p| p.va()).collect();
        for _ in 0..3 {
            for g in 0..r {
                devs[g].arm_broadcast_chunk(va[g], pv[g], 1, 0, n)?;
            }
            for g in 0..r {
                devs[g].wait(to)?;
            }
        }
        let s = Instant::now();
        for _ in 0..iters {
            for g in 0..r {
                devs[g].arm_broadcast_chunk(va[g], pv[g], 1, 0, n)?;
            }
            for g in 0..r {
                devs[g].wait(to)?;
            }
        }
        let t = s.elapsed().as_secs_f64() / iters as f64;
        println!(
            "C ring     : {:7.1} GB/s/gpu (1 in/1 out each) [{:.1} us]",
            gbps(bytes as f64, t),
            t * 1e6
        );
    }

    // D: all-to-all — each GPU writes to all peers concurrently
    {
        let ptrs: Vec<DeviceBuffer> = (0..r)
            .map(|g| {
                let targets: Vec<u64> = (0..r).filter(|&j| j != g).map(|j| va[j]).collect();
                build_ptrs(&devs[g], &targets)
            })
            .collect::<anyhow::Result<_>>()?;
        let pv: Vec<u64> = ptrs.iter().map(|p| p.va()).collect();
        let parts = (r - 1) as u32;
        for _ in 0..3 {
            for g in 0..r {
                devs[g].arm_broadcast_chunk(va[g], pv[g], parts, 0, n)?;
            }
            for g in 0..r {
                devs[g].wait(to)?;
            }
        }
        let s = Instant::now();
        for _ in 0..iters {
            for g in 0..r {
                devs[g].arm_broadcast_chunk(va[g], pv[g], parts, 0, n)?;
            }
            for g in 0..r {
                devs[g].wait(to)?;
            }
        }
        let t = s.elapsed().as_secs_f64() / iters as f64;
        let out = bytes as f64 * parts as f64;
        println!(
            "D all2all  : {:7.1} GB/s/gpu out ({:.1}/link) [{:.1} us]",
            gbps(out, t),
            gbps(bytes as f64, t),
            t * 1e6
        );
    }

    Ok(())
}

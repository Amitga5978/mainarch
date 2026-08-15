#![allow(clippy::needless_range_loop)]
// Validate the in-kernel cross-GPU barrier: all GPUs signal each other's
// arrival slots and spin-wait via system-scope atomics over XGMI. If the
// memory model carries the flag writes across GPUs, every GPU writes `seq` to
// its out[0]; a 0xDEAD means the spin cap hit (coherence failure / deadlock).
use std::sync::Arc;
use std::time::Duration;

use mainarch_core::{DeviceBuffer, GpuDevice, Kfd};

fn main() -> anyhow::Result<()> {
    let nodes: Vec<u32> = mainarch_core::enumerate_gpus()?
        .iter()
        .map(|g| g.node_id)
        .collect();
    let r = nodes.len();
    let kfd = Arc::new(Kfd::open()?);
    let mut devs: Vec<GpuDevice> = nodes
        .iter()
        .map(|&n| GpuDevice::open_shared(kfd.clone(), n))
        .collect::<anyhow::Result<_>>()?;

    // Per-GPU arrival flags (R u32) in VRAM, cross-mapped so peers can write.
    let mut flags: Vec<DeviceBuffer> = devs
        .iter()
        .map(|d| kfd.alloc_vram(d.node_id(), r * 4))
        .collect::<anyhow::Result<_>>()?;
    for f in &mut flags {
        let s = unsafe { f.as_mut_slice_of::<u32>() };
        for v in s.iter_mut() {
            *v = 0;
        }
    }
    for f in &flags {
        for d in &devs {
            let _ = kfd.map_buffer_to_peer(f, d.node_id());
        }
    }
    let flag_va: Vec<u64> = flags.iter().map(|f| f.va()).collect();

    // Per-GPU table of all peers' flag bases, and a host-visible out word.
    let mut ptrs: Vec<DeviceBuffer> = Vec::new();
    let mut outs: Vec<DeviceBuffer> = Vec::new();
    for d in &devs {
        let mut t = d.alloc(r * 8)?;
        {
            let s = unsafe { t.as_mut_slice_of::<u64>() };
            s[..r].copy_from_slice(&flag_va[..r]);
        }
        ptrs.push(t);
        outs.push(d.alloc(64)?);
    }

    let to = Duration::from_secs(10);
    for seq in 1u32..=5 {
        for g in 0..r {
            let mf = flag_va[g];
            let pp = ptrs[g].va();
            let ov = outs[g].va();
            devs[g].arm_xbarrier(mf, pp, r as u32, g as u32, seq, ov)?;
        }
        let mut ok = true;
        for g in 0..r {
            devs[g].wait(to)?;
            let got = outs[g].read_u32(0);
            if got != seq {
                println!("  seq {seq}: GPU {g} returned 0x{got:x} (expected {seq})");
                ok = false;
            }
        }
        println!(
            "seq {seq}: {}",
            if ok {
                "all GPUs passed the in-kernel cross-GPU barrier"
            } else {
                "FAILED"
            }
        );
        if !ok {
            break;
        }
    }
    Ok(())
}

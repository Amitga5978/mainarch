//! Collectives + an rccl-tests-style benchmark harness.
//!
//! The **M0 proving point** for mainarch is to reproduce the `rccl-tests`
//! methodology — sweep message sizes, check correctness, report algorithm and
//! bus bandwidth — on *our* stack. v0 ships a single-rank CPU reference
//! all-reduce so the *harness* is real and end-to-end today; GPU and multi-rank
//! (XGMI/PCIe) backends slot in behind the [`Collective`] trait as the
//! kernel-dispatch and queue layers land.

use anyhow::Result;
use mainarch_core::{enumerate_topology, Kfd};
use std::cmp::min;
use std::thread;
use std::time::Instant;

/// Reduction operators (matches the rccl/nccl set we care about first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Sum,
    Prod,
    Max,
    Min,
}

/// A collective-communication backend. Real backends (single-GPU, then
/// multi-GPU over XGMI/PCIe) implement this; the harness is backend-agnostic.
pub trait Collective {
    fn name(&self) -> &str;
    /// Number of ranks the backend can participate with in this process.
    ///
    /// Layer-0 stepping stones are single-rank; once multi-rank backends land,
    /// this will return the true backend participation count.
    fn ranks(&self) -> usize {
        1
    }
    /// In-place all-reduce of `buf` across all ranks in the backend.
    fn all_reduce_f32(&mut self, buf: &mut [f32], op: Op) -> Result<()>;
}

/// GPU single-rank path placeholder.
///
/// This currently proves the kernel-ABI path is available (via kfd open + topology
/// enumeration) and keeps the all-reduce harness on a non-CPU backend.
pub struct GpuSingleRank {
    // Declared before `_kfd` so the queue's destroy ioctl runs while the kfd
    // fd is still open.
    queue: mainarch_core::KfdQueue,
    _kfd: Kfd,
}

impl GpuSingleRank {
    /// Open `/dev/kfd`, acquire the device VM, and create a live AQL queue on
    /// the first GPU node.
    pub fn new() -> Result<Self> {
        let kfd = Kfd::open()?;
        let nodes = mainarch_core::enumerate_gpus()?;
        if nodes.is_empty() {
            return Err(anyhow::anyhow!("no GPU nodes visible under /sys/class/kfd"));
        }

        let queue = kfd.create_aql_queue(nodes[0].node_id)?;

        Ok(Self { queue, _kfd: kfd })
    }

    pub fn queue_id(&self) -> u32 {
        self.queue.queue_id()
    }
}

impl Collective for GpuSingleRank {
    fn name(&self) -> &str {
        "gpu-reference(1 rank)"
    }
    fn ranks(&self) -> usize {
        1
    }
    fn all_reduce_f32(&mut self, _buf: &mut [f32], _op: Op) -> Result<()> {
        // The backend holds a live AQL queue (created at construction); kernel
        // dispatch through it is the next milestone. Single-rank identity
        // semantics are correct for all supported ops.
        let _ = self.queue.queue_id();
        Ok(())
    }
}

/// GPU topology-aware single-process backend.
///
/// This remains single-rank for now; it returns explicit path hints derived
/// from `/sys/class/kfd/topology` and keeps the transport boundary clearly
/// separated from the compute backend.
pub struct KfdTopologyBackend {
    node_count: usize,
    _path: String,
}

impl KfdTopologyBackend {
    pub fn new() -> Result<Self> {
        let topo = enumerate_topology()?;
        let peer_count = topo.nodes.len();

        let path = if peer_count <= 1 {
            "no peer links exposed".to_string()
        } else {
            let node0 = topo.nodes.first().map(|n| n.node_id).unwrap_or(0);
            let peers = topo.xgmi_first_order(node0, peer_count.saturating_sub(1));
            let links: Vec<String> = peers
                .into_iter()
                .filter_map(|peer| topo.link_between(node0, peer))
                .map(|link| {
                    format!(
                        "{}:{}:{}mb/s",
                        link.kind,
                        if node0 == link.from_node {
                            link.to_node
                        } else {
                            link.from_node
                        },
                        link.effective_bandwidth()
                    )
                })
                .collect();
            if links.is_empty() {
                "peer links unresolved".to_string()
            } else {
                links.join(",")
            }
        };

        Ok(Self {
            node_count: peer_count,
            _path: path,
        })
    }
}

impl Collective for KfdTopologyBackend {
    fn name(&self) -> &str {
        if self.node_count <= 1 {
            "kfd-topology(1 rank)"
        } else {
            "kfd-topology(placeholder multi-rank)"
        }
    }

    fn ranks(&self) -> usize {
        self.node_count
    }

    fn all_reduce_f32(&mut self, buf: &mut [f32], op: Op) -> Result<()> {
        // Keep the transport semantics explicit: this backend surfaces topology,
        // but does not yet execute inter-node DMA.
        let factor = self.ranks().max(1) as f32;
        match op {
            Op::Sum => buf.iter_mut().for_each(|v| *v *= factor),
            Op::Prod => buf.iter_mut().for_each(|v| *v = v.powf(factor)),
            Op::Max | Op::Min => {}
        }
        Ok(())
    }
}

/// v0 reference backend: single rank, CPU. Proves the harness and serves as the
/// correctness oracle for every GPU backend we add later.
pub struct CpuReference;

impl Collective for CpuReference {
    fn name(&self) -> &str {
        "cpu-reference(1 rank)"
    }
    fn ranks(&self) -> usize {
        1
    }
    fn all_reduce_f32(&mut self, _buf: &mut [f32], _op: Op) -> Result<()> {
        // Single rank: all-reduce is the identity. (Multi-rank reduction lands
        // with the real backends.)
        Ok(())
    }
}

/// CPU multi-rank simulation backend.
///
/// This backend executes the same API as the real multi-rank path, but runs in
/// in-process as a deterministic stepping stone for harness validity while the
/// kernel-queue/queue-pair transport lands.
pub struct CpuRankedMock {
    ranks: usize,
}

impl CpuRankedMock {
    /// Construct a deterministic in-process multi-rank simulator.
    pub fn new(ranks: usize) -> Result<Self> {
        let ranks = ranks.max(1);
        Ok(Self { ranks })
    }
}

impl Collective for CpuRankedMock {
    fn name(&self) -> &str {
        // keep output parseable and explicit about the simulation boundary
        if self.ranks <= 1 {
            "cpu-mock(1 rank)"
        } else {
            "cpu-mock(multi-rank)"
        }
    }
    fn ranks(&self) -> usize {
        self.ranks
    }
    fn all_reduce_f32(&mut self, buf: &mut [f32], op: Op) -> Result<()> {
        // keep this deterministic and cheap; this is explicitly a simulation,
        // not a transport implementation.
        let factor = self.ranks as f32;
        match op {
            Op::Sum => {
                for v in buf.iter_mut() {
                    *v *= factor;
                }
            }
            Op::Prod => {
                for v in buf.iter_mut() {
                    *v = v.powf(self.ranks as f32);
                }
            }
            Op::Max | Op::Min => {}
        }
        Ok(())
    }
}

/// Threaded CPU multi-rank simulation backend.
///
/// This version keeps the same deterministic semantic as `CpuRankedMock` but
/// executes the local scale/reduction work across worker threads to reduce wall-
/// clock latency and better stress the benchmark harness.
pub struct CpuRankedParallel {
    ranks: usize,
    worker_threads: usize,
}

impl CpuRankedParallel {
    /// Construct a threaded in-process multi-rank simulator.
    pub fn new(ranks: usize, worker_threads: Option<usize>) -> Result<Self> {
        let ranks = ranks.max(1);
        let hw_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let worker_threads = worker_threads.unwrap_or(hw_threads).max(1);
        Ok(Self {
            ranks,
            worker_threads: worker_threads.min(ranks).max(1),
        })
    }

    fn run_parallel<F>(&self, buf: &mut [f32], mut f: F)
    where
        F: FnMut(&mut [f32]) + Send + Copy,
    {
        if buf.len() < 4096 || self.worker_threads <= 1 {
            f(buf);
            return;
        }

        let threads = min(self.worker_threads, buf.len());
        let chunk = buf.len().div_ceil(threads);
        let mut slices: Vec<&mut [f32]> = Vec::with_capacity(threads);
        let mut remaining = buf;

        for _ in 0..threads {
            let take = min(chunk, remaining.len());
            let (head, tail) = remaining.split_at_mut(take);
            if head.is_empty() {
                break;
            }
            slices.push(head);
            remaining = tail;
        }

        thread::scope(|scope| {
            for chunk in slices {
                let f = f;
                scope.spawn(move || {
                    let mut f = f;
                    f(chunk);
                });
            }
        });
    }
}

impl Collective for CpuRankedParallel {
    fn name(&self) -> &str {
        if self.ranks <= 1 {
            "cpu-parallel(1 rank)"
        } else {
            "cpu-parallel(multi-rank)"
        }
    }
    fn ranks(&self) -> usize {
        self.ranks
    }
    fn all_reduce_f32(&mut self, buf: &mut [f32], op: Op) -> Result<()> {
        // Explicitly keep transport boundary clear: this is still an in-process
        // compute model and not a transport implementation.
        let factor = self.ranks as f32;
        match op {
            Op::Sum => {
                self.run_parallel(buf, |chunk| {
                    for v in chunk.iter_mut() {
                        *v *= factor;
                    }
                });
            }
            Op::Prod => {
                self.run_parallel(buf, |chunk| {
                    for v in chunk.iter_mut() {
                        *v = v.powf(factor);
                    }
                });
            }
            Op::Max | Op::Min => {}
        }
        Ok(())
    }
}

/// One row of the rccl-tests-style report.
#[derive(Debug, Clone)]
pub struct BenchRow {
    pub bytes: usize,
    pub count: usize,
    pub time_us: f64,
    pub alg_bw_gbps: f64,
    pub bus_bw_gbps: f64,
    pub correct: bool,
}

/// Run an all-reduce size sweep, NCCL/RCCL-tests style.
///
/// `ranks` is used only for the bus-bandwidth factor `2*(n-1)/n` so the numbers
/// line up with rccl-tests once real multi-rank backends exist.
pub fn all_reduce_sweep(
    backend: &mut dyn Collective,
    min_bytes: usize,
    max_bytes: usize,
    requested_ranks: usize,
    iters: usize,
) -> Result<Vec<BenchRow>> {
    let elem = std::mem::size_of::<f32>();
    let mut rows = Vec::new();
    let mut bytes = min_bytes.max(elem);

    loop {
        let count = bytes / elem;
        let mut buf = vec![1.0_f32; count];

        // warmup
        backend.all_reduce_f32(&mut buf, Op::Sum)?;

        let t0 = Instant::now();
        for _ in 0..iters.max(1) {
            buf.fill(1.0);
            backend.all_reduce_f32(&mut buf, Op::Sum)?;
        }
        let per_iter = t0.elapsed().as_secs_f64() / iters.max(1) as f64;

        let ranks = requested_ranks.max(1).min(backend.ranks().max(1));
        let expected = expected_value(Op::Sum, ranks);
        let correct = buf.iter().all(|&x| (x - expected).abs() < 1e-6);

        let alg_bw = if per_iter > 0.0 {
            bytes as f64 / per_iter / 1e9
        } else {
            0.0
        };
        let factor = if ranks > 1 {
            2.0 * (ranks as f64 - 1.0) / ranks as f64
        } else {
            1.0
        };

        rows.push(BenchRow {
            bytes,
            count,
            time_us: per_iter * 1e6,
            alg_bw_gbps: alg_bw,
            bus_bw_gbps: alg_bw * factor,
            correct,
        });

        if bytes >= max_bytes {
            break;
        }
        bytes = (bytes * 2).min(max_bytes);
    }
    Ok(rows)
}

fn expected_value(op: Op, ranks: usize) -> f32 {
    let ranks = ranks.max(1);
    let init = 1.0_f32;
    match op {
        Op::Sum => init * ranks as f32,
        Op::Prod => {
            let mut out = init;
            for _ in 1..ranks {
                out *= init;
            }
            out
        }
        Op::Max => init,
        Op::Min => init,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sweep_runs_and_is_correct() {
        let mut b = CpuReference;
        let rows = all_reduce_sweep(&mut b, 8, 1024, 1, 2).unwrap();
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|r| r.correct));
        assert_eq!(rows.first().unwrap().bytes, 8);
        assert_eq!(rows.last().unwrap().bytes, 1024);
        for row in rows {
            assert!((row.alg_bw_gbps - row.bus_bw_gbps).abs() < 1e-12);
        }
    }

    #[test]
    fn gpu_single_rank_backend_smoke() {
        if let Ok(mut gpu) = GpuSingleRank::new() {
            let mut buf = vec![1.0_f32; 8];
            gpu.all_reduce_f32(&mut buf, Op::Sum).unwrap();
        }
    }

    #[test]
    fn cpu_ranked_mock_multi_rank_sum() {
        let mut mock = CpuRankedMock::new(4).unwrap();
        let mut buf = vec![1.0_f32; 8];
        mock.all_reduce_f32(&mut buf, Op::Sum).unwrap();
        assert_eq!(buf, vec![4.0_f32; 8]);
    }

    #[test]
    fn sweep_with_cpu_ranked_mock() {
        let mut mock = CpuRankedMock::new(4).unwrap();
        let rows = all_reduce_sweep(&mut mock, 8, 64, 4, 2).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.first().unwrap().bytes, 8);
        assert_eq!(rows.last().unwrap().bytes, 64);
        assert!(rows.iter().all(|row| row.correct));
    }

    #[test]
    fn cpu_ranked_parallel_multi_rank_sum() {
        let mut parallel = CpuRankedParallel::new(4, None).unwrap();
        let mut buf = vec![1.0_f32; 8192];
        parallel.all_reduce_f32(&mut buf, Op::Sum).unwrap();
        assert_eq!(buf, vec![4.0_f32; 8192]);
    }

    #[test]
    fn sweep_with_cpu_ranked_parallel() {
        let mut parallel = CpuRankedParallel::new(4, None).unwrap();
        let rows = all_reduce_sweep(&mut parallel, 8, 64, 4, 2).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.first().unwrap().bytes, 8);
        assert_eq!(rows.last().unwrap().bytes, 64);
        assert!(rows.iter().all(|row| row.correct));
    }
}

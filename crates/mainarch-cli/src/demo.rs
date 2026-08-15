use anyhow::{anyhow, Context, Result};
use mainarch_core as mcore;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::{Duration, Instant};

const LATEST_TARGET: &str = "public-preview";
const CONTROL_GBPS: u32 = 3405;
const MAINARCH_GBPS: u32 = 3770;
const DECODE_US_TOKEN: u32 = 671;
const KIMI_SPLIT_K: u32 = 64;
const KIMI_MLA_HOT_US: f64 = 144.202;
const KIMI_HOT_STEP_US_TOKEN: f64 = 188.471;
const KIMI_TOKENS_PER_S_PER_SEQUENCE: f64 = 5305.870;
const KIMI_SPLITK8_MLA_HOT_US: f64 = 443.739;
const KIMI_PRE_SPLIT_MLA_HOT_US: f64 = 3285.998;
const MAINARCH_TP8_GBPS: f64 = 10.70;
const RCCL_TP8_GBPS: f64 = 7.72;
const TOKENS: &str = "[46, 51, 90, 1]";
const MODEL_ID: &str = "mainarch-qwen3-235b-a22b-synthetic-proof";
const DEMO_APP_VERSION: &str = "sandbox-app-v8";
const LIVE_LANE_QUEUE_LIMIT: usize = 32;
const PROOF_STREAM_TOKENS: usize = 4;
const PROOF_KV_BLOCK_SIZE_TOKENS: usize = 16;
const PROOF_KV_LOGICAL_BLOCKS: usize = 33;
const LIVE_LANE_KV_BLOCK_CAPACITY: usize = PROOF_KV_LOGICAL_BLOCKS * 4;
const COLLECTIVE_PROOF_PAYLOAD_BYTES: usize = 262_144;
const COLLECTIVE_LARGE_UNFUSED_BYTES: usize = 1_048_576;
const COLLECTIVE_POLICY_STATE: &str = "armed_core_tp8_decode_policy_single_rank_proof";
const COLLECTIVE_GATE_LOGICAL_TP_RANKS: usize = 8;
const COLLECTIVE_GATE_ACTIVE_RANKS: usize = 1;
const COLLECTIVE_GATE_INJECT_DIVERGENCE_ENV: &str = "MAINARCH_DEMO_INJECT_COLLECTIVE_DIVERGENCE";

struct LiveLane {
    node_id: u32,
    in_flight: Arc<AtomicBool>,
    queued: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    cancelled: Arc<AtomicUsize>,
    failed: Arc<AtomicUsize>,
    queue_full: Arc<AtomicUsize>,
    wall_us_total: Arc<AtomicUsize>,
    queue_wait_us_total: Arc<AtomicUsize>,
    tpot_us_total: Arc<AtomicUsize>,
    tpot_samples: Arc<AtomicUsize>,
    stream_requests: Arc<AtomicUsize>,
    ttft_us_total: Arc<AtomicUsize>,
    ttft_samples: Arc<AtomicUsize>,
    stream_tokens: Arc<AtomicUsize>,
    itl_us_total: Arc<AtomicUsize>,
    itl_samples: Arc<AtomicUsize>,
    step_stops: Arc<AtomicUsize>,
    step_cancelled: Arc<AtomicUsize>,
    scheduler_decode_ticks: Arc<AtomicUsize>,
    scheduler_active_max: Arc<AtomicUsize>,
    shared_runner_creations: Arc<AtomicUsize>,
    request_state_allocations: Arc<AtomicUsize>,
    request_state_snapshots: Arc<AtomicUsize>,
    request_state_reused_snapshots: Arc<AtomicUsize>,
    request_state_restores: Arc<AtomicUsize>,
    request_state_bytes_high_watermark: Arc<AtomicUsize>,
    kv_blocks_in_use: Arc<AtomicUsize>,
    kv_leases_active: Arc<AtomicUsize>,
    kv_blocks_high_watermark: Arc<AtomicUsize>,
    kv_leases_acquired: Arc<AtomicUsize>,
    kv_leases_released: Arc<AtomicUsize>,
    kv_lease_denied: Arc<AtomicUsize>,
    kv_block_table_installs: Arc<AtomicUsize>,
    kv_prefill_page_writes: Arc<AtomicUsize>,
    scheduler_kv_ownership_checks: Arc<AtomicUsize>,
    scheduler_kv_ownership_failures: Arc<AtomicUsize>,
    scheduler_kv_active_blocks_checked: Arc<AtomicUsize>,
    kv_pool: Arc<KvBlockLeasePool>,
    tx: mpsc::SyncSender<LiveLaneRequest>,
}

struct LiveLaneRequest {
    enqueued_at: Instant,
    queue_depth_on_enqueue: usize,
    prompt: Option<String>,
    max_steps: usize,
    stream: Option<mpsc::Sender<LiveStreamEvent>>,
    cancelled: Arc<AtomicBool>,
    kv_lease: Option<KvBlockLeaseGuard>,
    reply: mpsc::Sender<LiveProofResult>,
}

struct LiveStreamEvent {
    request_id: u64,
    step_index: usize,
    token: u32,
    text: String,
    step_gpu_us: f64,
    elapsed_ms: f64,
    device_was_warm: bool,
    scheduler_tick: u64,
    scheduler_active: usize,
}

struct CollectiveGateDecision {
    state: &'static str,
    tp_ranks: usize,
    active_ranks: usize,
    local_ready: bool,
    expected_ready: bool,
    local_metadata_hash: u64,
    expected_metadata_hash: u64,
    rank_uniform: bool,
    enter_collective: bool,
    injected_divergence: bool,
    divergent_rank: Option<usize>,
    failure_reason: Option<&'static str>,
}

struct LiveProofResult {
    wall_ms: f64,
    error: Option<String>,
    device_was_warm: bool,
    runner_was_warm: bool,
    request_id: u64,
    gpu_us_per_token: Option<f64>,
    queue_wait_ms: f64,
    queue_depth_on_enqueue: usize,
    cancelled: bool,
    generated_steps: usize,
    stopped_early: bool,
    stop_reason: Option<String>,
    kv_blocks_leased: usize,
    kv_blocks_released: bool,
    kv_block_table: Vec<u32>,
}

enum LiveRun {
    QueueFull { queued: usize, capacity: usize },
    KvUnavailable { requested: usize, capacity: usize },
    Completed(LiveProofResult),
}

enum LiveStreamRun {
    QueueFull {
        queued: usize,
        capacity: usize,
    },
    KvUnavailable {
        requested: usize,
        capacity: usize,
    },
    Queued {
        events: mpsc::Receiver<LiveStreamEvent>,
        done: mpsc::Receiver<LiveProofResult>,
    },
}

impl LiveLane {
    fn start(node_id: u32) -> Arc<Self> {
        let (tx, rx) = mpsc::sync_channel(LIVE_LANE_QUEUE_LIMIT);
        let queued = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let queue_full = Arc::new(AtomicUsize::new(0));
        let wall_us_total = Arc::new(AtomicUsize::new(0));
        let queue_wait_us_total = Arc::new(AtomicUsize::new(0));
        let tpot_us_total = Arc::new(AtomicUsize::new(0));
        let tpot_samples = Arc::new(AtomicUsize::new(0));
        let stream_requests = Arc::new(AtomicUsize::new(0));
        let ttft_us_total = Arc::new(AtomicUsize::new(0));
        let ttft_samples = Arc::new(AtomicUsize::new(0));
        let stream_tokens = Arc::new(AtomicUsize::new(0));
        let itl_us_total = Arc::new(AtomicUsize::new(0));
        let itl_samples = Arc::new(AtomicUsize::new(0));
        let step_stops = Arc::new(AtomicUsize::new(0));
        let step_cancelled = Arc::new(AtomicUsize::new(0));
        let scheduler_decode_ticks = Arc::new(AtomicUsize::new(0));
        let scheduler_active_max = Arc::new(AtomicUsize::new(0));
        let shared_runner_creations = Arc::new(AtomicUsize::new(0));
        let request_state_allocations = Arc::new(AtomicUsize::new(0));
        let request_state_snapshots = Arc::new(AtomicUsize::new(0));
        let request_state_reused_snapshots = Arc::new(AtomicUsize::new(0));
        let request_state_restores = Arc::new(AtomicUsize::new(0));
        let request_state_bytes_high_watermark = Arc::new(AtomicUsize::new(0));
        let kv_blocks_in_use = Arc::new(AtomicUsize::new(0));
        let kv_leases_active = Arc::new(AtomicUsize::new(0));
        let kv_blocks_high_watermark = Arc::new(AtomicUsize::new(0));
        let kv_leases_acquired = Arc::new(AtomicUsize::new(0));
        let kv_leases_released = Arc::new(AtomicUsize::new(0));
        let kv_lease_denied = Arc::new(AtomicUsize::new(0));
        let kv_block_table_installs = Arc::new(AtomicUsize::new(0));
        let kv_prefill_page_writes = Arc::new(AtomicUsize::new(0));
        let scheduler_kv_ownership_checks = Arc::new(AtomicUsize::new(0));
        let scheduler_kv_ownership_failures = Arc::new(AtomicUsize::new(0));
        let scheduler_kv_active_blocks_checked = Arc::new(AtomicUsize::new(0));
        let kv_pool = Arc::new(KvBlockLeasePool::new(
            LIVE_LANE_KV_BLOCK_CAPACITY,
            Arc::clone(&kv_blocks_in_use),
            Arc::clone(&kv_leases_active),
            Arc::clone(&kv_blocks_high_watermark),
            Arc::clone(&kv_leases_acquired),
            Arc::clone(&kv_leases_released),
            Arc::clone(&kv_lease_denied),
        ));
        let worker_queued = Arc::clone(&queued);
        let worker_in_flight = Arc::clone(&in_flight);
        let worker_completed = Arc::clone(&completed);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker_failed = Arc::clone(&failed);
        let worker_wall_us_total = Arc::clone(&wall_us_total);
        let worker_queue_wait_us_total = Arc::clone(&queue_wait_us_total);
        let worker_tpot_us_total = Arc::clone(&tpot_us_total);
        let worker_tpot_samples = Arc::clone(&tpot_samples);
        let worker_step_stops = Arc::clone(&step_stops);
        let worker_step_cancelled = Arc::clone(&step_cancelled);
        let worker_scheduler_decode_ticks = Arc::clone(&scheduler_decode_ticks);
        let worker_scheduler_active_max = Arc::clone(&scheduler_active_max);
        let worker_shared_runner_creations = Arc::clone(&shared_runner_creations);
        let worker_request_state_allocations = Arc::clone(&request_state_allocations);
        let worker_request_state_snapshots = Arc::clone(&request_state_snapshots);
        let worker_request_state_reused_snapshots = Arc::clone(&request_state_reused_snapshots);
        let worker_request_state_restores = Arc::clone(&request_state_restores);
        let worker_request_state_bytes_high_watermark =
            Arc::clone(&request_state_bytes_high_watermark);
        let worker_kv_block_table_installs = Arc::clone(&kv_block_table_installs);
        let worker_kv_prefill_page_writes = Arc::clone(&kv_prefill_page_writes);
        let worker_scheduler_kv_ownership_checks = Arc::clone(&scheduler_kv_ownership_checks);
        let worker_scheduler_kv_ownership_failures = Arc::clone(&scheduler_kv_ownership_failures);
        let worker_scheduler_kv_active_blocks_checked =
            Arc::clone(&scheduler_kv_active_blocks_checked);
        std::thread::spawn(move || {
            live_lane_worker(
                node_id,
                rx,
                worker_queued,
                worker_in_flight,
                worker_completed,
                worker_cancelled,
                worker_failed,
                worker_wall_us_total,
                worker_queue_wait_us_total,
                worker_tpot_us_total,
                worker_tpot_samples,
                worker_step_stops,
                worker_step_cancelled,
                worker_scheduler_decode_ticks,
                worker_scheduler_active_max,
                worker_shared_runner_creations,
                worker_request_state_allocations,
                worker_request_state_snapshots,
                worker_request_state_reused_snapshots,
                worker_request_state_restores,
                worker_request_state_bytes_high_watermark,
                worker_kv_block_table_installs,
                worker_kv_prefill_page_writes,
                worker_scheduler_kv_ownership_checks,
                worker_scheduler_kv_ownership_failures,
                worker_scheduler_kv_active_blocks_checked,
            )
        });
        Arc::new(Self {
            node_id,
            in_flight,
            queued,
            completed,
            cancelled,
            failed,
            queue_full,
            wall_us_total,
            queue_wait_us_total,
            tpot_us_total,
            tpot_samples,
            stream_requests,
            ttft_us_total,
            ttft_samples,
            stream_tokens,
            itl_us_total,
            itl_samples,
            step_stops,
            step_cancelled,
            scheduler_decode_ticks,
            scheduler_active_max,
            shared_runner_creations,
            request_state_allocations,
            request_state_snapshots,
            request_state_reused_snapshots,
            request_state_restores,
            request_state_bytes_high_watermark,
            kv_blocks_in_use,
            kv_leases_active,
            kv_blocks_high_watermark,
            kv_leases_acquired,
            kv_leases_released,
            kv_lease_denied,
            kv_block_table_installs,
            kv_prefill_page_writes,
            scheduler_kv_ownership_checks,
            scheduler_kv_ownership_failures,
            scheduler_kv_active_blocks_checked,
            kv_pool,
            tx,
        })
    }

    fn node_id(&self) -> u32 {
        self.node_id
    }

    fn is_busy(&self) -> bool {
        self.in_flight.load(Ordering::Acquire)
    }

    fn queue_depth(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }

    fn completed_count(&self) -> usize {
        self.completed.load(Ordering::Acquire)
    }

    fn cancelled_count(&self) -> usize {
        self.cancelled.load(Ordering::Acquire)
    }

    fn failed_count(&self) -> usize {
        self.failed.load(Ordering::Acquire)
    }

    fn queue_full_count(&self) -> usize {
        self.queue_full.load(Ordering::Acquire)
    }

    fn wall_us_total(&self) -> usize {
        self.wall_us_total.load(Ordering::Acquire)
    }

    fn queue_wait_us_total(&self) -> usize {
        self.queue_wait_us_total.load(Ordering::Acquire)
    }

    fn tpot_us_total(&self) -> usize {
        self.tpot_us_total.load(Ordering::Acquire)
    }

    fn tpot_samples(&self) -> usize {
        self.tpot_samples.load(Ordering::Acquire)
    }

    fn stream_requests(&self) -> usize {
        self.stream_requests.load(Ordering::Acquire)
    }

    fn ttft_us_total(&self) -> usize {
        self.ttft_us_total.load(Ordering::Acquire)
    }

    fn ttft_samples(&self) -> usize {
        self.ttft_samples.load(Ordering::Acquire)
    }

    fn stream_tokens(&self) -> usize {
        self.stream_tokens.load(Ordering::Acquire)
    }

    fn itl_us_total(&self) -> usize {
        self.itl_us_total.load(Ordering::Acquire)
    }

    fn itl_samples(&self) -> usize {
        self.itl_samples.load(Ordering::Acquire)
    }

    fn step_stops(&self) -> usize {
        self.step_stops.load(Ordering::Acquire)
    }

    fn step_cancelled(&self) -> usize {
        self.step_cancelled.load(Ordering::Acquire)
    }

    fn scheduler_decode_ticks(&self) -> usize {
        self.scheduler_decode_ticks.load(Ordering::Acquire)
    }

    fn scheduler_active_max(&self) -> usize {
        self.scheduler_active_max.load(Ordering::Acquire)
    }

    fn shared_runner_creations(&self) -> usize {
        self.shared_runner_creations.load(Ordering::Acquire)
    }

    fn request_state_allocations(&self) -> usize {
        self.request_state_allocations.load(Ordering::Acquire)
    }

    fn request_state_snapshots(&self) -> usize {
        self.request_state_snapshots.load(Ordering::Acquire)
    }

    fn request_state_reused_snapshots(&self) -> usize {
        self.request_state_reused_snapshots.load(Ordering::Acquire)
    }

    fn request_state_restores(&self) -> usize {
        self.request_state_restores.load(Ordering::Acquire)
    }

    fn request_state_bytes_high_watermark(&self) -> usize {
        self.request_state_bytes_high_watermark
            .load(Ordering::Acquire)
    }

    fn kv_blocks_in_use(&self) -> usize {
        self.kv_blocks_in_use.load(Ordering::Acquire)
    }

    fn kv_leases_active(&self) -> usize {
        self.kv_leases_active.load(Ordering::Acquire)
    }

    fn kv_blocks_high_watermark(&self) -> usize {
        self.kv_blocks_high_watermark.load(Ordering::Acquire)
    }

    fn kv_leases_acquired(&self) -> usize {
        self.kv_leases_acquired.load(Ordering::Acquire)
    }

    fn kv_leases_released(&self) -> usize {
        self.kv_leases_released.load(Ordering::Acquire)
    }

    fn kv_lease_denied(&self) -> usize {
        self.kv_lease_denied.load(Ordering::Acquire)
    }

    fn kv_block_table_installs(&self) -> usize {
        self.kv_block_table_installs.load(Ordering::Acquire)
    }

    fn kv_prefill_page_writes(&self) -> usize {
        self.kv_prefill_page_writes.load(Ordering::Acquire)
    }

    fn scheduler_kv_ownership_checks(&self) -> usize {
        self.scheduler_kv_ownership_checks.load(Ordering::Acquire)
    }

    fn scheduler_kv_ownership_failures(&self) -> usize {
        self.scheduler_kv_ownership_failures.load(Ordering::Acquire)
    }

    fn scheduler_kv_active_blocks_checked(&self) -> usize {
        self.scheduler_kv_active_blocks_checked
            .load(Ordering::Acquire)
    }

    fn record_stream_first_chunk(&self, ttft_us: usize) {
        self.stream_requests.fetch_add(1, Ordering::AcqRel);
        self.ttft_us_total.fetch_add(ttft_us, Ordering::AcqRel);
        self.ttft_samples.fetch_add(1, Ordering::AcqRel);
    }

    fn record_stream_token(&self, itl_us: Option<usize>) {
        self.stream_tokens.fetch_add(1, Ordering::AcqRel);
        if let Some(itl_us) = itl_us {
            self.itl_us_total.fetch_add(itl_us, Ordering::AcqRel);
            self.itl_samples.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn run(&self, cancelled: Arc<AtomicBool>) -> LiveRun {
        let queued_before = self.queued.fetch_add(1, Ordering::AcqRel);
        if queued_before >= LIVE_LANE_QUEUE_LIMIT {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            self.queue_full.fetch_add(1, Ordering::AcqRel);
            return LiveRun::QueueFull {
                queued: queued_before,
                capacity: LIVE_LANE_QUEUE_LIMIT,
            };
        }
        let (reply, done) = mpsc::channel();
        let kv_lease = match self.kv_pool.try_acquire(0, PROOF_KV_LOGICAL_BLOCKS) {
            Some(lease) => lease,
            None => {
                self.queued.fetch_sub(1, Ordering::AcqRel);
                return LiveRun::KvUnavailable {
                    requested: PROOF_KV_LOGICAL_BLOCKS,
                    capacity: LIVE_LANE_KV_BLOCK_CAPACITY,
                };
            }
        };
        let req = LiveLaneRequest {
            enqueued_at: Instant::now(),
            queue_depth_on_enqueue: queued_before,
            prompt: None,
            max_steps: PROOF_STREAM_TOKENS,
            stream: None,
            cancelled,
            kv_lease: Some(kv_lease),
            reply: reply.clone(),
        };
        if let Err(err) = self.tx.try_send(req) {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            return LiveRun::Completed(LiveProofResult {
                wall_ms: 0.0,
                error: Some(format!("persistent live lane queue send failed: {err}")),
                device_was_warm: false,
                runner_was_warm: false,
                request_id: 0,
                gpu_us_per_token: None,
                queue_wait_ms: 0.0,
                queue_depth_on_enqueue: queued_before,
                cancelled: false,
                generated_steps: 0,
                stopped_early: false,
                stop_reason: None,
                kv_blocks_leased: 0,
                kv_blocks_released: false,
                kv_block_table: Vec::new(),
            });
        }
        match done.recv() {
            Ok(result) => LiveRun::Completed(result),
            Err(err) => LiveRun::Completed(LiveProofResult {
                wall_ms: 0.0,
                error: Some(format!("persistent live lane reply failed: {err}")),
                device_was_warm: false,
                runner_was_warm: false,
                request_id: 0,
                gpu_us_per_token: None,
                queue_wait_ms: 0.0,
                queue_depth_on_enqueue: queued_before,
                cancelled: false,
                generated_steps: 0,
                stopped_early: false,
                stop_reason: None,
                kv_blocks_leased: 0,
                kv_blocks_released: false,
                kv_block_table: Vec::new(),
            }),
        }
    }

    fn start_stream(
        &self,
        prompt: String,
        max_steps: usize,
        cancelled: Arc<AtomicBool>,
    ) -> LiveStreamRun {
        let queued_before = self.queued.fetch_add(1, Ordering::AcqRel);
        if queued_before >= LIVE_LANE_QUEUE_LIMIT {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            self.queue_full.fetch_add(1, Ordering::AcqRel);
            return LiveStreamRun::QueueFull {
                queued: queued_before,
                capacity: LIVE_LANE_QUEUE_LIMIT,
            };
        }
        let (reply, done) = mpsc::channel();
        let (stream_tx, events) = mpsc::channel();
        let kv_lease = match self.kv_pool.try_acquire(0, PROOF_KV_LOGICAL_BLOCKS) {
            Some(lease) => lease,
            None => {
                self.queued.fetch_sub(1, Ordering::AcqRel);
                return LiveStreamRun::KvUnavailable {
                    requested: PROOF_KV_LOGICAL_BLOCKS,
                    capacity: LIVE_LANE_KV_BLOCK_CAPACITY,
                };
            }
        };
        let req = LiveLaneRequest {
            enqueued_at: Instant::now(),
            queue_depth_on_enqueue: queued_before,
            prompt: Some(prompt),
            max_steps,
            stream: Some(stream_tx),
            cancelled,
            kv_lease: Some(kv_lease),
            reply: reply.clone(),
        };
        if self.tx.try_send(req).is_err() {
            self.queued.fetch_sub(1, Ordering::AcqRel);
            self.failed.fetch_add(1, Ordering::AcqRel);
            let _ = reply.send(LiveProofResult {
                wall_ms: 0.0,
                error: Some("persistent live lane stream queue send failed".to_string()),
                device_was_warm: false,
                runner_was_warm: false,
                request_id: 0,
                gpu_us_per_token: None,
                queue_wait_ms: 0.0,
                queue_depth_on_enqueue: queued_before,
                cancelled: false,
                generated_steps: 0,
                stopped_early: false,
                stop_reason: None,
                kv_blocks_leased: 0,
                kv_blocks_released: false,
                kv_block_table: Vec::new(),
            });
        }
        LiveStreamRun::Queued { events, done }
    }
}

fn live_lane_worker(
    node_id: u32,
    rx: mpsc::Receiver<LiveLaneRequest>,
    queued: Arc<AtomicUsize>,
    in_flight: Arc<AtomicBool>,
    completed: Arc<AtomicUsize>,
    cancelled_total: Arc<AtomicUsize>,
    failed_total: Arc<AtomicUsize>,
    wall_us_total: Arc<AtomicUsize>,
    queue_wait_us_total: Arc<AtomicUsize>,
    tpot_us_total: Arc<AtomicUsize>,
    tpot_samples: Arc<AtomicUsize>,
    step_stops_total: Arc<AtomicUsize>,
    step_cancelled_total: Arc<AtomicUsize>,
    scheduler_decode_ticks: Arc<AtomicUsize>,
    scheduler_active_max: Arc<AtomicUsize>,
    shared_runner_creations: Arc<AtomicUsize>,
    request_state_allocations: Arc<AtomicUsize>,
    request_state_snapshots: Arc<AtomicUsize>,
    request_state_reused_snapshots: Arc<AtomicUsize>,
    request_state_restores: Arc<AtomicUsize>,
    request_state_bytes_high_watermark: Arc<AtomicUsize>,
    kv_block_table_installs: Arc<AtomicUsize>,
    kv_prefill_page_writes: Arc<AtomicUsize>,
    scheduler_kv_ownership_checks: Arc<AtomicUsize>,
    scheduler_kv_ownership_failures: Arc<AtomicUsize>,
    scheduler_kv_active_blocks_checked: Arc<AtomicUsize>,
) {
    let counters = LiveWorkerCounters {
        queued,
        completed,
        cancelled_total,
        failed_total,
        wall_us_total,
        queue_wait_us_total,
        tpot_us_total,
        tpot_samples,
        step_stops_total,
        step_cancelled_total,
        shared_runner_creations,
        request_state_allocations,
        request_state_snapshots,
        request_state_reused_snapshots,
        request_state_restores,
        request_state_bytes_high_watermark,
        kv_block_table_installs,
        kv_prefill_page_writes,
        scheduler_kv_ownership_checks,
        scheduler_kv_ownership_failures,
        scheduler_kv_active_blocks_checked,
    };
    let mut dev = None;
    let mut runner = None;
    let mut request_id = 0u64;
    let mut active = Vec::new();
    let mut rx_disconnected = false;

    loop {
        if active.is_empty() {
            in_flight.store(false, Ordering::Release);
            if rx_disconnected {
                break;
            }
            match rx.recv() {
                Ok(req) => admit_live_request(
                    node_id,
                    req,
                    &mut request_id,
                    &mut dev,
                    &mut runner,
                    &mut active,
                    &counters,
                ),
                Err(_) => break,
            }
        }

        loop {
            match rx.try_recv() {
                Ok(req) => admit_live_request(
                    node_id,
                    req,
                    &mut request_id,
                    &mut dev,
                    &mut runner,
                    &mut active,
                    &counters,
                ),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    rx_disconnected = true;
                    break;
                }
            }
        }

        if active.is_empty() {
            continue;
        }

        in_flight.store(true, Ordering::Release);
        let active_count = active.len();
        let scheduler_tick = scheduler_decode_ticks.fetch_add(1, Ordering::AcqRel) as u64 + 1;
        record_scheduler_active_high_watermark(&scheduler_active_max, active_count);

        counters
            .scheduler_kv_ownership_checks
            .fetch_add(1, Ordering::AcqRel);
        match validate_active_kv_ownership(&active) {
            Ok(active_blocks) => {
                counters
                    .scheduler_kv_active_blocks_checked
                    .fetch_add(active_blocks, Ordering::AcqRel);
            }
            Err(err) => {
                counters
                    .scheduler_kv_ownership_failures
                    .fetch_add(1, Ordering::AcqRel);
                let error = format!(
                    "scheduler KV ownership validation failed before decode tick {scheduler_tick}: {err:#}"
                );
                while !active.is_empty() {
                    let active_req = active.remove(0);
                    finish_active_live_request(active_req, Some(error.clone()), false, &counters);
                }
                continue;
            }
        }

        let mut idx = 0;
        while idx < active.len() {
            let mut should_finish = false;
            let mut finish_error = None;
            let mut request_cancelled = false;
            let max_steps = active[idx].req.max_steps.min(PROOF_STREAM_TOKENS);

            if active[idx].req.cancelled.load(Ordering::Acquire) {
                active[idx].stopped_early = true;
                active[idx].stop_reason = Some("client_cancelled_before_decode_step".to_string());
                request_cancelled = true;
                should_finish = true;
            } else if active[idx].step_index >= max_steps {
                active[idx].stopped_early = true;
                active[idx].stop_reason = Some("max_tokens".to_string());
                should_finish = true;
            } else {
                let step_index = active[idx].step_index;
                let step = if let (Some(dev), Some(runner)) = (dev.as_mut(), runner.as_mut()) {
                    match runner.restore_stepwise_state(&active[idx].state) {
                        Ok(()) => {
                            counters
                                .request_state_restores
                                .fetch_add(1, Ordering::AcqRel);
                            runner.run_stepwise_next(dev, step_index)
                        }
                        Err(err) => Err(err),
                    }
                } else {
                    Err(anyhow::anyhow!(
                        "live lane active request without shared GPU runner"
                    ))
                };
                match step {
                    Ok(step) => {
                        active[idx].gpu_us_total += step.gpu_us;
                        active[idx].step_index += 1;
                        active[idx].generated_steps = active[idx].step_index;
                        if let Some(stream_tx) = active[idx].req.stream.as_ref().cloned() {
                            if stream_tx
                                .send(LiveStreamEvent {
                                    request_id: active[idx].request_id,
                                    step_index: step.step_index,
                                    token: step.token,
                                    text: stream_token_text(
                                        active[idx].request_id,
                                        &active[idx].prompt,
                                        step.step_index,
                                        step.token,
                                    ),
                                    step_gpu_us: step.gpu_us,
                                    elapsed_ms: active[idx].started.elapsed().as_secs_f64()
                                        * 1000.0,
                                    device_was_warm: active[idx].device_was_warm,
                                    scheduler_tick,
                                    scheduler_active: active_count,
                                })
                                .is_err()
                            {
                                active[idx].req.cancelled.store(true, Ordering::Release);
                                active[idx].stopped_early = true;
                                active[idx].stop_reason =
                                    Some("stream_receiver_disconnected".to_string());
                                request_cancelled = true;
                                should_finish = true;
                            }
                        }
                        if !should_finish && active[idx].generated_steps >= max_steps {
                            if active[idx].generated_steps < PROOF_STREAM_TOKENS {
                                active[idx].stopped_early = true;
                                active[idx].stop_reason = Some("max_tokens".to_string());
                            }
                            should_finish = true;
                        }
                        if !should_finish && active[idx].generated_steps >= PROOF_STREAM_TOKENS {
                            should_finish = true;
                        }
                        if !should_finish {
                            if let Some(runner) = runner.as_mut() {
                                match runner.capture_stepwise_state_into(&mut active[idx].state) {
                                    Ok(()) => {
                                        counters
                                            .request_state_snapshots
                                            .fetch_add(1, Ordering::AcqRel);
                                        counters
                                            .request_state_reused_snapshots
                                            .fetch_add(1, Ordering::AcqRel);
                                        record_scheduler_active_high_watermark(
                                            &counters.request_state_bytes_high_watermark,
                                            active[idx].state.byte_len(),
                                        );
                                    }
                                    Err(err) => {
                                        finish_error = Some(format!("{err:#}"));
                                        should_finish = true;
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        finish_error = Some(format!("{err:#}"));
                        should_finish = true;
                    }
                }
            }

            if should_finish {
                let active_req = active.remove(idx);
                finish_active_live_request(active_req, finish_error, request_cancelled, &counters);
            } else {
                idx += 1;
            }
        }
    }

    in_flight.store(false, Ordering::Release);
}

struct LiveWorkerCounters {
    queued: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    cancelled_total: Arc<AtomicUsize>,
    failed_total: Arc<AtomicUsize>,
    wall_us_total: Arc<AtomicUsize>,
    queue_wait_us_total: Arc<AtomicUsize>,
    tpot_us_total: Arc<AtomicUsize>,
    tpot_samples: Arc<AtomicUsize>,
    step_stops_total: Arc<AtomicUsize>,
    step_cancelled_total: Arc<AtomicUsize>,
    shared_runner_creations: Arc<AtomicUsize>,
    request_state_allocations: Arc<AtomicUsize>,
    request_state_snapshots: Arc<AtomicUsize>,
    request_state_reused_snapshots: Arc<AtomicUsize>,
    request_state_restores: Arc<AtomicUsize>,
    request_state_bytes_high_watermark: Arc<AtomicUsize>,
    kv_block_table_installs: Arc<AtomicUsize>,
    kv_prefill_page_writes: Arc<AtomicUsize>,
    scheduler_kv_ownership_checks: Arc<AtomicUsize>,
    scheduler_kv_ownership_failures: Arc<AtomicUsize>,
    scheduler_kv_active_blocks_checked: Arc<AtomicUsize>,
}

struct ActiveLiveRequest {
    req: LiveLaneRequest,
    request_id: u64,
    started: Instant,
    queue_wait_ms: f64,
    device_was_warm: bool,
    runner_was_warm: bool,
    state: mcore::model::CachedModelDecodeProofState,
    prompt: String,
    step_index: usize,
    gpu_us_total: f64,
    generated_steps: usize,
    stopped_early: bool,
    stop_reason: Option<String>,
    kv_blocks_leased: usize,
    kv_block_table: Vec<u32>,
}

fn admit_live_request(
    node_id: u32,
    req: LiveLaneRequest,
    request_id: &mut u64,
    dev: &mut Option<mcore::GpuDevice>,
    runner: &mut Option<mcore::model::CachedModelDecodeProof>,
    active: &mut Vec<ActiveLiveRequest>,
    counters: &LiveWorkerCounters,
) {
    let queue_wait_ms = req.enqueued_at.elapsed().as_secs_f64() * 1000.0;
    let device_was_warm = dev.is_some();
    let runner_was_warm = runner.is_some();
    if req.cancelled.load(Ordering::Acquire) {
        finish_unstarted_live_request(
            req,
            *request_id,
            queue_wait_ms,
            device_was_warm,
            runner_was_warm,
            "request cancelled before live decode execution".to_string(),
            true,
            counters,
        );
        return;
    }

    if dev.is_none() {
        match mcore::GpuDevice::open(node_id) {
            Ok(opened) => *dev = Some(opened),
            Err(err) => {
                finish_unstarted_live_request(
                    req,
                    *request_id,
                    queue_wait_ms,
                    device_was_warm,
                    runner_was_warm,
                    format!("{err:#}"),
                    false,
                    counters,
                );
                return;
            }
        }
    }

    *request_id = (*request_id).wrapping_add(1);
    let request_id = *request_id;
    let Some(kv_lease) = req.kv_lease.as_ref() else {
        finish_unstarted_live_request(
            req,
            request_id,
            queue_wait_ms,
            true,
            runner_was_warm,
            "live lane request reached execution without a KV block lease".to_string(),
            false,
            counters,
        );
        return;
    };
    let kv_blocks_leased = kv_lease.blocks();
    let kv_block_table = kv_lease.block_table().to_vec();

    let dev_ref = dev
        .as_mut()
        .expect("live lane opened GPU but device is absent");
    if runner.is_none() {
        match mcore::model::CachedModelDecodeProof::new(dev_ref) {
            Ok(opened) => {
                *runner = Some(opened);
                counters
                    .shared_runner_creations
                    .fetch_add(1, Ordering::AcqRel);
            }
            Err(err) => {
                finish_unstarted_live_request(
                    req,
                    request_id,
                    queue_wait_ms,
                    true,
                    runner_was_warm,
                    format!("{err:#}"),
                    false,
                    counters,
                );
                return;
            }
        }
    }
    let runner_ref = runner
        .as_mut()
        .expect("live lane opened shared runner but runner is absent");
    let prefill_pages_initialized =
        match runner_ref.begin_stepwise_request_with_block_table(dev_ref, &kv_block_table) {
            Ok(pages) => pages,
            Err(err) => {
                finish_unstarted_live_request(
                    req,
                    request_id,
                    queue_wait_ms,
                    true,
                    runner_was_warm,
                    format!("{err:#}"),
                    false,
                    counters,
                );
                return;
            }
        };
    counters
        .kv_block_table_installs
        .fetch_add(1, Ordering::AcqRel);
    counters
        .kv_prefill_page_writes
        .fetch_add(prefill_pages_initialized, Ordering::AcqRel);
    let state = runner_ref.capture_stepwise_state();
    counters
        .request_state_allocations
        .fetch_add(1, Ordering::AcqRel);
    counters
        .request_state_snapshots
        .fetch_add(1, Ordering::AcqRel);
    record_scheduler_active_high_watermark(
        &counters.request_state_bytes_high_watermark,
        state.byte_len(),
    );

    active.push(ActiveLiveRequest {
        prompt: req
            .prompt
            .as_deref()
            .unwrap_or("What makes mainarch different?")
            .to_string(),
        req,
        request_id,
        started: Instant::now(),
        queue_wait_ms,
        device_was_warm,
        runner_was_warm,
        state,
        step_index: 0,
        gpu_us_total: 0.0,
        generated_steps: 0,
        stopped_early: false,
        stop_reason: None,
        kv_blocks_leased,
        kv_block_table,
    });
}

fn finish_unstarted_live_request(
    mut req: LiveLaneRequest,
    request_id: u64,
    queue_wait_ms: f64,
    device_was_warm: bool,
    runner_was_warm: bool,
    error: String,
    request_cancelled: bool,
    counters: &LiveWorkerCounters,
) {
    let kv_block_table = req
        .kv_lease
        .as_ref()
        .map(|lease| lease.block_table().to_vec())
        .unwrap_or_default();
    let kv_blocks_leased = kv_block_table.len();
    let kv_blocks_released = req.kv_lease.take().is_some();
    if request_cancelled {
        counters.cancelled_total.fetch_add(1, Ordering::AcqRel);
        counters.queue_wait_us_total.fetch_add(
            (queue_wait_ms * 1000.0).max(0.0).round() as usize,
            Ordering::AcqRel,
        );
    } else {
        counters.failed_total.fetch_add(1, Ordering::AcqRel);
    }
    counters.queued.fetch_sub(1, Ordering::AcqRel);
    let _ = req.reply.send(LiveProofResult {
        wall_ms: 0.0,
        error: Some(error),
        device_was_warm,
        runner_was_warm,
        request_id,
        gpu_us_per_token: None,
        queue_wait_ms,
        queue_depth_on_enqueue: req.queue_depth_on_enqueue,
        cancelled: request_cancelled,
        generated_steps: 0,
        stopped_early: request_cancelled,
        stop_reason: if request_cancelled {
            Some("client_cancelled_before_execution".to_string())
        } else {
            None
        },
        kv_blocks_leased,
        kv_blocks_released,
        kv_block_table,
    });
}

fn finish_active_live_request(
    mut active: ActiveLiveRequest,
    error: Option<String>,
    request_cancelled: bool,
    counters: &LiveWorkerCounters,
) {
    let wall_ms = active.started.elapsed().as_secs_f64() * 1000.0;
    let gpu_us_per_token = if active.generated_steps == 0 {
        None
    } else {
        Some(active.gpu_us_total / active.generated_steps as f64)
    };
    let kv_blocks_released = active.req.kv_lease.take().is_some();
    if active.stopped_early {
        counters.step_stops_total.fetch_add(1, Ordering::AcqRel);
        if request_cancelled {
            counters.step_cancelled_total.fetch_add(1, Ordering::AcqRel);
        }
    }
    if error.is_none() {
        counters.queue_wait_us_total.fetch_add(
            (active.queue_wait_ms * 1000.0).max(0.0).round() as usize,
            Ordering::AcqRel,
        );
        if request_cancelled {
            counters.cancelled_total.fetch_add(1, Ordering::AcqRel);
        } else {
            counters.completed.fetch_add(1, Ordering::AcqRel);
            counters.wall_us_total.fetch_add(
                (wall_ms * 1000.0).max(0.0).round() as usize,
                Ordering::AcqRel,
            );
        }
        if let Some(tpot) = gpu_us_per_token {
            counters
                .tpot_us_total
                .fetch_add(tpot.max(0.0).round() as usize, Ordering::AcqRel);
            counters.tpot_samples.fetch_add(1, Ordering::AcqRel);
        }
    } else {
        counters.failed_total.fetch_add(1, Ordering::AcqRel);
    }
    counters.queued.fetch_sub(1, Ordering::AcqRel);
    let _ = active.req.reply.send(LiveProofResult {
        wall_ms,
        error,
        device_was_warm: active.device_was_warm,
        runner_was_warm: active.runner_was_warm,
        request_id: active.request_id,
        gpu_us_per_token,
        queue_wait_ms: active.queue_wait_ms,
        queue_depth_on_enqueue: active.req.queue_depth_on_enqueue,
        cancelled: request_cancelled,
        generated_steps: active.generated_steps,
        stopped_early: active.stopped_early,
        stop_reason: active.stop_reason,
        kv_blocks_leased: active.kv_blocks_leased,
        kv_blocks_released,
        kv_block_table: active.kv_block_table,
    });
}

fn record_scheduler_active_high_watermark(metric: &AtomicUsize, value: usize) {
    let mut prev = metric.load(Ordering::Acquire);
    while value > prev {
        match metric.compare_exchange(prev, value, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => break,
            Err(next) => prev = next,
        }
    }
}

struct KvBlockLeasePool {
    capacity: usize,
    free_blocks: Mutex<Vec<u32>>,
    in_use: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    high_watermark: Arc<AtomicUsize>,
    acquired: Arc<AtomicUsize>,
    released: Arc<AtomicUsize>,
    denied: Arc<AtomicUsize>,
}

impl KvBlockLeasePool {
    fn new(
        capacity: usize,
        in_use: Arc<AtomicUsize>,
        active: Arc<AtomicUsize>,
        high_watermark: Arc<AtomicUsize>,
        acquired: Arc<AtomicUsize>,
        released: Arc<AtomicUsize>,
        denied: Arc<AtomicUsize>,
    ) -> Self {
        let free_blocks = (0..capacity as u32).rev().collect();
        Self {
            capacity,
            free_blocks: Mutex::new(free_blocks),
            in_use,
            active,
            high_watermark,
            acquired,
            released,
            denied,
        }
    }

    fn try_acquire(self: &Arc<Self>, request_id: u64, blocks: usize) -> Option<KvBlockLeaseGuard> {
        let mut free_blocks = self.free_blocks.lock().ok()?;
        if free_blocks.len() < blocks {
            self.denied.fetch_add(1, Ordering::AcqRel);
            return None;
        }
        let mut block_table = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            if let Some(block_id) = free_blocks.pop() {
                block_table.push(block_id);
            }
        }
        self.in_use.fetch_add(block_table.len(), Ordering::AcqRel);
        self.active.fetch_add(1, Ordering::AcqRel);
        self.acquired.fetch_add(1, Ordering::AcqRel);
        self.record_high_watermark(self.capacity - free_blocks.len());
        Some(KvBlockLeaseGuard {
            request_id,
            block_table,
            pool: Arc::clone(self),
            released: false,
        })
    }

    fn release_blocks(&self, block_table: &mut Vec<u32>) {
        let blocks = block_table.len();
        if let Ok(mut free_blocks) = self.free_blocks.lock() {
            for block_id in block_table.drain(..).rev() {
                free_blocks.push(block_id);
            }
            self.in_use.fetch_sub(blocks, Ordering::AcqRel);
            self.active.fetch_sub(1, Ordering::AcqRel);
            self.released.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn record_high_watermark(&self, value: usize) {
        let mut current = self.high_watermark.load(Ordering::Acquire);
        while value > current {
            match self.high_watermark.compare_exchange(
                current,
                value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
    }

    fn validate_active_block_tables(&self, active: &[ActiveLiveRequest]) -> Result<usize> {
        let mut active_owner: Vec<Option<usize>> = vec![None; self.capacity];
        let mut active_blocks = 0usize;

        for (request_idx, request) in active.iter().enumerate() {
            if request.kv_block_table.len() != request.kv_blocks_leased {
                return Err(anyhow!(
                    "active request {request_idx} has {} block-table entries but {} leased blocks",
                    request.kv_block_table.len(),
                    request.kv_blocks_leased
                ));
            }
            if request.kv_block_table.len() != PROOF_KV_LOGICAL_BLOCKS {
                return Err(anyhow!(
                    "active request {request_idx} has {} block-table entries; expected {}",
                    request.kv_block_table.len(),
                    PROOF_KV_LOGICAL_BLOCKS
                ));
            }

            for &block in &request.kv_block_table {
                let block_idx = block as usize;
                if block_idx >= self.capacity {
                    return Err(anyhow!(
                        "active request {request_idx} owns out-of-range KV block {block}; capacity {}",
                        self.capacity
                    ));
                }
                if let Some(existing_request_idx) = active_owner[block_idx] {
                    return Err(anyhow!(
                        "KV block {block} is owned by active requests {existing_request_idx} and {request_idx}"
                    ));
                }
                active_owner[block_idx] = Some(request_idx);
                active_blocks += 1;
            }
        }

        let mut free_blocks_seen = vec![false; self.capacity];
        {
            let free_blocks = self
                .free_blocks
                .lock()
                .map_err(|_| anyhow!("KV block free list mutex poisoned"))?;
            for &block in free_blocks.iter() {
                let block_idx = block as usize;
                if block_idx >= self.capacity {
                    return Err(anyhow!(
                        "free list contains out-of-range KV block {block}; capacity {}",
                        self.capacity
                    ));
                }
                if free_blocks_seen[block_idx] {
                    return Err(anyhow!("free list contains duplicate KV block {block}"));
                }
                free_blocks_seen[block_idx] = true;
            }
        }

        for (block_idx, owner) in active_owner.iter().enumerate() {
            if owner.is_some() && free_blocks_seen[block_idx] {
                return Err(anyhow!(
                    "active KV block {block_idx} is also present in the free list"
                ));
            }
        }

        let in_use = self.in_use.load(Ordering::Acquire);
        if active_blocks > in_use {
            return Err(anyhow!(
                "active set references {active_blocks} KV blocks but only {in_use} are marked in use"
            ));
        }

        Ok(active_blocks)
    }
}

fn validate_active_kv_ownership(active: &[ActiveLiveRequest]) -> Result<usize> {
    let lease = active
        .first()
        .and_then(|request| request.req.kv_lease.as_ref())
        .ok_or_else(|| anyhow!("active scheduler set has no KV lease pool"))?;
    lease.pool.validate_active_block_tables(active)
}

struct KvBlockLeaseGuard {
    request_id: u64,
    block_table: Vec<u32>,
    pool: Arc<KvBlockLeasePool>,
    released: bool,
}

impl KvBlockLeaseGuard {
    fn blocks(&self) -> usize {
        self.block_table.len()
    }

    fn block_table(&self) -> &[u32] {
        &self.block_table
    }

    fn release_inner(&mut self) {
        if !self.released {
            self.pool.release_blocks(&mut self.block_table);
            self.released = true;
        }
    }
}

impl Drop for KvBlockLeaseGuard {
    fn drop(&mut self) {
        let _ = self.request_id;
        self.release_inner();
    }
}

struct CancellationWatch {
    cancelled: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
}

impl CancellationWatch {
    fn start(stream: &TcpStream) -> Self {
        let cancelled = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));
        if let Ok(monitor) = stream.try_clone() {
            let cancelled_worker = Arc::clone(&cancelled);
            let done_worker = Arc::clone(&done);
            std::thread::spawn(move || {
                let _ = monitor.set_read_timeout(Some(Duration::from_millis(10)));
                let mut buf = [0u8; 1];
                while !done_worker.load(Ordering::Acquire) {
                    match monitor.peek(&mut buf) {
                        Ok(0) => {
                            cancelled_worker.store(true, Ordering::Release);
                            break;
                        }
                        Ok(_) => std::thread::sleep(Duration::from_millis(10)),
                        Err(err)
                            if matches!(
                                err.kind(),
                                ErrorKind::WouldBlock
                                    | ErrorKind::TimedOut
                                    | ErrorKind::Interrupted
                            ) => {}
                        Err(_) => {
                            cancelled_worker.store(true, Ordering::Release);
                            break;
                        }
                    }
                }
            });
        }
        Self { cancelled, done }
    }
}

impl Drop for CancellationWatch {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
    }
}

pub fn serve(bind: &str, node: u32, olmo: Option<Arc<crate::olmo_lane::OlmoLane>>) -> Result<()> {
    let listener =
        TcpListener::bind(bind).with_context(|| format!("binding demo server to {bind}"))?;
    println!("mainarch demo-serve - one-page decode-latency demo");
    println!("  serving http://{bind}/");
    println!("  evidence endpoint http://{bind}/api/evidence");
    println!("  model discovery http://{bind}/v1/models");
    println!("  OpenAI-shaped endpoint http://{bind}/v1/chat/completions");
    println!("  persistent live proof lane node {node}");
    match &olmo {
        Some(lane) => {
            println!(
                "  OLMo 2 lane: {} resident on node {}, {} layers, hidden {}, vocab {}, {:.2} GiB",
                lane.model_id,
                lane.node,
                lane.layers,
                lane.hidden,
                lane.vocab,
                lane.device_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            println!("  /v1/chat/completions is served by the real model");
        }
        None => println!("  /v1/chat/completions is served by the synthetic proof lane"),
    }

    let live_lane = LiveLane::start(node);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let live_lane = Arc::clone(&live_lane);
                let olmo = olmo.clone();
                std::thread::spawn(move || {
                    if let Err(err) = handle_client(stream, live_lane, olmo) {
                        eprintln!("demo-serve request error: {err:#}");
                    }
                });
            }
            Err(err) => eprintln!("demo-serve accept error: {err:#}"),
        }
    }
    Ok(())
}

fn handle_client(
    mut stream: TcpStream,
    live_lane: Arc<LiveLane>,
    olmo: Option<Arc<crate::olmo_lane::OlmoLane>>,
) -> Result<()> {
    let req = read_http_request(&mut stream)?;
    let mut request_line = req.lines().next().unwrap_or("").split_whitespace();
    let method = request_line.next().unwrap_or("");
    let path = request_line.next().unwrap_or("/");
    let route = path.split('?').next().unwrap_or(path);
    let body = req.split("\r\n\r\n").nth(1).unwrap_or("").trim();

    if method == "OPTIONS" {
        respond(
            &mut stream,
            "204 No Content",
            "text/plain; charset=utf-8",
            "",
        )
    } else if route == "/" || route == "/index.html" {
        respond(&mut stream, "200 OK", "text/html; charset=utf-8", INDEX)
    } else if route == "/api/evidence" {
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &evidence_json(),
        )
    } else if route == "/api/compare" {
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &comparison_json(),
        )
    } else if route == "/metrics" {
        respond(
            &mut stream,
            "200 OK",
            "text/plain; version=0.0.4; charset=utf-8",
            &metrics_text(&live_lane),
        )
    } else if route == "/api/demo" {
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &demo_manifest_json(&live_lane),
        )
    } else if route == "/v1/models" {
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &olmo_aware_models_json(olmo.as_deref()),
        )
    } else if route.starts_with("/api/run") {
        let cancellation = CancellationWatch::start(&stream);
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &run_live_decode_json(&live_lane, Arc::clone(&cancellation.cancelled)),
        )
    } else if route.starts_with("/api/generate") {
        let cancellation = CancellationWatch::start(&stream);
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &run_prompt_generate_json(&live_lane, body, Arc::clone(&cancellation.cancelled)),
        )
    } else if route.starts_with("/v1/chat/completions") {
        let cancellation = CancellationWatch::start(&stream);
        let streaming = body_requests_stream(body);
        if streaming {
            if let Some(lane) = olmo.as_ref() {
                return respond_olmo_chat_stream(&mut stream, lane, body);
            }
            respond_chat_completion_stream(
                &mut stream,
                &live_lane,
                body,
                Arc::clone(&cancellation.cancelled),
            )
        } else if let Some(lane) = olmo.as_ref() {
            let response = run_olmo_chat_json(lane, body);
            respond(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                &response,
            )
        } else {
            let response = run_chat_completions_json(
                &live_lane,
                body,
                false,
                Arc::clone(&cancellation.cancelled),
            );
            respond(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                &response,
            )
        }
    } else if route == "/api/health" {
        respond(
            &mut stream,
            "200 OK",
            "application/json; charset=utf-8",
            &olmo_aware_health_json(&live_lane, olmo.as_deref()),
        )
    } else {
        respond(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            "not found\n",
        )
    }
}

fn read_http_request(stream: &mut TcpStream) -> Result<String> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let mut body_start = None;
    let mut content_length = 0usize;

    loop {
        let n = stream.read(&mut tmp).context("reading HTTP request")?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if body_start.is_none() {
            if let Some(pos) = find_http_header_end(&buf) {
                body_start = Some(pos + 4);
                content_length = parse_content_length(&buf[..pos]);
            }
        }
        if let Some(start) = body_start {
            if buf.len() >= start + content_length {
                break;
            }
        }
        if buf.len() > 1_048_576 {
            anyhow::bail!("demo HTTP request exceeded 1 MiB");
        }
    }

    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn find_http_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_content_length(header: &[u8]) -> usize {
    let Ok(header) = std::str::from_utf8(header) else {
        return 0;
    };
    header
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find_map(|(name, value)| {
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(0)
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Max-Age: 86400\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .context("writing HTTP headers")?;
    stream
        .write_all(body.as_bytes())
        .context("writing HTTP body")?;
    Ok(())
}

fn respond_sse_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Max-Age: 86400\r\nConnection: close\r\n\r\n"
    )
    .context("writing SSE headers")?;
    stream.flush().context("flushing SSE headers")?;
    Ok(())
}

fn write_sse_data(stream: &mut TcpStream, data: &str) -> Result<()> {
    write!(stream, "data: {data}\n\n").context("writing SSE data")?;
    stream.flush().context("flushing SSE data")?;
    Ok(())
}

fn respond_chat_completion_stream(
    stream: &mut TcpStream,
    live_lane: &LiveLane,
    body: &str,
    cancelled: Arc<AtomicBool>,
) -> Result<()> {
    respond_sse_headers(stream)?;
    let node_id = live_lane.node_id();
    let prompt = extract_chat_prompt(body);
    let max_steps = extract_max_completion_steps(body).unwrap_or(PROOF_STREAM_TOKENS);
    let stream_started = Instant::now();
    let collective_policy = collective_policy_json();
    let collective_gate = collective_entry_gate(&prompt, max_steps);
    let collective_gate_json = collective_gate_json(&collective_gate);
    if !collective_gate.enter_collective {
        write_sse_data(
            stream,
            &format!(
                r#"{{"error":{{"message":"rank-uniform collective gate rejected the request before decode","type":"mainarch_collective_gate_error","code":"rank_uniform_collective_gate_failed"}},"model":"{MODEL_ID}","mainarch":{{"ok":false,"busy":false,"node":{node_id},"stream":true,"stream_timing":"pre_decode_collective_gate","live_lane":"persistent-worker","model_runner":"cached-synthetic-proof","collective_policy":{collective_policy},"collective_gate":{collective_gate_json}}}}}"#
            ),
        )?;
        write_sse_data(stream, "[DONE]")?;
        return Ok(());
    }
    match live_lane.start_stream(prompt, max_steps, Arc::clone(&cancelled)) {
        LiveStreamRun::QueueFull { queued, capacity } => {
            write_sse_data(
                stream,
                &format!(
                    r#"{{"error":{{"message":"live decode proof queue is full","type":"server_queue_full","code":"mainarch_live_lane_queue_full"}},"model":"{MODEL_ID}","mainarch":{{"ok":false,"busy":true,"node":{node_id},"queue_depth":{queued},"queue_capacity":{capacity},"stream":true,"live_lane":"persistent-worker","model_runner":"cached-synthetic-proof","collective_policy":{collective_policy},"collective_gate":{collective_gate_json}}}}}"#
                ),
            )?;
            write_sse_data(stream, "[DONE]")?;
            return Ok(());
        }
        LiveStreamRun::KvUnavailable {
            requested,
            capacity,
        } => {
            write_sse_data(
                stream,
                &format!(
                    r#"{{"error":{{"message":"live decode proof KV block capacity is full","type":"server_kv_full","code":"mainarch_live_lane_kv_full"}},"model":"{MODEL_ID}","mainarch":{{"ok":false,"busy":true,"node":{node_id},"requested_kv_blocks":{requested},"kv_block_capacity":{capacity},"kv_blocks_per_request":{PROOF_KV_LOGICAL_BLOCKS},"stream":true,"live_lane":"persistent-worker","model_runner":"cached-synthetic-proof","collective_policy":{collective_policy},"collective_gate":{collective_gate_json}}}}}"#
                ),
            )?;
            write_sse_data(stream, "[DONE]")?;
            return Ok(());
        }
        LiveStreamRun::Queued { events, done } => {
            let mut first_chunk_us = None;
            let mut last_chunk_at = None;
            let mut token_chunks = 0usize;
            while let Ok(event) = events.recv() {
                if cancelled.load(Ordering::Acquire) {
                    break;
                }
                let role = if event.step_index == 0 {
                    "\"role\":\"assistant\","
                } else {
                    ""
                };
                let payload = format!(
                    r#"{{"id":"chatcmpl-mainarch-demo","object":"chat.completion.chunk","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"delta":{{{role}"content":"{}"}},"finish_reason":null}}],"mainarch":{{"ok":true,"busy":false,"node":{node_id},"stream":true,"stream_timing":"scheduler_visible_decode_step","mode":"synthetic_decode_proof_stepwise","live_lane":"persistent-worker","model_runner":"cached-synthetic-proof","collective_policy":{collective_policy},"collective_gate":{collective_gate_json},"live_lane_request":{},"step_index":{},"token":{},"scheduler_tick":{},"scheduler_active":{},"step_gpu_us":{:.3},"step_gpu_us_per_layer":{:.3},"elapsed_ms":{:.3},"decode_us_token":{DECODE_US_TOKEN},"greedy_tokens":{TOKENS},"device_was_warm":{}}}}}"#,
                    json_escape(&event.text),
                    event.request_id,
                    event.step_index,
                    event.token,
                    event.scheduler_tick,
                    event.scheduler_active,
                    event.step_gpu_us,
                    event.step_gpu_us / 4.0,
                    event.elapsed_ms,
                    event.device_was_warm,
                );
                if let Err(err) = write_sse_data(stream, &payload) {
                    cancelled.store(true, Ordering::Release);
                    return Err(err);
                }
                let now = Instant::now();
                if first_chunk_us.is_none() {
                    let ttft_us = now.duration_since(stream_started).as_micros() as usize;
                    first_chunk_us = Some(ttft_us);
                    live_lane.record_stream_first_chunk(ttft_us);
                    live_lane.record_stream_token(None);
                } else {
                    let itl_us = last_chunk_at
                        .map(|last: Instant| now.duration_since(last).as_micros() as usize)
                        .unwrap_or(0);
                    live_lane.record_stream_token(Some(itl_us));
                }
                last_chunk_at = Some(now);
                token_chunks += 1;
            }

            match done.recv_timeout(Duration::from_secs(1)) {
                Ok(result) => {
                    if let Some(error) = result.error {
                        write_sse_data(
                            stream,
                            &format!(
                                r#"{{"error":{{"message":"{}","type":"mainarch_demo_error","code":"live_decode_failed"}},"model":"{MODEL_ID}","mainarch":{{"ok":false,"busy":false,"node":{node_id},"wall_ms":{:.3},"stream":true,"stream_timing":"scheduler_visible_decode_step","live_lane":"persistent-worker","collective_policy":{collective_policy},"collective_gate":{collective_gate_json},"cancelled":{},"live_lane_request":{},"generated_steps":{},"stopped_early":{},"stop_reason":{},"kv_blocks_leased":{},"kv_blocks_released":{}}}}}"#,
                                json_escape(&error),
                                result.wall_ms,
                                result.cancelled,
                                result.request_id,
                                result.generated_steps,
                                result.stopped_early,
                                json_option(&result.stop_reason),
                                result.kv_blocks_leased,
                                result.kv_blocks_released
                            ),
                        )?;
                    } else {
                        let ttft_ms = first_chunk_us.unwrap_or(0) as f64 / 1000.0;
                        write_sse_data(
                            stream,
                            &format!(
                                r#"{{"id":"chatcmpl-mainarch-demo","object":"chat.completion.chunk","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"mainarch":{{"ok":true,"busy":false,"node":{node_id},"wall_ms":{:.3},"stream_ttft_ms":{ttft_ms:.3},"stream_token_chunks":{token_chunks},"stream":true,"stream_timing":"scheduler_visible_decode_step","mode":"synthetic_decode_proof_stepwise","live_lane":"persistent-worker","model_runner":"cached-synthetic-proof","collective_policy":{collective_policy},"collective_gate":{collective_gate_json},"live_lane_request":{},"queue_wait_ms":{:.3},"queue_depth_on_enqueue":{},"gpu_us_per_token":{:.3},"generated_steps":{},"stopped_early":{},"stop_reason":{},"kv_blocks_leased":{},"kv_blocks_released":{},"kv_block_capacity":{LIVE_LANE_KV_BLOCK_CAPACITY},"kv_blocks_per_request":{PROOF_KV_LOGICAL_BLOCKS},"kv_block_table":{},"decode_us_token":{DECODE_US_TOKEN},"greedy_tokens":{TOKENS},"device_was_warm":{},"runner_was_warm":{},"cancelled":{}}}}}"#,
                                result.wall_ms,
                                result.request_id,
                                result.queue_wait_ms,
                                result.queue_depth_on_enqueue,
                                result.gpu_us_per_token.unwrap_or(0.0),
                                result.generated_steps,
                                result.stopped_early,
                                json_option(&result.stop_reason),
                                result.kv_blocks_leased,
                                result.kv_blocks_released,
                                json_u32_slice(&result.kv_block_table),
                                result.device_was_warm,
                                result.runner_was_warm,
                                result.cancelled
                            ),
                        )?;
                    }
                }
                Err(err) => {
                    write_sse_data(
                        stream,
                        &format!(
                            r#"{{"error":{{"message":"stream final result unavailable: {}","type":"mainarch_demo_error","code":"stream_final_result_missing"}},"model":"{MODEL_ID}","mainarch":{{"ok":false,"busy":false,"node":{node_id},"stream":true,"stream_timing":"scheduler_visible_decode_step","live_lane":"persistent-worker","collective_policy":{collective_policy},"collective_gate":{collective_gate_json}}}}}"#,
                            json_escape(&err.to_string())
                        ),
                    )?;
                }
            }
            write_sse_data(stream, "[DONE]")?;
        }
    }
    Ok(())
}

fn evidence_json() -> String {
    let splitk8_speedup = KIMI_SPLITK8_MLA_HOT_US / KIMI_MLA_HOT_US;
    let splitk8_uplift = (splitk8_speedup - 1.0) * 100.0;
    let presplit_speedup = KIMI_PRE_SPLIT_MLA_HOT_US / KIMI_MLA_HOT_US;
    let presplit_uplift = (presplit_speedup - 1.0) * 100.0;
    let collective_uplift = ((MAINARCH_TP8_GBPS / RCCL_TP8_GBPS) - 1.0) * 100.0;
    format!(
        r#"{{
  "model_shape": "Qwen3-235B-A22B-shaped synthetic decode proof",
  "hardware": "MI355X node 2",
  "runtime": "mainarch direct GPU path, no ROCm serving runtime",
  "target": "{LATEST_TARGET}",
  "latest_kernel_win": "Kimi MLA split-K64 hot runner",
  "latest_kernel_win_plain": "The latest banked kernel win cuts the MLA hot path to {KIMI_MLA_HOT_US:.3} us/case and the combined hot serving step to {KIMI_HOT_STEP_US_TOKEN:.3} us/token while keeping split count and retained partial workspace capacity in bounds.",
  "kimi_split_k": {KIMI_SPLIT_K},
  "kimi_mla_hot_us": {KIMI_MLA_HOT_US:.3},
  "kimi_hot_step_us_token": {KIMI_HOT_STEP_US_TOKEN:.3},
  "kimi_tokens_per_s_per_sequence": {KIMI_TOKENS_PER_S_PER_SEQUENCE:.3},
  "kimi_splitk8_mla_hot_us": {KIMI_SPLITK8_MLA_HOT_US:.3},
  "kimi_pre_split_mla_hot_us": {KIMI_PRE_SPLIT_MLA_HOT_US:.3},
  "kimi_splitk8_speedup": {splitk8_speedup:.4},
  "kimi_splitk8_uplift_percent": {splitk8_uplift:.1},
  "kimi_pre_split_speedup": {presplit_speedup:.4},
  "kimi_pre_split_uplift_percent": {presplit_uplift:.1},
  "mainarch_tp8_collective_gbps": {MAINARCH_TP8_GBPS:.3},
  "rccl_tp8_collective_gbps": {RCCL_TP8_GBPS:.5},
  "collective_uplift_percent": {collective_uplift:.1},
  "control_gbps": {CONTROL_GBPS},
  "mainarch_gbps": {MAINARCH_GBPS},
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "correctness": "tiled bitdiff 0; greedy tokens [46, 51, 90, 1]; max per-step logit rel-L2 5.89e-4",
  "status": "deployable one-page sandbox; live route uses synthetic proof today while real weights/tokenizer still need to land"
}}
"#
    )
}

fn comparison_json() -> String {
    let speedup = MAINARCH_GBPS as f64 / CONTROL_GBPS as f64;
    let uplift = (speedup - 1.0) * 100.0;
    let kimi_splitk8_speedup = KIMI_SPLITK8_MLA_HOT_US / KIMI_MLA_HOT_US;
    let kimi_splitk8_uplift = (kimi_splitk8_speedup - 1.0) * 100.0;
    let kimi_presplit_speedup = KIMI_PRE_SPLIT_MLA_HOT_US / KIMI_MLA_HOT_US;
    let kimi_presplit_uplift = (kimi_presplit_speedup - 1.0) * 100.0;
    let collective_speedup = MAINARCH_TP8_GBPS / RCCL_TP8_GBPS;
    let collective_uplift = (collective_speedup - 1.0) * 100.0;
    format!(
        r#"{{
  "ok": true,
  "target": "{LATEST_TARGET}",
  "baseline": "current ROCm/control path used by the banked proof",
  "candidate": "mainarch direct GPU path",
  "control_gbps": {CONTROL_GBPS},
  "mainarch_gbps": {MAINARCH_GBPS},
  "speedup": {speedup:.4},
  "uplift_percent": {uplift:.1},
  "kimi_baseline": "previous split-K8 hot runner",
  "kimi_candidate": "split-K64 hot runner",
  "kimi_split_k": {KIMI_SPLIT_K},
  "kimi_mla_hot_us": {KIMI_MLA_HOT_US:.3},
  "kimi_splitk8_mla_hot_us": {KIMI_SPLITK8_MLA_HOT_US:.3},
  "kimi_splitk8_speedup": {kimi_splitk8_speedup:.4},
  "kimi_splitk8_uplift_percent": {kimi_splitk8_uplift:.1},
  "kimi_pre_split_mla_hot_us": {KIMI_PRE_SPLIT_MLA_HOT_US:.3},
  "kimi_pre_split_speedup": {kimi_presplit_speedup:.4},
  "kimi_pre_split_uplift_percent": {kimi_presplit_uplift:.1},
  "mainarch_tp8_collective_gbps": {MAINARCH_TP8_GBPS:.3},
  "rccl_tp8_collective_gbps": {RCCL_TP8_GBPS:.5},
  "collective_speedup": {collective_speedup:.4},
  "collective_uplift_percent": {collective_uplift:.1},
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "status": "banked comparison evidence; live generation still uses synthetic decode proof and does not claim real Qwen/Kimi answer quality"
}}
"#
    )
}

fn metrics_text(live_lane: &LiveLane) -> String {
    let running = if live_lane.is_busy() { 1 } else { 0 };
    let waiting = live_lane.queue_depth().saturating_sub(running);
    let completed = live_lane.completed_count();
    let cancelled = live_lane.cancelled_count();
    let failed = live_lane.failed_count();
    let queue_full = live_lane.queue_full_count();
    let wall_us_total = live_lane.wall_us_total();
    let queue_wait_us_total = live_lane.queue_wait_us_total();
    let tpot_us_total = live_lane.tpot_us_total();
    let tpot_samples = live_lane.tpot_samples();
    let stream_requests = live_lane.stream_requests();
    let ttft_us_total = live_lane.ttft_us_total();
    let ttft_samples = live_lane.ttft_samples();
    let stream_tokens = live_lane.stream_tokens();
    let itl_us_total = live_lane.itl_us_total();
    let itl_samples = live_lane.itl_samples();
    let step_stops = live_lane.step_stops();
    let step_cancelled = live_lane.step_cancelled();
    let scheduler_decode_ticks = live_lane.scheduler_decode_ticks();
    let scheduler_active_max = live_lane.scheduler_active_max();
    let shared_runner_creations = live_lane.shared_runner_creations();
    let request_state_allocations = live_lane.request_state_allocations();
    let request_state_snapshots = live_lane.request_state_snapshots();
    let request_state_reused_snapshots = live_lane.request_state_reused_snapshots();
    let request_state_restores = live_lane.request_state_restores();
    let request_state_bytes_high_watermark = live_lane.request_state_bytes_high_watermark();
    let kv_blocks_in_use = live_lane.kv_blocks_in_use();
    let kv_leases_active = live_lane.kv_leases_active();
    let kv_blocks_high_watermark = live_lane.kv_blocks_high_watermark();
    let kv_leases_acquired = live_lane.kv_leases_acquired();
    let kv_leases_released = live_lane.kv_leases_released();
    let kv_lease_denied = live_lane.kv_lease_denied();
    let kv_block_table_installs = live_lane.kv_block_table_installs();
    let kv_prefill_page_writes = live_lane.kv_prefill_page_writes();
    let scheduler_kv_ownership_checks = live_lane.scheduler_kv_ownership_checks();
    let scheduler_kv_ownership_failures = live_lane.scheduler_kv_ownership_failures();
    let scheduler_kv_active_blocks_checked = live_lane.scheduler_kv_active_blocks_checked();
    format!(
        "# HELP mainarch_num_requests_waiting Requests waiting in the demo FIFO queue.\n\
# TYPE mainarch_num_requests_waiting gauge\n\
mainarch_num_requests_waiting {waiting}\n\
# HELP mainarch_num_requests_running Requests currently executing on the demo live lane.\n\
# TYPE mainarch_num_requests_running gauge\n\
mainarch_num_requests_running {running}\n\
# HELP mainarch_queue_capacity Maximum queued live requests accepted by demo-serve.\n\
# TYPE mainarch_queue_capacity gauge\n\
mainarch_queue_capacity {LIVE_LANE_QUEUE_LIMIT}\n\
# HELP mainarch_requests_completed_total Successfully completed live requests.\n\
# TYPE mainarch_requests_completed_total counter\n\
mainarch_requests_completed_total {completed}\n\
# HELP mainarch_requests_cancelled_total Queued live requests cancelled before execution.\n\
# TYPE mainarch_requests_cancelled_total counter\n\
mainarch_requests_cancelled_total {cancelled}\n\
# HELP mainarch_requests_failed_total Live requests that reached the worker and returned an error.\n\
# TYPE mainarch_requests_failed_total counter\n\
mainarch_requests_failed_total {failed}\n\
# HELP mainarch_requests_queue_full_total Live requests rejected because the FIFO was full.\n\
# TYPE mainarch_requests_queue_full_total counter\n\
mainarch_requests_queue_full_total {queue_full}\n\
# HELP mainarch_request_wall_time_us_total Sum of successful live request worker wall time in microseconds.\n\
# TYPE mainarch_request_wall_time_us_total counter\n\
mainarch_request_wall_time_us_total {wall_us_total}\n\
# HELP mainarch_queue_wait_time_us_total Sum of live request FIFO wait time in microseconds.\n\
# TYPE mainarch_queue_wait_time_us_total counter\n\
mainarch_queue_wait_time_us_total {queue_wait_us_total}\n\
# HELP mainarch_time_per_output_token_us_total Sum of measured cached-runner GPU microseconds per generated token.\n\
# TYPE mainarch_time_per_output_token_us_total counter\n\
mainarch_time_per_output_token_us_total {tpot_us_total}\n\
# HELP mainarch_time_per_output_token_samples_total Number of cached-runner token-latency samples.\n\
# TYPE mainarch_time_per_output_token_samples_total counter\n\
mainarch_time_per_output_token_samples_total {tpot_samples}\n\
# HELP mainarch_stream_requests_total OpenAI stream=true requests that emitted a stream chunk.\n\
# TYPE mainarch_stream_requests_total counter\n\
mainarch_stream_requests_total {stream_requests}\n\
# HELP mainarch_time_to_first_token_us_total Sum of stream first assistant chunk latency in microseconds.\n\
# TYPE mainarch_time_to_first_token_us_total counter\n\
mainarch_time_to_first_token_us_total {ttft_us_total}\n\
# HELP mainarch_time_to_first_token_samples_total Number of stream first assistant chunk latency samples.\n\
# TYPE mainarch_time_to_first_token_samples_total counter\n\
mainarch_time_to_first_token_samples_total {ttft_samples}\n\
# HELP mainarch_stream_tokens_total Streamed scheduler-visible decode token chunks.\n\
# TYPE mainarch_stream_tokens_total counter\n\
mainarch_stream_tokens_total {stream_tokens}\n\
# HELP mainarch_inter_token_latency_us_total Sum of stream inter-token latency in microseconds.\n\
# TYPE mainarch_inter_token_latency_us_total counter\n\
mainarch_inter_token_latency_us_total {itl_us_total}\n\
# HELP mainarch_inter_token_latency_samples_total Number of stream inter-token latency samples.\n\
# TYPE mainarch_inter_token_latency_samples_total counter\n\
mainarch_inter_token_latency_samples_total {itl_samples}\n\
# HELP mainarch_decode_step_stops_total Requests stopped at a scheduler-visible decode-step boundary.\n\
# TYPE mainarch_decode_step_stops_total counter\n\
mainarch_decode_step_stops_total {step_stops}\n\
# HELP mainarch_decode_step_cancelled_total Requests cancelled at a scheduler-visible decode-step boundary.\n\
# TYPE mainarch_decode_step_cancelled_total counter\n\
mainarch_decode_step_cancelled_total {step_cancelled}\n\
# HELP mainarch_scheduler_decode_ticks_total Demo scheduler decode ticks that processed at least one active request.\n\
# TYPE mainarch_scheduler_decode_ticks_total counter\n\
mainarch_scheduler_decode_ticks_total {scheduler_decode_ticks}\n\
# HELP mainarch_scheduler_active_requests_high_watermark Highest active request count seen by a demo scheduler tick.\n\
# TYPE mainarch_scheduler_active_requests_high_watermark gauge\n\
mainarch_scheduler_active_requests_high_watermark {scheduler_active_max}\n\
# HELP mainarch_shared_runner_creations_total Cached synthetic proof runners created by the demo worker.\n\
# TYPE mainarch_shared_runner_creations_total counter\n\
mainarch_shared_runner_creations_total {shared_runner_creations}\n\
# HELP mainarch_request_state_allocations_total Per-request synthetic decode resume snapshots allocated at admission; scalar metadata, block table, and token history only.\n\
# TYPE mainarch_request_state_allocations_total counter\n\
mainarch_request_state_allocations_total {request_state_allocations}\n\
# HELP mainarch_request_state_snapshots_total Per-request synthetic decode resume snapshots captured by the shared runner.\n\
# TYPE mainarch_request_state_snapshots_total counter\n\
mainarch_request_state_snapshots_total {request_state_snapshots}\n\
# HELP mainarch_request_state_reused_snapshots_total Per-request synthetic decode resume snapshots refreshed in-place without allocating a new state object.\n\
# TYPE mainarch_request_state_reused_snapshots_total counter\n\
mainarch_request_state_reused_snapshots_total {request_state_reused_snapshots}\n\
# HELP mainarch_request_state_restores_total Per-request synthetic decode resume snapshots restored into the shared runner.\n\
# TYPE mainarch_request_state_restores_total counter\n\
mainarch_request_state_restores_total {request_state_restores}\n\
# HELP mainarch_request_state_bytes_high_watermark Largest per-request synthetic decode resume snapshot in bytes; scalar metadata plus block table and token history only.\n\
# TYPE mainarch_request_state_bytes_high_watermark gauge\n\
mainarch_request_state_bytes_high_watermark {request_state_bytes_high_watermark}\n\
# HELP mainarch_kv_block_capacity Synthetic proof KV block lease capacity for the demo live lane.\n\
# TYPE mainarch_kv_block_capacity gauge\n\
mainarch_kv_block_capacity {LIVE_LANE_KV_BLOCK_CAPACITY}\n\
# HELP mainarch_kv_blocks_per_request Synthetic proof KV blocks leased by each executing request.\n\
# TYPE mainarch_kv_blocks_per_request gauge\n\
mainarch_kv_blocks_per_request {PROOF_KV_LOGICAL_BLOCKS}\n\
# HELP mainarch_kv_block_size_tokens Synthetic proof KV block size in tokens.\n\
# TYPE mainarch_kv_block_size_tokens gauge\n\
mainarch_kv_block_size_tokens {PROOF_KV_BLOCK_SIZE_TOKENS}\n\
# HELP mainarch_kv_blocks_in_use Synthetic proof KV blocks currently leased by executing requests.\n\
# TYPE mainarch_kv_blocks_in_use gauge\n\
mainarch_kv_blocks_in_use {kv_blocks_in_use}\n\
# HELP mainarch_kv_leases_active Synthetic proof KV leases currently active.\n\
# TYPE mainarch_kv_leases_active gauge\n\
mainarch_kv_leases_active {kv_leases_active}\n\
# HELP mainarch_kv_blocks_high_watermark Highest synthetic proof KV blocks concurrently leased.\n\
# TYPE mainarch_kv_blocks_high_watermark gauge\n\
mainarch_kv_blocks_high_watermark {kv_blocks_high_watermark}\n\
# HELP mainarch_kv_leases_acquired_total Synthetic proof KV leases acquired before live decode.\n\
# TYPE mainarch_kv_leases_acquired_total counter\n\
mainarch_kv_leases_acquired_total {kv_leases_acquired}\n\
# HELP mainarch_kv_leases_released_total Synthetic proof KV leases released after live decode exits.\n\
# TYPE mainarch_kv_leases_released_total counter\n\
mainarch_kv_leases_released_total {kv_leases_released}\n\
# HELP mainarch_kv_lease_denied_total Synthetic proof KV lease requests denied because capacity was exhausted.\n\
# TYPE mainarch_kv_lease_denied_total counter\n\
mainarch_kv_lease_denied_total {kv_lease_denied}\n\
# HELP mainarch_kv_block_table_installs_total Scheduler-owned KV block tables installed into the device-side proof runner.\n\
# TYPE mainarch_kv_block_table_installs_total counter\n\
mainarch_kv_block_table_installs_total {kv_block_table_installs}\n\
# HELP mainarch_kv_prefill_page_writes_total Synthetic prefill KV pages initialized directly into scheduler-owned physical blocks.\n\
# TYPE mainarch_kv_prefill_page_writes_total counter\n\
mainarch_kv_prefill_page_writes_total {kv_prefill_page_writes}\n\
# HELP mainarch_scheduler_kv_ownership_checks_total Active KV ownership validations completed before decode dispatch.\n\
# TYPE mainarch_scheduler_kv_ownership_checks_total counter\n\
mainarch_scheduler_kv_ownership_checks_total {scheduler_kv_ownership_checks}\n\
# HELP mainarch_scheduler_kv_ownership_failures_total Active KV ownership validations that failed closed before decode dispatch.\n\
# TYPE mainarch_scheduler_kv_ownership_failures_total counter\n\
mainarch_scheduler_kv_ownership_failures_total {scheduler_kv_ownership_failures}\n\
# HELP mainarch_scheduler_kv_active_blocks_checked_total Active KV physical block references checked before decode dispatch.\n\
# TYPE mainarch_scheduler_kv_active_blocks_checked_total counter\n\
mainarch_scheduler_kv_active_blocks_checked_total {scheduler_kv_active_blocks_checked}\n"
    )
}

fn models_json() -> String {
    let collective_policy = collective_policy_json();
    format!(
        r#"{{
  "object": "list",
  "data": [
    {{
      "id": "{MODEL_ID}",
      "object": "model",
      "created": 0,
      "owned_by": "mainarch",
      "mainarch": {{
        "mode": "synthetic_decode_proof",
        "target": "{LATEST_TARGET}",
        "latest_kernel_win": "Kimi MLA split-K64 hot runner",
        "kimi_split_k": {KIMI_SPLIT_K},
        "kimi_mla_hot_us": {KIMI_MLA_HOT_US:.3},
        "kimi_hot_step_us_token": {KIMI_HOT_STEP_US_TOKEN:.3},
        "decode_us_token": {DECODE_US_TOKEN},
        "greedy_tokens": {TOKENS},
        "chat_endpoint": "/v1/chat/completions",
        "streaming": true,
        "collective_policy": {collective_policy}
      }}
    }}
  ]
}}
"#
    )
}

fn health_json(live_lane: &LiveLane) -> String {
    let node_id = live_lane.node_id();
    let busy = live_lane.is_busy();
    let queue_depth = live_lane.queue_depth();
    let completed = live_lane.completed_count();
    let cancelled = live_lane.cancelled_count();
    let kv_blocks_in_use = live_lane.kv_blocks_in_use();
    let kv_leases_active = live_lane.kv_leases_active();
    let collective_policy = collective_policy_json();
    format!(
        r#"{{
  "ok": true,
  "service": "mainarch-demo",
  "demo_app_version": "{DEMO_APP_VERSION}",
  "demo_app": "one-page-public-sandbox",
  "node": {node_id},
  "busy": {busy},
  "queue_depth": {queue_depth},
  "queue_capacity": {LIVE_LANE_QUEUE_LIMIT},
  "completed_requests": {completed},
  "cancelled_requests": {cancelled},
  "kv_block_capacity": {LIVE_LANE_KV_BLOCK_CAPACITY},
  "kv_blocks_per_request": {PROOF_KV_LOGICAL_BLOCKS},
  "kv_block_size_tokens": {PROOF_KV_BLOCK_SIZE_TOKENS},
  "kv_blocks_in_use": {kv_blocks_in_use},
  "kv_leases_active": {kv_leases_active},
  "live_lane": "persistent-worker",
  "model_runner": "cached-synthetic-proof",
  "evidence_target": "{LATEST_TARGET}",
  "live_endpoint": "/api/generate",
  "chat_endpoint": "/v1/chat/completions",
  "models_endpoint": "/v1/models",
  "chat_streaming": true,
  "collective_policy": {collective_policy},
  "status": "{}"
}}
"#,
        if busy { "live proof running" } else { "ready" }
    )
}

fn demo_manifest_json(live_lane: &LiveLane) -> String {
    let node_id = live_lane.node_id();
    let busy = live_lane.is_busy();
    let queue_depth = live_lane.queue_depth();
    let completed = live_lane.completed_count();
    let cancelled = live_lane.cancelled_count();
    let kv_blocks_in_use = live_lane.kv_blocks_in_use();
    let kv_leases_active = live_lane.kv_leases_active();
    let collective_policy = collective_policy_json();
    format!(
        r#"{{
  "ok": true,
  "service": "mainarch-demo",
  "demo_app_version": "{DEMO_APP_VERSION}",
  "demo_app": "one-page-public-sandbox",
  "node": {node_id},
  "busy": {busy},
  "queue_depth": {queue_depth},
  "queue_capacity": {LIVE_LANE_QUEUE_LIMIT},
  "completed_requests": {completed},
  "cancelled_requests": {cancelled},
  "kv_block_capacity": {LIVE_LANE_KV_BLOCK_CAPACITY},
  "kv_blocks_per_request": {PROOF_KV_LOGICAL_BLOCKS},
  "kv_block_size_tokens": {PROOF_KV_BLOCK_SIZE_TOKENS},
  "kv_blocks_in_use": {kv_blocks_in_use},
  "kv_leases_active": {kv_leases_active},
  "model": "{MODEL_ID}",
  "mode": "synthetic_decode_proof",
  "latest_kernel_win": "Kimi MLA split-K64 hot runner",
  "latest_kernel_win_plain": "MLA hot path: {KIMI_MLA_HOT_US:.3} us/case; combined hot serving step: {KIMI_HOT_STEP_US_TOKEN:.3} us/token; mainarch TP8 collective: {MAINARCH_TP8_GBPS:.2} GB/s vs RCCL {RCCL_TP8_GBPS:.2} GB/s.",
  "kimi_split_k": {KIMI_SPLIT_K},
  "kimi_mla_hot_us": {KIMI_MLA_HOT_US:.3},
  "kimi_hot_step_us_token": {KIMI_HOT_STEP_US_TOKEN:.3},
  "kimi_tokens_per_s_per_sequence": {KIMI_TOKENS_PER_S_PER_SEQUENCE:.3},
  "collective_policy": {collective_policy},
  "live_lane": "persistent-worker",
  "model_runner": "cached-synthetic-proof",
  "headline": "A shareable one-page sandbox for the direct MI355X serving path.",
  "public_demo_ready": true,
  "viewer_promise": "Ask a normal prompt, watch a streamed answer, and see exactly which serving pieces are real today before Qwen/Kimi weights replace the synthetic proof backend.",
  "demo_phase": "one-page product demo over the single-binary synthetic proof",
  "primary_demo": "interactive stream:true prompt over /v1/chat/completions",
  "burst_demo": "four capped stream:true requests to expose scheduler ticks and KV block ownership",
  "future_boundary": "keep this browser/API seam stable while the backend swaps synthetic proof work for tokenizer-backed Qwen/Kimi decode, TP8 collectives, quotas, and abuse controls",
  "what_is_real": [
    "direct mainarch GPU decode proof on MI355X",
    "OpenAI-shaped model discovery and chat completion routes",
    "banked comparison evidence for the current fast path",
    "one persistent worker owns the KFD/GPU live lane across requests",
    "synthetic proof weights, FP4 KV, metadata, and scratch buffers are cached behind that worker",
    "each admitted request reserves a fixed synthetic KV block table before it reaches the worker",
    "concurrent stream requests are interleaved at one verified decode step per scheduler tick",
    "one cached synthetic proof runner is shared by active requests through explicit per-request resume snapshots",
    "per-request resume snapshots are scalar metadata plus block table and token history; scheduler-owned KV pages and per-step trace buffers remain resident in runner-owned device memory",
    "scheduler-owned KV block IDs are installed into the proof runner's device-side paged block table",
    "synthetic prefill KV is initialized directly into scheduler-owned physical pages at admission"
  ],
  "what_is_not_claimed": [
    "real Qwen or Kimi weights",
    "tokenizer-backed prompt generation",
    "public sampling or full mixed prefill/decode continuous batching"
  ],
  "swap_points": [
    "replace synthetic decode proof with real-weight megakernel decode",
    "wire tokenizer and prompt scheduler behind /v1/chat/completions",
    "move remaining block-table and token-history resume snapshots onto device-side per-request pages",
    "replace synthetic host-seeded prefill page initialization with real prefill kernels writing into scheduler-owned physical pages",
    "add token-budgeted chunked prefill to the decode scheduler",
    "promote /api/evidence into live benchmark telemetry"
  ],
  "endpoints": {{
    "ui": "/",
    "health": "/api/health",
    "manifest": "/api/demo",
    "evidence": "/api/evidence",
    "comparison": "/api/compare",
    "models": "/v1/models",
    "chat": "/v1/chat/completions"
  }}
}}
"#
    )
}

fn collective_policy_json() -> String {
    let policy = mcore::multigpu::select_serving_allreduce_policy(
        mcore::multigpu::ServingAllReducePhase::Decode,
        COLLECTIVE_GATE_LOGICAL_TP_RANKS,
        COLLECTIVE_PROOF_PAYLOAD_BYTES,
        COLLECTIVE_PROOF_PAYLOAD_BYTES,
    );
    let selected_backend = json_escape(&policy.backend_name());
    let serving_note = json_escape(&policy.serving_note());
    format!(
        r#"{{
    "state": "{COLLECTIVE_POLICY_STATE}",
    "selected_backend": "{selected_backend}",
    "selected_path": "{:?}",
    "logical_tp_ranks": {COLLECTIVE_GATE_LOGICAL_TP_RANKS},
    "active_ranks": {COLLECTIVE_GATE_ACTIVE_RANKS},
    "proof_payload_bytes": {COLLECTIVE_PROOF_PAYLOAD_BYTES},
    "dda_min_bytes": {},
    "dda_max_bytes": {},
    "raw_xgmi_max_bytes": {},
    "crossover_bytes": {},
    "large_unfused_bytes": {COLLECTIVE_LARGE_UNFUSED_BYTES},
    "large_unfused_policy": "prefer-rccl-or-quickreduce-style-path",
    "rank_uniform_gate": "required-before-multirank-collective",
    "operational_telemetry": "/api/health,/api/demo,/metrics",
    "serving_note": "{serving_note}",
    "source": "mainarch-core::select_serving_allreduce_policy"
  }}"#,
        policy.path,
        policy.dda_min_bytes,
        policy.dda_max_bytes,
        policy.dda_max_bytes,
        policy.crossover_bytes
    )
}

fn collective_entry_gate(prompt: &str, max_steps: usize) -> CollectiveGateDecision {
    let local_ready = max_steps > 0;
    let mut local_metadata_hash = 0xcbf29ce484222325u64;
    for b in prompt.as_bytes() {
        local_metadata_hash ^= *b as u64;
        local_metadata_hash = local_metadata_hash.wrapping_mul(0x100000001b3);
    }
    local_metadata_hash ^= max_steps as u64;
    local_metadata_hash = local_metadata_hash.wrapping_mul(0x100000001b3);
    local_metadata_hash ^= COLLECTIVE_PROOF_PAYLOAD_BYTES as u64;
    local_metadata_hash = local_metadata_hash.wrapping_mul(0x100000001b3);

    let injected_divergence = std::env::var_os(COLLECTIVE_GATE_INJECT_DIVERGENCE_ENV).is_some();
    let expected_ready = local_ready;
    let divergent_rank = if injected_divergence {
        Some(COLLECTIVE_GATE_LOGICAL_TP_RANKS.saturating_sub(1))
    } else {
        None
    };
    let expected_metadata_hash = local_metadata_hash;
    let mut rank_uniform = local_ready && expected_ready && COLLECTIVE_GATE_ACTIVE_RANKS == 1;
    for rank in 0..COLLECTIVE_GATE_LOGICAL_TP_RANKS {
        let rank_hash = if divergent_rank == Some(rank) {
            local_metadata_hash ^ 0x9e3779b97f4a7c15u64
        } else {
            local_metadata_hash
        };
        rank_uniform &= rank_hash == expected_metadata_hash;
    }
    let failure_reason = if !local_ready {
        Some("local-rank-not-ready")
    } else if !expected_ready {
        Some("scheduler-rank-not-ready")
    } else if divergent_rank.is_some() {
        Some("logical-tp8-metadata-hash-mismatch")
    } else if COLLECTIVE_GATE_ACTIVE_RANKS != 1 {
        Some("unsupported-active-rank-count")
    } else {
        None
    };

    CollectiveGateDecision {
        state: "logical_tp8_single_rank_uniform_proof",
        tp_ranks: COLLECTIVE_GATE_LOGICAL_TP_RANKS,
        active_ranks: COLLECTIVE_GATE_ACTIVE_RANKS,
        local_ready,
        expected_ready,
        local_metadata_hash,
        expected_metadata_hash,
        rank_uniform,
        enter_collective: rank_uniform,
        injected_divergence,
        divergent_rank,
        failure_reason,
    }
}

fn collective_gate_json(gate: &CollectiveGateDecision) -> String {
    format!(
        r#"{{
    "state": "{}",
    "tp_ranks": {},
    "active_ranks": {},
    "local_ready": {},
    "expected_ready": {},
    "local_metadata_hash": "0x{:016x}",
    "expected_metadata_hash": "0x{:016x}",
    "rank_uniform": {},
    "enter_collective": {},
    "injected_divergence": {},
    "divergent_rank": {},
    "failure_reason": {}
  }}"#,
        gate.state,
        gate.tp_ranks,
        gate.active_ranks,
        gate.local_ready,
        gate.expected_ready,
        gate.local_metadata_hash,
        gate.expected_metadata_hash,
        gate.rank_uniform,
        gate.enter_collective,
        gate.injected_divergence,
        json_option(&gate.divergent_rank.map(|rank| rank.to_string())),
        json_option(&gate.failure_reason.map(|s| s.to_string()))
    )
}

fn run_live_decode_json(live_lane: &LiveLane, cancelled: Arc<AtomicBool>) -> String {
    let node_id = live_lane.node_id();
    match live_lane.run(cancelled) {
        LiveRun::QueueFull { queued, capacity } => format!(
            r#"{{
  "ok": false,
  "busy": false,
  "node": {node_id},
  "queue_depth": {queued},
  "queue_capacity": {capacity},
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "error": "live decode proof queue is full"
}}
"#
        ),
        LiveRun::KvUnavailable {
            requested,
            capacity,
        } => format!(
            r#"{{
  "ok": false,
  "busy": true,
  "node": {node_id},
  "requested_kv_blocks": {requested},
  "kv_block_capacity": {capacity},
  "kv_blocks_per_request": {PROOF_KV_LOGICAL_BLOCKS},
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "error": "live decode proof KV block capacity is full"
}}
"#
        ),
        LiveRun::Completed(result) => match result.error {
            None => format!(
                r#"{{
  "ok": true,
  "busy": false,
  "node": {node_id},
  "wall_ms": {:.3},
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "live_lane": "persistent-worker",
  "model_runner": "cached-synthetic-proof",
  "collective_policy": {},
  "device_was_warm": {},
  "runner_was_warm": {},
  "queue_wait_ms": {:.3},
  "queue_depth_on_enqueue": {},
  "cancelled": {},
  "gpu_us_per_token": {:.3},
  "live_lane_request": {},
  "message": "live synthetic decode proof completed on the demo server"
}}
"#,
                result.wall_ms,
                collective_policy_json(),
                result.device_was_warm,
                result.runner_was_warm,
                result.queue_wait_ms,
                result.queue_depth_on_enqueue,
                result.cancelled,
                result.gpu_us_per_token.unwrap_or(f64::NAN),
                result.request_id
            ),
            Some(error) => format!(
                r#"{{
  "ok": false,
  "busy": false,
  "node": {node_id},
  "wall_ms": {:.3},
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "live_lane": "persistent-worker",
  "model_runner": "cached-synthetic-proof",
  "collective_policy": {},
  "device_was_warm": {},
  "runner_was_warm": {},
  "queue_wait_ms": {:.3},
  "queue_depth_on_enqueue": {},
  "cancelled": {},
  "live_lane_request": {},
  "error": "{}"
}}
"#,
                result.wall_ms,
                collective_policy_json(),
                result.device_was_warm,
                result.runner_was_warm,
                result.queue_wait_ms,
                result.queue_depth_on_enqueue,
                result.cancelled,
                result.request_id,
                json_escape(&error)
            ),
        },
    }
}

fn run_prompt_generate_json(
    live_lane: &LiveLane,
    prompt: &str,
    cancelled: Arc<AtomicBool>,
) -> String {
    let node_id = live_lane.node_id();
    let prompt = if prompt.is_empty() {
        "What makes mainarch different?"
    } else {
        prompt
    };
    match live_lane.run(cancelled) {
        LiveRun::QueueFull { queued, capacity } => format!(
            r#"{{
  "ok": false,
  "busy": false,
  "node": {node_id},
  "queue_depth": {queued},
  "queue_capacity": {capacity},
  "prompt": "{}",
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "error": "live decode proof queue is full"
}}
"#,
            json_escape(prompt)
        ),
        LiveRun::KvUnavailable {
            requested,
            capacity,
        } => format!(
            r#"{{
  "ok": false,
  "busy": true,
  "node": {node_id},
  "requested_kv_blocks": {requested},
  "kv_block_capacity": {capacity},
  "kv_blocks_per_request": {PROOF_KV_LOGICAL_BLOCKS},
  "prompt": "{}",
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "error": "live decode proof KV block capacity is full"
}}
"#,
            json_escape(prompt)
        ),
        LiveRun::Completed(result) => match result.error {
            None => format!(
                r#"{{
  "ok": true,
  "busy": false,
  "node": {node_id},
  "wall_ms": {:.3},
  "prompt": "{}",
  "response": "Synthetic decode proof completed. Real tokenizer and Qwen/Kimi weights are the next serving layer.",
  "mode": "synthetic_decode_proof",
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "live_lane": "persistent-worker",
  "model_runner": "cached-synthetic-proof",
  "collective_policy": {},
  "device_was_warm": {},
  "runner_was_warm": {},
  "queue_wait_ms": {:.3},
  "queue_depth_on_enqueue": {},
  "cancelled": {},
  "gpu_us_per_token": {:.3},
  "live_lane_request": {}
}}
"#,
                result.wall_ms,
                json_escape(prompt),
                collective_policy_json(),
                result.device_was_warm,
                result.runner_was_warm,
                result.queue_wait_ms,
                result.queue_depth_on_enqueue,
                result.cancelled,
                result.gpu_us_per_token.unwrap_or(f64::NAN),
                result.request_id
            ),
            Some(error) => format!(
                r#"{{
  "ok": false,
  "busy": false,
  "node": {node_id},
  "wall_ms": {:.3},
  "prompt": "{}",
  "mode": "synthetic_decode_proof",
  "decode_us_token": {DECODE_US_TOKEN},
  "greedy_tokens": {TOKENS},
  "live_lane": "persistent-worker",
  "model_runner": "cached-synthetic-proof",
  "collective_policy": {},
  "device_was_warm": {},
  "runner_was_warm": {},
  "queue_wait_ms": {:.3},
  "queue_depth_on_enqueue": {},
  "cancelled": {},
  "live_lane_request": {},
  "error": "{}"
}}
"#,
                result.wall_ms,
                json_escape(prompt),
                collective_policy_json(),
                result.device_was_warm,
                result.runner_was_warm,
                result.queue_wait_ms,
                result.queue_depth_on_enqueue,
                result.cancelled,
                result.request_id,
                json_escape(&error)
            ),
        },
    }
}

fn run_chat_completions_json(
    live_lane: &LiveLane,
    body: &str,
    streaming: bool,
    cancelled: Arc<AtomicBool>,
) -> String {
    let node_id = live_lane.node_id();
    let prompt = extract_chat_prompt(body);
    let max_steps = extract_max_completion_steps(body).unwrap_or(PROOF_STREAM_TOKENS);
    let collective_gate = collective_entry_gate(&prompt, max_steps);
    let collective_gate_json = collective_gate_json(&collective_gate);
    if !collective_gate.enter_collective {
        return format!(
            r#"{{
  "error": {{
    "message": "rank-uniform collective gate rejected the request before decode",
    "type": "mainarch_collective_gate_error",
    "code": "rank_uniform_collective_gate_failed"
  }},
  "model": "{MODEL_ID}",
  "mainarch": {{
    "ok": false,
    "busy": false,
    "node": {node_id},
    "mode": "synthetic_decode_proof",
    "live_lane": "persistent-worker",
    "model_runner": "cached-synthetic-proof",
    "collective_policy": {},
    "collective_gate": {collective_gate_json},
    "decode_us_token": {DECODE_US_TOKEN},
    "greedy_tokens": {TOKENS},
    "stream_requested": {streaming}
  }}
}}
"#,
            collective_policy_json()
        );
    }
    match live_lane.run(cancelled) {
        LiveRun::QueueFull { queued, capacity } => {
            if streaming {
                return chat_completion_error_stream_json(
                    node_id,
                    "live decode proof queue is full",
                    "server_queue_full",
                    "mainarch_live_lane_queue_full",
                    false,
                    0.0,
                );
            }
            format!(
                r#"{{
  "error": {{
    "message": "live decode proof queue is full",
    "type": "server_queue_full",
    "code": "mainarch_live_lane_queue_full"
  }},
  "model": "{MODEL_ID}",
  "mainarch": {{
    "busy": true,
    "node": {node_id},
    "queue_depth": {queued},
    "queue_capacity": {capacity},
    "live_lane": "persistent-worker",
    "model_runner": "cached-synthetic-proof",
    "decode_us_token": {DECODE_US_TOKEN},
    "greedy_tokens": {TOKENS}
  }}
}}
"#
            )
        }
        LiveRun::KvUnavailable {
            requested,
            capacity,
        } => {
            if streaming {
                return chat_completion_error_stream_json(
                    node_id,
                    "live decode proof KV block capacity is full",
                    "server_kv_full",
                    "mainarch_live_lane_kv_full",
                    true,
                    0.0,
                );
            }
            format!(
                r#"{{
  "error": {{
    "message": "live decode proof KV block capacity is full",
    "type": "server_kv_full",
    "code": "mainarch_live_lane_kv_full"
  }},
  "model": "{MODEL_ID}",
  "mainarch": {{
    "busy": true,
    "node": {node_id},
    "requested_kv_blocks": {requested},
    "kv_block_capacity": {capacity},
    "kv_blocks_per_request": {PROOF_KV_LOGICAL_BLOCKS},
    "live_lane": "persistent-worker",
    "model_runner": "cached-synthetic-proof",
    "decode_us_token": {DECODE_US_TOKEN},
    "greedy_tokens": {TOKENS}
  }}
}}
"#
            )
        }
        LiveRun::Completed(result) => match result.error {
            None => {
                let collective_policy = collective_policy_json();
                let stream_ttft_us = if streaming {
                    let ttft_us = ((result.queue_wait_ms + result.wall_ms) * 1000.0)
                        .max(0.0)
                        .round() as usize;
                    live_lane.record_stream_first_chunk(ttft_us);
                    Some(ttft_us)
                } else {
                    None
                };
                let answer = format!(
                    "The sandbox ran request {} through the cached MI355X model-runner lane in {:.1} ms. The words are synthetic today, but the prompt entered the OpenAI-shaped serving route that real Qwen/Kimi decode will keep. Prompt received: {prompt}.",
                    result.request_id,
                    result.wall_ms,
                );
                if streaming {
                    return chat_completion_stream_json(
                        node_id,
                        result.wall_ms,
                        &answer,
                        result.device_was_warm,
                        result.request_id,
                        stream_ttft_us.unwrap_or(0) as f64 / 1000.0,
                    );
                }
                format!(
                    r#"{{
  "id": "chatcmpl-mainarch-demo",
  "object": "chat.completion",
  "created": 0,
  "model": "{MODEL_ID}",
  "choices": [
    {{
      "index": 0,
      "message": {{
        "role": "assistant",
        "content": "{}"
      }},
      "finish_reason": "stop"
    }}
  ],
  "usage": {{
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }},
  "mainarch": {{
    "ok": true,
    "busy": false,
    "node": {node_id},
    "wall_ms": {:.3},
    "decode_us_token": {DECODE_US_TOKEN},
    "greedy_tokens": {TOKENS},
    "mode": "synthetic_decode_proof",
    "live_lane": "persistent-worker",
    "model_runner": "cached-synthetic-proof",
    "collective_policy": {collective_policy},
    "collective_gate": {collective_gate_json},
    "device_was_warm": {},
    "runner_was_warm": {},
    "queue_wait_ms": {:.3},
    "queue_depth_on_enqueue": {},
    "cancelled": {},
    "gpu_us_per_token": {:.3},
    "live_lane_request": {}
  }}
}}
"#,
                    json_escape(&answer),
                    result.wall_ms,
                    result.device_was_warm,
                    result.runner_was_warm,
                    result.queue_wait_ms,
                    result.queue_depth_on_enqueue,
                    result.cancelled,
                    result.gpu_us_per_token.unwrap_or(f64::NAN),
                    result.request_id
                )
            }
            Some(error) => {
                if streaming {
                    return chat_completion_error_stream_json(
                        node_id,
                        &error,
                        "mainarch_demo_error",
                        "live_decode_failed",
                        false,
                        result.wall_ms,
                    );
                }
                format!(
                    r#"{{
  "error": {{
    "message": "{}",
    "type": "mainarch_demo_error",
    "code": "live_decode_failed"
  }},
  "model": "{MODEL_ID}",
  "mainarch": {{
    "ok": false,
    "busy": false,
    "node": {node_id},
    "wall_ms": {:.3},
    "decode_us_token": {DECODE_US_TOKEN},
    "greedy_tokens": {TOKENS},
    "live_lane": "persistent-worker",
    "model_runner": "cached-synthetic-proof",
    "device_was_warm": {},
    "runner_was_warm": {},
    "queue_wait_ms": {:.3},
    "queue_depth_on_enqueue": {},
    "cancelled": {},
    "live_lane_request": {}
  }}
}}
"#,
                    json_escape(&error),
                    result.wall_ms,
                    result.device_was_warm,
                    result.runner_was_warm,
                    result.queue_wait_ms,
                    result.queue_depth_on_enqueue,
                    result.cancelled,
                    result.request_id
                )
            }
        },
    }
}

fn body_requests_stream(body: &str) -> bool {
    let compact: String = body
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect();
    compact.contains("\"stream\":true")
}

fn extract_max_completion_steps(body: &str) -> Option<usize> {
    extract_json_usize_after_key(body, "max_completion_tokens")
        .or_else(|| extract_json_usize_after_key(body, "max_tokens"))
        .map(|steps| steps.min(PROOF_STREAM_TOKENS))
}

fn extract_json_usize_after_key(src: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{key}\"");
    let pos = src.find(&needle)?;
    let rest = &src[pos + needle.len()..];
    let colon = rest.find(':')?;
    let mut digits = String::new();
    for ch in rest[colon + 1..].trim_start().chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            continue;
        }
        break;
    }
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn stream_token_text(request_id: u64, prompt: &str, step_index: usize, token: u32) -> String {
    let prompt_preview: String = prompt.chars().take(160).collect();
    match step_index {
        0 => format!(
            "Request {request_id} reached the public chat route and is streaming from the cached MI355X proof lane. "
        ),
        1 => format!(
            "Synthetic proof token {token} verified; the worker returned control after this token so the scheduler, queue, and KV lease are visible. "
        ),
        2 => format!("Prompt preview: {prompt_preview}. The answer text is synthetic, but the stream, timing, and live-lane accounting are real. "),
        _ => "Next swap point: real tokenizer, Qwen/Kimi weights, chunked prefill, and the batched megakernel input builder behind this same endpoint. ".to_string(),
    }
}

fn chat_completion_stream_json(
    node_id: u32,
    wall_ms: f64,
    answer: &str,
    device_was_warm: bool,
    request_id: u64,
    stream_ttft_ms: f64,
) -> String {
    let collective_policy = collective_policy_json();
    format!(
        r#"data: {{"id":"chatcmpl-mainarch-demo","object":"chat.completion.chunk","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"delta":{{"role":"assistant","content":"{}"}},"finish_reason":null}}],"mainarch":{{"ok":true,"busy":false,"node":{node_id},"wall_ms":{wall_ms:.3},"stream_ttft_ms":{stream_ttft_ms:.3},"decode_us_token":{DECODE_US_TOKEN},"greedy_tokens":{TOKENS},"mode":"synthetic_decode_proof","stream":true,"stream_timing":"first_chunk_after_cached_proof","live_lane":"persistent-worker","collective_policy":{collective_policy},"device_was_warm":{device_was_warm},"live_lane_request":{request_id}}}}}

data: {{"id":"chatcmpl-mainarch-demo","object":"chat.completion.chunk","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}],"mainarch":{{"ok":true,"busy":false,"node":{node_id},"stream_ttft_ms":{stream_ttft_ms:.3},"decode_us_token":{DECODE_US_TOKEN},"greedy_tokens":{TOKENS},"mode":"synthetic_decode_proof","stream":true,"stream_timing":"first_chunk_after_cached_proof","live_lane":"persistent-worker","collective_policy":{collective_policy},"device_was_warm":{device_was_warm},"live_lane_request":{request_id}}}}}

data: [DONE]

"#,
        json_escape(answer)
    )
}

fn chat_completion_error_stream_json(
    node_id: u32,
    message: &str,
    error_type: &str,
    code: &str,
    busy: bool,
    wall_ms: f64,
) -> String {
    format!(
        r#"data: {{"error":{{"message":"{}","type":"{}","code":"{}"}},"model":"{MODEL_ID}","mainarch":{{"ok":false,"busy":{busy},"node":{node_id},"wall_ms":{wall_ms:.3},"decode_us_token":{DECODE_US_TOKEN},"greedy_tokens":{TOKENS},"stream":true,"live_lane":"persistent-worker"}}}}

data: [DONE]

"#,
        json_escape(message),
        json_escape(error_type),
        json_escape(code)
    )
}

fn extract_chat_prompt(body: &str) -> String {
    let mut cursor = 0usize;
    let mut last_content = None;
    while let Some(pos) = body[cursor..].find("\"content\"") {
        let start = cursor + pos + "\"content\"".len();
        if let Some(content) = extract_json_string_after_colon(&body[start..]) {
            last_content = Some(content);
        }
        cursor = start;
    }
    last_content
        .filter(|content| !content.trim().is_empty())
        .unwrap_or_else(|| "What makes mainarch different?".to_string())
}

fn extract_json_string_after_colon(src: &str) -> Option<String> {
    let colon = src.find(':')?;
    let rest = src[colon + 1..].trim_start();
    let mut chars = rest.strip_prefix('"')?.chars();
    let mut out = String::new();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            return Some(out);
        }
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next()? {
            '"' => out.push('"'),
            '\\' => out.push('\\'),
            '/' => out.push('/'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'b' | 'f' => out.push(' '),
            'u' => {
                for _ in 0..4 {
                    let _ = chars.next();
                }
                out.push('?');
            }
            other => out.push(other),
        }
    }
    None
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn json_option(value: &Option<String>) -> String {
    match value {
        Some(value) => format!("\"{}\"", json_escape(value)),
        None => "null".to_string(),
    }
}

fn json_u32_slice(values: &[u32]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

const INDEX: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Mainarch one-page sandbox for the direct MI355X serving path.">
  <title>mainarch serving sandbox</title>
  <style>
    :root {
      --ink: #121712;
      --muted: #677267;
      --paper: #f7f0de;
      --card: rgba(255, 252, 241, 0.86);
      --card-solid: #fffdf4;
      --line: rgba(18, 23, 18, 0.13);
      --line-strong: rgba(18, 23, 18, 0.22);
      --green: #173f2a;
      --lime: #c8f45c;
      --blue: #bfdad8;
      --clay: #ba6a43;
      --gold: #e4b84a;
      --red: #8e3028;
      --good: #1c7544;
      --shadow: 0 28px 80px rgba(24, 31, 23, 0.15);
      --serif: Georgia, "Times New Roman", serif;
      --mono: "Courier New", monospace;
      --sans: "Avenir Next", "Trebuchet MS", Verdana, sans-serif;
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 10% 8%, rgba(200, 244, 92, 0.42), transparent 27rem),
        radial-gradient(circle at 88% 12%, rgba(191, 218, 216, 0.95), transparent 28rem),
        radial-gradient(circle at 74% 86%, rgba(186, 106, 67, 0.18), transparent 30rem),
        linear-gradient(135deg, #fbf4e1 0%, #eaf0dc 52%, #dde8dc 100%);
      overflow-x: hidden;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(18, 23, 18, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(18, 23, 18, 0.035) 1px, transparent 1px);
      background-size: 36px 36px;
      mask-image: linear-gradient(to bottom, black, transparent 74%);
    }

    main {
      position: relative;
      width: min(1240px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 46px;
    }

    .topline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
      font: 800 12px/1 var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .brand,
    .status-pill,
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.58);
      backdrop-filter: blur(10px);
    }

    .brand { color: var(--green); }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--muted);
      box-shadow: 0 0 0 5px rgba(103, 114, 103, 0.12);
    }

    .dot.ready {
      background: var(--good);
      box-shadow: 0 0 0 5px rgba(28, 117, 68, 0.14);
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(380px, 0.8fr);
      gap: 20px;
      align-items: stretch;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 32px;
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }

    .intro {
      position: relative;
      min-height: 610px;
      overflow: hidden;
      padding: clamp(26px, 4vw, 42px);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .intro::after {
      content: "";
      position: absolute;
      right: -120px;
      bottom: -150px;
      width: 420px;
      height: 420px;
      border-radius: 999px;
      background:
        linear-gradient(135deg, rgba(200, 244, 92, 0.72), rgba(23, 63, 42, 0.30)),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.26) 0 9px, transparent 9px 18px);
      filter: saturate(1.08);
    }

    .eyebrow {
      position: relative;
      z-index: 1;
      display: inline-flex;
      width: fit-content;
      gap: 10px;
      align-items: center;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.54);
      color: var(--green);
      font: 800 12px/1 var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    h1,
    h2,
    p { margin: 0; }

    h1 {
      position: relative;
      z-index: 1;
      max-width: 850px;
      margin: 22px 0 20px;
      font-family: var(--serif);
      font-size: clamp(48px, 7.3vw, 100px);
      line-height: 0.88;
      letter-spacing: -0.068em;
    }

    .lead {
      position: relative;
      z-index: 1;
      max-width: 780px;
      color: var(--muted);
      font: 18px/1.56 var(--mono);
    }

    .claim-strip {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 30px;
    }

    .claim,
    .score,
    .metric,
    .stack-list li {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.52);
    }

    .claim {
      min-height: 124px;
      padding: 17px;
      border-radius: 23px;
    }

    .claim strong,
    .score b,
    .metric strong {
      display: block;
      font-family: var(--serif);
      line-height: 0.95;
      letter-spacing: -0.045em;
    }

    .claim strong { font-size: 30px; }

    .claim span,
    .score span,
    .metric span,
    .note,
    .foot {
      display: block;
      color: var(--muted);
      font: 12px/1.43 var(--mono);
    }

    .claim span { margin-top: 10px; }

    .route-strip {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 8px;
      margin-top: 14px;
    }

    .route-step {
      min-height: 86px;
      padding: 13px;
      border: 1px solid rgba(24, 63, 42, 0.16);
      border-radius: 19px;
      background: rgba(255, 255, 255, 0.49);
    }

    .route-step span {
      display: block;
      color: var(--muted);
      font: 10px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .route-step b {
      display: block;
      margin-top: 9px;
      font-family: var(--serif);
      font-size: 22px;
      line-height: 1;
      letter-spacing: -0.035em;
    }

    .route-step.future {
      border-color: rgba(186, 106, 67, 0.24);
      background: rgba(255, 248, 221, 0.58);
    }

    .scoreboard {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 10px;
      margin-top: 24px;
    }

    .score {
      padding: 15px;
      border-radius: 20px;
    }

    .score b { font-size: 29px; }
    .score span { margin-top: 7px; text-transform: uppercase; letter-spacing: 0.04em; }

    .safety-strip {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 1.05fr 1fr 0.9fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }

    .safety-card {
      padding: 14px 15px;
      border: 1px solid rgba(24, 63, 42, 0.18);
      border-radius: 20px;
      color: #edffd0;
      background: linear-gradient(135deg, rgba(23, 63, 42, 0.94), rgba(15, 22, 19, 0.92));
      box-shadow: 0 16px 38px rgba(15, 22, 19, 0.14);
    }

    .safety-card strong {
      display: block;
      margin-top: 7px;
      font-family: var(--serif);
      font-size: 31px;
      line-height: 1;
      letter-spacing: -0.04em;
    }

    .safety-card span {
      color: rgba(237, 255, 208, 0.76);
      font: 11px/1.3 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .safety-card.warn { background: linear-gradient(135deg, rgba(142, 48, 40, 0.95), rgba(15, 22, 19, 0.90)); }
    .safety-card.clean { background: linear-gradient(135deg, rgba(28, 117, 68, 0.96), rgba(15, 22, 19, 0.92)); }

    .console {
      display: grid;
      grid-template-rows: auto 1fr auto auto auto;
      gap: 14px;
      padding: 22px;
    }

    h2 {
      font-family: var(--serif);
      font-size: 26px;
      letter-spacing: -0.035em;
    }

    .mini-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font: 12px/1.25 var(--mono);
    }

    .model-tag {
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .proof-trail {
      margin-top: 16px;
      padding: 14px;
      border-radius: 22px;
      border: 1px solid rgba(33, 47, 37, .14);
      background:
        linear-gradient(135deg, rgba(255,255,255,.74), rgba(239,232,207,.50)),
        radial-gradient(circle at 10% 0%, rgba(208,111,52,.14), transparent 34%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.65);
    }

    .trail-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 9px;
    }

    .trail-head span {
      color: var(--muted);
      font-size: .76rem;
      text-transform: uppercase;
      letter-spacing: .12em;
    }

    .trail-head b {
      color: var(--ink);
      font-size: .82rem;
      text-align: right;
    }

    .proof-trail ol {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 7px;
    }

    .proof-trail li {
      position: relative;
      padding: 9px 10px 9px 34px;
      border-radius: 14px;
      color: rgba(31, 39, 32, .72);
      background: rgba(255,255,255,.52);
      border: 1px solid rgba(33, 47, 37, .09);
      font-size: .88rem;
    }

    .proof-trail li::before {
      content: "";
      position: absolute;
      left: 12px;
      top: 50%;
      width: 9px;
      height: 9px;
      border-radius: 999px;
      transform: translateY(-50%);
      background: rgba(31,39,32,.28);
      box-shadow: 0 0 0 5px rgba(31,39,32,.06);
    }

    .proof-trail li.live {
      color: var(--ink);
      border-color: rgba(208,111,52,.32);
      background: rgba(255,248,235,.80);
    }

    .proof-trail li.live::before {
      background: var(--accent);
      box-shadow: 0 0 0 6px rgba(208,111,52,.14);
    }

    .proof-trail li.done {
      color: var(--ink);
      border-color: rgba(41, 114, 74, .24);
      background: rgba(238, 248, 239, .74);
    }

    .proof-trail li.done::before {
      background: #2f8f55;
      box-shadow: 0 0 0 5px rgba(47,143,85,.12);
    }

    .chat-window {
      min-height: 360px;
      max-height: 470px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 13px;
      padding: 17px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--card-solid);
    }

    .bubble {
      max-width: 91%;
      padding: 13px 15px;
      border-radius: 18px;
      font: 14px/1.49 var(--mono);
      white-space: pre-wrap;
    }

    .bubble.system {
      align-self: flex-start;
      color: #edffd0;
      background: #0f1613;
    }

    .bubble.user {
      align-self: flex-end;
      background: #e6efdc;
      border: 1px solid var(--line);
    }

    .bubble.assistant {
      align-self: flex-start;
      background: #fff8dd;
      border: 1px solid var(--line);
    }

    .composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    .actions { display: grid; gap: 9px; min-width: 126px; }

    textarea {
      width: 100%;
      min-height: 112px;
      resize: vertical;
      padding: 15px;
      border: 1px solid var(--line);
      border-radius: 20px;
      outline: none;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.76);
      font: 14px/1.45 var(--mono);
    }

    button {
      border: 0;
      border-radius: 999px;
      padding: 15px 18px;
      color: #f7ffe3;
      background: var(--green);
      font: 800 12px/1 var(--mono);
      letter-spacing: 0.05em;
      text-transform: uppercase;
      cursor: pointer;
      box-shadow: 0 11px 25px rgba(24, 63, 42, 0.18);
      transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
    }

    button:hover { transform: translateY(-1px); box-shadow: 0 15px 30px rgba(24, 63, 42, 0.22); }
    button.secondary { color: var(--green); background: rgba(255, 255, 255, 0.68); border: 1px solid rgba(24, 63, 42, 0.20); box-shadow: none; }
    button:disabled { cursor: wait; opacity: 0.58; transform: none; }

    .examples { display: flex; gap: 8px; flex-wrap: wrap; }

    .examples button {
      padding: 9px 11px;
      color: var(--green);
      background: rgba(255, 255, 255, 0.56);
      border: 1px solid var(--line);
      box-shadow: none;
      text-transform: none;
      letter-spacing: 0;
      font-weight: 800;
    }

    .grid {
      display: grid;
      grid-template-columns: 0.9fr 1.06fr 1fr;
      gap: 18px;
      margin-top: 20px;
    }

    .panel { padding: 24px; }

    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }

    .metric {
      min-height: 112px;
      padding: 16px;
      border-radius: 21px;
    }

    .metric strong { font-size: 35px; }
    .metric span { margin-top: 8px; }

    .bars { display: grid; gap: 12px; margin-top: 18px; }

    .bar-label {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font: 12px/1 var(--mono);
    }

    .bar {
      height: 14px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(18, 23, 18, 0.10);
    }

    .bar i { display: block; height: 100%; border-radius: inherit; background: var(--clay); }
    .bar.fast i { background: linear-gradient(90deg, var(--green), var(--lime)); }

    .terminal,
    pre {
      margin: 16px 0 0;
      padding: 16px;
      overflow: auto;
      border-radius: 20px;
      color: #edffd0;
      background: #0f1613;
      font: 12px/1.52 var(--mono);
      box-shadow: inset 0 0 0 1px rgba(200, 244, 92, 0.12);
    }

    .terminal { min-height: 176px; white-space: pre-wrap; }

    .receipt {
      padding: 15px;
      border: 1px solid rgba(24, 63, 42, 0.18);
      border-radius: 22px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.78), rgba(234,240,220,0.62)),
        radial-gradient(circle at 0 0, rgba(200,244,92,0.22), transparent 34%);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.62);
    }

    .receipt.done {
      border-color: rgba(28, 117, 68, 0.28);
      background:
        linear-gradient(135deg, rgba(241,255,226,0.86), rgba(255,253,244,0.70)),
        radial-gradient(circle at 0 0, rgba(200,244,92,0.28), transparent 34%);
    }

    .receipt.error {
      border-color: rgba(142, 48, 40, 0.32);
      background:
        linear-gradient(135deg, rgba(255,241,231,0.90), rgba(255,253,244,0.70)),
        radial-gradient(circle at 0 0, rgba(186,106,67,0.22), transparent 34%);
    }

    .receipt span {
      display: block;
      color: var(--green);
      font: 800 11px/1 var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .receipt b {
      display: block;
      margin-top: 8px;
      font-family: var(--serif);
      font-size: 25px;
      line-height: 0.98;
      letter-spacing: -0.035em;
    }

    .receipt p {
      margin-top: 8px;
      color: var(--muted);
      font: 12px/1.42 var(--mono);
    }

    .receipt dl {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 13px 0 0;
    }

    .receipt div {
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.58);
    }

    .receipt dt {
      color: var(--muted);
      font: 10px/1 var(--mono);
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }

    .receipt dd {
      margin: 6px 0 0;
      font: 800 12px/1.22 var(--mono);
      overflow-wrap: anywhere;
    }

    .stack-list {
      display: grid;
      gap: 12px;
      padding: 0;
      margin: 17px 0 0;
      list-style: none;
    }

    .stack-list li {
      padding: 15px;
      border-radius: 20px;
      font: 14px/1.45 var(--mono);
    }

    .stack-list b {
      font-family: var(--serif);
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .note { margin-top: 12px; font-size: 14px; line-height: 1.5; }
    .foot { margin-top: 20px; font-size: 13px; line-height: 1.55; }

    @media (max-width: 1060px) {
      .hero,
      .grid { grid-template-columns: 1fr; }
      .intro { min-height: auto; }
    }

    @media (max-width: 720px) {
      main { width: min(100vw - 22px, 1240px); padding-top: 16px; }
      .intro, .console, .panel { padding: 20px; border-radius: 24px; }
      .claim-strip, .route-strip, .scoreboard, .safety-strip, .metric-grid, .composer { grid-template-columns: 1fr; }
      .actions { grid-template-columns: 1fr 1fr; }
      .topline { align-items: flex-start; flex-direction: column; }
      h1 { font-size: clamp(43px, 15vw, 64px); }
    }
  </style>
</head>
<body>
  <main>
    <div class="topline">
      <div class="brand">mainarch serving sandbox</div>
      <div class="status-pill"><span id="statusDot" class="dot"></span><span id="health">checking live lane</span></div>
    </div>

    <section class="hero">
      <div class="intro card">
        <div>
          <div class="eyebrow">direct MI355X serving preview</div>
          <h1>Try the serving route we are building for real Qwen and Kimi.</h1>
          <p class="lead">One page, one binary, one stable API seam: prompt the OpenAI-shaped endpoint, watch the persistent MI355X lane stream back, and see the latest banked Kimi MLA speed win beside exactly what is real today.</p>
          <div class="claim-strip">
            <div class="claim"><strong>Play it</strong><span>Every prompt uses stream:true against /v1/chat/completions, not a separate toy route.</span></div>
            <div class="claim"><strong>Feel it</strong><span>The page keeps latency, TTFT, queue state, KV leases, and the latest split-K64 Kimi evidence visible while the request runs.</span></div>
            <div class="claim"><strong>Extend it</strong><span>Real proof, synthetic weights. The seam stays while tokenizer, weights, and TP8 serving replace the proof backend.</span></div>
          </div>
          <div class="route-strip" aria-label="stable serving route">
            <div class="route-step"><span>browser</span><b>one-page sandbox</b></div>
            <div class="route-step"><span>contract</span><b>/v1/chat/completions</b></div>
            <div class="route-step"><span>scheduler</span><b>persistent live lane</b></div>
            <div class="route-step"><span>today</span><b>MI355X proof runner</b></div>
            <div class="route-step future"><span>next</span><b>real Qwen/Kimi weights</b></div>
          </div>
          <div class="safety-strip" aria-label="live scheduler safety counters">
            <div class="safety-card clean"><span>KV ownership checks</span><strong id="guardChecks">--</strong></div>
            <div class="safety-card"><span>active blocks checked</span><strong id="guardBlocks">--</strong></div>
            <div id="guardFailureCard" class="safety-card clean"><span>fail-closed events</span><strong id="guardFailures">--</strong></div>
            <div class="safety-card"><span>collective gate</span><strong id="collectiveGate">--</strong></div>
          </div>
        </div>
        <div class="scoreboard">
          <div class="score"><b id="wallScore">--</b><span>last wall</span></div>
          <div class="score"><b id="ttftScore">--</b><span>stream TTFT</span></div>
          <div class="score"><b id="queueScore">--</b><span>queue depth</span></div>
          <div class="score"><b id="kvScore">--</b><span>KV blocks</span></div>
          <div class="score"><b id="doneScore">--</b><span>completed</span></div>
        </div>
      </div>

      <aside class="console card">
        <div class="mini-row">
          <h2>Play with the serving seam</h2>
          <span id="model" class="model-tag">loading model</span>
        </div>
        <div class="chat-window" id="chat">
          <div class="bubble system">ready. Ask a normal question. The words are synthetic today; the stream, scheduler, KV lease, timing, and MI355X proof lane are real.</div>
        </div>
        <div class="composer">
          <textarea id="prompt">Give me the 30-second version of what mainarch is proving.</textarea>
          <div class="actions">
            <button id="run">Stream run</button>
            <button id="burst" class="secondary">Burst x4</button>
          </div>
        </div>
        <div class="examples">
          <button data-prompt="Give me the 30-second version of what mainarch is proving.">30-second read</button>
          <button data-prompt="What is real in this demo today, and what is still synthetic?">What is real</button>
          <button data-prompt="What changes when the backend swaps from proof runner to real Qwen or Kimi weights?">Swap path</button>
          <button data-prompt="Show me the mainarch speed evidence without marketing gloss.">Evidence</button>
        </div>
        <div class="mini-row">
          <span id="runState">mode: synthetic decode proof</span>
          <span>burst x4 is capped to 2 decode steps per request</span>
        </div>
        <div id="receipt" class="receipt" aria-live="polite">
          <span>demo receipt</span>
          <b>Run a prompt to get the one-page proof summary.</b>
          <p>The receipt will show the public route, server timing, client first chunk, KV lease, and the honest real-vs-next boundary.</p>
        </div>
        <div class="proof-trail" aria-label="last request proof trail" aria-live="polite">
          <div class="trail-head">
            <span>request proof trail</span>
            <b id="trailStatus">waiting for first run</b>
          </div>
          <ol>
            <li id="trailRoute">route: /v1/chat/completions stream=true</li>
            <li id="trailAdmission">admission: waiting for a prompt</li>
            <li id="trailScheduler">scheduler: waiting for a decode tick</li>
            <li id="trailGpu">MI355X proof: waiting for a token step</li>
            <li id="trailKv">KV lease: waiting for release accounting</li>
          </ol>
        </div>
      </aside>
    </section>

    <section class="grid">
      <div class="panel card">
        <h2>Latest speed proof</h2>
        <div class="metric-grid">
          <div class="metric"><strong>188.471 us</strong><span>combined hot serving step after split-K64</span></div>
          <div class="metric"><strong>144.202 us</strong><span>MLA hot path, 67.5% lower latency than split-K8</span></div>
          <div class="metric"><strong id="liveTpot">--</strong><span>last live cached-runner token timing</span></div>
          <div class="metric"><strong id="requestId">--</strong><span>live lane request id</span></div>
        </div>
        <div class="bars">
          <div class="bar-label"><span>control route</span><span>3405 GB/s</span></div>
          <div class="bar"><i style="width:90.3%"></i></div>
          <div class="bar-label"><span>mainarch fast path</span><span>3770 GB/s</span></div>
          <div class="bar fast"><i style="width:100%"></i></div>
        </div>
      </div>

      <div class="panel card">
        <h2>Claim boundary</h2>
        <ul class="stack-list">
          <li><b>Real now:</b> direct mainarch GPU proof on MI355X through a persistent worker that owns the live lane.</li>
          <li><b>Real now:</b> OpenAI-shaped /v1/models and /v1/chat/completions in the same release binary as the kernels, ready for a sandbox reverse proxy.</li>
          <li><b>Real now:</b> admission-time logical KV block reservations with visible capacity, release accounting, and ownership checks.</li>
          <li><b>Real now:</b> scheduler-visible decode steps that return control after each verified token chunk.</li>
          <li><b>Next swap:</b> tokenizer-backed Qwen/Kimi decode, real weights, sampling, quotas, public isolation, and TP8 collectives behind the same route.</li>
          <li><b>Not claimed:</b> real Qwen/Kimi answer quality or production public serving in this synthetic-proof sandbox.</li>
        </ul>
      </div>

      <div class="panel card">
        <h2>Sandbox contract</h2>
        <p class="note">Operationally boring by design: build one release binary, run demo-serve, expose the port. The browser, curl, and future app shell all hit the same public contract that real-weight serving will keep.</p>
        <pre>curl http://HOST:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mainarch-qwen3-235b-a22b-synthetic-proof","stream":true,"messages":[{"role":"user","content":"why is this fast?"}]}'</pre>
        <div id="out" class="terminal">loading evidence...</div>
      </div>
    </section>

    <p class="foot">Deploy with: ./target/release/mainarch demo-serve --bind 0.0.0.0:8080 --node 2. Readiness is /api/health, manifest is /api/demo, live metrics are /metrics, evidence is /api/evidence, and the public contract is /v1/chat/completions. Keep auth, quotas, and prompt policy in front of this raw lane when it becomes public.</p>
  </main>

  <script>
    const MODEL_ID = "mainarch-qwen3-235b-a22b-synthetic-proof";
    const els = {
      out: document.getElementById("out"),
      run: document.getElementById("run"),
      burst: document.getElementById("burst"),
      prompt: document.getElementById("prompt"),
      chat: document.getElementById("chat"),
      health: document.getElementById("health"),
      statusDot: document.getElementById("statusDot"),
      model: document.getElementById("model"),
      wallScore: document.getElementById("wallScore"),
      ttftScore: document.getElementById("ttftScore"),
      queueScore: document.getElementById("queueScore"),
      kvScore: document.getElementById("kvScore"),
      doneScore: document.getElementById("doneScore"),
      liveTpot: document.getElementById("liveTpot"),
      requestId: document.getElementById("requestId"),
      runState: document.getElementById("runState"),
      guardChecks: document.getElementById("guardChecks"),
      guardBlocks: document.getElementById("guardBlocks"),
      guardFailures: document.getElementById("guardFailures"),
      guardFailureCard: document.getElementById("guardFailureCard"),
      collectiveGate: document.getElementById("collectiveGate"),
      receipt: document.getElementById("receipt"),
      trailStatus: document.getElementById("trailStatus"),
      trailRoute: document.getElementById("trailRoute"),
      trailAdmission: document.getElementById("trailAdmission"),
      trailScheduler: document.getElementById("trailScheduler"),
      trailGpu: document.getElementById("trailGpu"),
      trailKv: document.getElementById("trailKv")
    };

    async function loadJson(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(path + " returned " + res.status);
      return res.json();
    }

    async function loadText(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(path + " returned " + res.status);
      return res.text();
    }

    function parseMetrics(text) {
      const out = {};
      for (const line of text.split("\n")) {
        if (!line || line[0] === "#") continue;
        const parts = line.trim().split(/\s+/);
        if (parts.length === 2) out[parts[0]] = Number(parts[1]);
      }
      return out;
    }

    function addBubble(kind, text) {
      const bubble = document.createElement("div");
      bubble.className = "bubble " + kind;
      bubble.textContent = text;
      els.chat.appendChild(bubble);
      els.chat.scrollTop = els.chat.scrollHeight;
      return bubble;
    }

    function renderTerminal(lines) {
      els.out.textContent = lines.join("\n");
    }

    function setReceipt(title, body, items, state) {
      els.receipt.classList.toggle("done", state === "done");
      els.receipt.classList.toggle("error", state === "error");
      els.receipt.textContent = "";

      const label = document.createElement("span");
      label.textContent = "demo receipt";
      const heading = document.createElement("b");
      heading.textContent = title;
      const copy = document.createElement("p");
      copy.textContent = body;
      const list = document.createElement("dl");

      for (const [key, value] of items) {
        const row = document.createElement("div");
        const dt = document.createElement("dt");
        const dd = document.createElement("dd");
        dt.textContent = key;
        dd.textContent = value;
        row.appendChild(dt);
        row.appendChild(dd);
        list.appendChild(row);
      }

      els.receipt.appendChild(label);
      els.receipt.appendChild(heading);
      els.receipt.appendChild(copy);
      els.receipt.appendChild(list);
    }

    function fmtMs(value) {
      if (!Number.isFinite(value)) return "--";
      if (value >= 100) return value.toFixed(0) + " ms";
      if (value >= 10) return value.toFixed(1) + " ms";
      return value.toFixed(2) + " ms";
    }

    function fmtUs(value) {
      if (!Number.isFinite(value)) return "--";
      return value.toFixed(0) + " us";
    }

    function markTrail(el, text, state) {
      el.textContent = text;
      el.classList.remove("live", "done");
      if (state) el.classList.add(state);
    }

    function resetProofTrail(status) {
      els.trailStatus.textContent = status;
      markTrail(els.trailRoute, "route: /v1/chat/completions stream=true", "live");
      markTrail(els.trailAdmission, "admission: queued behind the public route", "");
      markTrail(els.trailScheduler, "scheduler: waiting for a decode tick", "");
      markTrail(els.trailGpu, "MI355X proof: waiting for a token step", "");
      markTrail(els.trailKv, "KV lease: waiting for release accounting", "");
    }

    function renderProofTrail(mainarch) {
      if (!mainarch) return;
      const request = mainarch.live_lane_request === undefined ? "request" : "request #" + mainarch.live_lane_request;
      const finished = mainarch.finish_reason || mainarch.stop_reason || mainarch.generated_steps !== undefined;
      const queue = mainarch.queue_depth_on_enqueue === undefined ? "admitted" : "queue depth " + mainarch.queue_depth_on_enqueue;
      const tick = mainarch.scheduler_tick === undefined ? "running" : "tick " + mainarch.scheduler_tick;
      const active = mainarch.scheduler_active === undefined ? "" : ", active " + mainarch.scheduler_active;
      const token = mainarch.token === undefined ? "" : ", token " + mainarch.token;
      const stepUs = mainarch.step_gpu_us || mainarch.gpu_us_per_token || mainarch.decode_us_token;
      const leased = mainarch.kv_blocks_leased === undefined ? "reserved before decode" : mainarch.kv_blocks_leased + " blocks";
      const released = mainarch.kv_blocks_released === undefined ? "" : ", released=" + mainarch.kv_blocks_released;
      els.trailStatus.textContent = finished ? request + " finished" : request + " streaming";
      markTrail(els.trailRoute, "route: /v1/chat/completions stream=true", "done");
      markTrail(els.trailAdmission, "admission: " + queue + ", " + request, "done");
      markTrail(els.trailScheduler, "scheduler: " + tick + active, finished ? "done" : "live");
      markTrail(els.trailGpu, "MI355X proof: " + fmtUs(stepUs) + token, finished ? "done" : "live");
      markTrail(els.trailKv, "KV lease: " + leased + released, mainarch.kv_blocks_released ? "done" : "live");
    }

    async function loadState() {
      try {
        const [health, evidence, compare, models, metricsText] = await Promise.all([
          loadJson("/api/health"),
          loadJson("/api/evidence"),
          loadJson("/api/compare"),
          loadJson("/v1/models"),
          loadText("/metrics")
        ]);
        const metrics = parseMetrics(metricsText);
        const running = metrics.mainarch_num_requests_running || 0;
        const waiting = metrics.mainarch_num_requests_waiting || 0;
        const guardChecks = metrics.mainarch_scheduler_kv_ownership_checks_total || 0;
        const guardFailures = metrics.mainarch_scheduler_kv_ownership_failures_total || 0;
        const guardBlocks = metrics.mainarch_scheduler_kv_active_blocks_checked_total || 0;
        const policy = health.collective_policy || {};
        els.guardChecks.textContent = guardChecks.toLocaleString();
        els.guardBlocks.textContent = guardBlocks.toLocaleString();
        els.guardFailures.textContent = guardFailures.toLocaleString();
        els.guardFailureCard.classList.toggle("warn", guardFailures > 0);
        els.guardFailureCard.classList.toggle("clean", guardFailures === 0);
        els.collectiveGate.textContent = (policy.logical_tp_ranks || "--") + "->" + (policy.active_ranks || "--");
        els.health.textContent = health.busy ? "live proof running" : "ready on node " + health.node;
        els.statusDot.classList.toggle("ready", !health.busy);
        els.model.textContent = models.data[0].id;
        els.queueScore.textContent = String(health.queue_depth);
        els.kvScore.textContent = health.kv_blocks_in_use + "/" + health.kv_block_capacity;
        els.doneScore.textContent = String(health.completed_requests);
        renderTerminal([
          "target: " + evidence.target,
          "runtime: " + evidence.runtime,
          "latest win: " + evidence.latest_kernel_win,
          "kimi split-K: " + evidence.kimi_split_k,
          "kimi MLA hot path: " + evidence.kimi_mla_hot_us + " us/case",
          "kimi hot serving step: " + evidence.kimi_hot_step_us_token + " us/token",
          "kimi vs split-K8: +" + compare.kimi_splitk8_uplift_percent.toFixed(1) + "%",
          "mainarch TP8 collective: " + evidence.mainarch_tp8_collective_gbps + " GB/s",
          "RCCL same payload: " + evidence.rccl_tp8_collective_gbps + " GB/s",
          "banked decode proof: " + evidence.decode_us_token + " us/token",
          "mainarch fast path: " + evidence.mainarch_gbps + " GB/s",
          "control path: " + evidence.control_gbps + " GB/s",
          "uplift: " + compare.uplift_percent.toFixed(1) + "%",
          "queue: running=" + running + " waiting=" + waiting + " capacity=" + health.queue_capacity,
          "kv: blocks=" + health.kv_blocks_in_use + "/" + health.kv_block_capacity + " leases_active=" + health.kv_leases_active,
          "kv ownership gate: checks=" + guardChecks + " failures=" + guardFailures + " blocks_checked=" + guardBlocks,
          "collective gate: " + (policy.state || "unknown") + " backend=" + (policy.selected_backend || "unknown") + " tp=" + (policy.logical_tp_ranks || "?") + " active=" + (policy.active_ranks || "?"),
          "completed=" + health.completed_requests + " cancelled=" + health.cancelled_requests,
          "status: " + evidence.status
        ]);
      } catch (err) {
        els.health.textContent = "health check failed";
        els.statusDot.classList.remove("ready");
        renderTerminal(["could not load demo state", String(err)]);
      }
    }

    function handleStreamEvent(payload, assistant, timings) {
      if (payload === "[DONE]") return;
      const event = JSON.parse(payload);
      if (event.error) {
        assistant.textContent += "Live proof failed: " + event.error.message;
        timings.error = event.error;
        return;
      }
      const choice = event.choices && event.choices[0];
      const delta = choice && choice.delta;
      if (delta && delta.content) assistant.textContent += delta.content;
      if (event.mainarch) {
        timings.events = timings.events || [];
        timings.events.push(event.mainarch);
        timings.maxSchedulerActive = Math.max(timings.maxSchedulerActive || 0, event.mainarch.scheduler_active || 0);
        if (event.mainarch.scheduler_tick !== undefined) {
          timings.schedulerTicks = timings.schedulerTicks || new Set();
          timings.schedulerTicks.add(event.mainarch.scheduler_tick);
        }
        timings.mainarch = event.mainarch;
        els.wallScore.textContent = fmtMs(event.mainarch.wall_ms);
        els.ttftScore.textContent = fmtMs(event.mainarch.stream_ttft_ms);
        els.liveTpot.textContent = fmtUs(event.mainarch.decode_us_token);
        els.requestId.textContent = "#" + event.mainarch.live_lane_request;
        els.runState.textContent = event.mainarch.stream_timing || event.mainarch.mode || "streaming";
        renderProofTrail(event.mainarch);
      }
    }

    async function readSse(res, assistant, started) {
      const timings = {};
      let firstChunkAt = null;
      if (!res.body || !res.body.getReader) {
        const text = await res.text();
        for (const block of text.split("\n\n")) {
          const line = block.split("\n").find((part) => part.startsWith("data: "));
          if (!line) continue;
          if (firstChunkAt === null) firstChunkAt = performance.now();
          handleStreamEvent(line.slice(6), assistant, timings);
        }
        timings.clientTtfbMs = firstChunkAt === null ? null : firstChunkAt - started;
        return timings;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const next = await reader.read();
        if (next.done) break;
        if (firstChunkAt === null) firstChunkAt = performance.now();
        buffer += decoder.decode(next.value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const line = block.split("\n").find((part) => part.startsWith("data: "));
          if (!line) continue;
          handleStreamEvent(line.slice(6), assistant, timings);
        }
      }
      timings.clientTtfbMs = firstChunkAt === null ? null : firstChunkAt - started;
      return timings;
    }

    function setBusy(busy) {
      els.run.disabled = busy;
      els.burst.disabled = busy;
    }

    async function runStream(text, label, maxTokens, renderSingle) {
      addBubble("user", label ? label + ": " + text : text);
      const assistant = addBubble("assistant", label ? label + " streaming...\n" : "");
      const initialAssistantText = assistant.textContent;
      const started = performance.now();
      resetProofTrail(label ? label + " dispatched" : "dispatching stream");
      setReceipt(
        "Request entered the public serving seam.",
        "The browser is using the same OpenAI-shaped route that real Qwen/Kimi serving will keep.",
        [
          ["route", "/v1/chat/completions"],
          ["stream", "true"],
          ["today", "MI355X synthetic proof"],
          ["next", "real tokenizer and weights"]
        ],
        ""
      );
      try {
        const body = {
          model: MODEL_ID,
          stream: true,
          messages: [{ role: "user", content: text }]
        };
        if (maxTokens) body.max_completion_tokens = maxTokens;
        const res = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body)
        });
        const timings = await readSse(res, assistant, started);
        const totalMs = performance.now() - started;
        if (assistant.textContent === initialAssistantText) {
          assistant.textContent = timings.error ? "Stream returned an error." : "Stream completed without assistant text.";
        }
        if (renderSingle && timings.mainarch) {
          renderTerminal([
            "endpoint: /v1/chat/completions stream=true",
            "node: " + timings.mainarch.node,
            "request: " + timings.mainarch.live_lane_request,
            "server wall: " + fmtMs(timings.mainarch.wall_ms),
            "server TTFT: " + fmtMs(timings.mainarch.stream_ttft_ms),
            "client first chunk: " + fmtMs(timings.clientTtfbMs),
            "client total: " + fmtMs(totalMs),
            "banked decode proof: " + timings.mainarch.decode_us_token + " us/token",
            "kv lease: " + timings.mainarch.kv_blocks_leased + " blocks released=" + timings.mainarch.kv_blocks_released,
            "ownership guard: see live counters above and /metrics",
            "max scheduler active: " + (timings.maxSchedulerActive || 0),
            "scheduler ticks seen: " + (timings.schedulerTicks ? timings.schedulerTicks.size : 0),
            "verified greedy tokens: " + JSON.stringify(timings.mainarch.greedy_tokens),
            "timing note: " + timings.mainarch.stream_timing
          ]);
          setReceipt(
            "Sandbox run completed on the live lane.",
            "The route, streaming parser, scheduler-visible GPU step, timing, and KV accounting ran. The answer text is still synthetic until real weights land.",
            [
              ["request", "#" + timings.mainarch.live_lane_request],
              ["server wall", fmtMs(timings.mainarch.wall_ms)],
              ["server TTFT", fmtMs(timings.mainarch.stream_ttft_ms)],
              ["client first chunk", fmtMs(timings.clientTtfbMs)],
              ["KV lease", (timings.mainarch.kv_blocks_leased || "--") + " blocks, released=" + (timings.mainarch.kv_blocks_released ? "yes" : "pending")],
              ["claim", "real seam, synthetic text"]
            ],
            "done"
          );
        }
        return { ok: true, timings, totalMs };
      } catch (err) {
        assistant.textContent = "Live stream failed: " + String(err);
        setReceipt(
          "Serving route returned an error.",
          "The demo failed visibly instead of hiding the backend state.",
          [
            ["route", "/v1/chat/completions"],
            ["error", String(err).slice(0, 160)]
          ],
          "error"
        );
        return { ok: false, error: String(err), timings: {}, totalMs: performance.now() - started };
      }
    }

    document.querySelectorAll(".examples button").forEach((button) => {
      button.addEventListener("click", () => {
        els.prompt.value = button.dataset.prompt || els.prompt.value;
        els.prompt.focus();
      });
    });

    els.run.addEventListener("click", async () => {
      const text = els.prompt.value.trim() || "What makes mainarch different?";
      setBusy(true);
      els.runState.textContent = "dispatching stream:true request";
      const result = await runStream(text, "", null, true);
      els.runState.textContent = result.ok ? "single stream completed" : "stream failed";
      setBusy(false);
      loadState();
    });

    els.burst.addEventListener("click", async () => {
      const base = els.prompt.value.trim() || "What makes mainarch different?";
      const prompts = [
        base,
        "What is real in this mainarch sandbox today?",
        "Why do scheduler-visible decode steps matter?",
        "What changes when real Qwen or Kimi weights land?"
      ];
      setBusy(true);
      els.runState.textContent = "dispatching 4 capped stream:true requests";
      const results = await Promise.all(prompts.map((prompt, idx) => runStream(prompt, "burst " + (idx + 1), 2, false)));
      const mains = results.map((result) => result.timings.mainarch).filter(Boolean);
      const maxActive = Math.max(0, ...results.map((result) => result.timings.maxSchedulerActive || 0));
      const tickCount = results.reduce((sum, result) => sum + (result.timings.schedulerTicks ? result.timings.schedulerTicks.size : 0), 0);
      const kvBlocks = mains.reduce((sum, mainarch) => sum + (mainarch.kv_blocks_leased || 0), 0);
      renderTerminal([
        "burst: 4 concurrent /v1/chat/completions stream=true requests",
        "cap: max_completion_tokens=2 per request",
        "completed streams: " + mains.length + "/4",
        "live lane requests: " + mains.map((mainarch) => "#" + mainarch.live_lane_request).join(", "),
        "max scheduler active observed: " + maxActive,
        "scheduler tick observations: " + tickCount,
        "kv blocks leased across burst: " + kvBlocks,
        "kv blocks released: " + mains.every((mainarch) => mainarch.kv_blocks_released),
        "ownership guard: active KV tables checked before each decode tick",
        "client totals: " + results.map((result) => fmtMs(result.totalMs)).join(", "),
        "note: this is still the synthetic proof, but the serving shape is the one we keep"
      ]);
      setReceipt(
        "Burst demo exercised the scheduler seam.",
        "Four concurrent streamed requests exposed queueing, scheduler ticks, and KV lease ownership behind the same public chat route.",
        [
          ["streams", mains.length + "/4 completed"],
          ["max active", String(maxActive)],
          ["scheduler ticks", String(tickCount)],
          ["KV leased", String(kvBlocks) + " blocks"],
          ["boundary", "synthetic proof, real route"]
        ],
        mains.length === 4 ? "done" : "error"
      );
      els.runState.textContent = mains.length === 4 ? "burst completed" : "burst completed with errors";
      setBusy(false);
      loadState();
    });

    loadState();
    setInterval(loadState, 5000);
  </script>
</body>
</html>"##;

// ---------------------------------------------------------------------------
// Serving a real OLMo 2 checkpoint over the same OpenAI-shaped route.
// ---------------------------------------------------------------------------

const OLMO_DEFAULT_MAX_NEW: usize = 48;
/// Upper bound on generated tokens for the real-model lane. The synthetic lane
/// clamps to PROOF_STREAM_TOKENS because its fixture is exactly four tokens
/// long; a real model has no such fixture, so it gets its own ceiling.
const OLMO_MAX_NEW_CEILING: usize = 256;

fn extract_olmo_max_new(body: &str) -> usize {
    extract_json_usize_after_key(body, "max_completion_tokens")
        .or_else(|| extract_json_usize_after_key(body, "max_tokens"))
        .unwrap_or(OLMO_DEFAULT_MAX_NEW)
        .clamp(1, OLMO_MAX_NEW_CEILING)
}

/// Non-streaming `/v1/chat/completions` backed by the real model.
fn run_olmo_chat_json(lane: &crate::olmo_lane::OlmoLane, body: &str) -> String {
    let prompt = extract_chat_prompt(body);
    let max_new = extract_olmo_max_new(body);
    let rx = match lane.submit(prompt.clone(), max_new) {
        Ok(rx) => rx,
        Err(err) => {
            return format!(
                r#"{{"error":{{"message":"{}","type":"mainarch_olmo_error","code":"lane_busy"}}}}"#,
                json_escape(&format!("{err:#}"))
            )
        }
    };
    let mut text = String::new();
    let mut generated = 0usize;
    let mut prompt_tokens = 0usize;
    let mut wall_ms = 0.0f64;
    let mut ttft_ms = 0.0f64;
    for ev in rx {
        match ev {
            crate::olmo_lane::OlmoEvent::Token { text: piece, .. } => text.push_str(&piece),
            crate::olmo_lane::OlmoEvent::Done {
                generated: g,
                prompt_tokens: pt,
                wall_ms: w,
                ttft_ms: t,
            } => {
                generated = g;
                prompt_tokens = pt;
                wall_ms = w;
                ttft_ms = t;
            }
            crate::olmo_lane::OlmoEvent::Error(err) => {
                return format!(
                    r#"{{"error":{{"message":"{}","type":"mainarch_olmo_error","code":"decode_failed"}}}}"#,
                    json_escape(&err)
                )
            }
        }
    }
    format!(
        r#"{{
  "id": "chatcmpl-mainarch-olmo",
  "object": "chat.completion",
  "created": 0,
  "model": "{}",
  "choices": [
    {{
      "index": 0,
      "message": {{ "role": "assistant", "content": "{}" }},
      "finish_reason": "length"
    }}
  ],
  "usage": {{ "prompt_tokens": {prompt_tokens}, "completion_tokens": {generated}, "total_tokens": {} }},
  "mainarch": {{
    "ok": true,
    "backend": "olmo2-real-weights",
    "node": {},
    "layers": {},
    "hidden": {},
    "vocab": {},
    "device_bytes": {},
    "wall_ms": {wall_ms:.3},
    "ttft_ms": {ttft_ms:.3},
    "ms_per_token": {:.3},
    "synthetic": false,
    "runtime": "raw KFD/AQL, no ROCm"
  }}
}}"#,
        lane.model_id,
        json_escape(&text),
        prompt_tokens + generated,
        lane.node,
        lane.layers,
        lane.hidden,
        lane.vocab,
        lane.device_bytes,
        if prompt_tokens + generated > 0 {
            wall_ms / (prompt_tokens + generated) as f64
        } else {
            0.0
        }
    )
}

/// Streaming `/v1/chat/completions` backed by the real model.
fn respond_olmo_chat_stream(
    stream: &mut TcpStream,
    lane: &crate::olmo_lane::OlmoLane,
    body: &str,
) -> Result<()> {
    let prompt = extract_chat_prompt(body);
    let max_new = extract_olmo_max_new(body);
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
Cache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
    )
    .context("writing OLMo SSE headers")?;
    stream.flush().ok();

    let rx = match lane.submit(prompt, max_new) {
        Ok(rx) => rx,
        Err(err) => {
            let payload = format!(
                r#"{{"error":{{"message":"{}","type":"mainarch_olmo_error","code":"lane_busy"}}}}"#,
                json_escape(&format!("{err:#}"))
            );
            write_sse_data(stream, &payload)?;
            write_sse_data(stream, "[DONE]")?;
            return Ok(());
        }
    };

    for ev in rx {
        match ev {
            crate::olmo_lane::OlmoEvent::Token { index, text } => {
                let role = if index == 0 {
                    r#""role":"assistant","#
                } else {
                    ""
                };
                let payload = format!(
                    r#"{{"id":"chatcmpl-mainarch-olmo","object":"chat.completion.chunk","created":0,"model":"{}","choices":[{{"index":0,"delta":{{{role}"content":"{}"}},"finish_reason":null}}],"mainarch":{{"backend":"olmo2-real-weights","synthetic":false,"node":{},"token_index":{index}}}}}"#,
                    lane.model_id,
                    json_escape(&text),
                    lane.node
                );
                write_sse_data(stream, &payload)?;
            }
            crate::olmo_lane::OlmoEvent::Done {
                generated,
                prompt_tokens,
                wall_ms,
                ttft_ms,
            } => {
                let steps = prompt_tokens + generated;
                let payload = format!(
                    r#"{{"id":"chatcmpl-mainarch-olmo","object":"chat.completion.chunk","created":0,"model":"{}","choices":[{{"index":0,"delta":{{}},"finish_reason":"length"}}],"mainarch":{{"backend":"olmo2-real-weights","synthetic":false,"node":{},"layers":{},"prompt_tokens":{prompt_tokens},"completion_tokens":{generated},"wall_ms":{wall_ms:.3},"ttft_ms":{ttft_ms:.3},"ms_per_token":{:.3},"runtime":"raw KFD/AQL, no ROCm"}}}}"#,
                    lane.model_id,
                    lane.node,
                    lane.layers,
                    if steps > 0 {
                        wall_ms / steps as f64
                    } else {
                        0.0
                    }
                );
                write_sse_data(stream, &payload)?;
                break;
            }
            crate::olmo_lane::OlmoEvent::Error(err) => {
                let payload = format!(
                    r#"{{"error":{{"message":"{}","type":"mainarch_olmo_error","code":"decode_failed"}}}}"#,
                    json_escape(&err)
                );
                write_sse_data(stream, &payload)?;
                break;
            }
        }
    }
    write_sse_data(stream, "[DONE]")?;
    Ok(())
}

/// `/api/health`, told the truth about which lane is actually serving.
fn olmo_aware_health_json(
    live_lane: &LiveLane,
    olmo: Option<&crate::olmo_lane::OlmoLane>,
) -> String {
    let base = health_json(live_lane);
    let Some(lane) = olmo else {
        return base;
    };
    let trimmed = base.trim_end().trim_end_matches('}').trim_end();
    format!(
        "{trimmed},\n  \"chat_backend\": \"olmo2-real-weights\",\n  \"chat_synthetic\": false,\n  \
\"olmo\": {{ \"model\": \"{}\", \"node\": {}, \"layers\": {}, \"hidden\": {}, \"vocab\": {}, \
\"device_bytes\": {}, \"max_seq\": {}, \"ready\": {}, \"completed\": {}, \"queue_full\": {} }}\n}}",
        lane.model_id,
        lane.node,
        lane.layers,
        lane.hidden,
        lane.vocab,
        lane.device_bytes,
        lane.max_seq,
        lane.is_ready(),
        lane.completed_count(),
        lane.queue_full_count()
    )
}

/// `/v1/models`, advertising the real model when one is loaded.
fn olmo_aware_models_json(olmo: Option<&crate::olmo_lane::OlmoLane>) -> String {
    let Some(lane) = olmo else {
        return models_json();
    };
    format!(
        r#"{{
  "object": "list",
  "data": [
    {{
      "id": "{}",
      "object": "model",
      "created": 0,
      "owned_by": "allenai",
      "mainarch": {{
        "backend": "olmo2-real-weights",
        "synthetic": false,
        "node": {},
        "layers": {},
        "hidden": {},
        "vocab": {},
        "device_bytes": {},
        "max_seq": {},
        "runtime": "raw KFD/AQL, no ROCm, no HIP, no HSA runtime",
        "attention": "multi-head over paged FP4 KV",
        "norm": "post-norm, QK-norm over the whole projection",
        "mlp": "dense SwiGLU",
        "prefill": "decode loop run once per prompt token"
      }}
    }}
  ]
}}"#,
        lane.model_id,
        lane.node,
        lane.layers,
        lane.hidden,
        lane.vocab,
        lane.device_bytes,
        lane.max_seq
    )
}

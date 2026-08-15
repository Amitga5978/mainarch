//! A persistent serving lane backed by a real OLMo 2 checkpoint.
//!
//! This is the lane that makes `/v1/chat/completions` mean something. One
//! worker thread owns the GPU device and the resident model for the life of the
//! process, because loading 2.77 GiB of weights per request would dominate
//! everything else. Requests arrive on a bounded channel and are served one at
//! a time, which is honest for a single-sequence KV cache: there is no batching
//! here and pretending otherwise would be a lie the scheduler could not keep.
//!
//! It sits alongside the synthetic proof lane rather than replacing it. When no
//! checkpoint is configured the server behaves exactly as before.

use anyhow::{anyhow, Result};
use mainarch_core as mcore;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

/// Where the checkpoint lives and how much of it to run.
#[derive(Debug, Clone)]
pub struct OlmoLaneConfig {
    pub node: u32,
    pub config: PathBuf,
    pub index: PathBuf,
    pub tokenizer: PathBuf,
    pub max_seq: u32,
    pub layers: Option<usize>,
}

/// One streamed piece of a response.
pub enum OlmoEvent {
    /// Decoded text for one generated token.
    Token {
        index: usize,
        text: String,
    },
    Done {
        generated: usize,
        prompt_tokens: usize,
        wall_ms: f64,
        ttft_ms: f64,
    },
    Error(String),
}

struct OlmoRequest {
    prompt: String,
    max_new: usize,
    reply: mpsc::Sender<OlmoEvent>,
}

/// A live OLMo 2 serving lane.
pub struct OlmoLane {
    tx: mpsc::SyncSender<OlmoRequest>,
    pub model_id: String,
    pub node: u32,
    pub layers: usize,
    pub hidden: u64,
    pub vocab: u64,
    pub device_bytes: u64,
    pub max_seq: u32,
    ready: Arc<AtomicBool>,
    completed: Arc<AtomicUsize>,
    queue_full: Arc<AtomicUsize>,
}

impl OlmoLane {
    /// Start the worker and block until the model is resident.
    ///
    /// The device is opened inside the worker thread rather than handed to it,
    /// because `GpuDevice` owns raw pointers and is deliberately not `Send`.
    /// Startup still blocks on a readiness handshake, so a bad checkpoint fails
    /// here instead of on the first request.
    pub fn start(cfg: OlmoLaneConfig) -> Result<Arc<Self>> {
        let (tx, rx) = mpsc::sync_channel::<OlmoRequest>(8);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(usize, u64, u64, u64), String>>();
        let ready = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicUsize::new(0));
        let queue_full = Arc::new(AtomicUsize::new(0));
        let worker_completed = Arc::clone(&completed);
        let worker_ready = Arc::clone(&ready);
        let worker_cfg = cfg.clone();

        std::thread::spawn(move || {
            let boot = (|| -> Result<_> {
                let tokenizer =
                    tokenizers::Tokenizer::from_file(&worker_cfg.tokenizer).map_err(|err| {
                        anyhow!(
                            "loading tokenizer {} failed: {err}",
                            worker_cfg.tokenizer.display()
                        )
                    })?;
                let mut dev = mcore::GpuDevice::open(worker_cfg.node)?;
                let weights = mcore::olmo2::load_olmo2_weights(
                    &mut dev,
                    &worker_cfg.config,
                    &worker_cfg.index,
                    worker_cfg.layers,
                )?;
                let layers = weights.layers.len();
                let hidden = weights.config.hidden_size;
                let vocab = weights.config.vocab_size;
                let device_bytes = weights.device_bytes;
                let runner = mcore::olmo2::Olmo2Runner::new(&mut dev, weights, worker_cfg.max_seq)?;
                Ok((tokenizer, dev, runner, layers, hidden, vocab, device_bytes))
            })();

            let (tokenizer, mut dev, mut runner) = match boot {
                Ok((tk, dev, runner, layers, hidden, vocab, device_bytes)) => {
                    let _ = ready_tx.send(Ok((layers, hidden, vocab, device_bytes)));
                    worker_ready.store(true, Ordering::Release);
                    (tk, dev, runner)
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("{err:#}")));
                    return;
                }
            };

            while let Ok(req) = rx.recv() {
                let started = Instant::now();
                let encoding = match tokenizer.encode(req.prompt.as_str(), false) {
                    Ok(e) => e,
                    Err(err) => {
                        let _ = req
                            .reply
                            .send(OlmoEvent::Error(format!("tokenization failed: {err}")));
                        continue;
                    }
                };
                let ids = encoding.get_ids().to_vec();
                if ids.is_empty() {
                    let _ = req
                        .reply
                        .send(OlmoEvent::Error("prompt encoded to zero tokens".into()));
                    continue;
                }

                // Prefill: run the decode path once per prompt token so the KV
                // cache is built by the same code that generates.
                let mut pos = 0u32;
                let mut next = 0u32;
                let mut failed = false;
                for &t in &ids {
                    match runner.step(&mut dev, t, pos) {
                        Ok(n) => next = n,
                        Err(err) => {
                            let _ = req.reply.send(OlmoEvent::Error(format!("{err:#}")));
                            failed = true;
                            break;
                        }
                    }
                    pos += 1;
                }
                if failed {
                    continue;
                }

                let eos = runner.eos_token();
                let mut ttft_ms = 0.0f64;
                let mut generated = 0usize;
                for i in 0..req.max_new {
                    if i == 0 {
                        ttft_ms = started.elapsed().as_secs_f64() * 1e3;
                    }
                    // Stop before emitting the sentinel, not after.
                    if Some(next) == eos {
                        break;
                    }
                    let piece = tokenizer
                        .decode(&[next], false)
                        .unwrap_or_else(|_| String::new());
                    if req
                        .reply
                        .send(OlmoEvent::Token {
                            index: i,
                            text: piece,
                        })
                        .is_err()
                    {
                        break; // client hung up
                    }
                    generated += 1;
                    if pos >= worker_cfg.max_seq {
                        break;
                    }
                    match runner.step(&mut dev, next, pos) {
                        Ok(n) => next = n,
                        Err(err) => {
                            let _ = req.reply.send(OlmoEvent::Error(format!("{err:#}")));
                            failed = true;
                            break;
                        }
                    }
                    pos += 1;
                }
                if failed {
                    continue;
                }
                worker_completed.fetch_add(1, Ordering::AcqRel);
                let _ = req.reply.send(OlmoEvent::Done {
                    generated,
                    prompt_tokens: ids.len(),
                    wall_ms: started.elapsed().as_secs_f64() * 1e3,
                    ttft_ms,
                });
            }
        });

        let (layers, hidden, vocab, device_bytes) = ready_rx
            .recv()
            .map_err(|_| anyhow!("olmo lane worker exited before reporting readiness"))?
            .map_err(|err| anyhow!("olmo lane failed to start: {err}"))?;

        Ok(Arc::new(Self {
            tx,
            model_id: "olmo-2-0425-1b".to_string(),
            node: cfg.node,
            layers,
            hidden,
            vocab,
            device_bytes,
            max_seq: cfg.max_seq,
            ready,
            completed,
            queue_full,
        }))
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    pub fn completed_count(&self) -> usize {
        self.completed.load(Ordering::Acquire)
    }

    pub fn queue_full_count(&self) -> usize {
        self.queue_full.load(Ordering::Acquire)
    }

    /// Submit a prompt. The receiver yields tokens as they are generated.
    pub fn submit(&self, prompt: String, max_new: usize) -> Result<mpsc::Receiver<OlmoEvent>> {
        let (reply, rx) = mpsc::channel();
        self.tx
            .try_send(OlmoRequest {
                prompt,
                max_new,
                reply,
            })
            .map_err(|err| {
                self.queue_full.fetch_add(1, Ordering::AcqRel);
                anyhow!("olmo lane is busy: {err}")
            })?;
        Ok(rx)
    }
}

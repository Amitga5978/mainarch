//! OLMo 2 checkpoint support: a worked example of adding a model architecture.
//!
//! This module exists as much to be read as to be run. The rest of this crate
//! grew up around Qwen3, so OLMo 2 is the first architecture that had to be
//! *added*, and the three places it disagrees with Qwen3 are exactly the three
//! places a new architecture usually disagrees with the one you built first.
//!
//! # What is different, and why the runtime cares
//!
//! **Post-norm instead of pre-norm.** Qwen3 normalises the hidden state on the
//! way *into* attention and the MLP. OLMo 2 runs the sub-layer first and
//! normalises its output on the residual branch, so the layer reads
//!
//! ```text
//!   h = h + post_attention_layernorm(attention(h))
//!   h = h + post_feedforward_layernorm(mlp(h))
//! ```
//!
//! You can see this in the checkpoint without reading a line of modelling code.
//! A pre-norm model ships `input_layernorm`; OLMo 2 ships
//! `post_attention_layernorm` and `post_feedforward_layernorm` and no
//! `input_layernorm` at all. [`preflight_olmo2_checkpoint`] asserts that
//! absence, because a checkpoint carrying `input_layernorm` is not the
//! architecture this path implements and the failure should be loud and early
//! rather than a wrong number a thousand kernels later.
//!
//! **QK-norm over the whole projection.** Qwen3 normalises each attention head
//! independently, so its `q_norm` has `head_dim` elements. OLMo 2 normalises
//! across the entire projection, so its `q_norm` has `num_attention_heads *
//! head_dim` elements and its `k_norm` has `num_key_value_heads * head_dim`.
//! Same operation, different reduction width, and getting it wrong produces
//! plausible-looking garbage rather than an error.
//!
//! **Multi-head attention, not grouped-query.** OLMo 2 sets
//! `num_key_value_heads == num_attention_heads`, so every query head owns its
//! own KV head. In grouped-query terms that is a group size of one, which
//! matters because the decode attention kernel reads the KV cache once per
//! *group*, and a group of one removes the sharing the GQA kernel is built
//! around.
//!
//! # What is the same
//!
//! RMSNorm, rotary position embeddings, SwiGLU, and the `[out, in]` row-major
//! weight layout are all shared with the Qwen3 path, which is why the tensor
//! names for the projections are identical and
//! [`crate::weights::qwen_weight_placement`]'s Megatron-style rules carry over.

use crate::weights::{
    required_json_str, required_json_u64, JsonParser, QwenWeightPlacement, SafetensorsIndex,
};
use anyhow::{anyhow, Context, Result};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

/// Head dimension every decode attention kernel in this crate is built for.
///
/// This is not a tunable. The attention kernels address the KV cache with
/// 128-bit loads over a 128-wide head and map lanes accordingly, so a
/// checkpoint with a different head dimension needs a new kernel, not a new
/// constant. Preflight refuses rather than letting that surface as a wrong
/// answer.
pub const REQUIRED_HEAD_DIM: u64 = 128;

/// The eleven tensors every OLMo 2 decoder layer must provide.
pub const OLMO2_LAYER_TENSOR_SUFFIXES: [&str; 11] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
];

/// A tensor name that must NOT appear. Its presence means the checkpoint is
/// pre-norm, which is a different architecture from the one this path runs.
pub const PRE_NORM_MARKER_SUFFIX: &str = "input_layernorm.weight";

/// The fields of `config.json` this runtime actually depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Olmo2Config {
    pub model_type: String,
    pub num_hidden_layers: u64,
    pub hidden_size: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub rope_theta: u64,
    pub tie_word_embeddings: bool,
    /// End-of-sequence token, when config.json declares one.
    pub eos_token_id: Option<u64>,
}

impl Olmo2Config {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text, path)
    }

    /// Split out from [`Olmo2Config::open`] so the parse can be tested without
    /// putting a file on disk.
    pub fn parse(text: &str, path: &Path) -> Result<Self> {
        let root = JsonParser::new(text).parse()?;
        let obj = root.as_object("OLMo 2 config")?;
        let model_type = required_json_str(obj, "model_type", path)?.to_string();
        let num_attention_heads = required_json_u64(obj, "num_attention_heads", path)?;
        Ok(Self {
            model_type,
            num_hidden_layers: required_json_u64(obj, "num_hidden_layers", path)?,
            hidden_size: required_json_u64(obj, "hidden_size", path)?,
            num_attention_heads,
            num_key_value_heads: obj
                .get("num_key_value_heads")
                .map(|v| v.as_u64("num_key_value_heads"))
                .transpose()?
                .unwrap_or(num_attention_heads),
            intermediate_size: required_json_u64(obj, "intermediate_size", path)?,
            vocab_size: required_json_u64(obj, "vocab_size", path)?,
            rope_theta: obj
                .get("rope_theta")
                .map(|v| v.as_u64("rope_theta"))
                .transpose()?
                .unwrap_or(10_000),
            tie_word_embeddings: obj
                .get("tie_word_embeddings")
                .map(|v| v.as_bool("tie_word_embeddings"))
                .transpose()?
                .unwrap_or(false),
            eos_token_id: obj
                .get("eos_token_id")
                .map(|v| v.as_u64("eos_token_id"))
                .transpose()?,
        })
    }

    /// OLMo 2's config has no explicit `head_dim`, so it is derived.
    pub fn head_dim(&self) -> Result<u64> {
        if self.num_attention_heads == 0 {
            return Err(anyhow!("num_attention_heads must be non-zero"));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(anyhow!(
                "hidden_size={} is not divisible by num_attention_heads={}",
                self.hidden_size,
                self.num_attention_heads
            ));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }

    /// Query heads per KV head. One means multi-head attention.
    pub fn q_heads_per_kv(&self) -> Result<u64> {
        if self.num_key_value_heads == 0 {
            return Err(anyhow!("num_key_value_heads must be non-zero"));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(anyhow!(
                "num_attention_heads={} is not divisible by num_key_value_heads={}",
                self.num_attention_heads,
                self.num_key_value_heads
            ));
        }
        Ok(self.num_attention_heads / self.num_key_value_heads)
    }

    /// Elements in `q_norm`. OLMo 2 normalises the whole projection, so this is
    /// the full query width rather than one head.
    pub fn q_norm_elems(&self) -> Result<u64> {
        Ok(self.num_attention_heads * self.head_dim()?)
    }

    /// Elements in `k_norm`, the full key width.
    pub fn k_norm_elems(&self) -> Result<u64> {
        Ok(self.num_key_value_heads * self.head_dim()?)
    }
}

/// Serving placement for OLMo 2 tensor names.
///
/// The projections follow the same Megatron-style split as the Qwen3 path, so
/// that logic is reused rather than duplicated. The one OLMo 2 specific rule is
/// QK-norm: because `q_norm` and `k_norm` span the whole projection rather than
/// a single head, they have to be sliced along with the projection they
/// normalise instead of being replicated the way a per-head norm would be.
pub fn olmo2_weight_placement(name: &str) -> QwenWeightPlacement {
    if name.ends_with(".self_attn.q_norm.weight") || name.ends_with(".self_attn.k_norm.weight") {
        return QwenWeightPlacement::TensorParallel { axis: 0 };
    }
    crate::weights::qwen_weight_placement(name)
}

/// What preflight established about a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Olmo2CheckpointPreflight {
    pub model_type: String,
    pub num_hidden_layers: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub head_dim: u64,
    pub q_heads_per_kv: u64,
    pub is_multi_head: bool,
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub vocab_size: u64,
    pub rope_theta: u64,
    pub tie_word_embeddings: bool,
    pub q_norm_elems: u64,
    pub k_norm_elems: u64,
    pub tp_world: usize,
    pub tensor_count: usize,
    pub tensor_parallel_tensors: usize,
    pub replicated_tensors: usize,
    pub lm_head_present: bool,
    pub indexed_bytes: Option<u64>,
}

/// Validate an OLMo 2 checkpoint against what this runtime can actually execute.
///
/// CPU-only. Touches no GPU and allocates no device memory.
pub fn preflight_olmo2_checkpoint(
    config_path: impl AsRef<Path>,
    index_path: impl AsRef<Path>,
    tp_world: usize,
) -> Result<Olmo2CheckpointPreflight> {
    let config_path = config_path.as_ref();
    let index = SafetensorsIndex::open(index_path)?;
    if index.weight_map.is_empty() {
        return Err(anyhow!("{} has an empty weight_map", index.path.display()));
    }
    let names = index
        .weight_map
        .keys()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let indexed_bytes = index
        .metadata
        .get("total_size")
        .map(|s| {
            s.parse::<u64>()
                .with_context(|| format!("parsing {} metadata.total_size", index.path.display()))
        })
        .transpose()?;
    let config = Olmo2Config::open(config_path)?;
    preflight_olmo2_names(&config, &names, indexed_bytes, tp_world)
}

/// The checkable core, separated from file IO so it can be exercised against a
/// synthetic name set with no checkpoint present.
pub fn preflight_olmo2_names(
    config: &Olmo2Config,
    names: &HashSet<&str>,
    indexed_bytes: Option<u64>,
    tp_world: usize,
) -> Result<Olmo2CheckpointPreflight> {
    if tp_world == 0 {
        return Err(anyhow!("tensor-parallel world must be non-zero"));
    }
    if config.model_type != "olmo2" {
        return Err(anyhow!(
            "config model_type={} is not olmo2; this path implements OLMo 2's post-norm layer, \
             not a pre-norm Llama/Qwen layer",
            config.model_type
        ));
    }

    let head_dim = config.head_dim()?;
    if head_dim != REQUIRED_HEAD_DIM {
        return Err(anyhow!(
            "head_dim={head_dim} but every decode attention kernel in this crate is built for \
             head_dim={REQUIRED_HEAD_DIM}; this needs a new kernel, not a new parameter"
        ));
    }
    let q_heads_per_kv = config.q_heads_per_kv()?;

    if config.num_attention_heads % tp_world as u64 != 0 {
        return Err(anyhow!(
            "num_attention_heads={} is not divisible by tp_world={tp_world}",
            config.num_attention_heads
        ));
    }
    if config.num_key_value_heads % tp_world as u64 != 0 {
        return Err(anyhow!(
            "num_key_value_heads={} is not divisible by tp_world={tp_world}; OLMo 2 is multi-head, \
             so KV heads cannot be replicated across ranks the way a grouped-query model allows",
            config.num_key_value_heads
        ));
    }

    // The architectural assertion. A pre-norm checkpoint must not reach the
    // post-norm layer forward.
    if let Some(bad) = names
        .iter()
        .find(|n| n.ends_with(PRE_NORM_MARKER_SUFFIX))
        .copied()
    {
        return Err(anyhow!(
            "checkpoint contains {bad}, which means it is a pre-norm architecture; OLMo 2 \
             normalises on the residual branch and ships post_attention_layernorm plus \
             post_feedforward_layernorm instead"
        ));
    }

    // Every layer must be complete. A missing tensor here becomes a wrong
    // number much later, so it is worth the exhaustive check.
    let mut missing = Vec::new();
    for layer in 0..config.num_hidden_layers {
        for suffix in OLMO2_LAYER_TENSOR_SUFFIXES {
            let want = format!("model.layers.{layer}.{suffix}");
            if !names.contains(want.as_str()) {
                missing.push(want);
            }
        }
    }
    for global in ["model.embed_tokens.weight", "model.norm.weight"] {
        if !names.contains(global) {
            missing.push(global.to_string());
        }
    }
    let lm_head_present = names.contains("lm_head.weight");
    if !lm_head_present && !config.tie_word_embeddings {
        missing.push("lm_head.weight".to_string());
    }
    if !missing.is_empty() {
        let shown = missing.iter().take(8).cloned().collect::<Vec<_>>();
        return Err(anyhow!(
            "checkpoint is missing {} required tensor(s), first {}: {}",
            missing.len(),
            shown.len(),
            shown.join(", ")
        ));
    }

    let mut tensor_parallel_tensors = 0usize;
    let mut replicated_tensors = 0usize;
    for name in names {
        match olmo2_weight_placement(name) {
            QwenWeightPlacement::TensorParallel { .. } => tensor_parallel_tensors += 1,
            _ => replicated_tensors += 1,
        }
    }

    Ok(Olmo2CheckpointPreflight {
        model_type: config.model_type.clone(),
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim,
        q_heads_per_kv,
        is_multi_head: q_heads_per_kv == 1,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        rope_theta: config.rope_theta,
        tie_word_embeddings: config.tie_word_embeddings,
        q_norm_elems: config.q_norm_elems()?,
        k_norm_elems: config.k_norm_elems()?,
        tp_world,
        tensor_count: names.len(),
        tensor_parallel_tensors,
        replicated_tensors,
        lm_head_present,
        indexed_bytes,
    })
}

/// A synthetic OLMo 2 config shaped like OLMo-2-0425-1B but two layers deep,
/// used by the selftest so CI can exercise preflight with no checkpoint.
pub fn synthetic_olmo2_config(num_hidden_layers: u64) -> Olmo2Config {
    Olmo2Config {
        model_type: "olmo2".to_string(),
        num_hidden_layers,
        hidden_size: 2048,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        intermediate_size: 8192,
        vocab_size: 100_352,
        rope_theta: 500_000,
        tie_word_embeddings: false,
        eos_token_id: Some(100_257),
    }
}

/// Every tensor name a complete checkpoint for `config` would carry.
pub fn synthetic_olmo2_tensor_names(config: &Olmo2Config) -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    if !config.tie_word_embeddings {
        names.push("lm_head.weight".to_string());
    }
    for layer in 0..config.num_hidden_layers {
        for suffix in OLMO2_LAYER_TENSOR_SUFFIXES {
            names.push(format!("model.layers.{layer}.{suffix}"));
        }
    }
    names
}

/// CPU-only selftest. Proves preflight accepts a well-formed OLMo 2 checkpoint
/// and, more usefully, that it rejects each way one can be wrong.
pub fn selftest_olmo2_preflight() -> Result<()> {
    println!("mainarch olmo2-preflight-selftest: CPU-only OLMo 2 checkpoint contract gate");

    let config = synthetic_olmo2_config(2);
    let owned = synthetic_olmo2_tensor_names(&config);
    let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
    let ok = preflight_olmo2_names(&config, &names, Some(4_096), 1)?;
    if !ok.is_multi_head {
        return Err(anyhow!(
            "OLMo 2 is multi-head; preflight reported q_heads_per_kv={}",
            ok.q_heads_per_kv
        ));
    }
    if ok.head_dim != REQUIRED_HEAD_DIM {
        return Err(anyhow!("expected head_dim={REQUIRED_HEAD_DIM}"));
    }
    if ok.q_norm_elems != 2048 || ok.k_norm_elems != 2048 {
        return Err(anyhow!(
            "OLMo 2 QK-norm spans the whole projection; got q={} k={}",
            ok.q_norm_elems,
            ok.k_norm_elems
        ));
    }
    println!(
        "  accepted: layers={} heads={}/{} head_dim={} q_heads_per_kv={} (multi-head) \
hidden={} inter={} vocab={} rope_theta={} qk_norm={}/{} tensors={}",
        ok.num_hidden_layers,
        ok.num_attention_heads,
        ok.num_key_value_heads,
        ok.head_dim,
        ok.q_heads_per_kv,
        ok.hidden_size,
        ok.intermediate_size,
        ok.vocab_size,
        ok.rope_theta,
        ok.q_norm_elems,
        ok.k_norm_elems,
        ok.tensor_count
    );

    // A pre-norm checkpoint must be refused, loudly.
    let mut pre_norm = owned.clone();
    pre_norm.push("model.layers.0.input_layernorm.weight".to_string());
    let pre_norm_names = pre_norm.iter().map(String::as_str).collect::<HashSet<_>>();
    let err = preflight_olmo2_names(&config, &pre_norm_names, None, 1)
        .err()
        .ok_or_else(|| anyhow!("preflight accepted a pre-norm checkpoint"))?;
    if !format!("{err}").contains("pre-norm") {
        return Err(anyhow!("pre-norm rejection did not explain itself: {err}"));
    }
    println!("  rejected pre-norm checkpoint: {err}");

    // An incomplete layer must be refused.
    let truncated = owned
        .iter()
        .filter(|n| n.as_str() != "model.layers.1.self_attn.k_norm.weight")
        .cloned()
        .collect::<Vec<_>>();
    let truncated_names = truncated.iter().map(String::as_str).collect::<HashSet<_>>();
    let err = preflight_olmo2_names(&config, &truncated_names, None, 1)
        .err()
        .ok_or_else(|| anyhow!("preflight accepted a checkpoint with a missing tensor"))?;
    println!("  rejected incomplete layer: {err}");

    // A head dimension the kernels cannot run must be refused.
    let mut wrong_head_dim = synthetic_olmo2_config(2);
    wrong_head_dim.num_attention_heads = 32; // 2048 / 32 = 64
    wrong_head_dim.num_key_value_heads = 32;
    let err = preflight_olmo2_names(&wrong_head_dim, &names, None, 1)
        .err()
        .ok_or_else(|| anyhow!("preflight accepted head_dim=64"))?;
    if !format!("{err}").contains("head_dim") {
        return Err(anyhow!("head_dim rejection did not explain itself: {err}"));
    }
    println!("  rejected head_dim=64: {err}");

    // A non-OLMo config must be refused.
    let mut wrong_arch = synthetic_olmo2_config(2);
    wrong_arch.model_type = "qwen3".to_string();
    let err = preflight_olmo2_names(&wrong_arch, &names, None, 1)
        .err()
        .ok_or_else(|| anyhow!("preflight accepted model_type=qwen3"))?;
    println!("  rejected model_type=qwen3: {err}");

    println!("  olmo2 preflight selftest ok: 1 accept, 4 rejects");
    Ok(())
}

// ---------------------------------------------------------------------------
// Hardware gate for the fused QK-norm + RoPE + paged FP4 KV append kernel.
// ---------------------------------------------------------------------------

use crate::gemm::{f16_to_f32, f32_to_f16};
use crate::gpu::GpuDevice;
use std::time::Duration;

/// Query heads in OLMo-2-0425-1B.
pub const OLMO2_1B_Q_HEADS: usize = 16;
/// KV heads. Equal to the query head count, which is what makes it multi-head.
pub const OLMO2_1B_KV_HEADS: usize = 16;
/// Head dimension, and the only one the attention kernels implement.
pub const OLMO2_HEAD_DIM: usize = 128;

/// Validate OLMo 2's fused QK-norm + RoPE against an f64 reference, and prove
/// the normalisation really does span the whole projection.
///
/// The second half matters more than the first. A per-head norm and a
/// whole-projection norm produce numbers that look equally plausible, and the
/// only structural difference between them is *coupling*: under a
/// whole-projection norm every head shares one RMS, so changing the input of
/// head 5 must move the output of head 0. Under a per-head norm it cannot.
/// That single observation separates the two architectures, so the gate makes
/// it rather than trusting the arithmetic to be self-evidently right.
pub fn check_olmo2_qk_rope_on(dev: &mut GpuDevice, theta: f32, eps: f32, pos: u32) -> Result<()> {
    let qh = OLMO2_1B_Q_HEADS;
    let kvh = OLMO2_1B_KV_HEADS;
    let d = OLMO2_HEAD_DIM;
    let half_d = d / 2;
    let q_elems = qh * d;
    let k_elems = kvh * d;
    let block_size = 16usize;
    let physical_blocks = 4usize;
    let rows_per_head = physical_blocks * block_size;

    let mut q_src = dev.alloc(q_elems * 4)?;
    let mut q_w = dev.alloc(q_elems * 2)?;
    let mut q_dst = dev.alloc(q_elems * 2)?;
    let mut k_src = dev.alloc(k_elems * 4)?;
    let mut k_w = dev.alloc(k_elems * 2)?;
    let mut v_src = dev.alloc(k_elems * 4)?;
    let kcache = dev.alloc_device(kvh * rows_per_head * 64)?;
    let vcache = dev.alloc_device(kvh * rows_per_head * 64)?;
    let scale_k = dev.alloc_device(kvh * rows_per_head * 4)?;
    let scale_v = dev.alloc_device(kvh * rows_per_head * 4)?;
    let mut indices = dev.alloc(physical_blocks * 4)?;
    let mut indptr = dev.alloc(2 * 4)?;
    let mut last_page_len = dev.alloc(4)?;
    let mut batch_indices = dev.alloc(4)?;
    let mut positions = dev.alloc(4)?;

    // Addresses are captured up front so the buffers stay free to be refilled
    // between the two runs.
    let q_src_va = q_src.va();
    let q_w_va = q_w.va();
    let q_dst_va = q_dst.va();
    let k_src_va = k_src.va();
    let k_w_va = k_w.va();
    let v_src_va = v_src.va();
    let kcache_va = kcache.va();
    let vcache_va = vcache.va();
    let scale_k_va = scale_k.va();
    let scale_v_va = scale_v.va();
    let indices_va = indices.va();
    let indptr_va = indptr.va();
    let last_page_len_va = last_page_len.va();
    let batch_indices_va = batch_indices.va();
    let positions_va = positions.va();

    fn gen(i: usize, s: usize) -> f32 {
        let x = i
            .wrapping_mul(2_654_435_761)
            .wrapping_add(s.wrapping_mul(40_503));
        (((x >> 11) & 0x3ff) as f32) / 1024.0 - 0.5
    }

    // SAFETY: every buffer below was allocated by this function at the
    // element count being written, and no dispatch is in flight yet, so the host
    // has exclusive access to the mapping.
    unsafe {
        let idx = indices.as_mut_slice_of::<u32>();
        for (b, slot) in idx.iter_mut().enumerate().take(physical_blocks) {
            *slot = b as u32;
        }
        let ip = indptr.as_mut_slice_of::<u32>();
        ip[0] = 0;
        ip[1] = physical_blocks as u32;
        last_page_len.as_mut_slice_of::<u32>()[0] = block_size as u32;
        batch_indices.as_mut_slice_of::<u32>()[0] = 0;
        positions.as_mut_slice_of::<u32>()[0] = pos;
        let ks = k_src.as_mut_slice_of::<f32>();
        let kw = k_w.as_mut_slice_of::<u16>();
        let vs = v_src.as_mut_slice_of::<f32>();
        for e in 0..k_elems {
            ks[e] = gen(e, 17);
            kw[e] = f32_to_f16(0.5 + gen(e, 31) * 0.25);
            vs[e] = gen(e, 19);
        }
        let qw = q_w.as_mut_slice_of::<u16>();
        for e in 0..q_elems {
            qw[e] = f32_to_f16(0.5 + gen(e, 29) * 0.25);
        }
    }

    // `perturb_head` adds a large offset to one head's input. Under a shared
    // RMS that has to move every other head's output too.
    fn write_q(qs: &mut [f32], q_elems: usize, d: usize, perturb_head: Option<usize>) {
        for (e, slot) in qs.iter_mut().enumerate().take(q_elems) {
            let mut val = gen(e, 13);
            if perturb_head == Some(e / d) {
                val += 4.0;
            }
            *slot = val;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        dev: &mut GpuDevice,
        vas: [u64; 15],
        physical_blocks: u32,
        block_size: u32,
        theta: f32,
        eps: f32,
    ) -> Result<()> {
        // NOTE: no chain_next() here. chain_next() clears the completion signal
        // so several dispatches can be chained and waited on once. This gate
        // issues a single dispatch, so it must signal, or wait() would return
        // on a stale value and we would read the buffer before the GPU wrote it.
        dev.arm_cast_qk_rope_append_paged_fp4_vf32_q16_k16_d128_olmo2_meta(
            vas[0],
            vas[1],
            vas[2],
            vas[3],
            vas[4],
            vas[5],
            vas[6],
            vas[7],
            vas[8],
            vas[9],
            vas[10],
            vas[11],
            vas[12],
            vas[13],
            vas[14],
            physical_blocks,
            physical_blocks,
            block_size,
            theta,
            eps,
        )?;
        dev.wait(Duration::from_secs(20))?;
        Ok(())
    }

    let vas = [
        q_src_va,
        q_w_va,
        q_dst_va,
        k_src_va,
        k_w_va,
        v_src_va,
        kcache_va,
        vcache_va,
        scale_k_va,
        scale_v_va,
        indices_va,
        indptr_va,
        last_page_len_va,
        batch_indices_va,
        positions_va,
    ];

    // --- part one: agree with an f64 reference -----------------------------
    // SAFETY: `q_src` was allocated as `q_elems * 4` bytes above and no
    // dispatch is in flight, so a host-side f32 view of exactly `q_elems` is in
    // bounds and unaliased.
    unsafe { write_q(q_src.as_mut_slice_of::<f32>(), q_elems, d, None) };
    dispatch(
        dev,
        vas,
        physical_blocks as u32,
        block_size as u32,
        theta,
        eps,
    )?;
    // SAFETY: `dev.wait` returned, so the dispatch has retired and the host
    // may read `q_dst`, which was allocated as `q_elems` f16 elements.
    let baseline = unsafe { q_dst.as_mut_slice_of::<u16>()[..q_elems].to_vec() };
    // SAFETY: both buffers were allocated at these element counts and no
    // dispatch is in flight; the reads are copied out immediately.
    let (qs_host, qw_host) = unsafe {
        (
            q_src.as_mut_slice_of::<f32>()[..q_elems].to_vec(),
            q_w.as_mut_slice_of::<u16>()[..q_elems].to_vec(),
        )
    };

    // The kernel rounds each source element through f16 before squaring, so the
    // reference has to as well or the RMS will not match.
    let mut ss = 0.0f64;
    for &raw in qs_host.iter().take(q_elems) {
        let v = f16_to_f32(f32_to_f16(raw)) as f64;
        ss += v * v;
    }
    let rms = 1.0f64 / (ss / q_elems as f64 + eps as f64).sqrt();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for g in 0..qh {
        for i in 0..half_d {
            let base = g * d + i;
            let a = f16_to_f32(f32_to_f16(qs_host[base])) as f64
                * rms
                * f16_to_f32(qw_host[base]) as f64;
            let b = f16_to_f32(f32_to_f16(qs_host[base + half_d])) as f64
                * rms
                * f16_to_f32(qw_host[base + half_d]) as f64;
            let freq = (theta as f64).powf(-(i as f64) / half_d as f64);
            let ang = pos as f64 * freq;
            let (s, c) = ang.sin_cos();
            let want_lo = a * c - b * s;
            let want_hi = b * c + a * s;
            let got_lo = f16_to_f32(baseline[base]) as f64;
            let got_hi = f16_to_f32(baseline[base + half_d]) as f64;
            num += (want_lo - got_lo).powi(2) + (want_hi - got_hi).powi(2);
            den += want_lo * want_lo + want_hi * want_hi;
        }
    }
    if den == 0.0 {
        return Err(anyhow!(
            "reference produced an all-zero query, so the gate is vacuous"
        ));
    }
    let rel_l2 = (num / den).sqrt();
    // f16 storage plus the hardware's fast sin/cos put the floor near 1e-3.
    if !(rel_l2 < 5e-3) {
        let nonzero = baseline.iter().filter(|w| **w != 0).count();
        let sample: Vec<String> = (0..6)
            .map(|i| format!("{:.5}", f16_to_f32(baseline[i])))
            .collect();
        let want: Vec<String> = (0..6)
            .map(|i| {
                let a =
                    f16_to_f32(f32_to_f16(qs_host[i])) as f64 * rms * f16_to_f32(qw_host[i]) as f64;
                let b = f16_to_f32(f32_to_f16(qs_host[i + half_d])) as f64
                    * rms
                    * f16_to_f32(qw_host[i + half_d]) as f64;
                let freq = (theta as f64).powf(-(i as f64) / half_d as f64);
                let ang = pos as f64 * freq;
                let (sn, cs) = ang.sin_cos();
                format!("{:.5}", a * cs - b * sn)
            })
            .collect();
        return Err(anyhow!(
            "olmo2 qk-norm+rope rel-L2 {rel_l2:.3e} against the f64 reference exceeds 5e-3 \
(gpu non-zero {nonzero}/{q_elems}, rms={rms:.6}, gpu[0..6]=[{}], ref[0..6]=[{}])",
            sample.join(", "),
            want.join(", ")
        ));
    }

    // --- part two: prove the norm spans the whole projection ---------------
    // SAFETY: as above. The buffer is refilled between runs while the queue
    // is idle.
    unsafe { write_q(q_src.as_mut_slice_of::<f32>(), q_elems, d, Some(5)) };
    dispatch(
        dev,
        vas,
        physical_blocks as u32,
        block_size as u32,
        theta,
        eps,
    )?;
    // SAFETY: `dev.wait` returned, so the second dispatch has retired.
    let perturbed = unsafe { q_dst.as_mut_slice_of::<u16>()[..q_elems].to_vec() };

    if baseline[..d] == perturbed[..d] {
        return Err(anyhow!(
            "perturbing head 5's input left head 0's output unchanged, so this kernel normalises \
             per head rather than across the whole projection; that is Qwen3 semantics, not OLMo 2"
        ));
    }
    if baseline[5 * d..6 * d] == perturbed[5 * d..6 * d] {
        return Err(anyhow!("head 5 did not change when its own input changed"));
    }

    println!(
        "  olmo2-qk-rope: node {}, {qh}Q/{kvh}KV d{d} theta={theta} pos={pos}, rel-L2 \
{rel_l2:.3e} vs f64 reference",
        dev.node_id()
    );
    println!(
        "  olmo2-whole-projection-norm: perturbing head 5's input moved head 0's output, so one \
RMS is shared across all {q_elems} projection elements (a per-head norm cannot do that)"
    );
    Ok(())
}

/// Validate OLMo 2's post-norm residual update against an f64 reference, and
/// prove it is not quietly doing pre-norm.
///
/// Pre-norm and post-norm use the same three inputs and produce residual
/// streams of similar magnitude, so a magnitude check does not separate them.
/// What separates them is *which* value gets normalised:
///
/// ```text
///   pre-norm:   acc + x        then normalise  -> the residual is normalised
///   post-norm:  normalise x    then add to acc -> the residual is not
/// ```
///
/// So the gate computes both references and requires the kernel to match the
/// post-norm one and differ from the pre-norm one. If a future edit swaps the
/// order, the second assertion fires.
pub fn check_olmo2_post_norm_on(dev: &mut GpuDevice, h: usize, eps: f32) -> Result<()> {
    if h == 0 || h % 256 != 0 {
        return Err(anyhow!(
            "post-norm gate wants H a non-zero multiple of 256 (got {h})"
        ));
    }
    let mut acc = dev.alloc(h * 2)?;
    let mut x = dev.alloc(h * 4)?;
    let mut w = dev.alloc(h * 2)?;
    let (acc_va, x_va, w_va) = (acc.va(), x.va(), w.va());

    let gen = |i: usize, s: usize| -> f32 {
        let v = i
            .wrapping_mul(2_246_822_519)
            .wrapping_add(s.wrapping_mul(374_761_393));
        (((v >> 9) & 0x7ff) as f32) / 2048.0 - 0.5
    };

    // SAFETY: all three buffers were allocated at `h` elements of the widths
    // used here, and nothing is dispatched yet.
    let (acc_in, x_in, w_in) = unsafe {
        let a = acc.as_mut_slice_of::<u16>();
        let xs = x.as_mut_slice_of::<f32>();
        let ws = w.as_mut_slice_of::<u16>();
        for i in 0..h {
            a[i] = f32_to_f16(gen(i, 3) * 2.0);
            xs[i] = gen(i, 11);
            ws[i] = f32_to_f16(0.75 + gen(i, 23) * 0.5);
        }
        (a[..h].to_vec(), xs[..h].to_vec(), ws[..h].to_vec())
    };

    dev.arm_add_postnorm(acc_va, x_va, w_va, h as u32, eps)?;
    dev.wait(Duration::from_secs(20))?;
    // SAFETY: `dev.wait` returned, so `acc` holds the kernel's output.
    let got = unsafe { acc.as_mut_slice_of::<u16>()[..h].to_vec() };

    // Post-norm reference: normalise x, scale by w, add into the residual.
    let mut ss = 0.0f64;
    for &v in x_in.iter() {
        ss += (v as f64) * (v as f64);
    }
    let rms_x = 1.0f64 / (ss / h as f64 + eps as f64).sqrt();

    // Pre-norm reference, for contrast only: add first, then normalise the sum.
    let mut ss_pre = 0.0f64;
    for i in 0..h {
        let s = f16_to_f32(acc_in[i]) as f64 + x_in[i] as f64;
        ss_pre += s * s;
    }
    let rms_pre = 1.0f64 / (ss_pre / h as f64 + eps as f64).sqrt();

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut pre_num = 0.0f64;
    for i in 0..h {
        let wf = f16_to_f32(w_in[i]) as f64;
        let want_post = f16_to_f32(acc_in[i]) as f64 + x_in[i] as f64 * rms_x * wf;
        let want_pre = (f16_to_f32(acc_in[i]) as f64 + x_in[i] as f64) * rms_pre * wf;
        let g = f16_to_f32(got[i]) as f64;
        num += (want_post - g).powi(2);
        den += want_post * want_post;
        pre_num += (want_pre - g).powi(2);
    }
    if den == 0.0 {
        return Err(anyhow!("post-norm reference is all zero, gate is vacuous"));
    }
    let rel_post = (num / den).sqrt();
    let rel_pre = (pre_num / den).sqrt();

    if !(rel_post < 5e-3) {
        return Err(anyhow!(
            "add_postnorm_f16 rel-L2 {rel_post:.3e} against the post-norm reference exceeds 5e-3"
        ));
    }
    // If the kernel were pre-norm, rel_pre would be the small one. Require a
    // clear separation so the gate cannot pass both ways.
    if rel_pre < 10.0 * rel_post {
        return Err(anyhow!(
            "add_postnorm_f16 is not clearly distinguishable from pre-norm \
             (post rel-L2 {rel_post:.3e}, pre rel-L2 {rel_pre:.3e}); the gate cannot tell the two \
             orderings apart with these inputs"
        ));
    }

    println!(
        "  olmo2-post-norm: node {}, H={h} eps={eps}, rel-L2 {rel_post:.3e} vs the post-norm \
reference",
        dev.node_id()
    );
    println!(
        "  olmo2-post-norm-ordering: the same output is {rel_pre:.3e} away from the pre-norm \
reference, {:.0}x further, so acc = acc + rmsnorm(x) and not rmsnorm(acc + x)",
        rel_pre / rel_post.max(1e-12)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Real checkpoint -> VRAM.
// ---------------------------------------------------------------------------

/// Device addresses of one OLMo 2 decoder layer's weights.
///
/// Eleven entries, matching the eleven tensors the checkpoint carries per layer.
/// There is no `in_norm` because OLMo 2 is post-norm, and no router because it
/// is dense.
#[derive(Debug, Clone, Copy)]
pub struct Olmo2LayerVas {
    pub wq: u64,
    pub wk: u64,
    pub wv: u64,
    pub q_norm: u64,
    pub k_norm: u64,
    pub wo: u64,
    pub post_attn_norm: u64,
    pub gate: u64,
    pub up: u64,
    pub down: u64,
    pub post_ffn_norm: u64,
}

/// An OLMo 2 checkpoint resident in device memory.
pub struct Olmo2Weights {
    pub config: Olmo2Config,
    pub layers: Vec<Olmo2LayerVas>,
    pub embed: u64,
    pub final_norm: u64,
    pub lm_head: u64,
    /// f16 bytes actually uploaded.
    pub device_bytes: u64,
    /// f32 bytes read from the checkpoint.
    pub source_bytes: u64,
    /// Buffers are owned here so the addresses above stay valid.
    _keep: Vec<crate::DeviceBuffer>,
}

/// Read one tensor, convert f32 to f16, and upload it.
///
/// OLMo 2 ships fp32 weights and every kernel on this path consumes `half`, so
/// the conversion happens once here rather than on every token.
fn upload_f16(
    dev: &mut GpuDevice,
    shard: &crate::weights::SafetensorsShard,
    name: &str,
    expect_elems: usize,
    keep: &mut Vec<crate::DeviceBuffer>,
    source_bytes: &mut u64,
    device_bytes: &mut u64,
) -> Result<u64> {
    let meta = shard.tensor(name)?;
    if meta.dtype != "F32" {
        return Err(anyhow!(
            "{name} has dtype {}, but this loader converts F32 to f16; add a branch before \
             trusting anything downstream",
            meta.dtype
        ));
    }
    let raw = shard.read_tensor_bytes(name)?;
    if raw.len() % 4 != 0 {
        return Err(anyhow!(
            "{name} byte length {} is not f32-aligned",
            raw.len()
        ));
    }
    let elems = raw.len() / 4;
    if elems != expect_elems {
        return Err(anyhow!(
            "{name} has {elems} elements, expected {expect_elems} from config.json; the config and \
             the checkpoint disagree about the model's shape"
        ));
    }
    let mut buf = dev.alloc_device(elems * 2)?;
    // SAFETY: `buf` was allocated as `elems * 2` bytes immediately above and
    // is not yet visible to any dispatch.
    unsafe {
        let dst = buf.as_mut_slice_of::<u16>();
        for (i, chunk) in raw.chunks_exact(4).enumerate() {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            dst[i] = f32_to_f16(v);
        }
    }
    *source_bytes += raw.len() as u64;
    *device_bytes += (elems * 2) as u64;
    let va = buf.va();
    keep.push(buf);
    Ok(va)
}

/// Load a real OLMo 2 checkpoint into device memory.
///
/// `layer_limit` caps how many decoder layers are uploaded, which is what makes
/// bring-up tractable: a single layer is enough to validate the whole forward
/// path and costs a fraction of the load time.
pub fn load_olmo2_weights(
    dev: &mut GpuDevice,
    config_path: impl AsRef<Path>,
    index_path: impl AsRef<Path>,
    layer_limit: Option<usize>,
) -> Result<Olmo2Weights> {
    let config_path = config_path.as_ref();
    let index_path = index_path.as_ref();
    // Refuse before spending minutes on IO if the checkpoint is not what this
    // path implements.
    let pre = preflight_olmo2_checkpoint(config_path, index_path, 1)?;
    let config = Olmo2Config::open(config_path)?;
    let index = SafetensorsIndex::open(index_path)?;

    let h = config.hidden_size as usize;
    let inter = config.intermediate_size as usize;
    let vocab = config.vocab_size as usize;
    let head_dim = config.head_dim()? as usize;
    let q_width = config.num_attention_heads as usize * head_dim;
    let kv_width = config.num_key_value_heads as usize * head_dim;
    let n_layers = layer_limit
        .unwrap_or(config.num_hidden_layers as usize)
        .min(config.num_hidden_layers as usize);

    // One shard handle per file, opened once. SafetensorsShard keeps only
    // metadata, so this does not hold the checkpoint in memory.
    let mut shards: std::collections::HashMap<
        std::path::PathBuf,
        crate::weights::SafetensorsShard,
    > = std::collections::HashMap::new();
    for path in index.weight_map.values() {
        if !shards.contains_key(path) {
            shards.insert(path.clone(), crate::weights::SafetensorsShard::open(path)?);
        }
    }
    let shard_for = |name: &str| -> Result<&crate::weights::SafetensorsShard> {
        let path = index.shard_for(name)?;
        shards
            .get(path)
            .ok_or_else(|| anyhow!("no open shard for {name}"))
    };

    let mut keep = Vec::new();
    let mut source_bytes = 0u64;
    let mut device_bytes = 0u64;

    let started = std::time::Instant::now();
    let embed = upload_f16(
        dev,
        shard_for("model.embed_tokens.weight")?,
        "model.embed_tokens.weight",
        vocab * h,
        &mut keep,
        &mut source_bytes,
        &mut device_bytes,
    )?;
    let final_norm = upload_f16(
        dev,
        shard_for("model.norm.weight")?,
        "model.norm.weight",
        h,
        &mut keep,
        &mut source_bytes,
        &mut device_bytes,
    )?;
    let lm_head = if pre.lm_head_present {
        upload_f16(
            dev,
            shard_for("lm_head.weight")?,
            "lm_head.weight",
            vocab * h,
            &mut keep,
            &mut source_bytes,
            &mut device_bytes,
        )?
    } else {
        embed // tied embeddings
    };

    let mut layers = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let mut one = |suffix: &str, elems: usize| -> Result<u64> {
            let name = format!("model.layers.{layer}.{suffix}");
            let shard = shard_for(&name)?;
            upload_f16(
                dev,
                shard,
                &name,
                elems,
                &mut keep,
                &mut source_bytes,
                &mut device_bytes,
            )
        };
        layers.push(Olmo2LayerVas {
            wq: one("self_attn.q_proj.weight", q_width * h)?,
            wk: one("self_attn.k_proj.weight", kv_width * h)?,
            wv: one("self_attn.v_proj.weight", kv_width * h)?,
            q_norm: one("self_attn.q_norm.weight", q_width)?,
            k_norm: one("self_attn.k_norm.weight", kv_width)?,
            wo: one("self_attn.o_proj.weight", h * q_width)?,
            post_attn_norm: one("post_attention_layernorm.weight", h)?,
            gate: one("mlp.gate_proj.weight", inter * h)?,
            up: one("mlp.up_proj.weight", inter * h)?,
            down: one("mlp.down_proj.weight", h * inter)?,
            post_ffn_norm: one("post_feedforward_layernorm.weight", h)?,
        });
    }
    let elapsed = started.elapsed();

    println!(
        "  olmo2-weights: {} layer(s) of {} resident on node {}, {:.2} GiB f32 read, {:.2} GiB f16 \
uploaded, {} buffers, {:.1}s ({:.0} MiB/s)",
        n_layers,
        config.num_hidden_layers,
        dev.node_id(),
        source_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        device_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        keep.len(),
        elapsed.as_secs_f64(),
        source_bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64().max(1e-9),
    );

    Ok(Olmo2Weights {
        config,
        layers,
        embed,
        final_norm,
        lm_head,
        device_bytes,
        source_bytes,
        _keep: keep,
    })
}

/// Load a checkpoint and prove the bytes that landed in device memory are the
/// bytes the file holds.
///
/// The conversion from fp32 to f16 is lossy by design, so this does not compare
/// bit patterns. It re-reads a sample of source elements, converts them the same
/// way, and requires an exact match against what is resident. That catches the
/// failure modes that actually happen here: a wrong offset, a transposed shape,
/// a shard resolved to the wrong file, or a truncated upload.
pub fn check_olmo2_weight_load_on(
    dev: &mut GpuDevice,
    config_path: impl AsRef<Path>,
    index_path: impl AsRef<Path>,
    layer_limit: Option<usize>,
) -> Result<Olmo2Weights> {
    let config_path = config_path.as_ref();
    let index_path = index_path.as_ref();
    let w = load_olmo2_weights(dev, config_path, index_path, layer_limit)?;

    let index = SafetensorsIndex::open(index_path)?;
    let h = w.config.hidden_size as usize;
    let head_dim = w.config.head_dim()? as usize;
    let q_width = w.config.num_attention_heads as usize * head_dim;

    let mut checked = 0usize;
    let mut verify = |name: &str, va: u64, elems: usize| -> Result<()> {
        let path = index.shard_for(name)?;
        let shard = crate::weights::SafetensorsShard::open(path)?;
        let raw = shard.read_tensor_bytes(name)?;
        // SAFETY: `va` is the address of a live buffer held by `w._keep` for the
        // duration of this function, allocated at `elems` f16 elements. The slice is
        // read-only and does not outlive the borrow of `w`.
        let resident = unsafe { std::slice::from_raw_parts(va as *const u16, elems) };
        // Sample the ends and a deterministic scatter through the middle: a
        // truncated upload shows at the tail, a bad offset at the head.
        let mut probes: Vec<usize> = vec![0, 1, elems / 2, elems - 2, elems - 1];
        for k in 1..24 {
            probes.push((k * 2_654_435_761usize) % elems);
        }
        probes.sort_unstable();
        probes.dedup();
        for &i in &probes {
            let c = &raw[i * 4..i * 4 + 4];
            let want = f32_to_f16(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            if resident[i] != want {
                return Err(anyhow!(
                    "{name}[{i}] resident 0x{:04x} != 0x{:04x} converted from the checkpoint",
                    resident[i],
                    want
                ));
            }
            checked += 1;
        }
        Ok(())
    };

    verify("model.norm.weight", w.final_norm, h)?;
    verify(
        "model.layers.0.self_attn.q_proj.weight",
        w.layers[0].wq,
        q_width * h,
    )?;
    verify(
        "model.layers.0.self_attn.q_norm.weight",
        w.layers[0].q_norm,
        q_width,
    )?;
    verify(
        "model.layers.0.post_feedforward_layernorm.weight",
        w.layers[0].post_ffn_norm,
        h,
    )?;
    if w.layers.len() > 1 {
        let last = w.layers.len() - 1;
        verify(
            &format!("model.layers.{last}.mlp.down_proj.weight"),
            w.layers[last].down,
            h * w.config.intermediate_size as usize,
        )?;
    }

    println!(
        "  olmo2-weight-readback: {checked} sampled elements across {} tensors match the \
checkpoint after f32->f16 conversion",
        if w.layers.len() > 1 { 5 } else { 4 }
    );
    Ok(w)
}

// ---------------------------------------------------------------------------
// The decode runner: post-norm layer forward and an autoregressive token loop.
// ---------------------------------------------------------------------------

use crate::attn::{combine_groups, enqueue_combine_decode_gqa_f16, split_count};

/// Everything the decode loop needs besides the weights.
struct Olmo2Scratch {
    qf32: u64,
    kf32: u64,
    vf32: u64,
    qf16: u64,
    part: u64,
    inter_buf: u64,
    a16: u64,
    oproj: u64,
    gate_f32: u64,
    up_f32: u64,
    gate_f16: u64,
    up_f16: u64,
    sw_f16: u64,
    down_f32: u64,
    zeros_f32: u64,
    normed: u64,
    logits: u64,
}

/// Paged KV metadata shared by every layer.
struct Olmo2Paged {
    block_table: u64,
    indices: u64,
    indptr: u64,
    last_page_len: u64,
    batch_indices: u64,
    positions: u64,
    seq_lens: u64,
    block_size: u32,
    physical_blocks: u32,
    rows_per_head: u32,
}

/// One layer's FP4 paged KV cache.
struct Olmo2LayerKv {
    k: u64,
    v: u64,
    sk: u64,
    sv: u64,
}

/// A resident OLMo 2 model that can generate tokens.
pub struct Olmo2Runner {
    pub weights: Olmo2Weights,
    scratch: Olmo2Scratch,
    paged: Olmo2Paged,
    kv: Vec<Olmo2LayerKv>,
    hva: u64,
    max_seq: u32,
    num_splits: u32,
    /// Host mirror of the residual stream, used for the embedding lookup.
    embed_host: u64,
    _keep: Vec<crate::DeviceBuffer>,
}

impl Olmo2Runner {
    /// Allocate scratch and a paged KV cache for `max_seq` tokens.
    pub fn new(dev: &mut GpuDevice, weights: Olmo2Weights, max_seq: u32) -> Result<Self> {
        let cfg = &weights.config;
        let h = cfg.hidden_size as usize;
        let inter = cfg.intermediate_size as usize;
        let vocab = cfg.vocab_size as usize;
        let d = cfg.head_dim()? as usize;
        let nh = cfg.num_attention_heads as usize;
        let nkv = cfg.num_key_value_heads as usize;
        let q_width = nh * d;
        let kv_width = nkv * d;
        let block_size = 16u32;
        if max_seq == 0 || max_seq % block_size != 0 {
            return Err(anyhow!(
                "max_seq must be a non-zero multiple of {block_size} (got {max_seq})"
            ));
        }
        let physical_blocks = max_seq / block_size;
        let rows_per_head = physical_blocks * block_size;
        let num_splits = split_count(max_seq).max(1);
        let ps = d + 2;
        let inter_splits = combine_groups(num_splits).max(1) as usize;

        let mut keep = Vec::new();
        let dev_buf = |bytes: usize, keep: &mut Vec<crate::DeviceBuffer>| -> Result<u64> {
            let b = dev.alloc_device(bytes)?;
            let va = b.va();
            keep.push(b);
            Ok(va)
        };

        let scratch = Olmo2Scratch {
            qf32: dev_buf(q_width * 4, &mut keep)?,
            kf32: dev_buf(kv_width * 4, &mut keep)?,
            vf32: dev_buf(kv_width * 4, &mut keep)?,
            qf16: dev_buf(q_width * 2, &mut keep)?,
            part: dev_buf(nh * num_splits as usize * ps * 4, &mut keep)?,
            inter_buf: dev_buf(nh * inter_splits * ps * 4, &mut keep)?,
            a16: dev_buf(q_width * 2, &mut keep)?,
            oproj: dev_buf(h * 4, &mut keep)?,
            gate_f32: dev_buf(inter * 4, &mut keep)?,
            up_f32: dev_buf(inter * 4, &mut keep)?,
            gate_f16: dev_buf(inter * 2, &mut keep)?,
            up_f16: dev_buf(inter * 2, &mut keep)?,
            sw_f16: dev_buf(inter * 2, &mut keep)?,
            down_f32: dev_buf(h * 4, &mut keep)?,
            zeros_f32: dev_buf(h * 4, &mut keep)?,
            normed: dev_buf(h * 2, &mut keep)?,
            logits: dev_buf(vocab * 4, &mut keep)?,
        };
        let hva = dev_buf(h * 2, &mut keep)?;

        let paged = Olmo2Paged {
            block_table: dev_buf(physical_blocks as usize * 4, &mut keep)?,
            indices: dev_buf(physical_blocks as usize * 4, &mut keep)?,
            indptr: dev_buf(2 * 4, &mut keep)?,
            last_page_len: dev_buf(4, &mut keep)?,
            batch_indices: dev_buf(4, &mut keep)?,
            positions: dev_buf(4, &mut keep)?,
            seq_lens: dev_buf(4, &mut keep)?,
            block_size,
            physical_blocks,
            rows_per_head,
        };

        let mut kv = Vec::with_capacity(weights.layers.len());
        for _ in 0..weights.layers.len() {
            kv.push(Olmo2LayerKv {
                k: dev_buf(nkv * rows_per_head as usize * 64, &mut keep)?,
                v: dev_buf(nkv * rows_per_head as usize * 64, &mut keep)?,
                sk: dev_buf(nkv * rows_per_head as usize * 4, &mut keep)?,
                sv: dev_buf(nkv * rows_per_head as usize * 4, &mut keep)?,
            });
        }

        // Identity page mapping. A real scheduler would hand out physical pages
        // from a pool; a single-sequence demo does not need one.
        // SAFETY: these metadata buffers were allocated just above at the sizes
        // indexed here, and the runner has not dispatched anything yet.
        unsafe {
            let bt = std::slice::from_raw_parts_mut(
                paged.block_table as *mut u32,
                physical_blocks as usize,
            );
            let idx =
                std::slice::from_raw_parts_mut(paged.indices as *mut u32, physical_blocks as usize);
            for b in 0..physical_blocks as usize {
                bt[b] = b as u32;
                idx[b] = b as u32;
            }
            let ip = std::slice::from_raw_parts_mut(paged.indptr as *mut u32, 2);
            ip[0] = 0;
            ip[1] = physical_blocks;
            *(paged.batch_indices as *mut u32) = 0;
            std::ptr::write_bytes(scratch.zeros_f32 as *mut u8, 0, h * 4);
        }

        let embed_host = weights.embed;
        Ok(Self {
            weights,
            scratch,
            paged,
            kv,
            hva,
            max_seq,
            num_splits,
            embed_host,
            _keep: keep,
        })
    }

    /// One OLMo 2 decoder layer, in post-norm order.
    ///
    /// The shape of this function *is* the architecture. Note what is absent:
    /// there is no norm before attention and none before the MLP. The residual
    /// stream goes into each sub-layer raw, and only the sub-layer's output is
    /// normalised on its way back in.
    fn layer(&mut self, dev: &mut GpuDevice, li: usize) -> Result<()> {
        let cfg = &self.weights.config;
        let h = cfg.hidden_size as u32;
        let inter = cfg.intermediate_size as u32;
        let d = cfg.head_dim()? as u32;
        let nh = cfg.num_attention_heads as u32;
        let q_width = nh * d;
        let kv_width = cfg.num_key_value_heads as u32 * d;
        let eps = 1e-6f32;
        let theta = cfg.rope_theta as f32;
        let scale = 1.0f32 / (d as f32).sqrt();
        let w = self.weights.layers[li];
        let kv = &self.kv[li];
        let s = &self.scratch;
        let pm = &self.paged;

        // --- attention, no pre-norm ---------------------------------------
        dev.chain_next();
        dev.arm_gemv_qkv(
            w.wq, w.wk, w.wv, self.hva, s.qf32, s.kf32, s.vf32, q_width, kv_width, h,
        )?;
        dev.chain_next();
        dev.arm_cast_qk_rope_append_paged_fp4_vf32_q16_k16_d128_olmo2_meta(
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
            pm.physical_blocks,
            pm.physical_blocks,
            pm.block_size,
            theta,
            eps,
        )?;
        dev.chain_next();
        dev.arm_attn_decode_split_fp4_mha_paged_groups_meta(
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
            self.max_seq,
            d,
            scale,
            self.num_splits,
            nh, // one group per head
            1,  // multi-head: one query head per KV head
            pm.rows_per_head,
        )?;
        enqueue_combine_decode_gqa_f16(dev, s.part, s.inter_buf, s.a16, d, self.num_splits, nh)?;
        dev.chain_next();
        dev.arm_gemv(w.wo, s.a16, s.oproj, h, q_width)?;
        dev.chain_next();
        dev.arm_add_postnorm(self.hva, s.oproj, w.post_attn_norm, h, eps)?;

        // --- dense SwiGLU MLP, again no pre-norm --------------------------
        dev.chain_next();
        dev.arm_gemv(w.gate, self.hva, s.gate_f32, inter, h)?;
        dev.chain_next();
        dev.arm_gemv(w.up, self.hva, s.up_f32, inter, h)?;
        dev.chain_next();
        dev.arm_cast_f32_f16(s.gate_f32, s.gate_f16, inter)?;
        dev.chain_next();
        dev.arm_cast_f32_f16(s.up_f32, s.up_f16, inter)?;
        dev.chain_next();
        dev.arm_swiglu(s.gate_f16, s.up_f16, s.sw_f16, inter)?;
        dev.chain_next();
        dev.arm_gemv(w.down, s.sw_f16, s.down_f32, h, inter)?;
        dev.arm_add_postnorm(self.hva, s.down_f32, w.post_ffn_norm, h, eps)?;
        // Wait once per layer rather than once per token.
        //
        // Every dispatch above carries the barrier bit, so they retire in order
        // and one wait covers the whole layer. The reason not to chain the
        // entire model and wait once at the end is the kernarg ring: it holds
        // 256 slots and the AQL guard refuses to arm past it, so a model deep
        // enough to exceed that would fail to run at all. Sixteen layers times
        // roughly fourteen dispatches already sits at about 226. Waiting per
        // layer makes depth irrelevant, and at 125 ms/token the sixteen host
        // round trips are not what is costing the time.
        dev.wait(Duration::from_secs(30))?;
        Ok(())
    }

    /// Run one decode step for `token` at `pos` and return the greedy next token.
    pub fn step(&mut self, dev: &mut GpuDevice, token: u32, pos: u32) -> Result<u32> {
        let cfg = &self.weights.config;
        let h = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as u32;
        if token >= vocab {
            return Err(anyhow!("token {token} is outside vocab {vocab}"));
        }
        if pos >= self.max_seq {
            return Err(anyhow!(
                "position {pos} exceeds the {}-token KV cache; the prompt plus the tokens you \
                 asked for has to fit, so either shorten it or raise --olmo-max-seq (or max_seq \
                 if you are calling Olmo2Runner directly)",
                self.max_seq
            ));
        }

        // Embedding lookup on the host: 4 KiB per token, not worth a kernel.
        // SAFETY: `embed_host` addresses the resident embedding table, which is
        // `vocab * h` f16 elements, and `token < vocab` is checked above. `hva` is `h`
        // f16 elements. The paged metadata buffers are single u32 slots allocated in
        // `new`. No dispatch is in flight: `step` waits at the end of every layer.
        unsafe {
            let src = std::slice::from_raw_parts(
                (self.embed_host as *const u16).add(token as usize * h),
                h,
            );
            let dst = std::slice::from_raw_parts_mut(self.hva as *mut u16, h);
            dst.copy_from_slice(src);
            *(self.paged.positions as *mut u32) = pos;
            *(self.paged.seq_lens as *mut u32) = pos + 1;
            let page = pos / self.paged.block_size;
            *(self.paged.last_page_len as *mut u32) = (pos % self.paged.block_size) + 1;
            let ip = std::slice::from_raw_parts_mut(self.paged.indptr as *mut u32, 2);
            ip[0] = 0;
            ip[1] = page + 1;
        }

        for li in 0..self.weights.layers.len() {
            self.layer(dev, li)?;
        }

        // Final RMSNorm. add_rmsnorm with a zero contribution leaves the
        // residual untouched and writes the normalised copy to `normed`.
        let eps = 1e-6f32;
        dev.chain_next();
        dev.arm_add_rmsnorm(
            self.hva,
            self.scratch.zeros_f32,
            self.weights.final_norm,
            self.scratch.normed,
            h as u32,
            eps,
        )?;
        dev.arm_gemv(
            self.weights.lm_head,
            self.scratch.normed,
            self.scratch.logits,
            vocab,
            h as u32,
        )?;
        dev.wait(Duration::from_secs(60))?;

        // SAFETY: `dev.wait` returned above, so the LM head dispatch has retired
        // and `logits` holds `vocab` f32 values written by the GPU.
        let logits = unsafe {
            std::slice::from_raw_parts(self.scratch.logits as *const f32, vocab as usize)
        };
        let mut best = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i as u32;
            }
        }
        if !best_v.is_finite() {
            return Err(anyhow!(
                "logits are not finite at pos {pos} (max {best_v}); the forward path produced \
                 NaN or inf"
            ));
        }
        Ok(best)
    }

    /// The configured end-of-sequence token, if the checkpoint declares one.
    pub fn eos_token(&self) -> Option<u32> {
        self.weights.config.eos_token_id.map(|e| e as u32)
    }

    /// Greedy generation. The prompt is consumed one token at a time through the
    /// decode path, which is prefill done the slow honest way: no prefill GEMM
    /// kernel is needed because the decode loop already grows the KV cache.
    pub fn generate(
        &mut self,
        dev: &mut GpuDevice,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(anyhow!("prompt must have at least one token"));
        }
        let mut pos = 0u32;
        let mut next = 0u32;
        for &t in prompt {
            next = self.step(dev, t, pos)?;
            pos += 1;
        }
        let eos = self.weights.config.eos_token_id.map(|e| e as u32);
        let mut out = Vec::with_capacity(max_new);
        for _ in 0..max_new {
            // Stop *before* emitting EOS. A caller wants the text, not the
            // sentinel, and a base model that runs past it starts a new
            // document mid-response.
            if Some(next) == eos {
                break;
            }
            out.push(next);
            if pos as usize >= self.max_seq as usize {
                break;
            }
            next = self.step(dev, next, pos)?;
            pos += 1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `config.json` from allenai/OLMo-2-0425-1B, verbatim.
    const OLMO2_1B_CONFIG: &str = r#"{
      "architectures": ["Olmo2ForCausalLM"],
      "attention_bias": false,
      "eos_token_id": 100257,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "intermediate_size": 8192,
      "model_type": "olmo2",
      "num_attention_heads": 16,
      "num_hidden_layers": 16,
      "num_key_value_heads": 16,
      "pad_token_id": 100277,
      "rms_norm_eps": 1e-06,
      "rope_theta": 500000,
      "tie_word_embeddings": false,
      "vocab_size": 100352
    }"#;

    fn cfg() -> Olmo2Config {
        Olmo2Config::parse(OLMO2_1B_CONFIG, Path::new("config.json")).expect("parse")
    }

    #[test]
    fn parses_the_real_checkpoint_config() {
        let c = cfg();
        assert_eq!(c.model_type, "olmo2");
        assert_eq!(c.num_hidden_layers, 16);
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 16);
        assert_eq!(c.intermediate_size, 8192);
        assert_eq!(c.vocab_size, 100_352);
        assert_eq!(c.rope_theta, 500_000);
        assert!(!c.tie_word_embeddings);
        assert_eq!(c.eos_token_id, Some(100_257));
    }

    #[test]
    fn head_dim_is_derived_and_is_the_one_the_kernels_implement() {
        assert_eq!(cfg().head_dim().unwrap(), REQUIRED_HEAD_DIM);
    }

    #[test]
    fn head_dim_rejects_a_hidden_size_that_does_not_divide() {
        let mut c = cfg();
        c.hidden_size = 2050;
        assert!(c.head_dim().is_err());
    }

    #[test]
    fn olmo2_is_multi_head_so_the_group_is_one() {
        assert_eq!(cfg().q_heads_per_kv().unwrap(), 1);
    }

    #[test]
    fn qk_norm_spans_the_whole_projection_not_one_head() {
        let c = cfg();
        // The distinguishing fact. Per-head would be head_dim, 128.
        assert_eq!(c.q_norm_elems().unwrap(), 2048);
        assert_eq!(c.k_norm_elems().unwrap(), 2048);
        assert_ne!(c.q_norm_elems().unwrap(), c.head_dim().unwrap());
    }

    #[test]
    fn qk_norm_is_sliced_with_its_projection_not_replicated() {
        // The one placement rule OLMo 2 does not inherit from the Qwen path:
        // a projection-wide norm has to follow the projection under TP.
        for name in [
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.7.self_attn.k_norm.weight",
        ] {
            assert_eq!(
                olmo2_weight_placement(name),
                QwenWeightPlacement::TensorParallel { axis: 0 },
                "{name} must be sliced with the projection it normalises"
            );
        }
        // Post-norm weights are per-hidden-element and stay replicated.
        for name in [
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
            "model.norm.weight",
        ] {
            assert_eq!(
                olmo2_weight_placement(name),
                QwenWeightPlacement::Replicated
            );
        }
    }

    fn names_for(c: &Olmo2Config) -> Vec<String> {
        synthetic_olmo2_tensor_names(c)
    }

    #[test]
    fn accepts_a_complete_checkpoint() {
        let c = synthetic_olmo2_config(3);
        let owned = names_for(&c);
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        let r = preflight_olmo2_names(&c, &names, Some(1024), 1).expect("accept");
        assert!(r.is_multi_head);
        assert!(r.lm_head_present);
        assert_eq!(r.num_hidden_layers, 3);
        // 3 layers * 11 tensors + embed + norm + lm_head
        assert_eq!(r.tensor_count, 3 * 11 + 3);
    }

    #[test]
    fn rejects_a_pre_norm_checkpoint() {
        let c = synthetic_olmo2_config(1);
        let mut owned = names_for(&c);
        owned.push("model.layers.0.input_layernorm.weight".into());
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        let err = preflight_olmo2_names(&c, &names, None, 1).unwrap_err();
        assert!(
            format!("{err}").contains("pre-norm"),
            "rejection must name the reason: {err}"
        );
    }

    #[test]
    fn rejects_a_missing_layer_tensor() {
        let c = synthetic_olmo2_config(2);
        let owned: Vec<String> = names_for(&c)
            .into_iter()
            .filter(|n| n != "model.layers.1.mlp.down_proj.weight")
            .collect();
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        let err = preflight_olmo2_names(&c, &names, None, 1).unwrap_err();
        assert!(format!("{err}").contains("down_proj"), "{err}");
    }

    #[test]
    fn rejects_a_head_dim_the_kernels_cannot_run() {
        // SmolLM2-135M's shape: hidden 576 over 9 heads is head_dim 64.
        let mut c = synthetic_olmo2_config(1);
        c.hidden_size = 576;
        c.num_attention_heads = 9;
        c.num_key_value_heads = 3;
        let owned = names_for(&c);
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        let err = preflight_olmo2_names(&c, &names, None, 1).unwrap_err();
        assert!(format!("{err}").contains("head_dim"), "{err}");
    }

    #[test]
    fn rejects_a_non_olmo_architecture() {
        let mut c = synthetic_olmo2_config(1);
        c.model_type = "llama".into();
        let owned = names_for(&c);
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        assert!(preflight_olmo2_names(&c, &names, None, 1).is_err());
    }

    #[test]
    fn rejects_a_tp_world_that_does_not_divide_the_heads() {
        let c = synthetic_olmo2_config(1);
        let owned = names_for(&c);
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        // 16 heads over 5 ranks.
        assert!(preflight_olmo2_names(&c, &names, None, 5).is_err());
        // And zero is refused rather than dividing by it.
        assert!(preflight_olmo2_names(&c, &names, None, 0).is_err());
    }

    #[test]
    fn tied_embeddings_do_not_require_an_lm_head() {
        let mut c = synthetic_olmo2_config(1);
        c.tie_word_embeddings = true;
        let owned = names_for(&c);
        assert!(!owned.iter().any(|n| n == "lm_head.weight"));
        let names = owned.iter().map(String::as_str).collect::<HashSet<_>>();
        let r = preflight_olmo2_names(&c, &names, None, 1).expect("accept tied");
        assert!(!r.lm_head_present);
    }

    #[test]
    fn every_layer_tensor_the_preflight_wants_is_one_the_generator_emits() {
        let c = synthetic_olmo2_config(1);
        let owned = names_for(&c);
        for suffix in OLMO2_LAYER_TENSOR_SUFFIXES {
            let want = format!("model.layers.0.{suffix}");
            assert!(owned.contains(&want), "generator is missing {want}");
        }
    }
}

# mainarch roadmap

The plan is to climb from raw kernel ABI control to an inference-serving stack
one tested layer at a time. The README carries the headline measurements; this
roadmap is the status map and names the next gaps without turning them into
claims.

## Banked Layers

### M0 - Talk to the silicon

- [x] `mainarch-sys`: hand-encoded `amdkfd` ioctls for the initial UAPI surface.
- [x] `mainarch-core`: open `/dev/kfd`, acquire a VM, enumerate GPU topology,
  and expose the safe device/topology boundary.
- [x] Hardware probe path for MI355X/gfx950 hosts through `mainarch probe`.

### M1 - Memory, queues, and code objects

- [x] VRAM/GTT allocation and mapping through KFD.
- [x] AQL compute queue creation, packet construction, doorbell signaling, and
  completion-signal waiting without ROCm/HIP/HSA runtime.
- [x] HSA code-object loading through the in-repo ELF/code-object path.
- [x] CPU-readable kernel descriptor metadata for preflight inspection.

Open M1 hardening:

- [ ] Optional generated/bindgen audit surface for the full upstream
  `<linux/kfd_ioctl.h>` UAPI.
- [ ] Broader ABI drift tests across supported host kernel versions.

### M2 - Intra-node XGMI collectives

- [x] GPU-to-GPU peer mapping over XGMI.
- [x] Real multi-GPU `all_reduce_f32` with CPU-oracle correctness checks.
- [x] Small/mid-message fused reduce-scatter/all-gather path.
- [x] Large-message GPU-driven one-shot path with device-side synchronization.
- [x] RCCL comparison harness for apples-to-apples latency/bandwidth evidence.

Open M2 hardening:

- [ ] Reduce-scatter, all-gather, and broadcast as first-class public collective
  operations beyond the all-reduce proving path.
- [ ] Device-side SDMA triggering or equivalent large-message bandwidth closer.

### M3 - Decode primitives

- [x] FlashDecoding-style paged attention.
- [x] FP16, FP8, FP4, grouped-query, and paged KV decode attention variants.
- [x] RMSNorm, RoPE, SwiGLU, GEMV projections, KV quantization, and greedy
  sampling primitives.
- [x] MoE router and local routed MoE FFN for the Qwen-style decode layer.
- [x] End-to-end reduced-scale Qwen-style decode layer correctness against an
  f64 reference.

Open M3 hardening:

- [ ] Full prefill GEMM path, including FP8/MXFP4 matrix-core kernels.
- [ ] Persistent decode megakernel that removes per-token host launch overhead.
- [ ] Comm/compute fusion for tensor-parallel decode.

### M4 - Autoregressive model-decode loop

- [x] Multi-layer autoregressive `model-decode` loop on the raw ABI.
- [x] Growing KV cache across decode steps.
- [x] Per-step reduced-scale validation against an f64 reference.

Open M4 hardening:

- [x] Real checkpoint loading at full depth and full vocabulary through the
  serving path, done for OLMo 2 in M6 below. Qwen's own path is still
  reduced-scale.
- [ ] Tensor parallelism across all 8 GPUs for the model-decode loop.
- [ ] Production KV-cache manager with prefix reuse and XGMI movement.

### M6 - A real open-source model, end to end

- [x] OLMo 2 checkpoint contract preflight, CPU-only, refusing a pre-norm
  checkpoint, a head dimension the kernels cannot run, and a non-OLMo
  architecture, each by name.
- [x] Multi-head decode attention over paged FP4 KV, gated bitwise against the
  validated grouped-query kernel plus a head-mapping proof.
- [x] Fused QK-norm over the whole projection, RoPE, and paged FP4 KV append,
  gated at rel-L2 2-3e-4 against an f64 reference and proven to couple across
  heads the way a projection-wide norm must.
- [x] Post-norm residual update, gated against both orderings so pre-norm cannot
  pass.
- [x] Full-depth fp32 to f16 checkpoint load into VRAM with sampled readback.
- [x] Dense SwiGLU MLP from existing gemv, cast and swiglu kernels.
- [x] Greedy token loop with prefill done by running the decode path once per
  prompt token, and end-of-sequence stopping.
- [x] `/v1/chat/completions` served from the real model, streaming and not, with
  `synthetic: false` reported by `/v1/models` and `/api/health`.

Open M6 hardening:

- [ ] Batching. The lane serves one request at a time because the KV cache is
  single-sequence.
- [ ] Sampling beyond greedy argmax.
- [ ] A prefill kernel. The decode loop stands in for one correctly but pays
  O(prompt) decode steps to do it.
- [ ] Tensor parallelism, so a model larger than one card can run.
- [ ] Performance. About 125 ms/token, unfused and untuned.

### M5 - Model-facing API and CPU-only runtime metadata

- [x] Typed primitive graph API for model definitions.
- [x] Qwen-style reference MoE decoder graph using only the public primitive API.
- [x] External-style custom model example using `ModelDefinition`.
- [x] Static graph validation, stage coverage, tensor storage/access/lifetime
  manifests, checkpoint binding plans, lowering route manifests, runtime slot
  manifests, dispatch intent manifests, launch preflight reports, AQL packet
  templates, queue/staging/completion request plans, submission gates, and
  composed readiness reports.
- [x] Kernel-argument ABI named-schema preflight, size-compatibility receipts,
  verified-candidate recommendations for covered dispatches, and explicit
  missing-coverage counts.
- [x] Metadata-only semantic kernarg field comparison for covered
  host-launcher kernel symbols, with explicit missing-schema, descriptor-match,
  missing-field, field-mismatch, and extra-argument counts plus a dispatch-
  centric semantic gap report with primary semantic gap reason counts and a
  CPU-only semantic kernarg projection plan for descriptor-matching schemas.
- [x] Compact static handoff receipt for accepted external model packages,
  bundling manifest/summary fingerprints, metadata admission, synthetic pointer
  preflight, projection-selection counts, and explicit non-execution counters.
- [x] CPU-only runtime launch and submission handoff receipts for reference,
  custom, CLI selftest, and standalone external plugin examples, including
  launch request, submission gate, blocker report, prerequisite plan, resolved
  gate, resolved blocker-report, and resolved prerequisite-plan fixtures.
- [x] Non-submitting boundary helpers and release evidence for live-AQL
  proof-validation overlays, runtime component application overlays, and
  runtime component receipt-intake overlays, with explicit no-submit/no-queue-
  mutation assertions before those overlays feed submission gates.
- [x] CPU-only catalog/code-object kernel symbol coverage for every non-gap
  model API catalog case against the bundled gfx950 code object, including
  explicit unmapped-entrypoint and missing-kernel diagnostics.
- [x] CPU-only catalog ABI registry coverage for every catalog-required bundled
  gfx950 kernel symbol, joining code-object descriptor metadata to the named and
  semantic ABI schema registries with size/alignment mismatch diagnostics.

Not claimed for M5 yet:

- [ ] Executable graph lowering into live AQL packets.
- [ ] Full kernel argument ABI validation for every dispatch/candidate,
  including complete semantic argument order/type proof and kernel-specific
  argument-value translation beyond the covered static comparison/projection
  registries.
- [ ] Buffer allocation, residency, pointer lifetime validation, and live GPU
  execution through the new graph API.
- [ ] OpenAI-compatible serving or throughput/latency through the new graph API.

## Current Work

- [ ] Real safetensors mmap/stream loading for Qwen-class checkpoints.
- [ ] Convert the model API handoff receipts into live graph execution by
  resolving the documented submission blockers with hardware correctness
  evidence.
- [ ] Full-depth/full-vocab model execution instead of reduced-scale validation.
- [ ] Persistent decode megakernel and tensor-parallel execution across the 8-GPU
  MI355X node.
- [ ] KV-cache system: paged, quantized, prefix-cached, and movable over XGMI.
- [ ] Continuous batching, chunked prefill, scheduling, sampling, and admission
  control.
- [ ] OpenAI-compatible server surface.
- [ ] Multi-node RDMA transport and hierarchical collectives.
- [ ] Kubernetes operator, node agent, resource isolation, quotas, telemetry, and
  live introspection.

## Reference Inspirations

- **NCCL/RCCL** - collective algorithms and the rccl-tests bandwidth methodology.
- **MSCCL++** - GPU-driven collectives and device-side synchronization patterns.
- **vLLM / SGLang** - paged KV, prefix caching, scheduling, and serving API
  behavior to match or beat.
- **ROCm / CK / hipBLASLt** - reference behavior and performance baselines, not
  runtime dependencies.

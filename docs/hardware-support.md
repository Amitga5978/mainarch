# Hardware Support

`mainarch` is currently an MI355X/gfx950-first project. The low-level runtime is
designed around direct access to the AMD Linux kernel ABI through `/dev/kfd`,
`/dev/dri`, KFD topology sysfs, AQL queues, and GPU-visible memory mappings.

## Validation Tiers

### CPU-Only

These checks do not require an AMD GPU:

```bash
cargo check --workspace --all-targets
cargo test --workspace
python3 tools/check_model_api_public_examples.py
cargo run -p mainarch-cli --bin mainarch-model-api-selftest
cargo run -p mainarch-core --example reference_moe_model_api
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-launch-request-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-submission-gate-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-resolved-submission-gate-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-resolved-submission-blocker-report-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-submission-blocker-report-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-submission-prerequisite-plan-receipt
cargo run -p mainarch-core --example custom_model_api
cargo run -p mainarch-core --example custom_model_api -- --runtime-launch-request-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-submission-gate-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-resolved-submission-gate-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-resolved-submission-blocker-report-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-submission-blocker-report-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-submission-prerequisite-plan-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --model-api-contract-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --plugin-manifest-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --plugin-compatibility-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-launch-request-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-submission-gate-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-resolved-submission-gate-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-resolved-submission-blocker-report-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-submission-blocker-report-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-submission-prerequisite-plan-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --static-handoff-receipt
cargo test --locked --manifest-path examples/model-api-plugin/Cargo.toml
cargo run -p mainarch-core --example rejected_model_api
cargo run -p mainarch-core --example rejected_model_api -- --rejection-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-launch-request-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-submission-gate-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-resolved-submission-gate-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-resolved-submission-blocker-report-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-submission-blocker-report-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-submission-prerequisite-plan-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --static-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-staging-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-mapped-host-staging-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-copy-plan-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-destination-residency-proof-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-destination-residency-query-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-queue-reservation-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-queue-reservation-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-copy-completion-signal-binding-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-copy-completion-signal-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-materialization-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-materialization-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-validation-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-validation-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-cache-visibility-policy-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-synchronization-plan-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-schedule-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-prerequisite-plan-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-runtime-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-bound-runtime-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-mapped-host-staging-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-destination-residency-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-sdma-queue-reservation-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-copy-completion-signal-binding-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-packet-materialization-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-packet-validation-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-cache-visibility-policy-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-completion-synchronization-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-pin-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-pin-virtual-address-plan-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-userptr-pin-arguments-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-vm-acquire-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-userptr-alloc-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-userptr-alloc-result-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-map-memory-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-map-memory-argument-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-map-memory-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-pin-page-rounding-receipt
```

They validate Rust compilation, unit tests, the public model API contract, and
metadata/readiness reports, including safetensors checkpoint-binding metadata.
They do not prove GPU execution, serving, or performance.

### OLMo 2

The checkpoint contract runs anywhere, with no GPU and no checkpoint:

```bash
mainarch olmo2-preflight-selftest
mainarch olmo2-preflight --config <dir>/config.json --index <dir>/model.safetensors.index.json
```

The three kernel gates need an AMD GPU:

```bash
mainarch gpu-mha-attention-equivalence-selftest --node <n>
mainarch gpu-olmo2-qk-rope-selftest --node <n>
mainarch gpu-olmo2-post-norm-selftest --node <n>
```

And these need both a GPU and a checkpoint:

```bash
mainarch olmo2-weight-load --node <n> --config ... --index ...
mainarch olmo2-generate --node <n> --config ... --index ... --tokenizer ... --prompt "..."
```

Validated on 8x MI355X against `allenai/OLMo-2-0425-1B`: 16 layers, 179 tensors,
2.77 GiB resident, about 125 ms/token.

### Local GPU Hardware

The current hardware target is an AMD MI355X/gfx950 host with:

- `/dev/kfd` and `/dev/dri` passed into the development container
- host user membership in the `render` group before the container starts
- a Linux kernel and `amdgpu`/KFD stack that expose the MI355X topology
- upstream LLVM `amdgpu` codegen available for build-time code-object generation

Useful gates:

```bash
just probe
mainarch gpu-selftest
mainarch gpu-multi-check
mainarch gpu-allreduce-bench
mainarch attn-decode --node <node_id>
mainarch decode-layer --node <node_id>
mainarch model-decode --node <node_id>
```

Record the exact command, environment variables, node IDs, GPU model, kernel
version, and output in the pull request for any GPU execution or performance
claim.

### Baseline Comparison

Performance claims require an apples-to-apples baseline against the relevant
upstream stack. For collectives, use `bench/compare-rccl.sh` or the `just bench`
wrapper with a ROCm image that contains `rccl-tests`.

## Not Claimed

- NVIDIA/CUDA support.
- ROCm, HIP, HSA runtime, RCCL, PyTorch, vLLM, or SGLang as production runtime
  dependencies.
- Support for AMD GPUs other than the current MI355X/gfx950 target.
- Support for virtualized or container environments that do not expose `/dev/kfd`
  and `/dev/dri`.
- Production serving. The OLMo 2 path does serve real weights over an
  OpenAI-shaped endpoint, but one request at a time, with no batching, no
  scheduling across sequences, no auth and no quota.
- Continuous batching, or any OpenAI-compatible API behaviour through the model
  API. The model API is CPU-only metadata and preflight, and the OLMo path does
  not go through it.

## Reporting Hardware Results

When opening an issue or PR with hardware evidence, include:

- host kernel release
- GPU model and `gfx` target
- command line and environment variables
- whether the run used CPU simulator, single-GPU, or multi-GPU/XGMI paths
- benchmark baseline command and raw output when making a performance claim

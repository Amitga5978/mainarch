# mainarch. `just` with no args lists every task.

default:
    @just --list

# ─── start here ────────────────────────────────────────────────────────────────

# THE one command: build, prove the stack on this machine, serve the demo page.
demo:
    @bash scripts/demo.sh

# A guided CPU-only tour of what the stack is made of. No GPU required.
tour:
    @bash scripts/tour.sh

# ─── the real model ────────────────────────────────────────────────────────────

# Fetch allenai/OLMo-2-0425-1B. ~6 GB, fully open weights, data and training code.
olmo-fetch dir='./olmo-2-1b':
    @bash scripts/olmo_fetch.sh {{dir}}

# Generate with the real model on the raw KFD/AQL path. Needs an AMD GPU.
olmo prompt='The capital of France is' dir='./olmo-2-1b': build
    ./target/release/mainarch olmo2-generate \
      --config {{dir}}/config.json \
      --index {{dir}}/model.safetensors.index.json \
      --tokenizer {{dir}}/tokenizer.json \
      --prompt "{{prompt}}" --max-new 32

# Serve the real model over /v1/chat/completions.
olmo-serve dir='./olmo-2-1b' bind='127.0.0.1:8080': build
    ./target/release/mainarch demo-serve --bind {{bind}} \
      --olmo-config {{dir}}/config.json \
      --olmo-index {{dir}}/model.safetensors.index.json \
      --olmo-tokenizer {{dir}}/tokenizer.json

# CPU-only: does this checkpoint satisfy the contract this runtime implements?
olmo-preflight dir='./olmo-2-1b': build
    ./target/release/mainarch olmo2-preflight \
      --config {{dir}}/config.json \
      --index {{dir}}/model.safetensors.index.json

# The four OLMo gates, on hardware.
olmo-gates node='2': build
    ./target/release/mainarch olmo2-preflight-selftest
    ./target/release/mainarch gpu-mha-attention-equivalence-selftest --node {{node}}
    ./target/release/mainarch gpu-olmo2-qk-rope-selftest --node {{node}}
    ./target/release/mainarch gpu-olmo2-post-norm-selftest --node {{node}}

# ─── inner loop ────────────────────────────────────────────────────────────────

# fast type-check across the workspace
check:
    cargo check --workspace --all-targets

# clippy at CI grade
lint:
    cargo clippy --workspace --all-targets -- -D warnings

fmt:
    cargo fmt --all

# fast test runner (falls back to plain cargo test if nextest is absent)
test:
    cargo nextest run --workspace || cargo test --workspace

# re-check on every save, leave this running while you work
watch:
    cargo watch -x 'check --workspace --all-targets'

# build the release CLI
build:
    cargo build --release -p mainarch-cli

# ─── CPU-only gates (these are what CI runs; no GPU needed) ────────────────────

# the public model-API contract gate
check-model-api:
    python3 tools/check_model_api_public_examples.py
    python3 tools/check_model_api_boundary_helpers.py --self-test
    python3 tools/check_model_api_boundary_helpers.py

# the demo sandbox contract gate (starts the GPU-free server, asserts the seam)
check-demo:
    python3 tools/check_demo_sandbox_static.py

# "no ROCm/HIP/HSA/CUDA/PyTorch in a production path" is a policy, so it is a test
check-policy:
    python3 tools/check_runtime_dependency_policy.py --self-test
    python3 tools/check_runtime_dependency_policy.py
    python3 tools/check_crate_publish_policy.py --self-test
    python3 tools/check_crate_publish_policy.py

# the OLMo 2 checkpoint contract, no GPU and no checkpoint needed
check-olmo: build
    ./target/release/mainarch olmo2-preflight-selftest

# everything CI runs, in one go
ci: check-policy fmt-check check test check-model-api check-demo check-olmo

fmt-check:
    cargo fmt --all -- --check

# ─── hardware lane (needs an AMD GPU on /dev/kfd) ──────────────────────────────

# open /dev/kfd, read the amdkfd version, enumerate GPU nodes
probe: build
    ./target/release/mainarch probe

# prove live kernel execution: hand-built AQL packet + doorbell, no ROCm
gpu-selftest: build
    ./target/release/mainarch gpu-selftest

# 8-GPU all-reduce, bit-exact against a CPU oracle
gpu-multi-check: build
    ./target/release/mainarch gpu-multi-check

# busbw sweep in the rccl-tests table shape
gpu-allreduce-bench: build
    ./target/release/mainarch gpu-allreduce-bench

# FlashDecoding attention: correctness + FP16/FP8/FP4/GQA/paged sweep
attn-decode node='2': build
    ./target/release/mainarch attn-decode --node {{node}}

# decode-layer primitives: RMSNorm, RoPE, SwiGLU, KV quantization
decode-layer node='2': build
    ./target/release/mainarch decode-layer --node {{node}}

# the assembled multi-layer decode model, validated against an f64 reference
model-decode node='2': build
    ./target/release/mainarch model-decode --node {{node}}

# rccl-tests-style all-reduce sweep (add --backend gpu on hardware)
rccl-test *ARGS: build
    ./target/release/mainarch rccl-test all-reduce {{ARGS}}

# ─── containers ────────────────────────────────────────────────────────────────

# package the release CLI for sibling-container benchmark runs
build-mainarch-image image='localhost/mainarch:latest': build
    @${DOCKER_BIN:-docker} build --tag {{image}} -f Dockerfile.mainarch-bench .

# build and run the one-page demo as its own container image
build-demo-image image='localhost/mainarch-demo:latest': build
    @MAINARCH_DEMO_IMAGE={{image}} bash tools/demo_container.sh build

run-demo-container image='localhost/mainarch-demo:latest':
    @MAINARCH_DEMO_IMAGE={{image}} bash tools/demo_container.sh run

smoke-demo-container:
    @bash tools/demo_container.sh smoke

stop-demo-container:
    @bash tools/demo_container.sh stop

# ─── baselines ─────────────────────────────────────────────────────────────────

# how to capture an upstream rccl-tests reference run
baseline:
    @cat baseline/README.md

# side-by-side rccl: a ROCm baseline image vs mainarch, as sibling containers
# usage: just bench <rocm-image-with-rccl-tests> [mainarch-image] [ngpus]
bench rocm_image='' mainarch_image='localhost/mainarch:latest' ngpus='8' min_bytes='8' max_bytes='134217728':
    ROCM_IMAGE={{rocm_image}} MAINARCH_IMAGE={{mainarch_image}} bash bench/compare-rccl.sh {{ngpus}} {{min_bytes}} {{max_bytes}}

# dry-run the benchmark command line before burning cycles
bench-dry rocm_image='' mainarch_image='localhost/mainarch:latest' ngpus='8' min_bytes='8' max_bytes='134217728':
    DRY_RUN=1 ROCM_IMAGE={{rocm_image}} MAINARCH_IMAGE={{mainarch_image}} bash bench/compare-rccl.sh {{ngpus}} {{min_bytes}} {{max_bytes}}

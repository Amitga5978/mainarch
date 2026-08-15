#!/usr/bin/env bash
# A guided, CPU-only tour of what this stack is made of. No GPU required.
#
#   just tour
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BIN="$REPO/target/release/mainarch"

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
step() { printf '\n\033[2m────────────────────────────────────────────────────────────────────────\033[0m\n'; bold "$1"; dim "$2"; echo; }

[ -x "$BIN" ] || cargo build --release -p mainarch-cli || exit 1

step "1. The GPU binary, read by our own loader" \
     "184 kernels compiled to a gfx950 HSA code object and parsed here with a from-scratch ELF/code-object reader: kernel descriptors, kernarg sizes, LDS, wavefront size."
"$BIN" code-object-info | sed -n '1,20p'
echo "  ..."

step "2. Collectives, rccl-tests shaped" \
     "The same size sweep / correctness oracle / algbw+busbw table rccl-tests prints, against the CPU simulator backend. On hardware the GPU/XGMI backends slot in behind the same trait. Note that the harness says plainly this is a simulator, not a transport."
"$BIN" rccl-test all-reduce --backend cpu-mock --ranks 8 --max-bytes 262144

step "3. Real checkpoint plumbing, CPU side" \
     "SafeTensors header parsing, a Qwen config/index contract preflight, and tensor-parallel rank-shard conversion, all before a single byte reaches VRAM."
"$BIN" weights-qwen-preflight-selftest
"$BIN" weights-qwen-rank-shard-selftest

step "4. The model-facing API boundary" \
     "A typed primitive-graph API. A model definition is compiled, validated, and turned into deterministic, fingerprinted readiness receipts. This layer is CPU-only on purpose: it plans and proves, it does not dispatch."
cargo run --quiet --release -p mainarch-cli --bin mainarch-model-api-selftest 2>/dev/null | sed -n '1,30p'
echo "  ..."

step "5. What a rejected model looks like" \
     "The same boundary, refusing a definition it cannot lower, with typed blockers instead of a panic."
cargo run --quiet --release -p mainarch-core --example rejected_model_api 2>/dev/null | sed -n '1,20p'

printf '\n\033[2m────────────────────────────────────────────────────────────────────────\033[0m\n'
bold "Where to read next"
cat <<'TXT'
  crates/mainarch-sys/src/lib.rs      hand-encoded amdkfd ioctls, the bottom
  crates/mainarch-core/src/gpu.rs     queues, doorbells, AQL packets, signals
  crates/mainarch-core/src/attn.rs    FlashDecoding: split-KV, paged, GQA, FP8/FP4
  crates/mainarch-core/src/layer.rs   RMSNorm, RoPE, SwiGLU, GEMV, MoE FFN
  crates/mainarch-core/src/model.rs   the autoregressive decode loop
  kernels/mainarch_kernels.cl         every GPU kernel, in one file
  docs/model-api.md                   the model-facing API reference

  just demo                           serve the playable page
TXT
echo

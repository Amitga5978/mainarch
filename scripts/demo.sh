#!/usr/bin/env bash
# The one command: build mainarch, prove what this machine can actually do,
# then serve the playable OpenAI-shaped demo page.
#
#   just demo
#
# Works on any x86-64 Linux box. If an AMD GPU is reachable through /dev/kfd it
# runs the live raw-KFD/AQL lane; otherwise it runs the CPU-only lane and says
# so. Nothing here needs ROCm, HIP, HSA, PyTorch, or a GPU vendor runtime.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BIN="$REPO/target/release/mainarch"
BIND="${MAINARCH_DEMO_BIND:-127.0.0.1:8080}"
# Accept a bare port as well as host:port.
case "$BIND" in
  *:*) ;;
  *)   BIND="127.0.0.1:$BIND" ;;
esac
HOST="${BIND%:*}"
PORT="${BIND##*:}"
# 0.0.0.0 is a bind address, not something you type into a browser.
VIEW_HOST="$HOST"
[ "$VIEW_HOST" = "0.0.0.0" ] && VIEW_HOST="127.0.0.1"
[ "$VIEW_HOST" = "::" ] && VIEW_HOST="127.0.0.1"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
dim()   { printf '\033[2m%s\033[0m\n' "$*"; }
rule()  { printf '\033[2m%s\033[0m\n' "────────────────────────────────────────────────────────────────────────"; }

# ---------------------------------------------------------------- GPU detection
# A usable node is a KFD topology node with SIMDs and a real gfx target version.
gpu_node() {
  local n props
  [ -e /dev/kfd ] || return 1
  for n in /sys/class/kfd/kfd/topology/nodes/*/; do
    props="$n/properties"
    [ -r "$props" ] || continue
    awk '
      /^simd_count /          { simd = $2 }
      /^gfx_target_version /  { gfx  = $2 }
      END { if (simd > 0 && gfx > 0) exit 0; else exit 1 }
    ' "$props" && { basename "$n"; return 0; }
  done
  return 1
}

NODE="$(gpu_node || true)"

# ---------------------------------------------------------------------- build
rule
bold "1/3  Building mainarch (release)"
dim  "     one Rust workspace: /dev/kfd ioctls -> device layer -> kernels -> CLI"
echo
cargo build --release -p mainarch-cli || exit 1
echo

# --------------------------------------------------------------------- proofs
rule
bold "2/3  Proving the stack on this machine"
echo

dim "  \$ mainarch code-object-info"
dim "    parse the embedded gfx950 HSA code object with our own ELF loader"
echo
"$BIN" code-object-info 2>&1 | sed -n '1,12p'
echo "  ... (full table: mainarch code-object-info)"
echo

dim "  \$ mainarch rccl-test all-reduce --backend cpu-mock --ranks 8"
dim "    the rccl-tests-shaped harness, CPU simulator backend, correctness oracle"
echo
"$BIN" rccl-test all-reduce --backend cpu-mock --ranks 8 --max-bytes 262144 2>&1
echo

if [ -n "$NODE" ]; then
  dim "  \$ mainarch probe"
  dim "    open /dev/kfd, read the amdkfd version, enumerate the GPU topology"
  echo
  "$BIN" probe 2>&1 | sed -n '1,30p'
  echo

  dim "  \$ mainarch gpu-selftest"
  dim "    hand-built AQL dispatch packet + doorbell ring; the GPU stamps memory"
  echo
  if "$BIN" gpu-selftest > /tmp/mainarch-gpu-selftest.$$ 2>&1; then
    sed -n '1,10p' /tmp/mainarch-gpu-selftest.$$
  else
    sed -n '1,10p' /tmp/mainarch-gpu-selftest.$$
    echo
    dim "  the GPU is present but would not execute. Another process may hold"
    dim "  the device, or the container may lack render-group access."
    dim "  -> falling back to the CPU-only serving lane"
    NODE=""
  fi
  rm -f /tmp/mainarch-gpu-selftest.$$
  echo
else
  dim "  no AMD GPU reachable through /dev/kfd on this machine"
  dim "  -> skipping the live probe / AQL dispatch proofs"
  dim "  -> on an MI355X host, reopen in the .devcontainer/gpu configuration"
  echo
fi

# ----------------------------------------------------------------------- serve
rule
bold "3/3  Serving the demo"
echo

if [ -n "$NODE" ] && [ "${MAINARCH_DEMO_STATIC:-0}" != "1" ]; then
  dim "  live lane: mainarch demo-serve on KFD topology node $NODE"
  dim "  the same release binary that ran the proofs above owns the decode lane"
  echo
  bold "  open  http://${VIEW_HOST}:${PORT}/"
  echo
  exec "$BIN" demo-serve --bind "$BIND" --node "$NODE"
else
  dim "  CPU-only lane: the dependency-free Python sandbox in demo/sandbox"
  dim "  same page, same /v1/chat/completions contract, scripted responses"
  echo
  bold "  open  http://${VIEW_HOST}:${PORT}/"
  echo
  exec python3 demo/sandbox/server.py --host "$HOST" --port "$PORT"
fi

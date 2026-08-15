#!/usr/bin/env bash
# Compile mainarch GPU kernels to a gfx950 HSA code object.
#
# You do NOT need to run this to use mainarch: the compiled object is committed
# at crates/mainarch-core/artifacts/mainarch_kernels.gfx950.co and embedded in
# the binary. Run it only when you change mainarch_kernels.cl.
#
# It uses an LLVM that can target amdgcn purely as a build-time cross-compiler.
# No ROCm runtime is involved at mainarch execution time — the resulting .co is
# loaded and dispatched through the raw KFD/AQL path.
#
# Any container image with an amdgpu-capable clang works. Set:
#   MAINARCH_BUILD_IMAGE   image to compile in
#                          (default docker.io/rocm/dev-ubuntu-24.04:latest;
#                           rocm/dev-ubuntu-22.04:<ver> also works, as does any
#                           image with upstream LLVM >= 18 built with the
#                           AMDGPU target)
#   MAINARCH_CLANG         path to clang inside that image
#                          (default /opt/rocm/llvm/bin/clang)
#   MAINARCH_GFX           target arch (default gfx950)
#
# The source is piped in over stdin and the compiled object comes back
# base64-encoded over stdout, so no writable path has to be shared with the
# build container.
#
# Usage: kernels/build.sh [output_path]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="${1:-$REPO/crates/mainarch-core/artifacts/mainarch_kernels.gfx950.co}"
IMAGE="${MAINARCH_BUILD_IMAGE:-docker.io/rocm/dev-ubuntu-24.04:latest}"
MCPU="${MAINARCH_GFX:-gfx950}"
# Path to an amdgpu-capable clang *inside* the build image.
CLANG="${MAINARCH_CLANG:-/opt/rocm/llvm/bin/clang}"
ENGINE="${MAINARCH_CONTAINER_ENGINE:-}"
EXTRA_DEFS=()

if [[ -n "${MAINARCH_WPS_ATTN:-}" ]]; then
  if [[ ! "$MAINARCH_WPS_ATTN" =~ ^[0-9]+$ ]]; then
    echo "MAINARCH_WPS_ATTN must be an unsigned integer" >&2
    exit 2
  fi
  EXTRA_DEFS+=("-DMAINARCH_WPS_ATTN=$MAINARCH_WPS_ATTN")
fi
if [[ -n "${MAINARCH_UATTN:-}" ]]; then
  if [[ ! "$MAINARCH_UATTN" =~ ^[0-9]+$ ]]; then
    echo "MAINARCH_UATTN must be an unsigned integer" >&2
    exit 2
  fi
  EXTRA_DEFS+=("-DMAINARCH_UATTN=$MAINARCH_UATTN")
fi
EXTRA_DEFS_STR="${EXTRA_DEFS[*]}"

if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then
    ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then
    ENGINE=podman
  else
    echo "no container engine found; set MAINARCH_CONTAINER_ENGINE=docker or podman" >&2
    exit 2
  fi
fi

mkdir -p "$(dirname "$OUT")"
# Source is piped in over stdin (it has outgrown the env-arg size limit), the
# compiled .co comes back base64 over stdout. No ROCm runtime is involved.
# MAINARCH_CLANG overrides the compiler path for images that place clang elsewhere.
"$ENGINE" run --rm -i --entrypoint bash "$IMAGE" -lc '
set -euo pipefail
cat > /tmp/k.cl
'"$CLANG"' -x cl -cl-std=CL2.0 \
  --target=amdgcn-amd-amdhsa -mcpu='"$MCPU"' -O3 \
  '"$EXTRA_DEFS_STR"' \
  -Xclang -finclude-default-header \
  /tmp/k.cl -o /tmp/k.co
"$(dirname '"$CLANG"')/llvm-readelf" --notes /tmp/k.co 1>&2 | grep -E "kernarg_segment_size|amdhsa.target" 1>&2 || true
base64 -w0 /tmp/k.co
' < "$HERE/mainarch_kernels.cl" > "$OUT.b64"

base64 -d "$OUT.b64" > "$OUT"
rm -f "$OUT.b64"
echo "wrote $OUT"
ls -l "$OUT"
file "$OUT"

#!/usr/bin/env bash
# compare-rccl.sh: side-by-side rccl all-reduce, ROCm baseline vs mainarch.
#
# THE PATTERN (see README / docs/ROADMAP.md):
#   Run this ON THE HOST with access to the container engine.
#   It launches two *sibling* containers on the same engine:
#       1. a ROCm baseline image  -> upstream rccl-tests (all_reduce_perf)
#       2. the mainarch image     -> `mainarch rccl-test all-reduce`
#   Each in its own image (the two stacks never contaminate each other), each
#   with the GPUs passed through, run SEQUENTIALLY so each gets all the GPUs.
#
# Usage:
#   ROCM_IMAGE=<rocm img w/ rccl-tests> MAINARCH_IMAGE=<mainarch img> \
#     DRY_RUN=1 bench/compare-rccl.sh [NGPUS] [MIN_BYTES] [MAX_BYTES]
set -euo pipefail

NGPUS="${1:-8}"
MIN="${2:-8}"
MAX="${3:-134217728}"   # 128 MiB in bytes (same units for both tools)
BASELINE_USE_MPI="${BASELINE_USE_MPI:-auto}"  # auto|0|1
BASELINE_MPI_WRAPPER="${BASELINE_MPI_WRAPPER:-mpirun --allow-run-as-root}"
BASELINE_MPI_ARGS="${BASELINE_MPI_ARGS:---np ${NGPUS} --bind-to numa}"
BASELINE_MPIRUN_ENV="${BASELINE_MPIRUN_ENV:-NCCL_DEBUG=VERSION}"

DRY_RUN="${DRY_RUN:-0}"
DOCKER_BIN="${DOCKER_BIN:-${DOCKER:-docker}}" # override with DOCKER or DOCKER_BIN
if [ -z "$DOCKER_BIN" ] && [ -S /run/podman/podman.sock ] && command -v podman >/dev/null 2>&1; then
  DOCKER_BIN=podman
fi
ROCM_IMAGE="${ROCM_IMAGE:-}"               # must contain a built rccl-tests
MAINARCH_IMAGE="${MAINARCH_IMAGE:-localhost/mainarch:latest}"
MAINARCH_BACKEND="${MAINARCH_BACKEND:-gpu}"
OUTDIR="${OUTDIR:-/tmp/mainarch-rccl-compare}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BASELINE_OUT="${OUTDIR}/rccl-baseline-${TS}.txt"
MAINARCH_OUT="${OUTDIR}/mainarch-${TS}.txt"

if command -v "$DOCKER_BIN" >/dev/null 2>&1; then
  DOCKER="$DOCKER_BIN"
elif command -v podman >/dev/null 2>&1; then
  DOCKER=podman
elif [ "$DOCKER_BIN" = "podman" ] && command -v docker >/dev/null 2>&1; then
  DOCKER=docker
else
  echo "ERROR: container CLI '$DOCKER_BIN' not found on PATH."
  echo "Set DOCKER_BIN or DOCKER to an available binary (docker or podman)."
  exit 1
fi

ROOTLESS_ENGINE=0

GPU_ARGS=(
  --rm
  --device=/dev/kfd --device=/dev/dri
  --security-opt=seccomp=unconfined
  --ipc=host
  --network=host
  --cap-add=SYS_PTRACE
)

if security_options="$($DOCKER info --format '{{json .SecurityOptions}}' 2>/dev/null)"; then
  case "$security_options" in
    *'"name=rootless"'*) ROOTLESS_ENGINE=1;;
  esac
fi

if [ "$DOCKER" != "podman" ] && [ "$ROOTLESS_ENGINE" -eq 0 ]; then
  GPU_ARGS+=(--group-add=keep-groups)
fi

rootless_access_hint() {
  local file="$1"
  local label="$2"
  if [ "$ROOTLESS_ENGINE" -eq 1 ] && [ -f "$file" ] && grep -qiE 'no ROCm-capable device|permission denied|/dev/kfd' "$file"; then
    echo ""
    echo "  HINT (${label}): this host/container engine is rootless."
    echo "  Podman docs require supplementary-group passthrough for device ACLs; keep-groups is not supported in remote API mode."
    echo "  Sibling-container GPU passthrough is currently blocked on this setup; host fallback stays enabled."
  fi
}

hr() { printf '%.0s=' {1..78}; echo; }

has_mpi_runner() {
  local image="$1"
  if [ "$DRY_RUN" = "1" ]; then
    echo 0
    return 0
  fi
  local mpi_present
  mpi_present="$("$DOCKER" run --rm --entrypoint /bin/bash "$image" -lc 'command -v mpirun || command -v mpirun.mpich || command -v mpiexec || true' 2>/dev/null | tr -d '[:space:]' || true)"
  if [ -n "$mpi_present" ]; then
    echo 1
  else
    echo 0
  fi
}

extract_max_busbw() {
  local file="$1"
  awk '
    /^\s*Avg[[:space:]]+bus[[:space:]]+bandwidth/ {
      if ($NF ~ /^-?[0-9]+(\.[0-9]+)?$/) avg=$NF
    }
    /^[[:space:]]*[0-9]+(\.?[0-9]+)?[[:space:]]+[0-9]+/ {
      if (NF >= 12) {
        if ($8 ~ /^-?[0-9]+(\.[0-9]+)?$/ && $8 > max) max=$8
        if ($12 ~ /^-?[0-9]+(\.[0-9]+)?$/ && $12 > max) max=$12
      } else if (NF >= 5) {
        if ($5 ~ /^-?[0-9]+(\.[0-9]+)?$/ && $5 > max) max=$5
      }
    }
    END {
      if (max != "") { print max; exit 0 }
      if (avg != "") { print avg; exit 0 }
      exit 1
    }
  ' "$file"
}

summarize() {
  local label="$1"
  local file="$2"
  if [ -s "$file" ]; then
    local max_busbw
    if max_busbw=$(extract_max_busbw "$file"); then
      printf '%-11s max busbw: %8.2f GB/s\n' "$label" "$max_busbw"
    else
      printf '%-11s max busbw: %s\n' "$label" "n/a (parser)"
    fi
  else
    printf '%-11s max busbw: %s\n' "$label" "missing"
  fi
}

ratio() {
  local base="$1"
  local other="$2"
  if [ -z "$base" ] || [ -z "$other" ]; then
    echo "n/a"
    return 0
  fi
  awk -v base="$base" -v other="$other" 'BEGIN {
    if (base <= 0) { printf "n/a"; exit 0 }
    printf "%.2f", other / base
  }'
}

delta_pct() {
  local base="$1"
  local other="$2"
  if [ -z "$base" ] || [ -z "$other" ]; then
    echo "n/a"
    return 0
  fi
  awk -v base="$base" -v other="$other" 'BEGIN {
    if (base <= 0) { printf "n/a"; exit 0 }
    printf "%.2f", ((other - base) / base) * 100
  }'
}

has_image() {
  "$DOCKER" image inspect "$1" >/dev/null 2>&1
}

run_bench() {
  local image="$1"
  shift
  local docker_opts=()

  while [[ "$1" == --* ]]; do
    if [ "$1" = "--entrypoint" ]; then
      docker_opts+=("$1" "$2")
      shift 2
      continue
    fi

    docker_opts+=("$1")
    shift
  done

  local cmd=("$@")

  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY-RUN: $DOCKER run ${GPU_ARGS[*]} ${docker_opts[*]} $image ${cmd[*]}"
    return 0
  fi

  "$DOCKER" run "${GPU_ARGS[@]}" "${docker_opts[@]}" "$image" "${cmd[@]}" 2>&1
}

# Shared GPU passthrough for ANY benchmark container on the MI355 host.
# Mirrors the dev container's runArgs so the comparison sees the same devices.
gpu_info() {
  echo "host engine : $($DOCKER version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"
  echo "cli         : $DOCKER"
  if [ "$ROOTLESS_ENGINE" -eq 1 ]; then
    echo "engine mode : rootless (container device ACL passthrough may require keep-groups)"
  else
    echo "engine mode : rootful"
  fi
  echo "gpus=$NGPUS  sizes=${MIN}..${MAX} bytes"
}

gpu_info
mkdir -p "$OUTDIR"
hr
ROCM_MPI_CAPABLE=0
if [ -n "$ROCM_IMAGE" ] && [ "$BASELINE_USE_MPI" != "0" ]; then
  ROCM_MPI_CAPABLE="$(has_mpi_runner "$ROCM_IMAGE")"
fi

BASELINE_USE_MPI_MODE=0
if [ "$BASELINE_USE_MPI" = "1" ]; then
  BASELINE_USE_MPI_MODE=1
elif [ "$BASELINE_USE_MPI" = "auto" ] && [ "$ROCM_MPI_CAPABLE" -eq 1 ]; then
  BASELINE_USE_MPI_MODE=1
fi
if [ "$BASELINE_USE_MPI" = "auto" ] && [ "$ROCM_MPI_CAPABLE" -eq 0 ]; then
  echo "INFO: ROCM image does not expose mpirun; using single-process baseline."
fi

# ---- 1) ROCm baseline: upstream rccl-tests --------------------------------
echo ">> ROCm baseline  (${ROCM_IMAGE:-<unset>})"
BASELINE_OK=0
MAINARCH_COMPARE=0
if [ -z "$ROCM_IMAGE" ]; then
  echo "   SKIPPED. Set ROCM_IMAGE to a ROCm image with rccl-tests built."
  echo "   See baseline/README.md to build one (ROCm + RCCL + rccl-tests)."
else
  if [ "$DRY_RUN" != "1" ] && ! has_image "$ROCM_IMAGE"; then
    echo "WARNING: ROCM_IMAGE '$ROCM_IMAGE' is not present locally; docker/podman will try to pull it."
  fi
  if [ "$BASELINE_USE_MPI_MODE" -eq 1 ]; then
    BASELINE_CMD="$BASELINE_MPI_WRAPPER $BASELINE_MPI_ARGS -x $BASELINE_MPIRUN_ENV all_reduce_perf -b $MIN -e $MAX -f 2 -g 1"
    echo "  RUN MODE: baseline multi-process (--bind-to numa, -g 1)"
    if run_bench "$ROCM_IMAGE" --entrypoint /bin/bash -c "$BASELINE_CMD" | tee "$BASELINE_OUT"; then
      BASELINE_OK=1
    else
      echo "   WARN: ROCm baseline MPI run did not complete successfully."
      rootless_access_hint "$BASELINE_OUT" "baseline"
    fi
  else
    if run_bench "$ROCM_IMAGE" --entrypoint /bin/bash -c "all_reduce_perf -b $MIN -e $MAX -f 2 -g $NGPUS" | tee "$BASELINE_OUT"; then
      BASELINE_OK=1
    else
      echo "   WARN: ROCm baseline run did not complete successfully."
      rootless_access_hint "$BASELINE_OUT" "baseline"
    fi
  fi
fi

hr

# ---- 2) mainarch under test -----------------------------------------------
echo ">> mainarch  ($MAINARCH_IMAGE)"
MAINARCH_OK=0
MAINARCH_SIMULATED=0
if [ "$DRY_RUN" != "1" ] && ! has_image "$MAINARCH_IMAGE"; then
  echo "ERROR: MAINARCH_IMAGE '$MAINARCH_IMAGE' not found locally."
  echo "Build with: just build-mainarch-image mainarch_image=$MAINARCH_IMAGE"
  exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
  run_bench "$MAINARCH_IMAGE" rccl-test all-reduce --backend "$MAINARCH_BACKEND" --ranks "$NGPUS" --min-bytes "$MIN" --max-bytes "$MAX" | tee "$MAINARCH_OUT"
  MAINARCH_OK=1
  if [ -f "$MAINARCH_OUT" ] && grep -q "backend exposes 1" "$MAINARCH_OUT"; then
    MAINARCH_COMPARE=0
  else
    MAINARCH_COMPARE=1
  fi
  if [ -f "$MAINARCH_OUT" ] && grep -qi "placeholder transport\\|cpu-mock" "$MAINARCH_OUT"; then
    MAINARCH_SIMULATED=1
  fi
else
  if run_bench "$MAINARCH_IMAGE" rccl-test all-reduce --backend "$MAINARCH_BACKEND" --ranks "$NGPUS" --min-bytes "$MIN" --max-bytes "$MAX" | tee "$MAINARCH_OUT"; then
      MAINARCH_OK=1
      if [ -f "$MAINARCH_OUT" ] && grep -q "backend exposes 1" "$MAINARCH_OUT"; then
        MAINARCH_COMPARE=0
      else
        MAINARCH_COMPARE=1
      fi
      if [ -f "$MAINARCH_OUT" ] && grep -qi "placeholder transport\\|cpu-mock" "$MAINARCH_OUT"; then
        MAINARCH_SIMULATED=1
      fi
  else
    echo "   WARN: container mainarch run did not complete successfully."
    rootless_access_hint "$MAINARCH_OUT" "mainarch"
    echo "   FALLBACK: attempting host fallback (./target/release/mainarch) for signal-only comparison."
    if ./target/release/mainarch rccl-test all-reduce --backend "$MAINARCH_BACKEND" --ranks "$NGPUS" --min-bytes "$MIN" --max-bytes "$MAX" | tee "$MAINARCH_OUT"; then
      MAINARCH_OK=1
      if [ -f "$MAINARCH_OUT" ] && grep -q "backend exposes 1" "$MAINARCH_OUT"; then
        MAINARCH_COMPARE=0
      else
        MAINARCH_COMPARE=1
      fi
      if [ -f "$MAINARCH_OUT" ] && grep -qi "placeholder transport\\|cpu-mock" "$MAINARCH_OUT"; then
        MAINARCH_SIMULATED=1
      fi
    fi
  fi
fi
hr

echo "baseline -> $BASELINE_OUT"
echo "mainarch -> $MAINARCH_OUT"
  if [ -f "$BASELINE_OUT" ] || [ -f "$MAINARCH_OUT" ]; then
  echo ""
  summarize "baseline" "$BASELINE_OUT"
  summarize "mainarch" "$MAINARCH_OUT"

  if [ "$BASELINE_OK" -eq 1 ] && [ "$MAINARCH_OK" -eq 1 ] &&
     [ -f "$BASELINE_OUT" ] && [ -s "$BASELINE_OUT" ] &&
     [ -f "$MAINARCH_OUT" ] && [ -s "$MAINARCH_OUT" ]; then
    if [ "$MAINARCH_COMPARE" -ne 1 ]; then
      echo "  STATUS: MAINARCH is running single-rank backend; apples-to-apples ratio is not valid yet."
    elif [ "$MAINARCH_SIMULATED" -eq 1 ]; then
      echo "  STATUS: MAINARCH backend is simulated (cpu-mock); ratio is useful for harness trend but not transport parity."
    fi

    BASELINE_MAX=$(extract_max_busbw "$BASELINE_OUT" || true)
    MAINARCH_MAX=$(extract_max_busbw "$MAINARCH_OUT" || true)
    if [ "$MAINARCH_COMPARE" -eq 1 ] && [ "$MAINARCH_SIMULATED" -eq 0 ]; then
      SPEEDUP=$(ratio "$BASELINE_MAX" "$MAINARCH_MAX")
      DELTA=$(delta_pct "$BASELINE_MAX" "$MAINARCH_MAX")
      printf 'mainarch vs baseline peak busbw:  ratio=%s  delta=%s%%\n' "$SPEEDUP" "$DELTA"
      if [ "$SPEEDUP" != "n/a" ]; then
        if awk "BEGIN { exit !($SPEEDUP >= 1) }"; then
          echo "  STATUS: mainarch peak is ahead of baseline at this snapshot."
        else
          echo "  STATUS: mainarch peak is behind baseline at this snapshot."
        fi
      fi
    fi
  else
    echo "  STATUS: both outputs not present yet. Set ROCM_IMAGE and rerun for a true apples-to-apples ratio."
  fi
fi

echo ""
echo "Compare the busbw columns. If baseline and mainarch are present,"
echo "the ratio/delta printed above is the apples-to-apples signal for this run."

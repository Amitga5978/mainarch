#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
# Prefer whatever the machine actually has; override with
# MAINARCH_CONTAINER_ENGINE or DOCKER_BIN.
default_engine() {
  if [ -n "${DOCKER_BIN:-}" ]; then printf '%s' "$DOCKER_BIN"; return; fi
  if command -v docker >/dev/null 2>&1; then printf 'docker'; return; fi
  if command -v podman >/dev/null 2>&1; then printf 'podman'; return; fi
  printf 'docker'
}
ENGINE="${MAINARCH_CONTAINER_ENGINE:-$(default_engine)}"
IMAGE="${MAINARCH_DEMO_IMAGE:-localhost/mainarch-demo:latest}"
NAME="${MAINARCH_DEMO_CONTAINER_NAME:-mainarch-demo}"
BIND="${MAINARCH_DEMO_BIND:-0.0.0.0:8080}"
NODE="${MAINARCH_DEMO_NODE:-2}"
LIVE="${MAINARCH_DEMO_LIVE:-0}"
URL="${MAINARCH_DEMO_URL:-http://127.0.0.1:${BIND##*:}}"
BIN="${MAINARCH_DEMO_BIN:-./target/release/mainarch}"

usage() {
  cat <<USAGE
usage: bash tools/demo_container.sh [build|run|stop|smoke|logs|status]

Environment:
  MAINARCH_CONTAINER_ENGINE      podman or docker, default ${ENGINE}
  MAINARCH_DEMO_IMAGE            image tag, default ${IMAGE}
  MAINARCH_DEMO_CONTAINER_NAME   container name, default ${NAME}
  MAINARCH_DEMO_BIND             bind inside host network, default ${BIND}
  MAINARCH_DEMO_NODE             topology node, default ${NODE}
  MAINARCH_DEMO_LIVE=1           run the live mainarch binary inside the container
  MAINARCH_DEMO_URL              smoke URL, default ${URL}

Examples:
  bash tools/demo_container.sh build
  bash tools/demo_container.sh run
  bash tools/demo_container.sh smoke
  bash tools/demo_container.sh stop
USAGE
}

require_engine() {
  if ! command -v "${ENGINE}" >/dev/null 2>&1; then
    echo "missing container engine: ${ENGINE}" >&2
    exit 1
  fi
}

preflight_gpu() {
  if [ -d /sys/class/kfd/kfd/proc ] && [ "${MAINARCH_DEMO_FORCE:-0}" != "1" ]; then
    if find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
      echo "refusing to start: KFD process table is non-empty" >&2
      find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -print 2>/dev/null >&2 || true
      echo "set MAINARCH_DEMO_FORCE=1 only if you intentionally want to share the GPU node" >&2
      exit 1
    fi
  fi
}

preflight_port() {
  local port="${BIND##*:}"
  if [ "${MAINARCH_DEMO_FORCE:-0}" = "1" ]; then
    return
  fi
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      echo "refusing to start: host port ${port} is already listening" >&2
      ss -ltnp 2>/dev/null | grep -E "(^|:)${port}[[:space:]]" >&2 || true
      echo "set MAINARCH_DEMO_BIND=0.0.0.0:18080 or MAINARCH_DEMO_FORCE=1" >&2
      exit 1
    fi
  fi
}

container_exists() {
  "${ENGINE}" container exists "${NAME}" >/dev/null 2>&1
}

build_image() {
  require_engine
  cargo build --release -p mainarch-cli
  "${ENGINE}" build --tag "${IMAGE}" -f Dockerfile.mainarch-demo .
}

run_container() {
  require_engine
  local port="${BIND##*:}"
  local publish="${port}:${port}"
  case "${BIND%:*}" in
    127.0.0.1|localhost)
      publish="127.0.0.1:${port}:${port}"
      ;;
  esac
  if [ "${LIVE}" = "1" ]; then
    preflight_gpu
  fi
  preflight_port
  if container_exists; then
    "${ENGINE}" rm -f "${NAME}" >/dev/null
  fi
  local args=(
    run -d
    --name "${NAME}" \
    --publish "${publish}" \
    --ipc=host \
    -e MAINARCH_DEMO_BIND="${BIND}" \
    -e MAINARCH_DEMO_NODE="${NODE}" \
    -e MAINARCH_DEMO_LIVE="${LIVE}" \
    -e MAINARCH_DEMO_MODE="${MAINARCH_DEMO_MODE:-scripted}" \
    -e MAINARCH_DEMO_UPSTREAM="${MAINARCH_DEMO_UPSTREAM:-}" \
    -e MAINARCH_DEMO_MODEL="${MAINARCH_DEMO_MODEL:-}" \
    -e MAINARCH_DEMO_PUBLIC_URL="${MAINARCH_DEMO_PUBLIC_URL:-}" \
  )
  if [ "${LIVE}" = "1" ]; then
    args+=(
      --device=/dev/kfd
      --device=/dev/dri
      --security-opt seccomp=unconfined
      --security-opt label=disable
      --cap-add SYS_PTRACE
    )
  fi
  args+=("${IMAGE}" serve)
  "${ENGINE}" "${args[@]}"
  echo "mainarch demo container started: ${NAME}"
  echo "open: ${URL}"
}

smoke_container() {
  curl -fsS --max-time 20 "${URL}/api/health" >/dev/null
  curl -fsS --max-time 20 "${URL}/api/demo" >/dev/null
  curl -fsS --max-time 20 "${URL}/v1/models" >/dev/null
  if [ "${LIVE}" = "1" ]; then
    curl -fsS --max-time 20 "${URL}/api/evidence" >/dev/null
  else
    curl -fsS --max-time 20 "${URL}/api/status" >/dev/null
  fi
  curl -fsS --max-time 25 \
    -H 'Content-Type: application/json' \
    -d '{"model":"mainarch-qwen3-235b-a22b-synthetic-proof","stream":true,"messages":[{"role":"user","content":"what is this demo proving?"}]}' \
    "${URL}/v1/chat/completions" | sed -n '1,18p'
  printf '\nmainarch demo container smoke passed: %s\n' "${URL}"
}

case "${MODE}" in
  build) build_image ;;
  run) run_container ;;
  stop)
    require_engine
    container_exists && "${ENGINE}" rm -f "${NAME}" || true
    ;;
  smoke) smoke_container ;;
  logs)
    require_engine
    "${ENGINE}" logs --tail 80 "${NAME}"
    ;;
  status)
    require_engine
    "${ENGINE}" ps --filter "name=${NAME}"
    ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac

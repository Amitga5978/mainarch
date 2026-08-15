#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-serve}"
# Localhost by default. This serves an LLM with no auth, no quota and no rate
# limit, so binding every interface has to be a decision someone makes rather
# than one they inherit. Set MAINARCH_DEMO_BIND=0.0.0.0:8080 to expose it, and
# put a reverse proxy in front when you do.
BIND="${MAINARCH_DEMO_BIND:-127.0.0.1:8080}"
NODE="${MAINARCH_DEMO_NODE:-2}"
BIN="${MAINARCH_DEMO_BIN:-./target/release/mainarch}"
PORT="${BIND##*:}"
URL="${MAINARCH_DEMO_URL:-http://127.0.0.1:${PORT}}"
CURL_TIMEOUT="${MAINARCH_DEMO_CURL_TIMEOUT:-15}"
MODEL="mainarch-qwen3-235b-a22b-synthetic-proof"

usage() {
    cat <<EOF
usage: bash tools/demo_sandbox.sh [build|serve|smoke|env]

Environment:
  MAINARCH_DEMO_BIND         bind address for serve mode, default ${BIND}
  MAINARCH_DEMO_NODE         GPU topology node, default ${NODE}
  MAINARCH_DEMO_BIN          mainarch binary, default ${BIN}
  MAINARCH_DEMO_URL          base URL for smoke mode, default ${URL}
  MAINARCH_DEMO_FORCE=1      allow serve mode when /sys/class/kfd/kfd/proc is non-empty

Examples:
  bash tools/demo_sandbox.sh build
  MAINARCH_DEMO_BIND=0.0.0.0:8080 MAINARCH_DEMO_NODE=2 bash tools/demo_sandbox.sh serve   # expose it
  MAINARCH_DEMO_URL=http://127.0.0.1:8080 bash tools/demo_sandbox.sh smoke
EOF
}

preflight_serve() {
    if pgrep -af '[m]ainarch .*demo-serve' >/dev/null; then
        echo "refusing to start: a mainarch demo-serve process already appears to be running" >&2
        pgrep -af '[m]ainarch .*demo-serve' >&2 || true
        exit 1
    fi

    if [ "${MAINARCH_DEMO_FORCE:-0}" != "1" ] && [ -d /sys/class/kfd/kfd/proc ]; then
        if find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
            echo "refusing to start: KFD process table is non-empty" >&2
            find /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -print 2>/dev/null >&2 || true
            echo "set MAINARCH_DEMO_FORCE=1 only if you intentionally want to share the GPU node" >&2
            exit 1
        fi
    fi

    if command -v dmesg >/dev/null 2>&1; then
        recent_faults="$(dmesg -T 2>/dev/null | tail -n 180 | grep -Ei 'amdgpu|kfd|gpu reset|page fault|wedged|failed to quiesce' || true)"
        if [ -n "${recent_faults}" ]; then
            echo "recent amdgpu/kfd messages before demo launch:" >&2
            printf '%s\n' "${recent_faults}" >&2
        fi
    fi
}

smoke() {
    curl -fsS --max-time "${CURL_TIMEOUT}" "${URL}/" >/dev/null
    curl -fsS --max-time "${CURL_TIMEOUT}" "${URL}/api/health"
    printf '\n'
    curl -fsS --max-time "${CURL_TIMEOUT}" "${URL}/api/demo" >/dev/null
    curl -fsS --max-time "${CURL_TIMEOUT}" "${URL}/v1/models"
    printf '\n'
    curl -fsS --max-time "${CURL_TIMEOUT}" "${URL}/api/evidence" >/dev/null
    curl -fsS --max-time "${CURL_TIMEOUT}" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"why is this fast?\"}]}" \
        "${URL}/v1/chat/completions" | sed -n '1,24p'
    printf '\nmainarch sandbox smoke passed: %s\n' "${URL}"
}

case "${MODE}" in
    build)
        cargo build --release
        ;;
    serve)
        if [ ! -x "${BIN}" ]; then
            echo "missing executable ${BIN}; run: bash tools/demo_sandbox.sh build" >&2
            exit 1
        fi
        preflight_serve
        exec "${BIN}" demo-serve --bind "${BIND}" --node "${NODE}"
        ;;
    smoke)
        smoke
        ;;
    env)
        printf 'MAINARCH_DEMO_BIND=%s\n' "${BIND}"
        printf 'MAINARCH_DEMO_NODE=%s\n' "${NODE}"
        printf 'MAINARCH_DEMO_BIN=%s\n' "${BIN}"
        printf 'MAINARCH_DEMO_URL=%s\n' "${URL}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

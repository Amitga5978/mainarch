#!/usr/bin/env bash
set -euo pipefail

BIND="${MAINARCH_DEMO_BIND:-0.0.0.0:8080}"
NODE="${MAINARCH_DEMO_NODE:-2}"
HOST="${BIND%:*}"
PORT="${BIND##*:}"
MODE="${1:-serve}"

if [ "${HOST}" = "${PORT}" ]; then
  HOST="0.0.0.0"
fi

case "${MODE}" in
  serve)
    if [ "${MAINARCH_DEMO_LIVE:-0}" = "1" ]; then
      exec /usr/local/bin/mainarch demo-serve --bind "${BIND}" --node "${NODE}"
    fi
    cd /app/demo/sandbox
    exec python3 server.py --host "${HOST}" --port "${PORT}"
    ;;
  static)
    cd /app/demo/sandbox
    exec python3 server.py --host "${HOST}" --port "${PORT}"
    ;;
  live)
    exec /usr/local/bin/mainarch demo-serve --bind "${BIND}" --node "${NODE}"
    ;;
  mainarch)
    shift
    exec /usr/local/bin/mainarch "$@"
    ;;
  *)
    exec /usr/local/bin/mainarch "$@"
    ;;
esac

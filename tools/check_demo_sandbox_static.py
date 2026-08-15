#!/usr/bin/env python3
"""Smoke-test the CPU-only static sandbox demo boundary."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "demo/sandbox/server.py"
MODEL = "mainarch-static-sandbox-check"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, *, data: dict | None = None) -> dict:
    body = None
    headers = {}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read().decode("utf-8"))


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(detail)


def wait_for_server(base_url: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"static sandbox exited early with code {proc.returncode}")
        try:
            health = request_json(f"{base_url}/api/health")
            require(health.get("ok") is True, "health endpoint did not report ok=true")
            return
        except Exception as exc:  # noqa: BLE001 - keep polling until deadline.
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"static sandbox did not become ready: {last_error}")


def parse_openai_stream(url: str) -> tuple[str, list[dict], bool]:
    payload = {
        "model": MODEL,
        "stream": True,
        "messages": [{"role": "user", "content": "what is this demo proving?"}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    chunks: list[dict] = []
    text_parts: list[str] = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=10) as res:
        content_type = res.headers.get("content-type", "")
        require(
            content_type.startswith("text/event-stream"),
            f"stream content-type {content_type!r} is not text/event-stream",
        )
        for raw in res:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            choices = chunk.get("choices") or []
            if choices:
                text_parts.append(choices[0].get("delta", {}).get("content", ""))
    return "".join(text_parts), chunks, saw_done


def check_static_sandbox(base_url: str) -> None:
    health = request_json(f"{base_url}/api/health")
    require(health["service"] == "mainarch-demo-static", "health service drifted")
    require(health["mode"] == "scripted", "health mode is not scripted")
    require(health["backend_ready"] is True, "scripted backend should be ready")
    require(
        health["chat_endpoint"] == "/v1/chat/completions",
        "health chat endpoint drifted",
    )

    status = request_json(f"{base_url}/api/status")
    require(status["mode_label"] == "scripted preview", "status mode label drifted")
    require(status["model"] == MODEL, "status model drifted")
    require(status["backend_ready"] is True, "status backend is not ready")

    manifest = request_json(f"{base_url}/api/demo")
    require(manifest["ok"] is True, "manifest did not report ok=true")
    require(manifest["backend_ready"] is True, "manifest backend is not ready")
    require(
        "/v1/chat/completions"
        in " ".join(manifest.get("what_is_real", [])),
        "manifest does not advertise the OpenAI-shaped chat seam",
    )
    not_claimed = " ".join(manifest.get("what_is_not_claimed", []))
    require(
        "real Qwen or Kimi weights in scripted mode" in not_claimed,
        "manifest no longer names the scripted real-weight boundary",
    )

    models = request_json(f"{base_url}/v1/models")
    require(models["object"] == "list", "models object is not list")
    require(len(models["data"]) == 1, "models endpoint should expose one sandbox model")
    model = models["data"][0]
    require(model["id"] == MODEL, "models endpoint id drifted")
    require(
        model["mainarch"]["scripted_mode_is_synthetic"] is True,
        "models endpoint must expose scripted synthetic boundary",
    )

    full = request_json(
        f"{base_url}/v1/chat/completions",
        data={
            "model": MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Qwen boundary?"}],
        },
    )
    require(full["object"] == "chat.completion", "non-stream response object drifted")
    require(full["model"] == MODEL, "non-stream response model drifted")
    content = full["choices"][0]["message"]["content"]
    require("scripted preview mode" in content, "non-stream response hides synthetic boundary")

    streamed_text, chunks, saw_done = parse_openai_stream(f"{base_url}/v1/chat/completions")
    require(saw_done, "stream did not emit [DONE]")
    require(streamed_text.strip(), "stream emitted no text")
    first = next((chunk for chunk in chunks if chunk.get("mainarch")), None)
    require(first is not None, "stream chunks did not include mainarch metadata")
    require(first["mainarch"]["demo_mode"] == "scripted", "stream demo mode drifted")
    require(
        first["mainarch"]["proof"] == "synthetic-preview",
        "stream proof boundary drifted",
    )


def main() -> int:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "MAINARCH_DEMO_MODE": "scripted",
            "MAINARCH_DEMO_MODEL": MODEL,
            "MAINARCH_DEMO_TOKEN_DELAY": "0",
            "MAINARCH_DEMO_QUIET": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_server(base_url, proc)
        check_static_sandbox(base_url)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    print(f"static sandbox boundary ok: {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

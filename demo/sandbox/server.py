#!/usr/bin/env python3
import argparse
import json
import os
import select
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
DEFAULT_MODEL = "mainarch-preview"


def load_perf():
    path = Path(os.environ.get("MAINARCH_DEMO_PERF_JSON", ROOT / "perf-sample.json"))
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {
            "headline": "Perf export unavailable",
            "win": f"Could not load perf data from {path}: {exc}",
            "measurements": [],
        }


def json_bytes(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def public_endpoint():
    return os.environ.get("MAINARCH_DEMO_PUBLIC_URL", "").strip()


def backend_ready(mode):
    if mode == "openai":
        return bool(os.environ.get("MAINARCH_DEMO_UPSTREAM"))
    if mode == "command":
        return bool(os.environ.get("MAINARCH_DEMO_COMMAND"))
    return True


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "mainarch-demo/0.2"

    def log_message(self, fmt, *args):
        if os.environ.get("MAINARCH_DEMO_QUIET") == "1":
            return
        super().log_message(fmt, *args)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            return self.send_status()
        if path == "/api/health":
            return self.send_health()
        if path == "/api/demo":
            return self.send_demo_manifest()
        if path == "/v1/models":
            return self.send_models()
        if path == "/":
            return self.send_file(PUBLIC / "index.html", "text/html; charset=utf-8")
        return self.send_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in {"/api/chat", "/v1/chat/completions"}:
            return self.send_error(HTTPStatus.NOT_FOUND)

        try:
            length = min(int(self.headers.get("content-length", "0")), 128 * 1024)
            body = self.rfile.read(length)
            request = json.loads(body.decode("utf-8"))
        except Exception:
            return self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")

        if path == "/v1/chat/completions":
            return self.send_chat_completions(request)

        return self.send_chat_events(request)

    def send_chat_events(self, request):
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True

        mode = os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower()
        self.sse("meta", {
            "mode": mode,
            "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        try:
            if mode == "openai":
                self.stream_openai(request)
            elif mode == "command":
                self.stream_command(request)
            else:
                self.stream_scripted(request)
            self.sse("done", {"ok": True})
        except Exception as exc:
            self.sse("error", {"error": str(exc)})

    def send_status(self):
        mode = os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower()
        model = os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL)
        payload = {
            "mode": mode,
            "mode_label": self.mode_label(mode),
            "model": model,
            "perf": load_perf(),
            "served_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "public_url": public_endpoint(),
            "backend_ready": backend_ready(mode),
            "upstream_configured": bool(os.environ.get("MAINARCH_DEMO_UPSTREAM")),
            "command_configured": bool(os.environ.get("MAINARCH_DEMO_COMMAND")),
        }
        data = json_bytes(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_health(self):
        mode = os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower()
        payload = {
            "ok": True,
            "service": "mainarch-demo-static",
            "mode": mode,
            "backend_ready": backend_ready(mode),
            "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
            "chat_endpoint": "/v1/chat/completions",
            "models_endpoint": "/v1/models",
            "served_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.send_json(payload)

    def send_demo_manifest(self):
        mode = os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower()
        payload = {
            "ok": True,
            "service": "mainarch-demo-static",
            "headline": "One-page sandbox over the same OpenAI-shaped route the release binary serves.",
            "mode": mode,
            "backend_ready": backend_ready(mode),
            "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
            "perf": load_perf(),
            "what_is_real": [
                "browser chat surface",
                "OpenAI-shaped /v1/models and /v1/chat/completions contract",
                "scripted, OpenAI-compatible, and local-command backend modes",
                "visitor-facing receipt that explains route, backend mode, elapsed time, and synthetic/upstream boundary",
                "visible perf proof rail loaded from the current benchmark export",
            ],
            "what_is_not_claimed": [
                "real Qwen or Kimi weights in scripted mode",
                "production auth, quotas, or abuse controls",
            ],
            "swap_points": [
                "point MAINARCH_DEMO_UPSTREAM at a real model server",
                "or run the canonical release binary with tools/demo_sandbox.sh serve",
            ],
            "endpoints": {
                "ui": "/",
                "health": "/api/health",
                "manifest": "/api/demo",
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
            },
        }
        self.send_json(payload)

    def send_models(self):
        model = os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL)
        payload = {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "mainarch",
                    "mainarch": {
                        "mode": os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower(),
                        "target": "Qwen/Kimi direct serving sandbox",
                        "chat_endpoint": "/v1/chat/completions",
                        "scripted_mode_is_synthetic": True,
                    },
                }
            ],
        }
        self.send_json(payload)

    def send_json(self, payload):
        data = json_bytes(payload)
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_static(self, request_path):
        rel = request_path.lstrip("/")
        if not rel:
            rel = "index.html"
        target = (PUBLIC / rel).resolve()
        try:
            target.relative_to(PUBLIC.resolve())
        except ValueError:
            return self.send_error(HTTPStatus.NOT_FOUND)
        if not target.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        content_type = "text/plain; charset=utf-8"
        if target.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif target.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif target.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        return self.send_file(target, content_type)

    def send_file(self, path, content_type):
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def sse(self, event, payload):
        packet = f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(packet)
        self.wfile.flush()

    def openai_sse(self, payload):
        packet = f"data: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(packet)
        self.wfile.flush()

    def openai_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    @staticmethod
    def mode_label(mode):
        if mode == "openai":
            return "real backend: OpenAI-compatible"
        if mode == "command":
            return "real backend: local command"
        return "scripted preview"

    def stream_scripted(self, request):
        text = self.scripted_response_text(request)
        for chunk in text.split(" "):
            self.sse("token", {"text": chunk + " "})
            time.sleep(float(os.environ.get("MAINARCH_DEMO_TOKEN_DELAY", "0.018")))

    def scripted_response_text(self, request):
        prompt = str(request.get("prompt", "")).strip()
        if not prompt:
            messages = request.get("messages") or []
            if messages:
                prompt = str(messages[-1].get("content", "")).strip()
        perf = load_perf()
        best = ""
        measurements = perf.get("measurements") or []
        if measurements:
            first = measurements[0]
            best = (
                f" The visible proof rail starts with {first.get('name', 'a bench')}: "
                f"{first.get('mainarch')} {first.get('unit', '')} on mainarch versus "
                f"{first.get('baseline')} {first.get('unit', '')} on the comparison path."
            )

        if "qwen" in prompt.lower() or "kimi" in prompt.lower():
            text = (
                "The page is already shaped like the Qwen/Kimi product surface. This sandbox is in "
                "scripted preview mode, so it is not claiming a live model is attached. When the real "
                "backend is ready, the browser keeps calling /v1/chat/completions and this server swaps to "
                "MAINARCH_DEMO_MODE=openai or MAINARCH_DEMO_MODE=command."
                f"{best}"
            )
        elif "faster" in prompt.lower() or "beat" in prompt.lower():
            text = (
                "The speed claim is the architecture, not a decorative UI number: less host "
                "orchestration, fewer framework layers, and GPU-resident handoffs on the hot path. "
                "The proof rail stays next to the chat so every demo has to carry a measured local "
                "comparison instead of a hand-wavy benchmark slide."
                f"{best}"
            )
        else:
            text = (
                "This is the mainarch sandbox layer: a one-page chat surface, a proof rail, and a "
                "small HTTP seam for the future serving kernel. It has three backend modes: scripted "
                "for safe previews, OpenAI-compatible for a real model server, and command mode for "
                "a local mainarch bridge."
                f"{best}"
            )
        return text

    def send_chat_completions(self, request):
        stream = bool(request.get("stream", True))
        mode = os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower()
        if not stream:
            text = self.complete_text(request, mode)
            payload = {
                "id": "chatcmpl-mainarch-demo",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
            return self.send_json(payload)

        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            if mode == "openai":
                self.proxy_openai_completion(request)
            elif mode == "command":
                self.stream_command_openai(request)
            else:
                self.stream_scripted_openai(request)
            self.openai_done()
        except Exception as exc:
            self.openai_sse({
                "id": "chatcmpl-mainarch-demo",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
                "error": {"message": str(exc)},
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            })
            self.openai_done()

    def complete_text(self, request, mode):
        if mode == "command":
            return self.run_command_to_text(request)
        if mode == "openai":
            raise RuntimeError("non-streaming OpenAI upstream mode is not implemented in the static sandbox")
        return self.scripted_response_text(request)

    def stream_scripted_openai(self, request):
        text = self.scripted_response_text(request)
        for chunk in text.split(" "):
            self.openai_token(chunk + " ")
            time.sleep(float(os.environ.get("MAINARCH_DEMO_TOKEN_DELAY", "0.018")))
        self.openai_finish()

    def stream_command_openai(self, request):
        for chunk in self.run_command_to_text(request).split(" "):
            if chunk:
                self.openai_token(chunk + " ")
                time.sleep(0.004)
        self.openai_finish()

    def openai_token(self, text):
        self.openai_sse({
            "id": "chatcmpl-mainarch-demo",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            "mainarch": {
                "demo_mode": os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower(),
                "proof": "synthetic-preview" if os.environ.get("MAINARCH_DEMO_MODE", "scripted").strip().lower() == "scripted" else "backend-attached",
            },
        })

    def openai_finish(self):
        self.openai_sse({
            "id": "chatcmpl-mainarch-demo",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        })

    def proxy_openai_completion(self, request):
        upstream = os.environ.get("MAINARCH_DEMO_UPSTREAM")
        if not upstream:
            raise RuntimeError("MAINARCH_DEMO_UPSTREAM is not set")

        api_key = os.environ.get("MAINARCH_DEMO_API_KEY", "")
        payload = dict(request)
        payload["model"] = os.environ.get("MAINARCH_DEMO_MODEL", payload.get("model", DEFAULT_MODEL))
        payload["stream"] = True
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(upstream, data=json_bytes(payload), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=float(os.environ.get("MAINARCH_DEMO_TIMEOUT", "90"))) as res:
            for raw in res:
                line = raw.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()

    def run_command_to_text(self, request):
        command = os.environ.get("MAINARCH_DEMO_COMMAND")
        if not command:
            raise RuntimeError("MAINARCH_DEMO_COMMAND is not set")

        timeout = float(os.environ.get("MAINARCH_DEMO_TIMEOUT", "90"))
        proc = subprocess.run(
            shlex.split(command),
            input=json.dumps({
                "prompt": request.get("prompt", ""),
                "messages": request.get("messages") or [],
            }) + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or f"command exited {proc.returncode}")[:500])
        return proc.stdout

    def stream_openai(self, request):
        upstream = os.environ.get("MAINARCH_DEMO_UPSTREAM")
        if not upstream:
            raise RuntimeError("MAINARCH_DEMO_UPSTREAM is not set")

        api_key = os.environ.get("MAINARCH_DEMO_API_KEY", "")
        model = os.environ.get("MAINARCH_DEMO_MODEL", DEFAULT_MODEL)
        messages = request.get("messages") or [{"role": "user", "content": request.get("prompt", "")}]
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": float(os.environ.get("MAINARCH_DEMO_TEMPERATURE", "0.2")),
            "max_tokens": int(os.environ.get("MAINARCH_DEMO_MAX_TOKENS", "512")),
        }
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(upstream, data=json_bytes(payload), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=float(os.environ.get("MAINARCH_DEMO_TIMEOUT", "90"))) as res:
                for raw in res:
                    line = raw.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    obj = json.loads(data.decode("utf-8"))
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content") or ""
                    if text:
                        self.sse("token", {"text": text})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"upstream HTTP {exc.code}: {detail[:400]}") from exc

    def stream_command(self, request):
        command = os.environ.get("MAINARCH_DEMO_COMMAND")
        if not command:
            raise RuntimeError("MAINARCH_DEMO_COMMAND is not set")

        timeout = float(os.environ.get("MAINARCH_DEMO_TIMEOUT", "90"))
        proc = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps({
            "prompt": request.get("prompt", ""),
            "messages": request.get("messages", []),
        }) + "\n")
        proc.stdin.close()

        start = time.monotonic()
        while True:
            if time.monotonic() - start > timeout:
                proc.kill()
                raise RuntimeError("command backend timed out")
            ready, _, _ = select.select([proc.stdout], [], [], 0.05)
            if ready:
                ch = proc.stdout.read(1)
                if ch:
                    self.sse("token", {"text": ch})
                    continue
            if proc.poll() is not None:
                rest = proc.stdout.read()
                if rest:
                    self.sse("token", {"text": rest})
                break

        if proc.returncode:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()[:400]
            raise RuntimeError(f"command backend exited {proc.returncode}: {stderr}")


def main():
    parser = argparse.ArgumentParser(description="Run the mainarch sandbox demo")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"mainarch sandbox listening on http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

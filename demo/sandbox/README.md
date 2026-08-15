# The demo sandbox

A dependency-free, GPU-free server and one-page browser demo for the mainarch
serving seam.

Full documentation: [`docs/demo-sandbox.md`](../../docs/demo-sandbox.md).

```bash
just demo    # from the repo root, picks the GPU lane if one is available
```

Or run this lane directly:

```bash
python3 server.py --host 0.0.0.0 --port 8080
```

Then open <http://127.0.0.1:8080>.

## What is here

```
server.py          stdlib-only HTTP server: /api/health, /api/demo,
                   /v1/models, /v1/chat/completions
public/index.html  the page
public/app.js      talks to /v1/chat/completions and renders the proof rail
public/styles.css
perf-sample.json   the perf cards; every entry names its baseline config
```

No framework, no build step, no package manager. `python3 server.py` is the
whole dependency list.

## The point

The browser only ever speaks `/v1/chat/completions`. Swapping the backend,
whether to scripted, an OpenAI-compatible upstream, a local command, or the
real `mainarch demo-serve` binary on a GPU, changes nothing about the page. That is
the seam this exists to hold stable.

Responses in the default mode are **synthetic**. The page, `/api/demo`, and the
per-run receipt all say so.

## Contract gate

```bash
python3 ../../tools/check_demo_sandbox_static.py
```

Starts this server, exercises the endpoints, and asserts the synthetic
boundary. It runs in CI and needs no GPU.

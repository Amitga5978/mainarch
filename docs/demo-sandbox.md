# The demo sandbox

`just demo` serves a one-page browser sandbox in front of an OpenAI-shaped HTTP
seam. The point of it is the *seam*, not the answers: the browser talks to
`/v1/chat/completions` exactly the way it would talk to a real serving backend,
so the backend can be replaced without touching the page.

**By default the answers are synthetic.** The page says so on its face,
`/api/demo` reports `"mode": "synthetic_decode_proof"`, and `/v1/models`
advertises the model id `mainarch-qwen3-235b-a22b-synthetic-proof`. In that mode
nothing loads real weights or runs a tokenizer.

**Point it at a checkpoint and the same endpoint serves real completions.**

```bash
mainarch demo-serve --bind 127.0.0.1:8080 \
  --olmo-config <dir>/config.json \
  --olmo-index <dir>/model.safetensors.index.json \
  --olmo-tokenizer <dir>/tokenizer.json
```

`/v1/models` then advertises `olmo-2-0425-1b` and both it and `/api/health`
report `synthetic: false`, so a client can tell the two apart without reading the
source. See the OLMo 2 section of the README for what that path does.

## Two lanes

`just demo` picks one automatically.

### CPU lane (no GPU)

A dependency-free Python server, `demo/sandbox/server.py`. Serves the page and
the same public endpoints, with scripted responses. This is what runs on a
laptop, in CI, and in a sandbox deploy that must not touch the GPU.

```bash
python3 demo/sandbox/server.py --host 0.0.0.0 --port 8080
```

### GPU lane

The release binary itself, `mainarch demo-serve`. It serves an embedded page
from the same process that owns the GPU work, so every request drives a real
decode step on a real MI355X node: admission, a KV block-table reservation, a
scheduler tick, a live kernel dispatch, and KV release accounting, all visible
in the page's proof trail and in `/metrics`.

```bash
cargo build --release
./target/release/mainarch demo-serve --bind 0.0.0.0:8080 --node <gpu-node>
```

`--node` is a KFD topology node id; `mainarch probe` lists them. `just demo`
picks the first node with SIMDs and a real gfx target version. Force the CPU
lane on a GPU host with `MAINARCH_DEMO_STATIC=1 just demo`.

## Public endpoints

Both lanes serve:

```text
/                       one-page browser demo
/api/health             readiness
/api/demo               public sandbox manifest
/v1/models              OpenAI-shaped model discovery
/v1/chat/completions    OpenAI-shaped chat, streaming or not
```

The GPU lane adds `/api/evidence`, `/api/compare`, and `/metrics`
(Prometheus-style counters).

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mainarch-qwen3-235b-a22b-synthetic-proof","stream":true,
       "messages":[{"role":"user","content":"why is this fast?"}]}'
```

## Backend modes (CPU lane)

Default is `scripted`. To point the page at something real:

```bash
# any OpenAI-compatible server
MAINARCH_DEMO_MODE=openai \
MAINARCH_DEMO_UPSTREAM=http://127.0.0.1:8000/v1/chat/completions \
MAINARCH_DEMO_MODEL=my-model \
python3 demo/sandbox/server.py --host 0.0.0.0 --port 8080

# a local command; it gets one JSON line on stdin and streams stdout back
MAINARCH_DEMO_MODE=command \
MAINARCH_DEMO_COMMAND="/path/to/your/backend" \
python3 demo/sandbox/server.py --host 0.0.0.0 --port 8080
```

The receipt the page prints after every run states which route served it and
whether the text was synthetic or upstream-backed, so swapping the backend is
visible to the viewer rather than hidden.

## Perf cards

The page loads `demo/sandbox/perf-sample.json`. Every entry names the exact
baseline configuration in its `source` field. That is not decoration, it is the
difference between a measurement and a marketing number. Point it at your own
export with `MAINARCH_DEMO_PERF_JSON=/path/to/perf.json`:

```json
{
  "headline": "...",
  "win": "One plain sentence: what got better, and for whom.",
  "measurements": [
    { "name": "...", "mainarch": 21.7, "baseline": 29.2, "unit": "us",
      "source": "name the baseline's exact configuration here" }
  ]
}
```

## Container

The image defaults to the CPU lane so it can be deployed without device access:

```bash
just build-demo-image
just run-demo-container
just smoke-demo-container
just stop-demo-container
```

Set `MAINARCH_DEMO_LIVE=1` only when you deliberately want the container to run
`mainarch demo-serve` with GPU devices mounted.

## Safety guards

`tools/demo_sandbox.sh serve` refuses to start when another `mainarch
demo-serve` is already running, or when `/sys/class/kfd/kfd/proc` is non-empty,
i.e. when some other process already holds the GPU. Two processes fighting over
the same KFD node can wedge the device and force a reset. Set
`MAINARCH_DEMO_FORCE=1` only when you deliberately want to share the node.

```bash
MAINARCH_DEMO_URL=http://127.0.0.1:8080 bash tools/demo_sandbox.sh smoke
python3 tools/check_demo_sandbox_static.py    # CPU-only contract gate, in CI
```

## What the demo does and does not claim

This depends on which lane is serving, so it is worth splitting.

Real in every mode:

- the OpenAI-shaped route, streaming, and the receipt surface
- on the GPU lane, a real kernel dispatch per request through raw KFD/AQL, a
  persistent worker owning the device, admission-time KV block reservations with
  capacity and release accounting, and scheduler-visible KV ownership checks

Real only with `--olmo-config`:

- real weights, a real tokenizer, and text the model actually generated
- `synthetic: false` on `/v1/models` and `/api/health`

Not claimed in any mode:

- chat quality. `OLMo-2-0425-1B` is a base model, so it completes rather than
  converses, and no instruction tuning is involved here
- concurrency. One request is served at a time because the KV cache holds one
  sequence, so a second caller waits
- multi-user scheduling, auth, quota, or abuse controls
- any throughput number derived from this page

Keep auth, quota, prompt policy, and abuse controls in front of this process,
not inside it. It is a demonstration surface, not a public serving endpoint.

Everything here binds to `127.0.0.1` by default, deliberately. It serves a
language model with no authentication, no quota and no rate limit, so exposing
it should be something you choose rather than something you inherit. Set
`MAINARCH_DEMO_BIND=0.0.0.0:8080` when you mean it, and put a reverse proxy in
front. The container images bind `0.0.0.0` because that is correct inside a
container, where the decision is which ports you publish.

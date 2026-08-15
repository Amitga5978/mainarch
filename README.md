# mainarch

![A rack, a processor, the AQL queue drawn as a ring, the KV cache as a tape of
blocks, and a single token coming out the far end, in isometric line
work](docs/art/hero.png)

[![CI](https://github.com/MaincodeHQ/mainarch/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MaincodeHQ/mainarch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Running a large language model on an AMD GPU normally means going through ROCm,
and ROCm is a lot of software. There's the HIP runtime, the HSA runtime beneath
it, the thunk library beneath that, and only then the kernel driver that actually
talks to the card. Every one of those layers was written by people solving real
problems, and almost all of it was designed before anyone knew what inference
serving would look like.

mainarch skips them. It opens `/dev/kfd`, builds the dispatch packet by hand,
rings the doorbell, and waits on the signal the command processor decrements.
When this code runs a kernel on an MI355X, the only thing standing between it
and the silicon is the `amdkfd` kernel driver.

From there it goes up. GPU virtual memory, AQL queues, a code-object loader,
collectives over XGMI, attention, a decoder layer, and a token loop, all in one
typed Rust workspace with nothing borrowed from a vendor runtime.

At the top of it, a real model runs. Point this at
[OLMo 2](https://allenai.org/olmo) and ask it something.

```
The first person to walk on the moon was Neil Armstrong. He was an American
astronaut. He was born in 1930.
```

That completion came out of `/v1/chat/completions` on an MI355X, from weights
AI2 published alongside their training data and their training code, through a
stack where every layer between the token and the silicon is in this repository.

It's a reference architecture and a working prototype, published so you can read
how a serving stack looks when it's built from the ABI upward. It isn't a
production replacement for ROCm, vLLM, or SGLang, and the
[Scope and non-claims](#scope-and-non-claims) section says exactly where the
edges are. That section is short, and it's the honest heart of this README.

---

## Run it

Open the repo in the devcontainer, then run one command.

```bash
just demo
```

That's the whole thing. It builds the workspace, works out what your machine can
actually do, proves it in front of you, and serves a playable OpenAI-shaped demo
page at <http://127.0.0.1:8080>.

On any Linux box, with no GPU anywhere in sight, you get the CPU lane. The
embedded gfx950 GPU binary gets parsed by this repo's own ELF and code-object
reader, printing all 187 kernels with their descriptors, kernarg sizes, LDS
requirements and wavefront sizes. The collectives harness runs its size sweep
against the CPU simulator backend with the correctness oracle attached, and it
says plainly in its own output that a simulator isn't a transport. Then the demo
page comes up, serving `/v1/chat/completions` with scripted responses.

On a host with an AMD GPU on `/dev/kfd`, reopened using the `.devcontainer/gpu`
configuration, the same command does more. It opens the device, reads the
`amdkfd` version, and walks the topology and the XGMI links. It builds an
`hsa_kernel_dispatch_packet` by hand, rings the doorbell, and waits on an
`amd_signal_t` the command processor decrements, which is about as direct a proof
of live kernel execution without a vendor runtime as you can ask for. Then it
serves the demo from that same binary, with the decode lane running on a real
GPU.

### What you need

The CPU lane wants a Linux x86-64 host with Docker or Podman and about 6 GB of
disk for the build. No GPU, no ROCm, nothing else.

The GPU lane wants an AMD GPU reachable at `/dev/kfd` and `/dev/dri`, and your
user in the `render` group. The kernels are compiled for **gfx950**, which is
MI355X on CDNA4. Other architectures will load the wrong ISA, and
`mainarch probe` will tell you what you actually have.

Running OLMo 2 on top of that wants roughly 6 GB of disk for the checkpoint and
about 3 GB of VRAM once it is resident, which is nothing on a 288 GB card.

### Running the real model

`just demo` doesn't download six gigabytes behind your back, so the model is a
second and deliberate step.

```bash
just olmo-fetch                          # ~6 GB from Hugging Face
just olmo-preflight                      # CPU-only, no GPU needed
just olmo "The capital of France is"     # needs an AMD GPU
just olmo-serve                          # /v1/chat/completions, for real
```

Run the preflight first even without a GPU. It tells you whether a checkpoint is
something this runtime can actually execute, and says why when it isn't, which
is cheaper than finding out after six gigabytes and a load.

If you'd rather read than click, there's a slower narrated walk through the
pieces.

```bash
just tour        # CPU-only, no GPU required
just --list      # every task
```

<details>
<summary>Without the devcontainer</summary>

Nothing here actually needs the container. You'll want a stable Rust toolchain,
a C compiler because the `tokenizers` crate builds Oniguruma from source, and
Python 3.

```bash
sudo apt-get install build-essential python3    # or your distro's equivalent
cargo install just
just demo
```

On an AMD GPU host you also need read and write access to `/dev/kfd` and
`/dev/dri/renderD*`, which in practice means membership in the `render` group.

</details>

---

## Why build this

ROCm works, and that's worth saying out loud before criticising it. It's also a
large C and C++ stack layered over a decade and a half of accumulated
assumptions, and it wasn't shaped around what inference serving turned out to
need. mainarch is a bet that a clean-sheet design in Rust, speaking straight to
the `amdkfd` and `amdgpu` kernel ABI, makes the lower serving layers more
modular, more observable, and far easier to change, while still borrowing the
ideas that already work from SGLang, vLLM, RCCL, NCCL, and MSCCL++.

Four principles hold the thing together.

**Direct to the ABI.** The bottom crate issues the same ioctls that ROCm's thunk
library wraps, and nothing sits in between.

**Modular layers.** Every layer is a crate with a narrow, testable seam, so any
one of them can be replaced without disturbing the rest.

**Prove it at the bottom first.** Correctness and bandwidth get established with
low-level tests before anything taller is built on top of them.

**Every kernel checks against a reference.** Numerics are validated against an
`f64` host implementation rather than against whether the output looked about
right.

The only AMD-specific surface any of this depends on is the kernel UAPI, meaning
`/dev/kfd` ioctls and the `amdgpu` DRM interface, plus an LLVM with the `amdgpu`
backend used strictly as a build-time cross-compiler. That last part deserves
precision. The code object committed here was emitted by AMD's ROCm LLVM 7.2.4
distribution, and `kernels/build.sh` reaches for it by default because it's the
easiest way to get a working `amdgcn` target. Any LLVM built with the AMDGPU
backend does the job just as well, and the part that matters is that nothing from
ROCm is loaded, linked, or present at run time. The devcontainer image contains
none of it.

---

## The stack, bottom to top

| Layer | Where | What it does |
|---|---|---|
| Kernel ABI | `crates/mainarch-sys/src/lib.rs` | Hand-encoded `amdkfd` ioctls, no bindgen and no C shim |
| Device & memory | `crates/mainarch-core/src/gpu.rs` | `ACQUIRE_VM`, VRAM and GTT allocation, peer mapping, AQL queues, doorbells, completion signals, SDMA |
| Code objects | `crates/mainarch-core/src/codeobject.rs` | An ELF and HSA code-object loader written from scratch, covering kernel descriptors, kernarg layout, and relocation |
| Collectives | `crates/mainarch-collectives/src/lib.rs`, `crates/mainarch-core/src/multigpu.rs` | All-reduce over XGMI with size-selected algorithms, an rccl-tests-shaped harness, and a CPU correctness oracle |
| GEMM / GEMV | `crates/mainarch-core/src/gemm.rs` | FP16 matrix-core tiles, plus memory-bound GEMV for single-token projections including FP8 weights |
| Attention | `crates/mainarch-core/src/attn.rs` | FlashDecoding with split-KV online softmax, tree combine, paged KV, GQA, FP8 and FP4 KV, and the MLA latent-cache path with split-K |
| Decoder layer | `crates/mainarch-core/src/layer.rs` | RMSNorm, RoPE, SwiGLU, QK-norm, MoE router and FFN |
| Model loop | `crates/mainarch-core/src/model.rs` | Embedding, N layers with a growing KV cache, final norm, LM head, argmax, next token |
| Weights | `crates/mainarch-core/src/weights.rs` | SafeTensors parsing, Qwen checkpoint preflight, tensor-parallel rank sharding, SDMA upload to VRAM |
| OLMo 2 | `crates/mainarch-core/src/olmo2.rs` | Checkpoint preflight, fp32 to f16 load into VRAM, post-norm layer forward, and the token loop |
| Model API | `crates/mainarch-core/src/model_api.rs` | The typed primitive-graph authoring surface, CPU-only by design |
| Kernels | `kernels/mainarch_kernels.cl` | Every GPU kernel, in one file, compiled to a gfx950 HSA code object |

The compiled code object lives at
`crates/mainarch-core/artifacts/mainarch_kernels.gfx950.co` and gets embedded
straight into the binary, so building this repo never needs a GPU compiler.
`kernels/build.sh` rebuilds it if you change the kernel source.

---

## What's proven, and what it measured

Everything below was measured on one host, 8 MI355X cards on gfx950, with no
ROCm loaded at runtime. These are the numbers this repo's own commands print on
that machine, and your hardware will differ.

### Live execution on the raw ABI

Open `/dev/kfd`, acquire the VM, create an AQL compute queue, then dispatch a
kernel through a hand-built `hsa_kernel_dispatch_packet` and a doorbell write,
detecting completion through an `amd_signal_t` the command processor decrements.

```bash
mainarch probe          # driver version, topology, XGMI links
mainarch gpu-selftest   # dispatch a kernel, verify the GPU stamped memory
```

### Small-message all-reduce

Tensor-parallel decode hits a collective on every single token, and the messages
are small, usually somewhere between 8 and 64 KiB. At that size the cost isn't
bandwidth, it's overhead, so the algorithm changes with the message.

Below roughly 8 MiB, a fused reduce-scatter and all-gather runs across the fully
connected fabric. Every GPU owns one chunk and drives its own links, using remote
writes rather than reads because writes stream far better on Infinity Fabric,
with the reduce fused into the all-gather so no global barrier sits there idling
the links.

At 16 MiB and above, a GPU-driven one-shot kernel takes over. A single persistent
kernel per GPU runs the entire operation, scatter then reduce then all-gather,
synchronising on the device itself through a cross-GPU barrier built on
system-scope atomics over peer-mapped flags plus an intra-GPU grid barrier. One
launch streams all the XGMI traffic continuously, an idea borrowed from MSCCL++.

Data sits in device-local HBM the whole time and peer access is real GPU-to-GPU
XGMI. It's bit-exact against a CPU oracle from n=8 all the way through 64 MiB per
rank, odd sizes included.

Here's the comparison against RCCL `all_reduce_perf` on the same 8 GPUs, run at
RCCL's fastest configuration of `-t 8 -g 1`, meaning eight threads with one GPU
each. That choice matters. Single-threaded `-g 8` serialises RCCL's launch path
and would flatter us considerably. Latency in µs, lower is better.

| message | mainarch (1 thread) | RCCL `-t8 -g1` | |
|--------:|--------------------:|---------------:|---|
| 2 KiB | **~20** | ~26 | mainarch ~1.3× |
| 64 KiB | **21.7** | 29.2 | mainarch ~1.35× |
| 256 KiB – 2 MiB | ~31–38 | ~28–34 | about even |

At the sizes decode actually hits, mainarch sits around 20 µs from a single host
thread, where RCCL needs eight threads to reach 26. That 20 µs is the
host-synchronised latency floor we measured for this operation on this fabric,
one XGMI sync round plus one host round trip, and there's no clever trick
underneath it. It's just the absence of overhead.

RCCL wins above about 8 MiB, by roughly 1.3×. The GPU-driven one-shot reaches
around 277 GB/s busbw at 64 MiB and 307 GB/s at 256 MiB, and closing the
remaining gap needs device-side SDMA triggering on the same substrate, which
isn't built yet.

```bash
mainarch gpu-multi-check       # 8-GPU all-reduce, bit-exact vs CPU oracle
mainarch gpu-allreduce-bench   # busbw sweep in the rccl-tests table shape
bench/compare-allreduce.sh     # paired mainarch-vs-RCCL artifact
```

You can override the algorithm choice for experiments with
`MAINARCH_ALLREDUCE_ALGO=direct|rsag-read|rsag-write|oneshot`.

### Decode attention, the long-context core

Decode attention is one query attending over a long KV cache, and at long context
it's the thing that sets your tokens per second. It's memory-bandwidth bound,
which means the entire game is how fast you can pull the cache through and how
little you waste doing it. Built here from the ground up as FlashDecoding, using
split-KV online softmax plus a tree combine, on the raw KFD and AQL path at head
dimension 128.

A pure VRAM streaming read sustains about 5800 GB/s on this part, roughly 72% of
HBM peak, and that's the ceiling every kernel below gets measured against. All of
them are bit-close to an `f64` reference.

Single query, by KV length.

| KV len | FP16 | FP8 | FP4 |
|-------:|-----:|----:|----:|
| 64K | 39 µs (858 GB/s) | 1.01× | 1.03× |
| 256K | 56 µs (2379 GB/s) | 1.11× | 1.19× |
| 1M | 146 µs (3671 GB/s, 46% HBM) | **1.55×** | **1.92×** |

Grouped-query attention lets 8 query heads share one KV head, so the cache gets
read and dequantized once per head group instead of eight times over, and it
composes with quantized KV into the configurations people actually serve.

| config | 1M µs/head | vs FP16 single head | KV memory |
|---|---:|---:|---:|
| GQA-FP8 | 33.6 | 2.9× | ½ |
| GQA-FP4 | 31.9 | 4.5× | ¼ |
| **paged + GQA + FP4** | **33.3** | **4.5×** | **¼**, paged |

That last row is the real capstone. It's paged KV in the vLLM and SGLang sense,
where a block table maps logical blocks to physical ones, validated on the raw
ABI with a deliberately shuffled layout so nothing accidentally depends on
contiguity. It's 8 grouped queries. It's 4-bit E2M1 KV with E8M0 block scales
dequantized in the VALU through `cvt_scalef32_pk_f32_fp4`. All of it in one
kernel, with split and combine chained so the host waits exactly once.

Four things gate a memory-bound kernel here, and each was verified against the
streaming ceiling rather than assumed. Load width matters, so 128-bit `half8` and
packed FP8 and FP4. Memory-level parallelism matters, which means prefetching
many tokens before any cross-lane reduce. Every buffer has to live in VRAM, and
we learned that one the hard way when a single host-allocated scratch buffer
pinned the entire pipeline at 230 GB/s until somebody found it. And reductions
have to be parallel, a √N-way tree combine rather than one workgroup grinding
through it. FP8 here is OCP e4m3fn and FP4 is E2M1, both hardware-probed.

On realistic data, Gaussian with 2% outliers, relative L2 error comes out around
2% for FP8 and around 15% for raw FP4-E2M1. The compression and the speedups are
measured on the primitive, but what that costs a real model is workload-specific
and isn't claimed here. NVFP4 and Q-K smoothing are the open work.

```bash
mainarch attn-decode --node 2    # correctness plus FP16/FP8/FP4/GQA/paged sweep
mainarch decode-layer --node 2   # RMSNorm, RoPE, SwiGLU, KV-quant accuracy
```

### MLA, the other attention shape

GQA isn't the only long-context attention in use. Multi-head latent attention,
the shape DeepSeek-V3 and Kimi K2 use, compresses the KV cache into a single
low-rank latent per token instead of storing K and V per head. That changes the
kernel completely. The cache becomes a 512-dimensional compressed latent plus a
64-dimensional positional part, and the per-head projections happen after the
cache read rather than before it.

That path is built on the same raw ABI. There's a paged latent cache holding
FP8 E4M3 `ckv` alongside BF16 `kpe`, per-head dot scores, local softmax with
exported LSE, and softmax-weighted output tiles, then split-K across the sequence
with a stage-2 merge of the partial records. Correctness is gated across all 64
eight-wide windows of the 512-dimensional latent output and at a ragged page and
split boundary, not just at the aligned sizes where everything is easy.

Measured on 8 MI355X cards, 51,200 tokens across 800 pages, 8 heads, split-K
of 64.

| | µs/token |
|---|---:|
| MLA hot decode path | ~143–144 |
| the same path at split-K = 8 | ~444 |
| plus TP8 all-reduce, 256 KiB payload at ~10.4 GB/s busbw | ~188 combined |

That combined number is the honest shape of a tensor-parallel decode token,
the attention kernel plus the collective it has to wait on. It works out to
roughly 5,300 tokens per second per sequence, with the collective eating about a
quarter of the step.

```bash
mainarch gpu-paged-mla-fp8-splitk-e2e-selftest        # correctness, verified output
mainarch gpu-paged-mla-fp8-splitk-full-latent-sweep-gate
mainarch gpu-paged-mla-fp8-kimi-hot-decode-gate       # the timing gate above
```

Worth knowing that the hot-decode gate is a timing gate and reports
`output_verified false`. The correctness gates above it are where the numerics
get checked. That split is deliberate, and the output tells you which one you're
looking at.

### A full decoder layer, and a token loop

The rest of the decode step is built and validated on the same path. RMSNorm,
RoPE and SwiGLU are there, along with a memory-bound GEMV for the projections,
because single-token QKV, O and MLP are matrix-vector rather than matrix-matrix.
FP16 GEMV runs at 1.6 to 3.2 TB/s, and there's an FP8-weight variant that halves
weight traffic. The MoE FFN is there too, a router doing gate GEMV into a top-8
softmax, then a fused gate/up/SwiGLU and a weighted down-projection with the
expert index resolved on-device, landing at rel-L2 5.8e-4 on the real per-expert
dimensions of H=4096 and I=1536 while sustaining 872 GB/s.

So quantization is in place on both sides now, KV cache in FP8 and FP4, weights
in FP8. One complete Qwen3-235B-A22B-shaped decoder layer assembles and validates
end to end, running RMSNorm, QKV, QK-norm, RoPE at θ=5e6, GQA attention, O-proj,
residual, RMSNorm, MoE, residual, and coming out bit-close to an `f64` reference
at rel-L2 3.28e-5.

Those layers stack into a model that generates.

```bash
mainarch model-decode --node 2
```

That runs the full autoregressive loop on the raw ABI, embedding through N
decoder layers each carrying a growing KV cache, then final RMSNorm, LM head,
greedy argmax, next token, validating every decode step against an `f64`
reference with a maximum per-step logit rel-L2 of 9.5e-4. It runs at reduced
scale, and that qualifier is load-bearing. The head dimensions and the per-op
math are the real model's, but depth, vocabulary and expert count are cut down to
keep host memory sane. It's the leap from validated primitives to a transformer
that actually generates, which is not the same thing as a served model.

### OLMo 2, and why that model

The model here is [OLMo 2](https://allenai.org/olmo) from the Allen Institute
for AI, and the choice is deliberate. Most models you can download are open
*weights*, which means a student reading this stack hits a wall the moment they
reach the checkpoint. OLMo 2 is open *source*. AI2 publishes the weights, the
Dolma training corpus, the training code, and every intermediate checkpoint, so
there is no black box anywhere between the training data and a token coming out
of a hand-built AQL packet.

It also happens to fit. Every decode attention kernel here is built for head
dimension 128, and that is not a tunable, because the kernels address the KV
cache with 128-bit loads over a 128-wide head and map lanes accordingly. Most
small open models use 64 and would need a new kernel family. OLMo 2 uses 128.

What it did need was three changes, and they are worth reading because they are
the shape of what adding any architecture costs.

**It's multi-head, not grouped-query.** OLMo 2 gives every query head its own KV
head, which in grouped-query terms is a group of one. The decode kernel reads
the KV cache once per group, so a group of one removes the sharing the GQA
kernel exists to exploit. Group width sizes LDS and fixes the per-lane mapping,
so it has to be known at compile time, which makes this a sibling kernel rather
than a parameter.

**QK-norm spans the whole projection.** Qwen3 normalises each attention head
independently. OLMo 2 normalises across all of them, so `q_norm` is
`num_heads * head_dim` wide and each head reads its own slice. Getting this
wrong produces plausible numbers rather than an error, which is why the gate for
it perturbs one head's input and requires a different head's output to move.
Only a shared RMS can do that.

**It's post-norm.** Qwen3 normalises on the way into attention and the MLP.
OLMo 2 runs the sub-layer first and normalises its output on the residual
branch:

```text
  pre-norm:   h = rmsnorm(h + sublayer(h))
  post-norm:  h = h + rmsnorm(sublayer(h))
```

You can see which one a checkpoint is without reading any modelling code. A
pre-norm model ships `input_layernorm`. OLMo 2 ships `post_attention_layernorm`
and `post_feedforward_layernorm` and no `input_layernorm` at all, and preflight
refuses a checkpoint carrying one.

The MLP needed nothing new. It's dense SwiGLU, built from `gemv`, a cast and
`swiglu` so a reader sees gate, up, SwiGLU and down as four separate steps.

**Prefill is the decode loop.** There's no prefill GEMM kernel in this
repository and the roadmap still lists one as open, but none is needed to serve
this. The decode path already grows the KV cache one token at a time, so the
prompt is consumed by running it once per token. That's slow in the honest way
and it reuses every kernel that is already gated.

It runs at about 125 ms/token for 16 layers with FP4 KV, unfused and untuned.
Correctness came first and nothing here has been optimised.

One thing worth knowing before you type a question at it: `OLMo-2-0425-1B` is a
**base** model, not an instruction-tuned one, so it completes text rather than
holding a conversation. Give it `The capital of France is` and it does the right
thing. Give it a chat-style instruction and it will drift into whatever corpus
pattern looks likeliest. The Instruct variant is the same architecture and loads
without any change here if you want the other behaviour.

The model is © the Allen Institute for AI and released under Apache 2.0, the
same licence as this repository. `just olmo-fetch` downloads it from Hugging
Face and nothing about it is redistributed here. See
[allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) for the
model card, and [the OLMo 2 paper](https://arxiv.org/abs/2501.00656) for how it
was trained.

```bash
just olmo-gates            # the four gates, on hardware
mainarch olmo2-preflight-selftest                  # CPU-only
mainarch gpu-mha-attention-equivalence-selftest    # MHA vs the validated GQA kernel
mainarch gpu-olmo2-qk-rope-selftest                # whole-projection QK-norm + RoPE
mainarch gpu-olmo2-post-norm-selftest              # post-norm, and not pre-norm
```

### Checkpoints

SafeTensors parsing, a Qwen contract preflight over `config.json` and
`model.safetensors.index.json`, tensor-parallel rank-shard conversion, and raw
SDMA upload to VRAM with source-byte readback. The CPU-only halves run anywhere.

```bash
mainarch weights-qwen-preflight-selftest      # synthetic fixture, CPU-only
mainarch weights-qwen-rank-shard-selftest     # synthetic fixture, CPU-only
```

You can point the same code at a real Hugging Face Qwen3 directory. The preflight
and the shard conversion need no GPU, and only the VRAM load does.

```bash
# 1. does this checkpoint satisfy the contract, for a TP=8 split?
mainarch weights-qwen-preflight \
  --config  <qwen-dir>/config.json \
  --index   <qwen-dir>/model.safetensors.index.json \
  --tp-world 8

# 2. materialize the tensors rank 0 actually needs, into its own shard
mainarch weights-qwen-rank-shard \
  --config  <qwen-dir>/config.json \
  --index   <qwen-dir>/model.safetensors.index.json \
  --output  rank0.safetensors --tp-rank 0 --tp-world 8

# 3. upload that shard to VRAM over raw SDMA, with source-byte readback
mainarch weights-load-shard --source rank0.safetensors --node <n> --max-mb 512
```

A single-file checkpoint with no shard index uses the `-file-` variants,
`weights-qwen-file-preflight` and `weights-qwen-file-rank-shard`. What none of
this does is serve the model, which brings us to the part you should read
carefully.

---

## The model API

`docs/model-api.md` documents the model-facing layer, which is a typed primitive
graph, a Qwen-style reference MoE decoder written against it, an external-style
custom `ModelDefinition`, and a deterministic receipt system that fingerprints
what a model definition needs and reports precisely what isn't yet resolvable.

This layer is CPU-only on purpose. It compiles and validates model definitions,
plans checkpoint staging and dispatch, and emits fingerprinted non-execution
receipts. It doesn't lower the graph into executable AQL, allocate device
buffers, submit queues, run serving, or carry any performance claim. Its own
receipts say so, and `launch_executable: ready=false` is an asserted part of the
public contract rather than an oversight.

External packages import the authoring surface with one line.

```rust
use mainarch_core::model_api::prelude::*;
```

There's a complete standalone example including its contract test in
`examples/model-api-plugin/`, and one command checks the whole public surface.

```bash
just check-model-api
```

---

## Scope and non-claims

The numbers above are easy to over-read, so here's where the edges actually are.

**This isn't a production serving stack.** It does serve real weights over an
OpenAI-shaped endpoint, which is the whole point of the OLMo work, but it serves
exactly one request at a time. There's no batching, no continuous batching, no
scheduling across sequences, no multi-tenancy, no auth and no quota. The KV cache
holds one sequence, so a second concurrent request waits. That's honest for what
this is and it is nowhere near what a production server does.

**`just demo` on its own is still synthetic.** It serves a real HTTP seam, and on
hardware there's real GPU work behind it, but the text coming back is scripted.
Point the server at an OLMo checkpoint with `just olmo-serve` and the same
endpoint serves real completions instead. `/v1/models` and `/api/health` both
report `synthetic: false` when that's true, so a client can tell the two apart
without reading the source.

**Two model paths, and only one of them is real.** OLMo 2 runs at full depth and
full vocabulary from a real checkpoint. The Qwen3-235B-A22B path
(`mainarch model-decode`) is a synthetic-weight proof at reduced depth,
vocabulary and expert count, keeping the real head geometry and per-op math. It
demonstrates that the primitives compose into a working token loop. It is not a
94-layer real-checkpoint Qwen.

**The model API doesn't execute.** `mainarch-core::model_api` compiles and
validates model definitions and emits fingerprinted receipts. It does not lower
a graph into AQL, allocate buffers or submit queues, and its own receipts assert
`launch_executable: ready=false`. The OLMo path does not go through it.

**The numbers come from a single host.** One machine with 8 MI355X cards, one
software version. They're reproducible with the commands given, on that hardware.

**RCCL is faster above about 8 MiB.** Repeated here so it doesn't only live in a
table.

**FP4 accuracy is an open question.** Around 15% rel-L2 raw on the primitive, and
what that costs a real model isn't measured here. OLMo runs on FP4 KV today and
its output is coherent, but coherent is not the same as measured, and no
benchmark has been run against an FP16 baseline.

**Performance is unoptimised.** About 125 ms/token for a 1B model. Nothing on
the OLMo path is fused, batched or tuned, there's a host round trip per layer,
and prefill costs one decode step per prompt token. The roadmap lists what would
change that.

**MI355X and gfx950 only.** See `docs/hardware-support.md`.

---

## Hardware

[What you need](#what-you-need) covers the short version. Two things it leaves
out.

The multi-GPU paths, meaning the collectives and anything measured against RCCL,
assume eight XGMI-connected devices. Everything else runs on one.

No ROCm is needed anywhere, at build time or at run time, and the devcontainer
image contains none of it. The one place ROCm appears at all is
`kernels/build.sh`, which borrows an amdgpu-capable clang to recompile the kernel
source, and you only need that if you change a kernel. The compiled object is
committed.

`docs/hardware-support.md` has the validation tiers and what each one actually
proves.

---

## When it doesn't work

**`just probe` says the device is unavailable.** Your user needs to be in the
`render` group *before* the container starts, because `keep-groups` passes
through the groups you already had. Add yourself, then fully reconnect rather
than reopening a shell, since the group membership is captured at session start.

**`just demo` takes the CPU lane on a machine that has a GPU.** It looks for a
KFD topology node with SIMDs and a real gfx target version, so if `/dev/kfd`
isn't passed through it will quietly and correctly fall back. Reopen using the
`.devcontainer/gpu` configuration, and check `ls -l /dev/kfd /dev/dri/renderD*`
inside the container.

**`--node 2` doesn't exist on your machine.** That argument is a KFD topology
node id, not an ordinal GPU index, and the numbering depends on your host.
`mainarch probe` lists the real ones. `just demo` and `just olmo` work it out
for you.

**The port is already taken.** `MAINARCH_DEMO_BIND=127.0.0.1:9090 just demo`.

**`olmo2-preflight` refuses your checkpoint.** That is the preflight working. It
names the reason, and the three it will usually give are a `model_type` that
isn't `olmo2`, a `head_dim` that isn't 128, or an `input_layernorm` tensor, which
means the checkpoint is pre-norm and this path implements post-norm. None of
those are things a flag can force, because forcing them would produce wrong
numbers rather than an error.

**A kernel dispatch fails with a VA guard error.** That is also working. The
guard checks every buffer span against what the kernel will actually address,
and it fires before the GPU sees a bad pointer. The message names the kernel, the
argument, and the span it expected.

**Generation is slow.** It is. About 125 ms/token, unfused, untuned, with FP4 KV
and a host round trip per layer. Nothing in this path has been optimised, and the
roadmap is honest about what would change that.

## Reading it

If you want to understand how this works rather than just run it, here's the
order that makes each piece land before the next one needs it.

1. `crates/mainarch-sys/src/lib.rs`, the ioctl structs. Start at the bottom.
2. `crates/mainarch-core/src/gpu.rs`, where you find out how a queue, a doorbell
   and a completion signal actually work when nobody hands them to you.
3. `crates/mainarch-core/src/codeobject.rs`, reading an HSA code object without a
   loader library.
4. `crates/mainarch-core/src/attn.rs`, the memory-bound kernel playbook, with the
   reference checks sitting right next to the fast paths.
5. `crates/mainarch-core/src/multigpu.rs`, device-side cross-GPU synchronisation.
6. `crates/mainarch-core/src/olmo2.rs`, the whole of what adding a model
   architecture costs, in one file with the reasoning next to each check.
7. `kernels/mainarch_kernels.cl`, all of it, in one file.
8. `docs/ROADMAP.md`, where the layers are meant to go next.

---

### Two things you'll notice

`crates/mainarch-cli/src/main.rs` is about 136,000 lines, and
`model_api.rs` is about 68,000. That is not a good shape and nobody is pretending
otherwise. Both grew as harnesses, one holding 84 CLI subcommands that are mostly
hardware gates, the other holding a declarative metadata surface plus the tables
that pin it. They are wide rather than deep, so they read as a catalogue rather
than a call graph, but the honest summary is that they want splitting and have
not been split.

The library code is where the interesting work is, and it is normal sized.
`olmo2.rs` is about 1,600 lines, `attn.rs` about 3,500, `model.rs` about 2,100,
`gpu.rs` about 27,000 because it is the whole device layer. Start there. The
reading order above skips both large files deliberately.

The other thing is that `mainarch --help` lists 84 subcommands with names like
`gpu-paged-mla-fp8-splitk-ragged-full-latent-gate`. Those are hardware gates,
each pinning one thing that was hard to get right, and they are named after what
they gate rather than for a reader. `just --list` is the curated surface.

## Layout

```
crates/mainarch-sys           raw kernel ABI, amdkfd ioctls, hand-encoded
crates/mainarch-core          device, memory, queues, kernels, attention, model
crates/mainarch-core/src/olmo2.rs   OLMo 2: preflight, weight load, layer, token loop
crates/mainarch-collectives   collectives and the rccl-tests-shaped harness
crates/mainarch-cli           the `mainarch` binary
kernels/                      GPU kernel source and the cross-compile script
demo/sandbox/                 the GPU-free demo server and page
scripts/                      what `just demo`, `just tour` and `just olmo-fetch` run
examples/model-api-plugin/    a standalone external model package
docs/model-api.md             the model-facing API reference
docs/api-stability.md         stability tiers and breaking-change rules
docs/hardware-support.md      supported hardware and validation tiers
docs/ROADMAP.md               the layer-by-layer plan
bench/, baseline/             paired benchmarking against a ROCm baseline
.devcontainer/                CPU devcontainer, with .devcontainer/gpu for passthrough
```

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the detail. The short version is to keep
the ABI boundary explicit, never add a ROCm, HIP, HSA, CUDA or PyTorch runtime
dependency to a production path, and never claim GPU execution, correctness or
performance without the run that backs it.

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) covers behaviour.
[SECURITY.md](SECURITY.md) covers vulnerability reporting, and since this code
drives kernel interfaces directly, it's worth reading before you file anything
publicly.

## License

Apache License 2.0. See [LICENSE](LICENSE).

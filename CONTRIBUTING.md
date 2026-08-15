# Contributing to mainarch

`mainarch` is a Rust stack that talks directly to the Linux kernel ABI for AMD
GPUs. Contributions are welcome when they keep that boundary explicit, tested,
and honestly described.

## Ground rules

- **Keep the ABI boundary explicit.** `mainarch-sys` is hand-encoded ioctls; if
  a change needs a new one, add it there with the kernel UAPI struct it mirrors
  named in a comment.
- **Never add a ROCm, HIP, HSA runtime, RCCL, PyTorch, CUDA, or Python-serving
  runtime dependency to a production path.** That constraint is the project. It
  is enforced by `just check-policy`, not just by review.
- **Do not claim GPU execution, correctness, or performance without the run that
  backs it.** Paste the command and its output in the pull request. A
  performance claim needs an apples-to-apples baseline, meaning the relevant
  ROCm, RCCL, Composable Kernel, vLLM, or SGLang result, or a prior in-repo
  number, and it needs to say which configuration that baseline ran in.
- **Numerics go against a reference.** New kernels compare against an `f64` host
  implementation, with the tolerance stated.
- **Keep CPU-only work labelled CPU-only.** The model API layer plans, validates,
  and emits receipts; it does not dispatch. Work that does not submit GPU work
  should not read as if it does.
- **Keep changes narrow.** One behavioral atom per commit.
- **No generated output, local logs, benchmark scratch, or secrets** in commits.

## Local setup

Use the devcontainer.

- `.devcontainer/devcontainer.json` is the default. CPU only, works on any
  x86-64 Linux host, no GPU required.
- `.devcontainer/gpu/devcontainer.json` adds `/dev/kfd` and `/dev/dri`
  passthrough. Use this on an AMD GPU host. The host user must be in the
  `render` group before the container starts.

```bash
just demo     # build, prove the stack on this machine, serve the demo
just tour     # CPU-only walkthrough of the pieces
just --list   # everything else
```

## Validation

Everything CI runs, in one command:

```bash
just ci
```

That expands to:

```bash
just check-policy      # no ROCm/HIP/HSA/CUDA/PyTorch runtime deps; crate publish policy
just fmt-check         # cargo fmt --all -- --check
just check             # cargo check --workspace --all-targets
just test              # cargo test --workspace
just check-model-api   # the public model API contract gate
just check-demo        # the GPU-free demo sandbox contract gate
just check-olmo        # the OLMo 2 checkpoint contract gate
```

For model API changes, the focused checks are:

```bash
cargo test -p mainarch-core --test model_api_public_contract
python3 tools/check_model_api_boundary_helpers.py --self-test
python3 tools/check_model_api_public_examples.py
cargo test --locked --manifest-path examples/model-api-plugin/Cargo.toml
```

If you add or change a model architecture, the gates for it belong next to the
code and should be written so they fail when the claim is false rather than when
the arithmetic is merely off. The OLMo 2 gates are the worked example. The
attention gate would pass a kernel that ignored head indices, so it perturbs KV
heads and requires them to move. The QK-norm gate would pass Qwen3-style
per-head normalisation, so it perturbs one head and requires a different head's
output to change. Write the test that separates your architecture from the one
it most resembles.

```bash
mainarch olmo2-preflight-selftest                  # CPU-only
mainarch gpu-mha-attention-equivalence-selftest    # needs a GPU
mainarch gpu-olmo2-qk-rope-selftest
mainarch gpu-olmo2-post-norm-selftest
```

`tools/check_model_api_public_examples.py` runs every public model API command
and asserts the receipt lines, including the `launch_executable: ready=false`
boundary. If you extend the API, extend that gate in the same change.

For hardware changes, run the matching gate on real hardware and include the
output:

```bash
mainarch probe
mainarch gpu-selftest
mainarch attn-decode --node <n>
mainarch decode-layer --node <n>
mainarch model-decode --node <n>
mainarch gpu-multi-check          # multi-GPU
mainarch gpu-allreduce-bench      # multi-GPU
```

`tools/kfd_xgmi_preflight.sh` checks the multi-GPU preconditions before you
spend a run on them.

## Pull request checklist

- The change is scoped to one clear atom.
- Formatting, tests, and the gates appropriate to the change pass.
- New behavior has evidence: the command, the hardware, and the output.
- New claims state their negative scope, meaning what the change does *not* prove.
- CPU-only, preflight-only, or non-executing work is labelled as such.
- Documentation and README status text match the actual behavior.
- No unrelated local files, generated output, or secrets.

## Reporting problems

- Bugs: `.github/ISSUE_TEMPLATE/bug_report.md`.
- Gaps in the model API: `.github/ISSUE_TEMPLATE/model_api_gap.md`.
- Security: read [SECURITY.md](SECURITY.md) first. This code drives kernel
  interfaces directly, so please do not open a public issue with exploit
  details.

Behavior expectations are in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

# Pull Request

## Exact claim

State the narrow claim this PR makes. Keep it at the exact scale and behavior
the evidence validates.

## Change type

- [ ] Runtime / kernel ABI / GPU execution
- [ ] Kernels
- [ ] Model API metadata or preflight
- [ ] Checkpoint / weight loading
- [ ] Collectives / performance
- [ ] Documentation
- [ ] Tooling or tests

## Validation

- [ ] `just ci` passes
- [ ] Focused model API checks, if `model_api` changed
      (`python3 tools/check_model_api_public_examples.py`)
- [ ] Hardware gate, if the claim involves GPU execution
- [ ] Apples-to-apples baseline, if the claim involves performance.
      Say which configuration the baseline ran in

## Evidence

Paste the commands and their output. For hardware claims include the GPU model,
gfx target, and driver version.

```

```

## Negative scope

What this change does *not* prove.

## Notes for reviewers

Assumptions, unresolved blockers, unsupported hardware, or intentionally
deferred follow-up work.

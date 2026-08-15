# Model API Gap

## Model Or Architecture

Name the model family, architecture feature, or primitive pattern that is not
represented cleanly by the current `mainarch-core::model_api` surface.

## Exact Gap

- missing primitive
- missing tensor/cache shape contract
- missing lowering route metadata
- missing checkpoint binding metadata
- missing runtime preflight/reporting surface
- unclear public API ergonomics

## Contract Tier

Select the narrowest tier this gap affects:

- [ ] public model authoring contract
- [ ] static runtime metadata contract
- [ ] experimental execution boundary

## Minimal Graph Shape

List the smallest tensor shapes, cache layout, stage boundary, or op sequence
that demonstrates the gap.

```text

```

## Current Workaround

Describe whether the gap is currently represented as an explicit `Gap`, a custom
runner path, a fused native route, or not represented at all.

## Evidence

Include the command, test, example, or docs section that demonstrates the gap.
If the gap crosses into live graph execution, include the hardware correctness
oracle and name the submission blockers that would need to be removed. Otherwise
state that the gap is CPU-only metadata, preflight, or documentation scope.
Do not claim GPU execution or performance unless the matching hardware evidence
is included.

## Negative Scope

State what the issue does not prove or request: for example, no GPU allocation,
no queue submission, no kernel execution, no throughput claim, or no API change.

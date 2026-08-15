# Model API Plugin Example

This is a standalone sample crate that depends on `mainarch-core` by path and
uses `mainarch_core::model_api::prelude::*` the way an external model package
would. The reusable model definition is exported from `src/lib.rs` as
`ExternalMiniMoe`; the binary is only a deterministic CPU-side receipt printer.

It implements a compact MoE-style decoder with embedding, router top-k, local
MoE FFN, residual, output projection, and greedy sampling primitives. The sample
builds and inspects CPU-side metadata, emits the deterministic plugin manifest
and compatibility receipt an external package can pin, then derives the launch
semantic kernarg-projection and projection-aware candidate-selection request
plans plus a compact static handoff receipt from the bundled gfx950 code object
metadata. It does not allocate GPU buffers, bind live device pointers, submit
queues, launch kernels, serve tokens, or claim performance.
The summary also prints the static launch request descriptor table and the
static live-AQL proof-step descriptor subset from the public prelude, then
checks that the derived runtime request rows and proof surfaces use the same
ordered request-plan, requirement, and validation labels.
The package test also exercises the structured `(op_name, kernel_symbol)`
selection-request helpers and `(op_name, candidate_symbols)` host-branch helper
behind the displayed projection, kernel-selection, and branch-candidate label
lines, so external authors do not have to parse `op=symbol` output to audit
ready request bindings or branch alternatives.

Run it from the repository root:

```bash
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --model-api-contract-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --plugin-manifest-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --plugin-compatibility-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-launch-request-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-submission-gate-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-resolved-submission-gate-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-resolved-submission-blocker-report-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-submission-blocker-report-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --runtime-submission-prerequisite-plan-receipt
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --static-handoff-receipt
cargo test --locked --manifest-path examples/model-api-plugin/Cargo.toml
```

`expected-output.txt` is the exact CPU-only summary fixture checked by
`tools/check_model_api_public_examples.py`. `expected-contract.receipt` is the
exact minimal model API contract receipt fixture, `expected-manifest.receipt`
is the exact static plugin manifest receipt fixture,
`expected-compatibility.receipt` is the exact accepted compatibility receipt
fixture, `expected-runtime-launch-request.receipt` is the exact CPU-side runtime
launch request plan receipt fixture emitted through the report-level helper,
`expected-runtime-submission-gate.receipt` is the exact runtime submission gate
receipt fixture emitted through the report-level helper,
`expected-runtime-resolved-submission-gate.receipt` is the exact resolved runtime
submission gate receipt fixture,
`expected-runtime-resolved-submission-prerequisite-plan.receipt` is the exact
resolved runtime submission prerequisite plan receipt fixture with all
prerequisites satisfied and no pending next actions,
`expected-runtime-resolved-submission-blocker-report.receipt` is the exact
resolved runtime submission blocker report receipt fixture with zero blockers,
`expected-runtime-submission-blocker-report.receipt` is the exact runtime
submission blocker report receipt fixture emitted through the report-level
helper,
`expected-runtime-submission-prerequisite-plan.receipt` is the exact runtime
submission prerequisite plan receipt fixture, and
`expected-static-handoff.receipt` is the exact full static handoff receipt
fixture, including full manifest and compatibility receipt fingerprint
bindings, ordered unresolved execution requirement labels, and explicit
non-execution counters. The package test exercises the exported library model
through the public prelude and pins the static metadata, manifest, projection,
selection, contract receipt, compatibility receipt, report-level runtime launch request
helper receipt, report-level runtime submission gate helper receipt, resolved
runtime submission gate helper receipt, resolved runtime submission
prerequisite-plan helper receipt, resolved runtime submission blocker-report
helper receipt, report-level runtime submission blocker report helper receipt,
report-level runtime submission prerequisite plan helper receipt, report-level
live-AQL proof validation application helper, report-level
runtime-request component application helper, report-level runtime component
receipt-plan helper, report-level execution-readiness resolution helper,
report-level execution-readiness resolution receipt-plan helper, report-level
execution-readiness receipt prerequisite overlay helper, report-level
execution-readiness receipt submission-gate overlay helper, report-level
execution-readiness receipt blocker-report overlay helper, static
handoff helper, report-level static handoff readiness helper, static handoff
requirement lookup helper, static handoff non-execution boundary helper, launch
execution non-executable boundary helper, execution request non-submitting
boundary helper, submission gate non-submitting boundary helper, submission
blocker-report non-submitting boundary helper, submission prerequisite-plan
non-submitting boundary helper, runtime component application non-submitting
boundary helper, runtime component application receipt non-submitting boundary
helper, runtime component receipt-plan non-submitting boundary helper,
execution-readiness resolution non-submitting boundary helper,
execution-readiness resolution receipt non-submitting boundary helper,
execution-readiness resolution receipt-plan non-submitting boundary helper,
resolved prerequisite-plan helper, resolved submission-gate helper, resolved
blocker-report helper, and non-executable launch boundary. The
default smoke output
also
surfaces the static handoff's manifest
and compatibility receipt fingerprints, ordered unresolved execution requirement
labels, and non-execution counters on its compact handoff line. Update these
fixtures or test
expectations only when the public model API contract, plugin manifest fingerprint,
compatibility report, graph counts, launch projection counts, runtime launch
request receipt, runtime submission gate receipt, resolved runtime submission
gate receipt, resolved runtime submission prerequisite plan receipt, resolved
runtime submission blocker report receipt, runtime submission blocker report
receipt, runtime submission prerequisite plan receipt, handoff receipt, or
supported boundary intentionally changes.

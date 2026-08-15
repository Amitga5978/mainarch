# mainarch model API

`mainarch-core::model_api` is the model-facing boundary for describing a single
LLM decode step before the raw KFD/AQL runner sees device pointers.

The purpose is narrow: a model architecture should call a small set of typed
`mainarch` primitives instead of forcing the runtime to grow one bespoke Rust
runner per model family. The graph is declarative, CPU-safe metadata. It
allocates no GPU memory and launches no kernels.

The stability policy for this boundary is in `docs/api-stability.md`. In short:
model authoring traits and primitive descriptors are the public experiment
contract; static runtime metadata reports are deterministic tooling surfaces;
live graph execution remains experimental until backed by hardware evidence.
External model packages can start from
`use mainarch_core::model_api::prelude::*;`. The code-level descriptor is
exposed as `MODEL_API_CONTRACT`:
`mainarch-model-api version=0.1.0 stability=pre1-static-metadata
live_execution_supported=false`.
`ModelApiContractInfo::receipt_lines()`, `receipt_text()`, and
`receipt_fingerprint()` expose that descriptor as a deterministic contract
receipt for external package tests.

## What A Model Provides

A model definition implements `ModelDefinition` and receives a
`dyn ModelPrimitiveApi`. It declares tensors, annotates semantic decode stages,
and emits primitive ops:

```rust
api.begin_stage("layers.0.attention", ModelStageKind::Attention)?;
api.emit(PrimitiveOp::RmsNorm(...))?;
api.emit(PrimitiveOp::Linear(...))?;
api.emit(PrimitiveOp::ApplyRope(...))?;
api.emit(PrimitiveOp::KvCacheAppend(...))?;
api.emit(PrimitiveOp::PagedAttention(...))?;
api.emit(PrimitiveOp::Collective(...))?;
api.emit(PrimitiveOp::AddRmsNorm(...))?;
api.end_stage()?;
```

The current primitive vocabulary covers the decode-step pieces already present
in the native substrate:

- embedding lookup
- linear projection with replicated, tensor-parallel column, or tensor-parallel
  row placement
- RMSNorm and residual + RMSNorm
- RoPE with explicit mode
- paged KV-cache append
- paged attention through format-aware cache references
- MoE router top-k
- local routed MoE FFN
- residual add
- explicit collectives
- greedy argmax sampling

The API validates tensor names, dtypes, shapes, attention geometry, MoE geometry,
cache references, collective contracts, sampling contracts, stage ranges, and
stage coverage.

A small authoring example is `custom_model_api`: it defines a toy decoder
outside `mainarch-core::model_api`, implements `ModelDefinition`, emits
embedding, output projection, and sampling primitives through `dyn
ModelPrimitiveApi`, then runs the same readiness report used by the reference
MoE graph.

## Reference MoE Decoder

`ReferenceMoeDecoder` is a contemporary Qwen-style MoE decode graph, not a
model-specific runtime hook.

Its default shape is intentionally representative of the serving target:

- hidden size 4096
- 64 query heads, 4 KV heads, head dim 128
- 128 experts, top-8 routing
- intermediate dim 1536
- tensor parallel group size 8
- paged FP4 GQA KV cache with 16-token blocks
- 1M-token maximum context in the reference config

The reference model calls only the primitive API. It stages the graph as:

- `embedding`
- `layers.N.attention`
- `layers.N.moe`
- `output`
- `sampling`

For one decoder layer this is 20 primitive ops: one embedding op, 12 attention
stage ops, four MoE stage ops, two output ops, and one sampling op. Adding a
layer adds two stages and 16 ops without introducing a new primitive kind.

## Cache Boundary

KV cache is not hard-coded to a single tensor layout. `KvCacheRef` supports:

- separate K/V tensors for paged GQA cache
- opaque cache handles for MLA-style or future layouts

RoPE is a separate primitive so MLA/GQA placement remains a model/runtime
contract instead of being hidden inside attention.

## Lowering Readiness

`MainarchPrimitiveLoweringCatalog::mi355_reference()` maps primitive ops to the
current native mainarch substrate:

- `NativeGpu` means a standalone native route exists.
- `FusedNativeGpu` means the primitive is covered today through an existing
  fused native route.
- `Gap` means the API can represent the op, but this catalog does not yet lower
  it.

`MainarchPrimitiveLoweringCatalog::descriptor()` returns a compact
`ModelPrimitiveLoweringCatalogDescriptor` for authoring tools. It lists the
target, the canonical primitive vocabulary, per-primitive lowering cases,
stable lowering-status labels, native/fused/gap case counts, substrates,
entrypoints, and notes. The descriptor is intentionally parameterized: a
primitive can have both native cases and explicit gap cases when support depends
on dtype, cache format, RoPE mode, or collective kind.

`ModelPrimitiveGraph::lowering_plan()` validates a graph and reports per-op
routes. `ModelPrimitiveGraph::stage_lowering_plan()` folds those routes over the
model-declared stages and reports:

- native/fused/gap counts per stage
- named gap ops per stage
- named unstaged ops
- whole-graph native/fused/gap counts

For the Qwen-style reference MoE graph, the current catalog reports zero gaps
and no unstaged ops. Unsupported expert-parallel all-to-all and decoupled MLA
RoPE are still explicit gaps.

The native/fused route entrypoint metadata is tested against local source files
so catalog entries stay anchored to named Rust-side launcher functions or public
re-exports. `MainarchPrimitiveLoweringCatalog::code_object_kernel_coverage_report(&code_object)`
then maps every non-gap catalog case entrypoint to the conservative set of
code-object kernel symbols it can require and checks those symbols against a
CPU-readable `CodeObjectInfo`. Its `assert_complete()` guard rejects unmapped
entrypoints and missing bundled gfx950 kernel descriptors.

`MainarchPrimitiveLoweringCatalog::abi_registry_coverage_report(&code_object)`
builds on that symbol coverage and joins every catalog-required kernel symbol to
the static named ABI schema registry and semantic ABI schema registry. Its
`assert_complete()` guard rejects missing code-object descriptors, missing named
ABI schema rows, missing semantic ABI schema rows, and size/alignment mismatches
between the code-object descriptor, named schema, and semantic schema. This
proves static catalog-to-code-object-to-registry coverage only; it does not prove
complete kernel argument order/type semantics, kernel-specific argument-value
translation, packet lowering, runtime calling conventions, live execution, or
kernel performance.

## Composed Readiness Report

`ModelPrimitiveGraph::readiness_report(&catalog)` is the one-call summary for a
model author or integration test. It validates the graph and returns the same
CPU-side surfaces that can still be derived independently:

- graph validation report
- tensor storage footprint
- per-op tensor access plan
- per-tensor lifetime plan
- tensor binding manifest
- checkpoint binding manifest
- op-level lowering-readiness plan
- stage-level lowering-readiness plan
- primitive execution manifest
- runtime slot ABI manifest
- runtime dispatch intent manifest
- metadata slot-binding preflight
- runtime dispatch binding preflight
- runtime stage dispatch binding preflight
- runtime stage launch-candidate manifest
- runtime launch entrypoint provenance manifest
- runtime launch kernel-requirement manifest
- runtime launch kernel-metadata manifest
- runtime launch code-object load request plan
- runtime launch code-object base binding request plan
- runtime launch preflight report
- runtime launch AQL packet-field handoff
- runtime launch kernel-selection readiness report
- runtime launch kernel-candidate recommendation plan
- runtime launch kernel-candidate selection request plan
- runtime launch host-launcher branch resolution request plan
- runtime launch argument-binding manifest
- runtime launch device-argument manifest
- runtime launch kernarg layout plan
- runtime launch kernarg serialization plan
- runtime launch kernarg allocation request plan
- runtime launch kernel-argument ABI verification preflight plan
- runtime launch kernel-argument ABI size-compatibility receipt
- runtime launch kernel-argument ABI verification gap report
- runtime launch kernel-argument ABI capacity request plan
- runtime launch kernel-argument ABI schema request plan
- runtime launch kernel-argument ABI semantic plan
- runtime launch kernel-argument ABI semantic gap report
- runtime launch kernel-argument ABI semantic projection plan
- runtime launch kernel-argument ABI semantic projection gap report
- runtime launch kernel-argument ABI semantic projection candidate
  recommendation plan
- runtime launch kernel-argument ABI semantic projection candidate selection
  request plan
- runtime launch kernel-argument ABI semantic projection recommendation report
- runtime launch staging-footprint plan
- runtime launch staging-layout plan
- runtime launch completion-signal policy plan
- runtime launch completion-signal binding request plan
- runtime launch queue-slot plan
- runtime launch queue-reservation request plan
- runtime launch dispatch-geometry plan
- runtime launch AQL packet-template plan
- runtime launch AQL packet relocation-site plan
- runtime launch AQL packet byte-template plan
- runtime launch AQL packet materialization preflight plan
- runtime launch AQL live relocation binding-request plan
- runtime launch executable-readiness gate
- runtime launch execution request plan
- runtime checkpoint payload-to-resident-slot binding plan
- runtime launch window manifest
- runtime stage resource manifest
- runtime stage bundle manifest
- runtime stage dispatch manifest
- structured static readiness issue report
- runtime metadata admission report

`ModelGraphReadinessReport::assert_static_runtime_ready()` is the strongest
static gate for the current model API contract. It requires a binding manifest
with no role/access issues, every declared weight to have a checkpoint binding,
every primitive op to have a lowering route in the selected catalog, and every
op to be covered by a model-declared stage with no stage-local lowering gaps.
It also checks the primitive execution, runtime slot, and runtime dispatch
intent manifests derived from those surfaces, plus the stage resource and stage
bundle manifests.

`ModelGraphReadinessReport::assert_no_checkpoint_or_lowering_gaps()` remains as
the narrower checkpoint/lowering gate.

This report is an API ergonomics layer, not runtime proof. It does not inspect
safetensors headers, open checkpoint shards, allocate buffers, create residency
windows, lower the graph into AQL packets, or execute kernels.

## Plugin Manifest

`inspect_model_plugin(&model, &catalog)` is the one-call CPU-only inspection path
for an external `ModelDefinition`. It returns a `ModelPluginInspectionReport`
containing the full primitive/stage vocabulary descriptors, the target catalog
capability descriptor, the built `ModelPrimitiveGraph`, composed `ModelGraphReadinessReport`,
`ModelPluginManifest`, and `ModelPluginCompatibilityReport`. The helper is an
ergonomics layer over the existing graph/readiness/manifest calls; it does not
load a dynamic library, allocate runtime buffers, or execute GPU work.

`model_primitive_kind_descriptors()` and `model_stage_kind_descriptors()` expose
the full current model API vocabulary as stable labels plus short summaries.
These tables are contract vocabulary, not a claim that every possible parameter
combination for a primitive lowers natively in every catalog.

`ModelPluginInspectionReport::assert_consistent()` is the composite inspection
gate. It verifies that the report still carries the canonical primitive/stage
vocabulary, that the embedded graph validates to the embedded graph report, that
the catalog descriptor is internally consistent and target-matched, that the
manifest matches the readiness-derived manifest, and that compatibility matches
the manifest-derived compatibility report for the recorded target. This is useful
for cached/exported plugin metadata and authoring tests that want one fail-fast
check before inspecting the full graph. It still does not load code, allocate
buffers, submit AQL, execute kernels, or claim throughput.
`ModelPluginInspectionReport::is_static_handoff_ready()` and
`assert_static_handoff_ready()` provide the release-facing guard for accepted,
static-ready plugin inspections before package tests derive static handoff
receipts or default CPU-only launch request fixtures. The static handoff receipt
builder and synthetic CPU launch request helper both call this guard before
deriving fixture metadata.
The guard requires those same inspection consistency checks before trusting the
cached acceptance and static-readiness state.

`ModelPluginInspectionReport::summary()` returns a compact
`ModelPluginInspectionSummary` for package tests and release metadata that do not
need to clone the full graph or readiness report. The summary carries the
contract, model name, target, expected target, contract fingerprint,
accepted/static-ready status, vocabulary and catalog counts, graph manifest
counts, checkpoint coverage counts, runtime slot/dispatch counts, launch-step
counts, and the current `live_execution_supported=false` bit. It also exposes
deterministic `receipt_lines()`, newline-terminated `receipt_text()`, and
`receipt_fingerprint()` helpers so package tests can pin a small stable artifact
without depending on Rust debug formatting or adding a serialization dependency.
`ModelPluginInspectionSummary::assert_consistent_with(&report)` recomputes the
summary from the full report and fails if any exported count or contract field is
stale. The summary is derived metadata only; it does not add serialization,
dynamic plugin loading, checkpoint payload inspection, GPU allocation, AQL
materialization, queue submission, kernel execution, or throughput claims.

`ModelPluginInspectionReport::static_handoff_receipt(namespace, code_object,
window, base_va, alignment)` returns a deterministic
`ModelPluginStaticHandoffReceipt` for accepted static-ready plugins. It composes
the summary fingerprint, rejection fingerprint, manifest metadata fingerprint,
full manifest receipt fingerprint, compatibility receipt fingerprint, metadata
binding template, synthetic device-pointer preflight, metadata admission, launch
semantic projection, projection-selection requests, execution-readiness blockers,
ordered unresolved execution requirement labels, and launch execution request
counters into one compact package handoff. The receipt is for tests and
downstream tooling that need a stable CPU-only boundary before a live runner
exists. It still uses synthetic addresses only; it does not allocate GPU
buffers, prove KFD residency, load checkpoint payloads, patch live AQL packets,
submit queues, execute kernels, or claim serving throughput.
`ModelPluginStaticHandoffReceipt::unresolved_runtime_requirement_names()` returns
those ordered labels without parsing receipt text, and
`has_unresolved_runtime_requirement(label)` lets tests and downstream tooling
check for a specific blocker by stable requirement name.
`is_non_executing_boundary()` and `assert_non_executing_boundary()` provide a
focused guard for the CPU-only handoff boundary: non-executable launch, zero
dispatchable AQL packets, no live AQL submission surfaces, no live queue
mutation, no advertised live execution support, no GPU allocation, and no kernel
submission.
They require those same static-handoff consistency checks before accepting the
receipt as a non-executing CPU-side handoff. The static handoff builder asserts
that the derived launch-readiness report is non-executable, the derived launch
request plan is non-submitting, and the final handoff receipt is non-executing
before returning it.
`ModelPluginInspectionReport::synthetic_cpu_static_handoff_receipt(namespace)`
is the public convenience wrapper for that deterministic fixture path. It uses
the bundled gfx950 code-object metadata,
`DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES`,
`DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE`, and
`DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT`, then delegates to the lower-level
static handoff builder. The returned receipt is byte-equivalent to the explicit
builder for those inputs and keeps the same CPU-only boundary: no GPU buffer
allocation, live AQL packet materialization, queue mutation, kernel submission,
or execution proof.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_request_plan(namespace)`
is the matching convenience wrapper for the default unresolved launch request
fixture. It verifies the accepted static-ready inspection report, inspects the
bundled gfx950 code-object metadata, derives the default metadata bindings and
synthetic device-pointer preflight, then returns the same
`ModelRuntimeLaunchExecutionRequestPlan` that the explicit readiness-level
builder would produce for those inputs. This is a deterministic receipt path for
package tests and downstream tooling; it does not allocate GPU buffers,
materialize live AQL packets, mutate queues, submit kernels, or prove execution.
`ModelRuntimeLaunchExecutionRequestPlan::is_non_submitting_boundary()` and
`ModelRuntimeLaunchExecutionRequestPlan::assert_non_submitting_boundary()`
provide the same focused guard at the execution-request layer, checking
aggregate live AQL submitting surface and live queue mutation counts plus
row-level labels before the plan is folded into submission receipts. The
report-level synthetic launch request helper asserts that boundary before
returning the default fixture.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_gate(namespace)`
delegates through that launch request fixture and returns the default blocked
`ModelRuntimeLaunchSubmissionGate`. The returned gate is byte-equivalent to
calling `submission_gate()` on the explicit launch request plan for the same
inputs. It remains an unresolved static guardrail receipt, not a live execution
admission or queue-submission token, and the helper asserts the gate's
non-submitting boundary before returning it.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_blocker_report(namespace)`
continues that default fixture path and returns the expanded
`ModelRuntimeLaunchSubmissionBlockerReport` for the same blocked gate. It is
byte-equivalent to calling `blocker_report()` on the explicit default submission
gate and is intended for package tests that pin the blocked-submission
explanation before a live runner exists. The helper asserts the blocker-report
non-submitting boundary before returning it.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_prerequisite_plan(namespace)`
continues the same unresolved launch-request fixture path and returns the
expanded per-request `ModelRuntimeLaunchSubmissionPrerequisitePlan`. It is
byte-equivalent to calling `submission_prerequisite_plan()` on the explicit
default launch request plan and is intended for package tests that pin the
CPU-side prerequisite worklist before any live runtime applies those requests.
The helper asserts the prerequisite-plan non-submitting boundary before
returning it.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(namespace,
validations)` maps typed live-AQL proof validation receipts onto the same
default launch request's proof-surface worklist. It is byte-equivalent to
calling `live_aql_proof_validation_application_plan(validations)` on the
explicit default launch request plan and remains a CPU-only receipt overlay:
it does not construct proof inputs, reserve queues, materialize packets, submit
work, or mutate live queues. The helper asserts the validation-application
non-submitting boundary before returning it.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_runtime_request_component_application_plan(namespace,
validations)` continues that path by applying the typed proof validation
receipts to the default prerequisite worklist and returning the current
`ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan`. It is
byte-equivalent to calling `runtime_request_component_application_plan()` on the
explicit prerequisite overlay for those inputs. It remains CPU-only and does not
apply components, generate application receipts, load code objects, allocate
buffers, reserve queues, materialize or submit AQL packets, mutate queues, or
execute kernels.

`ModelPluginInspectionReport::rejection_report()` returns a compact
`ModelPluginRejectionReport` for failed plugin inspections. It includes the
summary, typed readiness issues, typed compatibility issues, total rejection
issue count, and grouped subject lists for lowering gaps, stage-local gaps,
unstaged ops, missing checkpoint weights, and tensor-binding issues. The report
has `assert_rejected()`, `assert_no_rejection()`, `assert_consistent()`, and
`assert_consistent_with(&report)` helpers so external model packages can fail CI
with a stable explanation when a new architecture needs a primitive or metadata
contract change, while stale exported rejection receipts fail before callers
trust the cached accepted/rejected bit. `ModelReadinessIssueKind::as_str()` and
`ModelPluginCompatibilityIssueKind::as_str()` provide stable lower-snake labels
for these receipts. The report also exposes deterministic `receipt_lines()`,
`receipt_text()`, and `receipt_fingerprint()` helpers over the grouped blockers
and typed issue rows. `cargo run -p mainarch-core --example rejected_model_api`
prints the shape using an unsupported expert-parallel all-to-all collective.
This is still static metadata only; it does not serialize packages, load
plugins, allocate GPU resources, or execute kernels.

`ModelPrimitiveGraph::plugin_manifest(&catalog)` and
`ModelGraphReadinessReport::plugin_manifest(model_name)` return a compact
`ModelPluginManifest` for external model packages. The manifest names the
`MODEL_API_CONTRACT`, a deterministic SHA-256 contract fingerprint, model name,
target catalog, primitive and stage vocabulary used by the graph, stable
lower-snake primitive/stage/launch-step labels, tensor/op/stage counts,
checkpoint weight coverage, runtime-slot and dispatch row counts, static
readiness status, and the canonical runtime launch request step descriptors.
The launch-step descriptors include the typed live-AQL proof kind, proof input,
validation method, and queue-mutation policy for rows that need proof
validation. It also carries the current `live_execution_supported=false`
contract bit.
Use `launch_request_step_for(...)` when tooling already has a typed
`RuntimeLaunchExecutionRequestStep`, or
`launch_request_step_for_request_plan(...)` when tooling starts from the stable
request-plan labels printed in receipts and fixtures.
`RuntimeLaunchLiveAqlProofKind` also exposes typed bridge methods for the two
current CPU-side proof validators:
`validate_batch_reservation_plan_proof(...)` and
`validate_materialized_packet_plan_proof(...)`. They call the concrete
`KfdQueueLiveAqlBatchReservationPlanProof::validate_ready` and
`KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready` implementations and
return `RuntimeLaunchLiveAqlProofValidation`, including deterministic
`receipt_lines()`, `receipt_text()`, and `receipt_fingerprint()` helpers. This
validation wrapper records `submits_work=false` and `mutates_live_queue=false`;
it does not construct proof inputs, reserve live queue slots, write AQL packets,
ring doorbells, or make launch submission ready.
`ModelRuntimeLaunchExecutionRequestPlan::live_aql_proof_validation_application_plan(...)`
maps those typed validation receipts onto the ordered pending live-AQL proof
surfaces and returns `ModelRuntimeLaunchLiveAqlProofValidationApplicationPlan`.
That plan records per-surface presence, pass/ready/no-mutation bits,
validation receipt fingerprints, applied/pending counts, rejection reasons, and
side-effect flags. It is an overlay receipt only: it does not construct proof
inputs, reserve queue slots, materialize packets, submit work, or mutate a live
queue. `is_non_submitting_boundary()` and `assert_non_submitting_boundary()`
require the same application-plan consistency checks and then verify the overlay
has no live AQL submitting validations and no live queue-mutating validations.
`submission_gate_with_live_aql_proof_validations(...)` and
`submission_gate_with_live_aql_proof_validation_application_plan(...)` can
consume that typed application state as a submission-gate overlay. Fully applied
proof validations remove only the `live_aql_proof_validation` blocker and set
`all_live_aql_proof_validations_applied=true`; execution-readiness blockers,
pending runtime-request components, and any side-effect or queue-mutation
blockers remain in force. The same validation state can be threaded into
`submission_prerequisite_plan_with_live_aql_proof_validations(...)` and
`submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(...)`
so validated live-AQL rows stop asking for `validate_live_aql_proof` and move to
the next blocked prerequisite action. The default `submission_gate()` and
`submission_prerequisite_plan()` remain unchanged.
`ModelRuntimeLaunchSubmissionPrerequisitePlan::is_non_submitting_boundary()` and
`assert_non_submitting_boundary()` provide a focused guard that the prerequisite
worklist has no live AQL submission side-effect next actions, no live AQL
submitting prerequisite rows, and no live queue mutation prerequisite rows.
`ModelRuntimeLaunchSubmissionGate::is_non_submitting_boundary()` and
`assert_non_submitting_boundary()` provide a focused guard that a gate still has
no live AQL submission side effects and no live queue mutation, independent of
whether its current metadata state is blocked or `submission_ready=true`.
They require those same submission-gate consistency checks before accepting a
blocked or resolved gate as a non-submitting CPU-side handoff.
`ModelRuntimeLaunchSubmissionBlockerReport` exposes the same helper pair for the
expanded blocker-report view, checking its side-effect and queue-mutation
counts without treating a zero-blocker receipt as live admission.
The blocker-report helpers require those same blocker-report consistency checks
before accepting the expanded report as a non-submitting CPU-side handoff.
`ModelRuntimeLaunchSubmissionPrerequisitePlan::runtime_request_component_application_plan()`
then exposes the current runtime-component application worklist, and
`ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::application_receipt_plan(...)`
validates externally supplied component-application receipts against that
worklist. `submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(...)`
and `submission_gate_with_runtime_request_component_application_receipt_plan(...)`
consume the validated receipt-intake state as another CPU-only overlay: matched
receipts clear the corresponding per-row component pending counts and move those
rows to execution-readiness cleanup, while deferred rows still waiting on
proof-validation or rejected receipt rows keep the runtime-component blocker in
force. A complete overlay can remove the aggregate `runtime_request_components`
blocker, but execution-readiness blockers still keep `submission_ready=false`.
`ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` require the same application-plan
consistency checks and then reject any live AQL submitting application row or
live queue-mutating application row. The receipt-intake plan exposes the same
`is_non_submitting_boundary()` and `assert_non_submitting_boundary()` shape for
caller-supplied runtime component receipts, including explicit false checks for
the live-submission and live-queue-mutation guard fields.
`RuntimeLaunchRuntimeRequestComponentApplicationReceipt::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` provide the individual caller-supplied
receipt guard before those receipts are aggregated into the receipt-intake
plan. The report-level synthetic CPU path asserts the receipt-plan boundary
before deriving downstream readiness metadata from caller-supplied component
receipts. There are no separate report-level gate or blocker-report helpers for
the runtime-component-only overlay; report-level gate and blocker-report
convenience returns are guarded on the execution-readiness receipt-overlay and
resolved submission paths below.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(namespace,
validations, receipts)` keeps that receipt-intake path available directly from
the accepted inspection report and bundled synthetic CPU fixture inputs. It
accepts caller-supplied runtime component application receipts unchanged and is
byte-equivalent to deriving the report-level application worklist first, then
calling `application_receipt_plan(receipts)`.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(namespace,
validations, runtime_component_receipts)` continues that path by applying typed
proof validations and caller-supplied runtime component receipts to the default
prerequisite worklist, then returning the current
`ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan`. It is
byte-equivalent to deriving the explicit prerequisite overlay first, then
calling `execution_readiness_blocker_resolution_plan()`.
`ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` require the same worklist consistency
checks, reject embedded resolution-receipt state, and keep execution readiness
plus submission readiness false while the worklist remains a CPU-only blocker
resolution handoff.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` maps
caller-supplied execution-readiness blocker resolution receipts onto that
report-level readiness-resolution worklist. It is byte-equivalent to deriving
the report-level readiness-resolution worklist first, then calling
`resolution_receipt_plan(resolution_receipts)`. It does not generate resolution
receipts, resolve blockers, mark execution readiness complete, or authorize
submission.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` overlays that
caller-supplied readiness-resolution receipt chain onto the report-level
prerequisite worklist. It is byte-equivalent to deriving the prerequisite
worklist after proof-validation and runtime-component receipt overlays, building
the readiness-resolution receipt plan, then calling
`submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(...)`.
It may produce a CPU-side `submission_ready=true` metadata handoff when every
receipt is matched, but it does not submit work or authorize live serving.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` returns the
corresponding report-level submission-gate metadata for the same caller-supplied
receipt chain. It is byte-equivalent to calling the prerequisite overlay helper
first and then `submission_gate()`. It does not generate receipts, materialize
AQL packets, submit queues, or turn the metadata handoff into live admission.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` expands that
report-level receipt-overlay gate into blocker-report metadata. It is
byte-equivalent to calling the gate overlay helper first and then
`blocker_report()`. It does not generate receipts, submit queues, or turn a
zero-blocker metadata report into live admission.
`ModelRuntimeLaunchSubmissionPrerequisitePlan::execution_readiness_blocker_resolution_plan()`
then groups the current `resolve_execution_readiness_blocker` prerequisite rows
by unique execution-readiness requirement in gate blocker order. The plan keeps
the source request-plan and typed step membership for each blocker, exposes
requirement and typed-step lookups, and has deterministic receipt helpers. It is
a CPU-only resolution worklist: it does not resolve blockers, create resolution
receipts, load code objects, allocate memory, reserve queues, materialize AQL,
submit work, mutate queues, or make submission ready.

`ModelPluginManifest::assert_static_metadata_ready()` verifies that the manifest
has no static readiness issues, lowering gaps, unstaged ops, binding issues, or
missing checkpoint weight bindings, and that its launch-step descriptor table
matches `RuntimeLaunchExecutionRequestStep::DESCRIPTORS`, including typed
live-AQL proof kind labels for the queue reservation and AQL live-relocation
rows. It also recomputes the fingerprint from stable metadata fields, not Rust
debug formatting. It is the small plugin-facing receipt for "this model can
target the static mainarch metadata contract." It still does not allocate
buffers, validate live device pointers, load code objects, materialize AQL
packets, submit work, execute kernels, or claim serving throughput.

`ModelPluginManifest::compatibility_report(&catalog)` returns a
`ModelPluginCompatibilityReport` for external model packages before they hand a
manifest to a later runtime layer. It records whether the manifest contract
matches `MODEL_API_CONTRACT`, the manifest target matches the selected lowering
catalog, the SHA-256 fingerprint recomputes cleanly, the static metadata gate is
clean, and the manifest has not advertised live execution support that the
contract does not provide. `assert_compatible_with(&catalog)` is the fail-fast
variant. `ModelPluginCompatibilityReport::assert_consistent()` checks that the
cached accepted bit, match booleans, fingerprint fields, live-execution support
state, and typed issue rows agree before callers trust
`ModelPluginCompatibilityReport::is_accepted()` or `assert_accepted()`. A
rejected report carries typed issue rows for manifest consistency, contract,
target, fingerprint, static metadata, and live-execution mismatches. The report
also exposes deterministic `receipt_lines()`, newline-terminated
`receipt_text()`, and `receipt_fingerprint()` helpers so external package tests
can pin the accepted or rejected compatibility decision as a small key/value
artifact.

## Runtime Metadata Admission Report

`ModelGraphReadinessReport::validate_metadata_runtime_admission(bindings)`
combines the strongest static readiness issue report, complete slot-binding
preflight, and aggregate stage binding preflight into
`ModelRuntimeMetadataAdmissionReport`. The report carries:

- static readiness issues
- complete runtime slot binding validation
- runtime dispatch binding validation
- per-stage slot binding validation
- per-stage dispatch binding validation
- aggregate admission status and issue count

`ModelRuntimeMetadataAdmissionReport::assert_consistent()` checks that the
admission report target matches the embedded static-readiness, dispatch-binding,
stage-binding, and stage-dispatch target provenance, and that the dispatch
binding report still carries the same slot-binding snapshot as the admission
report. `is_admitted()` and `assert_admitted()` require that consistency before
trusting a zero-issue metadata handoff.

This is the final metadata-only gate before a future runtime would replace
logical handles with real device allocations. It does not allocate buffers,
prove pointer validity, load checkpoints, create packet layouts, submit AQL, or
execute kernels.

## Structured Static Readiness Issues

`ModelGraphReadinessReport::static_readiness_issues()` folds the readiness
blockers that matter to the current static runtime contract into
`ModelReadinessIssueReport`. Each issue carries:

- issue kind
- source surface
- subject name
- human-readable message

The report covers tensor binding role/access issues, missing checkpoint weight
bindings, primitive lowering gaps, unstaged ops, and stage-local lowering gaps.
It is useful for model authoring tools and tests that need stable issue kinds
instead of parsing assertion text.

This is still a post-validation CPU-side report. It does not recover from an
invalid graph, inspect checkpoint files, validate proposed runtime buffer
handles, allocate memory, or execute work.

## Primitive Execution Manifest

`ModelPrimitiveGraph::primitive_execution_plan(&catalog)` composes the stage
metadata, tensor access plan, tensor binding manifest, and lowering routes into
one ordered static row per primitive op:

- op index, op name, and primitive kind
- optional stage name and kind
- selected lowering route and target catalog
- read tensors annotated with binding class, dtype, role, lifetime access, and
  storage bytes
- write tensors annotated with the same binding metadata
- whole-plan native/fused/gap counts, unstaged ops, and binding issues

This is the first runtime-facing manifest in the model API: a future executor can
consume one table instead of separately joining graph ops, stages, bindings, and
lowering routes. It is still not executable lowering. It does not build AQL
packets, assign device pointers, allocate scratch memory, create streams, prove
aliasing safety, load checkpoint bytes, or launch kernels.

## Runtime Slot ABI Manifest

`ModelPrimitiveGraph::runtime_slot_plan(&catalog)` turns the binding and
primitive execution manifests into a deterministic slot table:

- one tensor slot per declared tensor
- class-specific slot lists for external inputs, external outputs, checkpoint
  weights, persistent state, scratch tensors, and unused tensors
- one op row per primitive with read-slot and write-slot references
- each slot carries tensor name, dtype, role, binding class, lifetime access,
  storage bytes, and optional checkpoint key
- the same static issue surface for binding issues, lowering gaps, and unstaged
  ops

This gives a future direct-GPU runtime a stable ABI-shaped table before actual
device pointers exist. It still does not allocate memory, bind addresses, choose
scratch reuse, validate residency, create packet layouts, or execute work.

## Runtime Dispatch Intent Manifest

`ModelPrimitiveGraph::runtime_dispatch_intent_plan(&catalog)` projects the
runtime slot op rows into ordered dispatch-intent rows. Each row reports:

- op index, op name, and primitive kind
- optional stage name and kind
- selected lowering route and bare entrypoint symbols
- named tensor-slot arguments such as `input`, `weight`, `output`, `cache.key`,
  and `cache.value`
- primitive scalar metadata such as `vocab`, `hidden`, `head_dim`,
  `cache_dtype`, `weight_format`, and parallelism
- read slot IDs and write slot IDs
- the same static issue surface for binding issues, lowering gaps, and unstaged
  ops

This is packet-builder input metadata only. It gives a future runtime a compact
CPU-side intent table after tensor names have been resolved to slot IDs. It does
not translate scalar metadata into a kernel ABI, choose entrypoint variants,
bind pointers, produce kernel arguments, build AQL packet headers, create
queues, schedule dependencies, synchronize work, submit packets, execute
kernels, or prove performance.

## Metadata Slot-Binding Preflight

`ModelRuntimeSlotPlan::validate_complete_buffer_bindings()` validates a proposed
complete binding table against the slot ABI without touching device memory. Each
`RuntimeSlotBufferBinding` supplies:

- slot index
- tensor name
- byte capacity
- read/write access mode
- logical handle string

The preflight reports unknown slots, duplicate slots, tensor/slot mismatches,
undersized buffers, access modes that do not cover the tensor's lifetime access,
bindings for unused slots, and required non-unused slots that are missing.

`ModelRuntimeSlotPlan::metadata_binding_template("namespace")` builds a
deterministic complete metadata table for all required non-unused slots. The
template uses logical handles shaped as `namespace.slot.N`, exact slot byte
sizes, and access modes derived from the tensor lifetime access. It is intended
for examples, tests, and authoring tools that need a complete metadata table
before real device allocations exist.

`ModelRuntimeSlotPlan::validate_required_buffer_bindings(required_slots,
bindings)` applies the same metadata checks to a required slot subset. This lets
a stage-resource row validate only its `resource_slots` against a larger global
binding table. Rows for unrelated slots are ignored by this scoped preflight.

`ModelRuntimeStageSlotPlan::validate_stage_buffer_bindings(&slots, bindings)`
runs the scoped preflight for every stage resource row and returns per-stage
validation receipts. This is the aggregate stage-level admission check for a
logical binding table before any device pointers exist.

This is metadata validation only. It does not allocate buffers, prove that a
handle is a real GPU allocation, check pointer alignment, map peer memory, load
checkpoint payloads, enforce GPU lifetime safety, or execute any primitive.

## Runtime Device-Pointer Preflight

`ModelRuntimeSlotPlan::validate_complete_device_pointer_bindings()` validates a
proposed complete slot/device-VA table without touching GPU memory. Each
`RuntimeSlotDevicePointerBinding` supplies:

- slot index
- tensor name
- byte capacity
- read/write access mode
- device virtual address

The preflight applies the same slot metadata checks as logical handle bindings
and adds CPU-side pointer checks for null addresses, 16-byte alignment, address
span overflow, and overlapping spans. `validate_required_device_pointer_bindings`
applies those checks to a scoped slot subset, matching the required-slot
preflight used for stage resources.

`ModelRuntimeSlotPlan::device_pointer_binding_template(base_va, alignment)`
builds a deterministic synthetic table for examples and tests. The template
does not allocate memory; it only produces aligned, non-overlapping numeric VAs
large enough for the declared slot sizes.

`ModelRuntimeSlotPlan::validate_device_pointer_lifetimes(&lifetime,
&device_pointers)` joins a validated slot/device-VA table back to the tensor
lifetime manifest. `ModelGraphReadinessReport::validate_runtime_device_pointer_lifetimes`
is the convenience form for the composed readiness report. The resulting
`ModelRuntimeSlotDevicePointerLifetimePlan` records, per non-unused runtime
slot:

- tensor role, binding class, lifetime access, and required storage bytes
- whether a device pointer binding was present and CPU-valid
- device VA span from the proposed binding
- first and last op index/name from the static lifetime plan
- read/write counts and live-range op count
- per-slot pointer-validation and lifetime-metadata issue counts

`assert_cpu_lifetime_ready()` requires a complete device-pointer validation and
all accessed tensor lifetimes to have a CPU-valid binding. This is the
pre-submission correlation step needed before a future allocator can reason
about pointer reuse or residency windows.

This is still metadata validation. It does not prove that a VA is registered
with KFD, resident on the target GPU, peer-visible, initialized with checkpoint
bytes, synchronized with pending work, safe to reuse across live ranges, or safe
to pass to a live kernel.

## Runtime KFD Allocation/Residency Request Plan

`ModelRuntimeSlotDevicePointerLifetimePlan::kfd_allocation_residency_request_plan(resident_gpu_ids)`
turns a CPU-ready slot/device-pointer lifetime preflight into deterministic KFD
allocation and residency intent rows. `ModelGraphReadinessReport::runtime_slot_kfd_allocation_residency_request_plan`
is the report-level convenience form that first rebuilds the lifetime preflight
from a `ModelRuntimeSlotDevicePointerValidation`.

The resulting `ModelRuntimeSlotKfdAllocationResidencyRequestPlan` reports:

- the sorted resident KFD GPU IDs a future runtime must acquire/map against
- one request per non-unused runtime slot
- the selected allocation kind: host-visible coherent GTT for external
  input/output slots, or device-local VRAM preferring public coherent mapping
  for weights, persistent state, and scratch slots
- primary KFD allocation flags and optional fallback flags matching the
  existing raw KFD allocation helpers
- requested aligned allocation bytes, proposed device VA span, and static
  first/last op lifetime range
- per-slot readiness plus aggregate request, byte, residency-map, and issue
  counts

`assert_kfd_allocation_residency_request_ready()` requires the underlying
device-pointer lifetime preflight to be CPU-ready, at least one resident KFD GPU
ID, and zero request issues. This is an allocation/residency work order for a
future live runtime. It does not call `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, acquire
VMs, map handles into device VMs, prove peer visibility, load payload bytes,
submit AQL, or execute kernels. The plan keeps `allocation_performed=false`,
`residency_proven=false`, and `live_execution_supported=false`.

## Runtime KFD VM-Acquire Request Plan

`ModelRuntimeSlotKfdAllocationResidencyRequestPlan::kfd_vm_acquire_request_plan()`
derives the KFD VM-acquire precondition worklist from the same resident GPU IDs
used by the slot allocation/residency plan. `ModelGraphReadinessReport::runtime_kfd_vm_acquire_request_plan`
is the report-level convenience form that first rebuilds the slot lifetime and
allocation/residency request metadata from a
`ModelRuntimeSlotDevicePointerValidation`.

The resulting `ModelRuntimeKfdVmAcquireRequestPlan` reports:

- one VM-acquire request per sorted resident KFD GPU ID
- the KFD device path (`/dev/kfd`) and `AMDKFD_IOC_ACQUIRE_VM` ioctl number a
  future runtime must use
- per-GPU KFD fd and DRM render fd requirements
- per-GPU VM-acquire metadata readiness
- aggregate KFD fd, DRM fd, VM-acquire, readiness, and issue counts

`assert_kfd_vm_acquire_request_ready()` requires the upstream slot
allocation/residency request plan to be ready, at least one resident KFD GPU ID,
and zero VM-acquire request issues. This is still a CPU-only work order. It does
not open `/dev/kfd`, discover or open DRM render nodes, bind file descriptors,
call `AMDKFD_IOC_ACQUIRE_VM`, allocate memory, map memory, submit AQL, or execute
kernels. The plan keeps `kfd_fd_bound_count=0`, `drm_fd_bound_count=0`,
`vm_acquire_performed_count=0`, `all_vms_acquired=false`, and
`live_execution_supported=false`.

## Runtime KFD Alloc-Memory Request Plan

`ModelRuntimeSlotKfdAllocationResidencyRequestPlan::kfd_alloc_memory_request_plan(&vm_acquires)`
derives deterministic `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` argument rows from the
slot allocation/residency plan and the KFD VM-acquire request plan.
`ModelGraphReadinessReport::runtime_kfd_alloc_memory_request_plan` is the
report-level convenience form that rebuilds both upstream plans from a
`ModelRuntimeSlotDevicePointerValidation` and resident GPU ID list.

The resulting `ModelRuntimeKfdAllocMemoryRequestPlan` reports:

- one alloc-memory request per non-unused runtime slot
- the KFD device path and `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` ioctl number
- the deterministic allocation GPU ID selected as the first sorted resident KFD
  GPU ID
- per-slot ioctl argument fields: `va_addr`, `size`, `gpu_id`, primary flags,
  and optional fallback flags
- per-slot KFD fd, VM-acquire, allocation, handle, mmap offset, and map-to-GPU
  precondition state
- aggregate allocation kind, byte, map-to-GPU, handle, mmap offset, readiness,
  and issue counts

`assert_kfd_alloc_memory_request_ready()` requires the upstream slot
allocation/residency request plan and VM-acquire request metadata to be ready,
at least one allocation request, and zero alloc-memory request issues. This is
still a CPU-only ioctl work order. It does not open `/dev/kfd`, bind KFD or DRM
file descriptors, call `AMDKFD_IOC_ACQUIRE_VM`, call
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, receive allocation handles or mmap offsets,
map handles into device VMs, prove residency, load checkpoint payloads, submit
AQL, or execute kernels. The plan keeps `kfd_fd_bound_count=0`,
`vm_acquire_performed_count=0`, `allocation_performed_count=0`,
`handle_bound_count=0`, `mmap_offset_bound_count=0`,
`allocation_performed=false`, and `live_execution_supported=false`.

## Runtime KFD Alloc-Memory Result Binding Plan

`ModelRuntimeKfdAllocMemoryRequestPlan::kfd_alloc_memory_result_binding_plan(result_bindings)`
validates caller-supplied `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` result receipts
against the deterministic alloc-memory request rows. `ModelGraphReadinessReport::runtime_kfd_alloc_memory_result_binding_plan`
is the report-level convenience form that rebuilds the upstream lifetime,
allocation/residency, VM-acquire, and alloc-memory request metadata before
checking those supplied receipts.

The resulting `ModelRuntimeKfdAllocMemoryResultBindingPlan` reports:

- one result-binding validation row per alloc-memory request row
- the requested slot, tensor, allocation GPU ID, VA, size, and primary/fallback
  KFD allocation flags that a receipt must match
- the caller-supplied allocation handle and mmap offset, with mmap offset zero
  treated as bound when a receipt row is present
- per-slot primary/fallback flag match state, handle-bound state,
  allocation-performed receipt state, and metadata readiness
- aggregate matched, missing, duplicate, unmatched, handle-bound,
  mmap-offset-bound, allocation-performed, primary/fallback flag, byte,
  readiness, and issue counts

`assert_kfd_alloc_memory_result_bound()` requires the upstream alloc-memory
request metadata to be ready, every request row to have exactly one matching
result receipt, each receipt to match slot/tensor/GPU/VA/size and either primary
or fallback flags, and each receipt to carry a nonzero allocation handle. This
is a receipt-validation boundary for a future live runtime. It does not open
`/dev/kfd`, bind KFD or DRM file descriptors, call `AMDKFD_IOC_ACQUIRE_VM`, call
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, allocate memory, map handles into device VMs,
prove residency, load checkpoint payloads, submit AQL, or execute kernels. The
plan records caller-supplied allocation receipts only and keeps
`live_execution_supported=false`.

## Runtime KFD Map-Memory Request Plan

`ModelRuntimeKfdAllocMemoryRequestPlan::kfd_map_memory_request_plan()`
derives deterministic `AMDKFD_IOC_MAP_MEMORY_TO_GPU` metadata rows from the
alloc-memory request plan. `ModelGraphReadinessReport::runtime_kfd_map_memory_request_plan`
is the report-level convenience form that rebuilds the upstream lifetime,
allocation/residency, VM-acquire, and alloc-memory request metadata from a
`ModelRuntimeSlotDevicePointerValidation` plus resident GPU ID list.

The resulting `ModelRuntimeKfdMapMemoryRequestPlan` reports:

- one map-memory row per alloc-memory request row
- the KFD device path and `AMDKFD_IOC_MAP_MEMORY_TO_GPU` ioctl number
- the resident KFD GPU IDs that a future runtime must copy into the
  `device_ids_array_ptr` storage
- per-slot ioctl argument fields: `handle`, `device_ids_array_ptr`,
  `n_devices`, and `n_success`
- per-slot KFD fd, VM-acquire, allocation, allocation-handle, device-ID array,
  and map-memory side-effect state
- aggregate map-to-GPU, byte, device-ID, handle, device-ID array, success,
  readiness, and issue counts

`all_static_request_metadata_ready=true` means the upstream alloc-memory request
metadata is ready and every row has resident GPU IDs and deterministic
`n_devices` metadata. `assert_kfd_map_memory_request_ready()` is intentionally
stricter and remains false in the CPU-only API because the live allocation
handles and host storage pointers for `device_ids_array_ptr` are not bound.
`assert_consistent()` still permits a future caller-supplied bound overlay when
KFD fd, VM-acquire, allocation, handle, and device-ID array fields are coherent;
the default report-level helper does not create that overlay.
This plan does not open `/dev/kfd`, bind file descriptors, call
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, receive allocation handles or mmap offsets,
allocate or pin the device-ID array, call `AMDKFD_IOC_MAP_MEMORY_TO_GPU`, prove
residency or peer visibility, submit AQL, or execute kernels. It keeps
`handle_bound_count=0`, `device_ids_array_bound_count=0`,
`map_memory_performed_count=0`, `map_memory_success_count=0`,
`all_live_request_args_bound=false`, `all_request_args_ready=false`,
`map_memory_performed=false`, and `live_execution_supported=false`.

## Runtime KFD Map-Memory Argument Binding Plan

`ModelRuntimeKfdMapMemoryRequestPlan::kfd_map_memory_argument_binding_plan(alloc_memory_results, device_ids_array_bindings)`
binds CPU-validated map-memory ioctl arguments from two caller-supplied receipt
surfaces: allocation handles already checked by the alloc-memory result binding
plan, and host pointers to caller-owned `device_ids_array_ptr` storage containing
the resident KFD GPU ID list for each request row.
`ModelGraphReadinessReport::runtime_kfd_map_memory_argument_binding_plan` is the
report-level convenience form that rebuilds the upstream slot lifetime,
allocation/residency, VM-acquire, alloc-memory request, and map-memory request
metadata before checking the supplied alloc-memory receipts and device-ID array
bindings.

The resulting `ModelRuntimeKfdMapMemoryArgumentBindingPlan` reports:

- one argument-binding validation row per map-memory request row
- the `AMDKFD_IOC_MAP_MEMORY_TO_GPU` handle, `device_ids_array_ptr`,
  `n_devices`, and `n_success` argument values that a future runtime can use
- the caller-supplied device-ID array values and whether they exactly match the
  request row's resident GPU IDs
- per-slot allocation-result presence, allocation-performed receipt state,
  handle-bound state, device-ID array-bound state, argument-readiness state, and
  issue count
- aggregate device-ID array matched, missing, duplicate, unmatched, handle-bound,
  device-ID array-bound, allocation-performed, argument-ready, byte, device-ID,
  readiness, and issue counts

`assert_kfd_map_memory_arguments_bound()` requires static map-memory request
metadata to be ready, alloc-memory result receipts to be ready, every map-memory
request row to have a nonzero allocation handle, exactly one matching nonzero
`device_ids_array_ptr`, and device-ID array values equal to the resident GPU ID
list. This is still a CPU-only argument binding boundary. It does not open
`/dev/kfd`, bind KFD or DRM file descriptors, call `AMDKFD_IOC_ACQUIRE_VM`, call
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, allocate memory, allocate or pin the device-ID
array, call `AMDKFD_IOC_MAP_MEMORY_TO_GPU`, update `n_success`, prove residency
or peer visibility, submit AQL, or execute kernels. The plan records
caller-supplied argument bindings only and keeps `map_memory_performed=false`
and `live_execution_supported=false`.

## Runtime KFD Map-Memory Result Binding Plan

`ModelRuntimeKfdMapMemoryArgumentBindingPlan::kfd_map_memory_result_binding_plan(map_memory_results)`
validates caller-supplied `AMDKFD_IOC_MAP_MEMORY_TO_GPU` result receipt rows
against the already-bound map-memory argument rows. Each receipt must match the
request slot, tensor, allocation handle, `device_ids_array_ptr`, `n_devices`, and
must report `n_success` equal to the resident GPU ID count for that row.
`ModelGraphReadinessReport::runtime_kfd_map_memory_result_binding_plan` is the
report-level convenience form that rebuilds slot lifetime, allocation/residency,
VM-acquire, alloc-memory request/result, map-memory request, and map-memory
argument-binding metadata before checking the supplied result receipts.

The resulting `ModelRuntimeKfdMapMemoryResultBindingPlan` reports:

- one result-binding validation row per map-memory request row
- the expected map-memory arguments and caller-supplied result values for handle,
  `device_ids_array_ptr`, `n_devices`, and `n_success`
- per-slot argument-readiness, result-presence, allocation-performed,
  handle-bound, device-ID array-bound, map-memory-performed receipt,
  map-memory-success receipt, result-readiness, and issue state
- aggregate matched, missing, duplicate, unmatched, handle-bound, device-ID
  array-bound, allocation-performed, map-memory-performed, map-memory-success,
  byte, device-ID, readiness, and issue counts

`assert_kfd_map_memory_result_bound()` requires the upstream argument-binding
plan to be ready, every request row to have exactly one matching result receipt,
and every receipt to report successful mapping for every resident GPU ID. This
is still a CPU-only receipt-validation boundary. It does not open `/dev/kfd`,
bind KFD or DRM file descriptors, call `AMDKFD_IOC_ACQUIRE_VM`, call
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, allocate memory, allocate or pin the device-ID
array, call `AMDKFD_IOC_MAP_MEMORY_TO_GPU`, map or unmap memory, prove hardware
residency or peer visibility beyond caller-supplied successful receipt counts,
submit AQL, or execute kernels. The plan records caller-supplied result receipts
only and keeps `live_execution_supported=false`.

## Runtime Slot KFD Residency Binding Plan

`ModelRuntimeSlotKfdAllocationResidencyRequestPlan::kfd_residency_binding_plan(map_memory_results)`
correlates successful KFD alloc/map receipt validation back to the runtime slot
device-pointer lifetime rows. It checks that each allocation/residency request
slot has a matching map-memory result entry with the same tensor, class,
lifetime access, allocation kind, byte counts, resident GPU IDs, and a successful
`n_success` count for every resident KFD GPU ID.
`ModelGraphReadinessReport::runtime_slot_kfd_residency_binding_plan` is the
report-level convenience form that rebuilds slot lifetime, allocation/residency,
VM-acquire, alloc-memory request/result, map-memory request, argument-binding,
and result-binding metadata before producing the per-slot KFD residency receipt.

The resulting `ModelRuntimeSlotKfdResidencyBindingPlan` reports:

- one residency-binding row per runtime allocation/residency request row
- the CPU-validated slot device VA span plus the caller-supplied allocation
  handle, `device_ids_array_ptr`, `n_devices`, and `n_success` receipt values
- per-slot allocation-request readiness, map-memory result presence,
  device-pointer-bound, allocation-performed, handle-bound,
  device-ID-array-bound, map-memory-performed, map-memory-success,
  result-readiness, KFD-residency-proven receipt state, and issue count
- aggregate matched/missing map-memory result, allocation-kind,
  device-pointer-bound, allocation-performed, handle-bound, device-ID-array,
  map-memory-performed, map-memory-success, residency-proven, byte,
  residency-map, readiness, and issue counts

`assert_kfd_residency_bound()` requires the upstream allocation/residency request
plan and map-memory result binding plan to be ready, every requested slot to have
one successful map-memory result receipt, and every slot VA span to remain
CPU-valid. This is still a CPU-only receipt correlation boundary. It does not
open `/dev/kfd`, bind KFD or DRM file descriptors, call
`AMDKFD_IOC_ACQUIRE_VM`, call `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, allocate memory,
allocate or pin the device-ID array, call `AMDKFD_IOC_MAP_MEMORY_TO_GPU`, map or
unmap memory, load checkpoint payload bytes, submit AQL, synchronize queues, or
execute kernels. It proves only that caller-supplied KFD allocation and
map-memory receipts coherently cover every model API runtime slot, and it keeps
`live_execution_supported=false`.

## Runtime Checkpoint Payload Binding Plan

`ModelCheckpointBindingPlan::runtime_checkpoint_payload_binding_plan(slots, key_resolution, kfd_residency, payload_bindings)`
correlates caller-supplied checkpoint payload span metadata with the runtime
checkpoint-weight slots that already have CPU-validated KFD residency receipts.
It consumes a fully resolved `ModelCheckpointKeyResolution`, the runtime slot
ABI table, a `ModelRuntimeSlotKfdResidencyBindingPlan`, and payload span rows
provided by the caller. Exact checkpoint keys map one payload span to one weight
slot; wildcard packed expert keys require one payload row for every resolved
concrete checkpoint key and aggregate those byte spans back to the packed model
slot.

`ModelGraphReadinessReport::runtime_checkpoint_payload_binding_plan` is the
report-level convenience form that rebuilds the KFD residency chain from device
pointer, alloc-memory, device-ID-array, and map-memory result receipts before
binding the supplied payload metadata to checkpoint-weight slots.
`ModelGraphReadinessReport::synthetic_cpu_runtime_checkpoint_payload_binding_plan`
and the matching
`ModelPluginInspectionReport::synthetic_cpu_runtime_checkpoint_payload_binding_plan`
derive the same plan from caller-supplied available checkpoint keys, a payload
source label, resident GPU IDs, and deterministic synthetic CPU receipt values.
Those helpers are for examples and external-package smoke tests; they still
require callers to supply the checkpoint key inventory and do not inspect a
checkpoint file.

The existing safetensors metadata path can now supply those payload rows without
hand-written offsets. `SafetensorsShard::runtime_checkpoint_payload_bindings()`
and `SafetensorsIndex::runtime_checkpoint_payload_bindings()` validate the model
checkpoint bindings against parsed safetensors headers first, then emit
`RuntimeCheckpointPayloadBinding` rows with the safetensors source path,
absolute file offset, byte length, dtype, and expected concrete tensor shape.
Wildcard packed expert bindings produce one payload row per resolved expert key.
This is still header-derived metadata; it does not mmap the safetensors payload,
read tensor bytes, or allocate staging buffers.

The resulting `ModelRuntimeCheckpointPayloadBindingPlan` reports:

- one payload-binding row per model API checkpoint binding entry
- the runtime slot, checkpoint key pattern, resolved concrete checkpoint keys,
  dtype, shape, storage bytes, destination device VA span, and allocation handle
- caller-supplied payload source, offset, byte length, dtype, and shape rows
- per-slot slot-binding, KFD-residency, destination-span, payload metadata,
  payload byte-count, and payload-bound readiness state
- aggregate resolved/missing checkpoint key, expected/matched/missing,
  duplicate, and unmatched payload binding, residency-proven, payload-ready,
  payload-bound, byte, readiness, and issue counts

`assert_checkpoint_payload_bound()` requires every checkpoint weight to have a
resolved checkpoint key, every resolved concrete key to have exactly one payload
span with matching dtype, expected shape, non-overflowing source byte range, and
matching byte count, and every destination checkpoint-weight slot to have a
proven KFD residency receipt. This is still a CPU-only payload metadata
correlation boundary. It does not open checkpoint files, mmap safetensors
payloads, read payload bytes, allocate or pin staging buffers, copy bytes to
VRAM, flush caches, submit SDMA or AQL work, synchronize queues, execute
kernels, generate real KFD receipts, or change `live_execution_supported=false`.

## Checkpoint Payload Direct-Read Work Orders

`CheckpointPayloadDirectReadPlan::from_checkpoint_payload_binding_plan(plan, alignment)`
is a `weights` helper that consumes a bound
`ModelRuntimeCheckpointPayloadBindingPlan` and emits deterministic work orders
for a future direct checkpoint loader. Each work order records the runtime slot,
tensor, concrete checkpoint key, safetensors source path, payload file range,
enclosing direct-I/O-aligned read window, payload offset inside the staging
window, destination offset inside the model slot, destination device VA span,
dtype, and concrete shape.

Packed expert checkpoint bindings remain explicit: a wildcard model checkpoint
entry produces one safetensors payload row per concrete expert key, and the
direct-read plan assigns those payload rows sequential destination offsets
inside the packed model weight slot. The plan also reports source count, slot
count, work-order count, total payload bytes, total aligned read bytes, maximum
single staging window, and the direct-I/O alignment.

`CheckpointPayloadDirectReadPlan::staging_batch_plan(staging_slot_count)` and
`CheckpointPayloadDirectReadStagingPlan::from_direct_read_plan(plan, staging_slot_count)`
then coalesce those per-payload aligned windows into source-local staging
batches. Overlapping or touching aligned windows for the same safetensors source
become one batch with copy pieces for every payload in that read window. Each
batch records the aligned file range, read byte count, assigned reusable staging
slot, staging-slot byte offset, and the copy pieces that preserve the original
work-order index, model slot, checkpoint key, payload staging offset, payload
byte count, destination offset, and destination device VA span.

The staging plan reports source count, slot count, batch count, piece count,
total payload bytes, total coalesced read bytes, maximum batch read bytes,
staging slot count, total staging bytes, and a milli-scale read amplification.
This mirrors the two-slot overlap loaders in the existing `weights` module while
remaining only a deterministic schedule.
`CheckpointPayloadDirectReadStagingPlan::receipt_lines()`,
`receipt_text()`, and `receipt_fingerprint()` expose that schedule as a compact
line-oriented handoff receipt. The receipt pins the target, direct-I/O
alignment, source/slot/batch/piece counts, staging-slot footprint, payload/read
byte totals, read amplification, `live_execution_supported=false`, and explicit
false counters for opening checkpoint files, reading payload bytes, allocating
or pinning host staging, copying to VRAM, submitting SDMA or AQL, and executing
kernels. `assert_non_executing_boundary()` and `is_non_executing_boundary()`
provide the matching guard before an integration treats the receipt as a
metadata-only checkpoint-loader boundary.

This is still metadata only. Constructing the plan validates that checkpoint
payloads are already bound to proven model API residency receipts, but it does
not open checkpoint files, read payload bytes, allocate host staging, pin memory,
copy bytes to VRAM, submit SDMA or AQL work, synchronize queues, execute
kernels, or change `live_execution_supported=false`.

`CheckpointPayloadDirectReadStagingPlan::buffered_host_staging_receipt()` is the
next CPU-only checkpoint-loader boundary. It explicitly opens the planned
checkpoint payload sources, allocates reusable host staging bytes, reads the
file-backed portion of each aligned staging batch with buffered positional
reads, zero-fills any aligned EOF tail padding, and hashes the staged payload
copy spans in deterministic copy-piece order. The matching receipt records the
requested aligned read bytes, actual file bytes read, EOF tail-padding bytes,
payload byte total, payload fingerprint, and true counters for opened files,
payload bytes read, and host staging allocation. It still does not use
`O_DIRECT`, pin host staging, copy to VRAM, submit SDMA or AQL work,
synchronize queues, execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadDirectReadStagingPlan::mapped_host_staging_receipt()` is the
mmap-backed sibling of that CPU-only checkpoint-loader boundary. It opens the
same planned payload sources, maps only the readable file-backed portion of each
coalesced staging batch using page-aligned read-only mappings, copies mapped
bytes into reusable host staging slots, unmaps the batch before continuing, and
hashes the same staged payload copy spans as the buffered path. Its receipt
records host page size, mmap call count, mapped file bytes, mmap byte span,
tail-padding bytes, payload byte total, and payload fingerprint, with explicit
false counters for buffered reads, host pinning, VRAM copies, SDMA/AQL
submission, queue synchronization, kernel execution, and
`live_execution_supported=false`.

`CheckpointPayloadDirectReadStagingPlan::host_to_device_copy_plan()` is the
plan-only host-staging-to-device handoff manifest. It turns every staging copy
piece into an absolute host staging byte offset, payload byte count, and
destination device VA span while preserving the source, batch, work-order,
runtime slot, tensor, checkpoint key, and destination offset metadata. The
receipt pins the total copy byte count and true guards for bound host staging
offsets and destination device VAs. It does not require pinned host memory, copy
bytes to VRAM, submit SDMA or AQL work, synchronize queues, execute kernels, or
change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::destination_residency_proof_input()` is
the CPU-only proof-input for the destination residency prerequisite. It mirrors
each host-to-device copy row as a destination VA span, records aggregate minimum
and maximum destination VA bounds, and keeps KFD residency rows, allocation
handles, resident GPU IDs, KFD queries, executable proof state, and
`destination_residency_proven` false. It does not allocate VRAM, copy bytes to
VRAM, submit SDMA or AQL work, synchronize queues, execute kernels, or change
`live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::destination_residency_query_request(kfd_residency)`
is the CPU-only request table that joins those destination spans to
`ModelRuntimeSlotKfdResidencyBindingPlan` rows by runtime slot. It groups copy
destinations by KFD allocation, records allocation handles, resident GPU IDs,
KFD row tensor/class/access checks, per-copy destination spans,
destination-in-allocation checks, and metadata-ready state, and keeps KFD query
execution, executable query state, `destination_residency_proven`, VRAM
allocation, and VRAM copy false. It does not query KFD, allocate VRAM, copy
bytes to VRAM, submit SDMA or AQL work, synchronize queues, execute kernels, or
change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::sdma_queue_reservation_input()` is the
CPU-only proof-input for the SDMA queue reservation prerequisite. It derives the
upload waves from the copy plan, requests logical SDMA linear-copy packet
capacity for each copy row, requests one completion-fence packet per populated
wave, and records aggregate packet, dword, byte, and doorbell batch counts. The
receipt keeps queue IDs, queue rings, doorbells, completion signals, packet
materialization, applied reservations, SDMA submission, and VRAM copy all
unbound or false. It does not create a KFD queue, write packet bytes, ring a
doorbell, copy bytes to VRAM, submit SDMA or AQL work, synchronize queues,
execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadSdmaQueueReservationInput::sdma_queue_reservation_result_binding_plan(...)`
binds those requested wave rows to caller-supplied queue reservation result
metadata. The receipt checks each row's wave index, first packet, packet count,
packet dword span, nonzero queue ID, queue ring base/size, reserved write-index
range, and doorbell value, then records
`queue_reservation_prerequisite_satisfied_by_receipt=true` only when every wave
has exactly one valid result binding and every requested packet is reserved. It
keeps `queue_reservation_executed=false`, VRAM copy, SDMA submission, and
`live_execution_supported=false`; the CPU path reconciles queue-reservation
receipts and does not create a queue, write queue memory, ring a doorbell, or
submit work.

`CheckpointPayloadHostToDeviceCopyPlan::copy_completion_signal_binding_input()`
is the CPU-only proof-input for the copy completion signal binding prerequisite.
It mirrors the SDMA queue reservation's completion-fence packet rows, requests
one `amd_signal_t` completion signal per fence packet, and records the packet
index, packet dword offset, initial signal value, and completion value that a
future packet materializer must bind. The receipt keeps signal handles, signal
device VAs, completion signal bindings, queue reservations, materialized
packets, signal waits, SDMA submission, and VRAM copy all unbound or false. It
does not create an AMD signal, bind signal memory into a packet, write packet
bytes, ring a doorbell, wait on a signal, submit SDMA or AQL work, synchronize
queues, execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadCopyCompletionSignalBindingInput::copy_completion_signal_result_binding_plan(...)`
binds those requested completion-signal rows to caller-supplied signal handles
and signal device VAs. The receipt checks the binding index, completion signal
index, wave index, completion packet index, packet dword span, initial signal
value, completion value, and nonzero handle/device VA for each row, then records
`copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true` only
when every requested signal has exactly one valid result binding. It keeps
`completion_signal_binding_executed=false`, `completion_signals_created=false`,
`completion_signal_wait_issued=false`, VRAM copy, SDMA submission, and
`live_execution_supported=false`; the CPU path reconciles signal receipts and
does not create a signal, bind packet memory, write packet bytes, wait on a
signal, submit work, or execute a live upload.

`CheckpointPayloadHostToDeviceCopyPlan::sdma_copy_packet_materialization_input()`
is the CPU-only proof-input for the SDMA copy packet materialization
prerequisite. It walks the SDMA queue reservation and completion signal binding
inputs in queue order, assigns one packet row per linear-copy chunk and one
packet row per completion-fence packet, records packet kind labels, packet
dword offsets, packet byte sizes, source staging offsets, destination VAs, and
completion signal values, and asserts that packet rows and offsets are
contiguous. The receipt keeps host virtual addresses, completion signal device
VAs, completion signal bindings, queue packet reservations, applied
reservations, materialized packet bytes, SDMA submission, and VRAM copy all
unbound or false. It does not write SDMA packet bytes, reserve queue slots, bind
host or signal addresses, ring a doorbell, submit SDMA or AQL work, synchronize
queues, execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadSdmaCopyPacketMaterializationInput::sdma_copy_packet_materialization_result_binding_plan(...)`
binds those packet rows to caller-supplied queue packet slots, host virtual
addresses, destination device VAs, completion signal handles, and signal device
VAs. The receipt checks every copy packet and completion fence packet against
the materialization input, records that all SDMA packet bytes were materialized
by receipt, and sets
`upload_packet_materialization_prerequisite_satisfied_by_receipt=true` only
when every requested packet row has exactly one valid result binding. It keeps
`packet_materialization_executed=false`, `queue_memory_mutated=false`, VRAM
copy, SDMA submission, and `live_execution_supported=false`; the CPU path
reconciles materialized packet receipts and does not write queue memory, ring a
doorbell, submit SDMA or AQL work, synchronize queues, or execute kernels.

`CheckpointPayloadHostToDeviceCopyPlan::sdma_copy_packet_validation_input()` is
the CPU-only proof-input for the SDMA copy packet validation prerequisite. It
replays the materialization rows as validation rows and records row contiguity,
SDMA packet index contiguity, packet dword offsets, expected packet byte sizes,
copy payload spans, packet template requests, packet shape checks, and
completion signal value checks. The receipt keeps host virtual addresses,
completion signal bindings, signal device VAs, queue packet reservations,
applied reservations, materialized packet bytes, byte-validation counts, SDMA
submission, and VRAM copy all unbound or false. It does not read materialized
packet bytes, mutate queue memory, reserve queue slots, ring a doorbell, submit
SDMA or AQL work, synchronize queues, execute kernels, or change
`live_execution_supported=false`.

`CheckpointPayloadSdmaCopyPacketValidationInput::sdma_copy_packet_validation_result_binding_plan(...)`
binds those validation rows to caller-supplied queue packet slots, host virtual
addresses, signal handles, signal device VAs, materialized-packet state, and
packet byte validation evidence. The receipt checks each validation result
against the validation input, records packet template, shape, byte-count,
offset, copy-span, completion-signal-value, and byte-validation counts, and sets
`upload_packet_validation_prerequisite_satisfied_by_receipt=true` only when
every requested validation row has exactly one valid result binding. It keeps
`packet_validation_executed=false`, `queue_memory_mutated=false`, VRAM copy,
SDMA submission, and `live_execution_supported=false`; the CPU path reconciles
validation receipts and does not mutate queue memory, ring a doorbell, submit
SDMA or AQL work, synchronize queues, or execute kernels.

`CheckpointPayloadHostToDeviceCopyPlan::cache_visibility_policy_input()` is the
CPU-only cache-visibility handoff after SDMA packet validation. It groups
validated packet rows by upload wave, selects the device-scope VRAM visibility
policy that a future live uploader would use after its completion signal, and
records that no host visibility request, cache flush, cache invalidate, or VRAM
visibility proof has executed. The receipt keeps host staging pinning,
destination residency proof, queue reservations, applied reservations,
materialized packets, byte-validation counts, cache operations, SDMA
submission, and VRAM copy all unbound or false. It does not flush or invalidate
caches, prove VRAM visibility, submit SDMA or AQL work, synchronize queues,
execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::upload_synchronization_plan_input()` is
the CPU-only upload-completion synchronization handoff after cache visibility
policy selection. It joins completion-signal binding rows with cache-visibility
policy rows and records the wait rows a future live uploader would issue after
SDMA submission, including completion packet offsets, signal values, timeout
guard requests, policy row spans, and per-wave visibility preconditions. The
receipt keeps signal bindings, signal device VAs, queue reservations, applied
reservations, materialized packets, byte-validation counts, VRAM visibility
proof, issued waits, observed waits, queue synchronization, SDMA submission, and
VRAM copy all unbound or false. It does not wait on signals, observe
completion, synchronize queues, prove visibility, submit SDMA or AQL work,
execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceUploadCacheVisibilityPolicyHandoff::upload_completion_synchronization_handoff(...)`
is the final CPU-only upload prerequisite receipt bridge. It correlates the
cache-visibility handoff with the upload synchronization plan input, verifies
runtime input 7's receipt fingerprint, and advances the receipt prerequisite
chain to `8/8` while leaving `upload_ready=false`. The receipt records planned
completion waits and timeout guards, plus the prior receipt-bound queue packet,
SDMA packet, byte-validation, and completion-signal counts. It does not issue
waits, observe completion, synchronize queues, prove visibility, submit SDMA or
AQL work, execute kernels, or change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::host_to_device_upload_schedule()` is the
next plan-only uploader handoff. It groups copy rows into upload waves by
direct-read staging batch, preserves batch order, assigns a staging-slot reuse
epoch for every wave, and records per-wave copy counts, byte totals, host
staging spans, and destination VA bounds. The schedule receipt pins total copy
bytes and maximum wave size while asserting that batch order is preserved and
staging-slot reuse is serialized. It does not pin host memory, copy bytes to
VRAM, submit SDMA or AQL work, synchronize queues, execute kernels, or change
`live_execution_supported=false`.

`CheckpointPayloadHostToDeviceUploadSchedule::upload_prerequisite_plan()` names
the runtime work still required before a live checkpoint upload may happen. The
worklist currently includes host staging page pinning, destination VRAM
residency querying via `CheckpointPayloadDestinationResidencyQueryRequest`,
SDMA queue reservation, copy completion signal binding, SDMA copy packet
materialization and validation, cache visibility policy selection, and upload
completion synchronization. The receipt records every prerequisite as
unsatisfied with a next-action label and input type, plus ordered aggregate
requirement, unsatisfied-requirement, next-action, and next-action-input label
lists for runner worklist selection. The plan also exposes
`prerequisite_for(...)`, `prerequisite_requirement_names()`,
`unsatisfied_prerequisite_requirement_names()`,
`next_action_requirement_names()`, `next_action_labels()`, and
`next_action_input_labels()` for callers that want the same CPU-only ordering
without parsing receipt text. It pins `upload_ready=false` and explicitly
records that no next action has executed. It does not query KFD, pin memory,
copy bytes to VRAM, submit SDMA or AQL work, synchronize queues, execute
kernels, or change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::host_to_device_upload_runtime_handoff(...)`
turns that ordered prerequisite worklist into a compact CPU-only handoff for a
future upload runner. It builds the eight prerequisite input receipts in the
same order as the worklist, records each input requirement, next action, input
type, receipt kind, receipt fingerprint, and receipt line count, and includes
the upload schedule and prerequisite-plan receipt fingerprints. The handoff
keeps `upload_ready=false`, `next_actions_executed=false`, host staging pinning,
destination residency proof, VRAM copy, SDMA/AQL submission, and kernel
execution all false. It does not execute or submit the listed runtime actions,
synchronize queues, or change `live_execution_supported=false`.

`CheckpointPayloadHostToDeviceCopyPlan::host_to_device_upload_bound_runtime_handoff(...)`
adds the caller-supplied host-staging base VA to that CPU-only handoff and
fingerprints the derived host-staging pin virtual-address plan for prerequisite
input 0. The receipt ties the raw `CheckpointPayloadHostStagingPinRequest`
fingerprint to the bound virtual-address-plan fingerprint and records the
materialized host page-address state while keeping `pin_request_executable=false`,
`upload_ready=false`, `next_actions_executed=false`, `host_staging_pinned=false`,
VRAM copy, SDMA/AQL submission, and kernel execution all false. It does not pin
host memory, execute any prerequisite action, submit queues, or change
`live_execution_supported=false`. Unlike the standalone host-staging virtual
address plan, this aggregate handoff requires materialized host page addresses;
use the standalone plan when a runner needs to inspect a nonzero but
page-unaligned base VA as a non-executable preflight case.

`CheckpointPayloadHostToDeviceUploadBoundRuntimeHandoff::mapped_host_staging_upload_handoff(...)`
binds that bound upload handoff to a successful
`CheckpointPayloadHostStagingKfdMapMemoryResultBindingPlan`. The receipt proves
that prerequisite input 0 has a matching KFD USERPTR map-memory result for the
same host virtual page spans and total page-pin bytes, then records
`host_staging_pin_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=1`, and `unsatisfied_prerequisite_count=7`. It
keeps `upload_ready=false`, `next_actions_executed=false`,
`host_staging_pinned=false`, destination residency proof, VRAM copy, SDMA/AQL
submission, queue synchronization, and kernel execution all false; the CPU path
only reconciles receipts and does not execute a live upload.

`CheckpointPayloadHostToDeviceUploadMappedHostStagingHandoff::destination_residency_upload_handoff(...)`
binds that mapped host-staging handoff to a
`CheckpointPayloadDestinationResidencyQueryRequest` and the source
`ModelRuntimeSlotKfdResidencyBindingPlan`. The receipt proves that prerequisite
input 1's destination VRAM residency metadata matches the bound runtime input
fingerprint and still matches the KFD allocation handles, resident GPU IDs,
device-local VRAM allocation rows, and destination VA spans, then records
`destination_residency_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=2`, and `unsatisfied_prerequisite_count=6`. It
keeps `upload_ready=false`, `next_actions_executed=false`,
`destination_residency_proven=false`, VRAM copy, SDMA/AQL submission, queue
synchronization, and kernel execution all false; the CPU path reconciles
receipts and does not query KFD or execute a live upload.

`CheckpointPayloadHostToDeviceUploadDestinationResidencyHandoff::sdma_queue_reservation_upload_handoff(...)`
binds that destination-residency handoff to a successful
`CheckpointPayloadSdmaQueueReservationResultBindingPlan`. The receipt proves
that prerequisite input 2's SDMA queue reservation metadata matches the bound
runtime input fingerprint and the result-binding plan, then records
`sdma_queue_reservation_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=3`, and `unsatisfied_prerequisite_count=5`. It
keeps `upload_ready=false`, `next_actions_executed=false`,
`host_staging_pinned=false`, `destination_residency_proven=false`, VRAM copy,
SDMA/AQL submission, queue synchronization, and kernel execution all false; the
CPU path reconciles receipts and does not reserve live queue slots or execute a
live upload.

`CheckpointPayloadHostToDeviceUploadSdmaQueueReservationHandoff::copy_completion_signal_binding_upload_handoff(...)`
binds that SDMA queue reservation handoff to a successful
`CheckpointPayloadCopyCompletionSignalResultBindingPlan`. The receipt proves
that prerequisite input 3's copy-completion signal binding metadata matches the
bound runtime input fingerprint and the result-binding plan, then records
`copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=4`, and `unsatisfied_prerequisite_count=4`. It
keeps `upload_ready=false`, `next_actions_executed=false`,
`host_staging_pinned=false`, `destination_residency_proven=false`, VRAM copy,
SDMA/AQL submission, queue synchronization, and kernel execution all false; the
CPU path reconciles receipts and does not create live signals, bind packet
memory, wait on signals, or execute a live upload.

`CheckpointPayloadHostToDeviceUploadCopyCompletionSignalBindingHandoff::sdma_copy_packet_materialization_upload_handoff(...)`
binds that copy-completion signal handoff to a successful
`CheckpointPayloadSdmaCopyPacketMaterializationResultBindingPlan`. The receipt
proves that prerequisite input 4's SDMA copy-packet materialization metadata
matches the bound runtime input fingerprint, that copy packet host VAs match
the bound host-staging base VA, that queue packet writes match the SDMA queue
reservation receipt, and that completion fence packets use the bound signal
handles and signal device VAs. It records
`upload_packet_materialization_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=5`, and `unsatisfied_prerequisite_count=3` while
keeping `upload_ready=false`, `next_actions_executed=false`,
`host_staging_pinned=false`, `destination_residency_proven=false`, VRAM copy,
SDMA/AQL submission, queue synchronization, and kernel execution all false; the
CPU path reconciles receipts and does not write live queue memory, submit SDMA,
or execute a live upload.

`CheckpointPayloadHostToDeviceUploadPacketMaterializationHandoff::sdma_copy_packet_validation_upload_handoff(...)`
binds the packet-materialization upload handoff to a successful
`CheckpointPayloadSdmaCopyPacketValidationResultBindingPlan`. The receipt
proves that prerequisite input 5's SDMA copy-packet validation metadata matches
the bound runtime input fingerprint and that every validation row correlates
with the queue slot, host VA, destination VA, signal handle, and signal device
VA from the materialization receipt. It records
`upload_packet_validation_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=6`, and `unsatisfied_prerequisite_count=2` while
keeping `upload_ready=false`, `next_actions_executed=false`,
`host_staging_pinned=false`, `destination_residency_proven=false`, VRAM copy,
SDMA/AQL submission, queue synchronization, and kernel execution all false; the
CPU path reconciles receipts and does not validate live packet memory, submit
SDMA, or execute a live upload.

`CheckpointPayloadHostToDeviceUploadPacketValidationHandoff::cache_visibility_policy_upload_handoff(...)`
binds the packet-validation upload handoff to the cache visibility policy input
at runtime prerequisite input 6. The receipt proves that the policy input's
fingerprint matches the bound runtime handoff and that every policy wave
correlates with validated packet rows, including validation row spans, copy
counts, destination VA bounds, packet-offset checks, copy payload spans, and
completion signal values. It records
`cache_visibility_policy_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=7`, and `unsatisfied_prerequisite_count=1` while
keeping `upload_ready=false`, cache flush/invalidate, VRAM visibility proof,
VRAM copy, SDMA/AQL submission, queue synchronization, and kernel execution all
false; the CPU path selects the policy by receipt and does not issue live cache
operations or submit a live upload.

`CheckpointPayloadHostToDeviceUploadCacheVisibilityPolicyHandoff::upload_completion_synchronization_handoff(...)`
binds the cache-visibility upload handoff to the upload synchronization plan
input at runtime prerequisite input 7. The receipt proves that the wait-plan
input's fingerprint matches the bound runtime handoff and that every completion
wait row correlates with the selected cache-visibility policy wave and the
completion-signal request/result rows. It records
`upload_completion_synchronization_prerequisite_satisfied_by_receipt=true`,
`satisfied_prerequisite_count=8`, and `unsatisfied_prerequisite_count=0` while
keeping `upload_ready=false`, completion waits, queue synchronization, VRAM
visibility proof, VRAM copy, SDMA/AQL submission, and kernel execution all
false; the CPU path plans the synchronization by receipt and does not wait,
observe completion, synchronize queues, or submit a live upload.

`CheckpointPayloadHostToDeviceUploadSchedule::host_staging_pin_request()`
derives the first unresolved prerequisite input from the upload waves. The
request coalesces waves by reusable staging slot, records the staging byte range
that a future live runtime would pin, and derives merged page-rounded pin spans
from those raw ranges. The receipt records the page-sized host pin allocation,
total page-pin bytes, slack bytes introduced by page rounding, and whether the
page spans stay inside the host pin allocation and cover the raw ranges. The
receipt keeps host virtual address binding, page-address materialization,
executable pin requests, issued pin calls, and `host_staging_pinned` all false;
it does not copy bytes to VRAM, submit SDMA or AQL work, synchronize queues,
execute kernels, or change `live_execution_supported=false`.
The `--checkpoint-host-staging-pin-page-rounding-receipt` example command pins
the same boundary with a smaller CPU-only direct-read alignment so the raw range
is not page-aligned and the receipt must show nonzero page-pin slack.

`CheckpointPayloadHostStagingPinRequest::host_virtual_address_binding_plan(...)`
binds those raw and page-rounded pin byte ranges to a caller-supplied host
staging base virtual address. It records one host VA span per staging-slot range
and one page-aligned host VA span per merged page-pin span, proving that the
future pin call has concrete host addresses and page counts without issuing the
pin call. The receipt keeps `pin_request_executable=false`,
`pin_calls_issued=false`, and `host_staging_pinned=false`; it does not pin host
memory, copy bytes to VRAM, submit SDMA or AQL work, synchronize queues,
execute kernels, or change `live_execution_supported=false`. A nonzero but
page-unaligned base VA remains a valid preflight input, but the receipt records
`all_host_page_addresses_page_aligned=false` and does not materialize executable
pin-call inputs.

`CheckpointPayloadHostStagingPinVirtualAddressPlan::kfd_userptr_pin_argument_plan(...)`
derives CPU-only KFD USERPTR allocation argument rows from materialized host
page spans. It canonicalizes the resident KFD GPU IDs, records `/dev/kfd`,
`AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`, `USERPTR|WRITABLE|NO_SUBSTITUTE|COHERENT`
flags, host VA, pin size, and selected allocation GPU ID, and fingerprints the
source virtual-address plan. The receipt keeps KFD fd binding, VM acquire,
USERPTR allocation, handle binding, mmap-offset binding, issued pin calls,
host pinning, VRAM copy, SDMA/AQL submission, and kernel execution all false.
It requires materialized page addresses; use the standalone virtual-address
plan to inspect page-unaligned preflight inputs before deriving USERPTR rows.

`CheckpointPayloadHostStagingKfdUserptrPinArgumentPlan::kfd_vm_acquire_request_plan()`
derives one CPU-only `AMDKFD_IOC_ACQUIRE_VM` request row per resident KFD GPU ID
from those USERPTR pin arguments. The receipt records the `/dev/kfd` path, KFD
and DRM fd requirements, acquire-VM ioctl number, source USERPTR receipt
fingerprint, and request-readiness state while keeping fd binding, VM acquire,
USERPTR allocation, pin calls, host pinning, VRAM copy, SDMA/AQL submission,
queue synchronization, and kernel execution all false. It requires ready USERPTR
pin arguments so a future live runner can consume the VM-acquire worklist
without reparsing the allocation argument receipt.

`CheckpointPayloadHostStagingKfdVmAcquireRequestPlan::kfd_userptr_alloc_request_plan()`
derives one CPU-only `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` USERPTR allocation request
row per host page span after VM-acquire metadata is ready. The receipt records
the allocation GPU selection policy, `/dev/kfd`, alloc-memory ioctl number,
USERPTR flags, host VA, byte size, resident GPU IDs, upstream USERPTR and
VM-acquire receipt fingerprints, and request-readiness state while keeping fd
binding, VM acquire, USERPTR allocation, handle/mmap-offset binding, pin calls,
host pinning, VRAM copy, SDMA/AQL submission, queue synchronization, and kernel
execution all false. It is still a deterministic worklist for a future live
runner; it does not issue the alloc-memory ioctl.

`CheckpointPayloadHostStagingKfdUserptrAllocRequestPlan::kfd_userptr_alloc_result_binding_plan(...)`
binds caller-supplied `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` USERPTR allocation result
metadata back to those host-page request rows. It validates the request index,
argument index, page span, ioctl argument VA/size/GPU/flags, nonzero handle, and
mmap offset binding, then emits a deterministic result receipt tied to the
alloc-request receipt fingerprint. This result-binding helper still does not
open KFD or DRM fds, acquire VMs, issue ioctls, pin host memory, copy to VRAM,
submit SDMA/AQL, synchronize queues, execute kernels, or change
`live_execution_supported=false`; the supplied handle and mmap offset are
evidence from a future live runner, not side effects performed by this CPU-only
path.

`CheckpointPayloadHostStagingKfdUserptrAllocResultBindingPlan::kfd_map_memory_request_plan()`
derives one CPU-only `AMDKFD_IOC_MAP_MEMORY_TO_GPU` request row per bound
USERPTR allocation result. The receipt records the allocation handle,
resident GPU IDs, `n_devices`, host-page byte span, upstream alloc-result
receipt fingerprint, and map-memory readiness metadata while keeping KFD fd
binding, `device_ids_array_ptr` binding, `n_success`, map execution, pinning,
VRAM copy, SDMA/AQL submission, queue synchronization, and kernel execution
false or zero. It requires the alloc-result binding receipt to be complete and
side-effect-free before exposing the map-memory worklist.

`CheckpointPayloadHostStagingKfdMapMemoryRequestPlan::kfd_map_memory_argument_binding_plan(...)`
binds caller-supplied checkpoint map-memory argument metadata back to that
worklist. It validates one nonzero `device_ids_array_ptr` and resident GPU ID
array per request row, verifies the allocation handle and request indices still
match, records the upstream map-memory request receipt fingerprint, and exposes
whether every map-memory argument row is ready. This is still an
argument-binding-only boundary: it does not open KFD fds, issue
`AMDKFD_IOC_MAP_MEMORY_TO_GPU`, update `n_success`, map memory, pin host memory,
copy to VRAM, submit SDMA/AQL, synchronize queues, execute kernels, or change
`live_execution_supported=false`.

`CheckpointPayloadHostStagingKfdMapMemoryArgumentBindingPlan::kfd_map_memory_result_binding_plan(...)`
binds caller-supplied checkpoint map-memory result receipt metadata back to the
argument rows. It validates the request, allocation, argument, page-span,
handle, `device_ids_array_ptr`, and `n_devices` fields, then requires the
post-ioctl `n_success` value to equal the resident GPU ID count. The resulting
receipt records observed map-memory results, success counts, result metadata
readiness, and `residency_proven=true` only when every argument row has exactly
one successful result receipt. This remains a receipt-only boundary: it does not
open KFD fds, issue `AMDKFD_IOC_MAP_MEMORY_TO_GPU`, pin host memory, copy to
VRAM, submit SDMA/AQL, synchronize queues, execute kernels, or change
`live_execution_supported=false`.

## Runtime Dispatch Binding Preflight

`ModelGraphReadinessReport::validate_runtime_dispatch_bindings(bindings)` joins
the dispatch-intent rows with a logical `RuntimeSlotBufferBinding` table. The
receipt reports:

- one dispatch binding row per dispatch intent
- required slot IDs per op
- named slot and scalar arguments plus read and write binding refs with slot
  metadata and optional logical handles
- per-dispatch scoped slot-binding validation
- complete slot-binding validation for the whole slot table

The same receipt is included in
`ModelRuntimeMetadataAdmissionReport::dispatch_bindings`, so metadata admission
fails if a future packet builder would see an unbound dispatch read/write slot.
This is still a handle-resolution preflight only. It does not prove that handles
are GPU pointers, bind addresses, pack kernel arguments, choose packet formats,
submit AQL, synchronize queues, or execute work.

## Runtime Stage Resource Manifest

`ModelPrimitiveGraph::runtime_stage_slot_plan(&catalog)` groups the runtime slot
ABI rows by model-declared stage. Each stage row reports:

- stage index, name, kind, op range, and op names
- unique read slots and write slots
- stage input slots, computed as first-use reads before a slot is produced
  inside the stage
- resource slots grouped by binding class
- stage-local resource byte footprint by binding class
- native/fused/gap op counts and named gap ops

This gives a future runtime a deterministic resource table for stage bundling
and preflight. It does not choose stream placement, residency windows,
allocation/reuse, packet fusion, synchronization, or execution order beyond the
already declared op order.

## Runtime Stage Bundle Manifest

`ModelPrimitiveGraph::runtime_stage_bundle_plan(&catalog)` packages each
stage-resource row with the ordered runtime op slot rows that belong to that
stage. Each bundle row reports:

- stage index, name, kind, op range, and op count
- the exact `RuntimeStageSlotPlanEntry` resource row for that stage
- ordered `RuntimeOpSlotPlanEntry` rows, preserving graph op order
- named unstaged ops plus their op slot rows
- the same binding issue and lowering-gap readiness surface

This is the first CPU-side per-stage handoff shape that joins stage resources
and executable-candidate op metadata. It still does not construct AQL packets,
choose packet fusion, allocate or bind device buffers, assign stream placement,
build residency windows, synchronize work, or launch kernels.

## Runtime Stage Dispatch Manifest

`ModelPrimitiveGraph::runtime_stage_dispatch_plan(&catalog)` packages each
stage-resource row with the ordered dispatch-intent rows that belong to that
stage. Each stage-dispatch row reports:

- stage index, name, kind, op range, and op count
- the exact `RuntimeStageSlotPlanEntry` resource row for that stage
- ordered `RuntimeDispatchIntentEntry` rows with lowering routes, entrypoint
  symbols, named slot arguments, scalar metadata, and read/write slot IDs
- named unstaged ops plus their dispatch intent rows
- the same binding issue and lowering-gap readiness surface

This is a static handoff manifest for a future packet builder. It still does not
choose packet fusion, translate scalar metadata into kernargs, allocate or bind
device buffers, assign stream placement, build residency windows, synchronize
work, submit AQL, or execute kernels.

## Runtime Stage Dispatch Binding Preflight

`ModelGraphReadinessReport::validate_runtime_stage_dispatch_bindings(bindings)`
groups dispatch-binding receipts by model-declared stage. Each stage dispatch
binding row reports:

- stage index, name, kind, and required resource slots
- ordered dispatch-binding rows for that stage
- missing dispatch-binding rows, if any
- scoped slot-binding validation for the stage resource slots
- unstaged dispatch-binding rows that cannot be assigned to a stage

The same receipt is included in
`ModelRuntimeMetadataAdmissionReport::stage_dispatch_bindings`, so metadata
admission fails if a future stage packet builder would see an unbound resource
slot or an unstaged dispatch. This is still metadata validation only. It does
not allocate buffers, prove handles are GPU pointers, bind addresses, translate
scalar metadata into kernargs, build packet layouts, submit AQL, or execute
kernels.

## Runtime Stage Launch-Candidate Manifest

`ModelRuntimeMetadataAdmissionReport::runtime_stage_launch_candidate_plan()`
turns an admitted metadata report into a deterministic per-stage launch
candidate manifest. Each stage row reports:

- stage index, name, kind, and required resource slots
- bound logical resource handles for those slots
- ordered op names and ordered dispatch candidates
- dispatch entrypoint symbols, named slot arguments, scalar arguments, and
  bound read/write handles
- a whole-plan dispatch count

The method first requires metadata admission to pass, then checks that every
stage-dispatch binding row is complete and that every non-gap dispatch row has
at least one entrypoint symbol. This is still a CPU-side candidate receipt. It
does not allocate buffers, prove logical handles are device pointers, serialize
kernargs, choose packet fusion, create AQL packets, ring doorbells, synchronize
queues, execute kernels, or claim serving performance.

## Runtime Launch Entrypoint Provenance Manifest

`ModelRuntimeStageLaunchCandidatePlan::launch_entrypoint_provenance_plan()`
turns each launch candidate's route entrypoints into an explicit host-launcher
provenance manifest. Each dispatch row reports:

- op index, op name, primitive kind, stage name, and stage kind
- lowering route substrate
- qualified host launcher names and their bare host symbols
- a provenance kind of `HostLauncher`

The plan rejects mutated candidates whose route-derived host launcher symbols no
longer match the candidate entrypoint-symbol list. These are host launcher
identifiers such as `GpuDevice::arm_gemv`, not validated code-object kernel
symbols. This manifest does not inspect the embedded code object, prove kernel
ABI compatibility, serialize kernargs, create AQL packets, submit work, execute
kernels, or claim performance.

## Runtime Launch Kernel-Requirement Manifest

`ModelRuntimeStageLaunchCandidatePlan::launch_kernel_requirement_plan()` maps
known host launcher entrypoints to the conservative set of code-object kernel
symbols those launchers may request. Each dispatch row reports:

- op index, op name, primitive kind, stage name, and stage kind
- host launcher requirements and their mapped code-object kernel symbols
- deduplicated required kernel symbols for the dispatch, stage, and whole plan
- host launchers that do not yet have an explicit kernel-symbol mapping

`ModelRuntimeLaunchKernelRequirementPlan::validate_mapped_code_object_kernels(info)`
uses `CodeObjectInfo::validate_required_kernels(...)` to check that the
inspected code object exposes every mapped required kernel name.
`validate_code_object(info)` adds the stricter gate that every host launcher in
the plan must first have an explicit mapping. This is a CPU-only kernel presence
preflight. It does not prove kernel ABI argument order, select a runtime
specialization for dynamic dimensions, serialize kernargs, bind device pointers,
create AQL packets, submit work, execute kernels, or claim performance.
`CodeObjectInfo`, `CodeObjectKernelSetValidation`, `KernelInfo`,
`RuntimeLaunchKernelArgumentAbiSemanticEncoding`, `MAINARCH_KERNELS_GFX950`, and
`AQL_PACKET_BYTES` are re-exported through
`mainarch_core::model_api::prelude::*` so external packages can run these static
launch-readiness checks without importing runtime internals.
The aggregate `AllReduce::all_reduce_sum` selector is mapped to the conservative
union of kernels reachable through its size- and environment-selected branches;
the manifest still does not choose which branch will run for a specific payload.

## Runtime Launch Kernel-Metadata Manifest

`ModelRuntimeLaunchKernelRequirementPlan::kernel_metadata_plan(info)` validates
that every host launcher is explicitly mapped, checks that the inspected code
object exposes every required kernel, and materializes CPU-readable descriptor
metadata for each required kernel. Each kernel metadata row reports:

- code-object kernel symbol
- kernel descriptor virtual address inside the code object
- group/private segment sizes
- kernarg size and kernarg segment alignment
- wavefront size and maximum flat workgroup size from AMDHSA metadata

The plan also groups this metadata by dispatch and stage so a future packet
builder can inspect the required descriptor metadata before constructing
kernargs or AQL packets. It is still a CPU-only descriptor metadata handoff. It
does not prove named model arguments match the kernel ABI, allocate kernarg
memory, bind device pointers, create AQL packets, submit work, execute kernels,
or claim performance.

## Runtime Launch Code-Object Load Request Plan

`ModelRuntimeLaunchKernelMetadataPlan::code_object_load_request_plan()` consumes
the kernel-metadata manifest and turns it into explicit runtime work for a future
loader. The plan requests one code-object load and one kernel descriptor binding
request per required kernel, carrying descriptor metadata such as symbol,
descriptor virtual address, segment sizes, kernarg size/alignment, wavefront
size, and maximum flat workgroup size.

The current plan deliberately keeps `loaded_code_object_count=0`,
`code_object_base_bound=false`, `kernel_descriptor_bound_count=0`, and
`all_kernel_descriptors_bound=false`. Execution readiness exposes the same
counters so code-object loading and descriptor binding can be audited separately
from CPU-side kernel metadata inspection.

This is not a loader. It does not load GPU code objects, bind kernel descriptors,
patch AQL `kernel_object` fields, reserve queue slots, submit work, execute
kernels, or claim performance.

## Runtime Launch Code-Object Base Binding Request Plan

`ModelRuntimeLaunchCodeObjectLoadRequestPlan::code_object_base_binding_request_plan(live_relocation_bindings)`
joins the code-object load request plan with the AQL live relocation binding
plan. It requests one loaded code-object base, preserves the per-kernel
descriptor binding requests, and exposes one AQL `kernel_object` relocation
request per dispatch.

The current plan deliberately keeps `loaded_code_object_count=0`,
`loaded_code_object_base_bound_count=0`, `kernel_descriptor_bound_count=0`, and
`aql_kernel_object_relocation_bound_count=0`. Execution readiness reports those
loader-side and AQL-side counters independently so future runtime code can
separate loading the object, binding descriptors, and patching packet
`kernel_object` fields.

This is not a loader or relocator. It does not load GPU code objects, bind a
loaded base address, bind kernel descriptors, patch AQL packet bytes, reserve
queue slots, submit work, execute kernels, or claim performance.

## Runtime Launch Preflight Report

`ModelRuntimeMetadataAdmissionReport::runtime_launch_preflight_report(info,
max_dispatches)` composes the admitted metadata, stage launch candidates, launch
windows, entrypoint provenance, kernel requirements, code-object kernel
metadata, and launch arguments into one CPU-side handoff for a future packet
builder. `assert_ready()` checks count consistency across those subreports and
rejects any unmapped host launcher before the report is accepted.

This is still a metadata preflight. It does not serialize kernargs, prove
argument order against code-object metadata, bind device pointers, create AQL
packets, submit work, execute kernels, or claim performance.

## Runtime Launch AQL Packet-Field Handoff

`ModelRuntimeLaunchPreflightReport::aql_packet_field_plan()` joins the composed
preflight's per-dispatch launch arguments with each required code-object kernel
candidate. For every dispatch it reports:

- op index, op name, primitive kind, stage name, and stage kind
- selected entrypoint symbols and named launch arguments
- required kernel symbols and code-object-derived kernel candidates
- AQL packet fields already known from code-object metadata:
  `kernarg_size`, `private_segment_size`, and `group_segment_size`
- live runtime fields still unresolved: loaded `kernel_object`, `kernarg_va`,
  grid/workgroup geometry, and completion-signal binding

The rows are kernel candidates because some host launchers map to a conservative
kernel superset and still choose a concrete branch at runtime. Plan-level counts
therefore distinguish unique required kernel symbols from per-dispatch kernel
candidate rows. This handoff does not load the code object into GPU memory,
allocate kernarg buffers, prove code-object kernel argument ABI compatibility, build packet
bytes, reserve queue slots, ring a doorbell, submit AQL, execute kernels, or
claim performance.

## Runtime Launch Kernel-Selection Readiness Report

`ModelRuntimeLaunchPreflightReport::kernel_selection_readiness_plan()` consumes
the AQL packet-field handoff and classifies each dispatch's code-object kernel
candidate set. A dispatch is marked `SelectedSingleCandidate` only when the
metadata contains exactly one kernel candidate. Dispatches with multiple
candidates are marked `AmbiguousCandidateSet`, and missing candidate metadata is
represented separately.

The report aggregates selected, ambiguous, and missing dispatch counts by stage
and plan, keeps selected kernel metadata for singleton dispatches, and carries
unresolved candidate symbols for dispatches that still need runtime policy. Its
`assert_all_selected()` method is a future executable-launch gate: it succeeds
only when every dispatch has a single selected kernel.

This is still selection readiness, not executable lowering. It does not choose
between conservative host-launcher branches, prove a multi-kernel primitive
sequence, load kernel objects, allocate kernargs, serialize arguments, create
AQL packets, submit work, execute kernels, or claim performance.

## Runtime Launch Host-Launcher Branch Resolution Request Plan

`ModelRuntimeLaunchKernelSelectionReadinessPlan::host_launcher_branch_resolution_request_plan()`
turns ambiguous kernel-selection rows into an explicit host-launcher branch
resolution request manifest. It emits one request per ambiguous dispatch and
reports the unique unresolved candidate-symbol count carried by those requests.

The current plan deliberately keeps `branch_resolution_applied_count=0` and
`all_branches_resolved=false`; it is a CPU-side work manifest for a later
runtime policy, not an applied host-launcher decision. Execution readiness
reports the request, applied, unresolved-candidate, and ready counters while
keeping `host_launcher_runtime_branch_resolution` as an explicit blocker.
`branch_resolution_request_op_names()` exposes the launch-order operation names
behind the request count so downstream tooling can see which dispatches need
policy input without walking every candidate row.
`branch_resolution_request_candidate_symbol_sets()` exposes delimiter-safe
launch-order `(op_name, candidate_symbols)` pairs behind each request,
preserving the per-dispatch branch alternatives for CPU-side audit and external
runtime policy handoff without forcing consumers to parse text labels.
`branch_resolution_request_candidate_symbol_labels()` exposes the same sets as
display-oriented `op=symbol|symbol` labels for fixture output. Consumers that
need delimiter-safe structured data should use the structured helper or read the
request entries' `dispatches` and `candidate_symbols` directly.
`unresolved_candidate_symbols()` exposes the sorted unique kernel symbols behind
the unresolved-candidate count so tooling can identify the aggregate branch
policy surface without reconstructing it from each request entry.

This does not choose among conservative host-launcher branches, prove a
multi-kernel primitive sequence, select kernels for ambiguous dispatches, patch
AQL packets, submit work, execute kernels, or claim performance.

## Runtime Launch Kernel-Candidate Recommendation Plan

`ModelRuntimeLaunchKernelArgumentAbiVerificationPlan::kernel_candidate_recommendation_plan()`
uses a fixed CPU-only policy,
`first_verified_candidate_in_host_launcher_order`, to recommend one candidate
kernel per dispatch when at least one candidate passes named ABI schema
verification. A recommended candidate therefore has a matching named
descriptor schema and enough `kernarg_size` for the serialized Mainarch kernarg
bytes.

The plan reports source candidate counts, source ambiguous dispatch counts,
verified candidate counts, recommended dispatch counts, missing recommendation
counts, and `selection_applied_count`. Today `selection_applied_count` is
expected to remain zero: the recommendation is an auditable handoff for a
future runtime policy, not an applied launch decision.

This does not remove the `kernel_candidate_selection_policy` or
`host_launcher_runtime_branch_resolution` blockers. It does not prove semantic
argument order/types beyond the named descriptor schema, select a multi-kernel
primitive sequence, patch AQL packets, submit work, execute kernels, or claim
performance.

## Runtime Launch Kernel-Candidate Selection Request Plan

`ModelRuntimeLaunchKernelCandidateRecommendationPlan::kernel_candidate_selection_request_plan()`
turns recommendation rows into an explicit request manifest for a future runtime
selection policy. Dispatches with a verified recommendation carry the requested
kernel symbol, candidate index, kernarg size, and verified-candidate flag.
Dispatches without a verified recommendation remain counted as missing
selection requests.

The current plan deliberately keeps `selection_applied_count=0`.
`all_selection_requests_ready` is true only when no dispatch is missing a
selection request, and execution readiness exposes request, missing, and applied
counts separately from the recommendation counts.
`selection_request_op_names()`,
`selection_request_op_kernel_symbols()`,
`selection_request_op_kernel_symbol_labels()`, and
`missing_selection_request_op_names()` expose the launch-order operation names
and ready kernel-symbol request bindings behind those counts without applying a
policy. The tuple helper is delimiter-safe structured metadata; the label helper
is display-oriented fixture output.

This is not an applied kernel-selection policy. It does not choose ambiguous
host-launcher branches, bind a selected kernel into AQL templates, prove
semantic kernel ABI compatibility beyond the named descriptor schema,
materialize packets, submit work, execute kernels, or claim performance.

## Runtime Launch Argument-Binding Manifest

`ModelRuntimeStageLaunchCandidatePlan::launch_argument_plan()` joins each
launch candidate's named slot arguments with its bound logical read/write
handles, then appends the candidate's scalar arguments in declaration order.
Each dispatch argument row reports:

- op index, op name, primitive kind, stage name, and stage kind
- selected entrypoint symbols
- ordered argument names
- slot arguments resolved to logical handles, tensor metadata, and read/write
  argument access
- scalar argument values

The plan rejects duplicate argument names within one dispatch and rejects slot
arguments that cannot be resolved to the access-specific read or write handle.
This is a packet-builder input manifest, not a compiled kernel ABI. It does not
serialize kernargs, prove argument order against code-object metadata, bind
device pointers, create AQL packets, submit work, or execute kernels.

## Runtime Launch Device-Argument Manifest

`ModelRuntimeLaunchPreflightReport::device_argument_plan(device_pointers)` joins
the ordered launch-argument manifest with a completed device-pointer validation
receipt. Each slot argument is converted from a logical handle into a
`RuntimeLaunchDevicePointerArgument` carrying:

- slot index and tensor metadata from the logical launch handle
- requested read/write argument access
- the logical handle string used during metadata admission
- validated byte capacity, access mode, and device virtual address

Scalar arguments remain typed `RuntimeDispatchScalarValue` entries in their
existing declaration order. The plan validates count consistency and rejects
missing, mismatched, undersized, under-permissioned, null, or unaligned pointer
bindings before accepting the manifest.

This is still a CPU-side argument-binding handoff. It does not serialize
kernarg bytes, prove kernel ABI offsets or order, allocate kernarg memory, prove
that device VAs are live KFD registrations, build AQL packets, submit work,
execute kernels, or claim performance.

## Runtime Launch Kernarg Layout Plan

`ModelRuntimeLaunchPreflightReport::kernarg_layout_plan(device_pointers)` assigns
canonical Mainarch kernarg offsets to the bound launch arguments. Pointer
arguments are represented as 64-bit device virtual addresses. Scalar arguments
use fixed encodings such as `UsizeU64`, `F32BitsU32`, enum `U32`, and a compact
`LinearParallelism` tag/shard pair. The plan reports per-dispatch argument
offsets, sizes, alignments, payload bytes, span bytes including padding, and
remaining capacity or candidate capacity shortfall against the conservative
staging-layout kernarg region.

Plan-level capacity shortfall is the sum of per-dispatch shortfalls, because
alignment padding between staged dispatch regions is not reusable inside a
single dispatch's kernarg record.

This resolves Mainarch's CPU-side argument layout, not code-object ABI
verification. It does not serialize kernarg bytes, allocate kernarg memory, prove
that a particular kernel symbol consumes this exact ABI, materialize AQL packets,
submit work, execute kernels, or claim performance.

## Runtime Launch Kernarg Serialization Plan

`ModelRuntimeLaunchPreflightReport::kernarg_serialization_plan(device_pointers)`
serializes the canonical kernarg layout into deterministic CPU-side byte images.
Each pointer argument is encoded as a little-endian 64-bit device virtual
address. `usize` scalars are encoded as little-endian `u64`, `f32` bit patterns
and enum tags as little-endian `u32`, and `LinearParallelism` as an explicit
`u32` tag plus `u32` shard count.

The plan reports per-dispatch byte images sized to the canonical argument span,
the serialized byte count, per-argument byte slices, and the same candidate
capacity shortfall surfaced by the layout plan. Serialized images are allowed to
be larger than a candidate kernel's current `kernarg_size`; that remains a
code-object ABI verification failure, not a serialization failure.

This resolves CPU-side byte serialization only. It does not allocate GPU-visible
kernarg memory, copy these bytes into an allocation, prove code-object ABI
compatibility, materialize AQL packets, reserve queue slots, bind completion
signals, submit work, execute kernels, or claim performance.

## Runtime Launch Kernarg Allocation Request Plan

`ModelRuntimeLaunchKernargSerializationPlan::kernarg_allocation_request_plan()`
consumes the serialized kernarg byte images and turns them into an explicit
runtime allocation/copy request manifest. The plan requests one contiguous
GPU-visible kernarg backing allocation sized to the staged kernarg region and
one copy request per dispatch.

The current plan deliberately keeps `backing_allocation_bound_count=0`,
`backing_allocation_bound_bytes=0`, `dispatch_copy_applied_count=0`,
`dispatch_copy_applied_bytes=0`, `device_va_bound_dispatch_count=0`, and
`all_kernargs_allocated=false`. Execution readiness exposes those counters so
kernarg memory allocation and copy application can be audited independently from
CPU-side serialization.

This is not an allocator. It does not allocate GPU-visible memory, bind kernarg
device virtual addresses, copy bytes into an allocation, patch AQL packet
`kernarg_address` fields, reserve queue slots, submit work, execute kernels, or
claim performance.

## Runtime Launch Kernel-Argument ABI Verification Preflight Plan

`ModelRuntimeLaunchPreflightReport::kernel_argument_abi_verification_plan(device_pointers)`
consumes the AQL packet-template plan and audits each candidate kernel's
code-object `kernarg_size` and `kernarg_segment_align` against Mainarch's static
named kernarg descriptor schema registry. It reports candidate counts,
size-compatible candidate counts, size-shortfall candidate counts, total/max
shortfall bytes, schema availability, descriptor-match metadata, verified
candidate counts, and how many dispatches do or do not have at least one
verified candidate.

A candidate is `verification_ready` only when a named schema exists for its
kernel symbol, the candidate descriptor size/alignment match the schema, and the
serialized Mainarch kernarg bytes fit inside the candidate descriptor
`kernarg_size`. `abi_verification_ready` is true only when every candidate row is
verified. Partial verification keeps `kernel_argument_abi_verification` as an
explicit executable-readiness blocker.

The registry is inspectable through
`runtime_launch_kernel_argument_abi_schema_for(symbol)` and
`runtime_launch_kernel_argument_abi_schema_count()`.

The plan does not infer argument names from byte offsets, prove that a specific
kernel consumes Mainarch's canonical ABI, select ambiguous kernels, allocate or
copy kernarg memory, materialize AQL packets, submit work, execute kernels, or
claim performance.

`ModelRuntimeLaunchKernelArgumentAbiVerificationPlan::size_compatibility_receipt()`
turns the preflight rows into a first-class receipt for the part that is actually
checked today: named descriptor size/alignment metadata matches the candidate
kernel and serialized Mainarch kernarg bytes fit inside each candidate
descriptor. It reports dispatches with and without a size-compatible candidate
and dispatches with and without a verified candidate, while preserving candidate,
shortfall, named-schema, and verified-candidate totals.
This receipt is descriptor metadata verification, not a full kernel ABI proof.
`ModelRuntimeLaunchAqlPacketTemplatePlan::kernel_argument_abi_size_compatibility_receipt()`
provides the same receipt directly from an AQL packet-template plan.
`ModelRuntimeLaunchPreflightReport::kernel_argument_abi_size_compatibility_receipt(device_pointers)`
provides the same receipt directly from a launch preflight report.
`ModelRuntimeMetadataAdmissionReport::runtime_launch_kernel_argument_abi_size_compatibility_receipt(...)`
and
`ModelGraphReadinessReport::runtime_launch_kernel_argument_abi_size_compatibility_receipt(...)`
provide the same receipt as one-call convenience paths for integrations that
start from the admission or composed readiness surfaces. The CPU-only reference
example and CLI selftest exercise the readiness-level path; the custom-plugin
example exercises the admission-level path.

`ModelRuntimeLaunchKernelArgumentAbiVerificationPlan::kernel_argument_abi_verification_gap_report()`
filters the ABI preflight down to dispatches that still lack any verified
candidate. The report keeps source dispatch/candidate totals, per-gap dispatch
candidate symbols, named-schema availability, descriptor-match counts,
size-compatible versus shortfall counts, shortfall-byte totals, and
candidate-level rows with a deterministic primary failure reason:
`missing_named_abi_schema`, `named_abi_descriptor_mismatch`,
`kernarg_size_shortfall`, or `unknown_unverified_candidate`. Primary reason
counts are an explanation index; the raw schema, descriptor, and size counters
remain available because one candidate can fail more than one verification
predicate. The report is the dispatch-centric companion to the kernel-centric
schema request manifest.
`ModelRuntimeLaunchAqlPacketTemplatePlan`,
`ModelRuntimeLaunchPreflightReport`, `ModelRuntimeMetadataAdmissionReport`, and
`ModelGraphReadinessReport` expose matching convenience methods for callers that
start from those layers. The report is diagnostic only; it does not bind schemas,
select kernels, patch AQL packets, submit work, execute kernels, or claim
performance.

## Runtime Launch Kernel-Argument ABI Capacity Request Plan

`ModelRuntimeLaunchKernelArgumentAbiVerificationGapReport::kernel_argument_abi_capacity_request_plan()`
turns primary `kernarg_size_shortfall` candidate rows into a kernel-centric
capacity request manifest. It groups by unique kernel symbol, preserves
descriptor size/alignment and named-schema metadata, and reports the required
`kernarg_size` as the maximum serialized Mainarch kernarg byte count observed
for that symbol. The plan also keeps dispatch-reference counts, candidate
request counts, total/max shortfall bytes, the source primary-size-shortfall
count, and the inherited unresolved runtime requirements.

`ModelRuntimeLaunchKernelArgumentAbiVerificationPlan`,
`ModelRuntimeLaunchAqlPacketTemplatePlan`, `ModelRuntimeLaunchPreflightReport`,
`ModelRuntimeMetadataAdmissionReport`, and `ModelGraphReadinessReport` expose
matching convenience methods so integrations can obtain the same manifest from
the ABI plan, AQL template plan, preflight, admission, or readiness surface.

For the reference model and CLI selftest, the capacity request plan reports 13
unique kernel requests covering 59 primary size-shortfall candidate rows, with a
maximum shortfall of 52 bytes and total shortfall of 1340 bytes. The custom
plugin example reports 4 unique requests covering 4 primary size-shortfall
candidate rows, with a maximum shortfall of 24 bytes and total shortfall of 72
bytes.

This is a CPU-side work manifest for future kernel/code-object ABI updates. It
does not rewrite code-object descriptors, bind schemas, select kernels, patch
AQL packets, submit work, execute kernels, or claim that launch execution is
ready.

## Runtime Launch Kernel-Argument ABI Schema Request Plan

`ModelRuntimeLaunchKernelArgumentAbiVerificationPlan::kernel_argument_abi_schema_request_plan()`
turns the ABI verification preflight into an explicit named-schema request
manifest. It requests one named ABI schema per unique kernel symbol and one
candidate verification request per candidate ABI row, preserving the code-object
target and unresolved runtime requirements from the ABI preflight.

The static registry binds schema rows for current host-launcher kernel symbols.
For the reference model, the schema request plan reports all 44 unique requested
kernel schemas bound and 78 of 147 candidate rows verified; the remaining rows
still fail because the serialized argument payload is larger than those
candidates' descriptor `kernarg_size`. The ABI preflight and receipt also
report 16 dispatches with a verified candidate and 20 without one. The ABI gap
report narrows those 20 dispatches to 59 candidate rows, all with named schemas
bound and descriptor matches, and all primarily blocked by
`kernarg_size_shortfall`. Execution readiness reports schema-bound and
verification-applied counts separately from the remaining pending runtime work,
and its ABI blocker detail includes the verified dispatch coverage.

This is not full semantic ABI verification. It does not infer argument names,
prove argument order or types beyond descriptor size/alignment, select ambiguous
kernels, materialize AQL packets, submit work, execute kernels, or claim
performance.

## Runtime Launch Kernel-Argument ABI Semantic Plan

`ModelRuntimeLaunchAqlPacketTemplatePlan::kernel_argument_abi_semantic_plan(&serialization)`
compares AQL candidate rows against serialized Mainarch kernarg rows using a
static semantic schema registry for covered host-launcher kernel symbols. The
same comparison is available from
`ModelRuntimeLaunchKernargSerializationPlan::kernel_argument_abi_semantic_plan(&templates)`,
`ModelRuntimeLaunchPreflightReport::kernel_argument_abi_semantic_plan(device_pointers)`,
`ModelRuntimeMetadataAdmissionReport::runtime_launch_kernel_argument_abi_semantic_plan(...)`,
and
`ModelGraphReadinessReport::runtime_launch_kernel_argument_abi_semantic_plan(...)`.

The semantic registry is inspectable through
`runtime_launch_kernel_argument_abi_semantic_schema_for(symbol)` and
`runtime_launch_kernel_argument_abi_semantic_schema_count()`. Each schema row
names the kernel symbol, descriptor size/alignment, and ordered fields with the
kernel argument name, Mainarch model argument alias, argument kind, expected
encoding, offset, and size. Encodings distinguish compact kernel-side `u32`
parameters from Mainarch's current generic `usize_u64` scalar serialization.

The plan reports, per candidate, whether a semantic schema is available, whether
the descriptor matches that schema, whether the serialized Mainarch kernarg fits
the candidate descriptor, verified/missing/mismatched/extra field counts,
extra model argument names, per-field check rows, and a deterministic primary
gap reason:
`missing_semantic_schema`, `semantic_descriptor_mismatch`,
`missing_model_argument`, `field_shape_mismatch`, `extra_model_argument`,
`kernarg_size_shortfall`, or `unknown_unverified_semantic`. Aggregate rows
summarize schema coverage, descriptor matches, semantic-verified candidate
counts, field counts, dispatches with and without any semantic-verified
candidate, and whether every candidate is semantically verified.

`ModelRuntimeLaunchKernelArgumentAbiSemanticPlan::kernel_argument_abi_semantic_gap_report()`
derives a dispatch-centric gap report from the semantic plan. The report keeps
only dispatches that lack any semantic-verified candidate, preserves each
candidate semantic row, and aggregates source dispatch/candidate totals, gap
candidate totals, schema coverage, descriptor matches, verified field counts,
missing field counts, mismatched field counts, extra argument counts, and
primary semantic gap reasons. Convenience methods expose the same report from
`ModelRuntimeLaunchAqlPacketTemplatePlan::kernel_argument_abi_semantic_gap_report(&serialization)`,
`ModelRuntimeLaunchKernargSerializationPlan::kernel_argument_abi_semantic_gap_report(&templates)`,
`ModelRuntimeLaunchPreflightReport::kernel_argument_abi_semantic_gap_report(device_pointers)`,
`ModelRuntimeMetadataAdmissionReport::runtime_launch_kernel_argument_abi_semantic_gap_report(...)`,
and
`ModelGraphReadinessReport::runtime_launch_kernel_argument_abi_semantic_gap_report(...)`.
`all_dispatches_have_semantic_verified_candidate` is true exactly when the gap
report is empty.
`ModelRuntimeLaunchKernelArgumentAbiSemanticGapReport::missing_model_argument_requirements()`
returns structured rows for each missing model-side semantic field, including
dispatch identity, candidate kernel symbol, kernel argument name, model argument
alias, expected kind/encoding, offset, and size. This is the author-facing form
of the de-duplicated `missing_model_argument_names()` list and preserves
dispatch/candidate ordering for diagnostics.
`ModelRuntimeLaunchKernelArgumentAbiSemanticGapReport::field_mismatch_diagnostics()`
returns structured rows for schema fields whose model argument exists but does
not match the expected semantic ABI shape. Rows include the same dispatch and
candidate identity plus expected kind/encoding/offset/size, actual
kind/encoding/offset/size, and the individual match booleans so authors can
distinguish pointer/scalar, encoding, layout, and size drift without reverse
walking the full candidate table.

`ModelRuntimeLaunchAqlPacketTemplatePlan::kernel_argument_abi_semantic_projection_plan(&serialization)`
uses the same static semantic schema registry to build CPU-only,
candidate-specific projected kernarg byte images for descriptor-matching
semantic schemas. It lays out fields at kernel ABI offsets, narrows canonical
Mainarch `usize_u64` scalar arguments into kernel `u32` fields when the value
fits, copies pointer and exact-width scalar fields, and reports missing model
arguments, kind mismatches, unsupported encodings, scalar narrowing overflow,
and schema field range overflow as explicit blockers. Extra Mainarch arguments
are ignored by projection because the projected byte image is schema-shaped.
The same projection plan is available from
`ModelRuntimeLaunchKernargSerializationPlan::kernel_argument_abi_semantic_projection_plan(&templates)`,
`ModelRuntimeLaunchPreflightReport::kernel_argument_abi_semantic_projection_plan(device_pointers)`,
`ModelRuntimeMetadataAdmissionReport::runtime_launch_kernel_argument_abi_semantic_projection_plan(...)`,
and
`ModelGraphReadinessReport::runtime_launch_kernel_argument_abi_semantic_projection_plan(...)`.
This projection is still CPU-side metadata; it is not an allocation, copy, AQL
patch, submission path, or execution proof.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionPlan::kernel_argument_abi_semantic_projection_gap_report()`
derives a dispatch-centric gap report from the projection plan. It keeps only
dispatches that lack any projection-ready candidate, preserves the candidate
projection rows, and aggregates candidate counts, schema coverage, descriptor
matches, primary projection blockers, field-status counts, and projected byte
coverage. The same gap report is available from AQL template, kernarg
serialization, preflight, admission, and readiness paths.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionPlan::kernel_argument_abi_semantic_projection_candidate_recommendation_plan()`
uses the CPU-only policy
`first_projection_ready_candidate_in_host_launcher_order` to recommend the first
projection-ready candidate for each dispatch, when such a candidate exists. It
reports source candidate counts, schema coverage, projection-ready candidate
counts, source ambiguity, recommended/missing dispatch counts, projected
kernarg bytes for recommended rows, and `selection_applied_count=0`. This is a
projection-aware handoff surface; it does not replace the existing named-ABI
recommendation policy or apply a kernel selection.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationPlan::kernel_argument_abi_semantic_projection_candidate_selection_request_plan(&projection)`
turns those projection-aware recommendations into an explicit request manifest
for a future runtime selection policy. Ready rows carry the requested kernel
symbol, candidate index, kernarg size, projected kernarg byte count, and the
schema-shaped projected byte image. Missing rows remain empty and counted.
The builder derives counts and blockers from the source projection rows, and
`assert_consistent_with_projection(&projection)` checks the requested projected
byte image against the selected projection candidate. The plan keeps
`selection_applied_count=0` and is still CPU-side metadata; it does not bind
selected kernels into AQL templates, allocate or copy kernarg memory, submit
work, execute kernels, or claim performance.
`selection_request_op_names()`,
`selection_request_op_kernel_symbols()`,
`selection_request_op_kernel_symbol_labels()`, and
`missing_selection_request_op_names()` expose the launch-order operation names
and ready kernel-symbol request bindings so external plugins can audit the
manifest without walking every dispatch row. The tuple helper is delimiter-safe
structured metadata; the label helper is display-oriented fixture output.
`ModelRuntimeLaunchKernelCandidateRecommendationPlan::kernel_argument_abi_semantic_projection_recommendation_report(&projection)`
joins the named-ABI recommendation policy with the semantic projection plan. It
reports, per dispatch, whether the recommended kernel candidate is present in
the projection plan, whether that recommended candidate is projection-ready,
the primary projection blocker when it is not ready, and the ready projected
kernarg byte coverage. The same report is available from AQL template, kernarg
serialization, preflight, admission, and readiness paths. This is still an
auditable CPU-side join; it does not apply kernel selection, reorder
recommendations toward projection-ready candidates, patch AQL packets, submit
work, execute kernels, or claim performance.
`ModelRuntimeLaunchExecutionReadinessReport` mirrors the same projection
aggregates under `kernel_argument_abi_semantic_projection_*` fields and carries
`kernel_argument_abi_semantic_projection` in `blockers` and
`unresolved_runtime_requirements` until every candidate row has a
projection-ready schema-shaped byte image.

For the current reference MoE graph and CLI selftest, the semantic plan reports
49 static semantic schemas, 147 schema-covered candidate rows, 0 missing-schema
candidate rows, 147 descriptor matches, 0 semantic-verified candidates, 36
dispatches without a semantic-verified candidate, 1028 field-schema rows, 83
verified fields, 494 missing fields, 451 field mismatches, and 479 extra argument
instances. The custom plugin example reports 49 schemas, 8 schema-covered
candidate rows, 0 missing-schema candidate rows, 8 descriptor matches, 0
semantic-verified candidates, 3 dispatches without a semantic-verified
candidate, 51 field-schema rows, 8 verified fields, 20 missing fields, 23 field
mismatches, and 15 extra argument instances.

The derived gap report currently records 36 reference/CLI dispatch gaps across
147 candidate ABI rows: 147 schema-covered rows, 0 missing-schema rows, 147
descriptor matches, 0 semantic-verified candidates, 0 primary
missing-semantic-schema gaps, 114 primary missing-model-argument gaps, and 33
primary field-shape-mismatch gaps. The custom plugin example records 3 dispatch
gaps across 8 candidate ABI rows: 8 schema-covered rows, 0 missing-schema rows,
8 descriptor matches, 0 semantic-verified candidates, 0 primary
missing-semantic-schema gaps, 6 primary missing-model-argument gaps, and 2
primary field-shape-mismatch gaps.
`ModelRuntimeLaunchKernelArgumentAbiSemanticGapReport::missing_semantic_schema_kernel_symbols()`
returns the stable, de-duplicated missing semantic schema symbol list. The
reference/CLI and custom plugin lists are currently empty.
`ModelRuntimeLaunchKernelArgumentAbiSemanticGapReport::missing_model_argument_names()`
returns stable, de-duplicated model-side argument names still required by
covered semantic schemas. The reference/CLI semantic gap list currently has 63
names, while the custom plugin list has 13 names:
`base_pos`, `block_size`, `candidate_logits`, `candidate_token_ids`, `eps`,
`last_page_len`, `positions`, `rmsnorm_output`, `rmsnorm_weight`, `seq_lens`,
`slot`, `step`, and `token_ids`.

The semantic projection plan currently records 31 reference/CLI
projection-ready candidate ABI rows across 20 dispatches, 528 projected fields,
494 missing fields, 6 kind mismatches, 0 unsupported encodings, 0 scalar
narrowing overflows, 0 field range overflows, and 20284 projected kernarg bytes.
The custom plugin example records 2 projection-ready candidate ABI rows across 1
dispatch, 31 projected fields, 20 missing fields, and 340 projected kernarg
bytes. Both remain `ready=false` with 9 unresolved runtime requirements because
both paths still have schema fields without Mainarch model arguments. The
reference graph also keeps attention sequence length and RoPE position as
metadata pointers where
covered kernels currently require scalar sequence-length and position fields.
The fused allreduce plus residual RMSNorm schema also requires collective peer
pointer tables, grid-barrier and partial scratch buffers, and a workgroup count
that are not yet represented as model graph or runtime bindings.
The direct peer-reduce, scatter-to-staging, gather-reduce-local, reduce-scatter,
broadcast-chunk, broadcast-chunk-skip-owner, all-gather, one-shot allreduce,
dual-path allreduce, peer-broadcast, and P2P transfer schemas likewise require
local allreduce buffers, peer pointer tables, peer reduce-staging pointer
tables, reduce-staging buffers, local allreduce flags, peer allreduce flag
pointer tables, reduce-staging chunk strides, chunk offset/length, vec4 chunk
offset/length, P2P peer counts, peer vec4 strides, allreduce sequence bases,
tile counts, dual-path SDMA semaphores, dual-path CU split lengths, chunk owner,
self rank, or workgroup count that are not yet represented as branch-resolved
runtime bindings. The persistent allreduce schemas additionally require local
allreduce buffers, persistent DDA output buffers, peer allreduce pointer tables,
persistent allreduce control blocks, grid barriers, group size, and persistent
total-op counts that are not yet represented as branch-resolved runtime
bindings. Other covered kernels still report element-count model-argument gaps
where the graph does not supply `n`.
For one-shot allreduce specifically, the `n` element-count field is projected
where the graph already supplies `n`; the remaining one-shot collective pointer,
flag, barrier, rank/group, chunk, workgroup, sequence-base, and tile-count
fields stay visible as semantic projection blockers.
For persistent allreduce specifically, the `n` element-count fields are also
projected where the graph already supplies `n`; the remaining persistent
collective pointer, control, barrier, group-size, output-buffer, and total-op
fields stay visible as semantic projection blockers.
The derived projection gap report currently records 16 reference/CLI dispatch
gaps across 92 candidate ABI rows: 92 schema-covered rows, 0 missing-schema
rows, 92 descriptor matches, 0 projection-ready candidates, 0 primary
missing-semantic-schema blockers, 90 primary missing-model-argument blockers, 2
primary kind-mismatch blockers, 713 field-schema rows, 229 projected fields, 478
missing fields, 6 kind mismatches, and 17676 projected kernarg bytes. The custom
plugin example records 2 dispatch gaps across 4 candidate ABI rows: 4
schema-covered rows, 0 missing-schema rows, 4 descriptor matches, 0
projection-ready candidates, 0 primary missing-semantic-schema
blockers, 4 primary missing-model-argument blockers, 31 field-schema rows, 11
projected fields, 20 missing fields, and 204 projected kernarg bytes.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionGapReport::missing_semantic_schema_kernel_symbols()`
reports the projection-gap-only missing schema symbols. The reference/CLI and
custom plugin lists are currently empty.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionGapReport::missing_model_argument_names()`
reports the same kind of stable, de-duplicated names for projection-only gaps.
The reference/CLI projection gap list currently has 57 names; the custom plugin
list has the same 14 names as its semantic gap list. The standalone external
MoE plugin projection gap list has 17 names because its MoE dispatches add
`intermediate`, `n`, and `nthreads` to the custom list.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionGapReport::missing_model_argument_requirements()`
returns the projection-specific structured rows for fields whose projection
status is `missing_model_argument`, so plugin authors can identify which
candidate kernel and kernarg field still needs a model or runtime binding.
`ModelRuntimeLaunchKernelArgumentAbiSemanticProjectionGapReport::field_blocker_diagnostics()`
returns one row for every non-projected schema field in the projection gap
report. Each row carries dispatch and candidate identity, expected ABI shape,
available actual model-argument shape, projection status, and projected byte
metadata. It is the typed companion to the aggregate missing-field,
kind-mismatch, unsupported-encoding, narrowing-overflow, and range-overflow
counts.

The projection recommendation report currently records that the fixed
`first_verified_candidate_in_host_launcher_order` policy recommends 16
reference/CLI dispatches and 2 custom-plugin dispatches, but 0 recommended
dispatches are projection-ready. All recommended rows are present in the
projection plan and are blocked by semantic projection metadata or model
argument gaps: 16 blocked recommendation rows for the reference/CLI graph and 2
for the custom plugin. This separates "some candidate for a dispatch can be
projected" from "the candidate the recommendation policy would request can be
launched with a projected kernarg image."

The projection-aware candidate recommendation plan currently recommends 20
reference/CLI dispatches and leaves 16 without a projection-ready candidate.
Those recommended rows cover 656 recommended projected kernarg bytes; the
source projection plan has 31 projection-ready candidate rows. The custom plugin
example recommends 1 dispatch, leaves 2 without a projection-ready candidate,
and covers 32 recommended projected kernarg bytes while the source plan has 2
projection-ready candidate rows. Both plans keep `selection_applied_count=0`
and `all_recommended=false`.
The projection-aware candidate selection request plan mirrors those counts as
unapplied requests: 20 reference/CLI requests, 16 missing requests, 656
requested projected kernarg bytes, and `all_ready=false`; the custom plugin
example records 1 request, 2 missing requests, 32 requested projected kernarg
bytes, and `all_ready=false`. `request_plan_ready=true` only means the manifest
is well-formed for a future runtime policy. The standalone external MoE plugin
records ready operations `layers.0.router_topk,lm_head` and missing operations
`embed_tokens,layers.0.moe_local_ffn,layers.0.moe_residual,greedy_argmax`.
Its generic named-ABI kernel selection manifest records 4 ready operations
`embed_tokens,layers.0.router_topk,layers.0.moe_residual,greedy_argmax` and 2
missing operations `layers.0.moe_local_ffn,lm_head`.
Its host-launcher branch manifest records 4 requests for
`layers.0.router_topk,layers.0.moe_local_ffn,lm_head,greedy_argmax`.

`ModelRuntimeLaunchExecutionReadinessReport::unresolved_runtime_requirements`
keeps the ordered runtime blockers that still prevent launch submission.
`unresolved_runtime_requirement_names()` exposes that ordered list without
requiring callers to read the field directly, while
`blocker_requirement_names()` exposes the same order from the detailed blocker
rows. `has_blocker(...)` and `blocker_for(...)` let runtime code test or
retrieve a specific blocker row by requirement name. Use `blocker_for_step(...)`
when runtime code already has a typed `RuntimeLaunchExecutionRequestStep`; the
helper resolves the canonical step requirement before returning the blocker row.
The reference/CLI, custom plugin, and standalone external plugin examples
currently report the same 9 labels:
`kernel_candidate_selection_policy`,
`host_launcher_runtime_branch_resolution`, `loaded_code_object_base`,
`kernarg_allocation`, `kernel_argument_abi_verification`,
`kernel_argument_abi_semantic_projection`, `completion_signal_binding`,
`queue_reservation`, and `aql_packet_materialization`.

This is a metadata-only comparison and gap-report surface. It does not infer
schemas from code objects, rewrite kernel descriptors, choose candidates,
allocate or copy kernarg memory, patch AQL packets, submit work, execute
kernels, or claim that semantic ABI verification/projection is complete for
every dispatch/candidate.

## Runtime Launch Staging-Footprint Plan

`ModelRuntimeLaunchPreflightReport::staging_footprint_plan()` joins the composed
preflight's launch arguments, AQL packet-field handoff, and launch windows into a
CPU-only packet-builder sizing report. It reports:

- packet bytes per dispatch from `AQL_PACKET_BYTES`
- total packet bytes across the admitted dispatches
- dispatch-, stage-, and window-level packet byte totals
- each dispatch's maximum candidate `kernarg_size`
- conservative kernarg upper-bound bytes, using the maximum candidate kernarg
  size for each dispatch before allocator padding policy is chosen
- pointer/scalar launch argument counts
- maximum candidate kernarg segment alignment
- unresolved runtime requirements: kernel candidate selection, allocator
  alignment policy, kernarg layout serialization, and queue reservation

This plan is a staging footprint, not a runtime allocation or serialized launch
buffer. It does not allocate memory, reserve queue slots, choose one kernel
candidate, serialize kernarg bytes, materialize AQL packet bytes, submit work,
execute kernels, or claim performance.

## Runtime Launch Staging-Layout Plan

`ModelRuntimeLaunchPreflightReport::staging_layout_plan()` applies a deterministic
CPU-side layout policy to the staging-footprint plan. AQL packets are assigned
contiguous `AQL_PACKET_BYTES`-aligned offsets in a packet region. Kernargs are
assigned offsets in a separate kernarg region, with each dispatch aligned to its
maximum candidate kernarg segment alignment and sized by its conservative
kernarg upper bound.

The plan reports dispatch-, stage-, and window-level packet offsets, kernarg
offsets, kernarg spans, padded packet/kernarg region sizes, and total staging
bytes. It resolves the allocator alignment policy for CPU-side planning, but it
still does not allocate memory or write bytes.

This is a layout plan, not a staging allocation. It does not choose one kernel
candidate, serialize kernargs, materialize AQL packet bytes, reserve queue slots,
submit work, execute kernels, or claim performance.

## Runtime Launch Completion-Signal Policy Plan

`ModelRuntimeLaunchPreflightReport::completion_signal_plan()` applies a
deterministic CPU-side terminal-signal policy to launch windows. Each window gets
one logical terminal completion-signal slot, with a planned initial value of `1`
and completed value of `0`. The plan records the terminal dispatch for each
window and keeps the AQL packet `completion_signal` field unresolved until a
runtime binds actual HSA signal handles.

This resolves the policy choice, not the live signal object. It does not create
signals, allocate signal pools, write packet fields, reserve queue slots, submit
work, wait on signals, execute kernels, or claim performance.

## Runtime Launch Completion-Signal Binding Request Plan

`ModelRuntimeLaunchCompletionSignalPlan::completion_signal_binding_request_plan()`
turns the logical terminal-signal slots into explicit live-signal handle
requests. It requests one HSA completion-signal handle per logical terminal
signal slot and carries the planned initial/completed values plus the terminal
dispatch metadata for the window.

The current plan deliberately keeps `signal_handle_bound_count=0` and
`all_signal_handles_bound=false`. Execution readiness exposes those counters so
live signal creation and handle binding can be audited separately from the
CPU-side completion-signal policy and from AQL packet relocation binding.

This is not a signal allocator. It does not create HSA signals, bind signal
handles, patch AQL `completion_signal` fields, reserve queue slots, submit work,
wait on signals, execute kernels, or claim performance.

## Runtime Launch Queue-Slot Plan

`ModelRuntimeLaunchPreflightReport::queue_slot_plan()` assigns deterministic
logical AQL packet indices to each dispatch in launch-window order. It composes
the staging-layout plan with the completion-signal policy plan so the terminal
dispatch in each window carries the planned logical completion-signal slot.

The plan reports per-window first queue packet index, per-dispatch queue packet
index, window-local packet index, chain-barrier handoff, terminal completion
signal slot, queue packet count, and doorbell batch count. It is a queue layout
plan, not a live queue reservation.

This does not reserve AQL queue slots, write packets, bind completion signals,
ring a doorbell, submit work, execute kernels, or claim performance.

## Runtime Launch Queue-Reservation Request Plan

`ModelRuntimeLaunchQueueSlotPlan::queue_reservation_request_plan()` consumes the
logical queue-slot plan and reports the live queue work that a future runtime
must perform. It preserves the deterministic packet order while turning each
launch window into a reservation request with queue packet counts, packet byte
ranges, and one requested doorbell batch per window.

The current request plan deliberately keeps `queue_packet_reserved_count=0`,
`doorbell_batch_bound_count=0`, `reservation_applied_count=0`, and
`all_queue_packets_reserved=false`. Execution readiness exposes those counters
so queue reservation can be audited separately from logical packet indexing.

This is still not a queue reservation. It does not claim a live HSA queue slot,
bind a doorbell, copy packet bytes into a queue, ring a doorbell, submit work,
execute kernels, or claim performance.

## Runtime Launch Dispatch-Geometry Plan

`ModelRuntimeLaunchPreflightReport::dispatch_geometry_plan()` assigns
deterministic CPU-side AQL `dims`, `grid_*`, and `workgroup_*` fields for every
logical queue packet. It composes the queue-slot plan with launch-candidate
scalar metadata and uses a conservative one-dimensional policy:

- `workgroup_count_x = ceil(workload_items / 256)`
- `grid_size_x = workgroup_count_x * 256`
- `workgroup_size_x = 256`
- `grid_size_y = grid_size_z = workgroup_size_y = workgroup_size_z = 1`

`grid_size_x` is therefore the padded total AQL work-item count, not the number
of workgroups. The workload source is explicit per primitive. Examples include
`hidden` for embedding, `out_features` for linear output projection,
`normalized_dim` for RMS norms, `query_heads_x_head_dim` for paged attention,
`experts` for MoE routing, and max write-handle bytes for residual and
collective dispatches.

This resolves primitive-level packet geometry, not kernel-specific launch
tuning. It does not choose ambiguous host-launcher branches, write AQL packets,
reserve queues, submit work, execute kernels, or claim performance.

## Runtime Launch AQL Packet-Template Plan

`ModelRuntimeLaunchPreflightReport::aql_packet_template_plan(device_pointers)`
composes the AQL packet-field handoff, kernel-selection readiness report,
queue-slot plan, dispatch-geometry plan, and kernarg serialization plan into
non-dispatchable packet templates. Each template row records the logical packet
offset/index, deterministic geometry fields, kernarg staging offset/span,
serialized kernarg byte count, terminal logical completion-signal slot, and one
candidate template per possible kernel symbol. The plan keeps full staged
kernarg-region capacity separate from the sum of per-template kernarg regions so
allocator padding remains visible.

Candidate templates carry code-object descriptor metadata such as kernel symbol,
descriptor vaddr, `kernarg_size`, private/group segment sizes, wavefront size,
and whether the CPU-side serialized kernarg span fits that candidate's current
metadata. Ambiguous host launchers remain represented as multiple candidate
templates rather than being silently selected.

This resolves CPU-side packet shape only. It does not emit dispatchable AQL
packet bytes, add a loaded code-object base to `kernel_object`, allocate or copy
kernarg memory, bind live `completion_signal` handles, reserve queue slots, ring
a doorbell, submit work, execute kernels, or claim performance.

## Runtime Launch AQL Packet Relocation-Site Plan

`ModelRuntimeLaunchPreflightReport::aql_packet_relocation_plan(device_pointers)`
consumes the AQL packet-template plan and reports the standard 64-byte
`hsa_kernel_dispatch_packet_t` byte ranges each logical packet would eventually
write. It marks template-resolved fields such as setup, grid, workgroup, and
segment sizes, reserved-zero fields, and the three live relocation sites:

- `kernel_object` at byte offset 32, blocked on a loaded code-object base
- `kernarg_address` at byte offset 40, blocked on kernarg allocation
- `completion_signal` at byte offset 56, blocked on live signal binding

The plan validates full byte-range coverage from offset 0 through 63 for each
logical packet and reports total relocation-site counts. It still does not build
packet byte images, patch live virtual addresses, reserve queue slots, submit
work, execute kernels, or claim performance.

## Runtime Launch AQL Packet Byte-Template Plan

`ModelRuntimeLaunchPreflightReport::aql_packet_byte_template_plan(device_pointers)`
serializes each candidate packet template into a deterministic 64-byte CPU-side
byte image. Because ambiguous host launchers have not selected one kernel branch,
the plan emits one byte template per candidate kernel symbol. Resolved fields are
written in little-endian form, including header/setup, workgroup sizes, grid
sizes, and candidate-specific private/group segment sizes.

The live relocation fields remain zero-filled in every byte image:

- `kernel_object`
- `kernarg_address`
- `completion_signal`

These byte templates are therefore validation artifacts, not dispatchable AQL
packets. They do not apply a loaded code-object base, patch kernarg GPU VAs,
patch signal handles, reserve queue slots, copy bytes into a queue, ring a
doorbell, submit work, execute kernels, or claim performance.

## Runtime Launch AQL Packet Materialization Preflight Plan

`ModelRuntimeLaunchPreflightReport::aql_packet_materialization_plan(device_pointers)`
consumes the AQL packet byte-template plan and reports the remaining CPU-side
handoff needed before dispatchable packet bytes can exist. Single-candidate
dispatches carry their selected 64-byte template image forward. Ambiguous
dispatches report the number of candidate byte templates but intentionally do
not select one.

The plan reports selected and ambiguous dispatch counts, pending live relocation
site counts, live relocation byte counts, and dispatchable packet counts. Today
`dispatchable_packet_count` is expected to remain zero and
`packet_materialization_ready` is expected to remain false because the plan does
not patch `kernel_object`, `kernarg_address`, or `completion_signal`.

This is a materialization preflight report, not a packet submission path. It
does not choose ambiguous kernels, apply loaded code-object addresses, allocate
kernarg backing, bind signal handles, reserve queue slots, copy bytes into a
queue, ring a doorbell, submit work, execute kernels, or claim performance.

## Runtime Launch AQL Live Relocation Binding-Request Plan

`ModelRuntimeLaunchAqlPacketMaterializationPlan::aql_live_relocation_binding_plan()`
consumes the materialization preflight and turns each pending live relocation
site into an explicit runtime binding request. The plan reports request counts
for:

- loaded code-object base bindings for `kernel_object`
- kernarg allocation bindings for `kernarg_address`
- completion-signal bindings for `completion_signal`

The current plan is intentionally a request manifest only. It keeps
`bound_relocation_count=0`, `unbound_relocation_count` equal to the request
count, `dispatches_fully_bound_count=0`, and `all_relocations_bound=false`.
Execution readiness exposes the same counters so live address binding can be
audited independently from packet byte-template construction.

This plan does not choose ambiguous kernels, load code objects, allocate kernarg
backing, bind HSA signal handles, patch packet bytes, reserve queue slots, copy
bytes into a queue, ring a doorbell, submit work, execute kernels, or claim
performance.

## Runtime Launch Executable-Readiness Gate

`ModelRuntimeLaunchPreflightReport::execution_readiness_report(device_pointers)`
composes the packet-field handoff, kernel-selection readiness report,
device-argument manifest, staging-footprint/staging-layout plans,
completion-signal policy plan, queue-slot plan, queue-reservation request plan,
completion-signal binding request plan, code-object load and base binding request
plans,
dispatch-geometry plan, kernarg
layout/serialization and allocation request
plans, kernel-argument ABI schema and capacity request plans, AQL
packet-template, relocation-site, byte-template/materialization preflight, live
relocation binding-request plans, and launch-window counts into a single
CPU-side launch gate. It reports:

- whether the current metadata is executable
- selected, ambiguous, and missing kernel-candidate counts
- prepared kernel-candidate selection request, missing-request, and applied counts
- prepared host-launcher branch resolution request, applied, and unresolved
  candidate-symbol counts
- prepared code-object load and kernel descriptor binding request counts
- prepared code-object base binding and AQL `kernel_object` relocation request
  and bound counts
- prepared completion-signal handle request and bound-handle counts
- bound pointer/scalar launch argument counts
- packet bytes and conservative kernarg upper-bound bytes
- prepared kernarg allocation/copy request and applied-copy counts
- prepared kernel-argument ABI candidate totals, verified-dispatch coverage, and
  schema request/bound, verification request/applied, capacity request, and
  shortfall-byte counts
- prepared semantic projection candidate selection request, missing-request,
  requested projected kernarg byte, and applied counts
- prepared queue reservation request, reserved packet, and applied-window counts
- prepared CPU-side AQL packet-template, candidate-template, relocation-site, and
  candidate byte-template/materialization/live-binding request counts
- concrete runtime blockers such as kernel candidate selection, code-object
  loading, kernarg allocation, code-object ABI verification, completion signal
  binding, queue reservation, and AQL packet materialization

For the current reference MoE graph this gate is expected to report
`executable=false`: the model API has enough metadata to size and audit the
launch plan, but not enough runtime state to submit it.
`assert_executable()` is therefore a future runtime gate, not a claim that the
current model API launches kernels end-to-end.
`ModelRuntimeLaunchExecutionReadinessReport::is_non_executable_boundary()` and
`assert_non_executable_boundary()` provide the inverse release guard for today's
static metadata path: the report must remain blocked, non-dispatchable, and free
of live resource binding/application counts after passing the same report
consistency checks before it is treated as a CPU-only launch-readiness receipt.
`ModelRuntimeMetadataAdmissionReport::runtime_launch_execution_readiness_report(...)`
and `ModelGraphReadinessReport::runtime_launch_execution_readiness_report(...)`
return the same report from the admitted-model and readiness-level paths.

This report does not choose ambiguous kernel branches, load kernel objects,
allocate memory, prove code-object ABI compatibility, bind HSA signals, serialize
kernargs, create AQL packets, reserve queue slots, submit work, execute kernels,
or claim performance.

## Runtime Launch Execution Request Plan

`ModelRuntimeLaunchPreflightReport::execution_request_plan(device_pointers)`
packages the live runtime request surfaces into one CPU-side bundle. It carries
the code-object load request, code-object base binding request, completion-signal
binding request, queue-reservation request, kernarg allocation request,
kernel-argument ABI schema request, kernel-candidate selection request,
kernel-argument ABI semantic projection candidate selection request,
host-launcher branch request, AQL live relocation binding request, and the
executable-readiness report.
`ModelRuntimeMetadataAdmissionReport::runtime_launch_execution_request_plan(...)`
and `ModelGraphReadinessReport::runtime_launch_execution_request_plan(...)`
return the same request plan from the admitted-model and readiness-level paths.

The current bundle allows static metadata proof progress while keeping
`all_components_applied=false`. The kernel-argument ABI schema row can report
applied counts for bound named descriptor schemas and verified candidate rows,
while live runtime rows such as code-object loading, signal binding, queue
reservation, kernarg allocation, kernel selection, semantic projection
candidate selection, and live relocation remain unapplied. The
`component_request_count` is a structured sum of the included request counters
so downstream runtime code can audit the amount of work still needed before
launch submission. `component_request_plan_names()` exposes the execution
component request-plan names in typed launch order, and
`pending_component_request_plan_names()` exposes the subset that still has
pending component work.
For the semantic projection candidate-selection row, `request_count` is the
number of dispatches with projection-ready candidates. Missing projection-ready
dispatches remain visible in the embedded selection request plan and readiness
counts, so an all-missing model can still carry the typed row with zero
actionable selection requests while staying blocked on
`kernel_argument_abi_semantic_projection`.

The bundle also exposes ordered `RuntimeLaunchExecutionRequestComponent` rows.
The rows carry a typed `RuntimeLaunchExecutionRequestStep` plus the existing
step index and string request-plan label. `RuntimeLaunchExecutionRequestStep::ALL`
defines the canonical 10-step order from code-object load through AQL live
relocation binding, while `RuntimeLaunchExecutionRequestStep::DESCRIPTORS`
provides the matching static descriptor table. Each descriptor is the source of
truth for the step index, request-plan label, associated unresolved runtime
requirement, optional live-AQL proof kind, optional live-AQL proof input type,
optional proof validation method, and live-queue mutation policy.
`RuntimeLaunchExecutionRequestStep::from_request_plan(...)` maps a public
request-plan label back to the typed step, and
`descriptor_for_request_plan(...)` returns the matching descriptor for external
runtime tooling that starts from string labels in receipts or component rows.
`RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_STEPS` and
`LIVE_AQL_PROOF_DESCRIPTORS` expose the ordered subset that requires live-AQL
proof validation, and `requires_live_aql_proof()` lets runners test an individual
step before constructing proof-surface inputs.
The rows also keep the component
request/applied counters, whether that component's request plan is ready, the
pending count, and whether the executable gate still reports the matching
blocker. When a blocker is present, the row also carries the readiness report's
blocker detail string for that requirement. The component rows partition the
bundle count; `kernel_object` AQL relocation requests are counted in the
code-object base binding row, while the AQL live relocation row accounts for the
remaining `kernarg_address` and `completion_signal` live binding requests.
Use `live_aql_proof_surface_request_plan_names()` to list the ordered request
plans that require live-AQL proof surfaces.
`pending_live_aql_proof_surface_request_plan_names()` and
`pending_live_aql_proof_validation_request_plan_names()` expose the ordered
subsets with unapplied surface requests and pending proof validations, while
`live_aql_submitting_surface_request_plan_names()` and
`live_queue_mutating_component_request_plan_names()` expose the current
side-effect and live-queue mutation row sets.
`live_aql_proof_kind_labels()`, `live_aql_proof_input_labels()`, and
`live_aql_validation_method_labels()` expose the corresponding proof kind,
proof input, and validation method labels. Use `component_for(...)` and
`live_aql_proof_surface_for(...)` for request-plan string lookups, or
`component_for_step(...)` and `live_aql_proof_surface_for_step(...)` when
runtime code wants typed `RuntimeLaunchExecutionRequestStep` lookups.

Rows that already line up with non-submitting live AQL proof surfaces inherit
their proof kind, proof input, and validation method names from the descriptor
table. The current bridge maps `batch_reservation_plan` to
`KfdQueueLiveAqlBatchReservationPlanInput` for the queue-reservation row and
`materialized_packet_plan` to
`KfdQueueLiveAqlMaterializedPacketPlanInput` for the AQL live-relocation row.
The current pending proof-surface and pending proof-validation row sets contain
both rows; the current live-AQL submitting and live-queue mutating row sets are
empty. `ModelRuntimeLaunchExecutionRequestPlan::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` require those same request-plan
consistency checks before accepting the bundle as a non-submitting CPU-side
handoff.

The bundle derives a `RuntimeLaunchExecutionLiveAqlProofSurface` manifest from
those labelled rows. Each surface carries the component step, request plan,
requirement, typed proof kind, proof input label, proof type label, validation
type label, validation method label, validation-ready field name,
no-live-queue-mutation contract field name, request and pending counts,
ready/blocker booleans, and explicit `proof_input_constructed=false`,
`submits_work=false`, and `mutates_live_queue=false` flags. The reference MoE
reports two proof surfaces with 115 pending surface requests; the custom
example reports two proof surfaces with 12 pending surface requests. Each proof
surface also contributes one pending proof-validation request, so both examples
report
`live_aql_proof_validation_request_count=2`,
`live_aql_proof_validation_applied_count=0`, and
`live_aql_proof_validation_pending_count=2`.

For external tooling, `ModelRuntimeLaunchExecutionRequestPlan` also exposes
`receipt_lines()`, `receipt_text()`, and `receipt_fingerprint()`. The receipt is
a deterministic key/value snapshot of the aggregate execution-request counters,
unresolved runtime requirements, ordered component rows, live-AQL proof
surfaces, and semantic projection candidate-selection request counts. It is an
audit receipt for the static execution boundary, not a launch token and not
proof that any live runtime work has been performed. The reference MoE and CLI
self-test currently report a 220-line execution request receipt fingerprint
`20f630f7ffa1cdaf34d594cf6afd175c9d7e2d2d01ee564db243b4130729f5dd`; the
custom example reports
`dea2b05ca5f6fd6da0efc6c494f75562ad5fe24a2dbc13ce505ef34ea0036323`.
External plugin tests that already have a
`ModelPluginInspectionReport` can call
`synthetic_cpu_runtime_launch_execution_request_plan(namespace)` to emit the
same default unresolved launch-request receipt without reconstructing the
bundled gfx950 code-object, default launch window, metadata binding template,
and synthetic device-pointer validation by hand.
`is_non_submitting_boundary()` and `assert_non_submitting_boundary()` give
callers a focused execution-request check that no aggregate live AQL submitting
surface count, submitting proof-surface row, aggregate live queue mutation
component count, or queue-mutating component row has entered the static receipt.

`ModelRuntimeLaunchExecutionRequestPlan::submission_gate()` folds the execution
readiness blockers, pending request components, pending live AQL proof
validations, submitting proof-surface count, and queue-mutation count into a
single `ModelRuntimeLaunchSubmissionGate`. The current reference MoE reports
`submission_ready=false`, 11 submission blockers, 370 pending request
components, 2 pending proof validations, 0 submitting proof surfaces, and 0
queue-mutating components. The custom example reports the same 11 submission
blockers with 41 pending request components. The gate is a guardrail for future
runtime code; it does not submit work. `blocker_requirement_names()` exposes
the ordered blocker requirements behind the submission-blocker count, while
`has_blocker(...)`, `blocker_for(...)`, and `blocker_for_step(...)` let runtime
code test or retrieve a specific gate blocker row by requirement name or typed
launch request step.
`submission_gate_with_live_aql_proof_validations(...)` and
`submission_gate_with_live_aql_proof_validation_application_plan(...)` are
CPU-only overlays for typed proof validation receipts. When every pending
live-AQL proof surface has a passed, ready, no-queue-mutation validation, the
overlay gate reports `live_aql_proof_validation_pending_count=0` and omits the
`live_aql_proof_validation` blocker. It still reports `submission_ready=false`
while execution-readiness blockers or runtime-request component blockers remain,
rejected, missing, side-effecting, or queue-mutating validations continue to
produce the relevant blocker counts, and unexpected validation inputs are
rejected as mismatched overlay state.
For external tooling, `ModelRuntimeLaunchSubmissionGate` also exposes
`receipt_lines()`, `receipt_text()`, and `receipt_fingerprint()`. The receipt is
a deterministic key/value snapshot of the blocked submission decision, including
the target, code-object metadata, aggregate readiness booleans, pending counts,
submission-blocker count, and ordered blocker rows. It is audit metadata for the
static guardrail only; it is not a launch receipt and not evidence of live queue
submission. The reference MoE and CLI self-test currently report a 65-line
submission-gate receipt fingerprint
`795170f0210127c8a9deb2dd10f8d9426ff01b7dded75f9b33cd9c11ffd64299`; the
custom example reports
`e1012ae50e9c1f82b609cefc9c562d69ffd8454b7e1d0233e58392404de596c2`.
`ModelRuntimeLaunchPreflightReport::submission_gate(device_pointers)` returns
the same gate directly from the launch preflight report.
`ModelRuntimeMetadataAdmissionReport::runtime_launch_submission_gate(...)` and
`ModelGraphReadinessReport::runtime_launch_submission_gate(...)` expose the same
gate from the admitted-model and readiness-level paths.
External plugin tests that only need the default fixture can call
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_gate(namespace)`
to emit the same unresolved submission-gate receipt through the accepted
inspection report.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_blocker_report(namespace)`
does the same for the expanded blocker-report receipt.

`ModelRuntimeLaunchSubmissionGate::blocker_report()` expands the gate's blocker
rows into an ordered detail report. The report groups blocker counts by source,
keeps total pending runtime items, and distinguishes execution-readiness
blockers, unapplied runtime request components, pending live-AQL proof
validations, live-AQL submission side effects, and live queue mutation blockers.
The current reference MoE reports 11 blockers, 9 execution-readiness blockers,
372 pending blocker items, 370 pending runtime request components, 2 pending
proof validations, 0 submitting surfaces, and 0 queue-mutating components. The
custom example reports the same 11/9 blocker split with 43 pending blocker
items and 41 pending runtime request components.
For external tooling, `ModelRuntimeLaunchSubmissionBlockerReport` also exposes
`blocker_requirement_names()`, `blocker_for(...)`, `blocker_for_step(...)`,
`receipt_lines()`, `receipt_text()`, and `receipt_fingerprint()`.
`blocker_requirement_names()`
exposes the ordered requirements from the expanded blocker-report rows, and
`blocker_for(...)` returns the detailed row for a specific requirement name.
`blocker_for_step(...)` resolves a typed launch request step through its
canonical requirement label before returning the detailed blocker row.
`execution_readiness_blocker_requirement_names()`,
`runtime_request_component_blocker_requirement_names()`,
`live_aql_proof_validation_blocker_requirement_names()`,
`live_aql_submission_side_effect_blocker_requirement_names()`, and
`live_queue_mutation_blocker_requirement_names()` expose the ordered
requirements behind each grouped blocker class. The current
execution-readiness blocker rows are the nine launch-execution requirements;
the current runtime-component and proof-validation blocker row sets are
`runtime_request_components` and `live_aql_proof_validation`, respectively; the
current submission-side-effect and queue-mutation blocker row sets are empty.
The receipt is a
deterministic key/value snapshot of the blocker report, including grouped
blocker counts, readiness booleans, total pending count, and each ordered
blocker row's source, requirement, pending count, detail, and classification
flags. It is an audit receipt for the static blocked-submission explanation, not
a live execution receipt. The reference MoE and CLI self-test currently report a
132-line submission-blocker report receipt fingerprint
`0a1c07ce8e600644e2ae42d5b4052f9d0649934c0a4b17638632cee96da8e582`; the custom
example reports
`73212566f3e2187e9ef03cbf6c0f3ec21487d456f2d74cbff64e9585b2a6f526`.
`ModelRuntimeLaunchPreflightReport::submission_blocker_report(device_pointers)`
returns the same blocker report directly from the launch preflight report.
`ModelRuntimeMetadataAdmissionReport::runtime_launch_submission_blocker_report(...)`
and `ModelGraphReadinessReport::runtime_launch_submission_blocker_report(...)`
return the same blocker report from the admitted-model and readiness-level paths.

`ModelRuntimeLaunchExecutionRequestPlan::submission_prerequisite_plan()` breaks
the submission gate's aggregate runtime-component blocker back into per-request
prerequisites. Each row mirrors one execution request component, keeps the
request/applied/pending counts, blocker state and detail, live-AQL proof
requirement, proof kind label, proof input type label, validation method label,
live-AQL submission side-effect flag, live-AQL proof-validation pending count,
queue-mutation flag, and a derived `prerequisite_satisfied` boolean. The
current reference MoE reports 10 prerequisites, 0 satisfied, 10 unsatisfied, 370
pending component requests, 2 live-AQL proof prerequisites, 2 proof kind labels,
2 proof input labels, 2 validation method labels, 0 submitting prerequisites, 2
pending proof validations, and 0 queue-mutating prerequisites. The custom
example reports the same 10/0/10 prerequisite split with 41 pending component
requests.
`prerequisite_request_plan_names()` exposes the ordered prerequisite request
plans, and `unsatisfied_prerequisite_request_plan_names()` exposes the subset
whose `prerequisite_satisfied` flag is false.
Each unsatisfied prerequisite also carries a typed
`RuntimeLaunchSubmissionPrerequisiteNextAction`, `next_action_input`,
`next_action_pending_count`, `next_action_uses_live_aql_proof` flag, and
`next_action_live_aql_proof_kind` label for live-AQL proof-validation actions.
The current priority is live-AQL proof validation first, then runtime request
component application, then blocker-only execution-readiness cleanup, then
side-effect or queue-mutation rejection. The current reference MoE, CLI
self-test, custom example, and external plugin all expose 10 next actions: 8
`apply_runtime_request_component` rows and 2 `validate_live_aql_proof` rows.
`next_action_request_plan_names()`, `next_action_labels()`,
`next_action_input_labels()`, `next_action_live_aql_proof_kind_labels()`,
`runtime_request_component_next_action_request_plan_names()`, and
`live_aql_proof_validation_next_action_request_plan_names()` expose the ordered
worklist. The current live-AQL proof-validation next-action rows are queue
reservation and live relocation, with proof kinds `batch_reservation_plan` and
`materialized_packet_plan` and proof inputs
`KfdQueueLiveAqlBatchReservationPlanInput` and
`KfdQueueLiveAqlMaterializedPacketPlanInput`.
`live_aql_proof_prerequisite_request_plan_names()` exposes the prerequisite
request plans that require live-AQL proof surfaces, while
`live_aql_submitting_prerequisite_request_plan_names()`,
`pending_live_aql_proof_validation_prerequisite_request_plan_names()`, and
`live_queue_mutating_prerequisite_request_plan_names()` expose the ordered
prerequisite rows behind the corresponding side-effect, proof-validation, and
queue-mutation counts.
`live_aql_proof_kind_labels()`, `live_aql_proof_input_labels()`, and
`live_aql_validation_method_labels()` expose the corresponding labels.
Use `prerequisite_for(...)` for request-plan string lookups, or
`prerequisite_for_step(...)` when runtime code wants the typed
`RuntimeLaunchExecutionRequestStep` contract.
The live-AQL prerequisite label triples currently map queue reservation to
`batch_reservation_plan` /
`KfdQueueLiveAqlBatchReservationPlanInput` /
`KfdQueueLiveAqlBatchReservationPlanProof::validate_ready` and live relocation
to `materialized_packet_plan` /
`KfdQueueLiveAqlMaterializedPacketPlanInput` /
`KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready`.
External runners that have already built a concrete proof can call the matching
`RuntimeLaunchLiveAqlProofKind` typed validation method and receive a
`RuntimeLaunchLiveAqlProofValidation` receipt without stringly reconstructing
the validator. A proof kind mismatch returns an error. The receipt captures the
proof kind, proof input/type, validation type/method, printed-ready bit, `ready`
bit, `no_live_queue_mutation_contract` bit, `passed` bit, and side-effect
labels, while keeping both live submission and queue mutation disabled.
`live_aql_proof_validation_application_plan(...)` then turns a set of typed
validation receipts into a deterministic per-surface application worklist. The
plan distinguishes missing validations, duplicate proof-kind inputs, failed or
not-ready validation results, and applied non-mutating validation receipts. It
also exposes `applied_request_plan_names()`, `pending_request_plan_names()`,
`applied_proof_kind_labels()`, `application_for(...)`,
`application_for_step(...)`, and receipt/fingerprint helpers. This gives a
runner a typed bridge from validated proof outputs to the proof-validation
worklist without string reconstruction and without mutating the default
submission gate or prerequisite plan. The plan also exposes
`is_non_submitting_boundary()` and `assert_non_submitting_boundary()` so package
tests can reject proof-validation overlays that report live AQL submission or
queue mutation before the overlay is consumed by a submission gate or
prerequisite plan.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(namespace,
validations)` emits that same proof-validation application worklist through the
accepted inspection report and bundled synthetic CPU fixture inputs. It delegates
through the report-level launch request helper, accepts the caller's typed
validation receipts unchanged, and returns the same deterministic application
plan as the explicit launch-request path for those inputs.
`submission_prerequisite_plan_with_live_aql_proof_validations(...)` and
`submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(...)`
consume that application worklist as a CPU-only prerequisite overlay. When both
current proof validations are applied, the queue-reservation and live-relocation
rows report `live_aql_proof_validation_pending_count=0`, stop emitting
`validate_live_aql_proof`, and advance to `apply_runtime_request_component`.
The overlay still keeps `submission_ready=false` while runtime component
requests and execution-readiness blockers remain unresolved; rejected, missing,
or mismatched validation application state keeps the relevant proof-validation
work visible.
`ModelRuntimeLaunchSubmissionPrerequisitePlan::runtime_request_component_application_plan()`
turns the current `apply_runtime_request_component` next-action rows into a
deterministic CPU-only runtime request component application worklist. The
default prerequisite plan reports 8 application rows because queue reservation
and live relocation are still waiting on live-AQL proof validation. The
proof-validation overlay advances those two rows and reports 10 application
rows; a failed validation keeps the corresponding row out of the application
worklist. The worklist exposes ordered application request-plan names, ready and
blocked subsets, live-AQL proof application rows, typed request-plan and
`RuntimeLaunchExecutionRequestStep` lookups, and deterministic receipt helpers.
It does not allocate GPU memory, load code objects, reserve queues, materialize
AQL packets, submit packets, mutate queues, execute kernels, clear runtime
component blockers, mark all components applied, or authorize submission.
`is_non_submitting_boundary()` and `assert_non_submitting_boundary()` add a
focused guard for this worklist by composing the same consistency checks with
zero live AQL submitting application rows and zero live queue-mutating
application rows.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_runtime_request_component_application_plan(namespace,
validations)` emits that same application worklist through the accepted
inspection report and bundled synthetic CPU fixture inputs, after applying the
caller's typed live-AQL proof validation receipts to the default prerequisite
worklist.
External runtimes that perform those component applications can report
`RuntimeLaunchRuntimeRequestComponentApplicationReceipt` rows back to
`ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::application_receipt_plan(...)`.
That receipt-intake plan validates the supplied rows against the current
worklist and records complete, missing, unexpected, incomplete, launch-mismatched
or side-effecting receipts with deterministic receipt/fingerprint helpers. It
can report `all_applications_applied=true` for a fully matched external receipt
set without generating those receipts itself. Its
`is_non_submitting_boundary()` and `assert_non_submitting_boundary()` helpers
require receipt-plan consistency and then reject receipt overlays that report
live AQL submission or live queue mutation before the overlay is consumed by a
submission prerequisite plan or gate.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(namespace,
validations, receipts)` emits that same receipt-intake plan through the accepted
inspection report and bundled synthetic CPU fixture inputs. It applies the
caller's typed live-AQL proof validation receipts to select the default
application worklist and then validates the caller-supplied runtime component
application receipts against that worklist. It does not generate those receipts
or apply runtime components.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(namespace,
validations, runtime_component_receipts)` emits the current readiness-resolution
worklist through the accepted inspection report and bundled synthetic CPU fixture
inputs, after applying those proof-validation and runtime-component receipt
overlays to the default prerequisite worklist. It does not generate resolution
receipts, resolve blockers, mark execution readiness complete, authorize
submission, or execute kernels.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` emits the same
resolution receipt-intake plan through the accepted inspection report and
bundled synthetic CPU fixture inputs. It selects the default readiness-resolution
worklist after proof-validation and runtime-component receipt overlays, then
validates the caller-supplied resolution receipts against that worklist. It does
not generate resolution receipts or resolve blockers.
`RuntimeLaunchExecutionReadinessBlockerResolutionReceipt::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` reject individual caller-supplied
resolution receipts that report live AQL submission or live queue mutation.
`ModelRuntimeLaunchExecutionReadinessBlockerResolutionReceiptPlan::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` add the same focused guard to the
aggregate receipt-intake plan. The report-level synthetic CPU
prerequisite/gate/blocker helpers assert this boundary before deriving
downstream submission metadata from caller-supplied resolution receipts.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` emits the same
overlaid prerequisite plan through the accepted inspection report and bundled
synthetic CPU fixture inputs. It first selects the default prerequisite worklist
after proof-validation and runtime-component receipt overlays, validates the
caller-supplied readiness-resolution receipts against that derived resolution
worklist, then overlays the validated receipt plan back onto those
prerequisites. It does not create resolution receipts, clear blockers without a
matched receipt plan, or authorize live queue submission. It asserts the returned
prerequisite plan's non-submitting boundary before returning it.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` emits the matching
submission-gate metadata for that same report-level overlay chain. It delegates
through the prerequisite overlay helper and then derives `submission_gate()`, so
stale or unexpected receipt rows fail the same way as the explicit lower-level
path. It asserts the returned gate's non-submitting boundary before returning it
and does not perform live admission or submit work.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(namespace,
validations, runtime_component_receipts, resolution_receipts)` emits the
matching blocker-report metadata for the same report-level overlay chain. It
delegates through the gate overlay helper and then expands `blocker_report()`,
preserving the same stale/unexpected receipt failures as the explicit path. It
asserts the returned blocker report's non-submitting boundary before returning
it.
`submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(...)`
and `submission_gate_with_runtime_request_component_application_receipt_plan(...)`
consume that receipt-intake plan as a CPU-only overlay. Accepted receipt rows set
the matching prerequisite row's applied count to the request count, zero its
component pending count, and move that row to
`resolve_execution_readiness_blocker` when execution-readiness is still blocked.
Rows not present in the current application worklist, missing receipts,
incomplete receipts, launch-mismatched receipts, and side-effecting receipts do
not clear component pending state. The default prerequisite and gate surfaces
remain unchanged. A full proof-validation overlay plus a fully matched component
receipt overlay can remove the aggregate `runtime_request_components` blocker
and set `all_components_applied=true` on the overlaid gate, but existing
execution-readiness blockers still keep `submission_ready=false` and no live
submission is authorized.
`execution_readiness_blocker_resolution_plan()` converts the current
`resolve_execution_readiness_blocker` next-action rows into a deterministic
CPU-only worklist grouped by unique execution-readiness requirement. After both
live-AQL proof validation and component-application receipt overlays are fully
matched, the current reference MoE reports 10 source prerequisite rows grouped
into 9 readiness blockers because `code_object_load_request_plan` and
`code_object_base_binding_request_plan` share `loaded_code_object_base`. The
default prerequisite plan reports no readiness-resolution rows because proof
validation and runtime component application still have priority; a partial
component receipt overlay reports only the blockers whose prerequisite rows have
advanced to readiness cleanup. The resolution plan exposes requirement names,
flattened source request plans, `resolution_for(...)`,
`resolution_for_step(...)`, deterministic receipt helpers, and consistency
checks that reject stale counts or any claim that resolution receipts already
exist. External runtimes can return
`RuntimeLaunchExecutionReadinessBlockerResolutionReceipt` rows to
`ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan::resolution_receipt_plan(...)`.
That CPU-only intake validates complete, missing, unexpected, incomplete,
launch-mismatched, and side-effecting resolution receipts against the current
worklist and can report `all_resolutions_applied=true` for a complete matched
receipt set. It does not generate receipts, clear blockers, set
`execution_readiness_ready`, or authorize submission.
`is_non_submitting_boundary()` and `assert_non_submitting_boundary()` require
the same receipt-plan consistency and reject live AQL submitting receipt rows or
live queue-mutating receipt rows before the intake can feed the next overlay.
The separate
`ModelRuntimeLaunchSubmissionPrerequisitePlan::submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(...)`
and
`submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(...)`
helpers require that non-submitting receipt-plan boundary, then overlay an
already validated receipt plan onto the current
prerequisite worklist. Applied readiness receipts clear only their matching
blocker rows; missing, incomplete, mismatched, side-effecting, unexpected, or
stale-worklist receipt plans leave blockers in place or fail matching. A fully
matched proof-validation, component-application, and readiness-resolution chain
can produce a CPU-side `submission_ready=true` gate with zero blockers, but this
still represents a deterministic handoff contract rather than live AQL
submission.
`ModelRuntimeLaunchExecutionRequestPlan::synthetic_cpu_resolved_submission_prerequisite_plan(...)`
is the public convenience wrapper for this fixture/handoff path. It accepts
typed `RuntimeLaunchLiveAqlProofValidation` receipts plus a non-empty receipt
source label, validates the proof application worklist, synthesizes non-submitting
runtime component application receipts, synthesizes non-mutating execution
readiness blocker resolution receipts, and returns the same fully satisfied
`ModelRuntimeLaunchSubmissionPrerequisitePlan` that the lower-level overlay
chain would produce. The helper fails closed when proof validations are missing,
failed, stale, side-effecting, or leave any component or execution blocker
unresolved. It is CPU-only metadata for tests and downstream handoff fixtures; it
does not allocate buffers, materialize or submit AQL packets, mutate live queues,
or prove runtime execution. The returned plan satisfies
`assert_non_submitting_boundary()`.
`ModelRuntimeLaunchExecutionRequestPlan::synthetic_cpu_resolved_submission_gate(...)`
derives the zero-blocker `ModelRuntimeLaunchSubmissionGate` from that resolved
prerequisite plan, so the gate and prerequisite views stay byte-equivalent to
the same receipt-overlay chain. The returned gate satisfies
`assert_non_submitting_boundary()`: `submission_ready=true` here is still a
CPU-side metadata result, not a queue-submission token.
`ModelRuntimeLaunchExecutionRequestPlan::synthetic_cpu_resolved_submission_blocker_report(...)`
expands that deterministic resolved gate into the corresponding zero-blocker
`ModelRuntimeLaunchSubmissionBlockerReport`. It delegates through
`synthetic_cpu_resolved_submission_gate(...)` and then calls
`blocker_report()`, so it preserves the same proof-validation failures and
non-submitting boundary as the resolved gate helper.
`ModelPluginInspectionReport::synthetic_cpu_resolved_submission_prerequisite_plan(namespace,
validations, receipt_source)` is the report-level wrapper for the same resolved
prerequisite metadata. It rebuilds the default CPU-only launch request plan from
the accepted inspection report, delegates to the lower-level execution-request
helper, and asserts the returned prerequisite plan's non-submitting boundary.
`ModelPluginInspectionReport::synthetic_cpu_resolved_submission_gate(namespace,
validations, receipt_source)` is the report-level convenience wrapper for the
same deterministic fixture path. It rebuilds the default CPU-only launch request
plan from the accepted inspection report using the bundled gfx950 code-object
metadata, `DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES`,
`DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE`, and
`DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT`, then delegates to the lower-level
execution-request helper. The returned gate is byte-equivalent to the explicit
launch-request path for those inputs and asserts the same non-submitting
boundary.
`ModelPluginInspectionReport::synthetic_cpu_resolved_submission_blocker_report(namespace,
validations, receipt_source)` is the report-level wrapper for the same
zero-blocker blocker-report metadata. It rebuilds the same default CPU-only
launch request plan, delegates through the report-level resolved gate helper,
then expands `blocker_report()` without generating live work or authorizing
serving. It asserts the returned blocker report's non-submitting boundary before
returning it.
The current pending proof-validation prerequisite rows are queue reservation and
live relocation; the current live-AQL submitting and queue-mutating
prerequisite row sets are empty.
`ModelRuntimeLaunchSubmissionPrerequisitePlan::is_non_submitting_boundary()`
and `assert_non_submitting_boundary()` require those same prerequisite-plan
consistency checks before accepting blocked or resolved prerequisite metadata as
a non-submitting CPU-side handoff.
`ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_prerequisite_plan(namespace)`
emits the default unresolved prerequisite-plan receipt through the accepted
inspection report and bundled synthetic CPU fixture inputs. It delegates through
the report-level launch request helper and stays metadata-only: no GPU buffers
are allocated, no AQL packets are materialized or submitted, no queues are
mutated, and no runtime execution is claimed.
For external tooling, `ModelRuntimeLaunchSubmissionPrerequisitePlan` also
exposes `receipt_lines()`, `receipt_text()`, and `receipt_fingerprint()`. The
receipt is a deterministic key/value snapshot of the per-request prerequisite
plan, including aggregate prerequisite counts, readiness booleans, pending
request totals, live-AQL proof counts, next-action counts, and each ordered
prerequisite row's request plan, requirement, request/applied/pending counts,
blocker detail, proof kind and proof labels, queue-mutation flag, satisfaction
boolean, and next-action fields including the typed live-AQL proof kind for
proof-validation actions. It is static audit metadata only and does not
authorize live submission. The reference MoE and CLI self-test currently report
a 249-line submission-prerequisite plan receipt fingerprint
`3e12d3463820e350a5c163d41fe30a98c2e19c662049df29b0dd8099173f208e`; the
custom example reports
`28878a5e8a05eb5e53c4eb8adf8fcda66d0692481fa4d2f59b46f3f1fe3b6a91`.
`ModelRuntimeLaunchPreflightReport::submission_prerequisite_plan(device_pointers)`
returns the same prerequisite plan directly from the launch preflight report.
`ModelRuntimeMetadataAdmissionReport::runtime_launch_submission_prerequisite_plan(...)`
and `ModelGraphReadinessReport::runtime_launch_submission_prerequisite_plan(...)`
return the same prerequisite plan from the admitted-model and readiness-level
paths.
The reference MoE example and CLI self-test exercise the readiness-level
submission helpers; the custom plugin example exercises the admitted-model
helpers. All three assert equivalence with the preflight/execution-request path
before printing the existing `launch_submission_*` summaries.

This is not an executor. It does not choose kernels, load code objects, bind
device addresses or HSA signal handles, patch AQL packet bytes, copy kernargs,
reserve queues, ring a doorbell, submit work, execute kernels, or claim
performance.

## Runtime Launch Window Manifest

`ModelRuntimeStageLaunchCandidatePlan::launch_window_plan(max_dispatches)`
splits admitted stage launch candidates into deterministic per-stage dispatch
windows. Each window reports:

- global window index plus stage-local window index
- stage identity and first op index
- ordered dispatch names and deduplicated entrypoint symbols
- resource slots and bound logical handles used by that window
- dispatch count, chain-barrier count, and terminal-signal-required metadata

`DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES` is the example/default CPU-side
window capacity used by the selftests. Callers may choose a different non-zero
capacity. This is still metadata for a future packet builder. It does not
reserve queue slots, allocate kernarg memory, set packet barrier bits, choose
streams, prove residency, submit AQL, wait on signals, or execute kernels.

## Tensor Access Plan

`ModelPrimitiveGraph::tensor_access_plan()` validates the graph and returns one
read/write row per primitive op. The rows are ordered exactly like the graph ops
and include:

- op name and primitive kind
- tensors read by the op
- tensors written by the op

Cache refs are expanded into their concrete tensor handles. For example,
`KvCacheAppend` reads the new K/V row plus block-table and position metadata,
then writes the K/V cache tensors; `PagedAttention` reads the query, metadata,
and cache tensors, then writes the attention output.

The access plan is a runtime-bridge manifest for future buffer binding and
lifetime analysis. It does not decide allocation, aliasing, residency windows,
synchronization, stream placement, or in-place update legality.

## Tensor Lifetime Plan

`ModelPrimitiveGraph::tensor_lifetime_plan()` folds the storage plan and access
plan into one static row per declared tensor:

- tensor role, dtype, and storage bytes
- first and last op index/name that touch the tensor
- read/write counts
- reader and writer op names
- tensors that are declared but never accessed

This is a CPU-side lifetime manifest for later buffer binding and reuse
analysis. It does not allocate memory, alias tensors, prove in-place update
safety, split cache ranges by token/page, choose residency windows, or schedule
streams.

## Tensor Binding Manifest

`ModelPrimitiveGraph::tensor_binding_plan()` folds declared tensor roles and the
lifetime plan into a deterministic static binding contract:

- read-only external inputs such as token ids, positions, sequence length, and
  block tables
- write-only external outputs such as the next-token tensor
- checkpoint weight tensors and their logical checkpoint keys
- mutable persistent state such as KV-cache tensors
- scratch tensors for activations, routing state, and internally consumed logits
- unused tensors
- static role/access issues, such as scratch tensors that are read but never
  produced or checkpoint weights that are written by the graph

This is a manifest for future buffer binding and validation. It does not allocate
GPU memory, attach device pointers, decide scratch aliasing/reuse, page or shard
KV-cache storage, synchronize writes, load checkpoints, or execute any primitive.

## Tensor Storage Footprint

`ModelPrimitiveGraph::tensor_storage_plan()` validates the graph and returns a
deterministic storage manifest over the declared tensors:

- one row per tensor
- tensor role and dtype
- element count
- storage bytes
- aggregate bytes by role
- total declared bytes

The byte accounting is dtype-aware. FP4 uses four bits per element with byte
ceiling, so three FP4 elements occupy two bytes. This is still mathematical
storage accounting only: it does not model allocator alignment, padding,
aliasing, reuse, residency windows, or page-table overhead.

## Checkpoint Binding Manifest

Weight tensors can carry optional logical checkpoint keys. The reference
Qwen-style MoE graph binds each declared weight to a Qwen-style key, such as
`model.layers.0.self_attn.q_proj.weight`. Packed expert tensors use wildcard
patterns, such as `model.layers.0.mlp.experts.*.gate_proj.weight`, because the
graph declares one packed tensor for the expert projection family.

`ModelPrimitiveGraph::checkpoint_binding_plan()` validates the graph and returns
a deterministic CPU-side manifest:

- one row per weight tensor with a checkpoint key
- tensor dtype, shape, element count, and storage bytes
- missing weight tensors that do not have checkpoint keys
- total declared checkpoint bytes

`ModelCheckpointBindingPlan::resolve_against_available_keys()` can compare that
manifest with an already-materialized set of checkpoint keys. It matches exact
keys, expands one `*` wildcard through prefix/suffix matching for packed expert
patterns, reports missing checkpoint entries, carries unbound weight tensors,
and lists available checkpoint keys that no binding consumed.

The existing safetensors metadata parsers can feed this resolver directly:
`SafetensorsIndex::resolve_model_checkpoint_bindings()` uses names from a
sharded Hugging Face `weight_map`, and
`SafetensorsShard::resolve_model_checkpoint_bindings()` uses tensor names from a
single safetensors header.

For a parsed single shard,
`SafetensorsShard::validate_model_checkpoint_bindings()` also checks header
metadata after key resolution: dtype, concrete tensor shape, aggregate storage
bytes, and the packed-expert wildcard count/shape-tail contract.
`SafetensorsIndex::validate_model_checkpoint_bindings()` extends that check to a
sharded index: it opens each referenced shard header, reports missing shard
files, and validates dtype, shape, routed tensor presence, and matched byte
totals for the model API checkpoint plan.
`SafetensorsShard::runtime_checkpoint_payload_bindings()` and
`SafetensorsIndex::runtime_checkpoint_payload_bindings()` can then turn that
validated metadata into model API `RuntimeCheckpointPayloadBinding` rows using
the safetensors source path and absolute file payload offsets.
`CheckpointPayloadDirectReadPlan::from_checkpoint_payload_binding_plan()` can
then turn the bound model API payload plan into direct-I/O aligned read-window
work orders with staging prefix offsets and destination device VA spans, still
without opening the files or loading payload bytes.
`CheckpointPayloadDirectReadPlan::staging_batch_plan()` can then coalesce those
work orders into source-local staging batches with reusable staging-slot
assignments and copy pieces, still without opening the files, allocating staging
buffers, or submitting copies.

Checkpoint keys are metadata only. The model API itself does not read
safetensors payloads, bind destination buffers, or decide whether weights need
sharding, transposition, aliasing, or quantization sidecars. Sharded index
validation, runtime payload-row construction, and direct-read work-order
construction are header/receipt-derived; they do not read tensor payload bytes.

## Current Scope

Implemented and tested:

- typed primitive graph builder
- custom `ModelDefinition` authoring example
- reference Qwen-style MoE decoder graph
- graph validation
- tensor storage footprint
- per-op tensor access plan
- per-tensor lifetime plan
- tensor binding manifest
- checkpoint binding manifest
- checkpoint key resolution against available key sets
- safetensors metadata bridge into checkpoint key resolution
- single-shard safetensors metadata validation for checkpoint bindings
- sharded safetensors index header validation for checkpoint bindings and
  missing shard files
- safetensors header-derived runtime checkpoint payload binding rows
- CPU-only checkpoint payload direct-read work-order and staging-batch plans
  with direct-I/O aligned read windows, staging prefix offsets, reusable
  staging-slot assignments, copy pieces, and destination device VA spans
- semantic stage metadata
- op-level lowering-readiness plan
- stage-level lowering-readiness plan
- lowering route entrypoint provenance audit
- primitive execution manifest
- runtime slot ABI manifest
- runtime dispatch intent manifest
- metadata slot-binding template and complete/dispatch/stage/stage-dispatch
  scoped/aggregate preflight
- runtime device-pointer preflight
- runtime device-pointer lifetime preflight
- runtime KFD allocation/residency request plan
- runtime KFD VM-acquire request plan
- runtime KFD alloc-memory request plan
- runtime KFD alloc-memory result binding plan
- runtime KFD map-memory request plan
- runtime KFD map-memory argument binding plan
- runtime KFD map-memory result binding plan
- runtime slot KFD residency binding plan
- runtime stage resource manifest
- runtime stage bundle manifest
- runtime stage dispatch manifest
- runtime stage launch-candidate manifest
- runtime launch entrypoint provenance manifest
- runtime launch kernel-requirement manifest
- runtime launch kernel-metadata manifest
- runtime launch code-object load request plan
- runtime launch code-object base binding request plan
- runtime launch preflight report
- runtime launch AQL packet-field handoff
- runtime launch kernel-selection readiness report
- runtime launch kernel-candidate recommendation plan
- runtime launch kernel-candidate selection request plan
- runtime launch host-launcher branch resolution request plan
- runtime launch argument-binding manifest
- runtime launch device-argument manifest
- runtime launch kernarg layout plan
- runtime launch kernarg serialization plan
- runtime launch kernel-argument ABI verification preflight plan
- runtime launch kernel-argument ABI size-compatibility receipt
- runtime launch kernel-argument ABI verification gap report
- runtime launch kernel-argument ABI capacity request plan
- runtime launch kernel-argument ABI schema request plan
- runtime launch kernel-argument ABI semantic plan
- runtime launch kernel-argument ABI semantic gap report
- runtime launch kernel-argument ABI semantic projection plan
- runtime launch kernel-argument ABI semantic projection gap report
- runtime launch kernel-argument ABI semantic projection candidate
  recommendation plan
- runtime launch kernel-argument ABI semantic projection candidate selection
  request plan
- runtime launch kernel-argument ABI semantic projection recommendation report
- runtime launch kernarg allocation request plan
- runtime launch staging-footprint plan
- runtime launch staging-layout plan
- runtime launch completion-signal policy plan
- runtime launch completion-signal binding request plan
- runtime launch queue-slot plan
- runtime launch queue-reservation request plan
- runtime launch dispatch-geometry plan
- runtime launch AQL packet-template plan
- runtime launch AQL packet relocation-site plan
- runtime launch AQL packet byte-template plan
- runtime launch AQL packet materialization preflight plan
- runtime launch AQL live relocation binding-request plan
- runtime launch executable-readiness gate
- runtime launch execution request plan
- runtime launch live AQL proof-surface manifest and validation-request counters
- runtime launch submission gate
- runtime launch window manifest
- structured static readiness issue report
- runtime metadata admission report
- composed CPU-side graph readiness report

Not claimed yet:

- executable graph lowering into AQL packets
- complete code-object kernel ABI validation for every dispatch/candidate
  through the model API
- stream scheduling or residency windows
- executable buffer allocation, live pointer residency validation, or GPU
  lifetime validation
- live KFD VM acquisition, allocation, map/unmap, or peer visibility proof
- safetensors payload loading through the model API
- tensor payload validation against safetensors contents
- allocator padding, aliasing, or reuse decisions
- OpenAI-compatible serving
- runtime correctness through this new graph API
- throughput or latency performance through this new graph API

## Runnable Examples

These commands are CPU-only metadata gates. They compile model definitions,
validate the primitive graph, admit runtime metadata, and print static launch
readiness reports. They do not allocate GPU buffers, patch AQL, submit queues,
or run kernels.

Run the CI/release gate over the public examples:

```bash
python3 tools/check_model_api_public_examples.py
```

Build and summarize the compact reference MoE graph:

```bash
cargo run -p mainarch-core --example reference_moe_model_api
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-launch-request-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-submission-gate-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-resolved-submission-gate-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-resolved-submission-blocker-report-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-submission-blocker-report-receipt
cargo run -p mainarch-core --example reference_moe_model_api -- --runtime-submission-prerequisite-plan-receipt
```

Build a small custom model through the public `ModelDefinition` trait:

```bash
cargo run -p mainarch-core --example custom_model_api
cargo run -p mainarch-core --example custom_model_api -- --runtime-launch-request-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-submission-gate-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-resolved-submission-gate-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-resolved-submission-blocker-report-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-submission-blocker-report-receipt
cargo run -p mainarch-core --example custom_model_api -- --runtime-submission-prerequisite-plan-receipt
```

Build the standalone external package sample through the public prelude:

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
```

Exercise the same external model through the package library contract test:

```bash
cargo run --locked --manifest-path examples/model-api-plugin/Cargo.toml -- --static-handoff-receipt
cargo test --locked --manifest-path examples/model-api-plugin/Cargo.toml
```

The public gate compares the reference MoE launch request receipt-only command
against
`crates/mainarch-core/examples/expected-reference-moe-runtime-launch-request.receipt`,
which pins every CPU-side runtime launch request component, live-AQL proof
surface, and non-mutating launch boundary for the reduced reference graph. It
compares the reference MoE submission gate receipt-only command against
`crates/mainarch-core/examples/expected-reference-moe-runtime-submission-gate.receipt`,
which pins the aggregate blocked-submission guard and ordered submission
blockers for the same reduced graph. It compares the reference MoE resolved
submission gate receipt-only command against
`crates/mainarch-core/examples/expected-reference-moe-runtime-resolved-submission-gate.receipt`,
which pins the CPU-only proof-validation, runtime-component, and
execution-readiness receipt overlay with zero blockers and `submission_ready=true`.
It compares the reference MoE resolved submission prerequisite plan receipt-only
command against
`crates/mainarch-core/examples/expected-reference-moe-runtime-resolved-submission-prerequisite-plan.receipt`,
which pins the same resolved overlay as a fully satisfied
`ModelRuntimeLaunchSubmissionPrerequisitePlan` with no pending next actions.
It compares the reference MoE resolved submission blocker report receipt-only
command against
`crates/mainarch-core/examples/expected-reference-moe-runtime-resolved-submission-blocker-report.receipt`,
which pins the same resolved overlay as a zero-blocker
`ModelRuntimeLaunchSubmissionBlockerReport`.
It compares the reference MoE submission blocker report receipt-only command against
`crates/mainarch-core/examples/expected-reference-moe-runtime-submission-blocker-report.receipt`,
which pins the blocker-class counts, total pending count, and per-blocker
classification flags for that blocked-submission boundary. It compares the
reference MoE submission prerequisite plan receipt-only command against
`crates/mainarch-core/examples/expected-reference-moe-runtime-submission-prerequisite-plan.receipt`,
which pins every CPU-side runtime submission prerequisite, pending count,
live-AQL proof requirement, and non-mutating boundary for the reduced reference
graph. It compares the custom model launch request receipt-only command against
`crates/mainarch-core/examples/expected-custom-model-runtime-launch-request.receipt`,
which pins the same CPU-side runtime launch request boundary on the smaller
authoring example. It compares the custom model submission gate receipt-only
command against
`crates/mainarch-core/examples/expected-custom-model-runtime-submission-gate.receipt`,
which pins its aggregate blocked-submission guard and ordered submission
blockers. It compares the custom model resolved submission gate receipt-only
command against
`crates/mainarch-core/examples/expected-custom-model-runtime-resolved-submission-gate.receipt`,
which pins its zero-blocker CPU-only submission handoff on the smaller authoring
example. It compares the custom model resolved submission prerequisite plan
receipt-only command against
`crates/mainarch-core/examples/expected-custom-model-runtime-resolved-submission-prerequisite-plan.receipt`,
which pins the same resolved handoff as a fully satisfied
`ModelRuntimeLaunchSubmissionPrerequisitePlan` with no pending next actions on
the smaller authoring example. It compares the custom model resolved submission
blocker report receipt-only command against
`crates/mainarch-core/examples/expected-custom-model-runtime-resolved-submission-blocker-report.receipt`,
which pins the same resolved handoff as a zero-blocker
`ModelRuntimeLaunchSubmissionBlockerReport`. It compares the custom model
submission blocker report receipt-only
command against
`crates/mainarch-core/examples/expected-custom-model-runtime-submission-blocker-report.receipt`,
which pins its blocker-class counts, total pending count, and per-blocker
classification flags. It compares the custom model submission prerequisite plan
receipt-only command against
`crates/mainarch-core/examples/expected-custom-model-runtime-submission-prerequisite-plan.receipt`
for the same blocked-submission boundary on the smaller authoring example. It
compares the CLI selftest launch request receipt-only command against
`crates/mainarch-cli/expected-model-api-selftest-runtime-launch-request.receipt`,
which pins the same launch request boundary through the `mainarch-cli` package.
It compares the CLI selftest submission gate receipt-only command against
`crates/mainarch-cli/expected-model-api-selftest-runtime-submission-gate.receipt`,
which pins the same aggregate blocked-submission guard through the
`mainarch-cli` package.
It compares the CLI selftest resolved submission gate receipt-only command
against
`crates/mainarch-cli/expected-model-api-selftest-runtime-resolved-submission-gate.receipt`,
which pins the same zero-blocker CPU-only submission handoff through the
`mainarch-cli` package.
It compares the CLI selftest resolved submission prerequisite plan receipt-only
command against
`crates/mainarch-cli/expected-model-api-selftest-runtime-resolved-submission-prerequisite-plan.receipt`,
which pins the same resolved handoff as a fully satisfied
`ModelRuntimeLaunchSubmissionPrerequisitePlan` with no pending next actions
through the `mainarch-cli` package.
It compares the CLI selftest resolved submission blocker report receipt-only
command against
`crates/mainarch-cli/expected-model-api-selftest-runtime-resolved-submission-blocker-report.receipt`,
which pins the same resolved handoff as a zero-blocker
`ModelRuntimeLaunchSubmissionBlockerReport` through the `mainarch-cli` package.
It compares the CLI selftest submission blocker report receipt-only command
against
`crates/mainarch-cli/expected-model-api-selftest-runtime-submission-blocker-report.receipt`,
which pins the same blocker-class counts through the `mainarch-cli` package.
It compares the CLI selftest submission prerequisite plan receipt-only command
against
`crates/mainarch-cli/expected-model-api-selftest-runtime-submission-prerequisite-plan.receipt`,
which pins the same prerequisite receipt through the `mainarch-cli` package.
It compares the CLI selftest static handoff receipt-only command against
`crates/mainarch-cli/expected-model-api-selftest-static-handoff.receipt`, which
pins the same full static handoff boundary through the `mainarch-cli` package,
including ordered unresolved execution requirement labels and explicit
non-execution counters.
It compares the rejected model receipt-only command against
`crates/mainarch-core/examples/expected-rejected-model-api-rejection.receipt`,
which pins the typed negative-path rejection receipt for an unsupported
collective plugin. It
compares the standalone external package sample against
`examples/model-api-plugin/expected-output.txt`, an exact receipt fixture for
the external-package boundary. The fixture includes the deterministic plugin
manifest and compatibility receipt, static launch request descriptor table,
live-AQL proof descriptor labels, static graph summary, CPU-side launch semantic
projection counts, projection-aware candidate-selection counts, generic
kernel-selection counts, host-launcher branch request counts, launch-order
ready/missing selection and branch-request operation names, and the compact
static handoff receipt with manifest and compatibility receipt fingerprints plus
ordered unresolved requirement labels; it still reports
`aql_dispatchable_packets=0` and does not claim live execution. The static
handoff consistency check rejects an
executable launch claim, nonzero dispatchable AQL packet count, GPU-buffer
allocation claim, or kernel-submission claim while the contract reports
`live_execution_supported=false`.
`examples/model-api-plugin/expected-contract.receipt` is the exact minimal
model API contract receipt fixture for downstream packages that want to pin the
authoring contract before inspecting a model graph.
`examples/model-api-plugin/expected-manifest.receipt` is the exact external
plugin manifest receipt fixture for downstream packages that want to pin the
static graph/resource counts and canonical runtime launch request descriptor
table, including live-AQL proof kind/input/validation labels, before deriving a
static handoff receipt.
`examples/model-api-plugin/expected-compatibility.receipt` is the exact
manifest/catalog compatibility receipt fixture for downstream packages that
want to pin accepted contract, target, fingerprint, static metadata, and live
execution compatibility before deriving launch handoff metadata.
`examples/model-api-plugin/expected-runtime-launch-request.receipt` is the exact
runtime launch request plan receipt fixture for downstream packages that want to
pin the CPU-side launch work items, pending component counts, live-AQL proof
surfaces, and non-submitting launch boundary before a runtime implementation
starts applying those requests.
`examples/model-api-plugin/expected-runtime-submission-gate.receipt` is the exact
runtime submission gate receipt fixture for downstream packages that want to pin
submission blockers, unresolved execution readiness, and the zero live-AQL
submission/queue-mutation boundary before any GPU-launch integration.
`examples/model-api-plugin/expected-runtime-resolved-submission-gate.receipt` is
the exact resolved runtime submission gate receipt fixture for downstream
packages that want to pin the CPU-only proof-validation, runtime-component, and
execution-readiness receipt overlay with zero blockers and
`submission_ready=true`.
`examples/model-api-plugin/expected-runtime-resolved-submission-prerequisite-plan.receipt`
is the exact resolved runtime submission prerequisite plan fixture for
downstream packages that want to pin that same CPU-only handoff as a fully
satisfied prerequisite worklist with no pending next actions.
`examples/model-api-plugin/expected-runtime-resolved-submission-blocker-report.receipt`
is the exact resolved runtime submission blocker report fixture for downstream
packages that want to pin the matching zero-blocker blocker-report metadata
from that same CPU-only handoff path.
`examples/model-api-plugin/expected-runtime-submission-blocker-report.receipt`
is the exact runtime submission blocker report fixture for downstream packages
that want to pin blocker-class counts and per-blocker classifications before
turning CPU-side launch requests into live runtime work.
`examples/model-api-plugin/expected-runtime-submission-prerequisite-plan.receipt`
is the exact report-level runtime submission prerequisite plan helper fixture
for downstream packages that want to pin every launch request prerequisite,
pending count, live-AQL proof requirement, and non-mutating boundary before
runtime execution.
`examples/model-api-plugin/expected-static-handoff.receipt` is the exact full
static handoff receipt fixture for downstream tooling that wants the stable
line-oriented artifact, including full manifest and compatibility receipt
fingerprint bindings, ordered unresolved execution requirement labels, and
explicit non-execution counters instead of the summary line.
The gate also runs the package integration test against the exported
`ExternalMiniMoe` library model and guards that the external library, binary,
and test import mainarch only through the public prelude. It also runs the
checkpoint metadata example and guards that its model API symbols come from the
public prelude while its explicit safetensors metadata helpers come from the
weights module. That example now also prints the CPU-only direct-read work-order
and staging-batch summaries derived from safetensors payload rows and synthetic
residency receipts. The gate also checks the manifest descriptor table and
request-plan lookup round trips against the derived execution request rows
through the prelude.

Exercise the negative plugin path and verify the deterministic rejection receipt:

```bash
cargo run -p mainarch-core --example rejected_model_api
cargo run -p mainarch-core --example rejected_model_api -- --rejection-receipt
```

Run the same CPU-only readiness path through the CLI package:

```bash
cargo run -p mainarch-cli --bin mainarch-model-api-selftest
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-launch-request-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-submission-gate-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-resolved-submission-gate-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-resolved-submission-prerequisite-plan-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-resolved-submission-blocker-report-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-submission-blocker-report-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --runtime-submission-prerequisite-plan-receipt
cargo run -p mainarch-cli --bin mainarch-model-api-selftest -- --static-handoff-receipt
```

Build the compact reference MoE graph, synthesize matching safetensors metadata,
validate single-shard and index headers, and derive header-only payload span rows:

```bash
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-staging-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-mapped-host-staging-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-copy-plan-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-destination-residency-proof-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-destination-residency-query-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-queue-reservation-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-queue-reservation-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-copy-completion-signal-binding-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-copy-completion-signal-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-materialization-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-materialization-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-validation-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-sdma-copy-packet-validation-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-cache-visibility-policy-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-synchronization-plan-input-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-schedule-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-prerequisite-plan-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-runtime-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-bound-runtime-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-mapped-host-staging-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-destination-residency-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-sdma-queue-reservation-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-copy-completion-signal-binding-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-packet-materialization-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-packet-validation-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-cache-visibility-policy-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-upload-completion-synchronization-handoff-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-pin-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-pin-virtual-address-plan-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-userptr-pin-arguments-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-vm-acquire-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-userptr-alloc-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-userptr-alloc-result-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-map-memory-request-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-map-memory-argument-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-kfd-map-memory-result-binding-receipt
cargo run -p mainarch-core --example reference_moe_checkpoint_metadata -- --checkpoint-host-staging-pin-page-rounding-receipt
```

Current invariant summaries:

- Reference MoE example and CLI self-test:
  `graph tensors=75 ops=36 stages=7`, `lowering gap_ops=0`,
  `model_api_vocabulary primitive_kinds=12 stage_kinds=5`,
  `plugin_inspection consistent=true accepted=true`,
  `plugin_summary receipt_fingerprint=<64-hex> accepted=true static_ready=true compatibility_issues=0`,
  `catalog_capabilities primitive_kinds=12 cases=22 native_gpu_cases=16 fused_native_gpu_cases=1 gap_cases=5 parameterized=true`,
  `plugin_manifest fingerprint=<64-hex> primitive_kinds=12 stage_kinds=5 launch_steps=10 live_aql_proof_steps=2 static_ready=true live_execution_supported=false`,
  `plugin_compatibility accepted=true issues=0 target_matches=true fingerprint_matches=true static_metadata_ready=true live_execution_supported=false`,
  `launch_kernarg_abi dispatches_with_verified=14 dispatches_without_verified=22`,
  `launch_kernarg_abi_gaps dispatches_without_verified=22 candidate_abis=61 size_shortfall_candidates=61 primary_size_shortfalls=61`,
  `launch_kernarg_abi_semantics schemas=51 schema_candidates=147 missing_schema_candidates=0 descriptor_matches=147 verified_candidates=0 dispatches_with_verified=0 dispatches_without_verified=36 field_schemas=1028 verified_fields=83 missing_fields=494 field_mismatches=451 extra_arguments=479 ready=false unresolved_runtime_requirements=8`,
  `launch_kernarg_abi_semantic_gaps dispatches_without_verified=36 candidate_abis=147 schema_candidates=147 missing_schema_candidates=0 descriptor_matches=147 verified_candidates=0 primary_missing_schemas=0 primary_descriptor_mismatches=0 primary_missing_model_args=114 primary_field_mismatches=33 primary_extra_arguments=0 primary_size_shortfalls=0 primary_unknown=0 field_schemas=1028 verified_fields=83 missing_fields=494 field_mismatches=451 extra_arguments=479 all_dispatches_have_semantic_verified_candidate=false`,
  `launch_kernarg_abi_semantic_missing_schema_symbols: count=0 symbols=`,
  `launch_kernarg_abi_semantic_missing_model_arguments: count=63 names=allreduce_sequence_base,append_count,base_pos,batch_indices,batch_size,block_size,candidate_logits,candidate_token_ids,chunk_len,chunk_len_vec4,chunk_offset,chunk_offset_vec4,chunk_owner,dualpath_cu_chunk_len_vec4,dualpath_cu_owned_chunk_len_vec4,dualpath_sdma_semaphores,eps,expert_ids_history,expert_weights_history,gbar,group_size,history_steps,indptr,intermediate,last_page_len,layer,local_allreduce_buffer,local_allreduce_flags,nthreads,num_groups,num_layers,num_splits,num_tiles,num_wg,p2p_peer_count,partial,partials,peer_allreduce_flag_ptrs,peer_allreduce_ptrs,peer_reduce_staging_ptrs,peer_residual_ptrs,peer_stride_vec4,persistent_allreduce_ctrl,persistent_allreduce_output,persistent_allreduce_total_ops,physical_blocks,positions,q_heads_per_kv,reduce_staging_buffer,reduce_staging_chunk_len,rmsnorm_output,rmsnorm_weight,scale,scale_k,scale_v,self_rank,seq_lens,slot,src_scale_k,src_scale_v,step,token_ids,total_indices`,
  `launch_kernarg_abi_semantic_projection schema_candidates=147 missing_schema_candidates=0 descriptor_matches=147 projection_ready_candidates=31 dispatches_with_projection_ready=20 dispatches_without_projection_ready=16 field_schemas=1028 projected_fields=530 missing_fields=492 kind_mismatches=6 unsupported_encodings=0 scalar_narrowing_overflows=0 field_range_overflows=0 projected_kernarg_bytes=20284 ready=false unresolved_runtime_requirements=9`,
  `launch_kernarg_abi_semantic_projection_gaps dispatches_without_projection_ready=16 candidate_abis=92 schema_candidates=92 missing_schema_candidates=0 descriptor_matches=92 projection_ready_candidates=0 primary_missing_schemas=0 primary_descriptor_mismatches=0 primary_missing_model_args=90 primary_kind_mismatches=2 primary_unsupported_encodings=0 primary_scalar_narrowing_overflows=0 primary_field_range_overflows=0 primary_unknown=0 field_schemas=713 projected_fields=271 missing_fields=436 kind_mismatches=6 unsupported_encodings=0 scalar_narrowing_overflows=0 field_range_overflows=0 projected_kernarg_bytes=17676 all_dispatches_have_projection_ready_candidate=false`,
  `launch_kernarg_abi_semantic_projection_missing_schema_symbols: count=0 symbols=`,
  `launch_kernarg_abi_semantic_projection_missing_model_arguments: count=54 names=allreduce_sequence_base,append_count,base_pos,batch_indices,batch_size,block_size,candidate_logits,candidate_token_ids,chunk_len,chunk_len_vec4,chunk_offset,chunk_offset_vec4,chunk_owner,dualpath_cu_chunk_len_vec4,dualpath_cu_owned_chunk_len_vec4,dualpath_sdma_semaphores,eps,gbar,indptr,intermediate,last_page_len,local_allreduce_buffer,local_allreduce_flags,nthreads,num_groups,num_splits,num_tiles,num_wg,p2p_peer_count,partials,peer_allreduce_flag_ptrs,peer_allreduce_ptrs,peer_reduce_staging_ptrs,peer_stride_vec4,persistent_allreduce_ctrl,persistent_allreduce_output,persistent_allreduce_total_ops,physical_blocks,positions,reduce_staging_buffer,reduce_staging_chunk_len,rmsnorm_output,rmsnorm_weight,scale,scale_k,scale_v,self_rank,seq_lens,slot,src_scale_k,src_scale_v,step,token_ids,total_indices`,
  `launch_kernarg_abi_semantic_projection_recommendations recommended_dispatches=14 missing_recommendations=22 recommended_projection_ready=0 recommended_projection_blocked=14 recommended_projection_missing=0 recommended_without_projection_ready=14 dispatches_with_projection_ready=20 dispatches_without_projection_ready=16 all_recommended_projection_ready=false all_dispatches_have_projection_ready_recommendation=false ready_kernarg_bytes=0`,
  `launch_kernarg_abi_semantic_projection_candidate_recommendations recommended_dispatches=20 missing_recommendations=16 projection_ready_candidates=31 source_ambiguous_dispatches=24 recommended_projected_kernarg_bytes=656 all_recommended=false policy=first_projection_ready_candidate_in_host_launcher_order`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_requests requests=20 missing=16 projection_ready_candidates=31 source_ambiguous_dispatches=24 requested_projected_kernarg_bytes=656 applied=0 all_ready=false plan_ready=true policy=first_projection_ready_candidate_in_host_launcher_order`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_ready_ops: count=20 names=layers.0.input_rmsnorm,layers.0.q_proj,layers.0.k_proj,layers.0.v_proj,layers.0.q_rmsnorm,layers.0.k_rmsnorm,layers.0.o_proj,layers.0.attention_residual_rmsnorm,layers.0.router_topk,layers.1.input_rmsnorm,layers.1.q_proj,layers.1.k_proj,layers.1.v_proj,layers.1.q_rmsnorm,layers.1.k_rmsnorm,layers.1.o_proj,layers.1.attention_residual_rmsnorm,layers.1.router_topk,final_rmsnorm,lm_head`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_requested_symbols: count=20 labels=layers.0.input_rmsnorm=rmsnorm_f16,layers.0.q_proj=gemv_f16,layers.0.k_proj=gemv_f16,layers.0.v_proj=gemv_f16,layers.0.q_rmsnorm=rmsnorm_f16,layers.0.k_rmsnorm=rmsnorm_f16,layers.0.o_proj=gemv_f16,layers.0.attention_residual_rmsnorm=add_rmsnorm_f16,layers.0.router_topk=moe_router_topk,layers.1.input_rmsnorm=rmsnorm_f16,layers.1.q_proj=gemv_f16,layers.1.k_proj=gemv_f16,layers.1.v_proj=gemv_f16,layers.1.q_rmsnorm=rmsnorm_f16,layers.1.k_rmsnorm=rmsnorm_f16,layers.1.o_proj=gemv_f16,layers.1.attention_residual_rmsnorm=add_rmsnorm_f16,layers.1.router_topk=moe_router_topk,final_rmsnorm=rmsnorm_f16,lm_head=gemv_f16`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_missing_ops: count=16 names=embed_tokens,layers.0.rope,layers.0.kv_cache_append,layers.0.paged_gqa_attention,layers.0.o_proj_allreduce,layers.0.moe_local_ffn,layers.0.moe_allreduce,layers.0.moe_residual,layers.1.rope,layers.1.kv_cache_append,layers.1.paged_gqa_attention,layers.1.o_proj_allreduce,layers.1.moe_local_ffn,layers.1.moe_allreduce,layers.1.moe_residual,greedy_argmax`,
  `launch_kernarg_abi_capacity_requests requests=14 candidate_requests=61 primary_size_shortfalls=61`,
  `launch_kernel_selection_requests requests=14 missing=22 verified_candidates=68 applied=0 all_ready=false plan_ready=true policy=first_verified_candidate_in_host_launcher_order`,
  `launch_kernel_selection_ready_ops: count=14 names=embed_tokens,layers.0.kv_cache_append,layers.0.paged_gqa_attention,layers.0.o_proj_allreduce,layers.0.attention_residual_rmsnorm,layers.0.router_topk,layers.0.moe_allreduce,layers.1.kv_cache_append,layers.1.paged_gqa_attention,layers.1.o_proj_allreduce,layers.1.attention_residual_rmsnorm,layers.1.router_topk,layers.1.moe_allreduce,greedy_argmax`,
  `launch_kernel_selection_requested_symbols: count=14 labels=embed_tokens=decode_step_embed_rmsnorm_token_f16,layers.0.kv_cache_append=kv_append_paged_fp4,layers.0.paged_gqa_attention=attn_decode_split2_fp4_gqa_paged_groups_meta,layers.0.o_proj_allreduce=reduce_peers,layers.0.attention_residual_rmsnorm=allreduce_direct_residual_rmsnorm_grid,layers.0.router_topk=moe_router_gemv_topk_log_step,layers.0.moe_allreduce=reduce_peers,layers.1.kv_cache_append=kv_append_paged_fp4,layers.1.paged_gqa_attention=attn_decode_split2_fp4_gqa_paged_groups_meta,layers.1.o_proj_allreduce=reduce_peers,layers.1.attention_residual_rmsnorm=allreduce_direct_residual_rmsnorm_grid,layers.1.router_topk=moe_router_gemv_topk_log_step,layers.1.moe_allreduce=reduce_peers,greedy_argmax=argmax_f32_step`,
  `launch_kernel_selection_missing_ops: count=22 names=layers.0.input_rmsnorm,layers.0.q_proj,layers.0.k_proj,layers.0.v_proj,layers.0.q_rmsnorm,layers.0.k_rmsnorm,layers.0.rope,layers.0.o_proj,layers.0.moe_local_ffn,layers.0.moe_residual,layers.1.input_rmsnorm,layers.1.q_proj,layers.1.k_proj,layers.1.v_proj,layers.1.q_rmsnorm,layers.1.k_rmsnorm,layers.1.rope,layers.1.o_proj,layers.1.moe_local_ffn,layers.1.moe_residual,final_rmsnorm,lm_head`,
  `launch_host_launcher_branch_request_ops: count=24 names=layers.0.q_proj,layers.0.k_proj,layers.0.v_proj,layers.0.kv_cache_append,layers.0.paged_gqa_attention,layers.0.o_proj,layers.0.o_proj_allreduce,layers.0.attention_residual_rmsnorm,layers.0.router_topk,layers.0.moe_local_ffn,layers.0.moe_allreduce,layers.1.q_proj,layers.1.k_proj,layers.1.v_proj,layers.1.kv_cache_append,layers.1.paged_gqa_attention,layers.1.o_proj,layers.1.o_proj_allreduce,layers.1.attention_residual_rmsnorm,layers.1.router_topk,layers.1.moe_local_ffn,layers.1.moe_allreduce,lm_head,greedy_argmax`,
  `launch_host_launcher_branch_candidate_symbols: count=24 labels=layers.0.q_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.0.k_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.0.v_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.0.kv_cache_append=kv_append_paged_fp4|kv_append_paged_fp4_from_f16_vf32_heads,layers.0.paged_gqa_attention=attn_decode_split2_fp4_gqa_paged|attn_decode_split2_fp4_gqa_paged_groups_meta|attn_decode_combine_gqa_f16,layers.0.o_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.0.o_proj_allreduce=reduce_peers|broadcast_peers|broadcast_peers_skip0|allreduce_oneshot|scatter_to_staging|gather_reduce_local|reduce_scatter|broadcast_chunk|broadcast_chunk_skip_owner|all_gather|p2p_write|p2p_broadcast|allreduce_dualpath|allreduce_dda_persistent|allreduce_direct_persistent,layers.0.attention_residual_rmsnorm=add_rmsnorm_f16|add_rmsnorm_bf16_residual_f16_out|allreduce_direct_residual_rmsnorm_grid,layers.0.router_topk=moe_router_topk|moe_router_gemv_topk_log_step|moe_router_gemv_topk_log_step_e16_k4096_top8,layers.0.moe_local_ffn=moe_gate_up_swiglu|moe_gate_up_swiglu_slots|moe_gate_up_swiglu_slots_k4096|moe_down_accum|moe_down_accum_slots|moe_down_accum_slots_i1536|moe_down_accum_slots_i512,layers.0.moe_allreduce=reduce_peers|broadcast_peers|broadcast_peers_skip0|allreduce_oneshot|scatter_to_staging|gather_reduce_local|reduce_scatter|broadcast_chunk|broadcast_chunk_skip_owner|all_gather|p2p_write|p2p_broadcast|allreduce_dualpath|allreduce_dda_persistent|allreduce_direct_persistent,layers.1.q_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.1.k_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.1.v_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.1.kv_cache_append=kv_append_paged_fp4|kv_append_paged_fp4_from_f16_vf32_heads,layers.1.paged_gqa_attention=attn_decode_split2_fp4_gqa_paged|attn_decode_split2_fp4_gqa_paged_groups_meta|attn_decode_combine_gqa_f16,layers.1.o_proj=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,layers.1.o_proj_allreduce=reduce_peers|broadcast_peers|broadcast_peers_skip0|allreduce_oneshot|scatter_to_staging|gather_reduce_local|reduce_scatter|broadcast_chunk|broadcast_chunk_skip_owner|all_gather|p2p_write|p2p_broadcast|allreduce_dualpath|allreduce_dda_persistent|allreduce_direct_persistent,layers.1.attention_residual_rmsnorm=add_rmsnorm_f16|add_rmsnorm_bf16_residual_f16_out|allreduce_direct_residual_rmsnorm_grid,layers.1.router_topk=moe_router_topk|moe_router_gemv_topk_log_step|moe_router_gemv_topk_log_step_e16_k4096_top8,layers.1.moe_local_ffn=moe_gate_up_swiglu|moe_gate_up_swiglu_slots|moe_gate_up_swiglu_slots_k4096|moe_down_accum|moe_down_accum_slots|moe_down_accum_slots_i1536|moe_down_accum_slots_i512,layers.1.moe_allreduce=reduce_peers|broadcast_peers|broadcast_peers_skip0|allreduce_oneshot|scatter_to_staging|gather_reduce_local|reduce_scatter|broadcast_chunk|broadcast_chunk_skip_owner|all_gather|p2p_write|p2p_broadcast|allreduce_dualpath|allreduce_dda_persistent|allreduce_direct_persistent,lm_head=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,greedy_argmax=argmax_f32_step|argmax_f32_token_ids_write_candidate|argmax_f32_token_ids_write_candidate_n1187`,
  `launch_host_launcher_branch_unresolved_candidate_symbols: count=40 symbols=add_rmsnorm_bf16_residual_f16_out,add_rmsnorm_f16,all_gather,allreduce_dda_persistent,allreduce_direct_persistent,allreduce_direct_residual_rmsnorm_grid,allreduce_dualpath,allreduce_oneshot,argmax_f32_step,argmax_f32_token_ids_write_candidate,argmax_f32_token_ids_write_candidate_n1187,attn_decode_combine_gqa_f16,attn_decode_split2_fp4_gqa_paged,attn_decode_split2_fp4_gqa_paged_groups_meta,broadcast_chunk,broadcast_chunk_skip_owner,broadcast_peers,broadcast_peers_skip0,gather_reduce_local,gemv_f16,gemv_f16_k8192,gemv_f16_step,gemv_f16_step_k4096,kv_append_paged_fp4,kv_append_paged_fp4_from_f16_vf32_heads,moe_down_accum,moe_down_accum_slots,moe_down_accum_slots_i1536,moe_down_accum_slots_i512,moe_gate_up_swiglu,moe_gate_up_swiglu_slots,moe_gate_up_swiglu_slots_k4096,moe_router_gemv_topk_log_step,moe_router_gemv_topk_log_step_e16_k4096_top8,moe_router_topk,p2p_broadcast,p2p_write,reduce_peers,reduce_scatter,scatter_to_staging`,
  `launch_execution_requests request_plans=10 typed_steps=10 typed_step_descriptors=10 typed_live_aql_proof_steps=2 component_pending=378 blockers=9`,
  `launch_execution_request_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_execution_request_pending_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_execution_live_aql_proof_surface_plans: count=2 names=queue_reservation_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_execution_live_aql_proof_kinds: count=2 labels=batch_reservation_plan,materialized_packet_plan`,
  `launch_execution_live_aql_proof_inputs: count=2 labels=KfdQueueLiveAqlBatchReservationPlanInput,KfdQueueLiveAqlMaterializedPacketPlanInput`,
  `launch_execution_live_aql_validation_methods: count=2 labels=KfdQueueLiveAqlBatchReservationPlanProof::validate_ready,KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready`,
  `launch_execution_request_receipt fingerprint=2d09ee2b3bd4263fcbb65613577771567435eab182b802903dd845b0850f6c31 lines=220`,
  `launch_submission_gate blockers=11 execution_blockers=9 component_pending=378 proof_validation_pending=2`,
  `launch_submission_gate_blockers: count=11 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization,runtime_request_components,live_aql_proof_validation`,
  `launch_submission_gate_receipt fingerprint=3d8a40146f779f7415c8138f2381e31d83fda0689261aae8df637bf97dd5a55d lines=65`,
  `launch_submission_blocker_report_blockers: count=11 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization,runtime_request_components,live_aql_proof_validation`,
  `launch_submission_blocker_report_receipt fingerprint=88091d48ae40a5fbcc4b0159cccc074255388fef4eda0565832da1188f606afb lines=132`,
  `launch_submission_prerequisites prerequisites=10 live_aql_proof_inputs=2 live_aql_validation_methods=2`,
  `launch_submission_prerequisite_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_unsatisfied_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_next_action_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_next_action_labels: count=10 labels=apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,validate_live_aql_proof,apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,validate_live_aql_proof`,
  `launch_submission_prerequisite_runtime_component_next_action_plans: count=8 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan`,
  `launch_submission_prerequisite_live_aql_proof_validation_next_action_plans: count=2 names=queue_reservation_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_next_action_inputs: count=10 labels=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,KfdQueueLiveAqlBatchReservationPlanInput,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,KfdQueueLiveAqlMaterializedPacketPlanInput`,
  `launch_submission_prerequisite_next_action_live_aql_proof_kinds: count=2 labels=batch_reservation_plan,materialized_packet_plan`,
  `launch_submission_prerequisite_live_aql_proof_plans: count=2 names=queue_reservation_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_live_aql_proof_kinds: count=2 labels=batch_reservation_plan,materialized_packet_plan`,
  `launch_submission_prerequisite_live_aql_proof_inputs: count=2 labels=KfdQueueLiveAqlBatchReservationPlanInput,KfdQueueLiveAqlMaterializedPacketPlanInput`,
  `launch_submission_prerequisite_live_aql_validation_methods: count=2 labels=KfdQueueLiveAqlBatchReservationPlanProof::validate_ready,KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready`,
  `launch_submission_prerequisite_plan_receipt fingerprint=4d49217c527b339a78f91138bdb8398c6532106dfbcedafd5811745cf1b87d12 lines=249`,
  `launch_executable_semantic_projection ready=false candidates=147 schema_candidates=147 missing_schema_candidates=0 descriptor_matches=147 projection_ready_candidates=31 dispatches_with_ready=20 dispatches_without_ready=16 field_schemas=1028 projected_fields=530 missing_fields=492 kind_mismatches=6 unsupported_encodings=0 scalar_narrowing_overflows=0 field_range_overflows=0 projected_kernarg_bytes=20284`,
  `launch_executable_semantic_projection_selection requests=20 missing=16 requested_projected_kernarg_bytes=656 applied=0 plan_ready=true`,
  `launch_executable_blockers: count=9 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization`,
  `launch_executable_requirements: count=9 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization`,
  and `launch_executable ready=false blockers=9 kernel_argument_abi_capacity_requests=14`.
- Custom plugin example:
  `graph tensors=6 ops=3 stages=3`, `lowering gap_ops=0`,
  `model_api_vocabulary primitive_kinds=12 stage_kinds=5`,
  `plugin_inspection consistent=true accepted=true`,
  `plugin_summary receipt_fingerprint=<64-hex> accepted=true static_ready=true compatibility_issues=0`,
  `catalog_capabilities primitive_kinds=12 cases=22 native_gpu_cases=16 fused_native_gpu_cases=1 gap_cases=5 parameterized=true`,
  `plugin_manifest fingerprint=<64-hex> primitive_kinds=3 stage_kinds=3 launch_steps=10 live_aql_proof_steps=2 static_ready=true live_execution_supported=false`,
  `plugin_compatibility accepted=true issues=0 target_matches=true fingerprint_matches=true static_metadata_ready=true live_execution_supported=false`,
  `launch_kernarg_abi dispatches_with_verified=2 dispatches_without_verified=1`,
  `launch_kernarg_abi_gaps dispatches_without_verified=1 candidate_abis=4 size_shortfall_candidates=4 primary_size_shortfalls=4`,
  `launch_kernarg_abi_semantics schemas=51 schema_candidates=8 missing_schema_candidates=0 descriptor_matches=8 verified_candidates=0 dispatches_with_verified=0 dispatches_without_verified=3 field_schemas=51 verified_fields=8 missing_fields=20 field_mismatches=23 extra_arguments=15 ready=false unresolved_runtime_requirements=8`,
  `launch_kernarg_abi_semantic_gaps dispatches_without_verified=3 candidate_abis=8 schema_candidates=8 missing_schema_candidates=0 descriptor_matches=8 verified_candidates=0 primary_missing_schemas=0 primary_descriptor_mismatches=0 primary_missing_model_args=6 primary_field_mismatches=2 primary_extra_arguments=0 primary_size_shortfalls=0 primary_unknown=0 field_schemas=51 verified_fields=8 missing_fields=20 field_mismatches=23 extra_arguments=15 all_dispatches_have_semantic_verified_candidate=false`,
  `launch_kernarg_abi_semantic_missing_schema_symbols: count=0 symbols=`,
  `launch_kernarg_abi_semantic_missing_model_arguments: count=13 names=base_pos,block_size,candidate_logits,candidate_token_ids,eps,last_page_len,positions,rmsnorm_output,rmsnorm_weight,seq_lens,slot,step,token_ids`,
  `launch_kernarg_abi_semantic_projection schema_candidates=8 missing_schema_candidates=0 descriptor_matches=8 projection_ready_candidates=2 dispatches_with_projection_ready=1 dispatches_without_projection_ready=2 field_schemas=51 projected_fields=31 missing_fields=20 kind_mismatches=0 unsupported_encodings=0 scalar_narrowing_overflows=0 field_range_overflows=0 projected_kernarg_bytes=340 ready=false unresolved_runtime_requirements=9`,
  `launch_kernarg_abi_semantic_projection_gaps dispatches_without_projection_ready=2 candidate_abis=4 schema_candidates=4 missing_schema_candidates=0 descriptor_matches=4 projection_ready_candidates=0 primary_missing_schemas=0 primary_descriptor_mismatches=0 primary_missing_model_args=4 primary_kind_mismatches=0 primary_unsupported_encodings=0 primary_scalar_narrowing_overflows=0 primary_field_range_overflows=0 primary_unknown=0 field_schemas=31 projected_fields=13 missing_fields=18 kind_mismatches=0 unsupported_encodings=0 scalar_narrowing_overflows=0 field_range_overflows=0 projected_kernarg_bytes=204 all_dispatches_have_projection_ready_candidate=false`,
  `launch_kernarg_abi_semantic_projection_missing_schema_symbols: count=0 symbols=`,
  `launch_kernarg_abi_semantic_projection_missing_model_arguments: count=13 names=base_pos,block_size,candidate_logits,candidate_token_ids,eps,last_page_len,positions,rmsnorm_output,rmsnorm_weight,seq_lens,slot,step,token_ids`,
  `launch_kernarg_abi_semantic_projection_recommendations recommended_dispatches=2 missing_recommendations=1 recommended_projection_ready=0 recommended_projection_blocked=2 recommended_projection_missing=0 recommended_without_projection_ready=2 dispatches_with_projection_ready=1 dispatches_without_projection_ready=2 all_recommended_projection_ready=false all_dispatches_have_projection_ready_recommendation=false ready_kernarg_bytes=0`,
  `launch_kernarg_abi_semantic_projection_candidate_recommendations recommended_dispatches=1 missing_recommendations=2 projection_ready_candidates=2 source_ambiguous_dispatches=2 recommended_projected_kernarg_bytes=32 all_recommended=false policy=first_projection_ready_candidate_in_host_launcher_order`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_requests requests=1 missing=2 projection_ready_candidates=2 source_ambiguous_dispatches=2 requested_projected_kernarg_bytes=32 applied=0 all_ready=false plan_ready=true policy=first_projection_ready_candidate_in_host_launcher_order`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_ready_ops: count=1 names=lm_head`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_requested_symbols: count=1 labels=lm_head=gemv_f16`,
  `launch_kernarg_abi_semantic_projection_candidate_selection_missing_ops: count=2 names=embed_tokens,sample_argmax`,
  `launch_kernarg_abi_capacity_requests requests=4 candidate_requests=4 primary_size_shortfalls=4`,
  `launch_kernel_selection_requests requests=2 missing=1 verified_candidates=4 applied=0 all_ready=false plan_ready=true policy=first_verified_candidate_in_host_launcher_order`,
  `launch_kernel_selection_ready_ops: count=2 names=embed_tokens,sample_argmax`,
  `launch_kernel_selection_requested_symbols: count=2 labels=embed_tokens=decode_step_embed_rmsnorm_token_f16,sample_argmax=argmax_f32_step`,
  `launch_kernel_selection_missing_ops: count=1 names=lm_head`,
  `launch_host_launcher_branch_request_ops: count=2 names=lm_head,sample_argmax`,
  `launch_host_launcher_branch_candidate_symbols: count=2 labels=lm_head=gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096,sample_argmax=argmax_f32_step|argmax_f32_token_ids_write_candidate|argmax_f32_token_ids_write_candidate_n1187`,
  `launch_host_launcher_branch_unresolved_candidate_symbols: count=7 symbols=argmax_f32_step,argmax_f32_token_ids_write_candidate,argmax_f32_token_ids_write_candidate_n1187,gemv_f16,gemv_f16_k8192,gemv_f16_step,gemv_f16_step_k4096`,
  `launch_execution_requests request_plans=10 typed_steps=10 typed_step_descriptors=10 typed_live_aql_proof_steps=2 component_pending=41 blockers=9`,
  `launch_execution_request_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_execution_request_pending_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_execution_live_aql_proof_surface_plans: count=2 names=queue_reservation_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_execution_live_aql_proof_kinds: count=2 labels=batch_reservation_plan,materialized_packet_plan`,
  `launch_execution_live_aql_proof_inputs: count=2 labels=KfdQueueLiveAqlBatchReservationPlanInput,KfdQueueLiveAqlMaterializedPacketPlanInput`,
  `launch_execution_live_aql_validation_methods: count=2 labels=KfdQueueLiveAqlBatchReservationPlanProof::validate_ready,KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready`,
  `launch_execution_request_receipt fingerprint=dea2b05ca5f6fd6da0efc6c494f75562ad5fe24a2dbc13ce505ef34ea0036323 lines=220`,
  `launch_submission_gate blockers=11 execution_blockers=9 component_pending=41 proof_validation_pending=2`,
  `launch_submission_gate_blockers: count=11 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization,runtime_request_components,live_aql_proof_validation`,
  `launch_submission_gate_receipt fingerprint=e1012ae50e9c1f82b609cefc9c562d69ffd8454b7e1d0233e58392404de596c2 lines=65`,
  `launch_submission_blocker_report_blockers: count=11 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization,runtime_request_components,live_aql_proof_validation`,
  `launch_submission_blocker_report_receipt fingerprint=73212566f3e2187e9ef03cbf6c0f3ec21487d456f2d74cbff64e9585b2a6f526 lines=132`,
  `launch_submission_prerequisites prerequisites=10 live_aql_proof_inputs=2 live_aql_validation_methods=2`,
  `launch_submission_prerequisite_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_unsatisfied_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_next_action_plans: count=10 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,queue_reservation_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_next_action_labels: count=10 labels=apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,validate_live_aql_proof,apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,apply_runtime_request_component,validate_live_aql_proof`,
  `launch_submission_prerequisite_runtime_component_next_action_plans: count=8 names=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan`,
  `launch_submission_prerequisite_live_aql_proof_validation_next_action_plans: count=2 names=queue_reservation_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_next_action_inputs: count=10 labels=code_object_load_request_plan,code_object_base_binding_request_plan,completion_signal_binding_request_plan,KfdQueueLiveAqlBatchReservationPlanInput,kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,kernel_argument_abi_semantic_projection_candidate_selection_request_plan,host_launcher_branch_resolution_request_plan,KfdQueueLiveAqlMaterializedPacketPlanInput`,
  `launch_submission_prerequisite_next_action_live_aql_proof_kinds: count=2 labels=batch_reservation_plan,materialized_packet_plan`,
  `launch_submission_prerequisite_live_aql_proof_plans: count=2 names=queue_reservation_request_plan,aql_live_relocation_binding_request_plan`,
  `launch_submission_prerequisite_live_aql_proof_kinds: count=2 labels=batch_reservation_plan,materialized_packet_plan`,
  `launch_submission_prerequisite_live_aql_proof_inputs: count=2 labels=KfdQueueLiveAqlBatchReservationPlanInput,KfdQueueLiveAqlMaterializedPacketPlanInput`,
  `launch_submission_prerequisite_live_aql_validation_methods: count=2 labels=KfdQueueLiveAqlBatchReservationPlanProof::validate_ready,KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready`,
  `launch_submission_prerequisite_plan_receipt fingerprint=28878a5e8a05eb5e53c4eb8adf8fcda66d0692481fa4d2f59b46f3f1fe3b6a91 lines=249`,
  `launch_executable_semantic_projection ready=false candidates=8 schema_candidates=8 missing_schema_candidates=0 descriptor_matches=8 projection_ready_candidates=2 dispatches_with_ready=1 dispatches_without_ready=2 field_schemas=51 projected_fields=31 missing_fields=20 kind_mismatches=0 unsupported_encodings=0 scalar_narrowing_overflows=0 field_range_overflows=0 projected_kernarg_bytes=340`,
  `launch_executable_semantic_projection_selection requests=1 missing=2 requested_projected_kernarg_bytes=32 applied=0 plan_ready=true`,
  `launch_executable_blockers: count=9 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization`,
  `launch_executable_requirements: count=9 requirements=kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization`,
  and `launch_executable ready=false blockers=9 kernel_argument_abi_capacity_requests=4`.

Run the focused API tests with:

```bash
cargo test -p mainarch-core model_api
```

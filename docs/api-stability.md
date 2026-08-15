# API stability

`mainarch` is pre-1.0. The repository can still change quickly, but model-facing
changes should make it clear which surfaces are stable enough for external
experiments and which surfaces are scaffolding for the next runtime layer.

## Stability Tiers

### Public model authoring contract

These surfaces are intended for external model-definition experiments:

- `mainarch_core::model_api::prelude`, the curated import surface for the items
  below
- `ModelDefinition`
- `ModelPrimitiveApi`
- primitive descriptors passed through `PrimitiveOp`
- tensor, stage, dtype, placement, cache, and collective metadata types needed to
  build a `ModelPrimitiveGraph`
- graph validation and the composed readiness report entrypoints used by the
  public examples and CLI selftest
- `inspect_model_plugin`, `ModelPluginInspectionReport`, plugin manifests, and
  plugin compatibility reports
- `ModelPluginInspectionReport::assert_consistent`
- `ModelPluginCompatibilityReport::assert_consistent`
- `ModelPluginInspectionSummary` and `ModelPluginRejectionReport`, including
  their deterministic receipt text and receipt fingerprint helpers plus
  `ModelPluginRejectionReport::assert_consistent`
- `ModelPluginStaticHandoffReceipt`, including deterministic receipt text,
  receipt fingerprint, consistency helpers, and
  `ModelPluginStaticHandoffReceipt::unresolved_runtime_requirement_names()` plus
  `ModelPluginStaticHandoffReceipt::has_unresolved_runtime_requirement(...)`
  plus `ModelPluginStaticHandoffReceipt::is_non_executing_boundary()` and
  `ModelPluginStaticHandoffReceipt::assert_non_executing_boundary()` for accepted
  static-ready plugins after passing the same static-handoff consistency checks;
  static handoff builders assert the derived non-executable launch-readiness
  report, non-submitting launch request plan, and non-executing handoff receipt
  before returning
- primitive and stage vocabulary descriptor tables
- model primitive lowering catalog capability descriptors
- checkpoint binding plan handles re-exported by the prelude for metadata-only
  checkpoint key and safetensors preflights: `ModelCheckpointBindingPlan`
- CPU-only launch metadata handles re-exported by the prelude for external
  launch-readiness checks: `CodeObjectInfo`,
  `CodeObjectKernelSetValidation`, `KernelInfo`,
  `RuntimeLaunchKernelArgumentAbiSemanticEncoding`, `MAINARCH_KERNELS_GFX950`,
  and `AQL_PACKET_BYTES`

Changes to this tier should preserve source compatibility where practical. A
breaking change needs a migration note in the same pull request, updated public
examples, and focused model API tests.

The current code-level descriptor is `MODEL_API_CONTRACT`:
`mainarch-model-api version=0.1.0 stability=pre1-static-metadata
live_execution_supported=false`.
`ModelApiContractInfo::receipt_lines()`, `receipt_text()`, and
`receipt_fingerprint()` expose the same descriptor as a deterministic six-line
contract receipt that external model packages can pin before building a graph.

### Static runtime metadata contract

These reports are stable enough for tooling to inspect, but are not executable
runtime APIs yet:

- tensor storage, access, lifetime, binding, checkpoint, execution, slot,
  dispatch-intent, stage-resource, stage-bundle, and stage-dispatch manifests
- metadata binding templates and preflight receipts
- `ModelRuntimeMetadataAdmissionReport::assert_consistent`, for checking
  target provenance and embedded dispatch slot-binding snapshot consistency
  before treating a zero-issue admission report as a metadata handoff
- launch preflight, kernel requirement, kernel metadata, kernarg layout,
  kernel-argument ABI verification/receipt/schema-request/semantic
  comparison/semantic gap/semantic projection/projection gap/projection
  recommendation alignment/projection-aware recommendation/projection-aware
  selection request, missing-argument requirement diagnostics, and field
  mismatch/blocker diagnostics,
  kernel-candidate recommendation,
  kernel-candidate selection and projection-aware selection op-name,
  op/kernel-symbol tuple, and display-label helpers, host-launcher branch
  request op-name and candidate-symbol set/label helpers,
  staging, completion-signal, queue, dispatch-geometry, AQL packet-template,
  relocation, materialization, execution-readiness, execution-request,
  execution-request receipt, typed execution-request step descriptors and
  request-plan lookup helpers, live-AQL proof step descriptors, proof-surface,
  submission-gate,
  submission-gate receipt, submission-blocker report, submission-blocker report
  receipt, submission-prerequisite plan, and submission-prerequisite plan receipt
- `MainarchPrimitiveLoweringCatalog::code_object_kernel_coverage_report(...)`
  plus
  `ModelPrimitiveLoweringCatalogCodeObjectKernelCoverageReport::assert_complete()`,
  a CPU-only static metadata guard that maps every non-gap catalog case
  entrypoint to its required code-object kernel symbols and rejects unmapped
  entrypoints or missing bundled gfx950 kernel descriptors without validating
  kernel ABI semantics or authorizing live execution
- `MainarchPrimitiveLoweringCatalog::abi_registry_coverage_report(...)` plus
  `ModelPrimitiveLoweringCatalogAbiRegistryCoverageReport::assert_complete()`,
  a CPU-only static metadata guard that joins every catalog-required bundled
  code-object kernel symbol to the named ABI schema registry and semantic ABI
  schema registry, rejects missing registry rows, and checks code-object,
  named-schema, and semantic-schema kernarg size/alignment agreement without
  proving full argument semantics, packet lowering, or live execution
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_request_plan(...)`,
  a report-level CPU-only convenience helper that rebuilds the default
  unresolved launch-request fixture from an accepted inspection report and
  asserts the returned request plan's non-submitting boundary
- `ModelPluginInspectionReport::is_static_handoff_ready()` and
  `ModelPluginInspectionReport::assert_static_handoff_ready()`, a report-level
  guard for accepted, static-ready plugin inspections before deriving CPU-only
  static handoff or launch request fixtures; those default fixture builders use
  the guard as their report-level precondition after passing the same inspection consistency checks
- `ModelRuntimeLaunchExecutionRequestPlan::is_non_submitting_boundary()` and
  `ModelRuntimeLaunchExecutionRequestPlan::assert_non_submitting_boundary()`,
  for checking that execution-request metadata has no live AQL submitting
  proof-surface rows and no live queue-mutating component rows after passing the same request-plan consistency checks before deriving submission receipts
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_gate(...)`,
  a report-level CPU-only convenience helper that derives the default blocked
  submission-gate fixture from the same unresolved launch-request path and
  asserts the returned gate's non-submitting boundary
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_blocker_report(...)`,
  a report-level CPU-only convenience helper that derives the default expanded
  blocker-report fixture from the same blocked submission-gate path and asserts
  the returned blocker report's non-submitting boundary
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_prerequisite_plan(...)`,
  a report-level CPU-only convenience helper that derives the default
  per-request prerequisite-plan fixture from the same unresolved launch-request
  path and asserts the returned prerequisite plan's non-submitting boundary
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(...)`,
  a report-level CPU-only convenience helper that maps typed live-AQL proof
  validation receipts onto the default launch request's proof-surface worklist
  and asserts the returned application plan's non-submitting boundary
- `ModelRuntimeLaunchLiveAqlProofValidationApplicationPlan::is_non_submitting_boundary()`
  and
  `ModelRuntimeLaunchLiveAqlProofValidationApplicationPlan::assert_non_submitting_boundary()`,
  for checking that typed proof-validation overlay metadata has no live AQL
  submitting validation rows and no live queue-mutating validation rows after
  passing the same application-plan consistency checks
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_runtime_request_component_application_plan(...)`,
  a report-level CPU-only convenience helper that derives the default
  runtime-request component application worklist after mapping typed live-AQL
  proof validation receipts and asserts the intermediate proof/prerequisite
  boundaries plus the returned application worklist's non-submitting boundary
- `ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::is_non_submitting_boundary()`
  and
  `ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::assert_non_submitting_boundary()`,
  for checking that runtime-request component application worklists have no live
  AQL submitting application rows and no live queue-mutating application rows
  after passing the same application-plan consistency checks
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(...)`,
  a report-level CPU-only convenience helper that maps caller-supplied runtime
  component application receipts onto that default application worklist
- `RuntimeLaunchRuntimeRequestComponentApplicationReceipt::is_non_submitting_boundary()`
  and
  `RuntimeLaunchRuntimeRequestComponentApplicationReceipt::assert_non_submitting_boundary()`,
  for checking an individual caller-supplied runtime component application
  receipt has no live AQL submission side effect and no live queue mutation
  after passing receipt consistency checks
- `ModelRuntimeLaunchRuntimeRequestComponentApplicationReceiptPlan::is_non_submitting_boundary()`
  and
  `ModelRuntimeLaunchRuntimeRequestComponentApplicationReceiptPlan::assert_non_submitting_boundary()`,
  for checking that caller-supplied runtime component receipt overlays have no
  live AQL submitting receipt rows and no live queue-mutating receipt rows after
  passing the same receipt-plan consistency checks; the report-level synthetic
  CPU path asserts this boundary before deriving downstream readiness metadata
  from caller-supplied component receipts
- `ModelRuntimeLaunchExecutionReadinessReport::is_non_executable_boundary()`
  and `ModelRuntimeLaunchExecutionReadinessReport::assert_non_executable_boundary()`,
  for checking that launch-readiness metadata remains blocked,
  non-dispatchable, and free of live resource binding/application counts after
  passing the same report consistency checks before deriving execution request
  plans
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(...)`,
  a report-level CPU-only convenience helper that derives the default
  execution-readiness blocker resolution worklist after proof-validation and
  runtime-component receipt overlays
- `ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan::is_non_submitting_boundary()`
  and
  `ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan::assert_non_submitting_boundary()`,
  for checking that execution-readiness resolution worklists have no embedded
  resolution receipt rows and still claim neither execution readiness nor
  submission readiness after passing the same worklist consistency checks
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(...)`,
  a report-level CPU-only convenience helper that maps caller-supplied
  execution-readiness blocker resolution receipts onto that default resolution
  worklist
- `RuntimeLaunchExecutionReadinessBlockerResolutionReceipt::is_non_submitting_boundary()`
  and
  `RuntimeLaunchExecutionReadinessBlockerResolutionReceipt::assert_non_submitting_boundary()`,
  for checking an individual caller-supplied readiness-resolution receipt has
  no live AQL submission side effect and no live queue mutation after passing
  receipt consistency checks
- `ModelRuntimeLaunchExecutionReadinessBlockerResolutionReceiptPlan::is_non_submitting_boundary()`
  and
  `ModelRuntimeLaunchExecutionReadinessBlockerResolutionReceiptPlan::assert_non_submitting_boundary()`,
  for checking that caller-supplied readiness-resolution receipt overlays have
  no live AQL submitting receipt rows and no live queue-mutating receipt rows
  after passing the same receipt-plan consistency checks; the report-level
  synthetic CPU prerequisite/gate/blocker helpers assert this boundary before
  deriving downstream submission metadata from caller-supplied resolution
  receipts
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(...)`,
  a report-level CPU-only convenience helper that overlays caller-supplied
  execution-readiness blocker resolution receipts onto the derived default
  prerequisite worklist after proof-validation and runtime-component receipt
  overlays and asserts the returned prerequisite plan's non-submitting boundary
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(...)`,
  a report-level CPU-only convenience helper that turns that caller-supplied
  receipt overlay into the corresponding submission-gate metadata and asserts
  the returned gate's non-submitting boundary
- `ModelPluginInspectionReport::synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(...)`,
  a report-level CPU-only convenience helper that expands that caller-supplied
  receipt-overlay submission gate into blocker-report metadata and asserts the
  returned blocker report's non-submitting boundary
- `ModelRuntimeLaunchExecutionRequestPlan::synthetic_cpu_resolved_submission_prerequisite_plan(...)`,
  a CPU-only receipt-overlay convenience helper that returns the fully resolved
  prerequisite worklist for deterministic handoff fixtures; it is not a live
  execution admission path
- `ModelRuntimeLaunchExecutionRequestPlan::synthetic_cpu_resolved_submission_gate(...)`,
  a CPU-only receipt-overlay convenience helper for deterministic handoff
  fixtures that derives the zero-blocker gate from the resolved prerequisite
  worklist; it is not a live execution admission path
- `ModelRuntimeLaunchExecutionRequestPlan::synthetic_cpu_resolved_submission_blocker_report(...)`,
  a CPU-only convenience helper that expands the deterministic resolved
  submission gate into zero-blocker blocker-report metadata
- `ModelRuntimeLaunchSubmissionPrerequisitePlan::is_non_submitting_boundary()`
  and
  `ModelRuntimeLaunchSubmissionPrerequisitePlan::assert_non_submitting_boundary()`,
  for checking that blocked or resolved prerequisite metadata still has no live
  AQL submission side-effect next actions, no live AQL submitting prerequisite
  rows, and no live queue mutation prerequisite rows after passing the same prerequisite-plan consistency checks
- `ModelRuntimeLaunchSubmissionGate::is_non_submitting_boundary()` and
  `ModelRuntimeLaunchSubmissionGate::assert_non_submitting_boundary()`, plus
  `ModelRuntimeLaunchSubmissionBlockerReport::is_non_submitting_boundary()` and
  `ModelRuntimeLaunchSubmissionBlockerReport::assert_non_submitting_boundary()`,
  for checking that blocked or zero-blocker submission metadata still has no
  live AQL submission side effects and no live queue mutation after passing the same submission-gate consistency checks and passing the same blocker-report consistency checks
- `ModelPluginInspectionReport::synthetic_cpu_resolved_submission_prerequisite_plan(...)`,
  a report-level CPU-only convenience helper that rebuilds the default launch
  request fixture and delegates to the execution-request resolved prerequisite
  helper while preserving the returned prerequisite plan's non-submitting
  boundary
- `ModelPluginInspectionReport::synthetic_cpu_resolved_submission_gate(...)`, a
  report-level CPU-only convenience helper that rebuilds the default launch
  request fixture and delegates to the execution-request resolved gate helper
  while asserting the returned gate's non-submitting boundary
- `ModelPluginInspectionReport::synthetic_cpu_resolved_submission_blocker_report(...)`,
  a report-level CPU-only convenience helper that exposes the matching
  zero-blocker blocker-report metadata for that deterministic handoff fixture
  while asserting the returned blocker report's non-submitting boundary
- model plugin static handoff receipts, including the
  `manifest.receipt_fingerprint` and `compatibility.receipt_fingerprint`
  bindings to the full plugin manifest and compatibility receipts plus ordered
  `launch_execution.unresolved_runtime_requirements.*` labels for the live
  execution requirements that still block the handoff
- `ModelPluginInspectionReport::synthetic_cpu_static_handoff_receipt(...)`, a
  CPU-only convenience helper for deterministic static handoff fixtures that
  delegates to the explicit static handoff receipt builder with default bundled
  gfx950 fixture inputs
- `ModelRuntimeSlotDevicePointerLifetimePlan::kfd_allocation_residency_request_plan(...)`
  and
  `ModelGraphReadinessReport::runtime_slot_kfd_allocation_residency_request_plan(...)`,
  CPU-only KFD allocation/residency request metadata that maps CPU-validated
  slot lifetimes to deterministic allocation kinds, KFD flag intent, resident
  GPU ID targets, and readiness guards while keeping
  `allocation_performed=false`, `residency_proven=false`, and
  `live_execution_supported=false`
- `ModelRuntimeSlotKfdAllocationResidencyRequestPlan::kfd_vm_acquire_request_plan()`
  and `ModelGraphReadinessReport::runtime_kfd_vm_acquire_request_plan(...)`,
  CPU-only KFD VM-acquire request metadata that maps resident KFD GPU IDs to
  deterministic `/dev/kfd`, `AMDKFD_IOC_ACQUIRE_VM`, KFD fd, and DRM render fd
  precondition rows while keeping `kfd_fd_bound_count=0`,
  `drm_fd_bound_count=0`, `vm_acquire_performed_count=0`,
  `all_vms_acquired=false`, and `live_execution_supported=false`
- `ModelRuntimeSlotKfdAllocationResidencyRequestPlan::kfd_alloc_memory_request_plan(...)`
  and `ModelGraphReadinessReport::runtime_kfd_alloc_memory_request_plan(...)`,
  CPU-only KFD alloc-memory ioctl request metadata that maps each slot
  allocation/residency request to deterministic `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`
  argument rows with selected allocation GPU ID, VA, size, primary flags, and
  optional fallback flags while keeping `kfd_fd_bound_count=0`,
  `vm_acquire_performed_count=0`, `allocation_performed_count=0`,
  `handle_bound_count=0`, `mmap_offset_bound_count=0`,
  `allocation_performed=false`, and `live_execution_supported=false`
- `ModelRuntimeKfdAllocMemoryRequestPlan::kfd_alloc_memory_result_binding_plan(...)`
  and
  `ModelGraphReadinessReport::runtime_kfd_alloc_memory_result_binding_plan(...)`,
  CPU-only caller-supplied alloc-memory result receipt validation that checks
  slot, tensor, allocation GPU ID, VA, size, primary/fallback flags, nonzero
  handles, and mmap-offset binding against the deterministic
  `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` request rows without opening `/dev/kfd`,
  binding file descriptors, calling KFD ioctls, allocating memory, mapping
  buffers, submitting AQL, executing kernels, or changing
  `live_execution_supported=false`
- `ModelRuntimeKfdAllocMemoryRequestPlan::kfd_map_memory_request_plan()` and
  `ModelGraphReadinessReport::runtime_kfd_map_memory_request_plan(...)`,
  CPU-only KFD map-memory ioctl request metadata that maps each alloc-memory
  request row to deterministic `AMDKFD_IOC_MAP_MEMORY_TO_GPU` static argument
  metadata with resident GPU IDs and `n_devices` while keeping
  `handle_bound_count=0`, `device_ids_array_bound_count=0`,
  `map_memory_performed_count=0`, `map_memory_success_count=0`,
  `all_live_request_args_bound=false`, `all_request_args_ready=false`,
  `map_memory_performed=false`, and `live_execution_supported=false`
- `ModelRuntimeKfdMapMemoryRequestPlan::kfd_map_memory_argument_binding_plan(...)`
  and
  `ModelGraphReadinessReport::runtime_kfd_map_memory_argument_binding_plan(...)`,
  CPU-only caller-supplied map-memory argument binding that validates
  alloc-memory result handles plus nonzero `device_ids_array_ptr` bindings and
  resident GPU ID array values against deterministic
  `AMDKFD_IOC_MAP_MEMORY_TO_GPU` request rows without opening `/dev/kfd`,
  binding file descriptors, calling KFD ioctls, allocating memory, allocating or
  pinning device-ID arrays, mapping buffers, updating `n_success`, submitting
  AQL, executing kernels, or changing `live_execution_supported=false`
- `ModelRuntimeKfdMapMemoryArgumentBindingPlan::kfd_map_memory_result_binding_plan(...)`
  and
  `ModelGraphReadinessReport::runtime_kfd_map_memory_result_binding_plan(...)`,
  CPU-only caller-supplied map-memory result receipt validation that checks
  slot, tensor, map args handle, `device_ids_array_ptr`, `n_devices`, and
  successful `n_success` coverage for every resident GPU ID without opening
  `/dev/kfd`, binding file descriptors, calling KFD ioctls, allocating memory,
  allocating or pinning device-ID arrays, mapping or unmapping buffers, proving
  hardware residency beyond caller-supplied receipt counts, submitting AQL,
  executing kernels, or changing `live_execution_supported=false`
- `ModelRuntimeSlotKfdAllocationResidencyRequestPlan::kfd_residency_binding_plan(...)`
  and
  `ModelGraphReadinessReport::runtime_slot_kfd_residency_binding_plan(...)`,
  CPU-only per-slot KFD residency receipt correlation that checks successful
  alloc/map result receipts against runtime slot device-pointer lifetime rows,
  allocation kinds, VA spans, byte counts, and resident GPU IDs without opening
  `/dev/kfd`, binding file descriptors, calling KFD ioctls, allocating memory,
  loading checkpoint payloads, submitting AQL, synchronizing queues, executing
  kernels, or changing `live_execution_supported=false`
- `ModelCheckpointBindingPlan::runtime_checkpoint_payload_binding_plan(...)`
  and
  `ModelGraphReadinessReport::runtime_checkpoint_payload_binding_plan(...)`,
  CPU-only checkpoint payload span metadata correlation that checks a fully
  resolved checkpoint key plan, caller-supplied payload source/offset/byte
  metadata, runtime checkpoint-weight slots, and proven KFD residency receipts
  before reporting payload-bound weight slots without opening checkpoint files,
  mmapping safetensors payloads, reading payload bytes, allocating or pinning
  staging buffers, copying bytes to VRAM, submitting SDMA or AQL work,
  synchronizing queues, executing kernels, or changing
  `live_execution_supported=false`
- `ModelGraphReadinessReport::synthetic_cpu_runtime_checkpoint_payload_binding_plan(...)`
  and
  `ModelPluginInspectionReport::synthetic_cpu_runtime_checkpoint_payload_binding_plan(...)`,
  CPU-only convenience helpers for examples and external-package smoke tests
  that derive deterministic synthetic KFD result receipts and checkpoint payload
  span metadata from caller-supplied available checkpoint keys, payload source,
  resident GPU IDs, synthetic base VA, and pointer alignment before returning
  the same checkpoint payload-to-resident-slot binding plan without opening
  checkpoint files, generating real KFD receipts, loading payload bytes, copying
  to VRAM, submitting SDMA or AQL work, executing kernels, or changing
  `live_execution_supported=false`
- `SafetensorsShard::runtime_checkpoint_payload_bindings(...)` and
  `SafetensorsIndex::runtime_checkpoint_payload_bindings(...)`, header-only
  safetensors metadata bridges that first validate checkpoint bindings and then
  emit model API `RuntimeCheckpointPayloadBinding` rows with source path,
  absolute file offset, byte length, dtype, and expected concrete tensor shape,
  without mmapping safetensors payloads, reading tensor bytes, allocating staging
  buffers, copying to VRAM, submitting SDMA or AQL work, executing kernels, or
  changing `live_execution_supported=false`
- `CheckpointPayloadDirectReadPlan::from_checkpoint_payload_binding_plan(...)`
  `CheckpointPayloadDirectReadPlan::staging_batch_plan(...)`,
  `CheckpointPayloadDirectReadStagingPlan::from_direct_read_plan(...)`,
  `CheckpointPayloadDirectReadStagingPlan::receipt_lines()`,
  `CheckpointPayloadDirectReadStagingPlan::receipt_text()`,
  `CheckpointPayloadDirectReadStagingPlan::receipt_fingerprint()`,
  `CheckpointPayloadDirectReadStagingPlan::is_non_executing_boundary()`,
  `CheckpointPayloadDirectReadStagingPlan::assert_non_executing_boundary()`,
  `CheckpointPayloadDirectReadStagingPlan::buffered_host_staging_receipt()`,
  `CheckpointPayloadDirectReadStagingPlan::mapped_host_staging_receipt()`,
  `CheckpointPayloadDirectReadStagingPlan::host_to_device_copy_plan()`,
  `CheckpointPayloadHostToDeviceCopyPlan::destination_residency_proof_input()`,
  `CheckpointPayloadHostToDeviceCopyPlan::destination_residency_query_request()`,
  `CheckpointPayloadHostToDeviceCopyPlan::sdma_queue_reservation_input()`,
  `CheckpointPayloadHostToDeviceCopyPlan::copy_completion_signal_binding_input()`,
  `CheckpointPayloadSdmaQueueReservationInput::sdma_queue_reservation_result_binding_plan(...)`,
  `CheckpointPayloadCopyCompletionSignalBindingInput::copy_completion_signal_result_binding_plan(...)`,
  `CheckpointPayloadHostToDeviceCopyPlan::sdma_copy_packet_materialization_input()`,
  `CheckpointPayloadHostToDeviceCopyPlan::cache_visibility_policy_input()`,
  `CheckpointPayloadHostToDeviceCopyPlan::upload_synchronization_plan_input()`,
  `CheckpointPayloadHostToDeviceCopyPlan::host_to_device_upload_schedule()`,
  `CheckpointPayloadHostToDeviceCopyPlan::host_to_device_upload_runtime_handoff()`,
  `CheckpointPayloadHostToDeviceCopyPlan::host_to_device_upload_bound_runtime_handoff()`,
  `CheckpointPayloadHostToDeviceUploadBoundRuntimeHandoff::mapped_host_staging_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadMappedHostStagingHandoff::destination_residency_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadDestinationResidencyHandoff::sdma_queue_reservation_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadSdmaQueueReservationHandoff::copy_completion_signal_binding_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadCopyCompletionSignalBindingHandoff::sdma_copy_packet_materialization_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadPacketMaterializationHandoff::sdma_copy_packet_validation_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadPacketValidationHandoff::cache_visibility_policy_upload_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadCacheVisibilityPolicyHandoff::upload_completion_synchronization_handoff(...)`,
  `CheckpointPayloadHostToDeviceUploadSchedule::upload_prerequisite_plan()`,
  `CheckpointPayloadHostToDeviceUploadPrerequisitePlan::prerequisite_for()`,
  `CheckpointPayloadHostToDeviceUploadPrerequisitePlan::prerequisite_requirement_names()`,
  `CheckpointPayloadHostToDeviceUploadPrerequisitePlan::unsatisfied_prerequisite_requirement_names()`,
  `CheckpointPayloadHostToDeviceUploadPrerequisitePlan::next_action_requirement_names()`,
  `CheckpointPayloadHostToDeviceUploadPrerequisitePlan::next_action_labels()`,
  `CheckpointPayloadHostToDeviceUploadPrerequisitePlan::next_action_input_labels()`,
  `CheckpointPayloadHostToDeviceUploadSchedule::host_staging_pin_request()`,
  `CheckpointPayloadHostStagingPinRequest::host_virtual_address_binding_plan()`,
  `CheckpointPayloadHostStagingPinVirtualAddressPlan::kfd_userptr_pin_argument_plan()`,
  `CheckpointPayloadHostStagingKfdUserptrPinArgumentPlan::kfd_vm_acquire_request_plan()`,
  `CheckpointPayloadHostStagingKfdVmAcquireRequestPlan::kfd_userptr_alloc_request_plan()`,
  `CheckpointPayloadHostStagingKfdUserptrAllocRequestPlan::kfd_userptr_alloc_result_binding_plan(...)`,
  `CheckpointPayloadHostStagingKfdUserptrAllocResultBindingPlan::kfd_map_memory_request_plan()`,
  `CheckpointPayloadHostStagingKfdMapMemoryRequestPlan::kfd_map_memory_argument_binding_plan(...)`,
  `CheckpointPayloadHostStagingKfdMapMemoryArgumentBindingPlan::kfd_map_memory_result_binding_plan(...)`,
  `CheckpointPayloadDestinationResidencyQueryRequest`,
  `CheckpointPayloadDestinationResidencyQueryAllocation`,
  `CheckpointPayloadDirectReadWorkOrder`,
  `CheckpointPayloadDirectReadBatch`, and
  `CheckpointPayloadDirectReadStagingPiece`, CPU-only direct-read work-order
  and staging-batch metadata plus explicit buffered and mmap-backed
  host-staging reads and
  host-to-device copy-plan, destination-residency proof-input,
  destination-residency query-request, SDMA
  queue-reservation, SDMA queue-reservation result binding,
  copy-completion signal-binding, copy-completion signal result binding, SDMA
  copy-packet materialization, SDMA copy-packet materialization result binding,
  SDMA copy-packet validation, SDMA copy-packet validation result binding,
  cache-visibility policy input, upload-synchronization plan input,
  upload-schedule, upload-prerequisite, upload-runtime handoff, bound
  upload-runtime handoff, mapped host-staging upload handoff,
  destination-residency upload handoff, SDMA queue-reservation upload handoff,
  copy-completion signal-binding upload handoff, packet-materialization upload
  handoff, packet-validation upload handoff, cache-visibility policy upload
  handoff, upload-completion synchronization handoff, host-staging pin-request,
  host-staging pin virtual-address,
  host-staging KFD USERPTR pin-argument, and host-staging KFD VM-acquire,
  USERPTR alloc request, USERPTR alloc result binding, KFD
  map-memory request, KFD map-memory argument-binding, and KFD map-memory
  result-binding receipts for a bound model API checkpoint payload
  plan. The metadata reports direct-I/O
  aligned read windows, staging prefix offsets, coalesced source-local read
  batches, reusable staging-slot assignments, copy pieces, destination offsets,
  destination device VA spans, payload byte counts, aggregate aligned/coalesced
  read byte counts, deterministic receipt text, and explicit non-execution
  guards without opening checkpoint files, reading payload bytes, allocating or
  pinning staging buffers, copying to VRAM, submitting SDMA or AQL work,
  executing kernels, or changing `live_execution_supported=false`. The buffered
  host-staging receipt then explicitly opens the planned sources, allocates
  reusable host staging bytes, reads the file-backed portions of aligned staging
  batches, records EOF tail-padding bytes and a payload fingerprint, and still
  does not use `O_DIRECT`, pin host staging, copy to VRAM, submit SDMA/AQL,
  synchronize queues, execute kernels, or change `live_execution_supported=false`.
  The mmap-backed host-staging receipt maps page-aligned file spans batch by
  batch, copies mapped payload bytes into the same reusable staging shape,
  records mmap calls/bytes and the same payload fingerprint, and still does not
  pin host staging, copy to VRAM, submit SDMA/AQL, synchronize queues, execute
  kernels, or change `live_execution_supported=false`.
  The host-to-device copy-plan receipt resolves absolute host staging offsets,
  payload byte counts, and destination device VA spans for a future uploader. The
  destination-residency proof-input receipt mirrors those destinations as
  per-copy VA spans and aggregate VA bounds while still not binding KFD
  residency rows, allocation handles, resident GPU IDs, executing KFD queries,
  proving residency, allocating VRAM, copying to VRAM, submitting SDMA/AQL,
  synchronizing queues, or executing kernels. The SDMA queue-reservation input
  receipt maps those copy rows and upload waves to logical linear-copy packet
  requests, one completion-fence packet per populated wave, aggregate packet
  dword/byte totals, and doorbell batch requests while still not binding queue
  IDs, queue rings, doorbells, completion signals, materializing packets,
  applying reservations, copying to VRAM, submitting SDMA/AQL, synchronizing
  queues, or executing kernels. The SDMA queue-reservation result-binding
  receipt then validates caller-supplied queue IDs, queue rings, reserved packet
  ranges, and doorbell values while still not mutating queues, copying to VRAM,
  submitting SDMA/AQL, synchronizing queues, or executing kernels. The
  copy-completion signal input receipt requests one `amd_signal_t` binding per
  completion-fence packet, and the copy-completion signal result-binding receipt
  validates caller-supplied signal handles and signal device VAs while still not
  creating signals, binding packet memory, waiting on signals, submitting work,
  or executing kernels. The SDMA copy-packet materialization result-binding
  receipt validates caller-supplied host VAs, queue packet write slots,
  destination spans, signal handles, and signal device VAs while still not
  mutating queues, copying to VRAM, submitting SDMA/AQL, synchronizing queues,
  or executing kernels. The SDMA copy-packet validation input receipt
  replays materialization rows as validation rows and pins row/index/offset
  contiguity, packet byte counts, copy payload spans, and completion signal
  values while still not reading materialized packet bytes, validating packet
  bytes, mutating queues, copying to VRAM, submitting SDMA/AQL, synchronizing
  queues, or executing kernels. The SDMA copy-packet validation result-binding
  receipt validates caller-supplied queue slots, host VAs, signal metadata,
  materialized-packet state, packet template/shape/byte-count/offset checks,
  copy payload spans, completion signal values, and packet byte validation
  while still not mutating queues, copying to VRAM, submitting SDMA/AQL,
  synchronizing queues, or executing kernels. The cache-visibility policy input receipt
  groups validated packet rows by upload wave, selects the device-scope VRAM
  visibility policy, and still does not flush or invalidate caches, prove VRAM
  visibility, copy to VRAM, submit SDMA/AQL, synchronize queues, or execute
  kernels. The upload-synchronization plan input receipt joins completion-signal
  bindings with cache-visibility policy rows and records completion wait rows
  while still not binding signals, issuing or observing waits, proving
  visibility, synchronizing queues, copying to VRAM, submitting SDMA/AQL, or
  executing kernels. The upload-schedule receipt groups those copy rows into
  batch-ordered upload waves, records staging-slot reuse epochs, and pins
  per-wave maxima. The
  upload-prerequisite receipt lists the remaining host/GPU runtime worklist
  before any live copy attempt while still not executing next actions, pinning
  host memory, copying to VRAM, submitting SDMA/AQL, synchronizing queues, or
  executing kernels. The upload-runtime handoff receipt records the ordered
  prerequisite input receipt kinds, fingerprints, and line counts a future
  uploader would consume while still not executing next actions, pinning host
  memory, proving destination residency, copying to VRAM, submitting SDMA/AQL,
  synchronizing queues, or executing kernels. The mapped host-staging,
  destination-residency, SDMA queue-reservation, copy-completion signal
  binding, packet-materialization, packet-validation, cache-visibility policy,
  and upload-completion synchronization handoff receipts reconcile already-bound
  prerequisite receipts and advance the satisfied upload prerequisite count to
  8-of-8 while still not setting `upload_ready`, executing next actions,
  pinning live memory, copying to VRAM, submitting SDMA/AQL, synchronizing
  queues, or executing kernels. The host-staging pin-request
  receipt coalesces waves by reusable staging slot and records both the raw slot
  byte ranges and merged page-rounded pin spans a future live runtime would pin
  while still not binding
  host virtual addresses, materializing page addresses, issuing pin calls,
  copying to VRAM, submitting SDMA/AQL, synchronizing queues, or executing
  kernels. The host-staging pin virtual-address receipt binds those byte ranges
  to a caller-supplied host staging base VA and materializes page-aligned host
  VA spans while still not issuing pin calls, pinning host memory, copying to
  VRAM, submitting SDMA/AQL, synchronizing queues, or executing kernels. The
  host-staging KFD USERPTR pin-argument receipt derives `/dev/kfd`
  `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` USERPTR argument rows from materialized host
  page spans while still not opening KFD, acquiring VMs, issuing ioctls, binding
  handles or mmap offsets, pinning host memory, copying to VRAM, submitting
  SDMA/AQL, synchronizing queues, or executing kernels. The host-staging KFD
  VM-acquire request receipt derives one `AMDKFD_IOC_ACQUIRE_VM` row per
  resident GPU ID from the USERPTR argument receipt while still not opening KFD
  or DRM fds, acquiring VMs, issuing ioctls, allocating USERPTR memory, pinning
  host memory, copying to VRAM, submitting SDMA/AQL, synchronizing queues, or
  executing kernels. The host-staging KFD USERPTR alloc request receipt then
  derives one `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` USERPTR row per host page span
  from the VM-acquire receipt while still not binding fds, acquiring VMs,
  issuing ioctls, allocating USERPTR memory, binding handles or mmap offsets,
  pinning host memory, copying to VRAM, submitting SDMA/AQL, synchronizing
  queues, or executing kernels. The host-staging KFD USERPTR alloc result
  binding receipt validates caller-supplied alloc-memory handles and mmap
  offsets against those request rows while still not opening fds, acquiring VMs,
  issuing ioctls, pinning host memory, copying to VRAM, submitting SDMA/AQL,
  synchronizing queues, or executing kernels.
  The host-staging KFD map-memory request receipt derives one
  `AMDKFD_IOC_MAP_MEMORY_TO_GPU` row per bound USERPTR allocation result while
  still not binding KFD fds, binding `device_ids_array_ptr` storage, updating
  `n_success`, mapping memory, pinning host memory, copying to VRAM, submitting
  SDMA/AQL, synchronizing queues, or executing kernels.
  The host-staging KFD map-memory argument-binding receipt then validates
  caller-supplied `device_ids_array_ptr` storage and resident GPU ID arrays
  against those request rows while still not binding KFD fds, issuing ioctls,
  updating `n_success`, mapping memory, pinning host memory, copying to VRAM,
  submitting SDMA/AQL, synchronizing queues, or executing kernels.
  The host-staging KFD map-memory result-binding receipt validates
  caller-supplied post-ioctl `n_success` rows against the deterministic
  map-memory argument rows and marks residency proven only when every result
  reports successful coverage for all resident GPU IDs, while still not binding
  KFD fds, issuing ioctls, pinning host memory, copying to VRAM, submitting
  SDMA/AQL, synchronizing queues, or executing kernels.
- `CheckpointPayloadBufferedHostStagingReceipt`,
  `CheckpointPayloadBufferedHostStagingBatch`,
  `CheckpointPayloadMappedHostStagingReceipt`, and
  `CheckpointPayloadMappedHostStagingBatch`, CPU-only receipt rows for buffered
  and mmap-backed host-staging read boundaries
- `CheckpointPayloadHostToDeviceCopyPlan` and
  `CheckpointPayloadHostToDeviceCopy`, CPU-only manifest rows for a future
  host-staging-to-device copy boundary
- `CheckpointPayloadDestinationResidencyProofInput` and
  `CheckpointPayloadDestinationResidencySpan`, CPU-only destination-residency
  proof-input rows for a future checkpoint upload boundary
- `CheckpointPayloadSdmaQueueReservationInput`,
  `CheckpointPayloadSdmaQueueReservationWave`, and
  `CheckpointPayloadSdmaQueueReservationCopy`, CPU-only SDMA queue packet and
  doorbell reservation input rows for a future checkpoint upload boundary
- `CheckpointPayloadSdmaQueueReservationResultBinding`,
  `CheckpointPayloadSdmaQueueReservationResultBindingPlan`, and
  `CheckpointPayloadSdmaQueueReservationResult`, CPU-only receipt rows that bind
  SDMA queue reservation requests to caller-supplied queue IDs, ring spans,
  reserved packet ranges, and doorbell values without mutating a live queue
- `CheckpointPayloadCopyCompletionSignalBindingInput`,
  `CheckpointPayloadCopyCompletionSignalBindingRow`,
  `CheckpointPayloadCopyCompletionSignalResultBinding`,
  `CheckpointPayloadCopyCompletionSignalResultBindingPlan`, and
  `CheckpointPayloadCopyCompletionSignalResult`, CPU-only receipt rows that bind
  requested upload completion signals to caller-supplied signal handles and
  signal device VAs without creating signals, writing packets, waiting on
  signals, or mutating a live queue
- `CheckpointPayloadSdmaCopyPacketMaterializationInput`,
  `CheckpointPayloadSdmaCopyPacketMaterializationCopyPacket`, and
  `CheckpointPayloadSdmaCopyPacketMaterializationCompletionPacket`, CPU-only
  SDMA packet materialization input rows for a future checkpoint upload boundary
- `CheckpointPayloadSdmaCopyPacketMaterializationResultBinding`,
  `CheckpointPayloadSdmaCopyPacketMaterializationResultBindingPlan`,
  `CheckpointPayloadSdmaCopyPacketMaterializationResult`, and
  `CheckpointPayloadSdmaCopyPacketMaterializationResultBindingIssue`, CPU-only
  receipt rows that bind requested SDMA packet materialization rows to
  caller-supplied queue slots, host VAs, destination VAs, and completion signal
  metadata without mutating queue memory or submitting SDMA work
- `CheckpointPayloadSdmaCopyPacketValidationInput` and
  `CheckpointPayloadSdmaCopyPacketValidationRow`, CPU-only SDMA copy-packet
  validation input rows for a future checkpoint upload boundary
- `CheckpointPayloadSdmaCopyPacketValidationResultBinding`,
  `CheckpointPayloadSdmaCopyPacketValidationResultBindingPlan`,
  `CheckpointPayloadSdmaCopyPacketValidationResult`, and
  `CheckpointPayloadSdmaCopyPacketValidationResultBindingIssue`, CPU-only
  receipt rows that bind requested SDMA packet validation rows to
  caller-supplied queue slots, host VAs, signal metadata, materialized packet
  state, and packet byte validation without mutating queue memory or submitting
  SDMA work
- `CheckpointPayloadCacheVisibilityPolicyInput` and
  `CheckpointPayloadCacheVisibilityPolicyWave`, CPU-only cache-visibility
  policy input rows for a future checkpoint upload boundary
- `CheckpointPayloadUploadSynchronizationPlanInput` and
  `CheckpointPayloadUploadSynchronizationWait`, CPU-only upload completion wait
  plan rows for a future checkpoint upload boundary
- `CheckpointPayloadHostToDeviceUploadSchedule` and
  `CheckpointPayloadHostToDeviceUploadWave`, CPU-only wave ordering rows for a
  future host-staging-to-device uploader boundary
- `CheckpointPayloadHostToDeviceUploadPrerequisitePlan` and
  `CheckpointPayloadHostToDeviceUploadPrerequisite`, CPU-only unresolved runtime
  worklist rows for a future checkpoint upload boundary
- `CheckpointPayloadHostToDeviceUploadRuntimeHandoff` and
  `CheckpointPayloadHostToDeviceUploadRuntimeHandoffInputReceipt`, CPU-only
  ordered prerequisite input receipt rows for a future checkpoint upload boundary
- `CheckpointPayloadHostToDeviceUploadBoundRuntimeHandoff`, CPU-only
  host-staging VA-bound runtime handoff receipt metadata for a future checkpoint
  upload boundary
- `CheckpointPayloadHostToDeviceUploadMappedHostStagingHandoff` and
  `CheckpointPayloadHostToDeviceUploadDestinationResidencyHandoff`, and
  `CheckpointPayloadHostToDeviceUploadSdmaQueueReservationHandoff`, and
  `CheckpointPayloadHostToDeviceUploadCopyCompletionSignalBindingHandoff`,
  and `CheckpointPayloadHostToDeviceUploadPacketMaterializationHandoff`,
  and `CheckpointPayloadHostToDeviceUploadPacketValidationHandoff`,
  and `CheckpointPayloadHostToDeviceUploadCacheVisibilityPolicyHandoff`,
  and `CheckpointPayloadHostToDeviceUploadCompletionSynchronizationHandoff`,
  CPU-only receipt bridges that satisfy host-staging pinning,
  destination-residency, SDMA queue reservation, copy-completion signal
  binding, packet materialization, packet validation, cache-visibility policy,
  and upload-completion synchronization prerequisites by correlating
  already-bound KFD, queue, signal, packet, validation, policy, and wait-plan
  receipts without executing upload work
- `CheckpointPayloadHostStagingPinRequest` and
  `CheckpointPayloadHostStagingPinRange`, and
  `CheckpointPayloadHostStagingPinPageSpan`, CPU-only host-staging pin-request
  rows for a future checkpoint upload boundary
- `CheckpointPayloadHostStagingPinVirtualAddressPlan`,
  `CheckpointPayloadHostStagingPinVirtualRange`, and
  `CheckpointPayloadHostStagingPinVirtualPageSpan`, CPU-only host virtual
  address and page-span materialization rows for a future checkpoint upload pin
  boundary
- `CheckpointPayloadHostStagingKfdUserptrPinArgumentPlan` and
  `CheckpointPayloadHostStagingKfdUserptrPinArgument`, CPU-only KFD USERPTR
  alloc-memory argument rows for a future checkpoint upload pin boundary
- `CheckpointPayloadHostStagingKfdVmAcquireRequestPlan` and
  `CheckpointPayloadHostStagingKfdVmAcquireRequest`, CPU-only KFD VM-acquire
  request rows for the resident GPUs needed by future checkpoint host-staging
  USERPTR pinning
- `CheckpointPayloadHostStagingKfdUserptrAllocRequestPlan` and
  `CheckpointPayloadHostStagingKfdUserptrAllocRequest`, CPU-only KFD USERPTR
  alloc-memory request rows derived after VM-acquire metadata for future
  checkpoint host-staging USERPTR pinning
- `CheckpointPayloadHostStagingKfdUserptrAllocResultBindingPlan`,
  `CheckpointPayloadHostStagingKfdUserptrAllocResultBinding`,
  `CheckpointPayloadHostStagingKfdUserptrAllocResult`, and
  `CheckpointPayloadHostStagingKfdUserptrAllocResultIssue`, CPU-only validation
  rows for caller-supplied checkpoint host-staging KFD USERPTR alloc-memory
  result handles and mmap offsets
- `CheckpointPayloadHostStagingKfdMapMemoryRequestPlan` and
  `CheckpointPayloadHostStagingKfdMapMemoryRequest`, CPU-only KFD
  map-memory request rows derived from bound checkpoint host-staging USERPTR
  allocation results
- `CheckpointPayloadHostStagingKfdMapMemoryDeviceIdsArrayBinding`,
  `CheckpointPayloadHostStagingKfdMapMemoryArgumentBindingPlan`,
  `CheckpointPayloadHostStagingKfdMapMemoryArgumentBinding`, and
  `CheckpointPayloadHostStagingKfdMapMemoryArgumentBindingIssue`, CPU-only
  validation rows for caller-supplied checkpoint host-staging KFD map-memory
  device-ID array pointers and resident GPU ID arrays
- `CheckpointPayloadHostStagingKfdMapMemoryResultBindingPlan`,
  `CheckpointPayloadHostStagingKfdMapMemoryResultBinding`,
  `CheckpointPayloadHostStagingKfdMapMemoryResult`, and
  `CheckpointPayloadHostStagingKfdMapMemoryResultIssue`, CPU-only receipt
  validation rows for caller-supplied checkpoint host-staging KFD map-memory
  post-ioctl success counts

Rows and field names in this tier should stay deterministic. Additive fields are
preferred over reshaping existing rows. A breaking change must update the
reference MoE example, custom model example, CLI selftest, docs, and evidence.

The static handoff receipt is still a CPU-only metadata handoff boundary. Its
stable non-execution rows include `launch_execution.executable=false`,
`launch_execution.unresolved_runtime_requirement_count`,
`launch_execution.unresolved_runtime_requirements.*`,
`launch_execution.aql_dispatchable_packet_count=0`,
`live_aql_submitting_surface_count=0`,
`live_queue_mutating_component_count=0`, `live_execution_supported=false`,
`gpu_buffers_allocated=false`, and `kernels_submitted=false`.

### Experimental execution boundary

Anything that would turn the metadata graph into live GPU work is still
experimental until a change proves it with hardware evidence:

- live device allocation and hardware residency binding
- live checkpoint payload loading/copying into model API slots
- complete kernel-argument ABI coverage and semantic ABI validation against
  code-object metadata beyond the covered static semantic comparison registry
- live AQL packet construction from the graph API
- queue submission, synchronization, token generation, and performance through
  the graph API

Do not describe this tier as supported until the evidence includes the matching
hardware command, correctness oracle, and negative scope.

## Change Rules

- Keep model API changes one behavioral atom per commit.
- Run the focused model API tests and examples for any model-facing API change.
- Run the workspace test suite for shared validation/reporting changes.
- Update `docs/model-api.md` when a public surface is added, removed, renamed, or
  moved between stability tiers.
- Record the command, hardware, and output for claim-bearing changes, and state
  the negative scope, meaning what the change does not prove, before merging.

use anyhow::{anyhow, Result};
use mainarch_core::model_api::prelude::*;
use mainarch_core::weights::{
    CheckpointPayloadCopyCompletionSignalBindingInput,
    CheckpointPayloadCopyCompletionSignalResultBinding,
    CheckpointPayloadCopyCompletionSignalResultBindingPlan, CheckpointPayloadDirectReadPlan,
    CheckpointPayloadDirectReadStagingPlan,
    CheckpointPayloadHostStagingKfdMapMemoryDeviceIdsArrayBinding,
    CheckpointPayloadHostStagingKfdMapMemoryResultBinding,
    CheckpointPayloadHostStagingKfdMapMemoryResultBindingPlan,
    CheckpointPayloadHostStagingKfdUserptrAllocResultBinding,
    CheckpointPayloadSdmaCopyPacketMaterializationInput,
    CheckpointPayloadSdmaCopyPacketMaterializationResultBinding,
    CheckpointPayloadSdmaCopyPacketMaterializationResultBindingPlan,
    CheckpointPayloadSdmaCopyPacketValidationInput,
    CheckpointPayloadSdmaCopyPacketValidationResultBinding,
    CheckpointPayloadSdmaCopyPacketValidationResultBindingPlan,
    CheckpointPayloadSdmaQueueReservationInput, CheckpointPayloadSdmaQueueReservationResultBinding,
    CheckpointPayloadSdmaQueueReservationResultBindingPlan,
    SafetensorsCheckpointMetadataValidation, SafetensorsIndex,
    SafetensorsIndexCheckpointMetadataValidation, SafetensorsShard,
    CHECKPOINT_SDMA_COMPLETION_PACKET_VALIDATION_SCOPE_LABEL,
    CHECKPOINT_SDMA_COPY_PACKET_VALIDATION_SCOPE_LABEL, DEFAULT_DIRECT_IO_ALIGNMENT,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const SAFETENSORS_PREFIX_BYTES: usize = 8;
const PAGE_ROUNDING_DIRECT_IO_ALIGNMENT: u64 = 512;
const RESIDENT_GPU_IDS: [u32; 2] = [1002, 1001];
const HOST_STAGING_BASE_VA: u64 = 0x7f00_0000_0000;

fn main() -> Result<()> {
    let emit_checkpoint_staging_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-staging-receipt");
    let emit_checkpoint_host_staging_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-host-staging-receipt");
    let emit_checkpoint_mapped_host_staging_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-mapped-host-staging-receipt");
    let emit_checkpoint_copy_plan_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-copy-plan-receipt");
    let emit_checkpoint_destination_residency_proof_input_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-destination-residency-proof-input-receipt");
    let emit_checkpoint_destination_residency_query_request_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-destination-residency-query-request-receipt");
    let emit_checkpoint_sdma_queue_reservation_input_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-sdma-queue-reservation-input-receipt");
    let emit_checkpoint_sdma_queue_reservation_result_binding_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-sdma-queue-reservation-result-binding-receipt");
    let emit_checkpoint_copy_completion_signal_binding_input_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-copy-completion-signal-binding-input-receipt");
    let emit_checkpoint_copy_completion_signal_result_binding_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-copy-completion-signal-result-binding-receipt");
    let emit_checkpoint_sdma_copy_packet_materialization_input_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-sdma-copy-packet-materialization-input-receipt");
    let emit_checkpoint_sdma_copy_packet_materialization_result_binding_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-sdma-copy-packet-materialization-result-binding-receipt");
    let emit_checkpoint_sdma_copy_packet_validation_input_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-sdma-copy-packet-validation-input-receipt");
    let emit_checkpoint_sdma_copy_packet_validation_result_binding_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-sdma-copy-packet-validation-result-binding-receipt");
    let emit_checkpoint_cache_visibility_policy_input_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-cache-visibility-policy-input-receipt");
    let emit_checkpoint_upload_synchronization_plan_input_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-upload-synchronization-plan-input-receipt");
    let emit_checkpoint_upload_schedule_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-upload-schedule-receipt");
    let emit_checkpoint_upload_prerequisite_plan_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-upload-prerequisite-plan-receipt");
    let emit_checkpoint_upload_runtime_handoff_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-upload-runtime-handoff-receipt");
    let emit_checkpoint_upload_bound_runtime_handoff_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-upload-bound-runtime-handoff-receipt");
    let emit_checkpoint_upload_mapped_host_staging_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-mapped-host-staging-handoff-receipt");
    let emit_checkpoint_upload_destination_residency_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-destination-residency-handoff-receipt");
    let emit_checkpoint_upload_sdma_queue_reservation_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-sdma-queue-reservation-handoff-receipt");
    let emit_checkpoint_upload_copy_completion_signal_binding_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-copy-completion-signal-binding-handoff-receipt");
    let emit_checkpoint_upload_packet_materialization_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-packet-materialization-handoff-receipt");
    let emit_checkpoint_upload_packet_validation_handoff_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-upload-packet-validation-handoff-receipt");
    let emit_checkpoint_upload_cache_visibility_policy_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-cache-visibility-policy-handoff-receipt");
    let emit_checkpoint_upload_completion_synchronization_handoff_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-upload-completion-synchronization-handoff-receipt");
    let emit_checkpoint_host_staging_pin_request_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-host-staging-pin-request-receipt");
    let emit_checkpoint_host_staging_pin_virtual_address_plan_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-pin-virtual-address-plan-receipt");
    let emit_checkpoint_host_staging_userptr_pin_arguments_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-userptr-pin-arguments-receipt");
    let emit_checkpoint_host_staging_kfd_vm_acquire_request_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-kfd-vm-acquire-request-receipt");
    let emit_checkpoint_host_staging_kfd_userptr_alloc_request_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-kfd-userptr-alloc-request-receipt");
    let emit_checkpoint_host_staging_kfd_userptr_alloc_result_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-kfd-userptr-alloc-result-receipt");
    let emit_checkpoint_host_staging_kfd_map_memory_request_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-kfd-map-memory-request-receipt");
    let emit_checkpoint_host_staging_kfd_map_memory_argument_binding_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-kfd-map-memory-argument-binding-receipt");
    let emit_checkpoint_host_staging_kfd_map_memory_result_binding_receipt = std::env::args()
        .any(|arg| arg == "--checkpoint-host-staging-kfd-map-memory-result-binding-receipt");
    let emit_checkpoint_host_staging_pin_page_rounding_receipt =
        std::env::args().any(|arg| arg == "--checkpoint-host-staging-pin-page-rounding-receipt");

    let mut config = ReferenceMoeConfig::qwen3_moe_reference();
    config.layers = 1;
    config.hidden = 64;
    config.query_heads = 4;
    config.kv_heads = 2;
    config.head_dim = 16;
    config.intermediate = 32;
    config.experts = 4;
    config.top_k = 2;
    config.vocab = 128;
    config.max_context = 64;
    config.tensor_parallel = 2;

    let model = ReferenceMoeDecoder::new(config)?;
    let graph = build_model_graph(&model)?;
    let catalog = MainarchPrimitiveLoweringCatalog::mi355_reference();
    let readiness = graph.readiness_report(&catalog)?;
    let checkpoint = &readiness.checkpoint;
    checkpoint.assert_all_weights_bound()?;

    let shard_bytes = synthetic_safetensors_shard(checkpoint)?;
    let shard = SafetensorsShard::parse(
        PathBuf::from("synthetic-reference-moe-checkpoint.safetensors"),
        &shard_bytes,
    )?;
    let validation = shard.validate_model_checkpoint_bindings(&checkpoint)?;
    validation.assert_fully_valid()?;
    let shard_payload_bindings = shard.runtime_checkpoint_payload_bindings(&checkpoint)?;
    let shard_path = std::env::temp_dir().join(format!(
        "mainarch-reference-moe-checkpoint-{}.safetensors",
        std::process::id()
    ));
    fs::write(&shard_path, &shard_bytes)?;
    let index_json =
        synthetic_safetensors_index(&shard, &shard_path, checkpoint.total_checkpoint_bytes)?;
    let index = SafetensorsIndex::parse(
        shard_path.with_file_name("synthetic-reference-moe-checkpoint.index.json"),
        &index_json,
    )?;
    let index_validation = index.validate_model_checkpoint_bindings(&checkpoint)?;
    index_validation.assert_fully_valid()?;
    let index_payload_bindings = index.runtime_checkpoint_payload_bindings(&checkpoint)?;
    let (runtime_payload_plan, residency_plan) = synthetic_runtime_checkpoint_payload_plan(
        &readiness,
        &index_validation.resolution,
        &index_payload_bindings,
    )?;
    let direct_read_plan = CheckpointPayloadDirectReadPlan::from_checkpoint_payload_binding_plan(
        &runtime_payload_plan,
        DEFAULT_DIRECT_IO_ALIGNMENT,
    )?;
    let staging_plan = direct_read_plan.staging_batch_plan(2)?;
    staging_plan.assert_non_executing_boundary()?;

    if emit_checkpoint_staging_receipt {
        print!("{}", staging_plan.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_receipt {
        let host_staging_receipt = staging_plan.buffered_host_staging_receipt()?;
        host_staging_receipt.assert_cpu_only_host_staging_boundary()?;
        print!("{}", host_staging_receipt.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_mapped_host_staging_receipt {
        let mapped_host_staging_receipt = staging_plan.mapped_host_staging_receipt()?;
        mapped_host_staging_receipt.assert_cpu_only_mapped_host_staging_boundary()?;
        print!("{}", mapped_host_staging_receipt.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_copy_plan_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        copy_plan.assert_no_copy_side_effect_boundary()?;
        print!("{}", copy_plan.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_destination_residency_proof_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let residency_input = copy_plan.destination_residency_proof_input()?;
        residency_input.assert_no_residency_side_effect_boundary()?;
        print!("{}", residency_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_destination_residency_query_request_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let query_request = copy_plan.destination_residency_query_request(&residency_plan)?;
        query_request.assert_no_residency_query_side_effect_boundary()?;
        print!("{}", query_request.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_sdma_queue_reservation_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        queue_input.assert_no_queue_reservation_side_effect_boundary()?;
        print!("{}", queue_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_sdma_queue_reservation_result_binding_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        print!("{}", queue_results.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_copy_completion_signal_binding_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        signal_input.assert_no_completion_signal_binding_side_effect_boundary()?;
        print!("{}", signal_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_copy_completion_signal_result_binding_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        print!("{}", signal_results.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_sdma_copy_packet_materialization_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        packet_input.assert_no_packet_materialization_side_effect_boundary()?;
        print!("{}", packet_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_sdma_copy_packet_materialization_result_binding_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        let packet_results = synthetic_checkpoint_sdma_copy_packet_materialization_results(
            &packet_input,
            &queue_results,
            &signal_results,
            HOST_STAGING_BASE_VA,
        )?;
        print!("{}", packet_results.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_sdma_copy_packet_validation_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let validation_input = copy_plan.sdma_copy_packet_validation_input()?;
        validation_input.assert_no_packet_validation_side_effect_boundary()?;
        print!("{}", validation_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_sdma_copy_packet_validation_result_binding_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        let packet_results = synthetic_checkpoint_sdma_copy_packet_materialization_results(
            &packet_input,
            &queue_results,
            &signal_results,
            HOST_STAGING_BASE_VA,
        )?;
        let validation_input = copy_plan.sdma_copy_packet_validation_input()?;
        let validation_results = synthetic_checkpoint_sdma_copy_packet_validation_results(
            &validation_input,
            &packet_results,
        )?;
        print!("{}", validation_results.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_cache_visibility_policy_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let policy_input = copy_plan.cache_visibility_policy_input()?;
        policy_input.assert_no_cache_visibility_side_effect_boundary()?;
        print!("{}", policy_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_synchronization_plan_input_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let synchronization_input = copy_plan.upload_synchronization_plan_input()?;
        synchronization_input.assert_no_upload_synchronization_side_effect_boundary()?;
        print!("{}", synchronization_input.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_schedule_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        upload_schedule.assert_no_upload_side_effect_boundary()?;
        print!("{}", upload_schedule.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_prerequisite_plan_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let prerequisite_plan = upload_schedule.upload_prerequisite_plan()?;
        prerequisite_plan.assert_no_upload_side_effect_boundary()?;
        print!("{}", prerequisite_plan.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_runtime_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let runtime_handoff = copy_plan.host_to_device_upload_runtime_handoff(&residency_plan)?;
        runtime_handoff.assert_no_upload_side_effect_boundary()?;
        print!("{}", runtime_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_bound_runtime_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        bound_handoff.assert_no_upload_side_effect_boundary()?;
        print!("{}", bound_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_mapped_host_staging_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        mapped_handoff.assert_kfd_map_memory_receipt_boundary()?;
        print!("{}", mapped_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_destination_residency_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        destination_handoff.assert_destination_residency_receipt_boundary()?;
        print!("{}", destination_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_sdma_queue_reservation_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let queue_handoff =
            destination_handoff.sdma_queue_reservation_upload_handoff(&queue_results)?;
        queue_handoff.assert_sdma_queue_reservation_receipt_boundary()?;
        print!("{}", queue_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_copy_completion_signal_binding_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let queue_handoff =
            destination_handoff.sdma_queue_reservation_upload_handoff(&queue_results)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let signal_handoff =
            queue_handoff.copy_completion_signal_binding_upload_handoff(&signal_results)?;
        signal_handoff.assert_copy_completion_signal_binding_receipt_boundary()?;
        print!("{}", signal_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_packet_materialization_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let queue_handoff =
            destination_handoff.sdma_queue_reservation_upload_handoff(&queue_results)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let signal_handoff =
            queue_handoff.copy_completion_signal_binding_upload_handoff(&signal_results)?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        let packet_results = synthetic_checkpoint_sdma_copy_packet_materialization_results(
            &packet_input,
            &queue_results,
            &signal_results,
            HOST_STAGING_BASE_VA,
        )?;
        let packet_handoff =
            signal_handoff.sdma_copy_packet_materialization_upload_handoff(&packet_results)?;
        packet_handoff.assert_packet_materialization_receipt_boundary()?;
        print!("{}", packet_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_packet_validation_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let queue_handoff =
            destination_handoff.sdma_queue_reservation_upload_handoff(&queue_results)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let signal_handoff =
            queue_handoff.copy_completion_signal_binding_upload_handoff(&signal_results)?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        let packet_results = synthetic_checkpoint_sdma_copy_packet_materialization_results(
            &packet_input,
            &queue_results,
            &signal_results,
            HOST_STAGING_BASE_VA,
        )?;
        let packet_handoff =
            signal_handoff.sdma_copy_packet_materialization_upload_handoff(&packet_results)?;
        let validation_input = copy_plan.sdma_copy_packet_validation_input()?;
        let validation_results = synthetic_checkpoint_sdma_copy_packet_validation_results(
            &validation_input,
            &packet_results,
        )?;
        let validation_handoff =
            packet_handoff.sdma_copy_packet_validation_upload_handoff(&validation_results)?;
        validation_handoff.assert_packet_validation_receipt_boundary()?;
        print!("{}", validation_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_cache_visibility_policy_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let queue_handoff =
            destination_handoff.sdma_queue_reservation_upload_handoff(&queue_results)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let signal_handoff =
            queue_handoff.copy_completion_signal_binding_upload_handoff(&signal_results)?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        let packet_results = synthetic_checkpoint_sdma_copy_packet_materialization_results(
            &packet_input,
            &queue_results,
            &signal_results,
            HOST_STAGING_BASE_VA,
        )?;
        let packet_handoff =
            signal_handoff.sdma_copy_packet_materialization_upload_handoff(&packet_results)?;
        let validation_input = copy_plan.sdma_copy_packet_validation_input()?;
        let validation_results = synthetic_checkpoint_sdma_copy_packet_validation_results(
            &validation_input,
            &packet_results,
        )?;
        let validation_handoff =
            packet_handoff.sdma_copy_packet_validation_upload_handoff(&validation_results)?;
        let cache_policy_input = copy_plan.cache_visibility_policy_input()?;
        let cache_handoff =
            validation_handoff.cache_visibility_policy_upload_handoff(&cache_policy_input)?;
        cache_handoff.assert_cache_visibility_policy_receipt_boundary()?;
        print!("{}", cache_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_upload_completion_synchronization_handoff_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let bound_handoff = copy_plan
            .host_to_device_upload_bound_runtime_handoff(&residency_plan, HOST_STAGING_BASE_VA)?;
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        let mapped_handoff =
            bound_handoff.mapped_host_staging_upload_handoff(&map_memory_results)?;
        let destination_query = copy_plan.destination_residency_query_request(&residency_plan)?;
        let destination_handoff = mapped_handoff
            .destination_residency_upload_handoff(&destination_query, &residency_plan)?;
        let queue_input = copy_plan.sdma_queue_reservation_input()?;
        let queue_results = synthetic_checkpoint_sdma_queue_reservation_results(&queue_input)?;
        let queue_handoff =
            destination_handoff.sdma_queue_reservation_upload_handoff(&queue_results)?;
        let signal_input = copy_plan.copy_completion_signal_binding_input()?;
        let signal_results = synthetic_checkpoint_copy_completion_signal_results(&signal_input)?;
        let signal_handoff =
            queue_handoff.copy_completion_signal_binding_upload_handoff(&signal_results)?;
        let packet_input = copy_plan.sdma_copy_packet_materialization_input()?;
        let packet_results = synthetic_checkpoint_sdma_copy_packet_materialization_results(
            &packet_input,
            &queue_results,
            &signal_results,
            HOST_STAGING_BASE_VA,
        )?;
        let packet_handoff =
            signal_handoff.sdma_copy_packet_materialization_upload_handoff(&packet_results)?;
        let validation_input = copy_plan.sdma_copy_packet_validation_input()?;
        let validation_results = synthetic_checkpoint_sdma_copy_packet_validation_results(
            &validation_input,
            &packet_results,
        )?;
        let validation_handoff =
            packet_handoff.sdma_copy_packet_validation_upload_handoff(&validation_results)?;
        let cache_policy_input = copy_plan.cache_visibility_policy_input()?;
        let cache_handoff =
            validation_handoff.cache_visibility_policy_upload_handoff(&cache_policy_input)?;
        let synchronization_input = copy_plan.upload_synchronization_plan_input()?;
        let synchronization_handoff =
            cache_handoff.upload_completion_synchronization_handoff(&synchronization_input)?;
        synchronization_handoff.assert_upload_completion_synchronization_receipt_boundary()?;
        print!("{}", synchronization_handoff.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_pin_request_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        pin_request.assert_no_pin_side_effect_boundary()?;
        print!("{}", pin_request.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_pin_virtual_address_plan_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        virtual_address_plan.assert_no_pin_call_side_effect_boundary()?;
        print!("{}", virtual_address_plan.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_userptr_pin_arguments_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        let userptr_arguments =
            virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
        userptr_arguments.assert_no_userptr_pin_side_effect_boundary()?;
        print!("{}", userptr_arguments.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_kfd_vm_acquire_request_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        let userptr_arguments =
            virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
        let vm_acquire_requests = userptr_arguments.kfd_vm_acquire_request_plan()?;
        vm_acquire_requests.assert_no_vm_acquire_side_effect_boundary()?;
        print!("{}", vm_acquire_requests.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_kfd_userptr_alloc_request_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        let userptr_arguments =
            virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
        let vm_acquire_requests = userptr_arguments.kfd_vm_acquire_request_plan()?;
        let alloc_requests = vm_acquire_requests.kfd_userptr_alloc_request_plan()?;
        alloc_requests.assert_no_userptr_alloc_side_effect_boundary()?;
        print!("{}", alloc_requests.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_kfd_userptr_alloc_result_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        let userptr_arguments =
            virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
        let vm_acquire_requests = userptr_arguments.kfd_vm_acquire_request_plan()?;
        let alloc_requests = vm_acquire_requests.kfd_userptr_alloc_request_plan()?;
        let result_bindings = alloc_requests
            .requests
            .iter()
            .map(|request| {
                let handle = synthetic_slot_u64(
                    0x7b00_0000,
                    request.request_index,
                    1,
                    "synthetic KFD USERPTR allocation handle",
                )?;
                let mmap_offset = synthetic_slot_u64(
                    0x9b00_0000,
                    request.request_index,
                    0x1000,
                    "synthetic KFD USERPTR allocation mmap offset",
                )?;
                Ok(
                    CheckpointPayloadHostStagingKfdUserptrAllocResultBinding::new(
                        request.request_index,
                        request.argument_index,
                        request.page_span_index,
                        request.alloc_args_va_addr,
                        request.alloc_args_size,
                        request.alloc_args_gpu_id,
                        request.alloc_args_flags,
                        handle,
                        mmap_offset,
                    ),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let alloc_results = alloc_requests.kfd_userptr_alloc_result_binding_plan(&result_bindings);
        alloc_results.assert_kfd_userptr_alloc_result_bound()?;
        alloc_results.assert_userptr_alloc_result_binding_only_boundary()?;
        print!("{}", alloc_results.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_kfd_map_memory_request_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        let userptr_arguments =
            virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
        let vm_acquire_requests = userptr_arguments.kfd_vm_acquire_request_plan()?;
        let alloc_requests = vm_acquire_requests.kfd_userptr_alloc_request_plan()?;
        let result_bindings = alloc_requests
            .requests
            .iter()
            .map(|request| {
                let handle = synthetic_slot_u64(
                    0x7b00_0000,
                    request.request_index,
                    1,
                    "synthetic KFD USERPTR allocation handle",
                )?;
                let mmap_offset = synthetic_slot_u64(
                    0x9b00_0000,
                    request.request_index,
                    0x1000,
                    "synthetic KFD USERPTR allocation mmap offset",
                )?;
                Ok(
                    CheckpointPayloadHostStagingKfdUserptrAllocResultBinding::new(
                        request.request_index,
                        request.argument_index,
                        request.page_span_index,
                        request.alloc_args_va_addr,
                        request.alloc_args_size,
                        request.alloc_args_gpu_id,
                        request.alloc_args_flags,
                        handle,
                        mmap_offset,
                    ),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let alloc_results = alloc_requests.kfd_userptr_alloc_result_binding_plan(&result_bindings);
        alloc_results.assert_kfd_userptr_alloc_result_bound()?;
        alloc_results.assert_userptr_alloc_result_binding_only_boundary()?;
        let map_memory_requests = alloc_results.kfd_map_memory_request_plan()?;
        map_memory_requests.assert_no_map_memory_side_effect_boundary()?;
        print!("{}", map_memory_requests.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_kfd_map_memory_argument_binding_receipt {
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        let virtual_address_plan =
            pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
        let userptr_arguments =
            virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
        let vm_acquire_requests = userptr_arguments.kfd_vm_acquire_request_plan()?;
        let alloc_requests = vm_acquire_requests.kfd_userptr_alloc_request_plan()?;
        let result_bindings = alloc_requests
            .requests
            .iter()
            .map(|request| {
                let handle = synthetic_slot_u64(
                    0x7b00_0000,
                    request.request_index,
                    1,
                    "synthetic KFD USERPTR allocation handle",
                )?;
                let mmap_offset = synthetic_slot_u64(
                    0x9b00_0000,
                    request.request_index,
                    0x1000,
                    "synthetic KFD USERPTR allocation mmap offset",
                )?;
                Ok(
                    CheckpointPayloadHostStagingKfdUserptrAllocResultBinding::new(
                        request.request_index,
                        request.argument_index,
                        request.page_span_index,
                        request.alloc_args_va_addr,
                        request.alloc_args_size,
                        request.alloc_args_gpu_id,
                        request.alloc_args_flags,
                        handle,
                        mmap_offset,
                    ),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let alloc_results = alloc_requests.kfd_userptr_alloc_result_binding_plan(&result_bindings);
        alloc_results.assert_kfd_userptr_alloc_result_bound()?;
        alloc_results.assert_userptr_alloc_result_binding_only_boundary()?;
        let map_memory_requests = alloc_results.kfd_map_memory_request_plan()?;
        map_memory_requests.assert_no_map_memory_side_effect_boundary()?;
        let device_ids_array_bindings = map_memory_requests
            .requests
            .iter()
            .map(|request| {
                let device_ids_array_ptr = synthetic_slot_u64(
                    0x3c00_0000,
                    request.request_index,
                    0x100,
                    "synthetic KFD map-memory device ID array pointer",
                )?;
                Ok(
                    CheckpointPayloadHostStagingKfdMapMemoryDeviceIdsArrayBinding::new(
                        request.request_index,
                        request.alloc_result_index,
                        request.alloc_request_index,
                        request.argument_index,
                        request.page_span_index,
                        request.map_args_handle,
                        device_ids_array_ptr,
                        request.resident_gpu_ids.clone(),
                    ),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let map_memory_arguments =
            map_memory_requests.kfd_map_memory_argument_binding_plan(&device_ids_array_bindings);
        map_memory_arguments.assert_kfd_map_memory_arguments_bound()?;
        map_memory_arguments.assert_map_memory_argument_binding_only_boundary()?;
        print!("{}", map_memory_arguments.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_kfd_map_memory_result_binding_receipt {
        let map_memory_results =
            synthetic_checkpoint_host_staging_kfd_map_memory_results(&staging_plan)?;
        print!("{}", map_memory_results.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }
    if emit_checkpoint_host_staging_pin_page_rounding_receipt {
        let direct_read_plan =
            CheckpointPayloadDirectReadPlan::from_checkpoint_payload_binding_plan(
                &runtime_payload_plan,
                PAGE_ROUNDING_DIRECT_IO_ALIGNMENT,
            )?;
        let staging_plan = direct_read_plan.staging_batch_plan(2)?;
        let copy_plan = staging_plan.host_to_device_copy_plan()?;
        let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
        let pin_request = upload_schedule.host_staging_pin_request()?;
        pin_request.assert_no_pin_side_effect_boundary()?;
        print!("{}", pin_request.receipt_text());
        let _ = fs::remove_file(shard_path);
        return Ok(());
    }

    print_summary(
        checkpoint,
        &validation,
        &index_validation,
        &shard_payload_bindings,
        &index_payload_bindings,
        &direct_read_plan,
        &staging_plan,
        shard.tensors.len(),
        shard_bytes.len(),
    );
    let _ = fs::remove_file(shard_path);
    Ok(())
}

fn print_summary(
    checkpoint: &ModelCheckpointBindingPlan,
    validation: &SafetensorsCheckpointMetadataValidation,
    index_validation: &SafetensorsIndexCheckpointMetadataValidation,
    shard_payload_bindings: &[RuntimeCheckpointPayloadBinding],
    index_payload_bindings: &[RuntimeCheckpointPayloadBinding],
    direct_read_plan: &CheckpointPayloadDirectReadPlan,
    staging_plan: &CheckpointPayloadDirectReadStagingPlan,
    shard_tensors: usize,
    shard_bytes: usize,
) {
    println!("model: reference-moe-checkpoint-metadata");
    println!(
        "checkpoint: bound_weights={} checkpoint_bytes={}",
        checkpoint.entries.len(),
        checkpoint.total_checkpoint_bytes
    );
    println!(
        "safetensors: tensor_headers={} matched_tensors={} checked_entries={} mismatches={} file_bytes={}",
        shard_tensors,
        validation
            .resolution
            .resolved_entries
            .iter()
            .map(|entry| entry.matched_checkpoint_keys.len())
            .sum::<usize>(),
        validation.checked_entries.len(),
        validation.mismatches.len(),
        shard_bytes
    );
    println!(
        "safetensors_index: opened_shards={} missing_shards={} checked_entries={} mismatches={}",
        index_validation.opened_shards.len(),
        index_validation.missing_shards.len(),
        index_validation.checked_entries.len(),
        index_validation.mismatches.len()
    );
    println!(
        "safetensors_payload_spans: shard_bindings={} index_bindings={} payload_bytes={} first_offset={} header_only=true",
        shard_payload_bindings.len(),
        index_payload_bindings.len(),
        shard_payload_bindings
            .iter()
            .map(|binding| binding.payload_bytes)
            .sum::<usize>(),
        shard_payload_bindings
            .first()
            .map(|binding| binding.payload_offset)
            .unwrap_or(0)
    );
    println!(
        "checkpoint_payload_direct_reads: work_orders={} sources={} slots={} payload_bytes={} aligned_read_bytes={} max_staging_window={} direct_io_alignment={} cpu_only=true",
        direct_read_plan.work_order_count,
        direct_read_plan.source_count,
        direct_read_plan.slot_count,
        direct_read_plan.total_payload_bytes,
        direct_read_plan.total_aligned_read_bytes,
        direct_read_plan.max_staging_window_bytes,
        direct_read_plan.direct_io_alignment
    );
    println!(
        "checkpoint_payload_staging_batches: batches={} pieces={} sources={} slots={} staging_slots={} staging_bytes={} read_bytes={} max_batch_read_bytes={} read_amplification_milli={} cpu_only=true",
        staging_plan.batch_count,
        staging_plan.piece_count,
        staging_plan.source_count,
        staging_plan.slot_count,
        staging_plan.staging_slot_count,
        staging_plan.total_staging_bytes,
        staging_plan.total_read_bytes,
        staging_plan.max_batch_read_bytes,
        staging_plan.read_amplification_milli
    );
    println!(
        "checkpoint_payload_staging_receipt: fingerprint={} lines={} non_executing={} live_execution_supported=false cpu_only=true",
        staging_plan.receipt_fingerprint(),
        staging_plan.receipt_lines().len(),
        staging_plan.is_non_executing_boundary()
    );
}

fn synthetic_runtime_checkpoint_payload_plan(
    readiness: &ModelGraphReadinessReport,
    key_resolution: &ModelCheckpointKeyResolution,
    payload_bindings: &[RuntimeCheckpointPayloadBinding],
) -> Result<(
    ModelRuntimeCheckpointPayloadBindingPlan,
    ModelRuntimeSlotKfdResidencyBindingPlan,
)> {
    let device_pointer_bindings = readiness.slots.device_pointer_binding_template(
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    let device_pointer_validation = readiness
        .slots
        .validate_complete_device_pointer_bindings(&device_pointer_bindings);
    device_pointer_validation.assert_complete()?;

    let allocation_requests = readiness.runtime_slot_kfd_allocation_residency_request_plan(
        &device_pointer_validation,
        &RESIDENT_GPU_IDS,
    );
    let vm_requests = allocation_requests.kfd_vm_acquire_request_plan();
    let alloc_memory_requests = allocation_requests.kfd_alloc_memory_request_plan(&vm_requests);
    let alloc_result_bindings = alloc_memory_requests
        .entries
        .iter()
        .map(|entry| {
            RuntimeKfdAllocMemoryResultBinding::new(
                entry.slot,
                entry.tensor.as_str(),
                entry.allocation_gpu_id,
                entry.alloc_args_va_addr,
                entry.alloc_args_size,
                entry.alloc_args_flags,
                synthetic_slot_u64(
                    0x7a00_0000,
                    entry.slot,
                    1,
                    "synthetic KFD allocation handle",
                )?,
                synthetic_slot_u64(
                    0x9a00_0000,
                    entry.slot,
                    0x1000,
                    "synthetic KFD allocation mmap offset",
                )?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let alloc_result_plan =
        alloc_memory_requests.kfd_alloc_memory_result_binding_plan(&alloc_result_bindings);
    alloc_result_plan.assert_kfd_alloc_memory_result_bound()?;

    let map_memory_requests = alloc_memory_requests.kfd_map_memory_request_plan();
    let device_ids_array_bindings = map_memory_requests
        .entries
        .iter()
        .map(|entry| {
            RuntimeKfdMapMemoryDeviceIdsArrayBinding::new(
                entry.slot,
                entry.tensor.as_str(),
                synthetic_slot_u64(
                    0x3c00_0000,
                    entry.slot,
                    0x100,
                    "synthetic KFD map-memory device IDs array pointer",
                )?,
                entry.resident_gpu_ids.clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let argument_plan = map_memory_requests
        .kfd_map_memory_argument_binding_plan(&alloc_result_plan, &device_ids_array_bindings);
    argument_plan.assert_kfd_map_memory_arguments_bound()?;

    let map_result_bindings = argument_plan
        .entries
        .iter()
        .map(|entry| {
            RuntimeKfdMapMemoryResultBinding::new(
                entry.slot,
                entry.tensor.as_str(),
                entry.map_args_handle,
                entry.map_args_device_ids_array_ptr,
                entry.map_args_n_devices,
                entry.map_args_n_devices,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let map_result_plan = argument_plan.kfd_map_memory_result_binding_plan(&map_result_bindings);
    map_result_plan.assert_kfd_map_memory_result_bound()?;
    let residency_plan = allocation_requests.kfd_residency_binding_plan(&map_result_plan);
    residency_plan.assert_kfd_residency_bound()?;

    let payload_plan = readiness
        .checkpoint
        .runtime_checkpoint_payload_binding_plan(
            &readiness.slots,
            key_resolution,
            &residency_plan,
            payload_bindings,
        );
    payload_plan.assert_checkpoint_payload_bound()?;
    Ok((payload_plan, residency_plan))
}

fn synthetic_checkpoint_host_staging_kfd_map_memory_results(
    staging_plan: &CheckpointPayloadDirectReadStagingPlan,
) -> Result<CheckpointPayloadHostStagingKfdMapMemoryResultBindingPlan> {
    let copy_plan = staging_plan.host_to_device_copy_plan()?;
    let upload_schedule = copy_plan.host_to_device_upload_schedule()?;
    let pin_request = upload_schedule.host_staging_pin_request()?;
    let virtual_address_plan =
        pin_request.host_virtual_address_binding_plan(HOST_STAGING_BASE_VA)?;
    let userptr_arguments =
        virtual_address_plan.kfd_userptr_pin_argument_plan(&RESIDENT_GPU_IDS)?;
    let vm_acquire_requests = userptr_arguments.kfd_vm_acquire_request_plan()?;
    let alloc_requests = vm_acquire_requests.kfd_userptr_alloc_request_plan()?;
    let result_bindings = alloc_requests
        .requests
        .iter()
        .map(|request| {
            let handle = synthetic_slot_u64(
                0x7b00_0000,
                request.request_index,
                1,
                "synthetic KFD USERPTR allocation handle",
            )?;
            let mmap_offset = synthetic_slot_u64(
                0x9b00_0000,
                request.request_index,
                0x1000,
                "synthetic KFD USERPTR allocation mmap offset",
            )?;
            Ok(
                CheckpointPayloadHostStagingKfdUserptrAllocResultBinding::new(
                    request.request_index,
                    request.argument_index,
                    request.page_span_index,
                    request.alloc_args_va_addr,
                    request.alloc_args_size,
                    request.alloc_args_gpu_id,
                    request.alloc_args_flags,
                    handle,
                    mmap_offset,
                ),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let alloc_results = alloc_requests.kfd_userptr_alloc_result_binding_plan(&result_bindings);
    alloc_results.assert_kfd_userptr_alloc_result_bound()?;
    alloc_results.assert_userptr_alloc_result_binding_only_boundary()?;
    let map_memory_requests = alloc_results.kfd_map_memory_request_plan()?;
    map_memory_requests.assert_no_map_memory_side_effect_boundary()?;
    let device_ids_array_bindings = map_memory_requests
        .requests
        .iter()
        .map(|request| {
            let device_ids_array_ptr = synthetic_slot_u64(
                0x3c00_0000,
                request.request_index,
                0x100,
                "synthetic KFD map-memory device ID array pointer",
            )?;
            Ok(
                CheckpointPayloadHostStagingKfdMapMemoryDeviceIdsArrayBinding::new(
                    request.request_index,
                    request.alloc_result_index,
                    request.alloc_request_index,
                    request.argument_index,
                    request.page_span_index,
                    request.map_args_handle,
                    device_ids_array_ptr,
                    request.resident_gpu_ids.clone(),
                ),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let map_memory_arguments =
        map_memory_requests.kfd_map_memory_argument_binding_plan(&device_ids_array_bindings);
    map_memory_arguments.assert_kfd_map_memory_arguments_bound()?;
    map_memory_arguments.assert_map_memory_argument_binding_only_boundary()?;
    let map_memory_result_bindings = map_memory_arguments
        .bindings
        .iter()
        .map(|binding| {
            CheckpointPayloadHostStagingKfdMapMemoryResultBinding::new(
                binding.request_index,
                binding.alloc_result_index,
                binding.alloc_request_index,
                binding.argument_index,
                binding.page_span_index,
                binding.map_args_handle,
                binding.map_args_device_ids_array_ptr,
                binding.map_args_n_devices,
                binding.map_args_n_devices,
            )
        })
        .collect::<Vec<_>>();
    let map_memory_results =
        map_memory_arguments.kfd_map_memory_result_binding_plan(&map_memory_result_bindings);
    map_memory_results.assert_kfd_map_memory_result_bound()?;
    map_memory_results.assert_map_memory_result_binding_receipt_only_boundary()?;
    Ok(map_memory_results)
}

fn synthetic_checkpoint_sdma_queue_reservation_results(
    queue_input: &CheckpointPayloadSdmaQueueReservationInput,
) -> Result<CheckpointPayloadSdmaQueueReservationResultBindingPlan> {
    let queue_ring_size_bytes = queue_input
        .queue_packet_byte_count
        .next_power_of_two()
        .max(4096);
    let result_bindings = queue_input
        .waves
        .iter()
        .map(|wave| {
            let queue_id = 17u32
                .checked_add(
                    u32::try_from(wave.wave_index)
                        .map_err(|_| anyhow!("synthetic SDMA queue wave index exceeds u32"))?,
                )
                .ok_or_else(|| anyhow!("synthetic SDMA queue ID overflows"))?;
            let queue_ring_base_va = synthetic_slot_u64(
                0x4d00_0000,
                wave.wave_index,
                0x1_0000,
                "synthetic SDMA queue ring base VA",
            )?;
            let queue_write_end = wave
                .first_sdma_packet_index
                .checked_add(wave.packet_request_count)
                .ok_or_else(|| anyhow!("synthetic SDMA queue write index overflows"))?;
            Ok(CheckpointPayloadSdmaQueueReservationResultBinding::new(
                wave.wave_index,
                wave.first_sdma_packet_index,
                wave.packet_request_count,
                wave.packet_offset_dwords,
                wave.packet_dword_count,
                queue_id,
                queue_ring_base_va,
                queue_ring_size_bytes,
                wave.first_sdma_packet_index,
                queue_write_end,
                u64::try_from(queue_write_end)
                    .map_err(|_| anyhow!("synthetic SDMA queue doorbell exceeds u64"))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let queue_results = queue_input.sdma_queue_reservation_result_binding_plan(&result_bindings);
    queue_results.assert_sdma_queue_reservation_result_bound()?;
    queue_results.assert_queue_reservation_result_binding_receipt_only_boundary()?;
    Ok(queue_results)
}

fn synthetic_checkpoint_copy_completion_signal_results(
    signal_input: &CheckpointPayloadCopyCompletionSignalBindingInput,
) -> Result<CheckpointPayloadCopyCompletionSignalResultBindingPlan> {
    let result_bindings = signal_input
        .bindings
        .iter()
        .map(|binding| {
            let signal_handle = synthetic_slot_u64(
                0x5100_0000,
                binding.binding_index,
                1,
                "synthetic copy completion signal handle",
            )?;
            let signal_device_va = synthetic_slot_u64(
                0x5a00_0000,
                binding.binding_index,
                0x1000,
                "synthetic copy completion signal device VA",
            )?;
            Ok(CheckpointPayloadCopyCompletionSignalResultBinding::new(
                binding.binding_index,
                binding.completion_signal_index,
                binding.wave_index,
                binding.completion_packet_index,
                binding.completion_packet_offset_dwords,
                binding.completion_packet_dword_count,
                signal_handle,
                signal_device_va,
                binding.signal_initial_value,
                binding.signal_completion_value,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let signal_results = signal_input.copy_completion_signal_result_binding_plan(&result_bindings);
    signal_results.assert_copy_completion_signal_result_bound()?;
    signal_results.assert_copy_completion_signal_result_binding_receipt_only_boundary()?;
    Ok(signal_results)
}

fn synthetic_checkpoint_sdma_copy_packet_materialization_results(
    packet_input: &CheckpointPayloadSdmaCopyPacketMaterializationInput,
    queue_results: &CheckpointPayloadSdmaQueueReservationResultBindingPlan,
    signal_results: &CheckpointPayloadCopyCompletionSignalResultBindingPlan,
    host_staging_base_va: u64,
) -> Result<CheckpointPayloadSdmaCopyPacketMaterializationResultBindingPlan> {
    queue_results.assert_sdma_queue_reservation_result_bound()?;
    signal_results.assert_copy_completion_signal_result_bound()?;
    let queues_by_wave = queue_results
        .results
        .iter()
        .map(|result| (result.wave_index, result))
        .collect::<BTreeMap<_, _>>();
    let signals_by_index = signal_results
        .results
        .iter()
        .map(|result| (result.completion_signal_index, result))
        .collect::<BTreeMap<_, _>>();
    let mut result_bindings = Vec::with_capacity(packet_input.packet_row_count);

    for packet in &packet_input.copy_packets {
        let queue = queues_by_wave.get(&packet.wave_index).ok_or_else(|| {
            anyhow!(
                "synthetic packet materialization copy packet {} has no queue result for wave {}",
                packet.packet_row_index,
                packet.wave_index
            )
        })?;
        let host_staging_offset = u64::try_from(packet.host_staging_offset)
            .map_err(|_| anyhow!("synthetic packet materialization host offset exceeds u64"))?;
        let host_virtual_address = host_staging_base_va
            .checked_add(host_staging_offset)
            .ok_or_else(|| anyhow!("synthetic packet materialization host VA overflows"))?;
        result_bindings.push(
            CheckpointPayloadSdmaCopyPacketMaterializationResultBinding::new(
                packet.packet_row_index,
                packet.sdma_packet_index,
                packet.packet_kind,
                packet.packet_offset_dwords,
                packet.packet_dword_count,
                packet.packet_bytes,
                queue.queue_id,
                queue.queue_ring_base_va,
                packet.sdma_packet_index,
                host_virtual_address,
                packet.destination_device_va_begin,
                packet.destination_device_va_end,
                0,
                0,
                0,
                0,
                true,
            ),
        );
    }

    for packet in &packet_input.completion_packets {
        let queue = queues_by_wave.get(&packet.wave_index).ok_or_else(|| {
            anyhow!(
                "synthetic packet materialization completion packet {} has no queue result for wave {}",
                packet.packet_row_index,
                packet.wave_index
            )
        })?;
        let signal = signals_by_index
            .get(&packet.completion_signal_index)
            .ok_or_else(|| {
                anyhow!(
                    "synthetic packet materialization completion packet {} has no signal result {}",
                    packet.packet_row_index,
                    packet.completion_signal_index
                )
            })?;
        result_bindings.push(
            CheckpointPayloadSdmaCopyPacketMaterializationResultBinding::new(
                packet.packet_row_index,
                packet.sdma_packet_index,
                packet.packet_kind,
                packet.packet_offset_dwords,
                packet.packet_dword_count,
                packet.packet_bytes,
                queue.queue_id,
                queue.queue_ring_base_va,
                packet.sdma_packet_index,
                0,
                0,
                0,
                signal.signal_handle,
                signal.signal_device_va,
                packet.signal_initial_value,
                packet.signal_completion_value,
                true,
            ),
        );
    }

    let packet_results =
        packet_input.sdma_copy_packet_materialization_result_binding_plan(&result_bindings);
    packet_results.assert_sdma_copy_packet_materialization_result_bound()?;
    packet_results.assert_packet_materialization_result_binding_receipt_only_boundary()?;
    Ok(packet_results)
}

fn synthetic_checkpoint_sdma_copy_packet_validation_results(
    validation_input: &CheckpointPayloadSdmaCopyPacketValidationInput,
    packet_results: &CheckpointPayloadSdmaCopyPacketMaterializationResultBindingPlan,
) -> Result<CheckpointPayloadSdmaCopyPacketValidationResultBindingPlan> {
    packet_results.assert_sdma_copy_packet_materialization_result_bound()?;
    packet_results.assert_packet_materialization_result_binding_receipt_only_boundary()?;
    let materialized_by_packet_row = packet_results
        .results
        .iter()
        .map(|result| (result.packet_row_index, result))
        .collect::<BTreeMap<_, _>>();
    let mut result_bindings = Vec::with_capacity(validation_input.packet_validation_row_count);

    for row in &validation_input.validation_rows {
        let materialized = materialized_by_packet_row
            .get(&row.packet_row_index)
            .ok_or_else(|| {
                anyhow!(
                    "synthetic packet validation row {} has no materialization result for packet row {}",
                    row.validation_row_index,
                    row.packet_row_index
                )
            })?;
        let is_copy = row.validation_scope == CHECKPOINT_SDMA_COPY_PACKET_VALIDATION_SCOPE_LABEL;
        let is_completion =
            row.validation_scope == CHECKPOINT_SDMA_COMPLETION_PACKET_VALIDATION_SCOPE_LABEL;
        result_bindings.push(CheckpointPayloadSdmaCopyPacketValidationResultBinding::new(
            row.validation_row_index,
            row.packet_row_index,
            row.sdma_packet_index,
            row.validation_scope,
            row.packet_kind,
            row.packet_offset_dwords,
            row.packet_dword_count,
            row.packet_bytes,
            row.expected_packet_bytes,
            row.payload_bytes,
            row.host_staging_offset,
            row.destination_device_va_begin,
            row.destination_device_va_end,
            row.signal_initial_value,
            row.signal_completion_value,
            materialized.queue_id,
            materialized.queue_ring_base_va,
            materialized.queue_packet_write_index,
            if is_copy {
                materialized.host_virtual_address
            } else {
                0
            },
            if is_completion {
                materialized.signal_handle
            } else {
                0
            },
            if is_completion {
                materialized.signal_device_va
            } else {
                0
            },
            true,
            true,
            true,
            true,
            is_copy,
            is_completion,
            true,
            true,
        ));
    }

    let validation_results =
        validation_input.sdma_copy_packet_validation_result_binding_plan(&result_bindings);
    validation_results.assert_sdma_copy_packet_validation_result_bound()?;
    validation_results.assert_packet_validation_result_binding_receipt_only_boundary()?;
    Ok(validation_results)
}

fn synthetic_slot_u64(base: u64, slot: usize, stride: u64, label: &str) -> Result<u64> {
    let slot = u64::try_from(slot).map_err(|_| anyhow!("{label} slot does not fit in u64"))?;
    let offset = slot
        .checked_mul(stride)
        .ok_or_else(|| anyhow!("{label} slot offset overflows"))?;
    base.checked_add(offset)
        .ok_or_else(|| anyhow!("{label} value overflows"))
}

fn synthetic_safetensors_shard(plan: &ModelCheckpointBindingPlan) -> Result<Vec<u8>> {
    let mut tensors = Vec::new();
    let mut data_offset = 0usize;
    for entry in &plan.entries {
        if entry.dtype != DType::F16 {
            return Err(anyhow!(
                "synthetic reference shard supports F16 weights only, got {:?} for {}",
                entry.dtype,
                entry.tensor
            ));
        }
        if let Some((prefix, suffix)) = entry.checkpoint_key.split_once('*') {
            let expert_count = entry.shape.first().copied().ok_or_else(|| {
                anyhow!(
                    "wildcard checkpoint key {} has empty shape",
                    entry.checkpoint_key
                )
            })?;
            let expert_shape = &entry.shape[1..];
            let expert_bytes = storage_bytes(DType::F16, expert_shape)?;
            for expert in 0..expert_count {
                push_tensor_header(
                    &mut tensors,
                    format!("{prefix}{expert}{suffix}"),
                    expert_shape,
                    &mut data_offset,
                    expert_bytes,
                );
            }
        } else {
            let bytes = storage_bytes(entry.dtype, &entry.shape)?;
            push_tensor_header(
                &mut tensors,
                entry.checkpoint_key.clone(),
                &entry.shape,
                &mut data_offset,
                bytes,
            );
        }
    }

    let header = format!("{{{}}}", tensors.join(","));
    let mut out = Vec::with_capacity(SAFETENSORS_PREFIX_BYTES + header.len() + data_offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(out.len() + data_offset, 0);
    Ok(out)
}

fn synthetic_safetensors_index(
    shard: &SafetensorsShard,
    shard_path: &PathBuf,
    total_checkpoint_bytes: usize,
) -> Result<String> {
    let shard_file = shard_path
        .file_name()
        .ok_or_else(|| anyhow!("synthetic shard path has no file name"))?
        .to_string_lossy();
    let mut keys = shard.tensors.keys().cloned().collect::<Vec<_>>();
    keys.sort();
    let entries = keys
        .into_iter()
        .map(|key| format!("\"{key}\":\"{shard_file}\""))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"metadata\":{{\"total_size\":{total_checkpoint_bytes}}},\"weight_map\":{{{entries}}}}}"
    ))
}

fn push_tensor_header(
    tensors: &mut Vec<String>,
    name: String,
    shape: &[usize],
    data_offset: &mut usize,
    byte_len: usize,
) {
    let begin = *data_offset;
    let end = begin + byte_len;
    *data_offset = end;
    let shape = shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    tensors.push(format!(
        "\"{name}\":{{\"dtype\":\"F16\",\"shape\":[{shape}],\"data_offsets\":[{begin},{end}]}}"
    ));
}

fn storage_bytes(dtype: DType, shape: &[usize]) -> Result<usize> {
    let elements = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| anyhow!("synthetic safetensors shape {shape:?} overflows"))?;
    dtype.storage_bytes_for_elements(elements)
}

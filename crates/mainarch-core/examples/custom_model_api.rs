use anyhow::Result;
use mainarch_core::model_api::prelude::*;

struct ToyPluginDecoder {
    vocab: usize,
    hidden: usize,
}

impl Default for ToyPluginDecoder {
    fn default() -> Self {
        Self {
            vocab: 256,
            hidden: 128,
        }
    }
}

impl ModelDefinition for ToyPluginDecoder {
    fn name(&self) -> &str {
        "toy-plugin-decoder"
    }

    fn define(&self, api: &mut dyn ModelPrimitiveApi) -> Result<()> {
        api.declare_tensor(TensorSpec::new(
            "input_ids",
            DType::U32,
            vec![1],
            TensorRole::Token,
        ))?;
        api.declare_tensor(TensorSpec::new(
            "hidden",
            DType::F16,
            vec![self.hidden],
            TensorRole::Activation,
        ))?;
        api.declare_tensor(TensorSpec::new(
            "logits",
            DType::F16,
            vec![self.vocab],
            TensorRole::Logits,
        ))?;
        api.declare_tensor(TensorSpec::new(
            "next_token",
            DType::U32,
            vec![1],
            TensorRole::Token,
        ))?;
        api.declare_tensor(
            TensorSpec::new(
                "embed_weight",
                DType::F16,
                vec![self.vocab, self.hidden],
                TensorRole::Weight,
            )
            .with_checkpoint_key("toy.embed_tokens.weight")?,
        )?;
        api.declare_tensor(
            TensorSpec::new(
                "lm_head",
                DType::F16,
                vec![self.vocab, self.hidden],
                TensorRole::Weight,
            )
            .with_checkpoint_key("toy.lm_head.weight")?,
        )?;

        api.begin_stage("embedding", ModelStageKind::Embedding)?;
        api.emit(PrimitiveOp::EmbeddingLookup(EmbeddingLookup {
            name: "embed_tokens".to_string(),
            token_ids: "input_ids".into(),
            weight: "embed_weight".into(),
            output: "hidden".into(),
            vocab: self.vocab,
            hidden: self.hidden,
        }))?;
        api.end_stage()?;

        api.begin_stage("output", ModelStageKind::Output)?;
        api.emit(PrimitiveOp::Linear(Linear {
            name: "lm_head".to_string(),
            input: "hidden".into(),
            weight: "lm_head".into(),
            scale: None,
            output: "logits".into(),
            in_features: self.hidden,
            out_features: self.vocab,
            weight_format: WeightFormat::F16,
            parallelism: LinearParallelism::Replicated,
        }))?;
        api.end_stage()?;

        api.begin_stage("sampling", ModelStageKind::Sampling)?;
        api.emit(PrimitiveOp::ArgmaxSample(ArgmaxSample {
            name: "sample_argmax".to_string(),
            logits: "logits".into(),
            output_token: "next_token".into(),
            vocab: self.vocab,
        }))?;
        api.end_stage()?;

        Ok(())
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let emit_runtime_launch_request_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-launch-request-receipt");
    let emit_runtime_submission_gate_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-submission-gate-receipt");
    let emit_runtime_resolved_submission_gate_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-resolved-submission-gate-receipt");
    let emit_runtime_resolved_submission_prerequisite_plan_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-resolved-submission-prerequisite-plan-receipt");
    let emit_runtime_resolved_submission_blocker_report_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-resolved-submission-blocker-report-receipt");
    let emit_runtime_submission_blocker_report_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-submission-blocker-report-receipt");
    let emit_runtime_submission_prerequisite_plan_receipt = args
        .iter()
        .any(|arg| arg == "--runtime-submission-prerequisite-plan-receipt");

    let model = ToyPluginDecoder::default();
    let catalog = MainarchPrimitiveLoweringCatalog::mi355_reference();
    let plugin_inspection = inspect_model_plugin(&model, &catalog)?;
    plugin_inspection.assert_consistent()?;
    plugin_inspection.assert_accepted()?;
    let plugin_inspection_accepted = plugin_inspection.is_accepted();
    let plugin_summary = plugin_inspection.summary();
    plugin_summary.assert_consistent_with(&plugin_inspection)?;
    plugin_inspection.catalog.assert_consistent()?;
    assert_eq!(plugin_inspection.primitive_vocabulary.len(), 12);
    assert_eq!(plugin_inspection.stage_vocabulary.len(), 5);
    assert_eq!(plugin_inspection.catalog.primitive_kind_count, 12);
    assert_eq!(plugin_inspection.catalog.primitive_case_count, 22);
    assert_eq!(plugin_inspection.catalog.native_gpu_case_count, 16);
    assert_eq!(plugin_inspection.catalog.fused_native_gpu_case_count, 1);
    assert_eq!(plugin_inspection.catalog.gap_case_count, 5);
    let primitive_vocabulary_count = plugin_inspection.primitive_vocabulary.len();
    let stage_vocabulary_count = plugin_inspection.stage_vocabulary.len();
    let catalog_descriptor = plugin_inspection.catalog.clone();
    let graph = plugin_inspection.graph;
    let readiness = plugin_inspection.readiness;
    let plugin_manifest = plugin_inspection.manifest;
    let plugin_compatibility = plugin_inspection.compatibility;
    assert_eq!(graph.plugin_manifest(&catalog)?, plugin_manifest);
    plugin_manifest.assert_static_metadata_ready()?;
    let dispatch_slot_args = readiness
        .dispatch_intents
        .entries
        .iter()
        .map(|entry| entry.slot_arguments.len())
        .sum::<usize>();
    let dispatch_scalar_args = readiness
        .dispatch_intents
        .entries
        .iter()
        .map(|entry| entry.scalar_arguments.len())
        .sum::<usize>();
    let slot_bindings = readiness.slots.metadata_binding_template("custom")?;
    let device_pointer_bindings = readiness
        .slots
        .device_pointer_binding_template(0x1_0000_0000, DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT)?;
    let device_pointer_validation = readiness
        .slots
        .validate_complete_device_pointer_bindings(&device_pointer_bindings);
    device_pointer_validation.assert_complete()?;
    let admission = readiness.validate_metadata_runtime_admission(&slot_bindings);
    admission.assert_admitted()?;
    let stage_launch_candidates = admission.runtime_stage_launch_candidate_plan()?;
    let launch_windows =
        stage_launch_candidates.launch_window_plan(DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES)?;
    let launch_entrypoints = stage_launch_candidates.launch_entrypoint_provenance_plan()?;
    let launch_kernels = stage_launch_candidates.launch_kernel_requirement_plan()?;
    let code_object = CodeObjectInfo::inspect(MAINARCH_KERNELS_GFX950)?;
    let launch_kernel_metadata = launch_kernels.kernel_metadata_plan(&code_object)?;
    let launch_code_object_loads = launch_kernel_metadata.code_object_load_request_plan()?;
    launch_code_object_loads.assert_consistent()?;
    let launch_preflight = admission
        .runtime_launch_preflight_report(&code_object, DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES)?;
    launch_preflight.assert_ready()?;
    let launch_packet_fields = launch_preflight.aql_packet_field_plan()?;
    launch_packet_fields.assert_metadata_handoff_ready()?;
    let launch_kernel_selection = launch_preflight.kernel_selection_readiness_plan()?;
    launch_kernel_selection.assert_consistent()?;
    let launch_host_launcher_branch_requests =
        launch_kernel_selection.host_launcher_branch_resolution_request_plan()?;
    launch_host_launcher_branch_requests.assert_consistent()?;
    let launch_device_arguments =
        launch_preflight.device_argument_plan(&device_pointer_validation)?;
    launch_device_arguments.assert_bound()?;
    let launch_staging = launch_preflight.staging_footprint_plan()?;
    launch_staging.assert_consistent()?;
    let launch_staging_layout = launch_preflight.staging_layout_plan()?;
    launch_staging_layout.assert_consistent()?;
    let launch_completion_signals = launch_preflight.completion_signal_plan()?;
    launch_completion_signals.assert_consistent()?;
    let launch_completion_signal_bindings =
        launch_completion_signals.completion_signal_binding_request_plan()?;
    launch_completion_signal_bindings.assert_consistent()?;
    let launch_queue_slots = launch_preflight.queue_slot_plan()?;
    launch_queue_slots.assert_consistent()?;
    let launch_queue_reservations = launch_queue_slots.queue_reservation_request_plan()?;
    launch_queue_reservations.assert_consistent()?;
    let launch_geometry = launch_preflight.dispatch_geometry_plan()?;
    launch_geometry.assert_consistent()?;
    let launch_kernarg_layout = launch_preflight.kernarg_layout_plan(&device_pointer_validation)?;
    launch_kernarg_layout.assert_consistent()?;
    let launch_kernarg_serialization =
        launch_preflight.kernarg_serialization_plan(&device_pointer_validation)?;
    launch_kernarg_serialization.assert_consistent()?;
    let launch_kernarg_allocations =
        launch_kernarg_serialization.kernarg_allocation_request_plan()?;
    launch_kernarg_allocations.assert_consistent()?;
    let launch_aql_templates =
        launch_preflight.aql_packet_template_plan(&device_pointer_validation)?;
    launch_aql_templates.assert_consistent()?;
    let launch_kernel_argument_abi = admission
        .runtime_launch_kernel_argument_abi_verification_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi.assert_consistent()?;
    let launch_kernel_argument_abi_size_receipt = admission
        .runtime_launch_kernel_argument_abi_size_compatibility_receipt(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_size_receipt.assert_consistent()?;
    let launch_kernel_argument_abi_gaps = admission
        .runtime_launch_kernel_argument_abi_verification_gap_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_gaps.assert_consistent()?;
    assert_eq!(
        launch_kernel_argument_abi.kernel_argument_abi_verification_gap_report()?,
        launch_kernel_argument_abi_gaps
    );
    let launch_kernel_argument_abi_capacity_requests = admission
        .runtime_launch_kernel_argument_abi_capacity_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_capacity_requests.assert_consistent()?;
    assert_eq!(
        launch_kernel_argument_abi_gaps.kernel_argument_abi_capacity_request_plan()?,
        launch_kernel_argument_abi_capacity_requests
    );
    let launch_kernel_argument_abi_schema_requests =
        launch_kernel_argument_abi.kernel_argument_abi_schema_request_plan()?;
    launch_kernel_argument_abi_schema_requests.assert_consistent()?;
    let launch_kernel_argument_abi_semantics = admission
        .runtime_launch_kernel_argument_abi_semantic_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_semantics.assert_consistent()?;
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_plan(&device_pointer_validation)?,
        launch_kernel_argument_abi_semantics
    );
    assert_eq!(
        launch_aql_templates.kernel_argument_abi_semantic_plan(&launch_kernarg_serialization)?,
        launch_kernel_argument_abi_semantics
    );
    assert_eq!(
        launch_kernarg_serialization.kernel_argument_abi_semantic_plan(&launch_aql_templates)?,
        launch_kernel_argument_abi_semantics
    );
    let launch_kernel_argument_abi_semantic_gaps = admission
        .runtime_launch_kernel_argument_abi_semantic_gap_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_semantic_gaps.assert_consistent()?;
    assert_eq!(
        launch_kernel_argument_abi_semantics.kernel_argument_abi_semantic_gap_report()?,
        launch_kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_gap_report(&device_pointer_validation)?,
        launch_kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        launch_aql_templates
            .kernel_argument_abi_semantic_gap_report(&launch_kernarg_serialization)?,
        launch_kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        launch_kernarg_serialization
            .kernel_argument_abi_semantic_gap_report(&launch_aql_templates)?,
        launch_kernel_argument_abi_semantic_gaps
    );
    let launch_kernel_argument_abi_semantic_projection = admission
        .runtime_launch_kernel_argument_abi_semantic_projection_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_semantic_projection.assert_consistent()?;
    assert_eq!(
        launch_preflight
            .kernel_argument_abi_semantic_projection_plan(&device_pointer_validation)?,
        launch_kernel_argument_abi_semantic_projection
    );
    assert_eq!(
        launch_aql_templates
            .kernel_argument_abi_semantic_projection_plan(&launch_kernarg_serialization)?,
        launch_kernel_argument_abi_semantic_projection
    );
    assert_eq!(
        launch_kernarg_serialization
            .kernel_argument_abi_semantic_projection_plan(&launch_aql_templates)?,
        launch_kernel_argument_abi_semantic_projection
    );
    let launch_kernel_argument_abi_semantic_projection_gaps = admission
        .runtime_launch_kernel_argument_abi_semantic_projection_gap_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_kernel_argument_abi_semantic_projection_gaps.assert_consistent()?;
    assert_eq!(
        launch_kernel_argument_abi_semantic_projection
            .kernel_argument_abi_semantic_projection_gap_report()?,
        launch_kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        launch_preflight
            .kernel_argument_abi_semantic_projection_gap_report(&device_pointer_validation)?,
        launch_kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        launch_aql_templates
            .kernel_argument_abi_semantic_projection_gap_report(&launch_kernarg_serialization)?,
        launch_kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        launch_kernarg_serialization
            .kernel_argument_abi_semantic_projection_gap_report(&launch_aql_templates)?,
        launch_kernel_argument_abi_semantic_projection_gaps
    );
    let launch_semantic_projection_candidate_recommendations =
        launch_kernel_argument_abi_semantic_projection
            .kernel_argument_abi_semantic_projection_candidate_recommendation_plan()?;
    launch_semantic_projection_candidate_recommendations.assert_consistent()?;
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
            &device_pointer_validation,
        )?,
        launch_semantic_projection_candidate_recommendations
    );
    assert_eq!(
        launch_aql_templates
            .kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &launch_kernarg_serialization,
            )?,
        launch_semantic_projection_candidate_recommendations
    );
    assert_eq!(
        launch_kernarg_serialization
            .kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &launch_aql_templates,
            )?,
        launch_semantic_projection_candidate_recommendations
    );
    assert_eq!(
        admission
            .runtime_launch_kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &code_object,
                DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
                &device_pointer_validation,
            )?,
        launch_semantic_projection_candidate_recommendations
    );
    let launch_semantic_projection_candidate_selection_requests =
        launch_semantic_projection_candidate_recommendations
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &launch_kernel_argument_abi_semantic_projection,
            )?;
    launch_semantic_projection_candidate_selection_requests.assert_consistent()?;
    assert_eq!(
        launch_kernel_argument_abi_semantic_projection
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan()?,
        launch_semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
            &device_pointer_validation,
        )?,
        launch_semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        launch_aql_templates
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &launch_kernarg_serialization,
            )?,
        launch_semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        launch_kernarg_serialization
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &launch_aql_templates,
            )?,
        launch_semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        admission
            .runtime_launch_kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &code_object,
                DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
                &device_pointer_validation,
            )?,
        launch_semantic_projection_candidate_selection_requests
    );
    let launch_kernel_recommendations =
        launch_kernel_argument_abi.kernel_candidate_recommendation_plan()?;
    launch_kernel_recommendations.assert_consistent()?;
    let launch_semantic_projection_recommendations = launch_kernel_recommendations
        .kernel_argument_abi_semantic_projection_recommendation_report(
            &launch_kernel_argument_abi_semantic_projection,
        )?;
    launch_semantic_projection_recommendations.assert_consistent()?;
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_projection_recommendation_report(
            &device_pointer_validation,
        )?,
        launch_semantic_projection_recommendations
    );
    assert_eq!(
        launch_aql_templates.kernel_argument_abi_semantic_projection_recommendation_report(
            &launch_kernarg_serialization,
        )?,
        launch_semantic_projection_recommendations
    );
    assert_eq!(
        launch_kernarg_serialization
            .kernel_argument_abi_semantic_projection_recommendation_report(&launch_aql_templates)?,
        launch_semantic_projection_recommendations
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_semantic_projection_recommendation_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        launch_semantic_projection_recommendations
    );
    let launch_kernel_selection_requests =
        launch_kernel_recommendations.kernel_candidate_selection_request_plan()?;
    launch_kernel_selection_requests.assert_consistent()?;
    let launch_aql_relocations =
        launch_preflight.aql_packet_relocation_plan(&device_pointer_validation)?;
    launch_aql_relocations.assert_consistent()?;
    let launch_aql_byte_templates =
        launch_preflight.aql_packet_byte_template_plan(&device_pointer_validation)?;
    launch_aql_byte_templates.assert_consistent()?;
    let launch_aql_materialization = launch_aql_byte_templates.aql_packet_materialization_plan()?;
    launch_aql_materialization.assert_consistent()?;
    let launch_aql_live_bindings = launch_aql_materialization.aql_live_relocation_binding_plan()?;
    launch_aql_live_bindings.assert_consistent()?;
    let launch_code_object_base_bindings = launch_code_object_loads
        .code_object_base_binding_request_plan(&launch_aql_live_bindings)?;
    launch_code_object_base_bindings.assert_consistent()?;
    let launch_execution = admission.runtime_launch_execution_readiness_report(
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_execution.assert_consistent()?;
    assert_eq!(
        launch_preflight.execution_readiness_report(&device_pointer_validation)?,
        launch_execution
    );
    let launch_execution_requests = admission.runtime_launch_execution_request_plan(
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_execution_requests.assert_consistent()?;
    assert_eq!(
        launch_preflight.execution_request_plan(&device_pointer_validation)?,
        launch_execution_requests
    );
    if emit_runtime_launch_request_receipt {
        print!("{}", launch_execution_requests.receipt_text());
        return Ok(());
    }
    let launch_submission_gate = admission.runtime_launch_submission_gate(
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_submission_gate.assert_consistent()?;
    assert_eq!(
        launch_preflight.submission_gate(&device_pointer_validation)?,
        launch_submission_gate
    );
    assert_eq!(
        launch_execution_requests.submission_gate()?,
        launch_submission_gate
    );
    if emit_runtime_submission_gate_receipt {
        print!("{}", launch_submission_gate.receipt_text());
        return Ok(());
    }
    let launch_submission_blockers = admission.runtime_launch_submission_blocker_report(
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_submission_blockers.assert_consistent()?;
    assert_eq!(
        launch_preflight.submission_blocker_report(&device_pointer_validation)?,
        launch_submission_blockers
    );
    assert_eq!(
        launch_submission_gate.blocker_report()?,
        launch_submission_blockers
    );
    if emit_runtime_submission_blocker_report_receipt {
        print!("{}", launch_submission_blockers.receipt_text());
        return Ok(());
    }
    let launch_submission_prerequisites = admission.runtime_launch_submission_prerequisite_plan(
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_submission_prerequisites.assert_consistent()?;
    assert_eq!(
        launch_preflight.submission_prerequisite_plan(&device_pointer_validation)?,
        launch_submission_prerequisites
    );
    assert_eq!(
        launch_execution_requests.submission_prerequisite_plan()?,
        launch_submission_prerequisites
    );
    if emit_runtime_submission_prerequisite_plan_receipt {
        print!("{}", launch_submission_prerequisites.receipt_text());
        return Ok(());
    }
    let batch_validation = RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
        .validate_batch_reservation_plan_proof(live_aql_batch_plan_proof())?;
    let materialized_validation = RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
        .validate_materialized_packet_plan_proof(live_aql_materialized_packet_plan_proof())?;
    let live_aql_proof_validations = [batch_validation, materialized_validation];
    let launch_resolved_submission_prerequisites = launch_execution_requests
        .synthetic_cpu_resolved_submission_prerequisite_plan(
            &live_aql_proof_validations,
            "custom_model_public_example_cpu_receipt",
        )?;
    launch_resolved_submission_prerequisites.assert_consistent()?;
    assert_eq!(
        launch_resolved_submission_prerequisites.prerequisite_count,
        launch_submission_prerequisites.prerequisite_count
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.satisfied_prerequisite_count,
        launch_resolved_submission_prerequisites.prerequisite_count
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.unsatisfied_prerequisite_count,
        0
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.next_action_count,
        0
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.pending_component_request_count,
        0
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.live_aql_proof_validation_pending_count,
        0
    );
    assert!(launch_resolved_submission_prerequisites.request_plan_ready);
    assert!(launch_resolved_submission_prerequisites.execution_readiness_ready);
    assert!(launch_resolved_submission_prerequisites.all_prerequisites_satisfied);
    assert!(launch_resolved_submission_prerequisites.submission_ready);
    assert!(launch_resolved_submission_prerequisites
        .next_action_request_plan_names()
        .is_empty());
    if emit_runtime_resolved_submission_prerequisite_plan_receipt {
        print!(
            "{}",
            launch_resolved_submission_prerequisites.receipt_text()
        );
        return Ok(());
    }
    let launch_resolved_submission_gate =
        launch_resolved_submission_prerequisites.submission_gate()?;
    assert_eq!(
        launch_resolved_submission_gate,
        launch_execution_requests.synthetic_cpu_resolved_submission_gate(
            &live_aql_proof_validations,
            "custom_model_public_example_cpu_receipt",
        )?
    );
    launch_resolved_submission_gate.assert_consistent()?;
    assert!(launch_resolved_submission_gate.request_plan_ready);
    assert!(launch_resolved_submission_gate.execution_readiness_ready);
    assert!(launch_resolved_submission_gate.all_components_applied);
    assert!(launch_resolved_submission_gate.all_live_aql_proof_validations_applied);
    assert!(launch_resolved_submission_gate.no_live_aql_submission_side_effects);
    assert!(launch_resolved_submission_gate.no_live_queue_mutation);
    assert_eq!(launch_resolved_submission_gate.component_pending_count, 0);
    assert_eq!(
        launch_resolved_submission_gate.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        launch_resolved_submission_gate.live_aql_submitting_surface_count,
        0
    );
    assert_eq!(
        launch_resolved_submission_gate.live_queue_mutating_component_count,
        0
    );
    assert_eq!(launch_resolved_submission_gate.execution_blocker_count, 0);
    assert_eq!(launch_resolved_submission_gate.submission_blocker_count, 0);
    assert!(launch_resolved_submission_gate.blockers.is_empty());
    assert!(launch_resolved_submission_gate.submission_ready);
    let launch_resolved_submission_blocker_report = launch_execution_requests
        .synthetic_cpu_resolved_submission_blocker_report(
            &live_aql_proof_validations,
            "custom_model_public_example_cpu_receipt",
        )?;
    assert_eq!(
        launch_resolved_submission_blocker_report,
        launch_resolved_submission_gate.blocker_report()?
    );
    launch_resolved_submission_blocker_report.assert_consistent()?;
    assert!(launch_resolved_submission_blocker_report.submission_ready);
    assert_eq!(launch_resolved_submission_blocker_report.blocker_count, 0);
    if emit_runtime_resolved_submission_blocker_report_receipt {
        print!(
            "{}",
            launch_resolved_submission_blocker_report.receipt_text()
        );
        return Ok(());
    }
    if emit_runtime_resolved_submission_gate_receipt {
        print!("{}", launch_resolved_submission_gate.receipt_text());
        return Ok(());
    }
    let checkpoint_payloads = readiness.synthetic_cpu_runtime_checkpoint_payload_binding_plan(
        synthetic_available_checkpoint_keys(&readiness.checkpoint),
        "custom-model.synthetic.safetensors",
        &[1002, 1001],
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    checkpoint_payloads.assert_checkpoint_payload_bound()?;
    let launch_arguments = stage_launch_candidates.launch_argument_plan()?;
    let static_issues = &admission.static_issues;
    let slot_binding_validation = &admission.slot_bindings;
    let dispatch_binding_validation = &admission.dispatch_bindings;
    let stage_binding_validation = &admission.stage_bindings;
    let stage_dispatch_binding_validation = &admission.stage_dispatch_bindings;
    let typed_live_aql_proof_steps = RuntimeLaunchExecutionRequestStep::DESCRIPTORS
        .iter()
        .filter(|descriptor| descriptor.live_aql_proof_input.is_some())
        .count();

    println!(
        "model_api_contract: {} receipt_fingerprint={} lines={}",
        MODEL_API_CONTRACT,
        MODEL_API_CONTRACT.receipt_fingerprint(),
        MODEL_API_CONTRACT.receipt_lines().len()
    );
    println!("model: {}", graph.name);
    println!(
        "model_api_vocabulary: primitive_kinds={} stage_kinds={}",
        primitive_vocabulary_count, stage_vocabulary_count
    );
    println!(
        "plugin_inspection: consistent=true accepted={}",
        plugin_inspection_accepted
    );
    println!(
        "plugin_summary: receipt_fingerprint={} accepted={} static_ready={} compatibility_issues={} model_primitives={} model_stages={} tensors={} ops={} dispatches={} catalog_cases={} catalog_gaps={} live_execution_supported={}",
        plugin_summary.receipt_fingerprint(),
        plugin_summary.accepted,
        plugin_summary.static_ready,
        plugin_summary.compatibility_issue_count,
        plugin_summary.model_primitive_kind_count,
        plugin_summary.model_stage_kind_count,
        plugin_summary.tensor_count,
        plugin_summary.op_count,
        plugin_summary.runtime_dispatch_count,
        plugin_summary.catalog_primitive_case_count,
        plugin_summary.catalog_gap_case_count,
        plugin_summary.live_execution_supported
    );
    println!(
        "catalog_capabilities: target={} primitive_kinds={} cases={} native_gpu_cases={} fused_native_gpu_cases={} gap_cases={} parameterized={}",
        catalog_descriptor.target,
        catalog_descriptor.primitive_kind_count,
        catalog_descriptor.primitive_case_count,
        catalog_descriptor.native_gpu_case_count,
        catalog_descriptor.fused_native_gpu_case_count,
        catalog_descriptor.gap_case_count,
        catalog_descriptor.parameterized
    );
    println!(
        "plugin_manifest: contract={} fingerprint={} version={} stability={} target={} primitive_kinds={} stage_kinds={} tensors={} ops={} stages={} checkpoint_weights={} slots={} dispatches={} launch_steps={} live_aql_proof_steps={} static_ready={} live_execution_supported={}",
        plugin_manifest.contract.name,
        plugin_manifest.contract_fingerprint,
        plugin_manifest.contract.version,
        plugin_manifest.contract.stability,
        plugin_manifest.target,
        plugin_manifest.primitive_kinds.len(),
        plugin_manifest.stage_kinds.len(),
        plugin_manifest.tensor_count,
        plugin_manifest.op_count,
        plugin_manifest.stage_count,
        plugin_manifest.checkpoint_weight_count,
        plugin_manifest.runtime_slot_count,
        plugin_manifest.runtime_dispatch_count,
        plugin_manifest.runtime_launch_request_step_count,
        plugin_manifest.runtime_live_aql_proof_step_count,
        plugin_manifest.static_ready,
        plugin_manifest.live_execution_supported
    );
    println!(
        "plugin_compatibility: accepted={} issues={} target_matches={} fingerprint_matches={} static_metadata_ready={} live_execution_supported={}",
        plugin_compatibility.is_accepted(),
        plugin_compatibility.issues.len(),
        plugin_compatibility.target_matches,
        plugin_compatibility.fingerprint_matches,
        plugin_compatibility.static_metadata_ready,
        plugin_compatibility.live_execution_supported
    );
    println!(
        "graph: tensors={} ops={} stages={} staged_ops={} unstaged_ops={}",
        readiness.graph.tensors,
        readiness.graph.ops,
        readiness.graph.stages,
        readiness.graph.staged_ops,
        readiness.graph.unstaged_ops
    );
    println!(
        "readiness_issues: ready={} issues={}",
        static_issues.is_ready(),
        static_issues.issues.len()
    );
    println!(
        "runtime_admission: admitted={} issues={}",
        admission.is_admitted(),
        admission.issue_count()
    );
    println!(
        "checkpoint: bound_weights={} missing_weights={} checkpoint_bytes={}",
        readiness.checkpoint.entries.len(),
        readiness.checkpoint.missing_weight_tensors.len(),
        readiness.checkpoint.total_checkpoint_bytes
    );
    println!(
        "checkpoint_payloads: bound_weights={} expected_payloads={} matched_payloads={} residency_proven={} payload_bytes={} issues={} ready={} live_execution_supported={}",
        checkpoint_payloads.checkpoint_payload_bound_count,
        checkpoint_payloads.expected_payload_binding_count,
        checkpoint_payloads.matched_payload_binding_count,
        checkpoint_payloads.residency_proven_count,
        checkpoint_payloads.total_payload_bytes,
        checkpoint_payloads.binding_issue_count,
        checkpoint_payloads.checkpoint_payloads_bound,
        checkpoint_payloads.live_execution_supported
    );
    println!(
        "binding: inputs={} outputs={} checkpoint_weights={} scratch={} issues={}",
        readiness.binding.external_inputs.len(),
        readiness.binding.external_outputs.len(),
        readiness.binding.checkpoint_weights.len(),
        readiness.binding.scratch_tensors.len(),
        readiness.binding.issues.len()
    );
    println!(
        "lowering: target={} native_gpu_ops={} fused_native_gpu_ops={} gap_ops={}",
        readiness.lowering.target,
        readiness.lowering.native_gpu_ops,
        readiness.lowering.fused_native_gpu_ops,
        readiness.lowering.gap_ops
    );
    println!(
        "execution: ops={} unstaged_ops={} gap_ops={} binding_issues={}",
        readiness.execution.entries.len(),
        readiness.execution.unstaged_ops.len(),
        readiness.execution.gap_ops,
        readiness.execution.binding_issues.len()
    );
    println!(
        "slots: tensors={} op_rows={} inputs={} outputs={} checkpoint_weights={} scratch={}",
        readiness.slots.tensor_slots.len(),
        readiness.slots.op_slots.len(),
        readiness.slots.external_input_slots.len(),
        readiness.slots.external_output_slots.len(),
        readiness.slots.checkpoint_weight_slots.len(),
        readiness.slots.scratch_slots.len()
    );
    println!(
        "dispatch_intents: ops={} slot_args={} scalar_args={} unstaged_ops={} gap_ops={} binding_issues={}",
        readiness.dispatch_intents.entries.len(),
        dispatch_slot_args,
        dispatch_scalar_args,
        readiness.dispatch_intents.unstaged_ops.len(),
        readiness.dispatch_intents.gap_ops,
        readiness.dispatch_intents.binding_issues.len()
    );
    println!(
        "dispatch_bindings: ops={} issues={}",
        dispatch_binding_validation.entries.len(),
        dispatch_binding_validation.issue_count()
    );
    println!(
        "slot_bindings: bound_slots={} missing_slots={} issues={}",
        slot_binding_validation.bound_slots.len(),
        slot_binding_validation.missing_slots.len(),
        slot_binding_validation.issues.len()
    );
    println!(
        "device_pointer_template: bound_slots={} issues={} alignment={}",
        device_pointer_validation.bound_slots.len(),
        device_pointer_validation.issues.len(),
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT
    );
    println!(
        "stage_slot_bindings: stages={} issues={}",
        stage_binding_validation.stages.len(),
        stage_binding_validation.issue_count()
    );
    println!(
        "stage_slots: stages={} unstaged_ops={} gap_ops={} binding_issues={}",
        readiness.stage_slots.stages.len(),
        readiness.stage_slots.unstaged_ops.len(),
        readiness.stage_slots.gap_ops,
        readiness.stage_slots.binding_issues.len()
    );
    println!(
        "stage_bundles: stages={} unstaged_ops={} unstaged_op_slots={} gap_ops={} binding_issues={}",
        readiness.stage_bundles.stages.len(),
        readiness.stage_bundles.unstaged_ops.len(),
        readiness.stage_bundles.unstaged_op_slots.len(),
        readiness.stage_bundles.gap_ops,
        readiness.stage_bundles.binding_issues.len()
    );
    println!(
        "stage_dispatch: stages={} unstaged_ops={} unstaged_dispatches={} gap_ops={} binding_issues={}",
        readiness.stage_dispatch.stages.len(),
        readiness.stage_dispatch.unstaged_ops.len(),
        readiness.stage_dispatch.unstaged_dispatches.len(),
        readiness.stage_dispatch.gap_ops,
        readiness.stage_dispatch.binding_issues.len()
    );
    println!(
        "stage_dispatch_bindings: stages={} unstaged_bindings={} issues={}",
        stage_dispatch_binding_validation.stages.len(),
        stage_dispatch_binding_validation.unstaged_bindings.len(),
        stage_dispatch_binding_validation.issue_count()
    );
    println!(
        "stage_launch_candidates: stages={} dispatches={}",
        stage_launch_candidates.stages.len(),
        stage_launch_candidates.dispatch_count
    );
    println!(
        "launch_windows: windows={} dispatches={} max_dispatches_per_window={}",
        launch_windows.window_count,
        launch_windows.dispatch_count,
        launch_windows.max_dispatches_per_window
    );
    println!(
        "launch_entrypoints: dispatches={} host_launchers={}",
        launch_entrypoints.dispatch_count, launch_entrypoints.host_launcher_count
    );
    println!(
        "launch_kernels: dispatches={} required_kernels={} unmapped_host_launchers={}",
        launch_kernels.dispatch_count,
        launch_kernels.required_kernel_symbols.len(),
        launch_kernels.unmapped_host_launchers.len()
    );
    println!(
        "launch_kernel_metadata: kernels={} code_object_target={}",
        launch_kernel_metadata.required_kernel_count, launch_kernel_metadata.code_object_target
    );
    println!(
        "launch_code_object_loads: load_requests={} loaded={} descriptor_requests={} descriptor_bound={} base_bound={} all_descriptors_bound={} plan_ready={}",
        launch_code_object_loads.code_object_load_request_count,
        launch_code_object_loads.loaded_code_object_count,
        launch_code_object_loads.kernel_descriptor_binding_request_count,
        launch_code_object_loads.kernel_descriptor_bound_count,
        launch_code_object_loads.code_object_base_bound,
        launch_code_object_loads.all_kernel_descriptors_bound,
        launch_code_object_loads.request_plan_ready
    );
    println!(
        "launch_preflight: dispatches={} windows={} arguments={} kernels={}",
        launch_preflight.dispatch_count,
        launch_preflight.window_count,
        launch_preflight.argument_count,
        launch_preflight.required_kernel_count
    );
    println!(
        "launch_packet_fields: dispatches={} kernel_candidates={} required_kernels={} unresolved_runtime_requirements={}",
        launch_packet_fields.dispatch_count,
        launch_packet_fields.kernel_candidate_count,
        launch_packet_fields.required_kernel_count,
        launch_packet_fields.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_kernel_selection: selected_dispatches={} ambiguous_dispatches={} missing_dispatches={}",
        launch_kernel_selection.selected_dispatch_count,
        launch_kernel_selection.ambiguous_dispatch_count,
        launch_kernel_selection.missing_dispatch_count
    );
    println!(
        "launch_host_launcher_branch_requests: requests={} applied={} unresolved_candidates={} all_resolved={} plan_ready={}",
        launch_host_launcher_branch_requests.branch_resolution_request_count,
        launch_host_launcher_branch_requests.branch_resolution_applied_count,
        launch_host_launcher_branch_requests.unresolved_candidate_symbol_count,
        launch_host_launcher_branch_requests.all_branches_resolved,
        launch_host_launcher_branch_requests.request_plan_ready
    );
    let launch_host_launcher_branch_request_ops =
        launch_host_launcher_branch_requests.branch_resolution_request_op_names();
    println!(
        "launch_host_launcher_branch_request_ops: count={} names={}",
        launch_host_launcher_branch_request_ops.len(),
        launch_host_launcher_branch_request_ops.join(",")
    );
    let launch_host_launcher_branch_candidate_symbols =
        launch_host_launcher_branch_requests.branch_resolution_request_candidate_symbol_labels();
    println!(
        "launch_host_launcher_branch_candidate_symbols: count={} labels={}",
        launch_host_launcher_branch_candidate_symbols.len(),
        launch_host_launcher_branch_candidate_symbols.join(",")
    );
    let launch_host_launcher_branch_unresolved_candidate_symbols =
        launch_host_launcher_branch_requests.unresolved_candidate_symbols();
    println!(
        "launch_host_launcher_branch_unresolved_candidate_symbols: count={} symbols={}",
        launch_host_launcher_branch_unresolved_candidate_symbols.len(),
        launch_host_launcher_branch_unresolved_candidate_symbols.join(",")
    );
    println!(
        "launch_arguments: dispatches={} arguments={}",
        launch_arguments.dispatch_count, launch_arguments.argument_count
    );
    println!(
        "launch_device_arguments: dispatches={} pointer_arguments={} scalar_arguments={}",
        launch_device_arguments.dispatch_count,
        launch_device_arguments.pointer_argument_count,
        launch_device_arguments.scalar_argument_count
    );
    println!(
        "launch_staging: packet_bytes={} kernarg_upper_bound_bytes={} max_kernarg_size={}",
        launch_staging.packet_bytes,
        launch_staging.kernarg_bytes_upper_bound,
        launch_staging.max_kernarg_size
    );
    println!(
        "launch_staging_layout: packet_region_bytes={} kernarg_region_bytes={} total_staging_bytes={}",
        launch_staging_layout.packet_region_bytes,
        launch_staging_layout.kernarg_region_bytes,
        launch_staging_layout.total_staging_bytes
    );
    println!(
        "launch_completion_signals: windows={} signal_slots={} initial_value={} completed_value={}",
        launch_completion_signals.window_count,
        launch_completion_signals.logical_signal_slots,
        launch_completion_signals.signal_initial_value,
        launch_completion_signals.signal_completed_value
    );
    println!(
        "launch_completion_signal_bindings: signal_requests={} bound_handles={} all_bound={} plan_ready={}",
        launch_completion_signal_bindings.signal_handle_request_count,
        launch_completion_signal_bindings.signal_handle_bound_count,
        launch_completion_signal_bindings.all_signal_handles_bound,
        launch_completion_signal_bindings.request_plan_ready
    );
    println!(
        "launch_queue_slots: windows={} queue_packets={} doorbell_batches={}",
        launch_queue_slots.window_count,
        launch_queue_slots.queue_packet_count,
        launch_queue_slots.doorbell_batch_count
    );
    println!(
        "launch_queue_reservations: packet_requests={} reserved_packets={} doorbell_batches={} bound_doorbells={} applied_windows={} all_reserved={} plan_ready={}",
        launch_queue_reservations.queue_packet_request_count,
        launch_queue_reservations.queue_packet_reserved_count,
        launch_queue_reservations.doorbell_batch_request_count,
        launch_queue_reservations.doorbell_batch_bound_count,
        launch_queue_reservations.reservation_applied_count,
        launch_queue_reservations.all_queue_packets_reserved,
        launch_queue_reservations.request_plan_ready
    );
    println!(
        "launch_geometry: dispatches={} workgroups={} default_workgroup_size={}",
        launch_geometry.dispatch_count,
        launch_geometry.total_workgroups,
        launch_geometry.default_workgroup_size
    );
    println!(
        "launch_kernarg_layout: arguments={} payload_bytes={} span_bytes={} kernarg_region_bytes={} capacity_shortfall_bytes={}",
        launch_kernarg_layout.argument_count,
        launch_kernarg_layout.argument_payload_bytes,
        launch_kernarg_layout.argument_span_bytes,
        launch_kernarg_layout.kernarg_region_bytes,
        launch_kernarg_layout.candidate_capacity_shortfall_bytes
    );
    println!(
        "launch_kernarg_serialization: dispatches={} serialized_bytes={} argument_span_bytes={} capacity_shortfall_bytes={}",
        launch_kernarg_serialization.dispatch_count,
        launch_kernarg_serialization.serialized_kernarg_bytes,
        launch_kernarg_serialization.argument_span_bytes,
        launch_kernarg_serialization.candidate_capacity_shortfall_bytes
    );
    println!(
        "launch_kernarg_allocations: allocation_requests={} request_bytes={} bound_allocations={} bound_bytes={} copy_requests={} copy_request_bytes={} copies_applied={} copied_bytes={} all_allocated={} plan_ready={}",
        launch_kernarg_allocations.backing_allocation_request_count,
        launch_kernarg_allocations.backing_allocation_request_bytes,
        launch_kernarg_allocations.backing_allocation_bound_count,
        launch_kernarg_allocations.backing_allocation_bound_bytes,
        launch_kernarg_allocations.dispatch_copy_request_count,
        launch_kernarg_allocations.dispatch_copy_request_bytes,
        launch_kernarg_allocations.dispatch_copy_applied_count,
        launch_kernarg_allocations.dispatch_copy_applied_bytes,
        launch_kernarg_allocations.all_kernargs_allocated,
        launch_kernarg_allocations.request_plan_ready
    );
    println!(
        "launch_kernarg_abi: candidate_abis={} size_compatible_candidates={} verified_candidates={} dispatches_with_verified={} dispatches_without_verified={} ready={} unresolved_runtime_requirements={}",
        launch_kernel_argument_abi.kernel_candidate_count,
        launch_kernel_argument_abi.size_compatible_candidate_count,
        launch_kernel_argument_abi.verified_candidate_count,
        launch_kernel_argument_abi.dispatches_with_verified_candidate_count,
        launch_kernel_argument_abi.dispatches_without_verified_candidate_count,
        launch_kernel_argument_abi.abi_verification_ready,
        launch_kernel_argument_abi.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_kernarg_abi_size_receipt: checked={} dispatches_with_size_compatible={} dispatches_without_size_compatible={} dispatches_with_verified={} dispatches_without_verified={} candidate_abis={} size_compatible_candidates={} size_shortfall_candidates={} named_schemas={} named_verified={} named_ready={}",
        launch_kernel_argument_abi_size_receipt.size_compatibility_checked,
        launch_kernel_argument_abi_size_receipt.dispatches_with_size_compatible_candidate_count,
        launch_kernel_argument_abi_size_receipt.dispatches_without_size_compatible_candidate_count,
        launch_kernel_argument_abi_size_receipt.dispatches_with_verified_candidate_count,
        launch_kernel_argument_abi_size_receipt.dispatches_without_verified_candidate_count,
        launch_kernel_argument_abi_size_receipt.kernel_candidate_count,
        launch_kernel_argument_abi_size_receipt.size_compatible_candidate_count,
        launch_kernel_argument_abi_size_receipt.size_shortfall_candidate_count,
        launch_kernel_argument_abi_size_receipt.named_abi_schema_available_count,
        launch_kernel_argument_abi_size_receipt.named_abi_verified_candidate_count,
        launch_kernel_argument_abi_size_receipt.named_abi_verification_ready
    );
    println!(
        "launch_kernarg_abi_gaps: dispatches_without_verified={} candidate_abis={} size_compatible_candidates={} size_shortfall_candidates={} named_schemas={} descriptor_matches={} missing_named_schemas={} descriptor_mismatches={} verified_candidates={} primary_missing_named_schemas={} primary_descriptor_mismatches={} primary_size_shortfalls={} primary_unknown_unverified={} max_shortfall_bytes={} total_shortfall_bytes={} all_dispatches_have_verified_candidate={}",
        launch_kernel_argument_abi_gaps.dispatch_gap_count,
        launch_kernel_argument_abi_gaps.gap_kernel_candidate_count,
        launch_kernel_argument_abi_gaps.gap_size_compatible_candidate_count,
        launch_kernel_argument_abi_gaps.gap_size_shortfall_candidate_count,
        launch_kernel_argument_abi_gaps.gap_named_abi_schema_available_count,
        launch_kernel_argument_abi_gaps.gap_named_abi_descriptor_match_count,
        launch_kernel_argument_abi_gaps.gap_missing_named_abi_schema_count,
        launch_kernel_argument_abi_gaps.gap_descriptor_mismatch_candidate_count,
        launch_kernel_argument_abi_gaps.gap_verified_candidate_count,
        launch_kernel_argument_abi_gaps.gap_primary_missing_named_abi_schema_candidate_count,
        launch_kernel_argument_abi_gaps.gap_primary_descriptor_mismatch_candidate_count,
        launch_kernel_argument_abi_gaps.gap_primary_size_shortfall_candidate_count,
        launch_kernel_argument_abi_gaps.gap_primary_unknown_unverified_candidate_count,
        launch_kernel_argument_abi_gaps.max_capacity_shortfall_bytes,
        launch_kernel_argument_abi_gaps.total_capacity_shortfall_bytes,
        launch_kernel_argument_abi_gaps.all_dispatches_have_verified_candidate
    );
    println!(
        "launch_kernarg_abi_semantics: schemas={} schema_candidates={} missing_schema_candidates={} descriptor_matches={} verified_candidates={} dispatches_with_verified={} dispatches_without_verified={} field_schemas={} verified_fields={} missing_fields={} field_mismatches={} extra_arguments={} ready={} unresolved_runtime_requirements={}",
        runtime_launch_kernel_argument_abi_semantic_schema_count(),
        launch_kernel_argument_abi_semantics.semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantics.missing_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantics.semantic_descriptor_match_candidate_count,
        launch_kernel_argument_abi_semantics.semantic_verified_candidate_count,
        launch_kernel_argument_abi_semantics.dispatches_with_semantic_verified_candidate_count,
        launch_kernel_argument_abi_semantics.dispatches_without_semantic_verified_candidate_count,
        launch_kernel_argument_abi_semantics.field_schema_count,
        launch_kernel_argument_abi_semantics.verified_field_count,
        launch_kernel_argument_abi_semantics.missing_field_count,
        launch_kernel_argument_abi_semantics.field_mismatch_count,
        launch_kernel_argument_abi_semantics.extra_argument_count,
        launch_kernel_argument_abi_semantics.semantic_abi_ready,
        launch_kernel_argument_abi_semantics.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_kernarg_abi_semantic_gaps: dispatches_without_verified={} candidate_abis={} schema_candidates={} missing_schema_candidates={} descriptor_matches={} verified_candidates={} primary_missing_schemas={} primary_descriptor_mismatches={} primary_missing_model_args={} primary_field_mismatches={} primary_extra_arguments={} primary_size_shortfalls={} primary_unknown={} field_schemas={} verified_fields={} missing_fields={} field_mismatches={} extra_arguments={} all_dispatches_have_semantic_verified_candidate={}",
        launch_kernel_argument_abi_semantic_gaps.dispatch_gap_count,
        launch_kernel_argument_abi_semantic_gaps.gap_kernel_candidate_count,
        launch_kernel_argument_abi_semantic_gaps.gap_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_gaps.gap_missing_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_gaps.gap_semantic_descriptor_match_candidate_count,
        launch_kernel_argument_abi_semantic_gaps.gap_semantic_verified_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_missing_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_semantic_descriptor_mismatch_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_missing_model_argument_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_field_shape_mismatch_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_extra_model_argument_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_kernarg_size_shortfall_candidate_count,
        launch_kernel_argument_abi_semantic_gaps
            .gap_primary_unknown_unverified_semantic_candidate_count,
        launch_kernel_argument_abi_semantic_gaps.gap_field_schema_count,
        launch_kernel_argument_abi_semantic_gaps.gap_verified_field_count,
        launch_kernel_argument_abi_semantic_gaps.gap_missing_field_count,
        launch_kernel_argument_abi_semantic_gaps.gap_field_mismatch_count,
        launch_kernel_argument_abi_semantic_gaps.gap_extra_argument_count,
        launch_kernel_argument_abi_semantic_gaps
            .all_dispatches_have_semantic_verified_candidate
    );
    let launch_semantic_missing_schema_symbols =
        launch_kernel_argument_abi_semantic_gaps.missing_semantic_schema_kernel_symbols();
    println!(
        "launch_kernarg_abi_semantic_missing_schema_symbols: count={} symbols={}",
        launch_semantic_missing_schema_symbols.len(),
        launch_semantic_missing_schema_symbols.join(",")
    );
    let launch_semantic_missing_model_arguments =
        launch_kernel_argument_abi_semantic_gaps.missing_model_argument_names();
    println!(
        "launch_kernarg_abi_semantic_missing_model_arguments: count={} names={}",
        launch_semantic_missing_model_arguments.len(),
        launch_semantic_missing_model_arguments.join(",")
    );
    println!(
        "launch_kernarg_abi_semantic_projection: schema_candidates={} missing_schema_candidates={} descriptor_matches={} projection_ready_candidates={} dispatches_with_projection_ready={} dispatches_without_projection_ready={} field_schemas={} projected_fields={} missing_fields={} kind_mismatches={} unsupported_encodings={} scalar_narrowing_overflows={} field_range_overflows={} projected_kernarg_bytes={} ready={} unresolved_runtime_requirements={}",
        launch_kernel_argument_abi_semantic_projection.semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_projection.missing_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_projection.semantic_descriptor_match_candidate_count,
        launch_kernel_argument_abi_semantic_projection.projection_ready_candidate_count,
        launch_kernel_argument_abi_semantic_projection
            .dispatches_with_projection_ready_candidate_count,
        launch_kernel_argument_abi_semantic_projection
            .dispatches_without_projection_ready_candidate_count,
        launch_kernel_argument_abi_semantic_projection.field_schema_count,
        launch_kernel_argument_abi_semantic_projection.projected_field_count,
        launch_kernel_argument_abi_semantic_projection.missing_field_count,
        launch_kernel_argument_abi_semantic_projection.kind_mismatch_field_count,
        launch_kernel_argument_abi_semantic_projection.unsupported_encoding_field_count,
        launch_kernel_argument_abi_semantic_projection.scalar_narrowing_overflow_field_count,
        launch_kernel_argument_abi_semantic_projection.field_range_overflow_count,
        launch_kernel_argument_abi_semantic_projection.projected_kernarg_bytes,
        launch_kernel_argument_abi_semantic_projection.semantic_projection_ready,
        launch_kernel_argument_abi_semantic_projection
            .unresolved_runtime_requirements
            .len()
    );
    println!(
        "launch_kernarg_abi_semantic_projection_gaps: dispatches_without_projection_ready={} candidate_abis={} schema_candidates={} missing_schema_candidates={} descriptor_matches={} projection_ready_candidates={} primary_missing_schemas={} primary_descriptor_mismatches={} primary_missing_model_args={} primary_kind_mismatches={} primary_unsupported_encodings={} primary_scalar_narrowing_overflows={} primary_field_range_overflows={} primary_unknown={} field_schemas={} projected_fields={} missing_fields={} kind_mismatches={} unsupported_encodings={} scalar_narrowing_overflows={} field_range_overflows={} projected_kernarg_bytes={} all_dispatches_have_projection_ready_candidate={}",
        launch_kernel_argument_abi_semantic_projection_gaps.dispatch_gap_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_kernel_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_missing_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_semantic_descriptor_match_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_projection_ready_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_missing_semantic_schema_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_semantic_descriptor_mismatch_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_missing_model_argument_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_kind_mismatch_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_unsupported_encoding_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_scalar_narrowing_overflow_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_field_range_overflow_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_primary_unknown_unprojected_semantic_candidate_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_field_schema_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_projected_field_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_missing_field_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_kind_mismatch_field_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_unsupported_encoding_field_count,
        launch_kernel_argument_abi_semantic_projection_gaps
            .gap_scalar_narrowing_overflow_field_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_field_range_overflow_count,
        launch_kernel_argument_abi_semantic_projection_gaps.gap_projected_kernarg_bytes,
        launch_kernel_argument_abi_semantic_projection_gaps
            .all_dispatches_have_projection_ready_candidate
    );
    let launch_semantic_projection_missing_schema_symbols =
        launch_kernel_argument_abi_semantic_projection_gaps
            .missing_semantic_schema_kernel_symbols();
    println!(
        "launch_kernarg_abi_semantic_projection_missing_schema_symbols: count={} symbols={}",
        launch_semantic_projection_missing_schema_symbols.len(),
        launch_semantic_projection_missing_schema_symbols.join(",")
    );
    let launch_semantic_projection_missing_model_arguments =
        launch_kernel_argument_abi_semantic_projection_gaps.missing_model_argument_names();
    println!(
        "launch_kernarg_abi_semantic_projection_missing_model_arguments: count={} names={}",
        launch_semantic_projection_missing_model_arguments.len(),
        launch_semantic_projection_missing_model_arguments.join(",")
    );
    println!(
        "launch_kernarg_abi_semantic_projection_recommendations: recommended_dispatches={} missing_recommendations={} recommended_projection_ready={} recommended_projection_blocked={} recommended_projection_missing={} recommended_without_projection_ready={} dispatches_with_projection_ready={} dispatches_without_projection_ready={} all_recommended_projection_ready={} all_dispatches_have_projection_ready_recommendation={} ready_kernarg_bytes={}",
        launch_semantic_projection_recommendations.recommended_dispatch_count,
        launch_semantic_projection_recommendations.missing_recommendation_dispatch_count,
        launch_semantic_projection_recommendations.recommended_projection_ready_dispatch_count,
        launch_semantic_projection_recommendations.recommended_projection_blocked_dispatch_count,
        launch_semantic_projection_recommendations
            .recommended_missing_projection_candidate_count,
        launch_semantic_projection_recommendations
            .recommended_without_projection_ready_dispatch_count,
        launch_semantic_projection_recommendations
            .dispatches_with_projection_ready_candidate_count,
        launch_semantic_projection_recommendations
            .dispatches_without_projection_ready_candidate_count,
        launch_semantic_projection_recommendations.all_recommended_dispatches_projection_ready,
        launch_semantic_projection_recommendations
            .all_dispatches_have_projection_ready_recommendation,
        launch_semantic_projection_recommendations.recommended_projection_ready_kernarg_bytes
    );
    println!(
        "launch_kernarg_abi_semantic_projection_candidate_recommendations: recommended_dispatches={} missing_recommendations={} projection_ready_candidates={} source_ambiguous_dispatches={} recommended_projected_kernarg_bytes={} all_recommended={} policy={}",
        launch_semantic_projection_candidate_recommendations.recommended_dispatch_count,
        launch_semantic_projection_candidate_recommendations
            .missing_recommendation_dispatch_count,
        launch_semantic_projection_candidate_recommendations.projection_ready_candidate_count,
        launch_semantic_projection_candidate_recommendations.source_ambiguous_dispatch_count,
        launch_semantic_projection_candidate_recommendations.recommended_projected_kernarg_bytes,
        launch_semantic_projection_candidate_recommendations.all_dispatches_recommended,
        launch_semantic_projection_candidate_recommendations.policy
    );
    println!(
        "launch_kernarg_abi_semantic_projection_candidate_selection_requests: requests={} missing={} projection_ready_candidates={} source_ambiguous_dispatches={} requested_projected_kernarg_bytes={} applied={} all_ready={} plan_ready={} policy={}",
        launch_semantic_projection_candidate_selection_requests.selection_request_count,
        launch_semantic_projection_candidate_selection_requests.missing_selection_request_count,
        launch_semantic_projection_candidate_selection_requests.projection_ready_candidate_count,
        launch_semantic_projection_candidate_selection_requests.source_ambiguous_dispatch_count,
        launch_semantic_projection_candidate_selection_requests.requested_projected_kernarg_bytes,
        launch_semantic_projection_candidate_selection_requests.selection_applied_count,
        launch_semantic_projection_candidate_selection_requests.all_selection_requests_ready,
        launch_semantic_projection_candidate_selection_requests.request_plan_ready,
        launch_semantic_projection_candidate_selection_requests.policy
    );
    let launch_semantic_projection_selection_ready_ops =
        launch_semantic_projection_candidate_selection_requests.selection_request_op_names();
    println!(
        "launch_kernarg_abi_semantic_projection_candidate_selection_ready_ops: count={} names={}",
        launch_semantic_projection_selection_ready_ops.len(),
        launch_semantic_projection_selection_ready_ops.join(",")
    );
    let launch_semantic_projection_selection_requested_symbols =
        launch_semantic_projection_candidate_selection_requests
            .selection_request_op_kernel_symbol_labels();
    println!(
        "launch_kernarg_abi_semantic_projection_candidate_selection_requested_symbols: count={} labels={}",
        launch_semantic_projection_selection_requested_symbols.len(),
        launch_semantic_projection_selection_requested_symbols.join(",")
    );
    let launch_semantic_projection_selection_missing_ops =
        launch_semantic_projection_candidate_selection_requests
            .missing_selection_request_op_names();
    println!(
        "launch_kernarg_abi_semantic_projection_candidate_selection_missing_ops: count={} names={}",
        launch_semantic_projection_selection_missing_ops.len(),
        launch_semantic_projection_selection_missing_ops.join(",")
    );
    println!(
        "launch_kernarg_abi_capacity_requests: requests={} dispatch_references={} candidate_requests={} primary_size_shortfalls={} max_shortfall_bytes={} total_shortfall_bytes={} all_ready={} plan_ready={}",
        launch_kernel_argument_abi_capacity_requests.capacity_request_count,
        launch_kernel_argument_abi_capacity_requests.dispatch_reference_count,
        launch_kernel_argument_abi_capacity_requests.candidate_capacity_request_count,
        launch_kernel_argument_abi_capacity_requests
            .source_primary_size_shortfall_candidate_count,
        launch_kernel_argument_abi_capacity_requests.max_capacity_shortfall_bytes,
        launch_kernel_argument_abi_capacity_requests.total_capacity_shortfall_bytes,
        launch_kernel_argument_abi_capacity_requests.all_capacity_requests_ready,
        launch_kernel_argument_abi_capacity_requests.request_plan_ready
    );
    println!(
        "launch_kernarg_abi_schema_requests: schema_requests={} schema_bound={} verification_requests={} verified={} all_schemas_bound={} all_verified={} plan_ready={}",
        launch_kernel_argument_abi_schema_requests.schema_request_count,
        launch_kernel_argument_abi_schema_requests.schema_bound_count,
        launch_kernel_argument_abi_schema_requests.candidate_verification_request_count,
        launch_kernel_argument_abi_schema_requests.candidate_verified_count,
        launch_kernel_argument_abi_schema_requests.all_schemas_bound,
        launch_kernel_argument_abi_schema_requests.all_candidates_verified,
        launch_kernel_argument_abi_schema_requests.request_plan_ready
    );
    println!(
        "launch_kernel_recommendations: recommended_dispatches={} missing_recommendations={} source_ambiguous_dispatches={} verified_candidates={} selection_applied={} all_recommended={} policy={}",
        launch_kernel_recommendations.recommended_dispatch_count,
        launch_kernel_recommendations.missing_recommendation_dispatch_count,
        launch_kernel_recommendations.source_ambiguous_dispatch_count,
        launch_kernel_recommendations.verified_candidate_count,
        launch_kernel_recommendations.selection_applied_count,
        launch_kernel_recommendations.all_dispatches_recommended,
        launch_kernel_recommendations.policy
    );
    println!(
        "launch_kernel_selection_requests: requests={} missing={} verified_candidates={} applied={} all_ready={} plan_ready={} policy={}",
        launch_kernel_selection_requests.selection_request_count,
        launch_kernel_selection_requests.missing_selection_request_count,
        launch_kernel_selection_requests.verified_candidate_count,
        launch_kernel_selection_requests.selection_applied_count,
        launch_kernel_selection_requests.all_selection_requests_ready,
        launch_kernel_selection_requests.request_plan_ready,
        launch_kernel_selection_requests.policy
    );
    let launch_kernel_selection_ready_ops =
        launch_kernel_selection_requests.selection_request_op_names();
    println!(
        "launch_kernel_selection_ready_ops: count={} names={}",
        launch_kernel_selection_ready_ops.len(),
        launch_kernel_selection_ready_ops.join(",")
    );
    let launch_kernel_selection_requested_symbols =
        launch_kernel_selection_requests.selection_request_op_kernel_symbol_labels();
    println!(
        "launch_kernel_selection_requested_symbols: count={} labels={}",
        launch_kernel_selection_requested_symbols.len(),
        launch_kernel_selection_requested_symbols.join(",")
    );
    let launch_kernel_selection_missing_ops =
        launch_kernel_selection_requests.missing_selection_request_op_names();
    println!(
        "launch_kernel_selection_missing_ops: count={} names={}",
        launch_kernel_selection_missing_ops.len(),
        launch_kernel_selection_missing_ops.join(",")
    );
    println!(
        "launch_aql_templates: packets={} candidate_templates={} packet_bytes={} unresolved_runtime_requirements={}",
        launch_aql_templates.dispatch_count,
        launch_aql_templates.kernel_candidate_count,
        launch_aql_templates.packet_bytes,
        launch_aql_templates.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_aql_relocations: packets={} relocation_sites={} fields_per_packet={} unresolved_runtime_requirements={}",
        launch_aql_relocations.dispatch_count,
        launch_aql_relocations.total_relocation_sites,
        launch_aql_relocations.field_ranges_per_packet,
        launch_aql_relocations.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_aql_byte_templates: candidate_templates={} byte_template_bytes={} packet_bytes_per_template={} unresolved_runtime_requirements={}",
        launch_aql_byte_templates.candidate_byte_template_count,
        launch_aql_byte_templates.candidate_byte_template_bytes,
        AQL_PACKET_BYTES,
        launch_aql_byte_templates.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_aql_materialization: selected_dispatches={} ambiguous_dispatches={} live_relocation_sites={} dispatchable_packets={} ready={} unresolved_runtime_requirements={}",
        launch_aql_materialization.selected_dispatch_count,
        launch_aql_materialization.ambiguous_dispatch_count,
        launch_aql_materialization.live_relocation_patch_site_count,
        launch_aql_materialization.dispatchable_packet_count,
        launch_aql_materialization.packet_materialization_ready,
        launch_aql_materialization.unresolved_runtime_requirements.len()
    );
    println!(
        "launch_aql_live_bindings: requests={} bound={} unbound={} code_object={} kernarg={} completion_signal={} all_bound={} plan_ready={}",
        launch_aql_live_bindings.binding_request_count,
        launch_aql_live_bindings.bound_relocation_count,
        launch_aql_live_bindings.unbound_relocation_count,
        launch_aql_live_bindings.code_object_base_request_count,
        launch_aql_live_bindings.kernarg_allocation_request_count,
        launch_aql_live_bindings.completion_signal_request_count,
        launch_aql_live_bindings.all_relocations_bound,
        launch_aql_live_bindings.binding_request_plan_ready
    );
    println!(
        "launch_code_object_base_bindings: base_requests={} base_bound={} descriptor_requests={} descriptor_bound={} relocation_requests={} relocation_bound={} all_bound={} plan_ready={}",
        launch_code_object_base_bindings.loaded_code_object_base_request_count,
        launch_code_object_base_bindings.loaded_code_object_base_bound_count,
        launch_code_object_base_bindings.kernel_descriptor_binding_request_count,
        launch_code_object_base_bindings.kernel_descriptor_bound_count,
        launch_code_object_base_bindings.aql_kernel_object_relocation_request_count,
        launch_code_object_base_bindings.aql_kernel_object_relocation_bound_count,
        launch_code_object_base_bindings.all_code_object_base_bindings_bound,
        launch_code_object_base_bindings.request_plan_ready
    );
    println!(
        "launch_execution_requests: request_plans={} components={} typed_steps={} typed_step_descriptors={} typed_live_aql_proof_steps={} component_requests={} component_applied={} component_pending={} live_aql_proof_components={} live_aql_proof_surfaces={} live_aql_proof_pending={} live_aql_proof_validations={} live_aql_proof_validation_pending={} live_aql_submitting_surfaces={} live_queue_mutating_components={} blockers={} all_applied={} plan_ready={}",
        launch_execution_requests.runtime_request_plan_count,
        launch_execution_requests.components.len(),
        RuntimeLaunchExecutionRequestStep::ALL.len(),
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len(),
        typed_live_aql_proof_steps,
        launch_execution_requests.component_request_count,
        launch_execution_requests.component_applied_count,
        launch_execution_requests.component_pending_count,
        launch_execution_requests.live_aql_proof_component_count,
        launch_execution_requests.live_aql_proof_surface_count,
        launch_execution_requests.live_aql_proof_surface_pending_count,
        launch_execution_requests.live_aql_proof_validation_request_count,
        launch_execution_requests.live_aql_proof_validation_pending_count,
        launch_execution_requests.live_aql_submitting_surface_count,
        launch_execution_requests.live_queue_mutating_component_count,
        launch_execution_requests.execution_blocker_count,
        launch_execution_requests.all_components_applied,
        launch_execution_requests.request_plan_ready
    );
    let launch_execution_request_plans = launch_execution_requests.component_request_plan_names();
    println!(
        "launch_execution_request_plans: count={} names={}",
        launch_execution_request_plans.len(),
        launch_execution_request_plans.join(",")
    );
    let launch_execution_request_pending_plans =
        launch_execution_requests.pending_component_request_plan_names();
    println!(
        "launch_execution_request_pending_plans: count={} names={}",
        launch_execution_request_pending_plans.len(),
        launch_execution_request_pending_plans.join(",")
    );
    let launch_execution_live_aql_proof_surface_plans =
        launch_execution_requests.live_aql_proof_surface_request_plan_names();
    println!(
        "launch_execution_live_aql_proof_surface_plans: count={} names={}",
        launch_execution_live_aql_proof_surface_plans.len(),
        launch_execution_live_aql_proof_surface_plans.join(",")
    );
    let launch_execution_pending_live_aql_proof_surface_plans =
        launch_execution_requests.pending_live_aql_proof_surface_request_plan_names();
    println!(
        "launch_execution_pending_live_aql_proof_surface_plans: count={} names={}",
        launch_execution_pending_live_aql_proof_surface_plans.len(),
        launch_execution_pending_live_aql_proof_surface_plans.join(",")
    );
    let launch_execution_pending_live_aql_proof_validation_plans =
        launch_execution_requests.pending_live_aql_proof_validation_request_plan_names();
    println!(
        "launch_execution_pending_live_aql_proof_validation_plans: count={} names={}",
        launch_execution_pending_live_aql_proof_validation_plans.len(),
        launch_execution_pending_live_aql_proof_validation_plans.join(",")
    );
    let launch_execution_live_aql_proof_kinds =
        launch_execution_requests.live_aql_proof_kind_labels();
    println!(
        "launch_execution_live_aql_proof_kinds: count={} labels={}",
        launch_execution_live_aql_proof_kinds.len(),
        launch_execution_live_aql_proof_kinds.join(",")
    );
    let launch_execution_live_aql_submitting_surface_plans =
        launch_execution_requests.live_aql_submitting_surface_request_plan_names();
    println!(
        "launch_execution_live_aql_submitting_surface_plans: count={} names={}",
        launch_execution_live_aql_submitting_surface_plans.len(),
        launch_execution_live_aql_submitting_surface_plans.join(",")
    );
    let launch_execution_live_queue_mutating_component_plans =
        launch_execution_requests.live_queue_mutating_component_request_plan_names();
    println!(
        "launch_execution_live_queue_mutating_component_plans: count={} names={}",
        launch_execution_live_queue_mutating_component_plans.len(),
        launch_execution_live_queue_mutating_component_plans.join(",")
    );
    let launch_execution_live_aql_proof_inputs =
        launch_execution_requests.live_aql_proof_input_labels();
    println!(
        "launch_execution_live_aql_proof_inputs: count={} labels={}",
        launch_execution_live_aql_proof_inputs.len(),
        launch_execution_live_aql_proof_inputs.join(",")
    );
    let launch_execution_live_aql_validation_methods =
        launch_execution_requests.live_aql_validation_method_labels();
    println!(
        "launch_execution_live_aql_validation_methods: count={} labels={}",
        launch_execution_live_aql_validation_methods.len(),
        launch_execution_live_aql_validation_methods.join(",")
    );
    println!(
        "launch_execution_request_receipt: fingerprint={} lines={}",
        launch_execution_requests.receipt_fingerprint(),
        launch_execution_requests.receipt_lines().len()
    );
    println!(
        "launch_submission_gate: ready={} blockers={} execution_blockers={} component_pending={} proof_validation_pending={} submitting_surfaces={} queue_mutating_components={} request_plan_ready={} all_components_applied={}",
        launch_submission_gate.submission_ready,
        launch_submission_gate.submission_blocker_count,
        launch_submission_gate.execution_blocker_count,
        launch_submission_gate.component_pending_count,
        launch_submission_gate.live_aql_proof_validation_pending_count,
        launch_submission_gate.live_aql_submitting_surface_count,
        launch_submission_gate.live_queue_mutating_component_count,
        launch_submission_gate.request_plan_ready,
        launch_submission_gate.all_components_applied
    );
    let launch_submission_gate_blockers = launch_submission_gate.blocker_requirement_names();
    println!(
        "launch_submission_gate_blockers: count={} requirements={}",
        launch_submission_gate_blockers.len(),
        launch_submission_gate_blockers.join(",")
    );
    println!(
        "launch_submission_gate_receipt: fingerprint={} lines={}",
        launch_submission_gate.receipt_fingerprint(),
        launch_submission_gate.receipt_lines().len()
    );
    println!(
        "launch_submission_blockers: blockers={} execution_readiness={} total_pending={} runtime_pending={} proof_validation_pending={} submitting_surfaces={} queue_mutating_components={} ready={}",
        launch_submission_blockers.blocker_count,
        launch_submission_blockers.execution_readiness_blocker_count,
        launch_submission_blockers.total_pending_count,
        launch_submission_blockers.runtime_request_component_pending_count,
        launch_submission_blockers.live_aql_proof_validation_pending_count,
        launch_submission_blockers.live_aql_submission_side_effect_count,
        launch_submission_blockers.live_queue_mutation_count,
        launch_submission_blockers.submission_ready
    );
    let launch_submission_blocker_report_blockers =
        launch_submission_blockers.blocker_requirement_names();
    println!(
        "launch_submission_blocker_report_blockers: count={} requirements={}",
        launch_submission_blocker_report_blockers.len(),
        launch_submission_blocker_report_blockers.join(",")
    );
    let launch_submission_blocker_report_execution_readiness_blockers =
        launch_submission_blockers.execution_readiness_blocker_requirement_names();
    println!(
        "launch_submission_blocker_report_execution_readiness_blockers: count={} requirements={}",
        launch_submission_blocker_report_execution_readiness_blockers.len(),
        launch_submission_blocker_report_execution_readiness_blockers.join(",")
    );
    let launch_submission_blocker_report_runtime_component_blockers =
        launch_submission_blockers.runtime_request_component_blocker_requirement_names();
    println!(
        "launch_submission_blocker_report_runtime_component_blockers: count={} requirements={}",
        launch_submission_blocker_report_runtime_component_blockers.len(),
        launch_submission_blocker_report_runtime_component_blockers.join(",")
    );
    let launch_submission_blocker_report_live_aql_proof_validation_blockers =
        launch_submission_blockers.live_aql_proof_validation_blocker_requirement_names();
    println!(
        "launch_submission_blocker_report_live_aql_proof_validation_blockers: count={} requirements={}",
        launch_submission_blocker_report_live_aql_proof_validation_blockers.len(),
        launch_submission_blocker_report_live_aql_proof_validation_blockers.join(",")
    );
    let launch_submission_blocker_report_live_aql_submission_side_effect_blockers =
        launch_submission_blockers.live_aql_submission_side_effect_blocker_requirement_names();
    println!(
        "launch_submission_blocker_report_live_aql_submission_side_effect_blockers: count={} requirements={}",
        launch_submission_blocker_report_live_aql_submission_side_effect_blockers.len(),
        launch_submission_blocker_report_live_aql_submission_side_effect_blockers.join(",")
    );
    let launch_submission_blocker_report_live_queue_mutation_blockers =
        launch_submission_blockers.live_queue_mutation_blocker_requirement_names();
    println!(
        "launch_submission_blocker_report_live_queue_mutation_blockers: count={} requirements={}",
        launch_submission_blocker_report_live_queue_mutation_blockers.len(),
        launch_submission_blocker_report_live_queue_mutation_blockers.join(",")
    );
    println!(
        "launch_submission_blocker_report_receipt: fingerprint={} lines={}",
        launch_submission_blockers.receipt_fingerprint(),
        launch_submission_blockers.receipt_lines().len()
    );
    println!(
        "launch_submission_prerequisites: prerequisites={} satisfied={} unsatisfied={} pending_requests={} live_aql_proof_prerequisites={} live_aql_proof_inputs={} live_aql_validation_methods={} live_aql_submitting_prerequisites={} live_aql_proof_validation_pending={} live_queue_mutating_prerequisites={} ready={}",
        launch_submission_prerequisites.prerequisite_count,
        launch_submission_prerequisites.satisfied_prerequisite_count,
        launch_submission_prerequisites.unsatisfied_prerequisite_count,
        launch_submission_prerequisites.pending_component_request_count,
        launch_submission_prerequisites.live_aql_proof_prerequisite_count,
        launch_submission_prerequisites.live_aql_proof_input_count,
        launch_submission_prerequisites.live_aql_validation_method_count,
        launch_submission_prerequisites.live_aql_submitting_prerequisite_count,
        launch_submission_prerequisites.live_aql_proof_validation_pending_count,
        launch_submission_prerequisites.live_queue_mutating_prerequisite_count,
        launch_submission_prerequisites.submission_ready
    );
    let launch_submission_prerequisite_plans =
        launch_submission_prerequisites.prerequisite_request_plan_names();
    println!(
        "launch_submission_prerequisite_plans: count={} names={}",
        launch_submission_prerequisite_plans.len(),
        launch_submission_prerequisite_plans.join(",")
    );
    let launch_submission_prerequisite_unsatisfied_plans =
        launch_submission_prerequisites.unsatisfied_prerequisite_request_plan_names();
    println!(
        "launch_submission_prerequisite_unsatisfied_plans: count={} names={}",
        launch_submission_prerequisite_unsatisfied_plans.len(),
        launch_submission_prerequisite_unsatisfied_plans.join(",")
    );
    let launch_submission_prerequisite_next_action_plans =
        launch_submission_prerequisites.next_action_request_plan_names();
    println!(
        "launch_submission_prerequisite_next_action_plans: count={} names={}",
        launch_submission_prerequisite_next_action_plans.len(),
        launch_submission_prerequisite_next_action_plans.join(",")
    );
    let launch_submission_prerequisite_next_action_labels =
        launch_submission_prerequisites.next_action_labels();
    println!(
        "launch_submission_prerequisite_next_action_labels: count={} labels={}",
        launch_submission_prerequisite_next_action_labels.len(),
        launch_submission_prerequisite_next_action_labels.join(",")
    );
    let launch_submission_prerequisite_runtime_component_next_action_plans =
        launch_submission_prerequisites.runtime_request_component_next_action_request_plan_names();
    println!(
        "launch_submission_prerequisite_runtime_component_next_action_plans: count={} names={}",
        launch_submission_prerequisite_runtime_component_next_action_plans.len(),
        launch_submission_prerequisite_runtime_component_next_action_plans.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_validation_next_action_plans =
        launch_submission_prerequisites.live_aql_proof_validation_next_action_request_plan_names();
    println!(
        "launch_submission_prerequisite_live_aql_proof_validation_next_action_plans: count={} names={}",
        launch_submission_prerequisite_live_aql_proof_validation_next_action_plans.len(),
        launch_submission_prerequisite_live_aql_proof_validation_next_action_plans.join(",")
    );
    let launch_submission_prerequisite_next_action_inputs =
        launch_submission_prerequisites.next_action_input_labels();
    println!(
        "launch_submission_prerequisite_next_action_inputs: count={} labels={}",
        launch_submission_prerequisite_next_action_inputs.len(),
        launch_submission_prerequisite_next_action_inputs.join(",")
    );
    let launch_submission_prerequisite_next_action_live_aql_proof_kinds =
        launch_submission_prerequisites.next_action_live_aql_proof_kind_labels();
    println!(
        "launch_submission_prerequisite_next_action_live_aql_proof_kinds: count={} labels={}",
        launch_submission_prerequisite_next_action_live_aql_proof_kinds.len(),
        launch_submission_prerequisite_next_action_live_aql_proof_kinds.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_plans =
        launch_submission_prerequisites.live_aql_proof_prerequisite_request_plan_names();
    println!(
        "launch_submission_prerequisite_live_aql_proof_plans: count={} names={}",
        launch_submission_prerequisite_live_aql_proof_plans.len(),
        launch_submission_prerequisite_live_aql_proof_plans.join(",")
    );
    let launch_submission_prerequisite_live_aql_submitting_plans =
        launch_submission_prerequisites.live_aql_submitting_prerequisite_request_plan_names();
    println!(
        "launch_submission_prerequisite_live_aql_submitting_plans: count={} names={}",
        launch_submission_prerequisite_live_aql_submitting_plans.len(),
        launch_submission_prerequisite_live_aql_submitting_plans.join(",")
    );
    let launch_submission_prerequisite_pending_live_aql_proof_validation_plans =
        launch_submission_prerequisites
            .pending_live_aql_proof_validation_prerequisite_request_plan_names();
    println!(
        "launch_submission_prerequisite_pending_live_aql_proof_validation_plans: count={} names={}",
        launch_submission_prerequisite_pending_live_aql_proof_validation_plans.len(),
        launch_submission_prerequisite_pending_live_aql_proof_validation_plans.join(",")
    );
    let launch_submission_prerequisite_live_queue_mutating_plans =
        launch_submission_prerequisites.live_queue_mutating_prerequisite_request_plan_names();
    println!(
        "launch_submission_prerequisite_live_queue_mutating_plans: count={} names={}",
        launch_submission_prerequisite_live_queue_mutating_plans.len(),
        launch_submission_prerequisite_live_queue_mutating_plans.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_kinds =
        launch_submission_prerequisites.live_aql_proof_kind_labels();
    println!(
        "launch_submission_prerequisite_live_aql_proof_kinds: count={} labels={}",
        launch_submission_prerequisite_live_aql_proof_kinds.len(),
        launch_submission_prerequisite_live_aql_proof_kinds.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_inputs =
        launch_submission_prerequisites.live_aql_proof_input_labels();
    println!(
        "launch_submission_prerequisite_live_aql_proof_inputs: count={} labels={}",
        launch_submission_prerequisite_live_aql_proof_inputs.len(),
        launch_submission_prerequisite_live_aql_proof_inputs.join(",")
    );
    let launch_submission_prerequisite_live_aql_validation_methods =
        launch_submission_prerequisites.live_aql_validation_method_labels();
    println!(
        "launch_submission_prerequisite_live_aql_validation_methods: count={} labels={}",
        launch_submission_prerequisite_live_aql_validation_methods.len(),
        launch_submission_prerequisite_live_aql_validation_methods.join(",")
    );
    println!(
        "launch_submission_prerequisite_plan_receipt: fingerprint={} lines={}",
        launch_submission_prerequisites.receipt_fingerprint(),
        launch_submission_prerequisites.receipt_lines().len()
    );
    println!(
        "launch_executable: ready={} blockers={} unresolved_runtime_requirements={} code_object_load_request_ready={} code_object_base_binding_request_ready={} completion_signal_binding_request_ready={} kernarg_layout_ready={} kernarg_serialization_ready={} kernarg_allocation_request_ready={} kernel_argument_abi_ready={} kernel_argument_abi_schema_request_ready={} kernel_argument_abi_capacity_request_ready={} kernel_candidate_recommendation_ready={} kernel_candidate_selection_request_ready={} host_launcher_branch_request_ready={} queue_reservation_request_ready={} aql_packet_template_ready={} aql_packet_relocation_ready={} aql_packet_byte_template_ready={} aql_packet_materialization_ready={} aql_live_binding_plan_ready={} code_object_load_requests={} code_objects_loaded={} code_object_base_requests={} code_object_base_bound={} descriptor_requests={} descriptors_bound={} aql_kernel_object_relocation_requests={} aql_kernel_object_relocation_bound={} completion_signal_requests={} completion_signal_bound={} kernel_recommended_dispatches={} kernel_missing_recommendations={} kernel_recommendation_applied={} kernel_selection_requests={} kernel_selection_missing={} kernel_selection_applied={} host_launcher_branch_requests={} host_launcher_branch_applied={} host_launcher_branch_unresolved_candidates={} queue_reservation_requests={} queue_reserved_packets={} queue_reservation_applied={} kernarg_allocation_requests={} kernarg_allocation_bound={} kernarg_allocation_request_bytes={} kernarg_allocation_bound_bytes={} kernarg_copy_requests={} kernarg_copies_applied={} kernarg_copy_request_bytes={} kernarg_copied_bytes={} kernel_argument_abi_candidates={} kernel_argument_abi_size_compatible={} kernel_argument_abi_verified={} kernel_argument_abi_dispatches_with_verified={} kernel_argument_abi_dispatches_without_verified={} kernel_argument_abi_schema_requests={} kernel_argument_abi_schema_bound={} kernel_argument_abi_verification_requests={} kernel_argument_abi_verification_applied={} kernel_argument_abi_capacity_requests={} kernel_argument_abi_capacity_candidate_requests={} kernel_argument_abi_capacity_max_shortfall_bytes={} kernel_argument_abi_capacity_total_shortfall_bytes={} aql_packet_templates={} aql_candidate_templates={} aql_relocation_sites={} aql_byte_templates={} aql_byte_template_bytes={} aql_materialization_selected={} aql_materialization_ambiguous={} aql_materialization_relocation_sites={} aql_dispatchable_packets={} aql_live_binding_requests={} aql_live_binding_bound={} aql_live_binding_unbound={} kernarg_argument_bytes={} kernarg_argument_span_bytes={} kernarg_serialized_bytes={} kernarg_capacity_shortfall_bytes={}",
        launch_execution.executable,
        launch_execution.blockers.len(),
        launch_execution.unresolved_runtime_requirements.len(),
        launch_execution.code_object_load_request_plan_ready,
        launch_execution.code_object_base_binding_request_plan_ready,
        launch_execution.completion_signal_binding_request_plan_ready,
        launch_execution.kernarg_layout_ready,
        launch_execution.kernarg_serialization_ready,
        launch_execution.kernarg_allocation_request_plan_ready,
        launch_execution.kernel_argument_abi_preflight_ready,
        launch_execution.kernel_argument_abi_schema_request_plan_ready,
        launch_execution.kernel_argument_abi_capacity_request_plan_ready,
        launch_execution.kernel_candidate_recommendation_plan_ready,
        launch_execution.kernel_candidate_selection_request_plan_ready,
        launch_execution.host_launcher_branch_resolution_request_plan_ready,
        launch_execution.queue_reservation_request_plan_ready,
        launch_execution.aql_packet_template_ready,
        launch_execution.aql_packet_relocation_plan_ready,
        launch_execution.aql_packet_byte_template_ready,
        launch_execution.aql_packet_materialization_plan_ready,
        launch_execution.aql_live_relocation_binding_plan_ready,
        launch_execution.code_object_load_request_count,
        launch_execution.code_object_loaded_count,
        launch_execution.loaded_code_object_base_request_count,
        launch_execution.loaded_code_object_base_bound_count,
        launch_execution.kernel_descriptor_binding_request_count,
        launch_execution.kernel_descriptor_bound_count,
        launch_execution.aql_kernel_object_relocation_request_count,
        launch_execution.aql_kernel_object_relocation_bound_count,
        launch_execution.completion_signal_handle_request_count,
        launch_execution.completion_signal_handle_bound_count,
        launch_execution.kernel_candidate_recommended_dispatch_count,
        launch_execution.kernel_candidate_missing_recommendation_dispatch_count,
        launch_execution.kernel_candidate_recommendation_selection_applied_count,
        launch_execution.kernel_candidate_selection_request_count,
        launch_execution.kernel_candidate_selection_missing_request_count,
        launch_execution.kernel_candidate_selection_applied_count,
        launch_execution.host_launcher_branch_resolution_request_count,
        launch_execution.host_launcher_branch_resolution_applied_count,
        launch_execution.host_launcher_branch_resolution_unresolved_candidate_count,
        launch_execution.queue_reservation_packet_request_count,
        launch_execution.queue_reservation_packet_reserved_count,
        launch_execution.queue_reservation_applied_count,
        launch_execution.kernarg_allocation_request_count,
        launch_execution.kernarg_allocation_bound_count,
        launch_execution.kernarg_allocation_request_bytes,
        launch_execution.kernarg_allocation_bound_bytes,
        launch_execution.kernarg_copy_request_count,
        launch_execution.kernarg_copy_applied_count,
        launch_execution.kernarg_copy_request_bytes,
        launch_execution.kernarg_copy_applied_bytes,
        launch_execution.kernel_argument_abi_candidate_count,
        launch_execution.kernel_argument_abi_size_compatible_candidate_count,
        launch_execution.kernel_argument_abi_verified_candidate_count,
        launch_execution.kernel_argument_abi_dispatches_with_verified_candidate_count,
        launch_execution.kernel_argument_abi_dispatches_without_verified_candidate_count,
        launch_execution.kernel_argument_abi_schema_request_count,
        launch_execution.kernel_argument_abi_schema_bound_count,
        launch_execution.kernel_argument_abi_verification_request_count,
        launch_execution.kernel_argument_abi_verification_applied_count,
        launch_execution.kernel_argument_abi_capacity_request_count,
        launch_execution.kernel_argument_abi_capacity_candidate_request_count,
        launch_execution.kernel_argument_abi_capacity_max_shortfall_bytes,
        launch_execution.kernel_argument_abi_capacity_total_shortfall_bytes,
        launch_execution.aql_packet_template_count,
        launch_execution.aql_packet_template_candidate_count,
        launch_execution.aql_packet_relocation_site_count,
        launch_execution.aql_packet_byte_template_count,
        launch_execution.aql_packet_byte_template_bytes,
        launch_execution.aql_packet_materialization_selected_dispatch_count,
        launch_execution.aql_packet_materialization_ambiguous_dispatch_count,
        launch_execution.aql_packet_materialization_relocation_site_count,
        launch_execution.aql_packet_materialization_dispatchable_packet_count,
        launch_execution.aql_live_relocation_binding_request_count,
        launch_execution.aql_live_relocation_binding_bound_count,
        launch_execution.aql_live_relocation_binding_unbound_count,
        launch_execution.kernarg_argument_bytes,
        launch_execution.kernarg_argument_span_bytes,
        launch_execution.kernarg_serialized_bytes,
        launch_execution.kernarg_layout_capacity_shortfall_bytes
    );
    let launch_executable_blockers = launch_execution.blocker_requirement_names();
    println!(
        "launch_executable_blockers: count={} requirements={}",
        launch_executable_blockers.len(),
        launch_executable_blockers.join(",")
    );
    let launch_executable_requirements = launch_execution.unresolved_runtime_requirement_names();
    println!(
        "launch_executable_requirements: count={} requirements={}",
        launch_executable_requirements.len(),
        launch_executable_requirements.join(",")
    );
    println!(
        "launch_executable_semantic_projection: ready={} candidates={} schema_candidates={} missing_schema_candidates={} descriptor_matches={} projection_ready_candidates={} dispatches_with_ready={} dispatches_without_ready={} field_schemas={} projected_fields={} missing_fields={} kind_mismatches={} unsupported_encodings={} scalar_narrowing_overflows={} field_range_overflows={} projected_kernarg_bytes={}",
        launch_execution.kernel_argument_abi_semantic_projection_ready,
        launch_execution.kernel_argument_abi_semantic_projection_candidate_count,
        launch_execution.kernel_argument_abi_semantic_projection_schema_candidate_count,
        launch_execution.kernel_argument_abi_semantic_projection_missing_schema_candidate_count,
        launch_execution.kernel_argument_abi_semantic_projection_descriptor_match_candidate_count,
        launch_execution.kernel_argument_abi_semantic_projection_ready_candidate_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_dispatches_with_ready_candidate_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_dispatches_without_ready_candidate_count,
        launch_execution.kernel_argument_abi_semantic_projection_field_schema_count,
        launch_execution.kernel_argument_abi_semantic_projection_projected_field_count,
        launch_execution.kernel_argument_abi_semantic_projection_missing_field_count,
        launch_execution.kernel_argument_abi_semantic_projection_kind_mismatch_field_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_unsupported_encoding_field_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_scalar_narrowing_overflow_field_count,
        launch_execution.kernel_argument_abi_semantic_projection_field_range_overflow_count,
        launch_execution.kernel_argument_abi_semantic_projection_projected_kernarg_bytes
    );
    println!(
        "launch_executable_semantic_projection_selection: requests={} missing={} requested_projected_kernarg_bytes={} applied={} plan_ready={}",
        launch_execution
            .kernel_argument_abi_semantic_projection_candidate_selection_request_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_candidate_selection_requested_kernarg_bytes,
        launch_execution.kernel_argument_abi_semantic_projection_candidate_selection_applied_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan_ready
    );

    Ok(())
}

fn synthetic_available_checkpoint_keys(plan: &ModelCheckpointBindingPlan) -> Vec<String> {
    let mut keys = Vec::new();
    for entry in &plan.entries {
        if let Some((prefix, suffix)) = entry.checkpoint_key.split_once('*') {
            let expert_count = entry.shape.first().copied().unwrap_or_default();
            for expert in 0..expert_count {
                keys.push(format!("{prefix}{expert}{suffix}"));
            }
        } else {
            keys.push(entry.checkpoint_key.clone());
        }
    }
    keys
}

fn live_aql_batch_plan_proof() -> KfdQueueLiveAqlBatchReservationPlanProof {
    KfdQueueLiveAqlBatchReservationPlanInput {
        probe_version: 1,
        base_packet_id: 18,
        packet_count: 2,
        last_packet_id: 19,
        desired_write_index: 20,
        read_index: 15,
        inflight_packets: 3,
        capacity_ok: 1,
        slot0_va: 0x1000,
        slot1_va: 0x1040,
        slot0_offset: 1152,
        slot1_offset: 1216,
        slot0_index: 18,
        slot1_index: 19,
        slots_distinct: 1,
        slots_aligned64: 1,
        slot0_formula_ok: 1,
        slot1_formula_ok: 1,
        doorbell_packet_id: 19,
        doorbell_matches_last_packet: 1,
        single_doorbell_contract: 1,
        reserve_before_payload_contract: 1,
        payloads_before_headers_contract: 1,
        headers_before_doorbell_contract: 1,
        release_header_store_contract: 1,
        write_index_fetch_add_pending: 1,
        payload_writes_pending: 1,
        valid_headers_pending: 1,
        doorbell_pending: 1,
        no_live_queue_mutation_contract: 1,
        first_slot_matches_single_reservation: 1,
        observed_ready: 1,
        expected_restage_or_payload_ready: true,
        expected_capacity_ok: true,
        expected_slots_distinct: true,
        expected_slots_aligned64: true,
        expected_slot0_formula_ok: true,
        expected_slot1_formula_ok: true,
        expected_doorbell_matches_last_packet: true,
        expected_first_slot_matches_single_reservation: true,
    }
    .proof()
}

fn live_aql_materialized_packet_plan_proof() -> KfdQueueLiveAqlMaterializedPacketPlanProof {
    KfdQueueLiveAqlMaterializedPacketPlanInput {
        probe_version: 1,
        packet0_packet_id: 18,
        packet1_packet_id: 19,
        packet0_slot_va: 0x1000,
        packet1_slot_va: 0x1040,
        packet0_word0: 0x0001_1502,
        packet0_word4_kernel_object: 0x2000,
        packet0_word5_kernarg_va: 0x3000,
        packet1_word0: 0x0001_1502,
        packet1_word4_kernel_object: 0x2000,
        packet1_word5_kernarg_va: 0x3000,
        packet0_words_match_host_template: 1,
        packet1_words_match_host_template: 1,
        payload_words_match_host_template: 1,
        header_words_match_host_template: 1,
        target_slots_match_batch_plan: 1,
        packet0_slot_offset: 1152,
        packet1_slot_offset: 1216,
        packet_bytes: 64,
        packet_count: 2,
        batch_plan_ready: 1,
        reserve_first_restage_ready: 1,
        payloads_before_headers_contract: 1,
        release_header_store_contract: 1,
        doorbell_pending: 1,
        no_live_queue_mutation_contract: 1,
        packet_plan_ready: 1,
        publish_low32: 0x0001_1502,
        packet0_low32: 0x0001_1502,
        aql_packet_image_ready: 1,
        expected_batch_ready: true,
        expected_reserve_restage_plan_ready: true,
        expected_aql_packet_image_ready: true,
        expected_target_slots_match_batch_plan: true,
    }
    .proof()
}

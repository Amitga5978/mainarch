use anyhow::Result;
use mainarch_core::model_api::prelude::*;
use mainarch_model_api_plugin_example::{
    external_cpu_demo_live_aql_proof_validations, ExternalMiniMoe, EXTERNAL_PLUGIN_MODEL_NAME,
    EXTERNAL_PLUGIN_PACKAGE,
};

#[test]
fn external_mini_moe_library_exposes_static_model_api_contract() -> Result<()> {
    assert_eq!(
        MODEL_API_CONTRACT.receipt_text(),
        include_str!("../expected-contract.receipt")
    );
    assert_eq!(
        MODEL_API_CONTRACT.receipt_fingerprint(),
        "a97b03d8480b0c7b4ccbf88f312398f43ae9fd6d4040829c48eb08e230666b1c"
    );

    let model = ExternalMiniMoe::default();
    assert_eq!(model.name(), EXTERNAL_PLUGIN_MODEL_NAME);
    assert_eq!(EXTERNAL_PLUGIN_PACKAGE, "examples/model-api-plugin");

    let catalog = MainarchPrimitiveLoweringCatalog::mi355_reference();
    let report = inspect_model_plugin(&model, &catalog)?;
    report.assert_consistent()?;
    assert!(report.is_accepted());
    report.assert_accepted()?;
    let mut stale_accepted_report = report.clone();
    stale_accepted_report.primitive_vocabulary.pop();
    assert!(!stale_accepted_report.is_accepted());
    let stale_accepted_report_err = stale_accepted_report
        .assert_accepted()
        .unwrap_err()
        .to_string();
    assert!(stale_accepted_report_err.contains("not accepted"));
    assert!(stale_accepted_report_err.contains("consistency"));
    assert!(stale_accepted_report_err.contains("primitive vocabulary descriptors drifted"));
    assert!(report.is_static_handoff_ready());
    report.assert_static_handoff_ready()?;
    report.readiness.assert_static_runtime_ready()?;
    let mut stale_vocabulary_report = report.clone();
    stale_vocabulary_report.primitive_vocabulary.pop();
    assert!(!stale_vocabulary_report.is_static_handoff_ready());
    let stale_vocabulary_err = stale_vocabulary_report
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_vocabulary_err.contains("not static handoff ready"));
    assert!(stale_vocabulary_err.contains("consistency"));
    assert!(stale_vocabulary_err.contains("primitive vocabulary descriptors drifted"));
    let mut stale_compatibility_report = report.clone();
    stale_compatibility_report.compatibility.accepted = false;
    assert!(!stale_compatibility_report.is_static_handoff_ready());
    let stale_compatibility_err = stale_compatibility_report
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_compatibility_err.contains("not static handoff ready"));
    assert!(stale_compatibility_err.contains("consistency"));
    assert!(
        stale_compatibility_err.contains("compatibility report does not match manifest-derived")
    );
    let mut stale_readiness_report = report.clone();
    stale_readiness_report
        .readiness
        .checkpoint
        .missing_weight_tensors
        .push("unbound_static_handoff_weight".into());
    assert!(!stale_readiness_report.is_static_handoff_ready());
    let stale_readiness_err = stale_readiness_report
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_readiness_err.contains("not static handoff ready"));
    assert!(stale_readiness_err.contains("manifest does not match readiness-derived"));
    let plugin_rejection = report.rejection_report();
    plugin_rejection.assert_consistent()?;
    plugin_rejection.assert_no_rejection()?;
    assert!(!plugin_rejection.is_rejected());
    let mut stale_no_rejection = plugin_rejection.clone();
    stale_no_rejection.summary.accepted = false;
    assert!(!stale_no_rejection.is_rejected());
    let stale_no_rejection_err = stale_no_rejection
        .assert_rejected()
        .unwrap_err()
        .to_string();
    assert!(stale_no_rejection_err.contains("not rejected"));
    assert!(stale_no_rejection_err.contains("summary accepted false != expected true"));
    let metadata_bindings = report
        .readiness
        .slots
        .metadata_binding_template("external_plugin")?;
    let metadata_admission = report
        .readiness
        .validate_metadata_runtime_admission(&metadata_bindings);
    metadata_admission.assert_consistent()?;
    metadata_admission.assert_admitted()?;
    assert!(metadata_admission.is_admitted());
    let mut stale_stage_dispatch_target_admission = metadata_admission.clone();
    stale_stage_dispatch_target_admission
        .stage_dispatch_bindings
        .dispatch_target = "stale-stage-dispatch-target";
    assert!(!stale_stage_dispatch_target_admission.is_admitted());
    let stale_stage_dispatch_target_admission_err = stale_stage_dispatch_target_admission
        .assert_admitted()
        .unwrap_err()
        .to_string();
    assert!(stale_stage_dispatch_target_admission_err.contains("consistency"));
    assert!(stale_stage_dispatch_target_admission_err
        .contains("stage dispatch binding dispatch target"));

    let summary = report.summary();
    summary.assert_consistent_with(&report)?;
    assert_eq!(summary.model_primitive_kind_count, 6);
    assert_eq!(summary.model_stage_kind_count, 4);
    assert_eq!(summary.tensor_count, 14);
    assert_eq!(summary.op_count, 6);
    assert_eq!(summary.runtime_dispatch_count, 6);
    assert_eq!(summary.compatibility_issue_count, 0);
    assert!(!summary.live_execution_supported);

    let manifest = &report.manifest;
    manifest.assert_static_metadata_ready()?;
    manifest.assert_compatible_with(&catalog)?;
    assert_eq!(manifest.model_name, EXTERNAL_PLUGIN_MODEL_NAME);
    assert_eq!(
        manifest.contract_fingerprint,
        "2557fed1ee5509f978b91236fe5acfff40770129e6de7741303bccc645eff2e6"
    );
    assert_eq!(
        manifest.receipt_text(),
        include_str!("../expected-manifest.receipt")
    );
    assert_eq!(
        manifest.receipt_fingerprint(),
        "c0dc0712a0c5fac62f18b1895c9f0c0bdad8bfb4ffcbcdb9c14b9e0c9ead1458"
    );
    let compatibility = &report.compatibility;
    compatibility.assert_consistent()?;
    compatibility.assert_accepted()?;
    assert_eq!(
        compatibility.receipt_text(),
        include_str!("../expected-compatibility.receipt")
    );
    assert_eq!(
        compatibility.receipt_fingerprint(),
        "7e9fcc96f1d3072b1b136b30d737969960237c323a56ca63dd18739f74acab47"
    );
    let wrong_target_catalog = MainarchPrimitiveLoweringCatalog {
        target: "external-different-raw-aql-target",
    };
    let wrong_target_compatibility = manifest.compatibility_report(&wrong_target_catalog);
    wrong_target_compatibility.assert_consistent()?;
    assert!(!wrong_target_compatibility.is_accepted());
    let mut stale_accepted_compatibility = wrong_target_compatibility.clone();
    stale_accepted_compatibility.accepted = true;
    assert!(!stale_accepted_compatibility.is_accepted());
    let stale_accepted_compatibility_err = stale_accepted_compatibility
        .assert_accepted()
        .unwrap_err()
        .to_string();
    assert!(stale_accepted_compatibility_err.contains("not accepted"));
    assert!(stale_accepted_compatibility_err.contains("consistency"));
    assert!(stale_accepted_compatibility_err.contains("accepted true != expected false"));
    let static_launch_request_step_plans = RuntimeLaunchExecutionRequestStep::DESCRIPTORS
        .iter()
        .map(|descriptor| descriptor.request_plan)
        .collect::<Vec<_>>();
    let static_live_aql_proof_surface_plans = RuntimeLaunchExecutionRequestStep::DESCRIPTORS
        .iter()
        .filter(|descriptor| descriptor.step.requires_live_aql_proof())
        .map(|descriptor| descriptor.request_plan)
        .collect::<Vec<_>>();
    let static_launch_request_step_requirements = RuntimeLaunchExecutionRequestStep::DESCRIPTORS
        .iter()
        .map(|descriptor| descriptor.requirement)
        .collect::<Vec<_>>();
    assert_eq!(
        manifest.runtime_launch_request_step_count,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len()
    );
    assert_eq!(
        manifest.runtime_launch_request_steps.as_slice(),
        &RuntimeLaunchExecutionRequestStep::DESCRIPTORS[..]
    );
    assert_eq!(
        manifest.runtime_launch_request_step_labels.as_slice(),
        static_launch_request_step_plans.as_slice()
    );
    assert_eq!(
        static_launch_request_step_plans,
        [
            "code_object_load_request_plan",
            "code_object_base_binding_request_plan",
            "completion_signal_binding_request_plan",
            "queue_reservation_request_plan",
            "kernarg_allocation_request_plan",
            "kernel_argument_abi_schema_request_plan",
            "kernel_candidate_selection_request_plan",
            "kernel_argument_abi_semantic_projection_candidate_selection_request_plan",
            "host_launcher_branch_resolution_request_plan",
            "aql_live_relocation_binding_request_plan",
        ]
    );
    assert_eq!(
        static_launch_request_step_requirements,
        [
            "loaded_code_object_base",
            "loaded_code_object_base",
            "completion_signal_binding",
            "queue_reservation",
            "kernarg_allocation",
            "kernel_argument_abi_verification",
            "kernel_candidate_selection_policy",
            "kernel_argument_abi_semantic_projection",
            "host_launcher_runtime_branch_resolution",
            "aql_packet_materialization",
        ]
    );
    for descriptor in RuntimeLaunchExecutionRequestStep::DESCRIPTORS {
        assert_eq!(
            manifest.launch_request_step_for(descriptor.step),
            Some(&descriptor)
        );
        assert_eq!(
            manifest.launch_request_step_for_request_plan(descriptor.request_plan),
            Some(&descriptor)
        );
        assert_eq!(
            RuntimeLaunchExecutionRequestStep::from_request_plan(descriptor.request_plan),
            Some(descriptor.step)
        );
        assert_eq!(
            RuntimeLaunchExecutionRequestStep::descriptor_for_request_plan(descriptor.request_plan),
            Some(descriptor)
        );
    }
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::from_request_plan("missing_request_plan"),
        None
    );
    assert_eq!(
        manifest.launch_request_step_for_request_plan("missing_request_plan"),
        None
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::descriptor_for_request_plan("missing_request_plan"),
        None
    );
    assert_eq!(
        manifest.runtime_live_aql_proof_step_count,
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS.len()
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_STEPS,
        [
            RuntimeLaunchExecutionRequestStep::QueueReservation,
            RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding,
        ]
    );
    let static_live_aql_proof_step_plans =
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.request_plan)
            .collect::<Vec<_>>();
    let static_live_aql_proof_inputs =
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.live_aql_proof_input.unwrap())
            .collect::<Vec<_>>();
    let static_live_aql_proof_kinds = RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS
        .iter()
        .map(|descriptor| descriptor.live_aql_proof_kind.unwrap().as_str())
        .collect::<Vec<_>>();
    let static_live_aql_validation_methods =
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.live_aql_validation_method.unwrap())
            .collect::<Vec<_>>();
    assert_eq!(
        static_live_aql_proof_step_plans,
        [
            "queue_reservation_request_plan",
            "aql_live_relocation_binding_request_plan",
        ]
    );
    assert_eq!(
        static_live_aql_proof_inputs,
        [
            "KfdQueueLiveAqlBatchReservationPlanInput",
            "KfdQueueLiveAqlMaterializedPacketPlanInput",
        ]
    );
    assert_eq!(
        static_live_aql_proof_kinds,
        ["batch_reservation_plan", "materialized_packet_plan"]
    );
    assert_eq!(
        manifest.live_aql_proof_kind_labels(),
        static_live_aql_proof_kinds
    );
    assert_eq!(
        manifest.live_aql_proof_input_labels(),
        static_live_aql_proof_inputs
    );
    assert_eq!(
        manifest.live_aql_validation_method_labels(),
        static_live_aql_validation_methods
    );
    assert!(manifest
        .receipt_text()
        .contains("runtime_launch_request_steps.3.live_aql_proof_kind=batch_reservation_plan"));
    assert!(manifest
        .receipt_text()
        .contains("runtime_launch_request_steps.9.live_aql_proof_kind=materialized_packet_plan"));
    assert_eq!(
        static_live_aql_validation_methods,
        [
            "KfdQueueLiveAqlBatchReservationPlanProof::validate_ready",
            "KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready",
        ]
    );
    assert!(RuntimeLaunchExecutionRequestStep::QueueReservation.requires_live_aql_proof());
    assert!(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding.requires_live_aql_proof());
    assert!(!RuntimeLaunchExecutionRequestStep::KernelCandidateSelection.requires_live_aql_proof());
    assert!(!manifest.live_execution_supported);

    let slot_bindings = report
        .readiness
        .slots
        .metadata_binding_template("external_plugin")?;
    let device_pointer_bindings = report.readiness.slots.device_pointer_binding_template(
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    let device_pointer_validation = report
        .readiness
        .slots
        .validate_complete_device_pointer_bindings(&device_pointer_bindings);
    device_pointer_validation.assert_complete()?;
    report
        .readiness
        .validate_metadata_runtime_admission(&slot_bindings)
        .assert_admitted()?;

    let code_object = CodeObjectInfo::inspect(MAINARCH_KERNELS_GFX950)?;
    let launch_projection = report
        .readiness
        .runtime_launch_kernel_argument_abi_semantic_projection_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    launch_projection.assert_consistent()?;
    assert_eq!(launch_projection.kernel_candidate_count, 19);
    assert_eq!(launch_projection.semantic_schema_candidate_count, 19);
    assert_eq!(launch_projection.missing_semantic_schema_candidate_count, 0);
    assert_eq!(launch_projection.projection_ready_candidate_count, 3);
    assert_eq!(
        launch_projection.dispatches_with_projection_ready_candidate_count,
        2
    );
    assert_eq!(
        launch_projection.dispatches_without_projection_ready_candidate_count,
        4
    );
    assert!(!launch_projection.semantic_projection_ready);

    let projection_selection = launch_projection
        .kernel_argument_abi_semantic_projection_candidate_recommendation_plan()?
        .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
            &launch_projection,
        )?;
    projection_selection.assert_consistent()?;
    projection_selection.assert_consistent_with_projection(&launch_projection)?;
    assert_eq!(projection_selection.selection_request_count, 2);
    assert_eq!(projection_selection.missing_selection_request_count, 4);
    assert!(projection_selection.request_plan_ready);
    assert!(!projection_selection.all_selection_requests_ready);
    let projection_selection_symbol_pairs =
        projection_selection.selection_request_op_kernel_symbols();
    assert_eq!(
        projection_selection_symbol_pairs,
        vec![
            (
                "layers.0.router_topk".to_string(),
                "moe_router_topk".to_string()
            ),
            ("lm_head".to_string(), "gemv_f16".to_string()),
        ]
    );
    assert_eq!(
        projection_selection_symbol_pairs.len(),
        projection_selection.selection_request_count
    );
    assert_eq!(
        projection_selection_symbol_pairs
            .iter()
            .map(|(op_name, kernel_symbol)| format!("{op_name}={kernel_symbol}"))
            .collect::<Vec<_>>(),
        projection_selection.selection_request_op_kernel_symbol_labels()
    );

    let kernel_selection = report
        .readiness
        .runtime_launch_kernel_candidate_selection_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    kernel_selection.assert_consistent()?;
    assert_eq!(kernel_selection.selection_request_count, 3);
    assert_eq!(kernel_selection.missing_selection_request_count, 3);
    assert_eq!(kernel_selection.verified_candidate_count, 6);
    assert!(kernel_selection.request_plan_ready);
    assert!(!kernel_selection.all_selection_requests_ready);
    let kernel_selection_symbol_pairs = kernel_selection.selection_request_op_kernel_symbols();
    assert_eq!(
        kernel_selection_symbol_pairs,
        vec![
            (
                "embed_tokens".to_string(),
                "decode_step_embed_rmsnorm_token_f16".to_string(),
            ),
            (
                "layers.0.router_topk".to_string(),
                "moe_router_gemv_topk_log_step".to_string(),
            ),
            ("greedy_argmax".to_string(), "argmax_f32_step".to_string()),
        ]
    );
    assert_eq!(
        kernel_selection_symbol_pairs.len(),
        kernel_selection.selection_request_count
    );
    assert_eq!(
        kernel_selection_symbol_pairs
            .iter()
            .map(|(op_name, kernel_symbol)| format!("{op_name}={kernel_symbol}"))
            .collect::<Vec<_>>(),
        kernel_selection.selection_request_op_kernel_symbol_labels()
    );

    let host_launcher_branch_requests = report
        .readiness
        .runtime_launch_host_launcher_branch_resolution_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?;
    host_launcher_branch_requests.assert_consistent()?;
    assert_eq!(
        host_launcher_branch_requests.branch_resolution_request_count,
        4
    );
    assert_eq!(
        host_launcher_branch_requests.branch_resolution_applied_count,
        0
    );
    assert_eq!(
        host_launcher_branch_requests.unresolved_candidate_symbol_count,
        17
    );
    assert!(host_launcher_branch_requests.request_plan_ready);
    assert!(!host_launcher_branch_requests.all_branches_resolved);
    let host_launcher_branch_candidate_symbol_sets =
        host_launcher_branch_requests.branch_resolution_request_candidate_symbol_sets();
    assert_eq!(
        host_launcher_branch_candidate_symbol_sets,
        vec![
            (
                "layers.0.router_topk".to_string(),
                vec![
                    "moe_router_topk".to_string(),
                    "moe_router_gemv_topk_log_step".to_string(),
                    "moe_router_gemv_topk_log_step_e16_k4096_top8".to_string(),
                ],
            ),
            (
                "layers.0.moe_local_ffn".to_string(),
                vec![
                    "moe_gate_up_swiglu".to_string(),
                    "moe_gate_up_swiglu_slots".to_string(),
                    "moe_gate_up_swiglu_slots_k4096".to_string(),
                    "moe_down_accum".to_string(),
                    "moe_down_accum_slots".to_string(),
                    "moe_down_accum_slots_i1536".to_string(),
                    "moe_down_accum_slots_i512".to_string(),
                ],
            ),
            (
                "lm_head".to_string(),
                vec![
                    "gemv_f16".to_string(),
                    "gemv_f16_k8192".to_string(),
                    "gemv_f16_step".to_string(),
                    "gemv_f16_step_k4096".to_string(),
                ],
            ),
            (
                "greedy_argmax".to_string(),
                vec![
                    "argmax_f32_step".to_string(),
                    "argmax_f32_token_ids_write_candidate".to_string(),
                    "argmax_f32_token_ids_write_candidate_n1187".to_string(),
                ],
            ),
        ]
    );
    assert_eq!(
        host_launcher_branch_candidate_symbol_sets.len(),
        host_launcher_branch_requests.branch_resolution_request_count
    );
    assert_eq!(
        host_launcher_branch_candidate_symbol_sets
            .iter()
            .map(|(op_name, _)| op_name.clone())
            .collect::<Vec<_>>(),
        host_launcher_branch_requests.branch_resolution_request_op_names()
    );
    assert_eq!(
        host_launcher_branch_candidate_symbol_sets
            .iter()
            .map(|(op_name, candidate_symbols)| format!(
                "{op_name}={}",
                candidate_symbols.join("|")
            ))
            .collect::<Vec<_>>(),
        host_launcher_branch_requests.branch_resolution_request_candidate_symbol_labels()
    );
    assert_eq!(
        host_launcher_branch_requests
            .unresolved_candidate_symbols()
            .len(),
        host_launcher_branch_requests.unresolved_candidate_symbol_count
    );

    let launch_execution = report.readiness.runtime_launch_execution_readiness_report(
        &slot_bindings,
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_execution.assert_consistent()?;
    assert!(!launch_execution.executable);
    assert!(launch_execution.is_non_executable_boundary());
    launch_execution.assert_non_executable_boundary()?;
    let mut stale_count_launch_execution = launch_execution.clone();
    stale_count_launch_execution.argument_count += 1;
    assert!(!stale_count_launch_execution.is_non_executable_boundary());
    let stale_count_launch_execution_err = stale_count_launch_execution
        .assert_non_executable_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_launch_execution_err.contains("consistency"));
    assert!(stale_count_launch_execution_err.contains("pointer+scalar argument count"));
    macro_rules! assert_launch_execution_non_executable_rejected {
        ($field:ident = $value:expr, $needle:literal $(,)?) => {{
            let mut report = launch_execution.clone();
            report.$field = $value;
            assert!(!report.is_non_executable_boundary());
            assert!(report
                .assert_non_executable_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
        (clear_blockers, $needle:literal $(,)?) => {{
            let mut report = launch_execution.clone();
            report.blockers.clear();
            assert!(!report.is_non_executable_boundary());
            assert!(report
                .assert_non_executable_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
        (clear_unresolved_runtime_requirements, $needle:literal $(,)?) => {{
            let mut report = launch_execution.clone();
            report.unresolved_runtime_requirements.clear();
            assert!(!report.is_non_executable_boundary());
            assert!(report
                .assert_non_executable_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
    }
    assert_launch_execution_non_executable_rejected!(
        executable = true,
        "launch execution executable is true",
    );
    assert_launch_execution_non_executable_rejected!(
        clear_blockers,
        "launch execution blockers are empty",
    );
    assert_launch_execution_non_executable_rejected!(
        clear_unresolved_runtime_requirements,
        "unresolved runtime requirements are empty",
    );
    assert_launch_execution_non_executable_rejected!(
        aql_packet_materialization_dispatchable_packet_count = 1,
        "dispatchable AQL packet count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        code_object_loaded_count = 1,
        "loaded code objects 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        loaded_code_object_base_bound_count = 1,
        "loaded code object base bound count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        kernel_descriptor_bound_count = 1,
        "kernel descriptor bound count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        aql_kernel_object_relocation_bound_count = 1,
        "AQL kernel_object relocation bound count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        completion_signal_handle_bound_count = 1,
        "completion signal handle bound count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        queue_reservation_packet_reserved_count = 1,
        "queue reservation packet reserved count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        queue_reservation_doorbell_bound_count = 1,
        "queue reservation doorbell bound count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        queue_reservation_applied_count = 1,
        "queue reservation applied count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        kernarg_allocation_bound_count = 1,
        "kernarg allocation bound count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        kernarg_allocation_bound_bytes = 1,
        "kernarg allocation bound bytes 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        kernarg_copy_applied_count = 1,
        "kernarg copy applied count 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        kernarg_copy_applied_bytes = 1,
        "kernarg copy applied bytes 1 != 0",
    );
    assert_launch_execution_non_executable_rejected!(
        aql_live_relocation_binding_bound_count = 1,
        "AQL live relocation bound count 1 != 0",
    );
    assert_eq!(
        launch_execution
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation execution readiness blocker")
            .requirement,
        RuntimeLaunchExecutionRequestStep::QueueReservation.requirement()
    );
    assert_eq!(
        launch_execution.aql_packet_materialization_dispatchable_packet_count,
        0
    );
    let explicit_launch_execution_requests =
        report.readiness.runtime_launch_execution_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    explicit_launch_execution_requests.assert_consistent()?;
    let launch_execution_requests =
        report.synthetic_cpu_runtime_launch_execution_request_plan("external_plugin")?;
    assert_eq!(
        launch_execution_requests,
        explicit_launch_execution_requests
    );
    let launch_execution_component_request_plans =
        launch_execution_requests.component_request_plan_names();
    assert_eq!(
        launch_execution_component_request_plans,
        static_launch_request_step_plans
    );
    for descriptor in RuntimeLaunchExecutionRequestStep::DESCRIPTORS {
        let component = launch_execution_requests
            .component_for(descriptor.request_plan)
            .expect("runtime launch request component exists for static descriptor");
        assert_eq!(component.step, descriptor.step);
        assert_eq!(component.step_index, descriptor.step_index);
        assert_eq!(component.requirement, descriptor.requirement);
        assert_eq!(
            RuntimeLaunchExecutionRequestStep::descriptor_for_request_plan(component.request_plan),
            Some(descriptor)
        );
        assert_eq!(
            launch_execution_requests
                .component_for_step(descriptor.step)
                .expect("runtime launch request component exists for typed descriptor")
                .request_plan,
            component.request_plan
        );
    }
    assert!(launch_execution_requests.is_non_submitting_boundary());
    launch_execution_requests.assert_non_submitting_boundary()?;
    let mut stale_count_launch_execution_requests = launch_execution_requests.clone();
    stale_count_launch_execution_requests.component_pending_count = 0;
    assert!(!stale_count_launch_execution_requests.is_non_submitting_boundary());
    let stale_count_launch_execution_requests_err = stale_count_launch_execution_requests
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_launch_execution_requests_err.contains("consistency"));
    assert!(stale_count_launch_execution_requests_err.contains("component pending count"));
    macro_rules! assert_execution_request_non_submitting_rejected {
        ($plan:expr, $needle:literal $(,)?) => {{
            let plan = $plan;
            assert!(!plan.is_non_submitting_boundary());
            assert!(plan
                .assert_non_submitting_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
    }
    let mut submitting_surface_count_launch_execution_requests = launch_execution_requests.clone();
    submitting_surface_count_launch_execution_requests.live_aql_submitting_surface_count = 1;
    assert_execution_request_non_submitting_rejected!(
        submitting_surface_count_launch_execution_requests,
        "live AQL submitting surfaces 1 != 0",
    );
    let mut submitting_surface_row_launch_execution_requests = launch_execution_requests.clone();
    submitting_surface_row_launch_execution_requests
        .live_aql_proof_surfaces
        .iter_mut()
        .find(|surface| surface.request_plan == "queue_reservation_request_plan")
        .unwrap()
        .submits_work = true;
    assert_execution_request_non_submitting_rejected!(
        submitting_surface_row_launch_execution_requests,
        "live AQL submitting surface rows queue_reservation_request_plan",
    );
    let mut queue_mutating_component_count_launch_execution_requests =
        launch_execution_requests.clone();
    queue_mutating_component_count_launch_execution_requests.live_queue_mutating_component_count =
        1;
    assert_execution_request_non_submitting_rejected!(
        queue_mutating_component_count_launch_execution_requests,
        "live queue mutating components 1 != 0",
    );
    let mut queue_mutating_component_row_launch_execution_requests =
        launch_execution_requests.clone();
    queue_mutating_component_row_launch_execution_requests
        .components
        .iter_mut()
        .find(|component| component.request_plan == "code_object_load_request_plan")
        .unwrap()
        .mutates_live_queue = true;
    assert_execution_request_non_submitting_rejected!(
        queue_mutating_component_row_launch_execution_requests,
        "live queue mutating component rows code_object_load_request_plan",
    );
    let explicit_launch_submission_gate = launch_execution_requests.submission_gate()?;
    explicit_launch_submission_gate.assert_consistent()?;
    let launch_submission_gate =
        report.synthetic_cpu_runtime_launch_submission_gate("external_plugin")?;
    assert_eq!(launch_submission_gate, explicit_launch_submission_gate);
    assert!(launch_submission_gate.is_non_submitting_boundary());
    launch_submission_gate.assert_non_submitting_boundary()?;
    let mut stale_count_launch_submission_gate = launch_submission_gate.clone();
    stale_count_launch_submission_gate.submission_blocker_count = 0;
    assert!(!stale_count_launch_submission_gate.is_non_submitting_boundary());
    let stale_count_launch_submission_gate_err = stale_count_launch_submission_gate
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_launch_submission_gate_err.contains("consistency"));
    assert!(stale_count_launch_submission_gate_err.contains("submission blocker count"));
    macro_rules! assert_submission_gate_non_submitting_rejected {
        ($gate:expr, $needle:literal $(,)?) => {{
            let gate = $gate;
            assert!(!gate.is_non_submitting_boundary());
            assert!(gate
                .assert_non_submitting_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
    }
    let mut side_effect_guard_submission_gate = launch_submission_gate.clone();
    side_effect_guard_submission_gate.no_live_aql_submission_side_effects = false;
    assert_submission_gate_non_submitting_rejected!(
        side_effect_guard_submission_gate,
        "live AQL submission side-effect guard is false",
    );
    let mut submitting_surface_submission_gate = launch_submission_gate.clone();
    submitting_surface_submission_gate.live_aql_submitting_surface_count = 1;
    assert_submission_gate_non_submitting_rejected!(
        submitting_surface_submission_gate,
        "live AQL submitting surfaces 1 != 0",
    );
    let mut queue_mutation_guard_submission_gate = launch_submission_gate.clone();
    queue_mutation_guard_submission_gate.no_live_queue_mutation = false;
    assert_submission_gate_non_submitting_rejected!(
        queue_mutation_guard_submission_gate,
        "live queue mutation guard is false",
    );
    let mut queue_mutating_submission_gate = launch_submission_gate.clone();
    queue_mutating_submission_gate.live_queue_mutating_component_count = 1;
    assert_submission_gate_non_submitting_rejected!(
        queue_mutating_submission_gate,
        "live queue mutating components 1 != 0",
    );
    assert_eq!(
        launch_submission_gate
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation submission gate blocker")
            .requirement,
        RuntimeLaunchExecutionRequestStep::QueueReservation.requirement()
    );
    assert_eq!(
        launch_submission_gate.receipt_lines()[0],
        "receipt.kind=model_runtime_launch_submission_gate"
    );
    assert!(launch_submission_gate.receipt_text().ends_with('\n'));
    assert_eq!(
        launch_submission_gate.receipt_fingerprint(),
        "3af74757dc0bc16701c0902f811e293b6ee717fc86ed67a5467632a9289f7386"
    );
    assert_eq!(
        launch_submission_gate.receipt_text(),
        include_str!("../expected-runtime-submission-gate.receipt")
    );
    let explicit_launch_submission_blockers = launch_submission_gate.blocker_report()?;
    explicit_launch_submission_blockers.assert_consistent()?;
    let launch_submission_blockers =
        report.synthetic_cpu_runtime_launch_submission_blocker_report("external_plugin")?;
    assert_eq!(
        launch_submission_blockers,
        explicit_launch_submission_blockers
    );
    assert!(launch_submission_blockers.is_non_submitting_boundary());
    launch_submission_blockers.assert_non_submitting_boundary()?;
    let mut stale_count_launch_submission_blockers = launch_submission_blockers.clone();
    stale_count_launch_submission_blockers.blocker_count = 0;
    assert!(!stale_count_launch_submission_blockers.is_non_submitting_boundary());
    let stale_count_launch_submission_blockers_err = stale_count_launch_submission_blockers
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_launch_submission_blockers_err.contains("consistency"));
    assert!(stale_count_launch_submission_blockers_err.contains("submission blocker rows"));
    macro_rules! assert_submission_blocker_report_non_submitting_rejected {
        ($report:expr, $needle:literal $(,)?) => {{
            let report = $report;
            assert!(!report.is_non_submitting_boundary());
            assert!(report
                .assert_non_submitting_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
    }
    let mut side_effect_guard_submission_blockers = launch_submission_blockers.clone();
    side_effect_guard_submission_blockers.no_live_aql_submission_side_effects = false;
    assert_submission_blocker_report_non_submitting_rejected!(
        side_effect_guard_submission_blockers,
        "live AQL submission side-effect guard is false",
    );
    let mut side_effecting_submission_blockers = launch_submission_blockers.clone();
    side_effecting_submission_blockers.live_aql_submission_side_effect_count = 1;
    assert_submission_blocker_report_non_submitting_rejected!(
        side_effecting_submission_blockers,
        "live AQL submission side-effect count 1 != 0",
    );
    let mut queue_mutation_guard_submission_blockers = launch_submission_blockers.clone();
    queue_mutation_guard_submission_blockers.no_live_queue_mutation = false;
    assert_submission_blocker_report_non_submitting_rejected!(
        queue_mutation_guard_submission_blockers,
        "live queue mutation guard is false",
    );
    let mut queue_mutating_submission_blockers = launch_submission_blockers.clone();
    queue_mutating_submission_blockers.live_queue_mutation_count = 1;
    assert_submission_blocker_report_non_submitting_rejected!(
        queue_mutating_submission_blockers,
        "live queue mutation count 1 != 0",
    );
    assert!(
        launch_submission_blockers
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .expect("AQL live relocation submission blocker")
            .execution_readiness_blocker
    );
    assert_eq!(
        launch_submission_blockers.receipt_lines()[0],
        "receipt.kind=model_runtime_launch_submission_blocker_report"
    );
    assert!(launch_submission_blockers.receipt_text().ends_with('\n'));
    assert_eq!(
        launch_submission_blockers.receipt_fingerprint(),
        "6679269ebbd4f6cdafcda192ff6d1fb560ac0b4ed03d5e6313168753ce5aa305"
    );
    assert_eq!(
        launch_submission_blockers.receipt_text(),
        include_str!("../expected-runtime-submission-blocker-report.receipt")
    );
    assert_eq!(
        launch_execution_requests.live_aql_proof_surface_request_plan_names(),
        static_live_aql_proof_step_plans
    );
    assert_eq!(
        launch_execution_requests.live_aql_proof_input_labels(),
        static_live_aql_proof_inputs
    );
    assert_eq!(
        launch_execution_requests.live_aql_proof_kind_labels(),
        static_live_aql_proof_kinds
    );
    assert_eq!(
        launch_execution_requests.live_aql_validation_method_labels(),
        static_live_aql_validation_methods
    );
    for descriptor in RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS {
        let surface = launch_execution_requests
            .live_aql_proof_surface_for(descriptor.request_plan)
            .expect("live AQL proof surface exists for static descriptor");
        assert_eq!(surface.step, descriptor.step);
        assert_eq!(surface.step_index, descriptor.step_index);
        assert_eq!(surface.requirement, descriptor.requirement);
        assert_eq!(Some(surface.proof_kind), descriptor.live_aql_proof_kind);
        assert_eq!(Some(surface.proof_input), descriptor.live_aql_proof_input);
        assert!(!surface.proof_type.is_empty());
        assert!(!surface.validation_type.is_empty());
        assert_eq!(surface.validation_ready_field, "ready");
        assert_eq!(
            surface.no_live_queue_mutation_contract_field,
            "no_live_queue_mutation_contract"
        );
        assert_eq!(
            Some(surface.validation_method),
            descriptor.live_aql_validation_method
        );
        assert_eq!(
            launch_execution_requests
                .live_aql_proof_surface_for_step(descriptor.step)
                .expect("live AQL proof surface exists for typed descriptor")
                .request_plan,
            surface.request_plan
        );
    }
    assert!(launch_execution_requests
        .live_aql_proof_surface_for("missing_request_plan")
        .is_none());
    assert_eq!(
        launch_execution_requests
            .live_aql_proof_surface_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation live AQL proof surface")
            .proof_input,
        RuntimeLaunchExecutionRequestStep::QueueReservation
            .live_aql_proof_input()
            .expect("queue reservation proof input")
    );
    assert_eq!(
        launch_execution_requests
            .live_aql_proof_surface_for_step(
                RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding
            )
            .expect("AQL live relocation proof surface")
            .validation_method,
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding
            .live_aql_validation_method()
            .expect("AQL live relocation validation method")
    );
    assert!(launch_execution_requests
        .live_aql_proof_surface_for_step(RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSchema)
        .is_none());
    assert_eq!(
        launch_execution_requests.receipt_text(),
        include_str!("../expected-runtime-launch-request.receipt")
    );
    assert_eq!(
        launch_execution_requests.receipt_fingerprint(),
        "b5ecec66fe1ca766a26086e49826dd610ad84701a55b17e43dd3e23f5d4c45c5"
    );
    let explicit_launch_submission_prerequisites =
        launch_execution_requests.submission_prerequisite_plan()?;
    explicit_launch_submission_prerequisites.assert_consistent()?;
    let launch_submission_prerequisites =
        report.synthetic_cpu_runtime_launch_submission_prerequisite_plan("external_plugin")?;
    assert_eq!(
        launch_submission_prerequisites,
        explicit_launch_submission_prerequisites
    );
    launch_submission_prerequisites.assert_consistent()?;
    assert_eq!(
        launch_submission_prerequisites.prerequisite_request_plan_names(),
        static_launch_request_step_plans
    );
    assert_eq!(
        launch_submission_prerequisites.next_action_count,
        launch_submission_prerequisites.unsatisfied_prerequisite_count
    );
    assert_eq!(
        launch_submission_prerequisites.runtime_request_component_next_action_count,
        8
    );
    assert_eq!(
        launch_submission_prerequisites.live_aql_proof_validation_next_action_count,
        launch_execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        launch_submission_prerequisites.live_aql_proof_kind_labels(),
        static_live_aql_proof_kinds
    );
    assert_eq!(
        launch_submission_prerequisites.next_action_live_aql_proof_kind_labels(),
        static_live_aql_proof_kinds
    );
    assert_eq!(
        launch_submission_prerequisites.next_action_request_plan_names(),
        static_launch_request_step_plans
    );
    assert!(launch_submission_prerequisites.is_non_submitting_boundary());
    launch_submission_prerequisites.assert_non_submitting_boundary()?;
    let mut stale_count_submission_prerequisites = launch_submission_prerequisites.clone();
    stale_count_submission_prerequisites.unsatisfied_prerequisite_count = 0;
    assert!(!stale_count_submission_prerequisites.is_non_submitting_boundary());
    let stale_count_submission_prerequisites_err = stale_count_submission_prerequisites
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_submission_prerequisites_err.contains("consistency"));
    assert!(
        stale_count_submission_prerequisites_err.contains("satisfied+unsatisfied prerequisites")
    );
    macro_rules! assert_submission_prerequisite_non_submitting_rejected {
        ($plan:expr, $needle:literal $(,)?) => {{
            let plan = $plan;
            assert!(!plan.is_non_submitting_boundary());
            assert!(plan
                .assert_non_submitting_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
    }
    let mut side_effect_next_action_submission_prerequisites =
        launch_submission_prerequisites.clone();
    side_effect_next_action_submission_prerequisites
        .live_aql_submission_side_effect_next_action_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        side_effect_next_action_submission_prerequisites,
        "live AQL submission side-effect next actions 1 != 0",
    );
    let mut submitting_count_submission_prerequisites = launch_submission_prerequisites.clone();
    submitting_count_submission_prerequisites.live_aql_submitting_prerequisite_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        submitting_count_submission_prerequisites,
        "live AQL submitting prerequisites 1 != 0",
    );
    let mut submitting_row_submission_prerequisites = launch_submission_prerequisites.clone();
    submitting_row_submission_prerequisites.prerequisites[0].live_aql_submits_work = true;
    assert_submission_prerequisite_non_submitting_rejected!(
        submitting_row_submission_prerequisites,
        "live AQL submitting prerequisite rows code_object_load_request_plan",
    );
    let mut queue_mutation_next_action_submission_prerequisites =
        launch_submission_prerequisites.clone();
    queue_mutation_next_action_submission_prerequisites.live_queue_mutation_next_action_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        queue_mutation_next_action_submission_prerequisites,
        "live queue mutation next actions 1 != 0",
    );
    let mut queue_mutating_count_submission_prerequisites = launch_submission_prerequisites.clone();
    queue_mutating_count_submission_prerequisites.live_queue_mutating_prerequisite_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        queue_mutating_count_submission_prerequisites,
        "live queue mutating prerequisites 1 != 0",
    );
    let mut queue_mutating_row_submission_prerequisites = launch_submission_prerequisites.clone();
    queue_mutating_row_submission_prerequisites.prerequisites[0].mutates_live_queue = true;
    assert_submission_prerequisite_non_submitting_rejected!(
        queue_mutating_row_submission_prerequisites,
        "live queue mutating prerequisite rows code_object_load_request_plan",
    );
    assert_eq!(
        launch_submission_prerequisites.live_aql_proof_validation_next_action_request_plan_names(),
        static_live_aql_proof_surface_plans
    );
    for descriptor in RuntimeLaunchExecutionRequestStep::DESCRIPTORS {
        let prerequisite = launch_submission_prerequisites
            .prerequisite_for(descriptor.request_plan)
            .expect("submission prerequisite exists for static descriptor");
        assert_eq!(prerequisite.step, descriptor.step);
        assert_eq!(prerequisite.step_index, descriptor.step_index);
        assert_eq!(prerequisite.requirement, descriptor.requirement);
        assert_eq!(
            prerequisite.live_aql_proof_kind,
            descriptor.live_aql_proof_kind
        );
        assert_eq!(
            launch_submission_prerequisites
                .prerequisite_for_step(descriptor.step)
                .expect("submission prerequisite exists for typed descriptor")
                .request_plan,
            prerequisite.request_plan
        );
    }
    assert!(launch_submission_prerequisites
        .prerequisite_for("missing_request_plan")
        .is_none());
    assert_eq!(
        launch_submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation submission prerequisite")
            .request_plan,
        RuntimeLaunchExecutionRequestStep::QueueReservation.request_plan()
    );
    assert_eq!(
        launch_submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation submission prerequisite")
            .next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ValidateLiveAqlProof
    );
    assert_eq!(
        launch_submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation submission prerequisite")
            .next_action_live_aql_proof_kind,
        RuntimeLaunchExecutionRequestStep::QueueReservation.live_aql_proof_kind()
    );
    assert_eq!(
        launch_submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .expect("AQL live relocation submission prerequisite")
            .live_aql_validation_method,
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding.live_aql_validation_method()
    );
    assert_eq!(
        launch_submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .expect("AQL live relocation submission prerequisite")
            .next_action_input,
        "KfdQueueLiveAqlMaterializedPacketPlanInput"
    );
    assert_eq!(
        launch_submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .expect("AQL live relocation submission prerequisite")
            .next_action_live_aql_proof_kind,
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding.live_aql_proof_kind()
    );
    assert!(launch_submission_prerequisites
        .receipt_text()
        .contains("prerequisites.3.next_action_live_aql_proof_kind=batch_reservation_plan"));
    assert!(launch_submission_prerequisites
        .receipt_text()
        .contains("prerequisites.9.next_action_live_aql_proof_kind=materialized_packet_plan"));
    assert_eq!(
        launch_submission_prerequisites.receipt_lines()[0],
        "receipt.kind=model_runtime_launch_submission_prerequisite_plan"
    );
    assert!(launch_submission_prerequisites
        .receipt_text()
        .ends_with('\n'));
    assert_eq!(
        launch_submission_prerequisites.receipt_fingerprint(),
        "2d8c17a1f9d39fc405b7103b24a8ed7c8aad6fbaa77759b468050792b045349f"
    );
    assert_eq!(
        launch_submission_prerequisites.receipt_text(),
        include_str!("../expected-runtime-submission-prerequisite-plan.receipt")
    );

    let live_aql_proof_validations = external_cpu_demo_live_aql_proof_validations()?;
    let explicit_validation_application_plan = launch_execution_requests
        .live_aql_proof_validation_application_plan(&live_aql_proof_validations)?;
    explicit_validation_application_plan.assert_consistent()?;
    let validation_application_plan = report
        .synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(
            "external_plugin",
            &live_aql_proof_validations,
        )?;
    assert_eq!(
        validation_application_plan,
        explicit_validation_application_plan
    );
    validation_application_plan.assert_consistent()?;
    assert!(validation_application_plan.is_non_submitting_boundary());
    validation_application_plan.assert_non_submitting_boundary()?;
    assert!(validation_application_plan.all_validations_applied);
    assert!(validation_application_plan.no_live_aql_submission_side_effects);
    assert!(validation_application_plan.no_live_queue_mutation);
    let launch_submission_prerequisites_with_validations = launch_execution_requests
        .submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(
            &validation_application_plan,
        )?;
    launch_submission_prerequisites_with_validations.assert_consistent()?;
    assert_eq!(
        launch_submission_prerequisites_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    let explicit_runtime_component_applications = launch_submission_prerequisites_with_validations
        .runtime_request_component_application_plan()?;
    explicit_runtime_component_applications.assert_consistent()?;
    let runtime_component_applications = report
        .synthetic_cpu_runtime_launch_runtime_request_component_application_plan(
            "external_plugin",
            &live_aql_proof_validations,
        )?;
    assert_eq!(
        runtime_component_applications,
        explicit_runtime_component_applications
    );
    runtime_component_applications.assert_consistent()?;
    assert!(runtime_component_applications.is_non_submitting_boundary());
    runtime_component_applications.assert_non_submitting_boundary()?;
    assert_eq!(runtime_component_applications.application_count, 10);
    assert_eq!(
        runtime_component_applications.live_aql_proof_application_count,
        2
    );
    assert_eq!(
        runtime_component_applications.live_aql_proof_validation_pending_count,
        0
    );
    let runtime_component_application_receipts = runtime_component_applications
        .applications
        .iter()
        .map(
            |application| RuntimeLaunchRuntimeRequestComponentApplicationReceipt {
                target: runtime_component_applications.target,
                code_object_target: runtime_component_applications.code_object_target.clone(),
                code_object_sha256: runtime_component_applications.code_object_sha256.clone(),
                dispatch_count: runtime_component_applications.dispatch_count,
                window_count: runtime_component_applications.window_count,
                step: application.step,
                step_index: application.step_index,
                request_plan: application.request_plan,
                requirement: application.requirement,
                receipt_source: "external_plugin_cpu_receipt",
                requested_count: application.source_next_action_pending_count,
                applied_count: application.source_next_action_pending_count,
                pending_count: 0,
                live_aql_submits_work: false,
                mutates_live_queue: false,
            },
        )
        .collect::<Vec<_>>();
    for receipt in &runtime_component_application_receipts {
        receipt.assert_consistent()?;
        assert!(receipt.is_non_submitting_boundary());
        receipt.assert_non_submitting_boundary()?;
    }
    let explicit_runtime_component_application_receipt_plan = runtime_component_applications
        .application_receipt_plan(&runtime_component_application_receipts)?;
    explicit_runtime_component_application_receipt_plan.assert_consistent()?;
    let runtime_component_application_receipt_plan = report
        .synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
        )?;
    assert_eq!(
        runtime_component_application_receipt_plan,
        explicit_runtime_component_application_receipt_plan
    );
    runtime_component_application_receipt_plan.assert_consistent()?;
    assert!(runtime_component_application_receipt_plan.is_non_submitting_boundary());
    runtime_component_application_receipt_plan.assert_non_submitting_boundary()?;
    assert!(runtime_component_application_receipt_plan.all_applications_applied);
    assert!(runtime_component_application_receipt_plan.no_live_aql_submission_side_effects);
    assert!(runtime_component_application_receipt_plan.no_live_queue_mutation);
    let launch_submission_prerequisites_after_runtime_component_receipts =
        launch_submission_prerequisites_with_validations
            .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
                &runtime_component_application_receipt_plan,
            )?;
    launch_submission_prerequisites_after_runtime_component_receipts.assert_consistent()?;
    assert_eq!(
        launch_submission_prerequisites_after_runtime_component_receipts
            .pending_component_request_count,
        0
    );
    let explicit_execution_readiness_resolutions =
        launch_submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_blocker_resolution_plan()?;
    explicit_execution_readiness_resolutions.assert_consistent()?;
    let execution_readiness_resolutions = report
        .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
        )?;
    assert_eq!(
        execution_readiness_resolutions,
        explicit_execution_readiness_resolutions
    );
    execution_readiness_resolutions.assert_consistent()?;
    assert!(execution_readiness_resolutions.is_non_submitting_boundary());
    execution_readiness_resolutions.assert_non_submitting_boundary()?;
    assert_eq!(execution_readiness_resolutions.blocker_resolution_count, 9);
    assert_eq!(
        execution_readiness_resolutions.source_prerequisite_count,
        10
    );
    let execution_readiness_blocker_resolution_receipts = execution_readiness_resolutions
        .resolutions
        .iter()
        .map(
            |resolution| RuntimeLaunchExecutionReadinessBlockerResolutionReceipt {
                target: execution_readiness_resolutions.target,
                code_object_target: execution_readiness_resolutions.code_object_target.clone(),
                code_object_sha256: execution_readiness_resolutions.code_object_sha256.clone(),
                dispatch_count: execution_readiness_resolutions.dispatch_count,
                window_count: execution_readiness_resolutions.window_count,
                requirement: resolution.requirement,
                receipt_source: "external_plugin_cpu_receipt",
                source_prerequisite_count: resolution.source_prerequisite_count,
                resolved_count: resolution.source_prerequisite_count,
                pending_count: 0,
                live_aql_submits_work: false,
                mutates_live_queue: false,
            },
        )
        .collect::<Vec<_>>();
    for receipt in &execution_readiness_blocker_resolution_receipts {
        receipt.assert_consistent()?;
        assert!(receipt.is_non_submitting_boundary());
        receipt.assert_non_submitting_boundary()?;
    }
    let explicit_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_resolutions
            .resolution_receipt_plan(&execution_readiness_blocker_resolution_receipts)?;
    explicit_execution_readiness_blocker_resolution_receipt_plan.assert_consistent()?;
    let execution_readiness_blocker_resolution_receipt_plan = report
        .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )?;
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan,
        explicit_execution_readiness_blocker_resolution_receipt_plan
    );
    execution_readiness_blocker_resolution_receipt_plan.assert_consistent()?;
    assert!(execution_readiness_blocker_resolution_receipt_plan.is_non_submitting_boundary());
    execution_readiness_blocker_resolution_receipt_plan.assert_non_submitting_boundary()?;
    assert!(execution_readiness_blocker_resolution_receipt_plan.all_resolutions_applied);
    assert!(
        execution_readiness_blocker_resolution_receipt_plan.no_live_aql_submission_side_effects
    );
    assert!(execution_readiness_blocker_resolution_receipt_plan.no_live_queue_mutation);
    let explicit_launch_submission_prerequisites_after_execution_readiness_receipts =
        launch_submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &execution_readiness_blocker_resolution_receipt_plan,
            )?;
    explicit_launch_submission_prerequisites_after_execution_readiness_receipts
        .assert_consistent()?;
    let launch_submission_prerequisites_after_execution_readiness_receipts = report
        .synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )?;
    assert_eq!(
        launch_submission_prerequisites_after_execution_readiness_receipts,
        explicit_launch_submission_prerequisites_after_execution_readiness_receipts
    );
    launch_submission_prerequisites_after_execution_readiness_receipts.assert_consistent()?;
    assert!(launch_submission_prerequisites_after_execution_readiness_receipts.submission_ready);
    assert!(
        launch_submission_prerequisites_after_execution_readiness_receipts
            .is_non_submitting_boundary()
    );
    launch_submission_prerequisites_after_execution_readiness_receipts
        .assert_non_submitting_boundary()?;
    let launch_resolved_submission_prerequisites = report
        .synthetic_cpu_resolved_submission_prerequisite_plan(
            "external_plugin",
            &live_aql_proof_validations,
            "external_plugin_cpu_receipt",
        )?;
    let explicit_launch_resolved_submission_prerequisites = launch_execution_requests
        .synthetic_cpu_resolved_submission_prerequisite_plan(
            &live_aql_proof_validations,
            "external_plugin_cpu_receipt",
        )?;
    assert_eq!(
        launch_resolved_submission_prerequisites,
        explicit_launch_resolved_submission_prerequisites
    );
    assert_eq!(
        launch_resolved_submission_prerequisites,
        launch_submission_prerequisites_after_execution_readiness_receipts
    );
    launch_resolved_submission_prerequisites.assert_consistent()?;
    assert_eq!(
        launch_resolved_submission_prerequisites.prerequisite_count,
        launch_submission_prerequisites.prerequisite_count
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.prerequisite_count,
        10
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
    assert_eq!(
        launch_resolved_submission_prerequisites.receipt_lines()[0],
        "receipt.kind=model_runtime_launch_submission_prerequisite_plan"
    );
    assert!(launch_resolved_submission_prerequisites
        .receipt_text()
        .ends_with('\n'));
    let launch_resolved_submission_prerequisite_receipt_fingerprint =
        launch_resolved_submission_prerequisites.receipt_fingerprint();
    assert_eq!(
        launch_resolved_submission_prerequisite_receipt_fingerprint.len(),
        64
    );
    assert!(launch_resolved_submission_prerequisite_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    assert_eq!(
        launch_resolved_submission_prerequisite_receipt_fingerprint,
        "a68672deb3971a83998b67a27c03fed639525ad50c746475337abf26a162e0e1"
    );
    let mut stale_launch_resolved_submission_prerequisites =
        launch_resolved_submission_prerequisites.clone();
    stale_launch_resolved_submission_prerequisites.prerequisites[0].pending_count += 1;
    assert_ne!(
        stale_launch_resolved_submission_prerequisites.receipt_fingerprint(),
        launch_resolved_submission_prerequisite_receipt_fingerprint
    );
    assert_eq!(
        launch_resolved_submission_prerequisites.receipt_text(),
        include_str!("../expected-runtime-resolved-submission-prerequisite-plan.receipt")
    );
    let launch_resolved_submission_gate =
        launch_resolved_submission_prerequisites.submission_gate()?;
    let report_launch_submission_gate_after_execution_readiness_receipts = report
        .synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )?;
    assert_eq!(
        report_launch_submission_gate_after_execution_readiness_receipts,
        launch_resolved_submission_gate
    );
    launch_resolved_submission_gate.assert_consistent()?;
    let explicit_launch_submission_blocker_report_after_execution_readiness_receipts =
        launch_resolved_submission_gate.blocker_report()?;
    explicit_launch_submission_blocker_report_after_execution_readiness_receipts
        .assert_consistent()?;
    let report_launch_submission_blocker_report_after_execution_readiness_receipts = report
        .synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )?;
    assert_eq!(
        report_launch_submission_blocker_report_after_execution_readiness_receipts,
        explicit_launch_submission_blocker_report_after_execution_readiness_receipts
    );
    assert!(
        report_launch_submission_blocker_report_after_execution_readiness_receipts.submission_ready
    );
    assert!(
        report_launch_submission_blocker_report_after_execution_readiness_receipts
            .is_non_submitting_boundary()
    );
    report_launch_submission_blocker_report_after_execution_readiness_receipts
        .assert_non_submitting_boundary()?;
    assert_eq!(
        launch_submission_prerequisites_after_execution_readiness_receipts.submission_gate()?,
        launch_resolved_submission_gate
    );
    let report_launch_resolved_submission_gate = report.synthetic_cpu_resolved_submission_gate(
        "external_plugin",
        &live_aql_proof_validations,
        "external_plugin_cpu_receipt",
    )?;
    assert_eq!(
        report_launch_resolved_submission_gate,
        launch_resolved_submission_gate
    );
    let explicit_launch_resolved_submission_gate = launch_execution_requests
        .synthetic_cpu_resolved_submission_gate(
            &live_aql_proof_validations,
            "external_plugin_cpu_receipt",
        )?;
    assert_eq!(
        explicit_launch_resolved_submission_gate,
        launch_resolved_submission_gate
    );
    let report_launch_resolved_submission_blocker_report = report
        .synthetic_cpu_resolved_submission_blocker_report(
            "external_plugin",
            &live_aql_proof_validations,
            "external_plugin_cpu_receipt",
        )?;
    let explicit_launch_resolved_submission_blocker_report = launch_execution_requests
        .synthetic_cpu_resolved_submission_blocker_report(
            &live_aql_proof_validations,
            "external_plugin_cpu_receipt",
        )?;
    assert_eq!(
        report_launch_resolved_submission_blocker_report,
        explicit_launch_resolved_submission_blocker_report
    );
    assert_eq!(
        report_launch_resolved_submission_blocker_report,
        report_launch_submission_blocker_report_after_execution_readiness_receipts
    );
    report_launch_resolved_submission_blocker_report.assert_consistent()?;
    assert!(report_launch_resolved_submission_blocker_report.submission_ready);
    assert!(report_launch_resolved_submission_blocker_report.is_non_submitting_boundary());
    report_launch_resolved_submission_blocker_report.assert_non_submitting_boundary()?;
    assert_eq!(
        report_launch_resolved_submission_blocker_report.blocker_count,
        0
    );
    assert_eq!(
        report_launch_resolved_submission_blocker_report.receipt_lines()[0],
        "receipt.kind=model_runtime_launch_submission_blocker_report"
    );
    assert!(report_launch_resolved_submission_blocker_report
        .receipt_text()
        .ends_with('\n'));
    let report_launch_resolved_submission_blocker_receipt_fingerprint =
        report_launch_resolved_submission_blocker_report.receipt_fingerprint();
    assert_eq!(
        report_launch_resolved_submission_blocker_receipt_fingerprint.len(),
        64
    );
    assert!(
        report_launch_resolved_submission_blocker_receipt_fingerprint
            .chars()
            .all(|ch| ch.is_ascii_hexdigit())
    );
    assert_eq!(
        report_launch_resolved_submission_blocker_receipt_fingerprint,
        "953aaf5628d20a0e9de36aa908a614d6bfc4097d753d0e4a58b7932735cefd28"
    );
    let mut stale_report_launch_resolved_submission_blocker_report =
        report_launch_resolved_submission_blocker_report.clone();
    stale_report_launch_resolved_submission_blocker_report.total_pending_count += 1;
    assert_ne!(
        stale_report_launch_resolved_submission_blocker_report.receipt_fingerprint(),
        report_launch_resolved_submission_blocker_receipt_fingerprint
    );
    assert_eq!(
        report_launch_resolved_submission_blocker_report.receipt_text(),
        include_str!("../expected-runtime-resolved-submission-blocker-report.receipt")
    );
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
    assert!(launch_resolved_submission_gate.is_non_submitting_boundary());
    launch_resolved_submission_gate.assert_non_submitting_boundary()?;
    assert_eq!(launch_resolved_submission_gate.execution_blocker_count, 0);
    assert_eq!(launch_resolved_submission_gate.submission_blocker_count, 0);
    assert!(launch_resolved_submission_gate.blockers.is_empty());
    assert!(launch_resolved_submission_gate.submission_ready);
    assert_eq!(
        launch_resolved_submission_gate.receipt_lines()[0],
        "receipt.kind=model_runtime_launch_submission_gate"
    );
    assert!(launch_resolved_submission_gate
        .receipt_text()
        .ends_with('\n'));
    let launch_resolved_submission_gate_receipt_fingerprint =
        launch_resolved_submission_gate.receipt_fingerprint();
    assert_eq!(
        launch_resolved_submission_gate_receipt_fingerprint.len(),
        64
    );
    assert!(launch_resolved_submission_gate_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    assert_eq!(
        launch_resolved_submission_gate_receipt_fingerprint,
        "cf91c2b08af3072797c15656f23acf7119f37d5e5207599b946d0cfb63f1b9af"
    );
    let mut stale_launch_resolved_submission_gate = launch_resolved_submission_gate.clone();
    stale_launch_resolved_submission_gate.component_pending_count += 1;
    assert_ne!(
        stale_launch_resolved_submission_gate.receipt_fingerprint(),
        launch_resolved_submission_gate_receipt_fingerprint
    );
    assert_eq!(
        launch_resolved_submission_gate.receipt_text(),
        include_str!("../expected-runtime-resolved-submission-gate.receipt")
    );

    let static_handoff = report.synthetic_cpu_static_handoff_receipt("external_plugin")?;
    let explicit_static_handoff = report.static_handoff_receipt(
        "external_plugin",
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    assert_eq!(static_handoff, explicit_static_handoff);
    let mut stale_static_handoff_compatibility_report = report.clone();
    stale_static_handoff_compatibility_report
        .compatibility
        .accepted = false;
    let stale_static_handoff_compatibility_err = stale_static_handoff_compatibility_report
        .static_handoff_receipt(
            "external_plugin",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string();
    assert!(stale_static_handoff_compatibility_err.contains("not static handoff ready"));
    assert!(stale_static_handoff_compatibility_err
        .contains("compatibility report does not match manifest-derived"));
    let mut stale_static_handoff_readiness_report = report.clone();
    stale_static_handoff_readiness_report
        .readiness
        .checkpoint
        .missing_weight_tensors
        .push("unbound_static_handoff_receipt_weight".into());
    let stale_static_handoff_readiness_err = stale_static_handoff_readiness_report
        .static_handoff_receipt(
            "external_plugin",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string();
    assert!(stale_static_handoff_readiness_err.contains("not static handoff ready"));
    assert!(
        stale_static_handoff_readiness_err.contains("manifest does not match readiness-derived")
    );
    static_handoff.assert_consistent()?;
    static_handoff.assert_consistent_with(
        &report,
        "external_plugin",
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    assert_eq!(static_handoff.model_name, EXTERNAL_PLUGIN_MODEL_NAME);
    assert_eq!(
        static_handoff.manifest_fingerprint,
        manifest.contract_fingerprint
    );
    assert_eq!(
        static_handoff.summary_receipt_fingerprint,
        summary.receipt_fingerprint()
    );
    assert!(static_handoff.accepted);
    assert!(static_handoff.static_ready);
    assert_eq!(static_handoff.compatibility_issue_count, 0);
    assert_eq!(static_handoff.model_primitive_kind_count, 6);
    assert_eq!(static_handoff.model_stage_kind_count, 4);
    assert_eq!(static_handoff.tensor_count, 14);
    assert_eq!(static_handoff.op_count, 6);
    assert_eq!(static_handoff.runtime_slot_count, 14);
    assert_eq!(static_handoff.runtime_dispatch_count, 6);
    assert!(static_handoff.metadata_binding_complete);
    assert!(static_handoff.device_pointer_binding_complete);
    assert!(static_handoff.metadata_admitted);
    assert_eq!(
        static_handoff.launch_projection_kernel_candidate_count,
        launch_projection.kernel_candidate_count
    );
    assert_eq!(
        static_handoff.launch_projection_ready_candidate_count,
        launch_projection.projection_ready_candidate_count
    );
    assert_eq!(
        static_handoff.launch_projection_dispatches_with_ready_candidate_count,
        launch_projection.dispatches_with_projection_ready_candidate_count
    );
    assert_eq!(
        static_handoff.launch_projection_dispatches_without_ready_candidate_count,
        launch_projection.dispatches_without_projection_ready_candidate_count
    );
    assert!(!static_handoff.launch_projection_ready);
    assert_eq!(static_handoff.projection_selection_request_count, 2);
    assert_eq!(static_handoff.projection_selection_missing_request_count, 4);
    assert!(static_handoff.projection_selection_request_plan_ready);
    assert!(!static_handoff.projection_selection_all_requests_ready);
    assert!(!static_handoff.launch_execution_executable);
    assert_eq!(static_handoff.launch_execution_blocker_count, 9);
    assert_eq!(static_handoff.unresolved_runtime_requirement_count, 9);
    let expected_static_handoff_requirements = vec![
        "kernel_candidate_selection_policy",
        "host_launcher_runtime_branch_resolution",
        "loaded_code_object_base",
        "kernarg_allocation",
        "kernel_argument_abi_verification",
        "kernel_argument_abi_semantic_projection",
        "completion_signal_binding",
        "queue_reservation",
        "aql_packet_materialization",
    ];
    assert_eq!(
        static_handoff.unresolved_runtime_requirements,
        expected_static_handoff_requirements
    );
    assert_eq!(
        static_handoff.unresolved_runtime_requirement_names(),
        expected_static_handoff_requirements
    );
    assert!(static_handoff.has_unresolved_runtime_requirement("queue_reservation"));
    assert!(!static_handoff.has_unresolved_runtime_requirement("runtime_request_components"));
    assert_eq!(
        static_handoff
            .unresolved_runtime_requirement_names()
            .join(","),
        "kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization"
    );
    assert_eq!(
        static_handoff.aql_packet_materialization_dispatchable_packet_count,
        0
    );
    assert_eq!(static_handoff.live_aql_submitting_surface_count, 0);
    assert_eq!(static_handoff.live_queue_mutating_component_count, 0);
    assert!(!static_handoff.live_execution_supported);
    assert!(!static_handoff.gpu_buffers_allocated);
    assert!(!static_handoff.kernels_submitted);
    assert!(static_handoff.is_non_executing_boundary());
    static_handoff.assert_non_executing_boundary()?;
    let mut stale_count_static_handoff = static_handoff.clone();
    stale_count_static_handoff.unresolved_runtime_requirement_count = 0;
    assert!(!stale_count_static_handoff.is_non_executing_boundary());
    let stale_count_static_handoff_err = stale_count_static_handoff
        .assert_non_executing_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_static_handoff_err.contains("consistency"));
    assert!(stale_count_static_handoff_err.contains("unresolved runtime requirement"));
    let assert_non_execution_rejected = |receipt: ModelPluginStaticHandoffReceipt, needle: &str| {
        assert!(!receipt.is_non_executing_boundary());
        assert!(receipt
            .assert_non_executing_boundary()
            .unwrap_err()
            .to_string()
            .contains(needle));
    };
    let mut executable_static_handoff = static_handoff.clone();
    executable_static_handoff.launch_execution_executable = true;
    executable_static_handoff.launch_execution_blocker_count = 0;
    executable_static_handoff.unresolved_runtime_requirement_count = 0;
    assert_non_execution_rejected(
        executable_static_handoff.clone(),
        "launch execution executable is true",
    );
    assert!(executable_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("launch execution executable requires live execution support"));
    let mut dispatchable_static_handoff = static_handoff.clone();
    dispatchable_static_handoff.aql_packet_materialization_dispatchable_packet_count = 1;
    assert_non_execution_rejected(
        dispatchable_static_handoff.clone(),
        "dispatchable AQL packet count 1 != 0",
    );
    assert!(dispatchable_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("dispatchable AQL packet count 1 requires live execution support"));
    let mut submitting_static_handoff = static_handoff.clone();
    submitting_static_handoff.live_aql_submitting_surface_count = 1;
    assert_non_execution_rejected(
        submitting_static_handoff,
        "live AQL submitting surfaces 1 != 0",
    );
    let mut queue_mutating_static_handoff = static_handoff.clone();
    queue_mutating_static_handoff.live_queue_mutating_component_count = 1;
    assert_non_execution_rejected(
        queue_mutating_static_handoff,
        "live queue mutating components 1 != 0",
    );
    let mut live_supported_static_handoff = static_handoff.clone();
    live_supported_static_handoff.live_execution_supported = true;
    assert_non_execution_rejected(
        live_supported_static_handoff,
        "live execution support is true",
    );
    let mut allocated_static_handoff = static_handoff.clone();
    allocated_static_handoff.gpu_buffers_allocated = true;
    assert_non_execution_rejected(
        allocated_static_handoff.clone(),
        "GPU buffers allocated is true",
    );
    assert!(allocated_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("static handoff receipt claims GPU buffers were allocated"));
    let mut submitted_static_handoff = static_handoff.clone();
    submitted_static_handoff.kernels_submitted = true;
    assert_non_execution_rejected(
        submitted_static_handoff.clone(),
        "kernels submitted is true",
    );
    assert!(submitted_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("static handoff receipt claims kernels were submitted"));
    let mut requirement_count_mismatch_static_handoff = static_handoff.clone();
    requirement_count_mismatch_static_handoff.unresolved_runtime_requirement_count -= 1;
    assert!(requirement_count_mismatch_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("unresolved runtime requirement labels 9 != count 8"));
    let mut empty_requirement_static_handoff = static_handoff.clone();
    empty_requirement_static_handoff.unresolved_runtime_requirements[0] = String::new();
    assert!(empty_requirement_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("unresolved runtime requirement label is empty"));
    let mut whitespace_requirement_static_handoff = static_handoff.clone();
    whitespace_requirement_static_handoff.unresolved_runtime_requirements[0] =
        "runtime requirement".to_string();
    assert!(whitespace_requirement_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("contains whitespace"));
    let mut duplicate_requirement_static_handoff = static_handoff.clone();
    duplicate_requirement_static_handoff.unresolved_runtime_requirements[1] =
        duplicate_requirement_static_handoff.unresolved_runtime_requirements[0].clone();
    assert!(duplicate_requirement_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("appears more than once"));
    let mut stale_requirement_static_handoff = static_handoff.clone();
    stale_requirement_static_handoff.unresolved_runtime_requirements[0] =
        "stale_runtime_requirement".to_string();
    assert_ne!(
        stale_requirement_static_handoff.receipt_fingerprint(),
        static_handoff.receipt_fingerprint()
    );
    assert!(stale_requirement_static_handoff
        .assert_consistent_with(
            &report,
            "external_plugin",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string()
        .contains("unresolved runtime requirements"));
    assert_eq!(
        static_handoff.receipt_lines()[0],
        "receipt.kind=model_plugin_static_handoff"
    );
    assert!(static_handoff.receipt_text().ends_with('\n'));
    assert_eq!(
        static_handoff.receipt_fingerprint(),
        "30406e093c921f1929316780299682a10be718e4253b8cdee1fda32d27e19761"
    );
    assert_eq!(
        static_handoff.receipt_text(),
        include_str!("../expected-static-handoff.receipt")
    );

    Ok(())
}

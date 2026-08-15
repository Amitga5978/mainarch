use anyhow::Result;
use mainarch_core::model_api::prelude::*;
use mainarch_model_api_plugin_example::{
    external_cpu_demo_live_aql_proof_validations, ExternalMiniMoe, EXTERNAL_PLUGIN_PACKAGE,
};

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let emit_model_api_contract_receipt =
        args.iter().any(|arg| arg == "--model-api-contract-receipt");
    let emit_plugin_manifest_receipt = args.iter().any(|arg| arg == "--plugin-manifest-receipt");
    let emit_plugin_compatibility_receipt = args
        .iter()
        .any(|arg| arg == "--plugin-compatibility-receipt");
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
    let emit_static_handoff_receipt = args.iter().any(|arg| arg == "--static-handoff-receipt");
    if emit_model_api_contract_receipt {
        print!("{}", MODEL_API_CONTRACT.receipt_text());
        return Ok(());
    }
    if !emit_plugin_manifest_receipt
        && !emit_plugin_compatibility_receipt
        && !emit_runtime_launch_request_receipt
        && !emit_runtime_submission_gate_receipt
        && !emit_runtime_resolved_submission_gate_receipt
        && !emit_runtime_resolved_submission_prerequisite_plan_receipt
        && !emit_runtime_resolved_submission_blocker_report_receipt
        && !emit_runtime_submission_blocker_report_receipt
        && !emit_runtime_submission_prerequisite_plan_receipt
        && !emit_static_handoff_receipt
    {
        println!("mainarch-model-api-plugin-example - standalone external CPU-only smoke");
        println!(
            "model_api_contract: {} receipt_fingerprint={} lines={}",
            MODEL_API_CONTRACT,
            MODEL_API_CONTRACT.receipt_fingerprint(),
            MODEL_API_CONTRACT.receipt_lines().len()
        );
    }

    let model = ExternalMiniMoe::default();
    let catalog = MainarchPrimitiveLoweringCatalog::mi355_reference();
    let report = inspect_model_plugin(&model, &catalog)?;
    report.assert_consistent()?;
    report.assert_accepted()?;
    report.readiness.assert_static_runtime_ready()?;
    report.rejection_report().assert_no_rejection()?;
    let summary = report.summary();
    summary.assert_consistent_with(&report)?;
    let plugin_manifest = &report.manifest;
    plugin_manifest.assert_static_metadata_ready()?;
    plugin_manifest.assert_compatible_with(&catalog)?;
    let plugin_compatibility = &report.compatibility;
    plugin_compatibility.assert_accepted()?;
    if emit_plugin_manifest_receipt {
        print!("{}", plugin_manifest.receipt_text());
        return Ok(());
    }
    if emit_plugin_compatibility_receipt {
        print!("{}", plugin_compatibility.receipt_text());
        return Ok(());
    }
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
        plugin_manifest.runtime_launch_request_step_count,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len()
    );
    assert_eq!(
        plugin_manifest.runtime_launch_request_steps.as_slice(),
        &RuntimeLaunchExecutionRequestStep::DESCRIPTORS[..]
    );
    assert_eq!(
        plugin_manifest
            .runtime_launch_request_step_labels
            .as_slice(),
        static_launch_request_step_plans.as_slice()
    );
    for descriptor in RuntimeLaunchExecutionRequestStep::DESCRIPTORS {
        assert_eq!(
            plugin_manifest.launch_request_step_for(descriptor.step),
            Some(&descriptor)
        );
        assert_eq!(
            plugin_manifest.launch_request_step_for_request_plan(descriptor.request_plan),
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
        plugin_manifest.launch_request_step_for_request_plan("missing_request_plan"),
        None
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::descriptor_for_request_plan("missing_request_plan"),
        None
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
    let admission = report
        .readiness
        .validate_metadata_runtime_admission(&slot_bindings);
    admission.assert_admitted()?;
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
    let launch_projection_gaps =
        launch_projection.kernel_argument_abi_semantic_projection_gap_report()?;
    launch_projection_gaps.assert_consistent()?;
    let projection_recommendations = launch_projection
        .kernel_argument_abi_semantic_projection_candidate_recommendation_plan()?;
    projection_recommendations.assert_consistent()?;
    let projection_selection = projection_recommendations
        .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
            &launch_projection,
        )?;
    projection_selection.assert_consistent()?;
    projection_selection.assert_consistent_with_projection(&launch_projection)?;
    let kernel_selection = report
        .readiness
        .runtime_launch_kernel_candidate_selection_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?;
    kernel_selection.assert_consistent()?;
    let host_launcher_branch_requests = report
        .readiness
        .runtime_launch_host_launcher_branch_resolution_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?;
    host_launcher_branch_requests.assert_consistent()?;
    let launch_execution = report.readiness.runtime_launch_execution_readiness_report(
        &slot_bindings,
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        &device_pointer_validation,
    )?;
    launch_execution.assert_consistent()?;
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
    assert_eq!(
        launch_execution_requests.live_aql_proof_surface_request_plan_names(),
        static_live_aql_proof_step_plans
    );
    assert_eq!(
        launch_execution_requests.live_aql_proof_input_labels(),
        static_live_aql_proof_inputs
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
        assert_eq!(Some(surface.proof_input), descriptor.live_aql_proof_input);
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
    if emit_runtime_launch_request_receipt {
        print!("{}", launch_execution_requests.receipt_text());
        return Ok(());
    }
    let explicit_launch_submission_gate = launch_execution_requests.submission_gate()?;
    explicit_launch_submission_gate.assert_consistent()?;
    let launch_submission_gate =
        report.synthetic_cpu_runtime_launch_submission_gate("external_plugin")?;
    assert_eq!(launch_submission_gate, explicit_launch_submission_gate);
    assert_eq!(
        launch_submission_gate
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .expect("queue reservation submission gate blocker")
            .requirement,
        RuntimeLaunchExecutionRequestStep::QueueReservation.requirement()
    );
    if emit_runtime_submission_gate_receipt {
        print!("{}", launch_submission_gate.receipt_text());
        return Ok(());
    }
    let explicit_launch_submission_blockers = launch_submission_gate.blocker_report()?;
    explicit_launch_submission_blockers.assert_consistent()?;
    let launch_submission_blockers =
        report.synthetic_cpu_runtime_launch_submission_blocker_report("external_plugin")?;
    assert_eq!(
        launch_submission_blockers,
        explicit_launch_submission_blockers
    );
    assert!(
        launch_submission_blockers
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .expect("AQL live relocation submission blocker")
            .execution_readiness_blocker
    );
    if emit_runtime_submission_blocker_report_receipt {
        print!("{}", launch_submission_blockers.receipt_text());
        return Ok(());
    }
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
        launch_submission_prerequisites.next_action_request_plan_names(),
        static_launch_request_step_plans
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
    if emit_runtime_submission_prerequisite_plan_receipt {
        print!("{}", launch_submission_prerequisites.receipt_text());
        return Ok(());
    }
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
        runtime_component_applications.live_aql_proof_validation_pending_count,
        0
    );
    assert!(!runtime_component_applications.submission_ready);
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
    assert!(!execution_readiness_resolutions.submission_ready);
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
    let explicit_launch_submission_gate_after_execution_readiness_receipts =
        launch_submission_prerequisites_after_runtime_component_receipts
            .submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
                &execution_readiness_blocker_resolution_receipt_plan,
            )?;
    explicit_launch_submission_gate_after_execution_readiness_receipts.assert_consistent()?;
    let launch_submission_gate_after_execution_readiness_receipts = report
        .synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )?;
    assert_eq!(
        launch_submission_gate_after_execution_readiness_receipts,
        explicit_launch_submission_gate_after_execution_readiness_receipts
    );
    launch_submission_gate_after_execution_readiness_receipts.assert_consistent()?;
    assert!(launch_submission_gate_after_execution_readiness_receipts.submission_ready);
    let explicit_launch_submission_blocker_report_after_execution_readiness_receipts =
        launch_submission_gate_after_execution_readiness_receipts.blocker_report()?;
    explicit_launch_submission_blocker_report_after_execution_readiness_receipts
        .assert_consistent()?;
    let launch_submission_blocker_report_after_execution_readiness_receipts = report
        .synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(
            "external_plugin",
            &live_aql_proof_validations,
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )?;
    assert_eq!(
        launch_submission_blocker_report_after_execution_readiness_receipts,
        explicit_launch_submission_blocker_report_after_execution_readiness_receipts
    );
    launch_submission_blocker_report_after_execution_readiness_receipts.assert_consistent()?;
    assert!(launch_submission_blocker_report_after_execution_readiness_receipts.submission_ready);
    assert_eq!(
        launch_submission_blocker_report_after_execution_readiness_receipts.blocker_count,
        0
    );
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
    let report_launch_resolved_submission_gate = report.synthetic_cpu_resolved_submission_gate(
        "external_plugin",
        &live_aql_proof_validations,
        "external_plugin_cpu_receipt",
    )?;
    let explicit_launch_resolved_submission_gate = launch_execution_requests
        .synthetic_cpu_resolved_submission_gate(
            &live_aql_proof_validations,
            "external_plugin_cpu_receipt",
        )?;
    assert_eq!(
        launch_resolved_submission_gate,
        report_launch_resolved_submission_gate
    );
    assert_eq!(
        launch_resolved_submission_gate,
        explicit_launch_resolved_submission_gate
    );
    launch_resolved_submission_gate.assert_consistent()?;
    let launch_resolved_submission_blocker_report = report
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
        launch_resolved_submission_blocker_report,
        explicit_launch_resolved_submission_blocker_report
    );
    assert_eq!(
        launch_resolved_submission_blocker_report,
        launch_submission_blocker_report_after_execution_readiness_receipts
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
    if emit_runtime_resolved_submission_gate_receipt {
        print!("{}", launch_resolved_submission_gate.receipt_text());
        return Ok(());
    }
    let static_handoff = report.synthetic_cpu_static_handoff_receipt("external_plugin")?;
    static_handoff.assert_consistent_with(
        &report,
        "external_plugin",
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;

    if emit_static_handoff_receipt {
        print!("{}", static_handoff.receipt_text());
        return Ok(());
    }
    let checkpoint_payloads = report.synthetic_cpu_runtime_checkpoint_payload_binding_plan(
        synthetic_available_checkpoint_keys(&report.readiness.checkpoint),
        "external-plugin.synthetic.safetensors",
        &[1002, 1001],
    )?;
    checkpoint_payloads.assert_checkpoint_payload_bound()?;

    println!(
        "external_plugin: model={} package={} imported_prelude=true",
        report.graph.name, EXTERNAL_PLUGIN_PACKAGE
    );
    println!(
        "plugin_summary: receipt_fingerprint={} accepted={} static_ready={} compatibility_issues={} model_primitives={} model_stages={} tensors={} ops={} dispatches={} catalog_cases={} catalog_gaps={} live_execution_supported={}",
        summary.receipt_fingerprint(),
        summary.accepted,
        summary.static_ready,
        summary.compatibility_issue_count,
        summary.model_primitive_kind_count,
        summary.model_stage_kind_count,
        summary.tensor_count,
        summary.op_count,
        summary.runtime_dispatch_count,
        summary.catalog_primitive_case_count,
        summary.catalog_gap_case_count,
        summary.live_execution_supported
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
        "external_static_launch_request_steps: count={} names={} requirements={}",
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len(),
        static_launch_request_step_plans.join(","),
        static_launch_request_step_requirements.join(",")
    );
    println!(
        "external_static_live_aql_proof_steps: count={} names={} proof_kinds={} proof_inputs={} validation_methods={}",
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS.len(),
        static_live_aql_proof_step_plans.join(","),
        static_live_aql_proof_kinds.join(","),
        static_live_aql_proof_inputs.join(","),
        static_live_aql_validation_methods.join(",")
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
        report.readiness.graph.tensors,
        report.readiness.graph.ops,
        report.readiness.graph.stages,
        report.readiness.graph.staged_ops,
        report.readiness.graph.unstaged_ops
    );
    println!(
        "checkpoint: bound_weights={} missing_weights={} checkpoint_bytes={}",
        report.readiness.checkpoint.entries.len(),
        report.readiness.checkpoint.missing_weight_tensors.len(),
        report.readiness.checkpoint.total_checkpoint_bytes
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
        "lowering: target={} native_gpu_ops={} fused_native_gpu_ops={} gap_ops={}",
        report.readiness.lowering.target,
        report.readiness.lowering.native_gpu_ops,
        report.readiness.lowering.fused_native_gpu_ops,
        report.readiness.lowering.gap_ops
    );
    println!(
        "external_launch_projection: candidates={} schema_candidates={} missing_schema_candidates={} projection_ready_candidates={} dispatches_with_ready={} dispatches_without_ready={} projected_kernarg_bytes={} ready={}",
        launch_projection.kernel_candidate_count,
        launch_projection.semantic_schema_candidate_count,
        launch_projection.missing_semantic_schema_candidate_count,
        launch_projection.projection_ready_candidate_count,
        launch_projection.dispatches_with_projection_ready_candidate_count,
        launch_projection.dispatches_without_projection_ready_candidate_count,
        launch_projection.projected_kernarg_bytes,
        launch_projection.semantic_projection_ready
    );
    let launch_projection_missing_model_arguments =
        launch_projection_gaps.missing_model_argument_names();
    println!(
        "external_launch_projection_missing_model_arguments: count={} names={}",
        launch_projection_missing_model_arguments.len(),
        launch_projection_missing_model_arguments.join(",")
    );
    println!(
        "external_projection_selection: requests={} missing={} requested_projected_kernarg_bytes={} applied={} all_ready={} plan_ready={} policy={}",
        projection_selection.selection_request_count,
        projection_selection.missing_selection_request_count,
        projection_selection.requested_projected_kernarg_bytes,
        projection_selection.selection_applied_count,
        projection_selection.all_selection_requests_ready,
        projection_selection.request_plan_ready,
        projection_selection.policy
    );
    let projection_selection_ready_ops = projection_selection.selection_request_op_names();
    println!(
        "external_projection_selection_ready_ops: count={} names={}",
        projection_selection_ready_ops.len(),
        projection_selection_ready_ops.join(",")
    );
    let projection_selection_requested_symbol_pairs =
        projection_selection.selection_request_op_kernel_symbols();
    assert_eq!(
        projection_selection_requested_symbol_pairs,
        vec![
            (
                "layers.0.router_topk".to_string(),
                "moe_router_topk".to_string()
            ),
            ("lm_head".to_string(), "gemv_f16".to_string()),
        ]
    );
    let projection_selection_requested_symbols = projection_selection_requested_symbol_pairs
        .iter()
        .map(|(op_name, kernel_symbol)| format!("{op_name}={kernel_symbol}"))
        .collect::<Vec<_>>();
    assert_eq!(
        projection_selection_requested_symbols,
        projection_selection.selection_request_op_kernel_symbol_labels()
    );
    println!(
        "external_projection_selection_requested_symbols: count={} labels={}",
        projection_selection_requested_symbols.len(),
        projection_selection_requested_symbols.join(",")
    );
    let projection_selection_missing_ops =
        projection_selection.missing_selection_request_op_names();
    println!(
        "external_projection_selection_missing_ops: count={} names={}",
        projection_selection_missing_ops.len(),
        projection_selection_missing_ops.join(",")
    );
    println!(
        "external_kernel_selection: requests={} missing={} verified_candidates={} applied={} all_ready={} plan_ready={} policy={}",
        kernel_selection.selection_request_count,
        kernel_selection.missing_selection_request_count,
        kernel_selection.verified_candidate_count,
        kernel_selection.selection_applied_count,
        kernel_selection.all_selection_requests_ready,
        kernel_selection.request_plan_ready,
        kernel_selection.policy
    );
    let kernel_selection_ready_ops = kernel_selection.selection_request_op_names();
    println!(
        "external_kernel_selection_ready_ops: count={} names={}",
        kernel_selection_ready_ops.len(),
        kernel_selection_ready_ops.join(",")
    );
    let kernel_selection_requested_symbol_pairs =
        kernel_selection.selection_request_op_kernel_symbols();
    assert_eq!(
        kernel_selection_requested_symbol_pairs,
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
    let kernel_selection_requested_symbols = kernel_selection_requested_symbol_pairs
        .iter()
        .map(|(op_name, kernel_symbol)| format!("{op_name}={kernel_symbol}"))
        .collect::<Vec<_>>();
    assert_eq!(
        kernel_selection_requested_symbols,
        kernel_selection.selection_request_op_kernel_symbol_labels()
    );
    println!(
        "external_kernel_selection_requested_symbols: count={} labels={}",
        kernel_selection_requested_symbols.len(),
        kernel_selection_requested_symbols.join(",")
    );
    let kernel_selection_missing_ops = kernel_selection.missing_selection_request_op_names();
    println!(
        "external_kernel_selection_missing_ops: count={} names={}",
        kernel_selection_missing_ops.len(),
        kernel_selection_missing_ops.join(",")
    );
    println!(
        "external_host_launcher_branch_requests: requests={} applied={} unresolved_candidates={} all_resolved={} plan_ready={}",
        host_launcher_branch_requests.branch_resolution_request_count,
        host_launcher_branch_requests.branch_resolution_applied_count,
        host_launcher_branch_requests.unresolved_candidate_symbol_count,
        host_launcher_branch_requests.all_branches_resolved,
        host_launcher_branch_requests.request_plan_ready
    );
    let host_launcher_branch_request_ops =
        host_launcher_branch_requests.branch_resolution_request_op_names();
    println!(
        "external_host_launcher_branch_request_ops: count={} names={}",
        host_launcher_branch_request_ops.len(),
        host_launcher_branch_request_ops.join(",")
    );
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
    let host_launcher_branch_candidate_symbols = host_launcher_branch_candidate_symbol_sets
        .iter()
        .map(|(op_name, candidate_symbols)| format!("{op_name}={}", candidate_symbols.join("|")))
        .collect::<Vec<_>>();
    assert_eq!(
        host_launcher_branch_candidate_symbols,
        host_launcher_branch_requests.branch_resolution_request_candidate_symbol_labels()
    );
    println!(
        "external_host_launcher_branch_candidate_symbols: count={} labels={}",
        host_launcher_branch_candidate_symbols.len(),
        host_launcher_branch_candidate_symbols.join(",")
    );
    let host_launcher_branch_unresolved_candidate_symbols =
        host_launcher_branch_requests.unresolved_candidate_symbols();
    println!(
        "external_host_launcher_branch_unresolved_candidate_symbols: count={} symbols={}",
        host_launcher_branch_unresolved_candidate_symbols.len(),
        host_launcher_branch_unresolved_candidate_symbols.join(",")
    );
    println!(
        "external_launch_execution: executable={} blockers={} unresolved_runtime_requirements={} projection_selection_requests={} projection_selection_missing={} aql_dispatchable_packets={} live_aql_submitting_surfaces={} live_queue_mutating_components={}",
        launch_execution.executable,
        launch_execution.blockers.len(),
        launch_execution.unresolved_runtime_requirements.len(),
        launch_execution
            .kernel_argument_abi_semantic_projection_candidate_selection_request_count,
        launch_execution
            .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count,
        launch_execution.aql_packet_materialization_dispatchable_packet_count,
        launch_execution_requests.live_aql_submitting_surface_count,
        launch_execution_requests.live_queue_mutating_component_count
    );
    let launch_execution_request_plans = launch_execution_requests.component_request_plan_names();
    println!(
        "external_launch_execution_request_plans: count={} names={}",
        launch_execution_request_plans.len(),
        launch_execution_request_plans.join(",")
    );
    let launch_execution_request_pending_plans =
        launch_execution_requests.pending_component_request_plan_names();
    println!(
        "external_launch_execution_request_pending_plans: count={} names={}",
        launch_execution_request_pending_plans.len(),
        launch_execution_request_pending_plans.join(",")
    );
    let launch_execution_live_aql_proof_surface_plans =
        launch_execution_requests.live_aql_proof_surface_request_plan_names();
    println!(
        "external_launch_execution_live_aql_proof_surface_plans: count={} names={}",
        launch_execution_live_aql_proof_surface_plans.len(),
        launch_execution_live_aql_proof_surface_plans.join(",")
    );
    let launch_execution_pending_live_aql_proof_surface_plans =
        launch_execution_requests.pending_live_aql_proof_surface_request_plan_names();
    println!(
        "external_launch_execution_pending_live_aql_proof_surface_plans: count={} names={}",
        launch_execution_pending_live_aql_proof_surface_plans.len(),
        launch_execution_pending_live_aql_proof_surface_plans.join(",")
    );
    let launch_execution_pending_live_aql_proof_validation_plans =
        launch_execution_requests.pending_live_aql_proof_validation_request_plan_names();
    println!(
        "external_launch_execution_pending_live_aql_proof_validation_plans: count={} names={}",
        launch_execution_pending_live_aql_proof_validation_plans.len(),
        launch_execution_pending_live_aql_proof_validation_plans.join(",")
    );
    let launch_execution_live_aql_proof_kinds =
        launch_execution_requests.live_aql_proof_kind_labels();
    println!(
        "external_launch_execution_live_aql_proof_kinds: count={} labels={}",
        launch_execution_live_aql_proof_kinds.len(),
        launch_execution_live_aql_proof_kinds.join(",")
    );
    let launch_execution_live_aql_submitting_surface_plans =
        launch_execution_requests.live_aql_submitting_surface_request_plan_names();
    println!(
        "external_launch_execution_live_aql_submitting_surface_plans: count={} names={}",
        launch_execution_live_aql_submitting_surface_plans.len(),
        launch_execution_live_aql_submitting_surface_plans.join(",")
    );
    let launch_execution_live_queue_mutating_component_plans =
        launch_execution_requests.live_queue_mutating_component_request_plan_names();
    println!(
        "external_launch_execution_live_queue_mutating_component_plans: count={} names={}",
        launch_execution_live_queue_mutating_component_plans.len(),
        launch_execution_live_queue_mutating_component_plans.join(",")
    );
    let launch_execution_live_aql_proof_inputs =
        launch_execution_requests.live_aql_proof_input_labels();
    println!(
        "external_launch_execution_live_aql_proof_inputs: count={} labels={}",
        launch_execution_live_aql_proof_inputs.len(),
        launch_execution_live_aql_proof_inputs.join(",")
    );
    let launch_execution_live_aql_validation_methods =
        launch_execution_requests.live_aql_validation_method_labels();
    println!(
        "external_launch_execution_live_aql_validation_methods: count={} labels={}",
        launch_execution_live_aql_validation_methods.len(),
        launch_execution_live_aql_validation_methods.join(",")
    );
    let launch_submission_gate_blockers = launch_submission_gate.blocker_requirement_names();
    println!(
        "external_submission_gate_blockers: count={} requirements={}",
        launch_submission_gate_blockers.len(),
        launch_submission_gate_blockers.join(",")
    );
    let launch_submission_blocker_report_blockers =
        launch_submission_blockers.blocker_requirement_names();
    println!(
        "external_submission_blocker_report_blockers: count={} requirements={}",
        launch_submission_blocker_report_blockers.len(),
        launch_submission_blocker_report_blockers.join(",")
    );
    let launch_submission_blocker_report_execution_readiness_blockers =
        launch_submission_blockers.execution_readiness_blocker_requirement_names();
    println!(
        "external_submission_blocker_report_execution_readiness_blockers: count={} requirements={}",
        launch_submission_blocker_report_execution_readiness_blockers.len(),
        launch_submission_blocker_report_execution_readiness_blockers.join(",")
    );
    let launch_submission_blocker_report_runtime_component_blockers =
        launch_submission_blockers.runtime_request_component_blocker_requirement_names();
    println!(
        "external_submission_blocker_report_runtime_component_blockers: count={} requirements={}",
        launch_submission_blocker_report_runtime_component_blockers.len(),
        launch_submission_blocker_report_runtime_component_blockers.join(",")
    );
    let launch_submission_blocker_report_live_aql_proof_validation_blockers =
        launch_submission_blockers.live_aql_proof_validation_blocker_requirement_names();
    println!(
        "external_submission_blocker_report_live_aql_proof_validation_blockers: count={} requirements={}",
        launch_submission_blocker_report_live_aql_proof_validation_blockers.len(),
        launch_submission_blocker_report_live_aql_proof_validation_blockers.join(",")
    );
    let launch_submission_blocker_report_live_aql_submission_side_effect_blockers =
        launch_submission_blockers.live_aql_submission_side_effect_blocker_requirement_names();
    println!(
        "external_submission_blocker_report_live_aql_submission_side_effect_blockers: count={} requirements={}",
        launch_submission_blocker_report_live_aql_submission_side_effect_blockers.len(),
        launch_submission_blocker_report_live_aql_submission_side_effect_blockers.join(",")
    );
    let launch_submission_blocker_report_live_queue_mutation_blockers =
        launch_submission_blockers.live_queue_mutation_blocker_requirement_names();
    println!(
        "external_submission_blocker_report_live_queue_mutation_blockers: count={} requirements={}",
        launch_submission_blocker_report_live_queue_mutation_blockers.len(),
        launch_submission_blocker_report_live_queue_mutation_blockers.join(",")
    );
    let launch_submission_prerequisite_plans =
        launch_submission_prerequisites.prerequisite_request_plan_names();
    println!(
        "external_submission_prerequisite_plans: count={} names={}",
        launch_submission_prerequisite_plans.len(),
        launch_submission_prerequisite_plans.join(",")
    );
    let launch_submission_prerequisite_unsatisfied_plans =
        launch_submission_prerequisites.unsatisfied_prerequisite_request_plan_names();
    println!(
        "external_submission_prerequisite_unsatisfied_plans: count={} names={}",
        launch_submission_prerequisite_unsatisfied_plans.len(),
        launch_submission_prerequisite_unsatisfied_plans.join(",")
    );
    let launch_submission_prerequisite_next_action_plans =
        launch_submission_prerequisites.next_action_request_plan_names();
    println!(
        "external_submission_prerequisite_next_action_plans: count={} names={}",
        launch_submission_prerequisite_next_action_plans.len(),
        launch_submission_prerequisite_next_action_plans.join(",")
    );
    let launch_submission_prerequisite_next_action_labels =
        launch_submission_prerequisites.next_action_labels();
    println!(
        "external_submission_prerequisite_next_action_labels: count={} labels={}",
        launch_submission_prerequisite_next_action_labels.len(),
        launch_submission_prerequisite_next_action_labels.join(",")
    );
    let launch_submission_prerequisite_runtime_component_next_action_plans =
        launch_submission_prerequisites.runtime_request_component_next_action_request_plan_names();
    println!(
        "external_submission_prerequisite_runtime_component_next_action_plans: count={} names={}",
        launch_submission_prerequisite_runtime_component_next_action_plans.len(),
        launch_submission_prerequisite_runtime_component_next_action_plans.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_validation_next_action_plans =
        launch_submission_prerequisites.live_aql_proof_validation_next_action_request_plan_names();
    println!(
        "external_submission_prerequisite_live_aql_proof_validation_next_action_plans: count={} names={}",
        launch_submission_prerequisite_live_aql_proof_validation_next_action_plans.len(),
        launch_submission_prerequisite_live_aql_proof_validation_next_action_plans.join(",")
    );
    let launch_submission_prerequisite_next_action_inputs =
        launch_submission_prerequisites.next_action_input_labels();
    println!(
        "external_submission_prerequisite_next_action_inputs: count={} labels={}",
        launch_submission_prerequisite_next_action_inputs.len(),
        launch_submission_prerequisite_next_action_inputs.join(",")
    );
    let launch_submission_prerequisite_next_action_live_aql_proof_kinds =
        launch_submission_prerequisites.next_action_live_aql_proof_kind_labels();
    println!(
        "external_submission_prerequisite_next_action_live_aql_proof_kinds: count={} labels={}",
        launch_submission_prerequisite_next_action_live_aql_proof_kinds.len(),
        launch_submission_prerequisite_next_action_live_aql_proof_kinds.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_plans =
        launch_submission_prerequisites.live_aql_proof_prerequisite_request_plan_names();
    println!(
        "external_submission_prerequisite_live_aql_proof_plans: count={} names={}",
        launch_submission_prerequisite_live_aql_proof_plans.len(),
        launch_submission_prerequisite_live_aql_proof_plans.join(",")
    );
    let launch_submission_prerequisite_live_aql_submitting_plans =
        launch_submission_prerequisites.live_aql_submitting_prerequisite_request_plan_names();
    println!(
        "external_submission_prerequisite_live_aql_submitting_plans: count={} names={}",
        launch_submission_prerequisite_live_aql_submitting_plans.len(),
        launch_submission_prerequisite_live_aql_submitting_plans.join(",")
    );
    let launch_submission_prerequisite_pending_live_aql_proof_validation_plans =
        launch_submission_prerequisites
            .pending_live_aql_proof_validation_prerequisite_request_plan_names();
    println!(
        "external_submission_prerequisite_pending_live_aql_proof_validation_plans: count={} names={}",
        launch_submission_prerequisite_pending_live_aql_proof_validation_plans.len(),
        launch_submission_prerequisite_pending_live_aql_proof_validation_plans.join(",")
    );
    let launch_submission_prerequisite_live_queue_mutating_plans =
        launch_submission_prerequisites.live_queue_mutating_prerequisite_request_plan_names();
    println!(
        "external_submission_prerequisite_live_queue_mutating_plans: count={} names={}",
        launch_submission_prerequisite_live_queue_mutating_plans.len(),
        launch_submission_prerequisite_live_queue_mutating_plans.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_kinds =
        launch_submission_prerequisites.live_aql_proof_kind_labels();
    println!(
        "external_submission_prerequisite_live_aql_proof_kinds: count={} labels={}",
        launch_submission_prerequisite_live_aql_proof_kinds.len(),
        launch_submission_prerequisite_live_aql_proof_kinds.join(",")
    );
    let launch_submission_prerequisite_live_aql_proof_inputs =
        launch_submission_prerequisites.live_aql_proof_input_labels();
    println!(
        "external_submission_prerequisite_live_aql_proof_inputs: count={} labels={}",
        launch_submission_prerequisite_live_aql_proof_inputs.len(),
        launch_submission_prerequisite_live_aql_proof_inputs.join(",")
    );
    let launch_submission_prerequisite_live_aql_validation_methods =
        launch_submission_prerequisites.live_aql_validation_method_labels();
    println!(
        "external_submission_prerequisite_live_aql_validation_methods: count={} labels={}",
        launch_submission_prerequisite_live_aql_validation_methods.len(),
        launch_submission_prerequisite_live_aql_validation_methods.join(",")
    );
    let launch_execution_blockers = launch_execution.blocker_requirement_names();
    println!(
        "external_launch_execution_blockers: count={} requirements={}",
        launch_execution_blockers.len(),
        launch_execution_blockers.join(",")
    );
    let launch_execution_requirements = launch_execution.unresolved_runtime_requirement_names();
    println!(
        "external_launch_execution_requirements: count={} requirements={}",
        launch_execution_requirements.len(),
        launch_execution_requirements.join(",")
    );
    let static_handoff_requirements = static_handoff.unresolved_runtime_requirement_names();
    println!(
        "external_static_handoff: receipt_fingerprint={} manifest_receipt_fingerprint={} compatibility_receipt_fingerprint={} accepted={} static_ready={} metadata_admitted={} projection_ready={} selection_requests={} selection_missing={} executable={} blockers={} requirements={} aql_dispatchable_packets={} live_aql_submitting_surfaces={} live_queue_mutating_components={} gpu_buffers_allocated={} kernels_submitted={}",
        static_handoff.receipt_fingerprint(),
        static_handoff.manifest_receipt_fingerprint,
        static_handoff.compatibility_receipt_fingerprint,
        static_handoff.accepted,
        static_handoff.static_ready,
        static_handoff.metadata_admitted,
        static_handoff.launch_projection_ready,
        static_handoff.projection_selection_request_count,
        static_handoff.projection_selection_missing_request_count,
        static_handoff.launch_execution_executable,
        static_handoff.launch_execution_blocker_count,
        static_handoff_requirements.join(","),
        static_handoff.aql_packet_materialization_dispatchable_packet_count,
        static_handoff.live_aql_submitting_surface_count,
        static_handoff.live_queue_mutating_component_count,
        static_handoff.gpu_buffers_allocated,
        static_handoff.kernels_submitted
    );
    println!(
        "external_plugin_boundary: live_execution_supported={} launch_execution_supported=false gpu_buffers_allocated=false kernels_submitted=false",
        MODEL_API_CONTRACT.live_execution_supported
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

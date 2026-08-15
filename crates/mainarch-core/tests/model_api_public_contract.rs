use anyhow::Result;
use mainarch_core::model_api::prelude::*;
use std::path::{Path, PathBuf};

const EXPECTED_RUNTIME_LAUNCH_REQUEST_STEP_COUNT: usize = 10;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("mainarch-core crate lives under crates/mainarch-core")
        .to_path_buf()
}

fn markdown_normalized(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct ExternalToyDecoder {
    vocab: usize,
    hidden: usize,
}

impl ExternalToyDecoder {
    fn new() -> Self {
        Self {
            vocab: 256,
            hidden: 128,
        }
    }
}

impl ModelDefinition for ExternalToyDecoder {
    fn name(&self) -> &str {
        "external-toy-decoder"
    }

    fn define(&self, api: &mut dyn ModelPrimitiveApi) -> Result<()> {
        api.declare_tensor(TensorSpec::new(
            "tokens",
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
            .with_checkpoint_key("external.embed_tokens.weight")?,
        )?;
        api.declare_tensor(
            TensorSpec::new(
                "lm_head",
                DType::F16,
                vec![self.vocab, self.hidden],
                TensorRole::Weight,
            )
            .with_checkpoint_key("external.lm_head.weight")?,
        )?;

        api.begin_stage("embedding", ModelStageKind::Embedding)?;
        api.emit(PrimitiveOp::EmbeddingLookup(EmbeddingLookup {
            name: "embed_tokens".to_string(),
            token_ids: "tokens".into(),
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

struct ExternalUnsupportedCollectiveModel;

impl ModelDefinition for ExternalUnsupportedCollectiveModel {
    fn name(&self) -> &str {
        "external-unsupported-collective"
    }

    fn define(&self, api: &mut dyn ModelPrimitiveApi) -> Result<()> {
        api.declare_tensor(TensorSpec::new(
            "expert_input",
            DType::F16,
            vec![16],
            TensorRole::Activation,
        ))?;
        api.declare_tensor(TensorSpec::new(
            "expert_output",
            DType::F16,
            vec![16],
            TensorRole::Activation,
        ))?;

        api.begin_stage("expert_parallel", ModelStageKind::Moe)?;
        api.emit(PrimitiveOp::Collective(Collective {
            name: "ep_all_to_all".to_string(),
            kind: CollectiveKind::AllToAll,
            input: "expert_input".into(),
            output: "expert_output".into(),
            group_size: 2,
        }))?;
        api.end_stage()?;

        Ok(())
    }
}

fn public_live_aql_batch_reservation_plan_proof() -> KfdQueueLiveAqlBatchReservationPlanProof {
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

fn public_live_aql_materialized_packet_plan_proof() -> KfdQueueLiveAqlMaterializedPacketPlanProof {
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

#[test]
fn public_docs_pin_current_model_api_contract_descriptor() -> Result<()> {
    let descriptor = MODEL_API_CONTRACT.to_string();
    for doc in ["docs/model-api.md", "docs/api-stability.md"] {
        let path = repo_root().join(doc);
        let text = std::fs::read_to_string(&path)?;
        assert!(
            markdown_normalized(&text).contains(&descriptor),
            "{doc} does not document current MODEL_API_CONTRACT descriptor {descriptor:?}"
        );
    }
    Ok(())
}

#[test]
fn external_runner_uses_typed_live_aql_proof_validation_bridge() -> Result<()> {
    let batch_proof = public_live_aql_batch_reservation_plan_proof();
    let batch_validation = RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
        .validate_batch_reservation_plan_proof(batch_proof)?;
    assert!(matches!(
        batch_validation,
        RuntimeLaunchLiveAqlProofValidation::BatchReservationPlan(_)
    ));
    assert_eq!(
        batch_validation.kind(),
        RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
    );
    assert_eq!(
        batch_validation.proof_input(),
        "KfdQueueLiveAqlBatchReservationPlanInput"
    );
    assert_eq!(
        batch_validation.proof_type(),
        "KfdQueueLiveAqlBatchReservationPlanProof"
    );
    assert_eq!(
        batch_validation.validation_type(),
        "KfdQueueLiveAqlBatchReservationPlanValidation"
    );
    assert_eq!(
        batch_validation.validation_method(),
        "KfdQueueLiveAqlBatchReservationPlanProof::validate_ready"
    );
    assert_eq!(batch_validation.validation_ready_field(), "ready");
    assert_eq!(
        batch_validation.no_live_queue_mutation_contract_field(),
        "no_live_queue_mutation_contract"
    );
    assert!(batch_validation.printed_ready());
    assert!(batch_validation.ready());
    assert!(batch_validation.no_live_queue_mutation_contract());
    assert!(batch_validation.passed());
    assert!(!batch_validation.submits_work());
    assert!(!batch_validation.mutates_live_queue());
    assert_eq!(
        batch_validation.receipt_lines(),
        vec![
            "receipt.kind=model_runtime_launch_live_aql_proof_validation",
            "receipt.version=1",
            "proof_kind=batch_reservation_plan",
            "proof_input=KfdQueueLiveAqlBatchReservationPlanInput",
            "proof_type=KfdQueueLiveAqlBatchReservationPlanProof",
            "validation_type=KfdQueueLiveAqlBatchReservationPlanValidation",
            "validation_method=KfdQueueLiveAqlBatchReservationPlanProof::validate_ready",
            "validation_ready_field=ready",
            "no_live_queue_mutation_contract_field=no_live_queue_mutation_contract",
            "printed_ready=true",
            "ready=true",
            "no_live_queue_mutation_contract=true",
            "passed=true",
            "submits_work=false",
            "mutates_live_queue=false",
        ]
    );
    assert!(batch_validation.receipt_text().ends_with('\n'));
    assert_eq!(batch_validation.receipt_fingerprint().len(), 64);

    let materialized_proof = public_live_aql_materialized_packet_plan_proof();
    let materialized_validation = RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
        .validate_materialized_packet_plan_proof(materialized_proof)?;
    assert!(matches!(
        materialized_validation,
        RuntimeLaunchLiveAqlProofValidation::MaterializedPacketPlan(_)
    ));
    assert_eq!(
        materialized_validation.kind(),
        RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
    );
    assert_eq!(
        materialized_validation.proof_input(),
        "KfdQueueLiveAqlMaterializedPacketPlanInput"
    );
    assert_eq!(
        materialized_validation.proof_type(),
        "KfdQueueLiveAqlMaterializedPacketPlanProof"
    );
    assert_eq!(
        materialized_validation.validation_type(),
        "KfdQueueLiveAqlMaterializedPacketPlanValidation"
    );
    assert_eq!(
        materialized_validation.validation_method(),
        "KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready"
    );
    assert_eq!(materialized_validation.validation_ready_field(), "ready");
    assert!(materialized_validation.printed_ready());
    assert!(materialized_validation.ready());
    assert!(materialized_validation.no_live_queue_mutation_contract());
    assert!(materialized_validation.passed());
    assert!(!materialized_validation.submits_work());
    assert!(!materialized_validation.mutates_live_queue());

    let mismatch = RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
        .validate_batch_reservation_plan_proof(batch_proof)
        .unwrap_err();
    assert_eq!(
        mismatch.to_string(),
        "live-AQL proof kind materialized_packet_plan cannot validate KfdQueueLiveAqlBatchReservationPlanProof; expected batch_reservation_plan"
    );

    Ok(())
}

#[test]
fn external_model_definition_uses_public_api_readiness_contract() -> Result<()> {
    assert_eq!(MODEL_API_CONTRACT.name, MODEL_API_CONTRACT_NAME);
    assert_eq!(MODEL_API_CONTRACT.version, MODEL_API_CONTRACT_VERSION);
    assert_eq!(
        MODEL_API_CONTRACT.stability,
        ModelApiContractStability::Pre1StaticMetadata
    );
    assert_eq!(
        MODEL_API_CONTRACT.stability.as_str(),
        "pre1-static-metadata"
    );
    assert!(!MODEL_API_CONTRACT.live_execution_supported);
    assert_eq!(
        MODEL_API_CONTRACT.to_string(),
        "mainarch-model-api version=0.1.0 stability=pre1-static-metadata live_execution_supported=false"
    );
    let contract_receipt_lines = MODEL_API_CONTRACT.receipt_lines();
    assert_eq!(contract_receipt_lines.len(), 6);
    assert_eq!(contract_receipt_lines[0], "receipt.kind=model_api_contract");
    assert_eq!(contract_receipt_lines[1], "receipt.version=1");
    assert!(contract_receipt_lines
        .iter()
        .any(|line| line == "name=mainarch-model-api"));
    assert!(contract_receipt_lines
        .iter()
        .any(|line| line == "version=0.1.0"));
    assert!(contract_receipt_lines
        .iter()
        .any(|line| line == "stability=pre1-static-metadata"));
    assert!(contract_receipt_lines
        .iter()
        .any(|line| line == "live_execution_supported=false"));
    assert!(MODEL_API_CONTRACT.receipt_text().ends_with('\n'));
    let contract_receipt_fingerprint = MODEL_API_CONTRACT.receipt_fingerprint();
    assert_eq!(contract_receipt_fingerprint.len(), 64);
    assert!(contract_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_contract = MODEL_API_CONTRACT;
    stale_contract.version = "0.1.1";
    assert_ne!(
        stale_contract.receipt_fingerprint(),
        contract_receipt_fingerprint
    );
    assert_eq!(LoweringStatus::NativeGpu.as_str(), "native_gpu");
    assert_eq!(LoweringStatus::FusedNativeGpu.as_str(), "fused_native_gpu");
    assert_eq!(LoweringStatus::Gap.as_str(), "gap");
    assert_eq!(
        ModelReadinessIssueKind::LoweringGap.as_str(),
        "lowering_gap"
    );
    assert_eq!(
        ModelReadinessIssueKind::StageLoweringGap.as_str(),
        "stage_lowering_gap"
    );
    assert_eq!(
        ModelPluginCompatibilityIssueKind::StaticMetadata.as_str(),
        "static_metadata"
    );
    assert_eq!(PrimitiveKind::EmbeddingLookup.as_str(), "embedding_lookup");
    assert_eq!(ModelStageKind::Sampling.as_str(), "sampling");
    assert_eq!(PrimitiveKind::ALL.len(), 12);
    assert_eq!(ModelStageKind::ALL.len(), 5);
    assert_eq!(
        model_primitive_kind_descriptors()
            .iter()
            .map(|descriptor| descriptor.label)
            .collect::<Vec<_>>(),
        vec![
            "embedding_lookup",
            "linear",
            "rms_norm",
            "add_rms_norm",
            "apply_rope",
            "kv_cache_append",
            "paged_attention",
            "moe_router_topk",
            "moe_local_ffn",
            "residual_add",
            "collective",
            "argmax_sample"
        ]
    );
    assert!(model_primitive_kind_descriptors()
        .iter()
        .all(|descriptor| !descriptor.summary.is_empty()));
    assert_eq!(
        model_stage_kind_descriptors()
            .iter()
            .map(|descriptor| descriptor.label)
            .collect::<Vec<_>>(),
        vec!["embedding", "attention", "moe", "output", "sampling"]
    );
    assert!(model_stage_kind_descriptors()
        .iter()
        .all(|descriptor| !descriptor.summary.is_empty()));
    assert!(runtime_launch_kernel_argument_abi_schema_count() > 0);
    let gemv_schema = runtime_launch_kernel_argument_abi_schema_for("gemv_f16").unwrap();
    assert_eq!(gemv_schema.kernel_symbol, "gemv_f16");
    assert_eq!(gemv_schema.kernarg_size, 32);
    assert_eq!(gemv_schema.kernarg_segment_align, 8);
    assert!(runtime_launch_kernel_argument_abi_schema_for("not_a_mainarch_kernel").is_none());
    assert_eq!(
        runtime_launch_kernel_argument_abi_semantic_schema_count(),
        51
    );
    let rmsnorm_semantic_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("rmsnorm_f16").unwrap();
    assert_eq!(rmsnorm_semantic_schema.kernel_symbol, "rmsnorm_f16");
    assert_eq!(rmsnorm_semantic_schema.fields[3].kernel_argument_name, "H");
    assert_eq!(
        rmsnorm_semantic_schema.fields[3].model_argument_name,
        "normalized_dim"
    );
    assert_eq!(rmsnorm_semantic_schema.fields[3].encoding.as_str(), "u32");
    let add_rmsnorm_bf16_semantic_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("add_rmsnorm_bf16_residual_f16_out")
            .unwrap();
    assert_eq!(add_rmsnorm_bf16_semantic_schema.kernarg_size, 40);
    assert_eq!(
        add_rmsnorm_bf16_semantic_schema.fields[0].model_argument_name,
        "residual_output"
    );
    assert_eq!(
        add_rmsnorm_bf16_semantic_schema.fields[2].kernel_argument_name,
        "w"
    );
    assert_eq!(
        add_rmsnorm_bf16_semantic_schema.fields[3].model_argument_name,
        "norm_output"
    );
    let rope_semantic_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("rope_f16").unwrap();
    assert_eq!(rope_semantic_schema.kernarg_size, 20);
    assert_eq!(rope_semantic_schema.kernarg_segment_align, 8);
    assert_eq!(rope_semantic_schema.fields[0].kernel_argument_name, "x");
    assert_eq!(rope_semantic_schema.fields[0].model_argument_name, "query");
    assert_eq!(rope_semantic_schema.fields[1].kernel_argument_name, "H");
    assert_eq!(
        rope_semantic_schema.fields[1].model_argument_name,
        "head_dim"
    );
    assert_eq!(rope_semantic_schema.fields[1].offset, 8);
    assert_eq!(rope_semantic_schema.fields[2].kernel_argument_name, "pos");
    assert_eq!(
        rope_semantic_schema.fields[2].model_argument_name,
        "position"
    );
    assert_eq!(rope_semantic_schema.fields[2].offset, 12);
    assert_eq!(rope_semantic_schema.fields[3].kernel_argument_name, "theta");
    assert_eq!(rope_semantic_schema.fields[3].model_argument_name, "theta");
    assert_eq!(rope_semantic_schema.fields[3].offset, 16);
    let fused_allreduce_rmsnorm_schema = runtime_launch_kernel_argument_abi_semantic_schema_for(
        "allreduce_direct_residual_rmsnorm_grid",
    )
    .unwrap();
    assert_eq!(fused_allreduce_rmsnorm_schema.kernarg_size, 320);
    assert_eq!(fused_allreduce_rmsnorm_schema.kernarg_segment_align, 8);
    assert_eq!(fused_allreduce_rmsnorm_schema.fields.len(), 10);
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[0].kernel_argument_name,
        "own"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[2].kernel_argument_name,
        "residual_ptrs"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[2].model_argument_name,
        "peer_residual_ptrs"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[3].kernel_argument_name,
        "weight"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[3].model_argument_name,
        "weight"
    );
    assert_eq!(fused_allreduce_rmsnorm_schema.fields[3].offset, 24);
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[6].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[6].model_argument_name,
        "group_size"
    );
    assert_eq!(fused_allreduce_rmsnorm_schema.fields[6].offset, 48);
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[7].kernel_argument_name,
        "n"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[7].model_argument_name,
        "normalized_dim"
    );
    assert_eq!(fused_allreduce_rmsnorm_schema.fields[7].offset, 52);
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[8].kernel_argument_name,
        "eps"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[8].model_argument_name,
        "eps"
    );
    assert_eq!(fused_allreduce_rmsnorm_schema.fields[8].offset, 56);
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[9].kernel_argument_name,
        "num_wg"
    );
    assert_eq!(
        fused_allreduce_rmsnorm_schema.fields[9].model_argument_name,
        "num_wg"
    );
    assert_eq!(fused_allreduce_rmsnorm_schema.fields[9].offset, 60);
    let paged_mla_stage1_semantic_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("paged_mla_fp8_splitk_stage1_probe")
            .expect("missing semantic schema for MLA split-K stage1");
    assert_eq!(paged_mla_stage1_semantic_schema.kernarg_size, 400);
    assert_eq!(paged_mla_stage1_semantic_schema.kernarg_segment_align, 8);
    assert_eq!(paged_mla_stage1_semantic_schema.fields.len(), 25);
    assert_eq!(
        paged_mla_stage1_semantic_schema.fields[0].kernel_argument_name,
        "q_nope"
    );
    assert_eq!(
        paged_mla_stage1_semantic_schema.fields[10].kernel_argument_name,
        "fused_output_workspace"
    );
    assert_eq!(
        paged_mla_stage1_semantic_schema.fields[24].kernel_argument_name,
        "internal_split_mode"
    );
    assert_eq!(paged_mla_stage1_semantic_schema.fields[24].offset, 140);
    let paged_mla_stage2_semantic_schema = runtime_launch_kernel_argument_abi_semantic_schema_for(
        "paged_mla_fp8_splitk_stage2_merge_probe",
    )
    .expect("missing semantic schema for MLA split-K stage2");
    assert_eq!(paged_mla_stage2_semantic_schema.kernarg_size, 304);
    assert_eq!(paged_mla_stage2_semantic_schema.kernarg_segment_align, 8);
    assert_eq!(paged_mla_stage2_semantic_schema.fields.len(), 9);
    assert_eq!(
        paged_mla_stage2_semantic_schema.fields[0].kernel_argument_name,
        "partial_workspace"
    );
    assert_eq!(
        paged_mla_stage2_semantic_schema.fields[8].kernel_argument_name,
        "output_index"
    );
    assert_eq!(paged_mla_stage2_semantic_schema.fields[8].offset, 44);
    let reduce_peers_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("reduce_peers")
            .expect("missing semantic schema for reduce_peers");
    assert_eq!(reduce_peers_schema.kernarg_size, 280);
    assert_eq!(reduce_peers_schema.kernarg_segment_align, 8);
    assert_eq!(reduce_peers_schema.fields.len(), 4);
    assert_eq!(reduce_peers_schema.fields[0].kernel_argument_name, "dst");
    assert_eq!(
        reduce_peers_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(reduce_peers_schema.fields[0].offset, 0);
    assert_eq!(reduce_peers_schema.fields[1].kernel_argument_name, "ptrs");
    assert_eq!(
        reduce_peers_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(reduce_peers_schema.fields[1].offset, 8);
    assert_eq!(reduce_peers_schema.fields[2].kernel_argument_name, "parts");
    assert_eq!(
        reduce_peers_schema.fields[2].model_argument_name,
        "group_size"
    );
    assert_eq!(reduce_peers_schema.fields[2].offset, 16);
    assert_eq!(reduce_peers_schema.fields[3].kernel_argument_name, "n");
    assert_eq!(reduce_peers_schema.fields[3].model_argument_name, "n");
    assert_eq!(reduce_peers_schema.fields[3].offset, 20);
    let scatter_to_staging_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("scatter_to_staging")
            .expect("missing semantic schema for scatter_to_staging");
    assert_eq!(scatter_to_staging_schema.kernarg_size, 288);
    assert_eq!(scatter_to_staging_schema.kernarg_segment_align, 8);
    assert_eq!(scatter_to_staging_schema.fields.len(), 6);
    assert_eq!(
        scatter_to_staging_schema.fields[0].kernel_argument_name,
        "own"
    );
    assert_eq!(
        scatter_to_staging_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(scatter_to_staging_schema.fields[0].offset, 0);
    assert_eq!(
        scatter_to_staging_schema.fields[1].kernel_argument_name,
        "stage_ptrs"
    );
    assert_eq!(
        scatter_to_staging_schema.fields[1].model_argument_name,
        "peer_reduce_staging_ptrs"
    );
    assert_eq!(scatter_to_staging_schema.fields[1].offset, 8);
    assert_eq!(
        scatter_to_staging_schema.fields[2].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        scatter_to_staging_schema.fields[2].model_argument_name,
        "group_size"
    );
    assert_eq!(scatter_to_staging_schema.fields[2].offset, 16);
    assert_eq!(
        scatter_to_staging_schema.fields[3].kernel_argument_name,
        "cl"
    );
    assert_eq!(
        scatter_to_staging_schema.fields[3].model_argument_name,
        "chunk_len"
    );
    assert_eq!(scatter_to_staging_schema.fields[3].offset, 20);
    assert_eq!(
        scatter_to_staging_schema.fields[4].kernel_argument_name,
        "self_idx"
    );
    assert_eq!(
        scatter_to_staging_schema.fields[4].model_argument_name,
        "self_rank"
    );
    assert_eq!(scatter_to_staging_schema.fields[4].offset, 24);
    assert_eq!(
        scatter_to_staging_schema.fields[5].kernel_argument_name,
        "n"
    );
    assert_eq!(scatter_to_staging_schema.fields[5].model_argument_name, "n");
    assert_eq!(scatter_to_staging_schema.fields[5].offset, 28);
    let gather_reduce_local_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("gather_reduce_local")
            .expect("missing semantic schema for gather_reduce_local");
    assert_eq!(gather_reduce_local_schema.kernarg_size, 288);
    assert_eq!(gather_reduce_local_schema.kernarg_segment_align, 8);
    assert_eq!(gather_reduce_local_schema.fields.len(), 6);
    assert_eq!(
        gather_reduce_local_schema.fields[0].kernel_argument_name,
        "out"
    );
    assert_eq!(
        gather_reduce_local_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(gather_reduce_local_schema.fields[0].offset, 0);
    assert_eq!(
        gather_reduce_local_schema.fields[1].kernel_argument_name,
        "stage"
    );
    assert_eq!(
        gather_reduce_local_schema.fields[1].model_argument_name,
        "reduce_staging_buffer"
    );
    assert_eq!(gather_reduce_local_schema.fields[1].offset, 8);
    assert_eq!(
        gather_reduce_local_schema.fields[2].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        gather_reduce_local_schema.fields[2].model_argument_name,
        "group_size"
    );
    assert_eq!(gather_reduce_local_schema.fields[2].offset, 16);
    assert_eq!(
        gather_reduce_local_schema.fields[3].kernel_argument_name,
        "off"
    );
    assert_eq!(
        gather_reduce_local_schema.fields[3].model_argument_name,
        "chunk_offset"
    );
    assert_eq!(gather_reduce_local_schema.fields[3].offset, 20);
    assert_eq!(
        gather_reduce_local_schema.fields[4].kernel_argument_name,
        "cl"
    );
    assert_eq!(
        gather_reduce_local_schema.fields[4].model_argument_name,
        "reduce_staging_chunk_len"
    );
    assert_eq!(gather_reduce_local_schema.fields[4].offset, 24);
    assert_eq!(
        gather_reduce_local_schema.fields[5].kernel_argument_name,
        "len"
    );
    assert_eq!(
        gather_reduce_local_schema.fields[5].model_argument_name,
        "chunk_len"
    );
    assert_eq!(gather_reduce_local_schema.fields[5].offset, 28);
    let reduce_scatter_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("reduce_scatter")
            .expect("missing semantic schema for reduce_scatter");
    assert_eq!(reduce_scatter_schema.kernarg_size, 288);
    assert_eq!(reduce_scatter_schema.kernarg_segment_align, 8);
    assert_eq!(reduce_scatter_schema.fields.len(), 5);
    assert_eq!(reduce_scatter_schema.fields[0].kernel_argument_name, "out");
    assert_eq!(
        reduce_scatter_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(reduce_scatter_schema.fields[0].offset, 0);
    assert_eq!(reduce_scatter_schema.fields[1].kernel_argument_name, "ptrs");
    assert_eq!(
        reduce_scatter_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(reduce_scatter_schema.fields[1].offset, 8);
    assert_eq!(
        reduce_scatter_schema.fields[2].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        reduce_scatter_schema.fields[2].model_argument_name,
        "group_size"
    );
    assert_eq!(reduce_scatter_schema.fields[2].offset, 16);
    assert_eq!(reduce_scatter_schema.fields[3].kernel_argument_name, "off");
    assert_eq!(
        reduce_scatter_schema.fields[3].model_argument_name,
        "chunk_offset"
    );
    assert_eq!(reduce_scatter_schema.fields[3].offset, 20);
    assert_eq!(reduce_scatter_schema.fields[4].kernel_argument_name, "len");
    assert_eq!(
        reduce_scatter_schema.fields[4].model_argument_name,
        "chunk_len"
    );
    assert_eq!(reduce_scatter_schema.fields[4].offset, 24);
    let broadcast_chunk_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("broadcast_chunk")
            .expect("missing semantic schema for broadcast_chunk");
    assert_eq!(broadcast_chunk_schema.kernarg_size, 288);
    assert_eq!(broadcast_chunk_schema.kernarg_segment_align, 8);
    assert_eq!(broadcast_chunk_schema.fields.len(), 5);
    assert_eq!(broadcast_chunk_schema.fields[0].kernel_argument_name, "src");
    assert_eq!(
        broadcast_chunk_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(broadcast_chunk_schema.fields[0].offset, 0);
    assert_eq!(
        broadcast_chunk_schema.fields[1].kernel_argument_name,
        "ptrs"
    );
    assert_eq!(
        broadcast_chunk_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(broadcast_chunk_schema.fields[1].offset, 8);
    assert_eq!(
        broadcast_chunk_schema.fields[2].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        broadcast_chunk_schema.fields[2].model_argument_name,
        "group_size"
    );
    assert_eq!(broadcast_chunk_schema.fields[2].offset, 16);
    assert_eq!(broadcast_chunk_schema.fields[3].kernel_argument_name, "off");
    assert_eq!(
        broadcast_chunk_schema.fields[3].model_argument_name,
        "chunk_offset"
    );
    assert_eq!(broadcast_chunk_schema.fields[3].offset, 20);
    assert_eq!(broadcast_chunk_schema.fields[4].kernel_argument_name, "len");
    assert_eq!(
        broadcast_chunk_schema.fields[4].model_argument_name,
        "chunk_len"
    );
    assert_eq!(broadcast_chunk_schema.fields[4].offset, 24);
    let broadcast_chunk_skip_owner_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("broadcast_chunk_skip_owner")
            .expect("missing semantic schema for broadcast_chunk_skip_owner");
    assert_eq!(broadcast_chunk_skip_owner_schema.kernarg_size, 288);
    assert_eq!(broadcast_chunk_skip_owner_schema.kernarg_segment_align, 8);
    assert_eq!(broadcast_chunk_skip_owner_schema.fields.len(), 6);
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[0].kernel_argument_name,
        "src"
    );
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(broadcast_chunk_skip_owner_schema.fields[0].offset, 0);
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[1].kernel_argument_name,
        "ptrs"
    );
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(broadcast_chunk_skip_owner_schema.fields[1].offset, 8);
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[2].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[2].model_argument_name,
        "group_size"
    );
    assert_eq!(broadcast_chunk_skip_owner_schema.fields[2].offset, 16);
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[3].kernel_argument_name,
        "off"
    );
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[3].model_argument_name,
        "chunk_offset"
    );
    assert_eq!(broadcast_chunk_skip_owner_schema.fields[3].offset, 20);
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[4].kernel_argument_name,
        "len"
    );
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[4].model_argument_name,
        "chunk_len"
    );
    assert_eq!(broadcast_chunk_skip_owner_schema.fields[4].offset, 24);
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[5].kernel_argument_name,
        "owner"
    );
    assert_eq!(
        broadcast_chunk_skip_owner_schema.fields[5].model_argument_name,
        "chunk_owner"
    );
    assert_eq!(broadcast_chunk_skip_owner_schema.fields[5].offset, 28);
    let all_gather_schema = runtime_launch_kernel_argument_abi_semantic_schema_for("all_gather")
        .expect("missing semantic schema for all_gather");
    assert_eq!(all_gather_schema.kernarg_size, 288);
    assert_eq!(all_gather_schema.kernarg_segment_align, 8);
    assert_eq!(all_gather_schema.fields.len(), 6);
    assert_eq!(all_gather_schema.fields[0].kernel_argument_name, "out");
    assert_eq!(
        all_gather_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(all_gather_schema.fields[0].offset, 0);
    assert_eq!(all_gather_schema.fields[1].kernel_argument_name, "ptrs");
    assert_eq!(
        all_gather_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(all_gather_schema.fields[1].offset, 8);
    assert_eq!(
        all_gather_schema.fields[2].kernel_argument_name,
        "chunk_len"
    );
    assert_eq!(all_gather_schema.fields[2].model_argument_name, "chunk_len");
    assert_eq!(all_gather_schema.fields[2].offset, 16);
    assert_eq!(all_gather_schema.fields[3].kernel_argument_name, "parts");
    assert_eq!(
        all_gather_schema.fields[3].model_argument_name,
        "group_size"
    );
    assert_eq!(all_gather_schema.fields[3].offset, 20);
    assert_eq!(all_gather_schema.fields[4].kernel_argument_name, "n");
    assert_eq!(all_gather_schema.fields[4].model_argument_name, "n");
    assert_eq!(all_gather_schema.fields[4].offset, 24);
    assert_eq!(
        all_gather_schema.fields[5].kernel_argument_name,
        "own_chunk"
    );
    assert_eq!(
        all_gather_schema.fields[5].model_argument_name,
        "chunk_owner"
    );
    assert_eq!(all_gather_schema.fields[5].offset, 28);
    for symbol in ["broadcast_peers", "broadcast_peers_skip0"] {
        let broadcast_schema = runtime_launch_kernel_argument_abi_semantic_schema_for(symbol)
            .unwrap_or_else(|| panic!("missing semantic schema for {symbol}"));
        assert_eq!(broadcast_schema.kernarg_size, 280);
        assert_eq!(broadcast_schema.kernarg_segment_align, 8);
        assert_eq!(broadcast_schema.fields.len(), 4);
        assert_eq!(broadcast_schema.fields[0].kernel_argument_name, "src");
        assert_eq!(
            broadcast_schema.fields[0].model_argument_name,
            "local_allreduce_buffer"
        );
        assert_eq!(broadcast_schema.fields[0].offset, 0);
        assert_eq!(broadcast_schema.fields[1].kernel_argument_name, "ptrs");
        assert_eq!(
            broadcast_schema.fields[1].model_argument_name,
            "peer_allreduce_ptrs"
        );
        assert_eq!(broadcast_schema.fields[1].offset, 8);
        assert_eq!(broadcast_schema.fields[2].kernel_argument_name, "parts");
        assert_eq!(broadcast_schema.fields[2].model_argument_name, "group_size");
        assert_eq!(broadcast_schema.fields[2].offset, 16);
        assert_eq!(broadcast_schema.fields[3].kernel_argument_name, "n");
        assert_eq!(broadcast_schema.fields[3].model_argument_name, "n");
        assert_eq!(broadcast_schema.fields[3].offset, 20);
    }
    let allreduce_oneshot_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("allreduce_oneshot")
            .expect("missing semantic schema for allreduce_oneshot");
    assert_eq!(allreduce_oneshot_schema.kernarg_size, 344);
    assert_eq!(allreduce_oneshot_schema.kernarg_segment_align, 8);
    assert_eq!(allreduce_oneshot_schema.fields.len(), 14);
    assert_eq!(
        allreduce_oneshot_schema.fields[0].kernel_argument_name,
        "own"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(allreduce_oneshot_schema.fields[0].offset, 0);
    assert_eq!(
        allreduce_oneshot_schema.fields[1].kernel_argument_name,
        "peer_bufs"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(allreduce_oneshot_schema.fields[1].offset, 8);
    assert_eq!(
        allreduce_oneshot_schema.fields[2].kernel_argument_name,
        "stage_ptrs"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[2].model_argument_name,
        "peer_reduce_staging_ptrs"
    );
    assert_eq!(allreduce_oneshot_schema.fields[2].offset, 16);
    assert_eq!(
        allreduce_oneshot_schema.fields[3].kernel_argument_name,
        "my_stage"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[3].model_argument_name,
        "reduce_staging_buffer"
    );
    assert_eq!(allreduce_oneshot_schema.fields[3].offset, 24);
    assert_eq!(
        allreduce_oneshot_schema.fields[4].kernel_argument_name,
        "my_flags"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[4].model_argument_name,
        "local_allreduce_flags"
    );
    assert_eq!(allreduce_oneshot_schema.fields[4].offset, 32);
    assert_eq!(
        allreduce_oneshot_schema.fields[5].kernel_argument_name,
        "peer_flag_ptrs"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[5].model_argument_name,
        "peer_allreduce_flag_ptrs"
    );
    assert_eq!(allreduce_oneshot_schema.fields[5].offset, 40);
    assert_eq!(
        allreduce_oneshot_schema.fields[6].kernel_argument_name,
        "gbar"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[6].model_argument_name,
        "gbar"
    );
    assert_eq!(allreduce_oneshot_schema.fields[6].offset, 48);
    assert_eq!(
        allreduce_oneshot_schema.fields[7].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[7].model_argument_name,
        "group_size"
    );
    assert_eq!(allreduce_oneshot_schema.fields[7].offset, 56);
    assert_eq!(
        allreduce_oneshot_schema.fields[8].kernel_argument_name,
        "self_idx"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[8].model_argument_name,
        "self_rank"
    );
    assert_eq!(allreduce_oneshot_schema.fields[8].offset, 60);
    assert_eq!(
        allreduce_oneshot_schema.fields[9].kernel_argument_name,
        "cl"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[9].model_argument_name,
        "chunk_len"
    );
    assert_eq!(allreduce_oneshot_schema.fields[9].offset, 64);
    assert_eq!(
        allreduce_oneshot_schema.fields[10].kernel_argument_name,
        "n"
    );
    assert_eq!(allreduce_oneshot_schema.fields[10].model_argument_name, "n");
    assert_eq!(allreduce_oneshot_schema.fields[10].offset, 68);
    assert_eq!(
        allreduce_oneshot_schema.fields[11].kernel_argument_name,
        "num_wg"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[11].model_argument_name,
        "num_wg"
    );
    assert_eq!(allreduce_oneshot_schema.fields[11].offset, 72);
    assert_eq!(
        allreduce_oneshot_schema.fields[12].kernel_argument_name,
        "seq_base"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[12].model_argument_name,
        "allreduce_sequence_base"
    );
    assert_eq!(allreduce_oneshot_schema.fields[12].offset, 76);
    assert_eq!(
        allreduce_oneshot_schema.fields[13].kernel_argument_name,
        "num_tiles"
    );
    assert_eq!(
        allreduce_oneshot_schema.fields[13].model_argument_name,
        "num_tiles"
    );
    assert_eq!(allreduce_oneshot_schema.fields[13].offset, 80);
    let p2p_write_schema = runtime_launch_kernel_argument_abi_semantic_schema_for("p2p_write")
        .expect("missing semantic schema for p2p_write");
    assert_eq!(p2p_write_schema.kernarg_size, 36);
    assert_eq!(p2p_write_schema.kernarg_segment_align, 8);
    assert_eq!(p2p_write_schema.fields.len(), 7);
    assert_eq!(p2p_write_schema.fields[0].kernel_argument_name, "src");
    assert_eq!(
        p2p_write_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(p2p_write_schema.fields[0].offset, 0);
    assert_eq!(p2p_write_schema.fields[1].kernel_argument_name, "peer_bufs");
    assert_eq!(
        p2p_write_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(p2p_write_schema.fields[1].offset, 8);
    assert_eq!(p2p_write_schema.fields[2].kernel_argument_name, "npeers");
    assert_eq!(
        p2p_write_schema.fields[2].model_argument_name,
        "p2p_peer_count"
    );
    assert_eq!(p2p_write_schema.fields[2].offset, 16);
    assert_eq!(p2p_write_schema.fields[3].kernel_argument_name, "per4");
    assert_eq!(
        p2p_write_schema.fields[3].model_argument_name,
        "peer_stride_vec4"
    );
    assert_eq!(p2p_write_schema.fields[3].offset, 20);
    assert_eq!(p2p_write_schema.fields[4].kernel_argument_name, "start4");
    assert_eq!(
        p2p_write_schema.fields[4].model_argument_name,
        "chunk_offset_vec4"
    );
    assert_eq!(p2p_write_schema.fields[4].offset, 24);
    assert_eq!(p2p_write_schema.fields[5].kernel_argument_name, "count4");
    assert_eq!(
        p2p_write_schema.fields[5].model_argument_name,
        "chunk_len_vec4"
    );
    assert_eq!(p2p_write_schema.fields[5].offset, 28);
    assert_eq!(p2p_write_schema.fields[6].kernel_argument_name, "num_wg");
    assert_eq!(p2p_write_schema.fields[6].model_argument_name, "num_wg");
    assert_eq!(p2p_write_schema.fields[6].offset, 32);
    let p2p_broadcast_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("p2p_broadcast")
            .expect("missing semantic schema for p2p_broadcast");
    assert_eq!(p2p_broadcast_schema.kernarg_size, 32);
    assert_eq!(p2p_broadcast_schema.kernarg_segment_align, 8);
    assert_eq!(p2p_broadcast_schema.fields.len(), 6);
    assert_eq!(p2p_broadcast_schema.fields[0].kernel_argument_name, "src");
    assert_eq!(
        p2p_broadcast_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(p2p_broadcast_schema.fields[0].offset, 0);
    assert_eq!(
        p2p_broadcast_schema.fields[1].kernel_argument_name,
        "peer_bufs"
    );
    assert_eq!(
        p2p_broadcast_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(p2p_broadcast_schema.fields[1].offset, 8);
    assert_eq!(
        p2p_broadcast_schema.fields[2].kernel_argument_name,
        "npeers"
    );
    assert_eq!(
        p2p_broadcast_schema.fields[2].model_argument_name,
        "p2p_peer_count"
    );
    assert_eq!(p2p_broadcast_schema.fields[2].offset, 16);
    assert_eq!(
        p2p_broadcast_schema.fields[3].kernel_argument_name,
        "start4"
    );
    assert_eq!(
        p2p_broadcast_schema.fields[3].model_argument_name,
        "chunk_offset_vec4"
    );
    assert_eq!(p2p_broadcast_schema.fields[3].offset, 20);
    assert_eq!(
        p2p_broadcast_schema.fields[4].kernel_argument_name,
        "count4"
    );
    assert_eq!(
        p2p_broadcast_schema.fields[4].model_argument_name,
        "chunk_len_vec4"
    );
    assert_eq!(p2p_broadcast_schema.fields[4].offset, 24);
    assert_eq!(
        p2p_broadcast_schema.fields[5].kernel_argument_name,
        "num_wg"
    );
    assert_eq!(p2p_broadcast_schema.fields[5].model_argument_name, "num_wg");
    assert_eq!(p2p_broadcast_schema.fields[5].offset, 28);
    let allreduce_dualpath_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("allreduce_dualpath")
            .expect("missing semantic schema for allreduce_dualpath");
    assert_eq!(allreduce_dualpath_schema.kernarg_size, 352);
    assert_eq!(allreduce_dualpath_schema.kernarg_segment_align, 8);
    assert_eq!(allreduce_dualpath_schema.fields.len(), 16);
    assert_eq!(
        allreduce_dualpath_schema.fields[0].kernel_argument_name,
        "own"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(allreduce_dualpath_schema.fields[0].offset, 0);
    assert_eq!(
        allreduce_dualpath_schema.fields[1].kernel_argument_name,
        "peer_bufs"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(allreduce_dualpath_schema.fields[1].offset, 8);
    assert_eq!(
        allreduce_dualpath_schema.fields[2].kernel_argument_name,
        "stage_ptrs"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[2].model_argument_name,
        "peer_reduce_staging_ptrs"
    );
    assert_eq!(allreduce_dualpath_schema.fields[2].offset, 16);
    assert_eq!(
        allreduce_dualpath_schema.fields[3].kernel_argument_name,
        "my_stage"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[3].model_argument_name,
        "reduce_staging_buffer"
    );
    assert_eq!(allreduce_dualpath_schema.fields[3].offset, 24);
    assert_eq!(
        allreduce_dualpath_schema.fields[4].kernel_argument_name,
        "my_flags"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[4].model_argument_name,
        "local_allreduce_flags"
    );
    assert_eq!(allreduce_dualpath_schema.fields[4].offset, 32);
    assert_eq!(
        allreduce_dualpath_schema.fields[5].kernel_argument_name,
        "peer_flag_ptrs"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[5].model_argument_name,
        "peer_allreduce_flag_ptrs"
    );
    assert_eq!(allreduce_dualpath_schema.fields[5].offset, 40);
    assert_eq!(
        allreduce_dualpath_schema.fields[6].kernel_argument_name,
        "gbar"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[6].model_argument_name,
        "gbar"
    );
    assert_eq!(allreduce_dualpath_schema.fields[6].offset, 48);
    assert_eq!(
        allreduce_dualpath_schema.fields[7].kernel_argument_name,
        "sem"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[7].model_argument_name,
        "dualpath_sdma_semaphores"
    );
    assert_eq!(allreduce_dualpath_schema.fields[7].offset, 56);
    assert_eq!(
        allreduce_dualpath_schema.fields[8].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[8].model_argument_name,
        "group_size"
    );
    assert_eq!(allreduce_dualpath_schema.fields[8].offset, 64);
    assert_eq!(
        allreduce_dualpath_schema.fields[9].kernel_argument_name,
        "self_idx"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[9].model_argument_name,
        "self_rank"
    );
    assert_eq!(allreduce_dualpath_schema.fields[9].offset, 68);
    assert_eq!(
        allreduce_dualpath_schema.fields[10].kernel_argument_name,
        "cl"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[10].model_argument_name,
        "chunk_len"
    );
    assert_eq!(allreduce_dualpath_schema.fields[10].offset, 72);
    assert_eq!(
        allreduce_dualpath_schema.fields[11].kernel_argument_name,
        "n"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[11].model_argument_name,
        "n"
    );
    assert_eq!(allreduce_dualpath_schema.fields[11].offset, 76);
    assert_eq!(
        allreduce_dualpath_schema.fields[12].kernel_argument_name,
        "num_wg"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[12].model_argument_name,
        "num_wg"
    );
    assert_eq!(allreduce_dualpath_schema.fields[12].offset, 80);
    assert_eq!(
        allreduce_dualpath_schema.fields[13].kernel_argument_name,
        "seq_base"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[13].model_argument_name,
        "allreduce_sequence_base"
    );
    assert_eq!(allreduce_dualpath_schema.fields[13].offset, 84);
    assert_eq!(
        allreduce_dualpath_schema.fields[14].kernel_argument_name,
        "cu_clv"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[14].model_argument_name,
        "dualpath_cu_chunk_len_vec4"
    );
    assert_eq!(allreduce_dualpath_schema.fields[14].offset, 88);
    assert_eq!(
        allreduce_dualpath_schema.fields[15].kernel_argument_name,
        "cu_mylen4"
    );
    assert_eq!(
        allreduce_dualpath_schema.fields[15].model_argument_name,
        "dualpath_cu_owned_chunk_len_vec4"
    );
    assert_eq!(allreduce_dualpath_schema.fields[15].offset, 92);
    let allreduce_direct_persistent_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("allreduce_direct_persistent")
            .expect("missing semantic schema for allreduce_direct_persistent");
    assert_eq!(allreduce_direct_persistent_schema.kernarg_size, 304);
    assert_eq!(allreduce_direct_persistent_schema.kernarg_segment_align, 8);
    assert_eq!(allreduce_direct_persistent_schema.fields.len(), 7);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[0].kernel_argument_name,
        "own"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[0].model_argument_name,
        "local_allreduce_buffer"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[0].offset, 0);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[1].kernel_argument_name,
        "ptrs"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[1].offset, 8);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[2].kernel_argument_name,
        "ctrl"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[2].model_argument_name,
        "persistent_allreduce_ctrl"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[2].offset, 16);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[3].kernel_argument_name,
        "gbar"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[3].model_argument_name,
        "gbar"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[3].offset, 24);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[4].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[4].model_argument_name,
        "group_size"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[4].offset, 32);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[5].kernel_argument_name,
        "n"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[5].model_argument_name,
        "n"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[5].offset, 36);
    assert_eq!(
        allreduce_direct_persistent_schema.fields[6].kernel_argument_name,
        "total_ops"
    );
    assert_eq!(
        allreduce_direct_persistent_schema.fields[6].model_argument_name,
        "persistent_allreduce_total_ops"
    );
    assert_eq!(allreduce_direct_persistent_schema.fields[6].offset, 40);
    let allreduce_dda_persistent_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("allreduce_dda_persistent")
            .expect("missing semantic schema for allreduce_dda_persistent");
    assert_eq!(allreduce_dda_persistent_schema.kernarg_size, 304);
    assert_eq!(allreduce_dda_persistent_schema.kernarg_segment_align, 8);
    assert_eq!(allreduce_dda_persistent_schema.fields.len(), 7);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[0].kernel_argument_name,
        "out"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[0].model_argument_name,
        "persistent_allreduce_output"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[0].offset, 0);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[1].kernel_argument_name,
        "ptrs"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[1].model_argument_name,
        "peer_allreduce_ptrs"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[1].offset, 8);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[2].kernel_argument_name,
        "ctrl"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[2].model_argument_name,
        "persistent_allreduce_ctrl"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[2].offset, 16);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[3].kernel_argument_name,
        "gbar"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[3].model_argument_name,
        "gbar"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[3].offset, 24);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[4].kernel_argument_name,
        "parts"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[4].model_argument_name,
        "group_size"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[4].offset, 32);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[5].kernel_argument_name,
        "n"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[5].model_argument_name,
        "n"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[5].offset, 36);
    assert_eq!(
        allreduce_dda_persistent_schema.fields[6].kernel_argument_name,
        "total_ops"
    );
    assert_eq!(
        allreduce_dda_persistent_schema.fields[6].model_argument_name,
        "persistent_allreduce_total_ops"
    );
    assert_eq!(allreduce_dda_persistent_schema.fields[6].offset, 40);
    let kv_append_semantic_schema = runtime_launch_kernel_argument_abi_semantic_schema_for(
        "kv_append_paged_fp4_from_f16_vf32_heads",
    )
    .unwrap();
    assert_eq!(
        kv_append_semantic_schema.kernel_symbol,
        "kv_append_paged_fp4_from_f16_vf32_heads"
    );
    assert_eq!(kv_append_semantic_schema.kernarg_size, 116);
    assert_eq!(
        kv_append_semantic_schema.fields[0].model_argument_name,
        "cache.key"
    );
    assert_eq!(
        kv_append_semantic_schema.fields[7].model_argument_name,
        "indptr"
    );
    let attention_split_semantic_schema = runtime_launch_kernel_argument_abi_semantic_schema_for(
        "attn_decode_split2_fp4_gqa_paged_groups_meta",
    )
    .unwrap();
    assert_eq!(attention_split_semantic_schema.kernarg_size, 100);
    assert_eq!(
        attention_split_semantic_schema.fields[0].model_argument_name,
        "query"
    );
    assert_eq!(
        attention_split_semantic_schema.fields[8].model_argument_name,
        "partials"
    );
    assert_eq!(
        attention_split_semantic_schema.fields[9].model_argument_name,
        "seq_len"
    );
    assert_eq!(
        attention_split_semantic_schema.fields[16].model_argument_name,
        "max_context"
    );
    let attention_combine_semantic_schema =
        runtime_launch_kernel_argument_abi_semantic_schema_for("attn_decode_combine_gqa_f16")
            .unwrap();
    assert_eq!(attention_combine_semantic_schema.kernarg_size, 24);
    assert_eq!(
        attention_combine_semantic_schema.fields[0].model_argument_name,
        "partials"
    );
    assert_eq!(
        attention_combine_semantic_schema.fields[1].model_argument_name,
        "output"
    );
    assert!(
        runtime_launch_kernel_argument_abi_semantic_schema_for("not_a_mainarch_kernel").is_none()
    );

    let model = ExternalToyDecoder::new();
    let catalog = MainarchPrimitiveLoweringCatalog::mi355_reference();
    let catalog_descriptor = catalog.descriptor();
    catalog_descriptor.assert_consistent()?;
    assert_eq!(catalog_descriptor.target, catalog.target);
    assert_eq!(catalog_descriptor.primitive_kind_count, 12);
    assert_eq!(catalog_descriptor.primitive_case_count, 22);
    assert_eq!(catalog_descriptor.native_gpu_case_count, 16);
    assert_eq!(catalog_descriptor.fused_native_gpu_case_count, 1);
    assert_eq!(catalog_descriptor.gap_case_count, 5);
    assert!(catalog_descriptor.parameterized);
    assert_eq!(
        catalog_descriptor
            .primitives
            .iter()
            .map(|primitive| primitive.label)
            .collect::<Vec<_>>(),
        model_primitive_kind_descriptors()
            .iter()
            .map(|descriptor| descriptor.label)
            .collect::<Vec<_>>()
    );
    let embedding_catalog = catalog_descriptor
        .primitive_for(PrimitiveKind::EmbeddingLookup)
        .unwrap();
    assert_eq!(embedding_catalog.cases.len(), 1);
    assert_eq!(
        embedding_catalog.cases[0].status,
        LoweringStatus::FusedNativeGpu
    );
    assert_eq!(
        embedding_catalog.cases[0].case_label,
        "fused_decode_embedding"
    );
    let rope_catalog = catalog_descriptor
        .primitive_for(PrimitiveKind::ApplyRope)
        .unwrap();
    assert_eq!(rope_catalog.gap_cases().len(), 1);
    assert_eq!(rope_catalog.gap_cases()[0].case_label, "decoupled_mla");
    let collective_catalog = catalog_descriptor
        .primitive_for(PrimitiveKind::Collective)
        .unwrap();
    assert_eq!(collective_catalog.gap_cases().len(), 2);

    let code_object = CodeObjectInfo::inspect(MAINARCH_KERNELS_GFX950)?;
    let catalog_code_object_coverage = catalog.code_object_kernel_coverage_report(&code_object)?;
    catalog_code_object_coverage.assert_complete()?;
    assert!(catalog_code_object_coverage.is_complete());
    assert_eq!(catalog_code_object_coverage.target, catalog.target);
    assert_eq!(catalog_code_object_coverage.primitive_case_count, 22);
    assert_eq!(catalog_code_object_coverage.non_gap_case_count, 17);
    assert_eq!(catalog_code_object_coverage.unmapped_entrypoint_count, 0);
    assert_eq!(catalog_code_object_coverage.missing_kernel_count, 0);
    assert_eq!(
        catalog_code_object_coverage.present_kernel_count,
        catalog_code_object_coverage.required_kernel_count
    );
    for symbol in [
        "gemv_f16",
        "paged_mla_fp8_splitk_stage1_probe",
        "paged_mla_fp8_splitk_stage2_merge_probe",
        "allreduce_direct_persistent",
    ] {
        assert!(
            catalog_code_object_coverage
                .required_kernel_symbols
                .contains(&symbol.to_string()),
            "catalog coverage should require {symbol}"
        );
    }
    let mla_coverage = catalog_code_object_coverage
        .entries
        .iter()
        .find(|entry| entry.entrypoint == "gpu_paged_mla_fp8_splitk_e2e_selftest")
        .expect("missing MLA e2e coverage entry");
    assert_eq!(
        mla_coverage.kernel_symbols,
        vec![
            "paged_mla_fp8_splitk_stage1_probe".to_string(),
            "paged_mla_fp8_splitk_stage2_merge_probe".to_string(),
        ]
    );
    assert!(mla_coverage.is_complete());
    let mut missing_symbol_coverage = catalog_code_object_coverage.clone();
    missing_symbol_coverage.missing_kernel_count = 1;
    missing_symbol_coverage.present_kernel_count -= 1;
    missing_symbol_coverage
        .missing_kernel_symbols
        .push("not_a_mainarch_kernel".to_string());
    missing_symbol_coverage.entries[0]
        .missing_kernel_symbols
        .push("not_a_mainarch_kernel".to_string());
    let missing_symbol_err = missing_symbol_coverage
        .assert_complete()
        .unwrap_err()
        .to_string();
    assert!(missing_symbol_err.contains("missing code-object kernels 1"));
    assert!(missing_symbol_err.contains("not_a_mainarch_kernel"));

    let catalog_abi_registry_coverage = catalog.abi_registry_coverage_report(&code_object)?;
    catalog_abi_registry_coverage.assert_complete()?;
    assert!(catalog_abi_registry_coverage.is_complete());
    assert_eq!(catalog_abi_registry_coverage.target, catalog.target);
    assert_eq!(
        catalog_abi_registry_coverage.required_kernel_count,
        catalog_code_object_coverage.required_kernel_count
    );
    assert_eq!(
        catalog_abi_registry_coverage.present_code_object_kernel_count,
        catalog_abi_registry_coverage.required_kernel_count
    );
    assert_eq!(
        catalog_abi_registry_coverage.covered_named_abi_schema_count,
        catalog_abi_registry_coverage.required_kernel_count
    );
    assert_eq!(
        catalog_abi_registry_coverage.covered_semantic_abi_schema_count,
        catalog_abi_registry_coverage.required_kernel_count
    );
    assert_eq!(
        catalog_abi_registry_coverage.missing_code_object_kernel_count,
        0
    );
    assert_eq!(
        catalog_abi_registry_coverage.missing_named_abi_schema_count,
        0
    );
    assert_eq!(
        catalog_abi_registry_coverage.missing_semantic_abi_schema_count,
        0
    );
    assert_eq!(
        catalog_abi_registry_coverage.named_code_object_shape_mismatch_count,
        0
    );
    assert_eq!(
        catalog_abi_registry_coverage.semantic_code_object_shape_mismatch_count,
        0
    );
    assert_eq!(
        catalog_abi_registry_coverage.named_semantic_shape_mismatch_count,
        0
    );
    assert!(catalog_abi_registry_coverage
        .required_kernel_symbols
        .contains(&"paged_mla_fp8_splitk_stage1_probe".to_string()));
    assert!(catalog_abi_registry_coverage
        .required_kernel_symbols
        .contains(&"paged_mla_fp8_splitk_stage2_merge_probe".to_string()));
    let mla_stage1_abi_entry = catalog_abi_registry_coverage
        .entries
        .iter()
        .find(|entry| entry.kernel_symbol == "paged_mla_fp8_splitk_stage1_probe")
        .expect("missing MLA stage1 ABI registry coverage entry");
    assert!(mla_stage1_abi_entry.is_complete());
    assert_eq!(mla_stage1_abi_entry.code_object_kernarg_size, Some(400));
    assert_eq!(mla_stage1_abi_entry.named_kernarg_size, Some(400));
    assert_eq!(mla_stage1_abi_entry.semantic_kernarg_size, Some(400));
    let mla_stage2_abi_entry = catalog_abi_registry_coverage
        .entries
        .iter()
        .find(|entry| entry.kernel_symbol == "paged_mla_fp8_splitk_stage2_merge_probe")
        .expect("missing MLA stage2 ABI registry coverage entry");
    assert!(mla_stage2_abi_entry.is_complete());
    assert_eq!(mla_stage2_abi_entry.code_object_kernarg_size, Some(304));
    assert_eq!(mla_stage2_abi_entry.named_kernarg_size, Some(304));
    assert_eq!(mla_stage2_abi_entry.semantic_kernarg_size, Some(304));
    let mut missing_named_abi_coverage = catalog_abi_registry_coverage.clone();
    missing_named_abi_coverage.covered_named_abi_schema_count -= 1;
    missing_named_abi_coverage.missing_named_abi_schema_count = 1;
    missing_named_abi_coverage
        .missing_named_abi_schema_symbols
        .push("not_a_mainarch_kernel".to_string());
    missing_named_abi_coverage.entries[0].named_abi_schema_present = false;
    let missing_named_abi_err = missing_named_abi_coverage
        .assert_complete()
        .unwrap_err()
        .to_string();
    assert!(missing_named_abi_err.contains("missing named ABI schemas 1"));
    assert!(missing_named_abi_err.contains("not_a_mainarch_kernel"));

    let mut mismatched_abi_coverage = catalog_abi_registry_coverage.clone();
    let mismatch_symbol = mismatched_abi_coverage.entries[0].kernel_symbol.clone();
    mismatched_abi_coverage.named_semantic_shape_match_count -= 1;
    mismatched_abi_coverage.named_semantic_shape_mismatch_count = 1;
    mismatched_abi_coverage
        .named_semantic_shape_mismatch_symbols
        .push(mismatch_symbol.clone());
    mismatched_abi_coverage.entries[0].named_semantic_shape_matches = false;
    let mismatched_abi_err = mismatched_abi_coverage
        .assert_complete()
        .unwrap_err()
        .to_string();
    assert!(mismatched_abi_err.contains("ABI schema shape mismatches 1"));
    assert!(mismatched_abi_err.contains(&mismatch_symbol));

    let plugin_inspection = inspect_model_plugin(&model, &catalog)?;
    assert!(plugin_inspection.is_accepted());
    plugin_inspection.assert_consistent()?;
    plugin_inspection.assert_accepted()?;
    let mut stale_accepted_plugin_inspection = plugin_inspection.clone();
    stale_accepted_plugin_inspection.primitive_vocabulary.pop();
    assert!(!stale_accepted_plugin_inspection.is_accepted());
    let stale_accepted_plugin_inspection_err = stale_accepted_plugin_inspection
        .assert_accepted()
        .unwrap_err()
        .to_string();
    assert!(stale_accepted_plugin_inspection_err.contains("not accepted"));
    assert!(stale_accepted_plugin_inspection_err.contains("consistency"));
    assert!(
        stale_accepted_plugin_inspection_err.contains("primitive vocabulary descriptors drifted")
    );
    assert!(plugin_inspection.is_static_handoff_ready());
    plugin_inspection.assert_static_handoff_ready()?;
    let public_device_pointer_bindings = plugin_inspection
        .readiness
        .slots
        .device_pointer_binding_template(DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE, 64)?;
    let public_device_pointer_validation = plugin_inspection
        .readiness
        .slots
        .validate_complete_device_pointer_bindings(&public_device_pointer_bindings);
    public_device_pointer_validation.assert_complete()?;
    let public_kfd_requests: ModelRuntimeSlotKfdAllocationResidencyRequestPlan = plugin_inspection
        .readiness
        .runtime_slot_kfd_allocation_residency_request_plan(
            &public_device_pointer_validation,
            &[777],
        );
    public_kfd_requests.assert_kfd_allocation_residency_request_ready()?;
    assert!(public_kfd_requests.is_kfd_allocation_residency_request_ready());
    assert_eq!(public_kfd_requests.request_count, 6);
    assert_eq!(public_kfd_requests.host_visible_gtt_request_count, 2);
    assert_eq!(public_kfd_requests.device_local_vram_request_count, 4);
    assert_eq!(public_kfd_requests.resident_gpu_ids, vec![777]);
    assert_eq!(public_kfd_requests.residency_map_request_count, 6);
    assert!(!public_kfd_requests.allocation_performed);
    assert!(!public_kfd_requests.residency_proven);
    assert!(!public_kfd_requests.live_execution_supported);
    let public_hidden_request: &RuntimeSlotKfdAllocationResidencyRequestEntry =
        public_kfd_requests.entry_for("hidden").unwrap();
    assert_eq!(
        public_hidden_request.allocation_kind,
        RuntimeSlotKfdAllocationKind::DeviceLocalVramPreferPublicCoherent
    );
    assert_eq!(
        public_hidden_request.allocation_kind.as_str(),
        "device_local_vram_prefer_public_coherent"
    );
    assert!(public_hidden_request.request_ready);
    let public_vm_requests: ModelRuntimeKfdVmAcquireRequestPlan =
        public_kfd_requests.kfd_vm_acquire_request_plan();
    public_vm_requests.assert_kfd_vm_acquire_request_ready()?;
    assert!(public_vm_requests.is_kfd_vm_acquire_request_ready());
    assert_eq!(public_vm_requests.resident_gpu_ids, vec![777]);
    assert_eq!(public_vm_requests.vm_acquire_request_count, 1);
    assert_eq!(public_vm_requests.kfd_fd_request_count, 1);
    assert_eq!(public_vm_requests.drm_fd_request_count, 1);
    assert_eq!(public_vm_requests.kfd_fd_bound_count, 0);
    assert_eq!(public_vm_requests.drm_fd_bound_count, 0);
    assert_eq!(public_vm_requests.vm_acquire_performed_count, 0);
    assert!(public_vm_requests.all_request_metadata_ready);
    assert!(!public_vm_requests.all_kfd_fds_bound);
    assert!(!public_vm_requests.all_drm_fds_bound);
    assert!(!public_vm_requests.all_vms_acquired);
    assert!(!public_vm_requests.live_execution_supported);
    let public_vm_request: &RuntimeKfdVmAcquireRequestEntry =
        public_vm_requests.entry_for_gpu_id(777).unwrap();
    assert!(public_vm_request.kfd_fd_required);
    assert!(public_vm_request.drm_fd_required);
    assert!(public_vm_request.vm_acquire_required);
    assert!(!public_vm_request.kfd_fd_bound);
    assert!(!public_vm_request.drm_fd_bound);
    assert!(!public_vm_request.vm_acquire_performed);
    assert!(public_vm_request.request_metadata_ready);
    let public_alloc_memory_requests: ModelRuntimeKfdAllocMemoryRequestPlan =
        public_kfd_requests.kfd_alloc_memory_request_plan(&public_vm_requests);
    public_alloc_memory_requests.assert_kfd_alloc_memory_request_ready()?;
    assert!(public_alloc_memory_requests.is_kfd_alloc_memory_request_ready());
    assert_eq!(public_alloc_memory_requests.resident_gpu_ids, vec![777]);
    assert_eq!(public_alloc_memory_requests.allocation_gpu_id, Some(777));
    assert_eq!(public_alloc_memory_requests.allocation_request_count, 6);
    assert_eq!(
        public_alloc_memory_requests.host_visible_gtt_request_count,
        2
    );
    assert_eq!(
        public_alloc_memory_requests.device_local_vram_request_count,
        4
    );
    assert_eq!(public_alloc_memory_requests.map_to_gpu_request_count, 6);
    assert_eq!(public_alloc_memory_requests.kfd_fd_bound_count, 0);
    assert_eq!(public_alloc_memory_requests.vm_acquire_performed_count, 0);
    assert_eq!(public_alloc_memory_requests.allocation_performed_count, 0);
    assert_eq!(public_alloc_memory_requests.handle_bound_count, 0);
    assert_eq!(public_alloc_memory_requests.mmap_offset_bound_count, 0);
    assert!(public_alloc_memory_requests.all_request_metadata_ready);
    assert!(!public_alloc_memory_requests.allocation_performed);
    assert!(!public_alloc_memory_requests.live_execution_supported);
    let public_hidden_alloc_memory: &RuntimeKfdAllocMemoryRequestEntry =
        public_alloc_memory_requests.entry_for("hidden").unwrap();
    assert_eq!(public_hidden_alloc_memory.allocation_gpu_id, 777);
    assert_eq!(
        public_hidden_alloc_memory.allocation_kind,
        RuntimeSlotKfdAllocationKind::DeviceLocalVramPreferPublicCoherent
    );
    assert!(public_hidden_alloc_memory.kfd_fd_required);
    assert!(public_hidden_alloc_memory.vm_acquire_required);
    assert!(public_hidden_alloc_memory.allocation_required);
    assert!(public_hidden_alloc_memory.map_to_gpu_required);
    assert!(!public_hidden_alloc_memory.kfd_fd_bound);
    assert!(!public_hidden_alloc_memory.vm_acquire_performed);
    assert!(!public_hidden_alloc_memory.allocation_performed);
    assert!(!public_hidden_alloc_memory.handle_bound);
    assert!(!public_hidden_alloc_memory.mmap_offset_bound);
    assert!(public_hidden_alloc_memory.request_metadata_ready);
    let public_result_bindings: Vec<RuntimeKfdAllocMemoryResultBinding> =
        public_alloc_memory_requests
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
                    0x7300_0000 + entry.slot as u64,
                    0x9300_0000 + entry.slot as u64 * 0x1000,
                )
            })
            .collect::<Result<_>>()?;
    let public_result_binding_plan: ModelRuntimeKfdAllocMemoryResultBindingPlan =
        public_alloc_memory_requests.kfd_alloc_memory_result_binding_plan(&public_result_bindings);
    public_result_binding_plan.assert_consistent()?;
    public_result_binding_plan.assert_kfd_alloc_memory_result_bound()?;
    assert!(public_result_binding_plan.is_kfd_alloc_memory_result_bound());
    assert_eq!(
        public_result_binding_plan.result_binding_count,
        public_alloc_memory_requests.allocation_request_count
    );
    assert_eq!(
        public_result_binding_plan.matched_result_binding_count,
        public_alloc_memory_requests.allocation_request_count
    );
    assert_eq!(public_result_binding_plan.missing_result_binding_count, 0);
    assert_eq!(public_result_binding_plan.duplicate_result_binding_count, 0);
    assert_eq!(public_result_binding_plan.unmatched_result_binding_count, 0);
    assert_eq!(
        public_result_binding_plan.handle_bound_count,
        public_alloc_memory_requests.allocation_request_count
    );
    assert_eq!(
        public_result_binding_plan.mmap_offset_bound_count,
        public_alloc_memory_requests.allocation_request_count
    );
    assert_eq!(
        public_result_binding_plan.allocation_performed_count,
        public_alloc_memory_requests.allocation_request_count
    );
    assert_eq!(
        public_result_binding_plan.primary_flag_result_count,
        public_alloc_memory_requests.allocation_request_count
    );
    assert_eq!(public_result_binding_plan.fallback_flag_result_count, 0);
    assert_eq!(public_result_binding_plan.request_issue_count, 0);
    assert!(public_result_binding_plan.alloc_memory_request_metadata_ready);
    assert!(public_result_binding_plan.all_result_bindings_ready);
    assert!(public_result_binding_plan.allocation_performed);
    assert!(!public_result_binding_plan.live_execution_supported);
    let public_hidden_result: &RuntimeKfdAllocMemoryResultBindingEntry =
        public_result_binding_plan.entry_for("hidden").unwrap();
    assert_eq!(public_hidden_result.slot, public_hidden_alloc_memory.slot);
    assert_eq!(public_hidden_result.allocation_gpu_id, 777);
    assert_eq!(
        public_hidden_result.handle,
        0x7300_0000 + public_hidden_result.slot as u64
    );
    assert_eq!(
        public_hidden_result.mmap_offset,
        0x9300_0000 + public_hidden_result.slot as u64 * 0x1000
    );
    assert!(public_hidden_result.result_binding_present);
    assert!(public_hidden_result.handle_bound);
    assert!(public_hidden_result.mmap_offset_bound);
    assert!(public_hidden_result.allocation_performed);
    assert!(public_hidden_result.result_metadata_ready);
    let public_result_issues: Vec<RuntimeKfdAllocMemoryResultBindingIssue> =
        public_result_binding_plan.issues.clone();
    assert!(public_result_issues.is_empty());
    assert_eq!(
        plugin_inspection
            .readiness
            .runtime_kfd_alloc_memory_result_binding_plan(
                &public_device_pointer_validation,
                &[777],
                &public_result_bindings,
            ),
        public_result_binding_plan
    );
    let public_map_memory_requests: ModelRuntimeKfdMapMemoryRequestPlan =
        public_alloc_memory_requests.kfd_map_memory_request_plan();
    public_map_memory_requests.assert_consistent()?;
    assert!(!public_map_memory_requests.is_kfd_map_memory_request_ready());
    assert_eq!(public_map_memory_requests.resident_gpu_ids, vec![777]);
    assert_eq!(public_map_memory_requests.map_memory_request_count, 6);
    assert_eq!(public_map_memory_requests.map_to_gpu_required_count, 6);
    assert_eq!(public_map_memory_requests.host_visible_gtt_request_count, 2);
    assert_eq!(
        public_map_memory_requests.device_local_vram_request_count,
        4
    );
    assert_eq!(public_map_memory_requests.total_device_id_count, 6);
    assert_eq!(public_map_memory_requests.kfd_fd_bound_count, 0);
    assert_eq!(public_map_memory_requests.vm_acquire_performed_count, 0);
    assert_eq!(public_map_memory_requests.allocation_performed_count, 0);
    assert_eq!(public_map_memory_requests.handle_bound_count, 0);
    assert_eq!(public_map_memory_requests.device_ids_array_bound_count, 0);
    assert_eq!(public_map_memory_requests.map_memory_performed_count, 0);
    assert_eq!(public_map_memory_requests.map_memory_success_count, 0);
    assert!(public_map_memory_requests.alloc_memory_request_metadata_ready);
    assert!(public_map_memory_requests.all_static_request_metadata_ready);
    assert!(!public_map_memory_requests.all_live_request_args_bound);
    assert!(!public_map_memory_requests.all_request_args_ready);
    assert!(!public_map_memory_requests.map_memory_performed);
    assert!(!public_map_memory_requests.live_execution_supported);
    let public_hidden_map_memory: &RuntimeKfdMapMemoryRequestEntry =
        public_map_memory_requests.entry_for("hidden").unwrap();
    assert_eq!(public_hidden_map_memory.allocation_gpu_id, 777);
    assert_eq!(
        public_hidden_map_memory.allocation_kind,
        RuntimeSlotKfdAllocationKind::DeviceLocalVramPreferPublicCoherent
    );
    assert_eq!(public_hidden_map_memory.map_args_handle, 0);
    assert_eq!(public_hidden_map_memory.map_args_device_ids_array_ptr, 0);
    assert_eq!(public_hidden_map_memory.map_args_n_devices, 1);
    assert_eq!(public_hidden_map_memory.map_args_n_success, 0);
    assert!(public_hidden_map_memory.handle_required);
    assert!(!public_hidden_map_memory.handle_bound);
    assert!(public_hidden_map_memory.device_ids_array_required);
    assert!(!public_hidden_map_memory.device_ids_array_bound);
    assert!(public_hidden_map_memory.request_static_metadata_ready);
    assert!(!public_hidden_map_memory.request_args_ready);
    let public_device_ids_array_bindings: Vec<RuntimeKfdMapMemoryDeviceIdsArrayBinding> =
        public_map_memory_requests
            .entries
            .iter()
            .map(|entry| {
                RuntimeKfdMapMemoryDeviceIdsArrayBinding::new(
                    entry.slot,
                    entry.tensor.as_str(),
                    0x3700_0000 + entry.slot as u64 * 0x100,
                    entry.resident_gpu_ids.clone(),
                )
            })
            .collect::<Result<_>>()?;
    let public_map_argument_plan: ModelRuntimeKfdMapMemoryArgumentBindingPlan =
        public_map_memory_requests.kfd_map_memory_argument_binding_plan(
            &public_result_binding_plan,
            &public_device_ids_array_bindings,
        );
    public_map_argument_plan.assert_consistent()?;
    public_map_argument_plan.assert_kfd_map_memory_arguments_bound()?;
    assert!(public_map_argument_plan.is_kfd_map_memory_arguments_bound());
    assert_eq!(
        public_map_argument_plan.map_memory_request_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.alloc_memory_result_entry_count,
        public_result_binding_plan.entries.len()
    );
    assert_eq!(
        public_map_argument_plan.device_ids_array_binding_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.matched_device_ids_array_binding_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.missing_device_ids_array_binding_count,
        0
    );
    assert_eq!(
        public_map_argument_plan.duplicate_device_ids_array_binding_count,
        0
    );
    assert_eq!(
        public_map_argument_plan.unmatched_device_ids_array_binding_count,
        0
    );
    assert_eq!(
        public_map_argument_plan.handle_bound_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.device_ids_array_bound_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.device_ids_array_match_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.allocation_performed_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.map_memory_argument_ready_count,
        public_map_memory_requests.map_memory_request_count
    );
    assert_eq!(
        public_map_argument_plan.total_device_id_count,
        public_map_memory_requests.total_device_id_count
    );
    assert_eq!(public_map_argument_plan.binding_issue_count, 0);
    assert!(public_map_argument_plan.map_memory_request_static_metadata_ready);
    assert!(public_map_argument_plan.alloc_memory_result_binding_ready);
    assert!(public_map_argument_plan.all_map_memory_arguments_ready);
    assert!(!public_map_argument_plan.map_memory_performed);
    assert!(!public_map_argument_plan.live_execution_supported);
    let public_hidden_map_arguments: &RuntimeKfdMapMemoryArgumentBindingEntry =
        public_map_argument_plan.entry_for("hidden").unwrap();
    assert_eq!(
        public_hidden_map_arguments.map_args_handle,
        public_hidden_result.handle
    );
    assert_eq!(
        public_hidden_map_arguments.map_args_device_ids_array_ptr,
        0x3700_0000 + public_hidden_map_arguments.slot as u64 * 0x100
    );
    assert_eq!(public_hidden_map_arguments.map_args_n_devices, 1);
    assert_eq!(public_hidden_map_arguments.device_ids, vec![777]);
    assert!(public_hidden_map_arguments.alloc_result_binding_present);
    assert!(public_hidden_map_arguments.alloc_result_metadata_ready);
    assert!(public_hidden_map_arguments.allocation_performed);
    assert!(public_hidden_map_arguments.handle_bound);
    assert!(public_hidden_map_arguments.device_ids_array_binding_present);
    assert!(public_hidden_map_arguments.device_ids_array_bound);
    assert!(public_hidden_map_arguments.device_ids_match_resident_gpu_ids);
    assert!(public_hidden_map_arguments.map_memory_argument_ready);
    let public_map_argument_issues: Vec<RuntimeKfdMapMemoryArgumentBindingIssue> =
        public_map_argument_plan.issues.clone();
    assert!(public_map_argument_issues.is_empty());
    assert_eq!(
        plugin_inspection
            .readiness
            .runtime_kfd_map_memory_argument_binding_plan(
                &public_device_pointer_validation,
                &[777],
                &public_result_bindings,
                &public_device_ids_array_bindings,
            ),
        public_map_argument_plan
    );
    let public_map_result_bindings: Vec<RuntimeKfdMapMemoryResultBinding> =
        public_map_argument_plan
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
            .collect::<Result<_>>()?;
    let public_map_result_plan: ModelRuntimeKfdMapMemoryResultBindingPlan =
        public_map_argument_plan.kfd_map_memory_result_binding_plan(&public_map_result_bindings);
    public_map_result_plan.assert_consistent()?;
    public_map_result_plan.assert_kfd_map_memory_result_bound()?;
    assert!(public_map_result_plan.is_kfd_map_memory_result_bound());
    assert_eq!(
        public_map_result_plan.map_memory_request_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.argument_binding_count,
        public_map_argument_plan.entries.len()
    );
    assert_eq!(
        public_map_result_plan.result_binding_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.matched_result_binding_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(public_map_result_plan.missing_result_binding_count, 0);
    assert_eq!(public_map_result_plan.duplicate_result_binding_count, 0);
    assert_eq!(public_map_result_plan.unmatched_result_binding_count, 0);
    assert_eq!(
        public_map_result_plan.handle_bound_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.device_ids_array_bound_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.device_ids_array_match_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.allocation_performed_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.map_memory_performed_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(
        public_map_result_plan.map_memory_success_count,
        public_map_argument_plan.map_memory_request_count
    );
    assert_eq!(public_map_result_plan.result_issue_count, 0);
    assert!(public_map_result_plan.map_memory_argument_binding_ready);
    assert!(public_map_result_plan.all_result_bindings_ready);
    assert!(public_map_result_plan.map_memory_performed);
    assert!(public_map_result_plan.residency_proven);
    assert!(!public_map_result_plan.live_execution_supported);
    let public_hidden_map_result: &RuntimeKfdMapMemoryResultBindingEntry =
        public_map_result_plan.entry_for("hidden").unwrap();
    assert_eq!(
        public_hidden_map_result.map_args_handle,
        public_hidden_map_arguments.map_args_handle
    );
    assert_eq!(
        public_hidden_map_result.result_map_args_handle,
        public_hidden_map_arguments.map_args_handle
    );
    assert_eq!(
        public_hidden_map_result.map_args_device_ids_array_ptr,
        public_hidden_map_arguments.map_args_device_ids_array_ptr
    );
    assert_eq!(
        public_hidden_map_result.result_map_args_device_ids_array_ptr,
        public_hidden_map_arguments.map_args_device_ids_array_ptr
    );
    assert_eq!(public_hidden_map_result.map_args_n_devices, 1);
    assert_eq!(public_hidden_map_result.result_map_args_n_devices, 1);
    assert_eq!(public_hidden_map_result.map_args_n_success, 0);
    assert_eq!(public_hidden_map_result.result_map_args_n_success, 1);
    assert_eq!(public_hidden_map_result.device_ids, vec![777]);
    assert!(public_hidden_map_result.argument_binding_ready);
    assert!(public_hidden_map_result.result_binding_present);
    assert!(public_hidden_map_result.handle_bound);
    assert!(public_hidden_map_result.device_ids_array_bound);
    assert!(public_hidden_map_result.device_ids_match_resident_gpu_ids);
    assert!(public_hidden_map_result.allocation_performed);
    assert!(public_hidden_map_result.map_memory_performed);
    assert!(public_hidden_map_result.map_memory_successful);
    assert!(public_hidden_map_result.result_metadata_ready);
    let public_map_result_issues: Vec<RuntimeKfdMapMemoryResultBindingIssue> =
        public_map_result_plan.issues.clone();
    assert!(public_map_result_issues.is_empty());
    assert_eq!(
        plugin_inspection
            .readiness
            .runtime_kfd_map_memory_result_binding_plan(
                &public_device_pointer_validation,
                &[777],
                &public_result_bindings,
                &public_device_ids_array_bindings,
                &public_map_result_bindings,
            ),
        public_map_result_plan
    );
    let public_residency_plan: ModelRuntimeSlotKfdResidencyBindingPlan =
        public_kfd_requests.kfd_residency_binding_plan(&public_map_result_plan);
    public_residency_plan.assert_consistent()?;
    public_residency_plan.assert_kfd_residency_bound()?;
    assert!(public_residency_plan.is_kfd_residency_bound());
    assert_eq!(
        public_residency_plan.request_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.map_memory_result_entry_count,
        public_map_result_plan.entries.len()
    );
    assert_eq!(
        public_residency_plan.matched_map_memory_result_count,
        public_kfd_requests.request_count
    );
    assert_eq!(public_residency_plan.missing_map_memory_result_count, 0);
    assert_eq!(
        public_residency_plan.device_pointer_bound_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.allocation_performed_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.handle_bound_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.device_ids_array_bound_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.map_memory_performed_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.map_memory_success_count,
        public_kfd_requests.request_count
    );
    assert_eq!(
        public_residency_plan.residency_proven_count,
        public_kfd_requests.request_count
    );
    assert_eq!(public_residency_plan.binding_issue_count, 0);
    assert!(public_residency_plan.allocation_residency_request_ready);
    assert!(public_residency_plan.map_memory_result_binding_ready);
    assert!(public_residency_plan.all_residency_bindings_ready);
    assert!(public_residency_plan.allocation_performed);
    assert!(public_residency_plan.residency_proven);
    assert!(!public_residency_plan.live_execution_supported);
    let public_hidden_residency: &RuntimeSlotKfdResidencyBindingEntry =
        public_residency_plan.entry_for("hidden").unwrap();
    assert_eq!(
        public_hidden_residency.device_va,
        public_kfd_requests.entry_for("hidden").unwrap().device_va
    );
    assert_eq!(
        public_hidden_residency.allocation_handle,
        public_hidden_map_result.map_args_handle
    );
    assert_eq!(
        public_hidden_residency.device_ids_array_ptr,
        public_hidden_map_result.map_args_device_ids_array_ptr
    );
    assert_eq!(public_hidden_residency.map_args_n_devices, 1);
    assert_eq!(public_hidden_residency.result_map_args_n_success, 1);
    assert!(public_hidden_residency.allocation_request_ready);
    assert!(public_hidden_residency.map_memory_result_present);
    assert!(public_hidden_residency.device_pointer_bound);
    assert!(public_hidden_residency.allocation_performed);
    assert!(public_hidden_residency.handle_bound);
    assert!(public_hidden_residency.device_ids_array_bound);
    assert!(public_hidden_residency.map_memory_performed);
    assert!(public_hidden_residency.map_memory_successful);
    assert!(public_hidden_residency.result_metadata_ready);
    assert!(public_hidden_residency.kfd_residency_proven);
    let public_residency_issues: Vec<RuntimeSlotKfdResidencyBindingIssue> =
        public_residency_plan.issues.clone();
    assert!(public_residency_issues.is_empty());
    assert_eq!(
        plugin_inspection
            .readiness
            .runtime_slot_kfd_residency_binding_plan(
                &public_device_pointer_validation,
                &[777],
                &public_result_bindings,
                &public_device_ids_array_bindings,
                &public_map_result_bindings,
            ),
        public_residency_plan
    );
    let public_checkpoint_resolution: ModelCheckpointKeyResolution = plugin_inspection
        .readiness
        .checkpoint
        .resolve_against_available_keys([
            "external.embed_tokens.weight",
            "external.lm_head.weight",
        ])?;
    public_checkpoint_resolution.assert_fully_resolved()?;
    let mut payload_offset = 0u64;
    let public_checkpoint_payload_bindings: Vec<RuntimeCheckpointPayloadBinding> =
        public_checkpoint_resolution
            .resolved_entries
            .iter()
            .map(|entry| {
                let key = entry
                    .matched_checkpoint_keys
                    .first()
                    .expect("external toy weights resolve to exact checkpoint keys");
                let binding = RuntimeCheckpointPayloadBinding::new(
                    entry.tensor.as_str(),
                    key.as_str(),
                    "external-toy.safetensors",
                    payload_offset,
                    entry.storage_bytes,
                    entry.dtype,
                    entry.shape.clone(),
                );
                payload_offset += entry.storage_bytes as u64;
                binding
            })
            .collect::<Result<_>>()?;
    let public_payload_plan: ModelRuntimeCheckpointPayloadBindingPlan = plugin_inspection
        .readiness
        .checkpoint
        .runtime_checkpoint_payload_binding_plan(
            &plugin_inspection.readiness.slots,
            &public_checkpoint_resolution,
            &public_residency_plan,
            &public_checkpoint_payload_bindings,
        );
    public_payload_plan.assert_consistent()?;
    public_payload_plan.assert_checkpoint_payload_bound()?;
    assert!(public_payload_plan.is_checkpoint_payload_bound());
    assert_eq!(public_payload_plan.checkpoint_entry_count, 2);
    assert_eq!(public_payload_plan.checkpoint_weight_slot_count, 2);
    assert_eq!(public_payload_plan.resolved_checkpoint_entry_count, 2);
    assert_eq!(public_payload_plan.missing_checkpoint_entry_count, 0);
    assert_eq!(public_payload_plan.unbound_weight_tensor_count, 0);
    assert_eq!(public_payload_plan.expected_payload_binding_count, 2);
    assert_eq!(public_payload_plan.payload_binding_count, 2);
    assert_eq!(public_payload_plan.matched_payload_binding_count, 2);
    assert_eq!(public_payload_plan.missing_payload_binding_count, 0);
    assert_eq!(public_payload_plan.duplicate_payload_binding_count, 0);
    assert_eq!(public_payload_plan.unmatched_payload_binding_count, 0);
    assert_eq!(public_payload_plan.kfd_residency_request_count, 6);
    assert_eq!(public_payload_plan.slot_binding_count, 2);
    assert_eq!(public_payload_plan.residency_proven_count, 2);
    assert_eq!(public_payload_plan.payload_metadata_ready_count, 2);
    assert_eq!(public_payload_plan.checkpoint_payload_bound_count, 2);
    assert_eq!(
        public_payload_plan.total_checkpoint_bytes,
        plugin_inspection
            .readiness
            .checkpoint
            .total_checkpoint_bytes
    );
    assert_eq!(
        public_payload_plan.total_payload_bytes,
        plugin_inspection
            .readiness
            .checkpoint
            .total_checkpoint_bytes
    );
    assert_eq!(public_payload_plan.binding_issue_count, 0);
    assert!(public_payload_plan.checkpoint_binding_ready);
    assert!(public_payload_plan.checkpoint_key_resolution_ready);
    assert!(public_payload_plan.kfd_residency_binding_ready);
    assert!(public_payload_plan.all_payload_bindings_ready);
    assert!(public_payload_plan.checkpoint_payloads_bound);
    assert!(!public_payload_plan.live_execution_supported);
    let public_embed_payload: &RuntimeCheckpointPayloadSlotBindingEntry =
        public_payload_plan.entry_for("embed_weight").unwrap();
    assert_eq!(public_embed_payload.matched_payload_key_count, 1);
    assert_eq!(public_embed_payload.missing_payload_key_count, 0);
    assert_eq!(public_embed_payload.payload_source_count, 1);
    assert_eq!(
        public_embed_payload.payload_bytes,
        public_embed_payload.storage_bytes
    );
    assert!(public_embed_payload.slot_binding_ready);
    assert!(public_embed_payload.kfd_residency_present);
    assert!(public_embed_payload.kfd_residency_proven);
    assert!(public_embed_payload.destination_span_bound);
    assert!(public_embed_payload.payload_metadata_ready);
    assert!(public_embed_payload.payload_byte_count_matches);
    assert!(public_embed_payload.checkpoint_payload_bound);
    let public_payload_issues: Vec<RuntimeCheckpointPayloadSlotBindingIssue> =
        public_payload_plan.issues.clone();
    assert!(public_payload_issues.is_empty());
    assert_eq!(
        plugin_inspection
            .readiness
            .runtime_checkpoint_payload_binding_plan(
                &public_device_pointer_validation,
                &[777],
                &public_result_bindings,
                &public_device_ids_array_bindings,
                &public_map_result_bindings,
                &public_checkpoint_resolution,
                &public_checkpoint_payload_bindings,
            ),
        public_payload_plan
    );
    let public_synthetic_payload_plan: ModelRuntimeCheckpointPayloadBindingPlan = plugin_inspection
        .synthetic_cpu_runtime_checkpoint_payload_binding_plan(
            ["external.embed_tokens.weight", "external.lm_head.weight"],
            "external-toy.safetensors",
            &[777],
        )?;
    public_synthetic_payload_plan.assert_checkpoint_payload_bound()?;
    assert_eq!(
        public_synthetic_payload_plan.checkpoint_payload_bound_count,
        public_payload_plan.checkpoint_payload_bound_count
    );
    assert_eq!(
        public_synthetic_payload_plan.expected_payload_binding_count,
        public_payload_plan.expected_payload_binding_count
    );
    assert_eq!(
        public_synthetic_payload_plan.total_payload_bytes,
        public_payload_plan.total_payload_bytes
    );
    assert!(!public_synthetic_payload_plan.live_execution_supported);
    let public_map_memory_err = public_map_memory_requests
        .assert_kfd_map_memory_request_ready()
        .unwrap_err()
        .to_string();
    assert!(public_map_memory_err.contains("allocation handles are not bound"));
    assert!(public_map_memory_err.contains("device ID arrays are not bound"));
    let mut stale_vocabulary_static_handoff_inspection = plugin_inspection.clone();
    stale_vocabulary_static_handoff_inspection
        .primitive_vocabulary
        .pop();
    assert!(!stale_vocabulary_static_handoff_inspection.is_static_handoff_ready());
    let stale_vocabulary_static_handoff_err = stale_vocabulary_static_handoff_inspection
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_vocabulary_static_handoff_err.contains("not static handoff ready"));
    assert!(stale_vocabulary_static_handoff_err.contains("consistency"));
    assert!(
        stale_vocabulary_static_handoff_err.contains("primitive vocabulary descriptors drifted")
    );
    let mut stale_compatibility_plugin_inspection = plugin_inspection.clone();
    stale_compatibility_plugin_inspection.compatibility.accepted = false;
    assert!(!stale_compatibility_plugin_inspection.is_static_handoff_ready());
    let stale_compatibility_err = stale_compatibility_plugin_inspection
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_compatibility_err.contains("not static handoff ready"));
    assert!(stale_compatibility_err.contains("consistency"));
    assert!(
        stale_compatibility_err.contains("compatibility report does not match manifest-derived")
    );
    let mut stale_readiness_plugin_inspection = plugin_inspection.clone();
    stale_readiness_plugin_inspection
        .readiness
        .checkpoint
        .missing_weight_tensors
        .push("unbound_static_handoff_weight".into());
    assert!(!stale_readiness_plugin_inspection.is_static_handoff_ready());
    let stale_readiness_err = stale_readiness_plugin_inspection
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_readiness_err.contains("not static handoff ready"));
    assert!(stale_readiness_err.contains("manifest does not match readiness-derived"));
    let plugin_summary = plugin_inspection.summary();
    plugin_summary.assert_consistent_with(&plugin_inspection)?;
    assert_eq!(plugin_summary.contract, MODEL_API_CONTRACT);
    assert_eq!(plugin_summary.model_name, "external-toy-decoder");
    assert_eq!(plugin_summary.target, catalog.target);
    assert_eq!(plugin_summary.expected_target, catalog.target);
    assert!(plugin_summary.accepted);
    assert!(plugin_summary.static_ready);
    assert_eq!(plugin_summary.static_issue_count, 0);
    assert_eq!(plugin_summary.compatibility_issue_count, 0);
    assert_eq!(plugin_summary.primitive_vocabulary_count, 12);
    assert_eq!(plugin_summary.stage_vocabulary_count, 5);
    assert_eq!(plugin_summary.catalog_primitive_kind_count, 12);
    assert_eq!(plugin_summary.catalog_primitive_case_count, 22);
    assert_eq!(plugin_summary.catalog_native_gpu_case_count, 16);
    assert_eq!(plugin_summary.catalog_fused_native_gpu_case_count, 1);
    assert_eq!(plugin_summary.catalog_gap_case_count, 5);
    assert!(plugin_summary.catalog_parameterized);
    assert_eq!(plugin_summary.model_primitive_kind_count, 3);
    assert_eq!(plugin_summary.model_stage_kind_count, 3);
    assert_eq!(plugin_summary.tensor_count, 6);
    assert_eq!(plugin_summary.op_count, 3);
    assert_eq!(plugin_summary.stage_count, 3);
    assert_eq!(plugin_summary.checkpoint_weight_count, 2);
    assert_eq!(plugin_summary.missing_checkpoint_weight_count, 0);
    assert_eq!(plugin_summary.runtime_slot_count, 6);
    assert_eq!(plugin_summary.runtime_dispatch_count, 3);
    assert_eq!(
        plugin_summary.runtime_launch_request_step_count,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len()
    );
    assert_eq!(
        plugin_summary.runtime_launch_request_step_count,
        EXPECTED_RUNTIME_LAUNCH_REQUEST_STEP_COUNT
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len(),
        EXPECTED_RUNTIME_LAUNCH_REQUEST_STEP_COUNT
    );
    assert_eq!(plugin_summary.runtime_live_aql_proof_step_count, 2);
    assert_eq!(plugin_summary.runtime_live_queue_mutating_step_count, 0);
    assert!(!plugin_summary.live_execution_supported);
    assert_eq!(plugin_summary.contract_fingerprint.len(), 64);
    let plugin_summary_receipt_lines = plugin_summary.receipt_lines();
    assert_eq!(
        plugin_summary_receipt_lines[0],
        "receipt.kind=model_plugin_inspection_summary"
    );
    assert!(plugin_summary_receipt_lines
        .iter()
        .any(|line| line == "model_name=external-toy-decoder"));
    assert!(plugin_summary.receipt_text().ends_with('\n'));
    let plugin_summary_receipt_fingerprint = plugin_summary.receipt_fingerprint();
    assert_eq!(plugin_summary_receipt_fingerprint.len(), 64);
    assert!(plugin_summary_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_summary = plugin_summary.clone();
    stale_summary.tensor_count += 1;
    assert_ne!(
        stale_summary.receipt_fingerprint(),
        plugin_summary_receipt_fingerprint
    );
    assert!(stale_summary
        .assert_consistent_with(&plugin_inspection)
        .unwrap_err()
        .to_string()
        .contains("tensor count"));
    let plugin_manifest_receipt_lines = plugin_inspection.manifest.receipt_lines();
    assert_eq!(
        plugin_manifest_receipt_lines[0],
        "receipt.kind=model_plugin_manifest"
    );
    assert!(plugin_manifest_receipt_lines
        .iter()
        .any(|line| line == "model_name=external-toy-decoder"));
    assert!(plugin_manifest_receipt_lines.iter().any(|line| line
        == &format!(
            "manifest_fingerprint={}",
            plugin_inspection.manifest.contract_fingerprint
        )));
    assert!(plugin_manifest_receipt_lines
        .iter()
        .any(|line| line == "fingerprint_matches=true"));
    assert!(plugin_manifest_receipt_lines
        .iter()
        .any(|line| line == "runtime_launch_request_steps.count=10"));
    assert!(plugin_inspection.manifest.receipt_text().ends_with('\n'));
    let plugin_manifest_receipt_fingerprint = plugin_inspection.manifest.receipt_fingerprint();
    assert_eq!(plugin_manifest_receipt_fingerprint.len(), 64);
    assert!(plugin_manifest_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_manifest = plugin_inspection.manifest.clone();
    stale_manifest.tensor_count += 1;
    assert_ne!(
        stale_manifest.receipt_fingerprint(),
        plugin_manifest_receipt_fingerprint
    );
    assert!(stale_manifest
        .receipt_lines()
        .iter()
        .any(|line| line == "fingerprint_matches=false"));
    let plugin_rejection = plugin_inspection.rejection_report();
    plugin_rejection.assert_consistent()?;
    plugin_rejection.assert_consistent_with(&plugin_inspection)?;
    plugin_rejection.assert_no_rejection()?;
    let plugin_rejection_receipt_lines = plugin_rejection.receipt_lines();
    assert_eq!(
        plugin_rejection_receipt_lines[0],
        "receipt.kind=model_plugin_rejection"
    );
    assert!(plugin_rejection_receipt_lines.iter().any(|line| line
        == &format!(
            "summary.receipt_fingerprint={}",
            plugin_summary_receipt_fingerprint
        )));
    assert!(plugin_rejection.receipt_text().ends_with('\n'));
    let plugin_rejection_receipt_fingerprint = plugin_rejection.receipt_fingerprint();
    assert_eq!(plugin_rejection_receipt_fingerprint.len(), 64);
    assert!(!plugin_rejection.is_rejected());
    assert_eq!(plugin_rejection.rejection_issue_count, 0);
    assert!(plugin_rejection.readiness_issues.issues.is_empty());
    assert!(plugin_rejection.compatibility_issues.is_empty());
    assert!(plugin_rejection.lowering_gap_op_names.is_empty());
    assert!(plugin_rejection.stage_gap_names.is_empty());
    assert!(plugin_rejection.unstaged_op_names.is_empty());
    assert!(plugin_rejection.missing_checkpoint_weight_names.is_empty());
    assert!(plugin_rejection.binding_issue_tensor_names.is_empty());
    let mut stale_no_rejection = plugin_rejection.clone();
    stale_no_rejection.summary.accepted = false;
    assert!(!stale_no_rejection.is_rejected());
    let stale_no_rejection_err = stale_no_rejection
        .assert_rejected()
        .unwrap_err()
        .to_string();
    assert!(stale_no_rejection_err.contains("not rejected"));
    assert!(stale_no_rejection_err.contains("summary accepted false != expected true"));
    let stale_no_rejection_clear_err = stale_no_rejection
        .assert_no_rejection()
        .unwrap_err()
        .to_string();
    assert!(stale_no_rejection_clear_err.contains("has rejection issue"));
    assert!(stale_no_rejection_clear_err.contains("consistency"));
    assert!(stale_no_rejection_clear_err.contains("summary accepted false != expected true"));
    assert_eq!(
        plugin_inspection.primitive_vocabulary,
        model_primitive_kind_descriptors().to_vec()
    );
    assert_eq!(
        plugin_inspection.stage_vocabulary,
        model_stage_kind_descriptors().to_vec()
    );
    assert_eq!(plugin_inspection.catalog, catalog_descriptor);
    let mut stale_vocabulary_inspection = plugin_inspection.clone();
    stale_vocabulary_inspection.primitive_vocabulary.pop();
    assert!(stale_vocabulary_inspection
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("primitive vocabulary descriptors drifted"));
    let mut stale_manifest_inspection = plugin_inspection.clone();
    stale_manifest_inspection.manifest.tensor_count += 1;
    assert!(stale_manifest_inspection
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("manifest does not match readiness-derived manifest"));
    let mut stale_catalog_inspection = plugin_inspection.clone();
    stale_catalog_inspection.catalog.primitive_case_count += 1;
    assert!(stale_catalog_inspection
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("catalog descriptor failed consistency"));
    let mut stale_compatibility_inspection = plugin_inspection.clone();
    stale_compatibility_inspection.compatibility.target = "stale-target".to_string();
    assert!(stale_compatibility_inspection
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("compatibility report does not match manifest-derived compatibility"));
    let code_object = CodeObjectInfo::inspect(MAINARCH_KERNELS_GFX950)?;
    let plugin_static_handoff =
        plugin_inspection.synthetic_cpu_static_handoff_receipt("external")?;
    let explicit_plugin_static_handoff = plugin_inspection.static_handoff_receipt(
        "external",
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    assert_eq!(plugin_static_handoff, explicit_plugin_static_handoff);
    assert!(plugin_inspection
        .synthetic_cpu_static_handoff_receipt("")
        .unwrap_err()
        .to_string()
        .contains("model plugin static handoff namespace"));
    let mut stale_static_handoff_compatibility_inspection = plugin_inspection.clone();
    stale_static_handoff_compatibility_inspection
        .compatibility
        .accepted = false;
    let stale_static_handoff_compatibility_err = stale_static_handoff_compatibility_inspection
        .static_handoff_receipt(
            "external",
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
    let mut stale_static_handoff_readiness_inspection = plugin_inspection.clone();
    stale_static_handoff_readiness_inspection
        .readiness
        .checkpoint
        .missing_weight_tensors
        .push("unbound_static_handoff_receipt_weight".into());
    let stale_static_handoff_readiness_err = stale_static_handoff_readiness_inspection
        .static_handoff_receipt(
            "external",
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
    plugin_static_handoff.assert_consistent()?;
    plugin_static_handoff.assert_consistent_with(
        &plugin_inspection,
        "external",
        &code_object,
        DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    let plugin_static_handoff_receipt_lines = plugin_static_handoff.receipt_lines();
    assert_eq!(
        plugin_static_handoff_receipt_lines[0],
        "receipt.kind=model_plugin_static_handoff"
    );
    assert_eq!(plugin_static_handoff_receipt_lines[1], "receipt.version=1");
    assert!(plugin_static_handoff_receipt_lines
        .iter()
        .any(|line| line == "model_name=external-toy-decoder"));
    assert!(plugin_static_handoff_receipt_lines.iter().any(|line| line
        == &format!(
            "summary.receipt_fingerprint={}",
            plugin_summary_receipt_fingerprint
        )));
    assert!(plugin_static_handoff_receipt_lines.iter().any(|line| line
        == &format!(
            "manifest.receipt_fingerprint={}",
            plugin_inspection.manifest.receipt_fingerprint()
        )));
    assert!(plugin_static_handoff_receipt_lines.iter().any(|line| line
        == &format!(
            "compatibility.receipt_fingerprint={}",
            plugin_inspection.compatibility.receipt_fingerprint()
        )));
    assert!(plugin_static_handoff_receipt_lines
        .iter()
        .any(|line| line == "metadata_admitted=true"));
    assert!(plugin_static_handoff_receipt_lines
        .iter()
        .any(|line| line == "launch_execution.executable=false"));
    assert_eq!(
        plugin_static_handoff.summary_receipt_fingerprint,
        plugin_summary_receipt_fingerprint
    );
    assert_eq!(
        plugin_static_handoff.rejection_receipt_fingerprint,
        plugin_rejection_receipt_fingerprint
    );
    assert_eq!(
        plugin_static_handoff.manifest_fingerprint,
        plugin_inspection.manifest.contract_fingerprint
    );
    assert_eq!(
        plugin_static_handoff.manifest_receipt_fingerprint,
        plugin_inspection.manifest.receipt_fingerprint()
    );
    assert_eq!(
        plugin_static_handoff.compatibility_receipt_fingerprint,
        plugin_inspection.compatibility.receipt_fingerprint()
    );
    assert!(plugin_static_handoff.accepted);
    assert!(plugin_static_handoff.static_ready);
    assert_eq!(plugin_static_handoff.compatibility_issue_count, 0);
    assert_eq!(plugin_static_handoff.model_primitive_kind_count, 3);
    assert_eq!(plugin_static_handoff.model_stage_kind_count, 3);
    assert_eq!(plugin_static_handoff.tensor_count, 6);
    assert_eq!(plugin_static_handoff.op_count, 3);
    assert_eq!(plugin_static_handoff.runtime_slot_count, 6);
    assert_eq!(plugin_static_handoff.runtime_dispatch_count, 3);
    assert!(plugin_static_handoff.metadata_binding_complete);
    assert!(plugin_static_handoff.device_pointer_binding_complete);
    assert!(plugin_static_handoff.metadata_admitted);
    assert_eq!(
        plugin_static_handoff.launch_projection_dispatches_with_ready_candidate_count
            + plugin_static_handoff.launch_projection_dispatches_without_ready_candidate_count,
        plugin_static_handoff.runtime_dispatch_count
    );
    assert_eq!(
        plugin_static_handoff.projection_selection_request_count
            + plugin_static_handoff.projection_selection_missing_request_count,
        plugin_static_handoff.runtime_dispatch_count
    );
    assert!(plugin_static_handoff.projection_selection_request_plan_ready);
    assert!(!plugin_static_handoff.projection_selection_all_requests_ready);
    assert!(!plugin_static_handoff.launch_execution_executable);
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
        plugin_static_handoff.unresolved_runtime_requirements,
        expected_static_handoff_requirements
    );
    assert_eq!(
        plugin_static_handoff.unresolved_runtime_requirement_names(),
        expected_static_handoff_requirements
    );
    assert!(plugin_static_handoff.has_unresolved_runtime_requirement("queue_reservation"));
    assert!(!plugin_static_handoff.has_unresolved_runtime_requirement("runtime_request_components"));
    assert_eq!(
        plugin_static_handoff
            .unresolved_runtime_requirement_names()
            .join(","),
        "kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,kernel_argument_abi_semantic_projection,completion_signal_binding,queue_reservation,aql_packet_materialization"
    );
    assert_eq!(
        plugin_static_handoff.aql_packet_materialization_dispatchable_packet_count,
        0
    );
    assert_eq!(
        plugin_static_handoff.runtime_request_plan_count,
        EXPECTED_RUNTIME_LAUNCH_REQUEST_STEP_COUNT
    );
    assert_eq!(plugin_static_handoff.live_aql_submitting_surface_count, 0);
    assert_eq!(plugin_static_handoff.live_queue_mutating_component_count, 0);
    assert!(!plugin_static_handoff.live_execution_supported);
    assert!(!plugin_static_handoff.gpu_buffers_allocated);
    assert!(!plugin_static_handoff.kernels_submitted);
    assert!(plugin_static_handoff.is_non_executing_boundary());
    plugin_static_handoff.assert_non_executing_boundary()?;
    let mut stale_count_plugin_static_handoff = plugin_static_handoff.clone();
    stale_count_plugin_static_handoff.unresolved_runtime_requirement_count = 0;
    assert!(!stale_count_plugin_static_handoff.is_non_executing_boundary());
    let stale_count_plugin_static_handoff_err = stale_count_plugin_static_handoff
        .assert_non_executing_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_plugin_static_handoff_err.contains("consistency"));
    assert!(stale_count_plugin_static_handoff_err.contains("unresolved runtime requirement"));
    let assert_non_execution_rejected = |receipt: ModelPluginStaticHandoffReceipt, needle: &str| {
        assert!(!receipt.is_non_executing_boundary());
        assert!(receipt
            .assert_non_executing_boundary()
            .unwrap_err()
            .to_string()
            .contains(needle));
    };
    let mut executable_static_handoff = plugin_static_handoff.clone();
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
    let mut dispatchable_static_handoff = plugin_static_handoff.clone();
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
    let mut submitting_static_handoff = plugin_static_handoff.clone();
    submitting_static_handoff.live_aql_submitting_surface_count = 1;
    assert_non_execution_rejected(
        submitting_static_handoff,
        "live AQL submitting surfaces 1 != 0",
    );
    let mut queue_mutating_static_handoff = plugin_static_handoff.clone();
    queue_mutating_static_handoff.live_queue_mutating_component_count = 1;
    assert_non_execution_rejected(
        queue_mutating_static_handoff,
        "live queue mutating components 1 != 0",
    );
    let mut live_supported_static_handoff = plugin_static_handoff.clone();
    live_supported_static_handoff.live_execution_supported = true;
    assert_non_execution_rejected(
        live_supported_static_handoff,
        "live execution support is true",
    );
    let mut allocated_static_handoff = plugin_static_handoff.clone();
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
    let mut submitted_static_handoff = plugin_static_handoff.clone();
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
    let mut requirement_count_mismatch_static_handoff = plugin_static_handoff.clone();
    requirement_count_mismatch_static_handoff.unresolved_runtime_requirement_count -= 1;
    assert!(requirement_count_mismatch_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("unresolved runtime requirement labels 9 != count 8"));
    let mut empty_requirement_static_handoff = plugin_static_handoff.clone();
    empty_requirement_static_handoff.unresolved_runtime_requirements[0] = String::new();
    assert!(empty_requirement_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("unresolved runtime requirement label is empty"));
    let mut whitespace_requirement_static_handoff = plugin_static_handoff.clone();
    whitespace_requirement_static_handoff.unresolved_runtime_requirements[0] =
        "runtime requirement".to_string();
    assert!(whitespace_requirement_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("contains whitespace"));
    let mut duplicate_requirement_static_handoff = plugin_static_handoff.clone();
    duplicate_requirement_static_handoff.unresolved_runtime_requirements[1] =
        duplicate_requirement_static_handoff.unresolved_runtime_requirements[0].clone();
    assert!(duplicate_requirement_static_handoff
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("appears more than once"));
    assert!(plugin_static_handoff.receipt_text().ends_with('\n'));
    let plugin_static_handoff_receipt_fingerprint = plugin_static_handoff.receipt_fingerprint();
    assert_eq!(plugin_static_handoff_receipt_fingerprint.len(), 64);
    assert!(plugin_static_handoff_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_static_handoff = plugin_static_handoff.clone();
    stale_static_handoff.tensor_count += 1;
    assert_ne!(
        stale_static_handoff.receipt_fingerprint(),
        plugin_static_handoff_receipt_fingerprint
    );
    assert!(stale_static_handoff
        .assert_consistent_with(
            &plugin_inspection,
            "external",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string()
        .contains("tensor count"));
    let mut stale_requirement_static_handoff = plugin_static_handoff.clone();
    stale_requirement_static_handoff.unresolved_runtime_requirements[0] =
        "stale_runtime_requirement".to_string();
    assert_ne!(
        stale_requirement_static_handoff.receipt_fingerprint(),
        plugin_static_handoff_receipt_fingerprint
    );
    assert!(stale_requirement_static_handoff
        .assert_consistent_with(
            &plugin_inspection,
            "external",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string()
        .contains("unresolved runtime requirements"));
    let mut stale_manifest_receipt_static_handoff = plugin_static_handoff.clone();
    stale_manifest_receipt_static_handoff.manifest_receipt_fingerprint =
        plugin_summary_receipt_fingerprint.clone();
    assert!(stale_manifest_receipt_static_handoff
        .assert_consistent_with(
            &plugin_inspection,
            "external",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string()
        .contains("manifest receipt fingerprint"));
    let mut stale_compatibility_receipt_static_handoff = plugin_static_handoff.clone();
    stale_compatibility_receipt_static_handoff.compatibility_receipt_fingerprint =
        plugin_summary_receipt_fingerprint.clone();
    assert!(stale_compatibility_receipt_static_handoff
        .assert_consistent_with(
            &plugin_inspection,
            "external",
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
            DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
        )
        .unwrap_err()
        .to_string()
        .contains("compatibility receipt fingerprint"));
    let plugin_inspection_for_default_helpers = plugin_inspection.clone();
    let graph = plugin_inspection.graph;
    let readiness = plugin_inspection.readiness;
    let plugin_manifest = plugin_inspection.manifest;
    let plugin_compatibility = plugin_inspection.compatibility;

    readiness.assert_static_runtime_ready()?;
    assert_eq!(plugin_manifest, graph.plugin_manifest(&catalog)?);
    assert_eq!(graph, build_model_graph(&model)?);
    assert_eq!(readiness, graph.readiness_report(&catalog)?);
    assert_eq!(plugin_manifest, readiness.plugin_manifest(&graph.name)?);
    assert_eq!(
        plugin_compatibility,
        plugin_manifest.compatibility_report(&catalog)
    );
    plugin_manifest.assert_static_metadata_ready()?;
    assert_eq!(plugin_manifest.contract, MODEL_API_CONTRACT);
    assert_eq!(plugin_manifest.model_name, graph.name);
    assert_eq!(plugin_manifest.target, catalog.target);
    assert_eq!(
        plugin_manifest.primitive_kinds,
        vec![
            PrimitiveKind::EmbeddingLookup,
            PrimitiveKind::Linear,
            PrimitiveKind::ArgmaxSample
        ]
    );
    assert_eq!(
        plugin_manifest.primitive_kind_labels,
        vec!["embedding_lookup", "linear", "argmax_sample"]
    );
    assert_eq!(
        plugin_manifest.stage_kinds,
        vec![
            ModelStageKind::Embedding,
            ModelStageKind::Output,
            ModelStageKind::Sampling
        ]
    );
    assert_eq!(
        plugin_manifest.stage_kind_labels,
        vec!["embedding", "output", "sampling"]
    );
    assert_eq!(plugin_manifest.tensor_count, 6);
    assert_eq!(plugin_manifest.op_count, 3);
    assert_eq!(plugin_manifest.stage_count, 3);
    assert_eq!(plugin_manifest.checkpoint_weight_count, 2);
    assert_eq!(plugin_manifest.missing_checkpoint_weight_count, 0);
    assert_eq!(plugin_manifest.runtime_slot_count, 6);
    assert_eq!(plugin_manifest.runtime_dispatch_count, 3);
    assert_eq!(plugin_manifest.static_issue_count, 0);
    assert!(plugin_manifest.static_ready);
    assert_eq!(plugin_manifest.lowering_gap_count, 0);
    assert_eq!(plugin_manifest.unstaged_op_count, 0);
    assert_eq!(plugin_manifest.binding_issue_count, 0);
    assert_eq!(
        plugin_manifest.runtime_launch_request_steps,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.to_vec()
    );
    assert_eq!(
        plugin_manifest.runtime_launch_request_step_labels,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.request_plan)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        plugin_manifest.runtime_launch_request_step_count,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len()
    );
    assert_eq!(
        plugin_manifest.runtime_launch_request_step_count,
        EXPECTED_RUNTIME_LAUNCH_REQUEST_STEP_COUNT
    );
    assert_eq!(plugin_manifest.runtime_live_aql_proof_step_count, 2);
    assert_eq!(plugin_manifest.runtime_live_queue_mutating_step_count, 0);
    assert!(!plugin_manifest.live_execution_supported);
    assert_eq!(plugin_manifest.contract_fingerprint.len(), 64);
    assert!(plugin_manifest
        .contract_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    assert_eq!(
        plugin_manifest.contract_fingerprint,
        plugin_manifest.expected_contract_fingerprint()
    );
    let mut stale_static_metadata_manifest = plugin_manifest.clone();
    stale_static_metadata_manifest.contract_fingerprint = "0".repeat(64);
    let stale_static_metadata_manifest_err = stale_static_metadata_manifest
        .assert_static_metadata_ready()
        .unwrap_err()
        .to_string();
    assert!(stale_static_metadata_manifest_err.contains("not static metadata ready"));
    assert!(stale_static_metadata_manifest_err.contains("consistency"));
    assert!(stale_static_metadata_manifest_err.contains("contract fingerprint"));
    let plugin_compatibility = plugin_manifest.compatibility_report(&catalog);
    assert!(plugin_compatibility.is_accepted());
    plugin_compatibility.assert_consistent()?;
    plugin_compatibility.assert_accepted()?;
    assert_eq!(plugin_compatibility.model_name, graph.name);
    assert_eq!(plugin_compatibility.contract, MODEL_API_CONTRACT);
    assert_eq!(plugin_compatibility.expected_contract, MODEL_API_CONTRACT);
    assert!(plugin_compatibility.contract_matches);
    assert_eq!(plugin_compatibility.target, catalog.target);
    assert_eq!(plugin_compatibility.expected_target, catalog.target);
    assert!(plugin_compatibility.target_matches);
    assert_eq!(
        plugin_compatibility.contract_fingerprint,
        plugin_manifest.contract_fingerprint
    );
    assert_eq!(
        plugin_compatibility.expected_contract_fingerprint,
        plugin_manifest.expected_contract_fingerprint()
    );
    assert!(plugin_compatibility.fingerprint_matches);
    assert!(plugin_compatibility.static_metadata_ready);
    assert!(!plugin_compatibility.live_execution_supported);
    assert!(plugin_compatibility.issues.is_empty());
    let plugin_compatibility_receipt_lines = plugin_compatibility.receipt_lines();
    assert_eq!(
        plugin_compatibility_receipt_lines[0],
        "receipt.kind=model_plugin_compatibility"
    );
    let expected_plugin_compatibility_model_line = format!("model_name={}", graph.name);
    assert!(plugin_compatibility_receipt_lines
        .iter()
        .any(|line| line == &expected_plugin_compatibility_model_line));
    assert!(plugin_compatibility_receipt_lines
        .iter()
        .any(|line| line == "contract_matches=true"));
    assert!(plugin_compatibility_receipt_lines
        .iter()
        .any(|line| line == "target_matches=true"));
    assert!(plugin_compatibility_receipt_lines
        .iter()
        .any(|line| line == "fingerprint_matches=true"));
    assert!(plugin_compatibility_receipt_lines
        .iter()
        .any(|line| line == "accepted=true"));
    assert!(plugin_compatibility_receipt_lines
        .iter()
        .any(|line| line == "issues.count=0"));
    assert!(plugin_compatibility.receipt_text().ends_with('\n'));
    let plugin_compatibility_receipt_fingerprint = plugin_compatibility.receipt_fingerprint();
    assert_eq!(plugin_compatibility_receipt_fingerprint.len(), 64);
    assert!(plugin_compatibility_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    plugin_manifest.assert_compatible_with(&catalog)?;

    let wrong_target_catalog = MainarchPrimitiveLoweringCatalog {
        target: "different-raw-aql-target",
    };
    let wrong_target_compatibility = plugin_manifest.compatibility_report(&wrong_target_catalog);
    assert!(!wrong_target_compatibility.is_accepted());
    wrong_target_compatibility.assert_consistent()?;
    assert!(!wrong_target_compatibility.target_matches);
    assert_eq!(
        wrong_target_compatibility
            .issues_for_kind(ModelPluginCompatibilityIssueKind::Target)
            .len(),
        1
    );
    let wrong_target_receipt_lines = wrong_target_compatibility.receipt_lines();
    assert!(wrong_target_receipt_lines
        .iter()
        .any(|line| line == "target_matches=false"));
    assert!(wrong_target_receipt_lines
        .iter()
        .any(|line| line == "accepted=false"));
    assert!(wrong_target_receipt_lines
        .iter()
        .any(|line| line == "issues.count=1"));
    assert!(wrong_target_receipt_lines
        .iter()
        .any(|line| line == "issues.0.kind=target"));
    assert_ne!(
        wrong_target_compatibility.receipt_fingerprint(),
        plugin_compatibility_receipt_fingerprint
    );
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
    assert!(plugin_manifest
        .assert_compatible_with(&wrong_target_catalog)
        .is_err());
    assert_eq!(
        plugin_manifest
            .launch_request_step_for(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .unwrap()
            .requirement,
        "aql_packet_materialization"
    );
    assert_eq!(
        plugin_manifest
            .launch_request_step_for_request_plan("aql_live_relocation_binding_request_plan")
            .unwrap()
            .step,
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding
    );
    assert!(plugin_manifest
        .launch_request_step_for_request_plan("dispatch_geometry_request_plan")
        .is_none());
    let issue_report = readiness.static_readiness_issues();
    assert!(issue_report.is_ready());
    assert!(issue_report.issues.is_empty());
    issue_report.assert_ready()?;
    assert_eq!(graph.name, "external-toy-decoder");
    assert_eq!(readiness.graph.tensors, 6);
    assert_eq!(readiness.graph.ops, 3);
    assert_eq!(readiness.graph.stages, 3);
    assert_eq!(readiness.graph.staged_ops, 3);
    assert_eq!(readiness.graph.unstaged_ops, 0);
    assert_eq!(readiness.checkpoint.entries.len(), 2);
    assert!(readiness.checkpoint.missing_weight_tensors.is_empty());
    assert_eq!(
        readiness.binding.entry_for("tokens").unwrap().class,
        TensorBindingClass::ExternalInput
    );
    assert_eq!(
        readiness.binding.entry_for("next_token").unwrap().class,
        TensorBindingClass::ExternalOutput
    );
    assert_eq!(
        readiness.binding.entry_for("embed_weight").unwrap().class,
        TensorBindingClass::CheckpointWeight
    );
    assert_eq!(
        readiness.binding.entry_for("hidden").unwrap().class,
        TensorBindingClass::Scratch
    );
    let lm_head_execution = readiness.execution.entry_for("lm_head").unwrap();
    assert_eq!(lm_head_execution.stage_name.as_deref(), Some("output"));
    assert_eq!(lm_head_execution.route.status, LoweringStatus::NativeGpu);
    assert_eq!(
        lm_head_execution.read_binding_for("hidden").unwrap().class,
        TensorBindingClass::Scratch
    );
    assert_eq!(
        lm_head_execution.write_binding_for("logits").unwrap().class,
        TensorBindingClass::Scratch
    );
    let lm_head_slots = readiness.slots.op_entry_for("lm_head").unwrap();
    assert_eq!(lm_head_slots.stage_name.as_deref(), Some("output"));
    assert_eq!(lm_head_slots.route.status, LoweringStatus::NativeGpu);
    assert_eq!(
        lm_head_slots.read_slot_for("hidden").unwrap().slot,
        readiness.slots.tensor_slot_for("hidden").unwrap().slot
    );
    assert_eq!(
        lm_head_slots.write_slot_for("logits").unwrap().slot,
        readiness.slots.tensor_slot_for("logits").unwrap().slot
    );
    let lm_head_intent = readiness.dispatch_intents.entry_for("lm_head").unwrap();
    assert_eq!(lm_head_intent.route.status, LoweringStatus::NativeGpu);
    let embed_intent = readiness
        .dispatch_intents
        .entry_for("embed_tokens")
        .unwrap();
    assert_eq!(
        embed_intent
            .scalar_argument_for("token_count")
            .unwrap()
            .value
            .as_usize(),
        Some(1)
    );
    let sample_intent = readiness
        .dispatch_intents
        .entry_for("sample_argmax")
        .unwrap();
    assert_eq!(
        sample_intent
            .scalar_argument_for("token_count")
            .unwrap()
            .value
            .as_usize(),
        Some(1)
    );
    let hidden_slot = readiness.slots.tensor_slot_for("hidden").unwrap().slot;
    let logits_slot = readiness.slots.tensor_slot_for("logits").unwrap().slot;
    assert!(lm_head_intent.reads_slot(hidden_slot));
    assert!(lm_head_intent.writes_slot(logits_slot));
    assert_eq!(
        lm_head_intent.slot_argument_for("input").unwrap().slot,
        hidden_slot
    );
    assert_eq!(
        lm_head_intent.slot_argument_for("output").unwrap().slot,
        logits_slot
    );
    assert_eq!(
        lm_head_intent
            .scalar_argument_for("in_features")
            .unwrap()
            .value
            .as_usize(),
        Some(128)
    );
    assert_eq!(
        lm_head_intent
            .scalar_argument_for("out_features")
            .unwrap()
            .value
            .as_usize(),
        Some(256)
    );
    assert!(lm_head_intent
        .entrypoint_symbols
        .contains(&"arm_gemv".to_string()));
    let slot_bindings = readiness.slots.metadata_binding_template("external")?;
    readiness
        .slots
        .validate_complete_buffer_bindings(&slot_bindings)
        .assert_complete()?;
    let device_pointer_bindings = readiness.slots.device_pointer_binding_template(
        DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE,
        DEFAULT_RUNTIME_DEVICE_POINTER_ALIGNMENT,
    )?;
    let device_pointer_validation = readiness
        .slots
        .validate_complete_device_pointer_bindings(&device_pointer_bindings);
    device_pointer_validation.assert_complete()?;
    assert_eq!(
        device_pointer_validation.bound_slots.len(),
        slot_bindings.len()
    );
    let stage_binding_validation = readiness
        .stage_slots
        .validate_stage_buffer_bindings(&readiness.slots, &slot_bindings);
    assert_eq!(stage_binding_validation.issue_count(), 0);
    stage_binding_validation.assert_complete()?;
    let admission = readiness.validate_metadata_runtime_admission(&slot_bindings);
    assert!(admission.is_admitted());
    assert_eq!(admission.issue_count(), 0);
    assert!(admission.dispatch_bindings.is_complete());
    admission.assert_consistent()?;
    let lm_head_binding = admission.dispatch_bindings.entry_for("lm_head").unwrap();
    let hidden_handle = format!("external.slot.{hidden_slot}");
    let logits_handle = format!("external.slot.{logits_slot}");
    assert_eq!(
        lm_head_binding
            .read_binding_for(hidden_slot)
            .unwrap()
            .handle(),
        Some(hidden_handle.as_str())
    );
    assert_eq!(
        lm_head_binding
            .write_binding_for(logits_slot)
            .unwrap()
            .handle(),
        Some(logits_handle.as_str())
    );
    admission.assert_admitted()?;
    let mut stale_static_target_admission = admission.clone();
    stale_static_target_admission.static_issues.target = "stale-static-admission-target";
    assert!(!stale_static_target_admission.is_admitted());
    let stale_static_target_admission_err = stale_static_target_admission
        .assert_admitted()
        .unwrap_err()
        .to_string();
    assert!(stale_static_target_admission_err.contains("consistency"));
    assert!(stale_static_target_admission_err.contains("static readiness target"));
    assert!(stale_static_target_admission
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("static readiness target"));
    let mut stale_dispatch_snapshot_admission = admission.clone();
    stale_dispatch_snapshot_admission
        .dispatch_bindings
        .slot_bindings
        .missing_slots
        .push(usize::MAX);
    assert!(!stale_dispatch_snapshot_admission.is_admitted());
    assert!(stale_dispatch_snapshot_admission
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("slot-binding snapshot"));
    let output_stage_slots = readiness.stage_slots.stage_for("output").unwrap();
    assert_eq!(output_stage_slots.kind, ModelStageKind::Output);
    assert_eq!(output_stage_slots.op_names, vec!["lm_head".to_string()]);
    assert!(output_stage_slots
        .requires_stage_input_slot(readiness.slots.tensor_slot_for("hidden").unwrap().slot));
    assert!(output_stage_slots.writes_slot(readiness.slots.tensor_slot_for("logits").unwrap().slot));
    let output_bundle = readiness.stage_bundles.stage_for("output").unwrap();
    assert_eq!(output_bundle.resources, output_stage_slots.clone());
    assert_eq!(output_bundle.op_slots.len(), 1);
    let output_bundle_op = output_bundle.op_entry_for("lm_head").unwrap();
    assert_eq!(output_bundle_op.route.status, LoweringStatus::NativeGpu);
    assert!(output_bundle_op.read_slot_for("hidden").is_some());
    assert!(output_bundle_op.write_slot_for("logits").is_some());
    let output_stage_dispatch = readiness.stage_dispatch.stage_for("output").unwrap();
    assert_eq!(output_stage_dispatch.resources, output_stage_slots.clone());
    assert_eq!(output_stage_dispatch.dispatches.len(), 1);
    let output_dispatch = output_stage_dispatch.dispatch_for("lm_head").unwrap();
    assert_eq!(output_dispatch.route.status, LoweringStatus::NativeGpu);
    assert_eq!(
        output_dispatch.slot_argument_for("input").unwrap().slot,
        hidden_slot
    );
    assert_eq!(
        output_dispatch
            .scalar_argument_for("out_features")
            .unwrap()
            .value
            .as_usize(),
        Some(256)
    );
    assert!(admission.stage_dispatch_bindings.is_complete());
    let output_stage_dispatch_binding = admission
        .stage_dispatch_bindings
        .stage_for("output")
        .unwrap();
    assert_eq!(output_stage_dispatch_binding.dispatch_bindings.len(), 1);
    let lm_head_stage_dispatch_binding = output_stage_dispatch_binding
        .dispatch_binding_for("lm_head")
        .unwrap();
    assert_eq!(
        lm_head_stage_dispatch_binding
            .read_binding_for(hidden_slot)
            .unwrap()
            .handle(),
        Some(hidden_handle.as_str())
    );
    let launch_candidates = admission.runtime_stage_launch_candidate_plan()?;
    assert_eq!(
        launch_candidates.dispatch_count,
        readiness.dispatch_intents.entries.len()
    );
    assert_eq!(
        readiness.runtime_stage_launch_candidate_plan(&slot_bindings)?,
        launch_candidates
    );
    let output_launch = launch_candidates.stage_for("output").unwrap();
    assert_eq!(output_launch.kind, ModelStageKind::Output);
    assert_eq!(output_launch.op_names, vec!["lm_head".to_string()]);
    let lm_head_launch = output_launch.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_launch.entrypoint_symbols,
        output_dispatch.entrypoint_symbols
    );
    assert_eq!(
        lm_head_launch
            .read_handle_for(hidden_slot)
            .unwrap()
            .handle
            .as_str(),
        hidden_handle.as_str()
    );
    assert_eq!(
        lm_head_launch
            .write_handle_for(logits_slot)
            .unwrap()
            .handle
            .as_str(),
        logits_handle.as_str()
    );
    assert_eq!(
        lm_head_launch.scalar_argument_for("out_features").unwrap(),
        output_dispatch.scalar_argument_for("out_features").unwrap()
    );
    let launch_windows =
        admission.runtime_launch_window_plan(DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES)?;
    assert_eq!(
        launch_windows.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        launch_windows.window_count,
        readiness.stage_dispatch.stages.len()
    );
    assert_eq!(
        readiness
            .runtime_launch_window_plan(&slot_bindings, DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES)?,
        launch_windows
    );
    let output_windows = launch_windows.windows_for_stage("output");
    assert_eq!(output_windows.len(), 1);
    assert_eq!(output_windows[0].dispatch_names, output_launch.op_names);
    let launch_entrypoints = admission.runtime_launch_entrypoint_provenance_plan()?;
    assert_eq!(
        launch_entrypoints.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        readiness.runtime_launch_entrypoint_provenance_plan(&slot_bindings)?,
        launch_entrypoints
    );
    let lm_head_entrypoints = launch_entrypoints.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_entrypoints.host_launchers.len(),
        lm_head_launch.route.entrypoints.len()
    );
    let arm_gemv_launcher = lm_head_entrypoints.host_launcher_for("arm_gemv").unwrap();
    assert_eq!(
        arm_gemv_launcher.kind,
        RuntimeLaunchEntrypointProvenanceKind::HostLauncher
    );
    assert_eq!(
        arm_gemv_launcher.qualified_entrypoint,
        "GpuDevice::arm_gemv"
    );
    assert_eq!(
        arm_gemv_launcher.substrate,
        "crates/mainarch-core/src/gpu.rs"
    );
    let kernel_requirements = admission.runtime_launch_kernel_requirement_plan()?;
    assert_eq!(
        kernel_requirements.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert!(kernel_requirements.unmapped_host_launchers.is_empty());
    assert_eq!(
        readiness.runtime_launch_kernel_requirement_plan(&slot_bindings)?,
        kernel_requirements
    );
    let lm_head_kernels = kernel_requirements.dispatch_for("lm_head").unwrap();
    assert!(lm_head_kernels.requires_kernel("gemv_f16"));
    assert!(lm_head_kernels.requires_kernel("gemv_f16_step"));
    kernel_requirements
        .validate_code_object(&code_object)?
        .assert_complete()?;
    let kernel_metadata = kernel_requirements.kernel_metadata_plan(&code_object)?;
    assert_eq!(
        kernel_metadata.required_kernel_count,
        kernel_requirements.required_kernel_symbols.len()
    );
    assert_eq!(
        readiness.runtime_launch_kernel_metadata_plan(&slot_bindings, &code_object)?,
        kernel_metadata
    );
    assert!(
        kernel_metadata
            .dispatch_for("lm_head")
            .unwrap()
            .kernel_for("gemv_f16")
            .unwrap()
            .kernarg_size
            > 0
    );
    let code_object_loads = kernel_metadata.code_object_load_request_plan()?;
    assert_eq!(
        code_object_loads.required_kernel_count,
        kernel_metadata.required_kernel_count
    );
    assert_eq!(code_object_loads.code_object_load_request_count, 1);
    assert_eq!(code_object_loads.loaded_code_object_count, 0);
    assert!(!code_object_loads.code_object_base_bound);
    assert_eq!(
        code_object_loads.kernel_descriptor_binding_request_count,
        kernel_metadata.required_kernel_count
    );
    assert_eq!(code_object_loads.kernel_descriptor_bound_count, 0);
    assert!(!code_object_loads.all_kernel_descriptors_bound);
    assert!(code_object_loads.request_plan_ready);
    assert!(code_object_loads
        .unresolved_runtime_requirements
        .contains(&"loaded_code_object_base"));
    assert_eq!(
        admission.runtime_launch_code_object_load_request_plan(&code_object)?,
        code_object_loads
    );
    assert_eq!(
        readiness.runtime_launch_code_object_load_request_plan(&slot_bindings, &code_object)?,
        code_object_loads
    );
    let gemv_load_request = code_object_loads.kernel_for("gemv_f16").unwrap();
    assert_eq!(
        gemv_load_request.kernel_descriptor_vaddr,
        kernel_metadata
            .kernel_for("gemv_f16")
            .unwrap()
            .kernel_descriptor_vaddr
    );
    assert!(!gemv_load_request.descriptor_binding_bound);
    code_object_loads.assert_consistent()?;
    let launch_arguments = admission.runtime_launch_argument_plan()?;
    assert_eq!(
        launch_arguments.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        readiness.runtime_launch_argument_plan(&slot_bindings)?,
        launch_arguments
    );
    let launch_preflight = admission
        .runtime_launch_preflight_report(&code_object, DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES)?;
    assert_eq!(
        launch_preflight.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        launch_preflight.argument_count,
        launch_arguments.argument_count
    );
    assert_eq!(
        launch_preflight.required_kernel_count,
        kernel_metadata.required_kernel_count
    );
    assert_eq!(
        readiness.runtime_launch_preflight_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        launch_preflight
    );
    launch_preflight.assert_ready()?;
    let aql_packet_fields = launch_preflight.aql_packet_field_plan()?;
    assert_eq!(
        aql_packet_fields.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        aql_packet_fields.argument_count,
        launch_arguments.argument_count
    );
    assert_eq!(
        aql_packet_fields.required_kernel_count,
        kernel_metadata.required_kernel_count
    );
    assert!(aql_packet_fields.kernel_candidate_count >= aql_packet_fields.required_kernel_count);
    assert!(aql_packet_fields.has_unresolved_runtime_fields());
    assert!(aql_packet_fields
        .unresolved_runtime_requirements
        .contains(&"kernarg_allocation"));
    assert_eq!(
        admission.runtime_launch_aql_packet_field_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        aql_packet_fields
    );
    assert_eq!(
        readiness.runtime_launch_aql_packet_field_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        aql_packet_fields
    );
    assert!(aql_packet_fields
        .dispatch_for("lm_head")
        .unwrap()
        .kernel_candidate_for("gemv_f16")
        .is_some());
    aql_packet_fields.assert_metadata_handoff_ready()?;
    let kernel_selection = launch_preflight.kernel_selection_readiness_plan()?;
    assert_eq!(
        kernel_selection.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        kernel_selection.kernel_candidate_count,
        aql_packet_fields.kernel_candidate_count
    );
    assert_eq!(kernel_selection.selected_dispatch_count, 1);
    assert_eq!(kernel_selection.ambiguous_dispatch_count, 2);
    assert_eq!(kernel_selection.missing_dispatch_count, 0);
    assert!(kernel_selection
        .unresolved_runtime_requirements
        .contains(&"host_launcher_runtime_branch_resolution"));
    assert_eq!(
        admission.runtime_launch_kernel_selection_readiness_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        kernel_selection
    );
    assert_eq!(
        readiness.runtime_launch_kernel_selection_readiness_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        kernel_selection
    );
    let lm_head_selection = kernel_selection.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_selection.state,
        RuntimeLaunchKernelSelectionState::AmbiguousCandidateSet
    );
    assert!(lm_head_selection.candidate_for("gemv_f16_step_k4096"));
    let embed_selection = kernel_selection.dispatch_for("embed_tokens").unwrap();
    assert_eq!(
        embed_selection.state,
        RuntimeLaunchKernelSelectionState::SelectedSingleCandidate
    );
    assert_eq!(
        embed_selection
            .selected_kernel
            .as_ref()
            .unwrap()
            .kernel_symbol,
        "decode_step_embed_rmsnorm_token_f16"
    );
    assert!(kernel_selection.assert_all_selected().is_err());
    kernel_selection.assert_consistent()?;
    let host_launcher_branch_requests =
        kernel_selection.host_launcher_branch_resolution_request_plan()?;
    assert_eq!(
        host_launcher_branch_requests.dispatch_count,
        kernel_selection.dispatch_count
    );
    assert_eq!(
        host_launcher_branch_requests.branch_resolution_request_count,
        kernel_selection.ambiguous_dispatch_count
    );
    assert_eq!(
        host_launcher_branch_requests.branch_resolution_applied_count,
        0
    );
    assert!(host_launcher_branch_requests.unresolved_candidate_symbol_count > 0);
    assert!(!host_launcher_branch_requests.all_branches_resolved);
    assert!(host_launcher_branch_requests.request_plan_ready);
    assert_eq!(
        host_launcher_branch_requests
            .branch_resolution_request_op_names()
            .as_slice(),
        &["lm_head", "sample_argmax"]
    );
    assert_eq!(
        host_launcher_branch_requests
            .unresolved_candidate_symbols()
            .as_slice(),
        &[
            "argmax_f32_step",
            "argmax_f32_token_ids_write_candidate",
            "argmax_f32_token_ids_write_candidate_n1187",
            "gemv_f16",
            "gemv_f16_k8192",
            "gemv_f16_step",
            "gemv_f16_step_k4096",
        ]
    );
    assert_eq!(
        launch_preflight.host_launcher_branch_resolution_request_plan()?,
        host_launcher_branch_requests
    );
    assert_eq!(
        admission.runtime_launch_host_launcher_branch_resolution_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        host_launcher_branch_requests
    );
    assert_eq!(
        readiness.runtime_launch_host_launcher_branch_resolution_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        host_launcher_branch_requests
    );
    let lm_head_branch_request = host_launcher_branch_requests
        .dispatch_for("lm_head")
        .unwrap();
    assert!(lm_head_branch_request.branch_resolution_requested);
    assert!(!lm_head_branch_request.branch_resolution_applied);
    assert_eq!(
        lm_head_branch_request.candidate_count,
        lm_head_selection.candidate_count
    );
    assert!(host_launcher_branch_requests
        .dispatch_for("embed_tokens")
        .is_none());
    host_launcher_branch_requests.assert_consistent()?;
    let mut inconsistent_host_launcher_branch_requests = host_launcher_branch_requests.clone();
    inconsistent_host_launcher_branch_requests.branch_resolution_applied_count = 1;
    let err = inconsistent_host_launcher_branch_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("branch resolution applied count"));
    let launch_device_arguments =
        launch_preflight.device_argument_plan(&device_pointer_validation)?;
    assert_eq!(
        launch_device_arguments.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        launch_device_arguments.argument_count,
        launch_arguments.argument_count
    );
    assert_eq!(
        launch_device_arguments.pointer_argument_count,
        readiness
            .dispatch_intents
            .entries
            .iter()
            .map(|entry| entry.slot_arguments.len())
            .sum::<usize>()
    );
    assert_eq!(
        launch_device_arguments.scalar_argument_count,
        readiness
            .dispatch_intents
            .entries
            .iter()
            .map(|entry| entry.scalar_arguments.len())
            .sum::<usize>()
    );
    assert_eq!(
        admission.runtime_launch_device_argument_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        launch_device_arguments
    );
    assert_eq!(
        readiness.runtime_launch_device_argument_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        launch_device_arguments
    );
    let lm_head_device_arguments = launch_device_arguments.dispatch_for("lm_head").unwrap();
    match &lm_head_device_arguments
        .argument_for("weight")
        .unwrap()
        .value
    {
        RuntimeLaunchDeviceArgumentValue::Pointer(pointer) => {
            assert_eq!(pointer.logical_handle.tensor.as_str(), "lm_head");
            assert!(
                pointer.device_pointer.device_va >= DEFAULT_RUNTIME_SYNTHETIC_DEVICE_POINTER_BASE
            );
        }
        RuntimeLaunchDeviceArgumentValue::Scalar(_) => panic!("weight should bind a device VA"),
    }
    launch_device_arguments.assert_bound()?;
    let staging_footprint = launch_preflight.staging_footprint_plan()?;
    assert_eq!(
        staging_footprint.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        staging_footprint.argument_count,
        launch_arguments.argument_count
    );
    assert_eq!(
        staging_footprint.pointer_argument_count,
        launch_device_arguments.pointer_argument_count
    );
    assert_eq!(
        staging_footprint.scalar_argument_count,
        launch_device_arguments.scalar_argument_count
    );
    assert_eq!(
        staging_footprint.kernel_candidate_count,
        aql_packet_fields.kernel_candidate_count
    );
    assert_eq!(
        staging_footprint.packet_bytes,
        staging_footprint.dispatch_count * AQL_PACKET_BYTES as usize
    );
    assert!(staging_footprint.kernarg_bytes_upper_bound >= staging_footprint.max_kernarg_size);
    assert!(staging_footprint.max_kernarg_segment_align > 0);
    assert!(staging_footprint
        .unresolved_runtime_requirements
        .contains(&"queue_reservation"));
    assert_eq!(
        admission.runtime_launch_staging_footprint_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        staging_footprint
    );
    assert_eq!(
        readiness.runtime_launch_staging_footprint_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        staging_footprint
    );
    assert_eq!(
        staging_footprint
            .dispatch_for("lm_head")
            .unwrap()
            .packet_bytes,
        AQL_PACKET_BYTES as usize
    );
    staging_footprint.assert_consistent()?;
    let staging_layout = launch_preflight.staging_layout_plan()?;
    assert_eq!(
        staging_layout.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(staging_layout.window_count, staging_footprint.window_count);
    assert_eq!(staging_layout.packet_alignment, AQL_PACKET_BYTES as usize);
    assert_eq!(
        staging_layout.packet_region_bytes,
        staging_footprint.packet_bytes
    );
    assert!(staging_layout.kernarg_region_bytes >= staging_footprint.kernarg_bytes_upper_bound);
    assert_eq!(
        staging_layout.total_staging_bytes,
        staging_layout.packet_region_bytes + staging_layout.kernarg_region_bytes
    );
    assert!(!staging_layout
        .unresolved_runtime_requirements
        .contains(&"allocator_alignment_policy"));
    assert_eq!(
        admission.runtime_launch_staging_layout_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        staging_layout
    );
    assert_eq!(
        readiness.runtime_launch_staging_layout_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        staging_layout
    );
    let lm_head_layout = staging_layout.dispatch_for("lm_head").unwrap();
    assert_eq!(lm_head_layout.packet_bytes, AQL_PACKET_BYTES as usize);
    assert_eq!(
        lm_head_layout.kernarg_offset % lm_head_layout.kernarg_alignment,
        0
    );
    staging_layout.assert_consistent()?;
    let completion_signals = launch_preflight.completion_signal_plan()?;
    assert_eq!(
        completion_signals.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        completion_signals.window_count,
        launch_preflight.window_count
    );
    assert_eq!(
        completion_signals.terminal_signal_count,
        launch_preflight.window_count
    );
    assert_eq!(
        completion_signals.logical_signal_slots,
        completion_signals.terminal_signal_count
    );
    assert_eq!(completion_signals.signal_initial_value, 1);
    assert_eq!(completion_signals.signal_completed_value, 0);
    assert!(completion_signals
        .unresolved_runtime_requirements
        .contains(&"completion_signal_binding"));
    assert!(!completion_signals
        .unresolved_runtime_requirements
        .contains(&"completion_signal_policy"));
    assert_eq!(
        admission.runtime_launch_completion_signal_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        completion_signals
    );
    assert_eq!(
        readiness.runtime_launch_completion_signal_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        completion_signals
    );
    let first_signal_window = completion_signals.window_for(0).unwrap();
    assert_eq!(first_signal_window.logical_signal_slot, Some(0));
    assert_eq!(
        first_signal_window.unresolved_aql_packet_field,
        "completion_signal"
    );
    completion_signals.assert_consistent()?;
    let completion_signal_bindings = completion_signals.completion_signal_binding_request_plan()?;
    assert_eq!(
        completion_signal_bindings.dispatch_count,
        completion_signals.dispatch_count
    );
    assert_eq!(
        completion_signal_bindings.window_count,
        completion_signals.window_count
    );
    assert_eq!(
        completion_signal_bindings.signal_handle_request_count,
        completion_signals.terminal_signal_count
    );
    assert_eq!(completion_signal_bindings.signal_handle_bound_count, 0);
    assert!(!completion_signal_bindings.all_signal_handles_bound);
    assert!(completion_signal_bindings.request_plan_ready);
    assert_eq!(
        launch_preflight.completion_signal_binding_request_plan()?,
        completion_signal_bindings
    );
    assert_eq!(
        admission.runtime_launch_completion_signal_binding_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        completion_signal_bindings
    );
    assert_eq!(
        readiness.runtime_launch_completion_signal_binding_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        completion_signal_bindings
    );
    let first_signal_binding = completion_signal_bindings.window_for(0).unwrap();
    assert_eq!(first_signal_binding.logical_signal_slot, Some(0));
    assert!(first_signal_binding.signal_handle_requested);
    assert!(!first_signal_binding.signal_handle_bound);
    assert_eq!(
        first_signal_binding.terminal_dispatch_name,
        first_signal_window.terminal_dispatch_name
    );
    completion_signal_bindings.assert_consistent()?;
    let queue_slots = launch_preflight.queue_slot_plan()?;
    assert_eq!(queue_slots.dispatch_count, launch_candidates.dispatch_count);
    assert_eq!(queue_slots.window_count, launch_preflight.window_count);
    assert_eq!(
        queue_slots.queue_packet_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        queue_slots.doorbell_batch_count,
        launch_preflight.window_count
    );
    assert!(queue_slots
        .unresolved_runtime_requirements
        .contains(&"queue_reservation"));
    assert!(queue_slots
        .unresolved_runtime_requirements
        .contains(&"aql_packet_materialization"));
    assert_eq!(
        admission.runtime_launch_queue_slot_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        queue_slots
    );
    assert_eq!(
        readiness.runtime_launch_queue_slot_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        queue_slots
    );
    let first_queue_window = queue_slots.window_for(0).unwrap();
    assert_eq!(first_queue_window.first_queue_packet_index, 0);
    assert_eq!(
        first_queue_window.terminal_completion_signal_slot,
        first_signal_window.logical_signal_slot
    );
    let first_queue_dispatch = first_queue_window
        .dispatch_for(first_queue_window.dispatch_names.first().unwrap())
        .unwrap();
    assert_eq!(first_queue_dispatch.queue_packet_index, 0);
    assert_eq!(first_queue_dispatch.window_packet_index, 0);
    assert!(!first_queue_dispatch.chain_barrier_after_previous);
    queue_slots.assert_consistent()?;
    let queue_reservations = launch_preflight.queue_reservation_request_plan()?;
    assert_eq!(
        queue_reservations.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        queue_reservations.window_count,
        launch_preflight.window_count
    );
    assert_eq!(
        queue_reservations.queue_packet_request_count,
        queue_slots.queue_packet_count
    );
    assert_eq!(queue_reservations.queue_packet_reserved_count, 0);
    assert_eq!(
        queue_reservations.doorbell_batch_request_count,
        queue_slots.doorbell_batch_count
    );
    assert_eq!(queue_reservations.doorbell_batch_bound_count, 0);
    assert_eq!(queue_reservations.reservation_applied_count, 0);
    assert!(!queue_reservations.all_queue_packets_reserved);
    assert!(queue_reservations.request_plan_ready);
    assert!(queue_reservations
        .unresolved_runtime_requirements
        .contains(&"queue_reservation"));
    assert_eq!(
        admission.runtime_launch_queue_reservation_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        queue_reservations
    );
    assert_eq!(
        readiness.runtime_launch_queue_reservation_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        queue_reservations
    );
    let first_queue_reservation = queue_reservations.window_for(0).unwrap();
    assert_eq!(first_queue_reservation.first_queue_packet_index, 0);
    assert_eq!(
        first_queue_reservation.queue_packet_request_count,
        first_queue_window.queue_packet_count
    );
    assert!(first_queue_reservation.doorbell_batch_requested);
    assert!(!first_queue_reservation.doorbell_batch_bound);
    assert!(!first_queue_reservation.reservation_applied);
    queue_reservations.assert_consistent()?;
    let dispatch_geometry = launch_preflight.dispatch_geometry_plan()?;
    assert_eq!(
        dispatch_geometry.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        dispatch_geometry.window_count,
        launch_preflight.window_count
    );
    assert_eq!(
        dispatch_geometry.queue_packet_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(dispatch_geometry.default_workgroup_size, 256);
    assert!(dispatch_geometry.total_workgroups >= dispatch_geometry.dispatch_count);
    assert!(dispatch_geometry
        .unresolved_runtime_requirements
        .contains(&"kernel_specific_launch_tuning"));
    assert!(!dispatch_geometry
        .unresolved_runtime_requirements
        .contains(&"dispatch_geometry"));
    assert_eq!(
        admission.runtime_launch_dispatch_geometry_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        dispatch_geometry
    );
    assert_eq!(
        readiness.runtime_launch_dispatch_geometry_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
        )?,
        dispatch_geometry
    );
    let lm_head_geometry = dispatch_geometry.dispatch_for("lm_head").unwrap();
    assert_eq!(lm_head_geometry.workload_source, "out_features");
    assert!(lm_head_geometry.grid_size_x > 0);
    assert_eq!(lm_head_geometry.workgroup_size_x, 256);
    let first_geometry_window = dispatch_geometry.window_for(0).unwrap();
    assert_eq!(first_geometry_window.first_queue_packet_index, 0);
    assert_eq!(
        first_geometry_window.queue_packet_count,
        first_queue_window.queue_packet_count
    );
    dispatch_geometry.assert_consistent()?;
    let kernarg_layout = launch_preflight.kernarg_layout_plan(&device_pointer_validation)?;
    assert_eq!(
        kernarg_layout.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(kernarg_layout.window_count, launch_preflight.window_count);
    assert_eq!(
        kernarg_layout.argument_count,
        launch_device_arguments.argument_count
    );
    assert_eq!(
        kernarg_layout.pointer_argument_count,
        launch_device_arguments.pointer_argument_count
    );
    assert_eq!(
        kernarg_layout.scalar_argument_count,
        launch_device_arguments.scalar_argument_count
    );
    assert!(kernarg_layout.argument_payload_bytes > 0);
    assert!(kernarg_layout.argument_span_bytes >= kernarg_layout.argument_payload_bytes);
    let dispatch_capacity_shortfall_bytes = kernarg_layout
        .stages
        .iter()
        .flat_map(|stage| stage.dispatches.iter())
        .map(|dispatch| dispatch.candidate_capacity_shortfall_bytes)
        .sum::<usize>();
    assert_eq!(
        kernarg_layout.candidate_capacity_shortfall_bytes,
        dispatch_capacity_shortfall_bytes
    );
    assert_eq!(kernarg_layout.max_argument_alignment, 8);
    assert!(kernarg_layout
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert!(!kernarg_layout
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_layout"));
    assert_eq!(
        admission.runtime_launch_kernarg_layout_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernarg_layout
    );
    assert_eq!(
        readiness.runtime_launch_kernarg_layout_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernarg_layout
    );
    let lm_head_kernarg = kernarg_layout.dispatch_for("lm_head").unwrap();
    let lm_head_weight_kernarg = lm_head_kernarg.argument_for("weight").unwrap();
    assert_eq!(
        lm_head_weight_kernarg.encoding,
        RuntimeLaunchKernargArgumentEncoding::DevicePointerU64
    );
    assert_eq!(
        lm_head_weight_kernarg.offset % lm_head_weight_kernarg.alignment,
        0
    );
    let lm_head_out_features_kernarg = lm_head_kernarg.argument_for("out_features").unwrap();
    assert_eq!(
        lm_head_out_features_kernarg.encoding,
        RuntimeLaunchKernargArgumentEncoding::UsizeU64
    );
    kernarg_layout.assert_consistent()?;
    let kernarg_serialization =
        launch_preflight.kernarg_serialization_plan(&device_pointer_validation)?;
    assert_eq!(
        kernarg_serialization.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        kernarg_serialization.argument_count,
        launch_device_arguments.argument_count
    );
    assert_eq!(
        kernarg_serialization.argument_payload_bytes,
        kernarg_layout.argument_payload_bytes
    );
    assert_eq!(
        kernarg_serialization.argument_span_bytes,
        kernarg_layout.argument_span_bytes
    );
    assert_eq!(
        kernarg_serialization.serialized_kernarg_bytes,
        kernarg_layout.argument_span_bytes
    );
    assert_eq!(
        kernarg_serialization.candidate_capacity_shortfall_bytes,
        kernarg_layout.candidate_capacity_shortfall_bytes
    );
    assert!(kernarg_serialization
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert!(!kernarg_serialization
        .unresolved_runtime_requirements
        .contains(&"kernarg_layout_serialization"));
    assert_eq!(
        admission.runtime_launch_kernarg_serialization_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernarg_serialization
    );
    assert_eq!(
        readiness.runtime_launch_kernarg_serialization_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernarg_serialization
    );
    let lm_head_serialized = kernarg_serialization.dispatch_for("lm_head").unwrap();
    let lm_head_weight_serialized = lm_head_serialized.argument_for("weight").unwrap();
    let lm_head_weight_device_va = match &launch_device_arguments
        .dispatch_for("lm_head")
        .unwrap()
        .argument_for("weight")
        .unwrap()
        .value
    {
        RuntimeLaunchDeviceArgumentValue::Pointer(pointer) => pointer.device_pointer.device_va,
        RuntimeLaunchDeviceArgumentValue::Scalar(_) => panic!("weight should be a pointer"),
    };
    assert_eq!(
        lm_head_weight_serialized.bytes,
        lm_head_weight_device_va.to_le_bytes()
    );
    assert_eq!(
        &lm_head_serialized.byte_image[lm_head_weight_serialized.offset
            ..lm_head_weight_serialized.offset + lm_head_weight_serialized.size_bytes],
        lm_head_weight_serialized.bytes.as_slice()
    );
    let lm_head_out_features_serialized = lm_head_serialized.argument_for("out_features").unwrap();
    let lm_head_out_features = match &launch_device_arguments
        .dispatch_for("lm_head")
        .unwrap()
        .argument_for("out_features")
        .unwrap()
        .value
    {
        RuntimeLaunchDeviceArgumentValue::Scalar(RuntimeDispatchScalarValue::Usize(value)) => {
            *value
        }
        RuntimeLaunchDeviceArgumentValue::Scalar(_) => panic!("out_features should be usize"),
        RuntimeLaunchDeviceArgumentValue::Pointer(_) => panic!("out_features should be scalar"),
    };
    assert_eq!(
        lm_head_out_features_serialized.bytes,
        u64::try_from(lm_head_out_features)?.to_le_bytes()
    );
    kernarg_serialization.assert_consistent()?;
    let kernarg_allocations = kernarg_serialization.kernarg_allocation_request_plan()?;
    assert_eq!(
        kernarg_allocations.dispatch_count,
        kernarg_serialization.dispatch_count
    );
    assert_eq!(kernarg_allocations.backing_allocation_request_count, 1);
    assert_eq!(kernarg_allocations.backing_allocation_bound_count, 0);
    assert_eq!(
        kernarg_allocations.backing_allocation_request_bytes,
        kernarg_serialization.kernarg_region_bytes
    );
    assert_eq!(kernarg_allocations.backing_allocation_bound_bytes, 0);
    assert_eq!(
        kernarg_allocations.dispatch_copy_request_count,
        kernarg_serialization.dispatch_count
    );
    assert_eq!(kernarg_allocations.dispatch_copy_applied_count, 0);
    assert_eq!(
        kernarg_allocations.dispatch_copy_request_bytes,
        kernarg_serialization.serialized_kernarg_bytes
    );
    assert_eq!(kernarg_allocations.dispatch_copy_applied_bytes, 0);
    assert_eq!(kernarg_allocations.device_va_bound_dispatch_count, 0);
    assert!(!kernarg_allocations.all_kernargs_allocated);
    assert!(kernarg_allocations.request_plan_ready);
    assert!(kernarg_allocations
        .unresolved_runtime_requirements
        .contains(&"kernarg_allocation"));
    assert_eq!(
        admission.runtime_launch_kernarg_allocation_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernarg_allocations
    );
    assert_eq!(
        readiness.runtime_launch_kernarg_allocation_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernarg_allocations
    );
    let lm_head_allocation = kernarg_allocations.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_allocation.kernarg_region_offset,
        lm_head_serialized.kernarg_region_offset
    );
    assert_eq!(
        lm_head_allocation.copy_request_bytes,
        lm_head_serialized.serialized_kernarg_bytes
    );
    assert!(!lm_head_allocation.backing_allocation_bound);
    assert!(!lm_head_allocation.kernarg_device_va_bound);
    assert!(!lm_head_allocation.copy_applied);
    kernarg_allocations.assert_consistent()?;
    let aql_packet_templates =
        launch_preflight.aql_packet_template_plan(&device_pointer_validation)?;
    assert_eq!(
        aql_packet_templates.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        aql_packet_templates.window_count,
        launch_preflight.window_count
    );
    assert_eq!(
        aql_packet_templates.queue_packet_count,
        queue_slots.queue_packet_count
    );
    assert_eq!(
        aql_packet_templates.doorbell_batch_count,
        queue_slots.doorbell_batch_count
    );
    assert_eq!(
        aql_packet_templates.packet_bytes,
        staging_footprint.packet_bytes
    );
    assert_eq!(
        aql_packet_templates.kernarg_region_bytes,
        kernarg_serialization.kernarg_region_bytes
    );
    assert_eq!(
        aql_packet_templates.kernarg_template_region_bytes,
        aql_packet_templates
            .windows
            .iter()
            .map(|window| window.kernarg_region_bytes)
            .sum::<usize>()
    );
    assert_eq!(
        aql_packet_templates.kernarg_serialized_bytes,
        kernarg_serialization.serialized_kernarg_bytes
    );
    assert_eq!(
        aql_packet_templates.kernel_candidate_count,
        aql_packet_fields.kernel_candidate_count
    );
    assert!(aql_packet_templates
        .unresolved_aql_packet_fields
        .contains(&"kernel_object"));
    assert!(aql_packet_templates
        .unresolved_aql_packet_fields
        .contains(&"kernarg_va"));
    assert!(aql_packet_templates
        .unresolved_runtime_requirements
        .contains(&"queue_reservation"));
    assert!(aql_packet_templates
        .unresolved_runtime_requirements
        .contains(&"aql_packet_materialization"));
    assert!(!aql_packet_templates
        .unresolved_runtime_requirements
        .contains(&"dispatch_geometry"));
    assert_eq!(
        admission.runtime_launch_aql_packet_template_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_templates
    );
    assert_eq!(
        readiness.runtime_launch_aql_packet_template_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_templates
    );
    let lm_head_template = aql_packet_templates.dispatch_for("lm_head").unwrap();
    let lm_head_packet_fields = aql_packet_fields.dispatch_for("lm_head").unwrap();
    assert_eq!(lm_head_template.packet_bytes, AQL_PACKET_BYTES as usize);
    assert_eq!(lm_head_template.dimensions, lm_head_geometry.dimensions);
    assert_eq!(lm_head_template.grid_size_x, lm_head_geometry.grid_size_x);
    assert_eq!(
        lm_head_template.workgroup_size_x,
        lm_head_geometry.workgroup_size_x
    );
    assert_eq!(
        lm_head_template.kernarg_region_offset,
        lm_head_serialized.kernarg_region_offset
    );
    assert_eq!(
        lm_head_template.kernarg_serialized_bytes,
        lm_head_serialized.serialized_kernarg_bytes
    );
    assert_eq!(
        lm_head_template.kernel_candidate_count,
        lm_head_packet_fields.kernel_candidates.len()
    );
    assert!(lm_head_template
        .candidate_templates
        .iter()
        .all(|candidate| candidate.kernarg_serialized_bytes
            == lm_head_serialized.serialized_kernarg_bytes));
    let first_template_window = aql_packet_templates.window_for(0).unwrap();
    assert_eq!(
        first_template_window.dispatch_count,
        first_queue_window.dispatch_count
    );
    assert_eq!(
        first_template_window.queue_packet_count,
        first_geometry_window.queue_packet_count
    );
    aql_packet_templates.assert_consistent()?;
    let kernel_argument_abi = aql_packet_templates.kernel_argument_abi_verification_plan()?;
    assert_eq!(
        kernel_argument_abi.dispatch_count,
        aql_packet_templates.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi.kernel_candidate_count,
        aql_packet_templates.kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi.size_compatible_candidate_count
            + kernel_argument_abi.size_shortfall_candidate_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert!(kernel_argument_abi.named_abi_schema_available_count > 0);
    assert!(kernel_argument_abi.verified_candidate_count > 0);
    assert!(
        kernel_argument_abi.verified_candidate_count
            <= kernel_argument_abi.named_abi_schema_available_count
    );
    assert_eq!(
        kernel_argument_abi.dispatches_with_verified_candidate_count
            + kernel_argument_abi.dispatches_without_verified_candidate_count,
        kernel_argument_abi.dispatch_count
    );
    assert!(kernel_argument_abi.dispatches_with_verified_candidate_count > 0);
    assert!(kernel_argument_abi.dispatches_without_verified_candidate_count > 0);
    assert_eq!(
        kernel_argument_abi.abi_verification_ready,
        kernel_argument_abi.verified_candidate_count == kernel_argument_abi.kernel_candidate_count
    );
    assert!(!kernel_argument_abi.abi_verification_ready);
    assert!(kernel_argument_abi.total_capacity_shortfall_bytes > 0);
    assert!(kernel_argument_abi
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_verification_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_verification_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi
    );
    let lm_head_abi = kernel_argument_abi.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_abi.kernel_candidate_count,
        lm_head_template.kernel_candidate_count
    );
    assert!(lm_head_abi
        .candidates
        .iter()
        .all(|candidate| candidate.named_abi_schema_available
            && candidate.named_abi_schema_kernarg_size == Some(candidate.kernarg_size)
            && candidate.named_abi_schema_kernarg_segment_align
                == Some(candidate.kernarg_segment_align)
            && candidate.named_abi_descriptor_match
            && candidate.named_abi_verified == candidate.size_compatible
            && candidate.verification_ready == candidate.named_abi_verified));
    assert_eq!(
        lm_head_abi.has_verified_candidate,
        lm_head_abi.verified_candidate_count > 0
    );
    kernel_argument_abi.assert_consistent()?;
    let abi_size_receipt = kernel_argument_abi.size_compatibility_receipt()?;
    assert_eq!(abi_size_receipt.target, catalog.target);
    assert_eq!(
        abi_size_receipt.dispatch_count,
        kernel_argument_abi.dispatch_count
    );
    assert_eq!(
        abi_size_receipt.kernel_candidate_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert_eq!(
        abi_size_receipt.size_compatible_candidate_count,
        kernel_argument_abi.size_compatible_candidate_count
    );
    assert_eq!(
        abi_size_receipt.size_shortfall_candidate_count,
        kernel_argument_abi.size_shortfall_candidate_count
    );
    assert_eq!(
        abi_size_receipt.max_capacity_shortfall_bytes,
        kernel_argument_abi.max_capacity_shortfall_bytes
    );
    assert_eq!(
        abi_size_receipt.total_capacity_shortfall_bytes,
        kernel_argument_abi.total_capacity_shortfall_bytes
    );
    assert!(abi_size_receipt.dispatches_with_size_compatible_candidate_count > 0);
    assert!(abi_size_receipt.dispatches_without_size_compatible_candidate_count > 0);
    assert_eq!(
        abi_size_receipt.dispatches_with_verified_candidate_count,
        kernel_argument_abi.dispatches_with_verified_candidate_count
    );
    assert_eq!(
        abi_size_receipt.dispatches_without_verified_candidate_count,
        kernel_argument_abi.dispatches_without_verified_candidate_count
    );
    assert!(abi_size_receipt.size_compatibility_checked);
    assert_eq!(
        abi_size_receipt.named_abi_schema_available_count,
        kernel_argument_abi.named_abi_schema_available_count
    );
    assert_eq!(
        abi_size_receipt.named_abi_verified_candidate_count,
        kernel_argument_abi.verified_candidate_count
    );
    assert_eq!(
        abi_size_receipt.named_abi_verification_ready,
        kernel_argument_abi.abi_verification_ready
    );
    assert!(abi_size_receipt
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert_eq!(
        aql_packet_templates.kernel_argument_abi_size_compatibility_receipt()?,
        abi_size_receipt
    );
    assert_eq!(
        launch_preflight
            .kernel_argument_abi_size_compatibility_receipt(&device_pointer_validation)?,
        abi_size_receipt
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_size_compatibility_receipt(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        abi_size_receipt
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_size_compatibility_receipt(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        abi_size_receipt
    );
    let lm_head_size_receipt = abi_size_receipt.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_size_receipt.kernel_candidate_count,
        lm_head_abi.kernel_candidate_count
    );
    assert_eq!(
        lm_head_size_receipt.has_size_compatible_candidate,
        lm_head_abi.size_compatible_candidate_count > 0
    );
    assert_eq!(
        lm_head_size_receipt.verified_candidate_count,
        lm_head_abi.verified_candidate_count
    );
    assert_eq!(
        lm_head_size_receipt.has_verified_candidate,
        lm_head_abi.has_verified_candidate
    );
    assert_eq!(
        lm_head_size_receipt.total_capacity_shortfall_bytes,
        lm_head_abi
            .candidates
            .iter()
            .map(|candidate| candidate.capacity_shortfall_bytes)
            .sum::<usize>()
    );
    abi_size_receipt.assert_consistent()?;
    let kernel_argument_abi_gaps =
        kernel_argument_abi.kernel_argument_abi_verification_gap_report()?;
    assert_eq!(kernel_argument_abi_gaps.target, catalog.target);
    assert_eq!(
        kernel_argument_abi_gaps.source_dispatch_count,
        kernel_argument_abi.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_gaps.source_kernel_candidate_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_gaps.dispatch_gap_count,
        kernel_argument_abi.dispatches_without_verified_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_gaps.all_dispatches_have_verified_candidate,
        kernel_argument_abi.dispatches_without_verified_candidate_count == 0
    );
    assert!(kernel_argument_abi_gaps.dispatch_gap_count > 0);
    assert!(kernel_argument_abi_gaps.gap_kernel_candidate_count > 0);
    assert_eq!(kernel_argument_abi_gaps.gap_verified_candidate_count, 0);
    assert_eq!(
        kernel_argument_abi_gaps.gap_primary_missing_named_abi_schema_candidate_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_gaps.gap_primary_descriptor_mismatch_candidate_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_gaps.gap_primary_size_shortfall_candidate_count,
        kernel_argument_abi_gaps.gap_kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_gaps.gap_primary_unknown_unverified_candidate_count,
        0
    );
    assert!(kernel_argument_abi_gaps.total_capacity_shortfall_bytes > 0);
    assert!(kernel_argument_abi_gaps
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert_eq!(
        aql_packet_templates.kernel_argument_abi_verification_gap_report()?,
        kernel_argument_abi_gaps
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_verification_gap_report(&device_pointer_validation)?,
        kernel_argument_abi_gaps
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_verification_gap_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_gaps
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_verification_gap_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_gaps
    );
    let first_abi_gap = kernel_argument_abi_gaps.dispatches.first().unwrap();
    let first_gap_source = kernel_argument_abi
        .dispatch_for(&first_abi_gap.op_name)
        .unwrap();
    assert_eq!(
        kernel_argument_abi_gaps.dispatch_for(&first_abi_gap.op_name),
        Some(first_abi_gap)
    );
    assert!(!first_gap_source.has_verified_candidate);
    assert_eq!(
        first_abi_gap.kernel_candidate_count,
        first_gap_source.kernel_candidate_count
    );
    assert_eq!(first_abi_gap.verified_candidate_count, 0);
    assert_eq!(
        first_abi_gap.candidate_kernel_symbols.len(),
        first_gap_source.candidates.len()
    );
    assert_eq!(
        first_abi_gap.candidate_gaps.len(),
        first_gap_source.candidates.len()
    );
    assert_eq!(
        first_abi_gap.primary_size_shortfall_candidate_count,
        first_abi_gap.kernel_candidate_count
    );
    assert_eq!(
        first_abi_gap.primary_missing_named_abi_schema_candidate_count
            + first_abi_gap.primary_descriptor_mismatch_candidate_count
            + first_abi_gap.primary_size_shortfall_candidate_count
            + first_abi_gap.primary_unknown_unverified_candidate_count,
        first_abi_gap.kernel_candidate_count
    );
    let first_candidate_gap = first_abi_gap.candidate_gaps.first().unwrap();
    let first_candidate_source = &first_gap_source.candidates[0];
    assert_eq!(
        first_candidate_gap.kernel_symbol,
        first_candidate_source.kernel_symbol
    );
    assert_eq!(
        first_candidate_gap.primary_gap_reason,
        RuntimeLaunchKernelArgumentAbiVerificationGapReason::KernargSizeShortfall
    );
    assert_eq!(
        first_candidate_gap.primary_gap_reason.as_str(),
        "kernarg_size_shortfall"
    );
    assert!(!first_candidate_gap.verification_ready);
    assert!(!first_candidate_gap.named_abi_verified);
    assert!(first_candidate_gap.named_abi_schema_available);
    assert!(first_candidate_gap.named_abi_descriptor_match);
    assert!(!first_candidate_gap.size_compatible);
    assert!(first_candidate_gap.capacity_shortfall_bytes > 0);
    assert_eq!(
        first_abi_gap.total_capacity_shortfall_bytes,
        first_gap_source
            .candidates
            .iter()
            .map(|candidate| candidate.capacity_shortfall_bytes)
            .sum::<usize>()
    );
    kernel_argument_abi_gaps.assert_consistent()?;
    let mut inconsistent_kernel_argument_abi_gaps = kernel_argument_abi_gaps.clone();
    inconsistent_kernel_argument_abi_gaps.dispatch_gap_count += 1;
    let err = inconsistent_kernel_argument_abi_gaps
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("gap count"));
    let kernel_argument_abi_capacity_requests =
        kernel_argument_abi.kernel_argument_abi_capacity_request_plan()?;
    assert_eq!(
        kernel_argument_abi_capacity_requests.source_dispatch_count,
        kernel_argument_abi_gaps.source_dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_capacity_requests.dispatch_gap_count,
        kernel_argument_abi_gaps.dispatch_gap_count
    );
    assert_eq!(
        kernel_argument_abi_capacity_requests.source_kernel_candidate_count,
        kernel_argument_abi_gaps.source_kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_capacity_requests.gap_kernel_candidate_count,
        kernel_argument_abi_gaps.gap_kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_capacity_requests.source_primary_size_shortfall_candidate_count,
        kernel_argument_abi_gaps.gap_primary_size_shortfall_candidate_count
    );
    assert!(kernel_argument_abi_capacity_requests.capacity_request_count > 0);
    assert_eq!(
        kernel_argument_abi_capacity_requests.candidate_capacity_request_count,
        kernel_argument_abi_gaps.gap_primary_size_shortfall_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_capacity_requests.max_capacity_shortfall_bytes,
        kernel_argument_abi_gaps.max_capacity_shortfall_bytes
    );
    assert_eq!(
        kernel_argument_abi_capacity_requests.total_capacity_shortfall_bytes,
        kernel_argument_abi_gaps.total_capacity_shortfall_bytes
    );
    assert!(kernel_argument_abi_capacity_requests.all_capacity_requests_ready);
    assert!(kernel_argument_abi_capacity_requests.request_plan_ready);
    assert!(kernel_argument_abi_capacity_requests
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert_eq!(
        kernel_argument_abi_gaps.kernel_argument_abi_capacity_request_plan()?,
        kernel_argument_abi_capacity_requests
    );
    assert_eq!(
        aql_packet_templates.kernel_argument_abi_capacity_request_plan()?,
        kernel_argument_abi_capacity_requests
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_capacity_request_plan(&device_pointer_validation)?,
        kernel_argument_abi_capacity_requests
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_capacity_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_capacity_requests
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_capacity_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_capacity_requests
    );
    let first_capacity_request = kernel_argument_abi_capacity_requests
        .kernel_for(&first_candidate_gap.kernel_symbol)
        .unwrap();
    assert!(first_capacity_request.named_abi_schema_available);
    assert!(first_capacity_request.named_abi_descriptor_match);
    assert_eq!(
        first_capacity_request.max_serialized_kernarg_bytes,
        first_capacity_request.required_kernarg_size
    );
    assert!(
        first_capacity_request.required_kernarg_size > first_capacity_request.kernarg_size as usize
    );
    assert_eq!(
        first_capacity_request.max_capacity_shortfall_bytes,
        first_capacity_request.required_kernarg_size - first_capacity_request.kernarg_size as usize
    );
    assert!(first_capacity_request.dispatch_reference_count > 0);
    assert!(first_capacity_request.candidate_capacity_request_count > 0);
    assert!(first_capacity_request.capacity_request_ready);
    assert_eq!(
        first_capacity_request.capacity_request_reason,
        "candidate_kernarg_size_shortfall"
    );
    kernel_argument_abi_capacity_requests.assert_consistent()?;
    let mut inconsistent_kernel_argument_abi_capacity_requests =
        kernel_argument_abi_capacity_requests.clone();
    inconsistent_kernel_argument_abi_capacity_requests.candidate_capacity_request_count += 1;
    let err = inconsistent_kernel_argument_abi_capacity_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("candidate capacity request count"));
    let kernel_argument_abi_schema_requests =
        kernel_argument_abi.kernel_argument_abi_schema_request_plan()?;
    assert_eq!(
        kernel_argument_abi_schema_requests.dispatch_count,
        kernel_argument_abi.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_schema_requests.kernel_candidate_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert!(kernel_argument_abi_schema_requests.schema_request_count > 0);
    assert!(kernel_argument_abi_schema_requests.schema_bound_count > 0);
    assert_eq!(
        kernel_argument_abi_schema_requests.candidate_verification_request_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_schema_requests.candidate_verified_count,
        kernel_argument_abi.verified_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_schema_requests.all_schemas_bound,
        kernel_argument_abi_schema_requests.schema_bound_count
            == kernel_argument_abi_schema_requests.schema_request_count
    );
    assert!(!kernel_argument_abi_schema_requests.all_candidates_verified);
    assert!(kernel_argument_abi_schema_requests.request_plan_ready);
    assert_eq!(
        launch_preflight.kernel_argument_abi_schema_request_plan(&device_pointer_validation)?,
        kernel_argument_abi_schema_requests
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_schema_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_schema_requests
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_schema_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_schema_requests
    );
    let lm_head_kernel_symbol = &lm_head_abi.candidates[0].kernel_symbol;
    let lm_head_schema_request = kernel_argument_abi_schema_requests
        .kernel_for(lm_head_kernel_symbol)
        .unwrap();
    assert!(lm_head_schema_request.schema_requested);
    assert!(lm_head_schema_request.schema_bound);
    assert_eq!(
        lm_head_schema_request.named_abi_schema_kernarg_size,
        Some(lm_head_schema_request.kernarg_size)
    );
    assert_eq!(
        lm_head_schema_request.named_abi_schema_kernarg_segment_align,
        Some(lm_head_schema_request.kernarg_segment_align)
    );
    assert!(lm_head_schema_request.named_abi_descriptor_match);
    assert!(lm_head_schema_request.dispatch_reference_count > 0);
    kernel_argument_abi_schema_requests.assert_consistent()?;
    let mut inconsistent_kernel_argument_abi_schema_requests =
        kernel_argument_abi_schema_requests.clone();
    inconsistent_kernel_argument_abi_schema_requests.schema_bound_count += 1;
    let err = inconsistent_kernel_argument_abi_schema_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("schema bound count"));
    let kernel_argument_abi_semantics =
        aql_packet_templates.kernel_argument_abi_semantic_plan(&kernarg_serialization)?;
    assert_eq!(kernel_argument_abi_semantics.target, catalog.target);
    assert_eq!(
        kernel_argument_abi_semantics.dispatch_count,
        aql_packet_templates.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_semantics.kernel_candidate_count,
        aql_packet_templates.kernel_candidate_count
    );
    assert!(kernel_argument_abi_semantics.semantic_schema_candidate_count > 0);
    assert_eq!(
        kernel_argument_abi_semantics.missing_semantic_schema_candidate_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_semantics.semantic_schema_candidate_count,
        kernel_argument_abi_semantics.kernel_candidate_count
    );
    assert!(kernel_argument_abi_semantics.semantic_descriptor_match_candidate_count > 0);
    assert_eq!(
        kernel_argument_abi_semantics.semantic_schema_candidate_count
            + kernel_argument_abi_semantics.missing_semantic_schema_candidate_count,
        kernel_argument_abi_semantics.kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantics.dispatches_with_semantic_verified_candidate_count
            + kernel_argument_abi_semantics.dispatches_without_semantic_verified_candidate_count,
        kernel_argument_abi_semantics.dispatch_count
    );
    assert!(!kernel_argument_abi_semantics.semantic_abi_ready);
    assert!(kernel_argument_abi_semantics
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert_eq!(
        kernarg_serialization.kernel_argument_abi_semantic_plan(&aql_packet_templates)?,
        kernel_argument_abi_semantics
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_plan(&device_pointer_validation)?,
        kernel_argument_abi_semantics
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_semantic_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantics
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_semantic_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantics
    );
    let lm_head_semantic = kernel_argument_abi_semantics
        .dispatch_for("lm_head")
        .unwrap();
    assert_eq!(
        lm_head_semantic.kernel_candidate_count,
        lm_head_template.kernel_candidate_count
    );
    let gemv_semantic = lm_head_semantic
        .candidates
        .iter()
        .find(|candidate| candidate.kernel_symbol == "gemv_f16")
        .unwrap();
    assert!(gemv_semantic.semantic_schema_available);
    assert!(gemv_semantic.semantic_descriptor_match);
    assert!(!gemv_semantic.semantic_verified);
    assert_eq!(
        gemv_semantic.primary_gap_reason,
        Some(RuntimeLaunchKernelArgumentAbiSemanticGapReason::FieldShapeMismatch)
    );
    assert_eq!(
        gemv_semantic.primary_gap_reason.unwrap().as_str(),
        "field_shape_mismatch"
    );
    assert!(gemv_semantic.field_schema_count > 0);
    assert!(gemv_semantic.field_mismatch_count > 0);
    assert!(gemv_semantic.extra_argument_count > 0);
    assert!(gemv_semantic
        .extra_model_argument_names
        .iter()
        .any(|argument| argument == "weight_format"));
    let weight_field = gemv_semantic
        .fields
        .iter()
        .find(|field| field.model_argument_name == "weight")
        .unwrap();
    assert_eq!(weight_field.expected_offset, 0);
    assert_eq!(weight_field.actual_offset, Some(8));
    assert!(!weight_field.offset_matches);
    assert!(!weight_field.field_verified);
    let out_features_field = gemv_semantic
        .fields
        .iter()
        .find(|field| field.model_argument_name == "out_features")
        .unwrap();
    assert_eq!(out_features_field.expected_encoding.as_str(), "u32");
    assert_eq!(
        out_features_field.actual_encoding.unwrap().as_str(),
        "usize_u64"
    );
    assert_eq!(out_features_field.expected_size_bytes, 4);
    assert_eq!(out_features_field.actual_size_bytes, Some(8));
    assert!(!out_features_field.encoding_matches);
    assert!(!out_features_field.size_matches);
    assert!(!out_features_field.field_verified);
    let sample_argmax_semantic = kernel_argument_abi_semantics
        .dispatch_for("sample_argmax")
        .unwrap();
    let argmax_step_semantic = sample_argmax_semantic
        .candidates
        .iter()
        .find(|candidate| candidate.kernel_symbol == "argmax_f32_step")
        .unwrap();
    assert!(argmax_step_semantic.semantic_schema_available);
    assert!(argmax_step_semantic.missing_field_count > 0);
    assert!(argmax_step_semantic
        .fields
        .iter()
        .any(|field| { field.model_argument_name == "step" && !field.model_argument_present }));
    kernel_argument_abi_semantics.assert_consistent()?;
    let kernel_argument_abi_semantic_gaps =
        kernel_argument_abi_semantics.kernel_argument_abi_semantic_gap_report()?;
    assert_eq!(
        kernel_argument_abi_semantic_gaps.source_dispatch_count,
        kernel_argument_abi_semantics.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_gaps.source_kernel_candidate_count,
        kernel_argument_abi_semantics.kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_gaps.dispatch_gap_count,
        kernel_argument_abi_semantics.dispatches_without_semantic_verified_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_gaps.all_dispatches_have_semantic_verified_candidate,
        kernel_argument_abi_semantics.dispatches_without_semantic_verified_candidate_count == 0
    );
    assert_eq!(
        kernel_argument_abi_semantic_gaps.gap_semantic_verified_candidate_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_semantic_gaps.gap_semantic_schema_candidate_count
            + kernel_argument_abi_semantic_gaps.gap_missing_semantic_schema_candidate_count,
        kernel_argument_abi_semantic_gaps.gap_kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_gaps.gap_primary_missing_semantic_schema_candidate_count
            + kernel_argument_abi_semantic_gaps
                .gap_primary_semantic_descriptor_mismatch_candidate_count
            + kernel_argument_abi_semantic_gaps.gap_primary_missing_model_argument_candidate_count
            + kernel_argument_abi_semantic_gaps.gap_primary_field_shape_mismatch_candidate_count
            + kernel_argument_abi_semantic_gaps.gap_primary_extra_model_argument_candidate_count
            + kernel_argument_abi_semantic_gaps.gap_primary_kernarg_size_shortfall_candidate_count
            + kernel_argument_abi_semantic_gaps
                .gap_primary_unknown_unverified_semantic_candidate_count,
        kernel_argument_abi_semantic_gaps.gap_kernel_candidate_count
    );
    assert!(kernel_argument_abi_semantic_gaps.dispatch_gap_count > 0);
    assert_eq!(
        kernel_argument_abi_semantic_gaps.gap_primary_missing_semantic_schema_candidate_count,
        0
    );
    assert!(
        kernel_argument_abi_semantic_gaps.gap_primary_missing_model_argument_candidate_count > 0
    );
    assert!(kernel_argument_abi_semantic_gaps.gap_primary_field_shape_mismatch_candidate_count > 0);
    assert!(kernel_argument_abi_semantic_gaps
        .missing_semantic_schema_kernel_symbols()
        .is_empty());
    assert_eq!(
        kernel_argument_abi_semantic_gaps
            .missing_model_argument_names()
            .as_slice(),
        &[
            "base_pos",
            "block_size",
            "candidate_logits",
            "candidate_token_ids",
            "eps",
            "last_page_len",
            "positions",
            "rmsnorm_output",
            "rmsnorm_weight",
            "seq_lens",
            "slot",
            "step",
            "token_ids",
        ]
    );
    let semantic_missing_requirements =
        kernel_argument_abi_semantic_gaps.missing_model_argument_requirements();
    assert!(semantic_missing_requirements.len() > 14);
    let mut semantic_requirement_names = semantic_missing_requirements
        .iter()
        .map(|requirement| requirement.model_argument_name.to_string())
        .collect::<Vec<_>>();
    semantic_requirement_names.sort();
    semantic_requirement_names.dedup();
    assert_eq!(
        semantic_requirement_names,
        kernel_argument_abi_semantic_gaps.missing_model_argument_names()
    );
    let semantic_field_mismatches = kernel_argument_abi_semantic_gaps.field_mismatch_diagnostics();
    assert_eq!(
        semantic_field_mismatches.len(),
        kernel_argument_abi_semantic_gaps.gap_field_mismatch_count
    );
    assert_eq!(
        aql_packet_templates.kernel_argument_abi_semantic_gap_report(&kernarg_serialization)?,
        kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        kernarg_serialization.kernel_argument_abi_semantic_gap_report(&aql_packet_templates)?,
        kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_gap_report(&device_pointer_validation)?,
        kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_semantic_gap_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantic_gaps
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_semantic_gap_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantic_gaps
    );
    let lm_head_semantic_gap = kernel_argument_abi_semantic_gaps
        .dispatch_for("lm_head")
        .unwrap();
    assert_eq!(
        lm_head_semantic_gap.kernel_candidate_count,
        lm_head_semantic.kernel_candidate_count
    );
    let gemv_semantic_gap = lm_head_semantic_gap
        .candidate_gaps
        .iter()
        .find(|candidate| candidate.kernel_symbol == "gemv_f16")
        .unwrap();
    assert_eq!(gemv_semantic_gap, gemv_semantic);
    let out_features_mismatch = semantic_field_mismatches
        .iter()
        .find(|diagnostic| {
            diagnostic.op_name == "lm_head"
                && diagnostic.kernel_symbol == "gemv_f16"
                && diagnostic.model_argument_name == "out_features"
        })
        .unwrap();
    assert_eq!(
        out_features_mismatch.op_index,
        lm_head_semantic_gap.op_index
    );
    assert_eq!(out_features_mismatch.kind, PrimitiveKind::Linear);
    assert_eq!(out_features_mismatch.stage_name, "output");
    assert_eq!(out_features_mismatch.stage_kind, ModelStageKind::Output);
    assert_eq!(
        out_features_mismatch.window_index,
        lm_head_semantic_gap.window_index
    );
    assert_eq!(
        out_features_mismatch.queue_packet_index,
        lm_head_semantic_gap.queue_packet_index
    );
    assert_eq!(out_features_mismatch.field_index, 3);
    assert_eq!(out_features_mismatch.kernel_argument_name, "N");
    assert_eq!(out_features_mismatch.model_argument_name, "out_features");
    assert_eq!(
        out_features_mismatch.expected_kind,
        RuntimeLaunchKernargArgumentKind::Scalar
    );
    assert_eq!(
        out_features_mismatch.expected_encoding,
        RuntimeLaunchKernelArgumentAbiSemanticEncoding::U32
    );
    assert_eq!(out_features_mismatch.expected_offset, 24);
    assert_eq!(out_features_mismatch.expected_size_bytes, 4);
    assert_eq!(
        out_features_mismatch.actual_argument_index,
        out_features_field.actual_argument_index
    );
    assert_eq!(
        out_features_mismatch.actual_kind,
        Some(RuntimeLaunchKernargArgumentKind::Scalar)
    );
    assert_eq!(
        out_features_mismatch.actual_encoding,
        Some(RuntimeLaunchKernelArgumentAbiSemanticEncoding::UsizeU64)
    );
    assert_eq!(
        out_features_mismatch.actual_offset,
        out_features_field.actual_offset
    );
    assert_eq!(out_features_mismatch.actual_size_bytes, Some(8));
    assert!(out_features_mismatch.kind_matches);
    assert!(!out_features_mismatch.encoding_matches);
    assert_eq!(
        out_features_mismatch.offset_matches,
        out_features_field.offset_matches
    );
    assert!(!out_features_mismatch.size_matches);
    let sample_argmax_semantic_gap = kernel_argument_abi_semantic_gaps
        .dispatch_for("sample_argmax")
        .unwrap();
    let argmax_step_semantic_gap = sample_argmax_semantic_gap
        .candidate_gaps
        .iter()
        .find(|candidate| candidate.kernel_symbol == "argmax_f32_step")
        .unwrap();
    assert_eq!(argmax_step_semantic_gap, argmax_step_semantic);
    assert_eq!(
        argmax_step_semantic_gap.primary_gap_reason,
        Some(RuntimeLaunchKernelArgumentAbiSemanticGapReason::MissingModelArgument)
    );
    let semantic_step_requirement = semantic_missing_requirements
        .iter()
        .find(|requirement| {
            requirement.op_name == "sample_argmax"
                && requirement.kernel_symbol == "argmax_f32_step"
                && requirement.model_argument_name == "step"
        })
        .unwrap();
    assert_eq!(
        semantic_step_requirement.op_index,
        sample_argmax_semantic_gap.op_index
    );
    assert_eq!(semantic_step_requirement.kind, PrimitiveKind::ArgmaxSample);
    assert_eq!(semantic_step_requirement.stage_name, "sampling");
    assert_eq!(
        semantic_step_requirement.stage_kind,
        ModelStageKind::Sampling
    );
    assert_eq!(
        semantic_step_requirement.window_index,
        sample_argmax_semantic_gap.window_index
    );
    assert_eq!(
        semantic_step_requirement.queue_packet_index,
        sample_argmax_semantic_gap.queue_packet_index
    );
    assert_eq!(semantic_step_requirement.field_index, 2);
    assert_eq!(semantic_step_requirement.kernel_argument_name, "step");
    assert_eq!(
        semantic_step_requirement.expected_kind,
        RuntimeLaunchKernargArgumentKind::Pointer
    );
    assert_eq!(
        semantic_step_requirement.expected_encoding,
        RuntimeLaunchKernelArgumentAbiSemanticEncoding::DevicePointerU64
    );
    assert_eq!(semantic_step_requirement.expected_offset, 16);
    assert_eq!(semantic_step_requirement.expected_size_bytes, 8);
    kernel_argument_abi_semantic_gaps.assert_consistent()?;
    let mut inconsistent_kernel_argument_abi_semantic_gaps =
        kernel_argument_abi_semantic_gaps.clone();
    inconsistent_kernel_argument_abi_semantic_gaps.dispatch_gap_count += 1;
    let err = inconsistent_kernel_argument_abi_semantic_gaps
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic ABI gap dispatch entries"));
    let kernel_argument_abi_semantic_projection = aql_packet_templates
        .kernel_argument_abi_semantic_projection_plan(&kernarg_serialization)?;
    kernel_argument_abi_semantic_projection.assert_consistent()?;
    assert_eq!(
        kernel_argument_abi_semantic_projection.dispatch_count,
        kernel_argument_abi_semantics.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.kernel_candidate_count,
        kernel_argument_abi_semantics.kernel_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.semantic_schema_candidate_count,
        kernel_argument_abi_semantics.semantic_schema_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.missing_semantic_schema_candidate_count,
        kernel_argument_abi_semantics.missing_semantic_schema_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.semantic_descriptor_match_candidate_count,
        kernel_argument_abi_semantics.semantic_descriptor_match_candidate_count
    );
    assert!(kernel_argument_abi_semantic_projection.projection_ready_candidate_count > 0);
    assert!(
        kernel_argument_abi_semantic_projection.projection_ready_candidate_count
            > kernel_argument_abi_semantics.semantic_verified_candidate_count
    );
    assert!(
        kernel_argument_abi_semantic_projection.projected_field_count
            > kernel_argument_abi_semantics.verified_field_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.missing_field_count,
        kernel_argument_abi_semantics.missing_field_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.kind_mismatch_field_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.unsupported_encoding_field_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.scalar_narrowing_overflow_field_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection.field_range_overflow_count,
        0
    );
    assert!(!kernel_argument_abi_semantic_projection.semantic_projection_ready);
    assert!(kernel_argument_abi_semantic_projection
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_verification"));
    assert!(kernel_argument_abi_semantic_projection
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_semantic_projection"));
    assert_eq!(
        kernarg_serialization
            .kernel_argument_abi_semantic_projection_plan(&aql_packet_templates)?,
        kernel_argument_abi_semantic_projection
    );
    assert_eq!(
        launch_preflight
            .kernel_argument_abi_semantic_projection_plan(&device_pointer_validation)?,
        kernel_argument_abi_semantic_projection
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_semantic_projection_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantic_projection
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_semantic_projection_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantic_projection
    );
    let kernel_argument_abi_semantic_projection_gaps = kernel_argument_abi_semantic_projection
        .kernel_argument_abi_semantic_projection_gap_report()?;
    kernel_argument_abi_semantic_projection_gaps.assert_consistent()?;
    assert_eq!(
        kernel_argument_abi_semantic_projection_gaps.source_dispatch_count,
        kernel_argument_abi_semantic_projection.dispatch_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection_gaps.dispatch_gap_count,
        kernel_argument_abi_semantic_projection.dispatches_without_projection_ready_candidate_count
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection_gaps.gap_projection_ready_candidate_count,
        0
    );
    assert_eq!(
        kernel_argument_abi_semantic_projection_gaps
            .gap_primary_missing_semantic_schema_candidate_count,
        0
    );
    assert!(kernel_argument_abi_semantic_projection_gaps
        .missing_semantic_schema_kernel_symbols()
        .is_empty());
    assert_eq!(
        kernel_argument_abi_semantic_projection_gaps
            .missing_model_argument_names()
            .as_slice(),
        &[
            "base_pos",
            "block_size",
            "candidate_logits",
            "candidate_token_ids",
            "eps",
            "last_page_len",
            "positions",
            "rmsnorm_output",
            "rmsnorm_weight",
            "seq_lens",
            "slot",
            "step",
            "token_ids",
        ]
    );
    let projection_missing_requirements =
        kernel_argument_abi_semantic_projection_gaps.missing_model_argument_requirements();
    assert!(projection_missing_requirements.len() > 14);
    let mut projection_requirement_names = projection_missing_requirements
        .iter()
        .map(|requirement| requirement.model_argument_name.to_string())
        .collect::<Vec<_>>();
    projection_requirement_names.sort();
    projection_requirement_names.dedup();
    assert_eq!(
        projection_requirement_names,
        kernel_argument_abi_semantic_projection_gaps.missing_model_argument_names()
    );
    let projection_field_blockers =
        kernel_argument_abi_semantic_projection_gaps.field_blocker_diagnostics();
    assert_eq!(
        projection_field_blockers.len(),
        kernel_argument_abi_semantic_projection_gaps.gap_missing_field_count
            + kernel_argument_abi_semantic_projection_gaps.gap_kind_mismatch_field_count
            + kernel_argument_abi_semantic_projection_gaps.gap_unsupported_encoding_field_count
            + kernel_argument_abi_semantic_projection_gaps
                .gap_scalar_narrowing_overflow_field_count
            + kernel_argument_abi_semantic_projection_gaps.gap_field_range_overflow_count
    );
    assert_eq!(
        projection_field_blockers
            .iter()
            .filter(|diagnostic| {
                diagnostic.projection_status
                    == RuntimeLaunchKernelArgumentAbiSemanticProjectionStatus::MissingModelArgument
            })
            .count(),
        kernel_argument_abi_semantic_projection_gaps.gap_missing_field_count
    );
    assert_eq!(
        projection_field_blockers
            .iter()
            .filter(|diagnostic| {
                diagnostic.projection_status
                    == RuntimeLaunchKernelArgumentAbiSemanticProjectionStatus::KindMismatch
            })
            .count(),
        kernel_argument_abi_semantic_projection_gaps.gap_kind_mismatch_field_count
    );
    let projection_step_blocker = projection_field_blockers
        .iter()
        .find(|diagnostic| {
            diagnostic.op_name == "sample_argmax"
                && diagnostic.kernel_symbol == "argmax_f32_step"
                && diagnostic.model_argument_name == "step"
        })
        .unwrap();
    assert_eq!(
        projection_step_blocker.projection_status,
        RuntimeLaunchKernelArgumentAbiSemanticProjectionStatus::MissingModelArgument
    );
    assert!(!projection_step_blocker.model_argument_present);
    assert_eq!(
        projection_step_blocker.expected_kind,
        RuntimeLaunchKernargArgumentKind::Pointer
    );
    assert_eq!(
        projection_step_blocker.expected_encoding,
        RuntimeLaunchKernelArgumentAbiSemanticEncoding::DevicePointerU64
    );
    assert_eq!(projection_step_blocker.actual_argument_index, None);
    assert_eq!(projection_step_blocker.actual_kind, None);
    assert_eq!(projection_step_blocker.actual_encoding, None);
    assert_eq!(projection_step_blocker.actual_size_bytes, None);
    assert_eq!(projection_step_blocker.projected_offset, None);
    assert_eq!(projection_step_blocker.projected_size_bytes, 0);
    assert!(
        kernel_argument_abi_semantic_projection_gaps
            .gap_primary_missing_model_argument_candidate_count
            > 0
    );
    assert!(
        !kernel_argument_abi_semantic_projection_gaps
            .all_dispatches_have_projection_ready_candidate
    );
    assert!(kernel_argument_abi_semantic_projection_gaps
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_semantic_projection"));
    assert_eq!(
        aql_packet_templates
            .kernel_argument_abi_semantic_projection_gap_report(&kernarg_serialization)?,
        kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        kernarg_serialization
            .kernel_argument_abi_semantic_projection_gap_report(&aql_packet_templates)?,
        kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        launch_preflight
            .kernel_argument_abi_semantic_projection_gap_report(&device_pointer_validation)?,
        kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_semantic_projection_gap_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantic_projection_gaps
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_semantic_projection_gap_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_argument_abi_semantic_projection_gaps
    );
    let lm_head_projection = kernel_argument_abi_semantic_projection
        .dispatch_for("lm_head")
        .unwrap();
    let gemv_projection = lm_head_projection
        .candidates
        .iter()
        .find(|candidate| candidate.kernel_symbol == "gemv_f16")
        .unwrap();
    assert!(gemv_projection.projection_ready);
    assert_eq!(gemv_projection.primary_projection_blocker, None);
    assert_eq!(
        gemv_projection.projected_kernarg_bytes,
        gemv_projection.kernarg_size as usize
    );
    assert_eq!(
        gemv_projection.projected_byte_image.len(),
        gemv_projection.semantic_schema_kernarg_size.unwrap() as usize
    );
    assert!(gemv_projection.fields.iter().all(|field| {
        field.projection_status == RuntimeLaunchKernelArgumentAbiSemanticProjectionStatus::Projected
    }));
    let gemv_n_projection = gemv_projection
        .fields
        .iter()
        .find(|field| field.kernel_argument_name == "N")
        .unwrap();
    assert_eq!(gemv_n_projection.expected_size_bytes, 4);
    assert_eq!(gemv_n_projection.projected_bytes.len(), 4);
    assert_eq!(
        gemv_n_projection.actual_encoding,
        Some(RuntimeLaunchKernelArgumentAbiSemanticEncoding::UsizeU64)
    );
    let sample_argmax_projection = kernel_argument_abi_semantic_projection
        .dispatch_for("sample_argmax")
        .unwrap();
    let argmax_step_projection = sample_argmax_projection
        .candidates
        .iter()
        .find(|candidate| candidate.kernel_symbol == "argmax_f32_step")
        .unwrap();
    assert!(!argmax_step_projection.projection_ready);
    assert_eq!(
        argmax_step_projection.primary_projection_blocker,
        Some(RuntimeLaunchKernelArgumentAbiSemanticProjectionStatus::MissingModelArgument)
    );
    assert!(argmax_step_projection.fields.iter().any(|field| {
        field.model_argument_name == "step"
            && field.projection_status
                == RuntimeLaunchKernelArgumentAbiSemanticProjectionStatus::MissingModelArgument
    }));
    let projection_step_requirement = projection_missing_requirements
        .iter()
        .find(|requirement| {
            requirement.op_name == "sample_argmax"
                && requirement.kernel_symbol == "argmax_f32_step"
                && requirement.model_argument_name == "step"
        })
        .unwrap();
    assert_eq!(
        projection_step_requirement.op_index,
        sample_argmax_projection.op_index
    );
    assert_eq!(
        projection_step_requirement.kind,
        PrimitiveKind::ArgmaxSample
    );
    assert_eq!(projection_step_requirement.stage_name, "sampling");
    assert_eq!(
        projection_step_requirement.stage_kind,
        ModelStageKind::Sampling
    );
    assert_eq!(
        projection_step_requirement.window_index,
        sample_argmax_projection.window_index
    );
    assert_eq!(
        projection_step_requirement.queue_packet_index,
        sample_argmax_projection.queue_packet_index
    );
    assert_eq!(projection_step_requirement.field_index, 2);
    assert_eq!(projection_step_requirement.kernel_argument_name, "step");
    assert_eq!(
        projection_step_requirement.expected_kind,
        RuntimeLaunchKernargArgumentKind::Pointer
    );
    assert_eq!(
        projection_step_requirement.expected_encoding,
        RuntimeLaunchKernelArgumentAbiSemanticEncoding::DevicePointerU64
    );
    assert_eq!(projection_step_requirement.expected_offset, 16);
    assert_eq!(projection_step_requirement.expected_size_bytes, 8);
    let mut inconsistent_kernel_argument_abi_semantic_projection =
        kernel_argument_abi_semantic_projection.clone();
    inconsistent_kernel_argument_abi_semantic_projection.projected_field_count += 1;
    let err = inconsistent_kernel_argument_abi_semantic_projection
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic projection projected field count"));
    let mut inconsistent_kernel_argument_abi_semantic_projection_gaps =
        kernel_argument_abi_semantic_projection_gaps.clone();
    inconsistent_kernel_argument_abi_semantic_projection_gaps.dispatch_gap_count += 1;
    let err = inconsistent_kernel_argument_abi_semantic_projection_gaps
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic projection gap dispatch entries"));
    let semantic_projection_candidate_recommendations = kernel_argument_abi_semantic_projection
        .kernel_argument_abi_semantic_projection_candidate_recommendation_plan()?;
    semantic_projection_candidate_recommendations.assert_consistent()?;
    assert_eq!(
        semantic_projection_candidate_recommendations.dispatch_count,
        kernel_argument_abi_semantic_projection.dispatch_count
    );
    assert_eq!(
        semantic_projection_candidate_recommendations.source_kernel_candidate_count,
        kernel_argument_abi_semantic_projection.kernel_candidate_count
    );
    assert_eq!(
        semantic_projection_candidate_recommendations.recommended_dispatch_count,
        kernel_argument_abi_semantic_projection.dispatches_with_projection_ready_candidate_count
    );
    assert_eq!(
        semantic_projection_candidate_recommendations.missing_recommendation_dispatch_count,
        kernel_argument_abi_semantic_projection.dispatches_without_projection_ready_candidate_count
    );
    assert!(semantic_projection_candidate_recommendations.recommended_dispatch_count > 0);
    assert!(
        semantic_projection_candidate_recommendations.missing_recommendation_dispatch_count > 0
    );
    assert!(semantic_projection_candidate_recommendations.recommended_projected_kernarg_bytes > 0);
    assert_eq!(
        semantic_projection_candidate_recommendations.selection_applied_count,
        0
    );
    assert!(!semantic_projection_candidate_recommendations.all_dispatches_recommended);
    assert_eq!(
        semantic_projection_candidate_recommendations.policy,
        "first_projection_ready_candidate_in_host_launcher_order"
    );
    assert!(semantic_projection_candidate_recommendations
        .unresolved_runtime_requirements
        .contains(&"kernel_candidate_selection_policy"));
    assert!(semantic_projection_candidate_recommendations
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_semantic_projection"));
    assert_eq!(
        aql_packet_templates
            .kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &kernarg_serialization,
            )?,
        semantic_projection_candidate_recommendations
    );
    assert_eq!(
        kernarg_serialization
            .kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &aql_packet_templates,
            )?,
        semantic_projection_candidate_recommendations
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
            &device_pointer_validation,
        )?,
        semantic_projection_candidate_recommendations
    );
    assert_eq!(
        admission
            .runtime_launch_kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &code_object,
                DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
                &device_pointer_validation,
            )?,
        semantic_projection_candidate_recommendations
    );
    assert_eq!(
        readiness
            .runtime_launch_kernel_argument_abi_semantic_projection_candidate_recommendation_plan(
                &slot_bindings,
                &code_object,
                DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
                &device_pointer_validation,
            )?,
        semantic_projection_candidate_recommendations
    );
    let ready_projection_candidate_recommendation = semantic_projection_candidate_recommendations
        .dispatches
        .iter()
        .find(|dispatch| {
            dispatch.state
                == RuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationState::RecommendedProjectionReadyCandidate
        })
        .unwrap();
    assert_eq!(
        ready_projection_candidate_recommendation.recommendation_reason,
        "first_projection_ready_candidate"
    );
    assert!(ready_projection_candidate_recommendation.recommended_candidate_projection_ready);
    assert!(ready_projection_candidate_recommendation
        .recommended_kernel_symbol
        .is_some());
    assert!(!ready_projection_candidate_recommendation.selection_applied);
    let missing_projection_candidate_recommendation =
        semantic_projection_candidate_recommendations
            .dispatches
            .iter()
            .find(|dispatch| {
                dispatch.state
                    == RuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationState::NoProjectionReadyCandidate
            })
            .unwrap();
    assert_eq!(
        missing_projection_candidate_recommendation.recommendation_reason,
        "no_projection_ready_candidate"
    );
    assert!(missing_projection_candidate_recommendation
        .recommended_kernel_symbol
        .is_none());
    let mut inconsistent_semantic_projection_candidate_recommendations =
        semantic_projection_candidate_recommendations.clone();
    inconsistent_semantic_projection_candidate_recommendations.recommended_dispatch_count += 1;
    let err = inconsistent_semantic_projection_candidate_recommendations
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic projection candidate recommendation recommended dispatch count"));
    let mut inconsistent_semantic_projection_candidate_recommendations =
        semantic_projection_candidate_recommendations.clone();
    let invalid_index_projection_candidate_recommendation =
        inconsistent_semantic_projection_candidate_recommendations
            .dispatches
            .iter_mut()
            .find(|dispatch| {
                dispatch.state
                    == RuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationState::RecommendedProjectionReadyCandidate
            })
            .unwrap();
    invalid_index_projection_candidate_recommendation.recommended_candidate_index =
        Some(invalid_index_projection_candidate_recommendation.source_kernel_candidate_count);
    let err = inconsistent_semantic_projection_candidate_recommendations
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("recommended candidate index"));
    let mut inconsistent_semantic_projection_candidate_recommendations =
        semantic_projection_candidate_recommendations.clone();
    let missing_ready_projection_candidate_recommendation =
        inconsistent_semantic_projection_candidate_recommendations
            .dispatches
            .iter_mut()
            .find(|dispatch| {
                dispatch.state
                    == RuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationState::RecommendedProjectionReadyCandidate
            })
            .unwrap();
    let projected_bytes = missing_ready_projection_candidate_recommendation
        .recommended_projected_kernarg_bytes
        .unwrap();
    missing_ready_projection_candidate_recommendation.state =
        RuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationState::NoProjectionReadyCandidate;
    missing_ready_projection_candidate_recommendation.recommended_candidate_index = None;
    missing_ready_projection_candidate_recommendation.recommended_kernel_symbol = None;
    missing_ready_projection_candidate_recommendation.recommended_kernarg_size = None;
    missing_ready_projection_candidate_recommendation.recommended_projected_kernarg_bytes = None;
    missing_ready_projection_candidate_recommendation.recommended_candidate_projection_ready =
        false;
    missing_ready_projection_candidate_recommendation.recommendation_reason =
        "no_projection_ready_candidate";
    inconsistent_semantic_projection_candidate_recommendations.recommended_dispatch_count -= 1;
    inconsistent_semantic_projection_candidate_recommendations
        .missing_recommendation_dispatch_count += 1;
    inconsistent_semantic_projection_candidate_recommendations
        .recommended_projected_kernarg_bytes -= projected_bytes;
    let err = inconsistent_semantic_projection_candidate_recommendations
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("has projection-ready candidates but no projection candidate recommendation")
    );
    let semantic_projection_candidate_selection_requests =
        semantic_projection_candidate_recommendations
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &kernel_argument_abi_semantic_projection,
            )?;
    semantic_projection_candidate_selection_requests
        .assert_consistent_with_projection(&kernel_argument_abi_semantic_projection)?;
    assert_eq!(
        semantic_projection_candidate_selection_requests.dispatch_count,
        semantic_projection_candidate_recommendations.dispatch_count
    );
    assert_eq!(
        semantic_projection_candidate_selection_requests.selection_request_count,
        semantic_projection_candidate_recommendations.recommended_dispatch_count
    );
    assert_eq!(
        semantic_projection_candidate_selection_requests.missing_selection_request_count,
        semantic_projection_candidate_recommendations.missing_recommendation_dispatch_count
    );
    assert_eq!(
        semantic_projection_candidate_selection_requests.requested_projected_kernarg_bytes,
        semantic_projection_candidate_recommendations.recommended_projected_kernarg_bytes
    );
    assert_eq!(
        semantic_projection_candidate_selection_requests.selection_applied_count,
        0
    );
    assert!(semantic_projection_candidate_selection_requests.request_plan_ready);
    assert!(!semantic_projection_candidate_selection_requests.all_selection_requests_ready);
    assert_eq!(
        semantic_projection_candidate_selection_requests
            .selection_request_op_names()
            .as_slice(),
        &["lm_head"]
    );
    assert_eq!(
        semantic_projection_candidate_selection_requests
            .missing_selection_request_op_names()
            .as_slice(),
        &["embed_tokens", "sample_argmax"]
    );
    assert!(semantic_projection_candidate_selection_requests
        .unresolved_runtime_requirements
        .contains(&"kernel_candidate_selection_policy"));
    assert!(semantic_projection_candidate_selection_requests
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_semantic_projection"));
    let mut stale_semantic_projection_candidate_recommendations =
        semantic_projection_candidate_recommendations.clone();
    let stale_semantic_projection_candidate_recommendation =
        stale_semantic_projection_candidate_recommendations
            .dispatches
            .iter_mut()
            .find(|dispatch| dispatch.source_ambiguous_candidate_set)
            .unwrap();
    stale_semantic_projection_candidate_recommendation.source_kernel_candidate_count += 1;
    stale_semantic_projection_candidate_recommendation.missing_semantic_schema_candidate_count += 1;
    stale_semantic_projection_candidate_recommendations.source_kernel_candidate_count += 1;
    stale_semantic_projection_candidate_recommendations.missing_semantic_schema_candidate_count +=
        1;
    stale_semantic_projection_candidate_recommendations.assert_consistent()?;
    let err = stale_semantic_projection_candidate_recommendations
        .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
            &kernel_argument_abi_semantic_projection,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("recommendation/projection counts mismatch"));
    assert_eq!(
        kernel_argument_abi_semantic_projection
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan()?,
        semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        aql_packet_templates
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &kernarg_serialization,
            )?,
        semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        kernarg_serialization
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &aql_packet_templates,
            )?,
        semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
            &device_pointer_validation,
        )?,
        semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        admission
            .runtime_launch_kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &code_object,
                DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
                &device_pointer_validation,
            )?,
        semantic_projection_candidate_selection_requests
    );
    assert_eq!(
        readiness
            .runtime_launch_kernel_argument_abi_semantic_projection_candidate_selection_request_plan(
                &slot_bindings,
                &code_object,
                DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
                &device_pointer_validation,
            )?,
        semantic_projection_candidate_selection_requests
    );
    let ready_projection_candidate_selection_request =
        semantic_projection_candidate_selection_requests
            .dispatch_for(&ready_projection_candidate_recommendation.op_name)
            .unwrap();
    assert!(ready_projection_candidate_selection_request.selection_request_ready);
    assert!(ready_projection_candidate_selection_request.requested_candidate_projection_ready);
    assert_eq!(
        ready_projection_candidate_selection_request.selection_request_reason,
        "first_projection_ready_candidate"
    );
    assert_eq!(
        ready_projection_candidate_selection_request
            .requested_projected_byte_image
            .len(),
        ready_projection_candidate_selection_request
            .requested_projected_kernarg_bytes
            .unwrap()
    );
    assert_eq!(
        ready_projection_candidate_selection_request
            .requested_kernarg_size
            .unwrap() as usize,
        ready_projection_candidate_selection_request
            .requested_projected_byte_image
            .len()
    );
    assert!(!ready_projection_candidate_selection_request.selection_applied);
    let missing_projection_candidate_selection_request =
        semantic_projection_candidate_selection_requests
            .dispatch_for(&missing_projection_candidate_recommendation.op_name)
            .unwrap();
    assert!(!missing_projection_candidate_selection_request.selection_request_ready);
    assert_eq!(
        missing_projection_candidate_selection_request.selection_request_reason,
        "no_projection_ready_candidate"
    );
    assert!(missing_projection_candidate_selection_request
        .requested_projected_byte_image
        .is_empty());
    assert!(missing_projection_candidate_selection_request
        .requested_kernel_symbol
        .is_none());
    let mut inconsistent_semantic_projection_candidate_selection_requests =
        semantic_projection_candidate_selection_requests.clone();
    inconsistent_semantic_projection_candidate_selection_requests.selection_request_count += 1;
    let err = inconsistent_semantic_projection_candidate_selection_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic projection candidate selection request count"));
    let mut inconsistent_semantic_projection_candidate_selection_requests =
        semantic_projection_candidate_selection_requests.clone();
    let invalid_projection_candidate_selection_request =
        inconsistent_semantic_projection_candidate_selection_requests
            .dispatches
            .iter_mut()
            .find(|dispatch| dispatch.selection_request_ready)
            .unwrap();
    invalid_projection_candidate_selection_request
        .requested_projected_byte_image
        .push(0);
    let err = inconsistent_semantic_projection_candidate_selection_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("requested projected byte image len"));
    let mut inconsistent_semantic_projection_candidate_selection_requests =
        semantic_projection_candidate_selection_requests.clone();
    let invalid_projection_candidate_selection_request =
        inconsistent_semantic_projection_candidate_selection_requests
            .dispatches
            .iter_mut()
            .find(|dispatch| dispatch.selection_request_ready)
            .unwrap();
    invalid_projection_candidate_selection_request.requested_projected_byte_image[0] ^= 0xff;
    let err = inconsistent_semantic_projection_candidate_selection_requests
        .assert_consistent_with_projection(&kernel_argument_abi_semantic_projection)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requested projected byte image does not match projection candidate"));
    let mut inconsistent_kernel_argument_abi_semantics = kernel_argument_abi_semantics.clone();
    inconsistent_kernel_argument_abi_semantics.semantic_schema_candidate_count += 1;
    let err = inconsistent_kernel_argument_abi_semantics
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic schema candidate count"));
    let kernel_recommendations = kernel_argument_abi.kernel_candidate_recommendation_plan()?;
    assert_eq!(
        kernel_recommendations.dispatch_count,
        kernel_argument_abi.dispatch_count
    );
    assert_eq!(
        kernel_recommendations.source_kernel_candidate_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert_eq!(
        kernel_recommendations.size_compatible_candidate_count,
        kernel_argument_abi.size_compatible_candidate_count
    );
    assert_eq!(
        kernel_recommendations.verified_candidate_count,
        kernel_argument_abi.verified_candidate_count
    );
    assert_eq!(
        kernel_recommendations.recommended_dispatch_count
            + kernel_recommendations.missing_recommendation_dispatch_count,
        kernel_recommendations.dispatch_count
    );
    assert!(kernel_recommendations.recommended_dispatch_count > 0);
    assert_eq!(
        kernel_recommendations.all_dispatches_recommended,
        kernel_recommendations.missing_recommendation_dispatch_count == 0
    );
    assert_eq!(kernel_recommendations.selection_applied_count, 0);
    assert!(kernel_recommendations
        .unresolved_runtime_requirements
        .contains(&"kernel_candidate_selection_policy"));
    assert_eq!(
        admission.runtime_launch_kernel_candidate_recommendation_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_recommendations
    );
    assert_eq!(
        readiness.runtime_launch_kernel_candidate_recommendation_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_recommendations
    );
    let recommended_dispatch = kernel_recommendations
        .dispatches
        .iter()
        .find(|dispatch| {
            dispatch.state
                == RuntimeLaunchKernelCandidateRecommendationState::RecommendedVerifiedCandidate
        })
        .unwrap();
    assert_eq!(
        recommended_dispatch.state,
        RuntimeLaunchKernelCandidateRecommendationState::RecommendedVerifiedCandidate
    );
    assert_eq!(
        recommended_dispatch.recommended_capacity_shortfall_bytes,
        Some(0)
    );
    assert!(recommended_dispatch.recommended_candidate_verified);
    assert!(!recommended_dispatch.selection_applied);
    kernel_recommendations.assert_consistent()?;
    let semantic_projection_recommendations = kernel_recommendations
        .kernel_argument_abi_semantic_projection_recommendation_report(
            &kernel_argument_abi_semantic_projection,
        )?;
    semantic_projection_recommendations.assert_consistent()?;
    assert_eq!(
        semantic_projection_recommendations.dispatch_count,
        kernel_recommendations.dispatch_count
    );
    assert_eq!(
        semantic_projection_recommendations.source_kernel_candidate_count,
        kernel_recommendations.source_kernel_candidate_count
    );
    assert_eq!(
        semantic_projection_recommendations.projection_kernel_candidate_count,
        kernel_argument_abi_semantic_projection.kernel_candidate_count
    );
    assert_eq!(
        semantic_projection_recommendations.recommended_dispatch_count,
        kernel_recommendations.recommended_dispatch_count
    );
    assert_eq!(
        semantic_projection_recommendations.missing_recommendation_dispatch_count,
        kernel_recommendations.missing_recommendation_dispatch_count
    );
    assert_eq!(
        semantic_projection_recommendations.dispatches_with_projection_ready_candidate_count,
        kernel_argument_abi_semantic_projection.dispatches_with_projection_ready_candidate_count
    );
    assert_eq!(
        semantic_projection_recommendations.recommended_projection_ready_dispatch_count
            + semantic_projection_recommendations
                .recommended_without_projection_ready_dispatch_count,
        semantic_projection_recommendations.recommended_dispatch_count
    );
    assert!(
        semantic_projection_recommendations.recommended_without_projection_ready_dispatch_count > 0
    );
    assert!(
        !semantic_projection_recommendations.all_dispatches_have_projection_ready_recommendation
    );
    assert!(semantic_projection_recommendations
        .unresolved_runtime_requirements
        .contains(&"kernel_candidate_selection_policy"));
    assert!(semantic_projection_recommendations
        .unresolved_runtime_requirements
        .contains(&"kernel_argument_abi_semantic_projection"));
    assert_eq!(
        kernel_argument_abi_semantic_projection
            .kernel_argument_abi_semantic_projection_recommendation_report(
                &kernel_recommendations,
            )?,
        semantic_projection_recommendations
    );
    assert_eq!(
        aql_packet_templates.kernel_argument_abi_semantic_projection_recommendation_report(
            &kernarg_serialization
        )?,
        semantic_projection_recommendations
    );
    assert_eq!(
        kernarg_serialization
            .kernel_argument_abi_semantic_projection_recommendation_report(&aql_packet_templates)?,
        semantic_projection_recommendations
    );
    assert_eq!(
        launch_preflight.kernel_argument_abi_semantic_projection_recommendation_report(
            &device_pointer_validation,
        )?,
        semantic_projection_recommendations
    );
    assert_eq!(
        admission.runtime_launch_kernel_argument_abi_semantic_projection_recommendation_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        semantic_projection_recommendations
    );
    assert_eq!(
        readiness.runtime_launch_kernel_argument_abi_semantic_projection_recommendation_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        semantic_projection_recommendations
    );
    let recommended_projection = semantic_projection_recommendations
        .dispatch_for(&recommended_dispatch.op_name)
        .unwrap();
    assert_eq!(
        recommended_projection.recommended_kernel_symbol,
        recommended_dispatch.recommended_kernel_symbol
    );
    assert!(recommended_projection.recommended_candidate_projection_found);
    if semantic_projection_recommendations.recommended_projection_ready_dispatch_count > 0 {
        let ready_projection_recommendation = semantic_projection_recommendations
            .dispatches
            .iter()
            .find(|dispatch| dispatch.recommended_candidate_projection_ready)
            .unwrap();
        assert_eq!(
            ready_projection_recommendation.recommendation_projection_reason,
            "recommended_candidate_projection_ready"
        );
    }
    let blocked_projection_recommendation = semantic_projection_recommendations
        .dispatches
        .iter()
        .find(|dispatch| {
            dispatch.recommendation_state
                == RuntimeLaunchKernelCandidateRecommendationState::RecommendedVerifiedCandidate
                && !dispatch.recommended_candidate_projection_ready
        })
        .unwrap();
    assert_eq!(
        blocked_projection_recommendation.recommendation_projection_reason,
        "recommended_candidate_projection_blocked"
    );
    let missing_recommendation_projection = semantic_projection_recommendations
        .dispatches
        .iter()
        .find(|dispatch| {
            dispatch.recommendation_state
                == RuntimeLaunchKernelCandidateRecommendationState::NoVerifiedCandidate
        })
        .unwrap();
    assert_eq!(
        missing_recommendation_projection.recommendation_projection_reason,
        "no_recommended_candidate"
    );
    let mut inconsistent_semantic_projection_recommendations =
        semantic_projection_recommendations.clone();
    inconsistent_semantic_projection_recommendations.recommended_projection_ready_dispatch_count +=
        1;
    let err = inconsistent_semantic_projection_recommendations
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic projection recommendation ready dispatch count"));
    let kernel_selection_requests =
        kernel_recommendations.kernel_candidate_selection_request_plan()?;
    assert_eq!(
        kernel_selection_requests.dispatch_count,
        kernel_recommendations.dispatch_count
    );
    assert_eq!(
        kernel_selection_requests.selection_request_count,
        kernel_recommendations.recommended_dispatch_count
    );
    assert_eq!(
        kernel_selection_requests.missing_selection_request_count,
        kernel_recommendations.missing_recommendation_dispatch_count
    );
    assert_eq!(
        kernel_selection_requests.verified_candidate_count,
        kernel_recommendations.verified_candidate_count
    );
    assert_eq!(kernel_selection_requests.selection_applied_count, 0);
    let selection_request = kernel_selection_requests
        .dispatch_for(&recommended_dispatch.op_name)
        .unwrap();
    assert!(selection_request.selection_request_ready);
    assert_eq!(
        selection_request.requested_kernel_symbol,
        recommended_dispatch.recommended_kernel_symbol
    );
    assert!(selection_request.requested_candidate_verified);
    assert_eq!(
        kernel_selection_requests.all_selection_requests_ready,
        kernel_selection_requests.missing_selection_request_count == 0
    );
    assert!(kernel_selection_requests.request_plan_ready);
    assert_eq!(
        kernel_selection_requests
            .selection_request_op_names()
            .as_slice(),
        &["embed_tokens", "sample_argmax"]
    );
    assert_eq!(
        kernel_selection_requests
            .missing_selection_request_op_names()
            .as_slice(),
        &["lm_head"]
    );
    assert_eq!(
        launch_preflight.kernel_candidate_selection_request_plan(&device_pointer_validation)?,
        kernel_selection_requests
    );
    assert_eq!(
        admission.runtime_launch_kernel_candidate_selection_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_selection_requests
    );
    assert_eq!(
        readiness.runtime_launch_kernel_candidate_selection_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        kernel_selection_requests
    );
    let selection_request = kernel_selection_requests
        .dispatch_for(&recommended_dispatch.op_name)
        .unwrap();
    assert!(selection_request.selection_request_ready);
    assert_eq!(
        selection_request.requested_kernel_symbol,
        recommended_dispatch.recommended_kernel_symbol
    );
    assert!(!selection_request.selection_applied);
    kernel_selection_requests.assert_consistent()?;
    let aql_packet_relocations =
        launch_preflight.aql_packet_relocation_plan(&device_pointer_validation)?;
    assert_eq!(
        aql_packet_relocations.dispatch_count,
        aql_packet_templates.dispatch_count
    );
    assert_eq!(
        aql_packet_relocations.queue_packet_count,
        aql_packet_templates.queue_packet_count
    );
    assert_eq!(
        aql_packet_relocations.packet_bytes,
        aql_packet_templates.packet_bytes
    );
    assert_eq!(aql_packet_relocations.field_ranges_per_packet, 15);
    assert_eq!(aql_packet_relocations.relocation_sites_per_packet, 3);
    assert_eq!(
        aql_packet_relocations.total_relocation_sites,
        aql_packet_relocations.dispatch_count * aql_packet_relocations.relocation_sites_per_packet
    );
    assert!(aql_packet_relocations
        .unresolved_runtime_requirements
        .contains(&"loaded_code_object_base"));
    assert!(aql_packet_relocations
        .unresolved_runtime_requirements
        .contains(&"kernarg_allocation"));
    assert!(aql_packet_relocations
        .unresolved_runtime_requirements
        .contains(&"completion_signal_binding"));
    assert!(!aql_packet_relocations
        .unresolved_runtime_requirements
        .contains(&"dispatch_geometry"));
    assert_eq!(
        admission.runtime_launch_aql_packet_relocation_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_relocations
    );
    assert_eq!(
        readiness.runtime_launch_aql_packet_relocation_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_relocations
    );
    let lm_head_relocations = aql_packet_relocations.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_relocations.packet_offset,
        lm_head_template.packet_offset
    );
    assert_eq!(
        lm_head_relocations
            .relocation_sites
            .iter()
            .find(|site| site.field == "kernel_object")
            .unwrap()
            .packet_relative_offset,
        32
    );
    assert_eq!(
        lm_head_relocations
            .relocation_sites
            .iter()
            .find(|site| site.field == "kernarg_address")
            .unwrap()
            .packet_relative_offset,
        40
    );
    assert_eq!(
        lm_head_relocations
            .relocation_sites
            .iter()
            .find(|site| site.field == "completion_signal")
            .unwrap()
            .packet_relative_offset,
        56
    );
    assert_eq!(
        lm_head_relocations
            .field_ranges
            .iter()
            .find(|field| field.field == "reserved2")
            .unwrap()
            .resolution,
        RuntimeLaunchAqlPacketFieldResolution::ReservedZero
    );
    aql_packet_relocations.assert_consistent()?;
    let aql_packet_byte_templates =
        launch_preflight.aql_packet_byte_template_plan(&device_pointer_validation)?;
    assert_eq!(
        aql_packet_byte_templates.dispatch_count,
        aql_packet_templates.dispatch_count
    );
    assert_eq!(
        aql_packet_byte_templates.queue_packet_count,
        aql_packet_templates.queue_packet_count
    );
    assert_eq!(
        aql_packet_byte_templates.packet_bytes,
        aql_packet_templates.packet_bytes
    );
    assert_eq!(
        aql_packet_byte_templates.candidate_byte_template_count,
        aql_packet_templates.kernel_candidate_count
    );
    assert_eq!(
        aql_packet_byte_templates.candidate_byte_template_bytes,
        aql_packet_templates.kernel_candidate_count * AQL_PACKET_BYTES as usize
    );
    assert_eq!(
        aql_packet_byte_templates.relocation_sites_per_candidate,
        aql_packet_relocations.relocation_sites_per_packet
    );
    assert!(aql_packet_byte_templates
        .unresolved_runtime_requirements
        .contains(&"aql_packet_materialization"));
    assert_eq!(
        admission.runtime_launch_aql_packet_byte_template_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_byte_templates
    );
    assert_eq!(
        readiness.runtime_launch_aql_packet_byte_template_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_byte_templates
    );
    let lm_head_byte_templates = aql_packet_byte_templates.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_byte_templates.candidate_byte_template_count,
        lm_head_template.kernel_candidate_count
    );
    let lm_head_first_byte_template = lm_head_byte_templates
        .candidate_byte_templates
        .first()
        .unwrap();
    assert_eq!(
        lm_head_first_byte_template.byte_image.len(),
        AQL_PACKET_BYTES as usize
    );
    assert_ne!(
        u16::from_le_bytes([
            lm_head_first_byte_template.byte_image[0],
            lm_head_first_byte_template.byte_image[1],
        ]),
        0
    );
    assert_eq!(
        u32::from_le_bytes([
            lm_head_first_byte_template.byte_image[24],
            lm_head_first_byte_template.byte_image[25],
            lm_head_first_byte_template.byte_image[26],
            lm_head_first_byte_template.byte_image[27],
        ]),
        lm_head_first_byte_template.private_segment_size
    );
    for site in &lm_head_first_byte_template.zeroed_relocation_sites {
        let end = site.packet_relative_offset + site.size_bytes;
        assert!(
            lm_head_first_byte_template.byte_image[site.packet_relative_offset..end]
                .iter()
                .all(|byte| *byte == 0)
        );
    }
    aql_packet_byte_templates.assert_consistent()?;
    let aql_packet_materialization =
        launch_preflight.aql_packet_materialization_plan(&device_pointer_validation)?;
    assert_eq!(
        aql_packet_materialization.dispatch_count,
        aql_packet_byte_templates.dispatch_count
    );
    assert_eq!(
        aql_packet_materialization.queue_packet_count,
        aql_packet_byte_templates.queue_packet_count
    );
    assert_eq!(
        aql_packet_materialization.packet_bytes,
        aql_packet_byte_templates.packet_bytes
    );
    assert_eq!(
        aql_packet_materialization.candidate_byte_template_count,
        aql_packet_byte_templates.candidate_byte_template_count
    );
    assert_eq!(
        aql_packet_materialization.selected_dispatch_count,
        kernel_selection.selected_dispatch_count
    );
    assert_eq!(
        aql_packet_materialization.ambiguous_dispatch_count,
        kernel_selection.ambiguous_dispatch_count
    );
    assert_eq!(
        aql_packet_materialization.live_relocation_patch_site_count,
        aql_packet_relocations.total_relocation_sites
    );
    assert_eq!(aql_packet_materialization.dispatchable_packet_count, 0);
    assert!(!aql_packet_materialization.packet_materialization_ready);
    assert!(aql_packet_materialization
        .unresolved_runtime_requirements
        .contains(&"aql_packet_materialization"));
    assert_eq!(
        admission.runtime_launch_aql_packet_materialization_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_materialization
    );
    assert_eq!(
        readiness.runtime_launch_aql_packet_materialization_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_packet_materialization
    );
    let embed_materialization = aql_packet_materialization
        .dispatch_for("embed_tokens")
        .unwrap();
    assert_eq!(
        embed_materialization.state,
        RuntimeLaunchAqlPacketMaterializationState::SelectedCandidateByteTemplate
    );
    assert_eq!(
        embed_materialization
            .selected_candidate_kernel_symbol
            .as_deref(),
        Some("decode_step_embed_rmsnorm_token_f16")
    );
    assert_eq!(
        embed_materialization
            .selected_candidate_byte_image
            .as_ref()
            .unwrap()
            .len(),
        AQL_PACKET_BYTES as usize
    );
    assert_eq!(embed_materialization.live_relocation_patch_site_count, 3);
    let lm_head_materialization = aql_packet_materialization.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_materialization.state,
        RuntimeLaunchAqlPacketMaterializationState::AmbiguousCandidateByteTemplates
    );
    assert!(lm_head_materialization
        .selected_candidate_byte_image
        .is_none());
    assert_eq!(lm_head_materialization.live_relocation_patch_site_count, 3);
    aql_packet_materialization.assert_consistent()?;
    let aql_live_relocation_bindings =
        aql_packet_materialization.aql_live_relocation_binding_plan()?;
    assert_eq!(
        aql_live_relocation_bindings.dispatch_count,
        aql_packet_materialization.dispatch_count
    );
    assert_eq!(
        aql_live_relocation_bindings.binding_request_count,
        aql_packet_materialization.live_relocation_patch_site_count
    );
    assert_eq!(
        aql_live_relocation_bindings.binding_request_bytes,
        aql_packet_materialization.live_relocation_patch_bytes
    );
    assert_eq!(
        aql_live_relocation_bindings.code_object_base_request_count,
        launch_preflight.dispatch_count
    );
    assert_eq!(
        aql_live_relocation_bindings.kernarg_allocation_request_count,
        launch_preflight.dispatch_count
    );
    assert_eq!(
        aql_live_relocation_bindings.completion_signal_request_count,
        launch_preflight.dispatch_count
    );
    assert_eq!(aql_live_relocation_bindings.bound_relocation_count, 0);
    assert_eq!(
        aql_live_relocation_bindings.unbound_relocation_count,
        aql_live_relocation_bindings.binding_request_count
    );
    assert_eq!(aql_live_relocation_bindings.dispatches_fully_bound_count, 0);
    assert!(!aql_live_relocation_bindings.all_relocations_bound);
    assert!(aql_live_relocation_bindings.binding_request_plan_ready);
    assert!(aql_live_relocation_bindings
        .unresolved_runtime_requirements
        .contains(&"loaded_code_object_base"));
    assert!(aql_live_relocation_bindings
        .unresolved_runtime_requirements
        .contains(&"kernarg_allocation"));
    assert!(aql_live_relocation_bindings
        .unresolved_runtime_requirements
        .contains(&"completion_signal_binding"));
    assert_eq!(
        admission.runtime_launch_aql_live_relocation_binding_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_live_relocation_bindings
    );
    assert_eq!(
        readiness.runtime_launch_aql_live_relocation_binding_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        aql_live_relocation_bindings
    );
    let lm_head_live_bindings = aql_live_relocation_bindings
        .dispatch_for("lm_head")
        .unwrap();
    assert_eq!(
        lm_head_live_bindings.materialization_state,
        RuntimeLaunchAqlPacketMaterializationState::AmbiguousCandidateByteTemplates
    );
    assert_eq!(lm_head_live_bindings.binding_request_count, 3);
    assert_eq!(lm_head_live_bindings.bound_relocation_count, 0);
    assert!(lm_head_live_bindings
        .binding_requests
        .iter()
        .all(|request| !request.runtime_value_bound));
    let kernel_object_binding = lm_head_live_bindings
        .binding_requests
        .iter()
        .find(|request| {
            request.binding_kind == RuntimeLaunchAqlLiveRelocationBindingKind::LoadedCodeObjectBase
        })
        .unwrap();
    assert_eq!(kernel_object_binding.field, "kernel_object");
    assert_eq!(kernel_object_binding.packet_relative_offset, 32);
    assert_eq!(
        kernel_object_binding.unresolved_runtime_requirement,
        "loaded_code_object_base"
    );
    aql_live_relocation_bindings.assert_consistent()?;
    let code_object_base_bindings =
        code_object_loads.code_object_base_binding_request_plan(&aql_live_relocation_bindings)?;
    assert_eq!(
        code_object_base_bindings.dispatch_count,
        launch_preflight.dispatch_count
    );
    assert_eq!(
        code_object_base_bindings.required_kernel_count,
        code_object_loads.required_kernel_count
    );
    assert_eq!(
        code_object_base_bindings.code_object_load_request_count,
        code_object_loads.code_object_load_request_count
    );
    assert_eq!(code_object_base_bindings.loaded_code_object_count, 0);
    assert_eq!(
        code_object_base_bindings.loaded_code_object_base_request_count,
        1
    );
    assert_eq!(
        code_object_base_bindings.loaded_code_object_base_bound_count,
        0
    );
    assert_eq!(
        code_object_base_bindings.kernel_descriptor_binding_request_count,
        code_object_loads.kernel_descriptor_binding_request_count
    );
    assert_eq!(code_object_base_bindings.kernel_descriptor_bound_count, 0);
    assert_eq!(
        code_object_base_bindings.aql_kernel_object_relocation_request_count,
        aql_live_relocation_bindings.code_object_base_request_count
    );
    assert_eq!(
        code_object_base_bindings.aql_kernel_object_relocation_bound_count,
        0
    );
    assert!(!code_object_base_bindings.all_code_object_base_bindings_bound);
    assert!(code_object_base_bindings.request_plan_ready);
    assert_eq!(
        launch_preflight.code_object_base_binding_request_plan(&device_pointer_validation)?,
        code_object_base_bindings
    );
    assert_eq!(
        admission.runtime_launch_code_object_base_binding_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        code_object_base_bindings
    );
    assert_eq!(
        readiness.runtime_launch_code_object_base_binding_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        code_object_base_bindings
    );
    let lm_head_base_relocation = code_object_base_bindings.dispatch_for("lm_head").unwrap();
    assert_eq!(lm_head_base_relocation.packet_relative_offset, 32);
    assert!(!lm_head_base_relocation.runtime_value_bound);
    let gemv_descriptor = code_object_base_bindings
        .kernel_descriptor_for("gemv_f16")
        .unwrap();
    assert_eq!(
        gemv_descriptor.kernel_descriptor_vaddr,
        gemv_load_request.kernel_descriptor_vaddr
    );
    assert!(!gemv_descriptor.descriptor_binding_bound);
    code_object_base_bindings.assert_consistent()?;
    let mut inconsistent_code_object_base_bindings = code_object_base_bindings.clone();
    inconsistent_code_object_base_bindings.loaded_code_object_base_bound_count = 1;
    let err = inconsistent_code_object_base_bindings
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("loaded code object base bound count"));
    let execution_readiness =
        launch_preflight.execution_readiness_report(&device_pointer_validation)?;
    assert_eq!(
        execution_readiness.dispatch_count,
        launch_candidates.dispatch_count
    );
    assert_eq!(
        execution_readiness.argument_count,
        launch_device_arguments.argument_count
    );
    assert_eq!(
        execution_readiness.pointer_argument_count,
        launch_device_arguments.pointer_argument_count
    );
    assert_eq!(
        execution_readiness.scalar_argument_count,
        launch_device_arguments.scalar_argument_count
    );
    assert_eq!(
        execution_readiness.kernel_candidate_count,
        aql_packet_fields.kernel_candidate_count
    );
    assert_eq!(
        execution_readiness.code_object_load_request_count,
        code_object_loads.code_object_load_request_count
    );
    assert_eq!(execution_readiness.code_object_loaded_count, 0);
    assert_eq!(
        execution_readiness.loaded_code_object_base_request_count,
        code_object_base_bindings.loaded_code_object_base_request_count
    );
    assert_eq!(execution_readiness.loaded_code_object_base_bound_count, 0);
    assert_eq!(
        execution_readiness.kernel_descriptor_binding_request_count,
        code_object_loads.kernel_descriptor_binding_request_count
    );
    assert_eq!(execution_readiness.kernel_descriptor_bound_count, 0);
    assert_eq!(
        execution_readiness.aql_kernel_object_relocation_request_count,
        code_object_base_bindings.aql_kernel_object_relocation_request_count
    );
    assert_eq!(
        execution_readiness.aql_kernel_object_relocation_bound_count,
        0
    );
    assert_eq!(
        execution_readiness.selected_dispatch_count,
        kernel_selection.selected_dispatch_count
    );
    assert_eq!(
        execution_readiness.ambiguous_dispatch_count,
        kernel_selection.ambiguous_dispatch_count
    );
    assert_eq!(
        execution_readiness.kernel_candidate_recommended_dispatch_count,
        kernel_recommendations.recommended_dispatch_count
    );
    assert_eq!(
        execution_readiness.kernel_candidate_missing_recommendation_dispatch_count,
        kernel_recommendations.missing_recommendation_dispatch_count
    );
    assert_eq!(
        execution_readiness.kernel_candidate_recommendation_selection_applied_count,
        0
    );
    assert_eq!(
        execution_readiness.kernel_candidate_selection_request_count,
        kernel_selection_requests.selection_request_count
    );
    assert_eq!(
        execution_readiness.kernel_candidate_selection_missing_request_count,
        kernel_selection_requests.missing_selection_request_count
    );
    assert_eq!(
        execution_readiness.kernel_candidate_selection_applied_count,
        0
    );
    assert_eq!(
        execution_readiness.host_launcher_branch_resolution_request_count,
        host_launcher_branch_requests.branch_resolution_request_count
    );
    assert_eq!(
        execution_readiness.host_launcher_branch_resolution_applied_count,
        0
    );
    assert_eq!(
        execution_readiness.host_launcher_branch_resolution_unresolved_candidate_count,
        host_launcher_branch_requests.unresolved_candidate_symbol_count
    );
    assert_eq!(
        execution_readiness.packet_bytes,
        staging_footprint.packet_bytes
    );
    assert_eq!(
        execution_readiness.kernarg_bytes_upper_bound,
        staging_footprint.kernarg_bytes_upper_bound
    );
    assert_eq!(
        execution_readiness.packet_region_bytes,
        staging_layout.packet_region_bytes
    );
    assert_eq!(
        execution_readiness.kernarg_region_bytes,
        staging_layout.kernarg_region_bytes
    );
    assert_eq!(
        execution_readiness.total_staging_bytes,
        staging_layout.total_staging_bytes
    );
    assert_eq!(
        execution_readiness.completion_signal_count,
        completion_signals.terminal_signal_count
    );
    assert_eq!(
        execution_readiness.completion_signal_handle_request_count,
        completion_signal_bindings.signal_handle_request_count
    );
    assert_eq!(execution_readiness.completion_signal_handle_bound_count, 0);
    assert_eq!(
        execution_readiness.queue_packet_count,
        queue_slots.queue_packet_count
    );
    assert_eq!(
        execution_readiness.doorbell_batch_count,
        queue_slots.doorbell_batch_count
    );
    assert_eq!(
        execution_readiness.queue_reservation_packet_request_count,
        queue_reservations.queue_packet_request_count
    );
    assert_eq!(
        execution_readiness.queue_reservation_packet_reserved_count,
        0
    );
    assert_eq!(
        execution_readiness.queue_reservation_doorbell_request_count,
        queue_reservations.doorbell_batch_request_count
    );
    assert_eq!(
        execution_readiness.queue_reservation_doorbell_bound_count,
        0
    );
    assert_eq!(execution_readiness.queue_reservation_applied_count, 0);
    assert_eq!(
        execution_readiness.dispatch_workgroup_count,
        dispatch_geometry.total_workgroups
    );
    assert_eq!(
        execution_readiness.kernarg_argument_bytes,
        kernarg_layout.argument_payload_bytes
    );
    assert_eq!(
        execution_readiness.kernarg_argument_span_bytes,
        kernarg_layout.argument_span_bytes
    );
    assert_eq!(
        execution_readiness.kernarg_serialized_bytes,
        kernarg_serialization.serialized_kernarg_bytes
    );
    assert_eq!(
        execution_readiness.kernarg_layout_capacity_shortfall_bytes,
        kernarg_layout.candidate_capacity_shortfall_bytes
    );
    assert_eq!(
        execution_readiness.kernarg_allocation_request_count,
        kernarg_allocations.backing_allocation_request_count
    );
    assert_eq!(execution_readiness.kernarg_allocation_bound_count, 0);
    assert_eq!(
        execution_readiness.kernarg_allocation_request_bytes,
        kernarg_allocations.backing_allocation_request_bytes
    );
    assert_eq!(execution_readiness.kernarg_allocation_bound_bytes, 0);
    assert_eq!(
        execution_readiness.kernarg_copy_request_count,
        kernarg_allocations.dispatch_copy_request_count
    );
    assert_eq!(execution_readiness.kernarg_copy_applied_count, 0);
    assert_eq!(
        execution_readiness.kernarg_copy_request_bytes,
        kernarg_allocations.dispatch_copy_request_bytes
    );
    assert_eq!(execution_readiness.kernarg_copy_applied_bytes, 0);
    assert_eq!(
        execution_readiness.kernel_argument_abi_candidate_count,
        kernel_argument_abi.kernel_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_size_compatible_candidate_count,
        kernel_argument_abi.size_compatible_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_verified_candidate_count,
        kernel_argument_abi.verified_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_dispatches_with_verified_candidate_count,
        kernel_argument_abi.dispatches_with_verified_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_dispatches_without_verified_candidate_count,
        kernel_argument_abi.dispatches_without_verified_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_schema_request_count,
        kernel_argument_abi_schema_requests.schema_request_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_schema_bound_count,
        kernel_argument_abi_schema_requests.schema_bound_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_verification_request_count,
        kernel_argument_abi_schema_requests.candidate_verification_request_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_verification_applied_count,
        kernel_argument_abi_schema_requests.candidate_verified_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_capacity_request_count,
        kernel_argument_abi_capacity_requests.capacity_request_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_capacity_candidate_request_count,
        kernel_argument_abi_capacity_requests.candidate_capacity_request_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_capacity_max_shortfall_bytes,
        kernel_argument_abi_capacity_requests.max_capacity_shortfall_bytes
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_capacity_total_shortfall_bytes,
        kernel_argument_abi_capacity_requests.total_capacity_shortfall_bytes
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_candidate_count,
        kernel_argument_abi_semantic_projection.kernel_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_schema_candidate_count,
        kernel_argument_abi_semantic_projection.semantic_schema_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_missing_schema_candidate_count,
        kernel_argument_abi_semantic_projection.missing_semantic_schema_candidate_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_descriptor_match_candidate_count,
        kernel_argument_abi_semantic_projection.semantic_descriptor_match_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_ready_candidate_count,
        kernel_argument_abi_semantic_projection.projection_ready_candidate_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_dispatches_with_ready_candidate_count,
        kernel_argument_abi_semantic_projection.dispatches_with_projection_ready_candidate_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_dispatches_without_ready_candidate_count,
        kernel_argument_abi_semantic_projection.dispatches_without_projection_ready_candidate_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_field_schema_count,
        kernel_argument_abi_semantic_projection.field_schema_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_projected_field_count,
        kernel_argument_abi_semantic_projection.projected_field_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_missing_field_count,
        kernel_argument_abi_semantic_projection.missing_field_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_kind_mismatch_field_count,
        kernel_argument_abi_semantic_projection.kind_mismatch_field_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_unsupported_encoding_field_count,
        kernel_argument_abi_semantic_projection.unsupported_encoding_field_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_scalar_narrowing_overflow_field_count,
        kernel_argument_abi_semantic_projection.scalar_narrowing_overflow_field_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_field_range_overflow_count,
        kernel_argument_abi_semantic_projection.field_range_overflow_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_projected_kernarg_bytes,
        kernel_argument_abi_semantic_projection.projected_kernarg_bytes
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_request_count,
        semantic_projection_candidate_selection_requests.selection_request_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count,
        semantic_projection_candidate_selection_requests.missing_selection_request_count
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_requested_kernarg_bytes,
        semantic_projection_candidate_selection_requests.requested_projected_kernarg_bytes
    );
    assert_eq!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_applied_count,
        semantic_projection_candidate_selection_requests.selection_applied_count
    );
    assert_eq!(
        execution_readiness.kernel_argument_abi_semantic_projection_ready,
        kernel_argument_abi_semantic_projection.semantic_projection_ready
    );
    assert_eq!(
        execution_readiness.aql_packet_template_count,
        aql_packet_templates.dispatch_count
    );
    assert_eq!(
        execution_readiness.aql_packet_template_candidate_count,
        aql_packet_templates.kernel_candidate_count
    );
    assert_eq!(
        execution_readiness.aql_packet_relocation_site_count,
        aql_packet_relocations.total_relocation_sites
    );
    assert_eq!(
        execution_readiness.aql_packet_byte_template_count,
        aql_packet_byte_templates.candidate_byte_template_count
    );
    assert_eq!(
        execution_readiness.aql_packet_byte_template_bytes,
        aql_packet_byte_templates.candidate_byte_template_bytes
    );
    assert_eq!(
        execution_readiness.aql_packet_materialization_selected_dispatch_count,
        aql_packet_materialization.selected_dispatch_count
    );
    assert_eq!(
        execution_readiness.aql_packet_materialization_ambiguous_dispatch_count,
        aql_packet_materialization.ambiguous_dispatch_count
    );
    assert_eq!(
        execution_readiness.aql_packet_materialization_relocation_site_count,
        aql_packet_materialization.live_relocation_patch_site_count
    );
    assert_eq!(
        execution_readiness.aql_packet_materialization_dispatchable_packet_count,
        0
    );
    assert_eq!(
        execution_readiness.aql_live_relocation_binding_request_count,
        aql_live_relocation_bindings.binding_request_count
    );
    assert_eq!(
        execution_readiness.aql_live_relocation_binding_bound_count,
        0
    );
    assert_eq!(
        execution_readiness.aql_live_relocation_binding_unbound_count,
        aql_live_relocation_bindings.unbound_relocation_count
    );
    assert_eq!(
        execution_readiness.aql_live_relocation_code_object_request_count,
        aql_live_relocation_bindings.code_object_base_request_count
    );
    assert_eq!(
        execution_readiness.aql_live_relocation_kernarg_request_count,
        aql_live_relocation_bindings.kernarg_allocation_request_count
    );
    assert_eq!(
        execution_readiness.aql_live_relocation_completion_signal_request_count,
        aql_live_relocation_bindings.completion_signal_request_count
    );
    assert!(!execution_readiness.executable);
    assert!(execution_readiness.is_non_executable_boundary());
    execution_readiness.assert_non_executable_boundary()?;
    let mut stale_count_execution_readiness = execution_readiness.clone();
    stale_count_execution_readiness.selected_dispatch_count += 1;
    assert!(!stale_count_execution_readiness.is_non_executable_boundary());
    let stale_count_execution_readiness_err = stale_count_execution_readiness
        .assert_non_executable_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_execution_readiness_err.contains("consistency"));
    assert!(
        stale_count_execution_readiness_err.contains("selected+ambiguous+missing dispatch count")
    );
    macro_rules! assert_execution_readiness_non_executable_rejected {
        ($field:ident = $value:expr, $needle:literal $(,)?) => {{
            let mut report = execution_readiness.clone();
            report.$field = $value;
            assert!(!report.is_non_executable_boundary());
            assert!(report
                .assert_non_executable_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
        (clear_blockers, $needle:literal $(,)?) => {{
            let mut report = execution_readiness.clone();
            report.blockers.clear();
            assert!(!report.is_non_executable_boundary());
            assert!(report
                .assert_non_executable_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
        (clear_unresolved_runtime_requirements, $needle:literal $(,)?) => {{
            let mut report = execution_readiness.clone();
            report.unresolved_runtime_requirements.clear();
            assert!(!report.is_non_executable_boundary());
            assert!(report
                .assert_non_executable_boundary()
                .unwrap_err()
                .to_string()
                .contains($needle));
        }};
    }
    assert_execution_readiness_non_executable_rejected!(
        executable = true,
        "launch execution executable is true",
    );
    assert_execution_readiness_non_executable_rejected!(
        clear_blockers,
        "launch execution blockers are empty",
    );
    assert_execution_readiness_non_executable_rejected!(
        clear_unresolved_runtime_requirements,
        "unresolved runtime requirements are empty",
    );
    assert_execution_readiness_non_executable_rejected!(
        aql_packet_materialization_dispatchable_packet_count = 1,
        "dispatchable AQL packet count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        code_object_loaded_count = 1,
        "loaded code objects 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        loaded_code_object_base_bound_count = 1,
        "loaded code object base bound count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        kernel_descriptor_bound_count = 1,
        "kernel descriptor bound count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        aql_kernel_object_relocation_bound_count = 1,
        "AQL kernel_object relocation bound count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        completion_signal_handle_bound_count = 1,
        "completion signal handle bound count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        queue_reservation_packet_reserved_count = 1,
        "queue reservation packet reserved count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        queue_reservation_doorbell_bound_count = 1,
        "queue reservation doorbell bound count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        queue_reservation_applied_count = 1,
        "queue reservation applied count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        kernarg_allocation_bound_count = 1,
        "kernarg allocation bound count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        kernarg_allocation_bound_bytes = 1,
        "kernarg allocation bound bytes 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        kernarg_copy_applied_count = 1,
        "kernarg copy applied count 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        kernarg_copy_applied_bytes = 1,
        "kernarg copy applied bytes 1 != 0",
    );
    assert_execution_readiness_non_executable_rejected!(
        aql_live_relocation_binding_bound_count = 1,
        "AQL live relocation bound count 1 != 0",
    );
    let expected_launch_execution_requirements = [
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
        execution_readiness
            .unresolved_runtime_requirements
            .as_slice(),
        &expected_launch_execution_requirements
    );
    assert_eq!(
        execution_readiness
            .unresolved_runtime_requirement_names()
            .as_slice(),
        &expected_launch_execution_requirements
    );
    assert_eq!(
        execution_readiness.blocker_requirement_names().as_slice(),
        &expected_launch_execution_requirements
    );
    assert!(execution_readiness.staging_layout_ready);
    assert!(execution_readiness.completion_signal_policy_ready);
    assert!(execution_readiness.completion_signal_binding_request_plan_ready);
    assert!(execution_readiness.queue_slot_plan_ready);
    assert!(execution_readiness.queue_reservation_request_plan_ready);
    assert!(execution_readiness.dispatch_geometry_ready);
    assert!(execution_readiness.kernarg_layout_ready);
    assert!(execution_readiness.kernarg_serialization_ready);
    assert!(execution_readiness.kernarg_allocation_request_plan_ready);
    assert!(execution_readiness.kernel_argument_abi_preflight_ready);
    assert!(execution_readiness.kernel_argument_abi_schema_request_plan_ready);
    assert!(execution_readiness.kernel_argument_abi_capacity_request_plan_ready);
    assert!(!execution_readiness.kernel_argument_abi_semantic_projection_ready);
    assert!(
        execution_readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_request_plan_ready
    );
    assert!(execution_readiness.aql_packet_template_ready);
    assert!(execution_readiness.aql_packet_relocation_plan_ready);
    assert!(execution_readiness.aql_packet_byte_template_ready);
    assert!(execution_readiness.aql_packet_materialization_plan_ready);
    assert!(execution_readiness.code_object_load_request_plan_ready);
    assert!(execution_readiness.code_object_base_binding_request_plan_ready);
    assert!(execution_readiness.aql_live_relocation_binding_plan_ready);
    assert!(execution_readiness.kernel_candidate_recommendation_plan_ready);
    assert!(execution_readiness.kernel_candidate_selection_request_plan_ready);
    assert!(execution_readiness.host_launcher_branch_resolution_request_plan_ready);
    assert!(execution_readiness.has_blocker("kernel_candidate_selection_policy"));
    assert!(!execution_readiness.has_blocker("allocator_alignment_policy"));
    assert!(!execution_readiness.has_blocker("dispatch_geometry"));
    assert!(!execution_readiness.has_blocker("kernel_argument_abi_layout"));
    assert!(execution_readiness.has_blocker("kernel_argument_abi_verification"));
    assert!(execution_readiness.has_blocker("kernel_argument_abi_semantic_projection"));
    assert!(!execution_readiness.has_blocker("kernarg_layout_serialization"));
    assert!(!execution_readiness.has_blocker("completion_signal_policy"));
    assert!(execution_readiness.has_blocker("completion_signal_binding"));
    assert!(execution_readiness.has_blocker("queue_reservation"));
    assert!(execution_readiness.has_blocker("aql_packet_materialization"));
    let queue_readiness_blocker = execution_readiness
        .blocker_for("queue_reservation")
        .unwrap();
    assert_eq!(queue_readiness_blocker.requirement, "queue_reservation");
    assert_eq!(
        execution_readiness
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .requirement,
        queue_readiness_blocker.requirement
    );
    assert!(queue_readiness_blocker
        .detail
        .contains("queue packet reservation"));
    assert!(execution_readiness
        .blocker_for("dispatch_geometry")
        .is_none());
    assert_eq!(
        admission.runtime_launch_execution_readiness_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        execution_readiness
    );
    assert_eq!(
        readiness.runtime_launch_execution_readiness_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        execution_readiness
    );
    let execution_requests = launch_preflight.execution_request_plan(&device_pointer_validation)?;
    assert_eq!(
        execution_requests.dispatch_count,
        launch_preflight.dispatch_count
    );
    assert_eq!(
        execution_requests.runtime_request_plan_count,
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len()
    );
    assert_eq!(
        execution_requests.runtime_request_plan_count,
        EXPECTED_RUNTIME_LAUNCH_REQUEST_STEP_COUNT
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::DESCRIPTORS.len(),
        execution_requests.runtime_request_plan_count
    );
    let semantic_projection_selection_descriptor =
        RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSemanticProjectionCandidateSelection
            .descriptor();
    assert_eq!(semantic_projection_selection_descriptor.step_index, 7);
    assert_eq!(
        semantic_projection_selection_descriptor.request_plan,
        "kernel_argument_abi_semantic_projection_candidate_selection_request_plan"
    );
    assert_eq!(
        semantic_projection_selection_descriptor.requirement,
        "kernel_argument_abi_semantic_projection"
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::from_request_plan(
            "kernel_argument_abi_semantic_projection_candidate_selection_request_plan"
        ),
        Some(RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSemanticProjectionCandidateSelection)
    );
    let semantic_projection_selection_descriptor_from_label:
        RuntimeLaunchExecutionRequestStepDescriptor =
        RuntimeLaunchExecutionRequestStep::descriptor_for_request_plan(
            "kernel_argument_abi_semantic_projection_candidate_selection_request_plan",
        )
        .unwrap();
    assert_eq!(
        semantic_projection_selection_descriptor_from_label,
        semantic_projection_selection_descriptor
    );
    assert!(
        RuntimeLaunchExecutionRequestStep::from_request_plan("dispatch_geometry_request_plan")
            .is_none()
    );
    assert!(
        RuntimeLaunchExecutionRequestStep::descriptor_for_request_plan(
            "dispatch_geometry_request_plan"
        )
        .is_none()
    );
    for (expected_index, descriptor) in RuntimeLaunchExecutionRequestStep::DESCRIPTORS
        .iter()
        .enumerate()
    {
        assert_eq!(
            descriptor.step,
            RuntimeLaunchExecutionRequestStep::ALL[expected_index]
        );
        assert_eq!(descriptor.step_index, expected_index);
        assert_eq!(descriptor.request_plan, descriptor.step.request_plan());
        assert_eq!(descriptor.requirement, descriptor.step.requirement());
        assert_eq!(
            descriptor.live_aql_proof_kind,
            descriptor.step.live_aql_proof_kind()
        );
        assert_eq!(
            descriptor.live_aql_proof_input,
            descriptor.step.live_aql_proof_input()
        );
        assert_eq!(
            descriptor.live_aql_validation_method,
            descriptor.step.live_aql_validation_method()
        );
        assert_eq!(
            descriptor.mutates_live_queue,
            descriptor.step.mutates_live_queue()
        );
        assert_eq!(
            descriptor.live_aql_proof_input.is_some(),
            descriptor.step.requires_live_aql_proof()
        );
    }
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_STEPS,
        [
            RuntimeLaunchExecutionRequestStep::QueueReservation,
            RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding,
        ]
    );
    let live_aql_proof_descriptors_from_all = RuntimeLaunchExecutionRequestStep::DESCRIPTORS
        .iter()
        .copied()
        .filter(|descriptor| descriptor.step.requires_live_aql_proof())
        .collect::<Vec<_>>();
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS.to_vec(),
        live_aql_proof_descriptors_from_all
    );
    assert_eq!(
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS.len(),
        2
    );
    for descriptor in RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS {
        assert!(descriptor.step.requires_live_aql_proof());
        assert!(descriptor.live_aql_proof_kind.is_some());
        assert!(descriptor.live_aql_proof_input.is_some());
        assert!(descriptor.live_aql_validation_method.is_some());
        assert!(!descriptor.mutates_live_queue);
    }
    assert_eq!(
        RuntimeLaunchLiveAqlProofKind::DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.kind.as_str())
            .collect::<Vec<_>>()
            .as_slice(),
        &["batch_reservation_plan", "materialized_packet_plan"]
    );
    assert_eq!(
        RuntimeLaunchLiveAqlProofKind::from_proof_input("KfdQueueLiveAqlBatchReservationPlanInput"),
        Some(RuntimeLaunchLiveAqlProofKind::BatchReservationPlan)
    );
    assert_eq!(
        RuntimeLaunchLiveAqlProofKind::from_validation_method(
            "KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready"
        ),
        Some(RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan)
    );
    assert!(
        RuntimeLaunchLiveAqlProofKind::from_proof_input("KfdQueueLiveAqlUnknownInput").is_none()
    );
    assert!(!RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSchema.requires_live_aql_proof());
    assert_eq!(
        execution_requests.components.len(),
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        execution_requests.components.first().unwrap().request_plan,
        "code_object_load_request_plan"
    );
    assert_eq!(
        execution_requests.components.first().unwrap().step,
        RuntimeLaunchExecutionRequestStep::CodeObjectLoad
    );
    assert_eq!(
        execution_requests.components.last().unwrap().request_plan,
        "aql_live_relocation_binding_request_plan"
    );
    assert_eq!(
        execution_requests.components.last().unwrap().step,
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding
    );
    let expected_execution_request_plans = [
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
    ];
    let expected_live_aql_proof_surface_plans = [
        "queue_reservation_request_plan",
        "aql_live_relocation_binding_request_plan",
    ];
    let static_live_aql_proof_surface_plans =
        RuntimeLaunchExecutionRequestStep::LIVE_AQL_PROOF_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.request_plan)
            .collect::<Vec<_>>();
    assert_eq!(
        static_live_aql_proof_surface_plans.as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert_eq!(
        execution_requests.component_request_plan_names().as_slice(),
        &expected_execution_request_plans
    );
    assert_eq!(
        execution_requests
            .pending_component_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    let execution_request_receipt_lines = execution_requests.receipt_lines();
    assert_eq!(
        execution_request_receipt_lines[0],
        "receipt.kind=model_runtime_launch_execution_request_plan"
    );
    assert_eq!(execution_request_receipt_lines[1], "receipt.version=1");
    assert!(execution_request_receipt_lines
        .iter()
        .any(|line| line == "runtime_request_plan_count=10"));
    assert!(execution_request_receipt_lines.iter().any(|line| line
        == "components.7.request_plan=kernel_argument_abi_semantic_projection_candidate_selection_request_plan"));
    assert!(execution_request_receipt_lines.iter().any(|line| line
        == &format!(
            "kernel_argument_abi_semantic_projection_candidate_selection.request_count={}",
            semantic_projection_candidate_selection_requests.selection_request_count
        )));
    assert!(execution_requests.receipt_text().ends_with('\n'));
    let execution_request_receipt_fingerprint = execution_requests.receipt_fingerprint();
    assert_eq!(execution_request_receipt_fingerprint.len(), 64);
    assert!(execution_request_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_execution_request_receipt = execution_requests.clone();
    stale_execution_request_receipt.component_pending_count += 1;
    assert_ne!(
        stale_execution_request_receipt.receipt_fingerprint(),
        execution_request_receipt_fingerprint
    );
    assert!(stale_execution_request_receipt
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("component pending count"));
    assert!(execution_requests.components.iter().all(|component| {
        component.request_plan_ready
            && component.blocker_present
            && (component.applied_count == 0
                || component.request_plan == "kernel_argument_abi_schema_request_plan")
            && component.pending_count + component.applied_count == component.request_count
            && component.blocker_detail.is_some()
            && !component.mutates_live_queue
    }));
    let kernel_argument_abi_component = execution_requests
        .component_for("kernel_argument_abi_schema_request_plan")
        .unwrap();
    assert_eq!(
        kernel_argument_abi_component.applied_count,
        kernel_argument_abi_schema_requests.schema_bound_count
            + kernel_argument_abi_schema_requests.candidate_verified_count
    );
    assert!(kernel_argument_abi_component.applied_count > 0);
    assert!(execution_requests
        .component_for("dispatch_geometry_request_plan")
        .is_none());
    assert_eq!(
        execution_requests.kernel_argument_abi_semantic_projection_candidate_selection_requests,
        semantic_projection_candidate_selection_requests
    );
    let semantic_projection_selection_component = execution_requests
        .component_for("kernel_argument_abi_semantic_projection_candidate_selection_request_plan")
        .unwrap();
    assert_eq!(
        semantic_projection_selection_component.step,
        RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSemanticProjectionCandidateSelection
    );
    assert_eq!(
        semantic_projection_selection_component.request_count,
        semantic_projection_candidate_selection_requests.selection_request_count
    );
    assert_eq!(semantic_projection_selection_component.applied_count, 0);
    assert_eq!(
        semantic_projection_selection_component.pending_count,
        semantic_projection_candidate_selection_requests.selection_request_count
    );
    assert_eq!(
        execution_requests
            .component_for_step(
                RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSemanticProjectionCandidateSelection
            )
            .unwrap()
            .request_plan,
        semantic_projection_selection_component.request_plan
    );
    assert!(semantic_projection_selection_component
        .blocker_detail
        .as_ref()
        .unwrap()
        .contains(&format!(
            "selection_requests={}",
            semantic_projection_candidate_selection_requests.selection_request_count
        )));
    assert!(
        execution_requests
            .execution_readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count
            > 0
    );
    let mut readiness_drift_execution_requests = execution_requests.clone();
    readiness_drift_execution_requests
        .execution_readiness
        .kernel_argument_abi_semantic_projection_candidate_selection_request_count += 1;
    readiness_drift_execution_requests
        .execution_readiness
        .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count -= 1;
    readiness_drift_execution_requests
        .execution_readiness
        .kernel_argument_abi_semantic_projection_dispatches_with_ready_candidate_count += 1;
    readiness_drift_execution_requests
        .execution_readiness
        .kernel_argument_abi_semantic_projection_dispatches_without_ready_candidate_count -= 1;
    let err = readiness_drift_execution_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("semantic projection candidate selection request count"));
    assert!(err.contains("!= readiness"));

    let projection_selection_request_plan =
        "kernel_argument_abi_semantic_projection_candidate_selection_request_plan";
    let mut all_missing_execution_requests = execution_requests.clone();
    let removed_projection_selection_requests = {
        let all_missing_projection_selection = &mut all_missing_execution_requests
            .kernel_argument_abi_semantic_projection_candidate_selection_requests;
        assert!(all_missing_projection_selection.selection_request_count > 0);
        let removed_projection_selection_requests =
            all_missing_projection_selection.selection_request_count;
        all_missing_projection_selection.projection_ready_candidate_count = 0;
        all_missing_projection_selection.selection_request_count = 0;
        all_missing_projection_selection.missing_selection_request_count =
            all_missing_projection_selection.dispatch_count;
        all_missing_projection_selection.requested_projected_kernarg_bytes = 0;
        all_missing_projection_selection.all_selection_requests_ready = false;
        if !all_missing_projection_selection
            .unresolved_runtime_requirements
            .contains(&"kernel_argument_abi_semantic_projection")
        {
            all_missing_projection_selection
                .unresolved_runtime_requirements
                .push("kernel_argument_abi_semantic_projection");
        }
        for dispatch in &mut all_missing_projection_selection.dispatches {
            dispatch.projection_ready_candidate_count = 0;
            dispatch.has_projection_ready_candidate = false;
            dispatch.recommendation_state =
                RuntimeLaunchKernelArgumentAbiSemanticProjectionCandidateRecommendationState::NoProjectionReadyCandidate;
            dispatch.recommended_candidate_index = None;
            dispatch.requested_kernel_symbol = None;
            dispatch.requested_kernarg_size = None;
            dispatch.requested_projected_kernarg_bytes = None;
            dispatch.requested_projected_byte_image.clear();
            dispatch.requested_candidate_projection_ready = false;
            dispatch.selection_request_ready = false;
            dispatch.selection_applied = false;
            dispatch.selection_request_reason = "no_projection_ready_candidate";
        }
        all_missing_projection_selection.assert_consistent()?;
        removed_projection_selection_requests
    };
    {
        let readiness = &mut all_missing_execution_requests.execution_readiness;
        readiness.kernel_argument_abi_semantic_projection_ready_candidate_count = 0;
        readiness.kernel_argument_abi_semantic_projection_dispatches_with_ready_candidate_count = 0;
        readiness
            .kernel_argument_abi_semantic_projection_dispatches_without_ready_candidate_count =
            readiness.dispatch_count;
        readiness.kernel_argument_abi_semantic_projection_candidate_selection_request_count = 0;
        readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count =
            readiness.dispatch_count;
        readiness
            .kernel_argument_abi_semantic_projection_candidate_selection_requested_kernarg_bytes =
            0;
        readiness.kernel_argument_abi_semantic_projection_candidate_selection_applied_count = 0;
        readiness.kernel_argument_abi_semantic_projection_ready = false;
    }
    let all_missing_semantic_projection_blocker_detail = {
        let readiness = &all_missing_execution_requests.execution_readiness;
        format!(
            "{} projection-ready kernel argument ABI candidate(s) cover {}/{} dispatch(es); missing_ready_dispatches={}; selection_requests={} missing_selection_requests={} requested_projected_kernarg_bytes={} selection_applied={}; schema_candidates={} missing_schema_candidates={} descriptor_matches={} field_schemas={} projected_fields={} missing_fields={} kind_mismatches={} unsupported_encodings={} scalar_narrowing_overflows={} field_range_overflows={} projected_kernarg_bytes={}",
            readiness.kernel_argument_abi_semantic_projection_ready_candidate_count,
            readiness
                .kernel_argument_abi_semantic_projection_dispatches_with_ready_candidate_count,
            readiness.dispatch_count,
            readiness
                .kernel_argument_abi_semantic_projection_dispatches_without_ready_candidate_count,
            readiness
                .kernel_argument_abi_semantic_projection_candidate_selection_request_count,
            readiness
                .kernel_argument_abi_semantic_projection_candidate_selection_missing_request_count,
            readiness
                .kernel_argument_abi_semantic_projection_candidate_selection_requested_kernarg_bytes,
            readiness.kernel_argument_abi_semantic_projection_candidate_selection_applied_count,
            readiness.kernel_argument_abi_semantic_projection_schema_candidate_count,
            readiness.kernel_argument_abi_semantic_projection_missing_schema_candidate_count,
            readiness.kernel_argument_abi_semantic_projection_descriptor_match_candidate_count,
            readiness.kernel_argument_abi_semantic_projection_field_schema_count,
            readiness.kernel_argument_abi_semantic_projection_projected_field_count,
            readiness.kernel_argument_abi_semantic_projection_missing_field_count,
            readiness.kernel_argument_abi_semantic_projection_kind_mismatch_field_count,
            readiness.kernel_argument_abi_semantic_projection_unsupported_encoding_field_count,
            readiness
                .kernel_argument_abi_semantic_projection_scalar_narrowing_overflow_field_count,
            readiness.kernel_argument_abi_semantic_projection_field_range_overflow_count,
            readiness.kernel_argument_abi_semantic_projection_projected_kernarg_bytes
        )
    };
    all_missing_execution_requests
        .execution_readiness
        .blockers
        .iter_mut()
        .find(|blocker| blocker.requirement == "kernel_argument_abi_semantic_projection")
        .unwrap()
        .detail = all_missing_semantic_projection_blocker_detail.clone();
    let all_missing_projection_selection_component = all_missing_execution_requests
        .components
        .iter_mut()
        .find(|component| component.request_plan == projection_selection_request_plan)
        .unwrap();
    all_missing_projection_selection_component.request_count = 0;
    all_missing_projection_selection_component.applied_count = 0;
    all_missing_projection_selection_component.pending_count = 0;
    all_missing_projection_selection_component.blocker_present = true;
    all_missing_projection_selection_component.blocker_detail =
        Some(all_missing_semantic_projection_blocker_detail.clone());
    all_missing_execution_requests.component_request_count = all_missing_execution_requests
        .component_request_count
        .checked_sub(removed_projection_selection_requests)
        .unwrap();
    all_missing_execution_requests.component_pending_count = all_missing_execution_requests
        .component_pending_count
        .checked_sub(removed_projection_selection_requests)
        .unwrap();
    all_missing_execution_requests.assert_consistent()?;
    let all_missing_submission_gate = all_missing_execution_requests.submission_gate()?;
    assert!(!all_missing_submission_gate.submission_ready);
    assert!(all_missing_submission_gate.blockers.iter().any(|blocker| {
        blocker.source == "execution_readiness"
            && blocker.requirement == "kernel_argument_abi_semantic_projection"
            && blocker.detail.contains("selection_requests=0")
            && blocker.detail.contains(&format!(
                "missing_selection_requests={}",
                all_missing_execution_requests.dispatch_count
            ))
    }));
    let all_missing_submission_prerequisites =
        all_missing_execution_requests.submission_prerequisite_plan()?;
    let all_missing_projection_selection_prerequisite = all_missing_submission_prerequisites
        .prerequisites
        .iter()
        .find(|prerequisite| prerequisite.request_plan == projection_selection_request_plan)
        .unwrap();
    assert_eq!(
        all_missing_projection_selection_prerequisite.request_count,
        0
    );
    assert_eq!(
        all_missing_projection_selection_prerequisite.applied_count,
        0
    );
    assert_eq!(
        all_missing_projection_selection_prerequisite.pending_count,
        0
    );
    assert!(all_missing_projection_selection_prerequisite.blocker_present);
    assert!(!all_missing_projection_selection_prerequisite.prerequisite_satisfied);
    assert!(all_missing_projection_selection_prerequisite
        .blocker_detail
        .as_ref()
        .unwrap()
        .contains("selection_requests=0"));
    let aql_live_component = execution_requests
        .component_for("aql_live_relocation_binding_request_plan")
        .unwrap();
    let aql_live_descriptor =
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding.descriptor();
    assert_eq!(
        aql_live_descriptor.live_aql_proof_kind,
        Some(RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan)
    );
    assert_eq!(
        aql_live_component.live_aql_proof_kind,
        aql_live_descriptor.live_aql_proof_kind
    );
    assert_eq!(
        aql_live_component.live_aql_proof_input,
        aql_live_descriptor.live_aql_proof_input
    );
    assert_eq!(
        aql_live_component.live_aql_validation_method,
        aql_live_descriptor.live_aql_validation_method
    );
    assert_eq!(
        aql_live_component.mutates_live_queue,
        aql_live_descriptor.mutates_live_queue
    );
    assert_eq!(
        execution_requests
            .component_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .unwrap()
            .request_plan,
        aql_live_component.request_plan
    );
    assert_eq!(
        aql_live_component.step,
        RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding
    );
    assert_eq!(
        aql_live_component.request_count,
        aql_live_relocation_bindings.kernarg_allocation_request_count
            + aql_live_relocation_bindings.completion_signal_request_count
    );
    let queue_component = execution_requests
        .component_for("queue_reservation_request_plan")
        .unwrap();
    let queue_descriptor = RuntimeLaunchExecutionRequestStep::QueueReservation.descriptor();
    assert_eq!(
        queue_descriptor.live_aql_proof_kind,
        Some(RuntimeLaunchLiveAqlProofKind::BatchReservationPlan)
    );
    assert_eq!(
        queue_component.live_aql_proof_kind,
        queue_descriptor.live_aql_proof_kind
    );
    assert_eq!(
        queue_component.live_aql_proof_input,
        queue_descriptor.live_aql_proof_input
    );
    assert_eq!(
        queue_component.live_aql_validation_method,
        queue_descriptor.live_aql_validation_method
    );
    assert_eq!(
        queue_component.mutates_live_queue,
        queue_descriptor.mutates_live_queue
    );
    assert_eq!(
        execution_requests
            .component_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .request_plan,
        queue_component.request_plan
    );
    assert_eq!(
        queue_component.step,
        RuntimeLaunchExecutionRequestStep::QueueReservation
    );
    assert!(queue_component
        .blocker_detail
        .as_ref()
        .unwrap()
        .contains("queue packet reservation"));
    let proof_surface_request_count =
        queue_component.request_count + aql_live_component.request_count;
    assert_eq!(execution_requests.live_aql_proof_surface_count, 2);
    assert_eq!(execution_requests.live_aql_proof_surfaces.len(), 2);
    assert_eq!(
        execution_requests
            .live_aql_proof_surface_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert_eq!(
        execution_requests
            .pending_live_aql_proof_surface_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert_eq!(
        execution_requests
            .pending_live_aql_proof_validation_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert!(execution_requests
        .live_aql_submitting_surface_request_plan_names()
        .is_empty());
    assert!(execution_requests
        .live_queue_mutating_component_request_plan_names()
        .is_empty());
    assert_eq!(
        execution_requests.live_aql_proof_input_labels().as_slice(),
        &[
            "KfdQueueLiveAqlBatchReservationPlanInput",
            "KfdQueueLiveAqlMaterializedPacketPlanInput",
        ]
    );
    assert_eq!(
        execution_requests.live_aql_proof_kind_labels().as_slice(),
        &["batch_reservation_plan", "materialized_packet_plan"]
    );
    assert_eq!(
        execution_requests
            .live_aql_validation_method_labels()
            .as_slice(),
        &[
            "KfdQueueLiveAqlBatchReservationPlanProof::validate_ready",
            "KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready",
        ]
    );
    assert_eq!(
        execution_requests
            .live_aql_proof_surfaces
            .first()
            .unwrap()
            .request_plan,
        "queue_reservation_request_plan"
    );
    assert_eq!(
        execution_requests
            .live_aql_proof_surfaces
            .last()
            .unwrap()
            .request_plan,
        "aql_live_relocation_binding_request_plan"
    );
    assert_eq!(
        execution_requests.live_aql_proof_surface_request_count,
        proof_surface_request_count
    );
    assert_eq!(
        execution_requests.live_aql_proof_surface_pending_count,
        proof_surface_request_count
    );
    assert_eq!(
        execution_requests.live_aql_proof_validation_request_count,
        2
    );
    assert_eq!(
        execution_requests.live_aql_proof_validation_applied_count,
        0
    );
    assert_eq!(
        execution_requests.live_aql_proof_validation_pending_count,
        2
    );
    assert_eq!(execution_requests.live_aql_submitting_surface_count, 0);
    assert!(execution_requests
        .live_aql_proof_surfaces
        .iter()
        .all(|surface| {
            surface.request_plan_ready
                && surface.blocker_present
                && surface.pending_count == surface.request_count
                && surface.proof_validation_request_count == 1
                && surface.proof_validation_applied_count == 0
                && surface.proof_validation_pending_count == 1
                && !surface.proof_input_constructed
                && !surface.submits_work
                && !surface.mutates_live_queue
        }));
    let queue_surface = execution_requests
        .live_aql_proof_surface_for("queue_reservation_request_plan")
        .unwrap();
    assert_eq!(
        execution_requests
            .live_aql_proof_surface_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .request_plan,
        queue_surface.request_plan
    );
    assert_eq!(queue_surface.step, queue_component.step);
    assert_eq!(queue_surface.step_index, queue_component.step_index);
    assert_eq!(queue_surface.requirement, queue_component.requirement);
    assert_eq!(queue_surface.request_count, queue_component.request_count);
    assert_eq!(queue_surface.pending_count, queue_component.pending_count);
    assert_eq!(
        queue_surface.proof_kind,
        RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
    );
    assert_eq!(
        queue_surface.proof_type,
        "KfdQueueLiveAqlBatchReservationPlanProof"
    );
    assert_eq!(
        queue_surface.validation_type,
        "KfdQueueLiveAqlBatchReservationPlanValidation"
    );
    assert_eq!(queue_surface.validation_ready_field, "ready");
    assert_eq!(
        queue_surface.no_live_queue_mutation_contract_field,
        "no_live_queue_mutation_contract"
    );
    assert_eq!(queue_surface.proof_validation_request_count, 1);
    assert_eq!(queue_surface.proof_validation_applied_count, 0);
    assert_eq!(queue_surface.proof_validation_pending_count, 1);
    assert_eq!(
        queue_surface.proof_input,
        queue_descriptor.live_aql_proof_input.unwrap()
    );
    assert_eq!(
        queue_surface.validation_method,
        queue_descriptor.live_aql_validation_method.unwrap()
    );
    let aql_live_surface = execution_requests
        .live_aql_proof_surface_for("aql_live_relocation_binding_request_plan")
        .unwrap();
    assert_eq!(
        execution_requests
            .live_aql_proof_surface_for_step(
                RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding
            )
            .unwrap()
            .request_plan,
        aql_live_surface.request_plan
    );
    assert!(execution_requests
        .live_aql_proof_surface_for("kernel_argument_abi_schema_request_plan")
        .is_none());
    assert!(execution_requests
        .live_aql_proof_surface_for_step(RuntimeLaunchExecutionRequestStep::KernelArgumentAbiSchema)
        .is_none());
    assert_eq!(aql_live_surface.step, aql_live_component.step);
    assert_eq!(aql_live_surface.step_index, aql_live_component.step_index);
    assert_eq!(aql_live_surface.requirement, aql_live_component.requirement);
    assert_eq!(
        aql_live_surface.request_count,
        aql_live_component.request_count
    );
    assert_eq!(
        aql_live_surface.pending_count,
        aql_live_component.pending_count
    );
    assert_eq!(
        aql_live_surface.proof_kind,
        RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
    );
    assert_eq!(
        aql_live_surface.proof_type,
        "KfdQueueLiveAqlMaterializedPacketPlanProof"
    );
    assert_eq!(
        aql_live_surface.validation_type,
        "KfdQueueLiveAqlMaterializedPacketPlanValidation"
    );
    assert_eq!(aql_live_surface.validation_ready_field, "ready");
    assert_eq!(
        aql_live_surface.no_live_queue_mutation_contract_field,
        "no_live_queue_mutation_contract"
    );
    assert_eq!(aql_live_surface.proof_validation_request_count, 1);
    assert_eq!(aql_live_surface.proof_validation_applied_count, 0);
    assert_eq!(aql_live_surface.proof_validation_pending_count, 1);
    assert_eq!(
        aql_live_surface.proof_input,
        aql_live_descriptor.live_aql_proof_input.unwrap()
    );
    assert_eq!(
        aql_live_surface.validation_method,
        aql_live_descriptor.live_aql_validation_method.unwrap()
    );
    let missing_validation_application_plan =
        execution_requests.live_aql_proof_validation_application_plan(&[])?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(
                "external",
                &[],
            )?,
        missing_validation_application_plan
    );
    assert_eq!(
        missing_validation_application_plan.proof_surface_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        missing_validation_application_plan.validation_input_count,
        0
    );
    assert_eq!(
        missing_validation_application_plan.application_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        missing_validation_application_plan.validation_present_count,
        0
    );
    assert_eq!(
        missing_validation_application_plan.validation_applied_count,
        0
    );
    assert_eq!(
        missing_validation_application_plan.validation_pending_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        missing_validation_application_plan.missing_validation_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        missing_validation_application_plan.failed_validation_count,
        0
    );
    assert!(!missing_validation_application_plan.all_validations_present);
    assert!(!missing_validation_application_plan.all_validations_applied);
    assert_eq!(
        missing_validation_application_plan
            .pending_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert_eq!(
        missing_validation_application_plan
            .application_for("queue_reservation_request_plan")
            .unwrap()
            .rejection_reason,
        Some("missing_validation")
    );
    missing_validation_application_plan.assert_consistent()?;
    assert!(missing_validation_application_plan.is_non_submitting_boundary());
    missing_validation_application_plan.assert_non_submitting_boundary()?;

    let batch_validation = RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
        .validate_batch_reservation_plan_proof(public_live_aql_batch_reservation_plan_proof())?;
    let materialized_validation = RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
        .validate_materialized_packet_plan_proof(public_live_aql_materialized_packet_plan_proof())?;
    let validation_application_plan = execution_requests
        .live_aql_proof_validation_application_plan(&[batch_validation, materialized_validation])?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(
                "external",
                &[batch_validation, materialized_validation],
            )?,
        validation_application_plan
    );
    assert_eq!(validation_application_plan.target, catalog.target);
    assert_eq!(
        validation_application_plan.code_object_target,
        execution_requests.code_object_target
    );
    assert_eq!(
        validation_application_plan.code_object_sha256,
        execution_requests.code_object_sha256
    );
    assert_eq!(validation_application_plan.proof_surface_count, 2);
    assert_eq!(validation_application_plan.validation_input_count, 2);
    assert_eq!(validation_application_plan.application_count, 2);
    assert_eq!(validation_application_plan.validation_present_count, 2);
    assert_eq!(validation_application_plan.validation_applied_count, 2);
    assert_eq!(validation_application_plan.validation_pending_count, 0);
    assert_eq!(validation_application_plan.missing_validation_count, 0);
    assert_eq!(validation_application_plan.unexpected_validation_count, 0);
    assert_eq!(validation_application_plan.failed_validation_count, 0);
    assert_eq!(validation_application_plan.not_ready_validation_count, 0);
    assert_eq!(
        validation_application_plan.no_live_queue_mutation_contract_missing_count,
        0
    );
    assert_eq!(
        validation_application_plan.live_aql_submitting_validation_count,
        0
    );
    assert_eq!(
        validation_application_plan.live_queue_mutating_validation_count,
        0
    );
    assert!(validation_application_plan.all_validations_present);
    assert!(validation_application_plan.all_validations_applied);
    assert!(validation_application_plan.no_live_aql_submission_side_effects);
    assert!(validation_application_plan.no_live_queue_mutation);
    assert_eq!(
        validation_application_plan
            .applied_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert!(validation_application_plan
        .pending_request_plan_names()
        .is_empty());
    assert_eq!(
        validation_application_plan
            .applied_proof_kind_labels()
            .as_slice(),
        &["batch_reservation_plan", "materialized_packet_plan"]
    );
    assert!(validation_application_plan.is_non_submitting_boundary());
    validation_application_plan.assert_non_submitting_boundary()?;
    let mut stale_count_validation_application_plan = validation_application_plan.clone();
    stale_count_validation_application_plan.validation_present_count = 0;
    assert!(!stale_count_validation_application_plan.is_non_submitting_boundary());
    let stale_count_validation_application_plan_err = stale_count_validation_application_plan
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_validation_application_plan_err.contains("consistency"));
    assert!(stale_count_validation_application_plan_err
        .contains("validation present count 2 != plan 0"));
    macro_rules! assert_validation_application_non_submitting_rejected {
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
    let mut side_effect_guard_validation_application_plan = validation_application_plan.clone();
    side_effect_guard_validation_application_plan.no_live_aql_submission_side_effects = false;
    assert_validation_application_non_submitting_rejected!(
        side_effect_guard_validation_application_plan,
        "live AQL submission side-effect guard is false",
    );
    let mut submitting_count_validation_application_plan = validation_application_plan.clone();
    submitting_count_validation_application_plan.live_aql_submitting_validation_count = 1;
    assert_validation_application_non_submitting_rejected!(
        submitting_count_validation_application_plan,
        "live AQL submitting validations 1 != 0",
    );
    let mut submitting_row_validation_application_plan = validation_application_plan.clone();
    submitting_row_validation_application_plan.applications[0].live_aql_submits_work = true;
    assert_validation_application_non_submitting_rejected!(
        submitting_row_validation_application_plan,
        "live AQL submitting validation rows queue_reservation_request_plan",
    );
    let mut queue_mutation_guard_validation_application_plan = validation_application_plan.clone();
    queue_mutation_guard_validation_application_plan.no_live_queue_mutation = false;
    assert_validation_application_non_submitting_rejected!(
        queue_mutation_guard_validation_application_plan,
        "live queue mutation guard is false",
    );
    let mut queue_mutating_count_validation_application_plan = validation_application_plan.clone();
    queue_mutating_count_validation_application_plan.live_queue_mutating_validation_count = 1;
    assert_validation_application_non_submitting_rejected!(
        queue_mutating_count_validation_application_plan,
        "live queue mutating validations 1 != 0",
    );
    let mut queue_mutating_row_validation_application_plan = validation_application_plan.clone();
    queue_mutating_row_validation_application_plan.applications[0].mutates_live_queue = true;
    assert_validation_application_non_submitting_rejected!(
        queue_mutating_row_validation_application_plan,
        "live queue mutating validation rows queue_reservation_request_plan",
    );
    let queue_validation_application = validation_application_plan
        .application_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
        .unwrap();
    assert_eq!(
        queue_validation_application.request_plan,
        "queue_reservation_request_plan"
    );
    assert_eq!(
        queue_validation_application.proof_kind,
        RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
    );
    assert_eq!(
        queue_validation_application.validation_receipt_fingerprint,
        batch_validation.receipt_fingerprint()
    );
    assert!(queue_validation_application.validation_present);
    assert!(queue_validation_application.validation_passed);
    assert!(queue_validation_application.validation_ready);
    assert!(queue_validation_application.no_live_queue_mutation_contract);
    assert!(!queue_validation_application.live_aql_submits_work);
    assert!(!queue_validation_application.mutates_live_queue);
    assert!(queue_validation_application.validation_applied);
    assert_eq!(queue_validation_application.validation_pending_count, 0);
    assert_eq!(queue_validation_application.rejection_reason, None);
    let validation_application_receipt_lines = validation_application_plan.receipt_lines();
    assert_eq!(
        validation_application_receipt_lines[0],
        "receipt.kind=model_runtime_launch_live_aql_proof_validation_application_plan"
    );
    assert_eq!(validation_application_receipt_lines[1], "receipt.version=1");
    assert!(validation_application_receipt_lines
        .iter()
        .any(|line| line == "all_validations_applied=true"));
    assert!(validation_application_receipt_lines
        .iter()
        .any(|line| line == "applications.0.validation_applied=true"));
    assert!(validation_application_plan.receipt_text().ends_with('\n'));
    assert_eq!(validation_application_plan.receipt_fingerprint().len(), 64);
    validation_application_plan.assert_consistent()?;
    let submission_gate_after_validation_application = execution_requests.submission_gate()?;
    assert_eq!(
        submission_gate_after_validation_application.live_aql_proof_validation_pending_count,
        execution_requests.live_aql_proof_validation_pending_count
    );
    assert!(!submission_gate_after_validation_application.all_live_aql_proof_validations_applied);
    assert!(!submission_gate_after_validation_application.submission_ready);

    let mut failed_batch_proof = public_live_aql_batch_reservation_plan_proof();
    failed_batch_proof.ready = 0;
    let failed_batch_validation = RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
        .validate_batch_reservation_plan_proof(failed_batch_proof)?;
    let failed_validation_application_plan = execution_requests
        .live_aql_proof_validation_application_plan(&[
            failed_batch_validation,
            materialized_validation,
        ])?;
    assert_eq!(
        failed_validation_application_plan.validation_present_count,
        2
    );
    assert_eq!(
        failed_validation_application_plan.validation_applied_count,
        1
    );
    assert_eq!(
        failed_validation_application_plan.validation_pending_count,
        1
    );
    assert_eq!(
        failed_validation_application_plan.failed_validation_count,
        1
    );
    assert_eq!(
        failed_validation_application_plan.not_ready_validation_count,
        1
    );
    assert!(!failed_validation_application_plan.all_validations_applied);
    assert_eq!(
        failed_validation_application_plan
            .pending_request_plan_names()
            .as_slice(),
        &["queue_reservation_request_plan"]
    );
    assert_eq!(
        failed_validation_application_plan
            .application_for("queue_reservation_request_plan")
            .unwrap()
            .rejection_reason,
        Some("validation_not_passed")
    );
    failed_validation_application_plan.assert_consistent()?;
    assert!(failed_validation_application_plan.is_non_submitting_boundary());
    failed_validation_application_plan.assert_non_submitting_boundary()?;

    let duplicate_validation_err = execution_requests
        .live_aql_proof_validation_application_plan(&[batch_validation, batch_validation])
        .unwrap_err()
        .to_string();
    assert!(duplicate_validation_err.contains("batch_reservation_plan appears more than once"));
    assert!(execution_requests.component_request_count > execution_requests.dispatch_count);
    assert_eq!(
        execution_requests.component_applied_count,
        kernel_argument_abi_component.applied_count
    );
    assert_eq!(
        execution_requests.component_pending_count,
        execution_requests.component_request_count - execution_requests.component_applied_count
    );
    assert_eq!(execution_requests.live_aql_proof_component_count, 2);
    assert_eq!(execution_requests.live_queue_mutating_component_count, 0);
    assert!(execution_requests.is_non_submitting_boundary());
    execution_requests.assert_non_submitting_boundary()?;
    let mut stale_count_execution_requests = execution_requests.clone();
    stale_count_execution_requests.component_pending_count = 0;
    assert!(!stale_count_execution_requests.is_non_submitting_boundary());
    let stale_count_execution_requests_err = stale_count_execution_requests
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_execution_requests_err.contains("consistency"));
    assert!(stale_count_execution_requests_err.contains("component pending count"));
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
    let mut submitting_surface_count_execution_requests = execution_requests.clone();
    submitting_surface_count_execution_requests.live_aql_submitting_surface_count = 1;
    assert_execution_request_non_submitting_rejected!(
        submitting_surface_count_execution_requests,
        "live AQL submitting surfaces 1 != 0",
    );
    let mut submitting_surface_row_execution_requests = execution_requests.clone();
    submitting_surface_row_execution_requests
        .live_aql_proof_surfaces
        .iter_mut()
        .find(|surface| surface.request_plan == "queue_reservation_request_plan")
        .unwrap()
        .submits_work = true;
    assert_execution_request_non_submitting_rejected!(
        submitting_surface_row_execution_requests,
        "live AQL submitting surface rows queue_reservation_request_plan",
    );
    let mut queue_mutating_component_count_execution_requests = execution_requests.clone();
    queue_mutating_component_count_execution_requests.live_queue_mutating_component_count = 1;
    assert_execution_request_non_submitting_rejected!(
        queue_mutating_component_count_execution_requests,
        "live queue mutating components 1 != 0",
    );
    let mut queue_mutating_component_row_execution_requests = execution_requests.clone();
    queue_mutating_component_row_execution_requests
        .components
        .iter_mut()
        .find(|component| component.request_plan == "code_object_load_request_plan")
        .unwrap()
        .mutates_live_queue = true;
    assert_execution_request_non_submitting_rejected!(
        queue_mutating_component_row_execution_requests,
        "live queue mutating component rows code_object_load_request_plan",
    );
    assert_eq!(
        execution_requests.execution_blocker_count,
        execution_readiness.blockers.len()
    );
    let submission_gate = execution_requests.submission_gate().unwrap();
    assert_eq!(submission_gate.target, catalog.target);
    assert_eq!(
        submission_gate.code_object_target,
        execution_requests.code_object_target
    );
    assert_eq!(
        submission_gate.code_object_sha256,
        execution_requests.code_object_sha256
    );
    assert_eq!(
        submission_gate.dispatch_count,
        execution_requests.dispatch_count
    );
    assert_eq!(
        submission_gate.window_count,
        execution_requests.window_count
    );
    assert!(submission_gate.request_plan_ready);
    assert!(!submission_gate.execution_readiness_ready);
    assert!(!submission_gate.all_components_applied);
    assert!(!submission_gate.all_live_aql_proof_validations_applied);
    assert!(submission_gate.no_live_aql_submission_side_effects);
    assert!(submission_gate.no_live_queue_mutation);
    assert_eq!(
        submission_gate.component_pending_count,
        execution_requests.component_pending_count
    );
    assert_eq!(
        submission_gate.live_aql_proof_validation_pending_count,
        execution_requests.live_aql_proof_validation_pending_count
    );
    assert_eq!(submission_gate.live_aql_submitting_surface_count, 0);
    assert_eq!(submission_gate.live_queue_mutating_component_count, 0);
    assert!(submission_gate.is_non_submitting_boundary());
    submission_gate.assert_non_submitting_boundary()?;
    let mut stale_count_submission_gate = submission_gate.clone();
    stale_count_submission_gate.submission_blocker_count = 0;
    assert!(!stale_count_submission_gate.is_non_submitting_boundary());
    let stale_count_submission_gate_err = stale_count_submission_gate
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_submission_gate_err.contains("consistency"));
    assert!(stale_count_submission_gate_err.contains("submission blocker count"));
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
    let mut side_effect_guard_submission_gate = submission_gate.clone();
    side_effect_guard_submission_gate.no_live_aql_submission_side_effects = false;
    assert_submission_gate_non_submitting_rejected!(
        side_effect_guard_submission_gate,
        "live AQL submission side-effect guard is false",
    );
    let mut submitting_surface_submission_gate = submission_gate.clone();
    submitting_surface_submission_gate.live_aql_submitting_surface_count = 1;
    assert_submission_gate_non_submitting_rejected!(
        submitting_surface_submission_gate,
        "live AQL submitting surfaces 1 != 0",
    );
    let mut queue_mutation_guard_submission_gate = submission_gate.clone();
    queue_mutation_guard_submission_gate.no_live_queue_mutation = false;
    assert_submission_gate_non_submitting_rejected!(
        queue_mutation_guard_submission_gate,
        "live queue mutation guard is false",
    );
    let mut queue_mutating_submission_gate = submission_gate.clone();
    queue_mutating_submission_gate.live_queue_mutating_component_count = 1;
    assert_submission_gate_non_submitting_rejected!(
        queue_mutating_submission_gate,
        "live queue mutating components 1 != 0",
    );
    assert_eq!(
        submission_gate.execution_blocker_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        launch_preflight.submission_gate(&device_pointer_validation)?,
        submission_gate
    );
    assert_eq!(
        admission.runtime_launch_submission_gate(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        submission_gate
    );
    assert_eq!(
        readiness.runtime_launch_submission_gate(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        submission_gate
    );
    assert_eq!(
        submission_gate.submission_blocker_count,
        execution_readiness.blockers.len() + 2
    );
    assert!(!submission_gate.submission_ready);
    assert!(submission_gate.has_blocker("queue_reservation"));
    assert!(submission_gate.has_blocker("runtime_request_components"));
    assert!(submission_gate.has_blocker("live_aql_proof_validation"));
    let expected_submission_blockers = [
        "kernel_candidate_selection_policy",
        "host_launcher_runtime_branch_resolution",
        "loaded_code_object_base",
        "kernarg_allocation",
        "kernel_argument_abi_verification",
        "kernel_argument_abi_semantic_projection",
        "completion_signal_binding",
        "queue_reservation",
        "aql_packet_materialization",
        "runtime_request_components",
        "live_aql_proof_validation",
    ];
    assert_eq!(
        submission_gate.blocker_requirement_names().as_slice(),
        &expected_submission_blockers
    );
    assert_eq!(
        submission_gate
            .blocker_for("runtime_request_components")
            .unwrap()
            .pending_count,
        execution_requests.component_pending_count
    );
    assert_eq!(
        submission_gate
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .requirement,
        "queue_reservation"
    );
    assert_eq!(
        submission_gate
            .blocker_for("live_aql_proof_validation")
            .unwrap()
            .pending_count,
        2
    );
    let submission_gate_with_validations = execution_requests
        .submission_gate_with_live_aql_proof_validations(&[
            batch_validation,
            materialized_validation,
        ])?;
    assert_eq!(
        execution_requests.submission_gate_with_live_aql_proof_validation_application_plan(
            &validation_application_plan
        )?,
        submission_gate_with_validations
    );
    assert!(submission_gate_with_validations.request_plan_ready);
    assert!(!submission_gate_with_validations.execution_readiness_ready);
    assert!(!submission_gate_with_validations.all_components_applied);
    assert!(submission_gate_with_validations.all_live_aql_proof_validations_applied);
    assert!(submission_gate_with_validations.no_live_aql_submission_side_effects);
    assert!(submission_gate_with_validations.no_live_queue_mutation);
    assert_eq!(
        submission_gate_with_validations.component_pending_count,
        execution_requests.component_pending_count
    );
    assert_eq!(
        submission_gate_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        submission_gate_with_validations.submission_blocker_count,
        execution_readiness.blockers.len() + 1
    );
    assert!(!submission_gate_with_validations.submission_ready);
    assert!(submission_gate_with_validations.has_blocker("queue_reservation"));
    assert!(submission_gate_with_validations.has_blocker("runtime_request_components"));
    assert!(!submission_gate_with_validations.has_blocker("live_aql_proof_validation"));
    let expected_submission_blockers_with_validations = [
        "kernel_candidate_selection_policy",
        "host_launcher_runtime_branch_resolution",
        "loaded_code_object_base",
        "kernarg_allocation",
        "kernel_argument_abi_verification",
        "kernel_argument_abi_semantic_projection",
        "completion_signal_binding",
        "queue_reservation",
        "aql_packet_materialization",
        "runtime_request_components",
    ];
    assert_eq!(
        submission_gate_with_validations
            .blocker_requirement_names()
            .as_slice(),
        &expected_submission_blockers_with_validations
    );
    let submission_gate_with_validations_receipt_lines =
        submission_gate_with_validations.receipt_lines();
    assert!(submission_gate_with_validations_receipt_lines
        .iter()
        .any(|line| line == "live_aql_proof_validation_pending_count=0"));
    assert!(submission_gate_with_validations_receipt_lines
        .iter()
        .any(|line| line == "all_live_aql_proof_validations_applied=true"));
    assert!(!submission_gate_with_validations_receipt_lines
        .iter()
        .any(|line| line.contains("requirement=live_aql_proof_validation")));
    submission_gate_with_validations.assert_consistent()?;

    let failed_submission_gate = execution_requests
        .submission_gate_with_live_aql_proof_validation_application_plan(
            &failed_validation_application_plan,
        )?;
    assert!(!failed_submission_gate.all_live_aql_proof_validations_applied);
    assert_eq!(
        failed_submission_gate.live_aql_proof_validation_pending_count,
        1
    );
    assert!(failed_submission_gate.has_blocker("live_aql_proof_validation"));
    assert_eq!(
        failed_submission_gate
            .blocker_for("live_aql_proof_validation")
            .unwrap()
            .pending_count,
        1
    );
    assert!(!failed_submission_gate.submission_ready);

    let mut mismatched_validation_application_plan = validation_application_plan.clone();
    mismatched_validation_application_plan
        .code_object_sha256
        .push_str("stale");
    let mismatched_validation_application_err = execution_requests
        .submission_gate_with_live_aql_proof_validation_application_plan(
            &mismatched_validation_application_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(mismatched_validation_application_err.contains("does not match execution request plan"));
    let mut unexpected_validation_application_plan = validation_application_plan.clone();
    unexpected_validation_application_plan.validation_input_count += 1;
    unexpected_validation_application_plan.unexpected_validation_count = 1;
    unexpected_validation_application_plan.all_validations_present = false;
    unexpected_validation_application_plan.all_validations_applied = false;
    let unexpected_validation_application_err = execution_requests
        .submission_gate_with_live_aql_proof_validation_application_plan(
            &unexpected_validation_application_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(unexpected_validation_application_err.contains("unexpected validation input"));
    assert!(submission_gate.blocker_for("dispatch_geometry").is_none());
    let submission_gate_receipt_lines = submission_gate.receipt_lines();
    assert_eq!(
        submission_gate_receipt_lines[0],
        "receipt.kind=model_runtime_launch_submission_gate"
    );
    assert_eq!(submission_gate_receipt_lines[1], "receipt.version=1");
    assert!(submission_gate_receipt_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(submission_gate_receipt_lines.iter().any(|line| line
        == &format!(
            "submission_blocker_count={}",
            submission_gate.submission_blocker_count
        )));
    let runtime_request_component_blocker_index = submission_gate
        .blockers
        .iter()
        .position(|blocker| blocker.requirement == "runtime_request_components")
        .unwrap();
    assert!(submission_gate_receipt_lines.iter().any(|line| line
        == &format!(
            "blockers.{runtime_request_component_blocker_index}.requirement=runtime_request_components"
        )));
    assert!(submission_gate_receipt_lines.iter().any(|line| line
        == &format!(
            "blockers.{runtime_request_component_blocker_index}.pending_count={}",
            execution_requests.component_pending_count
        )));
    assert!(submission_gate.receipt_text().ends_with('\n'));
    let submission_gate_receipt_fingerprint = submission_gate.receipt_fingerprint();
    assert_eq!(submission_gate_receipt_fingerprint.len(), 64);
    assert!(submission_gate_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_submission_gate_receipt = submission_gate.clone();
    stale_submission_gate_receipt.component_pending_count += 1;
    assert_ne!(
        stale_submission_gate_receipt.receipt_fingerprint(),
        submission_gate_receipt_fingerprint
    );
    let submission_blockers = submission_gate.blocker_report().unwrap();
    assert_eq!(submission_blockers.target, catalog.target);
    assert_eq!(
        submission_blockers.blocker_count,
        submission_gate.submission_blocker_count
    );
    assert_eq!(
        submission_blockers.execution_readiness_blocker_count,
        submission_gate.execution_blocker_count
    );
    assert_eq!(
        submission_blockers.runtime_request_component_pending_count,
        submission_gate.component_pending_count
    );
    assert_eq!(
        submission_blockers.live_aql_proof_validation_pending_count,
        submission_gate.live_aql_proof_validation_pending_count
    );
    assert_eq!(submission_blockers.live_aql_submission_side_effect_count, 0);
    assert_eq!(submission_blockers.live_queue_mutation_count, 0);
    assert!(submission_blockers.is_non_submitting_boundary());
    submission_blockers.assert_non_submitting_boundary()?;
    let mut stale_count_submission_blockers = submission_blockers.clone();
    stale_count_submission_blockers.blocker_count = 0;
    assert!(!stale_count_submission_blockers.is_non_submitting_boundary());
    let stale_count_submission_blockers_err = stale_count_submission_blockers
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_count_submission_blockers_err.contains("consistency"));
    assert!(stale_count_submission_blockers_err.contains("submission blocker rows"));
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
    let mut side_effect_guard_submission_blockers = submission_blockers.clone();
    side_effect_guard_submission_blockers.no_live_aql_submission_side_effects = false;
    assert_submission_blocker_report_non_submitting_rejected!(
        side_effect_guard_submission_blockers,
        "live AQL submission side-effect guard is false",
    );
    let mut side_effecting_submission_blockers = submission_blockers.clone();
    side_effecting_submission_blockers.live_aql_submission_side_effect_count = 1;
    assert_submission_blocker_report_non_submitting_rejected!(
        side_effecting_submission_blockers,
        "live AQL submission side-effect count 1 != 0",
    );
    let mut queue_mutation_guard_submission_blockers = submission_blockers.clone();
    queue_mutation_guard_submission_blockers.no_live_queue_mutation = false;
    assert_submission_blocker_report_non_submitting_rejected!(
        queue_mutation_guard_submission_blockers,
        "live queue mutation guard is false",
    );
    let mut queue_mutating_submission_blockers = submission_blockers.clone();
    queue_mutating_submission_blockers.live_queue_mutation_count = 1;
    assert_submission_blocker_report_non_submitting_rejected!(
        queue_mutating_submission_blockers,
        "live queue mutation count 1 != 0",
    );
    assert_eq!(
        submission_blockers.total_pending_count,
        submission_gate.component_pending_count
            + submission_gate.live_aql_proof_validation_pending_count
    );
    assert!(!submission_blockers.submission_ready);
    assert_eq!(
        submission_blockers.blocker_requirement_names().as_slice(),
        &expected_submission_blockers
    );
    assert_eq!(
        submission_blockers
            .execution_readiness_blocker_requirement_names()
            .as_slice(),
        &expected_submission_blockers[..execution_readiness.blockers.len()]
    );
    assert_eq!(
        submission_blockers
            .runtime_request_component_blocker_requirement_names()
            .as_slice(),
        &["runtime_request_components"]
    );
    assert_eq!(
        submission_blockers
            .live_aql_proof_validation_blocker_requirement_names()
            .as_slice(),
        &["live_aql_proof_validation"]
    );
    assert!(submission_blockers
        .live_aql_submission_side_effect_blocker_requirement_names()
        .is_empty());
    assert!(submission_blockers
        .live_queue_mutation_blocker_requirement_names()
        .is_empty());
    assert!(
        submission_blockers
            .blocker_for("runtime_request_components")
            .unwrap()
            .runtime_request_component_blocker
    );
    assert!(
        submission_blockers
            .blocker_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .unwrap()
            .execution_readiness_blocker
    );
    assert!(
        submission_blockers
            .blocker_for("live_aql_proof_validation")
            .unwrap()
            .live_aql_proof_validation_blocker
    );
    assert!(submission_blockers
        .blocker_for("dispatch_geometry")
        .is_none());
    let submission_blocker_receipt_lines = submission_blockers.receipt_lines();
    assert_eq!(
        submission_blocker_receipt_lines[0],
        "receipt.kind=model_runtime_launch_submission_blocker_report"
    );
    assert_eq!(submission_blocker_receipt_lines[1], "receipt.version=1");
    assert!(submission_blocker_receipt_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(submission_blocker_receipt_lines
        .iter()
        .any(|line| line == &format!("blocker_count={}", submission_blockers.blocker_count)));
    assert!(submission_blocker_receipt_lines.iter().any(|line| line
        == &format!(
            "total_pending_count={}",
            submission_blockers.total_pending_count
        )));
    let runtime_request_component_report_blocker_index = submission_blockers
        .blockers
        .iter()
        .position(|blocker| blocker.requirement == "runtime_request_components")
        .unwrap();
    assert!(submission_blocker_receipt_lines.iter().any(|line| line
        == &format!(
            "blockers.{runtime_request_component_report_blocker_index}.requirement=runtime_request_components"
        )));
    assert!(submission_blocker_receipt_lines.iter().any(|line| line
        == &format!(
            "blockers.{runtime_request_component_report_blocker_index}.runtime_request_component_blocker=true"
        )));
    assert!(submission_blockers.receipt_text().ends_with('\n'));
    let submission_blocker_receipt_fingerprint = submission_blockers.receipt_fingerprint();
    assert_eq!(submission_blocker_receipt_fingerprint.len(), 64);
    assert!(submission_blocker_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_submission_blocker_receipt = submission_blockers.clone();
    stale_submission_blocker_receipt.blockers[runtime_request_component_report_blocker_index]
        .detail
        .push_str(" stale");
    assert_ne!(
        stale_submission_blocker_receipt.receipt_fingerprint(),
        submission_blocker_receipt_fingerprint
    );
    submission_blockers.assert_consistent()?;
    assert_eq!(
        launch_preflight.submission_blocker_report(&device_pointer_validation)?,
        submission_blockers
    );
    assert_eq!(
        admission.runtime_launch_submission_blocker_report(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        submission_blockers
    );
    assert_eq!(
        readiness.runtime_launch_submission_blocker_report(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        submission_blockers
    );
    let submission_prerequisites = execution_requests.submission_prerequisite_plan().unwrap();
    assert_eq!(submission_prerequisites.target, catalog.target);
    assert_eq!(
        submission_prerequisites.code_object_target,
        execution_requests.code_object_target
    );
    assert_eq!(
        submission_prerequisites.code_object_sha256,
        execution_requests.code_object_sha256
    );
    assert_eq!(
        submission_prerequisites.prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites.prerequisites.len(),
        execution_requests.components.len()
    );
    assert_eq!(submission_prerequisites.satisfied_prerequisite_count, 0);
    assert_eq!(
        submission_prerequisites.unsatisfied_prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites.next_action_count,
        submission_prerequisites.unsatisfied_prerequisite_count
    );
    assert_eq!(
        submission_prerequisites.request_plan_completion_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites.runtime_request_component_next_action_count,
        8
    );
    assert_eq!(
        submission_prerequisites.live_aql_proof_validation_next_action_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        submission_prerequisites.execution_readiness_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites.live_aql_submission_side_effect_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites.live_queue_mutation_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites.pending_component_request_count,
        execution_requests.component_pending_count
    );
    assert_eq!(
        submission_prerequisites.live_aql_proof_prerequisite_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        submission_prerequisites.live_aql_proof_input_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        submission_prerequisites.live_aql_validation_method_count,
        execution_requests.live_aql_proof_surface_count
    );
    assert_eq!(
        submission_prerequisites.live_aql_submitting_prerequisite_count,
        0
    );
    assert_eq!(
        submission_prerequisites.live_aql_proof_validation_pending_count,
        execution_requests.live_aql_proof_validation_pending_count
    );
    assert_eq!(
        submission_prerequisites.live_queue_mutating_prerequisite_count,
        0
    );
    assert!(submission_prerequisites.is_non_submitting_boundary());
    submission_prerequisites.assert_non_submitting_boundary()?;
    let mut stale_count_submission_prerequisites = submission_prerequisites.clone();
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
    let mut side_effect_next_action_submission_prerequisites = submission_prerequisites.clone();
    side_effect_next_action_submission_prerequisites
        .live_aql_submission_side_effect_next_action_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        side_effect_next_action_submission_prerequisites,
        "live AQL submission side-effect next actions 1 != 0",
    );
    let mut submitting_count_submission_prerequisites = submission_prerequisites.clone();
    submitting_count_submission_prerequisites.live_aql_submitting_prerequisite_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        submitting_count_submission_prerequisites,
        "live AQL submitting prerequisites 1 != 0",
    );
    let mut submitting_row_submission_prerequisites = submission_prerequisites.clone();
    submitting_row_submission_prerequisites.prerequisites[0].live_aql_submits_work = true;
    assert_submission_prerequisite_non_submitting_rejected!(
        submitting_row_submission_prerequisites,
        "live AQL submitting prerequisite rows code_object_load_request_plan",
    );
    let mut queue_mutation_next_action_submission_prerequisites = submission_prerequisites.clone();
    queue_mutation_next_action_submission_prerequisites.live_queue_mutation_next_action_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        queue_mutation_next_action_submission_prerequisites,
        "live queue mutation next actions 1 != 0",
    );
    let mut queue_mutating_count_submission_prerequisites = submission_prerequisites.clone();
    queue_mutating_count_submission_prerequisites.live_queue_mutating_prerequisite_count = 1;
    assert_submission_prerequisite_non_submitting_rejected!(
        queue_mutating_count_submission_prerequisites,
        "live queue mutating prerequisites 1 != 0",
    );
    let mut queue_mutating_row_submission_prerequisites = submission_prerequisites.clone();
    queue_mutating_row_submission_prerequisites.prerequisites[0].mutates_live_queue = true;
    assert_submission_prerequisite_non_submitting_rejected!(
        queue_mutating_row_submission_prerequisites,
        "live queue mutating prerequisite rows code_object_load_request_plan",
    );
    assert!(submission_prerequisites.request_plan_ready);
    assert!(!submission_prerequisites.execution_readiness_ready);
    assert!(!submission_prerequisites.all_prerequisites_satisfied);
    assert!(!submission_prerequisites.submission_ready);
    assert_eq!(
        submission_prerequisites.submission_ready,
        submission_gate.submission_ready
    );
    assert_eq!(
        submission_prerequisites
            .prerequisite_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert_eq!(
        submission_prerequisites
            .unsatisfied_prerequisite_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert_eq!(
        submission_prerequisites
            .next_action_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert_eq!(
        submission_prerequisites.next_action_labels().as_slice(),
        &[
            "apply_runtime_request_component",
            "apply_runtime_request_component",
            "apply_runtime_request_component",
            "validate_live_aql_proof",
            "apply_runtime_request_component",
            "apply_runtime_request_component",
            "apply_runtime_request_component",
            "apply_runtime_request_component",
            "apply_runtime_request_component",
            "validate_live_aql_proof",
        ]
    );
    let expected_runtime_component_next_action_plans = [
        "code_object_load_request_plan",
        "code_object_base_binding_request_plan",
        "completion_signal_binding_request_plan",
        "kernarg_allocation_request_plan",
        "kernel_argument_abi_schema_request_plan",
        "kernel_candidate_selection_request_plan",
        "kernel_argument_abi_semantic_projection_candidate_selection_request_plan",
        "host_launcher_branch_resolution_request_plan",
    ];
    assert_eq!(
        submission_prerequisites
            .runtime_request_component_next_action_request_plan_names()
            .as_slice(),
        &expected_runtime_component_next_action_plans
    );
    assert_eq!(
        submission_prerequisites
            .live_aql_proof_validation_next_action_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert!(submission_prerequisites
        .request_plan_completion_next_action_request_plan_names()
        .is_empty());
    assert!(submission_prerequisites
        .execution_readiness_next_action_request_plan_names()
        .is_empty());
    assert!(submission_prerequisites
        .live_aql_submission_side_effect_next_action_request_plan_names()
        .is_empty());
    assert!(submission_prerequisites
        .live_queue_mutation_next_action_request_plan_names()
        .is_empty());
    assert_eq!(
        submission_prerequisites
            .next_action_input_labels()
            .as_slice(),
        &[
            "code_object_load_request_plan",
            "code_object_base_binding_request_plan",
            "completion_signal_binding_request_plan",
            "KfdQueueLiveAqlBatchReservationPlanInput",
            "kernarg_allocation_request_plan",
            "kernel_argument_abi_schema_request_plan",
            "kernel_candidate_selection_request_plan",
            "kernel_argument_abi_semantic_projection_candidate_selection_request_plan",
            "host_launcher_branch_resolution_request_plan",
            "KfdQueueLiveAqlMaterializedPacketPlanInput",
        ]
    );
    assert_eq!(
        submission_prerequisites
            .live_aql_proof_prerequisite_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert!(submission_prerequisites
        .live_aql_submitting_prerequisite_request_plan_names()
        .is_empty());
    assert_eq!(
        submission_prerequisites
            .pending_live_aql_proof_validation_prerequisite_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert!(submission_prerequisites
        .live_queue_mutating_prerequisite_request_plan_names()
        .is_empty());
    assert_eq!(
        submission_prerequisites
            .live_aql_proof_input_labels()
            .as_slice(),
        &[
            "KfdQueueLiveAqlBatchReservationPlanInput",
            "KfdQueueLiveAqlMaterializedPacketPlanInput",
        ]
    );
    assert_eq!(
        submission_prerequisites
            .live_aql_proof_kind_labels()
            .as_slice(),
        &["batch_reservation_plan", "materialized_packet_plan"]
    );
    assert_eq!(
        submission_prerequisites
            .live_aql_validation_method_labels()
            .as_slice(),
        &[
            "KfdQueueLiveAqlBatchReservationPlanProof::validate_ready",
            "KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready",
        ]
    );
    assert!(submission_prerequisites
        .prerequisite_for("dispatch_geometry_request_plan")
        .is_none());
    let queue_prerequisite = submission_prerequisites
        .prerequisite_for("queue_reservation_request_plan")
        .unwrap();
    assert_eq!(
        submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .request_plan,
        queue_prerequisite.request_plan
    );
    assert_eq!(queue_prerequisite.step, queue_component.step);
    assert!(queue_prerequisite.live_aql_proof_required);
    assert_eq!(
        queue_prerequisite.live_aql_proof_kind,
        Some(RuntimeLaunchLiveAqlProofKind::BatchReservationPlan)
    );
    assert_eq!(
        queue_prerequisite.live_aql_proof_input,
        Some(queue_surface.proof_input)
    );
    assert_eq!(
        queue_prerequisite.live_aql_validation_method,
        Some(queue_surface.validation_method)
    );
    assert!(!queue_prerequisite.live_aql_submits_work);
    assert_eq!(
        queue_prerequisite.live_aql_proof_validation_pending_count,
        queue_surface.proof_validation_pending_count
    );
    assert!(!queue_prerequisite.prerequisite_satisfied);
    assert_eq!(
        queue_prerequisite.next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ValidateLiveAqlProof
    );
    assert_eq!(
        queue_prerequisite.next_action_input,
        queue_surface.proof_input
    );
    assert_eq!(
        queue_prerequisite.next_action_pending_count,
        queue_surface.proof_validation_pending_count
    );
    assert!(queue_prerequisite.next_action_uses_live_aql_proof);
    let aql_live_prerequisite = submission_prerequisites
        .prerequisite_for("aql_live_relocation_binding_request_plan")
        .unwrap();
    assert_eq!(
        submission_prerequisites
            .prerequisite_for_step(RuntimeLaunchExecutionRequestStep::AqlLiveRelocationBinding)
            .unwrap()
            .request_plan,
        aql_live_prerequisite.request_plan
    );
    assert_eq!(aql_live_prerequisite.step, aql_live_component.step);
    assert!(aql_live_prerequisite.live_aql_proof_required);
    assert_eq!(
        aql_live_prerequisite.live_aql_proof_kind,
        Some(RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan)
    );
    assert_eq!(
        aql_live_prerequisite.live_aql_proof_input,
        Some(aql_live_surface.proof_input)
    );
    assert_eq!(
        aql_live_prerequisite.live_aql_validation_method,
        Some(aql_live_surface.validation_method)
    );
    assert!(!aql_live_prerequisite.live_aql_submits_work);
    assert_eq!(
        aql_live_prerequisite.live_aql_proof_validation_pending_count,
        aql_live_surface.proof_validation_pending_count
    );
    assert!(!aql_live_prerequisite.prerequisite_satisfied);
    assert_eq!(
        aql_live_prerequisite.next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ValidateLiveAqlProof
    );
    assert_eq!(
        aql_live_prerequisite.next_action_input,
        aql_live_surface.proof_input
    );
    assert_eq!(
        aql_live_prerequisite.next_action_pending_count,
        aql_live_surface.proof_validation_pending_count
    );
    assert!(aql_live_prerequisite.next_action_uses_live_aql_proof);
    let submission_prerequisite_receipt_lines = submission_prerequisites.receipt_lines();
    assert_eq!(
        submission_prerequisite_receipt_lines[0],
        "receipt.kind=model_runtime_launch_submission_prerequisite_plan"
    );
    assert_eq!(
        submission_prerequisite_receipt_lines[1],
        "receipt.version=1"
    );
    assert!(submission_prerequisite_receipt_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisite_count={}",
            submission_prerequisites.prerequisite_count
        )));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "pending_component_request_count={}",
            submission_prerequisites.pending_component_request_count
        )));
    assert!(submission_prerequisite_receipt_lines
        .iter()
        .any(|line| line == "next_action_count=10"));
    assert!(submission_prerequisite_receipt_lines
        .iter()
        .any(|line| line == "runtime_request_component_next_action_count=8"));
    assert!(submission_prerequisite_receipt_lines
        .iter()
        .any(|line| line == "live_aql_proof_validation_next_action_count=2"));
    let queue_prerequisite_index = submission_prerequisites
        .prerequisites
        .iter()
        .position(|prerequisite| prerequisite.request_plan == "queue_reservation_request_plan")
        .unwrap();
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisites.{queue_prerequisite_index}.request_plan=queue_reservation_request_plan"
        )));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisites.{queue_prerequisite_index}.live_aql_proof_input={}",
            queue_surface.proof_input
        )));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisites.{queue_prerequisite_index}.live_aql_proof_kind={}",
            RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
        )));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisites.{queue_prerequisite_index}.live_aql_validation_method={}",
            queue_surface.validation_method
        )));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisites.{queue_prerequisite_index}.next_action={}",
            RuntimeLaunchSubmissionPrerequisiteNextAction::ValidateLiveAqlProof
        )));
    assert!(submission_prerequisite_receipt_lines.iter().any(|line| line
        == &format!(
            "prerequisites.{queue_prerequisite_index}.next_action_input={}",
            queue_surface.proof_input
        )));
    assert!(submission_prerequisites.receipt_text().ends_with('\n'));
    let submission_prerequisite_receipt_fingerprint =
        submission_prerequisites.receipt_fingerprint();
    assert_eq!(submission_prerequisite_receipt_fingerprint.len(), 64);
    assert!(submission_prerequisite_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_submission_prerequisite_receipt = submission_prerequisites.clone();
    stale_submission_prerequisite_receipt.prerequisites[queue_prerequisite_index].pending_count +=
        1;
    assert_ne!(
        stale_submission_prerequisite_receipt.receipt_fingerprint(),
        submission_prerequisite_receipt_fingerprint
    );
    submission_prerequisites.assert_consistent()?;
    let submission_prerequisites_with_validations = execution_requests
        .submission_prerequisite_plan_with_live_aql_proof_validations(&[
            batch_validation,
            materialized_validation,
        ])?;
    assert_eq!(
        execution_requests
            .submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(
                &validation_application_plan
            )?,
        submission_prerequisites_with_validations
    );
    assert_eq!(
        submission_prerequisites_with_validations.target,
        submission_prerequisites.target
    );
    assert_eq!(
        submission_prerequisites_with_validations.prerequisite_count,
        submission_prerequisites.prerequisite_count
    );
    assert_eq!(
        submission_prerequisites_with_validations.pending_component_request_count,
        submission_prerequisites.pending_component_request_count
    );
    assert_eq!(
        submission_prerequisites_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        submission_prerequisites_with_validations.live_aql_proof_validation_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites_with_validations.runtime_request_component_next_action_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites_with_validations.next_action_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites_with_validations.unsatisfied_prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert!(submission_prerequisites_with_validations.request_plan_ready);
    assert!(!submission_prerequisites_with_validations.execution_readiness_ready);
    assert!(!submission_prerequisites_with_validations.all_prerequisites_satisfied);
    assert!(!submission_prerequisites_with_validations.submission_ready);
    assert!(submission_prerequisites_with_validations
        .pending_live_aql_proof_validation_prerequisite_request_plan_names()
        .is_empty());
    assert_eq!(
        submission_prerequisites_with_validations
            .runtime_request_component_next_action_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert!(submission_prerequisites_with_validations
        .live_aql_proof_validation_next_action_request_plan_names()
        .is_empty());
    let prerequisite_next_action_labels_with_validations =
        submission_prerequisites_with_validations.next_action_labels();
    assert_eq!(
        prerequisite_next_action_labels_with_validations.len(),
        execution_requests.runtime_request_plan_count
    );
    assert!(prerequisite_next_action_labels_with_validations
        .iter()
        .all(|label| *label == "apply_runtime_request_component"));
    assert_eq!(
        submission_prerequisites_with_validations
            .next_action_input_labels()
            .as_slice(),
        &expected_execution_request_plans
    );
    let queue_prerequisite_with_validations = submission_prerequisites_with_validations
        .prerequisite_for("queue_reservation_request_plan")
        .unwrap();
    assert_eq!(
        queue_prerequisite_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        queue_prerequisite_with_validations.next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ApplyRuntimeRequestComponent
    );
    assert_eq!(
        queue_prerequisite_with_validations.next_action_input,
        "queue_reservation_request_plan"
    );
    assert_eq!(
        queue_prerequisite_with_validations.next_action_pending_count,
        queue_component.pending_count
    );
    assert!(!queue_prerequisite_with_validations.next_action_uses_live_aql_proof);
    assert_eq!(
        queue_prerequisite_with_validations.next_action_live_aql_proof_kind,
        None
    );
    let aql_live_prerequisite_with_validations = submission_prerequisites_with_validations
        .prerequisite_for("aql_live_relocation_binding_request_plan")
        .unwrap();
    assert_eq!(
        aql_live_prerequisite_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        aql_live_prerequisite_with_validations.next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ApplyRuntimeRequestComponent
    );
    assert_eq!(
        aql_live_prerequisite_with_validations.next_action_input,
        "aql_live_relocation_binding_request_plan"
    );
    assert_eq!(
        aql_live_prerequisite_with_validations.next_action_pending_count,
        aql_live_component.pending_count
    );
    assert!(!aql_live_prerequisite_with_validations.next_action_uses_live_aql_proof);
    assert_eq!(
        aql_live_prerequisite_with_validations.next_action_live_aql_proof_kind,
        None
    );
    let submission_prerequisite_receipt_lines_with_validations =
        submission_prerequisites_with_validations.receipt_lines();
    assert!(submission_prerequisite_receipt_lines_with_validations
        .iter()
        .any(|line| line == "live_aql_proof_validation_pending_count=0"));
    assert!(submission_prerequisite_receipt_lines_with_validations
        .iter()
        .any(|line| line == "runtime_request_component_next_action_count=10"));
    assert!(submission_prerequisite_receipt_lines_with_validations
        .iter()
        .any(|line| line == "live_aql_proof_validation_next_action_count=0"));
    assert!(!submission_prerequisite_receipt_lines_with_validations
        .iter()
        .any(|line| line.contains("next_action=validate_live_aql_proof")));
    submission_prerequisites_with_validations.assert_consistent()?;

    let failed_submission_prerequisites = execution_requests
        .submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(
            &failed_validation_application_plan,
        )?;
    assert_eq!(
        failed_submission_prerequisites.live_aql_proof_validation_pending_count,
        1
    );
    assert_eq!(
        failed_submission_prerequisites.live_aql_proof_validation_next_action_count,
        1
    );
    assert_eq!(
        failed_submission_prerequisites.runtime_request_component_next_action_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        failed_submission_prerequisites
            .pending_live_aql_proof_validation_prerequisite_request_plan_names()
            .as_slice(),
        &["queue_reservation_request_plan"]
    );
    assert_eq!(
        failed_submission_prerequisites
            .prerequisite_for("queue_reservation_request_plan")
            .unwrap()
            .next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ValidateLiveAqlProof
    );
    assert_eq!(
        failed_submission_prerequisites
            .prerequisite_for("aql_live_relocation_binding_request_plan")
            .unwrap()
            .next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ApplyRuntimeRequestComponent
    );
    failed_submission_prerequisites.assert_consistent()?;

    let runtime_component_applications =
        submission_prerequisites.runtime_request_component_application_plan()?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_runtime_request_component_application_plan(
                "external",
                &[],
            )?,
        runtime_component_applications
    );
    assert_eq!(
        runtime_component_applications.prerequisite_count,
        submission_prerequisites.prerequisite_count
    );
    assert_eq!(
        runtime_component_applications.application_count,
        expected_runtime_component_next_action_plans.len()
    );
    assert_eq!(
        runtime_component_applications.application_ready_count,
        runtime_component_applications.application_count
    );
    assert_eq!(runtime_component_applications.application_blocked_count, 0);
    assert_eq!(
        runtime_component_applications.pending_component_request_count,
        submission_prerequisites.pending_component_request_count
    );
    assert_eq!(
        runtime_component_applications.deferred_pending_component_request_count,
        queue_component.pending_count + aql_live_component.pending_count
    );
    assert_eq!(
        runtime_component_applications.application_pending_count,
        runtime_component_applications.pending_component_request_count
            - runtime_component_applications.deferred_pending_component_request_count
    );
    assert_eq!(
        runtime_component_applications.runtime_application_receipt_count,
        0
    );
    assert_eq!(
        runtime_component_applications.live_aql_proof_application_count,
        0
    );
    assert_eq!(
        runtime_component_applications.live_aql_proof_validation_pending_count,
        submission_prerequisites.live_aql_proof_validation_pending_count
    );
    assert_eq!(
        runtime_component_applications.live_aql_submitting_application_count,
        0
    );
    assert_eq!(
        runtime_component_applications.live_queue_mutating_application_count,
        0
    );
    assert!(runtime_component_applications.request_plan_ready);
    assert!(!runtime_component_applications.execution_readiness_ready);
    assert!(runtime_component_applications.all_application_requests_ready);
    assert!(!runtime_component_applications.all_components_applied);
    assert!(!runtime_component_applications.submission_ready);
    assert_eq!(
        runtime_component_applications
            .application_request_plan_names()
            .as_slice(),
        &expected_runtime_component_next_action_plans
    );
    assert_eq!(
        runtime_component_applications
            .ready_application_request_plan_names()
            .as_slice(),
        &expected_runtime_component_next_action_plans
    );
    assert!(runtime_component_applications
        .blocked_application_request_plan_names()
        .is_empty());
    assert!(runtime_component_applications
        .live_aql_proof_application_request_plan_names()
        .is_empty());
    assert!(runtime_component_applications
        .application_for("queue_reservation_request_plan")
        .is_none());
    assert!(runtime_component_applications
        .application_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
        .is_none());
    let code_object_load_application = runtime_component_applications
        .application_for("code_object_load_request_plan")
        .unwrap();
    assert_eq!(
        runtime_component_applications
            .application_for_step(RuntimeLaunchExecutionRequestStep::CodeObjectLoad)
            .unwrap()
            .request_plan,
        code_object_load_application.request_plan
    );
    assert_eq!(
        code_object_load_application.source_next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ApplyRuntimeRequestComponent
    );
    assert_eq!(
        code_object_load_application.source_next_action_input,
        code_object_load_application.request_plan
    );
    assert_eq!(
        code_object_load_application.source_next_action_pending_count,
        code_object_load_application.pending_count
    );
    assert!(code_object_load_application.application_ready);
    assert!(!code_object_load_application.application_blocked);
    assert_eq!(code_object_load_application.blocking_reason, None);
    assert!(code_object_load_application.blocker_present);
    assert!(!code_object_load_application.runtime_application_receipt_present);
    let runtime_component_application_receipt_lines =
        runtime_component_applications.receipt_lines();
    assert_eq!(
        runtime_component_application_receipt_lines[0],
        "receipt.kind=model_runtime_launch_runtime_request_component_application_plan"
    );
    assert_eq!(
        runtime_component_application_receipt_lines[1],
        "receipt.version=1"
    );
    assert!(runtime_component_application_receipt_lines
        .iter()
        .any(|line| line == "application_count=8"));
    assert!(runtime_component_application_receipt_lines
        .iter()
        .any(|line| line == "runtime_application_receipt_count=0"));
    assert!(runtime_component_application_receipt_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(runtime_component_application_receipt_lines
        .iter()
        .any(|line| line == "applications.count=8"));
    assert!(runtime_component_application_receipt_lines
        .iter()
        .any(|line| line == "applications.0.request_plan=code_object_load_request_plan"));
    assert!(runtime_component_applications
        .receipt_text()
        .ends_with('\n'));
    let runtime_component_application_receipt_fingerprint =
        runtime_component_applications.receipt_fingerprint();
    assert_eq!(runtime_component_application_receipt_fingerprint.len(), 64);
    assert!(runtime_component_application_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_runtime_component_applications = runtime_component_applications.clone();
    stale_runtime_component_applications.application_count += 1;
    assert_ne!(
        stale_runtime_component_applications.receipt_fingerprint(),
        runtime_component_application_receipt_fingerprint
    );
    assert!(stale_runtime_component_applications
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("application count"));
    runtime_component_applications.assert_consistent()?;
    assert!(runtime_component_applications.is_non_submitting_boundary());
    runtime_component_applications.assert_non_submitting_boundary()?;

    let runtime_component_applications_with_validations =
        submission_prerequisites_with_validations.runtime_request_component_application_plan()?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_runtime_request_component_application_plan(
                "external",
                &[batch_validation, materialized_validation],
            )?,
        runtime_component_applications_with_validations
    );
    assert_eq!(
        runtime_component_applications_with_validations.application_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        runtime_component_applications_with_validations.application_ready_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        runtime_component_applications_with_validations.application_blocked_count,
        0
    );
    assert_eq!(
        runtime_component_applications_with_validations.application_pending_count,
        submission_prerequisites_with_validations.pending_component_request_count
    );
    assert_eq!(
        runtime_component_applications_with_validations.deferred_pending_component_request_count,
        0
    );
    assert_eq!(
        runtime_component_applications_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        runtime_component_applications_with_validations.live_aql_proof_application_count,
        expected_live_aql_proof_surface_plans.len()
    );
    assert_eq!(
        runtime_component_applications_with_validations
            .application_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert_eq!(
        runtime_component_applications_with_validations
            .ready_application_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert_eq!(
        runtime_component_applications_with_validations
            .live_aql_proof_application_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert!(runtime_component_applications_with_validations.all_application_requests_ready);
    assert!(!runtime_component_applications_with_validations.all_components_applied);
    assert!(!runtime_component_applications_with_validations.submission_ready);
    let queue_application_with_validations = runtime_component_applications_with_validations
        .application_for("queue_reservation_request_plan")
        .unwrap();
    assert_eq!(
        runtime_component_applications_with_validations
            .application_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .request_plan,
        queue_application_with_validations.request_plan
    );
    assert_eq!(
        queue_application_with_validations.step,
        queue_component.step
    );
    assert!(queue_application_with_validations.live_aql_proof_required);
    assert_eq!(
        queue_application_with_validations.live_aql_proof_kind,
        Some(RuntimeLaunchLiveAqlProofKind::BatchReservationPlan)
    );
    assert_eq!(
        queue_application_with_validations.live_aql_proof_input,
        Some(queue_surface.proof_input)
    );
    assert_eq!(
        queue_application_with_validations.live_aql_validation_method,
        Some(queue_surface.validation_method)
    );
    assert_eq!(
        queue_application_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        queue_application_with_validations.source_next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ApplyRuntimeRequestComponent
    );
    assert_eq!(
        queue_application_with_validations.source_next_action_input,
        "queue_reservation_request_plan"
    );
    assert_eq!(
        queue_application_with_validations.source_next_action_pending_count,
        queue_component.pending_count
    );
    assert!(queue_application_with_validations.application_ready);
    assert!(!queue_application_with_validations.application_blocked);
    assert_eq!(queue_application_with_validations.blocking_reason, None);
    assert!(queue_application_with_validations.blocker_present);
    assert!(!queue_application_with_validations.live_aql_submits_work);
    assert!(!queue_application_with_validations.mutates_live_queue);
    assert!(!queue_application_with_validations.runtime_application_receipt_present);
    let aql_live_application_with_validations = runtime_component_applications_with_validations
        .application_for("aql_live_relocation_binding_request_plan")
        .unwrap();
    assert_eq!(
        aql_live_application_with_validations.live_aql_proof_kind,
        Some(RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan)
    );
    assert_eq!(
        aql_live_application_with_validations.live_aql_proof_input,
        Some(aql_live_surface.proof_input)
    );
    assert_eq!(
        aql_live_application_with_validations.live_aql_validation_method,
        Some(aql_live_surface.validation_method)
    );
    assert_eq!(
        aql_live_application_with_validations.live_aql_proof_validation_pending_count,
        0
    );
    let queue_application_index = runtime_component_applications_with_validations
        .applications
        .iter()
        .position(|application| application.request_plan == "queue_reservation_request_plan")
        .unwrap();
    let runtime_component_application_receipt_lines_with_validations =
        runtime_component_applications_with_validations.receipt_lines();
    assert!(runtime_component_application_receipt_lines_with_validations
        .iter()
        .any(|line| line == "application_count=10"));
    assert!(runtime_component_application_receipt_lines_with_validations
        .iter()
        .any(|line| line == "deferred_pending_component_request_count=0"));
    assert!(runtime_component_application_receipt_lines_with_validations
        .iter()
        .any(|line| line == "live_aql_proof_validation_pending_count=0"));
    assert!(runtime_component_application_receipt_lines_with_validations
        .iter()
        .any(|line| line
            == &format!(
                "applications.{queue_application_index}.live_aql_proof_validation_pending_count=0"
            )));
    assert!(runtime_component_application_receipt_lines_with_validations
        .iter()
        .any(|line| line
            == &format!(
                "applications.{queue_application_index}.runtime_application_receipt_present=false"
            )));
    runtime_component_applications_with_validations.assert_consistent()?;
    assert!(runtime_component_applications_with_validations.is_non_submitting_boundary());
    runtime_component_applications_with_validations.assert_non_submitting_boundary()?;
    let mut stale_count_runtime_component_applications_with_validations =
        runtime_component_applications_with_validations.clone();
    stale_count_runtime_component_applications_with_validations.application_count = 0;
    assert!(
        !stale_count_runtime_component_applications_with_validations.is_non_submitting_boundary()
    );
    let stale_count_runtime_component_applications_with_validations_err =
        stale_count_runtime_component_applications_with_validations
            .assert_non_submitting_boundary()
            .unwrap_err()
            .to_string();
    assert!(
        stale_count_runtime_component_applications_with_validations_err.contains("consistency")
    );
    assert!(
        stale_count_runtime_component_applications_with_validations_err
            .contains("application count")
    );
    macro_rules! assert_runtime_component_application_non_submitting_rejected {
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
    let mut submitting_count_runtime_component_applications =
        runtime_component_applications_with_validations.clone();
    submitting_count_runtime_component_applications.live_aql_submitting_application_count = 1;
    assert_runtime_component_application_non_submitting_rejected!(
        submitting_count_runtime_component_applications,
        "live AQL submitting applications 1 != 0",
    );
    let mut submitting_row_runtime_component_applications =
        runtime_component_applications_with_validations.clone();
    submitting_row_runtime_component_applications.applications[queue_application_index]
        .live_aql_submits_work = true;
    assert_runtime_component_application_non_submitting_rejected!(
        submitting_row_runtime_component_applications,
        "live AQL submitting application rows queue_reservation_request_plan",
    );
    let mut queue_mutating_count_runtime_component_applications =
        runtime_component_applications_with_validations.clone();
    queue_mutating_count_runtime_component_applications.live_queue_mutating_application_count = 1;
    assert_runtime_component_application_non_submitting_rejected!(
        queue_mutating_count_runtime_component_applications,
        "live queue mutating applications 1 != 0",
    );
    let mut queue_mutating_row_runtime_component_applications =
        runtime_component_applications_with_validations.clone();
    queue_mutating_row_runtime_component_applications.applications[queue_application_index]
        .mutates_live_queue = true;
    assert_runtime_component_application_non_submitting_rejected!(
        queue_mutating_row_runtime_component_applications,
        "live queue mutating application rows queue_reservation_request_plan",
    );
    let mut stale_runtime_component_application_receipt =
        runtime_component_applications_with_validations.clone();
    stale_runtime_component_application_receipt.applications[queue_application_index]
        .runtime_application_receipt_present = true;
    assert!(stale_runtime_component_application_receipt
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("runtime receipt"));
    let runtime_component_application_receipts = runtime_component_applications_with_validations
        .applications
        .iter()
        .map(
            |application| RuntimeLaunchRuntimeRequestComponentApplicationReceipt {
                target: runtime_component_applications_with_validations.target,
                code_object_target: runtime_component_applications_with_validations
                    .code_object_target
                    .clone(),
                code_object_sha256: runtime_component_applications_with_validations
                    .code_object_sha256
                    .clone(),
                dispatch_count: runtime_component_applications_with_validations.dispatch_count,
                window_count: runtime_component_applications_with_validations.window_count,
                step: application.step,
                step_index: application.step_index,
                request_plan: application.request_plan,
                requirement: application.requirement,
                receipt_source: "public_contract_external_runtime",
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
    let queue_receipt_index = runtime_component_application_receipts
        .iter()
        .position(|receipt| receipt.request_plan == "queue_reservation_request_plan")
        .unwrap();
    let queue_runtime_component_receipt =
        &runtime_component_application_receipts[queue_receipt_index];
    assert_eq!(
        queue_runtime_component_receipt.step,
        RuntimeLaunchExecutionRequestStep::QueueReservation
    );
    assert_eq!(
        queue_runtime_component_receipt.requested_count,
        queue_component.pending_count
    );
    assert_eq!(
        queue_runtime_component_receipt.applied_count,
        queue_component.pending_count
    );
    assert_eq!(queue_runtime_component_receipt.pending_count, 0);
    assert!(!queue_runtime_component_receipt.live_aql_submits_work);
    assert!(!queue_runtime_component_receipt.mutates_live_queue);
    let queue_runtime_component_receipt_lines = queue_runtime_component_receipt.receipt_lines();
    assert_eq!(
        queue_runtime_component_receipt_lines[0],
        "receipt.kind=runtime_launch_runtime_request_component_application_receipt"
    );
    assert_eq!(
        queue_runtime_component_receipt_lines[1],
        "receipt.version=1"
    );
    assert!(queue_runtime_component_receipt_lines
        .iter()
        .any(|line| line == "request_plan=queue_reservation_request_plan"));
    assert!(queue_runtime_component_receipt
        .receipt_text()
        .ends_with('\n'));
    let queue_runtime_component_receipt_fingerprint =
        queue_runtime_component_receipt.receipt_fingerprint();
    assert_eq!(queue_runtime_component_receipt_fingerprint.len(), 64);
    assert!(queue_runtime_component_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_runtime_component_receipt = queue_runtime_component_receipt.clone();
    stale_runtime_component_receipt.pending_count += 1;
    assert_ne!(
        stale_runtime_component_receipt.receipt_fingerprint(),
        queue_runtime_component_receipt_fingerprint
    );
    assert!(stale_runtime_component_receipt
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("pending count"));
    assert!(!stale_runtime_component_receipt.is_non_submitting_boundary());
    let stale_runtime_component_receipt_boundary_err = stale_runtime_component_receipt
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string();
    assert!(stale_runtime_component_receipt_boundary_err.contains("consistency"));
    assert!(stale_runtime_component_receipt_boundary_err.contains("pending count"));
    let mut submitting_runtime_component_receipt = queue_runtime_component_receipt.clone();
    submitting_runtime_component_receipt.live_aql_submits_work = true;
    assert!(!submitting_runtime_component_receipt.is_non_submitting_boundary());
    assert!(submitting_runtime_component_receipt
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string()
        .contains("live AQL submission side effect is true"));
    let mut queue_mutating_runtime_component_receipt = queue_runtime_component_receipt.clone();
    queue_mutating_runtime_component_receipt.mutates_live_queue = true;
    assert!(!queue_mutating_runtime_component_receipt.is_non_submitting_boundary());
    assert!(queue_mutating_runtime_component_receipt
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string()
        .contains("live queue mutation is true"));

    let runtime_component_application_receipt_plan =
        runtime_component_applications_with_validations
            .application_receipt_plan(&runtime_component_application_receipts)?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(
                "external",
                &[batch_validation, materialized_validation],
                &runtime_component_application_receipts,
            )?,
        runtime_component_application_receipt_plan
    );
    assert_eq!(
        runtime_component_application_receipt_plan.application_count,
        runtime_component_applications_with_validations.application_count
    );
    assert_eq!(
        runtime_component_application_receipt_plan.receipt_input_count,
        runtime_component_applications_with_validations.application_count
    );
    assert_eq!(
        runtime_component_application_receipt_plan.receipt_present_count,
        runtime_component_applications_with_validations.application_count
    );
    assert_eq!(
        runtime_component_application_receipt_plan.application_applied_count,
        runtime_component_applications_with_validations.application_count
    );
    assert_eq!(
        runtime_component_application_receipt_plan.application_pending_count,
        0
    );
    assert_eq!(
        runtime_component_application_receipt_plan.missing_receipt_count,
        0
    );
    assert_eq!(
        runtime_component_application_receipt_plan.unexpected_receipt_count,
        0
    );
    assert_eq!(
        runtime_component_application_receipt_plan.rejected_receipt_count,
        0
    );
    assert_eq!(
        runtime_component_application_receipt_plan.live_aql_submitting_receipt_count,
        0
    );
    assert_eq!(
        runtime_component_application_receipt_plan.live_queue_mutating_receipt_count,
        0
    );
    assert!(runtime_component_application_receipt_plan.all_receipts_present);
    assert!(runtime_component_application_receipt_plan.all_applications_applied);
    assert!(runtime_component_application_receipt_plan.no_live_aql_submission_side_effects);
    assert!(runtime_component_application_receipt_plan.no_live_queue_mutation);
    assert!(!runtime_component_application_receipt_plan.all_components_applied);
    assert!(!runtime_component_application_receipt_plan.submission_ready);
    assert_eq!(
        runtime_component_application_receipt_plan
            .applied_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    assert!(runtime_component_application_receipt_plan
        .pending_request_plan_names()
        .is_empty());
    let queue_receipt_application = runtime_component_application_receipt_plan
        .application_for("queue_reservation_request_plan")
        .unwrap();
    let queue_receipt_plan_application_index = runtime_component_application_receipt_plan
        .applications
        .iter()
        .position(|application| application.request_plan == "queue_reservation_request_plan")
        .unwrap();
    assert_eq!(
        runtime_component_application_receipt_plan
            .application_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
            .unwrap()
            .request_plan,
        queue_receipt_application.request_plan
    );
    assert!(queue_receipt_application.receipt_present);
    assert_eq!(
        queue_receipt_application.receipt_source,
        "public_contract_external_runtime"
    );
    assert!(queue_receipt_application.receipt_matches_launch);
    assert!(queue_receipt_application.receipt_matches_application);
    assert_eq!(
        queue_receipt_application.receipt_fingerprint,
        queue_runtime_component_receipt_fingerprint
    );
    assert_eq!(
        queue_receipt_application.expected_application_count,
        queue_component.pending_count
    );
    assert_eq!(
        queue_receipt_application.receipt_requested_count,
        queue_component.pending_count
    );
    assert_eq!(
        queue_receipt_application.receipt_applied_count,
        queue_component.pending_count
    );
    assert_eq!(queue_receipt_application.receipt_pending_count, 0);
    assert!(queue_receipt_application.application_applied);
    assert_eq!(queue_receipt_application.application_pending_count, 0);
    assert_eq!(queue_receipt_application.rejection_reason, None);
    let runtime_component_application_receipt_plan_lines =
        runtime_component_application_receipt_plan.receipt_lines();
    assert_eq!(
        runtime_component_application_receipt_plan_lines[0],
        "receipt.kind=model_runtime_launch_runtime_request_component_application_receipt_plan"
    );
    assert_eq!(
        runtime_component_application_receipt_plan_lines[1],
        "receipt.version=1"
    );
    assert!(runtime_component_application_receipt_plan_lines
        .iter()
        .any(|line| line == "all_applications_applied=true"));
    assert!(runtime_component_application_receipt_plan_lines
        .iter()
        .any(|line| line == "all_components_applied=false"));
    assert!(runtime_component_application_receipt_plan_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(runtime_component_application_receipt_plan
        .receipt_text()
        .ends_with('\n'));
    let runtime_component_application_receipt_plan_fingerprint =
        runtime_component_application_receipt_plan.receipt_fingerprint();
    assert_eq!(
        runtime_component_application_receipt_plan_fingerprint.len(),
        64
    );
    assert!(runtime_component_application_receipt_plan_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    stale_runtime_component_application_receipt_plan.application_count += 1;
    assert_ne!(
        stale_runtime_component_application_receipt_plan.receipt_fingerprint(),
        runtime_component_application_receipt_plan_fingerprint
    );
    assert!(stale_runtime_component_application_receipt_plan
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("application count"));
    runtime_component_application_receipt_plan.assert_consistent()?;
    assert!(runtime_component_application_receipt_plan.is_non_submitting_boundary());
    runtime_component_application_receipt_plan.assert_non_submitting_boundary()?;
    let mut stale_count_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    stale_count_runtime_component_application_receipt_plan.application_count = 0;
    assert!(!stale_count_runtime_component_application_receipt_plan.is_non_submitting_boundary());
    let stale_count_runtime_component_application_receipt_plan_err =
        stale_count_runtime_component_application_receipt_plan
            .assert_non_submitting_boundary()
            .unwrap_err()
            .to_string();
    assert!(stale_count_runtime_component_application_receipt_plan_err.contains("consistency"));
    assert!(
        stale_count_runtime_component_application_receipt_plan_err.contains("application count")
    );
    macro_rules! assert_runtime_component_receipt_non_submitting_rejected {
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
    let mut side_effect_guard_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    side_effect_guard_runtime_component_application_receipt_plan
        .no_live_aql_submission_side_effects = false;
    assert_runtime_component_receipt_non_submitting_rejected!(
        side_effect_guard_runtime_component_application_receipt_plan,
        "live AQL submission side-effect guard is false",
    );
    let mut submitting_count_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    submitting_count_runtime_component_application_receipt_plan.live_aql_submitting_receipt_count =
        1;
    assert_runtime_component_receipt_non_submitting_rejected!(
        submitting_count_runtime_component_application_receipt_plan,
        "live AQL submitting receipts 1 != 0",
    );
    let mut submitting_row_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    submitting_row_runtime_component_application_receipt_plan.applications
        [queue_receipt_plan_application_index]
        .live_aql_submits_work = true;
    assert_runtime_component_receipt_non_submitting_rejected!(
        submitting_row_runtime_component_application_receipt_plan,
        "live AQL submitting receipt rows queue_reservation_request_plan",
    );
    let mut queue_mutation_guard_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    queue_mutation_guard_runtime_component_application_receipt_plan.no_live_queue_mutation = false;
    assert_runtime_component_receipt_non_submitting_rejected!(
        queue_mutation_guard_runtime_component_application_receipt_plan,
        "live queue mutation guard is false",
    );
    let mut queue_mutating_count_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    queue_mutating_count_runtime_component_application_receipt_plan
        .live_queue_mutating_receipt_count = 1;
    assert_runtime_component_receipt_non_submitting_rejected!(
        queue_mutating_count_runtime_component_application_receipt_plan,
        "live queue mutating receipts 1 != 0",
    );
    let mut queue_mutating_row_runtime_component_application_receipt_plan =
        runtime_component_application_receipt_plan.clone();
    queue_mutating_row_runtime_component_application_receipt_plan.applications
        [queue_receipt_plan_application_index]
        .mutates_live_queue = true;
    assert_runtime_component_receipt_non_submitting_rejected!(
        queue_mutating_row_runtime_component_application_receipt_plan,
        "live queue mutating receipt rows queue_reservation_request_plan",
    );
    let mut submitting_runtime_component_application_receipts =
        runtime_component_application_receipts.clone();
    submitting_runtime_component_application_receipts[queue_receipt_plan_application_index]
        .live_aql_submits_work = true;
    let submitting_runtime_component_receipt_plan_err = runtime_component_applications
        .application_receipt_plan(&submitting_runtime_component_application_receipts)
        .unwrap_err()
        .to_string();
    assert!(submitting_runtime_component_receipt_plan_err.contains("not a non-submitting boundary"));
    assert!(submitting_runtime_component_receipt_plan_err
        .contains("live AQL submission side effect is true"));
    let mut queue_mutating_runtime_component_overlay_plan =
        runtime_component_application_receipt_plan.clone();
    queue_mutating_runtime_component_overlay_plan.applications
        [queue_receipt_plan_application_index]
        .mutates_live_queue = true;
    let queue_mutating_runtime_component_overlay_err = submission_prerequisites_with_validations
        .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
            &queue_mutating_runtime_component_overlay_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(queue_mutating_runtime_component_overlay_err.contains("not a non-submitting boundary"));
    assert!(queue_mutating_runtime_component_overlay_err
        .contains("live queue mutating receipt rows queue_reservation_request_plan"));
    assert_eq!(submission_prerequisites.submission_gate()?, submission_gate);
    assert_eq!(
        submission_prerequisites_with_validations.submission_gate()?,
        submission_gate_with_validations
    );
    let submission_prerequisites_after_runtime_component_receipts =
        submission_prerequisites_with_validations
            .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
                &runtime_component_application_receipt_plan,
            )?;
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts.prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts.pending_component_request_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts
            .runtime_request_component_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts
            .live_aql_proof_validation_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_next_action_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts.next_action_count,
        execution_requests.runtime_request_plan_count
    );
    assert!(submission_prerequisites_after_runtime_component_receipts.request_plan_ready);
    assert!(!submission_prerequisites_after_runtime_component_receipts.execution_readiness_ready);
    assert!(!submission_prerequisites_after_runtime_component_receipts.all_prerequisites_satisfied);
    assert!(!submission_prerequisites_after_runtime_component_receipts.submission_ready);
    assert!(submission_prerequisites_after_runtime_component_receipts
        .runtime_request_component_next_action_request_plan_names()
        .is_empty());
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_next_action_request_plan_names()
            .as_slice(),
        &expected_execution_request_plans
    );
    let next_action_labels_after_runtime_component_receipts =
        submission_prerequisites_after_runtime_component_receipts.next_action_labels();
    assert_eq!(
        next_action_labels_after_runtime_component_receipts.len(),
        execution_requests.runtime_request_plan_count
    );
    assert!(next_action_labels_after_runtime_component_receipts
        .iter()
        .all(|label| *label == "resolve_execution_readiness_blocker"));
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts
            .next_action_input_labels()
            .as_slice(),
        &[
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
    let queue_prerequisite_after_runtime_component_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .prerequisite_for("queue_reservation_request_plan")
            .unwrap();
    assert_eq!(
        queue_prerequisite_after_runtime_component_receipts.applied_count,
        queue_prerequisite_after_runtime_component_receipts.request_count
    );
    assert_eq!(
        queue_prerequisite_after_runtime_component_receipts.pending_count,
        0
    );
    assert_eq!(
        queue_prerequisite_after_runtime_component_receipts.next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ResolveExecutionReadinessBlocker
    );
    assert_eq!(
        queue_prerequisite_after_runtime_component_receipts.next_action_input,
        "queue_reservation"
    );
    assert_eq!(
        queue_prerequisite_after_runtime_component_receipts.next_action_pending_count,
        1
    );
    assert!(!queue_prerequisite_after_runtime_component_receipts.next_action_uses_live_aql_proof);
    assert_eq!(
        queue_prerequisite_after_runtime_component_receipts.next_action_live_aql_proof_kind,
        None
    );
    let runtime_component_applications_after_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .runtime_request_component_application_plan()?;
    assert_eq!(
        runtime_component_applications_after_receipts.application_count,
        0
    );
    assert!(!runtime_component_applications_after_receipts.all_application_requests_ready);
    assert!(!runtime_component_applications_after_receipts.all_components_applied);
    assert!(!runtime_component_applications_after_receipts.submission_ready);
    let submission_gate_after_runtime_component_receipts =
        submission_prerequisites_with_validations
            .submission_gate_with_runtime_request_component_application_receipt_plan(
                &runtime_component_application_receipt_plan,
            )?;
    assert_eq!(
        submission_prerequisites_after_runtime_component_receipts.submission_gate()?,
        submission_gate_after_runtime_component_receipts
    );
    assert!(submission_gate_after_runtime_component_receipts.request_plan_ready);
    assert!(!submission_gate_after_runtime_component_receipts.execution_readiness_ready);
    assert!(submission_gate_after_runtime_component_receipts.all_components_applied);
    assert!(
        submission_gate_after_runtime_component_receipts.all_live_aql_proof_validations_applied
    );
    assert!(submission_gate_after_runtime_component_receipts.no_live_aql_submission_side_effects);
    assert!(submission_gate_after_runtime_component_receipts.no_live_queue_mutation);
    assert_eq!(
        submission_gate_after_runtime_component_receipts.component_pending_count,
        0
    );
    assert_eq!(
        submission_gate_after_runtime_component_receipts.live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        submission_gate_after_runtime_component_receipts.submission_blocker_count,
        execution_readiness.blockers.len()
    );
    assert!(!submission_gate_after_runtime_component_receipts.submission_ready);
    assert!(submission_gate_after_runtime_component_receipts.has_blocker("queue_reservation"));
    assert!(
        !submission_gate_after_runtime_component_receipts.has_blocker("runtime_request_components")
    );
    assert!(
        !submission_gate_after_runtime_component_receipts.has_blocker("live_aql_proof_validation")
    );
    assert_eq!(
        submission_gate_after_runtime_component_receipts
            .blocker_requirement_names()
            .as_slice(),
        &expected_submission_blockers_with_validations[..execution_readiness.blockers.len()]
    );
    let submission_blockers_after_runtime_component_receipts =
        submission_gate_after_runtime_component_receipts.blocker_report()?;
    assert_eq!(
        submission_blockers_after_runtime_component_receipts
            .runtime_request_component_pending_count,
        0
    );
    assert_eq!(
        submission_blockers_after_runtime_component_receipts
            .live_aql_proof_validation_pending_count,
        0
    );
    assert_eq!(
        submission_blockers_after_runtime_component_receipts.total_pending_count,
        0
    );
    assert!(submission_blockers_after_runtime_component_receipts
        .runtime_request_component_blocker_requirement_names()
        .is_empty());
    assert!(submission_blockers_after_runtime_component_receipts
        .live_aql_proof_validation_blocker_requirement_names()
        .is_empty());
    assert!(!submission_blockers_after_runtime_component_receipts.submission_ready);
    submission_gate_after_runtime_component_receipts.assert_consistent()?;
    submission_prerequisites_after_runtime_component_receipts.assert_consistent()?;
    let execution_readiness_resolutions_before_component_receipts =
        submission_prerequisites.execution_readiness_blocker_resolution_plan()?;
    assert_eq!(
        execution_readiness_resolutions_before_component_receipts
            .execution_readiness_next_action_count,
        0
    );
    assert_eq!(
        execution_readiness_resolutions_before_component_receipts.blocker_resolution_count,
        0
    );
    assert_eq!(
        execution_readiness_resolutions_before_component_receipts.source_prerequisite_count,
        0
    );
    assert!(
        !execution_readiness_resolutions_before_component_receipts.all_blocker_resolutions_ready
    );
    assert!(!execution_readiness_resolutions_before_component_receipts.submission_ready);
    assert!(execution_readiness_resolutions_before_component_receipts
        .resolution_requirement_names()
        .is_empty());
    assert!(execution_readiness_resolutions_before_component_receipts
        .resolution_receipt_requirement_names()
        .is_empty());
    execution_readiness_resolutions_before_component_receipts.assert_consistent()?;
    assert!(execution_readiness_resolutions_before_component_receipts.is_non_submitting_boundary());
    execution_readiness_resolutions_before_component_receipts.assert_non_submitting_boundary()?;

    let execution_readiness_resolutions_after_component_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_blocker_resolution_plan()?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(
                "external",
                &[batch_validation, materialized_validation],
                &runtime_component_application_receipts,
            )?,
        execution_readiness_resolutions_after_component_receipts
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts.prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts
            .execution_readiness_next_action_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts.blocker_resolution_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts.source_prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts.resolution_ready_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts.resolution_receipt_count,
        0
    );
    assert!(execution_readiness_resolutions_after_component_receipts.request_plan_ready);
    assert!(!execution_readiness_resolutions_after_component_receipts.execution_readiness_ready);
    assert!(execution_readiness_resolutions_after_component_receipts.all_blocker_resolutions_ready);
    assert!(!execution_readiness_resolutions_after_component_receipts.submission_ready);
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts
            .resolution_requirement_names()
            .as_slice(),
        &expected_submission_blockers_with_validations[..execution_readiness.blockers.len()]
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts
            .source_request_plan_names()
            .as_slice(),
        &[
            "kernel_candidate_selection_request_plan",
            "host_launcher_branch_resolution_request_plan",
            "code_object_load_request_plan",
            "code_object_base_binding_request_plan",
            "kernarg_allocation_request_plan",
            "kernel_argument_abi_schema_request_plan",
            "kernel_argument_abi_semantic_projection_candidate_selection_request_plan",
            "completion_signal_binding_request_plan",
            "queue_reservation_request_plan",
            "aql_live_relocation_binding_request_plan",
        ]
    );
    assert!(execution_readiness_resolutions_after_component_receipts
        .resolution_receipt_requirement_names()
        .is_empty());
    assert!(execution_readiness_resolutions_after_component_receipts.is_non_submitting_boundary());
    execution_readiness_resolutions_after_component_receipts.assert_non_submitting_boundary()?;
    let loaded_code_object_resolution = execution_readiness_resolutions_after_component_receipts
        .resolution_for("loaded_code_object_base")
        .unwrap();
    assert_eq!(loaded_code_object_resolution.blocker_index, 2);
    assert_eq!(loaded_code_object_resolution.source_prerequisite_count, 2);
    assert_eq!(
        loaded_code_object_resolution
            .source_request_plans
            .as_slice(),
        &[
            "code_object_load_request_plan",
            "code_object_base_binding_request_plan"
        ]
    );
    assert_eq!(
        loaded_code_object_resolution.source_step_indices.as_slice(),
        &[0, 1]
    );
    assert_eq!(
        loaded_code_object_resolution.next_action,
        RuntimeLaunchSubmissionPrerequisiteNextAction::ResolveExecutionReadinessBlocker
    );
    assert_eq!(
        loaded_code_object_resolution.next_action_input,
        "loaded_code_object_base"
    );
    assert_eq!(loaded_code_object_resolution.next_action_pending_count, 2);
    assert!(loaded_code_object_resolution.resolution_ready);
    assert!(!loaded_code_object_resolution.resolution_receipt_present);
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts
            .resolution_for_step(RuntimeLaunchExecutionRequestStep::CodeObjectLoad)
            .unwrap()
            .requirement,
        "loaded_code_object_base"
    );
    assert_eq!(
        execution_readiness_resolutions_after_component_receipts
            .resolution_for_step(RuntimeLaunchExecutionRequestStep::CodeObjectBaseBinding)
            .unwrap()
            .requirement,
        "loaded_code_object_base"
    );
    assert!(execution_readiness_resolutions_after_component_receipts
        .resolution_for("runtime_request_components")
        .is_none());
    let execution_readiness_resolution_receipt_lines =
        execution_readiness_resolutions_after_component_receipts.receipt_lines();
    assert_eq!(
        execution_readiness_resolution_receipt_lines[0],
        "receipt.kind=model_runtime_launch_execution_readiness_blocker_resolution_plan"
    );
    assert_eq!(
        execution_readiness_resolution_receipt_lines[1],
        "receipt.version=1"
    );
    assert!(execution_readiness_resolution_receipt_lines
        .iter()
        .any(|line| line == "blocker_resolution_count=9"));
    assert!(execution_readiness_resolution_receipt_lines
        .iter()
        .any(|line| line == "source_prerequisite_count=10"));
    assert!(execution_readiness_resolution_receipt_lines
        .iter()
        .any(|line| line == "resolution_receipt_count=0"));
    assert!(execution_readiness_resolution_receipt_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(execution_readiness_resolutions_after_component_receipts
        .receipt_text()
        .ends_with('\n'));
    let execution_readiness_resolution_receipt_fingerprint =
        execution_readiness_resolutions_after_component_receipts.receipt_fingerprint();
    assert_eq!(execution_readiness_resolution_receipt_fingerprint.len(), 64);
    assert!(execution_readiness_resolution_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_execution_readiness_resolution_receipt =
        execution_readiness_resolutions_after_component_receipts.clone();
    stale_execution_readiness_resolution_receipt.blocker_resolution_count += 1;
    assert_ne!(
        stale_execution_readiness_resolution_receipt.receipt_fingerprint(),
        execution_readiness_resolution_receipt_fingerprint
    );
    assert!(stale_execution_readiness_resolution_receipt
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("blocker resolution count"));
    assert!(!stale_execution_readiness_resolution_receipt.is_non_submitting_boundary());
    let stale_execution_readiness_resolution_boundary_err =
        stale_execution_readiness_resolution_receipt
            .assert_non_submitting_boundary()
            .unwrap_err()
            .to_string();
    assert!(stale_execution_readiness_resolution_boundary_err.contains("consistency"));
    assert!(stale_execution_readiness_resolution_boundary_err.contains("blocker resolution count"));
    macro_rules! assert_execution_readiness_resolution_plan_non_submitting_rejected {
        ($plan:expr, $expected:expr) => {{
            let plan = $plan;
            assert!(!plan.is_non_submitting_boundary());
            let err = plan
                .assert_non_submitting_boundary()
                .unwrap_err()
                .to_string();
            assert!(
                err.contains($expected),
                "expected {:?} in {:?}",
                $expected,
                err
            );
        }};
    }
    let mut receipt_count_execution_readiness_resolution_plan =
        execution_readiness_resolutions_after_component_receipts.clone();
    receipt_count_execution_readiness_resolution_plan.resolution_receipt_count = 1;
    assert_execution_readiness_resolution_plan_non_submitting_rejected!(
        receipt_count_execution_readiness_resolution_plan,
        "resolution receipts 1 != 0"
    );
    let mut stale_execution_readiness_resolution_state =
        execution_readiness_resolutions_after_component_receipts.clone();
    stale_execution_readiness_resolution_state.resolutions[0].resolution_receipt_present = true;
    assert!(stale_execution_readiness_resolution_state
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("resolution receipt"));
    assert_eq!(
        stale_execution_readiness_resolution_state
            .resolution_receipt_requirement_names()
            .as_slice(),
        &["kernel_candidate_selection_policy"]
    );
    assert_execution_readiness_resolution_plan_non_submitting_rejected!(
        stale_execution_readiness_resolution_state,
        "resolution receipt rows kernel_candidate_selection_policy"
    );
    let mut ready_execution_readiness_resolution_plan =
        execution_readiness_resolutions_after_component_receipts.clone();
    ready_execution_readiness_resolution_plan.execution_readiness_ready = true;
    assert_execution_readiness_resolution_plan_non_submitting_rejected!(
        ready_execution_readiness_resolution_plan,
        "execution readiness guard is true"
    );
    let mut submission_ready_execution_readiness_resolution_plan =
        execution_readiness_resolutions_after_component_receipts.clone();
    submission_ready_execution_readiness_resolution_plan.submission_ready = true;
    assert_execution_readiness_resolution_plan_non_submitting_rejected!(
        submission_ready_execution_readiness_resolution_plan,
        "submission ready guard is true"
    );
    execution_readiness_resolutions_after_component_receipts.assert_consistent()?;

    let execution_readiness_blocker_resolution_receipts =
        execution_readiness_resolutions_after_component_receipts
            .resolutions
            .iter()
            .map(
                |resolution| RuntimeLaunchExecutionReadinessBlockerResolutionReceipt {
                    target: execution_readiness_resolutions_after_component_receipts.target,
                    code_object_target: execution_readiness_resolutions_after_component_receipts
                        .code_object_target
                        .clone(),
                    code_object_sha256: execution_readiness_resolutions_after_component_receipts
                        .code_object_sha256
                        .clone(),
                    dispatch_count: execution_readiness_resolutions_after_component_receipts
                        .dispatch_count,
                    window_count: execution_readiness_resolutions_after_component_receipts
                        .window_count,
                    requirement: resolution.requirement,
                    receipt_source: "public_contract_external_runtime",
                    source_prerequisite_count: resolution.source_prerequisite_count,
                    resolved_count: resolution.source_prerequisite_count,
                    pending_count: 0,
                    live_aql_submits_work: false,
                    mutates_live_queue: false,
                },
            )
            .collect::<Vec<_>>();
    for receipt in &execution_readiness_blocker_resolution_receipts {
        let receipt_lines = receipt.receipt_lines();
        assert_eq!(
            receipt_lines[0],
            "receipt.kind=runtime_launch_execution_readiness_blocker_resolution_receipt"
        );
        assert_eq!(receipt_lines[1], "receipt.version=1");
        assert!(receipt_lines
            .iter()
            .any(|line| line == "receipt_source=public_contract_external_runtime"));
        assert!(receipt.receipt_text().ends_with('\n'));
        let receipt_fingerprint = receipt.receipt_fingerprint();
        assert_eq!(receipt_fingerprint.len(), 64);
        assert!(receipt_fingerprint.chars().all(|ch| ch.is_ascii_hexdigit()));
        receipt.assert_consistent()?;
        assert!(receipt.is_non_submitting_boundary());
        receipt.assert_non_submitting_boundary()?;
    }
    let mut stale_execution_readiness_blocker_resolution_receipt =
        execution_readiness_blocker_resolution_receipts[0].clone();
    stale_execution_readiness_blocker_resolution_receipt.pending_count += 1;
    assert_ne!(
        stale_execution_readiness_blocker_resolution_receipt.receipt_fingerprint(),
        execution_readiness_blocker_resolution_receipts[0].receipt_fingerprint()
    );
    assert!(stale_execution_readiness_blocker_resolution_receipt
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("pending count"));
    let mut submitting_execution_readiness_blocker_resolution_receipt =
        execution_readiness_blocker_resolution_receipts[0].clone();
    submitting_execution_readiness_blocker_resolution_receipt.live_aql_submits_work = true;
    assert!(!submitting_execution_readiness_blocker_resolution_receipt.is_non_submitting_boundary());
    assert!(submitting_execution_readiness_blocker_resolution_receipt
        .assert_non_submitting_boundary()
        .unwrap_err()
        .to_string()
        .contains("live AQL submission side effect is true"));
    let mut queue_mutating_execution_readiness_blocker_resolution_receipt =
        execution_readiness_blocker_resolution_receipts[0].clone();
    queue_mutating_execution_readiness_blocker_resolution_receipt.mutates_live_queue = true;
    assert!(
        !queue_mutating_execution_readiness_blocker_resolution_receipt.is_non_submitting_boundary()
    );
    assert!(
        queue_mutating_execution_readiness_blocker_resolution_receipt
            .assert_non_submitting_boundary()
            .unwrap_err()
            .to_string()
            .contains("live queue mutation is true")
    );
    let mut submitting_execution_readiness_blocker_resolution_receipts =
        execution_readiness_blocker_resolution_receipts.clone();
    submitting_execution_readiness_blocker_resolution_receipts[0].live_aql_submits_work = true;
    let submitting_execution_readiness_receipt_plan_err =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&submitting_execution_readiness_blocker_resolution_receipts)
            .unwrap_err()
            .to_string();
    assert!(
        submitting_execution_readiness_receipt_plan_err.contains("not a non-submitting boundary")
    );
    assert!(submitting_execution_readiness_receipt_plan_err
        .contains("live AQL submission side effect is true"));

    let execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&execution_readiness_blocker_resolution_receipts)?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[batch_validation, materialized_validation],
                &runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )?,
        execution_readiness_blocker_resolution_receipt_plan
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.blocker_resolution_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.receipt_input_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.receipt_present_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.resolution_applied_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.resolution_pending_count,
        0
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.missing_receipt_count,
        0
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.unexpected_receipt_count,
        0
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.rejected_receipt_count,
        0
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.live_aql_submitting_receipt_count,
        0
    );
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan.live_queue_mutating_receipt_count,
        0
    );
    assert!(execution_readiness_blocker_resolution_receipt_plan.all_receipts_present);
    assert!(execution_readiness_blocker_resolution_receipt_plan.all_resolutions_applied);
    assert!(
        execution_readiness_blocker_resolution_receipt_plan.no_live_aql_submission_side_effects
    );
    assert!(execution_readiness_blocker_resolution_receipt_plan.no_live_queue_mutation);
    assert!(!execution_readiness_blocker_resolution_receipt_plan.execution_readiness_ready);
    assert!(!execution_readiness_blocker_resolution_receipt_plan.submission_ready);
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan
            .applied_requirement_names()
            .as_slice(),
        &expected_submission_blockers_with_validations[..execution_readiness.blockers.len()]
    );
    assert!(execution_readiness_blocker_resolution_receipt_plan
        .pending_requirement_names()
        .is_empty());
    assert!(execution_readiness_blocker_resolution_receipt_plan
        .live_aql_submitting_receipt_requirement_names()
        .is_empty());
    assert!(execution_readiness_blocker_resolution_receipt_plan
        .live_queue_mutating_receipt_requirement_names()
        .is_empty());
    assert!(execution_readiness_blocker_resolution_receipt_plan.is_non_submitting_boundary());
    execution_readiness_blocker_resolution_receipt_plan.assert_non_submitting_boundary()?;
    let loaded_code_object_resolution_receipt = execution_readiness_blocker_resolution_receipts
        .iter()
        .find(|receipt| receipt.requirement == "loaded_code_object_base")
        .unwrap();
    let loaded_code_object_receipt_resolution = execution_readiness_blocker_resolution_receipt_plan
        .resolution_for("loaded_code_object_base")
        .unwrap();
    assert_eq!(
        loaded_code_object_receipt_resolution.expected_source_prerequisite_count,
        2
    );
    assert_eq!(
        loaded_code_object_receipt_resolution.receipt_resolved_count,
        2
    );
    assert_eq!(
        loaded_code_object_receipt_resolution.receipt_fingerprint,
        loaded_code_object_resolution_receipt.receipt_fingerprint()
    );
    assert!(loaded_code_object_receipt_resolution.resolution_applied);
    let execution_readiness_blocker_resolution_receipt_plan_lines =
        execution_readiness_blocker_resolution_receipt_plan.receipt_lines();
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan_lines[0],
        "receipt.kind=model_runtime_launch_execution_readiness_blocker_resolution_receipt_plan"
    );
    assert!(execution_readiness_blocker_resolution_receipt_plan_lines
        .iter()
        .any(|line| line == "all_resolutions_applied=true"));
    assert!(execution_readiness_blocker_resolution_receipt_plan_lines
        .iter()
        .any(|line| line == "execution_readiness_ready=false"));
    assert!(execution_readiness_blocker_resolution_receipt_plan_lines
        .iter()
        .any(|line| line == "submission_ready=false"));
    assert!(execution_readiness_blocker_resolution_receipt_plan
        .receipt_text()
        .ends_with('\n'));
    let execution_readiness_blocker_resolution_receipt_plan_fingerprint =
        execution_readiness_blocker_resolution_receipt_plan.receipt_fingerprint();
    assert_eq!(
        execution_readiness_blocker_resolution_receipt_plan_fingerprint.len(),
        64
    );
    assert!(
        execution_readiness_blocker_resolution_receipt_plan_fingerprint
            .chars()
            .all(|ch| ch.is_ascii_hexdigit())
    );
    let mut stale_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    stale_execution_readiness_blocker_resolution_receipt_plan.blocker_resolution_count += 1;
    assert_ne!(
        stale_execution_readiness_blocker_resolution_receipt_plan.receipt_fingerprint(),
        execution_readiness_blocker_resolution_receipt_plan_fingerprint
    );
    assert!(stale_execution_readiness_blocker_resolution_receipt_plan
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("blocker resolution count"));
    assert!(!stale_execution_readiness_blocker_resolution_receipt_plan.is_non_submitting_boundary());
    let stale_execution_readiness_blocker_resolution_receipt_plan_boundary_err =
        stale_execution_readiness_blocker_resolution_receipt_plan
            .assert_non_submitting_boundary()
            .unwrap_err()
            .to_string();
    assert!(
        stale_execution_readiness_blocker_resolution_receipt_plan_boundary_err
            .contains("consistency")
    );
    assert!(
        stale_execution_readiness_blocker_resolution_receipt_plan_boundary_err
            .contains("blocker resolution count")
    );
    macro_rules! assert_execution_readiness_resolution_receipt_non_submitting_rejected {
        ($plan:expr, $expected:expr) => {{
            let plan = $plan;
            assert!(!plan.is_non_submitting_boundary());
            let err = plan
                .assert_non_submitting_boundary()
                .unwrap_err()
                .to_string();
            assert!(
                err.contains($expected),
                "expected {:?} in {:?}",
                $expected,
                err
            );
        }};
    }
    let mut submitting_guard_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    submitting_guard_execution_readiness_blocker_resolution_receipt_plan
        .no_live_aql_submission_side_effects = false;
    assert_execution_readiness_resolution_receipt_non_submitting_rejected!(
        submitting_guard_execution_readiness_blocker_resolution_receipt_plan,
        "live AQL submission side-effect guard is false"
    );
    let mut submitting_count_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    submitting_count_execution_readiness_blocker_resolution_receipt_plan
        .live_aql_submitting_receipt_count = 1;
    assert_execution_readiness_resolution_receipt_non_submitting_rejected!(
        submitting_count_execution_readiness_blocker_resolution_receipt_plan,
        "live AQL submitting receipts 1 != 0"
    );
    let mut submitting_row_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    submitting_row_execution_readiness_blocker_resolution_receipt_plan.resolutions[0]
        .live_aql_submits_work = true;
    assert_eq!(
        submitting_row_execution_readiness_blocker_resolution_receipt_plan
            .live_aql_submitting_receipt_requirement_names()
            .as_slice(),
        &["kernel_candidate_selection_policy"]
    );
    assert_execution_readiness_resolution_receipt_non_submitting_rejected!(
        submitting_row_execution_readiness_blocker_resolution_receipt_plan,
        "live AQL submitting receipt rows kernel_candidate_selection_policy"
    );
    let mut queue_guard_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    queue_guard_execution_readiness_blocker_resolution_receipt_plan.no_live_queue_mutation = false;
    assert_execution_readiness_resolution_receipt_non_submitting_rejected!(
        queue_guard_execution_readiness_blocker_resolution_receipt_plan,
        "live queue mutation guard is false"
    );
    let mut queue_count_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    queue_count_execution_readiness_blocker_resolution_receipt_plan
        .live_queue_mutating_receipt_count = 1;
    assert_execution_readiness_resolution_receipt_non_submitting_rejected!(
        queue_count_execution_readiness_blocker_resolution_receipt_plan,
        "live queue mutating receipts 1 != 0"
    );
    let mut queue_row_execution_readiness_blocker_resolution_receipt_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    queue_row_execution_readiness_blocker_resolution_receipt_plan.resolutions[0]
        .mutates_live_queue = true;
    assert_eq!(
        queue_row_execution_readiness_blocker_resolution_receipt_plan
            .live_queue_mutating_receipt_requirement_names()
            .as_slice(),
        &["kernel_candidate_selection_policy"]
    );
    assert_execution_readiness_resolution_receipt_non_submitting_rejected!(
        queue_row_execution_readiness_blocker_resolution_receipt_plan,
        "live queue mutating receipt rows kernel_candidate_selection_policy"
    );
    let mut queue_mutating_execution_readiness_overlay_plan =
        execution_readiness_blocker_resolution_receipt_plan.clone();
    queue_mutating_execution_readiness_overlay_plan.resolutions[0].mutates_live_queue = true;
    let queue_mutating_execution_readiness_overlay_err =
        submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &queue_mutating_execution_readiness_overlay_plan,
            )
            .unwrap_err()
            .to_string();
    assert!(
        queue_mutating_execution_readiness_overlay_err.contains("not a non-submitting boundary")
    );
    assert!(queue_mutating_execution_readiness_overlay_err
        .contains("live queue mutating receipt rows kernel_candidate_selection_policy"));

    let submission_prerequisites_after_execution_readiness_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &execution_readiness_blocker_resolution_receipt_plan,
            )?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[batch_validation, materialized_validation],
                &runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )?,
        submission_prerequisites_after_execution_readiness_receipts
    );
    assert_eq!(
        submission_prerequisites_after_execution_readiness_receipts.satisfied_prerequisite_count,
        execution_requests.runtime_request_plan_count
    );
    assert_eq!(
        submission_prerequisites_after_execution_readiness_receipts.unsatisfied_prerequisite_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_execution_readiness_receipts.next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_next_action_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_execution_readiness_receipts.pending_component_request_count,
        0
    );
    assert_eq!(
        submission_prerequisites_after_execution_readiness_receipts
            .live_aql_proof_validation_pending_count,
        0
    );
    assert!(submission_prerequisites_after_execution_readiness_receipts.execution_readiness_ready);
    assert!(
        submission_prerequisites_after_execution_readiness_receipts.all_prerequisites_satisfied
    );
    assert!(submission_prerequisites_after_execution_readiness_receipts.submission_ready);
    assert!(submission_prerequisites_after_execution_readiness_receipts
        .next_action_request_plan_names()
        .is_empty());
    for request_plan in
        execution_readiness_resolutions_after_component_receipts.source_request_plan_names()
    {
        let prerequisite = submission_prerequisites_after_execution_readiness_receipts
            .prerequisite_for(request_plan)
            .unwrap();
        assert!(prerequisite.prerequisite_satisfied);
        assert!(!prerequisite.blocker_present);
        assert_eq!(
            prerequisite.next_action,
            RuntimeLaunchSubmissionPrerequisiteNextAction::None
        );
    }
    submission_prerequisites_after_execution_readiness_receipts.assert_consistent()?;
    let resolved_submission_prerequisite_receipt_lines =
        submission_prerequisites_after_execution_readiness_receipts.receipt_lines();
    assert_eq!(
        resolved_submission_prerequisite_receipt_lines[0],
        "receipt.kind=model_runtime_launch_submission_prerequisite_plan"
    );
    assert!(submission_prerequisites_after_execution_readiness_receipts
        .receipt_text()
        .ends_with('\n'));
    let resolved_submission_prerequisite_receipt_fingerprint =
        submission_prerequisites_after_execution_readiness_receipts.receipt_fingerprint();
    assert_eq!(
        resolved_submission_prerequisite_receipt_fingerprint.len(),
        64
    );
    assert!(resolved_submission_prerequisite_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_resolved_submission_prerequisite_receipt =
        submission_prerequisites_after_execution_readiness_receipts.clone();
    stale_resolved_submission_prerequisite_receipt.prerequisites[0].pending_count += 1;
    assert_ne!(
        stale_resolved_submission_prerequisite_receipt.receipt_fingerprint(),
        resolved_submission_prerequisite_receipt_fingerprint
    );
    let synthetic_cpu_resolved_submission_prerequisites = execution_requests
        .synthetic_cpu_resolved_submission_prerequisite_plan(
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )?;
    assert_eq!(
        synthetic_cpu_resolved_submission_prerequisites,
        submission_prerequisites_after_execution_readiness_receipts
    );
    let report_synthetic_cpu_resolved_submission_prerequisites =
        plugin_inspection_for_default_helpers.synthetic_cpu_resolved_submission_prerequisite_plan(
            "external",
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )?;
    assert_eq!(
        report_synthetic_cpu_resolved_submission_prerequisites,
        submission_prerequisites_after_execution_readiness_receipts
    );
    assert!(
        submission_prerequisites_after_execution_readiness_receipts.is_non_submitting_boundary()
    );
    submission_prerequisites_after_execution_readiness_receipts.assert_non_submitting_boundary()?;

    let submission_gate_after_execution_readiness_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
                &execution_readiness_blocker_resolution_receipt_plan,
            )?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[batch_validation, materialized_validation],
                &runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )?,
        submission_gate_after_execution_readiness_receipts
    );
    assert!(submission_gate_after_execution_readiness_receipts.request_plan_ready);
    assert!(submission_gate_after_execution_readiness_receipts.execution_readiness_ready);
    assert!(submission_gate_after_execution_readiness_receipts.all_components_applied);
    assert!(
        submission_gate_after_execution_readiness_receipts.all_live_aql_proof_validations_applied
    );
    assert!(submission_gate_after_execution_readiness_receipts.no_live_aql_submission_side_effects);
    assert!(submission_gate_after_execution_readiness_receipts.no_live_queue_mutation);
    assert!(submission_gate_after_execution_readiness_receipts.is_non_submitting_boundary());
    submission_gate_after_execution_readiness_receipts.assert_non_submitting_boundary()?;
    assert_eq!(
        submission_gate_after_execution_readiness_receipts.submission_blocker_count,
        0
    );
    assert_eq!(
        submission_gate_after_execution_readiness_receipts.execution_blocker_count,
        0
    );
    assert!(submission_gate_after_execution_readiness_receipts
        .blockers
        .is_empty());
    assert!(submission_gate_after_execution_readiness_receipts.submission_ready);
    submission_gate_after_execution_readiness_receipts.assert_consistent()?;
    let resolved_submission_gate_receipt_lines =
        submission_gate_after_execution_readiness_receipts.receipt_lines();
    assert_eq!(
        resolved_submission_gate_receipt_lines[0],
        "receipt.kind=model_runtime_launch_submission_gate"
    );
    assert!(submission_gate_after_execution_readiness_receipts
        .receipt_text()
        .ends_with('\n'));
    let resolved_submission_gate_receipt_fingerprint =
        submission_gate_after_execution_readiness_receipts.receipt_fingerprint();
    assert_eq!(resolved_submission_gate_receipt_fingerprint.len(), 64);
    assert!(resolved_submission_gate_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_resolved_submission_gate_receipt =
        submission_gate_after_execution_readiness_receipts.clone();
    stale_resolved_submission_gate_receipt.component_pending_count += 1;
    assert_ne!(
        stale_resolved_submission_gate_receipt.receipt_fingerprint(),
        resolved_submission_gate_receipt_fingerprint
    );
    let synthetic_cpu_resolved_submission_gate = execution_requests
        .synthetic_cpu_resolved_submission_gate(
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )?;
    assert_eq!(
        synthetic_cpu_resolved_submission_gate,
        submission_gate_after_execution_readiness_receipts
    );
    let report_synthetic_cpu_resolved_submission_gate = plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_gate(
            "external",
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )?;
    assert_eq!(
        report_synthetic_cpu_resolved_submission_gate,
        submission_gate_after_execution_readiness_receipts
    );
    assert!(execution_requests
        .synthetic_cpu_resolved_submission_prerequisite_plan(&[], "contract_test_cpu_receipt")
        .unwrap_err()
        .to_string()
        .contains("is not fully resolved"));
    assert!(execution_requests
        .synthetic_cpu_resolved_submission_gate(&[], "contract_test_cpu_receipt")
        .unwrap_err()
        .to_string()
        .contains("is not fully resolved"));
    assert!(execution_requests
        .synthetic_cpu_resolved_submission_blocker_report(&[], "contract_test_cpu_receipt")
        .unwrap_err()
        .to_string()
        .contains("is not fully resolved"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_prerequisite_plan(
            "external",
            &[],
            "contract_test_cpu_receipt"
        )
        .unwrap_err()
        .to_string()
        .contains("is not fully resolved"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_gate("external", &[], "contract_test_cpu_receipt")
        .unwrap_err()
        .to_string()
        .contains("is not fully resolved"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_blocker_report(
            "external",
            &[],
            "contract_test_cpu_receipt"
        )
        .unwrap_err()
        .to_string()
        .contains("is not fully resolved"));
    let submission_blockers_after_execution_readiness_receipts =
        submission_gate_after_execution_readiness_receipts.blocker_report()?;
    assert_eq!(
        execution_requests.synthetic_cpu_resolved_submission_blocker_report(
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )?,
        submission_blockers_after_execution_readiness_receipts
    );
    assert_eq!(
        plugin_inspection_for_default_helpers.synthetic_cpu_resolved_submission_blocker_report(
            "external",
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )?,
        submission_blockers_after_execution_readiness_receipts
    );
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[batch_validation, materialized_validation],
                &runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )?,
        submission_blockers_after_execution_readiness_receipts
    );
    assert_eq!(
        submission_blockers_after_execution_readiness_receipts.blocker_count,
        0
    );
    assert!(submission_blockers_after_execution_readiness_receipts.submission_ready);
    assert!(submission_blockers_after_execution_readiness_receipts.is_non_submitting_boundary());
    submission_blockers_after_execution_readiness_receipts.assert_non_submitting_boundary()?;
    submission_blockers_after_execution_readiness_receipts.assert_consistent()?;
    let resolved_submission_blocker_receipt_lines =
        submission_blockers_after_execution_readiness_receipts.receipt_lines();
    assert_eq!(
        resolved_submission_blocker_receipt_lines[0],
        "receipt.kind=model_runtime_launch_submission_blocker_report"
    );
    assert!(submission_blockers_after_execution_readiness_receipts
        .receipt_text()
        .ends_with('\n'));
    let resolved_submission_blocker_receipt_fingerprint =
        submission_blockers_after_execution_readiness_receipts.receipt_fingerprint();
    assert_eq!(resolved_submission_blocker_receipt_fingerprint.len(), 64);
    assert!(resolved_submission_blocker_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_resolved_submission_blocker_receipt =
        submission_blockers_after_execution_readiness_receipts.clone();
    stale_resolved_submission_blocker_receipt.total_pending_count += 1;
    assert_ne!(
        stale_resolved_submission_blocker_receipt.receipt_fingerprint(),
        resolved_submission_blocker_receipt_fingerprint
    );

    let missing_queue_execution_readiness_resolution_receipts =
        execution_readiness_blocker_resolution_receipts
            .iter()
            .filter(|receipt| receipt.requirement != "queue_reservation")
            .cloned()
            .collect::<Vec<_>>();
    let missing_queue_execution_readiness_resolution_receipt_plan =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&missing_queue_execution_readiness_resolution_receipts)?;
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan.receipt_input_count,
        execution_readiness.blockers.len() - 1
    );
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan.receipt_present_count,
        execution_readiness.blockers.len() - 1
    );
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan.resolution_applied_count,
        execution_readiness.blockers.len() - 1
    );
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan.missing_receipt_count,
        1
    );
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan.rejected_receipt_count,
        1
    );
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan.resolution_pending_count,
        1
    );
    assert!(!missing_queue_execution_readiness_resolution_receipt_plan.all_receipts_present);
    assert!(!missing_queue_execution_readiness_resolution_receipt_plan.all_resolutions_applied);
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan
            .pending_requirement_names()
            .as_slice(),
        &["queue_reservation"]
    );
    assert_eq!(
        missing_queue_execution_readiness_resolution_receipt_plan
            .resolution_for("queue_reservation")
            .unwrap()
            .rejection_reason,
        Some("missing_receipt")
    );
    missing_queue_execution_readiness_resolution_receipt_plan.assert_consistent()?;
    let missing_queue_submission_prerequisites_after_execution_readiness_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &missing_queue_execution_readiness_resolution_receipt_plan,
            )?;
    assert_eq!(
        missing_queue_submission_prerequisites_after_execution_readiness_receipts
            .satisfied_prerequisite_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        missing_queue_submission_prerequisites_after_execution_readiness_receipts
            .unsatisfied_prerequisite_count,
        1
    );
    assert_eq!(
        missing_queue_submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_next_action_count,
        1
    );
    assert_eq!(
        missing_queue_submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_next_action_request_plan_names()
            .as_slice(),
        &["queue_reservation_request_plan"]
    );
    assert!(
        !missing_queue_submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_ready
    );
    assert!(
        !missing_queue_submission_prerequisites_after_execution_readiness_receipts
            .all_prerequisites_satisfied
    );
    assert!(
        !missing_queue_submission_prerequisites_after_execution_readiness_receipts.submission_ready
    );
    missing_queue_submission_prerequisites_after_execution_readiness_receipts
        .assert_consistent()?;
    let missing_queue_submission_gate_after_execution_readiness_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
                &missing_queue_execution_readiness_resolution_receipt_plan,
            )?;
    assert_eq!(
        missing_queue_submission_gate_after_execution_readiness_receipts.execution_blocker_count,
        1
    );
    assert!(
        missing_queue_submission_gate_after_execution_readiness_receipts
            .has_blocker("queue_reservation")
    );
    assert!(!missing_queue_submission_gate_after_execution_readiness_receipts.submission_ready);
    missing_queue_submission_gate_after_execution_readiness_receipts.assert_consistent()?;

    let loaded_code_object_resolution_receipt_index =
        execution_readiness_blocker_resolution_receipts
            .iter()
            .position(|receipt| receipt.requirement == "loaded_code_object_base")
            .unwrap();
    let mut incomplete_execution_readiness_resolution_receipts =
        execution_readiness_blocker_resolution_receipts.clone();
    incomplete_execution_readiness_resolution_receipts
        [loaded_code_object_resolution_receipt_index]
        .resolved_count = 1;
    incomplete_execution_readiness_resolution_receipts
        [loaded_code_object_resolution_receipt_index]
        .pending_count = 1;
    let incomplete_execution_readiness_resolution_receipt_plan =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&incomplete_execution_readiness_resolution_receipts)?;
    assert_eq!(
        incomplete_execution_readiness_resolution_receipt_plan.resolution_applied_count,
        execution_readiness.blockers.len() - 1
    );
    assert_eq!(
        incomplete_execution_readiness_resolution_receipt_plan.rejected_receipt_count,
        1
    );
    assert_eq!(
        incomplete_execution_readiness_resolution_receipt_plan.resolution_pending_count,
        2
    );
    assert_eq!(
        incomplete_execution_readiness_resolution_receipt_plan
            .resolution_for("loaded_code_object_base")
            .unwrap()
            .rejection_reason,
        Some("resolution_incomplete")
    );
    assert!(!incomplete_execution_readiness_resolution_receipt_plan.all_resolutions_applied);
    incomplete_execution_readiness_resolution_receipt_plan.assert_consistent()?;
    let incomplete_submission_prerequisites_after_execution_readiness_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &incomplete_execution_readiness_resolution_receipt_plan,
            )?;
    assert_eq!(
        incomplete_submission_prerequisites_after_execution_readiness_receipts
            .satisfied_prerequisite_count,
        execution_requests.runtime_request_plan_count - 2
    );
    assert_eq!(
        incomplete_submission_prerequisites_after_execution_readiness_receipts
            .unsatisfied_prerequisite_count,
        2
    );
    assert_eq!(
        incomplete_submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_next_action_count,
        2
    );
    assert_eq!(
        incomplete_submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_next_action_request_plan_names()
            .as_slice(),
        &[
            "code_object_load_request_plan",
            "code_object_base_binding_request_plan"
        ]
    );
    assert!(
        !incomplete_submission_prerequisites_after_execution_readiness_receipts
            .execution_readiness_ready
    );
    assert!(
        !incomplete_submission_prerequisites_after_execution_readiness_receipts.submission_ready
    );
    incomplete_submission_prerequisites_after_execution_readiness_receipts.assert_consistent()?;
    let incomplete_submission_gate_after_execution_readiness_receipts =
        submission_prerequisites_after_runtime_component_receipts
            .submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
                &incomplete_execution_readiness_resolution_receipt_plan,
            )?;
    assert_eq!(
        incomplete_submission_gate_after_execution_readiness_receipts.execution_blocker_count,
        1
    );
    assert!(
        incomplete_submission_gate_after_execution_readiness_receipts
            .has_blocker("loaded_code_object_base")
    );
    assert!(!incomplete_submission_gate_after_execution_readiness_receipts.submission_ready);
    incomplete_submission_gate_after_execution_readiness_receipts.assert_consistent()?;

    let mut mismatched_execution_readiness_resolution_receipts =
        execution_readiness_blocker_resolution_receipts.clone();
    mismatched_execution_readiness_resolution_receipts
        [loaded_code_object_resolution_receipt_index]
        .code_object_sha256 = "stale-execution-readiness-resolution-receipt".to_string();
    let mismatched_execution_readiness_resolution_receipt_plan =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&mismatched_execution_readiness_resolution_receipts)?;
    assert_eq!(
        mismatched_execution_readiness_resolution_receipt_plan
            .resolution_for("loaded_code_object_base")
            .unwrap()
            .rejection_reason,
        Some("launch_identity_mismatch")
    );
    assert!(!mismatched_execution_readiness_resolution_receipt_plan.all_resolutions_applied);
    mismatched_execution_readiness_resolution_receipt_plan.assert_consistent()?;

    let mut side_effecting_execution_readiness_resolution_receipts =
        execution_readiness_blocker_resolution_receipts.clone();
    side_effecting_execution_readiness_resolution_receipts
        [loaded_code_object_resolution_receipt_index]
        .live_aql_submits_work = true;
    let side_effecting_execution_readiness_resolution_receipt_plan_err =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&side_effecting_execution_readiness_resolution_receipts)
            .unwrap_err()
            .to_string();
    assert!(
        side_effecting_execution_readiness_resolution_receipt_plan_err
            .contains("not a non-submitting boundary")
    );
    assert!(
        side_effecting_execution_readiness_resolution_receipt_plan_err
            .contains("live AQL submission side effect is true")
    );

    let mut queue_mutating_execution_readiness_resolution_receipts =
        execution_readiness_blocker_resolution_receipts.clone();
    queue_mutating_execution_readiness_resolution_receipts
        [loaded_code_object_resolution_receipt_index]
        .mutates_live_queue = true;
    let queue_mutating_execution_readiness_resolution_receipt_plan_err =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&queue_mutating_execution_readiness_resolution_receipts)
            .unwrap_err()
            .to_string();
    assert!(
        queue_mutating_execution_readiness_resolution_receipt_plan_err
            .contains("not a non-submitting boundary")
    );
    assert!(
        queue_mutating_execution_readiness_resolution_receipt_plan_err
            .contains("live queue mutation is true")
    );

    let mut duplicate_execution_readiness_resolution_receipts =
        execution_readiness_blocker_resolution_receipts.clone();
    duplicate_execution_readiness_resolution_receipts.push(
        execution_readiness_blocker_resolution_receipts
            .first()
            .unwrap()
            .clone(),
    );
    let duplicate_execution_readiness_resolution_receipt_err =
        execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&duplicate_execution_readiness_resolution_receipts)
            .unwrap_err()
            .to_string();
    assert!(duplicate_execution_readiness_resolution_receipt_err.contains("appears more than once"));

    let default_runtime_component_application_receipts = runtime_component_applications
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
                receipt_source: "public_contract_external_runtime",
                requested_count: application.source_next_action_pending_count,
                applied_count: application.source_next_action_pending_count,
                pending_count: 0,
                live_aql_submits_work: false,
                mutates_live_queue: false,
            },
        )
        .collect::<Vec<_>>();
    let default_runtime_component_application_receipt_plan = runtime_component_applications
        .application_receipt_plan(&default_runtime_component_application_receipts)?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(
                "external",
                &[],
                &default_runtime_component_application_receipts,
            )?,
        default_runtime_component_application_receipt_plan
    );
    assert!(default_runtime_component_application_receipt_plan.all_applications_applied);
    let default_submission_prerequisites_after_runtime_component_receipts =
        submission_prerequisites
            .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
                &default_runtime_component_application_receipt_plan,
            )?;
    assert_eq!(
        default_submission_prerequisites_after_runtime_component_receipts
            .pending_component_request_count,
        queue_component.pending_count + aql_live_component.pending_count
    );
    assert_eq!(
        default_submission_prerequisites_after_runtime_component_receipts
            .runtime_request_component_next_action_count,
        0
    );
    assert_eq!(
        default_submission_prerequisites_after_runtime_component_receipts
            .live_aql_proof_validation_next_action_count,
        expected_live_aql_proof_surface_plans.len()
    );
    assert_eq!(
        default_submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_next_action_count,
        expected_runtime_component_next_action_plans.len()
    );
    assert_eq!(
        default_submission_prerequisites_after_runtime_component_receipts
            .live_aql_proof_validation_next_action_request_plan_names()
            .as_slice(),
        &expected_live_aql_proof_surface_plans
    );
    assert_eq!(
        default_submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_next_action_request_plan_names()
            .as_slice(),
        &expected_runtime_component_next_action_plans
    );
    let default_submission_gate_after_runtime_component_receipts = submission_prerequisites
        .submission_gate_with_runtime_request_component_application_receipt_plan(
            &default_runtime_component_application_receipt_plan,
        )?;
    assert!(!default_submission_gate_after_runtime_component_receipts.all_components_applied);
    assert!(
        !default_submission_gate_after_runtime_component_receipts
            .all_live_aql_proof_validations_applied
    );
    assert_eq!(
        default_submission_gate_after_runtime_component_receipts.component_pending_count,
        queue_component.pending_count + aql_live_component.pending_count
    );
    assert_eq!(
        default_submission_gate_after_runtime_component_receipts
            .live_aql_proof_validation_pending_count,
        expected_live_aql_proof_surface_plans.len()
    );
    assert!(default_submission_gate_after_runtime_component_receipts
        .has_blocker("runtime_request_components"));
    assert!(default_submission_gate_after_runtime_component_receipts
        .has_blocker("live_aql_proof_validation"));
    assert!(!default_submission_gate_after_runtime_component_receipts.submission_ready);
    default_submission_gate_after_runtime_component_receipts.assert_consistent()?;
    let default_execution_readiness_resolutions_after_component_receipts =
        default_submission_prerequisites_after_runtime_component_receipts
            .execution_readiness_blocker_resolution_plan()?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(
                "external",
                &[],
                &default_runtime_component_application_receipts,
            )?,
        default_execution_readiness_resolutions_after_component_receipts
    );
    assert_eq!(
        default_execution_readiness_resolutions_after_component_receipts
            .execution_readiness_next_action_count,
        expected_runtime_component_next_action_plans.len()
    );
    assert_eq!(
        default_execution_readiness_resolutions_after_component_receipts.blocker_resolution_count,
        execution_readiness.blockers.len() - 2
    );
    assert_eq!(
        default_execution_readiness_resolutions_after_component_receipts.source_prerequisite_count,
        expected_runtime_component_next_action_plans.len()
    );
    assert!(
        default_execution_readiness_resolutions_after_component_receipts
            .resolution_for("queue_reservation")
            .is_none()
    );
    assert!(
        default_execution_readiness_resolutions_after_component_receipts
            .resolution_for("aql_packet_materialization")
            .is_none()
    );
    assert!(!default_execution_readiness_resolutions_after_component_receipts.submission_ready);
    default_execution_readiness_resolutions_after_component_receipts.assert_consistent()?;
    let stale_execution_readiness_receipt_overlay_err =
        default_submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &execution_readiness_blocker_resolution_receipt_plan,
            )
            .unwrap_err()
            .to_string();
    assert!(stale_execution_readiness_receipt_overlay_err
        .contains("does not match submission prerequisites"));
    assert!(stale_execution_readiness_receipt_overlay_err.contains("blocker resolution count"));
    assert!(stale_execution_readiness_receipt_overlay_err
        .contains("stale execution readiness resolution queue_reservation"));
    let unexpected_execution_readiness_resolution_receipt_plan =
        default_execution_readiness_resolutions_after_component_receipts
            .resolution_receipt_plan(&execution_readiness_blocker_resolution_receipts)?;
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[],
                &default_runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )?,
        unexpected_execution_readiness_resolution_receipt_plan
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.blocker_resolution_count,
        execution_readiness.blockers.len() - 2
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.receipt_input_count,
        execution_readiness.blockers.len()
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.receipt_present_count,
        execution_readiness.blockers.len() - 2
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.resolution_applied_count,
        execution_readiness.blockers.len() - 2
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.resolution_pending_count,
        0
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.missing_receipt_count,
        0
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.unexpected_receipt_count,
        2
    );
    assert_eq!(
        unexpected_execution_readiness_resolution_receipt_plan.rejected_receipt_count,
        0
    );
    assert!(!unexpected_execution_readiness_resolution_receipt_plan.all_receipts_present);
    assert!(!unexpected_execution_readiness_resolution_receipt_plan.all_resolutions_applied);
    assert!(!unexpected_execution_readiness_resolution_receipt_plan.execution_readiness_ready);
    assert!(!unexpected_execution_readiness_resolution_receipt_plan.submission_ready);
    unexpected_execution_readiness_resolution_receipt_plan.assert_consistent()?;
    let unexpected_execution_readiness_receipt_overlay_err =
        default_submission_prerequisites_after_runtime_component_receipts
            .submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                &unexpected_execution_readiness_resolution_receipt_plan,
            )
            .unwrap_err()
            .to_string();
    assert!(unexpected_execution_readiness_receipt_overlay_err
        .contains("unexpected execution readiness resolution receipt input"));
    let report_unexpected_execution_readiness_receipt_overlay_err =
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[],
                &default_runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )
            .unwrap_err()
            .to_string();
    assert!(report_unexpected_execution_readiness_receipt_overlay_err
        .contains("unexpected execution readiness resolution receipt input"));
    let report_unexpected_execution_readiness_receipt_gate_overlay_err =
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[],
                &default_runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )
            .unwrap_err()
            .to_string();
    assert!(
        report_unexpected_execution_readiness_receipt_gate_overlay_err
            .contains("unexpected execution readiness resolution receipt input")
    );
    let report_unexpected_execution_readiness_receipt_blocker_report_overlay_err =
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(
                "external",
                &[],
                &default_runtime_component_application_receipts,
                &execution_readiness_blocker_resolution_receipts,
            )
            .unwrap_err()
            .to_string();
    assert!(
        report_unexpected_execution_readiness_receipt_blocker_report_overlay_err
            .contains("unexpected execution readiness resolution receipt input")
    );
    let unexpected_submission_prerequisite_receipt_overlay_err = submission_prerequisites
        .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
            &runtime_component_application_receipt_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(unexpected_submission_prerequisite_receipt_overlay_err
        .contains("does not match submission prerequisites"));
    assert!(unexpected_submission_prerequisite_receipt_overlay_err
        .contains("runtime component next-action count"));

    let missing_queue_runtime_component_receipts = runtime_component_application_receipts
        .iter()
        .filter(|receipt| receipt.request_plan != "queue_reservation_request_plan")
        .cloned()
        .collect::<Vec<_>>();
    let missing_queue_runtime_component_receipt_plan =
        runtime_component_applications_with_validations
            .application_receipt_plan(&missing_queue_runtime_component_receipts)?;
    assert_eq!(
        missing_queue_runtime_component_receipt_plan.receipt_input_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        missing_queue_runtime_component_receipt_plan.receipt_present_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        missing_queue_runtime_component_receipt_plan.application_applied_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        missing_queue_runtime_component_receipt_plan.missing_receipt_count,
        1
    );
    assert_eq!(
        missing_queue_runtime_component_receipt_plan.rejected_receipt_count,
        1
    );
    assert_eq!(
        missing_queue_runtime_component_receipt_plan.application_pending_count,
        queue_component.pending_count
    );
    assert!(!missing_queue_runtime_component_receipt_plan.all_receipts_present);
    assert!(!missing_queue_runtime_component_receipt_plan.all_applications_applied);
    assert_eq!(
        missing_queue_runtime_component_receipt_plan
            .pending_request_plan_names()
            .as_slice(),
        &["queue_reservation_request_plan"]
    );
    assert_eq!(
        missing_queue_runtime_component_receipt_plan
            .application_for("queue_reservation_request_plan")
            .unwrap()
            .rejection_reason,
        Some("missing_receipt")
    );
    missing_queue_runtime_component_receipt_plan.assert_consistent()?;
    let missing_queue_submission_prerequisites = submission_prerequisites_with_validations
        .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
            &missing_queue_runtime_component_receipt_plan,
        )?;
    assert_eq!(
        missing_queue_submission_prerequisites.pending_component_request_count,
        queue_component.pending_count
    );
    assert_eq!(
        missing_queue_submission_prerequisites.runtime_request_component_next_action_count,
        1
    );
    assert_eq!(
        missing_queue_submission_prerequisites.execution_readiness_next_action_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        missing_queue_submission_prerequisites
            .runtime_request_component_next_action_request_plan_names()
            .as_slice(),
        &["queue_reservation_request_plan"]
    );
    let missing_queue_submission_gate = submission_prerequisites_with_validations
        .submission_gate_with_runtime_request_component_application_receipt_plan(
            &missing_queue_runtime_component_receipt_plan,
        )?;
    assert!(!missing_queue_submission_gate.all_components_applied);
    assert!(missing_queue_submission_gate.all_live_aql_proof_validations_applied);
    assert_eq!(
        missing_queue_submission_gate.component_pending_count,
        queue_component.pending_count
    );
    assert!(missing_queue_submission_gate.has_blocker("runtime_request_components"));
    assert!(!missing_queue_submission_gate.has_blocker("live_aql_proof_validation"));
    assert!(!missing_queue_submission_gate.submission_ready);
    missing_queue_submission_gate.assert_consistent()?;
    let missing_queue_execution_readiness_resolutions =
        missing_queue_submission_prerequisites.execution_readiness_blocker_resolution_plan()?;
    assert_eq!(
        missing_queue_execution_readiness_resolutions.execution_readiness_next_action_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        missing_queue_execution_readiness_resolutions.blocker_resolution_count,
        execution_readiness.blockers.len() - 1
    );
    assert!(missing_queue_execution_readiness_resolutions
        .resolution_for("queue_reservation")
        .is_none());
    assert!(missing_queue_execution_readiness_resolutions.all_blocker_resolutions_ready);
    assert!(!missing_queue_execution_readiness_resolutions.submission_ready);
    missing_queue_execution_readiness_resolutions.assert_consistent()?;

    let unexpected_runtime_component_receipt_plan = runtime_component_applications
        .application_receipt_plan(&runtime_component_application_receipts)?;
    assert_eq!(
        unexpected_runtime_component_receipt_plan.application_count,
        expected_runtime_component_next_action_plans.len()
    );
    assert_eq!(
        unexpected_runtime_component_receipt_plan.unexpected_receipt_count,
        expected_live_aql_proof_surface_plans.len()
    );
    assert_eq!(
        unexpected_runtime_component_receipt_plan.application_applied_count,
        expected_runtime_component_next_action_plans.len()
    );
    assert_eq!(
        unexpected_runtime_component_receipt_plan.application_pending_count,
        0
    );
    assert!(!unexpected_runtime_component_receipt_plan.all_receipts_present);
    assert!(!unexpected_runtime_component_receipt_plan.all_applications_applied);
    assert!(!unexpected_runtime_component_receipt_plan.all_components_applied);
    assert!(!unexpected_runtime_component_receipt_plan.submission_ready);
    unexpected_runtime_component_receipt_plan.assert_consistent()?;
    let unexpected_runtime_component_receipt_overlay_err = submission_prerequisites
        .submission_prerequisite_plan_with_runtime_request_component_application_receipt_plan(
            &unexpected_runtime_component_receipt_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(unexpected_runtime_component_receipt_overlay_err
        .contains("unexpected runtime component receipt input"));

    let mut incomplete_runtime_component_receipts = runtime_component_application_receipts.clone();
    incomplete_runtime_component_receipts[queue_receipt_index].applied_count =
        incomplete_runtime_component_receipts[queue_receipt_index].requested_count - 1;
    incomplete_runtime_component_receipts[queue_receipt_index].pending_count = 1;
    let incomplete_runtime_component_receipt_plan = runtime_component_applications_with_validations
        .application_receipt_plan(&incomplete_runtime_component_receipts)?;
    assert_eq!(
        incomplete_runtime_component_receipt_plan.application_applied_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        incomplete_runtime_component_receipt_plan.rejected_receipt_count,
        1
    );
    assert_eq!(
        incomplete_runtime_component_receipt_plan.application_pending_count,
        queue_component.pending_count
    );
    assert_eq!(
        incomplete_runtime_component_receipt_plan
            .application_for("queue_reservation_request_plan")
            .unwrap()
            .rejection_reason,
        Some("application_incomplete")
    );
    assert!(!incomplete_runtime_component_receipt_plan.all_applications_applied);
    incomplete_runtime_component_receipt_plan.assert_consistent()?;

    let mut mismatched_runtime_component_receipts = runtime_component_application_receipts.clone();
    mismatched_runtime_component_receipts[queue_receipt_index].code_object_sha256 =
        "stale-runtime-component-application-receipt".to_string();
    let mismatched_runtime_component_receipt_plan = runtime_component_applications_with_validations
        .application_receipt_plan(&mismatched_runtime_component_receipts)?;
    assert_eq!(
        mismatched_runtime_component_receipt_plan
            .application_for("queue_reservation_request_plan")
            .unwrap()
            .rejection_reason,
        Some("launch_identity_mismatch")
    );
    assert!(!mismatched_runtime_component_receipt_plan.all_applications_applied);
    mismatched_runtime_component_receipt_plan.assert_consistent()?;
    let mut duplicate_runtime_component_receipts = runtime_component_application_receipts.clone();
    duplicate_runtime_component_receipts.push(
        runtime_component_application_receipts
            .first()
            .unwrap()
            .clone(),
    );
    let duplicate_runtime_component_receipt_err = runtime_component_applications_with_validations
        .application_receipt_plan(&duplicate_runtime_component_receipts)
        .unwrap_err()
        .to_string();
    assert!(duplicate_runtime_component_receipt_err.contains("appears more than once"));

    let failed_runtime_component_applications =
        failed_submission_prerequisites.runtime_request_component_application_plan()?;
    assert_eq!(
        failed_runtime_component_applications.application_count,
        execution_requests.runtime_request_plan_count - 1
    );
    assert_eq!(
        failed_runtime_component_applications.application_ready_count,
        failed_runtime_component_applications.application_count
    );
    assert_eq!(
        failed_runtime_component_applications.application_blocked_count,
        0
    );
    assert_eq!(
        failed_runtime_component_applications.deferred_pending_component_request_count,
        queue_component.pending_count
    );
    assert_eq!(
        failed_runtime_component_applications.application_pending_count,
        failed_runtime_component_applications.pending_component_request_count
            - queue_component.pending_count
    );
    assert_eq!(
        failed_runtime_component_applications.live_aql_proof_validation_pending_count,
        1
    );
    assert_eq!(
        failed_runtime_component_applications.live_aql_proof_application_count,
        1
    );
    assert!(failed_runtime_component_applications
        .application_for("queue_reservation_request_plan")
        .is_none());
    assert!(failed_runtime_component_applications
        .application_for_step(RuntimeLaunchExecutionRequestStep::QueueReservation)
        .is_none());
    assert!(failed_runtime_component_applications
        .application_for("aql_live_relocation_binding_request_plan")
        .is_some());
    assert_eq!(
        failed_runtime_component_applications
            .live_aql_proof_application_request_plan_names()
            .as_slice(),
        &["aql_live_relocation_binding_request_plan"]
    );
    assert!(!failed_runtime_component_applications.all_components_applied);
    assert!(!failed_runtime_component_applications.submission_ready);
    failed_runtime_component_applications.assert_consistent()?;

    let mismatched_submission_prerequisite_err = execution_requests
        .submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(
            &mismatched_validation_application_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(
        mismatched_submission_prerequisite_err.contains("does not match execution request plan")
    );
    let unexpected_submission_prerequisite_err = execution_requests
        .submission_prerequisite_plan_with_live_aql_proof_validation_application_plan(
            &unexpected_validation_application_plan,
        )
        .unwrap_err()
        .to_string();
    assert!(unexpected_submission_prerequisite_err.contains("unexpected validation input"));
    assert_eq!(
        launch_preflight.submission_prerequisite_plan(&device_pointer_validation)?,
        submission_prerequisites
    );
    assert_eq!(
        admission.runtime_launch_submission_prerequisite_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        submission_prerequisites
    );
    assert_eq!(
        readiness.runtime_launch_submission_prerequisite_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        submission_prerequisites
    );
    assert!(execution_requests.request_plan_ready);
    assert!(!execution_requests.all_components_applied);
    assert!(execution_requests.has_blocker("queue_reservation"));
    assert_eq!(execution_requests.code_object_loads, code_object_loads);
    assert_eq!(
        execution_requests.code_object_base_bindings,
        code_object_base_bindings
    );
    assert_eq!(
        execution_requests.completion_signal_bindings,
        completion_signal_bindings
    );
    assert_eq!(execution_requests.queue_reservations, queue_reservations);
    assert_eq!(execution_requests.kernarg_allocations, kernarg_allocations);
    assert_eq!(
        execution_requests.kernel_argument_abi_schema_requests,
        kernel_argument_abi_schema_requests
    );
    assert_eq!(
        execution_requests.kernel_candidate_selection_requests,
        kernel_selection_requests
    );
    assert_eq!(
        execution_requests.host_launcher_branch_requests,
        host_launcher_branch_requests
    );
    assert_eq!(
        execution_requests.aql_live_relocation_bindings,
        aql_live_relocation_bindings
    );
    assert_eq!(execution_requests.execution_readiness, execution_readiness);
    assert_eq!(
        execution_requests.unresolved_runtime_requirements,
        execution_readiness.unresolved_runtime_requirements
    );
    assert_eq!(
        admission.runtime_launch_execution_request_plan(
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        execution_requests
    );
    assert_eq!(
        readiness.runtime_launch_execution_request_plan(
            &slot_bindings,
            &code_object,
            DEFAULT_RUNTIME_LAUNCH_WINDOW_DISPATCHES,
            &device_pointer_validation,
        )?,
        execution_requests
    );
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_execution_request_plan("external")?,
        execution_requests
    );
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_gate("external")?,
        execution_requests.submission_gate()?
    );
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_blocker_report("external")?,
        execution_requests.submission_gate()?.blocker_report()?
    );
    assert_eq!(
        plugin_inspection_for_default_helpers
            .synthetic_cpu_runtime_launch_submission_prerequisite_plan("external")?,
        execution_requests.submission_prerequisite_plan()?
    );
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_execution_request_plan("")
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_submission_gate("")
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_submission_blocker_report("")
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_submission_prerequisite_plan("")
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan(
            "",
            &[batch_validation, materialized_validation],
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_runtime_request_component_application_plan(
            "",
            &[batch_validation, materialized_validation],
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan(
            "",
            &[batch_validation, materialized_validation],
            &runtime_component_application_receipts,
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan(
            "",
            &[batch_validation, materialized_validation],
            &runtime_component_application_receipts,
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan(
            "",
            &[batch_validation, materialized_validation],
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan(
            "",
            &[batch_validation, materialized_validation],
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan(
            "",
            &[batch_validation, materialized_validation],
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan(
            "",
            &[batch_validation, materialized_validation],
            &runtime_component_application_receipts,
            &execution_readiness_blocker_resolution_receipts,
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_prerequisite_plan(
            "",
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_gate(
            "",
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    assert!(plugin_inspection_for_default_helpers
        .synthetic_cpu_resolved_submission_blocker_report(
            "",
            &[batch_validation, materialized_validation],
            "contract_test_cpu_receipt",
        )
        .unwrap_err()
        .to_string()
        .contains("synthetic CPU runtime launch namespace"));
    execution_requests.assert_consistent()?;
    let mut inconsistent_execution_requests = execution_requests.clone();
    inconsistent_execution_requests.component_applied_count = 1;
    let err = inconsistent_execution_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("component applied count"));
    let mut inconsistent_execution_requests = execution_requests.clone();
    inconsistent_execution_requests.component_pending_count = 0;
    let err = inconsistent_execution_requests
        .assert_consistent()
        .unwrap_err()
        .to_string();
    assert!(err.contains("component pending count"));
    assert!(execution_readiness.assert_executable().is_err());
    execution_readiness.assert_consistent()?;
    let lm_head_arguments = launch_arguments.dispatch_for("lm_head").unwrap();
    assert_eq!(
        lm_head_arguments.arguments.len(),
        lm_head_launch.slot_arguments.len() + lm_head_launch.scalar_arguments.len()
    );
    assert_eq!(readiness.lowering.native_gpu_ops, 2);
    assert_eq!(readiness.lowering.fused_native_gpu_ops, 1);
    assert_eq!(readiness.lowering.gap_ops, 0);
    assert_eq!(
        readiness.lowering.route_for("embed_tokens").unwrap().status,
        LoweringStatus::FusedNativeGpu
    );
    assert_eq!(
        readiness.lowering.route_for("lm_head").unwrap().status,
        LoweringStatus::NativeGpu
    );
    assert_eq!(
        readiness
            .lowering
            .route_for("sample_argmax")
            .unwrap()
            .status,
        LoweringStatus::NativeGpu
    );

    let rejected_model = ExternalUnsupportedCollectiveModel;
    let rejected_inspection = inspect_model_plugin(&rejected_model, &catalog)?;
    assert!(!rejected_inspection.is_accepted());
    assert!(rejected_inspection.assert_accepted().is_err());
    assert!(!rejected_inspection.is_static_handoff_ready());
    let rejected_static_handoff_err = rejected_inspection
        .assert_static_handoff_ready()
        .unwrap_err()
        .to_string();
    assert!(rejected_static_handoff_err.contains("not static handoff ready"));
    assert!(rejected_static_handoff_err.contains("rejected"));
    let rejected_summary = rejected_inspection.summary();
    rejected_summary.assert_consistent_with(&rejected_inspection)?;
    assert!(!rejected_summary.accepted);
    assert!(!rejected_summary.static_ready);
    assert_eq!(
        rejected_summary.model_name,
        "external-unsupported-collective"
    );
    assert_eq!(rejected_summary.compatibility_issue_count, 1);
    assert_eq!(rejected_summary.model_primitive_kind_count, 1);
    assert_eq!(rejected_summary.model_stage_kind_count, 1);
    assert_eq!(rejected_summary.tensor_count, 2);
    assert_eq!(rejected_summary.op_count, 1);
    assert_eq!(rejected_summary.runtime_dispatch_count, 1);
    let rejected_summary_receipt_fingerprint = rejected_summary.receipt_fingerprint();
    assert_eq!(rejected_summary_receipt_fingerprint.len(), 64);
    assert!(rejected_summary_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let rejected_rejection = rejected_inspection.rejection_report();
    rejected_rejection.assert_consistent()?;
    rejected_rejection.assert_consistent_with(&rejected_inspection)?;
    rejected_rejection.assert_rejected()?;
    assert!(rejected_rejection.assert_no_rejection().is_err());
    assert!(rejected_rejection.is_rejected());
    assert_eq!(rejected_rejection.rejection_issue_count, 4);
    assert_eq!(rejected_rejection.readiness_issues.issues.len(), 3);
    assert_eq!(rejected_rejection.compatibility_issues.len(), 1);
    assert_eq!(
        rejected_rejection.lowering_gap_op_names,
        vec!["ep_all_to_all".to_string()]
    );
    assert_eq!(
        rejected_rejection.stage_gap_names,
        vec!["expert_parallel".to_string()]
    );
    assert!(rejected_rejection.unstaged_op_names.is_empty());
    assert!(rejected_rejection
        .missing_checkpoint_weight_names
        .is_empty());
    assert_eq!(
        rejected_rejection.binding_issue_tensor_names,
        vec!["expert_input".to_string()]
    );
    assert!(rejected_rejection
        .readiness_issues
        .issues_for_kind(ModelReadinessIssueKind::TensorBinding)
        .iter()
        .any(|issue| issue.subject == "expert_input"));
    assert!(rejected_rejection
        .readiness_issues
        .issues_for_kind(ModelReadinessIssueKind::LoweringGap)
        .iter()
        .any(|issue| issue.subject == "ep_all_to_all"));
    assert_eq!(
        rejected_rejection.compatibility_issues[0].kind,
        ModelPluginCompatibilityIssueKind::StaticMetadata
    );
    assert!(rejected_rejection.compatibility_issues[0]
        .message
        .contains("static_issues=3"));
    assert!(rejected_rejection.compatibility_issues[0]
        .message
        .contains("binding_issues=1"));
    let rejected_rejection_receipt_text = rejected_rejection.receipt_text();
    assert!(rejected_rejection_receipt_text.contains("receipt.kind=model_plugin_rejection\n"));
    assert!(rejected_rejection_receipt_text.contains("readiness_issues.0.kind=tensor_binding\n"));
    assert!(rejected_rejection_receipt_text.contains("readiness_issues.1.kind=lowering_gap\n"));
    assert!(
        rejected_rejection_receipt_text.contains("compatibility_issues.0.kind=static_metadata\n")
    );
    assert!(rejected_rejection_receipt_text.ends_with('\n'));
    let rejected_rejection_receipt_fingerprint = rejected_rejection.receipt_fingerprint();
    assert_eq!(rejected_rejection_receipt_fingerprint.len(), 64);
    assert!(rejected_rejection_receipt_fingerprint
        .chars()
        .all(|ch| ch.is_ascii_hexdigit()));
    let mut stale_rejection = rejected_rejection.clone();
    stale_rejection.rejection_issue_count += 1;
    assert_ne!(
        stale_rejection.receipt_fingerprint(),
        rejected_rejection_receipt_fingerprint
    );
    assert!(stale_rejection
        .assert_consistent_with(&rejected_inspection)
        .unwrap_err()
        .to_string()
        .contains("rejection issue count"));
    assert!(stale_rejection
        .assert_consistent()
        .unwrap_err()
        .to_string()
        .contains("rejection issue count"));
    let mut stale_rejected_summary = rejected_rejection.clone();
    stale_rejected_summary.summary.accepted = true;
    assert!(!stale_rejected_summary.is_rejected());
    let stale_rejected_summary_err = stale_rejected_summary
        .assert_rejected()
        .unwrap_err()
        .to_string();
    assert!(stale_rejected_summary_err.contains("not rejected"));
    assert!(stale_rejected_summary_err.contains("consistency"));
    assert!(stale_rejected_summary_err.contains("summary accepted true != expected false"));

    Ok(())
}

use anyhow::Result;
use mainarch_core::model_api::prelude::*;

pub const EXTERNAL_PLUGIN_MODEL_NAME: &str = "external-mini-moe";
pub const EXTERNAL_PLUGIN_PACKAGE: &str = "examples/model-api-plugin";

#[derive(Debug, Clone)]
pub struct ExternalMiniMoe {
    vocab: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
    top_k: usize,
}

impl Default for ExternalMiniMoe {
    fn default() -> Self {
        Self {
            vocab: 256,
            hidden: 128,
            intermediate: 64,
            experts: 4,
            top_k: 2,
        }
    }
}

impl ModelDefinition for ExternalMiniMoe {
    fn name(&self) -> &str {
        EXTERNAL_PLUGIN_MODEL_NAME
    }

    fn define(&self, api: &mut dyn ModelPrimitiveApi) -> Result<()> {
        declare(api, "input_ids", DType::U32, vec![1], TensorRole::Token)?;
        declare(api, "next_token", DType::U32, vec![1], TensorRole::Token)?;
        declare(
            api,
            "hidden.0",
            DType::F16,
            vec![self.hidden],
            TensorRole::Activation,
        )?;
        declare(
            api,
            "router.expert_ids",
            DType::U32,
            vec![self.top_k],
            TensorRole::Routing,
        )?;
        declare(
            api,
            "router.expert_weights",
            DType::F32,
            vec![self.top_k],
            TensorRole::Routing,
        )?;
        declare(
            api,
            "moe.out",
            DType::F16,
            vec![self.hidden],
            TensorRole::Activation,
        )?;
        declare(
            api,
            "hidden.1",
            DType::F16,
            vec![self.hidden],
            TensorRole::Activation,
        )?;
        declare(
            api,
            "logits",
            DType::F16,
            vec![self.vocab],
            TensorRole::Logits,
        )?;

        declare_weight(
            api,
            "embed_tokens.weight",
            "external.embed_tokens.weight",
            DType::F16,
            vec![self.vocab, self.hidden],
        )?;
        declare_weight(
            api,
            "router.weight",
            "external.layers.0.mlp.gate.weight",
            DType::F16,
            vec![self.experts, self.hidden],
        )?;
        declare_weight(
            api,
            "experts.gate.weight",
            "external.layers.0.mlp.experts.*.gate_proj.weight",
            DType::F16,
            vec![self.experts, self.intermediate, self.hidden],
        )?;
        declare_weight(
            api,
            "experts.up.weight",
            "external.layers.0.mlp.experts.*.up_proj.weight",
            DType::F16,
            vec![self.experts, self.intermediate, self.hidden],
        )?;
        declare_weight(
            api,
            "experts.down.weight",
            "external.layers.0.mlp.experts.*.down_proj.weight",
            DType::F16,
            vec![self.experts, self.hidden, self.intermediate],
        )?;
        declare_weight(
            api,
            "lm_head.weight",
            "external.lm_head.weight",
            DType::F16,
            vec![self.vocab, self.hidden],
        )?;

        api.begin_stage("embedding", ModelStageKind::Embedding)?;
        api.emit(PrimitiveOp::EmbeddingLookup(EmbeddingLookup {
            name: "embed_tokens".to_string(),
            token_ids: "input_ids".into(),
            weight: "embed_tokens.weight".into(),
            output: "hidden.0".into(),
            vocab: self.vocab,
            hidden: self.hidden,
        }))?;
        api.end_stage()?;

        api.begin_stage("layers.0.moe", ModelStageKind::Moe)?;
        api.emit(PrimitiveOp::MoeRouterTopK(MoeRouterTopK {
            name: "layers.0.router_topk".to_string(),
            hidden: "hidden.0".into(),
            gate_weight: "router.weight".into(),
            expert_ids: "router.expert_ids".into(),
            expert_weights: "router.expert_weights".into(),
            hidden_dim: self.hidden,
            experts: self.experts,
            top_k: self.top_k,
            group: None,
        }))?;
        api.emit(PrimitiveOp::MoeLocalFfn(MoeLocalFfn {
            name: "layers.0.moe_local_ffn".to_string(),
            hidden: "hidden.0".into(),
            gate_weight: "experts.gate.weight".into(),
            up_weight: "experts.up.weight".into(),
            down_weight: "experts.down.weight".into(),
            expert_ids: "router.expert_ids".into(),
            expert_weights: "router.expert_weights".into(),
            output: "moe.out".into(),
            hidden_dim: self.hidden,
            intermediate_dim: self.intermediate,
            experts: self.experts,
            top_k: self.top_k,
            activation: MoeActivation::SwiGlu,
        }))?;
        api.emit(PrimitiveOp::ResidualAdd(ResidualAdd {
            name: "layers.0.moe_residual".to_string(),
            lhs: "hidden.0".into(),
            rhs: "moe.out".into(),
            output: "hidden.1".into(),
        }))?;
        api.end_stage()?;

        api.begin_stage("output", ModelStageKind::Output)?;
        api.emit(PrimitiveOp::Linear(Linear {
            name: "lm_head".to_string(),
            input: "hidden.1".into(),
            weight: "lm_head.weight".into(),
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
            name: "greedy_argmax".to_string(),
            logits: "logits".into(),
            output_token: "next_token".into(),
            vocab: self.vocab,
        }))?;
        api.end_stage()?;

        Ok(())
    }
}

pub fn external_cpu_demo_live_aql_proof_validations(
) -> Result<[RuntimeLaunchLiveAqlProofValidation; 2]> {
    Ok([
        RuntimeLaunchLiveAqlProofKind::BatchReservationPlan
            .validate_batch_reservation_plan_proof(live_aql_batch_plan_proof())?,
        RuntimeLaunchLiveAqlProofKind::MaterializedPacketPlan
            .validate_materialized_packet_plan_proof(live_aql_materialized_packet_plan_proof())?,
    ])
}

fn declare(
    api: &mut dyn ModelPrimitiveApi,
    id: impl Into<TensorRef>,
    dtype: DType,
    shape: Vec<usize>,
    role: TensorRole,
) -> Result<()> {
    api.declare_tensor(TensorSpec::new(id, dtype, shape, role))
}

fn declare_weight(
    api: &mut dyn ModelPrimitiveApi,
    id: impl Into<TensorRef>,
    checkpoint_key: impl Into<String>,
    dtype: DType,
    shape: Vec<usize>,
) -> Result<()> {
    api.declare_tensor(
        TensorSpec::new(id, dtype, shape, TensorRole::Weight)
            .with_checkpoint_key(checkpoint_key)?,
    )
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

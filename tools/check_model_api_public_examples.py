#!/usr/bin/env python3
"""Run and verify the CPU-only public model API example gates."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEX64 = r"[0-9a-f]{64}"
CONTRACT_RE = re.compile(
    rf"model_api_contract: mainarch-model-api version=0\.1\.0 "
    rf"stability=pre1-static-metadata live_execution_supported=false "
    rf"receipt_fingerprint=({HEX64}) lines=6"
)
MAINARCH_CORE_PATH_RE = re.compile(r"\bmainarch_core\s*::")
MAINARCH_CORE_USE_RE = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?use\s+mainarch_core\b")
MAINARCH_CORE_EXTERN_RE = re.compile(r"^(?:pub\s+)?extern\s+crate\s+mainarch_core\b")
SHELL_ENV_PREFIX_RE = r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*"
LAUNCH_EXECUTION_REQUIREMENTS = (
    "kernel_candidate_selection_policy,host_launcher_runtime_branch_resolution,"
    "loaded_code_object_base,kernarg_allocation,kernel_argument_abi_verification,"
    "kernel_argument_abi_semantic_projection,completion_signal_binding,"
    "queue_reservation,aql_packet_materialization"
)
LAUNCH_SUBMISSION_GATE_BLOCKERS = (
    f"{LAUNCH_EXECUTION_REQUIREMENTS},runtime_request_components,"
    "live_aql_proof_validation"
)
LAUNCH_EXECUTION_REQUEST_PLANS = (
    "code_object_load_request_plan,code_object_base_binding_request_plan,"
    "completion_signal_binding_request_plan,queue_reservation_request_plan,"
    "kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,"
    "kernel_candidate_selection_request_plan,"
    "kernel_argument_abi_semantic_projection_candidate_selection_request_plan,"
    "host_launcher_branch_resolution_request_plan,"
    "aql_live_relocation_binding_request_plan"
)
LAUNCH_EXECUTION_REQUEST_STEP_REQUIREMENTS = (
    "loaded_code_object_base,loaded_code_object_base,completion_signal_binding,"
    "queue_reservation,kernarg_allocation,kernel_argument_abi_verification,"
    "kernel_candidate_selection_policy,kernel_argument_abi_semantic_projection,"
    "host_launcher_runtime_branch_resolution,aql_packet_materialization"
)
LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS = (
    "queue_reservation_request_plan,aql_live_relocation_binding_request_plan"
)
LAUNCH_LIVE_AQL_PROOF_INPUT_LABELS = (
    "KfdQueueLiveAqlBatchReservationPlanInput,"
    "KfdQueueLiveAqlMaterializedPacketPlanInput"
)
LAUNCH_LIVE_AQL_PROOF_KIND_LABELS = "batch_reservation_plan,materialized_packet_plan"
LAUNCH_LIVE_AQL_VALIDATION_METHOD_LABELS = (
    "KfdQueueLiveAqlBatchReservationPlanProof::validate_ready,"
    "KfdQueueLiveAqlMaterializedPacketPlanProof::validate_ready"
)
LAUNCH_RUNTIME_COMPONENT_NEXT_ACTION_PLANS = (
    "code_object_load_request_plan,code_object_base_binding_request_plan,"
    "completion_signal_binding_request_plan,kernarg_allocation_request_plan,"
    "kernel_argument_abi_schema_request_plan,kernel_candidate_selection_request_plan,"
    "kernel_argument_abi_semantic_projection_candidate_selection_request_plan,"
    "host_launcher_branch_resolution_request_plan"
)
LAUNCH_SUBMISSION_PREREQUISITE_NEXT_ACTION_LABELS = (
    "apply_runtime_request_component,apply_runtime_request_component,"
    "apply_runtime_request_component,validate_live_aql_proof,"
    "apply_runtime_request_component,apply_runtime_request_component,"
    "apply_runtime_request_component,apply_runtime_request_component,"
    "apply_runtime_request_component,validate_live_aql_proof"
)
LAUNCH_SUBMISSION_PREREQUISITE_NEXT_ACTION_INPUT_LABELS = (
    "code_object_load_request_plan,code_object_base_binding_request_plan,"
    "completion_signal_binding_request_plan,KfdQueueLiveAqlBatchReservationPlanInput,"
    "kernarg_allocation_request_plan,kernel_argument_abi_schema_request_plan,"
    "kernel_candidate_selection_request_plan,"
    "kernel_argument_abi_semantic_projection_candidate_selection_request_plan,"
    "host_launcher_branch_resolution_request_plan,"
    "KfdQueueLiveAqlMaterializedPacketPlanInput"
)


@dataclass(frozen=True)
class ExampleGate:
    name: str
    command: tuple[str, ...]
    required_patterns: tuple[re.Pattern[str], ...]
    expected_lines_file: Path | None = None


@dataclass(frozen=True)
class SourceImportSurfaceGuard:
    path: Path
    allowed_mainarch_core_lines: tuple[str, ...]


@dataclass(frozen=True)
class PublicCommandDocGuard:
    path: Path
    commands: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class PublicDocSnippetGuard:
    path: Path
    snippets: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class PublicSourcePatternGuard:
    path: Path
    patterns: tuple[re.Pattern[str], ...]
    label: str


def pattern(regex: str) -> re.Pattern[str]:
    return re.compile(regex)


RUNTIME_LAUNCH_REQUEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS = (
    pattern(r"components\.3\.live_aql_proof_kind=batch_reservation_plan"),
    pattern(r"components\.9\.live_aql_proof_kind=materialized_packet_plan"),
    pattern(r"live_aql_proof_surfaces\.0\.proof_kind=batch_reservation_plan"),
    pattern(r"live_aql_proof_surfaces\.1\.proof_kind=materialized_packet_plan"),
    pattern(
        r"live_aql_proof_surfaces\.0\.proof_type=KfdQueueLiveAqlBatchReservationPlanProof"
    ),
    pattern(
        r"live_aql_proof_surfaces\.1\.proof_type=KfdQueueLiveAqlMaterializedPacketPlanProof"
    ),
    pattern(
        r"live_aql_proof_surfaces\.0\.validation_type=KfdQueueLiveAqlBatchReservationPlanValidation"
    ),
    pattern(
        r"live_aql_proof_surfaces\.1\.validation_type=KfdQueueLiveAqlMaterializedPacketPlanValidation"
    ),
    pattern(r"live_aql_proof_surfaces\.0\.validation_ready_field=ready"),
    pattern(r"live_aql_proof_surfaces\.1\.validation_ready_field=ready"),
    pattern(
        r"live_aql_proof_surfaces\.0\.no_live_queue_mutation_contract_field=no_live_queue_mutation_contract"
    ),
    pattern(
        r"live_aql_proof_surfaces\.1\.no_live_queue_mutation_contract_field=no_live_queue_mutation_contract"
    ),
)

RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS = (
    pattern(r"prerequisites\.3\.live_aql_proof_kind=batch_reservation_plan"),
    pattern(r"prerequisites\.9\.live_aql_proof_kind=materialized_packet_plan"),
)

RUNTIME_SUBMISSION_PREREQUISITE_NEXT_ACTION_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS = (
    pattern(
        r"prerequisites\.3\.next_action_live_aql_proof_kind=batch_reservation_plan"
    ),
    pattern(
        r"prerequisites\.9\.next_action_live_aql_proof_kind=materialized_packet_plan"
    ),
)

PLUGIN_MANIFEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS = (
    pattern(
        r"runtime_launch_request_steps\.3\.live_aql_proof_kind=batch_reservation_plan"
    ),
    pattern(
        r"runtime_launch_request_steps\.9\.live_aql_proof_kind=materialized_packet_plan"
    ),
)


ACCEPTED_PATTERNS = (
    CONTRACT_RE,
    pattern(
        rf"plugin_summary: receipt_fingerprint={HEX64} accepted=true "
        rf"static_ready=true compatibility_issues=0 .*live_execution_supported=false"
    ),
    pattern(
        r"checkpoint_payloads: bound_weights=\d+ expected_payloads=\d+ "
        r"matched_payloads=\d+ residency_proven=\d+ payload_bytes=\d+ "
        r"issues=0 ready=true live_execution_supported=false"
    ),
    pattern(rf"launch_submission_prerequisite_plan_receipt: fingerprint={HEX64} lines=249"),
    pattern(r"launch_executable: ready=false\b"),
    pattern(rf"launch_submission_blocker_report_receipt: fingerprint={HEX64} lines=132"),
    pattern(
        r"launch_executable_blockers: "
        rf"count=9 requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)}"
    ),
    pattern(
        r"launch_executable_requirements: "
        rf"count=9 requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)}"
    ),
    pattern(
        r"launch_execution_request_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"launch_execution_request_pending_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"launch_execution_live_aql_proof_surface_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"launch_execution_pending_live_aql_proof_surface_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"launch_execution_pending_live_aql_proof_validation_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"launch_execution_live_aql_proof_kinds: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)}"
    ),
    pattern(r"launch_execution_live_aql_submitting_surface_plans: count=0 names="),
    pattern(r"launch_execution_live_queue_mutating_component_plans: count=0 names="),
    pattern(
        r"launch_execution_live_aql_proof_inputs: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_INPUT_LABELS)}"
    ),
    pattern(
        r"launch_execution_live_aql_validation_methods: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_VALIDATION_METHOD_LABELS)}"
    ),
    pattern(
        r"launch_submission_gate_blockers: "
        rf"count=11 requirements={re.escape(LAUNCH_SUBMISSION_GATE_BLOCKERS)}"
    ),
    pattern(
        r"launch_submission_blocker_report_blockers: "
        rf"count=11 requirements={re.escape(LAUNCH_SUBMISSION_GATE_BLOCKERS)}"
    ),
    pattern(
        r"launch_submission_blocker_report_execution_readiness_blockers: "
        rf"count=9 requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)}"
    ),
    pattern(
        r"launch_submission_blocker_report_runtime_component_blockers: "
        r"count=1 requirements=runtime_request_components"
    ),
    pattern(
        r"launch_submission_blocker_report_live_aql_proof_validation_blockers: "
        r"count=1 requirements=live_aql_proof_validation"
    ),
    pattern(
        r"launch_submission_blocker_report_live_aql_submission_side_effect_blockers: "
        r"count=0 requirements="
    ),
    pattern(
        r"launch_submission_blocker_report_live_queue_mutation_blockers: "
        r"count=0 requirements="
    ),
    pattern(
        r"launch_submission_prerequisite_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_unsatisfied_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_next_action_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_next_action_labels: "
        rf"count=10 labels={re.escape(LAUNCH_SUBMISSION_PREREQUISITE_NEXT_ACTION_LABELS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_runtime_component_next_action_plans: "
        rf"count=8 names={re.escape(LAUNCH_RUNTIME_COMPONENT_NEXT_ACTION_PLANS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_live_aql_proof_validation_next_action_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_next_action_inputs: "
        rf"count=10 labels={re.escape(LAUNCH_SUBMISSION_PREREQUISITE_NEXT_ACTION_INPUT_LABELS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_next_action_live_aql_proof_kinds: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_live_aql_proof_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_live_aql_proof_kinds: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)}"
    ),
    pattern(r"launch_submission_prerequisite_live_aql_submitting_plans: count=0 names="),
    pattern(
        r"launch_submission_prerequisite_pending_live_aql_proof_validation_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(r"launch_submission_prerequisite_live_queue_mutating_plans: count=0 names="),
    pattern(
        r"launch_submission_prerequisite_live_aql_proof_inputs: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_INPUT_LABELS)}"
    ),
    pattern(
        r"launch_submission_prerequisite_live_aql_validation_methods: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_VALIDATION_METHOD_LABELS)}"
    ),
)

REFERENCE_SEMANTIC_MISSING_SCHEMA_SYMBOLS = ""
REFERENCE_PROJECTION_MISSING_SCHEMA_SYMBOLS = ""
REFERENCE_SEMANTIC_MISSING_MODEL_ARGUMENTS = (
    "allreduce_sequence_base,append_count,base_pos,batch_indices,batch_size,"
    "block_size,candidate_logits,candidate_token_ids,chunk_len,chunk_len_vec4,"
    "chunk_offset,chunk_offset_vec4,chunk_owner,dualpath_cu_chunk_len_vec4,"
    "dualpath_cu_owned_chunk_len_vec4,dualpath_sdma_semaphores,eps,"
    "expert_ids_history,expert_weights_history,gbar,group_size,history_steps,"
    "indptr,intermediate,last_page_len,layer,local_allreduce_buffer,"
    "local_allreduce_flags,nthreads,num_groups,num_layers,num_splits,num_tiles,"
    "num_wg,p2p_peer_count,partial,partials,peer_allreduce_flag_ptrs,"
    "peer_allreduce_ptrs,peer_reduce_staging_ptrs,peer_residual_ptrs,"
    "peer_stride_vec4,persistent_allreduce_ctrl,persistent_allreduce_output,"
    "persistent_allreduce_total_ops,physical_blocks,positions,q_heads_per_kv,"
    "reduce_staging_buffer,reduce_staging_chunk_len,rmsnorm_output,"
    "rmsnorm_weight,scale,scale_k,scale_v,self_rank,seq_lens,slot,src_scale_k,"
    "src_scale_v,step,token_ids,total_indices"
)
REFERENCE_PROJECTION_MISSING_MODEL_ARGUMENTS = (
    "allreduce_sequence_base,append_count,base_pos,batch_indices,batch_size,"
    "block_size,candidate_logits,candidate_token_ids,chunk_len,chunk_len_vec4,"
    "chunk_offset,chunk_offset_vec4,chunk_owner,dualpath_cu_chunk_len_vec4,"
    "dualpath_cu_owned_chunk_len_vec4,dualpath_sdma_semaphores,eps,gbar,indptr,"
    "intermediate,last_page_len,local_allreduce_buffer,local_allreduce_flags,"
    "nthreads,num_groups,num_splits,num_tiles,num_wg,p2p_peer_count,partials,"
    "peer_allreduce_flag_ptrs,peer_allreduce_ptrs,peer_reduce_staging_ptrs,"
    "peer_stride_vec4,persistent_allreduce_ctrl,persistent_allreduce_output,"
    "persistent_allreduce_total_ops,physical_blocks,positions,"
    "reduce_staging_buffer,reduce_staging_chunk_len,rmsnorm_output,"
    "rmsnorm_weight,scale,scale_k,scale_v,self_rank,seq_lens,slot,src_scale_k,"
    "src_scale_v,step,token_ids,total_indices"
)
CUSTOM_MISSING_MODEL_ARGUMENTS = (
    "base_pos,block_size,candidate_logits,candidate_token_ids,eps,last_page_len,"
    "positions,rmsnorm_output,rmsnorm_weight,seq_lens,slot,step,token_ids"
)
EXTERNAL_PROJECTION_MISSING_MODEL_ARGUMENTS = (
    "base_pos,block_size,candidate_logits,candidate_token_ids,eps,intermediate,"
    "last_page_len,nthreads,positions,rmsnorm_output,rmsnorm_weight,seq_lens,"
    "slot,step,token_ids"
)
REFERENCE_PROJECTION_SELECTION_READY_OPS = (
    "layers.0.input_rmsnorm,layers.0.q_proj,layers.0.k_proj,layers.0.v_proj,"
    "layers.0.q_rmsnorm,layers.0.k_rmsnorm,layers.0.o_proj,"
    "layers.0.attention_residual_rmsnorm,layers.0.router_topk,"
    "layers.1.input_rmsnorm,layers.1.q_proj,layers.1.k_proj,layers.1.v_proj,"
    "layers.1.q_rmsnorm,layers.1.k_rmsnorm,layers.1.o_proj,"
    "layers.1.attention_residual_rmsnorm,layers.1.router_topk,final_rmsnorm,lm_head"
)
REFERENCE_PROJECTION_SELECTION_REQUESTED_SYMBOLS = (
    "layers.0.input_rmsnorm=rmsnorm_f16,layers.0.q_proj=gemv_f16,"
    "layers.0.k_proj=gemv_f16,layers.0.v_proj=gemv_f16,"
    "layers.0.q_rmsnorm=rmsnorm_f16,layers.0.k_rmsnorm=rmsnorm_f16,"
    "layers.0.o_proj=gemv_f16,"
    "layers.0.attention_residual_rmsnorm=add_rmsnorm_f16,"
    "layers.0.router_topk=moe_router_topk,"
    "layers.1.input_rmsnorm=rmsnorm_f16,layers.1.q_proj=gemv_f16,"
    "layers.1.k_proj=gemv_f16,layers.1.v_proj=gemv_f16,"
    "layers.1.q_rmsnorm=rmsnorm_f16,layers.1.k_rmsnorm=rmsnorm_f16,"
    "layers.1.o_proj=gemv_f16,"
    "layers.1.attention_residual_rmsnorm=add_rmsnorm_f16,"
    "layers.1.router_topk=moe_router_topk,final_rmsnorm=rmsnorm_f16,"
    "lm_head=gemv_f16"
)
REFERENCE_PROJECTION_SELECTION_MISSING_OPS = (
    "embed_tokens,layers.0.rope,layers.0.kv_cache_append,"
    "layers.0.paged_gqa_attention,layers.0.o_proj_allreduce,"
    "layers.0.moe_local_ffn,layers.0.moe_allreduce,layers.0.moe_residual,"
    "layers.1.rope,layers.1.kv_cache_append,layers.1.paged_gqa_attention,"
    "layers.1.o_proj_allreduce,layers.1.moe_local_ffn,layers.1.moe_allreduce,"
    "layers.1.moe_residual,greedy_argmax"
)
CUSTOM_PROJECTION_SELECTION_READY_OPS = "lm_head"
CUSTOM_PROJECTION_SELECTION_REQUESTED_SYMBOLS = "lm_head=gemv_f16"
CUSTOM_PROJECTION_SELECTION_MISSING_OPS = "embed_tokens,sample_argmax"
EXTERNAL_PROJECTION_SELECTION_READY_OPS = "layers.0.router_topk,lm_head"
EXTERNAL_PROJECTION_SELECTION_REQUESTED_SYMBOLS = (
    "layers.0.router_topk=moe_router_topk,lm_head=gemv_f16"
)
EXTERNAL_PROJECTION_SELECTION_MISSING_OPS = (
    "embed_tokens,layers.0.moe_local_ffn,layers.0.moe_residual,greedy_argmax"
)
REFERENCE_KERNEL_SELECTION_READY_OPS = (
    "embed_tokens,layers.0.kv_cache_append,layers.0.paged_gqa_attention,"
    "layers.0.o_proj_allreduce,layers.0.attention_residual_rmsnorm,"
    "layers.0.router_topk,layers.0.moe_allreduce,layers.1.kv_cache_append,"
    "layers.1.paged_gqa_attention,"
    "layers.1.o_proj_allreduce,layers.1.attention_residual_rmsnorm,"
    "layers.1.router_topk,layers.1.moe_allreduce,greedy_argmax"
)
REFERENCE_KERNEL_SELECTION_REQUESTED_SYMBOLS = (
    "embed_tokens=decode_step_embed_rmsnorm_token_f16,"
    "layers.0.kv_cache_append=kv_append_paged_fp4,"
    "layers.0.paged_gqa_attention=attn_decode_split2_fp4_gqa_paged_groups_meta,"
    "layers.0.o_proj_allreduce=reduce_peers,"
    "layers.0.attention_residual_rmsnorm=allreduce_direct_residual_rmsnorm_grid,"
    "layers.0.router_topk=moe_router_gemv_topk_log_step,"
    "layers.0.moe_allreduce=reduce_peers,"
    "layers.1.kv_cache_append=kv_append_paged_fp4,"
    "layers.1.paged_gqa_attention=attn_decode_split2_fp4_gqa_paged_groups_meta,"
    "layers.1.o_proj_allreduce=reduce_peers,"
    "layers.1.attention_residual_rmsnorm=allreduce_direct_residual_rmsnorm_grid,"
    "layers.1.router_topk=moe_router_gemv_topk_log_step,"
    "layers.1.moe_allreduce=reduce_peers,greedy_argmax=argmax_f32_step"
)
REFERENCE_KERNEL_SELECTION_MISSING_OPS = (
    "layers.0.input_rmsnorm,layers.0.q_proj,layers.0.k_proj,layers.0.v_proj,"
    "layers.0.q_rmsnorm,layers.0.k_rmsnorm,layers.0.rope,layers.0.o_proj,"
    "layers.0.moe_local_ffn,layers.0.moe_residual,layers.1.input_rmsnorm,"
    "layers.1.q_proj,layers.1.k_proj,layers.1.v_proj,layers.1.q_rmsnorm,"
    "layers.1.k_rmsnorm,layers.1.rope,layers.1.o_proj,layers.1.moe_local_ffn,"
    "layers.1.moe_residual,final_rmsnorm,lm_head"
)
CUSTOM_KERNEL_SELECTION_READY_OPS = "embed_tokens,sample_argmax"
CUSTOM_KERNEL_SELECTION_REQUESTED_SYMBOLS = (
    "embed_tokens=decode_step_embed_rmsnorm_token_f16,"
    "sample_argmax=argmax_f32_step"
)
CUSTOM_KERNEL_SELECTION_MISSING_OPS = "lm_head"
EXTERNAL_KERNEL_SELECTION_READY_OPS = (
    "embed_tokens,layers.0.router_topk,greedy_argmax"
)
EXTERNAL_KERNEL_SELECTION_REQUESTED_SYMBOLS = (
    "embed_tokens=decode_step_embed_rmsnorm_token_f16,"
    "layers.0.router_topk=moe_router_gemv_topk_log_step,"
    "greedy_argmax=argmax_f32_step"
)
EXTERNAL_KERNEL_SELECTION_MISSING_OPS = "layers.0.moe_local_ffn,layers.0.moe_residual,lm_head"
REFERENCE_HOST_LAUNCHER_BRANCH_REQUEST_OPS = (
    "layers.0.q_proj,layers.0.k_proj,layers.0.v_proj,layers.0.kv_cache_append,"
    "layers.0.paged_gqa_attention,layers.0.o_proj,layers.0.o_proj_allreduce,"
    "layers.0.attention_residual_rmsnorm,layers.0.router_topk,"
    "layers.0.moe_local_ffn,layers.0.moe_allreduce,layers.1.q_proj,"
    "layers.1.k_proj,layers.1.v_proj,layers.1.kv_cache_append,"
    "layers.1.paged_gqa_attention,layers.1.o_proj,layers.1.o_proj_allreduce,"
    "layers.1.attention_residual_rmsnorm,layers.1.router_topk,"
    "layers.1.moe_local_ffn,layers.1.moe_allreduce,lm_head,greedy_argmax"
)
CUSTOM_HOST_LAUNCHER_BRANCH_REQUEST_OPS = "lm_head,sample_argmax"
EXTERNAL_HOST_LAUNCHER_BRANCH_REQUEST_OPS = (
    "layers.0.router_topk,layers.0.moe_local_ffn,lm_head,greedy_argmax"
)
HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS = (
    "gemv_f16|gemv_f16_k8192|gemv_f16_step|gemv_f16_step_k4096"
)
HOST_LAUNCHER_BRANCH_KV_APPEND_CANDIDATE_SYMBOLS = (
    "kv_append_paged_fp4|kv_append_paged_fp4_from_f16_vf32_heads"
)
HOST_LAUNCHER_BRANCH_GQA_CANDIDATE_SYMBOLS = (
    "attn_decode_split2_fp4_gqa_paged|"
    "attn_decode_split2_fp4_gqa_paged_groups_meta|attn_decode_combine_gqa_f16"
)
HOST_LAUNCHER_BRANCH_ALLREDUCE_CANDIDATE_SYMBOLS = (
    "reduce_peers|broadcast_peers|broadcast_peers_skip0|allreduce_oneshot|"
    "scatter_to_staging|gather_reduce_local|reduce_scatter|broadcast_chunk|"
    "broadcast_chunk_skip_owner|all_gather|p2p_write|p2p_broadcast|"
    "allreduce_dualpath|allreduce_dda_persistent|allreduce_direct_persistent"
)
HOST_LAUNCHER_BRANCH_RESIDUAL_RMSNORM_CANDIDATE_SYMBOLS = (
    "add_rmsnorm_f16|add_rmsnorm_bf16_residual_f16_out|"
    "allreduce_direct_residual_rmsnorm_grid"
)
HOST_LAUNCHER_BRANCH_ROUTER_CANDIDATE_SYMBOLS = (
    "moe_router_topk|moe_router_gemv_topk_log_step|"
    "moe_router_gemv_topk_log_step_e16_k4096_top8"
)
HOST_LAUNCHER_BRANCH_MOE_FFN_CANDIDATE_SYMBOLS = (
    "moe_gate_up_swiglu|moe_gate_up_swiglu_slots|"
    "moe_gate_up_swiglu_slots_k4096|moe_down_accum|moe_down_accum_slots|"
    "moe_down_accum_slots_i1536|moe_down_accum_slots_i512"
)
HOST_LAUNCHER_BRANCH_ARGMAX_CANDIDATE_SYMBOLS = (
    "argmax_f32_step|argmax_f32_token_ids_write_candidate|"
    "argmax_f32_token_ids_write_candidate_n1187"
)
REFERENCE_HOST_LAUNCHER_BRANCH_CANDIDATE_SYMBOL_LABELS = (
    f"layers.0.q_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.0.k_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.0.v_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.0.kv_cache_append={HOST_LAUNCHER_BRANCH_KV_APPEND_CANDIDATE_SYMBOLS},"
    f"layers.0.paged_gqa_attention={HOST_LAUNCHER_BRANCH_GQA_CANDIDATE_SYMBOLS},"
    f"layers.0.o_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.0.o_proj_allreduce={HOST_LAUNCHER_BRANCH_ALLREDUCE_CANDIDATE_SYMBOLS},"
    "layers.0.attention_residual_rmsnorm="
    f"{HOST_LAUNCHER_BRANCH_RESIDUAL_RMSNORM_CANDIDATE_SYMBOLS},"
    f"layers.0.router_topk={HOST_LAUNCHER_BRANCH_ROUTER_CANDIDATE_SYMBOLS},"
    f"layers.0.moe_local_ffn={HOST_LAUNCHER_BRANCH_MOE_FFN_CANDIDATE_SYMBOLS},"
    f"layers.0.moe_allreduce={HOST_LAUNCHER_BRANCH_ALLREDUCE_CANDIDATE_SYMBOLS},"
    f"layers.1.q_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.1.k_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.1.v_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.1.kv_cache_append={HOST_LAUNCHER_BRANCH_KV_APPEND_CANDIDATE_SYMBOLS},"
    f"layers.1.paged_gqa_attention={HOST_LAUNCHER_BRANCH_GQA_CANDIDATE_SYMBOLS},"
    f"layers.1.o_proj={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"layers.1.o_proj_allreduce={HOST_LAUNCHER_BRANCH_ALLREDUCE_CANDIDATE_SYMBOLS},"
    "layers.1.attention_residual_rmsnorm="
    f"{HOST_LAUNCHER_BRANCH_RESIDUAL_RMSNORM_CANDIDATE_SYMBOLS},"
    f"layers.1.router_topk={HOST_LAUNCHER_BRANCH_ROUTER_CANDIDATE_SYMBOLS},"
    f"layers.1.moe_local_ffn={HOST_LAUNCHER_BRANCH_MOE_FFN_CANDIDATE_SYMBOLS},"
    f"layers.1.moe_allreduce={HOST_LAUNCHER_BRANCH_ALLREDUCE_CANDIDATE_SYMBOLS},"
    f"lm_head={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"greedy_argmax={HOST_LAUNCHER_BRANCH_ARGMAX_CANDIDATE_SYMBOLS}"
)
CUSTOM_HOST_LAUNCHER_BRANCH_CANDIDATE_SYMBOL_LABELS = (
    f"lm_head={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"sample_argmax={HOST_LAUNCHER_BRANCH_ARGMAX_CANDIDATE_SYMBOLS}"
)
EXTERNAL_HOST_LAUNCHER_BRANCH_CANDIDATE_SYMBOL_LABELS = (
    f"layers.0.router_topk={HOST_LAUNCHER_BRANCH_ROUTER_CANDIDATE_SYMBOLS},"
    f"layers.0.moe_local_ffn={HOST_LAUNCHER_BRANCH_MOE_FFN_CANDIDATE_SYMBOLS},"
    f"lm_head={HOST_LAUNCHER_BRANCH_GEMV_CANDIDATE_SYMBOLS},"
    f"greedy_argmax={HOST_LAUNCHER_BRANCH_ARGMAX_CANDIDATE_SYMBOLS}"
)
REFERENCE_HOST_LAUNCHER_BRANCH_UNRESOLVED_CANDIDATE_SYMBOLS = (
    "add_rmsnorm_bf16_residual_f16_out,add_rmsnorm_f16,all_gather,"
    "allreduce_dda_persistent,allreduce_direct_persistent,"
    "allreduce_direct_residual_rmsnorm_grid,allreduce_dualpath,"
    "allreduce_oneshot,argmax_f32_step,argmax_f32_token_ids_write_candidate,"
    "argmax_f32_token_ids_write_candidate_n1187,attn_decode_combine_gqa_f16,"
    "attn_decode_split2_fp4_gqa_paged,"
    "attn_decode_split2_fp4_gqa_paged_groups_meta,broadcast_chunk,"
    "broadcast_chunk_skip_owner,broadcast_peers,broadcast_peers_skip0,"
    "gather_reduce_local,gemv_f16,gemv_f16_k8192,gemv_f16_step,"
    "gemv_f16_step_k4096,kv_append_paged_fp4,"
    "kv_append_paged_fp4_from_f16_vf32_heads,moe_down_accum,"
    "moe_down_accum_slots,moe_down_accum_slots_i1536,"
    "moe_down_accum_slots_i512,moe_gate_up_swiglu,moe_gate_up_swiglu_slots,"
    "moe_gate_up_swiglu_slots_k4096,moe_router_gemv_topk_log_step,"
    "moe_router_gemv_topk_log_step_e16_k4096_top8,moe_router_topk,"
    "p2p_broadcast,p2p_write,reduce_peers,reduce_scatter,scatter_to_staging"
)
CUSTOM_HOST_LAUNCHER_BRANCH_UNRESOLVED_CANDIDATE_SYMBOLS = (
    "argmax_f32_step,argmax_f32_token_ids_write_candidate,"
    "argmax_f32_token_ids_write_candidate_n1187,gemv_f16,gemv_f16_k8192,"
    "gemv_f16_step,gemv_f16_step_k4096"
)
EXTERNAL_HOST_LAUNCHER_BRANCH_UNRESOLVED_CANDIDATE_SYMBOLS = (
    "argmax_f32_step,argmax_f32_token_ids_write_candidate,"
    "argmax_f32_token_ids_write_candidate_n1187,gemv_f16,gemv_f16_k8192,"
    "gemv_f16_step,gemv_f16_step_k4096,moe_down_accum,"
    "moe_down_accum_slots,moe_down_accum_slots_i1536,"
    "moe_down_accum_slots_i512,moe_gate_up_swiglu,moe_gate_up_swiglu_slots,"
    "moe_gate_up_swiglu_slots_k4096,moe_router_gemv_topk_log_step,"
    "moe_router_gemv_topk_log_step_e16_k4096_top8,moe_router_topk"
)

REFERENCE_MOE_GAP_SYMBOL_PATTERNS = (
    pattern(
        r"launch_kernarg_abi_semantic_missing_schema_symbols: "
        rf"count=0 symbols={re.escape(REFERENCE_SEMANTIC_MISSING_SCHEMA_SYMBOLS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_missing_schema_symbols: "
        rf"count=0 symbols={re.escape(REFERENCE_PROJECTION_MISSING_SCHEMA_SYMBOLS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_missing_model_arguments: "
        rf"count=63 names={re.escape(REFERENCE_SEMANTIC_MISSING_MODEL_ARGUMENTS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_missing_model_arguments: "
        rf"count=54 names={re.escape(REFERENCE_PROJECTION_MISSING_MODEL_ARGUMENTS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_candidate_selection_ready_ops: "
        rf"count=20 names={re.escape(REFERENCE_PROJECTION_SELECTION_READY_OPS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_candidate_selection_requested_symbols: "
        rf"count=20 labels={re.escape(REFERENCE_PROJECTION_SELECTION_REQUESTED_SYMBOLS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_candidate_selection_missing_ops: "
        rf"count=16 names={re.escape(REFERENCE_PROJECTION_SELECTION_MISSING_OPS)}"
    ),
    pattern(
        r"launch_kernel_selection_ready_ops: "
        rf"count=14 names={re.escape(REFERENCE_KERNEL_SELECTION_READY_OPS)}"
    ),
    pattern(
        r"launch_kernel_selection_requested_symbols: "
        rf"count=14 labels={re.escape(REFERENCE_KERNEL_SELECTION_REQUESTED_SYMBOLS)}"
    ),
    pattern(
        r"launch_kernel_selection_missing_ops: "
        rf"count=22 names={re.escape(REFERENCE_KERNEL_SELECTION_MISSING_OPS)}"
    ),
    pattern(
        r"launch_host_launcher_branch_request_ops: "
        rf"count=24 names={re.escape(REFERENCE_HOST_LAUNCHER_BRANCH_REQUEST_OPS)}"
    ),
    pattern(
        r"launch_host_launcher_branch_candidate_symbols: "
        rf"count=24 labels={re.escape(REFERENCE_HOST_LAUNCHER_BRANCH_CANDIDATE_SYMBOL_LABELS)}"
    ),
    pattern(
        r"launch_host_launcher_branch_unresolved_candidate_symbols: "
        rf"count=40 symbols={re.escape(REFERENCE_HOST_LAUNCHER_BRANCH_UNRESOLVED_CANDIDATE_SYMBOLS)}"
    ),
)

CUSTOM_MODEL_GAP_SYMBOL_PATTERNS = (
    pattern(r"launch_kernarg_abi_semantic_missing_schema_symbols: count=0 symbols="),
    pattern(
        r"launch_kernarg_abi_semantic_projection_missing_schema_symbols: count=0 symbols="
    ),
    pattern(
        r"launch_kernarg_abi_semantic_missing_model_arguments: "
        rf"count=13 names={re.escape(CUSTOM_MISSING_MODEL_ARGUMENTS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_missing_model_arguments: "
        rf"count=13 names={re.escape(CUSTOM_MISSING_MODEL_ARGUMENTS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_candidate_selection_ready_ops: "
        rf"count=1 names={re.escape(CUSTOM_PROJECTION_SELECTION_READY_OPS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_candidate_selection_requested_symbols: "
        rf"count=1 labels={re.escape(CUSTOM_PROJECTION_SELECTION_REQUESTED_SYMBOLS)}"
    ),
    pattern(
        r"launch_kernarg_abi_semantic_projection_candidate_selection_missing_ops: "
        rf"count=2 names={re.escape(CUSTOM_PROJECTION_SELECTION_MISSING_OPS)}"
    ),
    pattern(
        r"launch_kernel_selection_ready_ops: "
        rf"count=2 names={re.escape(CUSTOM_KERNEL_SELECTION_READY_OPS)}"
    ),
    pattern(
        r"launch_kernel_selection_requested_symbols: "
        rf"count=2 labels={re.escape(CUSTOM_KERNEL_SELECTION_REQUESTED_SYMBOLS)}"
    ),
    pattern(
        r"launch_kernel_selection_missing_ops: "
        rf"count=1 names={re.escape(CUSTOM_KERNEL_SELECTION_MISSING_OPS)}"
    ),
    pattern(
        r"launch_host_launcher_branch_request_ops: "
        rf"count=2 names={re.escape(CUSTOM_HOST_LAUNCHER_BRANCH_REQUEST_OPS)}"
    ),
    pattern(
        r"launch_host_launcher_branch_candidate_symbols: "
        rf"count=2 labels={re.escape(CUSTOM_HOST_LAUNCHER_BRANCH_CANDIDATE_SYMBOL_LABELS)}"
    ),
    pattern(
        r"launch_host_launcher_branch_unresolved_candidate_symbols: "
        rf"count=7 symbols={re.escape(CUSTOM_HOST_LAUNCHER_BRANCH_UNRESOLVED_CANDIDATE_SYMBOLS)}"
    ),
)

CHECKPOINT_METADATA_PATTERNS = (
    pattern(r"model: reference-moe-checkpoint-metadata"),
    pattern(r"checkpoint: bound_weights=15 checkpoint_bytes=107456"),
    pattern(
        r"safetensors: tensor_headers=24 matched_tensors=24 "
        r"checked_entries=15 mismatches=0 file_bytes=109913"
    ),
    pattern(
        r"safetensors_index: opened_shards=1 missing_shards=0 "
        r"checked_entries=15 mismatches=0"
    ),
    pattern(
        r"safetensors_payload_spans: shard_bindings=24 index_bindings=24 "
        r"payload_bytes=107456 first_offset=2457 header_only=true"
    ),
    pattern(
        r"checkpoint_payload_direct_reads: work_orders=24 sources=1 slots=15 "
        r"payload_bytes=107456 aligned_read_bytes=204800 max_staging_window=20480 "
        r"direct_io_alignment=4096 cpu_only=true"
    ),
    pattern(
        r"checkpoint_payload_staging_batches: batches=1 pieces=24 sources=1 slots=15 "
        r"staging_slots=2 staging_bytes=221184 read_bytes=110592 "
        r"max_batch_read_bytes=110592 read_amplification_milli=1029 cpu_only=true"
    ),
    pattern(
        r"checkpoint_payload_staging_receipt: "
        r"fingerprint=b710ed4ae9d8b89d27f97f50fb2de562e0dfbbb8271181eb9dc37d54594ab086 "
        r"lines=24 non_executing=true live_execution_supported=false cpu_only=true"
    ),
)

EXTERNAL_PLUGIN_PATTERNS = (
    CONTRACT_RE,
    pattern(
        r"external_plugin: model=external-mini-moe "
        r"package=examples/model-api-plugin imported_prelude=true"
    ),
    pattern(
        rf"plugin_summary: receipt_fingerprint={HEX64} accepted=true "
        rf"static_ready=true compatibility_issues=0 model_primitives=6 model_stages=4 "
        rf"tensors=14 ops=6 dispatches=6 catalog_cases=22 catalog_gaps=5 "
        rf"live_execution_supported=false"
    ),
    pattern(
        rf"plugin_manifest: contract=mainarch-model-api fingerprint={HEX64} "
        rf"version=0\.1\.0 stability=pre1-static-metadata "
        rf"target=mi355x-gfx950-raw-kfd-aql primitive_kinds=6 stage_kinds=4 "
        rf"tensors=14 ops=6 stages=4 checkpoint_weights=6 slots=14 "
        rf"dispatches=6 launch_steps=10 live_aql_proof_steps=2 static_ready=true "
        rf"live_execution_supported=false"
    ),
    pattern(
        r"external_static_launch_request_steps: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)} "
        rf"requirements={re.escape(LAUNCH_EXECUTION_REQUEST_STEP_REQUIREMENTS)}"
    ),
    pattern(
        r"external_static_live_aql_proof_steps: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)} "
        rf"proof_kinds={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)} "
        rf"proof_inputs={re.escape(LAUNCH_LIVE_AQL_PROOF_INPUT_LABELS)} "
        rf"validation_methods={re.escape(LAUNCH_LIVE_AQL_VALIDATION_METHOD_LABELS)}"
    ),
    pattern(
        r"plugin_compatibility: accepted=true issues=0 target_matches=true "
        r"fingerprint_matches=true static_metadata_ready=true "
        r"live_execution_supported=false"
    ),
    pattern(r"graph: tensors=14 ops=6 stages=4 staged_ops=6 unstaged_ops=0"),
    pattern(
        r"checkpoint_payloads: bound_weights=6 expected_payloads=15 "
        r"matched_payloads=15 residency_proven=6 payload_bytes=328704 "
        r"issues=0 ready=true live_execution_supported=false"
    ),
    pattern(
        r"external_launch_projection: candidates=19 schema_candidates=19 "
        r"missing_schema_candidates=0 projection_ready_candidates=3 "
        r"dispatches_with_ready=2 dispatches_without_ready=4 "
        r"projected_kernarg_bytes=908 ready=false"
    ),
    pattern(
        r"external_launch_projection_missing_model_arguments: "
        rf"count=15 names={re.escape(EXTERNAL_PROJECTION_MISSING_MODEL_ARGUMENTS)}"
    ),
    pattern(
        r"external_projection_selection: requests=2 missing=4 "
        r"requested_projected_kernarg_bytes=64 applied=0 all_ready=false "
        r"plan_ready=true policy=first_projection_ready_candidate_in_host_launcher_order"
    ),
    pattern(
        r"external_projection_selection_ready_ops: "
        rf"count=2 names={re.escape(EXTERNAL_PROJECTION_SELECTION_READY_OPS)}"
    ),
    pattern(
        r"external_projection_selection_requested_symbols: "
        rf"count=2 labels={re.escape(EXTERNAL_PROJECTION_SELECTION_REQUESTED_SYMBOLS)}"
    ),
    pattern(
        r"external_projection_selection_missing_ops: "
        rf"count=4 names={re.escape(EXTERNAL_PROJECTION_SELECTION_MISSING_OPS)}"
    ),
    pattern(
        r"external_kernel_selection: requests=3 missing=3 "
        r"verified_candidates=6 applied=0 all_ready=false plan_ready=true "
        r"policy=first_verified_candidate_in_host_launcher_order"
    ),
    pattern(
        r"external_kernel_selection_ready_ops: "
        rf"count=3 names={re.escape(EXTERNAL_KERNEL_SELECTION_READY_OPS)}"
    ),
    pattern(
        r"external_kernel_selection_requested_symbols: "
        rf"count=3 labels={re.escape(EXTERNAL_KERNEL_SELECTION_REQUESTED_SYMBOLS)}"
    ),
    pattern(
        r"external_kernel_selection_missing_ops: "
        rf"count=3 names={re.escape(EXTERNAL_KERNEL_SELECTION_MISSING_OPS)}"
    ),
    pattern(
        r"external_host_launcher_branch_requests: requests=4 applied=0 "
        r"unresolved_candidates=17 all_resolved=false plan_ready=true"
    ),
    pattern(
        r"external_host_launcher_branch_request_ops: "
        rf"count=4 names={re.escape(EXTERNAL_HOST_LAUNCHER_BRANCH_REQUEST_OPS)}"
    ),
    pattern(
        r"external_host_launcher_branch_candidate_symbols: "
        rf"count=4 labels={re.escape(EXTERNAL_HOST_LAUNCHER_BRANCH_CANDIDATE_SYMBOL_LABELS)}"
    ),
    pattern(
        r"external_host_launcher_branch_unresolved_candidate_symbols: "
        rf"count=17 symbols={re.escape(EXTERNAL_HOST_LAUNCHER_BRANCH_UNRESOLVED_CANDIDATE_SYMBOLS)}"
    ),
    pattern(
        r"external_launch_execution: executable=false blockers=9 "
        r"unresolved_runtime_requirements=9 projection_selection_requests=2 "
        r"projection_selection_missing=4 aql_dispatchable_packets=0 "
        r"live_aql_submitting_surfaces=0 live_queue_mutating_components=0"
    ),
    pattern(
        r"external_launch_execution_request_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"external_launch_execution_request_pending_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"external_launch_execution_live_aql_proof_surface_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"external_launch_execution_pending_live_aql_proof_surface_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"external_launch_execution_pending_live_aql_proof_validation_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"external_launch_execution_live_aql_proof_kinds: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)}"
    ),
    pattern(r"external_launch_execution_live_aql_submitting_surface_plans: count=0 names="),
    pattern(r"external_launch_execution_live_queue_mutating_component_plans: count=0 names="),
    pattern(
        r"external_launch_execution_live_aql_proof_inputs: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_INPUT_LABELS)}"
    ),
    pattern(
        r"external_launch_execution_live_aql_validation_methods: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_VALIDATION_METHOD_LABELS)}"
    ),
    pattern(
        r"external_submission_gate_blockers: "
        rf"count=11 requirements={re.escape(LAUNCH_SUBMISSION_GATE_BLOCKERS)}"
    ),
    pattern(
        r"external_submission_blocker_report_blockers: "
        rf"count=11 requirements={re.escape(LAUNCH_SUBMISSION_GATE_BLOCKERS)}"
    ),
    pattern(
        r"external_submission_blocker_report_execution_readiness_blockers: "
        rf"count=9 requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)}"
    ),
    pattern(
        r"external_submission_blocker_report_runtime_component_blockers: "
        r"count=1 requirements=runtime_request_components"
    ),
    pattern(
        r"external_submission_blocker_report_live_aql_proof_validation_blockers: "
        r"count=1 requirements=live_aql_proof_validation"
    ),
    pattern(
        r"external_submission_blocker_report_live_aql_submission_side_effect_blockers: "
        r"count=0 requirements="
    ),
    pattern(
        r"external_submission_blocker_report_live_queue_mutation_blockers: "
        r"count=0 requirements="
    ),
    pattern(
        r"external_submission_prerequisite_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"external_submission_prerequisite_unsatisfied_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"external_submission_prerequisite_next_action_plans: "
        rf"count=10 names={re.escape(LAUNCH_EXECUTION_REQUEST_PLANS)}"
    ),
    pattern(
        r"external_submission_prerequisite_next_action_labels: "
        rf"count=10 labels={re.escape(LAUNCH_SUBMISSION_PREREQUISITE_NEXT_ACTION_LABELS)}"
    ),
    pattern(
        r"external_submission_prerequisite_runtime_component_next_action_plans: "
        rf"count=8 names={re.escape(LAUNCH_RUNTIME_COMPONENT_NEXT_ACTION_PLANS)}"
    ),
    pattern(
        r"external_submission_prerequisite_live_aql_proof_validation_next_action_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"external_submission_prerequisite_next_action_inputs: "
        rf"count=10 labels={re.escape(LAUNCH_SUBMISSION_PREREQUISITE_NEXT_ACTION_INPUT_LABELS)}"
    ),
    pattern(
        r"external_submission_prerequisite_next_action_live_aql_proof_kinds: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)}"
    ),
    pattern(
        r"external_submission_prerequisite_live_aql_proof_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(
        r"external_submission_prerequisite_live_aql_proof_kinds: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_KIND_LABELS)}"
    ),
    pattern(r"external_submission_prerequisite_live_aql_submitting_plans: count=0 names="),
    pattern(
        r"external_submission_prerequisite_pending_live_aql_proof_validation_plans: "
        rf"count=2 names={re.escape(LAUNCH_LIVE_AQL_PROOF_SURFACE_PLANS)}"
    ),
    pattern(r"external_submission_prerequisite_live_queue_mutating_plans: count=0 names="),
    pattern(
        r"external_submission_prerequisite_live_aql_proof_inputs: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_PROOF_INPUT_LABELS)}"
    ),
    pattern(
        r"external_submission_prerequisite_live_aql_validation_methods: "
        rf"count=2 labels={re.escape(LAUNCH_LIVE_AQL_VALIDATION_METHOD_LABELS)}"
    ),
    pattern(
        r"external_launch_execution_blockers: "
        rf"count=9 requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)}"
    ),
    pattern(
        r"external_launch_execution_requirements: "
        rf"count=9 requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)}"
    ),
    pattern(
        rf"external_static_handoff: receipt_fingerprint={HEX64} "
        rf"manifest_receipt_fingerprint={HEX64} "
        rf"compatibility_receipt_fingerprint={HEX64} "
        r"accepted=true static_ready=true metadata_admitted=true projection_ready=false "
        r"selection_requests=2 selection_missing=4 executable=false blockers=9 "
        rf"requirements={re.escape(LAUNCH_EXECUTION_REQUIREMENTS)} "
        r"aql_dispatchable_packets=0 live_aql_submitting_surfaces=0 "
        r"live_queue_mutating_components=0 gpu_buffers_allocated=false "
        r"kernels_submitted=false"
    ),
    pattern(
        r"external_plugin_boundary: live_execution_supported=false "
        r"launch_execution_supported=false gpu_buffers_allocated=false kernels_submitted=false"
    ),
)

GATES = (
    ExampleGate(
        name="reference_moe_model_api",
        command=("cargo", "run", "-q", "-p", "mainarch-core", "--example", "reference_moe_model_api"),
        required_patterns=ACCEPTED_PATTERNS + REFERENCE_MOE_GAP_SYMBOL_PATTERNS,
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_launch_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-launch-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_execution_request_plan"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"runtime_request_plan_count=10"),
            pattern(r"component_pending_count=378"),
            pattern(r"live_aql_proof_surface_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"all_components_applied=false"),
        )
        + RUNTIME_LAUNCH_REQUEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-launch-request.receipt",
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_submission_gate_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"all_components_applied=false"),
            pattern(r"all_live_aql_proof_validations_applied=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=378"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=9"),
            pattern(r"submission_blocker_count=11"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-submission-gate.receipt",
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_resolved_submission_gate_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-resolved-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=0"),
            pattern(r"submission_blocker_count=0"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-resolved-submission-gate.receipt",
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_resolved_submission_blocker_report_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-resolved-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"blocker_count=0"),
            pattern(r"execution_readiness_blocker_count=0"),
            pattern(r"runtime_request_component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-resolved-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_resolved_submission_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-resolved-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"prerequisite_count=10"),
            pattern(r"satisfied_prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=0"),
            pattern(r"next_action_count=0"),
            pattern(r"runtime_request_component_next_action_count=0"),
            pattern(r"live_aql_proof_validation_next_action_count=0"),
            pattern(r"pending_component_request_count=0"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_prerequisites_satisfied=true"),
            pattern(r"submission_ready=true"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.9\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.3\.next_action=none"),
            pattern(r"prerequisites\.9\.next_action=none"),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-resolved-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_submission_blocker_report_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"blocker_count=11"),
            pattern(r"execution_readiness_blocker_count=9"),
            pattern(r"runtime_request_component_pending_count=378"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=380"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
            pattern(r"blockers\.9\.runtime_request_component_blocker=true"),
            pattern(r"blockers\.10\.live_aql_proof_validation_blocker=true"),
            pattern(r"live_aql_submission_side_effect_blocker=false"),
            pattern(r"live_queue_mutation_blocker=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="reference_moe_model_api_runtime_submission_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_model_api",
            "--",
            "--runtime-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=10"),
            pattern(r"next_action_count=10"),
            pattern(r"runtime_request_component_next_action_count=8"),
            pattern(r"live_aql_proof_validation_next_action_count=2"),
            pattern(r"pending_component_request_count=378"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"submission_ready=false"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.next_action=validate_live_aql_proof"),
            pattern(r"prerequisites\.3\.next_action_input=KfdQueueLiveAqlBatchReservationPlanInput"),
            pattern(r"prerequisites\.9\.next_action=validate_live_aql_proof"),
            pattern(
                r"prerequisites\.9\.next_action_input=KfdQueueLiveAqlMaterializedPacketPlanInput"
            ),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS
        + RUNTIME_SUBMISSION_PREREQUISITE_NEXT_ACTION_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-runtime-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="custom_model_api",
        command=("cargo", "run", "-q", "-p", "mainarch-core", "--example", "custom_model_api"),
        required_patterns=ACCEPTED_PATTERNS + CUSTOM_MODEL_GAP_SYMBOL_PATTERNS,
    ),
    ExampleGate(
        name="custom_model_api_runtime_launch_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-launch-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_execution_request_plan"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"runtime_request_plan_count=10"),
            pattern(r"component_pending_count=41"),
            pattern(r"live_aql_proof_surface_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"all_components_applied=false"),
        )
        + RUNTIME_LAUNCH_REQUEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-launch-request.receipt",
    ),
    ExampleGate(
        name="custom_model_api_runtime_submission_gate_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"all_components_applied=false"),
            pattern(r"all_live_aql_proof_validations_applied=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=41"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=9"),
            pattern(r"submission_blocker_count=11"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-submission-gate.receipt",
    ),
    ExampleGate(
        name="custom_model_api_runtime_resolved_submission_gate_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-resolved-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=0"),
            pattern(r"submission_blocker_count=0"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-resolved-submission-gate.receipt",
    ),
    ExampleGate(
        name="custom_model_api_runtime_resolved_submission_blocker_report_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-resolved-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"blocker_count=0"),
            pattern(r"execution_readiness_blocker_count=0"),
            pattern(r"runtime_request_component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-resolved-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="custom_model_api_runtime_resolved_submission_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-resolved-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"prerequisite_count=10"),
            pattern(r"satisfied_prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=0"),
            pattern(r"next_action_count=0"),
            pattern(r"runtime_request_component_next_action_count=0"),
            pattern(r"live_aql_proof_validation_next_action_count=0"),
            pattern(r"pending_component_request_count=0"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_prerequisites_satisfied=true"),
            pattern(r"submission_ready=true"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.9\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.3\.next_action=none"),
            pattern(r"prerequisites\.9\.next_action=none"),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-resolved-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="custom_model_api_runtime_submission_blocker_report_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"blocker_count=11"),
            pattern(r"execution_readiness_blocker_count=9"),
            pattern(r"runtime_request_component_pending_count=41"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=43"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
            pattern(r"blockers\.9\.runtime_request_component_blocker=true"),
            pattern(r"blockers\.10\.live_aql_proof_validation_blocker=true"),
            pattern(r"live_aql_submission_side_effect_blocker=false"),
            pattern(r"live_queue_mutation_blocker=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="custom_model_api_runtime_submission_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "custom_model_api",
            "--",
            "--runtime-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=3"),
            pattern(r"window_count=3"),
            pattern(r"prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=10"),
            pattern(r"next_action_count=10"),
            pattern(r"runtime_request_component_next_action_count=8"),
            pattern(r"live_aql_proof_validation_next_action_count=2"),
            pattern(r"pending_component_request_count=41"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"submission_ready=false"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.next_action=validate_live_aql_proof"),
            pattern(r"prerequisites\.3\.next_action_input=KfdQueueLiveAqlBatchReservationPlanInput"),
            pattern(r"prerequisites\.9\.next_action=validate_live_aql_proof"),
            pattern(
                r"prerequisites\.9\.next_action_input=KfdQueueLiveAqlMaterializedPacketPlanInput"
            ),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS
        + RUNTIME_SUBMISSION_PREREQUISITE_NEXT_ACTION_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-custom-model-runtime-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest",
        command=("cargo", "run", "-q", "-p", "mainarch-cli", "--bin", "mainarch-model-api-selftest"),
        required_patterns=ACCEPTED_PATTERNS + REFERENCE_MOE_GAP_SYMBOL_PATTERNS,
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-launch-request-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-launch-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_execution_request_plan"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"runtime_request_plan_count=10"),
            pattern(r"component_pending_count=378"),
            pattern(r"live_aql_proof_surface_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"all_components_applied=false"),
        )
        + RUNTIME_LAUNCH_REQUEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-launch-request.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-submission-gate-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"all_components_applied=false"),
            pattern(r"all_live_aql_proof_validations_applied=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=378"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=9"),
            pattern(r"submission_blocker_count=11"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-submission-gate.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-resolved-submission-gate-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-resolved-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=0"),
            pattern(r"submission_blocker_count=0"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-resolved-submission-gate.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-resolved-submission-prerequisite-plan-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-resolved-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"prerequisite_count=10"),
            pattern(r"satisfied_prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=0"),
            pattern(r"next_action_count=0"),
            pattern(r"runtime_request_component_next_action_count=0"),
            pattern(r"live_aql_proof_validation_next_action_count=0"),
            pattern(r"pending_component_request_count=0"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_prerequisites_satisfied=true"),
            pattern(r"submission_ready=true"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.9\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.3\.next_action=none"),
            pattern(r"prerequisites\.9\.next_action=none"),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-resolved-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-resolved-submission-blocker-report-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-resolved-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"blocker_count=0"),
            pattern(r"execution_readiness_blocker_count=0"),
            pattern(r"runtime_request_component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-resolved-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-submission-blocker-report-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"blocker_count=11"),
            pattern(r"execution_readiness_blocker_count=9"),
            pattern(r"runtime_request_component_pending_count=378"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=380"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
            pattern(r"blockers\.9\.runtime_request_component_blocker=true"),
            pattern(r"blockers\.10\.live_aql_proof_validation_blocker=true"),
            pattern(r"live_aql_submission_side_effect_blocker=false"),
            pattern(r"live_queue_mutation_blocker=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-runtime-submission-prerequisite-plan-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--runtime-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=36"),
            pattern(r"window_count=7"),
            pattern(r"prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=10"),
            pattern(r"next_action_count=10"),
            pattern(r"runtime_request_component_next_action_count=8"),
            pattern(r"live_aql_proof_validation_next_action_count=2"),
            pattern(r"pending_component_request_count=378"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"submission_ready=false"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.next_action=validate_live_aql_proof"),
            pattern(r"prerequisites\.3\.next_action_input=KfdQueueLiveAqlBatchReservationPlanInput"),
            pattern(r"prerequisites\.9\.next_action=validate_live_aql_proof"),
            pattern(
                r"prerequisites\.9\.next_action_input=KfdQueueLiveAqlMaterializedPacketPlanInput"
            ),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS
        + RUNTIME_SUBMISSION_PREREQUISITE_NEXT_ACTION_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-runtime-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="mainarch-model-api-selftest-static-handoff-receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-cli",
            "--bin",
            "mainarch-model-api-selftest",
            "--",
            "--static-handoff-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_plugin_static_handoff"),
            pattern(r"model_name=mainarch-reference-qwen3-style-moe"),
            pattern(r"receipt_namespace=selftest"),
            pattern(rf"manifest\.receipt_fingerprint={HEX64}"),
            pattern(rf"compatibility\.receipt_fingerprint={HEX64}"),
            pattern(r"metadata_admitted=true"),
            pattern(r"launch_execution\.executable=false"),
            pattern(r"launch_execution\.unresolved_runtime_requirement_count=9"),
            pattern(
                r"launch_execution\.unresolved_runtime_requirements\.0="
                r"kernel_candidate_selection_policy"
            ),
            pattern(
                r"launch_execution\.unresolved_runtime_requirements\.8="
                r"aql_packet_materialization"
            ),
            pattern(r"launch_execution\.aql_dispatchable_packet_count=0"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"live_execution_supported=false"),
            pattern(r"gpu_buffers_allocated=false"),
            pattern(r"kernels_submitted=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-cli/expected-model-api-selftest-static-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_metadata",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
        ),
        required_patterns=CHECKPOINT_METADATA_PATTERNS,
    ),
    ExampleGate(
        name="reference_moe_checkpoint_staging_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-staging-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_direct_read_staging_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"direct_io_alignment=4096"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"piece_count=24"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"total_payload_bytes=107456"),
            pattern(r"total_read_bytes=110592"),
            pattern(r"read_amplification_milli=1029"),
            pattern(r"live_execution_supported=false"),
            pattern(r"checkpoint_files_opened=false"),
            pattern(r"payload_bytes_read=false"),
            pattern(r"host_staging_allocated=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-staging.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_buffered_host_staging"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"direct_io_alignment=4096"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"piece_count=24"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"total_requested_read_bytes=110592"),
            pattern(r"total_file_bytes_read=109913"),
            pattern(r"total_tail_padding_bytes=679"),
            pattern(r"total_payload_bytes=107456"),
            pattern(
                r"payload_fingerprint=f4661d3f5241cc36ddd657d9da601a75a6876ae2f03238eb1b457fd884e11141"
            ),
            pattern(r"checkpoint_files_opened=true"),
            pattern(r"payload_bytes_read=true"),
            pattern(r"host_staging_allocated=true"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_mapped_host_staging_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-mapped-host-staging-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_mapped_host_staging"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"direct_io_alignment=4096"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"piece_count=24"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"total_requested_read_bytes=110592"),
            pattern(r"total_file_bytes_mapped=109913"),
            pattern(r"total_tail_padding_bytes=679"),
            pattern(r"mmap_call_count=1"),
            pattern(r"total_mmap_bytes=109913"),
            pattern(r"total_payload_bytes=107456"),
            pattern(
                r"payload_fingerprint=f4661d3f5241cc36ddd657d9da601a75a6876ae2f03238eb1b457fd884e11141"
            ),
            pattern(r"checkpoint_files_opened=true"),
            pattern(r"payload_bytes_mapped=true"),
            pattern(r"payload_bytes_staged=true"),
            pattern(r"buffered_reads_issued=false"),
            pattern(r"host_staging_allocated=true"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-mapped-host-staging.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_copy_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-copy-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_to_device_copy_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"staging_slot_count=2"),
            pattern(r"staging_slot_bytes=110592"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"host_staging_offsets_bound=true"),
            pattern(r"destination_device_va_bound=true"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-copy-plan.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_destination_residency_proof_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-destination-residency-proof-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_destination_residency_proof_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"destination_span_count=24"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"total_destination_span_bytes=107456"),
            pattern(r"destination_device_va_min=0x0000000100000010"),
            pattern(r"destination_device_va_max_end=0x000000010001b3f0"),
            pattern(r"destination_device_va_bound=true"),
            pattern(r"destination_span_rows_contiguous=true"),
            pattern(r"kfd_residency_rows_bound=false"),
            pattern(r"allocation_handles_bound=false"),
            pattern(r"resident_gpu_ids_bound=false"),
            pattern(r"residency_proof_executable=false"),
            pattern(r"kfd_query_executed=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_allocated=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"spans\.count=24"),
            pattern(r"spans\.0\.tensor=embed_tokens\.weight"),
            pattern(r"spans\.0\.destination_device_va_begin=0x0000000100000010"),
            pattern(r"spans\.10\.tensor=layers\.0\.experts\.up\.weight"),
            pattern(r"spans\.17\.payload_bytes=8192"),
            pattern(r"spans\.23\.tensor=lm_head\.weight"),
            pattern(r"spans\.23\.destination_device_va_end=0x000000010001b3f0"),
            pattern(r"spans\.23\.destination_span_bound=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-destination-residency-proof-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_destination_residency_query_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-destination-residency-query-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_destination_residency_query_request"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"destination_span_count=24"),
            pattern(r"allocation_count=15"),
            pattern(r"kfd_residency_row_count=43"),
            pattern(r"matched_allocation_count=15"),
            pattern(r"missing_allocation_count=0"),
            pattern(r"resident_gpu_ids=1001,1002"),
            pattern(r"resident_gpu_id_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"total_destination_span_bytes=107456"),
            pattern(r"total_allocation_span_bytes=107456"),
            pattern(r"destination_device_va_min=0x0000000100000010"),
            pattern(r"destination_device_va_max_end=0x000000010001b3f0"),
            pattern(r"kfd_residency_binding_ready=true"),
            pattern(r"destination_device_va_bound=true"),
            pattern(r"destination_span_rows_contiguous=true"),
            pattern(r"allocation_handles_bound=true"),
            pattern(r"resident_gpu_ids_bound=true"),
            pattern(r"destination_spans_within_allocations=true"),
            pattern(r"residency_query_metadata_ready=true"),
            pattern(r"residency_query_executable=false"),
            pattern(r"kfd_query_executed=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_allocated=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"allocations\.count=15"),
            pattern(r"allocations\.0\.tensor=embed_tokens\.weight"),
            pattern(r"allocations\.0\.kfd_tensor_matches_copy=true"),
            pattern(r"allocations\.0\.kfd_checkpoint_weight_slot=true"),
            pattern(r"allocations\.0\.copies\.0\.checkpoint_key=model\.embed_tokens\.weight"),
            pattern(r"allocations\.0\.copies\.0\.destination_within_allocation=true"),
            pattern(r"allocations\.2\.tensor=layers\.0\.experts\.down\.weight"),
            pattern(r"allocations\.2\.copy_count=4"),
            pattern(r"allocations\.2\.copies\.3\.destination_span_bound=true"),
            pattern(r"allocations\.4\.tensor=layers\.0\.experts\.up\.weight"),
            pattern(r"allocations\.8\.total_payload_bytes=8192"),
            pattern(r"allocations\.14\.tensor=lm_head\.weight"),
            pattern(r"allocations\.14\.destination_device_va_max_end=0x000000010001b3f0"),
            pattern(r"allocations\.14\.allocation_handle_bound=true"),
            pattern(r"allocations\.14\.kfd_read_only_access=true"),
            pattern(r"allocations\.14\.residency_query_row_ready=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-destination-residency-query-request.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_sdma_queue_reservation_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-sdma-queue-reservation-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_sdma_queue_reservation_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"max_copy_chunk_bytes=2097152"),
            pattern(r"copy_packet_dwords=7"),
            pattern(r"completion_fence_packet_dwords=4"),
            pattern(r"copy_packet_request_count=24"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_reserved_count=0"),
            pattern(r"queue_packet_dword_count=172"),
            pattern(r"queue_packet_byte_count=688"),
            pattern(r"doorbell_batch_request_count=1"),
            pattern(r"doorbell_batch_bound_count=0"),
            pattern(r"queue_id_bound_count=0"),
            pattern(r"queue_ring_bound_count=0"),
            pattern(r"completion_signal_bound_count=0"),
            pattern(r"sdma_packet_materialized_count=0"),
            pattern(r"reservation_applied_count=0"),
            pattern(r"all_queue_packets_reserved=false"),
            pattern(r"queue_reservation_executable=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"waves\.count=1"),
            pattern(r"waves\.0\.copy_packet_request_count=24"),
            pattern(r"waves\.0\.completion_packet_request_count=1"),
            pattern(r"waves\.0\.packet_request_count=25"),
            pattern(r"waves\.0\.first_completion_packet_index=24"),
            pattern(r"waves\.0\.completion_packet_offset_dwords=168"),
            pattern(r"waves\.0\.doorbell_batch_requested=true"),
            pattern(r"waves\.0\.doorbell_batch_bound=false"),
            pattern(r"waves\.0\.queue_packets_reserved=false"),
            pattern(r"copies\.count=24"),
            pattern(r"copies\.0\.tensor=embed_tokens\.weight"),
            pattern(r"copies\.0\.copy_packet_request_count=1"),
            pattern(r"copies\.0\.packet_offset_dwords=0"),
            pattern(r"copies\.17\.payload_bytes=8192"),
            pattern(r"copies\.17\.packet_offset_dwords=119"),
            pattern(r"copies\.23\.tensor=lm_head\.weight"),
            pattern(r"copies\.23\.first_sdma_packet_index=23"),
            pattern(r"copies\.23\.destination_span_bound=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-sdma-queue-reservation-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_sdma_queue_reservation_result_binding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-sdma-queue-reservation-result-binding-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_sdma_queue_reservation_result_binding_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"binding_source=checkpoint_sdma_queue_reservation_result_binding_v0"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(rf"sdma_queue_reservation_input_receipt_fingerprint={HEX64}"),
            pattern(r"sdma_queue_reservation_input_receipt_line_count=442"),
            pattern(r"queue_count_requested=1"),
            pattern(r"copy_packet_request_count=24"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_dword_count=172"),
            pattern(r"queue_packet_byte_count=688"),
            pattern(r"doorbell_batch_request_count=1"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"queue_id_bound_by_receipt_count=1"),
            pattern(r"queue_ring_bound_by_receipt_count=1"),
            pattern(r"doorbell_batch_bound_by_receipt_count=1"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"reservation_applied_by_receipt_count=1"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"all_queue_packets_reserved_by_receipt=true"),
            pattern(r"queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(r"queue_reservation_executed=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"results\.count=1"),
            pattern(r"results\.0\.queue_id=17"),
            pattern(r"results\.0\.queue_ring_base_va=0x000000004d000000"),
            pattern(r"results\.0\.queue_ring_size_bytes=4096"),
            pattern(r"results\.0\.queue_packet_write_index_end_exclusive=25"),
            pattern(r"results\.0\.doorbell_value=25"),
            pattern(r"results\.0\.queue_packets_reserved_by_receipt=true"),
            pattern(r"results\.0\.reservation_applied_by_receipt=true"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-sdma-queue-reservation-result-binding.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_copy_completion_signal_binding_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-copy-completion-signal-binding-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_copy_completion_signal_binding_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"signal_kind=amd_signal_t"),
            pattern(r"signal_initial_value=1"),
            pattern(r"signal_completion_value=0"),
            pattern(r"completion_signal_request_count=1"),
            pattern(r"completion_signal_bound_count=0"),
            pattern(r"signal_handle_bound_count=0"),
            pattern(r"signal_device_va_bound_count=0"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"completion_packet_dword_count=4"),
            pattern(r"completion_packet_byte_count=16"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_reserved_count=0"),
            pattern(r"reservation_applied_count=0"),
            pattern(r"sdma_packet_materialized_count=0"),
            pattern(r"signal_binding_rows_contiguous=true"),
            pattern(r"all_completion_signals_requested=true"),
            pattern(r"completion_signal_binding_executable=false"),
            pattern(r"completion_signals_created=false"),
            pattern(r"completion_signal_wait_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"bindings\.count=1"),
            pattern(r"bindings\.0\.wave_index=0"),
            pattern(r"bindings\.0\.completion_signal_index=0"),
            pattern(r"bindings\.0\.wave_completion_packet_ordinal=0"),
            pattern(r"bindings\.0\.completion_packet_index=24"),
            pattern(r"bindings\.0\.completion_packet_offset_dwords=168"),
            pattern(r"bindings\.0\.completion_packet_dword_count=4"),
            pattern(r"bindings\.0\.completion_packet_bytes=16"),
            pattern(r"bindings\.0\.queue_packet_request_count=25"),
            pattern(r"bindings\.0\.queue_packets_reserved=false"),
            pattern(r"bindings\.0\.signal_slot_requested=true"),
            pattern(r"bindings\.0\.signal_handle_bound=false"),
            pattern(r"bindings\.0\.signal_device_va_bound=false"),
            pattern(r"bindings\.0\.completion_signal_bound=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-copy-completion-signal-binding-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_copy_completion_signal_result_binding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-copy-completion-signal-result-binding-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_copy_completion_signal_result_binding_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"binding_source=checkpoint_copy_completion_signal_result_binding_v0"),
            pattern(r"signal_kind=amd_signal_t"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(rf"copy_completion_signal_binding_input_receipt_fingerprint={HEX64}"),
            pattern(r"copy_completion_signal_binding_input_receipt_line_count=58"),
            pattern(r"queue_count_requested=1"),
            pattern(r"signal_initial_value=1"),
            pattern(r"signal_completion_value=0"),
            pattern(r"completion_signal_request_count=1"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"completion_packet_dword_count=4"),
            pattern(r"completion_packet_byte_count=16"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"signal_handle_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"all_completion_signals_bound_by_receipt=true"),
            pattern(
                r"copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"completion_signal_binding_executed=false"),
            pattern(r"completion_signals_created=false"),
            pattern(r"completion_signal_wait_issued=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"results\.count=1"),
            pattern(r"results\.0\.binding_index=0"),
            pattern(r"results\.0\.wave_index=0"),
            pattern(r"results\.0\.completion_signal_index=0"),
            pattern(r"results\.0\.completion_packet_index=24"),
            pattern(r"results\.0\.completion_packet_offset_dwords=168"),
            pattern(r"results\.0\.completion_packet_dword_count=4"),
            pattern(r"results\.0\.completion_packet_bytes=16"),
            pattern(r"results\.0\.signal_handle=0x0000000051000000"),
            pattern(r"results\.0\.signal_device_va=0x000000005a000000"),
            pattern(r"results\.0\.signal_handle_bound_by_receipt=true"),
            pattern(r"results\.0\.signal_device_va_bound_by_receipt=true"),
            pattern(r"results\.0\.completion_signal_bound_by_receipt=true"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-copy-completion-signal-result-binding.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_sdma_copy_packet_materialization_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-sdma-copy-packet-materialization-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_sdma_copy_packet_materialization_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"max_copy_chunk_bytes=2097152"),
            pattern(r"copy_packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"completion_packet_kind=SDMA_OP_FENCE_SIGNAL"),
            pattern(r"copy_packet_dwords=7"),
            pattern(r"completion_fence_packet_dwords=4"),
            pattern(r"copy_packet_request_count=24"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_dword_count=172"),
            pattern(r"queue_packet_byte_count=688"),
            pattern(r"packet_row_count=25"),
            pattern(r"packet_template_request_count=25"),
            pattern(r"host_virtual_address_bound_count=0"),
            pattern(r"destination_device_va_bound_count=24"),
            pattern(r"completion_signal_bound_count=0"),
            pattern(r"signal_device_va_bound_count=0"),
            pattern(r"queue_packet_reserved_count=0"),
            pattern(r"reservation_applied_count=0"),
            pattern(r"sdma_packet_materialized_count=0"),
            pattern(r"packet_rows_contiguous=true"),
            pattern(r"packet_offsets_contiguous=true"),
            pattern(r"all_packet_templates_requested=true"),
            pattern(r"packet_materialization_executable=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"copy_packets\.count=24"),
            pattern(r"copy_packets\.0\.tensor=embed_tokens\.weight"),
            pattern(r"copy_packets\.0\.packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"copy_packets\.0\.packet_offset_dwords=0"),
            pattern(r"copy_packets\.0\.host_virtual_address_bound=false"),
            pattern(r"copy_packets\.0\.destination_device_va_bound=true"),
            pattern(r"copy_packets\.17\.copy_chunk_bytes=8192"),
            pattern(r"copy_packets\.17\.packet_offset_dwords=119"),
            pattern(r"copy_packets\.23\.tensor=lm_head\.weight"),
            pattern(r"copy_packets\.23\.sdma_packet_index=23"),
            pattern(r"copy_packets\.23\.packet_bytes_materialized=false"),
            pattern(r"completion_packets\.count=1"),
            pattern(r"completion_packets\.24\.sdma_packet_index=24"),
            pattern(r"completion_packets\.24\.packet_kind=SDMA_OP_FENCE_SIGNAL"),
            pattern(r"completion_packets\.24\.packet_offset_dwords=168"),
            pattern(r"completion_packets\.24\.packet_dword_count=4"),
            pattern(r"completion_packets\.24\.signal_initial_value=1"),
            pattern(r"completion_packets\.24\.signal_completion_value=0"),
            pattern(r"completion_packets\.24\.completion_signal_bound=false"),
            pattern(r"completion_packets\.24\.signal_device_va_bound=false"),
            pattern(r"completion_packets\.24\.packet_bytes_materialized=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-sdma-copy-packet-materialization-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_sdma_copy_packet_materialization_result_binding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-sdma-copy-packet-materialization-result-binding-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_sdma_copy_packet_materialization_result_binding_plan"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(
                r"binding_source=checkpoint_sdma_copy_packet_materialization_result_binding_v0"
            ),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"copy_count=24"),
            pattern(r"packet_row_count=25"),
            pattern(r"result_binding_count=25"),
            pattern(r"matched_result_binding_count=25"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"host_virtual_address_bound_by_receipt_count=24"),
            pattern(r"destination_device_va_bound_by_receipt_count=24"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"sdma_packet_materialized_by_receipt_count=25"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"all_sdma_packets_materialized_by_receipt=true"),
            pattern(
                r"upload_packet_materialization_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"packet_materialization_executed=false"),
            pattern(r"queue_memory_mutated=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"results\.count=25"),
            pattern(r"results\.0\.packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"results\.0\.host_virtual_address=0x00007f0000000999"),
            pattern(r"results\.0\.packet_materialization_bound_by_receipt=true"),
            pattern(r"results\.24\.packet_kind=SDMA_OP_FENCE_SIGNAL"),
            pattern(r"results\.24\.completion_signal_index=0"),
            pattern(r"results\.24\.signal_handle=0x0000000051000000"),
            pattern(r"results\.24\.packet_materialization_bound_by_receipt=true"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-sdma-copy-packet-materialization-result-binding.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_sdma_copy_packet_validation_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-sdma-copy-packet-validation-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_sdma_copy_packet_validation_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"copy_packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"completion_packet_kind=SDMA_OP_FENCE_SIGNAL"),
            pattern(r"copy_packet_dwords=7"),
            pattern(r"completion_fence_packet_dwords=4"),
            pattern(r"copy_packet_validation_row_count=24"),
            pattern(r"completion_packet_validation_row_count=1"),
            pattern(r"packet_validation_row_count=25"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_dword_count=172"),
            pattern(r"queue_packet_byte_count=688"),
            pattern(r"packet_template_request_count=25"),
            pattern(r"packet_template_valid_count=25"),
            pattern(r"packet_shape_valid_count=25"),
            pattern(r"packet_byte_count_valid_count=25"),
            pattern(r"packet_offset_valid_count=25"),
            pattern(r"copy_payload_span_valid_count=24"),
            pattern(r"completion_signal_value_valid_count=1"),
            pattern(r"host_virtual_address_bound_count=0"),
            pattern(r"destination_device_va_bound_count=24"),
            pattern(r"completion_signal_bound_count=0"),
            pattern(r"signal_device_va_bound_count=0"),
            pattern(r"queue_packet_reserved_count=0"),
            pattern(r"reservation_applied_count=0"),
            pattern(r"sdma_packet_materialized_count=0"),
            pattern(r"packet_bytes_validated_count=0"),
            pattern(r"packet_rows_contiguous=true"),
            pattern(r"sdma_packet_indices_contiguous=true"),
            pattern(r"packet_offsets_contiguous=true"),
            pattern(r"all_packet_templates_requested=true"),
            pattern(r"all_packet_templates_valid=true"),
            pattern(r"all_packet_shapes_valid=true"),
            pattern(r"all_packet_byte_counts_valid=true"),
            pattern(r"all_copy_payload_spans_valid=true"),
            pattern(r"all_completion_signal_values_valid=true"),
            pattern(r"packet_validation_executable=false"),
            pattern(r"packets_submittable=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"validation_rows\.count=25"),
            pattern(r"validation_rows\.0\.validation_scope=copy_payload_packet"),
            pattern(r"validation_rows\.0\.packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"validation_rows\.0\.packet_offset_dwords=0"),
            pattern(r"validation_rows\.0\.expected_packet_bytes=28"),
            pattern(r"validation_rows\.0\.packet_shape_valid=true"),
            pattern(r"validation_rows\.0\.packet_bytes_validated=false"),
            pattern(r"validation_rows\.17\.payload_bytes=8192"),
            pattern(r"validation_rows\.17\.packet_offset_dwords=119"),
            pattern(r"validation_rows\.23\.packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"validation_rows\.23\.packet_bytes_validated=false"),
            pattern(r"validation_rows\.24\.validation_scope=completion_fence_packet"),
            pattern(r"validation_rows\.24\.packet_kind=SDMA_OP_FENCE_SIGNAL"),
            pattern(r"validation_rows\.24\.packet_offset_dwords=168"),
            pattern(r"validation_rows\.24\.packet_dword_count=4"),
            pattern(r"validation_rows\.24\.signal_initial_value=1"),
            pattern(r"validation_rows\.24\.signal_completion_value=0"),
            pattern(r"validation_rows\.24\.completion_signal_value_valid=true"),
            pattern(r"validation_rows\.24\.completion_signal_bound=false"),
            pattern(r"validation_rows\.24\.packet_bytes_validated=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-sdma-copy-packet-validation-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_sdma_copy_packet_validation_result_binding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-sdma-copy-packet-validation-result-binding-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_sdma_copy_packet_validation_result_binding_plan"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"binding_source=checkpoint_sdma_copy_packet_validation_result_binding_v0"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"copy_count=24"),
            pattern(r"packet_validation_row_count=25"),
            pattern(r"result_binding_count=25"),
            pattern(r"matched_result_binding_count=25"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"host_virtual_address_bound_by_receipt_count=24"),
            pattern(r"destination_device_va_bound_by_receipt_count=24"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"sdma_packet_materialized_by_receipt_count=25"),
            pattern(r"packet_template_validated_by_receipt_count=25"),
            pattern(r"packet_shape_validated_by_receipt_count=25"),
            pattern(r"packet_byte_count_validated_by_receipt_count=25"),
            pattern(r"packet_offset_validated_by_receipt_count=25"),
            pattern(r"copy_payload_span_validated_by_receipt_count=24"),
            pattern(r"completion_signal_value_validated_by_receipt_count=1"),
            pattern(r"packet_bytes_validated_by_receipt_count=25"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"all_packets_validated_by_receipt=true"),
            pattern(r"packets_submittable_by_receipt=true"),
            pattern(r"upload_packet_validation_prerequisite_satisfied_by_receipt=true"),
            pattern(r"packet_validation_executed=false"),
            pattern(r"queue_memory_mutated=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"results\.count=25"),
            pattern(r"results\.0\.validation_scope=copy_payload_packet"),
            pattern(r"results\.0\.packet_kind=SDMA_OP_COPY_LINEAR"),
            pattern(r"results\.0\.host_virtual_address=0x00007f0000000999"),
            pattern(r"results\.0\.packet_bytes_validated_by_receipt=true"),
            pattern(r"results\.0\.packet_validation_bound_by_receipt=true"),
            pattern(r"results\.24\.validation_scope=completion_fence_packet"),
            pattern(r"results\.24\.packet_kind=SDMA_OP_FENCE_SIGNAL"),
            pattern(r"results\.24\.signal_handle=0x0000000051000000"),
            pattern(r"results\.24\.packet_bytes_validated_by_receipt=true"),
            pattern(r"results\.24\.packet_validation_bound_by_receipt=true"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-sdma-copy-packet-validation-result-binding.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_cache_visibility_policy_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-cache-visibility-policy-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_cache_visibility_policy_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"policy_kind=sdma_completion_signal_then_device_scope_cache_visibility"),
            pattern(r"visibility_scope=device_scope_vram_visibility"),
            pattern(r"copy_packet_validation_row_count=24"),
            pattern(r"completion_packet_validation_row_count=1"),
            pattern(r"packet_validation_row_count=25"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_reserved_count=0"),
            pattern(r"reservation_applied_count=0"),
            pattern(r"sdma_packet_materialized_count=0"),
            pattern(r"packet_bytes_validated_count=0"),
            pattern(r"packet_validation_ready_count=25"),
            pattern(r"packet_offset_valid_count=25"),
            pattern(r"copy_payload_span_valid_count=24"),
            pattern(r"completion_signal_value_valid_count=1"),
            pattern(r"policy_wave_count=1"),
            pattern(r"policy_selected_count=1"),
            pattern(r"device_visibility_request_count=1"),
            pattern(r"host_visibility_request_count=0"),
            pattern(r"cache_flush_request_count=0"),
            pattern(r"cache_invalidate_request_count=0"),
            pattern(r"vram_visibility_proven_count=0"),
            pattern(r"all_policy_rows_contiguous=true"),
            pattern(r"all_packet_offsets_contiguous=true"),
            pattern(r"all_completion_signal_values_valid=true"),
            pattern(r"all_device_visibility_requests_selected=true"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"cache_visibility_policy_executable=false"),
            pattern(r"cache_flush_issued=false"),
            pattern(r"cache_invalidate_issued=false"),
            pattern(r"vram_visibility_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"waves\.count=1"),
            pattern(r"waves\.0\.policy_wave_index=0"),
            pattern(r"waves\.0\.wave_index=0"),
            pattern(r"waves\.0\.batch_index=0"),
            pattern(r"waves\.0\.staging_slot=0"),
            pattern(r"waves\.0\.staging_slot_epoch=0"),
            pattern(r"waves\.0\.copy_begin_index=0"),
            pattern(r"waves\.0\.copy_count=24"),
            pattern(r"waves\.0\.copy_bytes=107456"),
            pattern(r"waves\.0\.validation_row_begin_index=0"),
            pattern(r"waves\.0\.validation_row_end_index_exclusive=25"),
            pattern(r"waves\.0\.validation_row_count=25"),
            pattern(r"waves\.0\.copy_validation_row_count=24"),
            pattern(r"waves\.0\.completion_validation_row_count=1"),
            pattern(r"waves\.0\.packet_validation_ready_count=25"),
            pattern(r"waves\.0\.destination_device_va_min=0x0000000100000010"),
            pattern(r"waves\.0\.destination_device_va_max_end=0x000000010001b3f0"),
            pattern(r"waves\.0\.packet_validation_rows_contiguous=true"),
            pattern(r"waves\.0\.packet_offsets_contiguous=true"),
            pattern(r"waves\.0\.completion_signal_values_valid=true"),
            pattern(r"waves\.0\.cache_visibility_policy_selected=true"),
            pattern(r"waves\.0\.device_visibility_required=true"),
            pattern(r"waves\.0\.host_visibility_required=false"),
            pattern(r"waves\.0\.cache_flush_requested=false"),
            pattern(r"waves\.0\.cache_invalidate_requested=false"),
            pattern(r"waves\.0\.vram_visibility_proven=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-cache-visibility-policy-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_synchronization_plan_input_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-synchronization-plan-input-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_upload_synchronization_plan_input"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"signal_kind=amd_signal_t"),
            pattern(r"completion_wait_kind=amd_signal_wait_acquire_cpu_only_plan"),
            pattern(r"synchronization_mode=sdma_completion_signal_wait_then_visibility_observation"),
            pattern(r"policy_kind=sdma_completion_signal_then_device_scope_cache_visibility"),
            pattern(r"visibility_scope=device_scope_vram_visibility"),
            pattern(r"completion_signal_request_count=1"),
            pattern(r"completion_signal_bound_count=0"),
            pattern(r"signal_device_va_bound_count=0"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_reserved_count=0"),
            pattern(r"reservation_applied_count=0"),
            pattern(r"sdma_packet_materialized_count=0"),
            pattern(r"packet_bytes_validated_count=0"),
            pattern(r"policy_wave_count=1"),
            pattern(r"policy_selected_count=1"),
            pattern(r"device_visibility_request_count=1"),
            pattern(r"vram_visibility_proven_count=0"),
            pattern(r"completion_wait_plan_count=1"),
            pattern(r"completion_wait_requested_count=1"),
            pattern(r"timeout_guard_request_count=1"),
            pattern(r"completion_wait_issued_count=0"),
            pattern(r"completion_wait_observed_count=0"),
            pattern(r"queue_synchronized_count=0"),
            pattern(r"copy_visible_to_device_count=0"),
            pattern(r"all_completion_signals_requested=true"),
            pattern(r"all_completion_waits_requested=true"),
            pattern(r"all_policy_waves_selected=true"),
            pattern(r"all_device_visibility_requests_selected=true"),
            pattern(r"all_completion_waits_observed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"cache_visibility_policy_executable=false"),
            pattern(r"upload_synchronization_executable=false"),
            pattern(r"completion_waits_issued=false"),
            pattern(r"queue_synchronized=false"),
            pattern(r"vram_visibility_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"waits\.count=1"),
            pattern(r"waits\.0\.wait_row_index=0"),
            pattern(r"waits\.0\.wave_index=0"),
            pattern(r"waits\.0\.batch_index=0"),
            pattern(r"waits\.0\.staging_slot=0"),
            pattern(r"waits\.0\.staging_slot_epoch=0"),
            pattern(r"waits\.0\.completion_signal_index=0"),
            pattern(r"waits\.0\.completion_packet_index=24"),
            pattern(r"waits\.0\.completion_packet_offset_dwords=168"),
            pattern(r"waits\.0\.completion_packet_dword_count=4"),
            pattern(r"waits\.0\.completion_packet_bytes=16"),
            pattern(r"waits\.0\.queue_packet_request_count=25"),
            pattern(r"waits\.0\.policy_wave_index=0"),
            pattern(r"waits\.0\.validation_row_begin_index=0"),
            pattern(r"waits\.0\.validation_row_end_index_exclusive=25"),
            pattern(r"waits\.0\.validation_row_count=25"),
            pattern(r"waits\.0\.signal_initial_value=1"),
            pattern(r"waits\.0\.signal_completion_value=0"),
            pattern(r"waits\.0\.signal_slot_requested=true"),
            pattern(r"waits\.0\.completion_signal_bound=false"),
            pattern(r"waits\.0\.signal_device_va_bound=false"),
            pattern(r"waits\.0\.cache_visibility_policy_selected=true"),
            pattern(r"waits\.0\.device_visibility_required=true"),
            pattern(r"waits\.0\.vram_visibility_proven=false"),
            pattern(r"waits\.0\.completion_wait_requested=true"),
            pattern(r"waits\.0\.timeout_guard_requested=true"),
            pattern(r"waits\.0\.completion_wait_issued=false"),
            pattern(r"waits\.0\.completion_wait_observed=false"),
            pattern(r"waits\.0\.queue_synchronized=false"),
            pattern(r"waits\.0\.copy_visible_to_device=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-synchronization-plan-input.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_schedule_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-schedule-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_to_device_upload_schedule"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"staging_slot_bytes=110592"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"max_wave_copy_count=24"),
            pattern(r"max_wave_copy_bytes=107456"),
            pattern(r"max_wave_host_staging_span_bytes=107456"),
            pattern(r"wave_order_preserves_batch_order=true"),
            pattern(r"staging_slot_reuse_serialized=true"),
            pattern(r"host_staging_offsets_bound=true"),
            pattern(r"destination_device_va_bound=true"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-schedule.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_to_device_upload_prerequisite_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"prerequisite_count=8"),
            pattern(r"satisfied_prerequisite_count=0"),
            pattern(r"unsatisfied_prerequisite_count=8"),
            pattern(r"next_action_count=8"),
            pattern(r"all_prerequisites_satisfied=false"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"prerequisite_requirements\.count=8"),
            pattern(
                r"prerequisite_requirements\.labels=host_staging_pin,destination_vram_residency,sdma_queue_reservation,copy_completion_signal_binding,upload_packet_materialization,upload_packet_validation,cache_visibility_policy,upload_completion_synchronization"
            ),
            pattern(r"unsatisfied_prerequisite_requirements\.count=8"),
            pattern(
                r"unsatisfied_prerequisite_requirements\.labels=host_staging_pin,destination_vram_residency,sdma_queue_reservation,copy_completion_signal_binding,upload_packet_materialization,upload_packet_validation,cache_visibility_policy,upload_completion_synchronization"
            ),
            pattern(r"next_action_requirements\.count=8"),
            pattern(
                r"next_action_requirements\.labels=host_staging_pin,destination_vram_residency,sdma_queue_reservation,copy_completion_signal_binding,upload_packet_materialization,upload_packet_validation,cache_visibility_policy,upload_completion_synchronization"
            ),
            pattern(r"next_actions\.count=8"),
            pattern(
                r"next_actions\.labels=pin_host_staging_pages,query_destination_device_residency,reserve_sdma_queue_slots,bind_copy_completion_signal,materialize_sdma_copy_packets,validate_sdma_copy_packets,select_cache_visibility_policy,plan_upload_completion_synchronization"
            ),
            pattern(r"next_action_inputs\.count=8"),
            pattern(
                r"next_action_inputs\.labels=CheckpointPayloadHostStagingPinRequest,CheckpointPayloadDestinationResidencyQueryRequest,CheckpointPayloadSdmaQueueReservationInput,CheckpointPayloadCopyCompletionSignalBindingInput,CheckpointPayloadSdmaCopyPacketMaterializationInput,CheckpointPayloadSdmaCopyPacketValidationInput,CheckpointPayloadCacheVisibilityPolicyInput,CheckpointPayloadUploadSynchronizationPlanInput"
            ),
            pattern(r"prerequisites\.count=8"),
            pattern(r"prerequisites\.0\.requirement=host_staging_pin"),
            pattern(r"prerequisites\.0\.next_action=pin_host_staging_pages"),
            pattern(r"prerequisites\.1\.requirement=destination_vram_residency"),
            pattern(r"prerequisites\.1\.next_action=query_destination_device_residency"),
            pattern(
                r"prerequisites\.1\.next_action_input=CheckpointPayloadDestinationResidencyQueryRequest"
            ),
            pattern(r"prerequisites\.2\.next_action=reserve_sdma_queue_slots"),
            pattern(r"prerequisites\.4\.requirement=upload_packet_materialization"),
            pattern(r"prerequisites\.5\.next_action=validate_sdma_copy_packets"),
            pattern(r"prerequisites\.7\.requirement=upload_completion_synchronization"),
            pattern(r"prerequisites\.7\.next_action=plan_upload_completion_synchronization"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_runtime_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-runtime-handoff-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_to_device_upload_runtime_handoff"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"prerequisite_count=8"),
            pattern(r"next_action_count=8"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(rf"schedule_receipt_fingerprint={HEX64}"),
            pattern(rf"prerequisite_plan_receipt_fingerprint={HEX64}"),
            pattern(r"runtime_inputs\.count=8"),
            pattern(r"runtime_inputs\.0\.requirement=host_staging_pin"),
            pattern(r"runtime_inputs\.0\.next_action=pin_host_staging_pages"),
            pattern(
                r"runtime_inputs\.0\.next_action_input=CheckpointPayloadHostStagingPinRequest"
            ),
            pattern(
                r"runtime_inputs\.0\.receipt_kind=checkpoint_payload_host_staging_pin_request"
            ),
            pattern(rf"runtime_inputs\.0\.receipt_fingerprint={HEX64}"),
            pattern(r"runtime_inputs\.1\.requirement=destination_vram_residency"),
            pattern(
                r"runtime_inputs\.1\.receipt_kind=checkpoint_payload_destination_residency_query_request"
            ),
            pattern(
                r"runtime_inputs\.4\.receipt_kind=checkpoint_payload_sdma_copy_packet_materialization_input"
            ),
            pattern(
                r"runtime_inputs\.7\.requirement=upload_completion_synchronization"
            ),
            pattern(
                r"runtime_inputs\.7\.receipt_kind=checkpoint_payload_upload_synchronization_plan_input"
            ),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-runtime-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_bound_runtime_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-bound-runtime-handoff-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_to_device_upload_bound_runtime_handoff"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_staging_va_end=139637976948736"),
            pattern(r"host_pin_allocation_va_end=139637976948736"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_virtual_address_bound=true"),
            pattern(r"host_page_addresses_materialized=true"),
            pattern(r"pin_request_executable=false"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(rf"runtime_handoff_receipt_fingerprint={HEX64}"),
            pattern(r"runtime_handoff_receipt_line_count=82"),
            pattern(r"host_staging_pin_input_index=0"),
            pattern(r"host_staging_pin_requirement=host_staging_pin"),
            pattern(r"host_staging_pin_next_action=pin_host_staging_pages"),
            pattern(
                r"host_staging_pin_next_action_input=CheckpointPayloadHostStagingPinRequest"
            ),
            pattern(
                r"host_staging_pin_request_receipt_kind=checkpoint_payload_host_staging_pin_request"
            ),
            pattern(rf"host_staging_pin_request_receipt_fingerprint={HEX64}"),
            pattern(r"host_staging_pin_request_receipt_line_count=57"),
            pattern(
                r"host_staging_pin_virtual_address_plan_receipt_kind=checkpoint_payload_host_staging_pin_virtual_address_plan"
            ),
            pattern(
                rf"host_staging_pin_virtual_address_plan_receipt_fingerprint={HEX64}"
            ),
            pattern(r"host_staging_pin_virtual_address_plan_receipt_line_count=65"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-bound-runtime-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_mapped_host_staging_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-mapped-host-staging-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_mapped_host_staging_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_staging_va_end=139637976948736"),
            pattern(r"host_pin_allocation_va_end=139637976948736"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_input_index=0"),
            pattern(r"host_staging_pin_requirement=host_staging_pin"),
            pattern(r"host_staging_pin_next_action=pin_host_staging_pages"),
            pattern(
                r"host_staging_pin_next_action_input=CheckpointPayloadHostStagingPinRequest"
            ),
            pattern(r"host_staging_virtual_address_bound=true"),
            pattern(r"host_page_addresses_materialized=true"),
            pattern(r"kfd_map_memory_result_bound=true"),
            pattern(r"map_memory_argument_binding_ready=true"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"all_map_memory_results_successful=true"),
            pattern(r"residency_proven=true"),
            pattern(r"pin_argument_count=1"),
            pattern(r"page_pin_span_count=1"),
            pattern(r"map_memory_request_count=1"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"map_memory_result_observed_count=1"),
            pattern(r"map_memory_success_count=1"),
            pattern(r"result_metadata_ready_count=1"),
            pattern(r"total_page_pin_bytes=110592"),
            pattern(r"total_map_memory_bytes=110592"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"satisfied_prerequisite_count=1"),
            pattern(r"unsatisfied_prerequisite_count=7"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"bound_runtime_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_bound_runtime_handoff"
            ),
            pattern(rf"bound_runtime_handoff_receipt_fingerprint={HEX64}"),
            pattern(r"bound_runtime_handoff_receipt_line_count=39"),
            pattern(
                r"kfd_map_memory_result_binding_plan_receipt_kind=checkpoint_payload_host_staging_kfd_map_memory_result_binding_plan"
            ),
            pattern(rf"kfd_map_memory_result_binding_plan_receipt_fingerprint={HEX64}"),
            pattern(r"kfd_map_memory_result_binding_plan_receipt_line_count=101"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-mapped-host-staging-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_destination_residency_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-destination-residency-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_destination_residency_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_staging_va_end=139637976948736"),
            pattern(r"host_pin_allocation_va_end=139637976948736"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"satisfied_prerequisite_count=2"),
            pattern(r"unsatisfied_prerequisite_count=6"),
            pattern(r"destination_residency_input_index=1"),
            pattern(r"destination_residency_requirement=destination_vram_residency"),
            pattern(
                r"destination_residency_next_action=query_destination_device_residency"
            ),
            pattern(
                r"destination_residency_next_action_input=CheckpointPayloadDestinationResidencyQueryRequest"
            ),
            pattern(r"destination_span_count=24"),
            pattern(r"allocation_count=15"),
            pattern(r"kfd_residency_row_count=43"),
            pattern(r"matched_allocation_count=15"),
            pattern(r"missing_allocation_count=0"),
            pattern(r"resident_gpu_ids=1001,1002"),
            pattern(r"resident_gpu_id_count=2"),
            pattern(r"total_destination_span_bytes=107456"),
            pattern(r"total_allocation_span_bytes=107456"),
            pattern(r"destination_device_va_min=0x0000000100000010"),
            pattern(r"destination_device_va_max_end=0x000000010001b3f0"),
            pattern(r"kfd_residency_binding_source=mainarch_kfd_slot_residency_binding_v0"),
            pattern(r"kfd_residency_request_count=43"),
            pattern(r"kfd_residency_proven_count=43"),
            pattern(r"kfd_all_residency_bindings_ready=true"),
            pattern(r"kfd_allocation_performed=true"),
            pattern(r"kfd_residency_proven=true"),
            pattern(r"kfd_residency_binding_ready=true"),
            pattern(r"destination_device_va_bound=true"),
            pattern(r"destination_span_rows_contiguous=true"),
            pattern(r"allocation_handles_bound=true"),
            pattern(r"resident_gpu_ids_bound=true"),
            pattern(r"destination_spans_within_allocations=true"),
            pattern(r"residency_query_metadata_ready=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"mapped_host_staging_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_mapped_host_staging_handoff"
            ),
            pattern(rf"mapped_host_staging_handoff_receipt_fingerprint={HEX64}"),
            pattern(r"mapped_host_staging_handoff_receipt_line_count=56"),
            pattern(
                r"destination_residency_query_request_receipt_kind=checkpoint_payload_destination_residency_query_request"
            ),
            pattern(rf"destination_residency_query_request_receipt_fingerprint={HEX64}"),
            pattern(r"destination_residency_query_request_receipt_line_count=760"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-destination-residency-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_sdma_queue_reservation_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-sdma-queue-reservation-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_sdma_queue_reservation_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"sdma_queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(r"satisfied_prerequisite_count=3"),
            pattern(r"unsatisfied_prerequisite_count=5"),
            pattern(r"sdma_queue_reservation_input_index=2"),
            pattern(r"sdma_queue_reservation_requirement=sdma_queue_reservation"),
            pattern(r"sdma_queue_reservation_next_action=reserve_sdma_queue_slots"),
            pattern(
                r"sdma_queue_reservation_next_action_input=CheckpointPayloadSdmaQueueReservationInput"
            ),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"copy_packet_request_count=24"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_dword_count=172"),
            pattern(r"queue_packet_byte_count=688"),
            pattern(r"doorbell_batch_request_count=1"),
            pattern(r"queue_id_bound_by_receipt_count=1"),
            pattern(r"queue_ring_bound_by_receipt_count=1"),
            pattern(r"doorbell_batch_bound_by_receipt_count=1"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"reservation_applied_by_receipt_count=1"),
            pattern(r"all_queue_packets_reserved_by_receipt=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"destination_residency_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_destination_residency_handoff"
            ),
            pattern(rf"destination_residency_handoff_receipt_fingerprint={HEX64}"),
            pattern(r"destination_residency_handoff_receipt_line_count=64"),
            pattern(
                r"sdma_queue_reservation_input_receipt_kind=checkpoint_payload_sdma_queue_reservation_input"
            ),
            pattern(rf"sdma_queue_reservation_input_receipt_fingerprint={HEX64}"),
            pattern(r"sdma_queue_reservation_input_receipt_line_count=442"),
            pattern(
                r"sdma_queue_reservation_result_binding_plan_receipt_kind=checkpoint_payload_sdma_queue_reservation_result_binding_plan"
            ),
            pattern(rf"sdma_queue_reservation_result_binding_plan_receipt_fingerprint={HEX64}"),
            pattern(r"sdma_queue_reservation_result_binding_plan_receipt_line_count=64"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-sdma-queue-reservation-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_copy_completion_signal_binding_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-copy-completion-signal-binding-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_copy_completion_signal_binding_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"sdma_queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(
                r"copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"satisfied_prerequisite_count=4"),
            pattern(r"unsatisfied_prerequisite_count=4"),
            pattern(r"copy_completion_signal_binding_input_index=3"),
            pattern(
                r"copy_completion_signal_binding_requirement=copy_completion_signal_binding"
            ),
            pattern(
                r"copy_completion_signal_binding_next_action=bind_copy_completion_signal"
            ),
            pattern(
                r"copy_completion_signal_binding_next_action_input=CheckpointPayloadCopyCompletionSignalBindingInput"
            ),
            pattern(r"queue_type=KFD_IOC_QUEUE_TYPE_SDMA"),
            pattern(r"queue_count_requested=1"),
            pattern(r"signal_kind=amd_signal_t"),
            pattern(r"signal_initial_value=1"),
            pattern(r"signal_completion_value=0"),
            pattern(r"completion_signal_request_count=1"),
            pattern(r"signal_handle_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"all_completion_signals_bound_by_receipt=true"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"completion_packet_dword_count=4"),
            pattern(r"completion_packet_byte_count=16"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"sdma_queue_reservation_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_sdma_queue_reservation_handoff"
            ),
            pattern(rf"sdma_queue_reservation_handoff_receipt_fingerprint={HEX64}"),
            pattern(r"sdma_queue_reservation_handoff_receipt_line_count=54"),
            pattern(
                r"copy_completion_signal_binding_input_receipt_kind=checkpoint_payload_copy_completion_signal_binding_input"
            ),
            pattern(rf"copy_completion_signal_binding_input_receipt_fingerprint={HEX64}"),
            pattern(r"copy_completion_signal_binding_input_receipt_line_count=58"),
            pattern(
                r"copy_completion_signal_result_binding_plan_receipt_kind=checkpoint_payload_copy_completion_signal_result_binding_plan"
            ),
            pattern(
                rf"copy_completion_signal_result_binding_plan_receipt_fingerprint={HEX64}"
            ),
            pattern(r"copy_completion_signal_result_binding_plan_receipt_line_count=58"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-copy-completion-signal-binding-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_packet_materialization_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-packet-materialization-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_packet_materialization_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"sdma_queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(
                r"copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(
                r"upload_packet_materialization_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"satisfied_prerequisite_count=5"),
            pattern(r"unsatisfied_prerequisite_count=3"),
            pattern(r"packet_materialization_input_index=4"),
            pattern(r"packet_materialization_requirement=upload_packet_materialization"),
            pattern(
                r"packet_materialization_next_action=materialize_sdma_copy_packets"
            ),
            pattern(
                r"packet_materialization_next_action_input=CheckpointPayloadSdmaCopyPacketMaterializationInput"
            ),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"packet_row_count=25"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"host_virtual_address_bound_by_receipt_count=24"),
            pattern(r"destination_device_va_bound_by_receipt_count=24"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"sdma_packet_materialized_by_receipt_count=25"),
            pattern(r"all_sdma_packets_materialized_by_receipt=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"copy_completion_signal_binding_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_copy_completion_signal_binding_handoff"
            ),
            pattern(
                rf"copy_completion_signal_binding_handoff_receipt_fingerprint={HEX64}"
            ),
            pattern(
                r"packet_materialization_input_receipt_kind=checkpoint_payload_sdma_copy_packet_materialization_input"
            ),
            pattern(rf"packet_materialization_input_receipt_fingerprint={HEX64}"),
            pattern(
                r"packet_materialization_result_binding_plan_receipt_kind=checkpoint_payload_sdma_copy_packet_materialization_result_binding_plan"
            ),
            pattern(
                rf"packet_materialization_result_binding_plan_receipt_fingerprint={HEX64}"
            ),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-packet-materialization-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_packet_validation_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-packet-validation-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_packet_validation_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"sdma_queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(
                r"copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(
                r"upload_packet_materialization_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"upload_packet_validation_prerequisite_satisfied_by_receipt=true"),
            pattern(r"satisfied_prerequisite_count=6"),
            pattern(r"unsatisfied_prerequisite_count=2"),
            pattern(r"packet_validation_input_index=5"),
            pattern(r"packet_validation_requirement=upload_packet_validation"),
            pattern(r"packet_validation_next_action=validate_sdma_copy_packets"),
            pattern(
                r"packet_validation_next_action_input=CheckpointPayloadSdmaCopyPacketValidationInput"
            ),
            pattern(r"packet_validation_row_count=25"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"host_virtual_address_bound_by_receipt_count=24"),
            pattern(r"destination_device_va_bound_by_receipt_count=24"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"sdma_packet_materialized_by_receipt_count=25"),
            pattern(r"packet_template_validated_by_receipt_count=25"),
            pattern(r"packet_shape_validated_by_receipt_count=25"),
            pattern(r"packet_byte_count_validated_by_receipt_count=25"),
            pattern(r"packet_offset_validated_by_receipt_count=25"),
            pattern(r"copy_payload_span_validated_by_receipt_count=24"),
            pattern(r"completion_signal_value_validated_by_receipt_count=1"),
            pattern(r"packet_bytes_validated_by_receipt_count=25"),
            pattern(r"all_packets_validated_by_receipt=true"),
            pattern(r"packets_submittable_by_receipt=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"packet_materialization_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_packet_materialization_handoff"
            ),
            pattern(rf"packet_materialization_handoff_receipt_fingerprint={HEX64}"),
            pattern(
                r"packet_validation_input_receipt_kind=checkpoint_payload_sdma_copy_packet_validation_input"
            ),
            pattern(rf"packet_validation_input_receipt_fingerprint={HEX64}"),
            pattern(
                r"packet_validation_result_binding_plan_receipt_kind=checkpoint_payload_sdma_copy_packet_validation_result_binding_plan"
            ),
            pattern(rf"packet_validation_result_binding_plan_receipt_fingerprint={HEX64}"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-packet-validation-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_cache_visibility_policy_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-cache-visibility-policy-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_cache_visibility_policy_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"sdma_queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(
                r"copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(
                r"upload_packet_materialization_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"upload_packet_validation_prerequisite_satisfied_by_receipt=true"),
            pattern(r"cache_visibility_policy_prerequisite_satisfied_by_receipt=true"),
            pattern(r"satisfied_prerequisite_count=7"),
            pattern(r"unsatisfied_prerequisite_count=1"),
            pattern(r"cache_visibility_policy_input_index=6"),
            pattern(r"cache_visibility_policy_requirement=cache_visibility_policy"),
            pattern(r"cache_visibility_policy_next_action=select_cache_visibility_policy"),
            pattern(
                r"cache_visibility_policy_next_action_input=CheckpointPayloadCacheVisibilityPolicyInput"
            ),
            pattern(r"policy_kind=sdma_completion_signal_then_device_scope_cache_visibility"),
            pattern(r"visibility_scope=device_scope_vram_visibility"),
            pattern(r"packet_validation_row_count=25"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"sdma_packet_materialized_by_receipt_count=25"),
            pattern(r"packet_bytes_validated_by_receipt_count=25"),
            pattern(r"packet_validation_ready_count=25"),
            pattern(r"packet_offset_valid_count=25"),
            pattern(r"copy_payload_span_valid_count=24"),
            pattern(r"completion_signal_value_valid_count=1"),
            pattern(r"policy_wave_count=1"),
            pattern(r"policy_selected_count=1"),
            pattern(r"device_visibility_request_count=1"),
            pattern(r"cache_flush_request_count=0"),
            pattern(r"cache_invalidate_request_count=0"),
            pattern(r"vram_visibility_proven_count=0"),
            pattern(r"all_packets_validated_by_receipt=true"),
            pattern(r"packets_submittable_by_receipt=true"),
            pattern(r"all_cache_visibility_policies_selected_by_receipt=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"cache_visibility_policy_executable=false"),
            pattern(r"cache_flush_issued=false"),
            pattern(r"cache_invalidate_issued=false"),
            pattern(r"vram_visibility_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"packet_validation_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_packet_validation_handoff"
            ),
            pattern(rf"packet_validation_handoff_receipt_fingerprint={HEX64}"),
            pattern(
                r"cache_visibility_policy_input_receipt_kind=checkpoint_payload_cache_visibility_policy_input"
            ),
            pattern(rf"cache_visibility_policy_input_receipt_fingerprint={HEX64}"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-cache-visibility-policy-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_upload_completion_synchronization_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-upload-completion-synchronization-handoff-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_to_device_upload_completion_synchronization_handoff"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"runtime_input_count=8"),
            pattern(r"all_runtime_inputs_ready=true"),
            pattern(r"host_staging_pin_prerequisite_satisfied_by_receipt=true"),
            pattern(r"destination_residency_prerequisite_satisfied_by_receipt=true"),
            pattern(r"sdma_queue_reservation_prerequisite_satisfied_by_receipt=true"),
            pattern(
                r"copy_completion_signal_binding_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(
                r"upload_packet_materialization_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"upload_packet_validation_prerequisite_satisfied_by_receipt=true"),
            pattern(r"cache_visibility_policy_prerequisite_satisfied_by_receipt=true"),
            pattern(
                r"upload_completion_synchronization_prerequisite_satisfied_by_receipt=true"
            ),
            pattern(r"satisfied_prerequisite_count=8"),
            pattern(r"unsatisfied_prerequisite_count=0"),
            pattern(r"upload_completion_synchronization_input_index=7"),
            pattern(
                r"upload_completion_synchronization_requirement=upload_completion_synchronization"
            ),
            pattern(
                r"upload_completion_synchronization_next_action=plan_upload_completion_synchronization"
            ),
            pattern(
                r"upload_completion_synchronization_next_action_input=CheckpointPayloadUploadSynchronizationPlanInput"
            ),
            pattern(r"signal_kind=amd_signal_t"),
            pattern(r"completion_wait_kind=amd_signal_wait_acquire_cpu_only_plan"),
            pattern(
                r"synchronization_mode=sdma_completion_signal_wait_then_visibility_observation"
            ),
            pattern(r"policy_kind=sdma_completion_signal_then_device_scope_cache_visibility"),
            pattern(r"visibility_scope=device_scope_vram_visibility"),
            pattern(r"completion_signal_request_count=1"),
            pattern(r"completion_signal_bound_by_receipt_count=1"),
            pattern(r"signal_device_va_bound_by_receipt_count=1"),
            pattern(r"completion_packet_request_count=1"),
            pattern(r"queue_packet_request_count=25"),
            pattern(r"queue_packet_reserved_by_receipt_count=25"),
            pattern(r"sdma_packet_materialized_by_receipt_count=25"),
            pattern(r"packet_bytes_validated_by_receipt_count=25"),
            pattern(r"policy_wave_count=1"),
            pattern(r"policy_selected_count=1"),
            pattern(r"device_visibility_request_count=1"),
            pattern(r"vram_visibility_proven_count=0"),
            pattern(r"completion_wait_plan_count=1"),
            pattern(r"completion_wait_requested_count=1"),
            pattern(r"timeout_guard_request_count=1"),
            pattern(r"completion_wait_issued_count=0"),
            pattern(r"completion_wait_observed_count=0"),
            pattern(r"queue_synchronized_count=0"),
            pattern(r"copy_visible_to_device_count=0"),
            pattern(r"all_completion_signals_requested=true"),
            pattern(r"all_completion_waits_requested=true"),
            pattern(r"all_policy_waves_selected=true"),
            pattern(r"all_device_visibility_requests_selected=true"),
            pattern(r"all_completion_waits_observed=false"),
            pattern(r"all_cache_visibility_policies_selected_by_receipt=true"),
            pattern(r"all_upload_completion_synchronization_planned_by_receipt=true"),
            pattern(r"upload_ready=false"),
            pattern(r"next_actions_executed=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"destination_residency_proven=false"),
            pattern(r"cache_visibility_policy_executable=false"),
            pattern(r"upload_synchronization_executable=false"),
            pattern(r"completion_waits_issued=false"),
            pattern(r"queue_synchronized=false"),
            pattern(r"vram_visibility_proven=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(
                r"cache_visibility_policy_handoff_receipt_kind=checkpoint_payload_host_to_device_upload_cache_visibility_policy_handoff"
            ),
            pattern(rf"cache_visibility_policy_handoff_receipt_fingerprint={HEX64}"),
            pattern(
                r"upload_synchronization_plan_input_receipt_kind=checkpoint_payload_upload_synchronization_plan_input"
            ),
            pattern(rf"upload_synchronization_plan_input_receipt_fingerprint={HEX64}"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-upload-completion-synchronization-handoff.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_pin_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-pin-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_pin_request"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"used_staging_slot_count=1"),
            pattern(r"staging_slot_bytes=110592"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"host_pin_allocation_bytes=221184"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"pin_range_count=1"),
            pattern(r"total_pin_bytes=110592"),
            pattern(r"max_pin_range_bytes=110592"),
            pattern(r"page_pin_span_count=1"),
            pattern(r"total_page_pin_bytes=110592"),
            pattern(r"total_page_pin_slack_bytes=0"),
            pattern(r"max_page_pin_span_bytes=110592"),
            pattern(r"all_pin_ranges_within_staging=true"),
            pattern(r"all_pin_ranges_page_aligned=true"),
            pattern(r"all_page_pin_spans_within_allocation=true"),
            pattern(r"all_page_pin_spans_page_aligned=true"),
            pattern(r"host_virtual_addresses_bound=false"),
            pattern(r"host_page_addresses_materialized=false"),
            pattern(r"pin_request_executable=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"ranges\.count=1"),
            pattern(r"ranges\.0\.staging_slot=0"),
            pattern(r"ranges\.0\.wave_begin_index=0"),
            pattern(r"ranges\.0\.wave_end_index_exclusive=1"),
            pattern(r"ranges\.0\.staging_slot_epoch_begin=0"),
            pattern(r"ranges\.0\.staging_slot_epoch_end_exclusive=1"),
            pattern(r"ranges\.0\.pin_offset_begin=0"),
            pattern(r"ranges\.0\.pin_offset_end=110592"),
            pattern(r"ranges\.0\.page_aligned=true"),
            pattern(r"page_spans\.count=1"),
            pattern(r"page_spans\.0\.page_offset_begin=0"),
            pattern(r"page_spans\.0\.page_offset_end=110592"),
            pattern(r"page_spans\.0\.page_bytes=110592"),
            pattern(r"page_spans\.0\.covers_pin_ranges=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-pin-request.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_pin_virtual_address_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-pin-virtual-address-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_pin_virtual_address_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"used_staging_slot_count=1"),
            pattern(r"staging_slot_bytes=110592"),
            pattern(r"total_staging_bytes=221184"),
            pattern(r"host_pin_allocation_bytes=221184"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_staging_va_end=139637976948736"),
            pattern(r"host_pin_allocation_va_end=139637976948736"),
            pattern(r"pin_range_count=1"),
            pattern(r"virtual_range_count=1"),
            pattern(r"page_pin_span_count=1"),
            pattern(r"virtual_page_span_count=1"),
            pattern(r"total_pin_bytes=110592"),
            pattern(r"total_page_pin_bytes=110592"),
            pattern(r"total_page_pin_slack_bytes=0"),
            pattern(r"all_virtual_ranges_bound=true"),
            pattern(r"all_host_page_addresses_page_aligned=true"),
            pattern(r"host_virtual_addresses_bound=true"),
            pattern(r"host_page_addresses_materialized=true"),
            pattern(r"pin_request_executable=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"virtual_ranges\.count=1"),
            pattern(r"virtual_ranges\.0\.pin_offset_begin=0"),
            pattern(r"virtual_ranges\.0\.pin_offset_end=110592"),
            pattern(r"virtual_ranges\.0\.host_va_begin=139637976727552"),
            pattern(r"virtual_ranges\.0\.host_va_end=139637976838144"),
            pattern(r"virtual_ranges\.0\.host_virtual_address_bound=true"),
            pattern(r"virtual_page_spans\.count=1"),
            pattern(r"virtual_page_spans\.0\.page_offset_begin=0"),
            pattern(r"virtual_page_spans\.0\.page_offset_end=110592"),
            pattern(r"virtual_page_spans\.0\.page_count=27"),
            pattern(r"virtual_page_spans\.0\.host_page_va_begin=139637976727552"),
            pattern(r"virtual_page_spans\.0\.host_page_va_end=139637976838144"),
            pattern(r"virtual_page_spans\.0\.host_page_va_page_aligned=true"),
            pattern(r"virtual_page_spans\.0\.host_page_address_materialized=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-pin-virtual-address-plan.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_userptr_pin_arguments_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-userptr-pin-arguments-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_kfd_userptr_pin_argument_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"allocation_gpu_selection_policy=first_sorted_resident_gpu_id"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_staging_va_end=139637976948736"),
            pattern(r"host_pin_allocation_va_end=139637976948736"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"allocation_gpu_id=1001"),
            pattern(r"userptr_alloc_flags=2483027972"),
            pattern(r"host_virtual_address_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"host_virtual_address_plan_receipt_line_count=65"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"vm_acquire_performed_count=0"),
            pattern(r"userptr_alloc_performed_count=0"),
            pattern(r"handle_bound_count=0"),
            pattern(r"mmap_offset_bound_count=0"),
            pattern(r"pin_argument_ready_count=1"),
            pattern(r"all_host_page_addresses_materialized=true"),
            pattern(r"all_pin_arguments_ready=true"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"arguments\.count=1"),
            pattern(r"arguments\.0\.argument_index=0"),
            pattern(r"arguments\.0\.page_span_index=0"),
            pattern(r"arguments\.0\.pin_range_begin_index=0"),
            pattern(r"arguments\.0\.pin_range_end_index_exclusive=1"),
            pattern(r"arguments\.0\.pin_range_count=1"),
            pattern(r"arguments\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"arguments\.0\.alloc_memory_ioctl=3223866134"),
            pattern(r"arguments\.0\.alloc_args_va_addr=139637976727552"),
            pattern(r"arguments\.0\.alloc_args_size=110592"),
            pattern(r"arguments\.0\.alloc_args_gpu_id=1001"),
            pattern(r"arguments\.0\.alloc_args_flags=2483027972"),
            pattern(r"arguments\.0\.host_page_va_begin=139637976727552"),
            pattern(r"arguments\.0\.host_page_va_end=139637976838144"),
            pattern(r"arguments\.0\.host_page_va_span_bytes=110592"),
            pattern(r"arguments\.0\.page_count=27"),
            pattern(r"arguments\.0\.resident_gpu_ids\.count=2"),
            pattern(r"arguments\.0\.resident_gpu_ids\.values=1001,1002"),
            pattern(r"arguments\.0\.kfd_fd_required=true"),
            pattern(r"arguments\.0\.kfd_fd_bound=false"),
            pattern(r"arguments\.0\.vm_acquire_required=true"),
            pattern(r"arguments\.0\.vm_acquire_performed=false"),
            pattern(r"arguments\.0\.userptr_alloc_required=true"),
            pattern(r"arguments\.0\.userptr_alloc_performed=false"),
            pattern(r"arguments\.0\.handle_bound=false"),
            pattern(r"arguments\.0\.mmap_offset_bound=false"),
            pattern(r"arguments\.0\.pin_argument_ready=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-userptr-pin-arguments.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_kfd_vm_acquire_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-kfd-vm-acquire-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_kfd_vm_acquire_request_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"request_source=checkpoint_host_staging_kfd_vm_acquire_request_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"pin_argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"userptr_pin_argument_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"userptr_pin_argument_plan_receipt_line_count=66"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"vm_acquire_request_count=2"),
            pattern(r"kfd_fd_request_count=2"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"drm_fd_request_count=2"),
            pattern(r"drm_fd_bound_count=0"),
            pattern(r"vm_acquire_performed_count=0"),
            pattern(r"request_metadata_ready_count=2"),
            pattern(r"pin_argument_plan_ready=true"),
            pattern(r"all_request_metadata_ready=true"),
            pattern(r"all_kfd_fds_bound=false"),
            pattern(r"all_drm_fds_bound=false"),
            pattern(r"all_vms_acquired=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"userptr_alloc_performed_count=0"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"requests\.count=2"),
            pattern(r"requests\.0\.request_index=0"),
            pattern(r"requests\.0\.gpu_id=1001"),
            pattern(r"requests\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"requests\.0\.acquire_vm_ioctl=1074285333"),
            pattern(r"requests\.0\.kfd_fd_required=true"),
            pattern(r"requests\.0\.kfd_fd_bound=false"),
            pattern(r"requests\.0\.drm_fd_required=true"),
            pattern(r"requests\.0\.drm_fd_bound=false"),
            pattern(r"requests\.0\.vm_acquire_required=true"),
            pattern(r"requests\.0\.vm_acquire_performed=false"),
            pattern(r"requests\.0\.request_metadata_ready=true"),
            pattern(r"requests\.1\.request_index=1"),
            pattern(r"requests\.1\.gpu_id=1002"),
            pattern(r"requests\.1\.kfd_device_path=/dev/kfd"),
            pattern(r"requests\.1\.acquire_vm_ioctl=1074285333"),
            pattern(r"requests\.1\.kfd_fd_required=true"),
            pattern(r"requests\.1\.kfd_fd_bound=false"),
            pattern(r"requests\.1\.drm_fd_required=true"),
            pattern(r"requests\.1\.drm_fd_bound=false"),
            pattern(r"requests\.1\.vm_acquire_required=true"),
            pattern(r"requests\.1\.vm_acquire_performed=false"),
            pattern(r"requests\.1\.request_metadata_ready=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-kfd-vm-acquire-request.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_kfd_userptr_alloc_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-kfd-userptr-alloc-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_kfd_userptr_alloc_request_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"request_source=checkpoint_host_staging_kfd_userptr_alloc_request_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"pin_argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"vm_acquire_request_source=checkpoint_host_staging_kfd_vm_acquire_request_v0"),
            pattern(r"allocation_gpu_selection_policy=first_sorted_resident_gpu_id"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"allocation_gpu_id=1001"),
            pattern(r"userptr_alloc_flags=2483027972"),
            pattern(r"userptr_pin_argument_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"userptr_pin_argument_plan_receipt_line_count=66"),
            pattern(r"kfd_vm_acquire_request_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"kfd_vm_acquire_request_plan_receipt_line_count=62"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"alloc_request_count=1"),
            pattern(r"total_alloc_request_bytes=110592"),
            pattern(r"vm_acquire_request_count=2"),
            pattern(r"kfd_fd_request_count=1"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"drm_fd_request_count=2"),
            pattern(r"drm_fd_bound_count=0"),
            pattern(r"vm_acquire_performed_count=0"),
            pattern(r"userptr_alloc_performed_count=0"),
            pattern(r"handle_bound_count=0"),
            pattern(r"mmap_offset_bound_count=0"),
            pattern(r"request_metadata_ready_count=1"),
            pattern(r"pin_argument_plan_ready=true"),
            pattern(r"vm_acquire_request_metadata_ready=true"),
            pattern(r"all_request_metadata_ready=true"),
            pattern(r"all_kfd_fds_bound=false"),
            pattern(r"all_drm_fds_bound=false"),
            pattern(r"all_vms_acquired=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"requests\.count=1"),
            pattern(r"requests\.0\.request_index=0"),
            pattern(r"requests\.0\.argument_index=0"),
            pattern(r"requests\.0\.page_span_index=0"),
            pattern(r"requests\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"requests\.0\.alloc_memory_ioctl=3223866134"),
            pattern(r"requests\.0\.alloc_args_va_addr=139637976727552"),
            pattern(r"requests\.0\.alloc_args_size=110592"),
            pattern(r"requests\.0\.alloc_args_gpu_id=1001"),
            pattern(r"requests\.0\.alloc_args_flags=2483027972"),
            pattern(r"requests\.0\.host_page_va_span_bytes=110592"),
            pattern(r"requests\.0\.page_count=27"),
            pattern(r"requests\.0\.resident_gpu_ids\.values=1001,1002"),
            pattern(r"requests\.0\.kfd_fd_required=true"),
            pattern(r"requests\.0\.kfd_fd_bound=false"),
            pattern(r"requests\.0\.vm_acquire_required=true"),
            pattern(r"requests\.0\.vm_acquire_performed=false"),
            pattern(r"requests\.0\.userptr_alloc_required=true"),
            pattern(r"requests\.0\.userptr_alloc_performed=false"),
            pattern(r"requests\.0\.handle_bound=false"),
            pattern(r"requests\.0\.mmap_offset_bound=false"),
            pattern(r"requests\.0\.request_metadata_ready=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-kfd-userptr-alloc-request.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_kfd_userptr_alloc_result_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-kfd-userptr-alloc-result-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_kfd_userptr_alloc_result_binding_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"result_source=checkpoint_host_staging_kfd_userptr_alloc_result_v0"),
            pattern(r"request_source=checkpoint_host_staging_kfd_userptr_alloc_request_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"pin_argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"vm_acquire_request_source=checkpoint_host_staging_kfd_vm_acquire_request_v0"),
            pattern(r"allocation_gpu_selection_policy=first_sorted_resident_gpu_id"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"allocation_gpu_id=1001"),
            pattern(r"userptr_alloc_flags=2483027972"),
            pattern(r"kfd_userptr_alloc_request_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"kfd_userptr_alloc_request_plan_receipt_line_count=[0-9]+"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"alloc_request_count=1"),
            pattern(r"total_alloc_request_bytes=110592"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"kfd_fd_request_count=1"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"drm_fd_request_count=2"),
            pattern(r"drm_fd_bound_count=0"),
            pattern(r"vm_acquire_performed_count=0"),
            pattern(r"userptr_alloc_performed_count=0"),
            pattern(r"handle_bound_count=1"),
            pattern(r"mmap_offset_bound_count=1"),
            pattern(r"result_metadata_ready_count=1"),
            pattern(r"issue_count=0"),
            pattern(r"alloc_request_metadata_ready=true"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"allocation_result_bound=true"),
            pattern(r"all_kfd_fds_bound=false"),
            pattern(r"all_drm_fds_bound=false"),
            pattern(r"all_vms_acquired=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"results\.count=1"),
            pattern(r"results\.0\.result_index=0"),
            pattern(r"results\.0\.request_index=0"),
            pattern(r"results\.0\.argument_index=0"),
            pattern(r"results\.0\.page_span_index=0"),
            pattern(r"results\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"results\.0\.alloc_memory_ioctl=3223866134"),
            pattern(r"results\.0\.alloc_args_va_addr=139637976727552"),
            pattern(r"results\.0\.alloc_args_size=110592"),
            pattern(r"results\.0\.alloc_args_gpu_id=1001"),
            pattern(r"results\.0\.alloc_args_flags=2483027972"),
            pattern(r"results\.0\.host_page_va_span_bytes=110592"),
            pattern(r"results\.0\.page_count=27"),
            pattern(r"results\.0\.resident_gpu_ids\.values=1001,1002"),
            pattern(r"results\.0\.handle=2063597568"),
            pattern(r"results\.0\.mmap_offset=2600468480"),
            pattern(r"results\.0\.result_binding_present=true"),
            pattern(r"results\.0\.kfd_fd_bound=false"),
            pattern(r"results\.0\.vm_acquire_performed=false"),
            pattern(r"results\.0\.userptr_alloc_performed=false"),
            pattern(r"results\.0\.handle_bound=true"),
            pattern(r"results\.0\.mmap_offset_bound=true"),
            pattern(r"results\.0\.result_metadata_ready=true"),
            pattern(r"results\.0\.issue_count=0"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-kfd-userptr-alloc-result.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_kfd_map_memory_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-kfd-map-memory-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_kfd_map_memory_request_plan"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"request_source=checkpoint_host_staging_kfd_map_memory_request_v0"),
            pattern(r"result_source=checkpoint_host_staging_kfd_userptr_alloc_result_v0"),
            pattern(r"alloc_request_source=checkpoint_host_staging_kfd_userptr_alloc_request_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"pin_argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"vm_acquire_request_source=checkpoint_host_staging_kfd_vm_acquire_request_v0"),
            pattern(r"allocation_gpu_selection_policy=first_sorted_resident_gpu_id"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"allocation_gpu_id=1001"),
            pattern(r"userptr_alloc_flags=2483027972"),
            pattern(r"kfd_userptr_alloc_result_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"kfd_userptr_alloc_result_plan_receipt_line_count=[0-9]+"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"alloc_request_count=1"),
            pattern(r"total_alloc_request_bytes=110592"),
            pattern(r"alloc_result_count=1"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"map_memory_request_count=1"),
            pattern(r"total_map_memory_bytes=110592"),
            pattern(r"total_device_id_count=2"),
            pattern(r"kfd_fd_request_count=1"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"allocation_handle_bound_count=1"),
            pattern(r"device_ids_array_request_count=1"),
            pattern(r"device_ids_array_bound_count=0"),
            pattern(r"map_memory_performed_count=0"),
            pattern(r"map_memory_success_count=0"),
            pattern(r"request_metadata_ready_count=1"),
            pattern(r"alloc_result_binding_ready=true"),
            pattern(r"all_request_metadata_ready=true"),
            pattern(r"all_kfd_fds_bound=false"),
            pattern(r"all_device_ids_arrays_bound=false"),
            pattern(r"map_memory_performed=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"requests\.count=1"),
            pattern(r"requests\.0\.request_index=0"),
            pattern(r"requests\.0\.alloc_result_index=0"),
            pattern(r"requests\.0\.alloc_request_index=0"),
            pattern(r"requests\.0\.argument_index=0"),
            pattern(r"requests\.0\.page_span_index=0"),
            pattern(r"requests\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"requests\.0\.map_memory_ioctl=3222817560"),
            pattern(r"requests\.0\.map_args_handle=2063597568"),
            pattern(r"requests\.0\.map_args_device_ids_array_ptr=0"),
            pattern(r"requests\.0\.map_args_n_devices=2"),
            pattern(r"requests\.0\.map_args_n_success=0"),
            pattern(r"requests\.0\.host_page_va_begin=139637976727552"),
            pattern(r"requests\.0\.host_page_va_end=139637976838144"),
            pattern(r"requests\.0\.host_page_va_span_bytes=110592"),
            pattern(r"requests\.0\.page_count=27"),
            pattern(r"requests\.0\.resident_gpu_ids\.count=2"),
            pattern(r"requests\.0\.resident_gpu_ids\.values=1001,1002"),
            pattern(r"requests\.0\.handle_required=true"),
            pattern(r"requests\.0\.handle_bound=true"),
            pattern(r"requests\.0\.kfd_fd_required=true"),
            pattern(r"requests\.0\.kfd_fd_bound=false"),
            pattern(r"requests\.0\.device_ids_array_required=true"),
            pattern(r"requests\.0\.device_ids_array_bound=false"),
            pattern(r"requests\.0\.map_to_gpu_required=true"),
            pattern(r"requests\.0\.map_memory_performed=false"),
            pattern(r"requests\.0\.map_memory_successful=false"),
            pattern(r"requests\.0\.request_metadata_ready=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-kfd-map-memory-request.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_kfd_map_memory_argument_binding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-kfd-map-memory-argument-binding-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_staging_kfd_map_memory_argument_binding_plan"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"binding_source=checkpoint_host_staging_kfd_map_memory_argument_binding_v0"),
            pattern(r"request_source=checkpoint_host_staging_kfd_map_memory_request_v0"),
            pattern(r"result_source=checkpoint_host_staging_kfd_userptr_alloc_result_v0"),
            pattern(r"alloc_request_source=checkpoint_host_staging_kfd_userptr_alloc_request_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"pin_argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"vm_acquire_request_source=checkpoint_host_staging_kfd_vm_acquire_request_v0"),
            pattern(r"allocation_gpu_selection_policy=first_sorted_resident_gpu_id"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"allocation_gpu_id=1001"),
            pattern(r"userptr_alloc_flags=2483027972"),
            pattern(r"kfd_map_memory_request_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"kfd_map_memory_request_plan_receipt_line_count=[0-9]+"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"alloc_request_count=1"),
            pattern(r"total_alloc_request_bytes=110592"),
            pattern(r"alloc_result_count=1"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"map_memory_request_count=1"),
            pattern(r"device_ids_array_binding_count=1"),
            pattern(r"matched_device_ids_array_binding_count=1"),
            pattern(r"missing_device_ids_array_binding_count=0"),
            pattern(r"duplicate_device_ids_array_binding_count=0"),
            pattern(r"unmatched_device_ids_array_binding_count=0"),
            pattern(r"total_map_memory_bytes=110592"),
            pattern(r"total_device_id_count=2"),
            pattern(r"kfd_fd_request_count=1"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"allocation_handle_bound_count=1"),
            pattern(r"device_ids_array_bound_count=1"),
            pattern(r"device_ids_array_match_count=1"),
            pattern(r"map_memory_argument_ready_count=1"),
            pattern(r"map_memory_performed_count=0"),
            pattern(r"map_memory_success_count=0"),
            pattern(r"request_metadata_ready_count=1"),
            pattern(r"issue_count=0"),
            pattern(r"map_memory_request_metadata_ready=true"),
            pattern(r"alloc_result_binding_ready=true"),
            pattern(r"all_map_memory_arguments_ready=true"),
            pattern(r"all_kfd_fds_bound=false"),
            pattern(r"all_device_ids_arrays_bound=true"),
            pattern(r"map_memory_performed=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"bindings\.count=1"),
            pattern(r"bindings\.0\.binding_index=0"),
            pattern(r"bindings\.0\.request_index=0"),
            pattern(r"bindings\.0\.alloc_result_index=0"),
            pattern(r"bindings\.0\.alloc_request_index=0"),
            pattern(r"bindings\.0\.argument_index=0"),
            pattern(r"bindings\.0\.page_span_index=0"),
            pattern(r"bindings\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"bindings\.0\.map_memory_ioctl=3222817560"),
            pattern(r"bindings\.0\.map_args_handle=2063597568"),
            pattern(r"bindings\.0\.map_args_device_ids_array_ptr=1006632960"),
            pattern(r"bindings\.0\.map_args_n_devices=2"),
            pattern(r"bindings\.0\.map_args_n_success=0"),
            pattern(r"bindings\.0\.device_ids\.count=2"),
            pattern(r"bindings\.0\.device_ids\.values=1001,1002"),
            pattern(r"bindings\.0\.host_page_va_span_bytes=110592"),
            pattern(r"bindings\.0\.page_count=27"),
            pattern(r"bindings\.0\.resident_gpu_ids\.values=1001,1002"),
            pattern(r"bindings\.0\.request_metadata_ready=true"),
            pattern(r"bindings\.0\.handle_bound=true"),
            pattern(r"bindings\.0\.kfd_fd_bound=false"),
            pattern(r"bindings\.0\.device_ids_array_binding_present=true"),
            pattern(r"bindings\.0\.device_ids_array_bound=true"),
            pattern(r"bindings\.0\.device_ids_match_resident_gpu_ids=true"),
            pattern(r"bindings\.0\.map_to_gpu_required=true"),
            pattern(r"bindings\.0\.map_memory_argument_ready=true"),
            pattern(r"bindings\.0\.map_memory_performed=false"),
            pattern(r"bindings\.0\.map_memory_successful=false"),
            pattern(r"bindings\.0\.issue_count=0"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-kfd-map-memory-argument-binding.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_kfd_map_memory_result_binding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-kfd-map-memory-result-binding-receipt",
        ),
        required_patterns=(
            pattern(
                r"receipt\.kind=checkpoint_payload_host_staging_kfd_map_memory_result_binding_plan"
            ),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"result_source=checkpoint_host_staging_kfd_map_memory_result_binding_v0"),
            pattern(
                r"argument_binding_source=checkpoint_host_staging_kfd_map_memory_argument_binding_v0"
            ),
            pattern(r"request_source=checkpoint_host_staging_kfd_map_memory_request_v0"),
            pattern(r"alloc_result_source=checkpoint_host_staging_kfd_userptr_alloc_result_v0"),
            pattern(r"alloc_request_source=checkpoint_host_staging_kfd_userptr_alloc_request_v0"),
            pattern(r"kfd_device_path=/dev/kfd"),
            pattern(r"pin_argument_source=checkpoint_host_staging_userptr_pin_arguments_v0"),
            pattern(r"vm_acquire_request_source=checkpoint_host_staging_kfd_vm_acquire_request_v0"),
            pattern(r"allocation_gpu_selection_policy=first_sorted_resident_gpu_id"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"host_staging_base_va=139637976727552"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"resident_gpu_ids\.count=2"),
            pattern(r"resident_gpu_ids\.values=1001,1002"),
            pattern(r"allocation_gpu_id=1001"),
            pattern(r"userptr_alloc_flags=2483027972"),
            pattern(r"kfd_map_memory_argument_binding_plan_receipt_fingerprint=[0-9a-f]{64}"),
            pattern(r"kfd_map_memory_argument_binding_plan_receipt_line_count=[0-9]+"),
            pattern(r"pin_argument_count=1"),
            pattern(r"total_userptr_pin_bytes=110592"),
            pattern(r"alloc_request_count=1"),
            pattern(r"total_alloc_request_bytes=110592"),
            pattern(r"alloc_result_count=1"),
            pattern(r"upstream_result_binding_count=1"),
            pattern(r"upstream_matched_result_binding_count=1"),
            pattern(r"upstream_missing_result_binding_count=0"),
            pattern(r"upstream_duplicate_result_binding_count=0"),
            pattern(r"upstream_unmatched_result_binding_count=0"),
            pattern(r"map_memory_request_count=1"),
            pattern(r"argument_binding_count=1"),
            pattern(r"result_binding_count=1"),
            pattern(r"matched_result_binding_count=1"),
            pattern(r"missing_result_binding_count=0"),
            pattern(r"duplicate_result_binding_count=0"),
            pattern(r"unmatched_result_binding_count=0"),
            pattern(r"total_map_memory_bytes=110592"),
            pattern(r"total_device_id_count=2"),
            pattern(r"kfd_fd_request_count=1"),
            pattern(r"kfd_fd_bound_count=0"),
            pattern(r"allocation_handle_bound_count=1"),
            pattern(r"device_ids_array_bound_count=1"),
            pattern(r"device_ids_array_match_count=1"),
            pattern(r"map_memory_result_observed_count=1"),
            pattern(r"map_memory_success_count=1"),
            pattern(r"result_metadata_ready_count=1"),
            pattern(r"issue_count=0"),
            pattern(r"map_memory_argument_binding_ready=true"),
            pattern(r"all_result_bindings_ready=true"),
            pattern(r"all_map_memory_results_successful=true"),
            pattern(r"residency_proven=true"),
            pattern(r"map_memory_performed=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"queues_synchronized=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"results\.count=1"),
            pattern(r"results\.0\.result_index=0"),
            pattern(r"results\.0\.request_index=0"),
            pattern(r"results\.0\.alloc_result_index=0"),
            pattern(r"results\.0\.alloc_request_index=0"),
            pattern(r"results\.0\.argument_index=0"),
            pattern(r"results\.0\.page_span_index=0"),
            pattern(r"results\.0\.kfd_device_path=/dev/kfd"),
            pattern(r"results\.0\.map_memory_ioctl=3222817560"),
            pattern(r"results\.0\.map_args_handle=2063597568"),
            pattern(r"results\.0\.result_map_args_handle=2063597568"),
            pattern(r"results\.0\.map_args_device_ids_array_ptr=1006632960"),
            pattern(r"results\.0\.result_map_args_device_ids_array_ptr=1006632960"),
            pattern(r"results\.0\.map_args_n_devices=2"),
            pattern(r"results\.0\.result_map_args_n_devices=2"),
            pattern(r"results\.0\.map_args_n_success=0"),
            pattern(r"results\.0\.result_map_args_n_success=2"),
            pattern(r"results\.0\.device_ids\.values=1001,1002"),
            pattern(r"results\.0\.host_page_va_span_bytes=110592"),
            pattern(r"results\.0\.page_count=27"),
            pattern(r"results\.0\.resident_gpu_ids\.values=1001,1002"),
            pattern(r"results\.0\.argument_binding_ready=true"),
            pattern(r"results\.0\.result_binding_present=true"),
            pattern(r"results\.0\.handle_bound=true"),
            pattern(r"results\.0\.kfd_fd_bound=false"),
            pattern(r"results\.0\.device_ids_array_bound=true"),
            pattern(r"results\.0\.device_ids_match_resident_gpu_ids=true"),
            pattern(r"results\.0\.map_memory_result_observed=true"),
            pattern(r"results\.0\.map_memory_successful=true"),
            pattern(r"results\.0\.result_metadata_ready=true"),
            pattern(r"results\.0\.issue_count=0"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-kfd-map-memory-result-binding.receipt",
    ),
    ExampleGate(
        name="reference_moe_checkpoint_host_staging_pin_page_rounding_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "reference_moe_checkpoint_metadata",
            "--",
            "--checkpoint-host-staging-pin-page-rounding-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=checkpoint_payload_host_staging_pin_request"),
            pattern(r"target=mi355x-gfx950-raw-kfd-aql"),
            pattern(r"source_count=1"),
            pattern(r"slot_count=15"),
            pattern(r"batch_count=1"),
            pattern(r"copy_count=24"),
            pattern(r"wave_count=1"),
            pattern(r"staging_slot_count=2"),
            pattern(r"used_staging_slot_count=1"),
            pattern(r"staging_slot_bytes=108032"),
            pattern(r"total_staging_bytes=216064"),
            pattern(r"host_pin_allocation_bytes=217088"),
            pattern(r"total_copy_bytes=107456"),
            pattern(r"host_page_size_bytes=4096"),
            pattern(r"pin_range_count=1"),
            pattern(r"total_pin_bytes=108032"),
            pattern(r"max_pin_range_bytes=108032"),
            pattern(r"page_pin_span_count=1"),
            pattern(r"total_page_pin_bytes=110592"),
            pattern(r"total_page_pin_slack_bytes=2560"),
            pattern(r"max_page_pin_span_bytes=110592"),
            pattern(r"all_pin_ranges_within_staging=true"),
            pattern(r"all_pin_ranges_page_aligned=false"),
            pattern(r"all_page_pin_spans_within_allocation=true"),
            pattern(r"all_page_pin_spans_page_aligned=true"),
            pattern(r"host_virtual_addresses_bound=false"),
            pattern(r"host_page_addresses_materialized=false"),
            pattern(r"pin_request_executable=false"),
            pattern(r"pin_calls_issued=false"),
            pattern(r"host_staging_pinned=false"),
            pattern(r"vram_copied=false"),
            pattern(r"sdma_submitted=false"),
            pattern(r"aql_submitted=false"),
            pattern(r"kernels_executed=false"),
            pattern(r"live_execution_supported=false"),
            pattern(r"ranges\.count=1"),
            pattern(r"ranges\.0\.pin_offset_begin=0"),
            pattern(r"ranges\.0\.pin_offset_end=108032"),
            pattern(r"ranges\.0\.pin_bytes=108032"),
            pattern(r"ranges\.0\.page_aligned=false"),
            pattern(r"page_spans\.count=1"),
            pattern(r"page_spans\.0\.page_offset_begin=0"),
            pattern(r"page_spans\.0\.page_offset_end=110592"),
            pattern(r"page_spans\.0\.page_bytes=110592"),
            pattern(r"page_spans\.0\.covers_pin_ranges=true"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-reference-moe-checkpoint-host-staging-pin-page-rounding.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
        ),
        required_patterns=EXTERNAL_PLUGIN_PATTERNS,
        expected_lines_file=ROOT / "examples/model-api-plugin/expected-output.txt",
    ),
    ExampleGate(
        name="external_model_api_plugin_contract_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--model-api-contract-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_api_contract"),
            pattern(r"name=mainarch-model-api"),
            pattern(r"stability=pre1-static-metadata"),
            pattern(r"live_execution_supported=false"),
        ),
        expected_lines_file=ROOT / "examples/model-api-plugin/expected-contract.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_manifest_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--plugin-manifest-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_plugin_manifest"),
            pattern(r"model_name=external-mini-moe"),
            pattern(rf"manifest_fingerprint={HEX64}"),
            pattern(r"fingerprint_matches=true"),
            pattern(r"runtime_launch_request_steps\.count=10"),
            pattern(r"live_execution_supported=false"),
        )
        + PLUGIN_MANIFEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT / "examples/model-api-plugin/expected-manifest.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_compatibility_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--plugin-compatibility-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_plugin_compatibility"),
            pattern(r"model_name=external-mini-moe"),
            pattern(r"contract_matches=true"),
            pattern(r"target_matches=true"),
            pattern(r"fingerprint_matches=true"),
            pattern(r"accepted=true"),
            pattern(r"issues\.count=0"),
        ),
        expected_lines_file=ROOT / "examples/model-api-plugin/expected-compatibility.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_launch_request_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-launch-request-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_execution_request_plan"),
            pattern(r"dispatch_count=6"),
            pattern(r"runtime_request_plan_count=10"),
            pattern(r"component_pending_count=82"),
            pattern(r"live_aql_proof_surface_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"all_components_applied=false"),
        )
        + RUNTIME_LAUNCH_REQUEST_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-launch-request.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_submission_gate_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=6"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"all_components_applied=false"),
            pattern(r"all_live_aql_proof_validations_applied=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=82"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=9"),
            pattern(r"submission_blocker_count=11"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
        ),
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-submission-gate.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_resolved_submission_gate_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-resolved-submission-gate-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_gate"),
            pattern(r"dispatch_count=6"),
            pattern(r"window_count=4"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submitting_surface_count=0"),
            pattern(r"live_queue_mutating_component_count=0"),
            pattern(r"execution_blocker_count=0"),
            pattern(r"submission_blocker_count=0"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-resolved-submission-gate.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_resolved_submission_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-resolved-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=6"),
            pattern(r"window_count=4"),
            pattern(r"prerequisite_count=10"),
            pattern(r"satisfied_prerequisite_count=10"),
            pattern(r"unsatisfied_prerequisite_count=0"),
            pattern(r"next_action_count=0"),
            pattern(r"runtime_request_component_next_action_count=0"),
            pattern(r"live_aql_proof_validation_next_action_count=0"),
            pattern(r"pending_component_request_count=0"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_prerequisites_satisfied=true"),
            pattern(r"submission_ready=true"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.9\.prerequisite_satisfied=true"),
            pattern(r"prerequisites\.3\.next_action=none"),
            pattern(r"prerequisites\.9\.next_action=none"),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-resolved-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_resolved_submission_blocker_report_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-resolved-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=6"),
            pattern(r"window_count=4"),
            pattern(r"blocker_count=0"),
            pattern(r"execution_readiness_blocker_count=0"),
            pattern(r"runtime_request_component_pending_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=0"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=true"),
            pattern(r"all_components_applied=true"),
            pattern(r"all_live_aql_proof_validations_applied=true"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=true"),
            pattern(r"blockers\.count=0"),
        ),
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-resolved-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_submission_blocker_report_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-submission-blocker-report-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_blocker_report"),
            pattern(r"dispatch_count=6"),
            pattern(r"blocker_count=11"),
            pattern(r"execution_readiness_blocker_count=9"),
            pattern(r"runtime_request_component_pending_count=82"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_aql_submission_side_effect_count=0"),
            pattern(r"live_queue_mutation_count=0"),
            pattern(r"total_pending_count=84"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"no_live_aql_submission_side_effects=true"),
            pattern(r"no_live_queue_mutation=true"),
            pattern(r"submission_ready=false"),
            pattern(r"blockers\.count=11"),
            pattern(r"blockers\.9\.runtime_request_component_blocker=true"),
            pattern(r"blockers\.10\.live_aql_proof_validation_blocker=true"),
            pattern(r"live_aql_submission_side_effect_blocker=false"),
            pattern(r"live_queue_mutation_blocker=false"),
        ),
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-submission-blocker-report.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_runtime_submission_prerequisite_plan_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--runtime-submission-prerequisite-plan-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_runtime_launch_submission_prerequisite_plan"),
            pattern(r"dispatch_count=6"),
            pattern(r"prerequisite_count=10"),
            pattern(r"satisfied_prerequisite_count=0"),
            pattern(r"unsatisfied_prerequisite_count=10"),
            pattern(r"next_action_count=10"),
            pattern(r"runtime_request_component_next_action_count=8"),
            pattern(r"live_aql_proof_validation_next_action_count=2"),
            pattern(r"pending_component_request_count=82"),
            pattern(r"live_aql_proof_prerequisite_count=2"),
            pattern(r"live_aql_proof_input_count=2"),
            pattern(r"live_aql_validation_method_count=2"),
            pattern(r"live_aql_submitting_prerequisite_count=0"),
            pattern(r"live_aql_proof_validation_pending_count=2"),
            pattern(r"live_queue_mutating_prerequisite_count=0"),
            pattern(r"request_plan_ready=true"),
            pattern(r"execution_readiness_ready=false"),
            pattern(r"all_prerequisites_satisfied=false"),
            pattern(r"submission_ready=false"),
            pattern(r"prerequisites\.count=10"),
            pattern(r"prerequisites\.3\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.9\.live_aql_proof_required=true"),
            pattern(r"prerequisites\.3\.next_action=validate_live_aql_proof"),
            pattern(r"prerequisites\.3\.next_action_input=KfdQueueLiveAqlBatchReservationPlanInput"),
            pattern(r"prerequisites\.9\.next_action=validate_live_aql_proof"),
            pattern(
                r"prerequisites\.9\.next_action_input=KfdQueueLiveAqlMaterializedPacketPlanInput"
            ),
            pattern(r"live_aql_submits_work=false"),
            pattern(r"mutates_live_queue=false"),
        )
        + RUNTIME_SUBMISSION_PREREQUISITE_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS
        + RUNTIME_SUBMISSION_PREREQUISITE_NEXT_ACTION_LIVE_AQL_PROOF_KIND_RECEIPT_PATTERNS,
        expected_lines_file=ROOT
        / "examples/model-api-plugin/expected-runtime-submission-prerequisite-plan.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_static_handoff_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
            "--",
            "--static-handoff-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_plugin_static_handoff"),
            pattern(r"model_name=external-mini-moe"),
            pattern(rf"manifest\.receipt_fingerprint={HEX64}"),
            pattern(rf"compatibility\.receipt_fingerprint={HEX64}"),
            pattern(r"metadata_admitted=true"),
            pattern(r"launch_execution\.executable=false"),
            pattern(r"launch_execution\.unresolved_runtime_requirement_count=9"),
            pattern(
                r"launch_execution\.unresolved_runtime_requirements\.0="
                r"kernel_candidate_selection_policy"
            ),
            pattern(
                r"launch_execution\.unresolved_runtime_requirements\.8="
                r"aql_packet_materialization"
            ),
            pattern(r"gpu_buffers_allocated=false"),
            pattern(r"kernels_submitted=false"),
        ),
        expected_lines_file=ROOT / "examples/model-api-plugin/expected-static-handoff.receipt",
    ),
    ExampleGate(
        name="external_model_api_plugin_tests",
        command=(
            "cargo",
            "test",
            "-q",
            "--locked",
            "--manifest-path",
            "examples/model-api-plugin/Cargo.toml",
        ),
        required_patterns=(pattern(r"test result: ok\. 1 passed; 0 failed"),),
    ),
    ExampleGate(
        name="rejected_model_api",
        command=("cargo", "run", "-q", "-p", "mainarch-core", "--example", "rejected_model_api"),
        required_patterns=(
            pattern(rf"plugin_rejection: receipt_fingerprint={HEX64} rejected=true\b"),
            pattern(r"readiness_issue: kind=lowering_gap .*subject=ep_all_to_all\b"),
            pattern(r"compatibility_issue: kind=static_metadata\b"),
        ),
    ),
    ExampleGate(
        name="rejected_model_api_rejection_receipt",
        command=(
            "cargo",
            "run",
            "-q",
            "-p",
            "mainarch-core",
            "--example",
            "rejected_model_api",
            "--",
            "--rejection-receipt",
        ),
        required_patterns=(
            pattern(r"receipt\.kind=model_plugin_rejection"),
            pattern(r"summary\.model_name=unsupported-collective-plugin"),
            pattern(r"summary\.accepted=false"),
            pattern(r"summary\.static_ready=false"),
            pattern(r"rejection_issue_count=4"),
            pattern(r"readiness_issue_count=3"),
            pattern(r"compatibility_issue_count=1"),
            pattern(r"lowering_gap_op_names\.count=1"),
            pattern(r"lowering_gap_op_names\.0=ep_all_to_all"),
            pattern(r"stage_gap_names\.0=expert_parallel"),
            pattern(r"binding_issue_tensor_names\.0=expert_input"),
            pattern(r"readiness_issues\.1\.kind=lowering_gap"),
            pattern(r"readiness_issues\.1\.subject=ep_all_to_all"),
            pattern(r"compatibility_issues\.0\.kind=static_metadata"),
        ),
        expected_lines_file=ROOT
        / "crates/mainarch-core/examples/expected-rejected-model-api-rejection.receipt",
    ),
)

PUBLIC_SOURCE_IMPORT_GUARDS = (
    SourceImportSurfaceGuard(
        path=ROOT / "examples/model-api-plugin/src/lib.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "examples/model-api-plugin/src/main.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "examples/model-api-plugin/tests/public_contract.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "crates/mainarch-core/tests/model_api_public_contract.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "crates/mainarch-core/examples/rejected_model_api.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "crates/mainarch-core/examples/reference_moe_model_api.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "crates/mainarch-core/examples/custom_model_api.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "crates/mainarch-core/examples/reference_moe_checkpoint_metadata.rs",
        allowed_mainarch_core_lines=(
            "use mainarch_core::model_api::prelude::*;",
            "use mainarch_core::weights::{",
        ),
    ),
    SourceImportSurfaceGuard(
        path=ROOT / "crates/mainarch-cli/src/bin/model_api_selftest.rs",
        allowed_mainarch_core_lines=("use mainarch_core::model_api::prelude::*;",),
    ),
)


PUBLIC_SOURCE_PATTERN_GUARDS = (
    PublicSourcePatternGuard(
        path=ROOT / "examples/model-api-plugin/src/main.rs",
        patterns=(
            pattern(r"\bprojection_selection\s*\.\s*selection_request_op_kernel_symbols\(\)"),
            pattern(
                r"\bprojection_selection\s*\.\s*selection_request_op_kernel_symbol_labels\(\)"
            ),
            pattern(r"\bkernel_selection\s*\.\s*selection_request_op_kernel_symbols\(\)"),
            pattern(
                r"\bkernel_selection\s*\.\s*selection_request_op_kernel_symbol_labels\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*branch_resolution_request_op_names\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*"
                r"branch_resolution_request_candidate_symbol_sets\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*"
                r"branch_resolution_request_candidate_symbol_labels\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*unresolved_candidate_symbols\(\)"
            ),
        ),
        label="external plugin structured helper calls",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "examples/model-api-plugin/tests/public_contract.rs",
        patterns=(
            pattern(r"\bprojection_selection\s*\.\s*selection_request_op_kernel_symbols\(\)"),
            pattern(
                r"\bprojection_selection\s*\.\s*selection_request_op_kernel_symbol_labels\(\)"
            ),
            pattern(r"\bkernel_selection\s*\.\s*selection_request_op_kernel_symbols\(\)"),
            pattern(
                r"\bkernel_selection\s*\.\s*selection_request_op_kernel_symbol_labels\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*"
                r"branch_resolution_request_candidate_symbol_sets\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*branch_resolution_request_op_names\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*"
                r"branch_resolution_request_candidate_symbol_labels\(\)"
            ),
            pattern(
                r"\bhost_launcher_branch_requests\s*\.\s*unresolved_candidate_symbols\(\)"
            ),
        ),
        label="external plugin structured helper contract coverage",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "examples/model-api-plugin/README.md",
        patterns=(
            pattern(re.escape("`(op_name, kernel_symbol)`")),
            pattern(re.escape("`(op_name, candidate_symbols)`")),
            pattern(re.escape("ready request bindings or branch alternatives")),
        ),
        label="external plugin structured helper docs",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "crates/mainarch-core/src/model_api.rs",
        patterns=(
            pattern(
                r"fn\s+model_plugin_static_handoff_receipt_from[\s\S]*?"
                r"launch_execution\s*\.\s*assert_non_executable_boundary\(\)\?;[\s\S]*?"
                r"execution_requests\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"receipt\s*\.\s*assert_non_executing_boundary\(\)\?;"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_execution_request_plan[\s\S]*?"
                r"execution_requests\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(execution_requests\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_submission_gate[\s\S]*?"
                r"submission_gate\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(submission_gate\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_submission_blocker_report[\s\S]*?"
                r"blocker_report\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(blocker_report\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_submission_prerequisite_plan[\s\S]*?"
                r"prerequisite_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(prerequisite_plan\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_live_aql_proof_validation_application_plan[\s\S]*?"
                r"validation_application_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(validation_application_plan\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_runtime_request_component_application_plan[\s\S]*?"
                r"validation_application_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"prerequisites_with_validations\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"runtime_component_application_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_runtime_request_component_application_receipt_plan[\s\S]*?"
                r"receipt_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(receipt_plan\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_plan[\s\S]*?"
                r"resolution_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(resolution_plan\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_execution_readiness_blocker_resolution_receipt_plan[\s\S]*?"
                r"receipt_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(receipt_plan\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_submission_prerequisite_plan_with_execution_readiness_blocker_resolution_receipt_plan[\s\S]*?"
                r"prerequisites_after_execution_readiness_receipts\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(prerequisites_after_execution_readiness_receipts\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_submission_gate_with_execution_readiness_blocker_resolution_receipt_plan[\s\S]*?"
                r"submission_gate\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(submission_gate\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_runtime_launch_submission_blocker_report_with_execution_readiness_blocker_resolution_receipt_plan[\s\S]*?"
                r"blocker_report\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(blocker_report\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_resolved_submission_gate[\s\S]*?"
                r"resolved_submission_gate\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(resolved_submission_gate\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_resolved_submission_prerequisite_plan[\s\S]*?"
                r"resolved_prerequisites\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(resolved_prerequisites\)"
            ),
            pattern(
                r"pub\s+fn\s+synthetic_cpu_resolved_submission_blocker_report[\s\S]*?"
                r"blocker_report\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"Ok\(blocker_report\)"
            ),
            pattern(
                r"fn\s+synthetic_cpu_runtime_launch_submission_prerequisite_plan_after_runtime_request_component_application_receipts[\s\S]*?"
                r"validation_application_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"prerequisites_with_validations\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"runtime_component_application_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"runtime_component_receipt_plan\s*\.\s*assert_non_submitting_boundary\(\)\?;[\s\S]*?"
                r"prerequisites_after_runtime_components\s*\.\s*assert_non_submitting_boundary\(\)\?;"
            ),
        ),
        label="model API synthetic CPU boundary assertions",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "crates/mainarch-core/src/model_api.rs",
        patterns=(
            pattern(
                r"pub\s+struct\s+ModelPrimitiveLoweringCatalogCodeObjectKernelCoverageReport[\s\S]*?"
                r"pub\s+unmapped_entrypoint_count:\s+usize,[\s\S]*?"
                r"pub\s+missing_kernel_count:\s+usize,[\s\S]*?"
                r"pub\s+missing_kernel_symbols:\s+Vec<String>,"
            ),
            pattern(
                r"impl\s+ModelPrimitiveLoweringCatalogCodeObjectKernelCoverageReport[\s\S]*?"
                r"pub\s+fn\s+assert_complete\(&self\)\s*->\s*Result<\(\)>[\s\S]*?"
                r"unmapped entrypoints[\s\S]*?"
                r"missing code-object kernels"
            ),
            pattern(
                r"pub\s+fn\s+code_object_kernel_coverage_report[\s\S]*?"
                r"descriptor\s*\.\s*assert_consistent\(\)\?;[\s\S]*?"
                r"kernel_symbols_for_host_launcher\(entrypoint\)[\s\S]*?"
                r"code_object\s*\.\s*contains_kernel\(symbol\)"
            ),
            pattern(
                r'"gpu_paged_mla_fp8_splitk_stage1_selftest"\s*=>\s*Some\(\&\["paged_mla_fp8_splitk_stage1_probe"\]\)'
            ),
            pattern(
                r'"gpu_paged_mla_fp8_splitk_stage2_selftest"[\s\S]*?'
                r'"paged_mla_fp8_splitk_stage2_merge_probe"'
            ),
            pattern(
                r'"gpu_paged_mla_fp8_splitk_e2e_selftest"[\s\S]*?'
                r'"paged_mla_fp8_splitk_stage1_probe"[\s\S]*?'
                r'"paged_mla_fp8_splitk_stage2_merge_probe"'
            ),
        ),
        label="model API catalog code-object kernel coverage",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "crates/mainarch-core/src/model_api.rs",
        patterns=(
            pattern(
                r"pub\s+struct\s+ModelPrimitiveLoweringCatalogAbiRegistryCoverageReport[\s\S]*?"
                r"pub\s+missing_named_abi_schema_count:\s+usize,[\s\S]*?"
                r"pub\s+missing_semantic_abi_schema_count:\s+usize,[\s\S]*?"
                r"pub\s+named_semantic_shape_mismatch_count:\s+usize,"
            ),
            pattern(
                r"impl\s+ModelPrimitiveLoweringCatalogAbiRegistryCoverageReport[\s\S]*?"
                r"pub\s+fn\s+assert_complete\(&self\)\s*->\s*Result<\(\)>[\s\S]*?"
                r"missing named ABI schemas[\s\S]*?"
                r"missing semantic ABI schemas[\s\S]*?"
                r"ABI schema shape mismatches"
            ),
            pattern(
                r"pub\s+fn\s+abi_registry_coverage_report[\s\S]*?"
                r"code_object_kernel_coverage_report\(code_object\)\?[\s\S]*?"
                r"runtime_launch_kernel_argument_abi_schema_for\(symbol\)[\s\S]*?"
                r"runtime_launch_kernel_argument_abi_semantic_schema_for\(symbol\)[\s\S]*?"
                r"named_semantic_shape_mismatch_symbols"
            ),
            pattern(
                r'kernel_symbol:\s*"paged_mla_fp8_splitk_stage1_probe"[\s\S]*?'
                r"kernarg_size:\s*400,[\s\S]*?"
                r"kernarg_segment_align:\s*8,"
            ),
            pattern(
                r'kernel_symbol:\s*"paged_mla_fp8_splitk_stage2_merge_probe"[\s\S]*?'
                r"kernarg_size:\s*304,[\s\S]*?"
                r"kernarg_segment_align:\s*8,"
            ),
            pattern(
                r'kernel_symbol:\s*"paged_mla_fp8_splitk_stage1_probe"[\s\S]*?'
                r'semantic_ptr_field\(0,\s*"q_nope",\s*"q_nope",\s*0\)[\s\S]*?'
                r'semantic_u32_field\(24,\s*"internal_split_mode",\s*"internal_split_mode",\s*140\)'
            ),
            pattern(
                r'kernel_symbol:\s*"paged_mla_fp8_splitk_stage2_merge_probe"[\s\S]*?'
                r'semantic_ptr_field\(0,\s*"partial_workspace",\s*"partial_workspace",\s*0\)[\s\S]*?'
                r'semantic_u32_field\(8,\s*"output_index",\s*"output_index",\s*44\)'
            ),
        ),
        label="model API catalog ABI registry coverage",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "crates/mainarch-core/tests/model_api_public_contract.rs",
        patterns=(
            pattern(
                r"catalog\s*\.\s*code_object_kernel_coverage_report\(&code_object\)\?"
            ),
            pattern(
                r"catalog_code_object_coverage\s*\.\s*assert_complete\(\)\?;[\s\S]*?"
                r"assert!\(catalog_code_object_coverage\s*\.\s*is_complete\(\)\)"
            ),
            pattern(r"catalog_code_object_coverage\.non_gap_case_count,\s*17"),
            pattern(
                r"gpu_paged_mla_fp8_splitk_e2e_selftest[\s\S]*?"
                r"paged_mla_fp8_splitk_stage1_probe[\s\S]*?"
                r"paged_mla_fp8_splitk_stage2_merge_probe"
            ),
            pattern(
                r"missing_symbol_coverage[\s\S]*?"
                r"not_a_mainarch_kernel[\s\S]*?"
                r"assert_complete\(\)[\s\S]*?"
                r"missing code-object kernels 1"
            ),
        ),
        label="model API catalog code-object coverage public contract",
    ),
    PublicSourcePatternGuard(
        path=ROOT / "crates/mainarch-core/tests/model_api_public_contract.rs",
        patterns=(
            pattern(
                r"catalog\s*\.\s*abi_registry_coverage_report\(&code_object\)\?"
            ),
            pattern(
                r"catalog_abi_registry_coverage\s*\.\s*assert_complete\(\)\?;[\s\S]*?"
                r"assert!\(catalog_abi_registry_coverage\s*\.\s*is_complete\(\)\)"
            ),
            pattern(
                r"catalog_abi_registry_coverage\.missing_named_abi_schema_count,\s*0[\s\S]*?"
                r"catalog_abi_registry_coverage\.missing_semantic_abi_schema_count,\s*0[\s\S]*?"
                r"catalog_abi_registry_coverage\.named_semantic_shape_mismatch_count,\s*0"
            ),
            pattern(
                r"paged_mla_fp8_splitk_stage1_probe[\s\S]*?"
                r"code_object_kernarg_size,\s*Some\(400\)[\s\S]*?"
                r"semantic_kernarg_size,\s*Some\(400\)"
            ),
            pattern(
                r"paged_mla_fp8_splitk_stage2_merge_probe[\s\S]*?"
                r"code_object_kernarg_size,\s*Some\(304\)[\s\S]*?"
                r"semantic_kernarg_size,\s*Some\(304\)"
            ),
            pattern(
                r"missing_named_abi_coverage[\s\S]*?"
                r"not_a_mainarch_kernel[\s\S]*?"
                r"missing named ABI schemas 1"
            ),
            pattern(
                r"mismatched_abi_coverage[\s\S]*?"
                r"named_semantic_shape_matches\s*=\s*false[\s\S]*?"
                r"ABI schema shape mismatches 1"
            ),
        ),
        label="model API catalog ABI registry coverage public contract",
    ),
)


PUBLIC_EXAMPLE_CHECK_COMMAND = "python3 tools/check_model_api_public_examples.py"


def public_gate_doc_command(gate: ExampleGate) -> str:
    return " ".join(token for token in gate.command if token != "-q")


def public_command_doc_requirements() -> tuple[str, ...]:
    return (PUBLIC_EXAMPLE_CHECK_COMMAND,) + tuple(
        public_gate_doc_command(gate) for gate in GATES
    )


def standalone_plugin_doc_requirements() -> tuple[str, ...]:
    return tuple(
        public_gate_doc_command(gate)
        for gate in GATES
        if "--manifest-path" in gate.command
        and "examples/model-api-plugin/Cargo.toml" in gate.command
    )


PUBLIC_COMMAND_DOC_GUARDS = (
    PublicCommandDocGuard(
        path=ROOT / "README.md",
        commands=public_command_doc_requirements(),
        label="public commands",
    ),
    PublicCommandDocGuard(
        path=ROOT / "CONTRIBUTING.md",
        commands=public_command_doc_requirements(),
        label="public commands",
    ),
    PublicCommandDocGuard(
        path=ROOT / "docs/model-api.md",
        commands=public_command_doc_requirements(),
        label="public commands",
    ),
    PublicCommandDocGuard(
        path=ROOT / "docs/release-checklist.md",
        commands=public_command_doc_requirements(),
        label="public commands",
    ),
    PublicCommandDocGuard(
        path=ROOT / "docs/hardware-support.md",
        commands=public_command_doc_requirements(),
        label="public commands",
    ),
    PublicCommandDocGuard(
        path=ROOT / "examples/model-api-plugin/README.md",
        commands=standalone_plugin_doc_requirements(),
        label="standalone plugin commands",
    ),
)


PUBLIC_DOC_SNIPPET_GUARDS = (
    PublicDocSnippetGuard(
        path=ROOT / "docs/release-checklist.md",
        snippets=(
            "The package carries exact contract, manifest, compatibility,\n"
            "runtime launch request, runtime submission gate, resolved runtime submission\n"
            "gate, resolved runtime submission prerequisite plan, resolved runtime\n"
            "submission blocker report, runtime submission blocker report, runtime\n"
            "submission prerequisite plan, and full static handoff receipt fixtures.",
        ),
        label="external package fixture summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "docs/release-checklist.md",
        snippets=(
            "It also checks source patterns in the external plugin binary and package\n"
            "integration test for receiver-qualified structured-helper call syntax, and that\n"
            "the plugin README documents `(op_name, kernel_symbol)`,\n"
            "`(op_name, candidate_symbols)`, and the ready request binding/branch-alternative\n"
            "boundary.",
        ),
        label="external plugin structured helper source guard summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "README.md",
        snippets=(
            "execution-request synthetic\n"
            "CPU-only resolved submission-prerequisite, submission-gate, and blocker-report\n"
            "helpers plus",
            "overlays, and resolved prerequisite-plan, submission-gate, and blocker-report\n"
            "helpers for deterministic handoff fixtures,",
            "prints its deterministic model API contract receipt, plugin manifest,\n"
            "compatibility receipt, runtime launch request receipt, runtime submission gate\n"
            "receipt, resolved runtime submission gate receipt, resolved runtime submission\n"
            "prerequisite plan receipt, resolved runtime submission blocker report receipt,\n"
            "runtime submission blocker report receipt, runtime submission prerequisite plan\n"
            "receipt, and report-level runtime submission prerequisite plan helper receipt;",
            "prerequisite/submission-gate/blocker-report overlay helpers plus the resolved\n"
            "prerequisite-plan, submission-gate, and blocker-report helpers, pins the",
        ),
        label="external package runtime receipt summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "README.md",
        snippets=(
            "The CLI selftest now also prints\n"
            "runtime launch request, runtime submission gate, resolved runtime submission\n"
            "gate, resolved runtime submission prerequisite plan, resolved runtime submission\n"
            "blocker report, runtime submission blocker report, runtime submission\n"
            "prerequisite plan, and full static handoff receipts for the reduced reference\n"
            "graph with the same manifest/compatibility bindings, ordered unresolved\n"
            "execution requirement labels, and explicit non-execution counters.",
        ),
        label="CLI selftest runtime receipt summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "docs/model-api.md",
        snippets=(
            "`crates/mainarch-cli/expected-model-api-selftest-static-handoff.receipt`, which\n"
            "pins the same full static handoff boundary through the `mainarch-cli` package,\n"
            "including ordered unresolved execution requirement labels and explicit\n"
            "non-execution counters.",
        ),
        label="CLI selftest static handoff fixture summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "docs/release-checklist.md",
        snippets=(
            "The CLI selftest static handoff\n"
            "fixture pins the same full static handoff boundary through the `mainarch-cli`\n"
            "package, including ordered unresolved execution requirement labels and explicit\n"
            "non-execution counters.",
        ),
        label="release CLI selftest static handoff summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "examples/model-api-plugin/README.md",
        snippets=(
            "helper receipt, report-level runtime submission gate helper receipt, resolved\n"
            "runtime submission gate helper receipt, resolved runtime submission\n"
            "prerequisite-plan helper receipt, resolved runtime submission blocker-report\n"
            "helper receipt, report-level runtime submission blocker report helper receipt,\n"
            "report-level runtime submission prerequisite plan helper receipt, report-level",
            "request receipt, runtime submission gate receipt, resolved runtime submission\n"
            "gate receipt, resolved runtime submission prerequisite plan receipt, resolved\n"
            "runtime submission blocker report receipt, runtime submission blocker report\n"
            "receipt, runtime submission prerequisite plan receipt, handoff receipt, or\n"
            "supported boundary intentionally changes.",
            "handoff helper, report-level static handoff readiness helper, static handoff\n"
            "requirement lookup helper, static handoff non-execution boundary helper, launch\n"
            "execution non-executable boundary helper, execution request non-submitting\n"
            "boundary helper, submission gate non-submitting boundary helper, submission\n"
            "blocker-report non-submitting boundary helper, submission prerequisite-plan\n"
            "non-submitting boundary helper, runtime component application non-submitting\n"
            "boundary helper, runtime component application receipt non-submitting boundary\n"
            "helper, runtime component receipt-plan non-submitting boundary helper,\n"
            "execution-readiness resolution non-submitting boundary helper,\n"
            "execution-readiness resolution receipt non-submitting boundary helper,\n"
            "execution-readiness resolution receipt-plan non-submitting boundary helper,\n"
            "resolved prerequisite-plan helper, resolved submission-gate helper, resolved\n"
            "blocker-report helper, and non-executable launch boundary.",
            "surfaces the static handoff's manifest\n"
            "and compatibility receipt fingerprints, ordered unresolved execution requirement\n"
            "labels, and non-execution counters on its compact handoff line.",
        ),
        label="standalone plugin runtime receipt summary",
    ),
    PublicDocSnippetGuard(
        path=ROOT / "docs/api-stability.md",
        snippets=(
            "`ModelPluginStaticHandoffReceipt`",
            "`ModelPluginInspectionReport::is_static_handoff_ready()`",
            "`ModelPluginInspectionReport::assert_static_handoff_ready()`",
            "passing the same inspection consistency checks",
            "`ModelPluginCompatibilityReport::assert_consistent`",
            "`ModelPluginRejectionReport::assert_consistent`",
            "`ModelRuntimeMetadataAdmissionReport::assert_consistent`",
            "passing the same report consistency checks",
            "`ModelPluginStaticHandoffReceipt::unresolved_runtime_requirement_names()`",
            "`ModelPluginStaticHandoffReceipt::has_unresolved_runtime_requirement(...)`",
            "`ModelPluginStaticHandoffReceipt::is_non_executing_boundary()`",
            "`ModelPluginStaticHandoffReceipt::assert_non_executing_boundary()`",
            "passing the same static-handoff consistency checks",
            "`ModelRuntimeLaunchExecutionReadinessReport::is_non_executable_boundary()`",
            "`ModelRuntimeLaunchExecutionReadinessReport::assert_non_executable_boundary()`",
            "`ModelRuntimeLaunchExecutionRequestPlan::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchExecutionRequestPlan::assert_non_submitting_boundary()`",
            "passing the same request-plan consistency checks",
            "`ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchRuntimeRequestComponentApplicationPlan::assert_non_submitting_boundary()`",
            "passing the same application-plan consistency checks",
            "`RuntimeLaunchRuntimeRequestComponentApplicationReceipt::is_non_submitting_boundary()`",
            "`RuntimeLaunchRuntimeRequestComponentApplicationReceipt::assert_non_submitting_boundary()`",
            "`ModelRuntimeLaunchRuntimeRequestComponentApplicationReceiptPlan::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchRuntimeRequestComponentApplicationReceiptPlan::assert_non_submitting_boundary()`",
            "passing the same receipt-plan consistency checks",
            "`ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchExecutionReadinessBlockerResolutionPlan::assert_non_submitting_boundary()`",
            "passing the same worklist consistency checks",
            "`RuntimeLaunchExecutionReadinessBlockerResolutionReceipt::is_non_submitting_boundary()`",
            "`RuntimeLaunchExecutionReadinessBlockerResolutionReceipt::assert_non_submitting_boundary()`",
            "`ModelRuntimeLaunchExecutionReadinessBlockerResolutionReceiptPlan::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchExecutionReadinessBlockerResolutionReceiptPlan::assert_non_submitting_boundary()`",
            "`ModelRuntimeLaunchSubmissionPrerequisitePlan::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchSubmissionPrerequisitePlan::assert_non_submitting_boundary()`",
            "passing the same prerequisite-plan consistency checks",
            "`ModelRuntimeLaunchSubmissionGate::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchSubmissionGate::assert_non_submitting_boundary()`",
            "passing the same submission-gate consistency checks",
            "`ModelRuntimeLaunchSubmissionBlockerReport::is_non_submitting_boundary()`",
            "`ModelRuntimeLaunchSubmissionBlockerReport::assert_non_submitting_boundary()`",
            "passing the same blocker-report consistency checks",
            "`manifest.receipt_fingerprint`",
            "`compatibility.receipt_fingerprint`",
            "`launch_execution.executable=false`",
            "`launch_execution.unresolved_runtime_requirement_count`",
            "`launch_execution.unresolved_runtime_requirements.*`",
            "`launch_execution.aql_dispatchable_packet_count=0`",
            "`live_aql_submitting_surface_count=0`",
            "`live_queue_mutating_component_count=0`",
            "`live_execution_supported=false`",
            "`gpu_buffers_allocated=false`",
            "`kernels_submitted=false`",
        ),
        label="static handoff stability rows",
    ),
)


# --- public-repository doc coupling -------------------------------------------
# Upstream, these guards also pinned exact prose in README.md, CONTRIBUTING.md,
# and a release checklist, so that the human-facing text could not drift from the
# API surface. In the public repository those two files are written for readers
# rather than as a machine-checked mirror of the receipt vocabulary, and the
# release checklist is an internal process document that is not published. The
# reference docs, meaning docs/model-api.md, docs/api-stability.md,
# docs/hardware-support.md and the standalone plugin README, stay coupled,
# because those *are* the API's written contract.
DOC_GUARD_UNCOUPLED_PATHS = frozenset(
    {
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "docs/release-checklist.md",
    }
)

PUBLIC_COMMAND_DOC_GUARDS = tuple(
    guard
    for guard in PUBLIC_COMMAND_DOC_GUARDS
    if guard.path not in DOC_GUARD_UNCOUPLED_PATHS
)
PUBLIC_DOC_SNIPPET_GUARDS = tuple(
    guard
    for guard in PUBLIC_DOC_SNIPPET_GUARDS
    if guard.path not in DOC_GUARD_UNCOUPLED_PATHS
)

# Filtering must never quietly empty a guard set: that would turn a real check
# into a silent pass.
if not PUBLIC_COMMAND_DOC_GUARDS or not PUBLIC_DOC_SNIPPET_GUARDS:
    raise RuntimeError("doc guard filtering removed every guard")
for _guard in (*PUBLIC_COMMAND_DOC_GUARDS, *PUBLIC_DOC_SNIPPET_GUARDS):
    if not _guard.path.exists():
        raise RuntimeError(f"doc guard targets a missing file: {_guard.path}")


def has_mainarch_core_source_reference(stripped: str) -> bool:
    return (
        MAINARCH_CORE_PATH_RE.search(stripped) is not None
        or MAINARCH_CORE_USE_RE.search(stripped) is not None
        or MAINARCH_CORE_EXTERN_RE.search(stripped) is not None
    )


def relative_source_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def disallowed_mainarch_core_source_references(
    path: Path,
    lines: list[str],
    allowed_mainarch_core_lines: tuple[str, ...],
) -> list[str]:
    disallowed = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if (
            has_mainarch_core_source_reference(stripped)
            and stripped not in allowed_mainarch_core_lines
        ):
            disallowed.append(f"{relative_source_label(path)}:{line_number}: {stripped}")
    return disallowed


def require_public_source_import_surface_guard(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"public source import surface guard self-test failed: {detail}")


def check_public_source_import_surface_guard() -> None:
    allowed = ("use mainarch_core::model_api::prelude::*;",)
    disallowed_cases = (
        "use mainarch_core :: model_api::prelude::*;",
        "pub use mainarch_core as core_api;",
        "extern crate mainarch_core;",
        "let _ = mainarch_core :: CodeObjectInfo;",
    )
    allowed_result = disallowed_mainarch_core_source_references(
        Path("guard.rs"),
        [allowed[0]],
        allowed,
    )
    require_public_source_import_surface_guard(
        allowed_result == [],
        f"allowed prelude import rejected as {allowed_result}",
    )
    for case in disallowed_cases:
        disallowed = disallowed_mainarch_core_source_references(
            Path("guard.rs"),
            [case],
            allowed,
        )
        expected = [f"guard.rs:1: {case}"]
        require_public_source_import_surface_guard(
            disallowed == expected,
            f"{case!r} produced {disallowed}, expected {expected}",
        )


def check_public_source_import_surface() -> None:
    print("model API public source import surface", flush=True)
    check_public_source_import_surface_guard()
    for guard in PUBLIC_SOURCE_IMPORT_GUARDS:
        lines = guard.path.read_text(encoding="utf-8").splitlines()
        disallowed = disallowed_mainarch_core_source_references(
            guard.path,
            lines,
            guard.allowed_mainarch_core_lines,
        )
        if disallowed:
            raise RuntimeError(
                "disallowed mainarch_core import surface:\n" + "\n".join(disallowed)
            )
        allowed = ", ".join(guard.allowed_mainarch_core_lines)
        print(f"  ok: {guard.path.relative_to(ROOT)} imports only {allowed}", flush=True)


def missing_public_source_patterns(
    text: str, patterns: tuple[re.Pattern[str], ...]
) -> list[str]:
    return [
        source_pattern.pattern
        for source_pattern in patterns
        if not source_pattern.search(text)
    ]


def require_public_source_pattern_guard(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"public source pattern guard self-test failed: {detail}")


def check_public_source_pattern_guard() -> None:
    receiver_call = pattern(r"\bguard\s*\.\s*helper\(\)")
    require_public_source_pattern_guard(
        missing_public_source_patterns("let _ = guard.helper();", (receiver_call,)) == [],
        "direct receiver call was rejected",
    )
    require_public_source_pattern_guard(
        missing_public_source_patterns("let _ = guard\n    .helper();", (receiver_call,))
        == [],
        "line-broken receiver call was rejected",
    )
    missing = missing_public_source_patterns("let _ = guard.other();", (receiver_call,))
    require_public_source_pattern_guard(
        missing == [receiver_call.pattern],
        f"unrelated receiver call produced {missing}",
    )
    for guard in PUBLIC_SOURCE_PATTERN_GUARDS:
        for source_pattern in guard.patterns:
            require_public_source_pattern_guard(
                bool(source_pattern.pattern),
                f"empty {guard.label} pattern",
            )


def check_public_source_patterns() -> None:
    print("model API public source patterns", flush=True)
    check_public_source_pattern_guard()
    for guard in PUBLIC_SOURCE_PATTERN_GUARDS:
        text = guard.path.read_text(encoding="utf-8")
        missing = missing_public_source_patterns(text, guard.patterns)
        if missing:
            missing_lines = "\n".join(
                f"  - {source_pattern}" for source_pattern in missing
            )
            raise RuntimeError(
                f"{guard.path.relative_to(ROOT)} missing {guard.label}:\n"
                f"{missing_lines}"
            )
        print(
            f"  ok: {guard.path.relative_to(ROOT)} carries "
            f"{len(guard.patterns)} {guard.label}",
            flush=True,
        )


def line_documents_public_command(line: str, command: str) -> bool:
    stripped = line.strip()
    command_line_re = re.compile(rf"^{SHELL_ENV_PREFIX_RE}{re.escape(command)}$")
    return command_line_re.fullmatch(stripped) is not None


def require_public_command_doc_guard(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"public command doc guard self-test failed: {detail}")


def check_public_command_doc_guard() -> None:
    require_public_command_doc_guard(
        public_gate_doc_command(
            ExampleGate(
                name="doc_guard",
                command=("cargo", "run", "-q", "-p", "mainarch-core"),
                required_patterns=(),
            )
        )
        == "cargo run -p mainarch-core",
        "quiet cargo flag was not removed from rendered doc command",
    )
    require_public_command_doc_guard(
        line_documents_public_command(
            "CARGO_INCREMENTAL=0 cargo run -p mainarch-core",
            "cargo run -p mainarch-core",
        ),
        "environment-prefixed command line was not accepted",
    )
    require_public_command_doc_guard(
        not line_documents_public_command(
            "cargo run -p mainarch-core -- --static-handoff-receipt",
            "cargo run -p mainarch-core",
        ),
        "longer command line incorrectly satisfied shorter command",
    )
    require_public_command_doc_guard(
        not line_documents_public_command(
            "To run it, use cargo run -p mainarch-core",
            "cargo run -p mainarch-core",
        ),
        "arbitrary prose prefix incorrectly satisfied command",
    )
    plugin_commands = standalone_plugin_doc_requirements()
    require_public_command_doc_guard(
        len(plugin_commands) == 13,
        f"standalone plugin docs expected 13 commands, found {plugin_commands}",
    )
    require_public_command_doc_guard(
        all(
            "examples/model-api-plugin/Cargo.toml" in command
            for command in plugin_commands
        ),
        f"standalone plugin docs included a non-plugin command: {plugin_commands}",
    )
    require_public_command_doc_guard(
        PUBLIC_EXAMPLE_CHECK_COMMAND not in plugin_commands,
        "standalone plugin docs unexpectedly require the repository-wide checker command",
    )


def check_public_command_docs() -> None:
    print("model API public command docs", flush=True)
    check_public_command_doc_guard()
    for guard in PUBLIC_COMMAND_DOC_GUARDS:
        required_commands = guard.commands
        lines = guard.path.read_text(encoding="utf-8").splitlines()
        missing = [
            command
            for command in required_commands
            if not any(line_documents_public_command(line, command) for line in lines)
        ]
        if missing:
            missing_lines = "\n".join(f"  - {command}" for command in missing)
            raise RuntimeError(
                f"{guard.path.relative_to(ROOT)} missing public command docs:\n"
                f"{missing_lines}"
            )
        print(
            f"  ok: {guard.path.relative_to(ROOT)} documents "
            f"{len(required_commands)} {guard.label}",
            flush=True,
        )


def missing_public_doc_snippets(text: str, snippets: tuple[str, ...]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def require_public_doc_snippet_guard(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"public doc snippet guard self-test failed: {detail}")


def drift_public_doc_snippet(snippet: str) -> str:
    require_public_doc_snippet_guard(
        bool(snippet),
        "empty snippets cannot be drift-tested",
    )
    if len(snippet) == 1:
        return "x" if snippet != "x" else "y"
    split_at = max(1, len(snippet) // 2)
    return f"{snippet[:split_at]}<drift>{snippet[split_at:]}"


def check_public_doc_snippet_guard() -> None:
    for guard in PUBLIC_DOC_SNIPPET_GUARDS:
        for snippet in guard.snippets:
            require_public_doc_snippet_guard(
                missing_public_doc_snippets(f"before\n{snippet}\nafter", (snippet,))
                == [],
                f"exact {guard.label} snippet was rejected",
            )
            missing = missing_public_doc_snippets(
                drift_public_doc_snippet(snippet),
                (snippet,),
            )
            require_public_doc_snippet_guard(
                missing == [snippet],
                f"drifted {guard.label} snippet was accepted",
            )


def check_public_doc_snippets() -> None:
    print("model API public snippet docs", flush=True)
    check_public_doc_snippet_guard()
    for guard in PUBLIC_DOC_SNIPPET_GUARDS:
        text = guard.path.read_text(encoding="utf-8")
        missing = missing_public_doc_snippets(text, guard.snippets)
        if missing:
            missing_lines = "\n".join(f"  - {snippet}" for snippet in missing)
            raise RuntimeError(
                f"{guard.path.relative_to(ROOT)} missing {guard.label}:\n"
                f"{missing_lines}"
            )
        print(
            f"  ok: {guard.path.relative_to(ROOT)} documents "
            f"{len(guard.snippets)} {guard.label}",
            flush=True,
        )


def expected_fixture_mismatch_detail(
    expected_text: str,
    actual_text: str,
    expected_path: Path,
    gate_name: str,
) -> str:
    if actual_text == expected_text:
        return ""

    expected_lines = expected_text.splitlines()
    actual_lines = actual_text.splitlines()
    diff = "\n".join(
        unified_diff(
            expected_lines,
            actual_lines,
            fromfile=str(expected_path),
            tofile=f"{gate_name} output",
            lineterm="",
        )
    )
    details = []
    if expected_lines == actual_lines:
        details.append("fixture text differs despite identical splitlines")
    if expected_text.endswith("\n") != actual_text.endswith("\n"):
        details.append(
            "final newline mismatch: "
            f"expected={expected_text.endswith(chr(10))} "
            f"actual={actual_text.endswith(chr(10))}"
        )
    details.append(
        f"text lengths: expected={len(expected_text)} actual={len(actual_text)}"
    )
    if diff and details:
        return diff + "\n" + "\n".join(details)
    return diff or "\n".join(details)


def require_expected_fixture_guard(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError(f"expected fixture guard self-test failed: {detail}")


def check_expected_fixture_guard() -> None:
    exact = "receipt.kind=example\nready=true\n"
    missing_final_newline = "receipt.kind=example\nready=true"
    require_expected_fixture_guard(
        expected_fixture_mismatch_detail(
            exact,
            exact,
            Path("expected.receipt"),
            "fixture_guard",
        )
        == "",
        "exact fixture text was reported as mismatched",
    )
    final_newline_detail = expected_fixture_mismatch_detail(
        exact,
        missing_final_newline,
        Path("expected.receipt"),
        "fixture_guard",
    )
    require_expected_fixture_guard(
        "fixture text differs despite identical splitlines" in final_newline_detail,
        "missing final newline did not report splitline-equivalent drift: "
        f"{final_newline_detail}",
    )
    require_expected_fixture_guard(
        "final newline mismatch: expected=True actual=False" in final_newline_detail,
        "missing final newline did not report final-newline drift: "
        f"{final_newline_detail}",
    )


def run_gate(gate: ExampleGate) -> tuple[str, str]:
    env = os.environ.copy()
    env.setdefault("CARGO_INCREMENTAL", "0")
    env.setdefault("CARGO_TARGET_DIR", str(ROOT / "target"))
    print(f"model API public example gate: {gate.name}", flush=True)
    completed = subprocess.run(
        gate.command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        print(output, file=sys.stderr, end="")
        raise RuntimeError(f"{gate.name} exited with {completed.returncode}")

    for required in gate.required_patterns:
        match = required.search(output)
        if match is None:
            print(output, file=sys.stderr, end="")
            raise RuntimeError(f"{gate.name} missing expected output: {required.pattern}")
        print(f"  ok: {match.group(0)}", flush=True)

    if gate.expected_lines_file is not None:
        expected_text = gate.expected_lines_file.read_text(encoding="utf-8")
        fixture_mismatch = expected_fixture_mismatch_detail(
            expected_text,
            output,
            gate.expected_lines_file,
            gate.name,
        )
        if fixture_mismatch:
            print(fixture_mismatch, file=sys.stderr)
            raise RuntimeError(
                f"{gate.name} output differs from {gate.expected_lines_file}"
            )
        print(f"  ok: exact fixture {gate.expected_lines_file}", flush=True)

    contract_match = CONTRACT_RE.search(output)
    return gate.name, contract_match.group(1) if contract_match else ""


def main() -> int:
    contract_fingerprints: dict[str, str] = {}
    try:
        check_expected_fixture_guard()
        check_public_command_docs()
        check_public_doc_snippets()
        check_public_source_import_surface()
        check_public_source_patterns()
        for gate in GATES:
            name, contract_fingerprint = run_gate(gate)
            if contract_fingerprint:
                contract_fingerprints[name] = contract_fingerprint
    except RuntimeError as err:
        print(f"model API public example gate failed: {err}", file=sys.stderr)
        return 1

    unique_contract_fingerprints = set(contract_fingerprints.values())
    if len(unique_contract_fingerprints) != 1:
        print(
            "model API public example gate failed: contract fingerprints disagree: "
            f"{contract_fingerprints}",
            file=sys.stderr,
        )
        return 1

    fingerprint = next(iter(unique_contract_fingerprints))
    print(
        "model API public example gate ok: "
        f"{len(GATES)} commands, contract_fingerprint={fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

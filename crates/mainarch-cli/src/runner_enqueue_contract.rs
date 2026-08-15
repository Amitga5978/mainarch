use anyhow::Result;

use crate::{
    qwen_resident_layer_runner_dispatch_kind_code,
    qwen_resident_layer_runner_dispatch_kind_from_code,
    qwen_resident_layer_runner_dispatch_kind_label, QwenResidentLayerRunnerDispatchKind,
};

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerMaterializedDispatchCursorHandoff {
    pub(crate) cursor_index: usize,
    pub(crate) ready_ordinal: usize,
    pub(crate) layer_idx: u32,
    pub(crate) kind_code: u64,
    pub(crate) kind: QwenResidentLayerRunnerDispatchKind,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerEnqueueReadyDispatchContract {
    pub(crate) frontier_edge_index: usize,
    pub(crate) cursor_index: usize,
    pub(crate) ready_ordinal: usize,
    pub(crate) layer_idx: u32,
    pub(crate) kind_code: u64,
    pub(crate) kind: QwenResidentLayerRunnerDispatchKind,
    pub(crate) wait_mask: u64,
    pub(crate) producer_signal_mask: u64,
    pub(crate) ready_mask: u64,
    pub(crate) transition_ready_count: u64,
    pub(crate) row_checksum: u64,
    pub(crate) frontier_checksum: u64,
    pub(crate) candidate_checksum: u64,
    pub(crate) launch_function: &'static str,
    pub(crate) capture_boundary: &'static str,
    pub(crate) dependency_packet_kind: &'static str,
    pub(crate) completion_signal_initial_value: u64,
    pub(crate) completion_signal_success_value: u64,
    pub(crate) aql_submitted: bool,
    pub(crate) persistent_batch_committed: bool,
    pub(crate) cleanup_preflight_gate_required: bool,
    pub(crate) cleanup_preflight_copy_decision_validated: bool,
    pub(crate) cleanup_preflight_validated: bool,
    pub(crate) enqueue_ready_gate_validated: bool,
}

impl QwenResidentLayerRunnerEnqueueReadyDispatchContract {
    pub(crate) fn require_validated_cleanup_preflight(
        mut self,
        cleanup_preflight_copy_decision_validated: bool,
        cleanup_preflight_validated: bool,
    ) -> Result<Self> {
        let enqueue_ready_gate_validated = cleanup_preflight_copy_decision_validated
            && cleanup_preflight_validated
            && !self.aql_submitted
            && !self.persistent_batch_committed;
        if !enqueue_ready_gate_validated {
            anyhow::bail!(
                "resident layer runner enqueue-ready contract requires validated cleanup preflight before live AQL ownership transfer: copy_decision_validated={} cleanup_preflight_validated={} submitted={} committed={}",
                cleanup_preflight_copy_decision_validated,
                cleanup_preflight_validated,
                self.aql_submitted,
                self.persistent_batch_committed
            );
        }
        self.cleanup_preflight_gate_required = true;
        self.cleanup_preflight_copy_decision_validated = cleanup_preflight_copy_decision_validated;
        self.cleanup_preflight_validated = cleanup_preflight_validated;
        self.enqueue_ready_gate_validated = enqueue_ready_gate_validated;
        Ok(self)
    }
}

pub(crate) fn qwen_resident_layer_runner_materialized_dispatch_cursor_handoff_from_probe(
    candidate_probe: &[u64],
    expected_layer_idx: u32,
    expected_kind: QwenResidentLayerRunnerDispatchKind,
) -> Result<Option<QwenResidentLayerRunnerMaterializedDispatchCursorHandoff>> {
    const SELECTED_EDGE_INDEX_WORD: usize = 2;
    const SELECTED_LAYER_KIND_WORD: usize = 3;
    const MATERIALIZED_STATUS_WORD: usize = 14;

    if candidate_probe.len() <= MATERIALIZED_STATUS_WORD {
        anyhow::bail!(
            "resident layer runner materialized cursor handoff probe has {} words, need at least {}",
            candidate_probe.len(),
            MATERIALIZED_STATUS_WORD + 1
        );
    }
    if candidate_probe[MATERIALIZED_STATUS_WORD] == 0 {
        return Ok(None);
    }

    let cursor_index = usize::try_from(candidate_probe[SELECTED_EDGE_INDEX_WORD])?;
    let layer_kind = candidate_probe[SELECTED_LAYER_KIND_WORD];
    let layer_idx = u32::try_from(layer_kind & 0xffff_ffff)?;
    let kind_code = layer_kind >> 32;
    let kind = qwen_resident_layer_runner_dispatch_kind_from_code(kind_code)?;
    if layer_idx != expected_layer_idx {
        anyhow::bail!(
            "resident layer runner materialized cursor handoff layer {} does not match expected layer {}",
            layer_idx,
            expected_layer_idx
        );
    }
    if kind_code != qwen_resident_layer_runner_dispatch_kind_code(expected_kind) {
        anyhow::bail!(
            "resident layer runner materialized cursor handoff kind code {} does not match expected kind code {}",
            kind_code,
            qwen_resident_layer_runner_dispatch_kind_code(expected_kind)
        );
    }

    Ok(Some(
        QwenResidentLayerRunnerMaterializedDispatchCursorHandoff {
            cursor_index,
            ready_ordinal: cursor_index,
            layer_idx,
            kind_code,
            kind,
        },
    ))
}

pub(crate) fn qwen_resident_layer_runner_materialized_dispatch_cursor_handoff_trace(
    handoff: QwenResidentLayerRunnerMaterializedDispatchCursorHandoff,
) -> String {
    format!(
        "{}:{}:{}:ready={}",
        handoff.cursor_index,
        handoff.layer_idx,
        qwen_resident_layer_runner_dispatch_kind_label(handoff.kind),
        handoff.ready_ordinal
    )
}

pub(crate) fn qwen_resident_layer_runner_dispatch_launch_function_name(
    kind: QwenResidentLayerRunnerDispatchKind,
) -> &'static str {
    match kind {
        QwenResidentLayerRunnerDispatchKind::DecodeAttention => {
            "qwen_resident_layer_runner_launch_decode_attention"
        }
        QwenResidentLayerRunnerDispatchKind::Mlp => "qwen_resident_layer_runner_launch_mlp",
    }
}

pub(crate) fn qwen_resident_layer_runner_enqueue_dependency_packet_kind(
    wait_mask: u64,
) -> &'static str {
    if wait_mask == 0 {
        "none"
    } else {
        "hsa_amd_barrier_value_wait_mask"
    }
}

pub(crate) fn qwen_resident_layer_runner_enqueue_ready_dispatch_contract_from_probe(
    candidate_probe: &[u64],
    handoff: QwenResidentLayerRunnerMaterializedDispatchCursorHandoff,
) -> Result<QwenResidentLayerRunnerEnqueueReadyDispatchContract> {
    const SELECTED_EDGE_INDEX_WORD: usize = 2;
    const SELECTED_LAYER_KIND_WORD: usize = 3;
    const SELECTED_WAIT_MASK_WORD: usize = 4;
    const SELECTED_SIGNAL_MASK_WORD: usize = 5;
    const TRANSITION_READY_MASK_WORD: usize = 7;
    const TRANSITION_READY_COUNT_WORD: usize = 9;
    const ROW_CHECKSUM_WORD: usize = 10;
    const FRONTIER_CHECKSUM_WORD: usize = 12;
    const CANDIDATE_CHECKSUM_WORD: usize = 13;
    const MATERIALIZED_STATUS_WORD: usize = 14;

    if candidate_probe.len() <= MATERIALIZED_STATUS_WORD {
        anyhow::bail!(
            "resident layer runner enqueue-ready dispatch contract probe has {} words, need at least {}",
            candidate_probe.len(),
            MATERIALIZED_STATUS_WORD + 1
        );
    }
    if candidate_probe[MATERIALIZED_STATUS_WORD] == 0 {
        anyhow::bail!(
            "resident layer runner enqueue-ready dispatch contract requires materialized candidate"
        );
    }
    let frontier_edge_index = usize::try_from(candidate_probe[SELECTED_EDGE_INDEX_WORD])?;
    if frontier_edge_index != handoff.cursor_index {
        anyhow::bail!(
            "resident layer runner enqueue-ready dispatch contract edge {} does not match handoff cursor {}",
            frontier_edge_index,
            handoff.cursor_index
        );
    }
    let layer_kind = candidate_probe[SELECTED_LAYER_KIND_WORD];
    let layer_idx = u32::try_from(layer_kind & 0xffff_ffff)?;
    let kind_code = layer_kind >> 32;
    if layer_idx != handoff.layer_idx {
        anyhow::bail!(
            "resident layer runner enqueue-ready dispatch contract layer {} does not match handoff layer {}",
            layer_idx,
            handoff.layer_idx
        );
    }
    if kind_code != handoff.kind_code {
        anyhow::bail!(
            "resident layer runner enqueue-ready dispatch contract kind code {} does not match handoff kind code {}",
            kind_code,
            handoff.kind_code
        );
    }
    let kind = qwen_resident_layer_runner_dispatch_kind_from_code(kind_code)?;
    if qwen_resident_layer_runner_dispatch_kind_code(kind) != handoff.kind_code {
        anyhow::bail!(
            "resident layer runner enqueue-ready dispatch contract decoded kind does not round-trip for code {}",
            handoff.kind_code
        );
    }

    Ok(QwenResidentLayerRunnerEnqueueReadyDispatchContract {
        frontier_edge_index,
        cursor_index: handoff.cursor_index,
        ready_ordinal: handoff.ready_ordinal,
        layer_idx,
        kind_code,
        kind,
        wait_mask: candidate_probe[SELECTED_WAIT_MASK_WORD],
        producer_signal_mask: candidate_probe[SELECTED_SIGNAL_MASK_WORD],
        ready_mask: candidate_probe[TRANSITION_READY_MASK_WORD],
        transition_ready_count: candidate_probe[TRANSITION_READY_COUNT_WORD],
        row_checksum: candidate_probe[ROW_CHECKSUM_WORD],
        frontier_checksum: candidate_probe[FRONTIER_CHECKSUM_WORD],
        candidate_checksum: candidate_probe[CANDIDATE_CHECKSUM_WORD],
        launch_function: qwen_resident_layer_runner_dispatch_launch_function_name(kind),
        capture_boundary: "piecewise_comm_break",
        dependency_packet_kind: qwen_resident_layer_runner_enqueue_dependency_packet_kind(
            candidate_probe[SELECTED_WAIT_MASK_WORD],
        ),
        completion_signal_initial_value: 1,
        completion_signal_success_value: 0,
        aql_submitted: false,
        persistent_batch_committed: false,
        cleanup_preflight_gate_required: false,
        cleanup_preflight_copy_decision_validated: false,
        cleanup_preflight_validated: false,
        enqueue_ready_gate_validated: false,
    })
}

pub(crate) fn qwen_resident_layer_runner_enqueue_ready_dispatch_contract_trace(
    contract: QwenResidentLayerRunnerEnqueueReadyDispatchContract,
) -> String {
    format!(
        "edge={}:cursor={}:layer={}:kind={}:ready={}:wait=0x{:016x}:signal=0x{:016x}:launch={}",
        contract.frontier_edge_index,
        contract.cursor_index,
        contract.layer_idx,
        qwen_resident_layer_runner_dispatch_kind_label(contract.kind),
        contract.ready_ordinal,
        contract.wait_mask,
        contract.producer_signal_mask,
        contract.launch_function
    )
}

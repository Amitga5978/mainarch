use super::{
    mcore, qwen_prepare_next_decode_layer_rank_h_buffers,
    qwen_resident_layer_runner_descriptor_checksum, qwen_resident_layer_runner_dispatch_kind_label,
    qwen_resident_layer_runner_dispatch_next_decode_attention,
    qwen_resident_layer_runner_dispatch_next_decode_mlp,
    qwen_resident_layer_runner_dispatch_plan_enabled,
    qwen_resident_layer_runner_mlp_dispatch_plan_for_layer,
    qwen_resident_layer_runner_mlp_launch_function,
    qwen_resident_layer_runner_validate_dispatch_buffer, QwenFp4KvCacheStagePlan,
    QwenFp4SingleRowAttentionStagePlan, QwenMlpStagePlan, QwenNextDecodeLayer0MlpAllReduceStage,
    QwenNextDecodeLayer0MlpResidualNormAllRankStage, QwenNextDecodeLayer1AttentionAllRankStage,
    QwenNextDecodeLayer1KvAppendAllRankStage, QwenNextDecodeLayer1OProjAllReduceStage,
    QwenNextDecodeLayer1PostAttnNormAllRankStage, QwenNextDecodeLayer1QkRopeAllRankStage,
    QwenNextDecodeLayer1QkvAllRankStage, QwenNextInputNormStagePlan, QwenOProjStagePlan,
    QwenPeerMlpWorkspace, QwenPostAttnNormStagePlan, QwenQkNormRopeStagePlan, QwenQkvProjStagePlan,
    QwenResidentLayerRunnerDagSchedulerContext, QwenResidentLayerRunnerDependencyId,
    QwenResidentLayerRunnerDispatchKind, QwenResidentLayerRunnerEntry,
    QwenResidentLayerRunnerLayerTopology, QwenResidentLayerRunnerPlanDescriptor,
    QwenResidentLayerRunnerRuntimeLayerObservation, QwenTpMlpPeerStagePlan,
    QwenTpOProjPeerStagePlan, QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_COUNT,
    QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_FLAG_HOST_LEGACY_BRIDGE,
    QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_LAYER_INDEX_SENTINEL,
    QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_MAGIC,
    QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_ROLE_MASK_ALL,
    QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_U64S,
    QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_VERSION,
};

#[derive(Clone, Copy)]
pub(crate) enum QwenResidentLayerRunnerDenseLayerHandoffKind {
    NextInputNorm,
}

impl QwenResidentLayerRunnerDenseLayerHandoffKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::NextInputNorm => "next_input_norm",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerHandoffContract {
    pub(crate) kind: QwenResidentLayerRunnerDenseLayerHandoffKind,
    pub(crate) source_layer_idx: u32,
    pub(crate) target_layer_idx: u32,
}

impl QwenResidentLayerRunnerDenseLayerHandoffContract {
    pub(crate) fn next_input_norm(source_layer_idx: u32, target_layer_idx: u32) -> Self {
        Self {
            kind: QwenResidentLayerRunnerDenseLayerHandoffKind::NextInputNorm,
            source_layer_idx,
            target_layer_idx,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        self.kind.label()
    }

    pub(crate) fn dispatch_payload_satisfies(self, dispatch_next_input_norm_present: bool) -> bool {
        match self.kind {
            QwenResidentLayerRunnerDenseLayerHandoffKind::NextInputNorm => {
                dispatch_next_input_norm_present
            }
        }
    }
}

fn qwen_resident_layer_runner_env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim();
            !(value.is_empty()
                || value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off"))
        }
        Err(_) => default,
    }
}

pub(crate) fn qwen_resident_layer_runner_dense_mlp_trial_enabled_from_env(
    layer_idx: u32,
    handoff_layer_idx: u32,
    handoff_output_owner: &str,
) -> bool {
    let env = format!("MAINARCH_QWEN_LAYER{layer_idx}_DENSE_MLP_TRIAL");
    let enabled = qwen_resident_layer_runner_env_flag(env.as_str(), false);
    println!("  resident_layer_runner_dense_mlp_trial_gate_stage:");
    println!("    source: resident_runner_dense_mlp_trial_gate");
    println!("    layer_idx: {layer_idx}");
    println!("    handoff_layer_idx: {handoff_layer_idx}");
    println!("    enabled: {enabled}");
    println!("    env: {env}");
    println!("    default_enabled: false");
    println!(
        "    default_path: layer{layer_idx}_attention_cursor_only_then_legacy_layer{layer_idx}_mlp"
    );
    println!("    trial_path: dense_cursor_attention_plus_generic_mlp");
    println!("    trial_handoff_output_owner: {handoff_output_owner}");
    println!("    local_trial_handoff_output_vector_removed: true");
    println!("    main_dense_mlp_trial_gate_callsite_removed: true");
    println!("    production_claim: false");
    println!("    graph_capture_started: false");
    enabled
}

pub(crate) fn qwen_resident_layer_runner_dense_mlp_layer_handoff(
    layer_idx: u32,
    handoff_layer_idx: u32,
    handoff_output_owner: &str,
    handoff_input_norm_plan_present: bool,
    output_ranks: usize,
    expected_ranks: usize,
) -> anyhow::Result<()> {
    if handoff_input_norm_plan_present && output_ranks != expected_ranks {
        anyhow::bail!(
            "next decode layer{} input norm output ranks {} do not match TP world {}",
            handoff_layer_idx,
            output_ranks,
            expected_ranks
        );
    }
    println!("  resident_layer_runner_dense_mlp_layer_handoff_stage:");
    println!("    source: resident_runner_dense_mlp_trial_layer_handoff");
    println!("    enabled: true");
    println!("    layer_idx: {layer_idx}");
    println!("    handoff_layer_idx: {handoff_layer_idx}");
    println!("    output_owner: {handoff_output_owner}");
    println!("    local_trial_vector_removed: true");
    println!("    output_ranks: {output_ranks}");
    println!("    expected_ranks: {expected_ranks}");
    println!("    ranks_match: {}", output_ranks == expected_ranks);
    println!("    handoff_input_norm_plan_present: {handoff_input_norm_plan_present}");
    println!("    stage_applied_to_legacy_layer_bundle: true");
    println!("    main_dense_mlp_handoff_validation_callsite_removed: true");
    println!("    default_path_changed: false");
    println!("    production_claim: false");
    println!("    graph_capture_started: false");
    Ok(())
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_observation_window_from_satisfied_dependency_ids(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    satisfied_dependency_ids: &[QwenResidentLayerRunnerDependencyId],
) -> anyhow::Result<Vec<QwenResidentLayerRunnerRuntimeLayerObservation>> {
    for left in 0..satisfied_dependency_ids.len() {
        for right in (left + 1)..satisfied_dependency_ids.len() {
            if satisfied_dependency_ids[left] == satisfied_dependency_ids[right] {
                anyhow::bail!(
                    "resident runtime satisfied dependency id table contains duplicate dependency {}",
                    satisfied_dependency_ids[left].label()
                );
            }
        }
    }

    let planned_dependencies =
        qwen_resident_layer_runner_planned_runtime_dependency_ids(runner_plans)?;

    for dependency_id in satisfied_dependency_ids {
        if !planned_dependencies
            .iter()
            .any(|planned_dependency_id| *planned_dependency_id == *dependency_id)
        {
            anyhow::bail!(
                "resident runtime satisfied dependency id {} has no planned dispatch dependency",
                dependency_id.label()
            );
        }
    }

    let mut observations = Vec::new();
    for plan in runner_plans {
        if plan.attention_dispatch.is_none() && plan.mlp_dispatch.is_none() {
            continue;
        }
        let attention_satisfied = match plan.attention_dispatch {
            Some(dispatch) => satisfied_dependency_ids
                .iter()
                .any(|dependency_id| *dependency_id == dispatch.dependency_id),
            None => false,
        };
        let mlp_satisfied = match plan.mlp_dispatch {
            Some(dispatch) => satisfied_dependency_ids
                .iter()
                .any(|dependency_id| *dependency_id == dispatch.dependency_id),
            None => false,
        };
        observations.push(QwenResidentLayerRunnerRuntimeLayerObservation {
            layer_idx: plan.layer_idx,
            attention_satisfied,
            mlp_satisfied,
        });
    }

    if observations.is_empty() {
        anyhow::bail!("resident runtime dependency presence window produced no observations");
    }

    let mut previous_layer_idx = None;
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    for observation in &observations {
        if let Some(previous_layer_idx) = previous_layer_idx {
            if observation.layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident runtime dependency presence observations must be strictly ascending"
                );
            }
            if previous_layer_idx.checked_add(1) != Some(observation.layer_idx) {
                anyhow::bail!(
                    "resident runtime dependency presence observations must be contiguous"
                );
            }
        }
        previous_layer_idx = Some(observation.layer_idx);
        first_layer_idx = first_layer_idx.min(observation.layer_idx);
        last_layer_idx = last_layer_idx.max(observation.layer_idx);
    }

    println!("  resident_layer_runner_runtime_layer_observation_window_stage:");
    println!("    source: resident_runner_runtime_layer_observation_window_from_satisfied_dependency_ids");
    println!("    observation_window_owner: resident_runner_module");
    println!("    satisfied_dependency_id_table_consumed: true");
    println!(
        "    planned_dependency_rows: {}",
        planned_dependencies.len()
    );
    println!(
        "    satisfied_dependency_id_rows: {}",
        satisfied_dependency_ids.len()
    );
    println!("    derived_observation_rows: {}", observations.len());
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: true");
    println!("    observation_rows_strictly_ascending: true");
    println!("    satisfied_dependency_ids_duplicate_free: true");
    println!("    plan_descriptor_dependency_ids_bound: true");
    println!("    main_runtime_dependency_presence_row_literals_removed: true");
    println!("    observation_derivation_owner: resident_runner_module");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(observations)
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerRuntimeDispatchSatisfactionRow {
    layer_idx: u32,
    kind: QwenResidentLayerRunnerDispatchKind,
    satisfied: bool,
}

impl QwenResidentLayerRunnerRuntimeDispatchSatisfactionRow {
    pub(crate) fn new(
        layer_idx: u32,
        kind: QwenResidentLayerRunnerDispatchKind,
        satisfied: bool,
    ) -> Self {
        Self {
            layer_idx,
            kind,
            satisfied,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerRuntimeLayerStateRow {
    layer_idx: u32,
    attention_stage_ready: bool,
    mlp_stage_ready: bool,
}

impl QwenResidentLayerRunnerRuntimeLayerStateRow {
    pub(crate) fn new(layer_idx: u32, attention_stage_ready: bool, mlp_stage_ready: bool) -> Self {
        Self {
            layer_idx,
            attention_stage_ready,
            mlp_stage_ready,
        }
    }
}

pub(crate) struct QwenResidentLayerRunnerRuntimeLayerStateWindow {
    rows: Vec<QwenResidentLayerRunnerRuntimeLayerStateRow>,
    first_layer_idx: u32,
    last_layer_idx: u32,
}

pub(crate) struct QwenResidentLayerRunnerRuntimeLayerStateTracker {
    rows: Vec<QwenResidentLayerRunnerRuntimeLayerStateRow>,
    first_layer_idx: u32,
    last_layer_idx: u32,
}

fn qwen_resident_layer_runner_runtime_layer_state_row_window_bounds(
    rows: &[QwenResidentLayerRunnerRuntimeLayerStateRow],
) -> anyhow::Result<(u32, u32, bool)> {
    if rows.is_empty() {
        anyhow::bail!("resident runtime layer-state window cannot be empty");
    }

    let mut first_layer_idx = rows[0].layer_idx;
    let mut last_layer_idx = rows[0].layer_idx;
    let mut previous_layer_idx = None;
    for (row_index, row) in rows.iter().enumerate() {
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
        for prior_row in rows[..row_index].iter() {
            if prior_row.layer_idx == row.layer_idx {
                anyhow::bail!(
                    "resident runtime layer-state window contains duplicate layer {}",
                    row.layer_idx
                );
            }
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if row.layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident runtime layer-state window is not strictly ascending: layer {} after layer {} at row {}",
                    row.layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(row.layer_idx);
    }

    let contiguous_layer_window =
        last_layer_idx - first_layer_idx + 1 == u32::try_from(rows.len()).unwrap_or(u32::MAX);

    Ok((first_layer_idx, last_layer_idx, contiguous_layer_window))
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_state_tracker_from_layer_indices(
    layer_indices: Vec<u32>,
) -> anyhow::Result<QwenResidentLayerRunnerRuntimeLayerStateTracker> {
    let tracker_rows = layer_indices.len();
    if tracker_rows == 0 {
        anyhow::bail!("resident runtime layer-state tracker cannot be empty");
    }

    let rows = layer_indices
        .into_iter()
        .map(|layer_idx| QwenResidentLayerRunnerRuntimeLayerStateRow::new(layer_idx, false, false))
        .collect::<Vec<_>>();
    let (first_layer_idx, last_layer_idx, contiguous_layer_window) =
        qwen_resident_layer_runner_runtime_layer_state_row_window_bounds(rows.as_slice())?;

    println!("  resident_layer_runner_runtime_layer_state_tracker_init_stage:");
    println!("    source: resident_runner_runtime_layer_state_tracker_from_layer_indices");
    println!("    tracker_owner: resident_runner_module");
    println!("    tracker_rows: {tracker_rows}");
    println!("    tracker_first_layer_idx: {first_layer_idx}");
    println!("    tracker_last_layer_idx: {last_layer_idx}");
    println!("    tracker_contiguous_layer_window: {contiguous_layer_window}");
    println!("    tracker_initialized_from_layer_indices: true");
    println!("    tracker_initial_state_all_false: true");
    println!("    tracker_rows_derived_from_layer_indices_len: true");
    println!("    fixed_tracker_window_row_count_removed: true");
    println!("    final_readiness_callsite_numbered_stage_checks_removed: true");
    println!("    legacy_stage_assignment_sources_retained: true");
    println!("    readiness_update_rows_supported: true");
    println!("    dense_readiness_batch_updates_supported: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerRuntimeLayerStateTracker {
        rows,
        first_layer_idx,
        last_layer_idx,
    })
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_state_tracker_from_layer_indices_and_readiness_update_rows(
    layer_indices: Vec<u32>,
    readiness_update_rows: Vec<QwenResidentLayerRunnerRuntimeReadinessUpdateRow>,
) -> anyhow::Result<QwenResidentLayerRunnerRuntimeLayerStateTracker> {
    let mut tracker =
        qwen_resident_layer_runner_runtime_layer_state_tracker_from_layer_indices(layer_indices)?;
    let tracker_rows = tracker.rows.len();
    let readiness_update_rows_len = readiness_update_rows.len();
    qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_readiness_update_rows(
        &mut tracker,
        readiness_update_rows,
    )?;

    println!("  resident_layer_runner_runtime_layer_state_tracker_seeded_init_stage:");
    println!(
        "    source: resident_runner_runtime_layer_state_tracker_from_layer_indices_and_readiness_update_rows"
    );
    println!("    tracker_owner: resident_runner_module");
    println!("    tracker_rows: {tracker_rows}");
    println!("    readiness_update_rows: {readiness_update_rows_len}");
    println!("    readiness_update_rows_dynamic_vec: true");
    println!("    fixed_readiness_update_row_count_removed: true");
    println!("    tracker_initialized_from_layer_indices: true");
    println!("    initial_readiness_update_rows_consumed: true");
    println!("    main_separate_seed_update_call_removed: true");
    println!("    legacy_stage_assignment_sources_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(tracker)
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_state_tracker_from_plan_descriptors_and_seeded_readiness_selection(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    readiness_update_selection: QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection,
) -> anyhow::Result<QwenResidentLayerRunnerRuntimeLayerStateTracker> {
    let layer_indices =
        qwen_resident_layer_runner_runtime_layer_state_tracker_layer_indices_from_plan_descriptors(
            runner_plans,
        )?;
    let readiness_update_row =
        qwen_resident_layer_runner_runtime_readiness_update_row_from_plan_descriptor_selection(
            runner_plans,
            readiness_update_selection,
        )?;

    qwen_resident_layer_runner_runtime_layer_state_tracker_from_layer_indices_and_readiness_update_rows(
        layer_indices,
        vec![readiness_update_row],
    )
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_state_tracker_layer_indices_from_plan_descriptors(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
) -> anyhow::Result<Vec<u32>> {
    let mut layer_indices = Vec::new();
    for plan in runner_plans {
        if plan.attention_dispatch.is_none() && plan.mlp_dispatch.is_none() {
            continue;
        }
        layer_indices.push(plan.layer_idx);
    }
    let derived_rows = layer_indices.len();
    if derived_rows == 0 {
        anyhow::bail!("resident runtime layer-state tracker layer-index window cannot be empty");
    }
    let state_rows = layer_indices
        .iter()
        .copied()
        .map(|layer_idx| QwenResidentLayerRunnerRuntimeLayerStateRow::new(layer_idx, false, false))
        .collect::<Vec<_>>();
    let (first_layer_idx, last_layer_idx, contiguous_layer_window) =
        qwen_resident_layer_runner_runtime_layer_state_row_window_bounds(state_rows.as_slice())?;

    println!("  resident_layer_runner_runtime_layer_state_tracker_layer_index_window_stage:");
    println!("    source: resident_runner_runtime_layer_state_tracker_layer_indices_from_plan_descriptors");
    println!("    tracker_owner: resident_runner_module");
    println!("    plan_descriptor_rows: {}", runner_plans.len());
    println!("    tracker_rows: {derived_rows}");
    println!("    derived_layer_index_rows: {derived_rows}");
    println!("    tracker_first_layer_idx: {first_layer_idx}");
    println!("    tracker_last_layer_idx: {last_layer_idx}");
    println!("    tracker_contiguous_layer_window: {contiguous_layer_window}");
    println!("    layer_indices_derived_from_plan_descriptors: true");
    println!("    tracker_row_count_derived_from_plan_descriptors: true");
    println!("    fixed_tracker_window_row_count_removed: true");
    println!("    main_literal_tracker_layer_index_array_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(layer_indices)
}

impl QwenResidentLayerRunnerRuntimeLayerStateTracker {
    fn row_mut(
        &mut self,
        layer_idx: u32,
    ) -> anyhow::Result<&mut QwenResidentLayerRunnerRuntimeLayerStateRow> {
        self.rows
            .iter_mut()
            .find(|row| row.layer_idx == layer_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resident runtime layer-state tracker missing layer {}",
                    layer_idx
                )
            })
    }

    fn mark_attention_stage_ready(&mut self, layer_idx: u32) -> anyhow::Result<()> {
        self.row_mut(layer_idx)?.attention_stage_ready = true;
        Ok(())
    }

    fn mark_mlp_stage_ready(&mut self, layer_idx: u32) -> anyhow::Result<()> {
        self.row_mut(layer_idx)?.mlp_stage_ready = true;
        Ok(())
    }

    pub(crate) fn into_window(
        self,
    ) -> anyhow::Result<QwenResidentLayerRunnerRuntimeLayerStateWindow> {
        let tracker_rows = self.rows.len();
        println!("  resident_layer_runner_runtime_layer_state_tracker_consume_stage:");
        println!("    source: resident_runner_runtime_layer_state_tracker_into_window");
        println!("    tracker_owner: resident_runner_module");
        println!("    tracker_rows: {tracker_rows}");
        println!("    tracker_first_layer_idx: {}", self.first_layer_idx);
        println!("    tracker_last_layer_idx: {}", self.last_layer_idx);
        println!("    layer_state_window_derived_from_tracker: true");
        println!("    tracker_rows_dynamic_vec: true");
        println!("    final_readiness_callsite_numbered_stage_checks_removed: true");
        println!("    final_readiness_callsite_uses_tracker_window: true");
        println!("    execution_path_changed: false");
        println!("    hip_graph_capture_started: false");
        qwen_resident_layer_runner_runtime_layer_state_window_from_rows(self.rows)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum QwenResidentLayerRunnerRuntimeReadinessUpdateKind {
    AttentionStageReady,
    MlpStageReady,
}

impl QwenResidentLayerRunnerRuntimeReadinessUpdateKind {
    fn index(self) -> u32 {
        match self {
            Self::AttentionStageReady => 0,
            Self::MlpStageReady => 1,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::AttentionStageReady => "attention_stage_ready",
            Self::MlpStageReady => "mlp_stage_ready",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerRuntimeReadinessUpdateRow {
    layer_idx: u32,
    kind: QwenResidentLayerRunnerRuntimeReadinessUpdateKind,
}

impl QwenResidentLayerRunnerRuntimeReadinessUpdateRow {
    pub(crate) fn attention_stage_ready(layer_idx: u32) -> Self {
        Self {
            layer_idx,
            kind: QwenResidentLayerRunnerRuntimeReadinessUpdateKind::AttentionStageReady,
        }
    }

    pub(crate) fn mlp_stage_ready(layer_idx: u32) -> Self {
        Self {
            layer_idx,
            kind: QwenResidentLayerRunnerRuntimeReadinessUpdateKind::MlpStageReady,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection {
    FirstAttentionDispatch,
    LastAttentionDispatch,
    LastMlpDispatch,
}

impl QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection {
    fn label(self) -> &'static str {
        match self {
            Self::FirstAttentionDispatch => "first_attention_dispatch",
            Self::LastAttentionDispatch => "last_attention_dispatch",
            Self::LastMlpDispatch => "last_mlp_dispatch",
        }
    }

    fn selected_boundary_label(self) -> &'static str {
        match self {
            Self::FirstAttentionDispatch => "first",
            Self::LastAttentionDispatch | Self::LastMlpDispatch => "last",
        }
    }

    fn dispatch_kind(self) -> QwenResidentLayerRunnerDispatchKind {
        match self {
            Self::FirstAttentionDispatch | Self::LastAttentionDispatch => {
                QwenResidentLayerRunnerDispatchKind::DecodeAttention
            }
            Self::LastMlpDispatch => QwenResidentLayerRunnerDispatchKind::Mlp,
        }
    }

    fn readiness_update_kind(self) -> QwenResidentLayerRunnerRuntimeReadinessUpdateKind {
        match self {
            Self::FirstAttentionDispatch | Self::LastAttentionDispatch => {
                QwenResidentLayerRunnerRuntimeReadinessUpdateKind::AttentionStageReady
            }
            Self::LastMlpDispatch => {
                QwenResidentLayerRunnerRuntimeReadinessUpdateKind::MlpStageReady
            }
        }
    }

    fn select_candidate_row(self, candidate_rows: &[(usize, u32)]) -> Option<(usize, u32)> {
        match self {
            Self::FirstAttentionDispatch => candidate_rows.first().copied(),
            Self::LastAttentionDispatch | Self::LastMlpDispatch => candidate_rows.last().copied(),
        }
    }
}

pub(crate) fn qwen_resident_layer_runner_runtime_readiness_update_row_from_plan_descriptor_selection(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    readiness_update_selection: QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection,
) -> anyhow::Result<QwenResidentLayerRunnerRuntimeReadinessUpdateRow> {
    if runner_plans.is_empty() {
        anyhow::bail!(
            "resident runtime readiness update plan selection cannot use empty plan descriptors"
        );
    }

    let dispatch_kind = readiness_update_selection.dispatch_kind();
    let readiness_update_kind = readiness_update_selection.readiness_update_kind();
    let mut candidate_rows = Vec::new();
    for (plan_row_index, plan) in runner_plans.iter().enumerate() {
        let dispatch_present = match dispatch_kind {
            QwenResidentLayerRunnerDispatchKind::DecodeAttention => {
                plan.attention_dispatch.is_some()
            }
            QwenResidentLayerRunnerDispatchKind::Mlp => plan.mlp_dispatch.is_some(),
        };
        if dispatch_present {
            candidate_rows.push((plan_row_index, plan.layer_idx));
        }
    }
    if candidate_rows.is_empty() {
        anyhow::bail!(
            "resident runtime readiness update plan selection {} found no {} dispatch rows",
            readiness_update_selection.label(),
            qwen_resident_layer_runner_dispatch_kind_label(dispatch_kind)
        );
    }

    let candidate_first_layer_idx = candidate_rows[0].1;
    let candidate_last_layer_idx = candidate_rows[candidate_rows.len() - 1].1;
    let (selected_plan_row_index, selected_layer_idx) = readiness_update_selection
        .select_candidate_row(candidate_rows.as_slice())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident runtime readiness update plan selection {} found no selected candidate row",
                readiness_update_selection.label()
            )
        })?;
    let readiness_update_row = match readiness_update_kind {
        QwenResidentLayerRunnerRuntimeReadinessUpdateKind::AttentionStageReady => {
            QwenResidentLayerRunnerRuntimeReadinessUpdateRow::attention_stage_ready(
                selected_layer_idx,
            )
        }
        QwenResidentLayerRunnerRuntimeReadinessUpdateKind::MlpStageReady => {
            QwenResidentLayerRunnerRuntimeReadinessUpdateRow::mlp_stage_ready(selected_layer_idx)
        }
    };

    println!("  resident_layer_runner_runtime_readiness_update_row_plan_selection_stage:");
    println!(
        "    source: resident_runner_runtime_readiness_update_row_from_plan_descriptor_selection"
    );
    println!("    tracker_owner: resident_runner_module");
    println!("    plan_descriptor_rows: {}", runner_plans.len());
    println!("    selection: {}", readiness_update_selection.label());
    println!(
        "    selected_boundary: {}",
        readiness_update_selection.selected_boundary_label()
    );
    println!(
        "    dispatch_kind: {}",
        qwen_resident_layer_runner_dispatch_kind_label(dispatch_kind)
    );
    println!("    candidate_dispatch_rows: {}", candidate_rows.len());
    println!("    candidate_first_layer_idx: {candidate_first_layer_idx}");
    println!("    candidate_last_layer_idx: {candidate_last_layer_idx}");
    println!("    selected_plan_row_index: {selected_plan_row_index}");
    println!("    target_layer_idx: {selected_layer_idx}");
    println!(
        "    readiness_update_kind: {}",
        readiness_update_kind.label()
    );
    println!("    readiness_update_row_derived_from_plan_descriptors: true");
    println!("    main_literal_readiness_update_layer_idx_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(readiness_update_row)
}

pub(crate) fn qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_plan_descriptor_selection(
    tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    readiness_update_selection: QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection,
) -> anyhow::Result<()> {
    let readiness_update_row =
        qwen_resident_layer_runner_runtime_readiness_update_row_from_plan_descriptor_selection(
            runner_plans,
            readiness_update_selection,
        )?;
    qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_readiness_update_rows(
        tracker,
        vec![readiness_update_row],
    )
}

pub(crate) fn qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_readiness_update_rows(
    tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    update_rows: Vec<QwenResidentLayerRunnerRuntimeReadinessUpdateRow>,
) -> anyhow::Result<()> {
    let update_rows_len = update_rows.len();
    if update_rows_len == 0 {
        anyhow::bail!("resident runtime layer-state readiness update rows cannot be empty");
    }
    let tracker_rows = tracker.rows.len();

    let mut update_first_layer_idx = update_rows[0].layer_idx;
    let mut update_last_layer_idx = update_rows[0].layer_idx;
    let mut previous_update_key = None;
    for (update_row_index, update_row) in update_rows.iter().copied().enumerate() {
        update_first_layer_idx = update_first_layer_idx.min(update_row.layer_idx);
        update_last_layer_idx = update_last_layer_idx.max(update_row.layer_idx);
        for prior_update_row in update_rows[..update_row_index].iter().copied() {
            if prior_update_row.layer_idx == update_row.layer_idx
                && prior_update_row.kind == update_row.kind
            {
                anyhow::bail!(
                    "resident runtime layer-state readiness update rows contain duplicate layer {} kind {}",
                    update_row.layer_idx,
                    update_row.kind.label()
                );
            }
        }
        let update_key = (update_row.layer_idx, update_row.kind.index());
        if let Some(previous_update_key) = previous_update_key {
            if update_key <= previous_update_key {
                anyhow::bail!(
                    "resident runtime layer-state readiness update rows are not strictly ascending: layer {} kind {} at row {}",
                    update_row.layer_idx,
                    update_row.kind.label(),
                    update_row_index
                );
            }
        }
        previous_update_key = Some(update_key);
    }

    println!("  resident_layer_runner_runtime_layer_state_tracker_readiness_update_batch_stage:");
    println!("    source: resident_runner_runtime_layer_state_tracker_from_readiness_update_rows");
    println!("    tracker_owner: resident_runner_module");
    println!("    tracker_rows: {tracker_rows}");
    println!("    readiness_update_rows: {update_rows_len}");
    println!("    update_first_layer_idx: {update_first_layer_idx}");
    println!("    update_last_layer_idx: {update_last_layer_idx}");
    println!("    readiness_update_rows_dynamic_vec: true");
    println!("    fixed_readiness_update_row_count_removed: true");
    println!("    update_rows_strictly_ascending: true");
    println!("    update_rows_duplicate_free: true");
    println!("    indexed_update_rows_consumed: true");
    println!("    main_direct_tracker_mark_calls_removed: true");
    println!("    legacy_stage_assignment_sources_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    for (update_row_index, update_row) in update_rows.iter().copied().enumerate() {
        let mut attention_stage_marked = false;
        let mut mlp_stage_marked = false;
        match update_row.kind {
            QwenResidentLayerRunnerRuntimeReadinessUpdateKind::AttentionStageReady => {
                tracker.mark_attention_stage_ready(update_row.layer_idx)?;
                attention_stage_marked = true;
            }
            QwenResidentLayerRunnerRuntimeReadinessUpdateKind::MlpStageReady => {
                tracker.mark_mlp_stage_ready(update_row.layer_idx)?;
                mlp_stage_marked = true;
            }
        }

        println!("  resident_layer_runner_runtime_layer_state_tracker_readiness_update_stage:");
        println!(
            "    source: resident_runner_runtime_layer_state_tracker_from_readiness_update_rows"
        );
        println!("    tracker_owner: resident_runner_module");
        println!("    update_row_index: {update_row_index}");
        println!("    target_layer_idx: {}", update_row.layer_idx);
        println!("    readiness_update_kind: {}", update_row.kind.label());
        println!("    attention_stage_marked: {attention_stage_marked}");
        println!("    mlp_stage_marked: {mlp_stage_marked}");
        println!("    indexed_update_rows_consumed: true");
        println!("    main_direct_tracker_mark_calls_removed: true");
        println!("    legacy_stage_assignment_sources_retained: true");
        println!("    execution_path_changed: false");
        println!("    hip_graph_capture_started: false");
    }

    println!(
        "  resident_layer_runner_runtime_layer_state_tracker_readiness_update_complete_stage:"
    );
    println!("    source: resident_runner_runtime_layer_state_tracker_from_readiness_update_rows");
    println!("    tracker_owner: resident_runner_module");
    println!("    readiness_update_rows: {update_rows_len}");
    println!("    tracker_rows: {tracker_rows}");
    println!("    readiness_update_rows_dynamic_vec: true");
    println!("    fixed_readiness_update_row_count_removed: true");
    println!("    all_update_rows_applied: true");
    println!("    main_direct_tracker_mark_calls_removed: true");
    println!("    legacy_stage_assignment_sources_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) enum QwenResidentLayerRunnerRuntimeMlpReadinessSource {
    MlpAllreduceStage,
    MlpResidualNormStage,
}

impl QwenResidentLayerRunnerRuntimeMlpReadinessSource {
    fn label(self) -> &'static str {
        match self {
            Self::MlpAllreduceStage => "mlp_allreduce_stage",
            Self::MlpResidualNormStage => "mlp_residual_norm_stage",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerRuntimeDenseReadinessUpdateRow {
    layer_idx: u32,
    mlp_readiness_source: QwenResidentLayerRunnerRuntimeMlpReadinessSource,
}

impl QwenResidentLayerRunnerRuntimeDenseReadinessUpdateRow {
    pub(crate) fn new(
        layer_idx: u32,
        mlp_readiness_source: QwenResidentLayerRunnerRuntimeMlpReadinessSource,
    ) -> Self {
        Self {
            layer_idx,
            mlp_readiness_source,
        }
    }
}

fn qwen_resident_layer_runner_runtime_dense_readiness_update_rows_from_stage_ref_window(
    stage_ref_window: &QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'_>,
) -> anyhow::Result<Vec<QwenResidentLayerRunnerRuntimeDenseReadinessUpdateRow>> {
    let stage_ref_window_rows = stage_ref_window.rows.len();
    let update_rows_expected = stage_ref_window_rows.checked_sub(1).ok_or_else(|| {
        anyhow::anyhow!("resident runtime dense readiness derivation requires stage-ref rows")
    })?;
    if update_rows_expected == 0 {
        anyhow::bail!(
            "resident runtime dense readiness derivation requires at least one update row"
        );
    }

    let (window_first_layer_idx, window_last_layer_idx) =
        qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_bounds(stage_ref_window)?;
    let mut update_rows = Vec::with_capacity(update_rows_expected);
    for (stage_row_index, stage_ref_row) in stage_ref_window
        .rows
        .iter()
        .take(update_rows_expected)
        .enumerate()
    {
        let mlp_readiness_source = if stage_row_index + 1 == update_rows_expected {
            QwenResidentLayerRunnerRuntimeMlpReadinessSource::MlpResidualNormStage
        } else {
            QwenResidentLayerRunnerRuntimeMlpReadinessSource::MlpAllreduceStage
        };
        update_rows.push(QwenResidentLayerRunnerRuntimeDenseReadinessUpdateRow::new(
            stage_ref_row.layer_idx,
            mlp_readiness_source,
        ));
    }
    let derived_rows = update_rows.len();
    let derived_update_first_layer_idx = update_rows[0].layer_idx;
    let derived_update_last_layer_idx = update_rows[update_rows_expected - 1].layer_idx;

    println!(
        "  resident_layer_runner_runtime_dense_readiness_update_rows_from_stage_ref_window_stage:"
    );
    println!(
        "    source: resident_runner_runtime_dense_readiness_update_rows_from_stage_ref_window"
    );
    println!("    tracker_owner: resident_runner_module");
    println!("    stage_ref_window_rows: {stage_ref_window_rows}");
    println!("    stage_ref_window_first_layer_idx: {window_first_layer_idx}");
    println!("    stage_ref_window_last_layer_idx: {window_last_layer_idx}");
    println!("    dense_readiness_update_rows: {update_rows_expected}");
    println!("    derived_dense_readiness_rows: {derived_rows}");
    println!("    derived_update_first_layer_idx: {derived_update_first_layer_idx}");
    println!("    derived_update_last_layer_idx: {derived_update_last_layer_idx}");
    println!("    dense_readiness_update_rows_derived_from_stage_ref_window_len: true");
    println!("    fixed_dense_readiness_update_row_count_removed: true");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    prefix_stage_ref_rows_consumed: true");
    println!("    final_stage_ref_row_reserved_for_seeded_last_mlp: true");
    println!("    final_prefix_update_row_uses_residual_norm_source: true");
    println!("    main_literal_dense_readiness_update_rows_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(update_rows)
}

pub(crate) fn qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_dense_stage_ref_window(
    tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    stage_ref_window: QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'_>,
) -> anyhow::Result<()> {
    let update_rows =
        qwen_resident_layer_runner_runtime_dense_readiness_update_rows_from_stage_ref_window(
            &stage_ref_window,
        )?;
    qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_dense_readiness_rows(
        tracker,
        stage_ref_window,
        update_rows.as_slice(),
    )
}

pub(crate) fn qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_dense_readiness_rows(
    tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    stage_ref_window: QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'_>,
    update_rows: &[QwenResidentLayerRunnerRuntimeDenseReadinessUpdateRow],
) -> anyhow::Result<()> {
    let update_rows_len = update_rows.len();
    if update_rows_len == 0 {
        anyhow::bail!("resident runtime layer-state dense readiness update rows cannot be empty");
    }
    let tracker_rows = tracker.rows.len();
    let stage_ref_window_rows = stage_ref_window.rows.len();

    let (window_first_layer_idx, window_last_layer_idx) =
        qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_bounds(&stage_ref_window)?;
    let mut update_first_layer_idx = update_rows[0].layer_idx;
    let mut update_last_layer_idx = update_rows[0].layer_idx;
    let mut previous_update_layer_idx = None;
    for (update_row_index, update_row) in update_rows.iter().enumerate() {
        update_first_layer_idx = update_first_layer_idx.min(update_row.layer_idx);
        update_last_layer_idx = update_last_layer_idx.max(update_row.layer_idx);
        for prior_update_row in update_rows[..update_row_index].iter() {
            if prior_update_row.layer_idx == update_row.layer_idx {
                anyhow::bail!(
                    "resident runtime layer-state dense readiness update rows contain duplicate layer {}",
                    update_row.layer_idx
                );
            }
        }
        if let Some(previous_update_layer_idx) = previous_update_layer_idx {
            if update_row.layer_idx <= previous_update_layer_idx {
                anyhow::bail!(
                    "resident runtime layer-state dense readiness update rows are not strictly ascending: layer {} after layer {} at row {}",
                    update_row.layer_idx,
                    previous_update_layer_idx,
                    update_row_index
                );
            }
        }
        previous_update_layer_idx = Some(update_row.layer_idx);
    }

    println!("  resident_layer_runner_runtime_layer_state_tracker_dense_readiness_batch_stage:");
    println!("    source: resident_runner_runtime_layer_state_tracker_from_dense_readiness_rows");
    println!("    tracker_owner: resident_runner_module");
    println!("    tracker_rows: {tracker_rows}");
    println!("    stage_ref_window_rows: {stage_ref_window_rows}");
    println!("    stage_ref_window_first_layer_idx: {window_first_layer_idx}");
    println!("    stage_ref_window_last_layer_idx: {window_last_layer_idx}");
    println!("    dense_readiness_update_rows: {update_rows_len}");
    println!("    update_first_layer_idx: {update_first_layer_idx}");
    println!("    update_last_layer_idx: {update_last_layer_idx}");
    println!("    update_rows_strictly_ascending: true");
    println!("    update_rows_duplicate_free: true");
    println!("    indexed_update_rows_consumed: true");
    println!("    main_per_layer_dense_tracker_update_calls_removed: true");
    println!("    legacy_stage_ref_window_consumed_once: true");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    legacy_stage_refs_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let mut stage_ref_rows = stage_ref_window.rows;
    for (update_row_index, update_row) in update_rows.iter().copied().enumerate() {
        let stage_ref_row = stage_ref_rows
            .iter_mut()
            .find(|stage_ref_row| stage_ref_row.layer_idx == update_row.layer_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resident runtime layer-state dense readiness update row layer {} has no legacy stage-ref row",
                    update_row.layer_idx
                )
            })?;
        let attention_stage_ready = stage_ref_row.refs.qkv_stage.is_some();
        let mlp_stage_ready = match update_row.mlp_readiness_source {
            QwenResidentLayerRunnerRuntimeMlpReadinessSource::MlpAllreduceStage => {
                stage_ref_row.refs.mlp_allreduce_stage.is_some()
            }
            QwenResidentLayerRunnerRuntimeMlpReadinessSource::MlpResidualNormStage => {
                stage_ref_row.refs.mlp_residual_norm_stage.is_some()
            }
        };
        if attention_stage_ready {
            tracker.mark_attention_stage_ready(update_row.layer_idx)?;
        }
        if mlp_stage_ready {
            tracker.mark_mlp_stage_ready(update_row.layer_idx)?;
        }

        println!(
            "  resident_layer_runner_runtime_layer_state_tracker_dense_stage_ref_update_stage:"
        );
        println!(
            "    source: resident_runner_runtime_layer_state_tracker_from_dense_readiness_rows"
        );
        println!("    tracker_owner: resident_runner_module");
        println!("    update_row_index: {update_row_index}");
        println!("    target_layer_idx: {}", update_row.layer_idx);
        println!("    attention_readiness_source: qkv_stage");
        println!("    attention_stage_ready: {attention_stage_ready}");
        println!(
            "    mlp_readiness_source: {}",
            update_row.mlp_readiness_source.label()
        );
        println!("    mlp_stage_ready: {mlp_stage_ready}");
        println!("    stage_ref_window_consumed: true");
        println!("    indexed_update_rows_consumed: true");
        println!("    main_per_layer_dense_tracker_update_calls_removed: true");
        println!(
            "    main_numbered_stage_is_some_checks_removed_from_final_readiness_callsite: true"
        );
        println!("    legacy_stage_refs_retained: true");
        println!("    execution_path_changed: false");
        println!("    hip_graph_capture_started: false");
    }

    println!("  resident_layer_runner_runtime_layer_state_tracker_dense_readiness_complete_stage:");
    println!("    source: resident_runner_runtime_layer_state_tracker_from_dense_readiness_rows");
    println!("    tracker_owner: resident_runner_module");
    println!("    dense_readiness_update_rows: {update_rows_len}");
    println!("    tracker_rows: {tracker_rows}");
    println!("    all_update_rows_applied: true");
    println!("    main_numbered_stage_is_some_checks_removed_from_final_readiness_callsite: true");
    println!("    main_per_layer_dense_tracker_update_calls_removed: true");
    println!("    legacy_stage_refs_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(())
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_state_window_from_rows(
    rows: Vec<QwenResidentLayerRunnerRuntimeLayerStateRow>,
) -> anyhow::Result<QwenResidentLayerRunnerRuntimeLayerStateWindow> {
    let layer_state_rows = rows.len();
    let (first_layer_idx, last_layer_idx, contiguous_layer_window) =
        qwen_resident_layer_runner_runtime_layer_state_row_window_bounds(rows.as_slice())?;

    println!("  resident_layer_runner_runtime_layer_state_window_stage:");
    println!("    source: resident_runner_runtime_layer_state_rows");
    println!("    layer_state_window_owner: resident_runner_module");
    println!("    layer_state_rows: {layer_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    layer_state_rows_strictly_ascending: true");
    println!("    layer_state_rows_duplicate_free: true");
    println!("    per_layer_runtime_state_storage: true");
    println!("    layer_state_rows_dynamic_vec: true");
    println!("    dispatch_kind_literals_in_main_readiness_callsite_removed: true");
    println!("    legacy_numbered_stage_assignment_sources_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerRuntimeLayerStateWindow {
        rows,
        first_layer_idx,
        last_layer_idx,
    })
}

pub(crate) fn qwen_resident_layer_runner_runtime_layer_observation_window_from_layer_state_window(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    window: QwenResidentLayerRunnerRuntimeLayerStateWindow,
) -> anyhow::Result<Vec<QwenResidentLayerRunnerRuntimeLayerObservation>> {
    let layer_state_rows = window.rows.len();
    let planned_dependency_ids =
        qwen_resident_layer_runner_planned_runtime_dependency_ids(runner_plans)?;

    let mut dispatch_rows = Vec::with_capacity(planned_dependency_ids.len());
    for plan in runner_plans {
        if plan.attention_dispatch.is_none() && plan.mlp_dispatch.is_none() {
            continue;
        }
        let layer_state = window
            .rows
            .iter()
            .find(|row| row.layer_idx == plan.layer_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resident runtime layer-state window is missing planned dispatch layer {}",
                    plan.layer_idx
                )
            })?;
        if plan.attention_dispatch.is_some() {
            dispatch_rows.push(QwenResidentLayerRunnerRuntimeDispatchSatisfactionRow::new(
                plan.layer_idx,
                QwenResidentLayerRunnerDispatchKind::DecodeAttention,
                layer_state.attention_stage_ready,
            ));
        }
        if plan.mlp_dispatch.is_some() {
            dispatch_rows.push(QwenResidentLayerRunnerRuntimeDispatchSatisfactionRow::new(
                plan.layer_idx,
                QwenResidentLayerRunnerDispatchKind::Mlp,
                layer_state.mlp_stage_ready,
            ));
        }
    }

    for row in window.rows.iter() {
        let plan = runner_plans
            .iter()
            .find(|plan| plan.layer_idx == row.layer_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resident runtime layer-state row layer {} has no plan descriptor",
                    row.layer_idx
                )
            })?;
        if plan.attention_dispatch.is_none() && plan.mlp_dispatch.is_none() {
            anyhow::bail!(
                "resident runtime layer-state row layer {} has no planned runtime dispatch",
                row.layer_idx
            );
        }
    }

    println!("  resident_layer_runner_runtime_layer_state_dispatch_satisfaction_stage:");
    println!(
        "    source: resident_runner_runtime_layer_state_window_to_dispatch_satisfaction_rows"
    );
    println!("    layer_state_window_owner: resident_runner_module");
    println!("    layer_state_rows: {layer_state_rows}");
    println!("    window_first_layer_idx: {}", window.first_layer_idx);
    println!("    window_last_layer_idx: {}", window.last_layer_idx);
    println!(
        "    planned_dependency_rows: {}",
        planned_dependency_ids.len()
    );
    println!(
        "    derived_dispatch_satisfaction_rows: {}",
        dispatch_rows.len()
    );
    println!("    dispatch_satisfaction_rows_derived_by_resident_runner: true");
    println!("    main_dispatch_satisfaction_row_literals_removed: true");
    println!("    main_dependency_id_enum_literals_removed_from_readiness_callsite: true");
    println!("    legacy_numbered_stage_bool_sources_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_runtime_layer_observation_window_from_dispatch_satisfaction_row_slice(
        runner_plans,
        dispatch_rows.as_slice(),
    )
}

fn qwen_resident_layer_runner_runtime_layer_observation_window_from_dispatch_satisfaction_row_slice(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    rows: &[QwenResidentLayerRunnerRuntimeDispatchSatisfactionRow],
) -> anyhow::Result<Vec<QwenResidentLayerRunnerRuntimeLayerObservation>> {
    if rows.is_empty() {
        anyhow::bail!("resident runtime dispatch satisfaction rows cannot be empty");
    }

    let planned_dependency_ids =
        qwen_resident_layer_runner_planned_runtime_dependency_ids(runner_plans)?;

    for left in 0..rows.len() {
        for right in (left + 1)..rows.len() {
            if rows[left].layer_idx == rows[right].layer_idx && rows[left].kind == rows[right].kind
            {
                anyhow::bail!(
                    "resident runtime dispatch satisfaction rows contain duplicate layer {} {} rows",
                    rows[left].layer_idx,
                    qwen_resident_layer_runner_dispatch_kind_label(rows[left].kind)
                );
            }
        }
    }

    for plan in runner_plans {
        if let Some(dispatch) = plan.attention_dispatch {
            if !rows.iter().any(|row| {
                row.layer_idx == plan.layer_idx
                    && row.kind == QwenResidentLayerRunnerDispatchKind::DecodeAttention
            }) {
                anyhow::bail!(
                    "resident runtime dispatch satisfaction rows are missing layer {} attention dependency {}",
                    plan.layer_idx,
                    dispatch.dependency_id.label()
                );
            }
        }
        if let Some(dispatch) = plan.mlp_dispatch {
            if !rows.iter().any(|row| {
                row.layer_idx == plan.layer_idx
                    && row.kind == QwenResidentLayerRunnerDispatchKind::Mlp
            }) {
                anyhow::bail!(
                    "resident runtime dispatch satisfaction rows are missing layer {} MLP dependency {}",
                    plan.layer_idx,
                    dispatch.dependency_id.label()
                );
            }
        }
    }

    let mut satisfied_dependency_ids = Vec::new();
    for row in rows.iter().copied() {
        let plan = runner_plans
            .iter()
            .find(|plan| plan.layer_idx == row.layer_idx)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "resident runtime dispatch satisfaction row layer {} {} has no plan descriptor",
                    row.layer_idx,
                    qwen_resident_layer_runner_dispatch_kind_label(row.kind)
                )
            })?;
        let dependency_id = match row.kind {
            QwenResidentLayerRunnerDispatchKind::DecodeAttention => plan
                .attention_dispatch
                .map(|dispatch| dispatch.dependency_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "resident runtime dispatch satisfaction row layer {} attention has no planned attention dispatch",
                        row.layer_idx
                    )
                })?,
            QwenResidentLayerRunnerDispatchKind::Mlp => plan
                .mlp_dispatch
                .map(|dispatch| dispatch.dependency_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "resident runtime dispatch satisfaction row layer {} MLP has no planned MLP dispatch",
                        row.layer_idx
                    )
                })?,
        };
        if row.satisfied {
            satisfied_dependency_ids.push(dependency_id);
        }
    }

    if rows.len() != planned_dependency_ids.len() {
        anyhow::bail!(
            "resident runtime dispatch satisfaction rows produced {} rows, expected {} planned dependencies",
            rows.len(),
            planned_dependency_ids.len()
        );
    }

    println!("  resident_layer_runner_runtime_dispatch_satisfaction_rows_stage:");
    println!("    source: resident_runner_runtime_dispatch_satisfaction_rows");
    println!(
        "    planned_dependency_rows: {}",
        planned_dependency_ids.len()
    );
    println!("    dispatch_satisfaction_rows: {}", rows.len());
    println!(
        "    satisfied_dependency_id_rows: {}",
        satisfied_dependency_ids.len()
    );
    println!("    dependency_id_mapping_owner: resident_runner_module");
    println!("    dispatch_satisfaction_rows_duplicate_free: true");
    println!("    plan_descriptor_dispatch_keys_bound: true");
    println!("    positional_dependency_contract: false");
    println!("    main_hardcoded_satisfied_dependency_push_blocks_removed: true");
    println!("    main_dependency_id_enum_literals_removed_from_readiness_callsite: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_runtime_layer_observation_window_from_satisfied_dependency_ids(
        runner_plans,
        satisfied_dependency_ids.as_slice(),
    )
}

fn qwen_resident_layer_runner_planned_runtime_dependency_ids(
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
) -> anyhow::Result<Vec<QwenResidentLayerRunnerDependencyId>> {
    let mut seen_plan_layers = std::collections::BTreeSet::new();
    let mut planned_dependencies = Vec::new();
    for plan in runner_plans {
        if !seen_plan_layers.insert(plan.layer_idx) {
            anyhow::bail!(
                "resident runtime dependency presence window has duplicate plan layer {}",
                plan.layer_idx
            );
        }
        if let Some(dispatch) = plan.attention_dispatch {
            planned_dependencies.push(dispatch.dependency_id);
        }
        if let Some(dispatch) = plan.mlp_dispatch {
            planned_dependencies.push(dispatch.dependency_id);
        }
    }
    for left in 0..planned_dependencies.len() {
        for right in (left + 1)..planned_dependencies.len() {
            if planned_dependencies[left] == planned_dependencies[right] {
                anyhow::bail!(
                    "resident runtime dependency presence window has duplicate planned dependency {}",
                    planned_dependencies[left].label()
                );
            }
        }
    }
    Ok(planned_dependencies)
}

#[derive(Clone, Copy)]
pub(crate) enum QwenResidentLayerRunnerRankHBufferRole {
    InputNorm,
    PostAttentionNorm,
}

impl QwenResidentLayerRunnerRankHBufferRole {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::InputNorm => "input_norm",
            Self::PostAttentionNorm => "post_attention_norm",
        }
    }

    fn index(self) -> u32 {
        match self {
            Self::InputNorm => 0,
            Self::PostAttentionNorm => 1,
        }
    }

    fn output_buffer_label(self) -> &'static str {
        match self {
            Self::InputNorm => "input norm",
            Self::PostAttentionNorm => "post-attention norm",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerRankHBufferWindowRow {
    pub(crate) layer_idx: u32,
    pub(crate) role: QwenResidentLayerRunnerRankHBufferRole,
    pub(crate) capacity_ranks: usize,
}

pub(crate) fn qwen_resident_layer_runner_dense_rank_h_buffer_window_rows<const ROWS: usize>(
    rows: [(u32, QwenResidentLayerRunnerRankHBufferRole); ROWS],
    tp_world: usize,
) -> anyhow::Result<[QwenResidentLayerRunnerRankHBufferWindowRow; ROWS]> {
    if ROWS == 0 {
        anyhow::bail!("resident dense rank-H output buffer window requires at least one row");
    }
    if tp_world == 0 {
        anyhow::bail!("resident dense rank-H output buffer window cannot use TP0");
    }

    let mut seen_rows = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut window_rows = Vec::with_capacity(ROWS);
    for (layer_idx, role) in rows {
        if !seen_rows.insert((layer_idx, role.index())) {
            anyhow::bail!(
                "resident dense rank-H output buffer window contains duplicate row layer {} role {}",
                layer_idx,
                role.label()
            );
        }
        first_layer_idx = first_layer_idx.min(layer_idx);
        last_layer_idx = last_layer_idx.max(layer_idx);
        window_rows.push(QwenResidentLayerRunnerRankHBufferWindowRow {
            layer_idx,
            role,
            capacity_ranks: tp_world,
        });
    }

    let row_count = window_rows.len();
    let window_rows = window_rows.try_into().map_err(
        |window_rows: Vec<QwenResidentLayerRunnerRankHBufferWindowRow>| {
            anyhow::anyhow!(
                "resident dense rank-H output buffer window produced {} rows, expected {}",
                window_rows.len(),
                ROWS
            )
        },
    )?;

    println!("  resident_layer_runner_dense_rank_h_buffer_window_stage:");
    println!("    source: resident_runner_dense_rank_h_buffer_window_rows");
    println!("    window_owner: resident_runner_module");
    println!("    row_count: {row_count}");
    println!("    first_layer_idx: {first_layer_idx}");
    println!("    last_layer_idx: {last_layer_idx}");
    println!("    capacity_ranks: {tp_world}");
    println!("    duplicate_layer_role_rows: false");
    println!("    main_repeated_vec_capacity_literals_replaced: true");
    println!("    legacy_vec_storage_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(window_rows)
}

pub(crate) fn qwen_resident_layer_runner_rank_h_buffer_vec_from_window_row(
    row: QwenResidentLayerRunnerRankHBufferWindowRow,
) -> Vec<mcore::DeviceBuffer> {
    println!("  resident_layer_runner_rank_h_buffer_vec_from_window_row_stage:");
    println!("    source: resident_runner_rank_h_buffer_vec_from_window_row");
    println!("    layer_idx: {}", row.layer_idx);
    println!("    role: {}", row.role.label());
    println!("    capacity_ranks: {}", row.capacity_ranks);
    println!("    legacy_vec_storage_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Vec::with_capacity(row.capacity_ranks)
}

pub(crate) fn qwen_resident_layer_runner_rank_h_buffer_vec_window_from_rows<const ROWS: usize>(
    rows: [QwenResidentLayerRunnerRankHBufferWindowRow; ROWS],
) -> anyhow::Result<[Vec<mcore::DeviceBuffer>; ROWS]> {
    if ROWS == 0 {
        anyhow::bail!("resident dense rank-H legacy vec window requires at least one row");
    }

    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut vecs = Vec::with_capacity(ROWS);
    for row in rows {
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
        vecs.push(qwen_resident_layer_runner_rank_h_buffer_vec_from_window_row(row));
    }
    let vec_count = vecs.len();
    let vecs = vecs
        .try_into()
        .map_err(|vecs: Vec<Vec<mcore::DeviceBuffer>>| {
            anyhow::anyhow!(
                "resident dense rank-H legacy vec window produced {} vecs, expected {}",
                vecs.len(),
                ROWS
            )
        })?;

    println!("  resident_layer_runner_rank_h_buffer_vec_window_stage:");
    println!("    source: resident_runner_rank_h_buffer_vec_window_from_rows");
    println!("    window_owner: resident_runner_module");
    println!("    row_count: {ROWS}");
    println!("    first_layer_idx: {first_layer_idx}");
    println!("    last_layer_idx: {last_layer_idx}");
    println!("    vec_count: {vec_count}");
    println!("    main_indexed_window_row_vec_calls_removed: true");
    println!("    legacy_vec_storage_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(vecs)
}

pub(crate) fn qwen_resident_layer_runner_prepare_dense_rank_h_buffer_from_window<
    const ROWS: usize,
>(
    rows: [QwenResidentLayerRunnerRankHBufferWindowRow; ROWS],
    target_layer_idx: u32,
    role: QwenResidentLayerRunnerRankHBufferRole,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    outputs: &mut Vec<mcore::DeviceBuffer>,
    h_bytes: usize,
) -> anyhow::Result<()> {
    if ROWS == 0 {
        anyhow::bail!("resident dense rank-H buffer prepare window requires at least one row");
    }

    let selected_row = rows
        .iter()
        .copied()
        .find(|row| row.layer_idx == target_layer_idx && row.role.index() == role.index())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense rank-H buffer prepare window missing layer {} role {}",
                target_layer_idx,
                role.label()
            )
        })?;
    qwen_prepare_next_decode_layer_rank_h_buffers(
        dev,
        peer_workspaces,
        outputs,
        selected_row.capacity_ranks,
        h_bytes,
        target_layer_idx,
        role.output_buffer_label(),
    )?;
    if outputs.len() != selected_row.capacity_ranks {
        anyhow::bail!(
            "resident dense rank-H buffer prepare layer {} role {} produced {} ranks, expected {}",
            target_layer_idx,
            role.label(),
            outputs.len(),
            selected_row.capacity_ranks
        );
    }

    println!("  resident_layer_runner_dense_rank_h_buffer_prepare_stage:");
    println!("    source: resident_runner_dense_rank_h_buffer_prepare_from_window");
    println!("    window_owner: resident_runner_module");
    println!("    target_layer_idx: {target_layer_idx}");
    println!("    role: {}", role.label());
    println!("    capacity_ranks: {}", selected_row.capacity_ranks);
    println!("    output_ranks: {}", outputs.len());
    println!("    dense_rank_h_buffer_window_consumed: true");
    println!("    main_manual_rank_h_buffer_prepare_loop_removed: true");
    println!("    legacy_vec_storage_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchWindow {
    pub(crate) dispatch_layer_idx: u32,
    pub(crate) lookahead_layer_idx: u32,
    pub(crate) handoff_contract: QwenResidentLayerRunnerDenseLayerHandoffContract,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QwenResidentLayerPlan {
    pub(crate) layer_idx: u32,
    pub(crate) stage_mask: QwenResidentLayerPlanStageMask,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QwenResidentLayerRunnerPlanState {
    pub(crate) layer_idx: u32,
    pub(crate) input_norm: bool,
    pub(crate) qkv: bool,
    pub(crate) qk_norm_rope: bool,
    pub(crate) fp4_kv_cache: bool,
    pub(crate) fp4_attention: bool,
    pub(crate) o_proj: bool,
    pub(crate) post_attn_norm: bool,
    pub(crate) tp_o_proj_peer: bool,
    pub(crate) mlp: bool,
    pub(crate) tp_mlp_peer: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QwenResidentLayerPlanStagePresence {
    pub(crate) input_norm: bool,
    pub(crate) qkv: bool,
    pub(crate) qk_norm_rope: bool,
    pub(crate) fp4_kv_cache: bool,
    pub(crate) fp4_attention: bool,
    pub(crate) o_proj: bool,
    pub(crate) post_attn_norm: bool,
    pub(crate) tp_o_proj_peer: bool,
    pub(crate) mlp: bool,
    pub(crate) tp_mlp_peer: bool,
}

impl QwenResidentLayerPlanStagePresence {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_norm: bool,
        qkv: bool,
        qk_norm_rope: bool,
        fp4_kv_cache: bool,
        fp4_attention: bool,
        o_proj: bool,
        post_attn_norm: bool,
        tp_o_proj_peer: bool,
        mlp: bool,
        tp_mlp_peer: bool,
    ) -> Self {
        Self {
            input_norm,
            qkv,
            qk_norm_rope,
            fp4_kv_cache,
            fp4_attention,
            o_proj,
            post_attn_norm,
            tp_o_proj_peer,
            mlp,
            tp_mlp_peer,
        }
    }

    pub(crate) fn stage_mask(self) -> QwenResidentLayerPlanStageMask {
        QwenResidentLayerPlanStageMask::from_stage_presence(
            self.input_norm,
            self.qkv,
            self.qk_norm_rope,
            self.fp4_kv_cache,
            self.fp4_attention,
            self.o_proj,
            self.post_attn_norm,
            self.tp_o_proj_peer,
            self.mlp,
            self.tp_mlp_peer,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QwenResidentLayerPlanStageMask(u16);

pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_INPUT_NORM: u16 = 1 << 0;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_QKV: u16 = 1 << 1;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_QK_NORM_ROPE: u16 = 1 << 2;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_FP4_KV_CACHE: u16 = 1 << 3;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_FP4_ATTENTION: u16 = 1 << 4;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_O_PROJ: u16 = 1 << 5;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_POST_ATTN_NORM: u16 = 1 << 6;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_TP_O_PROJ_PEER: u16 = 1 << 7;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_MLP: u16 = 1 << 8;
pub(crate) const QWEN_RESIDENT_LAYER_PLAN_STAGE_TP_MLP_PEER: u16 = 1 << 9;

impl QwenResidentLayerPlanStageMask {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_stage_presence(
        input_norm: bool,
        qkv: bool,
        qk_norm_rope: bool,
        fp4_kv_cache: bool,
        fp4_attention: bool,
        o_proj: bool,
        post_attn_norm: bool,
        tp_o_proj_peer: bool,
        mlp: bool,
        tp_mlp_peer: bool,
    ) -> Self {
        let mut mask = 0u16;
        if input_norm {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_INPUT_NORM;
        }
        if qkv {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_QKV;
        }
        if qk_norm_rope {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_QK_NORM_ROPE;
        }
        if fp4_kv_cache {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_FP4_KV_CACHE;
        }
        if fp4_attention {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_FP4_ATTENTION;
        }
        if o_proj {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_O_PROJ;
        }
        if post_attn_norm {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_POST_ATTN_NORM;
        }
        if tp_o_proj_peer {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_TP_O_PROJ_PEER;
        }
        if mlp {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_MLP;
        }
        if tp_mlp_peer {
            mask |= QWEN_RESIDENT_LAYER_PLAN_STAGE_TP_MLP_PEER;
        }
        Self(mask)
    }

    pub(crate) fn contains(self, stage: u16) -> bool {
        self.0 & stage != 0
    }

    pub(crate) fn raw(self) -> u16 {
        self.0
    }
}

pub(crate) fn qwen_resident_layer_runner_stage_masks_from_presence_rows<const N: usize>(
    rows: [QwenResidentLayerPlanStagePresence; N],
) -> anyhow::Result<[QwenResidentLayerPlanStageMask; N]> {
    if rows.is_empty() {
        anyhow::bail!("resident layer stage presence table must not be empty");
    }
    let masks = rows.map(QwenResidentLayerPlanStagePresence::stage_mask);
    let stage_mask_or = masks
        .iter()
        .fold(0u16, |mask, stage_mask| mask | stage_mask.raw());
    println!("  resident_layer_runner_stage_presence_table_stage:");
    println!("    source: resident_runner_stage_presence_rows");
    println!("    stage_presence_rows: {N}");
    println!("    stage_masks_derived_by_resident_runner_module: true");
    println!("    stage_mask_or: 0x{stage_mask_or:04x}");
    println!("    execution_path_changed: false");
    println!("    numerics_changed: false");
    println!("    graph_capture_started: false");
    Ok(masks)
}

pub(crate) fn qwen_resident_layer_runner_layer_plans_from_topology_stage_masks<const N: usize>(
    topology: [QwenResidentLayerRunnerLayerTopology; N],
    stage_masks: [QwenResidentLayerPlanStageMask; N],
) -> anyhow::Result<[QwenResidentLayerPlan; N]> {
    if topology.is_empty() {
        anyhow::bail!("resident layer topology table must not be empty");
    }
    let plans = std::array::from_fn(|index| QwenResidentLayerPlan {
        layer_idx: topology[index].layer_idx,
        stage_mask: stage_masks[index],
    });
    let first_layer_idx = plans[0].layer_idx;
    let last_layer_idx = plans[N - 1].layer_idx;
    let stage_mask_or = stage_masks
        .iter()
        .fold(0u16, |mask, stage_mask| mask | stage_mask.raw());
    println!("  resident_layer_runner_layer_plan_topology_source_stage:");
    println!("    source: resident_runner_layer_topology_plus_compact_stage_masks");
    println!("    topology_rows: {N}");
    println!("    stage_mask_rows: {N}");
    println!("    layer_indices_derived_from_topology: true");
    println!("    duplicated_layer_indices_in_plan_table_removed: true");
    println!("    compact_stage_mask_rows: true");
    println!("    stage_mask_bits_per_row: 10");
    println!("    stage_mask_or: 0x{stage_mask_or:04x}");
    println!("    first_layer_idx: {first_layer_idx}");
    println!("    last_layer_idx: {last_layer_idx}");
    println!("    resident_runner_module_owns_plan_metadata_builder: true");
    println!("    execution_path_changed: false");
    println!("    numerics_changed: false");
    println!("    graph_capture_started: false");
    Ok(plans)
}

pub(crate) fn qwen_resident_layer_runner_plan_state_from_layer_plan(
    plan: QwenResidentLayerPlan,
) -> QwenResidentLayerRunnerPlanState {
    let stage_mask = plan.stage_mask;
    QwenResidentLayerRunnerPlanState {
        layer_idx: plan.layer_idx,
        input_norm: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_INPUT_NORM),
        qkv: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_QKV),
        qk_norm_rope: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_QK_NORM_ROPE),
        fp4_kv_cache: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_FP4_KV_CACHE),
        fp4_attention: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_FP4_ATTENTION),
        o_proj: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_O_PROJ),
        post_attn_norm: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_POST_ATTN_NORM),
        tp_o_proj_peer: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_TP_O_PROJ_PEER),
        mlp: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_MLP),
        tp_mlp_peer: stage_mask.contains(QWEN_RESIDENT_LAYER_PLAN_STAGE_TP_MLP_PEER),
    }
}

pub(crate) fn qwen_resident_layer_runner_plan_states_from_layer_plans<const N: usize>(
    plans: [QwenResidentLayerPlan; N],
) -> anyhow::Result<[QwenResidentLayerRunnerPlanState; N]> {
    if plans.is_empty() {
        anyhow::bail!("resident layer plan table must not be empty");
    }
    let mut first_layer_idx = plans[0].layer_idx;
    let mut last_layer_idx = plans[0].layer_idx;
    for (row_index, plan) in plans.iter().enumerate() {
        first_layer_idx = first_layer_idx.min(plan.layer_idx);
        last_layer_idx = last_layer_idx.max(plan.layer_idx);
        if plans[..row_index]
            .iter()
            .any(|prior_plan| prior_plan.layer_idx == plan.layer_idx)
        {
            anyhow::bail!(
                "duplicate resident layer plan table row for layer {}",
                plan.layer_idx
            );
        }
    }
    let states = plans.map(qwen_resident_layer_runner_plan_state_from_layer_plan);
    println!("  resident_layer_runner_plan_table_stage:");
    println!("    source: resident_runner_layer_plan_table");
    println!("    plan_table_rows: {N}");
    println!("    first_layer_idx: {first_layer_idx}");
    println!("    last_layer_idx: {last_layer_idx}");
    println!("    compact_stage_mask_rows: true");
    println!("    layer_indices_derived_from_topology: true");
    println!("    plan_states_derived_from_table: true");
    println!("    plan_state_builder_owned_by_resident_runner_module: true");
    println!("    direct_plan_state_rows_in_main_removed: true");
    println!("    table_heap_allocation: false");
    println!("    execution_path_changed: false");
    println!("    numerics_changed: false");
    println!("    graph_capture_started: false");
    Ok(states)
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotMetadataRow {
    pub(crate) layer_idx: u32,
    pub(crate) role_mask: u32,
    pub(crate) slot_count: u32,
    pub(crate) flags: u64,
}

impl QwenResidentLayerRunnerDenseOutputSlotMetadataRow {
    fn legacy_host_bridge(layer_idx: u32) -> Self {
        Self {
            layer_idx,
            role_mask: QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_ROLE_MASK_ALL,
            slot_count: QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_COUNT,
            flags: QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_FLAG_HOST_LEGACY_BRIDGE,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotResolverTarget {
    pub(crate) layer_idx: u32,
    pub(crate) lookahead_layer_idx: u32,
    pub(crate) row_index: u32,
    pub(crate) row: QwenResidentLayerRunnerDenseOutputSlotMetadataRow,
    pub(crate) source: &'static str,
}

pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotHbmState {
    pub(crate) table_buffer: mcore::DeviceBuffer,
    pub(crate) table_rows: usize,
    pub(crate) table_bytes: usize,
    pub(crate) table_checksum: u64,
    pub(crate) table_va: u64,
    pub(crate) layer_index_buffer: mcore::DeviceBuffer,
    pub(crate) layer_index_entries: usize,
    pub(crate) layer_index_bytes: usize,
    pub(crate) layer_index_checksum: u64,
    pub(crate) layer_index_va: u64,
    pub(crate) source: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan {
    pub(crate) expected_layer_idx: u32,
    pub(crate) expected_row_index: u32,
    pub(crate) table_rows: usize,
    pub(crate) table_va: u64,
    pub(crate) table_checksum: u64,
    pub(crate) layer_index_entries: usize,
    pub(crate) layer_index_va: u64,
    pub(crate) layer_index_checksum: u64,
    pub(crate) derived_without_hbm_readback: bool,
    pub(crate) source: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContext {
    pub(crate) apply_plan: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan,
    pub(crate) resolver_target: QwenResidentLayerRunnerDenseOutputSlotResolverTarget,
    pub(crate) source: &'static str,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextRow {
    layer_idx: u32,
    context: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContext,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow {
    rows: Vec<QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextRow>,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseOutputSlotApplyTimeGuard {
    pub(crate) table_va_stable: bool,
    pub(crate) layer_index_va_stable: bool,
    pub(crate) table_rows_match: bool,
    pub(crate) layer_index_entries_match: bool,
    pub(crate) table_checksum_match: bool,
    pub(crate) layer_index_checksum_match: bool,
    pub(crate) source: &'static str,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerQkvAllRankStage {
    stage: QwenNextDecodeLayer1QkvAllRankStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage {
    stage: QwenNextDecodeLayer1QkRopeAllRankStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage {
    stage: QwenNextDecodeLayer1KvAppendAllRankStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerAttentionAllRankStage {
    stage: QwenNextDecodeLayer1AttentionAllRankStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerOProjAllReduceStage {
    stage: QwenNextDecodeLayer1OProjAllReduceStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage {
    stage: QwenNextDecodeLayer1PostAttnNormAllRankStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerMlpAllReduceStage {
    stage: QwenNextDecodeLayer0MlpAllReduceStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage {
    stage: QwenNextDecodeLayer0MlpResidualNormAllRankStage,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchResult {
    layer_idx: u32,
    qkv_stage: QwenResidentLayerRunnerDenseLayerQkvAllRankStage,
    qk_rope_stage: QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage,
    kv_append_stage: Option<QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage>,
    attention_stage: Option<QwenResidentLayerRunnerDenseLayerAttentionAllRankStage>,
    o_proj_stage: Option<QwenResidentLayerRunnerDenseLayerOProjAllReduceStage>,
    post_attn_norm_stage: Option<QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage>,
    mlp_allreduce_stage: QwenResidentLayerRunnerDenseLayerMlpAllReduceStage,
    next_input_norm_stage: Option<QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerStageSlots<'a> {
    qkv_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerQkvAllRankStage>,
    qk_rope_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage>,
    kv_append_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage>,
    attention_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerAttentionAllRankStage>,
    o_proj_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerOProjAllReduceStage>,
    post_attn_norm_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage>,
    mlp_allreduce_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerMlpAllReduceStage>,
    mlp_residual_norm_stage:
        &'a mut Option<QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'a> {
    qkv_stage: &'a mut Option<QwenNextDecodeLayer1QkvAllRankStage>,
    qk_rope_stage: &'a mut Option<QwenNextDecodeLayer1QkRopeAllRankStage>,
    kv_append_stage: &'a mut Option<QwenNextDecodeLayer1KvAppendAllRankStage>,
    attention_stage: &'a mut Option<QwenNextDecodeLayer1AttentionAllRankStage>,
    o_proj_stage: &'a mut Option<QwenNextDecodeLayer1OProjAllReduceStage>,
    post_attn_norm_stage: &'a mut Option<QwenNextDecodeLayer1PostAttnNormAllRankStage>,
    mlp_allreduce_stage: &'a mut Option<QwenNextDecodeLayer0MlpAllReduceStage>,
    mlp_residual_norm_stage: &'a mut Option<QwenNextDecodeLayer0MlpResidualNormAllRankStage>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerLegacyStageRefRow<'a> {
    layer_idx: u32,
    refs: QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'a>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'a> {
    rows: Vec<QwenResidentLayerRunnerDenseLayerLegacyStageRefRow<'a>>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'a> {
    stage_ref_window: QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'a>,
    apply_context_window: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow,
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_legacy_stage_refs_from_parts<'a>(
    qkv_stage: &'a mut Option<QwenNextDecodeLayer1QkvAllRankStage>,
    qk_rope_stage: &'a mut Option<QwenNextDecodeLayer1QkRopeAllRankStage>,
    kv_append_stage: &'a mut Option<QwenNextDecodeLayer1KvAppendAllRankStage>,
    attention_stage: &'a mut Option<QwenNextDecodeLayer1AttentionAllRankStage>,
    o_proj_stage: &'a mut Option<QwenNextDecodeLayer1OProjAllReduceStage>,
    post_attn_norm_stage: &'a mut Option<QwenNextDecodeLayer1PostAttnNormAllRankStage>,
    mlp_allreduce_stage: &'a mut Option<QwenNextDecodeLayer0MlpAllReduceStage>,
    mlp_residual_norm_stage: &'a mut Option<QwenNextDecodeLayer0MlpResidualNormAllRankStage>,
) -> QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'a> {
    QwenResidentLayerRunnerDenseLayerLegacyStageRefs {
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
        mlp_allreduce_stage,
        mlp_residual_norm_stage,
    }
}

#[inline]
pub(crate) fn qwen_resident_layer_runner_dense_layer_legacy_stage_ref_row<'a>(
    layer_idx: u32,
    refs: QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'a>,
) -> QwenResidentLayerRunnerDenseLayerLegacyStageRefRow<'a> {
    QwenResidentLayerRunnerDenseLayerLegacyStageRefRow { layer_idx, refs }
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_from_rows<'a>(
    rows: Vec<QwenResidentLayerRunnerDenseLayerLegacyStageRefRow<'a>>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'a>> {
    let row_count = rows.len();
    if row_count == 0 {
        anyhow::bail!("resident dense legacy stage-ref window requires at least one row");
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = rows[0].layer_idx;
    let mut last_layer_idx = rows[0].layer_idx;
    let mut previous_layer_idx = None;
    for (row_index, row) in rows.iter().enumerate() {
        if !seen_layers.insert(row.layer_idx) {
            anyhow::bail!(
                "resident dense legacy stage-ref window contains duplicate layer {}",
                row.layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if row.layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense legacy stage-ref window is not strictly ascending: layer {} after layer {} at row {}",
                    row.layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(row.layer_idx);
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
    }
    let contiguous_layer_window =
        last_layer_idx - first_layer_idx + 1 == u32::try_from(row_count).unwrap_or(u32::MAX);

    println!("  resident_layer_runner_dense_layer_legacy_stage_ref_window_stage:");
    println!("    source: resident_runner_dense_layer_legacy_stage_ref_window_from_rows");
    println!("    window_owner: resident_runner_module");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    window_layer_count: {row_count}");
    println!("    stage_ref_window_rows_derived_from_row_list_len: true");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    fixed_four_layer_stage_ref_window_constructor_removed: true");
    println!("    stage_ref_rows_strictly_ascending: true");
    println!("    stage_ref_rows_duplicate_free: true");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    semantic_stage_ref_window_constructed: true");
    println!("    main_loose_first_layer_and_four_ref_args_removed_from_apply_helper: true");
    println!("    main_repeated_stage_ref_row_literals_removed: true");
    println!("    host_side_stage_ref_window: true");
    println!("    device_resident_stage_ref_bundle: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow { rows })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_legacy_stage_refs_from_window<'a>(
    target_layer_idx: u32,
    stage_ref_window: QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'a>> {
    let window_layer_count = stage_ref_window.rows.len();
    let (first_layer_idx, last_layer_idx) =
        qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_bounds(&stage_ref_window)?;

    println!("  resident_layer_runner_dense_layer_legacy_stage_ref_window_select_stage:");
    println!("    source: resident_runner_dense_layer_legacy_stage_ref_window_select");
    println!("    window_owner: resident_runner_module");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    window_layer_count: {window_layer_count}");
    println!("    window_target_layer_idx: {target_layer_idx}");
    println!("    semantic_stage_ref_window_consumed: true");
    println!("    host_side_stage_ref_window: true");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    device_resident_stage_ref_bundle: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_legacy_stage_refs_from_indexed_rows(
        target_layer_idx,
        stage_ref_window.rows,
    )
}

pub(crate) fn qwen_resident_layer_runner_report_dense_layer_attention_descriptor_dispatch_from_stage_window<
    'a,
>(
    target_layer_idx: u32,
    stage_ref_window: QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'a>,
    dense_mlp_dispatch_promoted: bool,
    full_dense_dispatch_promoted: bool,
) -> anyhow::Result<()> {
    let legacy_stage_refs = qwen_resident_layer_runner_dense_layer_legacy_stage_refs_from_window(
        target_layer_idx,
        stage_ref_window,
    )?;

    println!(
        "  resident_layer_runner_layer{target_layer_idx}_attention_descriptor_dispatch_stage:"
    );
    println!(
        "    source: resident_runner_dense_layer_attention_descriptor_dispatch_from_stage_window"
    );
    println!("    target_layer_idx: {target_layer_idx}");
    println!("    stage_ref_window_consumed: true");
    println!("    selected_legacy_stage_refs_bound: true");
    println!("    stage_ref_window_owner: resident_runner_module");
    println!("    main_descriptor_dispatch_stage_ref_selector_removed: true");
    println!("    main_duplicate_stage_ref_window_literal_removed: true");
    println!("    layer{target_layer_idx}_attention_descriptor_dispatch_promoted: true");
    println!(
        "    layer{target_layer_idx}_dense_mlp_dispatch_promoted: {dense_mlp_dispatch_promoted}"
    );
    println!(
        "    layer{target_layer_idx}_full_dense_dispatch_promoted: {full_dense_dispatch_promoted}"
    );
    println!("    descriptor_dispatch_stage_report_owner: resident_runner_module");
    println!("    refactor_execution_path_changed: false");
    println!("    graph_capture_started: false");

    drop(legacy_stage_refs);
    Ok(())
}

fn qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_bounds(
    stage_ref_window: &QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'_>,
) -> anyhow::Result<(u32, u32)> {
    if stage_ref_window.rows.is_empty() {
        anyhow::bail!("resident dense legacy stage-ref window requires at least one row");
    }

    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    for row in &stage_ref_window.rows {
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
    }
    Ok((first_layer_idx, last_layer_idx))
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_legacy_stage_refs_from_indexed_rows<'a>(
    target_layer_idx: u32,
    rows: Vec<QwenResidentLayerRunnerDenseLayerLegacyStageRefRow<'a>>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'a>> {
    let indexed_stage_ref_row_count = rows.len();
    if indexed_stage_ref_row_count == 0 {
        anyhow::bail!("resident dense legacy stage-ref index requires at least one row");
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut selected = None;
    for row in rows {
        if !seen_layers.insert(row.layer_idx) {
            anyhow::bail!(
                "resident dense legacy stage-ref index contains duplicate layer {}",
                row.layer_idx
            );
        }
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
        if row.layer_idx == target_layer_idx {
            selected = Some(row.refs);
        }
    }
    let contiguous_layer_window = last_layer_idx - first_layer_idx + 1
        == u32::try_from(indexed_stage_ref_row_count).unwrap_or(u32::MAX);

    println!("  resident_layer_runner_dense_layer_legacy_stage_ref_index_stage:");
    println!("    source: resident_runner_dense_layer_legacy_stage_ref_index");
    println!("    index_owner: resident_runner_module");
    println!("    indexed_stage_ref_row_count: {indexed_stage_ref_row_count}");
    println!("    indexed_stage_ref_first_layer_idx: {first_layer_idx}");
    println!("    indexed_stage_ref_last_layer_idx: {last_layer_idx}");
    println!("    indexed_stage_ref_contiguous_layer_window: {contiguous_layer_window}");
    println!("    indexed_stage_ref_target_layer_idx: {target_layer_idx}");
    println!("    indexed_stage_ref_target_found: {}", selected.is_some());
    println!("    main_callsite_layer_suffix_selector_removed: true");
    println!("    host_side_indexed_stage_ref_table: true");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    device_resident_stage_ref_bundle: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    selected.ok_or_else(|| {
        anyhow::anyhow!(
            "resident dense legacy stage-ref index missing target layer {}",
            target_layer_idx
        )
    })
}

// TODO(layer-24-extension): delete this bridge when the main.rs harness emits
// resident semantic plan cursors instead of transient resident wrapper slots.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_stage_slots_from_refs<'a>(
    qkv_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerQkvAllRankStage>,
    qk_rope_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage>,
    kv_append_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage>,
    attention_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerAttentionAllRankStage>,
    o_proj_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerOProjAllReduceStage>,
    post_attn_norm_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage>,
    mlp_allreduce_stage: &'a mut Option<QwenResidentLayerRunnerDenseLayerMlpAllReduceStage>,
    mlp_residual_norm_stage: &'a mut Option<
        QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage,
    >,
) -> QwenResidentLayerRunnerDenseLayerStageSlots<'a> {
    QwenResidentLayerRunnerDenseLayerStageSlots {
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
        mlp_allreduce_stage,
        mlp_residual_norm_stage,
    }
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerLegacyApplyTask<'a> {
    result: Option<QwenResidentLayerRunnerDenseLayerDispatchResult>,
    hbm_state: &'a QwenResidentLayerRunnerDenseOutputSlotHbmState,
    apply_plan: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan,
    slots: QwenResidentLayerRunnerDenseLayerStageSlots<'a>,
}

pub(crate) enum QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry<'a> {
    Active(QwenResidentLayerRunnerDenseLayerLegacyApplyTask<'a>),
    PlannedNoop {
        layer_idx: u32,
        reason: &'static str,
    },
}

#[inline]
fn qwen_resident_layer_runner_dense_layer_legacy_apply_task_from_parts<'a>(
    result: Option<QwenResidentLayerRunnerDenseLayerDispatchResult>,
    hbm_state: &'a QwenResidentLayerRunnerDenseOutputSlotHbmState,
    apply_plan: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan,
    slots: QwenResidentLayerRunnerDenseLayerStageSlots<'a>,
) -> QwenResidentLayerRunnerDenseLayerLegacyApplyTask<'a> {
    QwenResidentLayerRunnerDenseLayerLegacyApplyTask {
        result,
        hbm_state,
        apply_plan,
        slots,
    }
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_apply_queue_from_slots_and_resolver<'a>(
    result: Option<QwenResidentLayerRunnerDenseLayerDispatchResult>,
    hbm_state: &'a QwenResidentLayerRunnerDenseOutputSlotHbmState,
    apply_plan: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan,
    slots: QwenResidentLayerRunnerDenseLayerStageSlots<'a>,
    resolver_target: QwenResidentLayerRunnerDenseOutputSlotResolverTarget,
) -> anyhow::Result<[QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry<'a>; 2]> {
    println!("  resident_layer_runner_dense_layer_legacy_boundary_builder_stage:");
    println!("    source: resident_runner_dense_layer_apply_queue_from_slots_and_resolver");
    println!("    boundary_owner: resident_runner_module");
    println!("    stage_slot_bundle_constructed_in_resident_runner: true");
    println!("    active_task_constructed_in_resident_runner: true");
    println!("    main_stage_slot_bundle_builder_removed: true");
    println!("    main_legacy_apply_task_builder_removed: true");
    println!("    public_stage_slot_builder_seam_removed: true");
    println!("    public_active_task_builder_seam_removed: true");
    println!("    public_apply_queue_signature_uses_numbered_stage_types: false");
    println!("    apply_queue_helper_name_contains_legacy_refs: false");
    println!("    opaque_semantic_stage_slots_boundary: true");
    println!("    legacy_numbered_stage_refs_cross_apply_queue_boundary: false");
    println!("    resident_stage_wrapper_slots_cross_apply_queue_boundary: true");
    println!("    active_layer_idx: {}", apply_plan.expected_layer_idx);
    println!(
        "    resolver_target_layer_idx: {}",
        resolver_target.layer_idx
    );
    println!(
        "    resolver_target_lookahead_layer_idx: {}",
        resolver_target.lookahead_layer_idx
    );
    println!("    queue_builder_heap_allocation: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let active_task = qwen_resident_layer_runner_dense_layer_legacy_apply_task_from_parts(
        result, hbm_state, apply_plan, slots,
    );
    qwen_resident_layer_runner_dense_layer_apply_queue_from_active_and_resolver(
        active_task,
        resolver_target,
    )
}

pub(crate) fn qwen_resident_layer_runner_apply_dense_layer_dispatch_result_to_legacy_stage_bundle(
    result: Option<QwenResidentLayerRunnerDenseLayerDispatchResult>,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    apply_context: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContext,
    legacy_stage_refs: QwenResidentLayerRunnerDenseLayerLegacyStageRefs<'_>,
) -> anyhow::Result<()> {
    let apply_plan = apply_context.apply_plan;
    let resolver_target = apply_context.resolver_target;
    println!("  resident_layer_runner_dense_layer_legacy_stage_apply_bridge_stage:");
    println!("    source: resident_runner_dense_layer_legacy_stage_apply_bridge");
    println!("    bridge_owner: resident_runner_module");
    println!("    result_layer_idx: {}", apply_plan.expected_layer_idx);
    println!(
        "    resolver_target_layer_idx: {}",
        resolver_target.layer_idx
    );
    println!(
        "    resolver_target_lookahead_layer_idx: {}",
        resolver_target.lookahead_layer_idx
    );
    println!("    legacy_stage_take_restore_owned_by_resident_runner: true");
    println!("    main_repeated_stage_apply_block_removed: true");
    println!("    semantic_stage_ref_bundle_boundary: true");
    println!("    helper_accepts_single_stage_ref_bundle: true");
    println!("    helper_signature_stage_ref_count: 1");
    println!("    output_slot_apply_context_boundary: true");
    println!("    apply_plan_and_resolver_target_bound: true");
    println!("    apply_context_source: {}", apply_context.source);
    println!("    legacy_numbered_stage_refs_cross_apply_helper_boundary: false");
    println!("    device_resident_stage_ref_bundle: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let mut resident_qkv_stage = legacy_stage_refs.qkv_stage.take().map(Into::into);
    let mut resident_qk_rope_stage = legacy_stage_refs.qk_rope_stage.take().map(Into::into);
    let mut resident_kv_append_stage = legacy_stage_refs.kv_append_stage.take().map(Into::into);
    let mut resident_attention_stage = legacy_stage_refs.attention_stage.take().map(Into::into);
    let mut resident_o_proj_stage = legacy_stage_refs.o_proj_stage.take().map(Into::into);
    let mut resident_post_attn_norm_stage = legacy_stage_refs
        .post_attn_norm_stage
        .take()
        .map(Into::into);
    let mut resident_mlp_allreduce_stage =
        legacy_stage_refs.mlp_allreduce_stage.take().map(Into::into);
    let mut resident_mlp_residual_norm_stage = legacy_stage_refs
        .mlp_residual_norm_stage
        .take()
        .map(Into::into);

    let resident_dense_stage_slots = qwen_resident_layer_runner_dense_layer_stage_slots_from_refs(
        &mut resident_qkv_stage,
        &mut resident_qk_rope_stage,
        &mut resident_kv_append_stage,
        &mut resident_attention_stage,
        &mut resident_o_proj_stage,
        &mut resident_post_attn_norm_stage,
        &mut resident_mlp_allreduce_stage,
        &mut resident_mlp_residual_norm_stage,
    );
    let resident_dense_apply_queue =
        qwen_resident_layer_runner_dense_layer_apply_queue_from_slots_and_resolver(
            result,
            hbm_state,
            apply_plan,
            resident_dense_stage_slots,
            resolver_target,
        )?;
    qwen_resident_layer_runner_apply_dense_layer_dispatch_result_tasks_to_legacy_slots(
        resident_dense_apply_queue,
    )?;

    *legacy_stage_refs.qkv_stage = resident_qkv_stage.take().map(Into::into);
    *legacy_stage_refs.qk_rope_stage = resident_qk_rope_stage.take().map(Into::into);
    *legacy_stage_refs.kv_append_stage = resident_kv_append_stage.take().map(Into::into);
    *legacy_stage_refs.attention_stage = resident_attention_stage.take().map(Into::into);
    *legacy_stage_refs.o_proj_stage = resident_o_proj_stage.take().map(Into::into);
    *legacy_stage_refs.post_attn_norm_stage = resident_post_attn_norm_stage.take().map(Into::into);
    *legacy_stage_refs.mlp_allreduce_stage = resident_mlp_allreduce_stage.take().map(Into::into);
    *legacy_stage_refs.mlp_residual_norm_stage =
        resident_mlp_residual_norm_stage.take().map(Into::into);

    Ok(())
}

pub(crate) fn qwen_resident_layer_runner_apply_dense_layer_dispatch_result_to_legacy_stage_window<
    'a,
>(
    result: Option<QwenResidentLayerRunnerDenseLayerDispatchResult>,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    target_layer_idx: u32,
    legacy_apply_window: QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'a>,
) -> anyhow::Result<()> {
    let QwenResidentLayerRunnerDenseLayerLegacyApplyWindow {
        stage_ref_window,
        apply_context_window,
    } = legacy_apply_window;
    let apply_context =
        qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_from_window(
            target_layer_idx,
            &apply_context_window,
        )?;
    let apply_plan = apply_context.apply_plan;
    let resolver_target = apply_context.resolver_target;
    if apply_plan.expected_layer_idx != target_layer_idx {
        anyhow::bail!(
            "resident dense legacy stage-window apply target {} does not match output-slot apply plan layer {}",
            target_layer_idx,
            apply_plan.expected_layer_idx
        );
    }
    if resolver_target.layer_idx != target_layer_idx {
        anyhow::bail!(
            "resident dense legacy stage-window apply target {} does not match resolver layer {}",
            target_layer_idx,
            resolver_target.layer_idx
        );
    }

    println!("  resident_layer_runner_dense_layer_legacy_stage_window_apply_target_guard_stage:");
    println!("    source: resident_runner_dense_layer_legacy_stage_window_apply_target_guard");
    println!("    guard_owner: resident_runner_module");
    println!("    target_layer_idx: {target_layer_idx}");
    println!(
        "    apply_plan_expected_layer_idx: {}",
        apply_plan.expected_layer_idx
    );
    println!(
        "    resolver_target_layer_idx: {}",
        resolver_target.layer_idx
    );
    println!(
        "    resolver_target_lookahead_layer_idx: {}",
        resolver_target.lookahead_layer_idx
    );
    println!("    target_matches_apply_plan: true");
    println!("    target_matches_resolver: true");
    println!("    apply_context_source: {}", apply_context.source);
    println!("    apply_plan_and_resolver_target_bound: true");
    println!("    apply_context_selected_from_window: true");
    println!("    legacy_apply_window_consumed: true");
    println!("    misrouted_stage_window_apply_fails_fast: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let (first_layer_idx, last_layer_idx) =
        qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_bounds(&stage_ref_window)?;
    let stage_ref_window_rows = stage_ref_window.rows.len();
    let apply_context_window_rows = apply_context_window.rows.len();

    println!("  resident_layer_runner_dense_layer_legacy_stage_window_apply_bridge_stage:");
    println!("    source: resident_runner_dense_layer_legacy_stage_window_apply_bridge");
    println!("    window_owner: resident_runner_module");
    println!("    target_layer_idx: {target_layer_idx}");
    println!("    first_layer_idx: {first_layer_idx}");
    println!("    last_layer_idx: {last_layer_idx}");
    println!("    stage_ref_window_rows: {stage_ref_window_rows}");
    println!("    apply_context_window_rows: {apply_context_window_rows}");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    apply_context_window_rows_dynamic_vec: true");
    println!("    stage_ref_window_argument: true");
    println!("    loose_first_layer_and_four_ref_apply_args_removed: true");
    println!("    output_slot_apply_context_argument: true");
    println!("    output_slot_apply_context_window_argument: true");
    println!("    legacy_apply_window_argument: true");
    println!("    loose_apply_plan_and_resolver_args_removed: true");
    println!("    loose_per_layer_apply_context_arg_removed: true");
    println!("    loose_stage_ref_and_apply_context_window_args_removed: true");
    println!("    main_explicit_stage_ref_selection_before_apply_removed: true");
    println!("    legacy_stage_refs_retained: true");
    println!("    helper_selects_stage_refs_and_applies_result: true");
    println!("    device_resident_stage_ref_bundle: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let legacy_stage_refs = qwen_resident_layer_runner_dense_layer_legacy_stage_refs_from_window(
        target_layer_idx,
        stage_ref_window,
    )?;
    qwen_resident_layer_runner_apply_dense_layer_dispatch_result_to_legacy_stage_bundle(
        result,
        hbm_state,
        apply_context,
        legacy_stage_refs,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_apply_queue_from_active_and_resolver<'a>(
    active_task: QwenResidentLayerRunnerDenseLayerLegacyApplyTask<'a>,
    resolver_target: QwenResidentLayerRunnerDenseOutputSlotResolverTarget,
) -> anyhow::Result<[QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry<'a>; 2]> {
    let active_layer_idx = active_task.apply_plan.expected_layer_idx;
    let lookahead_layer_idx = resolver_target.lookahead_layer_idx;
    let expected_lookahead_layer_idx = active_layer_idx + 1;
    if resolver_target.layer_idx != active_layer_idx {
        anyhow::bail!(
            "resident dense-layer apply queue builder expected active resolver layer {} but received {}",
            active_layer_idx,
            resolver_target.layer_idx
        );
    }
    if lookahead_layer_idx != expected_lookahead_layer_idx {
        anyhow::bail!(
            "resident dense-layer apply queue builder expected current+1 lookahead layer {} but received {}",
            expected_lookahead_layer_idx,
            lookahead_layer_idx
        );
    }

    println!("  resident_layer_runner_dense_layer_apply_queue_builder_stage:");
    println!("    source: resident_runner_dense_layer_apply_queue_from_active_and_resolver");
    println!("    boundary_owner: resident_runner_module");
    println!("    queue_constructed_from_resolver_target: true");
    println!("    queue_literal_in_main: false");
    println!("    active_row_source: legacy_active_task");
    println!("    planned_noop_row_source: resolver_target_lookahead_layer");
    println!("    apply_queue_builder_returns_fixed_array_len: 2");
    println!("    queue_builder_heap_allocation: false");
    println!("    active_layer_idx: {}", active_layer_idx);
    println!(
        "    resolver_target_layer_idx: {}",
        resolver_target.layer_idx
    );
    println!("    lookahead_noop_layer_idx: {}", lookahead_layer_idx);
    println!("    lookahead_validation_passed: true");
    println!("    terminal_layer_handling_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok([
        QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry::Active(active_task),
        QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry::PlannedNoop {
            layer_idx: lookahead_layer_idx,
            reason: "current_plus_one_lookahead_without_dense_result",
        },
    ])
}

// Temporary adapters while the main.rs harness still emits legacy numbered stage types.
// The resident finalizer API consumes the semantic resident wrappers and should keep
// these shims as the only numbered-stage ingress point.
impl From<QwenNextDecodeLayer1QkvAllRankStage>
    for QwenResidentLayerRunnerDenseLayerQkvAllRankStage
{
    fn from(stage: QwenNextDecodeLayer1QkvAllRankStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer1QkRopeAllRankStage>
    for QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage
{
    fn from(stage: QwenNextDecodeLayer1QkRopeAllRankStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer1KvAppendAllRankStage>
    for QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage
{
    fn from(stage: QwenNextDecodeLayer1KvAppendAllRankStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer1AttentionAllRankStage>
    for QwenResidentLayerRunnerDenseLayerAttentionAllRankStage
{
    fn from(stage: QwenNextDecodeLayer1AttentionAllRankStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer1OProjAllReduceStage>
    for QwenResidentLayerRunnerDenseLayerOProjAllReduceStage
{
    fn from(stage: QwenNextDecodeLayer1OProjAllReduceStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer1PostAttnNormAllRankStage>
    for QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage
{
    fn from(stage: QwenNextDecodeLayer1PostAttnNormAllRankStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer0MlpAllReduceStage>
    for QwenResidentLayerRunnerDenseLayerMlpAllReduceStage
{
    fn from(stage: QwenNextDecodeLayer0MlpAllReduceStage) -> Self {
        Self { stage }
    }
}

impl From<QwenNextDecodeLayer0MlpResidualNormAllRankStage>
    for QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage
{
    fn from(stage: QwenNextDecodeLayer0MlpResidualNormAllRankStage) -> Self {
        Self { stage }
    }
}

impl From<QwenResidentLayerRunnerDenseLayerQkvAllRankStage>
    for QwenNextDecodeLayer1QkvAllRankStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerQkvAllRankStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage>
    for QwenNextDecodeLayer1QkRopeAllRankStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage>
    for QwenNextDecodeLayer1KvAppendAllRankStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerAttentionAllRankStage>
    for QwenNextDecodeLayer1AttentionAllRankStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerAttentionAllRankStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerOProjAllReduceStage>
    for QwenNextDecodeLayer1OProjAllReduceStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerOProjAllReduceStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage>
    for QwenNextDecodeLayer1PostAttnNormAllRankStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerMlpAllReduceStage>
    for QwenNextDecodeLayer0MlpAllReduceStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerMlpAllReduceStage) -> Self {
        stage.stage
    }
}

impl From<QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage>
    for QwenNextDecodeLayer0MlpResidualNormAllRankStage
{
    fn from(stage: QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage) -> Self {
        stage.stage
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_result_from_stages<
    QkvStage,
    QkRopeStage,
    KvAppendStage,
    AttentionStage,
    OProjStage,
    PostAttnNormStage,
    MlpStage,
    NextInputNormStage,
>(
    layer_idx: u32,
    qkv_stage: QkvStage,
    qk_rope_stage: QkRopeStage,
    kv_append_stage: Option<KvAppendStage>,
    attention_stage: Option<AttentionStage>,
    o_proj_stage: Option<OProjStage>,
    post_attn_norm_stage: Option<PostAttnNormStage>,
    mlp_stage: MlpStage,
    next_input_norm_stage: Option<NextInputNormStage>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchResult>
where
    QkvStage: Into<QwenResidentLayerRunnerDenseLayerQkvAllRankStage>,
    QkRopeStage: Into<QwenResidentLayerRunnerDenseLayerQkRopeAllRankStage>,
    KvAppendStage: Into<QwenResidentLayerRunnerDenseLayerKvAppendAllRankStage>,
    AttentionStage: Into<QwenResidentLayerRunnerDenseLayerAttentionAllRankStage>,
    OProjStage: Into<QwenResidentLayerRunnerDenseLayerOProjAllReduceStage>,
    PostAttnNormStage: Into<QwenResidentLayerRunnerDenseLayerPostAttnNormAllRankStage>,
    MlpStage: Into<QwenResidentLayerRunnerDenseLayerMlpAllReduceStage>,
    NextInputNormStage: Into<QwenResidentLayerRunnerDenseLayerMlpResidualNormAllRankStage>,
{
    if layer_idx == u32::MAX {
        anyhow::bail!(
            "resident dense-layer dispatch result finalizer received invalid layer index"
        );
    }
    if post_attn_norm_stage.is_none() {
        anyhow::bail!(
            "resident dense-layer dispatch result finalizer requires post-attention norm before MLP result finalization for layer {}",
            layer_idx
        );
    }
    let qkv_stage = qkv_stage.into();
    let qk_rope_stage = qk_rope_stage.into();
    let kv_append_stage = kv_append_stage.map(|stage| stage.into());
    let attention_stage = attention_stage.map(|stage| stage.into());
    let o_proj_stage = o_proj_stage.map(|stage| stage.into());
    let post_attn_norm_stage = post_attn_norm_stage.map(|stage| stage.into());
    let mlp_stage = mlp_stage.into();
    let next_input_norm_stage = next_input_norm_stage.map(|stage| stage.into());

    println!("  resident_layer_runner_dense_layer_result_finalizer_stage:");
    println!("    source: resident_runner_dense_layer_dispatch_result_finalizer");
    println!("    result_finalizer_owner: resident_runner_module");
    println!("    result_finalizer_boundary: true");
    println!("    result_finalizer_layer_idx: {layer_idx}");
    println!("    result_finalizer_validates_layer_idx: true");
    println!("    result_finalizer_requires_post_attn_norm: true");
    println!("    qkv_stage_present: true");
    println!("    qk_rope_stage_present: true");
    println!("    kv_append_stage_present: {}", kv_append_stage.is_some());
    println!("    attention_stage_present: {}", attention_stage.is_some());
    println!("    o_proj_stage_present: {}", o_proj_stage.is_some());
    println!(
        "    post_attn_norm_stage_present: {}",
        post_attn_norm_stage.is_some()
    );
    println!("    mlp_allreduce_stage_present: true");
    println!(
        "    next_input_norm_stage_present: {}",
        next_input_norm_stage.is_some()
    );
    println!("    result_finalizer_signature_uses_resident_stage_wrappers: true");
    println!("    result_finalizer_legacy_numbered_stage_types_in_signature: false");
    println!("    result_fields_private_to_resident_runner: true");
    println!("    legacy_numbered_stage_adapters_temporary: true");
    println!("    manual_result_struct_literal_in_main: false");
    println!("    result_finalizer_behavior_changed: false");
    println!("    dispatch_scope_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchResult {
        layer_idx,
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
        mlp_allreduce_stage: mlp_stage,
        next_input_norm_stage,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_rows_from_plan_descriptors(
    descriptors: &[QwenResidentLayerRunnerPlanDescriptor],
) -> anyhow::Result<Vec<QwenResidentLayerRunnerDenseOutputSlotMetadataRow>> {
    if descriptors.is_empty() {
        anyhow::bail!("resident dense output slot table requires at least one plan descriptor");
    }

    let mut rows = Vec::with_capacity(descriptors.len());
    let mut seen_layers = std::collections::BTreeSet::new();
    let mut previous_layer = None;
    for descriptor in descriptors {
        if !seen_layers.insert(descriptor.layer_idx) {
            anyhow::bail!(
                "resident dense output slot table plan descriptors contain duplicate layer {}",
                descriptor.layer_idx
            );
        }
        if let Some(previous_layer) = previous_layer {
            if descriptor.layer_idx <= previous_layer {
                anyhow::bail!(
                    "resident dense output slot table plan descriptors are not strictly ascending: {} after {}",
                    descriptor.layer_idx,
                    previous_layer
                );
            }
        }
        previous_layer = Some(descriptor.layer_idx);
        rows.push(
            QwenResidentLayerRunnerDenseOutputSlotMetadataRow::legacy_host_bridge(
                descriptor.layer_idx,
            ),
        );
    }

    Ok(rows)
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_row_checksum(
    row_index: u32,
    row: QwenResidentLayerRunnerDenseOutputSlotMetadataRow,
) -> u64 {
    let words = [
        u64::from(row_index),
        u64::from(row.layer_idx),
        u64::from(row.slot_count),
        u64::from(row.role_mask),
        row.flags,
    ];
    let mut checksum = 0xcbf29ce484222325u64;
    for word in words {
        for byte in word.to_le_bytes() {
            checksum ^= byte as u64;
            checksum = checksum.wrapping_mul(0x100000001b3);
        }
    }
    checksum
}

fn qwen_resident_layer_runner_pack_dense_output_slot_descriptor(
    row: QwenResidentLayerRunnerDenseOutputSlotMetadataRow,
    row_idx: usize,
) -> anyhow::Result<[u64; QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_U64S]> {
    let row_idx_u32 = u32::try_from(row_idx)
        .map_err(|_| anyhow::anyhow!("resident dense output slot row index does not fit u32"))?;
    if row.role_mask != QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_ROLE_MASK_ALL {
        anyhow::bail!(
            "resident dense output slot table row for layer {} has role mask 0x{:x}, expected 0x{:x}",
            row.layer_idx,
            row.role_mask,
            QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_ROLE_MASK_ALL
        );
    }
    if row.slot_count != QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_COUNT {
        anyhow::bail!(
            "resident dense output slot table row for layer {} has slot count {}, expected {}",
            row.layer_idx,
            row.slot_count,
            QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_COUNT
        );
    }
    if row.flags & QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_FLAG_HOST_LEGACY_BRIDGE == 0 {
        anyhow::bail!(
            "resident dense output slot table row for layer {} is missing host legacy bridge flag",
            row.layer_idx
        );
    }
    let checksum = qwen_resident_layer_runner_dense_output_slot_row_checksum(row_idx_u32, row);
    Ok([
        QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_MAGIC,
        QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_VERSION | (u64::from(row_idx_u32) << 32),
        u64::from(row.layer_idx) | (u64::from(row.slot_count) << 32),
        u64::from(row.role_mask),
        0,
        0,
        row.flags,
        checksum,
    ])
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_descriptor_bytes(
    descriptors: &[[u64; QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_U64S]],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(
        descriptors.len()
            * QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_U64S
            * std::mem::size_of::<u64>(),
    );
    for descriptor in descriptors {
        for word in descriptor {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    bytes
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_descriptors_from_rows(
    rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
) -> anyhow::Result<Vec<[u64; QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_U64S]>> {
    if rows.is_empty() {
        anyhow::bail!("resident dense output slot table requires at least one row");
    }
    let mut seen_layers = std::collections::BTreeSet::new();
    for row in rows {
        if !seen_layers.insert(row.layer_idx) {
            anyhow::bail!(
                "resident dense output slot table has duplicate layer {}",
                row.layer_idx
            );
        }
    }
    rows.iter()
        .enumerate()
        .map(|(row_idx, row)| {
            qwen_resident_layer_runner_pack_dense_output_slot_descriptor(*row, row_idx)
        })
        .collect()
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_layer_index_rows(
    rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
) -> anyhow::Result<Vec<u32>> {
    if rows.is_empty() {
        anyhow::bail!("resident dense output slot layer index requires at least one row");
    }
    let max_layer = rows
        .iter()
        .map(|row| row.layer_idx)
        .max()
        .ok_or_else(|| anyhow::anyhow!("resident dense output slot layer index has no rows"))?;
    let entries = usize::try_from(max_layer)
        .map_err(|_| anyhow::anyhow!("resident dense output slot layer index max overflows"))?
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("resident dense output slot layer index size overflows"))?;
    let mut index =
        vec![QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_LAYER_INDEX_SENTINEL; entries];
    for (row_idx, row) in rows.iter().enumerate() {
        let row_idx_u32 = u32::try_from(row_idx).map_err(|_| {
            anyhow::anyhow!("resident dense output slot layer index row does not fit u32")
        })?;
        let layer_idx = usize::try_from(row.layer_idx).map_err(|_| {
            anyhow::anyhow!("resident dense output slot layer index layer does not fit usize")
        })?;
        if index[layer_idx] != QWEN_RESIDENT_LAYER_RUNNER_DENSE_OUTPUT_SLOT_LAYER_INDEX_SENTINEL {
            anyhow::bail!(
                "resident dense output slot layer index duplicate layer {}",
                row.layer_idx
            );
        }
        index[layer_idx] = row_idx_u32;
    }
    Ok(index)
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_layer_index_bytes(
    index: &[u32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(index.len() * std::mem::size_of::<u32>());
    for word in index {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

pub(crate) fn qwen_resident_layer_runner_upload_dense_output_slot_table(
    dev: &mut mcore::GpuDevice,
    rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
) -> anyhow::Result<(mcore::DeviceBuffer, usize, usize, u64, u64)> {
    let descriptors = qwen_resident_layer_runner_dense_output_slot_descriptors_from_rows(rows)?;
    let bytes = qwen_resident_layer_runner_dense_output_slot_descriptor_bytes(&descriptors);
    let checksum = qwen_resident_layer_runner_descriptor_checksum(&bytes);
    let mut buffer = dev.alloc_device(bytes.len())?;
    qwen_resident_layer_runner_validate_dispatch_buffer(
        "dense output slot descriptor upload",
        u32::MAX,
        "dense_output_slot_table",
        0,
        &mut buffer,
        bytes.len(),
    )?;
    unsafe {
        buffer.as_mut_slice_of::<u8>()[..bytes.len()].copy_from_slice(&bytes);
    }
    let va = buffer.va();
    Ok((buffer, descriptors.len(), bytes.len(), checksum, va))
}

pub(crate) fn qwen_resident_layer_runner_upload_dense_output_slot_layer_index_table(
    dev: &mut mcore::GpuDevice,
    rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
) -> anyhow::Result<(mcore::DeviceBuffer, usize, usize, u64, u64)> {
    let index = qwen_resident_layer_runner_dense_output_slot_layer_index_rows(rows)?;
    let bytes = qwen_resident_layer_runner_dense_output_slot_layer_index_bytes(&index);
    let checksum = qwen_resident_layer_runner_descriptor_checksum(&bytes);
    let mut buffer = dev.alloc_device(bytes.len())?;
    qwen_resident_layer_runner_validate_dispatch_buffer(
        "dense output slot layer index upload",
        u32::MAX,
        "dense_output_slot_layer_index",
        0,
        &mut buffer,
        bytes.len(),
    )?;
    unsafe {
        buffer.as_mut_slice_of::<u8>()[..bytes.len()].copy_from_slice(&bytes);
    }
    let va = buffer.va();
    Ok((buffer, index.len(), bytes.len(), checksum, va))
}

pub(crate) fn qwen_resident_layer_runner_upload_dense_output_slot_hbm_state(
    dev: &mut mcore::GpuDevice,
    rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotHbmState> {
    let (table_buffer, table_rows, table_bytes, table_checksum, table_va) =
        qwen_resident_layer_runner_upload_dense_output_slot_table(dev, rows)?;
    let (
        layer_index_buffer,
        layer_index_entries,
        layer_index_bytes,
        layer_index_checksum,
        layer_index_va,
    ) = qwen_resident_layer_runner_upload_dense_output_slot_layer_index_table(dev, rows)?;

    Ok(QwenResidentLayerRunnerDenseOutputSlotHbmState {
        table_buffer,
        table_rows,
        table_bytes,
        table_checksum,
        table_va,
        layer_index_buffer,
        layer_index_entries,
        layer_index_bytes,
        layer_index_checksum,
        layer_index_va,
        source: "resident_runner_dense_output_slot_hbm_state",
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_legacy_apply_plan_from_hbm_state(
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    target: QwenResidentLayerRunnerDenseOutputSlotResolverTarget,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan> {
    if target.row.layer_idx != target.layer_idx {
        anyhow::bail!(
            "resident dense output slot legacy apply target row layer {} does not match target layer {}",
            target.row.layer_idx,
            target.layer_idx
        );
    }
    let target_row_index = usize::try_from(target.row_index)
        .map_err(|_| anyhow::anyhow!("resident dense output slot target row overflows usize"))?;
    if target_row_index >= hbm_state.table_rows {
        anyhow::bail!(
            "resident dense output slot legacy apply target row {} is outside HBM table rows {}",
            target.row_index,
            hbm_state.table_rows
        );
    }
    let target_layer_index = usize::try_from(target.layer_idx)
        .map_err(|_| anyhow::anyhow!("resident dense output slot target layer overflows usize"))?;
    if target_layer_index >= hbm_state.layer_index_entries {
        anyhow::bail!(
            "resident dense output slot legacy apply target layer {} is outside HBM layer-index entries {}",
            target.layer_idx,
            hbm_state.layer_index_entries
        );
    }
    if hbm_state.table_buffer.va() != hbm_state.table_va {
        anyhow::bail!(
            "resident dense output slot legacy apply table VA changed from 0x{:x} to 0x{:x}",
            hbm_state.table_va,
            hbm_state.table_buffer.va()
        );
    }
    if hbm_state.layer_index_buffer.va() != hbm_state.layer_index_va {
        anyhow::bail!(
            "resident dense output slot legacy apply layer-index VA changed from 0x{:x} to 0x{:x}",
            hbm_state.layer_index_va,
            hbm_state.layer_index_buffer.va()
        );
    }

    Ok(QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan {
        expected_layer_idx: target.layer_idx,
        expected_row_index: target.row_index,
        table_rows: hbm_state.table_rows,
        table_va: hbm_state.table_va,
        table_checksum: hbm_state.table_checksum,
        layer_index_entries: hbm_state.layer_index_entries,
        layer_index_va: hbm_state.layer_index_va,
        layer_index_checksum: hbm_state.layer_index_checksum,
        derived_without_hbm_readback: true,
        source: "resident_runner_dense_output_slot_hbm_state_legacy_apply_plan",
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_apply_time_guard_from_hbm_state(
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    apply_plan: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotApplyTimeGuard> {
    let table_va_stable = hbm_state.table_va == apply_plan.table_va
        && hbm_state.table_buffer.va() == apply_plan.table_va;
    let layer_index_va_stable = hbm_state.layer_index_va == apply_plan.layer_index_va
        && hbm_state.layer_index_buffer.va() == apply_plan.layer_index_va;
    let table_rows_match = hbm_state.table_rows == apply_plan.table_rows;
    let layer_index_entries_match = hbm_state.layer_index_entries == apply_plan.layer_index_entries;
    let table_checksum_match = hbm_state.table_checksum == apply_plan.table_checksum;
    let layer_index_checksum_match =
        hbm_state.layer_index_checksum == apply_plan.layer_index_checksum;

    if !table_va_stable {
        anyhow::bail!(
            "resident dense output slot apply-time table VA changed: state_exported=0x{:x} retained=0x{:x} plan=0x{:x}",
            hbm_state.table_va,
            hbm_state.table_buffer.va(),
            apply_plan.table_va
        );
    }
    if !layer_index_va_stable {
        anyhow::bail!(
            "resident dense output slot apply-time layer-index VA changed: state_exported=0x{:x} retained=0x{:x} plan=0x{:x}",
            hbm_state.layer_index_va,
            hbm_state.layer_index_buffer.va(),
            apply_plan.layer_index_va
        );
    }
    if !table_rows_match {
        anyhow::bail!(
            "resident dense output slot apply-time table rows changed: state={} plan={}",
            hbm_state.table_rows,
            apply_plan.table_rows
        );
    }
    if !layer_index_entries_match {
        anyhow::bail!(
            "resident dense output slot apply-time layer-index entries changed: state={} plan={}",
            hbm_state.layer_index_entries,
            apply_plan.layer_index_entries
        );
    }
    if !table_checksum_match {
        anyhow::bail!(
            "resident dense output slot apply-time table checksum changed: state=0x{:016x} plan=0x{:016x}",
            hbm_state.table_checksum,
            apply_plan.table_checksum
        );
    }
    if !layer_index_checksum_match {
        anyhow::bail!(
            "resident dense output slot apply-time layer-index checksum changed: state=0x{:016x} plan=0x{:016x}",
            hbm_state.layer_index_checksum,
            apply_plan.layer_index_checksum
        );
    }

    Ok(QwenResidentLayerRunnerDenseOutputSlotApplyTimeGuard {
        table_va_stable,
        layer_index_va_stable,
        table_rows_match,
        layer_index_entries_match,
        table_checksum_match,
        layer_index_checksum_match,
        source: "resident_runner_dense_output_slot_apply_time_hbm_state_guard",
    })
}

pub(crate) fn qwen_resident_layer_runner_apply_dense_layer_dispatch_result_to_legacy_slots(
    result: QwenResidentLayerRunnerDenseLayerDispatchResult,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    apply_plan: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan,
    slots: QwenResidentLayerRunnerDenseLayerStageSlots<'_>,
) -> anyhow::Result<()> {
    let apply_time_guard =
        qwen_resident_layer_runner_dense_output_slot_apply_time_guard_from_hbm_state(
            hbm_state, apply_plan,
        )?;
    if result.layer_idx != apply_plan.expected_layer_idx {
        anyhow::bail!(
            "resident dense-layer result slot applier expected layer {} but received layer {}",
            apply_plan.expected_layer_idx,
            result.layer_idx
        );
    }

    println!("  resident_layer_runner_dense_layer_result_slot_applier_stage:");
    println!("    source: resident_runner_dense_layer_dispatch_result_slot_applier");
    println!("    boundary_owner: resident_runner_module");
    println!("    result_slot_applier_boundary: true");
    println!("    result_slot_applier_layer_idx: {}", result.layer_idx);
    println!("    apply_plan_source: {}", apply_plan.source);
    println!("    expected_layer_idx: {}", apply_plan.expected_layer_idx);
    println!("    expected_row_index: {}", apply_plan.expected_row_index);
    println!("    expected_layer_source: resident_dense_output_slot_legacy_apply_plan");
    println!("    legacy_stage_slot_builder_source: resident_runner_dense_layer_legacy_stage_slots_from_refs");
    println!("    legacy_stage_slot_builder_owned_by_resident_runner_module: true");
    println!("    legacy_stage_slot_struct_literal_in_main: false");
    println!("    legacy_stage_slot_fields_private_to_resident_runner: true");
    println!("    apply_plan_derived_from_hbm_state: true");
    println!(
        "    apply_plan_derived_without_hbm_readback: {}",
        apply_plan.derived_without_hbm_readback
    );
    println!("    apply_plan_table_rows: {}", apply_plan.table_rows);
    println!(
        "    apply_plan_table_device_va: 0x{:x}",
        apply_plan.table_va
    );
    println!(
        "    apply_plan_table_checksum_fnv1a64: 0x{:016x}",
        apply_plan.table_checksum
    );
    println!(
        "    apply_plan_layer_index_entries: {}",
        apply_plan.layer_index_entries
    );
    println!(
        "    apply_plan_layer_index_device_va: 0x{:x}",
        apply_plan.layer_index_va
    );
    println!(
        "    apply_plan_layer_index_checksum_fnv1a64: 0x{:016x}",
        apply_plan.layer_index_checksum
    );
    println!("    apply_time_guard_source: {}", apply_time_guard.source);
    println!(
        "    apply_time_table_va_stable: {}",
        apply_time_guard.table_va_stable
    );
    println!(
        "    apply_time_layer_index_va_stable: {}",
        apply_time_guard.layer_index_va_stable
    );
    println!(
        "    apply_time_table_rows_match: {}",
        apply_time_guard.table_rows_match
    );
    println!(
        "    apply_time_layer_index_entries_match: {}",
        apply_time_guard.layer_index_entries_match
    );
    println!(
        "    apply_time_table_checksum_match: {}",
        apply_time_guard.table_checksum_match
    );
    println!(
        "    apply_time_layer_index_checksum_match: {}",
        apply_time_guard.layer_index_checksum_match
    );
    println!("    apply_time_hbm_readback: false");
    println!("    apply_time_device_sync: false");
    println!("    legacy_stage_slot_count: 8");
    println!("    hardcoded_expected_layer_removed: true");
    println!("    dense_result_application_inlined_at_layer19_callsite: false");
    println!("    layer_specific_return_tuple_removed: true");
    println!("    legacy_callsite_stage_unwrap: false");
    println!("    dispatch_scope_changed: false");
    println!("    hip_graph_capture_started: false");

    *slots.qkv_stage = Some(result.qkv_stage);
    *slots.qk_rope_stage = Some(result.qk_rope_stage);
    *slots.kv_append_stage = result.kv_append_stage;
    *slots.attention_stage = result.attention_stage;
    *slots.o_proj_stage = result.o_proj_stage;
    *slots.post_attn_norm_stage = result.post_attn_norm_stage;
    *slots.mlp_allreduce_stage = Some(result.mlp_allreduce_stage);
    *slots.mlp_residual_norm_stage = result.next_input_norm_stage;

    Ok(())
}

pub(crate) fn qwen_resident_layer_runner_apply_dense_layer_dispatch_result_tasks_to_legacy_slots<
    'a,
    const TASKS: usize,
>(
    tasks: [QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry<'a>; TASKS],
) -> anyhow::Result<()> {
    println!("  resident_layer_runner_dense_layer_optional_apply_queue_stage:");
    println!("    source: resident_runner_dense_layer_dispatch_result_task_queue");
    println!("    boundary_owner: resident_runner_module");
    println!("    task_queue_topology: static_multi_row");
    println!("    optional_apply_decision_owned_by_resident_runner_module: true");
    println!("    optional_apply_loop_owned_by_resident_runner_module: true");
    println!("    main_optional_apply_branch_removed: true");
    println!("    task_queue_container: fixed_array");
    println!("    task_queue_heap_allocation: false");
    println!("    task_count: {}", TASKS);
    println!("    current_task_count_is_transitional_single_layer: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let mut applied_tasks = 0usize;
    let mut skipped_tasks = 0usize;
    let mut active_task_rows = 0usize;
    let mut planned_noop_rows = 0usize;
    let mut lookahead_noop_layer_idx = 0u32;
    let mut lookahead_noop_reason = "none";
    for task in tasks {
        match task {
            QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry::Active(task) => {
                active_task_rows += 1;
                if let Some(result) = task.result {
                    applied_tasks += 1;
                    qwen_resident_layer_runner_apply_dense_layer_dispatch_result_to_legacy_slots(
                        result,
                        task.hbm_state,
                        task.apply_plan,
                        task.slots,
                    )?;
                } else {
                    skipped_tasks += 1;
                }
            }
            QwenResidentLayerRunnerDenseLayerLegacyApplyQueueEntry::PlannedNoop {
                layer_idx,
                reason,
            } => {
                planned_noop_rows += 1;
                skipped_tasks += 1;
                lookahead_noop_layer_idx = layer_idx;
                lookahead_noop_reason = reason;
            }
        }
    }

    println!("  resident_layer_runner_dense_layer_optional_apply_queue_summary_stage:");
    println!("    source: resident_runner_dense_layer_dispatch_result_task_queue_summary");
    println!("    boundary_owner: resident_runner_module");
    println!("    task_queue_topology: static_multi_row");
    println!("    task_count: {}", TASKS);
    println!("    active_task_rows: {}", active_task_rows);
    println!("    planned_noop_rows: {}", planned_noop_rows);
    println!("    applied_tasks: {}", applied_tasks);
    println!("    skipped_tasks: {}", skipped_tasks);
    println!(
        "    queue_rows_include_current_plus_one_lookahead_noop: {}",
        planned_noop_rows > 0
    );
    println!("    lookahead_noop_layer_idx: {}", lookahead_noop_layer_idx);
    println!("    lookahead_noop_reason: {}", lookahead_noop_reason);
    println!("    no_op_decision_owned_by_resident_runner_module: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(())
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_window_descriptor(
    dispatch_layer_idx: u32,
    lookahead_layer_idx: u32,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchWindow> {
    if lookahead_layer_idx <= dispatch_layer_idx {
        anyhow::bail!(
            "resident dense-layer dispatch window lookahead layer {} must be after dispatch layer {}",
            lookahead_layer_idx,
            dispatch_layer_idx
        );
    }
    let lookahead_distance = lookahead_layer_idx - dispatch_layer_idx;
    if lookahead_distance != 1 {
        anyhow::bail!(
            "resident dense-layer dispatch window only supports current+1 lookahead, got distance {}",
            lookahead_distance
        );
    }
    let handoff_contract = QwenResidentLayerRunnerDenseLayerHandoffContract::next_input_norm(
        dispatch_layer_idx,
        lookahead_layer_idx,
    );

    println!("  resident_layer_runner_dense_layer_dispatch_window_descriptor_stage:");
    println!("    source: resident_runner_dense_layer_dispatch_window_descriptor");
    println!("    dispatch_window_descriptor: true");
    println!("    window_size: 2");
    println!("    dispatch_layer_idx: {dispatch_layer_idx}");
    println!("    lookahead_layer_idx: {lookahead_layer_idx}");
    println!("    lookahead_distance: {lookahead_distance}");
    println!("    current_plus_one_lookahead: true");
    println!("    lookup_indices_owned_by_descriptor: true");
    println!("    descriptor_owns_handoff_contract: true");
    println!("    descriptor_window_owned_by_resident_runner_module: true");
    println!("    handoff_contract_kind: {}", handoff_contract.label());
    println!(
        "    handoff_contract_source_layer_idx: {}",
        handoff_contract.source_layer_idx
    );
    println!(
        "    handoff_contract_target_layer_idx: {}",
        handoff_contract.target_layer_idx
    );
    println!("    dispatch_scope_changed: false");
    println!("    topology_routing_started: false");
    println!("    graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchWindow {
        dispatch_layer_idx,
        lookahead_layer_idx,
        handoff_contract,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_resolver_target_from_dispatch_window(
    rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotResolverTarget> {
    let row_index = rows
        .iter()
        .position(|row| row.layer_idx == dispatch_window.dispatch_layer_idx)
        .map(|row_idx| {
            u32::try_from(row_idx).map_err(|_| {
                anyhow::anyhow!("resident dense output slot row index does not fit u32")
            })
        })
        .transpose()?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense output slot resolver could not find row for dispatch layer {}",
                dispatch_window.dispatch_layer_idx
            )
        })?;
    let row = rows.get(row_index as usize).copied().ok_or_else(|| {
        anyhow::anyhow!(
            "resident dense output slot resolver row index {} is outside rows {}",
            row_index,
            rows.len()
        )
    })?;
    if row.layer_idx != dispatch_window.dispatch_layer_idx {
        anyhow::bail!(
            "resident dense output slot resolver target row layer {} does not match dispatch window layer {}",
            row.layer_idx,
            dispatch_window.dispatch_layer_idx
        );
    }
    Ok(QwenResidentLayerRunnerDenseOutputSlotResolverTarget {
        layer_idx: dispatch_window.dispatch_layer_idx,
        lookahead_layer_idx: dispatch_window.lookahead_layer_idx,
        row_index,
        row,
        source: "resident_runner_dense_layer_dispatch_window_descriptor",
    })
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourcePayload<'a> {
    pub(crate) qkv: Option<&'a QwenQkvProjStagePlan>,
    pub(crate) qk: Option<&'a QwenQkNormRopeStagePlan>,
    pub(crate) cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    pub(crate) attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    pub(crate) o_proj_plan: Option<&'a QwenOProjStagePlan>,
    pub(crate) post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    pub(crate) peer_plan: Option<&'a QwenTpOProjPeerStagePlan>,
    pub(crate) mlp: Option<&'a QwenMlpStagePlan>,
    pub(crate) peer_mlp: Option<&'a QwenTpMlpPeerStagePlan>,
    pub(crate) next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourcePayloadWindow<'a> {
    pub(crate) dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    pub(crate) dispatch_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
    pub(crate) lookahead_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_payload_window_from_dispatch_descriptor<
    'a,
>(
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    dispatch_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
    lookahead_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourcePayloadWindow<'a>> {
    let dispatch_payload_required_resources_present = dispatch_payload.qkv.is_some()
        && dispatch_payload.qk.is_some()
        && dispatch_payload.peer_plan.is_some()
        && dispatch_payload.mlp.is_some()
        && dispatch_payload.peer_mlp.is_some();
    let lookahead_payload_required_resources_present = lookahead_payload.qkv.is_some()
        && lookahead_payload.qk.is_some()
        && lookahead_payload.peer_plan.is_some()
        && lookahead_payload.mlp.is_some()
        && lookahead_payload.peer_mlp.is_some();
    let handoff_contract = dispatch_window.handoff_contract;
    let dispatch_next_input_norm_present = dispatch_payload.next_input_norm.is_some();
    let dispatch_next_input_norm_handoff_present =
        handoff_contract.dispatch_payload_satisfies(dispatch_next_input_norm_present);

    println!("  resident_layer_runner_dense_layer_descriptor_payload_window_stage:");
    println!("    source: resident_runner_dense_dispatch_window_payload_window_builder");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    descriptor_owns_payload_window: true");
    println!("    payload_window_owned_by_resident_runner_module: true");
    println!("    payload_window_size: 2");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    current_plus_one_lookahead: true");
    println!("    dispatch_payload_bound: true");
    println!("    lookahead_payload_bound: true");
    println!(
        "    dispatch_payload_required_resources_present: {}",
        dispatch_payload_required_resources_present
    );
    println!(
        "    lookahead_payload_required_resources_present: {}",
        lookahead_payload_required_resources_present
    );
    println!("    descriptor_payload_preflight_enforced: true");
    println!(
        "    dispatch_payload_preflight_passed: {}",
        dispatch_payload_required_resources_present
    );
    println!(
        "    lookahead_payload_preflight_passed: {}",
        lookahead_payload_required_resources_present
    );
    println!("    descriptor_handoff_preflight_enforced: true");
    println!("    descriptor_handoff_contract_consumed: true");
    println!("    handoff_contract_kind: {}", handoff_contract.label());
    println!(
        "    handoff_contract_source_layer_idx: {}",
        handoff_contract.source_layer_idx
    );
    println!(
        "    handoff_contract_target_layer_idx: {}",
        handoff_contract.target_layer_idx
    );
    println!(
        "    dispatch_next_input_norm_handoff_present: {}",
        dispatch_next_input_norm_handoff_present
    );
    println!(
        "    dispatch_to_lookahead_input_norm_handoff_preflight_passed: {}",
        dispatch_next_input_norm_handoff_present
    );
    println!("    callsite_direct_row_payload_binding_replaced: true");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    if !dispatch_payload_required_resources_present {
        anyhow::bail!(
            "resident dense-layer descriptor payload window missing required dispatch resources for layer {}",
            dispatch_window.dispatch_layer_idx
        );
    }
    if !lookahead_payload_required_resources_present {
        anyhow::bail!(
            "resident dense-layer descriptor payload window missing required lookahead resources for layer {}",
            dispatch_window.lookahead_layer_idx
        );
    }
    if !dispatch_next_input_norm_handoff_present {
        anyhow::bail!(
            "resident dense-layer descriptor payload window missing next-input-norm handoff from dispatch layer {} to lookahead layer {}",
            dispatch_window.dispatch_layer_idx,
            dispatch_window.lookahead_layer_idx
        );
    }

    Ok(QwenResidentLayerRunnerDenseLayerResourcePayloadWindow {
        dispatch_window,
        dispatch_payload,
        lookahead_payload,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_descriptor<
    'a,
>(
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    dispatch_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
    lookahead_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, 2>> {
    println!("  resident_layer_runner_dense_layer_resource_registry_stage:");
    println!("    source: resident_runner_dense_dispatch_descriptor_resource_registry");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    resource_registry_owned_by_resident_runner_module: true");
    println!("    payload_window_construction_owned_by_resident_runner_module: true");
    println!("    slot_table_construction_owned_by_resident_runner_module: true");
    println!("    main_payload_window_construction_removed: true");
    println!("    main_slot_table_construction_wrapper_removed: true");
    println!("    payload_row_binding_owner: upstream_registry_entrypoint");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    registry_complete: {registry_complete}");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    let payload_window =
        qwen_resident_layer_runner_dense_layer_resource_payload_window_from_dispatch_descriptor(
            dispatch_window,
            dispatch_payload,
            lookahead_payload,
        )?;
    qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_window_payload_window(
        payload_window,
        registry_complete,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_descriptor_resources<
    'a,
>(
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    dispatch_qkv: Option<&'a QwenQkvProjStagePlan>,
    dispatch_qk: Option<&'a QwenQkNormRopeStagePlan>,
    dispatch_cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    dispatch_attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    dispatch_o_proj_plan: Option<&'a QwenOProjStagePlan>,
    dispatch_post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    dispatch_peer_plan: Option<&'a QwenTpOProjPeerStagePlan>,
    dispatch_mlp: Option<&'a QwenMlpStagePlan>,
    dispatch_peer_mlp: Option<&'a QwenTpMlpPeerStagePlan>,
    dispatch_next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
    lookahead_qkv: Option<&'a QwenQkvProjStagePlan>,
    lookahead_qk: Option<&'a QwenQkNormRopeStagePlan>,
    lookahead_cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    lookahead_attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    lookahead_o_proj_plan: Option<&'a QwenOProjStagePlan>,
    lookahead_post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    lookahead_peer_plan: Option<&'a QwenTpOProjPeerStagePlan>,
    lookahead_mlp: Option<&'a QwenMlpStagePlan>,
    lookahead_peer_mlp: Option<&'a QwenTpMlpPeerStagePlan>,
    lookahead_next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, 2>> {
    println!("  resident_layer_runner_dense_layer_resource_row_binding_stage:");
    println!("    source: resident_runner_dense_dispatch_descriptor_resource_row_binding");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    resource_row_binding_owned_by_resident_runner_module: true");
    println!("    main_payload_row_struct_literals_removed: true");
    println!("    dispatch_payload_row_constructed_in_resident_runner: true");
    println!("    lookahead_payload_row_constructed_in_resident_runner: true");
    println!("    row_binding_table_len: 2");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    staged_device_table_copy_started: false");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_descriptor(
        dispatch_window,
        QwenResidentLayerRunnerDenseLayerResourcePayload {
            qkv: dispatch_qkv,
            qk: dispatch_qk,
            cache_plan: dispatch_cache_plan,
            attn_plan: dispatch_attn_plan,
            o_proj_plan: dispatch_o_proj_plan,
            post_attn_norm: dispatch_post_attn_norm,
            peer_plan: dispatch_peer_plan,
            mlp: dispatch_mlp,
            peer_mlp: dispatch_peer_mlp,
            next_input_norm: dispatch_next_input_norm,
        },
        QwenResidentLayerRunnerDenseLayerResourcePayload {
            qkv: lookahead_qkv,
            qk: lookahead_qk,
            cache_plan: lookahead_cache_plan,
            attn_plan: lookahead_attn_plan,
            o_proj_plan: lookahead_o_proj_plan,
            post_attn_norm: lookahead_post_attn_norm,
            peer_plan: lookahead_peer_plan,
            mlp: lookahead_mlp,
            peer_mlp: lookahead_peer_mlp,
            next_input_norm: lookahead_next_input_norm,
        },
        registry_complete,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceSlots<'a> {
    pub(crate) layer_idx: u32,
    pub(crate) qkv: Option<&'a QwenQkvProjStagePlan>,
    pub(crate) qk: Option<&'a QwenQkNormRopeStagePlan>,
    pub(crate) cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    pub(crate) attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    pub(crate) o_proj_plan: Option<&'a QwenOProjStagePlan>,
    pub(crate) post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    pub(crate) peer_plan: Option<&'a QwenTpOProjPeerStagePlan>,
    pub(crate) mlp: Option<&'a QwenMlpStagePlan>,
    pub(crate) peer_mlp: Option<&'a QwenTpMlpPeerStagePlan>,
    pub(crate) next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, const N: usize> {
    pub(crate) slots: [QwenResidentLayerRunnerDenseLayerResourceSlots<'a>; N],
    pub(crate) first_layer_idx: u32,
    pub(crate) last_layer_idx: u32,
    pub(crate) registry_complete: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceSlotTableRow<'a, const TABLE_ROWS: usize>
{
    dispatch_layer_idx: u32,
    slot_table: QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, TABLE_ROWS>,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceSlotTableWindow<
    'a,
    const TABLE_ROWS: usize,
> {
    rows: Vec<QwenResidentLayerRunnerDenseLayerResourceSlotTableRow<'a, TABLE_ROWS>>,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchContext<'a, const TABLE_ROWS: usize> {
    pub(crate) dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    pub(crate) resource_slot_table:
        QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, TABLE_ROWS>,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
    'a,
    const TABLE_ROWS: usize,
> {
    dispatch_windows: Vec<QwenResidentLayerRunnerDenseLayerDispatchWindow>,
    resource_slot_table_window:
        QwenResidentLayerRunnerDenseLayerResourceSlotTableWindow<'a, TABLE_ROWS>,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerContextWindows<'a> {
    pub(crate) dispatch_context_window:
        QwenResidentLayerRunnerDenseLayerDispatchContextWindow<'a, 2>,
    pub(crate) legacy_apply_context_window:
        QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceCatalog<'a> {
    pub(crate) rows: Vec<QwenResidentLayerRunnerDenseLayerResourceSlots<'a>>,
    pub(crate) first_layer_idx: u32,
    pub(crate) last_layer_idx: u32,
    pub(crate) registry_complete: bool,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceRegistry<'a> {
    pub(crate) rows: Vec<QwenResidentLayerRunnerDenseLayerResourceSlots<'a>>,
    pub(crate) first_layer_idx: u32,
    pub(crate) last_layer_idx: u32,
    pub(crate) registry_complete: bool,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerPreflightState<'a> {
    pub(crate) dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    pub(crate) resource_catalog: QwenResidentLayerRunnerDenseLayerResourceCatalog<'a>,
    pub(crate) output_slot_resolver_target: QwenResidentLayerRunnerDenseOutputSlotResolverTarget,
}

#[derive(Clone)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a> {
    rows: Vec<QwenResidentLayerRunnerDenseLayerPreflightState<'a>>,
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_registry_from_rows<'a>(
    rows: Vec<QwenResidentLayerRunnerDenseLayerResourceSlots<'a>>,
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceRegistry<'a>> {
    let resource_rows = rows.as_slice();
    if resource_rows.is_empty() {
        anyhow::bail!("resident runner dense resource registry has no rows");
    }

    let mut first_layer_idx = resource_rows[0].layer_idx;
    let mut last_layer_idx = resource_rows[0].layer_idx;
    for (row_index, row) in resource_rows.iter().enumerate() {
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
        if resource_rows[..row_index]
            .iter()
            .any(|prior_row| prior_row.layer_idx == row.layer_idx)
        {
            anyhow::bail!(
                "duplicate dense-layer resource registry row for layer {}",
                row.layer_idx
            );
        }
    }
    let contiguous_layer_window =
        last_layer_idx - first_layer_idx + 1 == resource_rows.len() as u32;

    println!("  resident_layer_runner_dense_layer_resource_registry_builder_stage:");
    println!("    source: resident_runner_dense_resource_registry_from_rows");
    println!("    resource_registry_owned_by_resident_runner_module: true");
    println!("    resource_registry_fixed_array: false");
    println!("    resource_registry_heap_allocation: true");
    println!("    resource_registry_row_count: {}", resource_rows.len());
    println!("    resource_registry_first_layer_idx: {first_layer_idx}");
    println!("    resource_registry_last_layer_idx: {last_layer_idx}");
    println!("    resource_registry_contiguous_layer_window: {contiguous_layer_window}");
    println!("    resource_registry_rows_dynamic_vec: true");
    println!("    fixed_resource_registry_row_count_removed: true");
    println!("    resource_registry_duplicate_layer_indices: false");
    println!("    resource_registry_complete: {registry_complete}");
    println!("    staged_device_table_copy_started: false");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerResourceRegistry {
        rows,
        first_layer_idx,
        last_layer_idx,
        registry_complete,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_preflight_dispatch_layer_indices_from_resource_registry(
    registry: &QwenResidentLayerRunnerDenseLayerResourceRegistry<'_>,
) -> anyhow::Result<Vec<u32>> {
    let registry_rows = registry.rows.len();
    if registry_rows < 2 {
        anyhow::bail!(
            "resident dense preflight dispatch layer derivation requires at least two resource registry rows"
        );
    }

    let mut dispatch_layer_indices = Vec::with_capacity(registry_rows - 1);
    for (row_index, pair) in registry.rows.windows(2).enumerate() {
        let dispatch_layer_idx = pair[0].layer_idx;
        let lookahead_layer_idx = pair[1].layer_idx;
        let expected_lookahead_layer_idx = dispatch_layer_idx.checked_add(1).ok_or_else(|| {
            anyhow::anyhow!("resident dense preflight dispatch layer index overflow")
        })?;
        if lookahead_layer_idx != expected_lookahead_layer_idx {
            anyhow::bail!(
                "resident dense preflight dispatch layer derivation requires adjacent current-plus-one registry rows, got {} then {} at row {}",
                dispatch_layer_idx,
                lookahead_layer_idx,
                row_index
            );
        }
        dispatch_layer_indices.push(dispatch_layer_idx);
    }

    let dispatch_layer_rows = dispatch_layer_indices.len();
    let dispatch_first_layer_idx = dispatch_layer_indices[0];
    let dispatch_last_layer_idx = dispatch_layer_indices[dispatch_layer_rows - 1];
    let lookahead_first_layer_idx = registry.rows[1].layer_idx;
    let lookahead_last_layer_idx = registry.rows[registry_rows - 1].layer_idx;
    let contiguous_dispatch_window = dispatch_last_layer_idx - dispatch_first_layer_idx + 1
        == u32::try_from(dispatch_layer_rows).unwrap_or(u32::MAX);

    println!("  resident_layer_runner_dense_layer_preflight_dispatch_layer_indices_stage:");
    println!(
        "    source: resident_runner_dense_preflight_dispatch_layer_indices_from_resource_registry"
    );
    println!("    preflight_dispatch_layer_indices_owner: resident_runner_module");
    println!("    registry_row_count: {registry_rows}");
    println!("    dispatch_layer_rows: {dispatch_layer_rows}");
    println!("    dispatch_first_layer_idx: {dispatch_first_layer_idx}");
    println!("    dispatch_last_layer_idx: {dispatch_last_layer_idx}");
    println!("    lookahead_first_layer_idx: {lookahead_first_layer_idx}");
    println!("    lookahead_last_layer_idx: {lookahead_last_layer_idx}");
    println!("    contiguous_dispatch_window: {contiguous_dispatch_window}");
    println!("    current_plus_one_lookahead_pairs: true");
    println!("    dispatch_layer_indices_derived_from_resource_registry: true");
    println!("    dispatch_layer_indices_dynamic_vec: true");
    println!("    resource_registry_rows_dynamic_vec: true");
    println!("    main_preflight_dispatch_layer_literal_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(dispatch_layer_indices)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_catalog_from_rows<'a>(
    rows: Vec<QwenResidentLayerRunnerDenseLayerResourceSlots<'a>>,
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceCatalog<'a>> {
    let resource_rows = rows.as_slice();
    if resource_rows.is_empty() {
        anyhow::bail!("resident runner dense resource catalog has no rows");
    }

    let mut first_layer_idx = resource_rows[0].layer_idx;
    let mut last_layer_idx = resource_rows[0].layer_idx;
    for (row_index, row) in resource_rows.iter().enumerate() {
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
        if resource_rows[..row_index]
            .iter()
            .any(|prior_row| prior_row.layer_idx == row.layer_idx)
        {
            anyhow::bail!(
                "duplicate dense-layer resource catalog row for layer {}",
                row.layer_idx
            );
        }
    }
    let contiguous_layer_window =
        last_layer_idx - first_layer_idx + 1 == resource_rows.len() as u32;

    println!("  resident_layer_runner_dense_layer_resource_catalog_builder_stage:");
    println!("    source: resident_runner_dense_resource_catalog_from_rows");
    println!("    resource_catalog_owned_by_resident_runner_module: true");
    println!("    resource_catalog_fixed_array: false");
    println!("    resource_catalog_heap_allocation: true");
    println!("    resource_catalog_row_count: {}", resource_rows.len());
    println!("    resource_catalog_first_layer_idx: {first_layer_idx}");
    println!("    resource_catalog_last_layer_idx: {last_layer_idx}");
    println!("    resource_catalog_contiguous_layer_window: {contiguous_layer_window}");
    println!("    resource_catalog_rows_dynamic_vec: true");
    println!("    fixed_resource_catalog_row_count_removed: true");
    println!("    resource_catalog_duplicate_layer_indices: false");
    println!("    resource_catalog_registry_complete: {registry_complete}");
    println!("    staged_device_table_copy_started: false");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerResourceCatalog {
        rows,
        first_layer_idx,
        last_layer_idx,
        registry_complete,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_catalog_from_registry<'a>(
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    registry: &QwenResidentLayerRunnerDenseLayerResourceRegistry<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceCatalog<'a>> {
    let dispatch_row = registry
        .rows
        .iter()
        .copied()
        .find(|row| row.layer_idx == dispatch_window.dispatch_layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense resource registry missing dispatch row for layer {}",
                dispatch_window.dispatch_layer_idx
            )
        })?;
    let lookahead_row = registry
        .rows
        .iter()
        .copied()
        .find(|row| row.layer_idx == dispatch_window.lookahead_layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense resource registry missing lookahead row for layer {}",
                dispatch_window.lookahead_layer_idx
            )
        })?;

    println!("  resident_layer_runner_dense_layer_resource_catalog_from_registry_stage:");
    println!("    source: resident_runner_dense_resource_catalog_from_registry");
    println!("    resource_registry_owned_by_resident_runner_module: true");
    println!("    resource_catalog_derived_from_registry: true");
    println!("    setup_time_two_row_catalog_literal_removed: true");
    println!("    dispatch_window_descriptor_consumed: true");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    registry_first_layer_idx: {}", registry.first_layer_idx);
    println!("    registry_last_layer_idx: {}", registry.last_layer_idx);
    println!("    registry_row_count: {}", registry.rows.len());
    println!("    resource_registry_rows_dynamic_vec: true");
    println!("    dispatch_row_found: true");
    println!("    lookahead_row_found: true");
    println!("    staged_device_table_copy_started: false");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_resource_catalog_from_rows(
        vec![dispatch_row, lookahead_row],
        registry.registry_complete,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_preflight_state_from_registry<'a>(
    dispatch_layer_idx: u32,
    output_slot_rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
    registry: &QwenResidentLayerRunnerDenseLayerResourceRegistry<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerPreflightState<'a>> {
    let lookahead_layer_idx = dispatch_layer_idx
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("resident dense preflight dispatch layer overflow"))?;
    let dispatch_window = qwen_resident_layer_runner_dense_layer_dispatch_window_descriptor(
        dispatch_layer_idx,
        lookahead_layer_idx,
    )?;
    let resource_catalog = qwen_resident_layer_runner_dense_layer_resource_catalog_from_registry(
        dispatch_window,
        registry,
    )?;
    let output_slot_resolver_target =
        qwen_resident_layer_runner_dense_output_slot_resolver_target_from_dispatch_window(
            output_slot_rows,
            dispatch_window,
        )?;

    println!("  resident_layer_runner_dense_layer_preflight_state_stage:");
    println!("    source: resident_runner_dense_layer_preflight_state_from_registry");
    println!("    preflight_state_owner: resident_runner_module");
    println!("    dispatch_layer_idx: {dispatch_layer_idx}");
    println!("    lookahead_layer_idx: {lookahead_layer_idx}");
    println!("    dispatch_window_derived_by_preflight_state: true");
    println!("    resource_catalog_derived_by_preflight_state: true");
    println!("    output_slot_resolver_derived_by_preflight_state: true");
    println!("    main_repeated_window_catalog_resolver_blocks_removed: true");
    println!(
        "    output_slot_row_index: {}",
        output_slot_resolver_target.row_index
    );
    println!("    registry_first_layer_idx: {}", registry.first_layer_idx);
    println!("    registry_last_layer_idx: {}", registry.last_layer_idx);
    println!("    registry_row_count: {}", registry.rows.len());
    println!("    resource_registry_rows_dynamic_vec: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerPreflightState {
        dispatch_window,
        resource_catalog,
        output_slot_resolver_target,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_preflight_state_window_from_registry<'a>(
    dispatch_layer_indices: Vec<u32>,
    output_slot_rows: &[QwenResidentLayerRunnerDenseOutputSlotMetadataRow],
    registry: &QwenResidentLayerRunnerDenseLayerResourceRegistry<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a>> {
    let preflight_state_rows = dispatch_layer_indices.len();
    if preflight_state_rows == 0 {
        anyhow::bail!("resident dense preflight-state window requires at least one layer");
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut previous_layer_idx = None;
    let mut states = Vec::with_capacity(preflight_state_rows);
    for (row_index, dispatch_layer_idx) in dispatch_layer_indices.into_iter().enumerate() {
        if !seen_layers.insert(dispatch_layer_idx) {
            anyhow::bail!(
                "resident dense preflight-state window contains duplicate layer {}",
                dispatch_layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if dispatch_layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense preflight-state window is not strictly ascending: layer {} after layer {} at row {}",
                    dispatch_layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(dispatch_layer_idx);
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
        states.push(
            qwen_resident_layer_runner_dense_layer_preflight_state_from_registry(
                dispatch_layer_idx,
                output_slot_rows,
                registry,
            )?,
        );
    }

    let contiguous_layer_window = last_layer_idx - first_layer_idx + 1
        == u32::try_from(preflight_state_rows).unwrap_or(u32::MAX);

    println!("  resident_layer_runner_dense_layer_preflight_state_window_stage:");
    println!("    source: resident_runner_dense_layer_preflight_state_window_from_registry");
    println!("    preflight_state_window_owner: resident_runner_module");
    println!("    window_row_count: {preflight_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    preflight_state_rows_derived_from_dispatch_layer_indices_len: true");
    println!("    preflight_state_window_rows_dynamic_vec: true");
    println!("    fixed_preflight_state_window_row_count_removed: true");
    println!("    dispatch_layer_indices_strictly_ascending: true");
    println!("    dispatch_layer_indices_duplicate_free: true");
    println!("    main_repeated_preflight_state_calls_removed: true");
    println!("    per_layer_preflight_state_builder_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerPreflightStateWindow { rows: states })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_preflight_state_from_window<'a>(
    target_layer_idx: u32,
    preflight_state_window: &QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerPreflightState<'a>> {
    let preflight_state_rows = preflight_state_window.rows.len();
    if preflight_state_rows == 0 {
        anyhow::bail!("resident dense preflight-state window select requires at least one row");
    }

    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut selected_state = None;
    for preflight_state in &preflight_state_window.rows {
        let dispatch_layer_idx = preflight_state.dispatch_window.dispatch_layer_idx;
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
        if dispatch_layer_idx == target_layer_idx {
            selected_state = Some(preflight_state.clone());
        }
    }
    let selected_state = selected_state.ok_or_else(|| {
        anyhow::anyhow!(
            "resident dense preflight-state window missing target layer {} in {} rows",
            target_layer_idx,
            preflight_state_rows
        )
    })?;

    println!("  resident_layer_runner_dense_layer_preflight_state_window_select_stage:");
    println!("    source: resident_runner_dense_layer_preflight_state_from_window");
    println!("    preflight_state_window_owner: resident_runner_module");
    println!("    window_row_count: {preflight_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    target_layer_idx: {target_layer_idx}");
    println!(
        "    selected_dispatch_layer_idx: {}",
        selected_state.dispatch_window.dispatch_layer_idx
    );
    println!(
        "    selected_lookahead_layer_idx: {}",
        selected_state.dispatch_window.lookahead_layer_idx
    );
    println!("    target_preflight_state_found: true");
    println!("    preflight_state_window_rows_dynamic_vec: true");
    println!("    semantic_preflight_state_window_consumed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(selected_state)
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_legacy_apply_plan_from_preflight_state<
    'a,
>(
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    preflight_state: &QwenResidentLayerRunnerDenseLayerPreflightState<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotLegacyApplyPlan> {
    let apply_plan = qwen_resident_layer_runner_dense_output_slot_legacy_apply_plan_from_hbm_state(
        hbm_state,
        preflight_state.output_slot_resolver_target,
    )?;

    println!("  resident_layer_runner_dense_layer_preflight_apply_plan_stage:");
    println!(
        "    source: resident_runner_dense_output_slot_legacy_apply_plan_from_preflight_state"
    );
    println!("    apply_plan_owner: resident_runner_module");
    println!(
        "    dispatch_layer_idx: {}",
        preflight_state.dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        preflight_state.dispatch_window.lookahead_layer_idx
    );
    println!(
        "    output_slot_expected_layer_idx: {}",
        apply_plan.expected_layer_idx
    );
    println!(
        "    output_slot_expected_row_index: {}",
        apply_plan.expected_row_index
    );
    println!("    apply_plan_derived_from_preflight_state: true");
    println!("    apply_plan_derived_from_hbm_state: true");
    println!("    main_repeated_apply_plan_derivation_removed: true");
    println!("    hbm_table_rows: {}", apply_plan.table_rows);
    println!(
        "    layer_index_entries: {}",
        apply_plan.layer_index_entries
    );
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(apply_plan)
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_from_preflight_state<
    'a,
>(
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    preflight_state: QwenResidentLayerRunnerDenseLayerPreflightState<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContext> {
    let resolver_target = preflight_state.output_slot_resolver_target;
    let apply_plan =
        qwen_resident_layer_runner_dense_output_slot_legacy_apply_plan_from_preflight_state(
            hbm_state,
            &preflight_state,
        )?;
    if apply_plan.expected_layer_idx != resolver_target.layer_idx {
        anyhow::bail!(
            "resident dense output slot apply context layer mismatch: plan {} resolver {}",
            apply_plan.expected_layer_idx,
            resolver_target.layer_idx
        );
    }
    if apply_plan.expected_row_index != resolver_target.row_index {
        anyhow::bail!(
            "resident dense output slot apply context row mismatch: plan {} resolver {}",
            apply_plan.expected_row_index,
            resolver_target.row_index
        );
    }

    println!("  resident_layer_runner_dense_output_slot_apply_context_stage:");
    println!(
        "    source: resident_runner_dense_output_slot_legacy_apply_context_from_preflight_state"
    );
    println!("    apply_context_owner: resident_runner_module");
    println!("    apply_plan_and_resolver_target_bound: true");
    println!(
        "    dispatch_layer_idx: {}",
        preflight_state.dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        preflight_state.dispatch_window.lookahead_layer_idx
    );
    println!(
        "    apply_plan_expected_layer_idx: {}",
        apply_plan.expected_layer_idx
    );
    println!(
        "    resolver_target_layer_idx: {}",
        resolver_target.layer_idx
    );
    println!(
        "    resolver_target_row_index: {}",
        resolver_target.row_index
    );
    println!("    main_separate_apply_plan_resolver_pair_removed_from_apply_calls: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContext {
        apply_plan,
        resolver_target,
        source: "resident_runner_dense_output_slot_legacy_apply_context",
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_window_from_preflight_states<
    'a,
>(
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    preflight_state_window: &QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow> {
    let preflight_state_rows = preflight_state_window.rows.len();
    if preflight_state_rows == 0 {
        anyhow::bail!("resident dense apply-context window requires at least one preflight state");
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut previous_layer_idx = None;
    let mut apply_context_rows = Vec::with_capacity(preflight_state_rows);
    for (row_index, preflight_state) in preflight_state_window.rows.iter().cloned().enumerate() {
        let dispatch_layer_idx = preflight_state.dispatch_window.dispatch_layer_idx;
        if !seen_layers.insert(dispatch_layer_idx) {
            anyhow::bail!(
                "resident dense apply-context window contains duplicate layer {}",
                dispatch_layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if dispatch_layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense apply-context window is not strictly ascending: layer {} after layer {} at row {}",
                    dispatch_layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(dispatch_layer_idx);
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
        let apply_context =
            qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_from_preflight_state(
                hbm_state,
                preflight_state,
            )?;
        apply_context_rows.push(
            QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextRow {
                layer_idx: dispatch_layer_idx,
                context: apply_context,
            },
        );
    }

    let contiguous_layer_window = last_layer_idx - first_layer_idx + 1
        == u32::try_from(preflight_state_rows).unwrap_or(u32::MAX);
    let apply_context_window_rows = apply_context_rows.len();

    println!("  resident_layer_runner_dense_output_slot_apply_context_window_stage:");
    println!("    source: resident_runner_dense_output_slot_legacy_apply_context_window_from_preflight_states");
    println!("    apply_context_window_owner: resident_runner_module");
    println!("    window_row_count: {apply_context_window_rows}");
    println!("    preflight_state_rows: {preflight_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    apply_context_window_rows_derived_from_preflight_states_len: true");
    println!("    apply_context_window_rows_dynamic_vec: true");
    println!("    fixed_apply_context_window_row_count_removed: true");
    println!("    preflight_state_window_rows_dynamic_vec: true");
    println!("    preflight_state_window_consumed: true");
    println!("    dispatch_layer_indices_strictly_ascending: true");
    println!("    dispatch_layer_indices_duplicate_free: true");
    println!("    semantic_apply_context_window_constructed: true");
    println!("    main_apply_plan_resolver_pair_window_removed: true");
    println!("    main_per_layer_apply_context_vars_removed_from_apply_calls: true");
    println!("    per_layer_apply_context_builder_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(
        QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow {
            rows: apply_context_rows,
        },
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_legacy_apply_window_from_windows<'a>(
    stage_ref_window: QwenResidentLayerRunnerDenseLayerLegacyStageRefWindow<'a>,
    apply_context_window: QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'a>> {
    let stage_ref_window_rows = stage_ref_window.rows.len();
    let apply_context_window_rows = apply_context_window.rows.len();
    if stage_ref_window_rows != apply_context_window_rows {
        anyhow::bail!(
            "resident dense legacy apply window row-count mismatch: stage refs {} contexts {}",
            stage_ref_window_rows,
            apply_context_window_rows
        );
    }
    let (stage_first_layer_idx, stage_last_layer_idx) =
        qwen_resident_layer_runner_dense_layer_legacy_stage_ref_window_bounds(&stage_ref_window)?;
    let (context_first_layer_idx, context_last_layer_idx) =
        qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_window_bounds(
            &apply_context_window,
        )?;
    if stage_first_layer_idx != context_first_layer_idx
        || stage_last_layer_idx != context_last_layer_idx
    {
        anyhow::bail!(
            "resident dense legacy apply window bounds mismatch: stage refs {}..{} contexts {}..{}",
            stage_first_layer_idx,
            stage_last_layer_idx,
            context_first_layer_idx,
            context_last_layer_idx
        );
    }

    println!("  resident_layer_runner_dense_layer_legacy_apply_window_stage:");
    println!("    source: resident_runner_dense_layer_legacy_apply_window_from_windows");
    println!("    legacy_apply_window_owner: resident_runner_module");
    println!("    window_row_count: {stage_ref_window_rows}");
    println!("    stage_ref_window_rows: {stage_ref_window_rows}");
    println!("    apply_context_window_rows: {apply_context_window_rows}");
    println!("    window_first_layer_idx: {stage_first_layer_idx}");
    println!("    window_last_layer_idx: {stage_last_layer_idx}");
    println!("    stage_ref_window_bound: true");
    println!("    apply_context_window_bound: true");
    println!("    stage_ref_and_apply_context_row_counts_match: true");
    println!("    stage_ref_and_apply_context_bounds_match: true");
    println!("    stage_ref_window_rows_dynamic_vec: true");
    println!("    apply_context_window_rows_dynamic_vec: true");
    println!("    semantic_legacy_apply_window_constructed: true");
    println!("    main_loose_stage_ref_and_apply_context_window_pair_removed: true");
    println!("    legacy_stage_refs_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerLegacyApplyWindow {
        stage_ref_window,
        apply_context_window,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_from_window(
    target_layer_idx: u32,
    apply_context_window: &QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow,
) -> anyhow::Result<QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContext> {
    let apply_context_window_rows = apply_context_window.rows.len();
    if apply_context_window_rows == 0 {
        anyhow::bail!("resident dense apply-context window select requires at least one row");
    }

    let (first_layer_idx, last_layer_idx) =
        qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_window_bounds(
            &apply_context_window,
        )?;
    let mut selected_context = None;
    for row in apply_context_window.rows.iter().copied() {
        if row.layer_idx == target_layer_idx {
            selected_context = Some(row.context);
        }
    }
    let selected_context = selected_context.ok_or_else(|| {
        anyhow::anyhow!(
            "resident dense apply-context window missing target layer {} in {} rows",
            target_layer_idx,
            apply_context_window_rows
        )
    })?;
    if selected_context.apply_plan.expected_layer_idx != target_layer_idx {
        anyhow::bail!(
            "resident dense apply-context window target {} does not match selected plan layer {}",
            target_layer_idx,
            selected_context.apply_plan.expected_layer_idx
        );
    }
    if selected_context.resolver_target.layer_idx != target_layer_idx {
        anyhow::bail!(
            "resident dense apply-context window target {} does not match selected resolver layer {}",
            target_layer_idx,
            selected_context.resolver_target.layer_idx
        );
    }

    println!("  resident_layer_runner_dense_output_slot_apply_context_window_select_stage:");
    println!("    source: resident_runner_dense_output_slot_legacy_apply_context_from_window");
    println!("    apply_context_window_owner: resident_runner_module");
    println!("    window_row_count: {apply_context_window_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    target_layer_idx: {target_layer_idx}");
    println!(
        "    selected_apply_plan_expected_layer_idx: {}",
        selected_context.apply_plan.expected_layer_idx
    );
    println!(
        "    selected_resolver_target_layer_idx: {}",
        selected_context.resolver_target.layer_idx
    );
    println!(
        "    selected_resolver_target_row_index: {}",
        selected_context.resolver_target.row_index
    );
    println!("    target_context_found: true");
    println!("    apply_plan_and_resolver_target_bound: true");
    println!("    semantic_apply_context_window_consumed: true");
    println!("    apply_context_window_rows_dynamic_vec: true");
    println!("    main_per_layer_apply_context_vars_removed_from_apply_calls: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(selected_context)
}

fn qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_window_bounds(
    apply_context_window: &QwenResidentLayerRunnerDenseOutputSlotLegacyApplyContextWindow,
) -> anyhow::Result<(u32, u32)> {
    if apply_context_window.rows.is_empty() {
        anyhow::bail!("resident dense apply-context window bounds require at least one row");
    }
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    for row in &apply_context_window.rows {
        first_layer_idx = first_layer_idx.min(row.layer_idx);
        last_layer_idx = last_layer_idx.max(row.layer_idx);
    }
    Ok((first_layer_idx, last_layer_idx))
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_catalog<'a>(
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    catalog: &QwenResidentLayerRunnerDenseLayerResourceCatalog<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, 2>> {
    let dispatch_row = catalog
        .rows
        .iter()
        .copied()
        .find(|row| row.layer_idx == dispatch_window.dispatch_layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense resource catalog missing dispatch row for layer {}",
                dispatch_window.dispatch_layer_idx
            )
        })?;
    let lookahead_row = catalog
        .rows
        .iter()
        .copied()
        .find(|row| row.layer_idx == dispatch_window.lookahead_layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense resource catalog missing lookahead row for layer {}",
                dispatch_window.lookahead_layer_idx
            )
        })?;

    println!("  resident_layer_runner_dense_layer_resource_catalog_lookup_stage:");
    println!("    source: resident_runner_dense_resource_catalog_descriptor_lookup");
    println!("    resource_catalog_owned_by_resident_runner_module: true");
    println!("    resource_catalog_lookup_owned_by_resident_runner_module: true");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    main_per_layer_resource_refs_at_dispatch_removed: true");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    catalog_first_layer_idx: {}", catalog.first_layer_idx);
    println!("    catalog_last_layer_idx: {}", catalog.last_layer_idx);
    println!("    catalog_row_count: {}", catalog.rows.len());
    println!("    resource_catalog_rows_dynamic_vec: true");
    println!("    dispatch_row_found: true");
    println!("    lookahead_row_found: true");
    println!("    staged_device_table_copy_started: false");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_descriptor_resources(
        dispatch_window,
        dispatch_row.qkv,
        dispatch_row.qk,
        dispatch_row.cache_plan,
        dispatch_row.attn_plan,
        dispatch_row.o_proj_plan,
        dispatch_row.post_attn_norm,
        dispatch_row.peer_plan,
        dispatch_row.mlp,
        dispatch_row.peer_mlp,
        dispatch_row.next_input_norm,
        lookahead_row.qkv,
        lookahead_row.qk,
        lookahead_row.cache_plan,
        lookahead_row.attn_plan,
        lookahead_row.o_proj_plan,
        lookahead_row.post_attn_norm,
        lookahead_row.peer_plan,
        lookahead_row.mlp,
        lookahead_row.peer_mlp,
        lookahead_row.next_input_norm,
        catalog.registry_complete,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_window_from_preflight_states<
    'a,
>(
    preflight_state_window: &QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTableWindow<'a, 2>> {
    let preflight_state_rows = preflight_state_window.rows.len();
    if preflight_state_rows == 0 {
        anyhow::bail!(
            "resident dense resource-slot-table window requires at least one preflight state"
        );
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut previous_layer_idx = None;
    let mut slot_table_rows = Vec::with_capacity(preflight_state_rows);
    for (row_index, preflight_state) in preflight_state_window.rows.iter().enumerate() {
        let dispatch_layer_idx = preflight_state.dispatch_window.dispatch_layer_idx;
        if !seen_layers.insert(dispatch_layer_idx) {
            anyhow::bail!(
                "resident dense resource-slot-table window contains duplicate layer {}",
                dispatch_layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if dispatch_layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense resource-slot-table window is not strictly ascending: layer {} after layer {} at row {}",
                    dispatch_layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(dispatch_layer_idx);
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
        let slot_table = qwen_resident_layer_runner_dense_layer_resource_slot_table_from_catalog(
            preflight_state.dispatch_window,
            &preflight_state.resource_catalog,
        )?;
        slot_table_rows.push(QwenResidentLayerRunnerDenseLayerResourceSlotTableRow {
            dispatch_layer_idx,
            slot_table,
        });
    }

    let contiguous_layer_window = last_layer_idx - first_layer_idx + 1
        == u32::try_from(preflight_state_rows).unwrap_or(u32::MAX);
    let resource_slot_table_window_rows = slot_table_rows.len();

    println!("  resident_layer_runner_dense_layer_resource_slot_table_window_stage:");
    println!("    source: resident_runner_dense_resource_slot_table_window_from_preflight_states");
    println!("    resource_slot_table_window_owner: resident_runner_module");
    println!("    window_row_count: {resource_slot_table_window_rows}");
    println!("    preflight_state_rows: {preflight_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    resource_slot_table_window_rows_derived_from_preflight_states_len: true");
    println!("    resource_slot_table_window_rows_dynamic_vec: true");
    println!("    fixed_resource_slot_table_window_row_count_removed: true");
    println!("    preflight_state_window_rows_dynamic_vec: true");
    println!("    dispatch_layer_indices_strictly_ascending: true");
    println!("    dispatch_layer_indices_duplicate_free: true");
    println!("    preflight_state_window_consumed: true");
    println!("    per_layer_resource_slot_tables_prebuilt: true");
    println!("    main_per_layer_resource_slot_table_from_catalog_calls_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerResourceSlotTableWindow {
        rows: slot_table_rows,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_window<
    'a,
    const TABLE_ROWS: usize,
>(
    target_layer_idx: u32,
    slot_table_window: &QwenResidentLayerRunnerDenseLayerResourceSlotTableWindow<'a, TABLE_ROWS>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, TABLE_ROWS>> {
    let resource_slot_table_window_rows = slot_table_window.rows.len();
    if resource_slot_table_window_rows == 0 {
        anyhow::bail!("resident dense resource-slot-table window select requires at least one row");
    }

    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut selected_slot_table = None;
    for row in slot_table_window.rows.iter().copied() {
        first_layer_idx = first_layer_idx.min(row.dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(row.dispatch_layer_idx);
        if row.dispatch_layer_idx == target_layer_idx {
            selected_slot_table = Some(row.slot_table);
        }
    }
    let selected_slot_table = selected_slot_table.ok_or_else(|| {
        anyhow::anyhow!(
            "resident dense resource-slot-table window missing target layer {} in {} rows",
            target_layer_idx,
            resource_slot_table_window_rows
        )
    })?;
    if !selected_slot_table
        .slots
        .iter()
        .any(|slot| slot.layer_idx == target_layer_idx)
    {
        anyhow::bail!(
            "resident dense resource-slot-table window target {} not present in selected slot table {}..{}",
            target_layer_idx,
            selected_slot_table.first_layer_idx,
            selected_slot_table.last_layer_idx
        );
    }

    println!("  resident_layer_runner_dense_layer_resource_slot_table_window_select_stage:");
    println!("    source: resident_runner_dense_resource_slot_table_from_window");
    println!("    resource_slot_table_window_owner: resident_runner_module");
    println!("    window_row_count: {resource_slot_table_window_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    target_layer_idx: {target_layer_idx}");
    println!(
        "    selected_slot_table_first_layer_idx: {}",
        selected_slot_table.first_layer_idx
    );
    println!(
        "    selected_slot_table_last_layer_idx: {}",
        selected_slot_table.last_layer_idx
    );
    println!(
        "    selected_slot_table_rows: {}",
        selected_slot_table.slots.len()
    );
    println!("    selected_slot_table_contains_target: true");
    println!(
        "    selected_slot_table_registry_complete: {}",
        selected_slot_table.registry_complete
    );
    println!("    semantic_resource_slot_table_window_consumed: true");
    println!("    resource_slot_table_window_rows_dynamic_vec: true");
    println!("    main_per_layer_resource_slot_table_from_catalog_calls_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(selected_slot_table)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_context_window_from_preflight_states<
    'a,
>(
    preflight_state_window: &QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchContextWindow<'a, 2>> {
    let preflight_state_rows = preflight_state_window.rows.len();
    if preflight_state_rows == 0 {
        anyhow::bail!(
            "resident dense dispatch-context window requires at least one preflight state"
        );
    }

    let resource_slot_table_window =
        qwen_resident_layer_runner_dense_layer_resource_slot_table_window_from_preflight_states(
            preflight_state_window,
        )?;
    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut previous_layer_idx = None;
    let mut dispatch_windows = Vec::with_capacity(preflight_state_rows);
    for (row_index, preflight_state) in preflight_state_window.rows.iter().enumerate() {
        let dispatch_window = preflight_state.dispatch_window;
        let dispatch_layer_idx = dispatch_window.dispatch_layer_idx;
        if !seen_layers.insert(dispatch_layer_idx) {
            anyhow::bail!(
                "resident dense dispatch-context window contains duplicate layer {}",
                dispatch_layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if dispatch_layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense dispatch-context window is not strictly ascending: layer {} after layer {} at row {}",
                    dispatch_layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(dispatch_layer_idx);
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
        dispatch_windows.push(dispatch_window);
    }

    let contiguous_layer_window = last_layer_idx - first_layer_idx + 1
        == u32::try_from(preflight_state_rows).unwrap_or(u32::MAX);
    let dispatch_context_window_rows = dispatch_windows.len();

    println!("  resident_layer_runner_dense_layer_dispatch_context_window_stage:");
    println!("    source: resident_runner_dense_dispatch_context_window_from_preflight_states");
    println!("    dispatch_context_window_owner: resident_runner_module");
    println!("    window_row_count: {dispatch_context_window_rows}");
    println!("    preflight_state_rows: {preflight_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    dispatch_context_window_rows_derived_from_preflight_states_len: true");
    println!("    dispatch_context_window_rows_dynamic_vec: true");
    println!("    fixed_dispatch_context_window_row_count_removed: true");
    println!("    preflight_state_window_rows_dynamic_vec: true");
    println!("    dispatch_layer_indices_strictly_ascending: true");
    println!("    dispatch_layer_indices_duplicate_free: true");
    println!("    preflight_state_window_consumed: true");
    println!("    dispatch_windows_bound: true");
    println!("    resource_slot_table_window_bound: true");
    println!("    main_loose_dispatch_window_and_slot_table_pair_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchContextWindow {
        dispatch_windows,
        resource_slot_table_window,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_context_windows_from_preflight_states<'a>(
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    preflight_state_window: &QwenResidentLayerRunnerDenseLayerPreflightStateWindow<'a>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerContextWindows<'a>> {
    let preflight_state_rows = preflight_state_window.rows.len();
    if preflight_state_rows == 0 {
        anyhow::bail!("resident dense context window requires at least one preflight state");
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut previous_layer_idx = None;
    for (row_index, preflight_state) in preflight_state_window.rows.iter().enumerate() {
        let dispatch_layer_idx = preflight_state.dispatch_window.dispatch_layer_idx;
        if !seen_layers.insert(dispatch_layer_idx) {
            anyhow::bail!(
                "resident dense context window contains duplicate layer {}",
                dispatch_layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if dispatch_layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense context window is not strictly ascending: layer {} after layer {} at row {}",
                    dispatch_layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(dispatch_layer_idx);
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
    }
    let contiguous_layer_window = last_layer_idx - first_layer_idx + 1
        == u32::try_from(preflight_state_rows).unwrap_or(u32::MAX);

    let dispatch_context_window =
        qwen_resident_layer_runner_dense_layer_dispatch_context_window_from_preflight_states(
            preflight_state_window,
        )?;
    let legacy_apply_context_window =
        qwen_resident_layer_runner_dense_output_slot_legacy_apply_context_window_from_preflight_states(
            hbm_state,
            preflight_state_window,
        )?;

    println!("  resident_layer_runner_dense_layer_context_windows_stage:");
    println!("    source: resident_runner_dense_layer_context_windows_from_preflight_states");
    println!("    context_window_owner: resident_runner_module");
    println!("    window_row_count: {preflight_state_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    preflight_state_window_consumed: true");
    println!("    preflight_state_window_rows_dynamic_vec: true");
    println!("    context_window_rows_derived_from_preflight_states_len: true");
    println!("    dispatch_context_window_bound: true");
    println!("    apply_context_window_bound: true");
    println!("    dispatch_and_apply_context_bounds_match: true");
    println!("    main_duplicate_preflight_state_array_removed: true");
    println!("    main_loose_context_window_builders_bound: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerContextWindows {
        dispatch_context_window,
        legacy_apply_context_window,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_context_from_window<
    'a,
    const TABLE_ROWS: usize,
>(
    target_layer_idx: u32,
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'a,
        TABLE_ROWS,
    >,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchContext<'a, TABLE_ROWS>> {
    let dispatch_context_window_rows = dispatch_context_window.dispatch_windows.len();
    if dispatch_context_window_rows == 0 {
        anyhow::bail!("resident dense dispatch-context window select requires at least one row");
    }

    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut selected_dispatch_window = None;
    for dispatch_window in dispatch_context_window.dispatch_windows.iter().copied() {
        first_layer_idx = first_layer_idx.min(dispatch_window.dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_window.dispatch_layer_idx);
        if dispatch_window.dispatch_layer_idx == target_layer_idx {
            selected_dispatch_window = Some(dispatch_window);
        }
    }
    let selected_dispatch_window = selected_dispatch_window.ok_or_else(|| {
        anyhow::anyhow!(
            "resident dense dispatch-context window missing target layer {} in {} rows",
            target_layer_idx,
            dispatch_context_window_rows
        )
    })?;
    let resource_slot_table =
        qwen_resident_layer_runner_dense_layer_resource_slot_table_from_window(
            target_layer_idx,
            &dispatch_context_window.resource_slot_table_window,
        )?;
    if !resource_slot_table
        .slots
        .iter()
        .any(|slot| slot.layer_idx == selected_dispatch_window.dispatch_layer_idx)
    {
        anyhow::bail!(
            "resident dense dispatch-context window target {} not present in selected resource-slot table {}..{}",
            target_layer_idx,
            resource_slot_table.first_layer_idx,
            resource_slot_table.last_layer_idx
        );
    }

    println!("  resident_layer_runner_dense_layer_dispatch_context_window_select_stage:");
    println!("    source: resident_runner_dense_dispatch_context_from_window");
    println!("    dispatch_context_window_owner: resident_runner_module");
    println!("    window_row_count: {dispatch_context_window_rows}");
    println!("    window_first_layer_idx: {first_layer_idx}");
    println!("    window_last_layer_idx: {last_layer_idx}");
    println!("    target_layer_idx: {target_layer_idx}");
    println!(
        "    selected_dispatch_layer_idx: {}",
        selected_dispatch_window.dispatch_layer_idx
    );
    println!(
        "    selected_lookahead_layer_idx: {}",
        selected_dispatch_window.lookahead_layer_idx
    );
    println!(
        "    selected_handoff_target_layer_idx: {}",
        selected_dispatch_window.handoff_contract.target_layer_idx
    );
    println!(
        "    selected_slot_table_first_layer_idx: {}",
        resource_slot_table.first_layer_idx
    );
    println!(
        "    selected_slot_table_last_layer_idx: {}",
        resource_slot_table.last_layer_idx
    );
    println!("    selected_slot_table_contains_target: true");
    println!("    dispatch_window_selected_from_context_window: true");
    println!("    resource_slot_table_selected_from_context_window: true");
    println!("    semantic_dispatch_context_window_consumed: true");
    println!("    dispatch_context_window_rows_dynamic_vec: true");
    println!("    resource_slot_table_window_rows_dynamic_vec: true");
    println!("    main_loose_dispatch_window_and_slot_table_pair_removed: true");
    println!("    main_per_layer_dispatch_window_vars_removed_from_dispatch_calls: true");
    println!("    main_resource_slot_window_select_calls_removed_from_dispatch_calls: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchContext {
        dispatch_window: selected_dispatch_window,
        resource_slot_table,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_window_payload_window<
    'a,
>(
    payload_window: QwenResidentLayerRunnerDenseLayerResourcePayloadWindow<'a>,
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, 2>> {
    println!("  resident_layer_runner_dense_layer_descriptor_payload_window_consumption_stage:");
    println!("    source: resident_runner_dense_dispatch_window_payload_window_slot_table_driver");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    descriptor_payload_window_consumed: true");
    println!("    descriptor_payload_window_to_slot_table: true");
    println!("    slot_table_owned_by_resident_runner_module: true");
    println!("    payload_window_size: 2");
    println!(
        "    dispatch_layer_idx: {}",
        payload_window.dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        payload_window.dispatch_window.lookahead_layer_idx
    );
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_window_payloads(
        payload_window.dispatch_window,
        payload_window.dispatch_payload,
        payload_window.lookahead_payload,
        registry_complete,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_dispatch_window_payloads<
    'a,
>(
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    dispatch_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
    lookahead_payload: QwenResidentLayerRunnerDenseLayerResourcePayload<'a>,
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, 2>> {
    println!("  resident_layer_runner_dense_layer_descriptor_row_materialization_stage:");
    println!("    source: resident_runner_dense_dispatch_window_payload_row_builder");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    payloads_indexless: true");
    println!("    resource_rows_built_by_descriptor: true");
    println!("    resource_rows_owned_by_resident_runner_module: true");
    println!("    row_count: 2");
    println!(
        "    dispatch_row_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_row_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    telemetry_layer_ids_from_descriptor: true");
    println!("    dispatch_scope_changed: false");
    println!("    topology_routing_started: false");
    println!("    graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_resource_slot_table_from_rows(
        [
            QwenResidentLayerRunnerDenseLayerResourceSlots {
                layer_idx: dispatch_window.dispatch_layer_idx,
                qkv: dispatch_payload.qkv,
                qk: dispatch_payload.qk,
                cache_plan: dispatch_payload.cache_plan,
                attn_plan: dispatch_payload.attn_plan,
                o_proj_plan: dispatch_payload.o_proj_plan,
                post_attn_norm: dispatch_payload.post_attn_norm,
                peer_plan: dispatch_payload.peer_plan,
                mlp: dispatch_payload.mlp,
                peer_mlp: dispatch_payload.peer_mlp,
                next_input_norm: dispatch_payload.next_input_norm,
            },
            QwenResidentLayerRunnerDenseLayerResourceSlots {
                layer_idx: dispatch_window.lookahead_layer_idx,
                qkv: lookahead_payload.qkv,
                qk: lookahead_payload.qk,
                cache_plan: lookahead_payload.cache_plan,
                attn_plan: lookahead_payload.attn_plan,
                o_proj_plan: lookahead_payload.o_proj_plan,
                post_attn_norm: lookahead_payload.post_attn_norm,
                peer_plan: lookahead_payload.peer_plan,
                mlp: lookahead_payload.mlp,
                peer_mlp: lookahead_payload.peer_mlp,
                next_input_norm: lookahead_payload.next_input_norm,
            },
        ],
        registry_complete,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_slot_table_from_rows<
    'a,
    const N: usize,
>(
    slots: [QwenResidentLayerRunnerDenseLayerResourceSlots<'a>; N],
    registry_complete: bool,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, N>> {
    let resource_slots = slots.as_slice();
    if resource_slots.is_empty() {
        anyhow::bail!("resident runner dense resource slot table has no rows");
    }

    let mut first_layer_idx = resource_slots[0].layer_idx;
    let mut last_layer_idx = resource_slots[0].layer_idx;
    for (slot_index, slot) in resource_slots.iter().enumerate() {
        first_layer_idx = first_layer_idx.min(slot.layer_idx);
        last_layer_idx = last_layer_idx.max(slot.layer_idx);
        if resource_slots[..slot_index]
            .iter()
            .any(|prior_slot| prior_slot.layer_idx == slot.layer_idx)
        {
            anyhow::bail!(
                "duplicate dense-layer resource slot table row for layer {}",
                slot.layer_idx
            );
        }
    }

    let contiguous_layer_window =
        last_layer_idx - first_layer_idx + 1 == resource_slots.len() as u32;

    println!("  resident_layer_runner_dense_layer_resource_slot_table_builder_stage:");
    println!("    source: resident_runner_prebuilt_dense_resource_slot_table_builder");
    println!("    typed_resource_slot_table: true");
    println!("    table_builder_rows_prebuilt: true");
    println!("    slot_table_builder_owned_by_resident_runner_module: true");
    println!("    table_builder_layer_count: {}", resource_slots.len());
    println!("    table_builder_first_layer_idx: {first_layer_idx}");
    println!("    table_builder_last_layer_idx: {last_layer_idx}");
    println!("    table_builder_contiguous_layer_window: {contiguous_layer_window}");
    println!("    table_builder_duplicate_layer_indices: false");
    println!("    table_builder_named_19_20_constructor_removed: true");
    println!("    table_builder_registry_complete: {registry_complete}");

    Ok(QwenResidentLayerRunnerDenseLayerResourceSlotTable {
        slots,
        first_layer_idx,
        last_layer_idx,
        registry_complete,
    })
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceFrame<'a> {
    pub(crate) layer_idx: u32,
    pub(crate) qkv: &'a QwenQkvProjStagePlan,
    pub(crate) qk: &'a QwenQkNormRopeStagePlan,
    pub(crate) cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    pub(crate) attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    pub(crate) o_proj_plan: Option<&'a QwenOProjStagePlan>,
    pub(crate) post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    pub(crate) peer_plan: &'a QwenTpOProjPeerStagePlan,
    pub(crate) mlp: &'a QwenMlpStagePlan,
    pub(crate) peer_mlp: &'a QwenTpMlpPeerStagePlan,
    pub(crate) next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerStepPlanHandle<'a> {
    pub(crate) layer_idx: u32,
    pub(crate) qkv: &'a QwenQkvProjStagePlan,
    pub(crate) qk: &'a QwenQkNormRopeStagePlan,
    pub(crate) cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    pub(crate) attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    pub(crate) o_proj_plan: Option<&'a QwenOProjStagePlan>,
    pub(crate) post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    pub(crate) peer_plan: &'a QwenTpOProjPeerStagePlan,
    pub(crate) mlp: &'a QwenMlpStagePlan,
    pub(crate) peer_mlp: &'a QwenTpMlpPeerStagePlan,
    pub(crate) next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
    pub(crate) attention_dependency_id: QwenResidentLayerRunnerDependencyId,
    pub(crate) mlp_dependency_id: QwenResidentLayerRunnerDependencyId,
}

#[derive(Clone, Copy)]
enum QwenResidentLayerRunnerDenseLayerStepCursorOp {
    DenseStep,
}

impl QwenResidentLayerRunnerDenseLayerStepCursorOp {
    fn label(self) -> &'static str {
        match self {
            Self::DenseStep => "dense_step",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a> {
    cursor_index: usize,
    table_len: usize,
    op: QwenResidentLayerRunnerDenseLayerStepCursorOp,
    pub(crate) handle: QwenResidentLayerRunnerDenseLayerStepPlanHandle<'a>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_step_cursor_frame_from_index<'a>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    layer_idx: u32,
    qkv: &'a QwenQkvProjStagePlan,
    qk: &'a QwenQkNormRopeStagePlan,
    cache_plan: Option<&'a QwenFp4KvCacheStagePlan>,
    attn_plan: Option<&'a QwenFp4SingleRowAttentionStagePlan>,
    o_proj_plan: Option<&'a QwenOProjStagePlan>,
    post_attn_norm: Option<&'a QwenPostAttnNormStagePlan>,
    peer_plan: &'a QwenTpOProjPeerStagePlan,
    mlp: &'a QwenMlpStagePlan,
    peer_mlp: &'a QwenTpMlpPeerStagePlan,
    next_input_norm: Option<&'a QwenNextInputNormStagePlan>,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>> {
    let plan = runner_plans
        .iter()
        .find(|plan| plan.layer_idx == layer_idx)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident layer runner dense layer{layer_idx} indexed step handle missing layer plan descriptor"
            )
        })?;
    let attention_dispatch = plan.attention_dispatch.ok_or_else(|| {
        anyhow::anyhow!(
            "resident layer runner dense layer{layer_idx} indexed step handle missing attention dispatch plan"
        )
    })?;
    let mlp_dispatch = plan.mlp_dispatch.ok_or_else(|| {
        anyhow::anyhow!(
            "resident layer runner dense layer{layer_idx} indexed step handle missing MLP dispatch plan"
        )
    })?;
    if !qwen_resident_layer_runner_dispatch_plan_enabled(attention_dispatch) {
        anyhow::bail!(
            "resident layer runner dense layer{layer_idx} indexed step handle attention dispatch plan disabled"
        );
    }
    if !qwen_resident_layer_runner_dispatch_plan_enabled(mlp_dispatch) {
        anyhow::bail!(
            "resident layer runner dense layer{layer_idx} indexed step handle MLP dispatch plan disabled"
        );
    }

    Ok(QwenResidentLayerRunnerDenseLayerStepCursorFrame {
        cursor_index: 0,
        table_len: 1,
        op: QwenResidentLayerRunnerDenseLayerStepCursorOp::DenseStep,
        handle: QwenResidentLayerRunnerDenseLayerStepPlanHandle {
            layer_idx,
            qkv,
            qk,
            cache_plan,
            attn_plan,
            o_proj_plan,
            post_attn_norm,
            peer_plan,
            mlp,
            peer_mlp,
            next_input_norm,
            attention_dependency_id: attention_dispatch.dependency_id,
            mlp_dependency_id: mlp_dispatch.dependency_id,
        },
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_frame_from_index<
    'a,
    const N: usize,
>(
    resource_slot_table: &QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, N>,
    layer_idx: u32,
) -> anyhow::Result<Option<QwenResidentLayerRunnerDenseLayerResourceFrame<'a>>> {
    println!("  resident_layer_runner_dense_layer_resource_frame_builder_stage:");
    println!("    source: resident_layer_runner_indexed_dense_layer_resource_frame_builder");
    println!("    requested_layer_idx: {layer_idx}");
    println!("    typed_resource_slot_table_consumed: true");
    println!("    resource_frame_owned_by_resident_runner_module: true");
    println!(
        "    resource_table_first_layer_idx: {}",
        resource_slot_table.first_layer_idx
    );
    println!(
        "    resource_table_last_layer_idx: {}",
        resource_slot_table.last_layer_idx
    );
    println!(
        "    resource_table_registry_complete: {}",
        resource_slot_table.registry_complete
    );
    let resource_slots = resource_slot_table.slots.as_slice();
    println!("    resource_slot_count: {}", resource_slots.len());
    println!(
        "    resource_table_multi_entry: {}",
        resource_slots.len() > 1
    );

    let mut matching_slots = resource_slots
        .iter()
        .copied()
        .filter(|slots| slots.layer_idx == layer_idx);
    let Some(resource_slots) = matching_slots.next() else {
        println!("    resource_frame_found: false");
        return Ok(None);
    };
    if matching_slots.next().is_some() {
        anyhow::bail!("duplicate dense-layer resource slots for resident runner layer {layer_idx}");
    }

    println!("    resource_frame_found: true");
    println!("    resource_frame_indexed_lookup: true");

    let (Some(qkv), Some(qk), Some(peer_plan), Some(mlp), Some(peer_mlp)) = (
        resource_slots.qkv,
        resource_slots.qk,
        resource_slots.peer_plan,
        resource_slots.mlp,
        resource_slots.peer_mlp,
    ) else {
        println!("    required_resources_bound: false");
        return Ok(None);
    };

    println!("    required_resources_bound: true");
    println!("    resource_frame_bound_by_index: true");

    Ok(Some(QwenResidentLayerRunnerDenseLayerResourceFrame {
        layer_idx,
        qkv,
        qk,
        cache_plan: resource_slots.cache_plan,
        attn_plan: resource_slots.attn_plan,
        o_proj_plan: resource_slots.o_proj_plan,
        post_attn_norm: resource_slots.post_attn_norm,
        peer_plan,
        mlp,
        peer_mlp,
        next_input_norm: resource_slots.next_input_norm,
    }))
}

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerResourceLookupWindow<'a> {
    pub(crate) dispatch_frame: Option<QwenResidentLayerRunnerDenseLayerResourceFrame<'a>>,
    pub(crate) _lookahead_frame: Option<QwenResidentLayerRunnerDenseLayerResourceFrame<'a>>,
}

#[derive(Clone, Copy)]
enum QwenResidentLayerRunnerDenseLayerResourceLookupRole {
    LookaheadOnly,
    DispatchCandidate,
}

impl QwenResidentLayerRunnerDenseLayerResourceLookupRole {
    fn label(self) -> &'static str {
        match self {
            Self::LookaheadOnly => "lookahead_only",
            Self::DispatchCandidate => "dispatch_candidate",
        }
    }

    fn dispatch_deferred_to_cursor_table(self) -> bool {
        matches!(self, Self::DispatchCandidate)
    }
}

fn qwen_resident_layer_runner_dense_layer_resource_lookup_boundary<'a, const N: usize>(
    resource_slot_table: &QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, N>,
    layer_idx: u32,
    role: QwenResidentLayerRunnerDenseLayerResourceLookupRole,
) -> anyhow::Result<Option<QwenResidentLayerRunnerDenseLayerResourceFrame<'a>>> {
    let resource_frame = qwen_resident_layer_runner_dense_layer_resource_frame_from_index(
        resource_slot_table,
        layer_idx,
    )?;

    println!("  resident_layer_runner_dense_layer_resource_lookup_boundary_stage:");
    println!("    source: typed_dense_resource_slot_table_lookup_boundary");
    println!("    reusable_lookup_boundary: true");
    println!("    lookup_boundary_owned_by_resident_runner_module: true");
    println!("    requested_layer_idx: {layer_idx}");
    println!("    lookup_role: {}", role.label());
    println!("    lookup_materialized: {}", resource_frame.is_some());
    println!("    dispatch_submitted_by_boundary: false");
    println!(
        "    dispatch_submission_deferred_to_cursor_table: {}",
        role.dispatch_deferred_to_cursor_table()
    );
    println!("    layer_dispatch_path_changed: false");
    println!("    graph_capture_started: false");

    Ok(resource_frame)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_resource_lookup_window_from_dispatch_descriptor<
    'a,
    const N: usize,
>(
    resource_slot_table: &QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, N>,
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerResourceLookupWindow<'a>> {
    let lookahead_frame = qwen_resident_layer_runner_dense_layer_resource_lookup_boundary(
        resource_slot_table,
        dispatch_window.lookahead_layer_idx,
        QwenResidentLayerRunnerDenseLayerResourceLookupRole::LookaheadOnly,
    )?;
    let dispatch_frame = qwen_resident_layer_runner_dense_layer_resource_lookup_boundary(
        resource_slot_table,
        dispatch_window.dispatch_layer_idx,
        QwenResidentLayerRunnerDenseLayerResourceLookupRole::DispatchCandidate,
    )?;

    println!("  resident_layer_runner_dense_layer_descriptor_lookup_window_stage:");
    println!("    source: resident_runner_dense_dispatch_window_lookup_driver");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    descriptor_drives_lookup_boundaries: true");
    println!("    lookup_window_owned_by_resident_runner_module: true");
    println!("    lookup_window_size: 2");
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!("    lookahead_lookup_role: lookahead_only");
    println!("    dispatch_lookup_role: dispatch_candidate");
    println!(
        "    lookahead_lookup_materialized: {}",
        lookahead_frame.is_some()
    );
    println!(
        "    dispatch_lookup_materialized: {}",
        dispatch_frame.is_some()
    );
    println!("    dispatch_frame_returned_to_cursor_table: true");
    println!("    lookup_boundary_callsite_replaced: true");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerResourceLookupWindow {
        dispatch_frame,
        _lookahead_frame: lookahead_frame,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_step_cursor_table_from_dispatch_descriptor<
    'a,
    const N: usize,
>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    resource_slot_table: &QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, N>,
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
) -> anyhow::Result<Option<[QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>; 1]>> {
    let dense_layer_lookup_window =
        qwen_resident_layer_runner_dense_layer_resource_lookup_window_from_dispatch_descriptor(
            resource_slot_table,
            dispatch_window,
        )?;
    let Some(dispatch_frame) = dense_layer_lookup_window.dispatch_frame else {
        println!("  resident_layer_runner_dense_layer_descriptor_cursor_table_stage:");
        println!("    source: resident_runner_dense_dispatch_window_cursor_table_driver");
        println!("    dispatch_window_descriptor_consumed: true");
        println!("    descriptor_drives_cursor_table_binding: true");
        println!("    cursor_table_binding_owned_by_resident_runner_module: true");
        println!("    lookup_window_consumed_by_cursor_table: true");
        println!(
            "    dispatch_layer_idx: {}",
            dispatch_window.dispatch_layer_idx
        );
        println!(
            "    lookahead_layer_idx: {}",
            dispatch_window.lookahead_layer_idx
        );
        println!("    dispatch_frame_materialized: false");
        println!("    cursor_table_returned_to_dispatch: false");
        println!("    callsite_dispatch_frame_unwrap_replaced: true");
        println!("    dispatch_scope_changed: false");
        println!("    async_materialization_started: false");
        println!("    graph_capture_started: false");
        return Ok(None);
    };

    let cursor_table =
        qwen_resident_layer_runner_dense_layer_step_cursor_table_from_resource_frame(
            runner_plans,
            dispatch_frame,
        )?;

    println!("  resident_layer_runner_dense_layer_descriptor_cursor_table_stage:");
    println!("    source: resident_runner_dense_dispatch_window_cursor_table_driver");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    descriptor_drives_cursor_table_binding: true");
    println!("    cursor_table_binding_owned_by_resident_runner_module: true");
    println!("    lookup_window_consumed_by_cursor_table: true");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    dispatch_frame_materialized: true");
    println!("    cursor_table_len: {}", cursor_table.len());
    println!("    cursor_table_returned_to_dispatch: true");
    println!("    callsite_dispatch_frame_unwrap_replaced: true");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    Ok(Some(cursor_table))
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_dispatch_descriptor<
    'a,
    const N: usize,
    T,
    F,
>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    resource_slot_table: &QwenResidentLayerRunnerDenseLayerResourceSlotTable<'a, N>,
    dispatch_window: QwenResidentLayerRunnerDenseLayerDispatchWindow,
    dispatch_cursor_table: F,
) -> anyhow::Result<Option<T>>
where
    F: FnOnce([QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>; 1]) -> anyhow::Result<T>,
{
    let Some(cursor_table) =
        qwen_resident_layer_runner_dense_layer_step_cursor_table_from_dispatch_descriptor(
            runner_plans,
            resource_slot_table,
            dispatch_window,
        )?
    else {
        println!("  resident_layer_runner_dense_layer_descriptor_dispatch_boundary_stage:");
        println!("    source: resident_runner_dense_dispatch_window_dispatch_boundary");
        println!("    dispatch_window_descriptor_consumed: true");
        println!("    descriptor_drives_cursor_table_binding: true");
        println!("    dispatch_boundary_owned_by_resident_runner_module: true");
        println!("    cursor_table_dispatched_by_descriptor_boundary: false");
        println!(
            "    dispatch_layer_idx: {}",
            dispatch_window.dispatch_layer_idx
        );
        println!(
            "    lookahead_layer_idx: {}",
            dispatch_window.lookahead_layer_idx
        );
        println!("    cursor_table_materialized: false");
        println!("    dispatch_result_returned_to_callsite: false");
        println!("    callsite_cursor_table_dispatch_replaced: true");
        println!("    dispatch_scope_changed: false");
        println!("    async_materialization_started: false");
        println!("    graph_capture_started: false");
        return Ok(None);
    };

    println!("  resident_layer_runner_dense_layer_descriptor_dispatch_boundary_stage:");
    println!("    source: resident_runner_dense_dispatch_window_dispatch_boundary");
    println!("    dispatch_window_descriptor_consumed: true");
    println!("    descriptor_drives_cursor_table_binding: true");
    println!("    dispatch_boundary_owned_by_resident_runner_module: true");
    println!("    cursor_table_dispatched_by_descriptor_boundary: true");
    println!(
        "    dispatch_layer_idx: {}",
        dispatch_window.dispatch_layer_idx
    );
    println!(
        "    lookahead_layer_idx: {}",
        dispatch_window.lookahead_layer_idx
    );
    println!("    cursor_table_materialized: true");
    println!("    cursor_table_len: {}", cursor_table.len());
    println!("    dispatch_result_returned_to_callsite: true");
    println!("    callsite_cursor_table_dispatch_replaced: true");
    println!("    dispatch_scope_changed: false");
    println!("    async_materialization_started: false");
    println!("    graph_capture_started: false");

    dispatch_cursor_table(cursor_table).map(Some)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_context_window<
    'a,
    const TABLE_ROWS: usize,
    T,
    F,
>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'a,
        TABLE_ROWS,
    >,
    target_layer_idx: u32,
    dispatch_cursor_table: F,
) -> anyhow::Result<Option<T>>
where
    F: FnOnce([QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>; 1]) -> anyhow::Result<T>,
{
    let dense_layer_dispatch_context =
        qwen_resident_layer_runner_dense_layer_dispatch_context_from_window(
            target_layer_idx,
            dispatch_context_window,
        )?;
    let dense_layer_dispatch_window = dense_layer_dispatch_context.dispatch_window;
    let dense_layer_resource_slots = dense_layer_dispatch_context.resource_slot_table;

    println!("  resident_layer_runner_dense_layer_context_window_dispatch_boundary_stage:");
    println!("    source: resident_runner_dense_dispatch_context_window_dispatch_boundary");
    println!("    dispatch_context_window_owned_by_resident_runner_module: true");
    println!("    dispatch_context_window_consumed_by_dispatch_boundary: true");
    println!("    target_layer_idx: {target_layer_idx}");
    println!(
        "    selected_dispatch_layer_idx: {}",
        dense_layer_dispatch_window.dispatch_layer_idx
    );
    println!(
        "    selected_lookahead_layer_idx: {}",
        dense_layer_dispatch_window.lookahead_layer_idx
    );
    println!(
        "    selected_slot_table_first_layer_idx: {}",
        dense_layer_resource_slots.first_layer_idx
    );
    println!(
        "    selected_slot_table_last_layer_idx: {}",
        dense_layer_resource_slots.last_layer_idx
    );
    println!("    context_select_and_descriptor_dispatch_bound: true");
    println!("    main_dispatch_context_manual_destructure_removed: true");
    println!("    main_descriptor_dispatch_callsite_replaced: true");
    println!("    manual_layer_order_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_dispatch_from_dispatch_descriptor(
        runner_plans,
        &dense_layer_resource_slots,
        dense_layer_dispatch_window,
        dispatch_cursor_table,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_target_indices_from_context_window<
    const TABLE_ROWS: usize,
>(
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        '_,
        TABLE_ROWS,
    >,
) -> anyhow::Result<Vec<u32>> {
    let target_rows = dispatch_context_window.dispatch_windows.len();
    if target_rows == 0 {
        anyhow::bail!("resident dense dispatch target index window requires at least one row");
    }

    let mut seen_layers = std::collections::BTreeSet::new();
    let mut target_layer_indices = Vec::with_capacity(target_rows);
    let mut first_layer_idx = u32::MAX;
    let mut last_layer_idx = 0u32;
    let mut previous_layer_idx = None;
    for (row_index, dispatch_window) in dispatch_context_window
        .dispatch_windows
        .iter()
        .copied()
        .enumerate()
    {
        let dispatch_layer_idx = dispatch_window.dispatch_layer_idx;
        if !seen_layers.insert(dispatch_layer_idx) {
            anyhow::bail!(
                "resident dense dispatch target index window contains duplicate layer {}",
                dispatch_layer_idx
            );
        }
        if let Some(previous_layer_idx) = previous_layer_idx {
            if dispatch_layer_idx <= previous_layer_idx {
                anyhow::bail!(
                    "resident dense dispatch target index window is not strictly ascending: layer {} after layer {} at row {}",
                    dispatch_layer_idx,
                    previous_layer_idx,
                    row_index
                );
            }
        }
        previous_layer_idx = Some(dispatch_layer_idx);
        first_layer_idx = first_layer_idx.min(dispatch_layer_idx);
        last_layer_idx = last_layer_idx.max(dispatch_layer_idx);
        target_layer_indices.push(dispatch_layer_idx);
    }

    let contiguous_layer_window =
        last_layer_idx - first_layer_idx + 1 == u32::try_from(target_rows).unwrap_or(u32::MAX);

    println!("  resident_layer_runner_dense_layer_dispatch_target_indices_stage:");
    println!("    source: resident_runner_dense_dispatch_target_indices_from_context_window");
    println!("    dispatch_context_window_owned_by_resident_runner_module: true");
    println!("    dispatch_context_window_consumed_for_target_indices: true");
    println!("    dispatch_target_rows: {target_rows}");
    println!("    dispatch_target_first_layer_idx: {first_layer_idx}");
    println!("    dispatch_target_last_layer_idx: {last_layer_idx}");
    println!("    contiguous_layer_window: {contiguous_layer_window}");
    println!("    dispatch_target_indices_strictly_ascending: true");
    println!("    dispatch_target_indices_duplicate_free: true");
    println!("    dispatch_target_indices_derived_from_context_window: true");
    println!("    dispatch_context_window_rows_dynamic_vec: true");
    println!("    main_dispatch_target_layer_literals_removed: true");
    println!("    manual_layer_order_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(target_layer_indices)
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchTargetCursor {
    target_layer_indices: Vec<u32>,
}

#[derive(Clone, Copy)]
pub(crate) enum QwenResidentLayerRunnerDenseLayerDispatchTargetRole {
    FirstDenseDispatch,
    SecondDenseDispatch,
    ThirdDenseDispatch,
    FourthDenseDispatch,
}

impl QwenResidentLayerRunnerDenseLayerDispatchTargetRole {
    fn label(self) -> &'static str {
        match self {
            Self::FirstDenseDispatch => "first_dense_dispatch",
            Self::SecondDenseDispatch => "second_dense_dispatch",
            Self::ThirdDenseDispatch => "third_dense_dispatch",
            Self::FourthDenseDispatch => "fourth_dense_dispatch",
        }
    }

    fn row_index(self) -> usize {
        match self {
            Self::FirstDenseDispatch => 0,
            Self::SecondDenseDispatch => 1,
            Self::ThirdDenseDispatch => 2,
            Self::FourthDenseDispatch => 3,
        }
    }
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_target_cursor_from_context_window<
    const TABLE_ROWS: usize,
>(
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        '_,
        TABLE_ROWS,
    >,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchTargetCursor> {
    let target_layer_indices =
        qwen_resident_layer_runner_dense_layer_dispatch_target_indices_from_context_window(
            dispatch_context_window,
        )?;
    let target_rows = target_layer_indices.len();
    let first_layer_idx = target_layer_indices[0];
    let last_layer_idx = target_layer_indices[target_rows - 1];

    println!("  resident_layer_runner_dense_layer_dispatch_target_cursor_stage:");
    println!("    source: resident_runner_dense_dispatch_target_cursor_from_context_window");
    println!("    dispatch_context_window_owned_by_resident_runner_module: true");
    println!("    dispatch_target_indices_derived_from_context_window: true");
    println!("    cursor_target_rows: {target_rows}");
    println!("    cursor_first_target_layer_idx: {first_layer_idx}");
    println!("    cursor_last_target_layer_idx: {last_layer_idx}");
    println!("    cursor_stateful: false");
    println!("    cursor_reset_required: false");
    println!("    main_dispatch_target_row_index_variables_removed: true");
    println!("    main_dispatch_target_layer_literals_removed: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchTargetCursor {
        target_layer_indices,
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_target_from_cursor_role(
    cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
) -> anyhow::Result<u32> {
    let target_rows = cursor.target_layer_indices.len();
    if target_rows == 0 {
        anyhow::bail!("resident dense dispatch target cursor requires at least one row");
    }
    let row_index = dispatch_target_role.row_index();
    let target_layer_idx = cursor
        .target_layer_indices
        .get(row_index)
        .copied()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense dispatch target cursor row {} missing from {} rows",
                row_index,
                target_rows
            )
        })?;
    let first_layer_idx = cursor.target_layer_indices[0];
    let last_layer_idx = cursor.target_layer_indices[target_rows - 1];

    println!("  resident_layer_runner_dense_layer_dispatch_target_index_select_stage:");
    println!("    source: resident_runner_dense_dispatch_target_from_cursor_role");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    target_rows: {target_rows}");
    println!("    row_index: {row_index}");
    println!("    selected_target_layer_idx: {target_layer_idx}");
    println!("    first_target_layer_idx: {first_layer_idx}");
    println!("    last_target_layer_idx: {last_layer_idx}");
    println!("    dispatch_target_indices_derived_from_context_window: true");
    println!("    selected_target_layer_idx_derived_from_cursor_role: true");
    println!("    cursor_stateful: false");
    println!("    cursor_reset_required: false");
    println!("    main_dispatch_target_row_index_variables_removed: true");
    println!("    main_dispatch_target_layer_literal_removed: true");
    println!("    manual_layer_order_retained: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(target_layer_idx)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role<
    'a,
    const TABLE_ROWS: usize,
    T,
    F,
>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'a,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    dispatch_cursor_table: F,
) -> anyhow::Result<Option<T>>
where
    F: FnOnce(
        [QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>; 1],
        QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    ) -> anyhow::Result<T>,
{
    let target_layer_idx = qwen_resident_layer_runner_dense_layer_dispatch_target_from_cursor_role(
        target_cursor,
        dispatch_target_role,
    )?;

    println!("  resident_layer_runner_dense_layer_dispatch_cursor_role_boundary_stage:");
    println!("    source: resident_runner_dense_dispatch_from_context_window_cursor_role");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    selected_target_layer_idx: {target_layer_idx}");
    println!("    dispatch_target_indices_derived_from_context_window: true");
    println!("    target_role_lookup_bound_to_context_dispatch: true");
    println!("    dispatch_target_role_forwarded_to_cursor_table: true");
    println!("    main_target_lookup_callsite_removed: true");
    println!("    main_dispatch_calls_unrolled: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_dispatch_from_context_window(
        runner_plans,
        dispatch_context_window,
        target_layer_idx,
        |cursor_table| dispatch_cursor_table(cursor_table, dispatch_target_role),
    )
}

struct QwenResidentLayerRunnerDenseLayerDispatchIoOptions<'a> {
    attention_allreduce_nodes: Option<&'a [u32]>,
    residual_replicas: Option<&'a mut [mcore::DeviceBuffer]>,
}

fn qwen_resident_layer_runner_dense_layer_dispatch_io_options_from_cursor_table_role<'a>(
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    cursor_table: &[QwenResidentLayerRunnerDenseLayerStepCursorFrame<'_>],
    tp_allreduce_node_order: &'a [u32],
    residual_replicas: &'a mut [mcore::DeviceBuffer],
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchIoOptions<'a>> {
    let target_layer_idx = qwen_resident_layer_runner_dense_layer_dispatch_target_from_cursor_role(
        target_cursor,
        dispatch_target_role,
    )?;
    if cursor_table.len() != 1 {
        anyhow::bail!(
            "resident dense dispatch I/O options expected one cursor row for {}, got {}",
            dispatch_target_role.label(),
            cursor_table.len()
        );
    }
    let cursor_frame = &cursor_table[0];
    let cursor_layer_idx = cursor_frame.handle.layer_idx;
    if cursor_layer_idx != target_layer_idx {
        anyhow::bail!(
            "resident dense dispatch I/O options cursor layer {} does not match selected target layer {} for {}",
            cursor_layer_idx,
            target_layer_idx,
            dispatch_target_role.label()
        );
    }
    let o_proj_plan_present = cursor_frame.handle.o_proj_plan.is_some();
    let attention_allreduce_nodes = if o_proj_plan_present {
        Some(tp_allreduce_node_order)
    } else {
        None
    };
    let attention_allreduce_nodes_present = attention_allreduce_nodes.is_some();
    let residual_replica_rows = residual_replicas.len();

    println!("  resident_layer_runner_dense_layer_dispatch_io_options_stage:");
    println!("    source: resident_runner_dense_dispatch_io_options_from_cursor_table_role");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    selected_target_layer_idx: {target_layer_idx}");
    println!("    cursor_table_len: {}", cursor_table.len());
    println!("    cursor_layer_idx: {cursor_layer_idx}");
    println!("    o_proj_plan_present: {o_proj_plan_present}");
    println!("    o_proj_plan_present_source: dense_step_cursor_frame_handle");
    println!("    attention_allreduce_nodes_present: {attention_allreduce_nodes_present}");
    println!(
        "    tp_allreduce_node_order_len: {}",
        tp_allreduce_node_order.len()
    );
    println!("    residual_replicas_present: true");
    println!("    residual_replica_rows: {residual_replica_rows}");
    println!("    dispatch_target_indices_derived_from_context_window: true");
    println!("    target_role_lookup_bound_to_dispatch_io_options: true");
    println!("    target_role_lookup_bound_to_allreduce_nodes: true");
    println!("    target_role_lookup_bound_to_residual_replicas: true");
    println!("    cursor_table_bound_to_dispatch_io_options: true");
    println!("    cursor_table_layer_matches_selected_target: true");
    println!("    main_per_layer_o_proj_option_branch_removed: true");
    println!("    main_per_layer_o_proj_plan_presence_arg_removed: true");
    println!("    main_per_layer_residual_replicas_option_removed: true");
    println!("    separate_main_dispatch_io_option_callsites_removed: true");
    println!("    main_dispatch_calls_unrolled: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchIoOptions {
        attention_allreduce_nodes,
        residual_replicas: Some(residual_replicas),
    })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_and_update_runtime_readiness<
    'a,
    const TABLE_ROWS: usize,
    T,
    F,
>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'a,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    runtime_layer_state_tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    readiness_update_selection: QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection,
    dispatch_cursor_table: F,
) -> anyhow::Result<Option<T>>
where
    F: FnOnce(
        [QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>; 1],
        QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    ) -> anyhow::Result<T>,
{
    let dispatch_result =
        qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role(
            runner_plans,
            dispatch_context_window,
            target_cursor,
            dispatch_target_role,
            dispatch_cursor_table,
        )?;
    let dispatch_result_present = dispatch_result.is_some();

    println!("  resident_layer_runner_dense_layer_dispatch_role_readiness_boundary_stage:");
    println!("    source: resident_runner_dense_dispatch_role_boundary_readiness_update");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!(
        "    readiness_update_selection: {}",
        readiness_update_selection.label()
    );
    println!("    readiness_update_bound_to_dispatch_role: true");
    println!("    readiness_update_after_dispatch_result: true");
    println!("    dispatch_result_present: {dispatch_result_present}");
    println!("    readiness_update_applied: {dispatch_result_present}");
    println!("    main_last_attention_readiness_update_callsite_removed: true");
    println!("    main_dispatch_calls_unrolled: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    if dispatch_result_present {
        qwen_resident_layer_runner_update_runtime_layer_state_tracker_from_plan_descriptor_selection(
            runtime_layer_state_tracker,
            runner_plans,
            readiness_update_selection,
        )?;
    }

    Ok(dispatch_result)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_and_apply_legacy_window<
    'dispatch,
    'legacy,
    const TABLE_ROWS: usize,
    F,
>(
    runner_plans: &'dispatch [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'dispatch,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    legacy_apply_window: QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'legacy>,
    dispatch_cursor_table: F,
) -> anyhow::Result<bool>
where
    F: FnOnce(
        [QwenResidentLayerRunnerDenseLayerStepCursorFrame<'dispatch>; 1],
        QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    ) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchResult>,
{
    let target_layer_idx = qwen_resident_layer_runner_dense_layer_dispatch_target_from_cursor_role(
        target_cursor,
        dispatch_target_role,
    )?;

    println!("  resident_layer_runner_dense_layer_dispatch_apply_boundary_stage:");
    println!("    source: resident_runner_dense_dispatch_role_boundary_legacy_apply");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    selected_target_layer_idx: {target_layer_idx}");
    println!("    dispatch_target_indices_derived_from_context_window: true");
    println!("    target_role_lookup_bound_to_context_dispatch: true");
    println!("    dispatch_target_role_forwarded_to_cursor_table: true");
    println!("    legacy_apply_bound_to_dispatch_role: true");
    println!("    main_post_dispatch_legacy_apply_callsite_removed: true");
    println!("    main_apply_target_layer_literal_removed: true");
    println!("    main_dispatch_calls_unrolled: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let dispatch_result = qwen_resident_layer_runner_dense_layer_dispatch_from_context_window(
        runner_plans,
        dispatch_context_window,
        target_layer_idx,
        |cursor_table| dispatch_cursor_table(cursor_table, dispatch_target_role),
    )?;
    let dispatch_result_present = dispatch_result.is_some();

    println!("  resident_layer_runner_dense_layer_dispatch_apply_result_stage:");
    println!("    source: resident_runner_dense_dispatch_role_boundary_legacy_apply_result");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    selected_target_layer_idx: {target_layer_idx}");
    println!("    dispatch_result_present: {dispatch_result_present}");
    println!("    legacy_apply_after_dispatch_result: true");
    println!("    legacy_apply_uses_role_target_layer_idx: true");
    println!("    main_post_dispatch_legacy_apply_callsite_removed: true");
    println!("    main_apply_target_layer_literal_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_apply_dense_layer_dispatch_result_to_legacy_stage_window(
        dispatch_result,
        hbm_state,
        target_layer_idx,
        legacy_apply_window,
    )?;

    Ok(dispatch_result_present)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_apply_next_decode_dense_cursor_table<
    'dispatch,
    'legacy,
    const TABLE_ROWS: usize,
>(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &'dispatch [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'dispatch,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    legacy_apply_window: QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'legacy>,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    next_input_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
) -> anyhow::Result<bool> {
    println!(
        "  resident_layer_runner_dense_layer_dispatch_apply_dense_cursor_table_boundary_stage:"
    );
    println!("    source: resident_runner_dense_dispatch_apply_dense_cursor_table_boundary");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    context_window_role_apply_bound_to_dense_cursor_table_dispatch: true");
    println!("    dispatch_io_options_bound_inside_cursor_table_dispatch_helper: true");
    println!("    legacy_apply_after_dense_cursor_table_dispatch: true");
    println!("    main_dense_cursor_table_dispatch_closure_removed: true");
    println!("    main_dispatch_calls_unrolled: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_and_apply_legacy_window(
        runner_plans,
        dispatch_context_window,
        target_cursor,
        dispatch_target_role,
        hbm_state,
        legacy_apply_window,
        |dense_step_cursor_table, dense_dispatch_target_role| {
            qwen_resident_layer_runner_dispatch_next_decode_dense_layer_from_cursor_table(
                runner_entries,
                runner_plans,
                dense_step_cursor_table.as_slice(),
                dag_scheduler,
                dev,
                peer_workspaces,
                target_cursor,
                dense_dispatch_target_role,
                tp_allreduce_node_order,
                residual_replicas,
                post_attn_norm_outputs,
                input_norm_outputs,
                next_input_norm_outputs,
                position,
                tp_world,
                h_size,
                h_bytes,
                poison_h,
            )
        },
    )
}

#[derive(Clone, Copy)]
struct QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepDescriptor {
    sequence_step_index: usize,
    sequence_stage_label: &'static str,
    source_label: &'static str,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    post_attn_norm_layer_idx: u32,
    input_norm_layer_idx: u32,
    next_input_norm_layer_idx: u32,
    main_callsite_removed_label: &'static str,
    future_sequence_roles: &'static str,
}

const QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS:
    [QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepDescriptor; 3] = [
    QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepDescriptor {
        sequence_step_index: 0,
        sequence_stage_label:
            "resident_layer_runner_dense_layer_dispatch_apply_sequence_indexed_step_stage",
        source_label: "resident_runner_dense_dispatch_indexed_apply_sequence_step",
        dispatch_target_role:
            QwenResidentLayerRunnerDenseLayerDispatchTargetRole::FirstDenseDispatch,
        post_attn_norm_layer_idx: 19,
        input_norm_layer_idx: 19,
        next_input_norm_layer_idx: 20,
        main_callsite_removed_label: "main_first_apply_cursor_table_callsite_removed",
        future_sequence_roles: "second_dense_dispatch,third_dense_dispatch",
    },
    QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepDescriptor {
        sequence_step_index: 1,
        sequence_stage_label:
            "resident_layer_runner_dense_layer_dispatch_apply_sequence_indexed_step_stage",
        source_label: "resident_runner_dense_dispatch_indexed_apply_sequence_step",
        dispatch_target_role:
            QwenResidentLayerRunnerDenseLayerDispatchTargetRole::SecondDenseDispatch,
        post_attn_norm_layer_idx: 20,
        input_norm_layer_idx: 20,
        next_input_norm_layer_idx: 21,
        main_callsite_removed_label: "main_second_apply_cursor_table_callsite_removed",
        future_sequence_roles: "third_dense_dispatch",
    },
    QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepDescriptor {
        sequence_step_index: 2,
        sequence_stage_label:
            "resident_layer_runner_dense_layer_dispatch_apply_sequence_indexed_step_stage",
        source_label: "resident_runner_dense_dispatch_indexed_apply_sequence_step",
        dispatch_target_role:
            QwenResidentLayerRunnerDenseLayerDispatchTargetRole::ThirdDenseDispatch,
        post_attn_norm_layer_idx: 21,
        input_norm_layer_idx: 21,
        next_input_norm_layer_idx: 22,
        main_callsite_removed_label: "main_third_apply_cursor_table_callsite_removed",
        future_sequence_roles: "none",
    },
];

#[derive(Clone, Copy)]
pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindowRow {
    sequence_window_row_index: usize,
    sequence_step_index: usize,
    post_attn_norm_layer_idx: u32,
    input_norm_layer_idx: u32,
    next_input_norm_layer_idx: u32,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindow {
    rows: Vec<QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindowRow>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepCursor {
    next_window_row_index: usize,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferRow<'a> {
    post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &'a mut [mcore::DeviceBuffer],
    next_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
}

pub(crate) struct QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferWindow<'a> {
    layer19_post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer19_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer20_post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer20_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer21_post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer21_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer22_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
}

fn qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_descriptor_for_index(
    sequence_step_index: usize,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepDescriptor> {
    let descriptor = QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS
        .iter()
        .copied()
        .find(|descriptor| descriptor.sequence_step_index == sequence_step_index)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident dense apply sequence step index {} outside descriptor table rows {}",
                sequence_step_index,
                QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS.len()
            )
        })?;
    let role_row_index = descriptor.dispatch_target_role.row_index();
    if role_row_index != sequence_step_index {
        anyhow::bail!(
            "resident dense apply sequence descriptor index {} does not match role row index {}",
            sequence_step_index,
            role_row_index
        );
    }
    Ok(descriptor)
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_window_from_descriptor_table(
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindow> {
    let descriptor_rows =
        QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS.len();
    let mut rows = Vec::with_capacity(descriptor_rows);
    for (sequence_window_row_index, descriptor) in
        QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS
            .iter()
            .copied()
            .enumerate()
    {
        if descriptor.sequence_step_index != sequence_window_row_index {
            anyhow::bail!(
                "resident dense apply sequence descriptor row {} does not match step index {}",
                sequence_window_row_index,
                descriptor.sequence_step_index
            );
        }
        let role_row_index = descriptor.dispatch_target_role.row_index();
        if role_row_index != descriptor.sequence_step_index {
            anyhow::bail!(
                "resident dense apply sequence descriptor step index {} does not match role row index {}",
                descriptor.sequence_step_index,
                role_row_index
            );
        }
        rows.push(
            QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindowRow {
                sequence_window_row_index,
                sequence_step_index: descriptor.sequence_step_index,
                post_attn_norm_layer_idx: descriptor.post_attn_norm_layer_idx,
                input_norm_layer_idx: descriptor.input_norm_layer_idx,
                next_input_norm_layer_idx: descriptor.next_input_norm_layer_idx,
            },
        );
    }

    println!("  resident_layer_runner_dense_layer_dispatch_apply_sequence_step_window_stage:");
    println!("    source: resident_runner_dense_dispatch_apply_sequence_step_window_from_descriptor_table");
    println!("    window_row_count: {descriptor_rows}");
    println!("    first_sequence_step_index: 0");
    println!(
        "    last_sequence_step_index: {}",
        descriptor_rows.saturating_sub(1)
    );
    println!("    descriptor_table_rows_bound: true");
    println!("    descriptor_rows_strictly_ascending: true");
    println!("    descriptor_role_rows_match_sequence_indices: true");
    println!("    buffer_window_rows_bound_to_sequence_metadata: true");
    println!("    buffer_window_first_row_layers: post_attn=19,input=19,next_input=20");
    println!("    buffer_window_second_row_layers: post_attn=20,input=20,next_input=21");
    println!("    buffer_window_third_row_layers: post_attn=21,input=21,next_input=22");
    println!("    sequence_step_window_constructed: true");
    println!("    main_apply_sequence_step_index_literals_removed: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindow { rows })
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_cursor_for_window(
    sequence_step_window: &QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindow,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepCursor> {
    let window_row_count = sequence_step_window.rows.len();
    let descriptor_rows =
        QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS.len();
    if window_row_count != descriptor_rows {
        anyhow::bail!(
            "resident dense apply sequence cursor window rows {} do not match descriptor rows {}",
            window_row_count,
            descriptor_rows
        );
    }

    println!("  resident_layer_runner_dense_layer_dispatch_apply_sequence_step_cursor_stage:");
    println!("    source: resident_runner_dense_dispatch_apply_sequence_step_cursor_for_window");
    println!("    window_row_count: {window_row_count}");
    println!("    next_window_row_index: 0");
    println!("    sequence_step_window_bound_to_cursor: true");
    println!("    main_apply_sequence_step_index_literals_removed: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(
        QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepCursor {
            next_window_row_index: 0,
        },
    )
}

fn qwen_resident_layer_runner_dense_layer_dispatch_next_apply_sequence_step_window_row(
    sequence_step_window: &QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindow,
    sequence_step_cursor: &mut QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepCursor,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindowRow> {
    let window_row_count = sequence_step_window.rows.len();
    if sequence_step_cursor.next_window_row_index >= window_row_count {
        anyhow::bail!(
            "resident dense apply sequence cursor exhausted at row {} of {}",
            sequence_step_cursor.next_window_row_index,
            window_row_count
        );
    }
    let sequence_window_row_index = sequence_step_cursor.next_window_row_index;
    let row = sequence_step_window.rows[sequence_window_row_index];
    let descriptor =
        qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_descriptor_for_index(
            row.sequence_step_index,
        )?;
    sequence_step_cursor.next_window_row_index += 1;

    println!("  resident_layer_runner_dense_layer_dispatch_apply_sequence_step_cursor_next_stage:");
    println!("    source: resident_runner_dense_dispatch_next_apply_sequence_step_window_row");
    println!("    sequence_window_row_index: {sequence_window_row_index}");
    println!("    sequence_step_index: {}", row.sequence_step_index);
    println!(
        "    dispatch_role: {}",
        descriptor.dispatch_target_role.label()
    );
    println!(
        "    next_window_row_index_after_advance: {}",
        sequence_step_cursor.next_window_row_index
    );
    println!("    sequence_step_window_cursor_used: true");
    println!("    sequence_step_index_selected_from_window: true");
    println!("    sequence_step_index_validated_against_role_row: true");
    println!("    buffer_window_row_metadata_selected: true");
    println!("    main_apply_sequence_step_index_literals_removed: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(row)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_buffer_window_from_layer19_22_outputs<
    'a,
>(
    layer19_post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer19_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer20_post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer20_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer21_post_attn_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer21_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
    layer22_input_norm_outputs: &'a mut Vec<mcore::DeviceBuffer>,
) -> QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferWindow<'a> {
    println!("  resident_layer_runner_dense_layer_dispatch_apply_sequence_buffer_window_stage:");
    println!("    source: resident_runner_dense_dispatch_apply_sequence_buffer_window_from_layer19_22_outputs");
    println!("    layer19_post_attn_norm_outputs_bound: true");
    println!("    layer19_input_norm_outputs_bound: true");
    println!("    layer20_post_attn_norm_outputs_bound: true");
    println!("    layer20_input_norm_outputs_bound: true");
    println!("    layer21_post_attn_norm_outputs_bound: true");
    println!("    layer21_input_norm_outputs_bound: true");
    println!("    layer22_input_norm_outputs_bound: true");
    println!("    buffer_window_rows_bound_to_sequence_metadata: true");
    println!("    main_apply_sequence_per_call_buffer_row_selection_removed: true");
    println!("    main_apply_sequence_driver_buffer_triple_args_removed: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferWindow {
        layer19_post_attn_norm_outputs,
        layer19_input_norm_outputs,
        layer20_post_attn_norm_outputs,
        layer20_input_norm_outputs,
        layer21_post_attn_norm_outputs,
        layer21_input_norm_outputs,
        layer22_input_norm_outputs,
    }
}

fn qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_buffer_row_from_window<'a>(
    sequence_step_row: QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindowRow,
    apply_sequence_buffer_window: QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferWindow<
        'a,
    >,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferRow<'a>> {
    let descriptor =
        qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_descriptor_for_index(
            sequence_step_row.sequence_step_index,
        )?;
    if descriptor.post_attn_norm_layer_idx != sequence_step_row.post_attn_norm_layer_idx
        || descriptor.input_norm_layer_idx != sequence_step_row.input_norm_layer_idx
        || descriptor.next_input_norm_layer_idx != sequence_step_row.next_input_norm_layer_idx
    {
        anyhow::bail!(
            "resident dense apply sequence buffer metadata row {} did not match descriptor step {}",
            sequence_step_row.sequence_window_row_index,
            sequence_step_row.sequence_step_index
        );
    }

    let QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferWindow {
        layer19_post_attn_norm_outputs,
        layer19_input_norm_outputs,
        layer20_post_attn_norm_outputs,
        layer20_input_norm_outputs,
        layer21_post_attn_norm_outputs,
        layer21_input_norm_outputs,
        layer22_input_norm_outputs,
    } = apply_sequence_buffer_window;

    let buffer_row = match (
        sequence_step_row.post_attn_norm_layer_idx,
        sequence_step_row.input_norm_layer_idx,
        sequence_step_row.next_input_norm_layer_idx,
    ) {
        (19, 19, 20) => QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferRow {
            post_attn_norm_outputs: layer19_post_attn_norm_outputs,
            input_norm_outputs: layer19_input_norm_outputs.as_mut_slice(),
            next_input_norm_outputs: layer20_input_norm_outputs,
        },
        (20, 20, 21) => QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferRow {
            post_attn_norm_outputs: layer20_post_attn_norm_outputs,
            input_norm_outputs: layer20_input_norm_outputs.as_mut_slice(),
            next_input_norm_outputs: layer21_input_norm_outputs,
        },
        (21, 21, 22) => QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferRow {
            post_attn_norm_outputs: layer21_post_attn_norm_outputs,
            input_norm_outputs: layer21_input_norm_outputs.as_mut_slice(),
            next_input_norm_outputs: layer22_input_norm_outputs,
        },
        (post_attn_norm_layer_idx, input_norm_layer_idx, next_input_norm_layer_idx) => {
            anyhow::bail!(
                "resident dense apply sequence buffer metadata row {} has unsupported layer triple post_attn={}, input={}, next_input={}",
                sequence_step_row.sequence_window_row_index,
                post_attn_norm_layer_idx,
                input_norm_layer_idx,
                next_input_norm_layer_idx
            );
        }
    };

    println!(
        "  resident_layer_runner_dense_layer_dispatch_apply_sequence_buffer_window_row_stage:"
    );
    println!("    source: resident_runner_dense_dispatch_apply_sequence_buffer_row_from_window");
    println!(
        "    sequence_window_row_index: {}",
        sequence_step_row.sequence_window_row_index
    );
    println!(
        "    sequence_step_index: {}",
        sequence_step_row.sequence_step_index
    );
    println!(
        "    dispatch_role: {}",
        descriptor.dispatch_target_role.label()
    );
    println!(
        "    post_attn_norm_layer_idx: {}",
        sequence_step_row.post_attn_norm_layer_idx
    );
    println!(
        "    input_norm_layer_idx: {}",
        sequence_step_row.input_norm_layer_idx
    );
    println!(
        "    next_input_norm_layer_idx: {}",
        sequence_step_row.next_input_norm_layer_idx
    );
    println!("    buffer_window_row_selected_from_sequence_metadata: true");
    println!("    apply_sequence_buffer_row_bound_to_step_window_row: true");
    println!("    main_apply_sequence_per_call_buffer_row_selection_removed: true");
    println!("    main_apply_sequence_driver_buffer_triple_args_removed: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    Ok(buffer_row)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_next_decode_dense_cursor_table<
    'dispatch,
    'legacy,
    const TABLE_ROWS: usize,
>(
    sequence_step_row: QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindowRow,
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &'dispatch [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'dispatch,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    legacy_apply_window: QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'legacy>,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    next_input_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
) -> anyhow::Result<bool> {
    let sequence_step_index = sequence_step_row.sequence_step_index;
    let descriptor =
        qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_descriptor_for_index(
            sequence_step_index,
        )?;
    println!("  {}:", descriptor.sequence_stage_label);
    println!("    source: {}", descriptor.source_label);
    println!(
        "    dispatch_role: {}",
        descriptor.dispatch_target_role.label()
    );
    println!("    sequence_step_index: {sequence_step_index}");
    println!(
        "    sequence_window_row_index: {}",
        sequence_step_row.sequence_window_row_index
    );
    println!(
        "    sequence_step_count_bound: {}",
        QWEN_RESIDENT_LAYER_RUNNER_DENSE_LAYER_DISPATCH_APPLY_SEQUENCE_STEPS.len()
    );
    println!("    apply_sequence_roles_bound: first_dense_dispatch,second_dense_dispatch,third_dense_dispatch");
    println!("    remaining_apply_dispatch_calls_unrolled: 0");
    println!(
        "    future_sequence_roles: {}",
        descriptor.future_sequence_roles
    );
    println!("    {}: true", descriptor.main_callsite_removed_label);
    println!("    main_apply_cursor_table_callsites_removed: true");
    println!("    main_role_specific_sequence_step_wrappers_removed: true");
    println!("    main_dispatch_sequence_step_callsites_unrolled: true");
    println!("    apply_sequence_step_descriptor_table_used: true");
    println!("    apply_sequence_step_window_row_used: true");
    println!("    sequence_step_window_cursor_used: true");
    println!("    main_apply_sequence_step_index_literals_removed: true");
    println!("    sequence_step_index_validated_against_role_row: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_apply_next_decode_dense_cursor_table(
        runner_entries,
        runner_plans,
        dispatch_context_window,
        target_cursor,
        descriptor.dispatch_target_role,
        hbm_state,
        legacy_apply_window,
        dag_scheduler,
        dev,
        peer_workspaces,
        tp_allreduce_node_order,
        residual_replicas,
        post_attn_norm_outputs,
        input_norm_outputs,
        next_input_norm_outputs,
        position,
        tp_world,
        h_size,
        h_bytes,
        poison_h,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_apply_next_sequence_step_next_decode_dense_cursor_table<
    'dispatch,
    'legacy,
    'buffers,
    const TABLE_ROWS: usize,
>(
    sequence_step_window: &QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepWindow,
    sequence_step_cursor: &mut QwenResidentLayerRunnerDenseLayerDispatchApplySequenceStepCursor,
    apply_sequence_buffer_window: QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferWindow<
        'buffers,
    >,
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &'dispatch [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'dispatch,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    legacy_apply_window: QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'legacy>,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
) -> anyhow::Result<bool> {
    let sequence_step_row =
        qwen_resident_layer_runner_dense_layer_dispatch_next_apply_sequence_step_window_row(
            sequence_step_window,
            sequence_step_cursor,
        )?;
    let descriptor =
        qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_descriptor_for_index(
            sequence_step_row.sequence_step_index,
        )?;

    println!("  resident_layer_runner_dense_layer_dispatch_apply_sequence_driver_stage:");
    println!("    source: resident_runner_dense_dispatch_apply_next_sequence_step_driver");
    println!(
        "    sequence_window_row_index: {}",
        sequence_step_row.sequence_window_row_index
    );
    println!(
        "    sequence_step_index: {}",
        sequence_step_row.sequence_step_index
    );
    println!(
        "    dispatch_role: {}",
        descriptor.dispatch_target_role.label()
    );
    println!(
        "    next_window_row_index_after_driver_advance: {}",
        sequence_step_cursor.next_window_row_index
    );
    println!("    cursor_advance_bound_to_dispatch: true");
    println!("    indexed_dispatch_bound_to_sequence_driver: true");
    println!("    main_apply_sequence_cursor_next_callsites_removed: true");
    println!("    main_apply_sequence_step_index_literals_removed: true");
    println!("    apply_sequence_buffer_window_bound_to_driver: true");
    println!("    apply_sequence_buffer_row_selected_from_step_window_row: true");
    println!("    apply_sequence_buffer_row_bound_to_driver: true");
    println!("    main_apply_sequence_driver_buffer_triple_args_removed: true");
    println!("    sequence_step_window_cursor_used: true");
    println!("    sequence_loop_started: false");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let apply_sequence_buffer_row =
        qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_buffer_row_from_window(
            sequence_step_row,
            apply_sequence_buffer_window,
        )?;

    let QwenResidentLayerRunnerDenseLayerDispatchApplySequenceBufferRow {
        post_attn_norm_outputs,
        input_norm_outputs,
        next_input_norm_outputs,
    } = apply_sequence_buffer_row;

    qwen_resident_layer_runner_dense_layer_dispatch_apply_sequence_step_next_decode_dense_cursor_table(
        sequence_step_row,
        runner_entries,
        runner_plans,
        dispatch_context_window,
        target_cursor,
        hbm_state,
        legacy_apply_window,
        dag_scheduler,
        dev,
        peer_workspaces,
        tp_allreduce_node_order,
        residual_replicas,
        post_attn_norm_outputs,
        input_norm_outputs,
        next_input_norm_outputs,
        position,
        tp_world,
        h_size,
        h_bytes,
        poison_h,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_update_runtime_readiness_next_decode_attention_cursor_table<
    'dispatch,
    const TABLE_ROWS: usize,
>(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &'dispatch [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'dispatch,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    runtime_layer_state_tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    readiness_update_selection: QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
) -> anyhow::Result<
    Option<(
        QwenNextDecodeLayer1QkvAllRankStage,
        QwenNextDecodeLayer1QkRopeAllRankStage,
        Option<QwenNextDecodeLayer1KvAppendAllRankStage>,
        Option<QwenNextDecodeLayer1AttentionAllRankStage>,
        Option<QwenNextDecodeLayer1OProjAllReduceStage>,
        Option<QwenNextDecodeLayer1PostAttnNormAllRankStage>,
    )>,
> {
    println!(
        "  resident_layer_runner_dense_layer_dispatch_readiness_attention_cursor_table_boundary_stage:"
    );
    println!(
        "    source: resident_runner_dense_dispatch_readiness_attention_cursor_table_boundary"
    );
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!(
        "    readiness_update_selection: {}",
        readiness_update_selection.label()
    );
    println!("    context_window_role_readiness_bound_to_attention_cursor_table_dispatch: true");
    println!("    dispatch_io_options_bound_inside_cursor_table_dispatch_helper: true");
    println!("    runtime_readiness_after_attention_cursor_table_dispatch: true");
    println!("    main_attention_cursor_table_dispatch_closure_removed: true");
    println!("    main_dispatch_calls_unrolled: true");
    println!("    manual_dispatch_calls_retained: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_and_update_runtime_readiness(
        runner_plans,
        dispatch_context_window,
        target_cursor,
        dispatch_target_role,
        runtime_layer_state_tracker,
        readiness_update_selection,
        |dense_step_cursor_table, dense_dispatch_target_role| {
            qwen_resident_layer_runner_dispatch_next_decode_attention_from_dense_cursor_table(
                runner_entries,
                runner_plans,
                dense_step_cursor_table.as_slice(),
                dag_scheduler,
                dev,
                peer_workspaces,
                target_cursor,
                dense_dispatch_target_role,
                tp_allreduce_node_order,
                residual_replicas,
                post_attn_norm_outputs,
                input_norm_outputs,
                position,
                tp_world,
                h_size,
                h_bytes,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dense_layer_dispatch_trial_or_attention_update_from_context_window_cursor_role<
    'dispatch,
    'legacy,
    const TABLE_ROWS: usize,
>(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &'dispatch [QwenResidentLayerRunnerPlanDescriptor],
    dispatch_context_window: &QwenResidentLayerRunnerDenseLayerDispatchContextWindow<
        'dispatch,
        TABLE_ROWS,
    >,
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    runtime_layer_state_tracker: &mut QwenResidentLayerRunnerRuntimeLayerStateTracker,
    readiness_update_selection: QwenResidentLayerRunnerRuntimeReadinessUpdatePlanSelection,
    hbm_state: &QwenResidentLayerRunnerDenseOutputSlotHbmState,
    legacy_apply_window: QwenResidentLayerRunnerDenseLayerLegacyApplyWindow<'legacy>,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    next_input_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
    layer_idx: u32,
    handoff_layer_idx: u32,
    handoff_output_owner: &str,
    handoff_input_norm_plan_present: bool,
) -> anyhow::Result<
    Option<(
        QwenNextDecodeLayer1QkvAllRankStage,
        QwenNextDecodeLayer1QkRopeAllRankStage,
        Option<QwenNextDecodeLayer1KvAppendAllRankStage>,
        Option<QwenNextDecodeLayer1AttentionAllRankStage>,
        Option<QwenNextDecodeLayer1OProjAllReduceStage>,
        Option<QwenNextDecodeLayer1PostAttnNormAllRankStage>,
    )>,
> {
    println!("  resident_layer_runner_dense_layer_dispatch_trial_or_attention_boundary_stage:");
    println!("    source: resident_runner_dense_dispatch_trial_or_attention_boundary");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    layer_idx: {layer_idx}");
    println!("    handoff_layer_idx: {handoff_layer_idx}");
    println!(
        "    readiness_update_selection: {}",
        readiness_update_selection.label()
    );
    println!("    trial_handoff_output_owner: {handoff_output_owner}");
    println!("    main_dense_mlp_trial_branch_removed: true");
    println!("    default_attention_readiness_path_preserved: true");
    println!("    trial_dense_apply_path_preserved: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");

    let dense_mlp_trial_enabled = qwen_resident_layer_runner_dense_mlp_trial_enabled_from_env(
        layer_idx,
        handoff_layer_idx,
        handoff_output_owner,
    );
    println!("    trial_enabled: {dense_mlp_trial_enabled}");
    if dense_mlp_trial_enabled {
        let dense_layer_dispatch_result_present =
            qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_apply_next_decode_dense_cursor_table(
                runner_entries,
                runner_plans,
                dispatch_context_window,
                target_cursor,
                dispatch_target_role,
                hbm_state,
                legacy_apply_window,
                dag_scheduler,
                dev,
                peer_workspaces,
                tp_allreduce_node_order,
                residual_replicas,
                post_attn_norm_outputs,
                input_norm_outputs,
                next_input_norm_outputs,
                position,
                tp_world,
                h_size,
                h_bytes,
                poison_h,
            )?;
        if dense_layer_dispatch_result_present {
            qwen_resident_layer_runner_dense_mlp_layer_handoff(
                layer_idx,
                handoff_layer_idx,
                handoff_output_owner,
                handoff_input_norm_plan_present,
                next_input_norm_outputs.len(),
                tp_world,
            )?;
        }
        Ok(None)
    } else {
        qwen_resident_layer_runner_dense_layer_dispatch_from_context_window_cursor_role_update_runtime_readiness_next_decode_attention_cursor_table(
            runner_entries,
            runner_plans,
            dispatch_context_window,
            target_cursor,
            dispatch_target_role,
            runtime_layer_state_tracker,
            readiness_update_selection,
            dag_scheduler,
            dev,
            peer_workspaces,
            tp_allreduce_node_order,
            residual_replicas,
            post_attn_norm_outputs,
            input_norm_outputs,
            position,
            tp_world,
            h_size,
            h_bytes,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dispatch_next_decode_dense_layer_from_cursor_table(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    cursor_table: &[QwenResidentLayerRunnerDenseLayerStepCursorFrame<'_>],
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    next_input_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchResult> {
    let dispatch_io_options =
        qwen_resident_layer_runner_dense_layer_dispatch_io_options_from_cursor_table_role(
            target_cursor,
            dispatch_target_role,
            cursor_table,
            tp_allreduce_node_order,
            residual_replicas,
        )?;
    println!("  resident_layer_runner_dense_layer_cursor_table_dispatch_io_boundary_stage:");
    println!("    source: resident_runner_dense_cursor_table_dispatch_io_boundary");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    cursor_table_bound_to_dispatch_io_options_inside_dispatch_helper: true");
    println!("    main_dispatch_io_options_helper_callsite_removed: true");
    println!("    main_dispatch_io_options_argument_removed: true");
    println!("    main_dense_mlp_nodes_duplicate_argument_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");
    let QwenResidentLayerRunnerDenseLayerDispatchIoOptions {
        attention_allreduce_nodes,
        mut residual_replicas,
    } = dispatch_io_options;
    qwen_resident_layer_runner_dense_layer_step_cursor_dispatch_loop(cursor_table, |cursor_frame| {
        qwen_resident_layer_runner_dispatch_next_decode_dense_layer_from_layer_plan(
            runner_entries,
            runner_plans,
            cursor_frame.handle,
            dag_scheduler,
            dev,
            peer_workspaces,
            attention_allreduce_nodes,
            residual_replicas.as_deref_mut(),
            post_attn_norm_outputs,
            input_norm_outputs,
            tp_allreduce_node_order,
            next_input_norm_outputs,
            position,
            tp_world,
            h_size,
            h_bytes,
            poison_h,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dispatch_next_decode_attention_from_dense_cursor_table(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    cursor_table: &[QwenResidentLayerRunnerDenseLayerStepCursorFrame<'_>],
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    target_cursor: &QwenResidentLayerRunnerDenseLayerDispatchTargetCursor,
    dispatch_target_role: QwenResidentLayerRunnerDenseLayerDispatchTargetRole,
    tp_allreduce_node_order: &[u32],
    residual_replicas: &mut [mcore::DeviceBuffer],
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
) -> anyhow::Result<(
    QwenNextDecodeLayer1QkvAllRankStage,
    QwenNextDecodeLayer1QkRopeAllRankStage,
    Option<QwenNextDecodeLayer1KvAppendAllRankStage>,
    Option<QwenNextDecodeLayer1AttentionAllRankStage>,
    Option<QwenNextDecodeLayer1OProjAllReduceStage>,
    Option<QwenNextDecodeLayer1PostAttnNormAllRankStage>,
)> {
    let dispatch_io_options =
        qwen_resident_layer_runner_dense_layer_dispatch_io_options_from_cursor_table_role(
            target_cursor,
            dispatch_target_role,
            cursor_table,
            tp_allreduce_node_order,
            residual_replicas,
        )?;
    println!(
        "  resident_layer_runner_dense_layer_attention_cursor_table_dispatch_io_boundary_stage:"
    );
    println!("    source: resident_runner_dense_attention_cursor_table_dispatch_io_boundary");
    println!("    dispatch_role: {}", dispatch_target_role.label());
    println!("    cursor_table_bound_to_dispatch_io_options_inside_dispatch_helper: true");
    println!("    main_dispatch_io_options_helper_callsite_removed: true");
    println!("    main_dispatch_io_options_argument_removed: true");
    println!("    execution_path_changed: false");
    println!("    hip_graph_capture_started: false");
    let QwenResidentLayerRunnerDenseLayerDispatchIoOptions {
        attention_allreduce_nodes,
        residual_replicas,
    } = dispatch_io_options;
    qwen_resident_layer_runner_dense_layer_step_cursor_dispatch_loop(cursor_table, |cursor_frame| {
        let handle = cursor_frame.handle;
        println!("  resident_layer_runner_dense_layer_attention_cursor_dispatch_stage:");
        println!("    source: resident_runner_dense_step_cursor_table");
        println!("    dispatch_layer_idx: {}", handle.layer_idx);
        println!(
            "    attention_dependency_id: {}",
            handle.attention_dependency_id.label()
        );
        println!("    dense_cursor_consumed_for_attention_dispatch: true");
        println!("    dense_cursor_mlp_leg_dispatched: false");
        println!("    full_dense_layer_dispatch_promoted: false");
        println!("    hip_graph_capture_started: false");
        qwen_resident_layer_runner_dispatch_next_decode_attention_from_layer_plan(
            runner_entries,
            runner_plans,
            handle.layer_idx,
            dag_scheduler,
            dev,
            peer_workspaces,
            handle.qkv,
            handle.qk,
            handle.cache_plan,
            handle.attn_plan,
            handle.o_proj_plan,
            attention_allreduce_nodes,
            residual_replicas,
            handle.post_attn_norm,
            post_attn_norm_outputs,
            handle.peer_plan,
            input_norm_outputs,
            position,
            tp_world,
            h_size,
            h_bytes,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn qwen_resident_layer_runner_dispatch_next_decode_attention_from_layer_plan(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    layer_idx: u32,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    qkv: &QwenQkvProjStagePlan,
    qk: &QwenQkNormRopeStagePlan,
    cache_plan: Option<&QwenFp4KvCacheStagePlan>,
    attn_plan: Option<&QwenFp4SingleRowAttentionStagePlan>,
    o_proj_plan: Option<&QwenOProjStagePlan>,
    allreduce_nodes: Option<&[u32]>,
    residual_replicas: Option<&mut [mcore::DeviceBuffer]>,
    post_attn_norm: Option<&QwenPostAttnNormStagePlan>,
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    peer_plan: &QwenTpOProjPeerStagePlan,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
) -> anyhow::Result<(
    QwenNextDecodeLayer1QkvAllRankStage,
    QwenNextDecodeLayer1QkRopeAllRankStage,
    Option<QwenNextDecodeLayer1KvAppendAllRankStage>,
    Option<QwenNextDecodeLayer1AttentionAllRankStage>,
    Option<QwenNextDecodeLayer1OProjAllReduceStage>,
    Option<QwenNextDecodeLayer1PostAttnNormAllRankStage>,
)> {
    let dispatch_plan = runner_plans
        .iter()
        .find(|plan| plan.layer_idx == layer_idx)
        .and_then(|plan| plan.attention_dispatch)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "resident layer runner layer{layer_idx} attention table dispatch plan is missing"
            )
        })?;
    let dispatch_enabled = qwen_resident_layer_runner_dispatch_plan_enabled(dispatch_plan);
    if !dispatch_enabled {
        anyhow::bail!(
            "resident layer runner layer{layer_idx} attention table dispatch plan is disabled"
        );
    }
    if post_attn_norm.is_some() {
        qwen_prepare_next_decode_layer_rank_h_buffers(
            dev,
            peer_workspaces,
            post_attn_norm_outputs,
            tp_world,
            h_bytes,
            layer_idx,
            "post-attention norm",
        )?;
    }

    println!("  resident_layer_runner_table_attention_dispatch_plan_stage:");
    println!("    source: per_layer_resident_dispatch_plan");
    println!("    layer_idx: {layer_idx}");
    println!("    generic_helper: qwen_resident_layer_runner_dispatch_next_decode_attention_from_layer_plan");
    println!("    generic_helper_owner: resident_runner_module");
    println!("    main_owned_generic_helper_removed_from_dense_dispatch_path: true");
    println!("    manual_unrolled_callsite_replaced: true");
    println!("    dependency_id: {}", dispatch_plan.dependency_id.label());
    println!("    wait_mask: 0x{:016x}", dispatch_plan.wait_mask);
    println!("    signal_mask: 0x{:016x}", dispatch_plan.signal_mask);
    println!(
        "    launch_function: {}",
        dispatch_plan
            .launch_function
            .unwrap_or("qwen_resident_layer_runner_launch_decode_attention")
    );
    println!(
        "    capture_boundary: {}",
        dispatch_plan.capture_boundary.label()
    );
    println!("    next_required: {}", dispatch_plan.next_required);
    println!("    default_enabled: {}", dispatch_plan.default_enabled);
    println!(
        "    disable_env: {}",
        dispatch_plan.disable_env.unwrap_or("<none>")
    );
    println!(
        "    post_attn_norm_outputs_prepared_by_helper: {}",
        post_attn_norm.is_some()
    );
    println!("    hip_graph_capture_started: false");

    let post_attn_norm_outputs_arg = if post_attn_norm.is_some() {
        Some(post_attn_norm_outputs.as_mut_slice())
    } else {
        None
    };
    let (
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
    ) = qwen_resident_layer_runner_dispatch_next_decode_attention(
        runner_entries,
        layer_idx,
        dag_scheduler,
        dev,
        peer_workspaces,
        qkv,
        qk,
        cache_plan,
        attn_plan,
        o_proj_plan,
        allreduce_nodes,
        residual_replicas,
        post_attn_norm,
        post_attn_norm_outputs_arg,
        peer_plan,
        input_norm_outputs,
        position,
        h_size,
        h_bytes,
    )?;
    if post_attn_norm_stage.is_some() && post_attn_norm_outputs.len() != tp_world {
        anyhow::bail!(
            "next decode layer{layer_idx} post-attention norm output ranks {} do not match TP world {}",
            post_attn_norm_outputs.len(),
            tp_world
        );
    }
    Ok((
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
    ))
}

#[allow(clippy::too_many_arguments)]
fn qwen_resident_layer_runner_dispatch_next_decode_mlp_from_layer_plan(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    layer_idx: u32,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    mlp: &QwenMlpStagePlan,
    peer_mlp: &QwenTpMlpPeerStagePlan,
    nodes: &[u32],
    post_norm_outputs: &mut [mcore::DeviceBuffer],
    residual_replicas: Option<&mut [mcore::DeviceBuffer]>,
    next_input_norm: Option<&QwenNextInputNormStagePlan>,
    next_input_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
) -> anyhow::Result<(
    QwenNextDecodeLayer0MlpAllReduceStage,
    Option<QwenNextDecodeLayer0MlpResidualNormAllRankStage>,
)> {
    let dispatch_plan = qwen_resident_layer_runner_mlp_dispatch_plan_for_layer(layer_idx)?;
    let dispatch_enabled = qwen_resident_layer_runner_dispatch_plan_enabled(dispatch_plan);
    if !dispatch_enabled {
        anyhow::bail!("resident layer runner layer{layer_idx} MLP table dispatch plan is disabled");
    }
    if next_input_norm.is_some() {
        qwen_prepare_next_decode_layer_rank_h_buffers(
            dev,
            peer_workspaces,
            next_input_norm_outputs,
            tp_world,
            h_bytes,
            layer_idx + 1,
            "input norm",
        )?;
    }

    println!("  resident_layer_runner_table_mlp_dispatch_plan_stage:");
    println!("    source: per_layer_resident_dispatch_plan");
    println!("    layer_idx: {layer_idx}");
    println!(
        "    generic_helper: qwen_resident_layer_runner_dispatch_next_decode_mlp_from_layer_plan"
    );
    println!("    generic_helper_owner: resident_runner_module");
    println!("    main_owned_generic_helper_removed_from_dense_dispatch_path: true");
    println!("    manual_unrolled_callsite_replaced: true");
    println!("    dependency_id: {}", dispatch_plan.dependency_id.label());
    println!("    wait_mask: 0x{:016x}", dispatch_plan.wait_mask);
    println!("    signal_mask: 0x{:016x}", dispatch_plan.signal_mask);
    println!(
        "    launch_function: {}",
        qwen_resident_layer_runner_mlp_launch_function(dispatch_plan)
    );
    println!(
        "    capture_boundary: {}",
        dispatch_plan.capture_boundary.label()
    );
    println!("    next_required: {}", dispatch_plan.next_required);
    println!("    default_enabled: {}", dispatch_plan.default_enabled);
    println!(
        "    disable_env: {}",
        dispatch_plan.disable_env.unwrap_or("<none>")
    );
    println!(
        "    next_input_norm_outputs_prepared_by_helper: {}",
        next_input_norm.is_some()
    );
    println!("    hip_graph_capture_started: false");

    let next_input_norm_outputs_arg = if next_input_norm.is_some() {
        Some(next_input_norm_outputs.as_mut_slice())
    } else {
        None
    };
    let (mlp_stage, next_input_norm_stage) = qwen_resident_layer_runner_dispatch_next_decode_mlp(
        runner_entries,
        layer_idx,
        dag_scheduler,
        dev,
        peer_workspaces,
        mlp,
        peer_mlp,
        nodes,
        post_norm_outputs,
        residual_replicas,
        next_input_norm,
        next_input_norm_outputs_arg,
        h_size,
        h_bytes,
        poison_h,
    )?;
    if next_input_norm_stage.is_some() && next_input_norm_outputs.len() != tp_world {
        anyhow::bail!(
            "next decode layer{} input norm output ranks {} do not match TP world {}",
            layer_idx + 1,
            next_input_norm_outputs.len(),
            tp_world
        );
    }
    Ok((mlp_stage, next_input_norm_stage))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_resident_layer_runner_dispatch_next_decode_dense_layer_from_layer_plan(
    runner_entries: &[QwenResidentLayerRunnerEntry],
    runner_plans: &[QwenResidentLayerRunnerPlanDescriptor],
    handle: QwenResidentLayerRunnerDenseLayerStepPlanHandle<'_>,
    dag_scheduler: QwenResidentLayerRunnerDagSchedulerContext<'_>,
    dev: &mut mcore::GpuDevice,
    peer_workspaces: &mut [QwenPeerMlpWorkspace],
    attention_allreduce_nodes: Option<&[u32]>,
    mut residual_replicas: Option<&mut [mcore::DeviceBuffer]>,
    post_attn_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    input_norm_outputs: &mut [mcore::DeviceBuffer],
    mlp_nodes: &[u32],
    next_input_norm_outputs: &mut Vec<mcore::DeviceBuffer>,
    position: u32,
    tp_world: usize,
    h_size: u32,
    h_bytes: usize,
    poison_h: u16,
) -> anyhow::Result<QwenResidentLayerRunnerDenseLayerDispatchResult> {
    let layer_idx = handle.layer_idx;
    println!("  resident_layer_runner_dense_layer_step_dispatch_stage:");
    println!("    source: per_layer_resident_dispatch_plan");
    println!("    layer_idx: {layer_idx}");
    println!("    indexed_layer_plan_handle: true");
    println!("    handle_layer_idx: {}", handle.layer_idx);
    println!(
        "    attention_dependency_id: {}",
        handle.attention_dependency_id.label()
    );
    println!(
        "    mlp_dependency_id: {}",
        handle.mlp_dependency_id.label()
    );
    println!("    generic_helper: qwen_resident_layer_runner_dispatch_next_decode_dense_layer_from_layer_plan");
    println!("    attention_helper: qwen_resident_layer_runner_dispatch_next_decode_attention_from_layer_plan");
    println!("    mlp_helper: qwen_resident_layer_runner_dispatch_next_decode_mlp_from_layer_plan");
    println!("    dense_dispatch_wrapper_owner: resident_runner_module");
    println!("    main_dense_cursor_dispatch_wrapper_removed: true");
    println!("    manual_attention_mlp_sequence_replaced: true");
    println!("    hip_graph_capture_started: false");

    let (
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
    ) = qwen_resident_layer_runner_dispatch_next_decode_attention_from_layer_plan(
        runner_entries,
        runner_plans,
        layer_idx,
        dag_scheduler,
        dev,
        peer_workspaces,
        handle.qkv,
        handle.qk,
        handle.cache_plan,
        handle.attn_plan,
        handle.o_proj_plan,
        attention_allreduce_nodes,
        residual_replicas.as_deref_mut(),
        handle.post_attn_norm,
        post_attn_norm_outputs,
        handle.peer_plan,
        input_norm_outputs,
        position,
        tp_world,
        h_size,
        h_bytes,
    )?;
    if post_attn_norm_stage.is_none() {
        anyhow::bail!(
            "resident layer runner dense layer{layer_idx} step requires post-attention norm before MLP dispatch"
        );
    }

    let (mlp_stage, next_input_norm_stage) =
        qwen_resident_layer_runner_dispatch_next_decode_mlp_from_layer_plan(
            runner_entries,
            layer_idx,
            dag_scheduler,
            dev,
            peer_workspaces,
            handle.mlp,
            handle.peer_mlp,
            mlp_nodes,
            post_attn_norm_outputs.as_mut_slice(),
            residual_replicas.as_deref_mut(),
            handle.next_input_norm,
            next_input_norm_outputs,
            tp_world,
            h_size,
            h_bytes,
            poison_h,
        )?;

    qwen_resident_layer_runner_dense_layer_dispatch_result_from_stages(
        layer_idx,
        qkv_stage,
        qk_rope_stage,
        kv_append_stage,
        attention_stage,
        o_proj_stage,
        post_attn_norm_stage,
        mlp_stage,
        next_input_norm_stage,
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_step_cursor_dispatch_loop<'a, T, F>(
    cursor_table: &[QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>],
    dispatch_cursor_frame: F,
) -> anyhow::Result<T>
where
    F: FnOnce(QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>) -> anyhow::Result<T>,
{
    if cursor_table.is_empty() {
        anyhow::bail!("resident layer runner dense-step cursor table is empty");
    }

    let mut cursor_index = 0usize;
    while cursor_index < cursor_table.len() {
        let cursor_frame = cursor_table[cursor_index];
        if cursor_frame.cursor_index != cursor_index {
            anyhow::bail!(
                "resident layer runner dense-step cursor index mismatch: table slot {} has cursor index {}",
                cursor_index,
                cursor_frame.cursor_index
            );
        }
        if cursor_frame.table_len != cursor_table.len() {
            anyhow::bail!(
                "resident layer runner dense-step cursor table length mismatch: frame={} actual={}",
                cursor_frame.table_len,
                cursor_table.len()
            );
        }

        println!("  resident_layer_runner_dense_layer_step_cursor_dispatch_stage:");
        println!("    source: resident_layer_runner_dense_step_cursor_table");
        println!("    cursor_dispatch_loop_owned_by_resident_runner_module: true");
        println!("    cursor_table_len: {}", cursor_table.len());
        println!("    cursor_index: {cursor_index}");
        println!("    cursor_op: {}", cursor_frame.op.label());
        println!("    cursor_layer_idx: {}", cursor_frame.handle.layer_idx);
        println!("    table_dispatched_dense_step: true");
        println!("    hip_graph_capture_started: false");

        let result = match cursor_frame.op {
            QwenResidentLayerRunnerDenseLayerStepCursorOp::DenseStep => {
                dispatch_cursor_frame(cursor_frame)?
            }
        };
        cursor_index += 1;
        println!("    cursor_loop_iterations: {cursor_index}");
        return Ok(result);
    }

    anyhow::bail!(
        "resident layer runner dense-step cursor table reached terminal cursor without dispatch"
    )
}

pub(crate) fn qwen_resident_layer_runner_dense_layer_step_cursor_table_from_resource_frame<'a>(
    runner_plans: &'a [QwenResidentLayerRunnerPlanDescriptor],
    resource_frame: QwenResidentLayerRunnerDenseLayerResourceFrame<'a>,
) -> anyhow::Result<[QwenResidentLayerRunnerDenseLayerStepCursorFrame<'a>; 1]> {
    let layer_idx = resource_frame.layer_idx;
    println!("  resident_layer_runner_dense_layer_step_cursor_table_builder_stage:");
    println!("    source: resident_layer_runner_layer_indexed_cursor_table_builder");
    println!("    builder_layer_idx: {layer_idx}");
    println!("    builder_keyed_by_layer_idx: true");
    println!("    resource_frame_bound: true");
    println!("    cursor_table_builder_owned_by_resident_runner_module: true");
    println!("    cursor_table_len: 1");
    println!("    cursor_op: dense_step");
    println!("    topology_routing_started: false");
    println!("    hip_graph_capture_started: false");

    Ok([
        qwen_resident_layer_runner_dense_layer_step_cursor_frame_from_index(
            runner_plans,
            layer_idx,
            resource_frame.qkv,
            resource_frame.qk,
            resource_frame.cache_plan,
            resource_frame.attn_plan,
            resource_frame.o_proj_plan,
            resource_frame.post_attn_norm,
            resource_frame.peer_plan,
            resource_frame.mlp,
            resource_frame.peer_mlp,
            resource_frame.next_input_norm,
        )?,
    ])
}

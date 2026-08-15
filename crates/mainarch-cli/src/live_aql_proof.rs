use mainarch_core as mcore;

pub(crate) fn qwen_print_live_aql_reservation_proof(
    proof: &mcore::KfdQueueLiveAqlReservationProof,
) {
    println!(
        "    reservation_operands_probe_version: {}",
        proof.operands_probe_version
    );
    println!("    reservation_packet_id: {}", proof.packet_id);
    println!("    reservation_read_index: {}", proof.read_index);
    println!(
        "    reservation_packet_id_matches_host_snapshot: {}",
        proof.packet_id_matches_host_snapshot
    );
    println!(
        "    reservation_read_index_matches_host_snapshot: {}",
        proof.read_index_matches_host_snapshot
    );
    println!(
        "    reservation_inflight_packets: {}",
        proof.inflight_packets
    );
    println!("    reservation_capacity_ok: {}", proof.capacity_ok);
    println!("    reservation_slot_index: {}", proof.slot_index);
    println!("    reservation_slot_offset: {}", proof.slot_offset);
    println!("    reservation_slot_va: 0x{:016x}", proof.slot_va);
    println!(
        "    reservation_slot_va_aligned64: {}",
        proof.slot_va_aligned64
    );
    println!(
        "    reservation_desired_write_index: {}",
        proof.desired_write_index
    );
    println!("    reservation_packet_count: {}", proof.packet_count);
    println!(
        "    reservation_doorbell_packet_id: {}",
        proof.doorbell_packet_id
    );
    println!(
        "    reservation_doorbell_matches_last_packet: {}",
        proof.doorbell_matches_last_packet
    );
    println!(
        "    reservation_publish_low32: 0x{:08x}",
        proof.publish_low32
    );
    println!(
        "    reservation_header_release_width_bits: {}",
        proof.header_release_width_bits
    );
    println!("    reservation_live_low32: 0x{:08x}", proof.live_low32);
    println!(
        "    reservation_valid_header_not_stored: {}",
        proof.valid_header_not_stored
    );
    println!(
        "    reservation_fetch_add_not_performed: {}",
        proof.fetch_add_not_performed
    );
    println!(
        "    reservation_doorbell_not_written: {}",
        proof.doorbell_not_written
    );
    println!(
        "    reservation_capacity_formula_ok: {}",
        proof.capacity_formula_ok
    );
    println!("    reservation_slot_formula_ok: {}", proof.slot_formula_ok);
    println!(
        "    reservation_non_consuming_contract: {}",
        proof.non_consuming_contract
    );
    println!("    reservation_ready: {}", proof.observed_ready);
}

pub(crate) fn qwen_print_live_aql_reserve_before_stage_proof(
    prefix: &str,
    proof: &mcore::KfdQueueLiveAqlReserveBeforeStageProof,
) {
    println!("    {prefix}reserve_before_stage_contract: staged_payload_not_publishable_unless_it_matches_reserved_slot");
    println!(
        "    {prefix}reserve_before_stage_probe_version: {}",
        proof.probe_version
    );
    println!(
        "    {prefix}reserve_before_stage_staged_packet_id: {}",
        proof.staged_packet_id
    );
    println!(
        "    {prefix}reserve_before_stage_reserved_packet_id: {}",
        proof.reserved_packet_id
    );
    println!(
        "    {prefix}reserve_before_stage_staged_slot_va: 0x{:016x}",
        proof.staged_slot_va
    );
    println!(
        "    {prefix}reserve_before_stage_reserved_slot_va: 0x{:016x}",
        proof.reserved_slot_va
    );
    println!(
        "    {prefix}reserve_before_stage_staged_slot_offset: {}",
        proof.staged_slot_offset
    );
    println!(
        "    {prefix}reserve_before_stage_reserved_slot_offset: {}",
        proof.reserved_slot_offset
    );
    println!(
        "    {prefix}reserve_before_stage_same_packet_id: {}",
        proof.same_packet_id
    );
    println!(
        "    {prefix}reserve_before_stage_same_slot: {}",
        proof.same_slot
    );
    println!(
        "    {prefix}reserve_before_stage_old_payload_write_ready: {}",
        proof.old_payload_write_ready
    );
    println!(
        "    {prefix}reserve_before_stage_old_payload_publishable: {}",
        proof.old_payload_publishable
    );
    println!(
        "    {prefix}reserve_before_stage_must_restage_after_reserve: {}",
        proof.must_restage_after_reserve
    );
    println!(
        "    {prefix}reserve_before_stage_publish_blocked_until_restage: {}",
        proof.publish_blocked_until_restage
    );
    println!(
        "    {prefix}reserve_before_stage_old_slot_still_invalid: {}",
        proof.old_slot_still_invalid
    );
    println!(
        "    {prefix}reserve_before_stage_reservation_ready_dependency: {}",
        proof.reservation_ready_dependency
    );
    println!(
        "    {prefix}reserve_before_stage_valid_header_not_stored: {}",
        proof.valid_header_not_stored
    );
    println!(
        "    {prefix}reserve_before_stage_publish_low32: 0x{:08x}",
        proof.publish_low32
    );
    println!(
        "    {prefix}reserve_before_stage_live_low32: 0x{:08x}",
        proof.live_low32
    );
    println!(
        "    {prefix}reserve_before_stage_slot_progress_observed: {}",
        proof.slot_progress_observed
    );
    println!(
        "    {prefix}reserve_before_stage_desired_write_index: {}",
        proof.desired_write_index
    );
    println!(
        "    {prefix}reserve_before_stage_doorbell_packet_id: {}",
        proof.doorbell_packet_id
    );
    println!(
        "    {prefix}reserve_before_stage_capacity_ok: {}",
        proof.capacity_ok
    );
    println!(
        "    {prefix}reserve_before_stage_slot_formula_ok: {}",
        proof.slot_formula_ok
    );
    println!(
        "    {prefix}reserve_before_stage_fetch_add_not_performed: {}",
        proof.fetch_add_not_performed
    );
    println!(
        "    {prefix}reserve_before_stage_reserved_slot_not_written: {}",
        proof.reserved_slot_not_written
    );
    println!(
        "    {prefix}reserve_before_stage_header_not_published: {}",
        proof.header_not_published
    );
    println!(
        "    {prefix}reserve_before_stage_doorbell_not_written: {}",
        proof.doorbell_not_written
    );
    println!(
        "    {prefix}reserve_before_stage_reserve_first_contract: {}",
        proof.reserve_first_contract
    );
    println!(
        "    {prefix}reserve_before_stage_reserved_slot_stage_required: {}",
        proof.reserved_slot_stage_required
    );
    println!(
        "    {prefix}reserve_before_stage_non_consuming_contract: {}",
        proof.non_consuming_contract
    );
    println!(
        "    {prefix}reserve_before_stage_sequence_ready: {}",
        proof.sequence_ready
    );
    println!(
        "    {prefix}reserve_before_stage_ready: {}",
        proof.observed_ready
    );
}

pub(crate) fn qwen_print_live_aql_reserve_first_restage_proof(
    prefix: &str,
    proof: &mcore::KfdQueueLiveAqlReserveFirstRestageProof,
) {
    println!("    {prefix}reserve_first_restage_contract: reserved_slot_is_the_only_payload_restage_target_before_publish");
    println!(
        "    {prefix}reserve_first_restage_probe_version: {}",
        proof.probe_version
    );
    println!(
        "    {prefix}reserve_first_restage_target_packet_id: {}",
        proof.target_packet_id
    );
    println!(
        "    {prefix}reserve_first_restage_target_slot_va: 0x{:016x}",
        proof.target_slot_va
    );
    println!(
        "    {prefix}reserve_first_restage_target_slot_offset: {}",
        proof.target_slot_offset
    );
    println!(
        "    {prefix}reserve_first_restage_reservation_packet_id: {}",
        proof.reservation_packet_id
    );
    println!(
        "    {prefix}reserve_first_restage_reservation_slot_va: 0x{:016x}",
        proof.reservation_slot_va
    );
    println!(
        "    {prefix}reserve_first_restage_reservation_slot_offset: {}",
        proof.reservation_slot_offset
    );
    println!(
        "    {prefix}reserve_first_restage_target_matches_reservation: {}",
        proof.target_matches_reservation
    );
    println!(
        "    {prefix}reserve_first_restage_old_packet_id: {}",
        proof.old_packet_id
    );
    println!(
        "    {prefix}reserve_first_restage_old_slot_va: 0x{:016x}",
        proof.old_slot_va
    );
    println!(
        "    {prefix}reserve_first_restage_old_slot_bypassed: {}",
        proof.old_slot_bypassed
    );
    println!(
        "    {prefix}reserve_first_restage_payload_inputs_ready: {}",
        proof.payload_inputs_ready
    );
    println!(
        "    {prefix}reserve_first_restage_publish_low32: 0x{:08x}",
        proof.publish_low32
    );
    println!(
        "    {prefix}reserve_first_restage_live_low32: 0x{:08x}",
        proof.live_low32
    );
    println!(
        "    {prefix}reserve_first_restage_valid_header_store_pending: {}",
        proof.valid_header_store_pending
    );
    println!(
        "    {prefix}reserve_first_restage_reserved_slot_write_pending: {}",
        proof.reserved_slot_write_pending
    );
    println!(
        "    {prefix}reserve_first_restage_write_index_fetch_add_pending: {}",
        proof.write_index_fetch_add_pending
    );
    println!(
        "    {prefix}reserve_first_restage_doorbell_pending: {}",
        proof.doorbell_pending
    );
    println!(
        "    {prefix}reserve_first_restage_release_header_after_payload_contract: {}",
        proof.release_header_after_payload_contract
    );
    println!(
        "    {prefix}reserve_first_restage_reserve_before_payload_contract: {}",
        proof.reserve_before_payload_contract
    );
    println!(
        "    {prefix}reserve_first_restage_doorbell_after_header_contract: {}",
        proof.doorbell_after_header_contract
    );
    println!(
        "    {prefix}reserve_first_restage_no_live_queue_mutation_contract: {}",
        proof.no_live_queue_mutation_contract
    );
    println!(
        "    {prefix}reserve_first_restage_plan_ready: {}",
        proof.observed_plan_ready
    );
    println!(
        "    {prefix}reserve_first_restage_capacity_ok: {}",
        proof.capacity_ok
    );
    println!(
        "    {prefix}reserve_first_restage_slot_formula_ok: {}",
        proof.slot_formula_ok
    );
    println!(
        "    {prefix}reserve_first_restage_desired_write_index: {}",
        proof.desired_write_index
    );
    println!(
        "    {prefix}reserve_first_restage_doorbell_packet_id: {}",
        proof.doorbell_packet_id
    );
    println!(
        "    {prefix}reserve_first_restage_packet_bytes: {}",
        proof.packet_bytes
    );
    println!(
        "    {prefix}reserve_first_restage_ring_slots: {}",
        proof.ring_slots
    );
    println!(
        "    {prefix}reserve_first_restage_slot_mask: 0x{:016x}",
        proof.slot_mask
    );
    println!(
        "    {prefix}reserve_first_restage_publish_blocked_before_restage: {}",
        proof.publish_blocked_before_restage
    );
    println!(
        "    {prefix}reserve_first_restage_ready: {}",
        proof.observed_ready
    );
}

pub(crate) fn qwen_print_live_aql_batch_reservation_plan_proof(
    prefix: &str,
    proof: &mcore::KfdQueueLiveAqlBatchReservationPlanProof,
) {
    println!(
        "    {prefix}batch_reservation_plan_contract: two_packets_one_doorbell_after_all_headers"
    );
    println!(
        "    {prefix}batch_reservation_plan_probe_version: {}",
        proof.probe_version
    );
    println!(
        "    {prefix}batch_reservation_plan_base_packet_id: {}",
        proof.base_packet_id
    );
    println!(
        "    {prefix}batch_reservation_plan_packet_count: {}",
        proof.packet_count
    );
    println!(
        "    {prefix}batch_reservation_plan_last_packet_id: {}",
        proof.last_packet_id
    );
    println!(
        "    {prefix}batch_reservation_plan_desired_write_index: {}",
        proof.desired_write_index
    );
    println!(
        "    {prefix}batch_reservation_plan_read_index: {}",
        proof.read_index
    );
    println!(
        "    {prefix}batch_reservation_plan_inflight_packets: {}",
        proof.inflight_packets
    );
    println!(
        "    {prefix}batch_reservation_plan_capacity_ok: {}",
        proof.capacity_ok
    );
    println!(
        "    {prefix}batch_reservation_plan_slot0_va: 0x{:016x}",
        proof.slot0_va
    );
    println!(
        "    {prefix}batch_reservation_plan_slot1_va: 0x{:016x}",
        proof.slot1_va
    );
    println!(
        "    {prefix}batch_reservation_plan_slot0_offset: {}",
        proof.slot0_offset
    );
    println!(
        "    {prefix}batch_reservation_plan_slot1_offset: {}",
        proof.slot1_offset
    );
    println!(
        "    {prefix}batch_reservation_plan_slot0_index: {}",
        proof.slot0_index
    );
    println!(
        "    {prefix}batch_reservation_plan_slot1_index: {}",
        proof.slot1_index
    );
    println!(
        "    {prefix}batch_reservation_plan_slots_distinct: {}",
        proof.slots_distinct
    );
    println!(
        "    {prefix}batch_reservation_plan_slots_aligned64: {}",
        proof.slots_aligned64
    );
    println!(
        "    {prefix}batch_reservation_plan_slot0_formula_ok: {}",
        proof.slot0_formula_ok
    );
    println!(
        "    {prefix}batch_reservation_plan_slot1_formula_ok: {}",
        proof.slot1_formula_ok
    );
    println!(
        "    {prefix}batch_reservation_plan_doorbell_packet_id: {}",
        proof.doorbell_packet_id
    );
    println!(
        "    {prefix}batch_reservation_plan_doorbell_matches_last_packet: {}",
        proof.doorbell_matches_last_packet
    );
    println!(
        "    {prefix}batch_reservation_plan_single_doorbell_contract: {}",
        proof.single_doorbell_contract
    );
    println!(
        "    {prefix}batch_reservation_plan_reserve_before_payload_contract: {}",
        proof.reserve_before_payload_contract
    );
    println!(
        "    {prefix}batch_reservation_plan_payloads_before_headers_contract: {}",
        proof.payloads_before_headers_contract
    );
    println!(
        "    {prefix}batch_reservation_plan_headers_before_doorbell_contract: {}",
        proof.headers_before_doorbell_contract
    );
    println!(
        "    {prefix}batch_reservation_plan_release_header_store_contract: {}",
        proof.release_header_store_contract
    );
    println!(
        "    {prefix}batch_reservation_plan_write_index_fetch_add_pending: {}",
        proof.write_index_fetch_add_pending
    );
    println!(
        "    {prefix}batch_reservation_plan_payload_writes_pending: {}",
        proof.payload_writes_pending
    );
    println!(
        "    {prefix}batch_reservation_plan_valid_headers_pending: {}",
        proof.valid_headers_pending
    );
    println!(
        "    {prefix}batch_reservation_plan_doorbell_pending: {}",
        proof.doorbell_pending
    );
    println!(
        "    {prefix}batch_reservation_plan_no_live_queue_mutation_contract: {}",
        proof.no_live_queue_mutation_contract
    );
    println!(
        "    {prefix}batch_reservation_plan_first_slot_matches_single_reservation: {}",
        proof.first_slot_matches_single_reservation
    );
    println!(
        "    {prefix}batch_reservation_plan_ready: {}",
        proof.observed_ready
    );
}

pub(crate) fn qwen_print_live_aql_materialized_packet_plan_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlMaterializedPacketPlanProof,
) {
    println!(
        "  resident_layer_runner_{target_stage_label}_live_aql_materialized_packet_plan_stage:"
    );
    println!(
        "    source: resident_layer_runner_{target_stage_label}_live_aql_reservation_plan_stage"
    );
    println!("    proof: gpu_materializes_two_reservation_scoped_aql_packet_images_without_live_queue_mutation");
    println!("    materialized_packet_plan_contract: two_64b_packets_payloads_before_release_headers_no_doorbell");
    println!("    materialized_probe_version: {}", proof.probe_version);
    println!(
        "    materialized_packet0_packet_id: {}",
        proof.packet0_packet_id
    );
    println!(
        "    materialized_packet1_packet_id: {}",
        proof.packet1_packet_id
    );
    println!(
        "    materialized_packet0_slot_va: 0x{:016x}",
        proof.packet0_slot_va
    );
    println!(
        "    materialized_packet1_slot_va: 0x{:016x}",
        proof.packet1_slot_va
    );
    println!(
        "    materialized_packet0_word0: 0x{:016x}",
        proof.packet0_word0
    );
    println!(
        "    materialized_packet0_word4_kernel_object: 0x{:016x}",
        proof.packet0_word4_kernel_object
    );
    println!(
        "    materialized_packet0_word5_kernarg_va: 0x{:016x}",
        proof.packet0_word5_kernarg_va
    );
    println!(
        "    materialized_packet1_word0: 0x{:016x}",
        proof.packet1_word0
    );
    println!(
        "    materialized_packet1_word4_kernel_object: 0x{:016x}",
        proof.packet1_word4_kernel_object
    );
    println!(
        "    materialized_packet1_word5_kernarg_va: 0x{:016x}",
        proof.packet1_word5_kernarg_va
    );
    println!(
        "    materialized_packet0_words_match_host_template: {}",
        proof.packet0_words_match_host_template
    );
    println!(
        "    materialized_packet1_words_match_host_template: {}",
        proof.packet1_words_match_host_template
    );
    println!(
        "    materialized_payload_words_match_host_template: {}",
        proof.payload_words_match_host_template
    );
    println!(
        "    materialized_header_words_match_host_template: {}",
        proof.header_words_match_host_template
    );
    println!(
        "    materialized_target_slots_match_batch_plan: {}",
        proof.target_slots_match_batch_plan
    );
    println!(
        "    materialized_packet0_slot_offset: {}",
        proof.packet0_slot_offset
    );
    println!(
        "    materialized_packet1_slot_offset: {}",
        proof.packet1_slot_offset
    );
    println!("    materialized_packet_bytes: {}", proof.packet_bytes);
    println!("    materialized_packet_count: {}", proof.packet_count);
    println!(
        "    materialized_batch_plan_ready: {}",
        proof.batch_plan_ready
    );
    println!(
        "    materialized_reserve_first_restage_ready: {}",
        proof.reserve_first_restage_ready
    );
    println!(
        "    materialized_payloads_before_headers_contract: {}",
        proof.payloads_before_headers_contract
    );
    println!(
        "    materialized_release_header_store_contract: {}",
        proof.release_header_store_contract
    );
    println!(
        "    materialized_doorbell_pending: {}",
        proof.doorbell_pending
    );
    println!(
        "    materialized_no_live_queue_mutation_contract: {}",
        proof.no_live_queue_mutation_contract
    );
    println!(
        "    materialized_packet_plan_ready: {}",
        proof.packet_plan_ready
    );
    println!(
        "    materialized_publish_low32: 0x{:08x}",
        proof.publish_low32
    );
    println!(
        "    materialized_packet0_low32: 0x{:08x}",
        proof.packet0_low32
    );
    println!(
        "    materialized_aql_packet_image_ready: {}",
        proof.aql_packet_image_ready
    );
}

pub(crate) fn qwen_print_live_aql_shadow_packet_store_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlShadowPacketStoreProof,
) {
    println!("  resident_layer_runner_{target_stage_label}_live_aql_shadow_packet_store_stage:");
    println!("    source: resident_layer_runner_{target_stage_label}_live_aql_materialized_packet_plan_stage");
    println!(
        "    proof: gpu_stores_two_materialized_aql_packets_to_dedicated_shadow_buffer_header_last"
    );
    println!("    shadow_packet_store_contract: dedicated_128b_packet_region_payload_words_before_low32_release_headers");
    println!(
        "    shadow_packet_store_device_va: 0x{:016x}",
        proof.device_va
    );
    println!(
        "    shadow_packet_store_requested_iterations: {}",
        proof.requested_iterations
    );
    println!(
        "    shadow_packet_store_executed_iterations: {}",
        proof.executed_iterations
    );
    println!(
        "    shadow_packet_store_present: {}",
        proof.observed_present
    );
    println!("    shadow_packet0_word0: 0x{:016x}", proof.packet0_word0);
    println!("    shadow_packet1_word0: 0x{:016x}", proof.packet1_word0);
    println!(
        "    shadow_packet_words_match_host_template: {}",
        proof.words_match_host_template
    );
    println!(
        "    shadow_packet_payload_words_match_host_template: {}",
        proof.payload_words_match_host_template
    );
    println!(
        "    shadow_packet_header_words_match_host_template: {}",
        proof.header_words_match_host_template
    );
    println!(
        "    shadow_packet_materialized_source_ready: {}",
        proof.materialized_source_ready
    );
    println!(
        "    shadow_packet_payloads_before_headers_contract: {}",
        proof.payloads_before_headers_contract
    );
    println!(
        "    shadow_packet_low32_release_headers_last_contract: {}",
        proof.low32_release_headers_last_contract
    );
    println!(
        "    shadow_packet_doorbell_pending: {}",
        proof.doorbell_pending
    );
    println!(
        "    shadow_packet_no_live_queue_mutation_contract: {}",
        proof.no_live_queue_mutation_contract
    );
    println!("    shadow_packet_region_bytes: {}", proof.region_bytes);
    println!("    shadow_packet_count: {}", proof.packet_count);
    println!("    shadow_packet_store_ready: {}", proof.store_ready);
    println!(
        "    shadow_packet_batch_plan_ready: {}",
        proof.batch_plan_ready
    );
    println!("    shadow_packet_handoff_ready: {}", proof.handoff_ready);
}

pub(crate) fn qwen_print_live_aql_host_poll_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlHostPollProof,
) {
    let validation = (*proof).validate_acquire_only();
    println!(
        "  resident_layer_runner_{target_stage_label}_live_aql_shadow_packet_host_poll_stage:"
    );
    println!(
        "    source: resident_layer_runner_{target_stage_label}_live_aql_shadow_packet_store_stage"
    );
    println!("    proof: host_acquire_polls_shadow_packet_headers_and_sequence_before_device_wait");
    println!("    host_poll_contract: acquire_header_loads_timeout_no_fetch_add_no_doorbell");
    println!(
        "    host_poll_expected_low32_header: 0x{:08x}",
        proof.expected_low32_header
    );
    println!("    host_poll_header0: 0x{:08x}", proof.header0);
    println!("    host_poll_header1: 0x{:08x}", proof.header1);
    println!("    host_poll_sequence: {}", proof.sequence);
    println!(
        "    host_poll_expected_sequence: {}",
        proof.expected_sequence
    );
    println!("    host_poll_sentinel: 0x{:016x}", proof.sentinel);
    println!("    host_poll_spins: {}", proof.spins);
    println!("    host_poll_elapsed_us: {:.3}", proof.elapsed_us);
    println!("    host_poll_timeout_ms: {:.3}", proof.timeout_ms);
    println!(
        "    host_poll_ready_before_device_wait: {}",
        proof.ready_before_device_wait == 1
    );
    println!(
        "    host_poll_fetch_add_performed: {}",
        proof.fetch_add_performed
    );
    println!("    host_poll_doorbell_written: {}", proof.doorbell_written);
    println!(
        "    host_poll_live_queue_mutated: {}",
        proof.live_queue_mutated
    );
    println!("    host_poll_validation_passed: {}", validation.passed);
}

pub(crate) fn qwen_print_live_aql_admission_guard_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlAdmissionGuardProof,
) {
    println!("  resident_layer_runner_{target_stage_label}_live_aql_admission_guard_stage:");
    println!("    source: resident_layer_runner_{target_stage_label}_live_aql_shadow_packet_host_poll_stage");
    println!(
        "    proof: host_derives_non_submitting_live_aql_admission_token_from_shadow_packet_gate"
    );
    println!("    admission_guard_contract: fail_closed_shadow_validated_no_fetch_add_no_doorbell");
    println!(
        "    admission_shadow_words_match: {}",
        proof.shadow_words_match
    );
    println!(
        "    admission_header_acquire_match: {}",
        proof.header_acquire_match
    );
    println!("    admission_sequence_match: {}", proof.sequence_match);
    println!(
        "    admission_reservation_ready: {}",
        proof.reservation_ready
    );
    println!("    admission_restage_ready: {}", proof.restage_ready);
    println!("    admission_batch_ready: {}", proof.batch_ready);
    println!(
        "    admission_materialized_ready: {}",
        proof.materialized_ready
    );
    println!(
        "    admission_shadow_store_ready: {}",
        proof.shadow_store_ready
    );
    println!("    admission_host_poll_ready: {}", proof.host_poll_ready);
    println!(
        "    admission_host_poll_validated: {}",
        proof.host_poll_validated
    );
    println!("    admission_prereqs_ready: {}", proof.prereqs_ready);
    println!(
        "    admission_no_live_queue_mutation_contract: {}",
        proof.no_live_mutation_contract
    );
    println!("    admission_token_ready: {}", proof.token_ready);
    println!("    admission_submit_enabled: {}", proof.submit_enabled);
    println!("    admission_submit_allowed: {}", proof.submit_allowed);
    println!("    admission_status: armed_shadow_validated_not_submitted");
    println!("    admission_fetch_add_performed: false");
    println!("    admission_doorbell_written: false");
    println!("    admission_live_queue_mutated: false");
}

pub(crate) fn qwen_print_live_aql_slot_preflight_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlSlotPreflightProof,
) {
    println!("  resident_layer_runner_{target_stage_label}_live_aql_live_slot_preflight_stage:");
    println!(
        "    source: resident_layer_runner_{target_stage_label}_live_aql_admission_guard_stage"
    );
    println!(
        "    proof: host_derives_disabled_live_slot_preflight_without_queue_ownership_transfer"
    );
    println!("    live_slot_preflight_contract: offline_invalid_template_admitted_shadow_packet_future_write_blocked");
    println!(
        "    live_slot_preflight_offline_template_header_invalid: {}",
        proof.offline_template_header_invalid
    );
    println!(
        "    live_slot_preflight_packet_template_ready: {}",
        proof.packet_template_ready
    );
    println!(
        "    live_slot_preflight_admission_token_ready: {}",
        proof.admission_token_ready
    );
    println!(
        "    live_slot_preflight_admission_validated: {}",
        proof.admission_validated
    );
    println!(
        "    live_slot_preflight_future_write_blocked: {}",
        proof.future_write_blocked
    );
    println!(
        "    live_slot_preflight_no_ownership_transfer: {}",
        proof.no_ownership_transfer
    );
    println!(
        "    live_slot_preflight_first_slot_matches_reservation: {}",
        proof.first_slot_matches_reservation
    );
    println!(
        "    live_slot_preflight_reservation_ready: {}",
        proof.reservation_ready
    );
    println!("    live_slot_preflight_ready: {}", proof.ready);
    println!(
        "    live_slot_preflight_live_write_allowed: {}",
        proof.live_write_allowed
    );
    println!("    live_slot_preflight_status: dry_run_blocked_live_write_not_submitted");
    println!("    live_slot_preflight_fetch_add_performed: false");
    println!("    live_slot_preflight_doorbell_written: false");
    println!("    live_slot_preflight_live_queue_mutated: false");
}

pub(crate) fn qwen_print_live_aql_header_probe_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlHeaderProbeProof,
) {
    println!("  resident_layer_runner_{target_stage_label}_live_aql_live_slot_header_probe_stage:");
    println!(
        "    source: resident_layer_runner_{target_stage_label}_live_aql_live_slot_preflight_stage"
    );
    println!(
        "    proof: gpu_acquire_reads_planned_live_slot_headers_without_queue_ownership_transfer"
    );
    println!("    live_slot_header_probe_contract: read_only_headers_future_copy_blocked_no_fetch_add_no_doorbell");
    println!(
        "    live_slot_header_probe_slot0_va: 0x{:016x}",
        proof.slot0_va
    );
    println!(
        "    live_slot_header_probe_slot1_va: 0x{:016x}",
        proof.slot1_va
    );
    println!(
        "    live_slot_header_probe_slot0_offset: {}",
        proof.slot0_offset
    );
    println!(
        "    live_slot_header_probe_slot1_offset: {}",
        proof.slot1_offset
    );
    println!(
        "    live_slot_header_probe_slot0_low32: 0x{:08x}",
        proof.slot0_low32
    );
    println!(
        "    live_slot_header_probe_slot1_low32: 0x{:08x}",
        proof.slot1_low32
    );
    println!(
        "    live_slot_header_probe_slot0_type: {}",
        proof.slot0_type
    );
    println!(
        "    live_slot_header_probe_slot1_type: {}",
        proof.slot1_type
    );
    println!(
        "    live_slot_header_probe_slot0_not_target_publish: {}",
        proof.slot0_not_target_publish
    );
    println!(
        "    live_slot_header_probe_slot1_not_target_publish: {}",
        proof.slot1_not_target_publish
    );
    println!(
        "    live_slot_header_probe_targets_match_batch_plan: {}",
        proof.targets_match_batch_plan
    );
    println!(
        "    live_slot_header_probe_read_only_contract: {}",
        proof.read_only_contract
    );
    println!(
        "    live_slot_header_probe_fetch_add_not_performed: {}",
        proof.fetch_add_not_performed
    );
    println!(
        "    live_slot_header_probe_doorbell_not_written: {}",
        proof.doorbell_not_written
    );
    println!(
        "    live_slot_header_probe_live_slot_not_written: {}",
        proof.live_slot_not_written
    );
    println!(
        "    live_slot_header_probe_future_copy_blocked: {}",
        proof.future_copy_blocked
    );
    println!(
        "    live_slot_header_probe_preflight_ready: {}",
        proof.live_slot_preflight_ready
    );
    println!(
        "    live_slot_header_probe_preflight_validated: {}",
        proof.live_slot_preflight_validated
    );
    println!("    live_slot_header_probe_ready: {}", proof.ready);
    println!(
        "    live_slot_header_probe_expected_publish_low32: 0x{:08x}",
        proof.expected_publish_low32
    );
    println!(
        "    live_slot_header_probe_live_write_allowed: {}",
        proof.live_write_allowed
    );
    println!(
        "    live_slot_header_probe_no_mutation_contract: {}",
        proof.no_mutation_contract
    );
}

pub(crate) fn qwen_print_live_aql_copy_decision_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueLiveAqlCopyDecisionProof,
) {
    println!(
        "  resident_layer_runner_{target_stage_label}_live_aql_live_slot_copy_decision_stage:"
    );
    println!("    source: resident_layer_runner_{target_stage_label}_live_aql_live_slot_header_probe_stage");
    println!("    proof: host_fail_closed_blocks_live_copy_from_observed_live_slot_headers");
    println!("    live_slot_copy_decision_contract: block_on_target_publish_or_unknown_header_no_reset_no_copy");
    println!(
        "    live_slot_copy_decision_slot0_reason_code: {}",
        proof.slot0_reason
    );
    println!(
        "    live_slot_copy_decision_slot1_reason_code: {}",
        proof.slot1_reason
    );
    println!(
        "    live_slot_copy_decision_slot0_reason: {}",
        qwen_live_aql_copy_reason_label(proof.slot0_reason)
    );
    println!(
        "    live_slot_copy_decision_slot1_reason: {}",
        qwen_live_aql_copy_reason_label(proof.slot1_reason)
    );
    println!(
        "    live_slot_copy_decision_any_header_block: {}",
        proof.any_header_block
    );
    println!(
        "    live_slot_copy_decision_requires_cleanup: {}",
        proof.requires_cleanup
    );
    println!(
        "    live_slot_copy_decision_header_probe_ready: {}",
        proof.header_probe_ready
    );
    println!(
        "    live_slot_copy_decision_header_probe_validated: {}",
        proof.header_probe_validated
    );
    println!(
        "    live_slot_copy_decision_header_reset_allowed: {}",
        proof.header_reset_allowed
    );
    println!(
        "    live_slot_copy_decision_copy_allowed: {}",
        proof.copy_allowed
    );
    println!("    live_slot_copy_decision_ready: {}", proof.ready);
    println!("    live_slot_copy_decision_status: blocked_header_observed_not_reset_not_copied");
    println!("    live_slot_copy_decision_fetch_add_performed: false");
    println!("    live_slot_copy_decision_doorbell_written: false");
    println!("    live_slot_copy_decision_live_queue_mutated: false");
}

fn qwen_live_aql_copy_reason_label(reason: u64) -> &'static str {
    match reason {
        0 => "slot_header_zero",
        1 => "target_publish_header_present",
        _ => "unknown_nonzero_header_present",
    }
}

pub(crate) fn qwen_print_live_aql_cleanup_preflight_proof(
    target_stage_label: &str,
    proof: &mcore::KfdQueueCleanupPreflightProof,
) {
    println!("  resident_layer_runner_{target_stage_label}_live_aql_cleanup_preflight_stage:");
    println!("    source: resident_layer_runner_{target_stage_label}_live_aql_live_slot_copy_decision_stage");
    println!("    proof: host_derives_cleanup_eligibility_from_read_index_without_reset");
    println!("    cleanup_preflight_contract: reset_only_after_read_index_passes_blocked_packet_no_reset_no_copy");
    println!(
        "    cleanup_preflight_gpu_read_index: {}",
        proof.gpu_read_index
    );
    println!(
        "    cleanup_preflight_host_snapshot_read_index: {}",
        proof.host_snapshot_read_index
    );
    println!(
        "    cleanup_preflight_gpu_read_index_matches_host_snapshot: {}",
        proof.gpu_read_index_matches_host_snapshot
    );
    println!(
        "    cleanup_preflight_gpu_read_index_matches_reference_lane: {}",
        proof.gpu_read_index_matches_reference
    );
    println!(
        "    cleanup_preflight_gpu_read_index_not_behind_host_snapshot: {}",
        proof.gpu_read_index_not_behind_host_snapshot
    );
    println!("    cleanup_preflight_ring_slots: {}", proof.ring_slots);
    println!(
        "    cleanup_preflight_slot0_target_packet_id: {}",
        proof.slot0_target_packet_id
    );
    println!(
        "    cleanup_preflight_slot1_target_packet_id: {}",
        proof.slot1_target_packet_id
    );
    println!(
        "    cleanup_preflight_slot0_blocked_packet_id: {}",
        proof.slot0_blocked_packet_id
    );
    println!(
        "    cleanup_preflight_slot1_blocked_packet_id: {}",
        proof.slot1_blocked_packet_id
    );
    println!(
        "    cleanup_preflight_slot0_blocked_id_known: {}",
        proof.slot0_blocked_id_known
    );
    println!(
        "    cleanup_preflight_slot1_blocked_id_known: {}",
        proof.slot1_blocked_id_known
    );
    println!(
        "    cleanup_preflight_slot0_read_index_passed: {}",
        proof.slot0_read_index_passed
    );
    println!(
        "    cleanup_preflight_slot1_read_index_passed: {}",
        proof.slot1_read_index_passed
    );
    println!(
        "    cleanup_preflight_observed_block: {}",
        proof.observed_block
    );
    println!(
        "    cleanup_preflight_copy_decision_ready: {}",
        proof.copy_decision_ready
    );
    println!(
        "    cleanup_preflight_copy_decision_validated: {}",
        proof.copy_decision_validated
    );
    println!(
        "    cleanup_preflight_any_reset_eligible: {}",
        proof.any_reset_eligible
    );
    println!(
        "    cleanup_preflight_reset_allowed: {}",
        proof.reset_allowed
    );
    println!("    cleanup_preflight_copy_allowed: {}", proof.copy_allowed);
    println!("    cleanup_preflight_ready: {}", proof.ready);
    println!("    cleanup_preflight_status: observed_block_not_reset_not_copied");
    println!("    cleanup_preflight_fetch_add_performed: false");
    println!("    cleanup_preflight_doorbell_written: false");
    println!("    cleanup_preflight_live_queue_mutated: false");
    println!("    aql_submitted: false");
    println!("    real_doorbell_rung: false");
    println!("    live_hsa_queue_mutated: false");
    println!(
        "    consumer_status: gpu_live_aql_reservation_restage_batch_plan_computed_not_submitted"
    );
}

# SPDX-License-Identifier: Apache-2.0
"""Scheduler handling for request-terminal PE-to-DE ABORT notices."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_empty_scheduler_output,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_read_recovery import (
    _RESUME_BLOCKS,
    _admit_de_read,
    _make_de_read_request,
    _resume,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)

from tests.ut.distributed.kv_transfer.dual_path.test_reverse_send_completion import (
    _admit_decode_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_prefill_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import (
    path_decision as decision_model,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import (
    scheduler as scheduler_module,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.completion_tracker import (
    CompletionKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    ReverseAttemptKey,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
)


_PREFILL_CONTROL_PORT = 24002
_PREFILL_CONTROL_ENDPOINT = DecodeControlEndpoint(
    host="192.0.2.44",
    port=_PREFILL_CONTROL_PORT,
)
_EXPECTED_REVERSE_INVALID_BLOCKS = (71, 72)
_EXPECTED_REFRESH_REVERSE_INVALID_BLOCKS = (82,)
_ABSENT_REVERSE_ADMISSION_TERMINAL = object()


def _notice(
    request_key,
    reason: str = "REQUEST_ABORTED",
    *,
    reverse_attempt_id: int | None = None,
    may_have_started_through_attempt_id=_ABSENT_REVERSE_ADMISSION_TERMINAL,
):
    return decision_model.PathAbortNotice(
        request_key=request_key,
        reason=decision_model.PathAbortReason(reason),
        reverse_terminal=(
            None
            if reverse_attempt_id is None
            else decision_model.ReverseTerminalNotice(
                reverse_attempt_id=reverse_attempt_id,
                state=decision_model.ReverseTerminalState.TERMINALIZED,
            )
        ),
        reverse_admission_terminal=(
            None
            if may_have_started_through_attempt_id
            is _ABSENT_REVERSE_ADMISSION_TERMINAL
            else decision_model.ReverseAdmissionTerminalNotice(
                may_have_started_through_attempt_id
            )
        ),
    )


def test_pending_decode_abort_stages_peer_abort_failure(decode_scheduler_factory, decode_control_seams) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decode_control_seams.decode_coordinator.take_received_decisions.return_value = []
    decode_control_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert state.reverse_admission_terminal == (
        decision_model.ReverseAdmissionTerminalNotice(None)
    )
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.PEER_ABORT,
        )
    ]
    assert scheduler._decode_control_failures == {}
    decode_control_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)
    decode_control_seams.decode_coordinator.submit_abort.assert_not_called()


def test_same_drain_commits_decision_then_aborts_without_touching_completion_tracker(
    decode_scheduler_factory,
    decode_control_seams,
) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decision = _de_read_decision()
    decode_control_seams.decode_coordinator.take_received_decisions.return_value = [decision]
    decode_control_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert len(metadata.reverse_plans) == 1
    completion_id = metadata.reverse_plans[0].reverse_send_completion_id
    assert completion_id is not None
    completion = scheduler._completion_tracker.get(completion_id)
    assert completion is not None
    assert completion.closed is False
    assert completion.failed is False
    assert scheduler._completion_tracker.open_count() == 1
    assert len(scheduler._reverse_send_completion_ids) == 1
    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert state.reverse_admission_terminal == (
        decision_model.ReverseAdmissionTerminalNotice(0)
    )
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.PEER_ABORT,
        )
    ]


def test_decode_final_publication_gate_rejects_admission_frozen_during_validation(
    decode_scheduler_factory,
) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decision = _de_read_decision()
    snapshot = scheduler._decode_kv_snapshots[request.request_id]
    validate = scheduler._validate_committed_decision

    def validate_then_freeze(*args, **kwargs):
        validated = validate(*args, **kwargs)
        scheduler._fail_decode_admission(
            state,
            snapshot,
            local_reason=DualPathControlFailureReason.ACTIVATION_FAILED,
            peer_endpoint=decision.prefill_control_endpoint,
            reverse_attempt_id=decision.result.reverse_attempt_id,
        )
        return validated

    with patch.object(scheduler, "_validate_committed_decision", side_effect=validate_then_freeze):
        metadata = scheduler_module.DualPathConnectorMetadata()
        scheduler._activate_received_decision(decision, metadata)

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert state.reverse_admission_terminal == decision_model.ReverseAdmissionTerminalNotice(None)
    assert metadata.reverse_plans == []
    assert scheduler._completion_tracker.open_count() == 0
    assert scheduler._reverse_send_completion_ids == {}
    scheduler._kvpool_adapter.commit_after_alloc.assert_not_called()


def test_decode_final_publication_gate_rejects_admission_frozen_during_store_commit(
    decode_scheduler_factory,
) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decision = _de_read_decision()
    snapshot = scheduler._decode_kv_snapshots[request.request_id]

    def commit_then_freeze(*args, **kwargs):
        scheduler._fail_decode_admission(
            state,
            snapshot,
            local_reason=DualPathControlFailureReason.ACTIVATION_FAILED,
            peer_endpoint=decision.prefill_control_endpoint,
            reverse_attempt_id=decision.result.reverse_attempt_id,
        )

    scheduler._kvpool_adapter.commit_after_alloc.side_effect = commit_then_freeze
    metadata = scheduler_module.DualPathConnectorMetadata()

    scheduler._activate_received_decision(decision, metadata)

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert state.reverse_admission_terminal == (
        decision_model.ReverseAdmissionTerminalNotice(None)
    )
    assert metadata.reverse_plans == []
    assert scheduler._completion_tracker.open_count() == 0
    assert scheduler._reverse_send_completion_ids == {}
    scheduler._kvpool_adapter.commit_after_alloc.assert_called_once()


def test_unknown_decode_abort_is_ignored(decode_scheduler_factory, decode_control_seams) -> None:
    scheduler = decode_scheduler_factory()
    unknown_key = decision_model.DualPathRequestKey(
        decode_control_seams.decode_coordinator.decode_engine_instance_id,
        "unknown-request",
        0,
    )

    scheduler._handle_received_abort(_notice(unknown_key))

    assert scheduler._decode_control_failures == {}
    decode_control_seams.decode_coordinator.unregister.assert_not_called()


def test_already_terminal_decode_abort_is_ignored(decode_scheduler_factory, decode_control_seams) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    state.status = scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED

    scheduler._handle_received_abort(_notice(state.request_key))

    assert scheduler._decode_control_failures == {}
    decode_control_seams.decode_coordinator.unregister.assert_not_called()


def test_queued_old_admission_abort_does_not_fail_reused_request_id(
    decode_scheduler_factory,
    decode_control_seams,
) -> None:
    scheduler = decode_scheduler_factory()
    first_request = _admit_decode_request(scheduler)
    first_state = scheduler._decode_decision_states[first_request.request_id]
    old_notice = _notice(first_state.request_key)
    scheduler._release_scheduler_request_state(first_request)

    second_request = _admit_decode_request(scheduler)
    second_state = scheduler._decode_decision_states[second_request.request_id]
    assert second_state.request_key.admission_id == first_state.request_key.admission_id + 1
    decode_control_seams.decode_coordinator.unregister.reset_mock()
    decode_control_seams.decode_coordinator.take_received_decisions.return_value = []
    decode_control_seams.decode_coordinator.take_received_aborts.return_value = [old_notice]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert second_state.status is scheduler_module._DecodeDecisionStatus.PENDING
    assert metadata.control_failures == []
    decode_control_seams.decode_coordinator.unregister.assert_not_called()


def test_decode_abort_keeps_terminal_state_when_failure_metadata_cannot_be_built(
    decode_scheduler_factory,
    decode_control_seams,
) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decode_control_seams.decode_coordinator.take_received_decisions.return_value = []
    decode_control_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

    with (
        patch.object(
            scheduler,
            "_build_decode_control_failure",
            side_effect=RuntimeError("unaligned snapshot"),
        ),
        patch.object(scheduler_module.logger, "error") as log_error,
    ):
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert metadata.control_failures == []
    decode_control_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)
    log_error.assert_called_once()


def _prefill_de_read_scheduler(pe_scheduler_factory, *, expected_worker_count: int = 1):
    scheduler, coordinator = pe_scheduler_factory(
        decision_model.PathKind.DE_READ,
        pool=make_block_pool(),
        prefill_control_port=_PREFILL_CONTROL_PORT,
    )
    scheduler._expected_worker_count = expected_worker_count
    request = _admit_de_read(scheduler)
    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    return scheduler, coordinator, request, binding


def test_prefill_admission_registers_abort_key_before_decision_submit(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    request_key = binding.request_key

    register_call = call.register_prefill_abort_key(request_key)
    submit_call = coordinator.submit.call_args
    submit_method_call = call.submit(*submit_call.args, **submit_call.kwargs)
    assert register_call in coordinator.method_calls
    assert coordinator.method_calls.index(register_call) < coordinator.method_calls.index(submit_method_call)
    assert scheduler._prefill_control_endpoint == _PREFILL_CONTROL_ENDPOINT
    assert submit_call.args[1].prefill_control_endpoint == _PREFILL_CONTROL_ENDPOINT
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}


def test_prefill_received_abort_with_absent_ceiling_invalidates_without_closing_completion(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    coordinator.take_received_aborts.return_value = [
        _notice(binding.request_key, "ACTIVATION_FAILED")
    ]

    metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

    completion = scheduler._completion_tracker.get(binding.reverse_receive_completion_id)
    assert metadata.reverse_receive_bindings == [binding]
    assert metadata.reverse_receive_failure_terminals == []
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=_EXPECTED_REVERSE_INVALID_BLOCKS,
            reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
        )
    ]
    assert completion is not None
    assert completion.dispatched is True
    assert completion.failed is False
    assert completion.closed is False
    assert scheduler._waiting_reverse_attempt_ids[request.request_id] == ReverseAttemptKey(
        binding.request_key,
        binding.reverse_attempt_id,
    )
    assert scheduler._scheduler_side_finished_recving == set()
    assert binding.request_key in scheduler._prefill_invalid_request_keys


def test_prefill_received_terminalized_abort_fail_closes_unstarted_reverse_receive(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    scheduler._pending_finished_recving.add(request.request_id)
    coordinator.take_received_aborts.return_value = [
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=binding.reverse_attempt_id,
        )
    ]

    metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

    assert metadata.reverse_receive_bindings == []
    assert metadata.reverse_receive_failure_terminals == []
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=_EXPECTED_REVERSE_INVALID_BLOCKS,
            reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
        )
    ]
    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None
    assert request.request_id not in scheduler._waiting_reverse_attempt_ids
    assert request.request_id not in scheduler._pending_finished_recving
    assert scheduler._scheduler_side_finished_recving == {request.request_id}
    assert binding.request_key in scheduler._prefill_invalid_request_keys


def test_prefill_peer_abort_does_not_bypass_started_worker_barrier(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(
        pe_scheduler_factory,
        expected_worker_count=2,
    )
    binding_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    first_worker = _make_prefill_worker()
    second_worker = _make_prefill_worker()
    first_worker.start_load_kv(binding_metadata)
    second_worker.start_load_kv(binding_metadata)
    coordinator.take_received_aborts.return_value = [
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=binding.reverse_attempt_id,
        )
    ]

    terminal_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

    completion = scheduler._completion_tracker.get(binding.reverse_receive_completion_id)
    assert completion is not None
    assert completion.completed_worker_count == 0
    assert completion.failed is True
    assert completion.closed is False
    assert binding.request_key in scheduler._prefill_invalid_request_keys
    assert scheduler._scheduler_side_finished_recving == set()
    assert scheduler._waiting_reverse_attempt_ids[request.request_id].request_key == binding.request_key
    assert len(terminal_metadata.reverse_receive_failure_terminals) == 1

    first_worker.start_load_kv(terminal_metadata)
    second_worker.start_load_kv(terminal_metadata)
    first_report = first_worker.build_connector_worker_meta()
    second_report = second_worker.build_connector_worker_meta()
    assert first_report is not None
    assert second_report is not None
    assert first_report.failure_reports == {
        binding.reverse_receive_completion_id: 1,
    }
    assert second_report.failure_reports == {
        binding.reverse_receive_completion_id: 1,
    }

    first_output = KVConnectorOutput(kv_connector_worker_meta=first_report)
    scheduler.update_connector_output(first_output)

    completion = scheduler._completion_tracker.get(binding.reverse_receive_completion_id)
    assert completion is not None
    assert completion.completed_worker_count == 1
    assert completion.failed is True
    assert completion.closed is False
    assert first_output.finished_recving is None

    final_output = KVConnectorOutput(kv_connector_worker_meta=second_report)
    scheduler.update_connector_output(final_output)
    assert final_output.finished_recving == {request.request_id}
    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None


def test_prefill_peer_abort_after_binding_dispatch_closes_through_real_worker_report(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    first_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    assert first_metadata.reverse_receive_bindings == [binding]
    worker = _make_prefill_worker()
    worker.start_load_kv(first_metadata)
    request.status = RequestStatus.FINISHED_STOPPED
    assert scheduler.request_finished(request, []) == (True, None)
    coordinator.take_received_aborts.return_value = [
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=binding.reverse_attempt_id,
        )
    ]

    second_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    assert second_metadata.reverse_receive_bindings == []
    worker.start_load_kv(second_metadata)
    worker_report = worker.build_connector_worker_meta()
    assert worker_report is not None

    final_output = KVConnectorOutput(kv_connector_worker_meta=worker_report)
    scheduler.update_connector_output(final_output)
    assert final_output.finished_recving == {request.request_id}
    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None


def test_duplicate_terminalized_abort_does_not_redispatch_before_worker_report(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    binding_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    worker = _make_prefill_worker()
    worker.start_load_kv(binding_metadata)
    notice = _notice(
        binding.request_key,
        "ACTIVATION_FAILED",
        reverse_attempt_id=binding.reverse_attempt_id,
    )

    coordinator.take_received_aborts.return_value = [notice]
    first_terminal_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    coordinator.take_received_aborts.return_value = [notice]
    duplicate_terminal_metadata = scheduler.build_connector_meta(
        make_empty_scheduler_output()
    )

    attempt_key = ReverseAttemptKey(
        binding.request_key,
        binding.reverse_attempt_id,
    )
    assert len(first_terminal_metadata.reverse_receive_failure_terminals) == 1
    assert duplicate_terminal_metadata.reverse_receive_failure_terminals == []
    assert scheduler._prefill_staged_or_delivered_reverse_terminals == {attempt_key}

    worker.start_load_kv(first_terminal_metadata)
    worker_report = worker.build_connector_worker_meta()
    assert worker_report is not None
    output = KVConnectorOutput(kv_connector_worker_meta=worker_report)
    scheduler.update_connector_output(output)

    assert output.finished_recving == {request.request_id}
    assert scheduler._prefill_staged_or_delivered_reverse_terminals == set()


def test_exact_stale_attempt_proof_invalidates_current_plan_without_closing_current_attempt(
    pe_scheduler_factory,
) -> None:
    scheduler, _, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    scheduler.build_connector_meta(make_empty_scheduler_output())
    stale_attempt = ReverseAttemptKey(
        binding.request_key,
        binding.reverse_attempt_id,
    )
    assert _resume(scheduler, request, 32, _RESUME_BLOCKS) == (16, True)
    live_binding = scheduler._prefill_pending_reverse_receive_bindings[
        request.request_id
    ]
    live_attempt = ReverseAttemptKey(
        live_binding.request_key,
        live_binding.reverse_attempt_id,
    )
    live_completion = scheduler._completion_tracker.get(
        live_binding.reverse_receive_completion_id
    )
    assert live_completion is not None
    assert scheduler._prefill_reverse_plans[
        request.request_id
    ].destination_block_ids == (tuple(_RESUME_BLOCKS),)

    scheduler._handle_received_peer_abort(
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=stale_attempt.reverse_attempt_id,
        )
    )
    terminal_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

    stale_completion = scheduler._completion_tracker.get(
        binding.reverse_receive_completion_id
    )
    assert stale_completion is not None
    assert stale_completion.failed is True
    assert stale_completion.closed is False
    assert live_completion.failed is False
    assert live_completion.closed is False
    assert scheduler._waiting_reverse_attempt_ids[request.request_id] == live_attempt
    assert [
        terminal.reverse_attempt_id
        for terminal in terminal_metadata.reverse_receive_failure_terminals
    ] == [stale_attempt.reverse_attempt_id]
    assert terminal_metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=_EXPECTED_REFRESH_REVERSE_INVALID_BLOCKS,
            reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
        )
    ]

    stale_output = KVConnectorOutput(
        kv_connector_worker_meta=make_worker_metadata(
            failure_reports={binding.reverse_receive_completion_id: 1},
        )
    )
    scheduler.update_connector_output(stale_output)

    assert stale_output.finished_recving is None
    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None
    assert scheduler._completion_tracker.get(live_completion.completion_id) is live_completion
    assert scheduler._waiting_reverse_attempt_ids[request.request_id] == live_attempt
    assert stale_attempt not in scheduler._prefill_staged_or_delivered_reverse_terminals


def test_prefill_ceiling_zero_authorizes_only_attempts_above_zero(
    pe_scheduler_factory,
) -> None:
    scheduler, _, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    attempt_0 = ReverseAttemptKey(binding.request_key, 0)
    attempt_0_completion = scheduler._completion_tracker.get(
        binding.reverse_receive_completion_id
    )
    assert attempt_0_completion is not None
    assert attempt_0_completion.dispatched is False
    attempt_1 = ReverseAttemptKey(binding.request_key, 1)
    attempt_1_completion = scheduler._completion_tracker.open_completion(
        CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=1,
        reverse_attempt_key=attempt_1,
    )
    scheduler._waiting_reverse_attempt_ids[request.request_id] = attempt_1

    scheduler._handle_received_peer_abort(
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            may_have_started_through_attempt_id=0,
        )
    )

    assert scheduler._completion_tracker.get(
        attempt_0_completion.completion_id
    ) is attempt_0_completion
    assert attempt_0_completion.failed is False
    assert attempt_0_completion.closed is False
    assert scheduler._completion_tracker.get(attempt_1_completion.completion_id) is None
    assert scheduler._scheduler_side_finished_recving == {request.request_id}
    assert scheduler._prefill_pending_reverse_receive_failure_terminals == {}
    assert scheduler._prefill_control_failures == {
        request.request_id: DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=_EXPECTED_REVERSE_INVALID_BLOCKS,
            reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
        )
    }
    assert scheduler._completion_tracker.open_attempt_keys(
        CompletionKind.REVERSE_RECEIVE,
        binding.request_key,
    ) == (attempt_0,)


def test_prefill_ceiling_keeps_dispatched_attempt_at_or_below_ceiling_open(
    pe_scheduler_factory,
) -> None:
    scheduler, _, request, binding = _prefill_de_read_scheduler(
        pe_scheduler_factory
    )
    attempt_0 = ReverseAttemptKey(binding.request_key, 0)
    scheduler.build_connector_meta(make_empty_scheduler_output())
    completion = scheduler._completion_tracker.get(
        binding.reverse_receive_completion_id
    )
    assert completion is not None
    assert completion.dispatched is True

    scheduler._handle_received_peer_abort(
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            may_have_started_through_attempt_id=0,
        )
    )

    assert scheduler._completion_tracker.get(completion.completion_id) is completion
    assert completion.failed is False
    assert completion.closed is False
    assert scheduler._prefill_pending_reverse_receive_failure_terminals == {}
    assert scheduler._prefill_staged_or_delivered_reverse_terminals == set()
    assert scheduler._completion_tracker.open_attempt_keys(
        CompletionKind.REVERSE_RECEIVE,
        binding.request_key,
    ) == (attempt_0,)


def test_prefill_ignores_exact_terminal_authority_without_terminalized_state(
    pe_scheduler_factory,
) -> None:
    scheduler, _, _, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    completion = scheduler._completion_tracker.get(
        binding.reverse_receive_completion_id
    )
    assert completion is not None
    reverse_terminal = decision_model.ReverseTerminalNotice(
        reverse_attempt_id=0,
        state=decision_model.ReverseTerminalState.TERMINALIZED,
    )
    object.__setattr__(reverse_terminal, "state", object())

    scheduler._handle_received_peer_abort(
        decision_model.PathAbortNotice(
            request_key=binding.request_key,
            reason=decision_model.PathAbortReason.ACTIVATION_FAILED,
            reverse_terminal=reverse_terminal,
        )
    )

    assert scheduler._completion_tracker.get(completion.completion_id) is completion
    assert completion.failed is False
    assert completion.closed is False
    assert scheduler._prefill_pending_reverse_receive_failure_terminals == {}


def test_prefill_explicit_null_ceiling_authorizes_every_open_attempt(
    pe_scheduler_factory,
) -> None:
    scheduler, _, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    attempt_1 = ReverseAttemptKey(binding.request_key, 1)
    attempt_1_completion = scheduler._completion_tracker.open_completion(
        CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=1,
        reverse_attempt_key=attempt_1,
    )
    scheduler._waiting_reverse_attempt_ids[request.request_id] = attempt_1

    scheduler._handle_received_peer_abort(
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            may_have_started_through_attempt_id=None,
        )
    )

    assert scheduler._completion_tracker.open_attempt_keys(
        CompletionKind.REVERSE_RECEIVE,
        binding.request_key,
    ) == ()
    assert scheduler._completion_tracker.get(
        binding.reverse_receive_completion_id
    ) is None
    assert scheduler._completion_tracker.get(attempt_1_completion.completion_id) is None
    assert scheduler._waiting_reverse_attempt_ids.get(request.request_id) is None
    assert scheduler._scheduler_side_finished_recving == {request.request_id}
    assert scheduler._prefill_pending_reverse_receive_failure_terminals == {}
    assert binding.request_key in scheduler._prefill_invalid_request_keys


def test_prefill_combined_exact_and_ceiling_closes_two_dispatched_attempts_through_real_tp2_barrier(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding_0 = _prefill_de_read_scheduler(
        pe_scheduler_factory,
        expected_worker_count=2,
    )
    workers = (_make_prefill_worker(), _make_prefill_worker())
    binding_0_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    for worker in workers:
        worker.start_load_kv(binding_0_metadata)

    assert _resume(scheduler, request, 32, _RESUME_BLOCKS) == (16, True)
    binding_1 = scheduler._prefill_pending_reverse_receive_bindings[
        request.request_id
    ]
    binding_1_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    for worker in workers:
        worker.start_load_kv(binding_1_metadata)

    coordinator.take_received_aborts.return_value = [
        _notice(
            binding_0.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=0,
            may_have_started_through_attempt_id=0,
        )
    ]
    terminal_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

    assert terminal_metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=_EXPECTED_REFRESH_REVERSE_INVALID_BLOCKS,
            reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
        )
    ]
    assert [
        terminal.reverse_attempt_id
        for terminal in terminal_metadata.reverse_receive_failure_terminals
    ] == [0, 1]
    for completion_id in (
        binding_0.reverse_receive_completion_id,
        binding_1.reverse_receive_completion_id,
    ):
        completion = scheduler._completion_tracker.get(completion_id)
        assert completion is not None
        assert completion.dispatched is True
        assert completion.failed is True
        assert completion.closed is False

    reports = []
    for worker in workers:
        worker.start_load_kv(terminal_metadata)
        report = worker.build_connector_worker_meta()
        assert report is not None
        assert report.failure_reports == {
            binding_0.reverse_receive_completion_id: 1,
            binding_1.reverse_receive_completion_id: 1,
        }
        reports.append(report)

    first_output = KVConnectorOutput(kv_connector_worker_meta=reports[0])
    scheduler.update_connector_output(first_output)
    assert first_output.finished_recving is None
    for completion_id in (
        binding_0.reverse_receive_completion_id,
        binding_1.reverse_receive_completion_id,
    ):
        completion = scheduler._completion_tracker.get(completion_id)
        assert completion is not None
        assert completion.completed_worker_count == 1
        assert completion.closed is False

    second_output = KVConnectorOutput(kv_connector_worker_meta=reports[1])
    scheduler.update_connector_output(second_output)
    assert second_output.finished_recving == {request.request_id}
    assert scheduler._completion_tracker.get(
        binding_0.reverse_receive_completion_id
    ) is None
    assert scheduler._completion_tracker.get(
        binding_1.reverse_receive_completion_id
    ) is None


def test_prefill_first_ceiling_survives_release_and_conflict_does_not_grant_authority(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding_0 = _prefill_de_read_scheduler(
        pe_scheduler_factory
    )
    worker = _make_prefill_worker()
    binding_0_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    worker.start_load_kv(binding_0_metadata)
    assert _resume(scheduler, request, 32, _RESUME_BLOCKS) == (16, True)
    binding_1 = scheduler._prefill_pending_reverse_receive_bindings[
        request.request_id
    ]
    binding_1_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
    worker.start_load_kv(binding_1_metadata)
    request_key = binding_0.request_key

    scheduler._handle_received_peer_abort(
        _notice(
            request_key,
            "ACTIVATION_FAILED",
            may_have_started_through_attempt_id=1,
        )
    )
    assert scheduler._prefill_observed_reverse_admission_terminals == {
        request_key: decision_model.ReverseAdmissionTerminalNotice(1)
    }
    request.status = RequestStatus.FINISHED_STOPPED
    assert scheduler.request_finished(request, []) == (True, None)
    assert request.request_id not in scheduler._prefill_request_keys
    assert scheduler._prefill_observed_reverse_admission_terminals == {
        request_key: decision_model.ReverseAdmissionTerminalNotice(1)
    }

    scheduler._handle_received_peer_abort(
        _notice(
            request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=0,
            may_have_started_through_attempt_id=0,
        )
    )
    attempt_0_completion = scheduler._completion_tracker.get(
        binding_0.reverse_receive_completion_id
    )
    attempt_1_completion = scheduler._completion_tracker.get(
        binding_1.reverse_receive_completion_id
    )
    assert scheduler._prefill_observed_reverse_admission_terminals[
        request_key
    ] == decision_model.ReverseAdmissionTerminalNotice(1)
    assert attempt_0_completion is not None
    assert attempt_0_completion.failed is True
    assert attempt_0_completion.closed is False
    assert attempt_1_completion is not None
    assert attempt_1_completion.failed is False
    assert attempt_1_completion.closed is False

    attempt_0_terminal_metadata = scheduler.build_connector_meta(
        make_empty_scheduler_output()
    )
    assert [
        terminal.reverse_attempt_id
        for terminal in attempt_0_terminal_metadata.reverse_receive_failure_terminals
    ] == [0]
    worker.start_load_kv(attempt_0_terminal_metadata)
    attempt_0_report = worker.build_connector_worker_meta()
    assert attempt_0_report is not None
    attempt_0_output = KVConnectorOutput(kv_connector_worker_meta=attempt_0_report)
    scheduler.update_connector_output(attempt_0_output)
    assert attempt_0_output.finished_recving is None
    assert request_key in scheduler._prefill_abort_request_ids
    assert request_key in scheduler._prefill_observed_reverse_admission_terminals

    scheduler._handle_received_peer_abort(
        _notice(
            request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=1,
            may_have_started_through_attempt_id=1,
        )
    )
    attempt_1_terminal_metadata = scheduler.build_connector_meta(
        make_empty_scheduler_output()
    )
    assert [
        terminal.reverse_attempt_id
        for terminal in attempt_1_terminal_metadata.reverse_receive_failure_terminals
    ] == [1]
    worker.start_load_kv(attempt_1_terminal_metadata)
    attempt_1_report = worker.build_connector_worker_meta()
    assert attempt_1_report is not None
    attempt_1_output = KVConnectorOutput(kv_connector_worker_meta=attempt_1_report)
    scheduler.update_connector_output(attempt_1_output)

    assert attempt_1_output.finished_recving == {request.request_id}
    assert request_key not in scheduler._prefill_abort_request_ids
    assert request_key not in scheduler._prefill_observed_reverse_admission_terminals
    coordinator.unregister_prefill_abort_key.assert_called_once_with(request_key)

    replacement = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
        admission_id=1,
    )
    assert scheduler.get_num_new_matched_tokens(replacement, 16) == (16, True)
    replacement_key = scheduler._prefill_request_keys[replacement.request_id]
    assert replacement_key != request_key
    assert replacement_key not in scheduler._prefill_observed_reverse_admission_terminals


@pytest.mark.parametrize(
    "abort_before_replacement_attempt",
    [False, True],
    ids=["abort-after-attempt", "abort-before-attempt"],
)
def test_prefill_same_id_replacement_waits_until_old_abort_terminal_is_delivered(
    pe_scheduler_factory,
    abort_before_replacement_attempt,
) -> None:
    scheduler, coordinator, first_request, first_binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    first_request.status = RequestStatus.FINISHED_STOPPED
    assert scheduler.request_finished(first_request, []) == (True, None)
    second_request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
        admission_id=1,
    )
    notice = _notice(
        first_binding.request_key,
        "ACTIVATION_FAILED",
        reverse_attempt_id=first_binding.reverse_attempt_id,
    )

    if abort_before_replacement_attempt:
        coordinator.take_received_aborts.return_value = [notice]
        scheduler.build_connector_meta(make_empty_scheduler_output())

    assert scheduler.get_num_new_matched_tokens(second_request, 16) == (None, True)
    assert second_request.request_id not in scheduler._prefill_request_keys
    if abort_before_replacement_attempt:
        assert first_binding.request_key not in scheduler._prefill_abort_request_ids
    else:
        assert scheduler._prefill_abort_request_ids.get(first_binding.request_key) == first_request.request_id

    if not abort_before_replacement_attempt:
        coordinator.take_received_aborts.return_value = [notice]
        scheduler.build_connector_meta(make_empty_scheduler_output())

    assert scheduler._scheduler_side_finished_recving == {first_request.request_id}
    assert scheduler.get_num_new_matched_tokens(second_request, 16) == (None, True)
    assert second_request.request_id not in scheduler._prefill_request_keys

    terminal_output = KVConnectorOutput(kv_connector_worker_meta=None)
    scheduler.update_connector_output(terminal_output)
    assert terminal_output.finished_recving == {first_request.request_id}

    assert scheduler.get_num_new_matched_tokens(second_request, 16) == (16, True)
    second_key = scheduler._prefill_request_keys[second_request.request_id]
    assert second_key.admission_id == 1


def test_old_admission_terminalized_abort_does_not_mutate_same_id_replacement(
    pe_scheduler_factory,
) -> None:
    scheduler, _, first_request, first_binding = _prefill_de_read_scheduler(
        pe_scheduler_factory
    )
    first_request.status = RequestStatus.FINISHED_STOPPED
    assert scheduler.request_finished(first_request, []) == (True, None)
    scheduler._handle_received_peer_abort(
        _notice(
            first_binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=first_binding.reverse_attempt_id,
        )
    )
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=None))

    replacement = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
        admission_id=1,
    )
    assert scheduler.get_num_new_matched_tokens(replacement, 16) == (16, True)
    scheduler.update_state_after_alloc(
        replacement,
        _blocks(([80, 81, 82, 83],)),
        16,
    )
    replacement_binding = scheduler._prefill_pending_reverse_receive_bindings[
        replacement.request_id
    ]
    replacement_completion = scheduler._completion_tracker.get(
        replacement_binding.reverse_receive_completion_id
    )
    assert replacement_binding.request_key.admission_id == 1
    assert replacement_completion is not None

    scheduler._handle_received_peer_abort(
        _notice(
            first_binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=first_binding.reverse_attempt_id,
        )
    )

    assert replacement_completion.failed is False
    assert replacement_completion.closed is False
    assert scheduler._waiting_reverse_attempt_ids[replacement.request_id] == ReverseAttemptKey(
        replacement_binding.request_key,
        replacement_binding.reverse_attempt_id,
    )
    assert scheduler._prefill_pending_reverse_receive_failure_terminals == {}
    assert replacement_binding.request_key not in scheduler._prefill_invalid_request_keys


def test_prefill_abort_after_request_finished_still_finds_delayed_completion(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    request_key = binding.request_key

    assert scheduler.request_finished(request, []) == (True, None)
    assert request.request_id not in scheduler._prefill_request_keys
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()

    coordinator.take_received_aborts.return_value = [
        _notice(
            request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=binding.reverse_attempt_id,
        )
    ]
    scheduler.build_connector_meta(make_empty_scheduler_output())

    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None
    assert scheduler._scheduler_side_finished_recving == {request.request_id}
    assert scheduler._prefill_abort_request_ids == {}
    assert request_key not in scheduler._prefill_invalid_request_keys
    coordinator.unregister_prefill_abort_key.assert_called_once_with(request_key)


def test_prefill_abort_registry_retires_only_after_request_release_and_last_completion_close(
    pe_scheduler_factory,
) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    request_key = binding.request_key

    scheduler._handle_received_peer_abort(
        _notice(
            request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=binding.reverse_attempt_id,
        )
    )

    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None
    assert request_key in scheduler._prefill_invalid_request_keys
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()

    scheduler.request_finished(request, [])

    assert scheduler._prefill_abort_request_ids == {}
    coordinator.unregister_prefill_abort_key.assert_called_once_with(request_key)


def test_prefill_received_abort_for_unknown_or_same_id_foreign_key_is_noop(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    request_key = binding.request_key
    foreign_key = DualPathRequestKey(
        request_key.decode_engine_instance_id,
        request_key.decode_request_id,
        request_key.admission_id + 1,
    )

    scheduler._handle_received_peer_abort(_notice(foreign_key, "ACTIVATION_FAILED"))

    completion = scheduler._completion_tracker.get(binding.reverse_receive_completion_id)
    assert completion is not None and completion.closed is False and completion.failed is False
    assert scheduler._prefill_invalid_request_keys == set()
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()

    malformed_finisher = SimpleNamespace(
        request_id=request.request_id,
        kv_transfer_params={"dual_path": {"unexpected": "shape"}},
    )
    scheduler._release_scheduler_request_state(malformed_finisher)
    assert scheduler._prefill_request_keys == {request.request_id: request_key}
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()


def test_scheduler_side_finished_recving_injects_without_worker_metadata(pe_scheduler_factory) -> None:
    scheduler, _, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    scheduler._handle_received_peer_abort(
        _notice(
            binding.request_key,
            "ACTIVATION_FAILED",
            reverse_attempt_id=binding.reverse_attempt_id,
        )
    )
    output = KVConnectorOutput(kv_connector_worker_meta=None)

    scheduler.update_connector_output(output)

    assert output.finished_recving == {request.request_id}
    assert scheduler._scheduler_side_finished_recving == set()


def test_prefill_stale_attempt_completion_close_retires_abort_registry(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    request_key = binding.request_key
    stale_completion = scheduler._completion_tracker.get(binding.reverse_receive_completion_id)
    assert stale_completion is not None
    live_attempt = ReverseAttemptKey(request_key, binding.reverse_attempt_id + 1)
    live_completion = scheduler._completion_tracker.open_completion(
        CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=1,
        reverse_attempt_key=live_attempt,
    )
    scheduler._waiting_reverse_attempt_ids[request.request_id] = live_attempt

    assert scheduler._completion_tracker.tally_reports(stale_completion.completion_id, success_count=1)
    assert scheduler._run_completion_close_action(stale_completion) == (set(), set())
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()

    assert scheduler.request_finished(request, []) == (True, None)
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()

    assert scheduler._completion_tracker.tally_reports(live_completion.completion_id, success_count=1)
    assert scheduler._run_completion_close_action(live_completion) == (set(), {request.request_id})
    assert scheduler._prefill_abort_request_ids == {}
    coordinator.unregister_prefill_abort_key.assert_called_once_with(request_key)


def test_prefill_activation_failure_retires_abort_key_before_any_submit(pe_scheduler_factory) -> None:
    scheduler, coordinator = pe_scheduler_factory(
        decision_model.PathKind.DE_READ,
        pool=make_block_pool(),
        prefill_control_port=_PREFILL_CONTROL_PORT,
    )
    request = _make_de_read_request()
    assert scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    request_key = scheduler._prefill_request_keys[request.request_id]

    with patch.object(
        scheduler,
        "_activate_de_read_path",
        side_effect=RuntimeError("activation failed before submit"),
    ):
        scheduler.update_state_after_alloc(request, _blocks(([70, 71, 72, 73, 74],)), 32)

    coordinator.submit.assert_not_called()
    assert scheduler._prefill_abort_request_ids == {}
    coordinator.unregister_prefill_abort_key.assert_called_once_with(request_key)

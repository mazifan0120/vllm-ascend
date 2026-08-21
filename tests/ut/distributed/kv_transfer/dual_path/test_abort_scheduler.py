# SPDX-License-Identifier: Apache-2.0
"""Scheduler handling for request-terminal PE-to-DE ABORT notices."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from vllm.v1.outputs import KVConnectorOutput

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_empty_scheduler_output,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_read_recovery import (
    _admit_de_read,
    _make_de_read_request,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import _blocks

from tests.ut.distributed.kv_transfer.dual_path.test_reverse_send_completion import (
    _admit_decode_request,
    _de_read_decision,
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


def _notice(request_key, reason: str = "REQUEST_ABORTED"):
    return decision_model.PathAbortNotice(
        request_key=request_key,
        reason=decision_model.PathAbortReason(reason),
    )


def test_pending_decode_abort_stages_peer_abort_failure(decode_scheduler_factory, decode_control_seams) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decode_control_seams.decode_coordinator.take_received_decisions.return_value = []
    decode_control_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.PEER_ABORT,
        )
    ]
    assert scheduler._decode_control_failures == {}
    decode_control_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)


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
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.PEER_ABORT,
        )
    ]


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


def test_prefill_received_abort_fail_closes_unstarted_reverse_receive(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    scheduler._pending_finished_recving.add(request.request_id)
    coordinator.take_received_aborts.return_value = [_notice(binding.request_key, "ACTIVATION_FAILED")]

    metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

    assert metadata.reverse_receive_bindings == []
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
    scheduler.build_connector_meta(make_empty_scheduler_output())
    first_output = KVConnectorOutput(
        kv_connector_worker_meta=make_worker_metadata(
            completion_reports={binding.reverse_receive_completion_id: 1},
        )
    )
    scheduler.update_connector_output(first_output)
    coordinator.take_received_aborts.return_value = [_notice(binding.request_key, "ACTIVATION_FAILED")]

    scheduler.build_connector_meta(make_empty_scheduler_output())

    completion = scheduler._completion_tracker.get(binding.reverse_receive_completion_id)
    assert completion is not None
    assert completion.completed_worker_count == 1
    assert completion.failed is True
    assert completion.closed is False
    assert binding.request_key in scheduler._prefill_invalid_request_keys
    assert scheduler._scheduler_side_finished_recving == set()
    assert scheduler._waiting_reverse_attempt_ids[request.request_id].request_key == binding.request_key

    final_output = KVConnectorOutput(
        kv_connector_worker_meta=make_worker_metadata(
            completion_reports={binding.reverse_receive_completion_id: 1},
        )
    )
    scheduler.update_connector_output(final_output)
    assert final_output.finished_recving == {request.request_id}
    assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None


def test_prefill_abort_after_request_finished_still_finds_delayed_completion(pe_scheduler_factory) -> None:
    scheduler, coordinator, request, binding = _prefill_de_read_scheduler(pe_scheduler_factory)
    request_key = binding.request_key

    assert scheduler.request_finished(request, []) == (True, None)
    assert request.request_id not in scheduler._prefill_request_keys
    assert scheduler._prefill_abort_request_ids == {request_key: request.request_id}
    coordinator.unregister_prefill_abort_key.assert_not_called()

    coordinator.take_received_aborts.return_value = [_notice(request_key, "ACTIVATION_FAILED")]
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

    scheduler._handle_received_peer_abort(_notice(request_key, "ACTIVATION_FAILED"))

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
    scheduler._handle_received_peer_abort(_notice(binding.request_key, "ACTIVATION_FAILED"))
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

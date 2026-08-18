# SPDX-License-Identifier: Apache-2.0
"""Decode Reverse sender completion reporting via the completion tracker."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    DECODE_TEST_INSTANCE_ID,
    layerwise_module,
    make_sender_req_meta,
    make_sending_layer_thread,
    make_worker_metadata,
    successful_terminal_ack_zmq_ctx,
)
from tests.ut.distributed.kv_transfer.dual_path.test_reverse_attempt_identity import (
    _attempt_key,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_prefill_worker,
    _make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision as path_decision_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    DualPathControlFailureReason,
    ReversePlan,
    ReverseReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionResult,
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    PathDecision,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)

_REQUEST_ID = "decode-request-7"
_REQUEST_KEY = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, _REQUEST_ID, 0)


def _admit_decode_request(scheduler, request_id: str = _REQUEST_ID) -> SimpleNamespace:
    scheduler._kvpool_adapter.lookup.return_value = LoadSpec(
        vllm_cached_tokens=16,
        kvpool_cached_tokens=32,
        can_load=False,
    )
    request = SimpleNamespace(
        request_id=request_id,
        num_tokens=49,
        prompt_token_ids=list(range(49)),
        kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://proxy.example/v1/kv"},
    )
    assert scheduler.get_num_new_matched_tokens(request, 16) == (33, True)
    blocks = MagicMock(name="admission_blocks")
    blocks.get_block_ids.return_value = ([41, 42, 43, 44],)
    scheduler.update_state_after_alloc(request, blocks, 33)
    return request


def _de_read_decision(reverse_attempt_id: int = 0, remote_tp_size: int = 1) -> PathDecision:
    return PathDecision(
        result=PathDecisionResult(
            request_key=_REQUEST_KEY,
            path=PathKind.DE_READ,
            reverse_attempt_id=reverse_attempt_id,
            prefill_local_tokens=16,
        ),
        reverse_plan=ReversePlan(
            request_key=_REQUEST_KEY,
            wire_request_id=path_decision_module.reverse_wire_id(_attempt_key(reverse_attempt_id, _REQUEST_KEY)),
            token_start=16,
            token_end=32,
            source_block_ids=((41, 42, 43, 44),),
            destination_block_ids=((71, 72, 73, 74),),
            remote_engine_id="prefill-engine",
            remote_host="198.51.100.10",
            remote_port=6000,
            remote_block_sizes=(16,),
            remote_tp_size=remote_tp_size,
            remote_pcp_size=1,
            remote_dcp_size=1,
            reverse_attempt_id=reverse_attempt_id,
            prefill_local_tokens=16,
            reverse_send_completion_id=None,
        ),
    )


def _activate_decision(scheduler, decode_control_seams, decision: PathDecision):
    decode_control_seams.decode_coordinator.take_received_decisions.return_value = [decision]
    return scheduler.build_connector_meta(MagicMock(name="scheduler_output"))


def _seed_reverse_send_tracker(worker, reverse_send_completion_id: int, reverse_attempt_id: int = 0):
    attempt_key = _attempt_key(reverse_attempt_id, _REQUEST_KEY)
    plan = replace(
        _de_read_decision(reverse_attempt_id).reverse_plan,
        reverse_send_completion_id=reverse_send_completion_id,
    )
    tracker = worker_module._SplitTracker(
        store_phase=worker_module._SplitPhase.SKIPPED,
        reverse_phase=worker_module._SplitPhase.PENDING,
        forward_phase=worker_module._SplitPhase.PENDING,
        store_destination_slice=(),
        forward_destination_slice=(70, 71),
        reverse_plan=plan,
        reverse_submitted_attempt=attempt_key,
        store_load_failed=False,
        terminal_reported=False,
    )
    worker._split_trackers[_REQUEST_ID] = tracker
    return tracker


def _activate_two_reverse_attempts(scheduler, decode_control_seams):
    request = _admit_decode_request(scheduler)
    first_metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(0))
    second_metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(1))
    first_completion_id = first_metadata.reverse_plans[0].reverse_send_completion_id
    second_completion_id = second_metadata.reverse_plans[0].reverse_send_completion_id
    assert first_completion_id is not None
    assert second_completion_id is not None
    return request, first_completion_id, second_completion_id


def _close_send_completion(scheduler, completion_id: int) -> KVConnectorOutput:
    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
    scheduler.update_connector_output(output)
    return output


def _fail_send_completion(scheduler, completion_id: int) -> KVConnectorOutput:
    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(failure_reports={completion_id: 1}))
    scheduler.update_connector_output(output)
    return output


def test_reverse_send_completion_allocated_at_attempt_creation_and_carried_on_plan(
    decode_scheduler_factory, decode_control_seams
):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)

    metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision())

    assert len(metadata.reverse_plans) == 1
    carried_plan = metadata.reverse_plans[0]
    assert carried_plan.reverse_send_completion_id is not None
    attempt_key = _attempt_key(0, _REQUEST_KEY)
    completion = scheduler._completion_tracker.get(carried_plan.reverse_send_completion_id)
    assert completion.completion_kind.name == "REVERSE_SEND"
    assert completion.reverse_attempt_key == attempt_key
    assert completion.expected_worker_count == 1
    assert scheduler._reverse_send_completion_ids[attempt_key] == completion.completion_id
    send_completion = scheduler._completion_tracker.get(scheduler._reverse_send_completion_ids[attempt_key])
    assert not (send_completion.closed and not send_completion.failed)


def test_partial_tp_completion_never_sender_complete(decode_scheduler_factory, decode_control_seams):
    scheduler = decode_scheduler_factory(world_size=2)
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(0, remote_tp_size=2))
    completion_id = metadata.reverse_plans[0].reverse_send_completion_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
    scheduler.update_connector_output(output)

    completion = scheduler._completion_tracker.get(completion_id)
    assert completion.completed_worker_count == 1
    assert completion.closed is False
    send_completion = scheduler._completion_tracker.get(scheduler._reverse_send_completion_ids[attempt_key])
    assert not (send_completion.closed and not send_completion.failed)


def test_terminal_ack_failure_marks_completion_failed_never_success(decode_scheduler_factory, decode_control_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision())
    completion_id = metadata.reverse_plans[0].reverse_send_completion_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    worker = _make_worker()
    _seed_reverse_send_tracker(worker, completion_id)
    worker.kv_send_layer_thread = make_sending_layer_thread()
    with patch.object(layerwise_module, "zmq_ctx", side_effect=RuntimeError("no route to host")):
        worker.send_done_send_signal(_REQUEST_ID, make_sender_req_meta(), 0, trans_flag=True)

    # Core finish lands before the failed Reverse terminal is drained. The
    # attempt report still closes, but request-level done_recving must not race
    # the scheduler's finished_sending release.
    assert worker.get_finished({_REQUEST_ID}, DualPathConnectorMetadata()) == (set(), set())
    assert _REQUEST_ID not in worker._split_trackers
    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.failure_reports == {completion_id: 1}
    assert worker_metadata.completion_reports == {}

    output = KVConnectorOutput(kv_connector_worker_meta=worker_metadata)
    scheduler.update_connector_output(output)
    completion = scheduler._completion_tracker.get(completion_id)
    assert completion is None
    assert attempt_key not in scheduler._reverse_send_completion_ids


def test_reverse_success_after_core_finish_reports_completion_without_done_recving():
    worker = _make_worker()
    completion_id = 17
    _seed_reverse_send_tracker(worker, completion_id)
    with patch.object(
        layerwise_module.MooncakeLayerwiseConnectorWorker,
        "send_done_send_signal",
        return_value=True,
    ):
        worker.send_done_send_signal(_REQUEST_ID, make_sender_req_meta(), 0, trans_flag=True)

    assert worker.get_finished({_REQUEST_ID}, DualPathConnectorMetadata()) == (set(), set())
    assert _REQUEST_ID not in worker._split_trackers
    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.completion_reports == {completion_id: 1}
    assert worker_metadata.failure_reports == {}


def test_abort_before_final_layer_leaves_completion_incomplete(decode_scheduler_factory, decode_control_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision())
    completion_id = metadata.reverse_plans[0].reverse_send_completion_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    # The abort lands before the final layer: send_done_send_signal never
    # fires, so no worker report exists for the reverse-send completion.
    worker = _make_worker()
    _seed_reverse_send_tracker(worker, completion_id)
    worker.kv_send_layer_thread = make_sending_layer_thread()
    assert worker.build_connector_worker_meta() is None

    completion = scheduler._completion_tracker.get(completion_id)
    assert completion.closed is False
    send_completion = scheduler._completion_tracker.get(scheduler._reverse_send_completion_ids[attempt_key])
    assert not (send_completion.closed and not send_completion.failed)


def test_close_marks_the_send_completion_closed_and_not_failed(decode_scheduler_factory, decode_control_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision())
    completion_id = metadata.reverse_plans[0].reverse_send_completion_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    worker = _make_worker()
    _seed_reverse_send_tracker(worker, completion_id)
    worker.kv_send_layer_thread = make_sending_layer_thread()
    with patch.object(layerwise_module, "zmq_ctx", successful_terminal_ack_zmq_ctx):
        worker.send_done_send_signal(_REQUEST_ID, make_sender_req_meta(), 0, trans_flag=True)

    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.completion_reports == {completion_id: 1}
    assert worker_metadata.failure_reports == {}

    output = KVConnectorOutput(kv_connector_worker_meta=worker_metadata)
    scheduler.update_connector_output(output)
    assert attempt_key not in scheduler._reverse_send_completion_ids
    assert scheduler._completion_tracker.get(completion_id) is None


@pytest.mark.parametrize(
    ("trans_flag", "expected_message_type"),
    [
        (True, layerwise_module.DONE_SENDING_MSG),
        (False, layerwise_module.FAILED_SENDING_MSG),
    ],
)
def test_reverse_terminal_wire_payload_closes_prefill_receive_attempt(
    trans_flag: bool,
    expected_message_type: bytes,
):
    reverse_receive_completion_id = 91
    expected_wire_id = "ra:decode-engine:2:boot-7:decode-request-7:0:0"
    worker = _make_worker()
    _seed_reverse_send_tracker(worker, reverse_send_completion_id=17)
    sockets = []

    @contextlib.contextmanager
    def capture_terminal_socket(_socket_type, _addr):
        sock = MagicMock(name="terminal_ack_socket")
        sock.poll.return_value = True
        sock.recv.return_value = b"ACK"
        sockets.append(sock)
        yield sock

    with patch.object(layerwise_module, "zmq_ctx", capture_terminal_socket):
        worker.send_done_send_signal(_REQUEST_ID, make_sender_req_meta(), 0, trans_flag=trans_flag)

    encoded_payload = sockets[0].send.call_args.args[0]
    terminal = layerwise_module.msgspec.msgpack.Decoder(type=tuple).decode(encoded_payload)
    assert terminal[0] == expected_message_type
    assert terminal[1] == expected_wire_id

    prefill_worker = _make_prefill_worker()
    binding = ReverseReceiveBinding(
        request_key=_REQUEST_KEY,
        wire_request_id=expected_wire_id,
        prefill_request_id="prefill-request-7",
        destination_block_ids=((71, 72, 73, 74),),
        token_start=16,
        token_end=32,
        reverse_attempt_id=0,
        prefill_local_tokens=16,
        reverse_receive_completion_id=reverse_receive_completion_id,
    )
    prefill_worker._install_reverse_receive_binding(binding)
    prefill_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
        {terminal[1]} if trans_flag else set()
    )
    prefill_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
        set() if trans_flag else {terminal[1]}
    )

    assert prefill_worker.get_finished(set(), DualPathConnectorMetadata()) == (set(), set())
    worker_metadata = prefill_worker.build_connector_worker_meta()
    expected_report = {reverse_receive_completion_id: 1}
    assert worker_metadata.completion_reports == (expected_report if trans_flag else {})
    assert worker_metadata.failure_reports == ({} if trans_flag else expected_report)
    assert prefill_worker._pending_forward_done_wire_ids == set()
    assert prefill_worker._pending_forward_failed_wire_ids == set()


def test_reverse_terminal_is_not_visible_while_terminal_ack_is_blocked():
    worker = _make_worker()
    tracker = _seed_reverse_send_tracker(worker, reverse_send_completion_id=17)
    ack_entered = threading.Event()
    release_ack = threading.Event()

    def blocking_ack(*_args, **_kwargs):
        ack_entered.set()
        assert release_ack.wait(timeout=5)
        return True

    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal", blocking_ack):
        callback = threading.Thread(
            target=worker.send_done_send_signal,
            args=(_REQUEST_ID, make_sender_req_meta(), 0, True),
        )
        callback.start()
        assert ack_entered.wait(timeout=5)

        observed_while_blocked = worker.get_finished(set(), DualPathConnectorMetadata())
        reverse_phase_while_blocked = tracker.reverse_phase
        pending_terminals_while_blocked = dict(worker._pending_local_reverse_terminals)
        release_ack.set()
        callback.join(timeout=5)
        assert not callback.is_alive()

    assert observed_while_blocked == (set(), set())
    assert reverse_phase_while_blocked is worker_module._SplitPhase.PENDING
    assert pending_terminals_while_blocked == {}
    assert worker.get_finished(set(), DualPathConnectorMetadata()) == (set(), set())
    assert tracker.reverse_phase is worker_module._SplitPhase.DONE
    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.completion_reports == {17: 1}
    assert worker_metadata.failure_reports == {}


def test_old_reverse_send_attempt_closes_without_releasing_open_latest_attempt(
    decode_scheduler_factory, decode_control_seams
):
    scheduler = decode_scheduler_factory()
    request, old_completion_id, latest_completion_id = _activate_two_reverse_attempts(scheduler, decode_control_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)
    assert scheduler._delay_free_for_connector(request) is True

    old_output = _close_send_completion(scheduler, old_completion_id)

    assert old_output.finished_sending is None
    assert old_attempt not in scheduler._reverse_send_completion_ids
    assert scheduler._completion_tracker.get(old_completion_id) is None
    assert scheduler._reverse_send_completion_ids[latest_attempt] == latest_completion_id
    assert request.request_id in scheduler._pending_finished_sending
    assert scheduler._latest_reverse_attempt_ids[request.request_id] == 1

    latest_output = _close_send_completion(scheduler, latest_completion_id)
    assert latest_output.finished_sending == {request.request_id}


def test_latest_reverse_send_attempt_closes_without_releasing_open_old_attempt(
    decode_scheduler_factory, decode_control_seams
):
    scheduler = decode_scheduler_factory()
    request, old_completion_id, latest_completion_id = _activate_two_reverse_attempts(scheduler, decode_control_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)
    assert scheduler._delay_free_for_connector(request) is True

    latest_output = _close_send_completion(scheduler, latest_completion_id)

    assert latest_output.finished_sending is None
    assert latest_attempt not in scheduler._reverse_send_completion_ids
    assert scheduler._completion_tracker.get(latest_completion_id) is None
    assert scheduler._reverse_send_completion_ids[old_attempt] == old_completion_id
    assert request.request_id in scheduler._pending_finished_sending
    assert scheduler._latest_reverse_attempt_ids[request.request_id] == 1

    old_output = _close_send_completion(scheduler, old_completion_id)
    assert old_output.finished_sending == {request.request_id}


@pytest.mark.parametrize("first_failure", ["older", "latest"])
def test_request_finished_retains_delayed_free_until_last_exact_send_attempt_fails(
    decode_scheduler_factory,
    decode_control_seams,
    first_failure,
):
    scheduler = decode_scheduler_factory()
    request, old_completion_id, latest_completion_id = _activate_two_reverse_attempts(scheduler, decode_control_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)
    request.status = RequestStatus.FINISHED_STOPPED

    assert scheduler.request_finished(request, []) == (True, None)
    assert request.request_id not in scheduler._decode_decision_states
    first_completion_id, first_attempt, final_completion_id, final_attempt = (
        (old_completion_id, old_attempt, latest_completion_id, latest_attempt)
        if first_failure == "older"
        else (latest_completion_id, latest_attempt, old_completion_id, old_attempt)
    )

    first_output = _fail_send_completion(scheduler, first_completion_id)

    assert first_output.finished_sending is None
    assert first_attempt not in scheduler._reverse_send_completion_ids
    assert scheduler._completion_tracker.get(first_completion_id) is None
    assert scheduler._reverse_send_completion_ids[final_attempt] == final_completion_id
    assert scheduler._completion_tracker.get(final_completion_id) is not None
    assert request.request_id in scheduler._pending_finished_sending

    final_output = _fail_send_completion(scheduler, final_completion_id)

    assert final_output.finished_sending == {request.request_id}
    assert final_attempt not in scheduler._reverse_send_completion_ids
    assert scheduler._completion_tracker.get(final_completion_id) is None
    assert request.request_id not in scheduler._pending_finished_sending


def test_failed_reverse_send_releases_delayed_free_only_after_every_attempt_closes(
    decode_scheduler_factory, decode_control_seams
):
    scheduler = decode_scheduler_factory()
    request, old_completion_id, latest_completion_id = _activate_two_reverse_attempts(scheduler, decode_control_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)
    assert scheduler._delay_free_for_connector(request) is True

    old_output = _fail_send_completion(scheduler, old_completion_id)

    assert old_output.finished_sending is None
    assert old_attempt not in scheduler._reverse_send_completion_ids
    assert scheduler._reverse_send_completion_ids[latest_attempt] == latest_completion_id
    assert request.request_id in scheduler._pending_finished_sending

    latest_output = _fail_send_completion(scheduler, latest_completion_id)
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert latest_output.finished_sending == {request.request_id}
    assert request.request_id not in scheduler._pending_finished_sending
    assert latest_attempt not in scheduler._reverse_send_completion_ids
    assert metadata.control_failures[0].reason is DualPathControlFailureReason.REVERSE_JOB_FAILED
    assert scheduler._decode_control_failures == {}


def test_finish_delays_when_latest_send_completion_closed_but_older_attempt_is_open(
    decode_scheduler_factory, decode_control_seams
):
    scheduler = decode_scheduler_factory()
    request, old_completion_id, latest_completion_id = _activate_two_reverse_attempts(scheduler, decode_control_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)

    latest_output = _close_send_completion(scheduler, latest_completion_id)
    assert latest_output.finished_sending is None
    assert latest_attempt not in scheduler._reverse_send_completion_ids
    assert scheduler._reverse_send_completion_ids[old_attempt] == old_completion_id

    assert scheduler._delay_free_for_connector(request) is True
    assert request.request_id in scheduler._pending_finished_sending

    old_output = _close_send_completion(scheduler, old_completion_id)
    assert old_output.finished_sending == {request.request_id}


def test_active_request_retains_closed_attempt_epoch_for_later_refresh(decode_scheduler_factory, decode_control_seams):
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    first_metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(0))
    first_completion_id = first_metadata.reverse_plans[0].reverse_send_completion_id
    assert first_completion_id is not None
    assert request.request_id not in scheduler._pending_finished_sending

    first_output = _close_send_completion(scheduler, first_completion_id)

    assert first_output.finished_sending is None
    assert scheduler._latest_reverse_attempt_ids[request.request_id] == 0

    second_metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(1))
    assert len(second_metadata.reverse_plans) == 1
    second_completion_id = second_metadata.reverse_plans[0].reverse_send_completion_id
    assert second_completion_id is not None
    assert second_completion_id != first_completion_id
    assert scheduler._latest_reverse_attempt_ids[request.request_id] == 1

# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W3: DE Reverse sender completion proof via the job ledger."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.outputs import KVConnectorOutput

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
    _make_reverse_plan,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision as path_decision_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ReversePlan,
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
_REQUEST_KEY = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, _REQUEST_ID)


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
            reverse_send_job_id=None,
        ),
    )


def _activate_decision(scheduler, decode_task04_seams, decision: PathDecision):
    decode_task04_seams.decode_coordinator.take_received_decisions.return_value = [decision]
    return scheduler.build_connector_meta(MagicMock(name="scheduler_output"))


def _seed_reverse_send_tracker(worker, reverse_send_job_id: int, reverse_attempt_id: int = 0):
    attempt_key = _attempt_key(reverse_attempt_id, _REQUEST_KEY)
    plan = _make_reverse_plan(reverse_attempt_id=reverse_attempt_id, reverse_send_job_id=reverse_send_job_id)
    tracker = worker_module._SplitTracker(
        store_phase=worker_module._SplitPhase.SKIPPED,
        reverse_phase=worker_module._SplitPhase.PENDING,
        forward_phase=worker_module._SplitPhase.PENDING,
        store_destination_slice=(),
        forward_destination_slice=(70, 71),
        reverse_plan=plan,
        reverse_submitted_attempt=attempt_key,
        store_load_failed=False,
        terminal_published=False,
    )
    worker._split_trackers[_REQUEST_ID] = tracker
    return tracker


def _activate_two_reverse_attempts(scheduler, decode_task04_seams):
    request = _admit_decode_request(scheduler)
    first_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
    second_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(1))
    first_job_id = first_metadata.reverse_plans[0].reverse_send_job_id
    second_job_id = second_metadata.reverse_plans[0].reverse_send_job_id
    assert first_job_id is not None
    assert second_job_id is not None
    return request, first_job_id, second_job_id


def _close_send_job(scheduler, job_id: int) -> KVConnectorOutput:
    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={job_id: 1}))
    scheduler.update_connector_output(output)
    return output


def test_reverse_send_job_allocated_at_attempt_creation_and_carried_on_plan(
    decode_scheduler_factory, decode_task04_seams
):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)

    metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision())

    assert len(metadata.reverse_plans) == 1
    carried_plan = metadata.reverse_plans[0]
    assert carried_plan.reverse_send_job_id is not None
    attempt_key = _attempt_key(0, _REQUEST_KEY)
    job = scheduler._job_ledger.get(carried_plan.reverse_send_job_id)
    assert job.job_kind.name == "REVERSE_SEND"
    assert job.reverse_attempt_key == attempt_key
    assert job.expected_worker_count == 1
    assert scheduler._reverse_send_job_ids[attempt_key] == job.job_id
    send_job = scheduler._job_ledger.get(scheduler._reverse_send_job_ids[attempt_key])
    assert not (send_job.closed and not send_job.failed)


def test_partial_tp_completion_never_sender_complete(decode_scheduler_factory, decode_task04_seams):
    scheduler = decode_scheduler_factory(world_size=2)
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0, remote_tp_size=2))
    job_id = metadata.reverse_plans[0].reverse_send_job_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={job_id: 1}))
    scheduler.update_connector_output(output)

    job = scheduler._job_ledger.get(job_id)
    assert job.completed_worker_count == 1
    assert job.closed is False
    send_job = scheduler._job_ledger.get(scheduler._reverse_send_job_ids[attempt_key])
    assert not (send_job.closed and not send_job.failed)


def test_terminal_ack_failure_marks_job_failed_never_success(decode_scheduler_factory, decode_task04_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision())
    job_id = metadata.reverse_plans[0].reverse_send_job_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    worker = _make_worker()
    _seed_reverse_send_tracker(worker, job_id)
    worker.kv_send_layer_thread = make_sending_layer_thread()
    with patch.object(layerwise_module, "zmq_ctx", side_effect=RuntimeError("no route to host")):
        worker.send_done_send_signal(_REQUEST_ID, make_sender_req_meta(), 0, trans_flag=True)

    # The failed ACK closes the split through the failure path; it never turns
    # the Reverse phase into DONE.
    assert worker.get_finished(set(), DualPathConnectorMetadata()) == (set(), {_REQUEST_ID})
    assert worker._split_trackers[_REQUEST_ID].reverse_phase is worker_module._SplitPhase.FAILED
    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.failed_jobs == {job_id: 1}
    assert worker_metadata.completed_jobs == {}

    output = KVConnectorOutput(kv_connector_worker_meta=worker_metadata)
    scheduler.update_connector_output(output)
    job = scheduler._job_ledger.get(job_id)
    assert job.closed is True
    assert job.failed is True
    send_job = scheduler._job_ledger.get(scheduler._reverse_send_job_ids[attempt_key])
    assert not (send_job.closed and not send_job.failed)


def test_abort_before_final_layer_leaves_job_incomplete(decode_scheduler_factory, decode_task04_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision())
    job_id = metadata.reverse_plans[0].reverse_send_job_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    # The abort lands before the final layer: send_done_send_signal never
    # fires, so no worker report exists for the reverse-send job.
    worker = _make_worker()
    _seed_reverse_send_tracker(worker, job_id)
    worker.kv_send_layer_thread = make_sending_layer_thread()
    assert worker.build_connector_worker_meta() is None

    job = scheduler._job_ledger.get(job_id)
    assert job.closed is False
    send_job = scheduler._job_ledger.get(scheduler._reverse_send_job_ids[attempt_key])
    assert not (send_job.closed and not send_job.failed)


def test_job_close_marks_the_send_job_closed_and_not_failed(decode_scheduler_factory, decode_task04_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision())
    job_id = metadata.reverse_plans[0].reverse_send_job_id
    attempt_key = _attempt_key(0, _REQUEST_KEY)

    worker = _make_worker()
    _seed_reverse_send_tracker(worker, job_id)
    worker.kv_send_layer_thread = make_sending_layer_thread()
    with patch.object(layerwise_module, "zmq_ctx", successful_terminal_ack_zmq_ctx):
        worker.send_done_send_signal(_REQUEST_ID, make_sender_req_meta(), 0, trans_flag=True)

    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.completed_jobs == {job_id: 1}
    assert worker_metadata.failed_jobs == {}

    output = KVConnectorOutput(kv_connector_worker_meta=worker_metadata)
    scheduler.update_connector_output(output)
    assert attempt_key not in scheduler._reverse_send_job_ids
    assert scheduler._job_ledger.get(job_id) is None


def test_reverse_terminal_is_not_visible_while_terminal_ack_is_blocked():
    worker = _make_worker()
    tracker = _seed_reverse_send_tracker(worker, reverse_send_job_id=17)
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
    assert worker_metadata.completed_jobs == {17: 1}
    assert worker_metadata.failed_jobs == {}


def test_old_reverse_send_attempt_closes_without_releasing_open_latest_attempt(
    decode_scheduler_factory, decode_task04_seams
):
    scheduler = decode_scheduler_factory()
    request, old_job_id, latest_job_id = _activate_two_reverse_attempts(scheduler, decode_task04_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)
    assert scheduler._delay_free_for_connector(request) is True

    old_output = _close_send_job(scheduler, old_job_id)

    assert old_output.finished_sending is None
    assert old_attempt not in scheduler._reverse_send_job_ids
    assert scheduler._job_ledger.get(old_job_id) is None
    assert scheduler._reverse_send_job_ids[latest_attempt] == latest_job_id
    assert request.request_id in scheduler._pending_finished_sending
    assert scheduler._latest_reverse_attempt_ids[request.request_id] == 1

    latest_output = _close_send_job(scheduler, latest_job_id)
    assert latest_output.finished_sending == {request.request_id}


def test_latest_reverse_send_attempt_closes_without_releasing_open_old_attempt(
    decode_scheduler_factory, decode_task04_seams
):
    scheduler = decode_scheduler_factory()
    request, old_job_id, latest_job_id = _activate_two_reverse_attempts(scheduler, decode_task04_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)
    assert scheduler._delay_free_for_connector(request) is True

    latest_output = _close_send_job(scheduler, latest_job_id)

    assert latest_output.finished_sending is None
    assert latest_attempt not in scheduler._reverse_send_job_ids
    assert scheduler._job_ledger.get(latest_job_id) is None
    assert scheduler._reverse_send_job_ids[old_attempt] == old_job_id
    assert request.request_id in scheduler._pending_finished_sending
    assert scheduler._latest_reverse_attempt_ids[request.request_id] == 1

    old_output = _close_send_job(scheduler, old_job_id)
    assert old_output.finished_sending == {request.request_id}


def test_finish_delays_when_latest_send_job_closed_but_older_attempt_is_open(
    decode_scheduler_factory, decode_task04_seams
):
    scheduler = decode_scheduler_factory()
    request, old_job_id, latest_job_id = _activate_two_reverse_attempts(scheduler, decode_task04_seams)
    state = scheduler._decode_decision_states[request.request_id]
    old_attempt = _attempt_key(0, state.request_key)
    latest_attempt = _attempt_key(1, state.request_key)

    latest_output = _close_send_job(scheduler, latest_job_id)
    assert latest_output.finished_sending is None
    assert latest_attempt not in scheduler._reverse_send_job_ids
    assert scheduler._reverse_send_job_ids[old_attempt] == old_job_id

    assert scheduler._delay_free_for_connector(request) is True
    assert request.request_id in scheduler._pending_finished_sending

    old_output = _close_send_job(scheduler, old_job_id)
    assert old_output.finished_sending == {request.request_id}

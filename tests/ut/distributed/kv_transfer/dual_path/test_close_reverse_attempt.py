# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W7: CloseReverseAttempt close matrix, durable proofs, dual release."""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock

from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    DECODE_TEST_INSTANCE_ID,
    make_block_pool,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import (
    _deliver_frames,
    _make_receiver,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_reverse_send_proof import (
    _admit_decode_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel as channel
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathKind,
    ReverseAttemptKey,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    CloseReverseAttempt,
)

_KEY = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, "decode-request-9")


def _close(receiver, attempt_id: int, key: DualPathRequestKey = _KEY) -> channel.CloseReplyStatus:
    close = CloseReverseAttempt(request_key=key, reverse_attempt_id=attempt_id)
    reply = _deliver_frames(
        receiver,
        channel.encode_control_message(channel.ControlMessageKind.CLOSE_REVERSE_ATTEMPT, close.to_dict()),
    )
    assert reply is not None
    return channel.decode_close_reply(reply)


def _receive_decision(receiver, attempt_id: int = 0) -> None:
    from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _decision

    _deliver_frames(receiver, channel.encode_path_decision(_decision(attempt_id)))


def _record(receiver, attempt_id: int, key: DualPathRequestKey = _KEY):
    return receiver._closed_reverse_records.get(ReverseAttemptKey(key, attempt_id))


class TestCloseMatrix:
    def test_retained_safe_close_proof_checked_first_returns_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

        receiver.take_received_decisions()
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_admission_absent_returns_not_safe_no_record_created(self):
        receiver = _make_receiver()
        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE
        assert receiver._closed_reverse_records == {}

    def test_attempt_never_received_closes_advances_watermark_returns_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)

        assert _close(receiver, 2) is channel.CloseReplyStatus.SAFE
        assert receiver._closed_through_attempt_ids[_KEY] == 2
        record = _record(receiver, 2)
        assert record is not None and record.safe_close_proof is True

        # A delayed Decision at the closed number is rejected as stale.
        from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _decision

        reply = _deliver_frames(receiver, channel.encode_path_decision(_decision(2)))
        assert channel.decode_decision_reply(reply) is channel.DecisionReplyStatus.STALE_CLOSED

    def test_decision_received_activation_unclaimed_returns_safe_activation_suppressed(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)

        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        # The persisted proof suppresses a later activation claim.
        assert receiver.claim_reverse_activation(_KEY, 0) is False
        assert receiver.take_received_decisions() != []

    def test_activation_claimed_publication_cancelled_returns_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert receiver.claim_reverse_activation(_KEY, 0) is True
        receiver.cancel_reverse_publication(ReverseAttemptKey(_KEY, 0))

        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        assert _record(receiver, 0).safe_close_proof is True

    def test_activation_claimed_or_publication_uncertain_incomplete_work_returns_not_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert receiver.claim_reverse_activation(_KEY, 0) is True
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)

        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE
        record = _record(receiver, 0)
        assert record is not None
        assert record.safe_close_proof is False
        assert record.reverse_send_job_id == 7

    def test_published_work_all_workers_complete_returns_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)
        receiver.mark_reverse_send_complete(ReverseAttemptKey(_KEY, 0))

        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        assert _record(receiver, 0).safe_close_proof is True

    def test_close_covered_only_by_watermark_without_proof_returns_not_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        # Accepting attempt 1 advances the watermark past 0 without any proof.
        _receive_decision(receiver, 1)

        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE
        record = _record(receiver, 0)
        assert record is not None and record.safe_close_proof is False

    def test_not_safe_record_converts_to_safe_when_job_closes_not_failed(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)
        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE

        receiver.mark_reverse_send_complete(ReverseAttemptKey(_KEY, 0))

        assert _record(receiver, 0).safe_close_proof is True
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_record_survives_worker_request_cleanup_and_keeps_no_counter(self):
        from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import _make_worker

        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)
        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE

        worker = _make_worker()
        worker._release_split_request_state({_KEY.decode_request_id})

        record = _record(receiver, 0)
        assert record is not None
        assert set(vars(record)) == {
            "attempt_key",
            "activation_claimed",
            "worker_work_published",
            "reverse_send_job_id",
            "safe_close_proof",
        }
        receiver.mark_reverse_send_complete(ReverseAttemptKey(_KEY, 0))
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_retained_safe_proof_survives_until_pe_ack_or_teardown(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

        receiver.unregister(_KEY)
        # Admission teardown retains the independent minimal proof.
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_admission_teardown_retains_independent_minimal_proof(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)
        receiver.mark_reverse_send_complete(ReverseAttemptKey(_KEY, 0))
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

        receiver.unregister(_KEY)
        assert receiver._accepted_decisions == {}
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        # A different attempt of the torn-down admission stays fail-closed.
        assert _close(receiver, 1) is channel.CloseReplyStatus.NOT_SAFE


class TestLostResponseDurability:
    def test_dropped_safe_retry_never_received_still_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        assert _close(receiver, 3) is channel.CloseReplyStatus.SAFE
        # The first SAFE is dropped on the wire; the identical retry reads the
        # persisted proof.
        assert _close(receiver, 3) is channel.CloseReplyStatus.SAFE

    def test_dropped_safe_retry_unclaimed_still_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_dropped_safe_retry_publication_cancelled_still_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.cancel_reverse_publication(ReverseAttemptKey(_KEY, 0))
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_dropped_safe_retry_sender_complete_still_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)
        receiver.mark_reverse_send_complete(ReverseAttemptKey(_KEY, 0))
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_not_safe_to_safe_conversion_retry_returns_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=7)
        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE
        receiver.mark_reverse_send_complete(ReverseAttemptKey(_KEY, 0))
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_dropped_safe_retry_does_not_reevaluate_worker_state(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

        # Worker-state transitions after the persisted proof must not flip the
        # answer: the retained proof is authoritative.
        assert receiver.claim_reverse_activation(_KEY, 0) is False
        receiver.mark_reverse_work_published(ReverseAttemptKey(_KEY, 0), reverse_send_job_id=9)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_admission_cleanup_racing_dropped_safe_still_returns_safe(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        receiver.unregister(_KEY)
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE


class TestRouterContract:
    def test_router_handler_never_blocks_on_worker_queue_or_future(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)

        class PoisonedQueue:
            def __getattr__(self, name):
                raise AssertionError(f"close handler touched the receive queue via {name}")

        receiver._received_decisions = PoisonedQueue()
        poisoned_future = MagicMock(name="future")
        poisoned_future.result.side_effect = AssertionError("close handler waited on a Future")
        poisoned_future.exception.side_effect = AssertionError("close handler waited on a Future")

        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE
        poisoned_future.result.assert_not_called()


class TestAbortDualRelease:
    def _admit_waiting_de_read(self, scheduler):
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)
        return request

    def test_abort_waiting_de_read_injects_finished_recving_and_frees_ordinary_ownership(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = self._admit_waiting_de_read(scheduler)
        close_future: Future = Future()
        coordinator.submit_close.return_value = close_future

        request.status = RequestStatus.FINISHED_ABORTED
        delay_free, params = scheduler.request_finished(request, [])
        assert (delay_free, params) == (False, None)
        assert request.request_id in scheduler._pending_ordinary_release
        coordinator.submit_close.assert_called_once()
        close_message = coordinator.submit_close.call_args.args[1]
        assert isinstance(close_message, CloseReverseAttempt)
        assert close_message.reverse_attempt_id == 0

        output = KVConnectorOutput()
        scheduler.update_connector_output(output)
        assert output.finished_recving == {request.request_id}
        assert request.request_id not in scheduler._pending_ordinary_release
        # The destination hold is retained until the close proves SAFE.
        assert pool.blocks[71].ref_cnt == 1

    def test_reverse_destination_hold_retained_until_safe(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = self._admit_waiting_de_read(scheduler)
        close_future: Future = Future()
        coordinator.submit_close.return_value = close_future
        request.status = RequestStatus.FINISHED_ABORTED
        scheduler.request_finished(request, [])

        close_future.set_result(channel.CloseReplyStatus.SAFE)
        scheduler.update_connector_output(KVConnectorOutput())
        assert pool.blocks[71].ref_cnt == 0

    def test_not_safe_forever_hold_survives_request_failure(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = self._admit_waiting_de_read(scheduler)
        close_future: Future = Future()
        coordinator.submit_close.return_value = close_future
        request.status = RequestStatus.FINISHED_ABORTED
        scheduler.request_finished(request, [])

        close_future.set_result(channel.CloseReplyStatus.NOT_SAFE)
        scheduler.update_connector_output(KVConnectorOutput())
        assert pool.blocks[71].ref_cnt == 1
        scheduler.update_connector_output(KVConnectorOutput())
        assert pool.blocks[71].ref_cnt == 1

    def test_only_request_abort_no_forward_step_still_injects(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = self._admit_waiting_de_read(scheduler)
        close_future: Future = Future()
        coordinator.submit_close.return_value = close_future
        request.status = RequestStatus.FINISHED_ABORTED
        scheduler.request_finished(request, [])

        # The aborted waiter is the engine's only request: a no-forward output
        # step (no worker metadata) still carries the injected id.
        output = KVConnectorOutput()
        scheduler.update_connector_output(output)
        assert output.finished_recving == {request.request_id}


def test_de_sole_request_zero_token_steps_harvest_reports_and_close_retry_returns_safe(
    decode_scheduler_factory, decode_task04_seams
):
    from tests.ut.distributed.kv_transfer.dual_path.conftest import DECODE_TEST_CONTROL_ENDPOINT

    scheduler = decode_scheduler_factory()
    receiver = _make_receiver()
    receiver._decode_control_endpoint = DECODE_TEST_CONTROL_ENDPOINT
    receiver._context = MagicMock(name="zmq_context")
    receiver._receiver_thread = MagicMock(name="receiver_thread")
    scheduler._path_decision_coordinator = receiver
    request_key = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, "decode-request-7")

    _admit_decode_request(scheduler)
    # The Decision arrives over the real registry path: accepted, enqueued,
    # then claimed and published by the scheduler's activation drain.
    _deliver_frames(receiver, channel.encode_path_decision(_de_read_decision(0)))
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
    job_id = metadata.reverse_plans[0].reverse_send_job_id

    # The initial close finds published-but-incomplete work: NOT_SAFE.
    assert _close(receiver, 0, request_key) is channel.CloseReplyStatus.NOT_SAFE
    assert _record(receiver, 0, request_key).safe_close_proof is False

    # Zero-token steps harvest the all-worker reverse-send report; the locked
    # transition converts the record to the retained SAFE proof.
    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={job_id: 1}))
    scheduler.update_connector_output(output)
    assert _record(receiver, 0, request_key).safe_close_proof is True

    assert _close(receiver, 0, request_key) is channel.CloseReplyStatus.SAFE

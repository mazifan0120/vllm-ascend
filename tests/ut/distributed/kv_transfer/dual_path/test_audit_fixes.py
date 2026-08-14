# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W9: implementation-audit regression tests (B1, M2, M3, M4, m6, m7)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    DECODE_TEST_INSTANCE_ID,
    make_block_pool,
    make_empty_scheduler_output,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_reverse_send_proof import (
    _activate_decision,
    _admit_decode_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_i4_gate import (
    _admit_de_read_request,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
)
from tests.ut.distributed.kv_transfer.dual_path.test_resume_admission import (
    _admit_de_read,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel as channel
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.close_registry import (
    ReverseAttemptRegistryState,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.ledgers import (
    HoldKind,
    HoldLedger,
    JobKind,
    JobLedger,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathKind,
    ReverseAttemptKey,
)

_KEY = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, "decode-request-9")


def _close(receiver, attempt_id: int, key: DualPathRequestKey = _KEY) -> channel.CloseReplyStatus:
    from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _deliver_frames

    close = channel.CloseReverseAttempt(request_key=key, reverse_attempt_id=attempt_id)
    reply = _deliver_frames(
        receiver,
        channel.encode_control_message(channel.ControlMessageKind.CLOSE_REVERSE_ATTEMPT, close.to_dict()),
    )
    assert reply is not None
    return channel.decode_close_reply(reply)


def _receive_decision(receiver, attempt_id: int = 0) -> None:
    from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _decision, _deliver_frames

    _deliver_frames(receiver, channel.encode_path_decision(_decision(attempt_id)))


class TestDecodeEngineProgress:
    def test_de_final_request_delayed_free_until_reverse_send_job_closes(
        self, decode_scheduler_factory, decode_task04_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
        job_id = metadata.reverse_plans[0].reverse_send_job_id

        # The final Decode request finishes with its reverse-send job open:
        # the connector delays the free so zero-token steps keep harvesting.
        request.status = RequestStatus.FINISHED_STOPPED
        delay_free, params = scheduler.request_finished(request, [])
        assert (delay_free, params) == (True, None)

        idle_output = KVConnectorOutput()
        scheduler.update_connector_output(idle_output)
        assert idle_output.finished_sending is None

        harvest_output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={job_id: 1}))
        scheduler.update_connector_output(harvest_output)
        assert harvest_output.finished_sending == {request.request_id}

        # With the job closed, the request no longer delays the free and the
        # attempt-owned job mappings are retired.
        delay_free_after, _ = scheduler.request_finished(request, [])
        assert delay_free_after is False
        attempt_key = ReverseAttemptKey(
            DualPathRequestKey(scheduler._path_decision_coordinator.decode_engine_instance_id, request.request_id),
            0,
        )
        assert attempt_key not in scheduler._reverse_send_job_ids
        assert request.request_id not in scheduler._latest_reverse_attempt_ids


class TestBoundedCompletionJobs:
    def test_failed_reverse_completion_job_surfaces_control_failure_without_release(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(failed_jobs={binding.reverse_completion_job_id: 1})
        )
        scheduler.update_connector_output(output)
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        assert len(metadata.control_failures) == 1
        failure = metadata.control_failures[0]
        assert failure.request_id == request.request_id
        assert failure.reason is DualPathControlFailureReason.RECOVERY_TIMEOUT
        assert pool.blocks[71].ref_cnt == 1

    def test_missing_reverse_completion_deadline_expiry_surfaces_control_failure(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        clock = MagicMock(name="monotonic_clock")
        clock.monotonic.return_value = 2000.0
        from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module

        with patch.object(scheduler_module, "time", clock):
            request = _admit_de_read_request(scheduler)
            clock.monotonic.return_value = 2000.0 + scheduler._recovery_watchdog_s + 1.0
            metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        assert len(metadata.control_failures) == 1
        assert metadata.control_failures[0].request_id == request.request_id
        assert pool.blocks[71].ref_cnt == 1


class TestSingleUnresolvedDeliveryFuture:
    def test_replacement_delivery_defers_until_prior_future_resolves(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        pending_future = MagicMock(name="delivery_future_attempt_0")
        pending_future.done.return_value = False
        pending_future.cancelled.return_value = False
        pending_future.exception.return_value = None
        resolved_future = MagicMock(name="delivery_future_attempt_1")
        resolved_future.done.return_value = True
        resolved_future.cancelled.return_value = False
        resolved_future.exception.return_value = None
        coordinator.submit.side_effect = [pending_future, resolved_future]

        request = _admit_de_read(scheduler)
        assert coordinator.submit.call_count == 1
        first_future = scheduler._prefill_delivery_futures[request.request_id]

        request.num_preemptions += 1
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 16)
        # Attempt 1 is installed, but its delivery is deferred: no second
        # submit and no overwrite of the unresolved attempt-0 Future.
        assert scheduler._prefill_reverse_plans[request.request_id].reverse_attempt_id == 1
        assert coordinator.submit.call_count == 1
        assert scheduler._prefill_delivery_futures[request.request_id] is first_future

        # Once attempt 0's delivery resolves, the next build pass delivers.
        pending_future.done.return_value = True
        scheduler.build_connector_meta(make_empty_scheduler_output())
        assert coordinator.submit.call_count == 2
        second_decision = coordinator.submit.call_args_list[1].args[1]
        assert second_decision.result.reverse_attempt_id == 1
        assert scheduler._prefill_delivery_futures[request.request_id] is not first_future


class TestFailedPriorFutureCancelsDeferredReplacement:
    def _defer_replacement(self, scheduler, coordinator):
        pending_future = MagicMock(name="delivery_future_attempt_0")
        pending_future.done.return_value = False
        pending_future.cancelled.return_value = False
        pending_future.exception.return_value = None
        coordinator.submit.side_effect = [pending_future]
        request = _admit_de_read(scheduler)
        assert coordinator.submit.call_count == 1
        request.num_preemptions += 1
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 16)
        assert request.request_id in scheduler._prefill_deferred_deliveries
        return request, pending_future

    def test_failed_prior_future_cancels_deferred_replacement(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request, pending_future = self._defer_replacement(scheduler, coordinator)

        pending_future.done.return_value = True
        pending_future.exception.return_value = RuntimeError("delivery exhausted")
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        assert coordinator.submit.call_count == 1
        assert request.request_id in scheduler._prefill_invalid_request_ids
        assert request.request_id not in scheduler._prefill_deferred_deliveries
        assert len(metadata.control_failures) == 1
        # The invalid request can never be delivered later, even by a direct call.
        scheduler._deliver_prefill_decision(request.request_id)
        assert coordinator.submit.call_count == 1

    def test_cancelled_prior_future_cancels_deferred_replacement(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request, pending_future = self._defer_replacement(scheduler, coordinator)

        pending_future.done.return_value = True
        pending_future.cancelled.return_value = True
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        assert coordinator.submit.call_count == 1
        assert request.request_id in scheduler._prefill_invalid_request_ids
        assert request.request_id not in scheduler._prefill_deferred_deliveries
        assert len(metadata.control_failures) == 1


class TestLedgerRetirementWiring:
    def test_reverse_completion_close_retires_job_and_hold_records(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
        hold_id = scheduler._reverse_destination_holds[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(completed_jobs={binding.reverse_completion_job_id: 1})
        )
        scheduler.update_connector_output(output)

        assert output.finished_recving == {request.request_id}
        assert scheduler._job_ledger.get(binding.reverse_completion_job_id) is None
        assert scheduler._hold_ledger.get(hold_id) is None

    def test_failed_job_and_held_records_are_never_retired_early(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
        hold_id = scheduler._reverse_destination_holds[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(failed_jobs={binding.reverse_completion_job_id: 1})
        )
        scheduler.update_connector_output(output)

        assert scheduler._job_ledger.get(binding.reverse_completion_job_id).failed is True
        assert scheduler._hold_ledger.get(hold_id).released is False
        assert pool.blocks[71].ref_cnt == 1

    def test_normal_reverse_send_completion_without_close_retires_state_with_proof(self):
        from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import (
            _make_receiver,
        )

        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        attempt_key = ReverseAttemptKey(_KEY, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(attempt_key, reverse_send_job_id=3)

        receiver.mark_reverse_send_complete(attempt_key)

        # No close ever arrived: the minimal completed-send proof is persisted
        # and the live state slot retires with the attempt.
        assert attempt_key not in receiver._reverse_attempt_states
        record = receiver._closed_reverse_records[attempt_key]
        assert record.safe_close_proof is True
        assert _close(receiver, 0) is channel.CloseReplyStatus.SAFE

    def test_state_with_pending_close_record_is_not_retired_early(self):
        from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import (
            _make_receiver,
        )

        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        _receive_decision(receiver, 0)
        attempt_key = ReverseAttemptKey(_KEY, 0)
        receiver.claim_reverse_activation(_KEY, 0)
        receiver.mark_reverse_work_published(attempt_key, reverse_send_job_id=3)
        assert _close(receiver, 0) is channel.CloseReplyStatus.NOT_SAFE

        # A NOT_SAFE record is pending the send job: the live state stays.
        assert attempt_key in receiver._reverse_attempt_states

        receiver.mark_reverse_send_complete(attempt_key)
        assert attempt_key not in receiver._reverse_attempt_states
        assert receiver._closed_reverse_records[attempt_key].safe_close_proof is True


class TestLifecycleRetirement:
    def test_ledgers_retire_only_closed_and_released_records(self):
        pool = make_block_pool(num_blocks=8)
        hold_ledger = HoldLedger()
        job_ledger = JobLedger()
        released = hold_ledger.acquire(pool, (1,), HoldKind.REVERSE_DESTINATION)
        retained = hold_ledger.acquire(pool, (2,), HoldKind.REVERSE_DESTINATION)
        hold_ledger.release(pool, released.hold_id)

        closed_job = job_ledger.create_job(JobKind.REVERSE_COMPLETION, expected_worker_count=1)
        open_job = job_ledger.create_job(JobKind.REVERSE_COMPLETION, expected_worker_count=1)
        job_ledger.record_reports(closed_job.job_id, 1)

        assert not job_ledger.discard(open_job.job_id)
        assert job_ledger.discard(closed_job.job_id)
        assert job_ledger.get(closed_job.job_id) is None
        assert not hold_ledger.discard(retained.hold_id)
        assert hold_ledger.discard(released.hold_id)
        assert hold_ledger.get(released.hold_id) is None

    def test_coordinator_attempt_states_retire_after_safe_proof(self):
        from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _make_receiver

        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        assert _close(receiver, 0) is not None
        attempt_key = ReverseAttemptKey(_KEY, 0)
        assert attempt_key in receiver._closed_reverse_records
        # The live state slot is retired once the proof is authoritative; the
        # record survives until teardown.
        receiver._reverse_attempt_states.setdefault(attempt_key, ReverseAttemptRegistryState())
        receiver.mark_reverse_work_published(attempt_key, reverse_send_job_id=3)
        receiver.mark_reverse_send_complete(attempt_key)
        assert attempt_key not in receiver._reverse_attempt_states
        assert receiver._closed_reverse_records[attempt_key].safe_close_proof is True


class TestStage2EnvValidation:
    @pytest.mark.parametrize(
        "name",
        [
            "VLLM_ASCEND_DUALPATH_RECOVERY_WATCHDOG_S",
            "VLLM_ASCEND_DUALPATH_DE_PROGRESS_WATCHDOG_S",
        ],
    )
    def test_non_positive_timeouts_rejected(self, monkeypatch, pe_scheduler_factory, name):
        monkeypatch.setenv(name, "0")
        with pytest.raises(ValueError, match=name):
            pe_scheduler_factory(PathKind.PE_READ)

    @pytest.mark.parametrize(
        "name",
        [
            "VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS",
            "VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS",
        ],
    )
    def test_negative_budgets_rejected(self, monkeypatch, pe_scheduler_factory, name):
        monkeypatch.setenv(name, "-1")
        with pytest.raises(ValueError, match=name):
            pe_scheduler_factory(PathKind.PE_READ)


def _deliver_decision_to(receiver, attempt_id: int) -> None:
    from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _deliver_frames
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel as channel

    _deliver_frames(receiver, channel.encode_path_decision(_de_read_decision(attempt_id)))


class TestReverseSendJobRetirement:
    def test_delayed_free_reverse_send_close_retires_job_record(self, decode_scheduler_factory, decode_task04_seams):
        from tests.ut.distributed.kv_transfer.dual_path.conftest import (
            DECODE_TEST_CONTROL_ENDPOINT,
            DECODE_TEST_INSTANCE_ID,
        )
        from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _make_receiver

        scheduler = decode_scheduler_factory()
        receiver = _make_receiver()
        receiver._decode_control_endpoint = DECODE_TEST_CONTROL_ENDPOINT
        receiver._context = MagicMock(name="zmq_context")
        receiver._receiver_thread = MagicMock(name="receiver_thread")
        scheduler._path_decision_coordinator = receiver
        request_key = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, "decode-request-7")

        request = _admit_decode_request(scheduler)
        _deliver_decision_to(receiver, 0)
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        job_id = metadata.reverse_plans[0].reverse_send_job_id

        # Delayed-free flow: release runs while the job is open and skips its
        # retirement; the job close must retire the record itself.
        request.status = RequestStatus.FINISHED_STOPPED
        delay_free, _ = scheduler.request_finished(request, [])
        assert delay_free is True
        assert scheduler._job_ledger.get(job_id) is not None

        output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={job_id: 1}))
        scheduler.update_connector_output(output)

        assert output.finished_sending == {request.request_id}
        assert scheduler._job_ledger.get(job_id) is None
        # The coordinator's persisted proof survives; a later close is SAFE.
        attempt_key = ReverseAttemptKey(request_key, 0)
        assert receiver._closed_reverse_records[attempt_key].safe_close_proof is True
        assert _close(receiver, 0, request_key) is channel.CloseReplyStatus.SAFE

    def test_normal_reverse_send_close_retires_at_release_and_predicate_reads_ledger(
        self, decode_scheduler_factory, decode_task04_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
        job_id = metadata.reverse_plans[0].reverse_send_job_id
        attempt_key = ReverseAttemptKey(
            DualPathRequestKey(
                scheduler._path_decision_coordinator.decode_engine_instance_id,
                request.request_id,
            ),
            0,
        )

        output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={job_id: 1}))
        scheduler.update_connector_output(output)

        # Between close and release the predicate still reads the ledger.
        assert scheduler._is_reverse_send_complete(attempt_key) is True
        assert scheduler._job_ledger.get(job_id) is not None

        request.status = RequestStatus.FINISHED_STOPPED
        delay_free, _ = scheduler.request_finished(request, [])
        assert delay_free is False
        assert scheduler._job_ledger.get(job_id) is None

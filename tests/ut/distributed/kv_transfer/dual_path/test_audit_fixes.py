# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W9: implementation-audit regression tests (B1, M2, M3, M4, m6, m7)."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
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
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathKind,
    ReverseAttemptKey,
)


class TestDecodeEngineProgress:
    def test_de_final_request_delayed_free_until_reverse_send_job_closes(
        self, decode_scheduler_factory, decode_task04_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
        completion_id = metadata.reverse_plans[0].reverse_send_completion_id

        # The final Decode request finishes with its reverse-send completion open:
        # the connector delays the free so zero-token steps keep harvesting.
        request.status = RequestStatus.FINISHED_STOPPED
        delay_free, params = scheduler.request_finished(request, [])
        assert (delay_free, params) == (True, None)

        idle_output = KVConnectorOutput()
        scheduler.update_connector_output(idle_output)
        assert idle_output.finished_sending is None

        harvest_output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1})
        )
        scheduler.update_connector_output(harvest_output)
        assert harvest_output.finished_sending == {request.request_id}

        # With the completion closed, the request no longer delays the free and the
        # attempt-owned completion mappings are retired.
        delay_free_after, _ = scheduler.request_finished(request, [])
        assert delay_free_after is False
        attempt_key = ReverseAttemptKey(
            DualPathRequestKey(
                scheduler._path_decision_coordinator.decode_engine_instance_id,
                request.request_id,
                0,
            ),
            0,
        )
        assert attempt_key not in scheduler._reverse_send_completion_ids
        assert request.request_id not in scheduler._latest_reverse_attempt_ids


class TestBoundedCompletionJobs:
    def test_failed_reverse_completion_job_surfaces_control_failure(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(failure_reports={binding.reverse_receive_completion_id: 1})
        )
        scheduler.update_connector_output(output)
        assert binding.request_key in scheduler._prefill_invalid_request_keys
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        assert len(metadata.control_failures) == 1
        failure = metadata.control_failures[0]
        assert failure.request_id == request.request_id
        assert failure.reason is DualPathControlFailureReason.REVERSE_JOB_FAILED


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
        request_key = scheduler._prefill_path_results[request.request_id].request_key
        first_future = scheduler._prefill_delivery_futures[request_key]

        request.num_preemptions += 1
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 16)
        # Attempt 1 is installed, but its delivery is deferred: no second
        # submit and no overwrite of the unresolved attempt-0 Future.
        assert scheduler._prefill_reverse_plans[request.request_id].reverse_attempt_id == 1
        assert coordinator.submit.call_count == 1
        assert scheduler._prefill_delivery_futures[request_key] is first_future

        # Once attempt 0's delivery resolves, the next build pass delivers.
        pending_future.done.return_value = True
        scheduler.build_connector_meta(make_empty_scheduler_output())
        assert coordinator.submit.call_count == 2
        second_decision = coordinator.submit.call_args_list[1].args[1]
        assert second_decision.result.reverse_attempt_id == 1
        assert scheduler._prefill_delivery_futures[request_key] is not first_future


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
        request_key = scheduler._prefill_path_results[request.request_id].request_key
        assert request_key in scheduler._prefill_deferred_deliveries
        return request, pending_future

    def test_failed_prior_future_cancels_deferred_replacement(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request, pending_future = self._defer_replacement(scheduler, coordinator)

        pending_future.done.return_value = True
        pending_future.exception.return_value = RuntimeError("delivery exhausted")
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        request_key = scheduler._prefill_request_keys[request.request_id]
        assert coordinator.submit.call_count == 1
        assert request_key in scheduler._prefill_invalid_request_keys
        assert request_key not in scheduler._prefill_deferred_deliveries
        assert len(metadata.control_failures) == 1
        # The invalid request can never be delivered later, even by a direct call.
        scheduler._deliver_prefill_decision(request_key)
        assert coordinator.submit.call_count == 1

    def test_cancelled_prior_future_cancels_deferred_replacement(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request, pending_future = self._defer_replacement(scheduler, coordinator)

        pending_future.done.return_value = True
        pending_future.cancelled.return_value = True
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        request_key = scheduler._prefill_request_keys[request.request_id]
        assert coordinator.submit.call_count == 1
        assert request_key in scheduler._prefill_invalid_request_keys
        assert request_key not in scheduler._prefill_deferred_deliveries
        assert len(metadata.control_failures) == 1


class TestJobLedgerRetirementWiring:
    def test_reverse_completion_close_retires_job_record(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(completion_reports={binding.reverse_receive_completion_id: 1})
        )
        scheduler.update_connector_output(output)

        assert output.finished_recving == {request.request_id}
        assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None

    def test_failed_job_record_is_retained(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(failure_reports={binding.reverse_receive_completion_id: 1})
        )
        scheduler.update_connector_output(output)

        assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id).failed is True


def _deliver_decision_to(receiver, attempt_id: int) -> None:
    from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _deliver_frames
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel as channel

    _deliver_frames(receiver, channel.encode_path_decision(_de_read_decision(attempt_id)))


class TestReverseSendJobRetirement:
    def test_delayed_free_reverse_send_close_retires_job_record(self, decode_scheduler_factory, decode_task04_seams):
        from tests.ut.distributed.kv_transfer.dual_path.conftest import (
            DECODE_TEST_CONTROL_ENDPOINT,
        )
        from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _make_receiver

        scheduler = decode_scheduler_factory()
        receiver = _make_receiver()
        receiver._decode_control_endpoint = DECODE_TEST_CONTROL_ENDPOINT
        receiver._context = MagicMock(name="zmq_context")
        receiver._receiver_thread = MagicMock(name="receiver_thread")
        scheduler._path_decision_coordinator = receiver
        request = _admit_decode_request(scheduler)
        _deliver_decision_to(receiver, 0)
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        completion_id = metadata.reverse_plans[0].reverse_send_completion_id

        # Delayed-free flow: release runs while the completion is open and skips its
        # retirement; the completion close must retire the record itself.
        request.status = RequestStatus.FINISHED_STOPPED
        delay_free, _ = scheduler.request_finished(request, [])
        assert delay_free is True
        assert scheduler._completion_tracker.get(completion_id) is not None

        output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
        scheduler.update_connector_output(output)

        assert output.finished_sending == {request.request_id}
        assert scheduler._completion_tracker.get(completion_id) is None

    def test_normal_reverse_send_close_retires_its_attempt_immediately(
        self, decode_scheduler_factory, decode_task04_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
        completion_id = metadata.reverse_plans[0].reverse_send_completion_id
        attempt_key = ReverseAttemptKey(
            DualPathRequestKey(
                scheduler._path_decision_coordinator.decode_engine_instance_id,
                request.request_id,
                0,
            ),
            0,
        )

        output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
        scheduler.update_connector_output(output)

        assert output.finished_sending is None
        assert attempt_key not in scheduler._reverse_send_completion_ids
        assert scheduler._completion_tracker.get(completion_id) is None

        request.status = RequestStatus.FINISHED_STOPPED
        delay_free, _ = scheduler.request_finished(request, [])
        assert delay_free is False

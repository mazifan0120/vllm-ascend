# SPDX-License-Identifier: Apache-2.0
"""Completion and delivery recovery regression tests."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_completed_future,
    make_empty_scheduler_output,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_current_reverse_attempt_completion import (
    _admit_de_read_request,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
)
from tests.ut.distributed.kv_transfer.dual_path.test_resume_admission import (
    _admit_de_read,
)
from tests.ut.distributed.kv_transfer.dual_path.test_reverse_send_completion import (
    _REQUEST_KEY,
    _activate_decision,
    _admit_decode_request,
    _de_read_decision,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathAbortNotice,
    PathAbortReason,
    PathKind,
    ReverseAttemptKey,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
)


_PREFILL_ENDPOINT_A = DecodeControlEndpoint(host="198.51.100.41", port=24041)
_PREFILL_ENDPOINT_B = DecodeControlEndpoint(host="198.51.100.42", port=24042)


class TestDecodeEngineProgress:
    def test_de_final_request_delayed_free_until_reverse_send_completion_closes(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(0))
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


class TestDecodeFailureRelay:
    @staticmethod
    def _decision_with_endpoint(
        attempt_id: int,
        endpoint: DecodeControlEndpoint | None,
        *,
        remote_tp_size: int = 1,
    ):
        return replace(
            _de_read_decision(attempt_id, remote_tp_size=remote_tp_size),
            prefill_control_endpoint=endpoint,
        )

    @staticmethod
    def _expected_abort() -> PathAbortNotice:
        return PathAbortNotice(
            request_key=_REQUEST_KEY,
            reason=PathAbortReason.ACTIVATION_FAILED,
        )

    def test_decode_activation_validation_failure_sends_abort_without_persisting_untrusted_endpoint(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        state = scheduler._decode_decision_states[request.request_id]

        metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A, remote_tp_size=2),
        )

        assert state.status.value == "ACTIVATION_FAILED"
        assert state.prefill_control_endpoint is None
        decode_control_seams.decode_coordinator.unregister.assert_called_with(state.request_key)
        assert [failure.reason for failure in metadata.control_failures] == [
            DualPathControlFailureReason.ACTIVATION_FAILED
        ]
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            self._expected_abort(),
        )

    def test_decode_worker_failure_uses_persisted_prefill_endpoint(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        state = scheduler._decode_decision_states[request.request_id]
        metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A),
        )
        completion_id = metadata.reverse_plans[0].reverse_send_completion_id

        assert state.prefill_control_endpoint == _PREFILL_ENDPOINT_A
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=make_worker_metadata(
                    failure_reports={completion_id: 1}
                )
            )
        )

        assert state.status.value == "ACTIVATION_FAILED"
        assert scheduler._decode_control_failures[request.request_id].reason is (
            DualPathControlFailureReason.REVERSE_JOB_FAILED
        )
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            self._expected_abort(),
        )

    def test_decode_failure_still_sends_abort_when_control_failure_build_raises(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        state = scheduler._decode_decision_states[request.request_id]

        with patch.object(
            scheduler,
            "_build_decode_control_failure",
            side_effect=RuntimeError("control failure unavailable"),
        ):
            metadata = _activate_decision(
                scheduler,
                decode_control_seams,
                self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A, remote_tp_size=2),
            )

        assert metadata.control_failures == []
        assert state.status.value == "ACTIVATION_FAILED"
        decode_control_seams.decode_coordinator.unregister.assert_called_with(state.request_key)
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            self._expected_abort(),
        )

    def test_decode_failure_without_endpoint_is_local_only(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        state = scheduler._decode_decision_states[request.request_id]

        metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, None, remote_tp_size=2),
        )

        assert state.status.value == "ACTIVATION_FAILED"
        assert state.prefill_control_endpoint is None
        assert [failure.reason for failure in metadata.control_failures] == [
            DualPathControlFailureReason.ACTIVATION_FAILED
        ]
        decode_control_seams.decode_coordinator.submit_abort.assert_not_called()

    def test_failed_attempt_refresh_does_not_force_close_older_inflight_attempt(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory(world_size=2)
        request = _admit_decode_request(scheduler)
        first_metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A, remote_tp_size=2),
        )
        completion_id = first_metadata.reverse_plans[0].reverse_send_completion_id

        refresh_metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(1, _PREFILL_ENDPOINT_A, remote_tp_size=1),
        )

        state = scheduler._decode_decision_states[request.request_id]
        completion = scheduler._completion_tracker.get(completion_id)
        assert state.status.value == "ACTIVATION_FAILED"
        assert refresh_metadata.reverse_plans == []
        assert completion is not None and completion.closed is False
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            self._expected_abort(),
        )
        request.status = RequestStatus.FINISHED_STOPPED
        assert scheduler.request_finished(request, []) == (True, None)

        first_report = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(
                completion_reports={completion_id: 1}
            )
        )
        scheduler.update_connector_output(first_report)
        assert first_report.finished_sending is None
        completion = scheduler._completion_tracker.get(completion_id)
        assert completion is not None and completion.closed is False

        second_report = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(
                completion_reports={completion_id: 1}
            )
        )
        scheduler.update_connector_output(second_report)
        assert second_report.finished_sending == {request.request_id}
        assert scheduler._completion_tracker.get(completion_id) is None

    def test_attempt_refresh_endpoint_mismatch_fails_closed_to_original_endpoint(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A),
        )

        refresh_metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(1, _PREFILL_ENDPOINT_B),
        )

        state = scheduler._decode_decision_states[request.request_id]
        assert state.status.value == "ACTIVATION_FAILED"
        assert state.prefill_control_endpoint == _PREFILL_ENDPOINT_A
        assert refresh_metadata.reverse_plans == []
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            self._expected_abort(),
        )

    def test_decode_late_worker_failure_after_request_finish_still_notifies_prefill(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        first_metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A),
        )
        second_metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(1, _PREFILL_ENDPOINT_A),
        )
        first_completion_id = first_metadata.reverse_plans[0].reverse_send_completion_id
        second_completion_id = second_metadata.reverse_plans[0].reverse_send_completion_id
        request_key = scheduler._decode_decision_states[request.request_id].request_key
        request.status = RequestStatus.FINISHED_STOPPED

        assert scheduler.request_finished(request, []) == (True, None)
        assert scheduler._decode_late_abort_endpoints == {
            request_key: _PREFILL_ENDPOINT_A
        }

        with (
            patch.object(scheduler, "_build_decode_control_failure") as build_failure,
            patch(
                "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.logger"
            ) as scheduler_logger,
        ):
            scheduler.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=make_worker_metadata(
                        failure_reports={first_completion_id: 1}
                    )
                )
            )
            scheduler.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=make_worker_metadata(
                        failure_reports={second_completion_id: 1}
                    )
                )
            )

        build_failure.assert_not_called()
        assert request_key not in scheduler._decode_late_abort_endpoints
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            self._expected_abort(),
        )
        assert any(
            "has no Decode state or snapshot" in str(call)
            for call in scheduler_logger.error.call_args_list
        )

    def test_late_failure_for_old_admission_does_not_fail_same_id_replacement(
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        old_request = _admit_decode_request(scheduler)
        old_state = scheduler._decode_decision_states[old_request.request_id]
        old_key = old_state.request_key
        activation_metadata = _activate_decision(
            scheduler,
            decode_control_seams,
            self._decision_with_endpoint(0, _PREFILL_ENDPOINT_A),
        )
        completion_id = activation_metadata.reverse_plans[0].reverse_send_completion_id
        old_request.status = RequestStatus.FINISHED_STOPPED
        assert scheduler.request_finished(old_request, []) == (True, None)
        assert scheduler._decode_late_abort_endpoints == {
            old_key: _PREFILL_ENDPOINT_A
        }

        replacement = _admit_decode_request(scheduler)
        replacement_state = scheduler._decode_decision_states[replacement.request_id]
        replacement_key = replacement_state.request_key
        assert replacement_key != old_key
        assert replacement_state.status.value == "PENDING"
        decode_control_seams.decode_coordinator.register_pending.assert_called_with(
            replacement_key
        )
        decode_control_seams.decode_coordinator.unregister.reset_mock()

        with patch.object(
            scheduler,
            "_build_decode_control_failure",
        ) as build_failure:
            scheduler.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=make_worker_metadata(
                        failure_reports={completion_id: 1}
                    )
                )
            )

        assert replacement_state.status.value == "PENDING"
        assert scheduler._decode_decision_states[replacement.request_id] is replacement_state
        decode_control_seams.decode_coordinator.unregister.assert_not_called()
        build_failure.assert_not_called()
        assert replacement.request_id not in scheduler._decode_control_failures
        assert old_key not in scheduler._decode_late_abort_endpoints
        decode_control_seams.decode_coordinator.submit_abort.assert_called_once_with(
            _PREFILL_ENDPOINT_A,
            PathAbortNotice(
                request_key=old_key,
                reason=PathAbortReason.ACTIVATION_FAILED,
            ),
        )


class TestBoundedCompletions:
    def test_failed_reverse_completion_surfaces_control_failure(self, pe_scheduler_factory):
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

    def test_mixed_worker_terminals_close_failed_reverse_receive_once(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        scheduler._expected_worker_count = 2
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(
                completion_reports={binding.reverse_receive_completion_id: 1},
                failure_reports={binding.reverse_receive_completion_id: 1},
            )
        )
        scheduler.update_connector_output(output)

        assert output.finished_recving == {request.request_id}
        assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None
        assert binding.request_key in scheduler._prefill_invalid_request_keys


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


class TestReplacementActivationRollback:
    @staticmethod
    def _install_attempt_zero(scheduler):
        request = _admit_de_read(scheduler)
        scheduler.build_connector_meta(make_empty_scheduler_output())
        request_id = request.request_id
        return (
            request,
            scheduler._prefill_reverse_plans[request_id],
            scheduler._waiting_reverse_attempt_ids[request_id],
        )

    @staticmethod
    def _assert_attempt_zero_restored(scheduler, request, old_plan, old_waiting):
        request_id = request.request_id
        assert scheduler._prefill_reverse_plans[request_id] is old_plan
        assert scheduler._waiting_reverse_attempt_ids[request_id] == old_waiting
        assert scheduler._prefill_pending_reverse_receive_bindings == {}
        assert scheduler._completion_tracker.open_count() == 1

    def test_sync_replacement_submit_failure_restores_previous_attempt(self, pe_scheduler_factory):
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=make_block_pool())
        coordinator.submit.side_effect = [make_completed_future(), RuntimeError("coordinator closed")]
        request, old_plan, old_waiting = self._install_attempt_zero(scheduler)

        request.num_preemptions += 1
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 16)

        self._assert_attempt_zero_restored(scheduler, request, old_plan, old_waiting)

    def test_post_install_activation_failure_restores_previous_attempt(self, pe_scheduler_factory):
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=make_block_pool())
        request, old_plan, old_waiting = self._install_attempt_zero(scheduler)

        request.num_preemptions += 1
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        with patch.object(scheduler, "_may_install_forward_plan", side_effect=RuntimeError("forward conflict")):
            scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 16)

        self._assert_attempt_zero_restored(scheduler, request, old_plan, old_waiting)

    def test_deferred_submit_failure_restores_previous_attempt_without_binding(self, pe_scheduler_factory):
        pending_future = MagicMock(name="delivery_future_attempt_0")
        pending_future.done.return_value = False
        pending_future.cancelled.return_value = False
        pending_future.exception.return_value = None
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=make_block_pool())
        coordinator.submit.side_effect = [pending_future, RuntimeError("coordinator closed")]

        request, old_plan, old_waiting = self._install_attempt_zero(scheduler)
        request_id = request.request_id
        old_forward_plan = scheduler._prefill_forward_plans[request_id]
        old_forward_epoch = scheduler._prefill_forward_plan_epochs[request_id]
        old_send_info = scheduler._reqs_need_send_layerwise[request_id]

        request.num_preemptions += 1
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 16)
        assert coordinator.submit.call_count == 1
        assert scheduler._prefill_reverse_plans[request_id].reverse_attempt_id == 1

        pending_future.done.return_value = True
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        assert coordinator.submit.call_count == 2
        self._assert_attempt_zero_restored(scheduler, request, old_plan, old_waiting)
        assert scheduler._prefill_forward_plans[request_id] is old_forward_plan
        assert scheduler._prefill_forward_plan_epochs[request_id] == old_forward_epoch
        assert scheduler._reqs_need_send_layerwise[request_id] is old_send_info
        assert metadata.reverse_receive_bindings == []


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
        metadata = scheduler.build_connector_meta(make_empty_scheduler_output())
        assert metadata.reverse_receive_bindings == []
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
        assert request_key not in scheduler._prefill_deferred_activations
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
        assert request_key not in scheduler._prefill_deferred_activations
        assert len(metadata.control_failures) == 1


class TestCompletionTrackerRetirementWiring:
    def test_reverse_completion_close_retires_tracker_record(self, pe_scheduler_factory):
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

    def test_failed_completion_closes_and_retires_tracker_record(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read_request(scheduler)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]

        output = KVConnectorOutput(
            kv_connector_worker_meta=make_worker_metadata(failure_reports={binding.reverse_receive_completion_id: 1})
        )
        scheduler.update_connector_output(output)

        assert output.finished_recving == {request.request_id}
        assert scheduler._completion_tracker.get(binding.reverse_receive_completion_id) is None


def _deliver_decision_to(receiver, attempt_id: int) -> None:
    from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import _deliver_frames
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel as channel

    _deliver_frames(receiver, channel.encode_path_decision(_de_read_decision(attempt_id)))


class TestReverseSendCompletionRetirement:
    def test_delayed_free_reverse_send_close_retires_tracker_record(
        self, decode_scheduler_factory, decode_control_seams
    ):
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
        self, decode_scheduler_factory, decode_control_seams
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        metadata = _activate_decision(scheduler, decode_control_seams, _de_read_decision(0))
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

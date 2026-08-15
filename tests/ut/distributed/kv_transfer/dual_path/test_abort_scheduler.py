# SPDX-License-Identifier: Apache-2.0
"""Scheduler handling for request-terminal PE-to-DE ABORT notices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.ut.distributed.kv_transfer.dual_path.test_de_reverse_send_proof import (
    _admit_decode_request,
    _de_read_decision,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import (
    path_decision as decision_model,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import (
    scheduler as scheduler_module,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
)


def _notice(request_key, reason: str = "REQUEST_ABORTED"):
    return decision_model.PathAbortNotice(
        request_key=request_key,
        reason=decision_model.PathAbortReason(reason),
    )


def test_pending_decode_abort_stages_peer_abort_failure(decode_scheduler_factory, decode_task04_seams) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decode_task04_seams.decode_coordinator.take_received_decisions.return_value = []
    decode_task04_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

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
    decode_task04_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)


def test_same_drain_commits_decision_then_aborts_without_touching_job_ledger(
    decode_scheduler_factory,
    decode_task04_seams,
) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decision = _de_read_decision()
    decode_task04_seams.decode_coordinator.take_received_decisions.return_value = [decision]
    decode_task04_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert len(metadata.reverse_plans) == 1
    job_id = metadata.reverse_plans[0].reverse_send_job_id
    assert job_id is not None
    job = scheduler._job_ledger.get(job_id)
    assert job is not None
    assert job.closed is False
    assert job.failed is False
    assert scheduler._job_ledger.open_count() == 1
    assert len(scheduler._reverse_send_job_ids) == 1
    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.PEER_ABORT,
        )
    ]


def test_unknown_decode_abort_is_ignored(decode_scheduler_factory, decode_task04_seams) -> None:
    scheduler = decode_scheduler_factory()
    unknown_key = decision_model.DualPathRequestKey(
        decode_task04_seams.decode_coordinator.decode_engine_instance_id,
        "unknown-request",
    )

    scheduler._handle_received_abort(_notice(unknown_key))

    assert scheduler._decode_control_failures == {}
    decode_task04_seams.decode_coordinator.unregister.assert_not_called()


def test_already_terminal_decode_abort_is_ignored(decode_scheduler_factory, decode_task04_seams) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    state.status = scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED

    scheduler._handle_received_abort(_notice(state.request_key))

    assert scheduler._decode_control_failures == {}
    decode_task04_seams.decode_coordinator.unregister.assert_not_called()


def test_decode_abort_keeps_terminal_state_when_failure_metadata_cannot_be_built(
    decode_scheduler_factory,
    decode_task04_seams,
) -> None:
    scheduler = decode_scheduler_factory()
    request = _admit_decode_request(scheduler)
    state = scheduler._decode_decision_states[request.request_id]
    decode_task04_seams.decode_coordinator.take_received_decisions.return_value = []
    decode_task04_seams.decode_coordinator.take_received_aborts.return_value = [_notice(state.request_key)]

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
    decode_task04_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)
    log_error.assert_called_once()

# SPDX-License-Identifier: Apache-2.0
"""ClosedReverseAttemptRecord and the pure close-matrix evaluator.

The evaluator is deliberately free of locks, sockets, and scheduler/worker
objects so the receiver thread performs bounded in-memory work only: the
coordinator supplies registry state under its lock and applies the returned
mutations before replying.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    ReverseAttemptKey,
)

__all__ = [
    "CloseAttemptEvaluation",
    "ClosedReverseAttemptRecord",
    "ReverseAttemptRegistryState",
    "evaluate_close_attempt",
]


@dataclass
class ClosedReverseAttemptRecord:
    """One record per closed attempt; no completion counter — the job ledger
    is the sole counting authority for the referenced reverse-send job."""

    attempt_key: ReverseAttemptKey
    activation_claimed: bool
    worker_work_published: bool
    reverse_send_job_id: int | None
    safe_close_proof: bool


@dataclass
class ReverseAttemptRegistryState:
    """Live attempt facts fed by short locked scheduler transitions."""

    activation_claimed: bool = False
    publication_cancelled: bool = False
    worker_work_published: bool = False
    reverse_send_job_id: int | None = None
    sender_complete: bool = False


@dataclass(frozen=True)
class CloseAttemptEvaluation:
    safe: bool
    record: ClosedReverseAttemptRecord | None
    new_closed_through: int | None
    suppress_activation: bool


def evaluate_close_attempt(
    attempt_key: ReverseAttemptKey,
    *,
    record: ClosedReverseAttemptRecord | None,
    admission_present: bool,
    decision_received: bool,
    state: ReverseAttemptRegistryState | None,
    closed_through: int | None,
) -> CloseAttemptEvaluation:
    """The close matrix, in exact row order; a persisted proof always wins."""
    if record is not None and record.safe_close_proof:
        return CloseAttemptEvaluation(True, record, None, False)
    if record is None and not admission_present and state is None:
        return CloseAttemptEvaluation(False, None, None, False)

    if state is not None and state.worker_work_published and state.sender_complete:
        closed_record = ClosedReverseAttemptRecord(attempt_key, True, True, state.reverse_send_job_id, True)
        return CloseAttemptEvaluation(True, closed_record, attempt_key.reverse_attempt_id, False)

    if not decision_received and not (state and state.activation_claimed):
        if closed_through is not None and attempt_key.reverse_attempt_id <= closed_through:
            # Watermark-only coverage: "cannot reopen" is not "writer stopped".
            stale_record = ClosedReverseAttemptRecord(attempt_key, False, False, None, False)
            return CloseAttemptEvaluation(False, stale_record, None, False)
        never_received = ClosedReverseAttemptRecord(attempt_key, False, False, None, True)
        return CloseAttemptEvaluation(True, never_received, attempt_key.reverse_attempt_id, False)

    if not (state and state.activation_claimed):
        unclaimed = ClosedReverseAttemptRecord(attempt_key, False, False, None, True)
        return CloseAttemptEvaluation(True, unclaimed, attempt_key.reverse_attempt_id, True)

    if state.publication_cancelled and not state.worker_work_published:
        cancelled = ClosedReverseAttemptRecord(attempt_key, True, False, state.reverse_send_job_id, True)
        return CloseAttemptEvaluation(True, cancelled, attempt_key.reverse_attempt_id, False)

    incomplete = ClosedReverseAttemptRecord(
        attempt_key,
        True,
        state.worker_work_published,
        state.reverse_send_job_id,
        False,
    )
    return CloseAttemptEvaluation(False, incomplete, None, False)

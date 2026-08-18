# SPDX-License-Identifier: Apache-2.0
"""Track per-attempt DualPath transfer completion across workers.

Each reverse attempt opens one completion on the Scheduler; every
participating worker reports it exactly once through
``DualPathWorkerMetadata``; at ``expected_worker_count`` the close action
runs exactly once. Reports arriving for a closed completion are ignored.
Failure is latched immediately, but does not close the completion before every
worker has reported a terminal result.

Close actions run on the Scheduler: they unpark via ``finished_recving``,
authorize delayed free via ``finished_sending``, or fail the owning request.
Concurrently open completions are tolerated and each closes independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    ReverseAttemptKey,
)

__all__ = [
    "CompletionKind",
    "TransferCompletionTracker",
    "CompletionRecord",
]


class CompletionKind(str, Enum):
    REVERSE_RECEIVE = "REVERSE_RECEIVE"
    REVERSE_SEND = "REVERSE_SEND"


@dataclass
class CompletionRecord:
    completion_id: int
    completion_kind: CompletionKind
    expected_worker_count: int
    reverse_attempt_key: ReverseAttemptKey | None = None
    completed_worker_count: int = 0
    failed: bool = False
    closed: bool = False


@dataclass
class TransferCompletionTracker:
    """Monotonic completion-id allocator with the all-worker completion rule.

    Each worker contributes at most one report per completion; aggregated
    counts must not exceed ``expected_worker_count``, and the kind-specific
    close action runs exactly once. Reports arriving for a ``closed``
    completion are ignored. Failure is latched while all-worker closure is
    still enforced.
    """

    _records: dict[int, CompletionRecord] = field(default_factory=dict)
    _next_completion_id: int = 0

    def open_completion(
        self,
        completion_kind: CompletionKind,
        *,
        expected_worker_count: int,
        reverse_attempt_key: ReverseAttemptKey | None = None,
    ) -> CompletionRecord:
        if isinstance(expected_worker_count, bool) or not isinstance(expected_worker_count, int):
            raise TypeError("expected_worker_count must be an integer")
        if expected_worker_count <= 0:
            raise ValueError("expected_worker_count must be greater than zero")
        record = CompletionRecord(
            completion_id=self._next_completion_id,
            completion_kind=completion_kind,
            expected_worker_count=expected_worker_count,
            reverse_attempt_key=reverse_attempt_key,
        )
        self._next_completion_id += 1
        self._records[record.completion_id] = record
        return record

    def get(self, completion_id: int) -> CompletionRecord | None:
        return self._records.get(completion_id)

    def find_open_completion(
        self,
        completion_kind: CompletionKind,
        reverse_attempt_key: ReverseAttemptKey,
    ) -> CompletionRecord | None:
        """Return the unique open completion for one exact Reverse attempt."""
        matches = [
            record
            for record in self._records.values()
            if not record.closed
            and record.completion_kind is completion_kind
            and record.reverse_attempt_key == reverse_attempt_key
        ]
        if len(matches) > 1:
            raise RuntimeError("multiple open completions exist for one Reverse attempt")
        return matches[0] if matches else None

    def discard(self, completion_id: int) -> bool:
        """Retire a closed record with its owning attempt; open completions stay."""
        record = self._records.get(completion_id)
        if record is None or not record.closed:
            return False
        del self._records[completion_id]
        return True

    def discard_unstarted(
        self,
        completion_id: int,
        *,
        expected_kind: CompletionKind,
        expected_attempt_key: ReverseAttemptKey,
    ) -> bool:
        """Discard one exact completion that provably never reached a worker.

        This is a close-without-action operation: it removes the record without
        marking it closed, running a close action, or producing a core terminal.
        """
        record = self._records.get(completion_id)
        if record is None or record.closed:
            return False
        if record.completion_kind is not expected_kind or record.reverse_attempt_key != expected_attempt_key:
            raise RuntimeError("completion identity mismatch while discarding unstarted attempt")
        if record.completed_worker_count != 0:
            raise RuntimeError("cannot discard completion that already received terminal reports")
        del self._records[completion_id]
        return True

    def discard_closed_completions(self, completion_kind: CompletionKind, request_key: DualPathRequestKey) -> None:
        """Retire every closed completion of ``completion_kind`` owned by ``request_key``."""
        for completion_id, record in list(self._records.items()):
            attempt_key = record.reverse_attempt_key
            if (
                record.completion_kind is completion_kind
                and attempt_key is not None
                and attempt_key.request_key == request_key
                and record.closed
            ):
                del self._records[completion_id]

    def open_count(self) -> int:
        return sum(1 for record in self._records.values() if not record.closed)

    def tally_reports(
        self,
        completion_id: int,
        success_count: int = 0,
        failure_count: int = 0,
    ) -> bool:
        """Accumulate worker terminals; return True iff closure happens now."""
        record = self._records.get(completion_id)
        if record is None or record.closed:
            return False
        if success_count < 0 or failure_count < 0:
            raise ValueError("completion report counts must be non-negative")
        report_count = success_count + failure_count
        if report_count == 0:
            return False
        completed_worker_count = record.completed_worker_count + report_count
        if completed_worker_count > record.expected_worker_count:
            raise RuntimeError("completion reports exceed expected worker count")
        record.completed_worker_count = completed_worker_count
        if failure_count:
            record.failed = True
        if record.completed_worker_count < record.expected_worker_count:
            return False
        record.closed = True
        return True

    def fail_completion(self, completion_id: int) -> bool:
        """Record one failed worker terminal; True iff this closes the record."""
        return self.tally_reports(completion_id, failure_count=1)

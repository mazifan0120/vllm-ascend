# SPDX-License-Identifier: Apache-2.0
"""Completion-job ledger for DualPath Stage-2.

Both roles aggregate all-worker completion proofs through this ledger: the PE
role gates a parked DE_READ request on its Reverse completion job, and the DE
role gates the source blocks of a finishing request on its Reverse-send job.
Reports are counted exactly once per worker and capped at the expected count,
so a duplicated report can never close a job early or twice.

No KV block is pinned here. Both directions follow the parent's immediate-free
semantics: Forward is the ordinary Layerwise push, and an aborted Reverse
destination is released with the request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    ReverseAttemptKey,
)

__all__ = [
    "JobKind",
    "JobLedger",
    "JobRecord",
]


class JobKind(str, Enum):
    REVERSE_COMPLETION = "REVERSE_COMPLETION"
    REVERSE_SEND = "REVERSE_SEND"


@dataclass
class JobRecord:
    job_id: int
    job_kind: JobKind
    expected_worker_count: int
    reverse_attempt_key: ReverseAttemptKey | None = None
    completed_worker_count: int = 0
    failed: bool = False
    closed: bool = False


@dataclass
class JobLedger:
    """Monotonic job-id allocator with the all-worker completion rule.

    Each worker contributes at most one report per job; aggregated counts are
    capped at ``expected_worker_count`` so a duplicated report can never
    overshoot, and the kind-specific close action runs exactly once. Reports
    arriving for a ``closed`` job are ignored. A failure report closes the job
    as ``failed``.
    """

    _records: dict[int, JobRecord] = field(default_factory=dict)
    _next_job_id: int = 0

    def create_job(
        self,
        job_kind: JobKind,
        *,
        expected_worker_count: int,
        reverse_attempt_key: ReverseAttemptKey | None = None,
    ) -> JobRecord:
        if isinstance(expected_worker_count, bool) or not isinstance(expected_worker_count, int):
            raise TypeError("expected_worker_count must be an integer")
        if expected_worker_count <= 0:
            raise ValueError("expected_worker_count must be greater than zero")
        record = JobRecord(
            job_id=self._next_job_id,
            job_kind=job_kind,
            expected_worker_count=expected_worker_count,
            reverse_attempt_key=reverse_attempt_key,
        )
        self._next_job_id += 1
        self._records[record.job_id] = record
        return record

    def get(self, job_id: int) -> JobRecord | None:
        return self._records.get(job_id)

    def discard(self, job_id: int) -> bool:
        """Retire a closed record with its owning attempt; open jobs stay."""
        record = self._records.get(job_id)
        if record is None or not record.closed:
            return False
        del self._records[job_id]
        return True

    def open_count(self) -> int:
        return sum(1 for record in self._records.values() if not record.closed)

    def record_reports(self, job_id: int, report_count: int) -> bool:
        """Accumulate worker reports; returns True iff the job closes now."""
        record = self._records.get(job_id)
        if record is None or record.closed or report_count <= 0:
            return False
        record.completed_worker_count = min(
            record.completed_worker_count + report_count,
            record.expected_worker_count,
        )
        if record.completed_worker_count < record.expected_worker_count:
            return False
        record.closed = True
        return True

    def record_failure(self, job_id: int) -> bool:
        """Close the job as failed; True iff newly closed."""
        record = self._records.get(job_id)
        if record is None or record.closed:
            return False
        record.failed = True
        record.closed = True
        return True

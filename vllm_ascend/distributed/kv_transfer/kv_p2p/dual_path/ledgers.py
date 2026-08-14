# SPDX-License-Identifier: Apache-2.0
"""Decision-complete hold and completion-job ledgers for DualPath Stage-2.

The PE role binds the block pool and pins Reverse destination blocks until the
all-worker completion proof for the gating job arrives; the DE role keeps the
same job ledger for its Reverse-send completion proofs. Each acquisition and
release flows through exactly one ledger record so overlapping physical sets,
retries, and cleanup cannot unbalance ``BlockPool`` refcounts.

Forward source blocks are deliberately not pinned. The Forward direction is the
ordinary Layerwise push, whose completion is reported to the peer over the
control channel rather than to the local scheduler, so its blocks follow the
parent's immediate-free semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    ReverseAttemptKey,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool

__all__ = [
    "HoldKind",
    "HoldLedger",
    "HoldRecord",
    "JobKind",
    "JobLedger",
    "JobRecord",
]


class HoldKind(str, Enum):
    REVERSE_DESTINATION = "REVERSE_DESTINATION"


@dataclass
class HoldRecord:
    """Exactly one record per held reference."""

    hold_id: int
    block_ids: tuple[int, ...]
    hold_kind: HoldKind
    released: bool = False


class HoldLedger:
    def __init__(self) -> None:
        self._records: dict[int, HoldRecord] = {}
        self._next_hold_id: int = 0

    def acquire(
        self,
        block_pool: BlockPool,
        block_ids: tuple[int, ...],
        hold_kind: HoldKind,
    ) -> HoldRecord:
        record = HoldRecord(
            hold_id=self._next_hold_id,
            block_ids=tuple(block_ids),
            hold_kind=hold_kind,
        )
        self._next_hold_id += 1
        block_pool.touch([block_pool.blocks[block_id] for block_id in record.block_ids])
        self._records[record.hold_id] = record
        return record

    def get(self, hold_id: int) -> HoldRecord | None:
        return self._records.get(hold_id)

    def is_released(self, hold_id: int) -> bool:
        record = self._records.get(hold_id)
        return record is None or record.released

    def release(self, block_pool: BlockPool, hold_id: int) -> bool:
        """Release one held reference; repeated releases are no-ops."""
        record = self._records.get(hold_id)
        if record is None or record.released:
            return False
        record.released = True
        block_pool.free_blocks([block_pool.blocks[block_id] for block_id in record.block_ids])
        return True

    def discard(self, hold_id: int) -> bool:
        """Retire a released record with its owning attempt; open holds stay."""
        record = self._records.get(hold_id)
        if record is None or not record.released:
            return False
        del self._records[hold_id]
        return True

    def held_block_count(self) -> int:
        return sum(len(record.block_ids) for record in self._records.values() if not record.released)

    def unreleased_count(self) -> int:
        return sum(1 for record in self._records.values() if not record.released)


class JobKind(str, Enum):
    REVERSE_COMPLETION = "REVERSE_COMPLETION"
    REVERSE_SEND = "REVERSE_SEND"


@dataclass
class JobRecord:
    job_id: int
    job_kind: JobKind
    expected_worker_count: int
    affected_hold_ids: tuple[int, ...] = ()
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
    as ``failed`` without releasing any hold.
    """

    _records: dict[int, JobRecord] = field(default_factory=dict)
    _next_job_id: int = 0

    def create_job(
        self,
        job_kind: JobKind,
        *,
        expected_worker_count: int,
        affected_hold_ids: tuple[int, ...] = (),
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
            affected_hold_ids=tuple(affected_hold_ids),
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
        """Close the job as failed without releasing holds; True iff newly closed."""
        record = self._records.get(job_id)
        if record is None or record.closed:
            return False
        record.failed = True
        record.closed = True
        return True

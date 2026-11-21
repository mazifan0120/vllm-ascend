# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side retention and rank fan-in for layerwise decode targets."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_plan import (
    CatalogRef,
    RegisteredLayout,
    TargetAllocation,
    TargetId,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True)
class DecodeWorkerEvent:
    kind: Literal["prepared", "ready", "closed", "failed"]
    target: TargetId
    worker: CatalogRef


@dataclass(frozen=True)
class DecodeWorkerLink:
    """Enqueue-only control callbacks owned by the connector's control loop."""

    prepare: Callable[[TargetAllocation, tuple[CatalogRef, ...]], None]
    close: Callable[[TargetId, tuple[CatalogRef, ...]], None]
    publish: Callable[["Request", TargetAllocation, tuple[CatalogRef, ...]], None]


@dataclass
class _Retained:
    allocation: TargetAllocation
    prepared: set[int] = field(default_factory=set)
    ready: set[int] = field(default_factory=set)
    closed: set[int] = field(default_factory=set)
    published: bool = False
    closing: bool = False


class DecodeAdmission:
    """Runs on the scheduler thread, with worker responses delivered as events."""

    def __init__(
        self,
        layout: RegisteredLayout,
        hold_blocks: Callable[[TargetAllocation], None],
        release_blocks: Callable[[TargetAllocation], None],
    ) -> None:
        self._refs = layout.refs
        self._hold_blocks = hold_blocks
        self._release_blocks = release_blocks
        self._retained: dict[TargetId, _Retained] = {}
        self._by_request: dict[str, TargetId] = {}
        self._released: set[TargetId] = set()

    @property
    def worker_count(self) -> int:
        return len(self._refs)

    def retain(self, allocation: TargetAllocation) -> tuple[CatalogRef, ...]:
        target = allocation.target
        existing = self._by_request.get(target.request)
        if target in self._released or (existing is not None and existing != target):
            raise RuntimeError("Cannot replace a released or still-retained target generation")
        previous = self._retained.get(target)
        if previous is None:
            self._hold_blocks(allocation)
            self._retained[target] = _Retained(allocation)
            self._by_request[target.request] = target
        elif previous.allocation != allocation or previous.closing:
            raise RuntimeError("Cannot change or reuse a closing target allocation")
        return self._refs

    def prepared(self, target: TargetId, ref: CatalogRef) -> None:
        record = self._for_event(target, ref)
        if record is not None and not record.closing:
            record.prepared.add(ref.rank)

    def publish(self, target: TargetId) -> tuple[TargetAllocation, tuple[CatalogRef, ...]] | None:
        record = self._retained.get(target)
        if record is None or record.published or record.closing or len(record.prepared) != len(self._refs):
            return None
        record.published = True
        return record.allocation, self._refs

    def ready(self, target: TargetId, ref: CatalogRef) -> bool:
        record = self._for_event(target, ref)
        if record is None or record.closing:
            return False
        if not record.published or ref.rank not in record.prepared:
            raise RuntimeError("Decode worker reported ready before publication")
        if ref.rank in record.ready:
            return False
        record.ready.add(ref.rank)
        return len(record.ready) == self.worker_count

    def may_decode(self, target: TargetId) -> bool:
        record = self._retained.get(target)
        return record is not None and not record.closing and len(record.ready) == len(self._refs)

    def begin_close(self, target: TargetId) -> tuple[CatalogRef, ...]:
        record = self._retained.get(target)
        if record is None:
            return ()
        record.closing = True
        # Send to every potentially addressed rank, including late Prepare commands.
        return tuple(ref for ref in self._refs if ref.rank not in record.closed)

    def closed(self, target: TargetId, ref: CatalogRef) -> bool:
        record = self._for_event(target, ref)
        if record is None:
            return target in self._released
        if not record.closing:
            raise RuntimeError("Unsolicited target close acknowledgment")
        record.closed.add(ref.rank)
        if len(record.closed) != len(self._refs):
            return False
        self._release_blocks(record.allocation)
        del self._retained[target]
        del self._by_request[target.request]
        self._released.add(target)
        return True

    def validate_worker(self, ref: CatalogRef) -> None:
        if not 0 <= ref.rank < len(self._refs) or self._refs[ref.rank] != ref:
            raise ValueError("Target event belongs to a different worker registration")

    def _for_event(self, target: TargetId, ref: CatalogRef) -> _Retained | None:
        self.validate_worker(ref)
        return self._retained.get(target)

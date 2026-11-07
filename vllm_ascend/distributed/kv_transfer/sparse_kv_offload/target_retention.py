# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned allocation retention, consumed through an owner-side client.

The scheduler issues records from its actual KVCacheManager allocation and must
close them before freeing/preempting blocks. This does not touch block ref_cnt:
extra cache references can incorrectly trigger decode copy-on-write.
"""

import threading
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol
from uuid import uuid4

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    ContractError,
    ReleaseAck,
)


class RetentionGuard(Protocol):
    def validate(self) -> None: ...
    def release(self) -> None: ...


class RetentionAuthority(Protocol):
    """Implement in the scheduler or a real transport to that owner."""

    def retain(self, allocation: BlockAllocation) -> RetentionGuard: ...


@dataclass
class _RetentionRecord:
    allocation: BlockAllocation
    internal_request_id: str
    guards: int = 0


class _RegistryGuard:
    def __init__(self, registry: "SchedulerRetentionRegistry", allocation: BlockAllocation) -> None:
        self._registry = registry
        self._allocation = allocation
        self._lock = threading.Lock()
        self._released = False

    def validate(self) -> None:
        with self._lock:
            if self._released:
                raise ContractError("Scheduler retention guard has been released")
            self._registry.validate(self._allocation)

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self._registry._release_guard(self._allocation)
                self._released = True

    def __getstate__(self):
        raise TypeError("Retention guards are owner-local, not wire values")


class SchedulerRetentionRegistry:
    """Authoritative records checked against the scheduler's KVCacheManager.

    issue/validate/retire must execute on the scheduler's serialized allocation
    path (or its serialized RPC dispatcher). Runtime callers in another process
    need a RetentionAuthority client, not a pickled registry. B owns wiring the
    normal/cancel/preemption free paths through retire().
    """

    def __init__(self, kv_cache_manager, *, allocation_lock=None) -> None:
        if not callable(getattr(kv_cache_manager, "get_block_ids", None)):
            raise TypeError("Retention requires the real scheduler allocation lookup")
        self._manager = kv_cache_manager
        self._lock = threading.RLock() if allocation_lock is None else allocation_lock
        self._records: dict[str, _RetentionRecord] = {}
        self._requests: dict[object, str] = {}
        self._internal_requests: dict[str, str] = {}

    def _check_blocks(self, allocation: BlockAllocation, request_id: str) -> None:
        current = tuple(tuple(group) for group in self._manager.get_block_ids(request_id))
        expected = allocation.group_block_ids
        # Decode can append blocks. It may not replace the retained prompt map.
        if len(current) != len(expected) or any(
            tuple(actual[: len(retained)]) != tuple(retained) for actual, retained in zip(current, expected)
        ):
            raise ContractError("Scheduler no longer owns the retained prompt block mapping")

    def issue(self, internal_request_id: str, allocation: BlockAllocation) -> BlockAllocation:
        if not internal_request_id or allocation.retention_token:
            raise ContractError("Issue retention exactly once from an unretained allocation")
        with self._lock:
            if allocation.request in self._requests or internal_request_id in self._internal_requests:
                raise ContractError("Request already has scheduler retention")
            self._check_blocks(allocation, internal_request_id)
            token = uuid4().hex
            retained = replace(allocation, retention_token=token)
            self._records[token] = _RetentionRecord(retained, internal_request_id)
            self._requests[allocation.request] = token
            self._internal_requests[internal_request_id] = token
            return retained

    def _record(self, allocation: BlockAllocation) -> _RetentionRecord:
        record = self._records.get(allocation.retention_token or "")
        if record is None or record.allocation != allocation:
            raise ContractError("Unknown, changed or retired scheduler retention")
        self._check_blocks(allocation, record.internal_request_id)
        return record

    def validate(self, allocation: BlockAllocation) -> None:
        with self._lock:
            self._record(allocation)

    def validate_scheduled(self, request_ids: Iterable[str]) -> dict[str, BlockAllocation]:
        """Validate each retained allocation once for this scheduler output."""
        with self._lock:
            allocations = {}
            for request_id in request_ids:
                token = self._internal_requests.get(request_id)
                if token is None:
                    continue  # A local request has no receipt lease.
                record = self._records[token]
                if not record.guards:
                    raise ContractError("Scheduled receipt target has no active owner retention")
                self._check_blocks(record.allocation, request_id)
                allocations[request_id] = record.allocation
            return allocations

    def retain(self, allocation: BlockAllocation) -> RetentionGuard:
        with self._lock:
            self._record(allocation).guards += 1
            return _RegistryGuard(self, allocation)

    def release_guard(self, allocation: BlockAllocation) -> None:
        """Public release of one guard counted by :meth:`retain`.

        A wire retention dispatcher mirrors the local ``_RegistryGuard`` release
        through this entry, so the owner-side guard never needs the registry
        object itself across the process boundary.
        """
        self._release_guard(allocation)

    def _release_guard(self, allocation: BlockAllocation) -> None:
        with self._lock:
            record = self._record(allocation)
            if record.guards < 1:
                raise ContractError("Scheduler retention guard underflow")
            record.guards -= 1

    def retire(self, allocation: BlockAllocation, ack: ReleaseAck) -> None:
        """Authorize the scheduler's existing free path, never free blocks here.

        The connector must aggregate remote DMA quiescence before producing the
        matching safe ReleaseAck. A runtime release alone is insufficient.
        """
        with self._lock:
            record = self._record(allocation)
            if not ack.safe_to_release or ack.retention_token != allocation.retention_token:
                raise ContractError("Scheduler needs a matching safe release acknowledgement")
            if record.guards:
                raise ContractError("Target owner still retains the scheduler allocation")
            del self._records[allocation.retention_token]
            del self._requests[allocation.request]
            del self._internal_requests[record.internal_request_id]

    def has_request(self, internal_request_id: str) -> bool:
        with self._lock:
            return internal_request_id in self._internal_requests

    def assert_can_free(self, internal_request_id: str) -> None:
        if self.has_request(internal_request_id):
            raise ContractError("Cannot free/pop scheduler blocks while their target allocation is retained")

    def __getstate__(self):
        raise TypeError("Scheduler retention registry must remain in the scheduler")


class SerializedKVCacheManager:
    """Scheduler instance adapter sharing the reverse authority's same lock.

    Serializes method calls rather than holding a lock across scheduler RPCs:
    owner acquire/release may nest a reverse retention call on another thread.
    Allocation methods keep their normal ref counts; retained requests cannot
    enter the real free/pop path until the scheduler retires their authority.
    """

    def __init__(self, manager, allocation_lock) -> None:
        self._manager = manager
        self._lock = allocation_lock
        self._registry = None

    def bind_retention_registry(self, registry) -> None:
        with self._lock:
            if self._registry is not None:
                raise ContractError("Scheduler allocation authority cannot be replaced")
            self._registry = registry

    def __getattr__(self, name):
        value = getattr(self._manager, name)
        if not callable(value):
            return value

        def serialized(*args, **kwargs):
            with self._lock:
                if name in ("free", "pop_blocks_for_free"):
                    if self._registry is None:
                        raise ContractError("Scheduler allocation authority has not been installed")
                    request = args[0] if args else kwargs.get("request")
                    self._registry.assert_can_free(request if isinstance(request, str) else request.request_id)
                return value(*args, **kwargs)

        return serialized

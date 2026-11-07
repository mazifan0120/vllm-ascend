# SPDX-License-Identifier: Apache-2.0
"""Transport-independent retention records, owner guard and scheduler dispatcher.

RPC whitelists record names rather than module paths; moving these records does
not change the versioned frame schema. Live guards remain owner-local.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import BlockAllocation, ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import (
    RetentionGuard,
    SchedulerRetentionRegistry,
)


@dataclass(frozen=True)
class RetentionOp:
    """Wire frame for a reverse retention request from owner to scheduler."""

    op_id: str
    operation: str
    allocation: BlockAllocation


@dataclass(frozen=True)
class RetentionResult:
    """Wire frame for the scheduler's answer to a ``RetentionOp``."""

    op_id: str
    ok: bool
    error_type: str = ""
    error_message: str = ""


class _RemoteRetentionGuard:
    """Owner-local guard proxy: each method travels back over the configured transport."""

    def __init__(self, authority: "RemoteRetentionAuthority", allocation: BlockAllocation) -> None:
        self._authority = authority
        self._allocation = allocation
        self._lock = threading.Lock()
        self._released = False

    def validate(self) -> None:
        with self._lock:
            if self._released:
                raise ContractError("Scheduler retention guard has been released")
            self._authority._invoke("validate", self._allocation)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._authority._invoke("release", self._allocation)
            self._released = True

    def __getstate__(self):
        raise TypeError("Remote retention guards are owner-local, not wire values")


class RemoteRetentionAuthority(ABC):
    """Shared guard semantics with an explicit transport operation boundary."""

    def retain(self, allocation: BlockAllocation) -> RetentionGuard:
        self._invoke("retain", allocation)
        return _RemoteRetentionGuard(self, allocation)

    @abstractmethod
    def _invoke(self, operation: str, allocation: BlockAllocation) -> None:
        """Complete one reverse operation or retain ownership on uncertainty."""

    def __getstate__(self):
        raise TypeError("Remote retention authority holds an owner-local transport")


class RetentionDispatcher:
    """Map a ``RetentionOp`` onto the scheduler registry's authoritative path."""

    def __init__(self, registry: SchedulerRetentionRegistry) -> None:
        self._registry = registry
        self._ops = {
            "retain": registry.retain,
            "validate": registry.validate,
            "release": registry.release_guard,
        }

    def dispatch(self, op: RetentionOp) -> RetentionResult:
        try:
            if not isinstance(op, RetentionOp) or not op.op_id:
                raise ContractError("Malformed retention command")
            if op.operation not in self._ops:
                raise ContractError("Unsupported retention operation")
            self._ops[op.operation](op.allocation)
            return RetentionResult(op.op_id, True)
        except Exception as error:
            return RetentionResult(op.op_id, False, type(error).__name__, str(error))

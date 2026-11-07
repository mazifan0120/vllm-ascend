# SPDX-License-Identifier: Apache-2.0
"""Owner-side target lifetime and real stream dependencies for D offload.

The connector supplies receive dependencies directly, or confirms exact native
sync-WRITE receipts before this runtime records the D consumption dependency.
Request end alone never proves completion. The connector still owns remote-DMA
quiescence and scheduler block-free permission.
"""

import math
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    ContractError,
    ReceiveReceipt,
    RequestKey,
    TargetLease,
    copy_target_lease,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import RetentionAuthority


@dataclass(frozen=True)
class ReceivedTargetWrites:
    """Local dependencies for one exact lease, supplied by the transfer owner.

    host_visible.wait(timeout) must return True only after remote Host writes
    are visible (not just source-read completion). indexer_event is the real
    device-write event. Both objects remain local and retain their resources.
    """

    lease: TargetLease
    host_visible: object
    indexer_event: object

    def __post_init__(self) -> None:
        if not callable(getattr(self.host_visible, "wait", None)):
            raise TypeError("A real Host-visible completion fence is required")
        if not callable(getattr(self.indexer_event, "query", None)):
            raise TypeError("A real Indexer device completion event is required")
        object.__setattr__(self, "lease", copy_target_lease(self.lease))

    def __getstate__(self):
        raise TypeError("Receive dependencies must stay in the target owner process")


class NpuTargetStreams:
    """Bind visibility and completion to the actual worker consumption stream."""

    def __init__(self, manager, device_id: int, consumption_stream) -> None:
        if type(device_id) is not int or device_id < 0 or consumption_stream is None:
            raise ValueError("Target runtime needs an explicit NPU and consumption stream")
        if not callable(getattr(manager, "target_runtime_streams", None)):
            raise TypeError("Sparse manager must expose its compute/writeback streams")
        import torch

        stream_device = getattr(getattr(consumption_stream, "device", None), "index", None)
        if stream_device != device_id:
            raise ValueError("Target consumption stream belongs to another NPU")
        self._manager = manager
        self._device_id = device_id
        self._stream = consumption_stream
        self._npu = torch.npu

    @contextmanager
    def consumption(self):
        with self._npu.device(self._device_id), self._npu.stream(self._stream):
            yield

    def install_visibility(self, dependency: ReceivedTargetWrites) -> None:
        if not isinstance(dependency.indexer_event, self._npu.Event):
            raise TypeError("Indexer visibility requires a native NPU event")
        if not dependency.indexer_event.npu_event:
            raise ContractError("Indexer visibility event has not been recorded")
        with self._npu.device(self._device_id):
            self._stream.wait_event(dependency.indexer_event)

    def confirmed_receive_dependency(self, lease: TargetLease) -> ReceivedTargetWrites:
        """Called only after exact entry receipts prove remote sync WRITE completion.

        The Host fence reflects that observed native completion, not a guessed
        request state. Record a native event on the actual D consumption stream
        for future work; no synthetic already-completed device event is used.
        """
        with self._npu.device(self._device_id), self._npu.stream(self._stream):
            indexer_event = self._npu.Event()
            indexer_event.record()
        host_visible = threading.Event()
        host_visible.set()
        return ReceivedTargetWrites(lease, host_visible, indexer_event)

    def record_use(self) -> tuple:
        # Sparse fused writeback can run on a distinct stream. Fence every
        # stream that could have used this request before dropping owner refs.
        streams = [self._stream, *self._manager.target_runtime_streams()]
        events = []
        seen = set()
        with self._npu.device(self._device_id):
            for stream in streams:
                if stream is None or id(stream) in seen:
                    continue
                seen.add(id(stream))
                if getattr(getattr(stream, "device", None), "index", None) != self._device_id:
                    raise RuntimeError("Target manager exposed a stream from another NPU")
                with self._npu.stream(stream):
                    event = self._npu.Event()
                    event.record()
                    events.append(event)
        return tuple(events)

    def wait_quiescent(self, events: Iterable, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._npu.device(self._device_id):
            for event in events:
                while not event.query():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Target compute/writeback is not quiescent; retain allocations")
                    time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
                event.synchronize()


@dataclass
class _TargetUse:
    lease: TargetLease
    resources: object
    retention: object
    dependency: ReceivedTargetWrites | None = None
    visible: bool = False
    preparing: bool = False
    closing: bool = False
    releasing: bool = False
    users: int = 0
    events: list = field(default_factory=list)
    needs_recovery_fence: bool = False


class OwnerTargetRuntime:
    """Concrete retain/visibility/release implementation over registered memory.

    resources is the live RegisteredTargetResources, not a second allocator.
    RetentionAuthority executes in the scheduler or a real transport to it.
    All callbacks and waits execute outside the state lock.
    """

    def __init__(self, resources, retention: RetentionAuthority, streams, *, timeout: float) -> None:
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Target runtime needs a positive bounded wait timeout")
        for owner, methods in (
            (resources, ("retain", "validate")),
            (retention, ("retain",)),
            (streams, ("consumption", "install_visibility", "record_use", "wait_quiescent")),
        ):
            if any(not callable(getattr(owner, name, None)) for name in methods):
                raise TypeError("Target runtime requires real resource, retention and stream owners")
        self._resources = resources
        self._retention = retention
        self._streams = streams
        self._timeout = float(timeout)
        self._lock = threading.RLock()
        self._states: dict[RequestKey, _TargetUse] = {}
        self._acquiring: set[RequestKey] = set()
        self._quarantine: list = []
        self._retrying_rollbacks = False

    def _state(self, lease: TargetLease) -> _TargetUse:
        state = self._states.get(lease.request)
        if state is None or state.lease != lease:
            raise ContractError("Unknown or changed owner target lease")
        return state

    def retain(self, lease: TargetLease) -> None:
        with self._lock:
            if self._quarantine or self._retrying_rollbacks:
                raise RuntimeError("Target retain rollback is incomplete; owner is quarantined")
            if lease.request in self._states or lease.request in self._acquiring:
                raise ContractError("Target already has an owner operation")
            self._acquiring.add(lease.request)
        guard = None
        resource_pin = None
        try:
            self._resources.validate(lease)
            guard = self._retention.retain(lease.allocation)
            if any(not callable(getattr(guard, name, None)) for name in ("validate", "release")):
                raise TypeError("Retention authority did not return a live guard")
            guard.validate()
            resource_pin = self._resources.retain(lease)
            if not callable(getattr(resource_pin, "release", None)):
                raise TypeError("Target resource owner did not return a real pin")
            with self._lock:
                self._states[lease.request] = _TargetUse(copy_target_lease(lease), resource_pin, guard)
        except Exception:
            try:
                if resource_pin is not None:
                    resource_pin.release()
                if guard is not None:
                    guard.release()
            except Exception:
                with self._lock:
                    self._quarantine.append((resource_pin, guard))
                raise RuntimeError("Target retain failed and rollback remains quarantined") from None
            raise
        finally:
            with self._lock:
                self._acquiring.discard(lease.request)

    def retry_failed_retains(self) -> None:
        """Retry only unpublished acquisitions; never forge transfer quiescence."""
        with self._lock:
            if self._retrying_rollbacks:
                raise ContractError("Target retain rollback retry is already running")
            self._retrying_rollbacks = True
            pending, self._quarantine = self._quarantine, []
        try:
            for index, (pin, guard) in enumerate(pending):
                try:
                    if pin is not None:
                        pin.release()
                    if guard is not None:
                        guard.release()
                except Exception:
                    with self._lock:
                        self._quarantine.extend(pending[index:])
                    raise
        finally:
            with self._lock:
                self._retrying_rollbacks = False

    def publish_received(self, lease: TargetLease, dependency: ReceivedTargetWrites) -> None:
        if dependency.lease != lease:
            raise ContractError("Receive dependencies belong to another target allocation")
        with self._lock:
            state = self._state(lease)
            if state.closing or state.preparing or state.visible or state.dependency is not None:
                raise ContractError("Receive dependencies cannot replace an active target generation")
            state.dependency = dependency

    def confirm_native_received(self, lease: TargetLease, receipt: ReceiveReceipt) -> None:
        """Worker-local bridge from judged native results to real receive fences.

        Not an owner RPC operation. Only the actual receive ledger may supply
        this receipt; a scheduler cannot manufacture a native receive fence.
        """
        if (
            receipt.request != lease.request
            or receipt.target_generation != lease.allocation.allocation_generation
            or receipt.visible
            or not (receipt.main_union_ok and receipt.indexer_ranks_ok and receipt.terminals_ok)
        ):
            raise ContractError("Native receive confirmation requires the exact complete entry receipt")
        with self._lock:
            state = self._state(lease)
            if state.closing or state.preparing:
                raise ContractError("Target receive confirmation races with lifecycle closure")
            if state.dependency is not None:
                return
            state.preparing = True
        try:
            state.retention.validate()
            self._resources.validate(lease)
            dependency = self._streams.confirmed_receive_dependency(lease)
            if type(dependency) is not ReceivedTargetWrites or dependency.lease != lease:
                raise ContractError("Native receive dependency belongs to another target")
            with self._lock:
                state.dependency = dependency
        finally:
            with self._lock:
                state.preparing = False

    def make_visible(self, lease: TargetLease, receipt: ReceiveReceipt) -> None:
        if (
            receipt.request != lease.request
            or receipt.target_generation != lease.allocation.allocation_generation
            or receipt.visible
            or not (receipt.main_union_ok and receipt.indexer_ranks_ok and receipt.terminals_ok)
        ):
            raise ContractError("Target visibility needs the matching complete, not-yet-visible receipt")
        with self._lock:
            state = self._state(lease)
            if state.visible:
                return
            if state.closing or state.preparing or state.dependency is None:
                raise ContractError("Actual receive dependencies must precede target visibility")
            state.preparing = True
            dependency = state.dependency
        try:
            state.retention.validate()
            self._resources.validate(lease)
            if dependency.host_visible.wait(self._timeout) is not True:
                raise TimeoutError("Remote Host writes are not visible; target remains retained")
            state.retention.validate()
            self._streams.install_visibility(dependency)
            with self._lock:
                state.visible = True
        finally:
            with self._lock:
                state.preparing = False

    @contextmanager
    def use(self, lease: TargetLease, *, scheduled_allocation: BlockAllocation | None = None):
        """Cover a real model use, including its queued asynchronous writeback."""
        with self._lock:
            state = self._state(lease)
            if not state.visible or state.closing or state.preparing:
                raise ContractError("Target is not available for decode")
            if scheduled_allocation is not None and (
                type(scheduled_allocation) is not BlockAllocation or scheduled_allocation != state.lease.allocation
            ):
                raise ContractError("Scheduler step allocation does not match the retained target")
            state.users += 1
        try:
            # Production scheduler metadata has already validated the block
            # map for this step. The local live lease and users gate retain
            # both owner resources and the scheduler guard during execution.
            if scheduled_allocation is None:
                state.retention.validate()
            self._resources.validate(lease)
            with self._streams.consumption():
                self._streams.install_visibility(state.dependency)
                yield
        finally:
            try:
                events = self._streams.record_use()
                if not events:
                    raise RuntimeError("Target consumption produced no real completion fences")
                with self._lock:
                    state.events.extend(events)
            except Exception:
                with self._lock:
                    state.needs_recovery_fence = True
                raise
            finally:
                with self._lock:
                    state.users -= 1

    def release(self, lease: TargetLease) -> None:
        with self._lock:
            state = self._state(lease)
            if state.preparing or state.users or state.releasing:
                raise ContractError("Target still has active compute/writeback users or release")
            state.closing = True
            state.releasing = True
            recover = state.needs_recovery_fence
        try:
            if recover:
                events = self._streams.record_use()
                if not events:
                    raise RuntimeError("Cannot prove failed target use is quiescent")
                with self._lock:
                    state.events.extend(events)
                    state.needs_recovery_fence = False
            # Also fence already queued manager work at the release boundary.
            # The connector must stop future submissions before requesting it.
            final_events = self._streams.record_use()
            if not final_events:
                raise RuntimeError("Target owner cannot prove its final stream quiescence")
            events = tuple(state.events) + tuple(final_events)
            if state.dependency is not None:
                events += (state.dependency.indexer_event,)
            self._streams.wait_quiescent(events, self._timeout)
            state.retention.validate()
            state.resources.release()
            state.retention.release()
            with self._lock:
                del self._states[lease.request]
        finally:
            with self._lock:
                state.releasing = False
        # The connector must still prove remote DMA quiescence before allowing
        # SchedulerRetentionRegistry.retire and the scheduler's block free.

    def assert_can_close(self) -> None:
        with self._lock:
            if self._states or self._acquiring or self._quarantine or self._retrying_rollbacks:
                raise ContractError("Target runtime still holds live or quarantined allocations")

    def __getstate__(self):
        raise TypeError("Target runtime belongs to its worker owner process")

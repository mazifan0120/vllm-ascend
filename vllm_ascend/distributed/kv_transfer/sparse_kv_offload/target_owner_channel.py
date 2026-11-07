# SPDX-License-Identifier: Apache-2.0
"""Single-process scheduler-worker channel for the target owner IPC.

Co-located deployment (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) keeps the decode
scheduler and the worker owner in one process. The scheduler facade
(``TargetProviderClient``) reaches the worker owner (``TargetOwnerDispatcher``)
through a module-level rendezvous instead of a wire transport. The scheduler
retention registry is shared by reference: the owner runtime's nested
``retain()`` during ``acquire_target`` calls it directly, in the calling thread.

A multi-process deployment needs a real wire transport (zmq/pipe) and a wire
retention client; that is wired by the deployment wrapper and is not guessed
here. The rendezvous, transport and retention authority defined here are the
co-located reference for that follow-up.
"""

import threading

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetOwnerReply,
)

#: Bounded wait for the peer to publish its half of the rendezvous. The
#: scheduler role probes the dispatcher eagerly during admission and must fail
#: closed fast when the worker owner never published, so the default is short;
#: callers that need to outlive a slow worker registration pass a larger value.
DEFAULT_BIND_TIMEOUT_SECONDS = 5.0


class _CoLocatedRendezvous:
    """Single-slot pairing of the scheduler client and the worker owner.

    Either side may publish first; the other side blocks on a bounded wait. The
    rendezvous is not rebindable across owner generations: a closed or bound
    slot must be reset explicitly before reuse.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._dispatcher: TargetOwnerDispatcher | None = None
        self._registry: object | None = None
        self._dispatcher_event = threading.Event()
        self._registry_event = threading.Event()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def bind_dispatcher(self, dispatcher: TargetOwnerDispatcher) -> None:
        with self._lock:
            if self._closed:
                raise ContractError("Target owner channel is already closed")
            if self._dispatcher is not None:
                raise ContractError("Target owner dispatcher is already bound")
            self._dispatcher = dispatcher
            self._dispatcher_event.set()

    def bind_registry(self, registry: object) -> None:
        with self._lock:
            if self._closed:
                raise ContractError("Target owner channel is already closed")
            if self._registry is not None:
                raise ContractError("Scheduler retention authority is already bound")
            self._registry = registry
            self._registry_event.set()

    def dispatcher(self, timeout: float = DEFAULT_BIND_TIMEOUT_SECONDS) -> TargetOwnerDispatcher:
        if not self._dispatcher_event.wait(timeout):
            raise ContractError("Target owner dispatcher was not bound in time")
        with self._lock:
            if self._closed or self._dispatcher is None:
                raise ContractError("Target owner dispatcher is not available")
            return self._dispatcher

    def registry(self, timeout: float = DEFAULT_BIND_TIMEOUT_SECONDS) -> object:
        if not self._registry_event.wait(timeout):
            raise ContractError("Scheduler retention authority was not bound in time")
        with self._lock:
            if self._closed or self._registry is None:
                raise ContractError("Scheduler retention authority is not available")
            return self._registry

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._dispatcher = None
            self._registry = None
        self._dispatcher_event.set()
        self._registry_event.set()

    def reset(self) -> None:
        """Return the slot to its unbound, open state for a new owner generation."""
        with self._lock:
            self._dispatcher = None
            self._registry = None
            self._closed = False
        self._dispatcher_event.clear()
        self._registry_event.clear()


_active_lock = threading.Lock()
_active: _CoLocatedRendezvous | None = None


def active_channel() -> _CoLocatedRendezvous:
    """Module-level single slot shared by the scheduler and worker roles."""
    global _active
    with _active_lock:
        if _active is None or _active.closed:
            _active = _CoLocatedRendezvous()
        return _active


def reset_active_channel() -> None:
    """Drop the module-level slot. Tests call this between cases."""
    global _active
    with _active_lock:
        if _active is not None:
            _active.close()
        _active = None


class LocalOwnerTransport:
    """Scheduler-side transport: dispatch the call on the bound owner.

    The call runs synchronously in the calling thread. Nested reverse calls
    from the owner runtime (retention ``retain`` during ``acquire_target``)
    execute on the same thread and the same process, so no re-entrancy guard is
    needed beyond the owner's own locks.
    """

    def __init__(self, channel: _CoLocatedRendezvous | None = None) -> None:
        self._channel = channel if channel is not None else active_channel()

    def call(self, call: TargetOwnerCall) -> TargetOwnerReply:
        dispatcher = self._channel.dispatcher()
        reply = dispatcher.dispatch(call)
        if not isinstance(reply, TargetOwnerReply):
            raise ContractError("Target owner dispatcher returned a malformed reply")
        return reply


def bind_worker_dispatcher(facade) -> TargetOwnerDispatcher:
    """Build the dispatcher from a bound owner facade and publish it."""
    dispatcher = TargetOwnerDispatcher(facade)
    active_channel().bind_dispatcher(dispatcher)
    return dispatcher


def bind_scheduler_retention_authority(registry: object) -> None:
    """Publish the scheduler retention registry for the worker owner to consume."""
    active_channel().bind_registry(registry)


def scheduler_retention_authority(timeout: float = DEFAULT_BIND_TIMEOUT_SECONDS) -> object:
    """Fetch the scheduler retention authority shared by reference.

    The co-located owner passes this directly to ``configure_target_owner``: the
    registry already implements the ``retain(allocation) -> RetentionGuard``
    protocol and its guards call back into the scheduler process synchronously.
    """
    return active_channel().registry(timeout)

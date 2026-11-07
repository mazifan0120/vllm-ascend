# SPDX-License-Identifier: Apache-2.0
"""Multi-process wire transport for the target owner IPC.

The co-located channel (``target_owner_channel``) shares the scheduler retention
registry by reference. A multi-process deployment cannot share that reference:
the decode scheduler and the worker owner live in separate processes. This
module provides the wire replacement so the scheduler facade
(``TargetProviderClient``) reaches the worker owner over a pipe, while the
owner runtime's nested ``retain()`` during ``acquire_target`` travels back to
the scheduler over a second pipe.

Two directions, two pipes:

* forward call pipe: scheduler -> worker (``TargetOwnerCall``),
  worker -> scheduler (``TargetOwnerReply``).
* reverse retain pipe: worker -> scheduler (``RetentionOp``),
  scheduler -> worker (``RetentionResult``).

Only typed catalog/lease/allocation values cross the boundary: they are frozen
dataclasses already whitelisted by ``_assert_wire_value``. No local runtime
object, tensor or event is ever pickled. The deployment wrapper creates the
pipes, binds each side and starts the server threads; nothing here guesses a
transport or fabricates an authority.
"""

import threading
from dataclasses import dataclass
from uuid import uuid4

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    ContractError,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetOwnerReply,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import (
    RetentionGuard,
    SchedulerRetentionRegistry,
)

#: Forward call wait. Admission and release are serialized, so this only bounds
#: an unresponsive owner; the receipt watchdog bounds the higher-level flow.
DEFAULT_CALL_TIMEOUT_SECONDS = 300.0

#: Reverse retain wait. ``retain`` runs nested inside ``acquire_target`` and
#: must fail closed fast when the scheduler retention server is stuck, rather
#: than holding the owner's allocation path.
DEFAULT_RETAIN_TIMEOUT_SECONDS = 10.0

#: Poll interval for the server loops so a stop event is observed promptly
#: without busy-waiting.
_POLL_INTERVAL_SECONDS = 0.1

#: Sentinel shutdown message. Neither ``TargetOwnerCall`` nor ``RetentionOp``
#: is ``None``, so it cannot be confused with a real frame.
_STOP = None


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


def _poll(conn, timeout: float) -> bool:
    """``multiprocessing.Connection.poll`` with a fallback for duck-typed ends."""
    poll = getattr(conn, "poll", None)
    if callable(poll):
        return poll(timeout)
    raise TypeError("wire transport requires a pollable connection end")


class PipeOwnerTransport:
    """Scheduler-side forward transport: send a call, wait for its reply.

    Holds the parent end of the forward call pipe. Calls are serialized by a
    lock so the call_id/reply pairing on the pipe stays unambiguous even when
    the scheduler admits several requests concurrently.
    """

    def __init__(self, call_conn, *, timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS) -> None:
        self._call = call_conn
        self._timeout = timeout
        self._lock = threading.Lock()

    def call(self, call: TargetOwnerCall) -> TargetOwnerReply:
        with self._lock:
            try:
                self._call.send(call)
                if not _poll(self._call, self._timeout):
                    raise ContractError("Target owner call timed out waiting for a reply")
                reply = self._call.recv()
            except (EOFError, BrokenPipeError, OSError) as error:
                raise ContractError(f"Target owner pipe closed: {error}") from None
        if not isinstance(reply, TargetOwnerReply):
            raise ContractError("Target owner returned a malformed reply")
        return reply

    def stop(self) -> None:
        """Signal the worker dispatch loop to exit and wait for its ack."""
        with self._lock:
            self._call.send(_STOP)
            if _poll(self._call, self._timeout):
                self._call.recv()


class _WireRetentionGuard:
    """Owner-local guard proxy: each method travels back over the retain pipe."""

    def __init__(self, authority: "WireRetentionAuthority", allocation: BlockAllocation) -> None:
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
        raise TypeError("Wire retention guards are owner-local, not wire values")


class WireRetentionAuthority:
    """Owner-side ``RetentionAuthority`` backed by the reverse retain pipe.

    The owner runtime calls ``retain(allocation)`` nested inside
    ``acquire_target``. In a co-located deployment that hits the scheduler
    registry in-thread; here it serializes to a ``RetentionOp`` and waits for
    the scheduler's ``RetentionResult``. The returned guard mirrors the local
    registry guard's validate/release over the same pipe.
    """

    def __init__(self, retain_conn, *, timeout: float = DEFAULT_RETAIN_TIMEOUT_SECONDS) -> None:
        self._retain = retain_conn
        self._timeout = timeout
        self._lock = threading.Lock()

    def retain(self, allocation: BlockAllocation) -> RetentionGuard:
        self._invoke("retain", allocation)
        return _WireRetentionGuard(self, allocation)

    def _invoke(self, operation: str, allocation: BlockAllocation) -> None:
        op_id = uuid4().hex
        op = RetentionOp(op_id, operation, allocation)
        with self._lock:
            try:
                self._retain.send(op)
                if not _poll(self._retain, self._timeout):
                    raise ContractError("Scheduler retention authority timed out")
                result = self._retain.recv()
            except (EOFError, BrokenPipeError, OSError) as error:
                raise ContractError(f"Retention authority pipe closed: {error}") from None
        if not isinstance(result, RetentionResult) or result.op_id != op_id:
            raise ContractError("Retention authority returned a mismatched result")
        if not result.ok:
            error = ContractError if result.error_type == "ContractError" else RuntimeError
            raise error(f"Retention {result.error_type}: {result.error_message}")

    def stop(self) -> None:
        """Signal the scheduler retain loop to exit and wait for its ack."""
        with self._lock:
            self._retain.send(_STOP)
            if _poll(self._retain, self._timeout):
                self._retain.recv()

    def __getstate__(self):
        raise TypeError("Wire retention authority holds a local pipe endpoint")


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


def serve_retention_ops(conn, registry: SchedulerRetentionRegistry, stop_event: threading.Event | None = None) -> None:
    """Scheduler-side server loop for the reverse retain pipe.

    Runs in the scheduler process alongside the forward call path. Each
    ``RetentionOp`` is dispatched on the authoritative registry and answered
    with a ``RetentionResult``. ``None`` shuts the loop down. A closed peer
    (EOFError/BrokenPipeError/OSError) ends the loop cleanly rather than
    surfacing as thread noise.
    """
    dispatcher = RetentionDispatcher(registry)
    while True:
        try:
            if not _poll(conn, _POLL_INTERVAL_SECONDS):
                if stop_event is not None and stop_event.is_set():
                    break
                continue
            msg = conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            break
        if msg is _STOP:
            conn.send(_STOP)
            break
        if not isinstance(msg, RetentionOp):
            continue
        conn.send(dispatcher.dispatch(msg))


def serve_owner_dispatch(conn, dispatcher: TargetOwnerDispatcher, stop_event: threading.Event | None = None) -> None:
    """Worker-side server loop for the forward call pipe.

    Runs in the owner process. Each ``TargetOwnerCall`` is dispatched on the
    owner facade; the dispatch may itself nest a reverse ``retain`` back to the
    scheduler over the retain pipe, which the scheduler's retention server
    answers on a different thread. ``None`` or a closed peer ends the loop.
    """
    while True:
        try:
            if not _poll(conn, _POLL_INTERVAL_SECONDS):
                if stop_event is not None and stop_event.is_set():
                    break
                continue
            msg = conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            break
        if msg is _STOP:
            conn.send(_STOP)
            break
        if not isinstance(msg, TargetOwnerCall):
            continue
        conn.send(dispatcher.dispatch(msg))


class PipeOwnerChannel:
    """Reference wiring for a parent-child pipe owner channel.

    The scheduler process owns the retention registry; the worker process owns
    the target owner facade. ``for_scheduler`` binds the forward transport and
    starts the reverse retain server; ``for_worker`` binds the wire retention
    authority and starts the forward dispatch server. A deployment wrapper
    creates the two pipes, hands each end to the matching side, and calls these
    helpers exactly once per process.
    """

    @staticmethod
    def for_scheduler(
        connector,
        kv_cache_manager,
        *,
        call_conn,
        retain_conn,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        retain_timeout: float = DEFAULT_RETAIN_TIMEOUT_SECONDS,
    ):
        """Bind the scheduler facade and start the reverse retain server.

        ``connector.bind_scheduler_retention`` still publishes the registry for
        co-located fallback; the wire transport is bound through the scheduler
        facade. Returns ``(stop_event, thread)`` so the caller can drain the
        reverse retain loop on shutdown.
        """
        connector.bind_scheduler_retention(kv_cache_manager)
        transport = PipeOwnerTransport(call_conn, timeout=call_timeout)
        connector.bind_target_owner_transport(transport)
        stop_event = threading.Event()
        thread = threading.Thread(
            target=serve_retention_ops,
            args=(retain_conn, connector._retention_registry, stop_event),
            name="target-owner-retain-server",
            daemon=True,
        )
        thread.start()
        return stop_event, thread

    @staticmethod
    def for_worker(
        connector,
        manager,
        *,
        call_conn,
        retain_conn,
        consumption_stream,
        timeout: float,
        te=None,
        call_timeout: float = DEFAULT_CALL_TIMEOUT_SECONDS,
        retain_timeout: float = DEFAULT_RETAIN_TIMEOUT_SECONDS,
    ):
        """Bind the worker owner facade and start the forward dispatch server.

        ``configure_target_owner`` receives the wire retention authority so the
        owner runtime's nested ``retain`` travels back to the scheduler. After
        ``register_kv_caches`` publishes the dispatcher, the caller passes it to
        ``serve_owner_dispatch``; this helper returns ``(authority, stop_event,
        thread)`` and leaves dispatcher publication to the connector's normal
        registration flow.
        """
        authority = WireRetentionAuthority(retain_conn, timeout=retain_timeout)
        connector.configure_target_owner(
            manager,
            retention=authority,
            consumption_stream=consumption_stream,
            timeout=timeout,
            te=te,
        )
        stop_event = threading.Event()
        thread = threading.Thread(
            target=_serve_worker_dispatch_after_registration,
            args=(connector, call_conn, stop_event),
            name="target-owner-call-server",
            daemon=True,
        )
        thread.start()
        return authority, stop_event, thread


def _serve_worker_dispatch_after_registration(connector, call_conn, stop_event) -> None:
    """Wait for ``register_kv_caches`` to publish the dispatcher, then serve.

    The dispatcher is built from the bound owner facade during
    ``register_kv_caches``; the deployment wrapper starts this thread before
    that call returns. It polls the facade until the dispatcher is published.
    """
    deadline = None
    while True:
        dispatcher = getattr(connector, "_target_owner_dispatcher", None)
        if dispatcher is not None:
            serve_owner_dispatch(call_conn, dispatcher, stop_event)
            return
        if stop_event.is_set():
            return
        if deadline is None:
            deadline = True
        threading.Event().wait(_POLL_INTERVAL_SECONDS)

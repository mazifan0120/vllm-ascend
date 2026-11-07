# SPDX-License-Identifier: Apache-2.0
"""Bidirectional Pipe reference transport for protocol/fault tests only."""

import threading
from uuid import uuid4

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import BlockAllocation, ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetOwnerReply,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SchedulerRetentionRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention_protocol import (
    RemoteRetentionAuthority,
    RetentionDispatcher,
    RetentionOp,
    RetentionResult,
)

DEFAULT_CALL_TIMEOUT_SECONDS = 300.0
DEFAULT_RETAIN_TIMEOUT_SECONDS = 10.0
_POLL_INTERVAL_SECONDS = 0.1
_STOP = None


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


class WireRetentionAuthority(RemoteRetentionAuthority):
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

# SPDX-License-Identifier: Apache-2.0
"""Layerwise control messages over the existing rank-local Mooncake side channel."""

import logging
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Event, get_ident
from time import monotonic

import msgspec
import zmq

from vllm_ascend.core.layerwise_offload_scheduler import DecodeWorkerEvent, DecodeWorkerLink

from ..kv_pool.ascend_store.prefill_reuse import SlotBarrier
from .layerwise_offload_plan import (
    CatalogRef,
    PeerAddress,
    RegisteredLayout,
    TargetAllocation,
    TargetId,
    Write,
    WriteResult,
)
from .layerwise_offload_sender import LayerwiseSender, NativeResult, PrefillChunk
from .layerwise_offload_wire import (
    decode_addresses,
    decode_catalog,
    decode_worker_status,
    encode_allocation,
    encode_catalog_query,
    encode_results,
    encode_terminal,
    encode_worker_query,
    encode_writer_query,
)

logger = logging.getLogger(__name__)


class SideChannelClient:
    """One socket per rank, confined to the connector's owning control thread."""

    def __init__(self, endpoints: tuple[str, ...], timeout_ms: int):
        if not endpoints or timeout_ms <= 0:
            raise ValueError("Layerwise side channel needs endpoints and a positive timeout")
        self._endpoints = endpoints
        self._timeout_ms = timeout_ms
        self._thread: int | None = None
        self._sockets: dict[int, zmq.Socket] = {}

    def request(self, rank: int, frame: tuple) -> tuple:
        thread = get_ident()
        if self._thread is None:
            self._thread = thread
        if self._thread != thread:
            raise RuntimeError("Layerwise side-channel socket crossed control threads")
        sock = self._sockets.get(rank)
        if sock is None:
            sock = zmq.Context.instance().socket(zmq.REQ)
            sock.linger = 0
            sock.rcvtimeo = self._timeout_ms
            sock.sndtimeo = self._timeout_ms
            sock.connect(self._endpoints[rank])
            self._sockets[rank] = sock
        try:
            sock.send(msgspec.msgpack.encode(frame))
            reply = msgspec.msgpack.decode(sock.recv(), type=tuple)
        except zmq.Again as exc:
            # An unanswered REQ cannot send another request. A retry gets a fresh socket.
            self._sockets.pop(rank).close()
            raise TimeoutError(f"No layerwise reply from rank {rank}") from exc
        if reply and reply[0] == "error":
            raise ValueError(f"Layerwise rank {rank} rejected {frame[0]}")
        return reply

    def close(self) -> None:
        if self._thread is not None and self._thread != get_ident():
            raise RuntimeError("Layerwise side channel must close on its owning thread")
        for sock in self._sockets.values():
            sock.close()
        self._sockets.clear()
        self._thread = None


class DecodeControl:
    """Fan out Prepare/Close, then poll worker state independently of model execution."""

    def __init__(self, client: SideChannelClient, post: Callable[[DecodeWorkerEvent], None], timeout_s: float):
        if timeout_s <= 0:
            raise ValueError("Layerwise target timeout must be positive")
        self._client = client
        self._post = post
        self._timeout = timeout_s
        self._commands: SimpleQueue[tuple[TargetId, CatalogRef, tuple, float]] = SimpleQueue()
        self._pending: dict[tuple[TargetId, CatalogRef], tuple[tuple, float]] = {}

    def link(self, publish: Callable) -> DecodeWorkerLink:
        return DecodeWorkerLink(self.prepare, self.close_target, publish)

    def prepare(self, allocation: TargetAllocation, refs: tuple[CatalogRef, ...]) -> None:
        frame = encode_allocation(allocation, refs)
        deadline = monotonic() + self._timeout
        for ref in refs:
            self._commands.put((allocation.target, ref, frame, deadline))

    def close_target(self, target: TargetId, refs: tuple[CatalogRef, ...]) -> None:
        for ref in refs:
            self._commands.put((target, ref, encode_worker_query("close-target", target, ref), float("inf")))

    def step(self) -> None:
        """Idempotent retries. A missing close reply leaves the scheduler lease held."""
        while True:
            try:
                target, ref, frame, deadline = self._commands.get_nowait()
            except Empty:
                break
            self._pending[target, ref] = frame, deadline
        unreachable = set()
        for (target, ref), (frame, deadline) in list(self._pending.items()):
            try:
                if monotonic() >= deadline:
                    raise ValueError("Layerwise target preparation or receipt timed out")
                if ref.rank in unreachable:
                    continue
                state, _ = decode_worker_status(self._client.request(ref.rank, frame), target, ref)
                if frame[0] == "close-target":
                    if state != "closed":
                        continue
                elif state == "prepared" and frame[0] == "target":
                    self._post(DecodeWorkerEvent("prepared", target, ref))
                    self._pending[target, ref] = encode_worker_query("poll-target", target, ref), deadline
                    continue
                elif state == "waiting" and frame[0] == "poll-target":
                    continue
                elif state not in ("ready", "failed") or frame[0] != "poll-target":
                    raise ValueError("Unexpected worker state for layerwise command")
            except TimeoutError:
                unreachable.add(ref.rank)
                continue
            except (ValueError, msgspec.DecodeError, zmq.ZMQError) as exc:
                if frame[0] == "close-target":
                    continue
                logger.warning("Layerwise target %s failed at rank %s: %s", target, ref.rank, exc)
                state = "failed"
            self._post(DecodeWorkerEvent(state, target, ref))
            del self._pending[target, ref]

    def run(self, stop: Event, interval_s: float = 0.02) -> None:
        try:
            while not stop.is_set():
                self.step()
                stop.wait(interval_s)
        finally:
            self._client.close()


class WriterControl:
    """Uses the writer's D owner as its cancellation mailbox; no reverse server on P."""

    def __init__(self, client: SideChannelClient, refs: tuple[CatalogRef, ...]):
        self._client = client
        self._refs = refs
        self._acknowledged_uncertain: set[tuple[TargetId, int]] = set()

    def read_addresses(self, owner: int, target: TargetId, writer: int, ref: CatalogRef) -> tuple[PeerAddress, ...]:
        frame = encode_writer_query("get-addresses", target, writer, ref)
        return decode_addresses(self._client.request(owner, frame), target)

    def report(self, rank: int, target: TargetId, frames: tuple[WriteResult, ...]) -> None:
        self._send(rank, encode_results(target, frames))

    def progress(
        self, sender: LayerwiseSender, target: TargetId, *, finished: bool = False, cancel: bool = False
    ) -> bool:
        key = target, sender.rank
        acknowledged = key in self._acknowledged_uncertain
        state = "closing" if cancel else None
        if not cancel and not acknowledged:
            ref = self._refs[sender.rank]
            try:
                frame = encode_writer_query("poll-writer", target, sender.rank, ref)
                state, _ = decode_worker_status(self._client.request(ref.rank, frame), target, ref)
                if state not in ("waiting", "closing"):
                    raise ValueError("Unexpected layerwise writer state")
            except (TimeoutError, zmq.ZMQError):
                return False
            except (ValueError, msgspec.DecodeError):
                state = "closing"
        if state == "waiting":
            sender.flush_proofs(target)
        terminal = sender.finish(target, cancel=state == "closing", end_of_input=finished)
        if terminal is None or (acknowledged and not terminal.quiesced):
            return False
        if not self.send_terminal(sender.rank, target, terminal):
            return False
        if terminal.quiesced:
            self._acknowledged_uncertain.discard(key)
        else:
            self._acknowledged_uncertain.add(key)
        return terminal.quiesced

    def send_terminal(self, rank: int, target: TargetId, terminal: NativeResult) -> bool:
        frame = encode_terminal(target, rank, terminal.error, terminal.quiesced)
        try:
            for worker in self._refs:
                self._send(worker.rank, frame)
        except (TimeoutError, ValueError, msgspec.DecodeError, zmq.ZMQError):
            return False  # Retry the same terminal; no worker is inferred to be closed.
        return True

    def close(self) -> None:
        self._client.close()

    def _send(self, rank: int, frame: tuple) -> None:
        if self._client.request(rank, frame) != ("ack",):
            raise ValueError("Invalid layerwise proof acknowledgement")


@dataclass
class _WriterTask:
    sender: LayerwiseSender | None
    control: WriterControl
    finished: bool = False
    cancel: bool = False
    refusal: NativeResult | None = None


class PrefillControl:
    """One control thread owns the sockets for all D teams addressed by this P rank."""

    def __init__(self, rank: int, tp_size: int, slots: SlotBarrier, write: Callable[[Write], NativeResult]):
        self._rank, self._tp_size = rank, tp_size
        self._slots, self._write = slots, write
        self._sessions = {}
        self._active: dict[TargetId, _WriterTask] = {}
        self._commands: SimpleQueue[tuple[Callable, Future | None]] = SimpleQueue()

    def prepare(self, chunks: tuple[PrefillChunk, ...]) -> tuple[tuple[LayerwiseSender, PrefillChunk], ...]:
        if not chunks or all(self._rank >= len(chunk.refs) for chunk in chunks):
            return ()
        answer = Future()
        self._commands.put((lambda: self._attach(chunks), answer))
        return answer.result()

    def _attach(self, chunks):
        active = []
        for chunk in chunks:
            if self._rank >= len(chunk.refs):
                continue
            target = chunk.allocation.target
            previous = self._active.get(target)
            if previous is not None and previous.refusal is not None:
                continue
            key = (chunk.refs, chunk.endpoints, chunk.parallel_main)
            if key not in self._sessions:
                client = SideChannelClient(chunk.endpoints, 1000)
                try:
                    catalogs = tuple(
                        decode_catalog(client.request(ref.rank, encode_catalog_query(ref))) for ref in chunk.refs
                    )
                    layout = RegisteredLayout(catalogs, self._tp_size, parallel_main=chunk.parallel_main)
                    if layout.refs != chunk.refs:
                        raise ValueError("D worker returned a different registered catalog")
                except (TimeoutError, ValueError, msgspec.DecodeError, zmq.ZMQError) as error:
                    reason = str(error) or type(error).__name__
                    logger.warning("P rank %d could not read layerwise catalog for %s: %s", self._rank, target, reason)
                    if previous is None:
                        # Only the receiver can confirm whether it issued an address grant.
                        self._active[target] = _WriterTask(
                            None, WriterControl(client, chunk.refs), refusal=NativeResult(reason, False)
                        )
                    else:
                        client.close()
                        previous.cancel = True
                    continue
                except Exception:
                    client.close()
                    raise
                control = WriterControl(client, layout.refs)
                sender = LayerwiseSender(
                    self._rank, layout, self._slots, control.read_addresses, self._write, control.report
                )
                self._sessions[key] = client, control, sender
            _, control, sender = self._sessions[key]
            try:
                attached = sender.attach(chunk.allocation, chunk.refs)
            except (TimeoutError, ValueError, msgspec.DecodeError, zmq.ZMQError) as error:
                logger.warning("P rank %d could not attach layerwise target %s: %s", self._rank, target, error)
                self._active[target] = _WriterTask(sender, control, cancel=True)
                continue
            if attached:
                self._active.setdefault(target, _WriterTask(sender, control))
                active.append((sender, chunk))
        return tuple(active)

    def finish(self, targets: tuple[TargetId, ...], *, cancel: bool = False) -> None:
        def mark():
            for target in targets:
                if (task := self._active.get(target)) is not None:
                    task.finished = True
                    task.cancel |= cancel

        if targets:
            self._commands.put((mark, None))

    def _run_command(self, timeout: float) -> None:
        try:
            action, answer = self._commands.get(timeout=timeout)
        except Empty:
            return
        if answer is None:
            action()
        else:
            try:
                answer.set_result(action())
            except Exception as error:
                answer.set_exception(error)

    def run(self, stop: Event) -> None:
        try:
            while not stop.is_set():
                self._run_command(0.02)
                for target in tuple(self._active):
                    # Model steps and cancellation need progress while other targets report proofs.
                    if not self._commands.empty():
                        self._run_command(0)
                    task = self._active[target]
                    if task.refusal is not None:
                        if task.control.send_terminal(self._rank, target, task.refusal):
                            task.control.close()
                            del self._active[target]
                        continue
                    assert task.sender is not None
                    if task.control.progress(task.sender, target, finished=task.finished, cancel=task.cancel):
                        task.sender.forget(target)
                        del self._active[target]
        finally:
            for client, _, _ in self._sessions.values():
                client.close()
            for task in self._active.values():
                if task.refusal is not None:
                    task.control.close()

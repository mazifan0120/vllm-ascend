# SPDX-License-Identifier: Apache-2.0
"""P worker submission, native completion and proof emission."""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Lock

from ..kv_pool.ascend_store.prefill_reuse import ReadOutcome, ReadTicket, SlotBarrier, SlotEpoch
from .layerwise_offload_plan import (
    CatalogRef,
    EntryId,
    PeerAddress,
    RegisteredLayout,
    Span,
    TargetAllocation,
    TargetId,
    Write,
    WriteResult,
    plan_writes,
)
from .mooncake_dsa_transfer import DsaCacheLayout


@dataclass(frozen=True)
class NativeResult:
    error: str | None
    quiesced: bool

    def __post_init__(self) -> None:
        if self.error is None and not self.quiesced:
            raise ValueError("Successful native write must stop reading the source")


def mooncake_write_once(engine, write: Write) -> NativeResult:
    # The sync wrapper can retry a failed batch without proving it has stopped.
    batch = engine.batch_transfer_async_write(write.endpoint, write.sources, write.targets, write.lengths)
    if batch == 0:
        return NativeResult(f"Mooncake write submission failed for {write.endpoint}", False)
    result = engine.get_batch_transfer_status([batch])
    if result == 0:
        return NativeResult(None, True)
    return NativeResult(f"Mooncake write completion failed for {write.endpoint}: {result}", False)


@dataclass(frozen=True)
class LayerWrite:
    target: TargetId
    key: tuple[int, Span]
    writes: tuple[Write, ...]
    read: ReadTicket
    wait_ready: Callable[[], None] | None


@dataclass(frozen=True)
class PrefillChunk:
    allocation: TargetAllocation
    refs: tuple[CatalogRef, ...]
    endpoints: tuple[str, ...]
    parallel_main: bool
    blocks: tuple[tuple[int, ...], ...]
    span: Span

    @property
    def finished(self) -> bool:
        return self.span.end == self.allocation.prompt_tokens


@dataclass
class _Transfer:
    allocation: TargetAllocation
    addresses: dict[tuple[int, str], PeerAddress]
    scheduled: set[tuple[int, Span]] = field(default_factory=set)
    pending: dict[tuple[int, Span], LayerWrite] = field(default_factory=dict)
    running: set[tuple[int, Span]] = field(default_factory=set)
    unknown: dict[tuple[int, Span], tuple[ReadTicket, str]] = field(default_factory=dict)
    proofs: deque[tuple[int, tuple[WriteResult, ...]]] = field(default_factory=deque)
    reporting: bool = False
    closing: bool = False
    final: bool = False
    error: str | None = None
    terminal: NativeResult | None = None


class LayerwiseSender:
    def __init__(
        self,
        rank: int,
        layout: RegisteredLayout,
        barrier: SlotBarrier,
        read_addresses: Callable[[int, TargetId, int, CatalogRef], tuple[PeerAddress, ...]],
        write: Callable[[Write], NativeResult],
        report: Callable[[int, TargetId, tuple[WriteResult, ...]], None],
    ) -> None:
        if rank not in layout.writers:
            raise ValueError("Rank is not an active layerwise writer")
        self.rank = rank
        self._layout = layout
        self._barrier = barrier
        self._read_addresses = read_addresses
        self._write = write
        self._report = report
        self._lock = Lock()
        self._transfers: dict[TargetId, _Transfer] = {}
        self._closed: set[TargetId] = set()

    def attach(self, allocation: TargetAllocation, refs: tuple[CatalogRef, ...]) -> bool:
        if refs != self._layout.refs:
            raise ValueError("Published target uses different registered layouts")
        self._layout.validate_allocation(allocation)
        target = allocation.target
        with self._lock:
            if target in self._closed:
                return False
            previous = self._transfers.get(target)
            if previous is not None:
                if previous.allocation != allocation:
                    raise RuntimeError("The admitted target has changed")
                return not (previous.closing or previous.final or previous.error is not None)
        catalog = self._layout.catalogs[self.rank]
        addresses = self._read_addresses(self.rank, target, self.rank, catalog.ref)
        expected = {region.name: region.length for region in catalog.regions}
        if len(addresses) != len(expected) or any(row.catalog != catalog.ref or row.base <= 0 for row in addresses):
            raise ValueError("Address response does not match the registered owner")
        if {row.region: row.length for row in addresses} != expected:
            raise ValueError("Address response changed the admitted region set")
        with self._lock:
            if target in self._closed:
                return False
            previous = self._transfers.setdefault(
                target, _Transfer(allocation, {(self.rank, row.region): row for row in addresses})
            )
            if previous.allocation != allocation:
                raise RuntimeError("The admitted target has changed")
            return not (previous.closing or previous.final or previous.error is not None)

    def reserve_layer(
        self,
        target: TargetId,
        epoch: SlotEpoch,
        span: Span,
        sources: Mapping[EntryId, tuple[DsaCacheLayout, tuple[int, ...]]],
        wait_ready: Callable[[], None] | None,
    ) -> LayerWrite | None:
        key = (epoch.layer, span)
        with self._lock:
            if target in self._closed:
                return None
            transfer = self._transfers[target]
            if transfer.closing or transfer.final or transfer.error is not None:
                return None
            if key in transfer.scheduled:
                return None
        writes = plan_writes(
            self._layout, transfer.allocation, sources, transfer.addresses, self.rank, epoch.layer, span
        )
        with self._lock:
            if transfer.closing or transfer.final or transfer.error is not None:
                return None
            if key in transfer.scheduled:
                return None
            transfer.scheduled.add(key)
            if not writes:
                return None
            ticket = self._barrier.register(epoch, "transfer", (target, key))
            job = LayerWrite(target, key, writes, ticket, wait_ready)
            transfer.pending[key] = job
            return job

    def execute(self, job: LayerWrite) -> None:
        with self._lock:
            transfer = self._transfers[job.target]
            if transfer.pending.get(job.key) is not job or job.key in transfer.running:
                raise RuntimeError("Unregistered or completed layer write")
            if transfer.closing or transfer.error is not None:
                self._barrier.complete(job.read, ReadOutcome(True))
                del transfer.pending[job.key]
                return
            transfer.running.add(job.key)
        results: dict[int, list[WriteResult]] = {}
        error = None
        quiesced = True
        try:
            if job.wait_ready is not None:
                job.wait_ready()
            for write in job.writes:
                with self._lock:
                    if transfer.closing or transfer.error is not None:
                        raise RuntimeError(transfer.error or "Target cancelled")
                quiesced = False
                result = self._write(write)
                quiesced = result.quiesced
                if result.error is not None:
                    raise RuntimeError(result.error)
                for receiver in write.receivers:
                    results.setdefault(receiver, []).extend(write.results)
        except Exception as exc:
            error = str(exc) or type(exc).__name__

        # The native send loop never waits for proof delivery.
        self._barrier.complete(job.read, ReadOutcome(quiesced, error if not quiesced else None))
        with self._lock:
            if error is not None and transfer.error is None:
                transfer.error = error
            if not quiesced:
                assert error is not None
                transfer.unknown[job.key] = (job.read, error)
            if error is None and not transfer.closing and transfer.error is None:
                transfer.proofs.extend((rank, tuple(frames)) for rank, frames in results.items())
            transfer.running.remove(job.key)
            del transfer.pending[job.key]

    def flush_proofs(self, target: TargetId) -> None:
        """Called by the control loop; native work can continue while peers reply."""
        with self._lock:
            transfer = self._transfers[target]
            if transfer.reporting:
                return
            transfer.reporting = True
        try:
            while True:
                with self._lock:
                    if transfer.closing or transfer.error is not None or not transfer.proofs:
                        return
                    receiver, frames = transfer.proofs.popleft()
                try:
                    self._report(receiver, target, frames)
                except TimeoutError:
                    # The receiver may already have applied these ranges.
                    with self._lock:
                        if not transfer.closing and transfer.error is None:
                            transfer.proofs.appendleft((receiver, frames))
                    return
        except Exception as exc:
            with self._lock:
                transfer.error = transfer.error or str(exc) or type(exc).__name__
        finally:
            with self._lock:
                transfer.reporting = False

    def finish(self, target: TargetId, *, cancel: bool = False, end_of_input: bool = True) -> NativeResult | None:
        """Return a terminal when no queued job can start another native write."""
        with self._lock:
            transfer = self._transfers.get(target)
            if transfer is None:
                if not cancel:
                    raise KeyError(target)
                self._closed.add(target)
                return NativeResult("Target cancelled", True)
            if transfer.terminal is not None:
                return transfer.terminal
            if not (end_of_input or cancel or transfer.final or transfer.error is not None):
                return None
            transfer.final = True
            if cancel:
                transfer.closing = True
            if transfer.closing or transfer.error is not None:
                transfer.proofs.clear()
            if transfer.pending or transfer.proofs or transfer.reporting:
                return None
            error = transfer.error or ("Target cancelled" if transfer.closing else None)
            transfer.terminal = NativeResult(error, not transfer.unknown)
            return transfer.terminal

    def confirm_native_quiescence(self, target: TargetId) -> None:
        """Update a failed transfer only after the backend independently proves it is idle."""
        with self._lock:
            transfer = self._transfers[target]
            if not transfer.final or transfer.pending:
                raise RuntimeError("Cannot fence a writer that may still issue native writes")
            for ticket, error in transfer.unknown.values():
                self._barrier.complete(ticket, ReadOutcome(True, error))
            transfer.unknown.clear()
            if transfer.terminal is not None:
                transfer.terminal = NativeResult(transfer.terminal.error, True)

    def forget(self, target: TargetId) -> None:
        """Drop a finished transfer after its terminal has been acknowledged."""
        with self._lock:
            transfer = self._transfers.get(target)
            if transfer is None and target in self._closed:
                return
            if transfer is None:
                raise KeyError(target)
            if (
                transfer.terminal is None
                or transfer.pending
                or transfer.unknown
                or transfer.proofs
                or transfer.reporting
            ):
                raise RuntimeError("Cannot forget a transfer with potentially live DMA")
            self._closed.add(target)
            del self._transfers[target]

# SPDX-License-Identifier: Apache-2.0
"""Worker-local read barrier for layerwise prefill."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from threading import Condition
from time import perf_counter
from typing import Literal

Reader = Literal["attention", "store", "transfer", "restore"]


@dataclass(frozen=True)
class SlotEpoch:
    slot: int
    generation: int
    layer: int


@dataclass(frozen=True)
class ReadTicket:
    epoch: SlotEpoch
    reader: Reader
    operation: Hashable


@dataclass(frozen=True)
class ReadOutcome:
    quiesced: bool
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.quiesced and self.error is None:
            raise ValueError("A successful read must prove source quiescence")


@dataclass
class _Occupant:
    epoch: SlotEpoch
    readers: dict[ReadTicket, ReadOutcome | None] = field(default_factory=dict)
    sealed: bool = False
    error: str | None = None
    attention_done: Callable[[], None] | None = None

    @property
    def idle(self) -> bool:
        return self.sealed and all(result is not None and result.quiesced for result in self.readers.values())


class SlotBarrier:
    """One writer opens each physical slot; attention, Store and P/D retire reads independently."""

    def __init__(self, slots: tuple[tuple[int, ...], ...]):
        self._slots = {layer: slot for slot, layers in enumerate(slots) for layer in layers}
        self._occupants: dict[int, _Occupant] = {}
        self._changed = Condition()
        self._aborted: str | None = None

    def begin_write(self, layer: int, timeout: float | None = None) -> SlotEpoch:
        slot = self._slots[layer]
        with self._changed:
            previous = self._occupants.get(slot)
            attention_done = None
            if previous is not None:
                if not self._changed.wait_for(lambda: previous.sealed or previous.error or self._aborted, timeout):
                    raise TimeoutError(f"Readers of {previous.epoch} are still being registered")
                self._check_error(previous)
                attention_done = previous.attention_done
        if previous is not None and attention_done is not None:
            ticket = ReadTicket(previous.epoch, "attention", previous.epoch.layer)
            try:
                attention_done()
            except Exception as exc:
                self.complete(ticket, ReadOutcome(False, str(exc) or type(exc).__name__))
            else:
                self.complete(ticket, ReadOutcome(True))
        with self._changed:
            if previous is not None:
                if attention_done is not None:
                    previous.attention_done = None
                if not self._changed.wait_for(lambda: previous.idle or previous.error or self._aborted, timeout):
                    raise TimeoutError(f"Readers of {previous.epoch} have not stopped")
            self._check_error(previous)
            if self._occupants.get(slot) is not previous:
                raise RuntimeError("Concurrent attempt to open a Main slot")
            epoch = SlotEpoch(slot, 1 if previous is None else previous.epoch.generation + 1, layer)
            self._occupants[slot] = _Occupant(epoch)
            return epoch

    def epoch(self, layer: int) -> SlotEpoch:
        with self._changed:
            occupant = self._occupants.get(self._slots[layer])
            self._check_error(occupant)
            if occupant is None or occupant.epoch.layer != layer:
                raise RuntimeError(f"Store has not opened the Main slot for layer {layer}")
            return occupant.epoch

    def register(self, epoch: SlotEpoch, reader: Reader, operation: Hashable) -> ReadTicket:
        ticket = ReadTicket(epoch, reader, operation)
        with self._changed:
            occupant = self._require(epoch)
            if ticket in occupant.readers:
                return ticket
            if occupant.sealed or occupant.error is not None:
                raise RuntimeError("Cannot add a reader to a sealed or failed Main slot")
            occupant.readers[ticket] = None
            return ticket

    def seal(self, epoch: SlotEpoch, attention_done: Callable[[], None] | None = None) -> None:
        with self._changed:
            occupant = self._require(epoch)
            if attention_done is not None:
                self.register(epoch, "attention", epoch.layer)
                occupant.attention_done = attention_done
            occupant.sealed = True
            self._changed.notify_all()

    def complete(self, ticket: ReadTicket, result: ReadOutcome) -> None:
        epoch = ticket.epoch
        with self._changed:
            current = self._occupants.get(epoch.slot)
            if current is not None and current.epoch.generation > epoch.generation:
                return  # A prior generation could only have retired after all successful reads stopped.
            occupant = self._require(epoch)
            previous = occupant.readers[ticket]
            if previous is not None and (previous.error != result.error or (previous.quiesced and not result.quiesced)):
                raise RuntimeError("Conflicting outcome for the same Main slot reader")
            occupant.readers[ticket] = result
            if result.error is not None and occupant.error is None:
                occupant.error = result.error
            self._changed.notify_all()

    def fail(self, epoch: SlotEpoch, error: str) -> None:
        with self._changed:
            occupant = self._require(epoch)
            if occupant.error is None:
                occupant.error = error
            occupant.sealed = True
            self._changed.notify_all()

    def may_dispose(self) -> bool:
        """For shutdown only. A failed epoch may be disposed after all readers stop."""
        with self._changed:
            return all(occupant.idle for occupant in self._occupants.values())

    def abort(self, error: str) -> None:
        with self._changed:
            self._aborted = error
            self._changed.notify_all()

    def raise_if_failed(self) -> None:
        with self._changed:
            self._check_error(None)
            for occupant in self._occupants.values():
                self._check_error(occupant)

    def _check_error(self, occupant: _Occupant | None) -> None:
        if error := self._aborted or (occupant.error if occupant is not None else None):
            raise RuntimeError(error)

    def _require(self, epoch: SlotEpoch) -> _Occupant:
        occupant = self._occupants.get(epoch.slot)
        if occupant is None or occupant.epoch != epoch:
            raise RuntimeError(f"Unknown Main slot epoch {epoch}")
        return occupant


def restore_into_slot(
    barrier: SlotBarrier,
    layer: int,
    restore: Callable[[], ReadOutcome | None] | None,
    timeout: float | None = None,
    *,
    record_operation: Callable[[str, float, int], None] | None = None,
) -> SlotEpoch:
    """The Store receiver opens the Main epoch even when it has no GET rows."""
    start = perf_counter() if record_operation is not None else 0.0
    epoch = barrier.begin_write(layer, timeout)
    if record_operation is not None:
        record_operation("slot_acquire", perf_counter() - start, 0)
    if restore is None:
        return epoch
    ticket = barrier.register(epoch, "restore", layer)
    try:
        result = restore() or ReadOutcome(True)
    except Exception as exc:
        result = ReadOutcome(False, str(exc) or type(exc).__name__)
    barrier.complete(ticket, result)
    if result.error is not None:
        barrier.fail(epoch, result.error)
        raise RuntimeError(result.error)
    return epoch

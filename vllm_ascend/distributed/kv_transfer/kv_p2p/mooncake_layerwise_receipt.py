# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Receipt/Slot ledgers for layerwise PD transfer (pure state machines).

- :class:`ReceiptLedger` (D side): judge readiness by identity's
  four conditions + protocol-error clause. Replaces the old
  "count DONE messages reported by P" logic, which deadlocks or silently
  passes once DONE becomes per-(layer, component, rank, range).
- :class:`SlotLedger` (P side): reuse barrier for shared physical KV slots,
  keyed by *slot id* — the address-derived identity from the baseline
  ``_init_layerwise_reuse_state`` (the planner is the only
  source of the layer→slot mapping; never re-derive by modulo).

Hard rule: pure stdlib. No torch / torch_npu / zmq / vllm / msgspec imports.
"""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

try:  # in-repo import (real deployment and stub-bootstrapped unit tests)
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
        Component,
        ReceiveReceipt,
        RequestKey,
        Status,
        TransferKey,
        TransferResult,
    )
except ModuleNotFoundError:  # direct top-level loading (bare test harnesses)
    from mooncake_layerwise_contract import (  # type: ignore[no-redef]
        Component,
        ReceiveReceipt,
        RequestKey,
        Status,
        TransferKey,
        TransferResult,
    )

logger = logging.getLogger(__name__)


def _ranges_cover(ranges: Iterable[tuple[int, int]], total: int) -> bool:
    """True if the merged half-open ranges cover [0, total). Sort by start,
    keep current end; a start beyond current means a gap."""
    if total <= 0:
        return True
    cur = 0
    for start, end in sorted(ranges):
        if start > cur:
            return False
        if end > cur:
            cur = end
        if cur >= total:
            return True
    return cur >= total


def is_no_compute_layer(computed_tokens: Iterable[int], prompt_tokens: Iterable[int]) -> bool:
    """True when EVERY request on this layer was fully prefix-hit
    (computed >= prompt > 0): nothing was computed here, so the layer has no
    compute reader and is marked ``no_compute_reader`` —
    its send path must never wait on a compute-done event.

    An empty request set (or a degenerate prompt) gives no evidence -> False,
    i.e. treat the layer as compute-present and record the event. Pure stdlib
    so it is testable on any box; the connector calls this from save_kv_layer.
    """
    found_any = False
    for computed, prompt in zip(computed_tokens, prompt_tokens):
        found_any = True
        if prompt <= 0 or computed < prompt:
            return False
    return found_any


class ReceiptLedger:
    """D-side readiness ledger for one request instance.

    Created after ``publish_target`` succeeds — ``expected_ranks`` comes from
    the TargetManifest, which is only published after registration and
    allocation retention are in place.
    """

    def __init__(
        self,
        request: RequestKey,
        expected_ranks: Iterable[int],
        main_backings: Iterable[str],
        indexer_ranks: Iterable[int],
        prompt_len: int,
        required_layers: Iterable[int] = (),
        indexer_backings: Iterable[str] = (),
    ) -> None:
        self.request = request
        self.expected_ranks = frozenset(expected_ranks)
        self.main_backings = tuple(main_backings)
        self.indexer_ranks = frozenset(indexer_ranks)
        # Wire contract identity for INDEXER transfers: the sender does not
        # know D's backing layout in v1, so the D side resolves an empty wire
        # backing_id against this tuple (never a fabricated ID).
        self.indexer_backings = tuple(indexer_backings)
        # Required layer plan: when non-empty, coverage is judged per logical
        # layer, so a layer that never reports cannot hide inside the union.
        self.required_layers = frozenset(int(layer) for layer in required_layers)
        self.prompt_len = int(prompt_len)
        self._entries: dict[TransferKey, _ResultEntry] = {}
        self._terminals: dict[int, Status] = {}
        self._unexpected: set[int] = set()
        self._visible: set[int] = set()
        self._conflicts = 0
        self._stale_dropped = 0
        self._invalid_results = 0
        # spans carry the logical layer: backing/rank -> {(layer, start, end)}
        self._main_ranges: dict[str, set[tuple[int, int, int]]] = {}
        self._indexer_ranges: dict[int, set[tuple[int, int, int]]] = {}
        # Failed transfers keep their scope blocked: a FAILED write may have
        # touched the target and v1 has no retry policy, so readiness stays
        # False until the bounded-failure path retires the request.
        self._failed_scopes: set[tuple[str, int]] = set()

    # ------------------------------------------------------------- record

    def record(self, result: TransferResult) -> None:
        """Judge one transfer result. Order is fixed, do not reorder:
        1. identity mismatch -> stale, dropped, only old references cleared
        2. same key exists -> conflicts never flip an existing verdict;
           only source_quiesced False->True may be merged
        3. new key -> store; only SUCCESS is credited, MAIN by backing_id and
           INDEXER by consumer_rank (un-enrolled target -> invalid)
        """
        key = result.key
        if key.request != self.request:
            # Results from another request lifetime (preempt/retry) can never
            # belong here; we only store own-instance keys, so dropping is the
            # full "clear old references" step.
            self._stale_dropped += 1
            logger.debug(
                "ReceiptLedger dropped stale result (instance/generation mismatch): key=%s expected=%s",
                key,
                self.request,
            )
            return

        existing = self._entries.get(key)
        if existing is not None:
            if existing.status is Status.SUCCESS and result.status is not Status.SUCCESS:
                self._conflicts += 1
                logger.warning("ReceiptLedger conflict: SUCCESS then %s for key=%s", result.status.value, key)
            elif existing.status is not Status.SUCCESS and result.status is Status.SUCCESS:
                self._conflicts += 1
                logger.warning("ReceiptLedger conflict: %s then SUCCESS for key=%s", existing.status.value, key)
            elif result.source_quiesced and not existing.source_quiesced:
                existing.source_quiesced = True
            return

        self._entries[key] = _ResultEntry(status=result.status, source_quiesced=result.source_quiesced)
        if self.required_layers and key.logical_layer not in self.required_layers:
            self._invalid_results += 1
            logger.warning(
                "ReceiptLedger invalid result: layer %s is outside the required plan (key=%s)", key.logical_layer, key
            )
            return
        scope = key.target_scope
        if key.component is Component.MAIN:
            scope_key = (f"main:{scope.backing_id}", key.logical_layer)
        else:
            assert scope.consumer_rank is not None  # guaranteed by TransferKey validation
            scope_key = (f"indexer:{scope.consumer_rank}", key.logical_layer)
        if result.status is not Status.SUCCESS:
            self._failed_scopes.add(scope_key)
            return
        span = (key.logical_layer, key.token_range.start, key.token_range.end)
        if key.component is Component.MAIN:
            if scope.backing_id in self.main_backings:
                self._main_ranges.setdefault(scope.backing_id, set()).add(span)
            else:
                self._invalid_results += 1
                logger.warning("ReceiptLedger invalid result: unknown MAIN backing %r (key=%s)", scope.backing_id, key)
        else:
            if scope.consumer_rank in self.indexer_ranks:
                self._indexer_ranges.setdefault(scope.consumer_rank, set()).add(span)
            else:
                self._invalid_results += 1
                logger.warning(
                    "ReceiptLedger invalid result: unknown INDEXER consumer rank %s (key=%s)",
                    scope.consumer_rank,
                    key,
                )

    def finish_rank(self, rank: int, status: Status) -> None:
        """Rank terminal. First write wins; identical repeats are idempotent;
        inconsistent repeats count as conflict; unexpected ranks never count."""
        if not isinstance(status, Status):
            raise TypeError(f"finish_rank status must be Status, got {type(status).__name__}")
        if rank not in self.expected_ranks:
            if rank not in self._unexpected:
                self._unexpected.add(rank)
                logger.warning(
                    "ReceiptLedger got terminal from unexpected rank %s (expected=%s)",
                    rank,
                    sorted(self.expected_ranks),
                )
            return
        prev = self._terminals.get(rank)
        if prev is None:
            self._terminals[rank] = status
        elif prev is not status:
            self._conflicts += 1
            logger.warning("ReceiptLedger terminal conflict for rank %s: %s then %s", rank, prev.value, status.value)

    def mark_visible(self, rank: int) -> None:
        """Record that the consumer stream dependency is wired for ``rank``.
        Only call after prepare_decode returned normally."""
        self._visible.add(rank)

    # ----------------------------------------------------------- judgment

    def _data_ready(self) -> bool:
        """Data-side conditions only (unions + terminals + no protocol errors),
        without visibility. Gates building the receipt for prepare_decode —
        visibility is the *outcome* of that call, so it must not be
        a precondition here."""
        if self._conflicts or self._invalid_results:
            return False
        if self._failed_scopes:
            return False
        if set(self._terminals) != self.expected_ranks:
            return False
        if any(status is not Status.SUCCESS for status in self._terminals.values()):
            return False
        if self.required_layers:
            return all(
                _ranges_cover(
                    {(s, e) for (lay, s, e) in self._main_ranges.get(backing, ()) if lay == layer}, self.prompt_len
                )
                for layer in self.required_layers
                for backing in self.main_backings
            ) and all(
                _ranges_cover(
                    {(s, e) for (lay, s, e) in self._indexer_ranges.get(rank, ()) if lay == layer}, self.prompt_len
                )
                for layer in self.required_layers
                for rank in self.indexer_ranks
            )
        return all(
            _ranges_cover({(s, e) for (_, s, e) in self._main_ranges.get(backing, ())}, self.prompt_len)
            for backing in self.main_backings
        ) and all(
            _ranges_cover({(s, e) for (_, s, e) in self._indexer_ranges.get(rank, ())}, self.prompt_len)
            for rank in self.indexer_ranks
        )

    def should_publish(self) -> bool:
        """Full conjunction (data conditions + every expected rank
        visible + no protocol errors); pure query, no side effects. This is the
        gate for handing the request to ``done_requests``/``get_finished``."""
        return self._data_ready() and self.expected_ranks <= self._visible

    def build_receipt(self) -> ReceiveReceipt | None:
        """Build the ReceiveReceipt once the data conditions hold (visible is
        still False at that point); None when not ready. ``visible`` carries the
        current visibility value (caller flips it after prepare_decode)."""
        if not self._data_ready():
            return None
        return ReceiveReceipt(
            request=self.request,
            target_generation=self.request.transfer_generation,
            main_union_ok=all(
                _ranges_cover({(s, e) for (_, s, e) in self._main_ranges.get(b, ())}, self.prompt_len)
                for b in self.main_backings
            ),
            indexer_ranks_ok=all(
                _ranges_cover({(s, e) for (_, s, e) in self._indexer_ranges.get(r, ())}, self.prompt_len)
                for r in self.indexer_ranks
            ),
            terminals_ok=(
                set(self._terminals) == set(self.expected_ranks)
                and all(status is Status.SUCCESS for status in self._terminals.values())
            ),
            visible=self.expected_ranks <= self._visible,
        )

    # --------------------------------------------------------- introspection

    @property
    def conflicts(self) -> int:
        return self._conflicts

    @property
    def stale_dropped(self) -> int:
        return self._stale_dropped

    @property
    def invalid_results(self) -> int:
        return self._invalid_results

    @property
    def pending_ranks(self) -> frozenset[int]:
        """Expected ranks that have not delivered a terminal yet (logging/UT)."""
        return frozenset(self.expected_ranks - set(self._terminals))

    @property
    def unexpected_ranks(self) -> frozenset[int]:
        return frozenset(self._unexpected)


@dataclass
class _ResultEntry:
    status: Status
    source_quiesced: bool


# --------------------------------------------------------------------------
# SlotLedger (P side reuse barrier) — slot-keyed per
# --------------------------------------------------------------------------


class SlotError(Exception):
    """Protocol-level slot misuse (e.g. advancing a generation while the old
    one still has in-flight readers, or a failed generation being awaited)."""


@dataclass
class _Ticket:
    """One reader of one layer generation. ``quiesced`` releases the source;
    a FAILED + quiesced=False ticket must keep holding the source."""

    key: TransferKey
    done: bool = False
    failed: bool = False
    quiesced: bool = False

    @property
    def pending(self) -> bool:
        return not self.quiesced


@dataclass
class _Generation:
    """One generation of readers on one physical slot. Sealed when the batch
    plan is complete; a generation may only be replaced once fully quiesced."""

    epoch: int
    sealed: bool
    tickets: dict[TransferKey, _Ticket] = field(default_factory=dict)


class SlotLedger:
    """Per-slot generation + ticket ledger.

    Keys are *slot ids* — the same address-derived identity as the baseline
    ``layer_storage_slots`` — never layer numbers. One layer's slots share one
    ticket per key, so a layer mapping to several slots still counts once.

    ``wait`` implements the snapshot rule:
      1. current epoch == snapshot epoch and sealed with nothing pending -> done
      2. current epoch > snapshot epoch -> old generation fully ended
         (advancing requires all tickets quiesced), return without chasing
         the new generation
      3. current epoch == snapshot epoch and still pending -> wait;
         any failed ticket -> SlotError
    """

    def __init__(self, slots_for_layer: Callable[[int], tuple[int, ...]]) -> None:
        self._slots_for_layer = slots_for_layer
        self._cond = threading.Condition(threading.Lock())
        self._current: dict[int, _Generation] = {}
        self._next_epoch: dict[int, int] = {}

    # ------------------------------------------------------------ helpers

    def _slots(self, layer: int) -> tuple[int, ...]:
        slots = tuple(self._slots_for_layer(layer))
        if not slots:
            raise SlotError(f"no storage slot mapped for layer {layer}")
        return slots

    # ------------------------------------------------------------ mutate

    def register_batch(
        self,
        layer: int,
        keys: Iterable[TransferKey],
        *,
        seal_immediately: bool = True,
    ) -> tuple[tuple[int, int], ...]:
        """Register one layer's whole send batch and (by default) seal at once:
        advance each slot's epoch, register all tickets, then seal — the batch
        plan is complete at this point.

        Raises SlotError when a slot's current generation still has pending
        (non-quiesced) tickets: that means the reuse wait did not take effect.

        ``seal_immediately=False`` exists only for incremental flows/tests; the
        default matches the spec's atomic register+seal.
        """
        key_list = list(keys)
        slots = self._slots(layer)
        created: list[tuple[int, int]] = []
        with self._cond:
            for slot in slots:
                gen = self._current.get(slot)
                if gen is not None and any(t.pending for t in gen.tickets.values()):
                    raise SlotError(
                        f"slot {slot}: generation {gen.epoch} still has "
                        f"{sum(1 for t in gen.tickets.values() if t.pending)} pending tickets; "
                        "register_batch requires the previous generation to be quiesced (wait first)"
                    )
            for slot in slots:
                epoch = self._next_epoch.get(slot, 0) + 1
                self._next_epoch[slot] = epoch
                # One shared ticket per key across all slots of this layer.
                gen = _Generation(epoch=epoch, sealed=seal_immediately)
                self._current[slot] = gen
                created.append((slot, epoch))
            if key_list:
                ticket_map: dict[TransferKey, _Ticket] = {}
                for key in key_list:
                    ticket_map[key] = _Ticket(key=key)
                for slot in slots:
                    self._current[slot].tickets = dict(ticket_map)
            self._cond.notify_all()
        return tuple(created)

    def reserve(self, layer: int, key: TransferKey) -> _Ticket:
        """Incremental registration into the current generation. Duplicate key
        returns the same ticket; different keys count separately. Requires an
        open (registered, unsealed) generation — with the default atomic
        register_batch this is only usable in incremental flows."""
        slots = self._slots(layer)
        with self._cond:
            ticket: _Ticket | None = None
            for slot in slots:
                gen = self._current.get(slot)
                if gen is None:
                    raise SlotError(f"slot {slot}: no generation for layer {layer}; register_batch first")
                if gen.sealed:
                    raise SlotError(f"slot {slot}: generation {gen.epoch} already sealed")
                existing = gen.tickets.get(key)
                if ticket is None:
                    ticket = existing if existing is not None else _Ticket(key=key)
                gen.tickets[key] = ticket
            self._cond.notify_all()
            assert ticket is not None
            return ticket

    def seal(self, layer: int) -> None:
        """Seal the current generation(s): no more readers may register. A
        zero-transfer batch must still be sealed — ``register_batch`` with no
        keys seals empty generations. Idempotent."""
        slots = self._slots(layer)
        with self._cond:
            for slot in slots:
                gen = self._current.get(slot)
                if gen is None:
                    raise SlotError(f"slot {slot}: no generation to seal for layer {layer}")
                gen.sealed = True
            self._cond.notify_all()

    def complete(self, layer: int, key: TransferKey, status: Status, source_quiesced: bool) -> bool:
        """Transfer-completion callback. Unknown (stale-generation) keys are
        ignored with a log. Only ``source_quiesced=True`` releases the source
        (removes the ticket from pending); FAILED + quiesced=False keeps it.

        Returns True when any current-generation ticket was updated.
        """
        slots = self._slots(layer)
        touched = False
        with self._cond:
            for slot in slots:
                gen = self._current.get(slot)
                if gen is None:
                    continue
                ticket = gen.tickets.get(key)
                if ticket is None:
                    continue  # old-generation late arrival: ignore, keep new generation intact
                if status is Status.SUCCESS:
                    ticket.done = True
                elif status in (Status.FAILED, Status.CANCELLED):
                    ticket.failed = True
                if source_quiesced:
                    ticket.quiesced = True
                touched = True
            if touched:
                self._cond.notify_all()
        if not touched:
            logger.debug("SlotLedger.complete ignored unknown key (stale generation?): layer=%s key=%s", layer, key)
        return touched

    # -------------------------------------------------------------- wait

    def wait(self, layer: int, timeout: float) -> None:
        """Block until every slot of ``layer`` has its *snapshot* generation
        quiesced (three-branch rule). Cheap and idempotent when nothing
        is pending. Raises SlotError on failure, TimeoutError on time.

        The snapshot is taken once at entry and never refreshed: a newer
        generation may only exist after the snapshot generation was fully
        quiesced (see ``register_batch`` guard), so branch 2 may return while a
        newer generation is still in flight — it has no readers of the
        snapshot generation and chasing it would deadlock ."""
        slots = self._slots(layer)
        with self._cond:
            snapshot: dict[int, int | None] = {}
            for slot in slots:
                gen = self._current.get(slot)
                snapshot[slot] = gen.epoch if gen is not None else None
            deadline = time.monotonic() + max(float(timeout), 0.0)
            while True:
                ready = True
                for slot in slots:
                    gen = self._current.get(slot)
                    snap = snapshot[slot]
                    if gen is None or snap is None:
                        continue  # nothing was ever registered on this slot
                    if gen.epoch > snap:
                        continue  # branch 2: snapshot generation fully ended; do not chase
                    if gen.epoch < snap:
                        raise SlotError(f"slot {slot}: epoch regressed ({gen.epoch} < snapshot {snap})")
                    if not gen.sealed:
                        ready = False
                    for ticket in gen.tickets.values():
                        if ticket.failed:
                            raise SlotError(
                                f"slot {slot}: generation {gen.epoch} has failed transfers (key={ticket.key})"
                            )
                    if any(t.pending for t in gen.tickets.values()):
                        ready = False
                if ready:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"layerwise slot reuse wait timed out for layer {layer} (slots={slots}, snapshot={snapshot})"
                    )
                self._cond.wait(remaining)

    # ----------------------------------------------------- introspection

    def epoch_of(self, layer: int) -> tuple[tuple[int, int | None], ...]:
        """Current generation per slot (logging/UT)."""
        slots = self._slots(layer)
        with self._cond:
            return tuple((slot, gen.epoch if (gen := self._current.get(slot)) is not None else None) for slot in slots)

    def pending_count(self, layer: int) -> int:
        """Distinct tickets still holding their source (logging/UT)."""
        slots = self._slots(layer)
        with self._cond:
            seen: deque[_Ticket] = []
            for slot in slots:
                gen = self._current.get(slot)
                if gen is None:
                    continue
                for ticket in gen.tickets.values():
                    if ticket.pending and all(ticket is not s for s in seen):
                        seen.append(ticket)
            return len(seen)


__all__ = ["ReceiptLedger", "SlotError", "SlotLedger", "is_no_compute_layer"]

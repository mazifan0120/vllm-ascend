# SPDX-License-Identifier: Apache-2.0
"""Entry-specific receive coverage and remote-DMA retirement proofs.

The required keys come from an admitted manifest, not a Cartesian product of
layers and backings. A completed Main view cannot credit its sibling view.
All methods are local, bounded state operations; no owner or peer calls occur
under the ledger lock. This module intentionally depends only on stdlib and
the value contract, so control tests need no device/runtime imports.
"""

import threading
from dataclasses import replace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    Component,
    ContractError,
    ReceiveReceipt,
    RequestKey,
    Status,
    TokenRange,
    TransferKey,
    TransferResult,
)


def _scope(key: TransferKey) -> TransferKey:
    return replace(key, token_range=TokenRange(0, 0))


def _covers(required: TokenRange, spans) -> bool:
    end = required.start
    for start, stop in sorted(spans):
        if start > end:
            return False
        end = max(end, stop)
        if end >= required.end:
            return True
    return end >= required.end


class ExactReceiptLedger:
    def __init__(
        self,
        request: RequestKey,
        required_keys: tuple[TransferKey, ...],
        *,
        expected_ranks: tuple[int, ...],
        target_generation: int,
        visibility_rank: int,
    ) -> None:
        if (
            type(request) is not RequestKey
            or type(required_keys) is not tuple
            or not required_keys
            or type(expected_ranks) is not tuple
            or not expected_ranks
            or any(type(rank) is not int or rank < 0 for rank in expected_ranks)
            or len(set(expected_ranks)) != len(expected_ranks)
            or type(target_generation) is not int
            or target_generation < 0
            or type(visibility_rank) is not int
            or visibility_rank not in expected_ranks
        ):
            raise ContractError("Exact receipt needs an explicit request, target generation and TP owner plan")
        self.request = request
        self.expected_ranks = frozenset(expected_ranks)
        self.target_generation = target_generation
        self.visibility_rank = visibility_rank
        self._required = {}
        for key in required_keys:
            if (
                type(key) is not TransferKey
                or key.request != request
                or key.rank not in self.expected_ranks
                or key.token_range.start >= key.token_range.end
                or _scope(key) in self._required
            ):
                raise ContractError("Exact receipt plan repeats or misidentifies an entry/range/writer")
            self._required[_scope(key)] = key.token_range
        if {key.component for key in required_keys} != set(Component):
            raise ContractError("Exact receipt plan must include both Main and Indexer entries")
        self._lock = threading.RLock()
        self._results: dict[TransferKey, TransferResult] = {}
        self._ranges: dict[TransferKey, set[tuple[int, int]]] = {}
        self._terminals: dict[int, tuple[Status, bool]] = {}
        self._visible = False
        self._aborted = False
        self._protocol_errors = 0
        self._stale_dropped = 0

    def record(self, result: TransferResult) -> None:
        if type(result) is not TransferResult:
            raise ContractError("Exact receipt accepts typed transfer results")
        with self._lock:
            key = result.key
            if key.request != self.request:
                self._stale_dropped += 1
                return
            scope = _scope(key)
            required = self._required.get(scope)
            if required is None or not required.start <= key.token_range.start < key.token_range.end <= required.end:
                self._protocol_errors += 1
                return
            previous = self._results.get(key)
            if previous is not None:
                if previous.status is not result.status:
                    self._protocol_errors += 1
                    return
                if previous.source_quiesced:
                    return  # Exact duplicates and delayed False cannot revoke a fence.
            self._results[key] = result
            if result.status is Status.SUCCESS and result.source_quiesced:
                self._ranges.setdefault(scope, set()).add((key.token_range.start, key.token_range.end))

    def finish_rank(self, request: RequestKey, rank: int, status: Status, source_quiesced: bool) -> None:
        """A terminal promises no future writes, in addition to native read completion.

        A failed terminal may still prove safe retirement. It never proves
        decode readiness. A timeout/unknown native outcome must pass False.
        """
        if (
            type(request) is not RequestKey
            or type(rank) is not int
            or type(status) is not Status
            or type(source_quiesced) is not bool
        ):
            raise ContractError("Exact rank terminal requires typed identity/status/quiescence")
        with self._lock:
            if request != self.request:
                self._stale_dropped += 1
                return
            if rank not in self.expected_ranks:
                self._protocol_errors += 1
                return
            previous = self._terminals.get(rank)
            if previous is not None:
                if previous[0] is not status:
                    self._protocol_errors += 1
                    return
                source_quiesced = previous[1] or source_quiesced
            self._terminals[rank] = (status, source_quiesced)

    def remote_quiesced(self) -> bool:
        with self._lock:
            return (
                not self._protocol_errors
                and self.expected_ranks == self._terminals.keys()
                and all(quiesced for _, quiesced in self._terminals.values())
                and all(result.source_quiesced for result in self._results.values())
            )

    def build_receipt(self) -> ReceiveReceipt | None:
        with self._lock:
            if (
                self._aborted
                or not self.remote_quiesced()
                or any(status is not Status.SUCCESS for status, _ in self._terminals.values())
                or any(result.status is not Status.SUCCESS for result in self._results.values())
                or any(not _covers(span, self._ranges.get(scope, ())) for scope, span in self._required.items())
            ):
                return None
            return ReceiveReceipt(self.request, self.target_generation, True, True, True, self._visible)

    def abort_delivery(self) -> None:
        """Prevent decode permanently; continue judging late DMA completion."""
        with self._lock:
            self._aborted = True

    @property
    def delivery_aborted(self):
        with self._lock:
            return self._aborted

    @property
    def visibility_installed(self):
        with self._lock:
            return self._visible

    @property
    def failed(self):
        with self._lock:
            return bool(
                self._protocol_errors
                or any(r.status is not Status.SUCCESS for r in self._results.values())
                or any(status is not Status.SUCCESS for status, _ in self._terminals.values())
            )

    def accepts_source_lookup(self, writer_rank: int) -> bool:
        with self._lock:
            return (
                not self._aborted
                and not self._protocol_errors
                and writer_rank in self.expected_ranks
                and writer_rank not in self._terminals
            )

    def mark_visible(self, rank: int) -> None:
        with self._lock:
            if type(rank) is not int or rank != self.visibility_rank or self.build_receipt() is None:
                raise ContractError("Visibility requires this owner's complete admitted entry plan")
            self._visible = True

    def should_publish(self) -> bool:
        with self._lock:
            return self._visible and self.build_receipt() is not None

    @property
    def protocol_errors(self) -> int:
        with self._lock:
            return self._protocol_errors

    @property
    def stale_dropped(self) -> int:
        with self._lock:
            return self._stale_dropped

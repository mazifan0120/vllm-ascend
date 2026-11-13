# SPDX-License-Identifier: Apache-2.0
"""Local source contract for a connector reading Prefill reuse allocations.

Events and views are worker-local. The consumer owns transport tickets, seals
and source-quiescence; Store keys and snapshot generations never cross here.
"""

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class SourceView:
    layer: int
    component: str
    group: int
    slot: str
    address: int
    length_bytes: int
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    dtype: str
    manager_block_tokens: int
    page_tokens: int
    # Entry within the component (e.g. Indexer data and scale).
    entry: int


@dataclass(frozen=True)
class SlotEpoch:
    worker: str
    slot: str
    generation: int
    layer: int


@dataclass(frozen=True)
class LayerSource:
    layer: int
    epochs: tuple[SlotEpoch, ...]
    views: tuple[SourceView, ...]


@dataclass(frozen=True)
class RequestSource:
    request_id: str
    request_instance: str
    start_token: int
    end_token: int
    block_ids_by_group: tuple[tuple[int, ...], ...]


class SourceConsumer(Protocol):
    """Callbacks may run on receiver or model threads; must be thread-safe.

    open_source is a nonblocking local registration, before the first write.
    source_ready must register ALL batch readers before making work runnable;
    consumers wait ready_event before device reads. seal_source closes additions,
    including an empty batch. wait_source succeeds only after sealing AND all
    readers quiesce successfully. Failures must never be converted to success.
    """

    def open_source(self, source: LayerSource) -> None: ...
    def source_ready(self, source: LayerSource, requests: tuple[RequestSource, ...], ready_event: Any) -> None: ...
    def seal_source(self, source: LayerSource) -> None: ...
    def wait_source(self, source: LayerSource) -> None: ...
    def abort_source(self, source: LayerSource, error: Exception) -> None: ...


class ReuseSourceRegistry:
    """One ordered writer, with concurrent immutable consumer callbacks.

    No locks are held across consumer waits. Caller must serialize begin_write;
    the Store's single receive queue provides that ordering. Keeping a failed
    source current means a later attempt must wait (and fail) again, not reuse it.
    """

    def __init__(self, views: dict[int, tuple[SourceView, ...]], consumer: SourceConsumer):
        if not views or any(not entries for entries in views.values()):
            raise ValueError("A source registry requires every layer's actual views")
        self.views = views
        self.consumer = consumer
        self.worker = uuid4().hex
        self._generation: dict[str, int] = {}
        self._occupants: dict[str, LayerSource] = {}
        self._layers: dict[int, LayerSource] = {}
        self._published: set[tuple[SlotEpoch, ...]] = set()
        self._error: Exception | None = None

    def _check_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("Reuse source contract failed; allocations must remain retained") from self._error

    def begin_write(self, layer: int) -> None:
        self._check_failed()
        views = self.views[layer]
        slots = tuple(dict.fromkeys(view.slot for view in views))
        previous = {self._occupants[slot].epochs: self._occupants[slot] for slot in slots if slot in self._occupants}
        for source in previous.values():
            self.consumer.wait_source(source)
            self._published.discard(source.epochs)
        epochs = tuple(SlotEpoch(self.worker, slot, self._generation.get(slot, 0) + 1, layer) for slot in slots)
        source = LayerSource(layer, epochs, views)
        # Install BEFORE notification: an open failure must not lose ownership.
        for epoch in epochs:
            self._generation[epoch.slot] = epoch.generation
            self._occupants[epoch.slot] = source
        self._layers[layer] = source
        try:
            self.consumer.open_source(source)
        except Exception as exc:
            self._error = exc
            self.consumer.abort_source(source, exc)
            raise

    def publish(self, layer: int, requests: tuple[RequestSource, ...], ready_event: Any) -> None:
        self._check_failed()
        source = self._layers[layer]
        if source.epochs in self._published:
            return
        if any(self._occupants[epoch.slot] is not source for epoch in source.epochs):
            raise RuntimeError("Cannot publish an overwritten source epoch")
        try:
            if any(
                not request.request_instance or request.start_token < 0 or request.end_token < request.start_token
                for request in requests
            ):
                raise ValueError("Source requests require an instance and valid range")
            for request in requests:
                for view in source.views:
                    if view.group >= len(request.block_ids_by_group):
                        raise ValueError("Source request is missing a component block map")
                    ids = request.block_ids_by_group[view.group]
                    capacity = view.shape[0] * view.page_tokens // view.manager_block_tokens
                    if request.end_token > len(ids) * view.manager_block_tokens or any(
                        block < 0 or block >= capacity for block in ids
                    ):
                        raise ValueError("Source request exceeds its cache capacity")
        except Exception as exc:
            self.fail(exc)
            raise
        # Mark before callbacks so failure/repeated hooks cannot duplicate sends.
        self._published.add(source.epochs)
        try:
            self.consumer.source_ready(source, requests, ready_event)
            self.consumer.seal_source(source)
        except Exception as exc:
            self._error = exc
            self.consumer.abort_source(source, exc)
            raise

    def fail(self, error: Exception) -> None:
        """Publish failure without claiming DMA quiescence or freeing views."""
        if self._error is not None:
            return
        self._error = error
        sources = {source.epochs: source for source in self._occupants.values()}
        for source in sources.values():
            self.consumer.abort_source(source, error)

    def drain(self) -> None:
        self._check_failed()
        sources = {source.epochs: source for source in self._occupants.values()}
        for source in sources.values():
            self.consumer.wait_source(source)

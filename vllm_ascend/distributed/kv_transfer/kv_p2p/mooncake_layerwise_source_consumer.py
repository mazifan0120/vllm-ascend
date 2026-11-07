# SPDX-License-Identifier: Apache-2.0
"""Exact source-epoch readers for native manifest and ordinary send tasks.

All readers are reserved before queue publication. Queue, RPC and native work
run outside the generation lock. A legacy hook arriving after sealing uses its
already-reserved ticket. Old completed epochs are remembered per physical slot.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import LayerSource, RequestSource

SOURCE_WAIT_TIMEOUT_SECONDS = 300.0


@dataclass
class _Generation:
    source: LayerSource
    ready: bool = False
    sealed: bool = False
    pending: set[int] = field(default_factory=set)
    completed: set[int] = field(default_factory=set)
    ready_event: Any = None
    error: Exception | None = None
    legacy_expected: bool = False
    legacy_received: bool = False
    legacy_task: Any = None


class LayerwiseSourceConsumer:
    def __init__(self, worker: Any, *, timeout: float = SOURCE_WAIT_TIMEOUT_SECONDS) -> None:
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Source wait timeout must be positive and finite")
        self._worker = worker
        self._timeout = timeout
        self._condition = threading.Condition(threading.RLock())
        self._generations = {}
        self._layer_source = {}
        self._closed_epochs = {}
        self._ever_opened = set()

    def _record(self, source):
        record = self._generations.get(source.epochs)
        if record is None or record.source != source:
            raise RuntimeError("Unknown or changed exact source generation")
        return record

    def open_source(self, source: LayerSource) -> None:
        with self._condition:
            if (
                not source.epochs
                or source.epochs in self._generations
                or any(self._closed_epochs.get((e.worker, e.slot), 0) >= e.generation for e in source.epochs)
            ):
                raise RuntimeError("Source generation is empty, repeated or already retired")
            new_slots = {(e.worker, e.slot) for e in source.epochs}
            for record in self._generations.values():
                if new_slots.intersection((e.worker, e.slot) for e in record.source.epochs):
                    raise RuntimeError("Physical source slot still has an unretired generation")
            self._generations[source.epochs] = _Generation(source)
            self._layer_source[source.layer] = source.epochs
            self._ever_opened.add(source.layer)

    def source_ready(self, source: LayerSource, requests: tuple[RequestSource, ...], ready_event: Any) -> None:
        with self._condition:
            record = self._record(source)
            if record.ready:
                return
            if record.sealed or record.error is not None:
                raise RuntimeError("Cannot publish a sealed or failed source generation")
        prepare = getattr(self._worker, "prepare_source_batch", None)
        jobs, legacy_expected = prepare(source, requests, ready_event) if callable(prepare) else ((), bool(requests))
        task = None
        with self._condition:
            record = self._record(source)
            if record.ready or record.sealed or record.error is not None:
                raise RuntimeError("Concurrent or failed source publication")
            record.ready = True
            record.ready_event = ready_event
            record.legacy_expected = legacy_expected or record.legacy_task is not None
            # Zero is the one legacy task; positive tickets are direct jobs.
            # Install the complete reader set before releasing any queue item.
            record.pending = set(range(1, len(jobs) + 1))
            if record.legacy_expected:
                record.pending.add(0)
            if record.legacy_task is not None:
                task, record.legacy_task = record.legacy_task, None
            self._condition.notify_all()
        try:
            for ticket, job in enumerate(jobs, 1):
                job.publish(self._completion(source, ticket))
            if task is not None:
                self._enqueue(task, ready_event, self._completion(source, 0))
        except Exception as error:
            self.abort_source(source, error)
            raise

    def _completion(self, source, ticket):
        def complete(error, quiesced):
            with self._condition:
                record = self._record(source)
                if ticket in record.completed or ticket not in record.pending or type(quiesced) is not bool:
                    raise RuntimeError("Unexpected or repeated exact source reader completion")
                record.completed.add(ticket)
                if error is not None:
                    record.error = error
                if quiesced:
                    record.pending.remove(ticket)
                elif record.error is None:
                    record.error = RuntimeError("Native reader outcome is unknown; retain source")
                self._condition.notify_all()

        return complete

    def seal_source(self, source: LayerSource) -> None:
        with self._condition:
            record = self._record(source)
            if not record.ready:
                raise RuntimeError("Seal requires a published complete source reader plan")
            record.sealed = True
            self._condition.notify_all()

    def _retire(self, record):
        source = record.source
        if not record.ready or not record.sealed or record.pending or record.error is not None:
            raise RuntimeError("Cannot retire an unsealed, pending or failed source generation")
        self._generations.pop(source.epochs)
        if self._layer_source.get(source.layer) == source.epochs:
            self._layer_source.pop(source.layer)
        for epoch in source.epochs:
            identity = (epoch.worker, epoch.slot)
            self._closed_epochs[identity] = max(epoch.generation, self._closed_epochs.get(identity, 0))

    def wait_source(self, source: LayerSource) -> None:
        deadline = time.monotonic() + self._timeout
        with self._condition:
            while True:
                record = self._generations.get(source.epochs)
                if record is None:
                    if source.epochs and all(
                        self._closed_epochs.get((e.worker, e.slot), 0) >= e.generation for e in source.epochs
                    ):
                        return
                    raise RuntimeError("wait_source for an unknown exact source generation")
                if record.source != source:
                    raise RuntimeError("wait_source view changed within an epoch")
                if record.error is not None:
                    raise RuntimeError("Reuse source generation failed; allocations retained") from record.error
                if record.ready and record.sealed and not record.pending:
                    self._retire(record)
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Exact source generation is unsealed or still has native readers")
                self._condition.wait(remaining)

    def abort_source(self, source: LayerSource, error: Exception) -> None:
        with self._condition:
            record = self._generations.get(source.epochs)
            if record is not None:
                record.error = error
            self._condition.notify_all()
        complete = getattr(self._worker, "_complete_layer_reuse", None)
        if callable(complete):
            complete(source.layer, str(error))

    def hand_off_send_task(self, layer: int, task: Any) -> None:
        ready = None
        completion = None
        with self._condition:
            record = self._generations.get(self._layer_source.get(layer))
            if record is None:
                if layer in self._ever_opened and getattr(task, "send_request", {}):
                    raise RuntimeError("Legacy reader lost its registered source generation")
            else:
                if record.legacy_received:
                    raise RuntimeError("Legacy source task handed off twice")
                record.legacy_received = True
                if not record.ready:
                    record.legacy_task = task
                    return
                if not record.legacy_expected and getattr(task, "send_request", {}):
                    raise RuntimeError("Legacy reader was absent from the sealed source batch")
                ready = record.ready_event
                if record.legacy_expected:
                    completion = self._completion(record.source, 0)
        try:
            self._enqueue(task, ready, completion)
        except Exception as error:
            if record is not None:
                self.abort_source(record.source, error)
            raise

    def _enqueue(self, task, ready_event, completion=None):
        if ready_event is not None:
            task.source_ready_event = ready_event
        task.source_completion = completion
        send_thread = getattr(self._worker, "kv_send_layer_thread", None)
        if send_thread is None:
            raise RuntimeError("Layerwise send thread is not available")
        send_thread.send_queue.put(task)

    def is_published(self, layer: int) -> bool:
        with self._condition:
            record = self._generations.get(self._layer_source.get(layer))
            return bool(record and record.ready)

    def retire(self, layer: int) -> None:
        with self._condition:
            record = self._generations.get(self._layer_source.get(layer))
            if record is not None:
                self._retire(record)

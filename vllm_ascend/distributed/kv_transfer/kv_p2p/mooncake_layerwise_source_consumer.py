# SPDX-License-Identifier: Apache-2.0
"""P-side source consumer adapting the reuse source contract to the
layerwise connector's existing slot ledger and send path.

The Store-side ``ReuseSourceRegistry`` drives this consumer through the five
source-lifecycle callbacks. The adapter maps them onto the connector worker's
already-established slot wait / send-done machinery so that enabling the source
contract does not start a second, parallel send path.

Send ownership and deduplication: when the contract is active, the layer's
transfer is released through ``source_ready`` (readers registered before the
async task is released). The legacy ``save_kv_layer`` hook still owns the NPU
data preparation (resharding, quantization, NZ layout) that cannot move off the
attention path, so it hands the assembled ``SendTask`` to this consumer instead
of enqueueing it directly. The consumer publishes it to the send queue when
``source_ready`` fires — or immediately if ``source_ready`` already fired for
this generation (the two callbacks both run after attention, in either order).
Plain mode bypasses the consumer and enqueues directly, preserving the legacy
path. ``wait_source`` delegates to the worker's ``wait_for_layer_reuse``, which
already enforces the slot ledger's three-branch rule and bounded wait.
"""

import threading
from typing import Any

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import (
    LayerSource,
    RequestSource,
)


class LayerwiseSourceConsumer:
    """Adapts the five-method source contract to the layerwise send path.

    The consumer is installed on the producer connector and called by the
    Store registry. Callbacks may run on receiver or model threads, so the
    per-generation state is guarded by a lock.
    """

    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self._lock = threading.RLock()
        # epochs -> generation record dict
        self._generations: dict[tuple, dict] = {}
        # layer -> active epochs, so wait_source / hand_off resolve a layer
        self._layer_source: dict[int, tuple] = {}

    def _record(self, source: LayerSource) -> dict | None:
        return self._generations.get(self._key(source))

    def _key(self, source: LayerSource) -> tuple:
        return source.epochs

    def open_source(self, source: LayerSource) -> None:
        """Nonblocking registration of a new source generation before first write."""
        with self._lock:
            if self._key(source) in self._generations:
                raise RuntimeError("open_source called twice for the same source generation")
            self._generations[self._key(source)] = {
                "sealed": False,
                "pending": 0,
                "ready_event": None,
                "error": None,
                "published": False,
                "send_task": None,
            }
            self._layer_source[source.layer] = self._key(source)

    def source_ready(self, source: LayerSource, requests: tuple[RequestSource, ...], ready_event: Any) -> None:
        """Register all batch readers, then release the async send task.

        ``ready_event`` is the KV-ready fence recorded by the Store. If
        ``save_kv_layer`` already handed off the assembled ``SendTask``, it is
        published to the send queue here; otherwise the task is published when
        it arrives through ``hand_off_send_task``. Either ordering is correct
        since both callbacks run after attention for the same layer.
        """
        with self._lock:
            record = self._record(source)
            if record is None:
                raise RuntimeError("source_ready before open_source for the layer generation")
            if record["published"]:
                return
            record["pending"] = len(requests)
            record["ready_event"] = ready_event
            record["published"] = True
            task = record.get("send_task")
            if task is not None:
                record["send_task"] = None
                self._enqueue(task, ready_event)

    def seal_source(self, source: LayerSource) -> None:
        """Close additions for this generation, including an empty batch."""
        with self._lock:
            record = self._record(source)
            if record is None:
                raise RuntimeError("seal_source before open_source for the layer generation")
            record["sealed"] = True

    def wait_source(self, source: LayerSource) -> None:
        """Succeed only after sealing AND all readers quiesce successfully.

        Delegates to the worker's existing layer reuse wait, which enforces the
        slot ledger's three-branch rule (current/older/pending) and a bounded
        timeout. A failed generation surfaces its error through that path.
        """
        with self._lock:
            record = self._record(source)
            error = record["error"] if record is not None else None
        if error is not None:
            raise RuntimeError("Reuse source generation failed; allocations retained") from error
        self._worker.wait_for_layer_reuse(source.layer)

    def abort_source(self, source: LayerSource, error: Exception) -> None:
        """Mark the generation failed while retaining in-flight references."""
        with self._lock:
            record = self._record(source)
            if record is not None:
                record["error"] = error
        complete = getattr(self._worker, "_complete_layer_reuse", None)
        if callable(complete):
            complete(source.layer, str(error))

    def hand_off_send_task(self, layer: int, task: Any) -> None:
        """Receive the assembled ``SendTask`` from ``save_kv_layer``.

        When the source contract is active, ``save_kv_layer`` calls this instead
        of enqueueing directly. If ``source_ready`` has already published the
        generation, the task is released now; otherwise it is held until
        ``source_ready`` fires. This is the deduplication point: the task is
        released exactly once, through the source contract, never both here and
        via a direct enqueue.
        """
        with self._lock:
            key = self._layer_source.get(layer)
            if key is None:
                # No open generation for this layer: the source contract is not
                # driving it. Fall back to a direct enqueue so a layer without a
                # published source is still sent (and fail loudly if the send
                # queue is gone).
                self._enqueue(task, None)
                return
            record = self._generations.get(key)
            if record is None:
                self._enqueue(task, None)
                return
            if record.get("published"):
                ready_event = record.get("ready_event")
                self._enqueue(task, ready_event)
            else:
                record["send_task"] = task

    def _enqueue(self, task: Any, ready_event: Any) -> None:
        """Release the task to the worker's send thread, with the KV-ready fence."""
        if ready_event is not None:
            # Store readiness is an additional dependency. The task's existing
            # wait_event can cover later quantization/resharding, so replacing
            # it could let DMA read before that work completes.
            task.source_ready_event = ready_event
        send_thread = getattr(self._worker, "kv_send_layer_thread", None)
        if send_thread is None:
            raise RuntimeError("Layerwise send thread is not available when releasing a source task")
        send_thread.send_queue.put(task)

    def is_published(self, layer: int) -> bool:
        """True when source_ready has fired for the layer's current generation."""
        with self._lock:
            key = self._layer_source.get(layer)
            if key is None:
                return False
            record = self._generations.get(key)
            return bool(record and record.get("published"))

    def retire(self, layer: int) -> None:
        """Drop the generation record after the layer's reuse is complete."""
        with self._lock:
            key = self._layer_source.pop(layer, None)
            if key is not None:
                self._generations.pop(key, None)

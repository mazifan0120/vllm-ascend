# SPDX-License-Identifier: Apache-2.0
"""Layerwise source consumer adapter tests.

The adapter maps the five-method source contract onto the worker's existing
slot reuse wait. These tests cover the lifecycle state machine and delegation
without torch or an NPU, using a stub worker that records wait calls.
"""

import queue
import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_source_consumer import (
    LayerwiseSourceConsumer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import (
    LayerSource,
    RequestSource,
    SlotEpoch,
)


def _source(layer: int = 0) -> LayerSource:
    epoch = SlotEpoch("worker-0", "main:0", 1, layer)
    return LayerSource(layer, (epoch,), ())


class StubWorker:
    """Records wait_for_layer_reuse and _complete_layer_reuse; owns a queue."""

    def __init__(self):
        self.waits = []
        self.completions = []
        self.send_queue: queue.Queue = queue.Queue()
        self.kv_send_layer_thread = SimpleNamespace(send_queue=self.send_queue)

    def wait_for_layer_reuse(self, layer_idx: int) -> None:
        self.waits.append(layer_idx)

    def _complete_layer_reuse(self, layer_idx: int, error):
        self.completions.append((layer_idx, error))


class TestLayerwiseSourceConsumer(unittest.TestCase):
    def test_lifecycle_records_publish_and_seal_before_wait(self):
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        source = _source(2)

        consumer.open_source(source)
        self.assertFalse(consumer.is_published(2))

        requests = (RequestSource("r1", "inst-1", 0, 4, ((1, 3),)),)
        consumer.source_ready(source, requests, ready_event=SimpleNamespace())
        self.assertTrue(consumer.is_published(2))

        consumer.seal_source(source)
        consumer.wait_source(source)
        self.assertEqual(worker.waits, [2])

    def test_wait_before_open_raises(self):
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        # No open_source: wait_source has no error record, so it delegates to
        # the worker wait (which on a real worker would raise for a layer with
        # no slot mapping). Here the stub just records the call.
        consumer.wait_source(_source(5))
        self.assertEqual(worker.waits, [5])

    def test_abort_marks_failure_and_completes_worker_reuse(self):
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        source = _source(3)
        consumer.open_source(source)
        consumer.abort_source(source, RuntimeError("injected"))

        # A later wait on the aborted generation surfaces the error before
        # delegating to the worker wait.
        with self.assertRaisesRegex(RuntimeError, "source generation failed"):
            consumer.wait_source(source)
        self.assertEqual(worker.completions, [(3, "injected")])

    def test_retire_drops_generation_record(self):
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        source = _source(1)
        consumer.open_source(source)
        consumer.source_ready(source, (), ready_event=SimpleNamespace())
        self.assertTrue(consumer.is_published(1))
        consumer.retire(1)
        self.assertFalse(consumer.is_published(1))

    def test_source_ready_before_open_raises(self):
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        with self.assertRaisesRegex(RuntimeError, "source_ready before open_source"):
            consumer.source_ready(_source(0), (), ready_event=None)

    def test_hand_off_after_source_ready_releases_immediately(self):
        # source_ready fires first; save_kv_layer hands off afterwards. The task
        # is released right away with the KV-ready fence attached.
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        source = _source(4)
        consumer.open_source(source)
        ready = SimpleNamespace(name="ready-event")
        consumer.source_ready(source, (), ready_event=ready)
        original = SimpleNamespace(name="attention-or-quantization-event")
        task = SimpleNamespace(wait_event=original)
        consumer.hand_off_send_task(4, task)
        self.assertIs(worker.send_queue.get_nowait(), task)
        self.assertIs(task.source_ready_event, ready)
        self.assertIs(task.wait_event, original)

    def test_hand_off_before_source_ready_holds_then_releases(self):
        # save_kv_layer hands off first; source_ready releases it. The task is
        # not enqueued until source_ready fires (readers registered first).
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        source = _source(6)
        consumer.open_source(source)
        task = SimpleNamespace(wait_event=None)
        consumer.hand_off_send_task(6, task)
        self.assertTrue(worker.send_queue.empty())
        ready = SimpleNamespace(name="ready-event")
        consumer.source_ready(source, (), ready_event=ready)
        self.assertIs(worker.send_queue.get_nowait(), task)
        self.assertIs(task.source_ready_event, ready)
        self.assertIsNone(task.wait_event)

    def test_hand_off_without_open_generation_falls_back_to_direct_enqueue(self):
        # No open_source for the layer: the consumer must not swallow the task.
        worker = StubWorker()
        consumer = LayerwiseSourceConsumer(worker)
        task = SimpleNamespace(wait_event=None)
        consumer.hand_off_send_task(9, task)
        self.assertIs(worker.send_queue.get_nowait(), task)


if __name__ == "__main__":
    unittest.main()

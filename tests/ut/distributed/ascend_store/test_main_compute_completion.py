# SPDX-License-Identifier: Apache-2.0
"""Prove Store completion cannot replace attention's last-read dependency."""

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import AttentionComputeCompletion
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerLoadTask,
    LayerRangeReqMeta,
    LayerTransferTask,
    SharedBlockData,
)


class TestAttentionComputeCompletion(unittest.TestCase):
    def test_unrecorded_event_is_not_synchronized_or_treated_as_complete(self):
        event = MagicMock()
        completion = AttentionComputeCompletion(event)
        self.assertFalse(completion.wait(timeout=0))
        event.synchronize.assert_not_called()

    def test_record_is_idempotent_and_wait_synchronizes_device_work(self):
        event = MagicMock()
        completion = AttentionComputeCompletion(event)
        completion.record()
        completion.record()
        self.assertTrue(completion.wait(timeout=0))
        event.record.assert_called_once_with()
        event.synchronize.assert_called_once_with()

    def test_failure_wakes_waiters_without_reporting_reuse_success(self):
        event = MagicMock()
        completion = AttentionComputeCompletion(event)
        error = RuntimeError("attention failed")
        completion.fail(error)
        for action in (lambda: completion.wait(timeout=0), completion.record):
            with self.assertRaisesRegex(RuntimeError, "Attention completion failed") as raised:
                action()
            self.assertIs(raised.exception.__cause__, error)
        event.record.assert_not_called()
        event.synchronize.assert_not_called()

    def test_record_failure_is_retained(self):
        event = MagicMock()
        event.record.side_effect = RuntimeError("NPU record failed")
        completion = AttentionComputeCompletion(event)
        with self.assertRaisesRegex(RuntimeError, "NPU record failed"):
            completion.record()
        with self.assertRaisesRegex(RuntimeError, "Attention completion failed") as raised:
            completion.wait(timeout=0)
        self.assertIs(raised.exception.__cause__, event.record.side_effect)

    def test_old_step_record_does_not_release_new_step(self):
        old = AttentionComputeCompletion(MagicMock())
        current = AttentionComputeCompletion(MagicMock())
        old.record()
        self.assertTrue(old.wait(timeout=0))
        self.assertFalse(current.wait(timeout=0))


class TestMainComputeReceiver(unittest.TestCase):
    def test_store_done_and_recorded_event_do_not_allow_get_until_attention_finishes(self):
        from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerReceivingTaskOwnership

        thread, finished, saves, _ = TestGVALayerReceivingTaskOwnership()._make_thread()
        saves[0].set()
        wait_entered = threading.Event()
        device_wait_entered = threading.Event()
        device_done = threading.Event()
        errors = []
        order = []
        event = MagicMock()

        def synchronize():
            device_wait_entered.set()
            if not device_done.wait(timeout=5):
                raise RuntimeError("test device work did not finish")
            order.append("compute")

        event.synchronize.side_effect = synchronize

        class ObservedCompletion(AttentionComputeCompletion):
            def wait(self, timeout=10.0):
                wait_entered.set()
                return super().wait(timeout)

        completion = ObservedCompletion(event)
        thread.external_slot_release_waiter = lambda layer: order.append("pd")
        thread.group_builders[0].build_addrs.return_value = LayerRangeReqMeta(
            req_ids=["r1"],
            layer_id=1,
            block_ids=[0],
            keys=["key"],
            all_buffers=[[10]],
            all_sizes=[[16]],
            all_offsets=[[0]],
        )

        def copy_get(*args):
            order.append("get")
            return [16]

        thread.m_store.batch_copy_get.side_effect = copy_get
        task = LayerTransferTask(
            layer_id=1,
            block_ranges=[],
            use_key_major_ranges=True,
            shared_block_data=SharedBlockData(
                block_ids_arr=np.asarray([0]),
                block_gvas_arr=None,
                req_ids=["r1"],
                is_last_chunks=[False],
            ),
        )
        load = LayerLoadTask(
            layer_id=1,
            wait_for_save_layer=0,
            transfer_tasks=[task],
            previous_compute_done=completion,
        )
        thread.request_queue.put(load)

        def receive():
            try:
                thread._handle_request(load)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=receive)
        worker.start()
        try:
            self.assertTrue(wait_entered.wait(timeout=5))
            event.synchronize.assert_not_called()
            self.assertFalse(finished[1].is_set())
            thread.m_store.batch_copy_get.assert_not_called()
            completion.record()
            self.assertTrue(device_wait_entered.wait(timeout=5))
            self.assertFalse(finished[1].is_set())
            thread.m_store.batch_copy_get.assert_not_called()
        finally:
            completion.record()
            device_done.set()
            worker.join(timeout=5)
            if worker.is_alive():
                thread.stop()
                worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(order, ["compute", "pd", "get"])
        self.assertTrue(finished[1].is_set())

    def test_empty_get_still_propagates_compute_failure(self):
        from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerReceivingTaskOwnership

        thread, _, _, _ = TestGVALayerReceivingTaskOwnership()._make_thread()
        completion = AttentionComputeCompletion(MagicMock())
        completion.fail(RuntimeError("previous attention failed"))
        load = LayerLoadTask(
            layer_id=1,
            wait_for_save_layer=None,
            transfer_tasks=[],
            previous_compute_done=completion,
        )
        thread.request_queue.put(load)
        with self.assertRaisesRegex(RuntimeError, "Attention completion failed"):
            thread._handle_request(load)
        thread.m_store.batch_copy_get.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            thread.raise_if_failed()

    def test_completed_compute_precedes_external_wait_for_empty_get(self):
        from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerReceivingTaskOwnership

        order = []
        thread, finished, _, _ = TestGVALayerReceivingTaskOwnership()._make_thread(
            external_slot_release_waiter=lambda layer: order.append("pd")
        )
        event = MagicMock()
        event.synchronize.side_effect = lambda: order.append("compute")
        completion = AttentionComputeCompletion(event)
        completion.record()
        load = LayerLoadTask(
            layer_id=1,
            wait_for_save_layer=None,
            transfer_tasks=[],
            previous_compute_done=completion,
        )
        thread.request_queue.put(load)
        thread._handle_request(load)
        self.assertEqual(order, ["compute", "pd"])
        self.assertTrue(finished[1].is_set())

    def test_receiver_stop_cannot_turn_unfinished_attention_into_success(self):
        from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerReceivingTaskOwnership

        thread, _, _, _ = TestGVALayerReceivingTaskOwnership()._make_thread()
        completion = AttentionComputeCompletion(MagicMock())
        load = LayerLoadTask(
            layer_id=1,
            wait_for_save_layer=None,
            transfer_tasks=[],
            previous_compute_done=completion,
        )
        thread.request_queue.put(load)
        thread.stop()
        with self.assertRaisesRegex(RuntimeError, "Receiver stopped before previous attention completed"):
            thread._handle_request(load)
        self.assertFalse(completion.wait(timeout=0))
        thread.m_store.batch_copy_get.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            thread.raise_if_failed()


class TestMainComputeWorker(unittest.TestCase):
    def test_step_drain_does_not_rewait_a_consumed_reuse_save_event(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        worker = make_worker(self, num_layers=2, use_layerwise=True)
        worker.use_block_key_layerwise = True
        worker.block_key_hybrid = False
        worker.prefetch_layer_map = {1: 0}
        consumed = MagicMock()
        consumed.wait.side_effect = AssertionError("Reuse already consumed this event")
        final = threading.Event()
        final.set()
        worker.layer_save_finished_events = [consumed, final]
        sender = MagicMock(request_queue=queue.Queue())
        worker._wait_for_final_layer_save(2, sender)
        consumed.wait.assert_not_called()
        self.assertFalse(final.is_set())

    def test_reuse_without_external_waiter_fails_before_starting_transfer_threads(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        worker = make_worker(self, num_layers=2, use_layerwise=True)
        worker._main_buffer_layer_slots = {0: 0, 1: 0}
        self.assertFalse(worker._transfer_threads_started)
        with self.assertRaisesRegex(RuntimeError, "requires an external slot-release waiter"):
            worker._start_kv_transfer_threads()
        self.assertFalse(worker._transfer_threads_started)
        self.assertIsNone(worker.kv_recv_thread)
        self.assertTrue(worker.set_external_slot_release_waiter(lambda layer: None))
        worker.assert_reuse_wired()

    def test_running_receiver_must_have_the_registered_waiter(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        worker = make_worker(self, num_layers=2, use_layerwise=True)
        worker._main_buffer_layer_slots = {0: 0, 1: 0}
        worker.external_slot_release_waiter = lambda layer: None
        worker._transfer_threads_started = True
        worker.kv_recv_thread = None
        with self.assertRaisesRegex(RuntimeError, "not installed on the Store receiver"):
            worker.assert_reuse_wired()

    def test_slot_tails_are_carried_across_steps_without_reusing_completion_objects(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        worker = make_worker(self, num_layers=2, use_layerwise=True)
        worker._main_buffer_layer_slots = {0: 0, 1: 0}
        worker.process_layer_data = MagicMock()
        worker.kv_recv_thread = MagicMock()
        worker.num_prefetch_layers = 1
        metadata = SimpleNamespace(requests=[])

        worker.prepare_layerwise_step(metadata)
        old = dict(worker._main_layer_compute_done)
        worker._submit_ready_layer_loads()
        tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
        self.assertIsNone(tasks[0].previous_compute_done)
        self.assertIs(tasks[1].previous_compute_done, old[0])

        worker.kv_recv_thread.reset_mock()
        worker.prepare_layerwise_step(metadata)
        current = worker._main_layer_compute_done
        worker._submit_ready_layer_loads()
        tasks = [call.args[0] for call in worker.kv_recv_thread.add_request.call_args_list]
        self.assertIs(tasks[0].previous_compute_done, old[1])
        self.assertIs(tasks[1].previous_compute_done, current[0])
        old[1].record()
        self.assertFalse(current[1].wait(timeout=0))

    def test_post_attention_hook_records_even_when_store_was_submitted_early(self):
        from tests.ut.distributed.ascend_store.test_pool_worker import make_worker

        worker = make_worker(self, num_layers=2, use_layerwise=True)
        completion = AttentionComputeCompletion(MagicMock())
        worker._main_buffer_layer_slots = {0: 0, 1: 0}
        worker._main_layer_compute_done = {0: completion}
        worker._attention_saved_layers = {0}
        worker.kv_send_thread = MagicMock()
        worker.layer_save_finished_events = [threading.Event(), threading.Event()]
        worker.sync_save_events = [MagicMock(), MagicMock()]
        worker.save_kv_layer(SimpleNamespace())
        self.assertTrue(completion.wait(timeout=0))
        worker.sync_save_events[0].record.assert_not_called()

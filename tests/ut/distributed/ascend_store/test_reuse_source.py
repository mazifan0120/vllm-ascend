# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import (
    RequestSource,
    ReuseSourceRegistry,
    SourceView,
)


class ControlledReader:
    def __init__(self):
        self.states = {}
        self.ready = []
        self.error = None

    def open_source(self, source):
        self.states[source.epochs] = [False, 2]

    def source_ready(self, source, requests, event):
        self.ready.append((source, requests, event))

    def seal_source(self, source):
        self.states[source.epochs][0] = True

    def wait_source(self, source):
        sealed, pending = self.states[source.epochs]
        if self.error:
            raise self.error
        if not sealed or pending:
            raise RuntimeError("reader not quiescent")

    def abort_source(self, source, error):
        self.error = error


class TestReuseSource(unittest.TestCase):
    def make(self):
        view = SourceView(0, "MAIN", 0, "main:0", 100, 256, (8, 32), (32, 1), "int8", 32, 32, 0)
        reader = ControlledReader()
        registry = ReuseSourceRegistry({0: (view,), 1: (replace(view, layer=1),)}, reader)
        request = RequestSource("external", "instance", 0, 47, ((3, 5),))
        return registry, reader, request

    def test_open_and_partial_batch_completion_cannot_release_slot(self):
        registry, reader, request = self.make()
        registry.begin_write(0)
        old = registry._layers[0]
        with self.assertRaisesRegex(RuntimeError, "not quiescent"):
            registry.begin_write(1)
        event = object()
        registry.publish(0, (request, replace(request, request_instance="second")), event)
        self.assertIs(reader.ready[0][2], event)
        reader.states[old.epochs][1] = 1
        with self.assertRaisesRegex(RuntimeError, "not quiescent"):
            registry.begin_write(1)
        reader.states[old.epochs][1] = 0
        registry.begin_write(1)
        self.assertEqual(registry._layers[1].epochs[0].generation, 2)
        # A late completion belongs to the old immutable epoch only.
        reader.states[old.epochs][1] = 0
        with self.assertRaisesRegex(RuntimeError, "not quiescent"):
            registry.begin_write(0)

    def test_duplicate_ready_does_not_resend_or_seal_twice(self):
        registry, reader, request = self.make()
        registry.begin_write(0)
        registry.publish(0, (request,), object())
        registry.publish(0, (request,), object())
        self.assertEqual(len(reader.ready), 1)

    def test_failed_publish_retains_failure_and_forbids_future_write(self):
        registry, reader, request = self.make()
        registry.begin_write(0)
        reader.source_ready = MagicMock(side_effect=RuntimeError("enqueue failed"))
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            registry.publish(0, (request,), object())
        for action in (
            lambda: registry.begin_write(1),
            registry.drain,
            lambda: registry.publish(0, (request,), object()),
        ):
            with self.assertRaisesRegex(RuntimeError, "remain retained"):
                action()
        self.assertEqual(len(registry._occupants), 1)

    def test_invalid_source_range_is_rejected_before_sender_sees_it(self):
        registry, reader, request = self.make()
        for invalid in (
            replace(request, end_token=65),
            replace(request, block_ids_by_group=((3, 8),)),
            replace(request, request_instance=""),
        ):
            registry, reader, _ = self.make()
            registry.begin_write(0)
            with self.assertRaises(ValueError):
                registry.publish(0, (invalid,), object())
            self.assertEqual(reader.ready, [])
            with self.assertRaisesRegex(RuntimeError, "remain retained"):
                registry.drain()

    def test_abort_notifies_consumer_without_turning_failure_into_release(self):
        registry, reader, _ = self.make()
        registry.begin_write(0)
        source = registry._layers[0]
        error = RuntimeError("GET failed")
        registry.fail(error)
        self.assertIs(reader.error, error)
        reader.states[source.epochs] = [True, 0]
        with self.assertRaisesRegex(RuntimeError, "remain retained"):
            registry.drain()
        with self.assertRaisesRegex(RuntimeError, "remain retained"):
            registry.begin_write(1)

    def test_zero_payload_still_needs_seal(self):
        registry, reader, _ = self.make()
        registry.begin_write(0)
        reader.states[registry._layers[0].epochs][1] = 0
        with self.assertRaisesRegex(RuntimeError, "not quiescent"):
            registry.drain()
        registry.publish(0, (), object())
        registry.drain()

    def test_worker_exports_component_views_from_actual_layout_and_tensor_stride(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker._reuse_source_consumer = ControlledReader()
        worker._main_buffer_layer_slots = {0: 0, 1: 0}
        worker._global_to_local_layer = {0: 0, 1: 1}
        spec = SimpleNamespace(block_size=32)
        worker._layerwise_reuse_layout = SimpleNamespace(
            layer_cache_specs={
                0: SimpleNamespace(
                    main=SimpleNamespace(layer_name="main0", spec=spec),
                    indexer=SimpleNamespace(layer_name="index0", spec=spec),
                ),
                1: SimpleNamespace(main=SimpleNamespace(layer_name="main1", spec=spec), indexer=None),
            }
        )
        worker.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(layer_names=["main0", "main1", "index0"])]
        )
        main = MagicMock(shape=(8, 16, 2), dtype="int8")
        main.data_ptr.return_value = 100
        main.stride.return_value = (32, 2, 1)
        main.element_size.return_value = 1
        index = MagicMock(shape=(8, 16, 2), dtype="int8")
        index.data_ptr.return_value = 500
        index.stride.return_value = (32, 2, 1)
        index.element_size.return_value = 1
        worker.kv_caches = {"main0": main, "main1": main, "index0": index}
        worker._as_cache_tuple = lambda cache: (cache,)
        worker._get_cache_block_metadata = lambda cache: (32, 32, 256, 2)
        worker._get_storage_key = lambda cache: cache.data_ptr()
        worker._register_reuse_source_views()
        views = worker._reuse_source_registry.views
        self.assertEqual(views[0][0].slot, views[1][0].slot)
        self.assertEqual(views[0][0].address, views[1][0].address)
        self.assertEqual(views[0][1].component, "INDEXER")
        self.assertNotEqual(views[0][0].slot, views[0][1].slot)
        self.assertEqual(views[0][1].strides_bytes, (32, 2, 1))
        self.assertEqual(views[0][0].page_tokens, 16)

    def test_worker_source_range_includes_the_valid_non_block_aligned_tail(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        request = ReqMeta(
            "internal-request",
            save_end_token=32,
            target_token_len=47,
            block_ids_by_group=[[1, 3]],
            request_instance="private-store-instance",
        )
        source = KVPoolWorker._source_requests([request])[0]
        self.assertEqual(source.end_token, 47)
        self.assertEqual(source.block_ids_by_group, ((1, 3),))
        self.assertEqual(source.request_instance, "private-store-instance")
        self.assertEqual(request.save_end_token, 32)

    def test_restore_only_runs_all_layers_without_forward_and_rejects_new_tokens(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker._reuse_source_registry = object()
        worker.num_layers = 3
        worker.current_layer = 0
        worker._attention_saved_layers = set()
        order = []
        worker.drain_reuse_sources = lambda: order.append("drain")
        worker.prepare_layerwise_step = lambda metadata: order.append("prepare")
        worker.wait_for_layer_load = lambda: order.append(("load", worker.current_layer))
        worker._submit_attention_save = lambda layer: order.append(("ready", layer))

        def save(metadata):
            order.append(("done", worker.current_layer))
            worker.current_layer += 1

        worker.save_kv_layer = save
        request = SimpleNamespace(request_instance="instance", restore_end_token=47, token_len_chunk=47)
        metadata = SimpleNamespace(requests=[request])
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_attention_compute_start_gate"
        ) as gate:
            worker.restore_kv_for_transfer(metadata)
            self.assertEqual(gate.return_value.record.call_count, 3)
        self.assertEqual(
            order,
            [
                "drain",
                "prepare",
                ("load", 0),
                ("ready", 0),
                ("done", 0),
                ("load", 1),
                ("ready", 1),
                ("done", 1),
                ("load", 2),
                ("ready", 2),
                ("done", 2),
                "drain",
            ],
        )
        request.restore_end_token = 32
        with self.assertRaisesRegex(ValueError, "entire valid"):
            worker.restore_kv_for_transfer(metadata)
        # The public save range may end at the preceding block. The driver
        # checks actual restored valid tokens, including a private tail.
        request.token_len_chunk = 32
        request.target_token_len = request.restore_end_token = 47
        worker.current_layer = 0
        order.clear()
        with patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.get_attention_compute_start_gate"
        ):
            worker.restore_kv_for_transfer(metadata)
        self.assertIn(("load", 2), order)

    def test_multi_connector_installs_only_explicit_version_one_consumer(self):
        # Only the unavailable upstream base is stubbed; execute the real
        # AscendMultiConnector registration method rather than copying it.
        module = SimpleNamespace(MultiConnector=type("MultiConnector", (), {}))
        with patch.dict(sys.modules, {"vllm.distributed.kv_transfer.kv_connector.v1.multi_connector": module}):
            from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector

        provider = SimpleNamespace(reuse_source_contract_version=1)
        sink = SimpleNamespace(set_reuse_source_consumer=MagicMock())
        multi = AscendMultiConnector.__new__(AscendMultiConnector)
        multi._layerwise_slot_release_providers = [provider]
        multi._connectors = [provider, sink]
        multi._external_slot_release_sink_configured = True
        multi._configure_reuse_source_contract()
        sink.set_reuse_source_consumer.assert_called_once_with(provider)
        multi._layerwise_slot_release_providers.append(SimpleNamespace(reuse_source_contract_version=1))
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            multi._configure_reuse_source_contract()
        multi._layerwise_slot_release_providers = [provider]
        multi._external_slot_release_sink_configured = False
        with self.assertRaisesRegex(RuntimeError, "wired Store"):
            multi._configure_reuse_source_contract()

    def test_plain_storeless_transport_capability_does_not_opt_in_to_source_reuse(self):
        module = SimpleNamespace(MultiConnector=type("MultiConnector", (), {}))
        with patch.dict(sys.modules, {"vllm.distributed.kv_transfer.kv_connector.v1.multi_connector": module}):
            from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector

        provider = SimpleNamespace(reuse_source_contract_version=1)
        multi = AscendMultiConnector.__new__(AscendMultiConnector)
        multi._layerwise_slot_release_providers = [provider]
        multi._connectors = [provider]
        multi._external_slot_release_sink_configured = False
        multi._configure_reuse_source_contract()
        # A regular Store also exposes the setter, but has no shared Main
        # layout from which SourceViews could be exported.
        store = SimpleNamespace(wants_reuse_source_contract=lambda: False, set_reuse_source_consumer=MagicMock())
        multi._connectors.append(store)
        multi._configure_reuse_source_contract()
        store.set_reuse_source_consumer.assert_not_called()

    def test_empty_get_dispatch_opens_epoch_after_external_wait(self):
        from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerReceivingTaskOwnership
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LayerLoadTask

        receiver, finished, _, _ = TestGVALayerReceivingTaskOwnership()._make_thread()
        order = []
        receiver.external_slot_release_waiter = lambda layer: order.append("external")
        task = LayerLoadTask(None, [], 0, before_slot_write=lambda: order.append("open"))
        receiver.request_queue.put(task)
        receiver._handle_request(task)
        self.assertEqual(order, ["external", "open"])
        self.assertTrue(finished[0].is_set())

    def test_receiver_opens_epoch_after_all_waits_before_get_and_propagates_failure(self):
        from tests.ut.distributed.ascend_store.test_kv_transfer import TestGVALayerReceivingTaskOwnership
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LayerLoadTask

        receiver, _, _, _ = TestGVALayerReceivingTaskOwnership()._make_thread()
        order = []
        receiver.external_slot_release_waiter = lambda layer: order.append("external")
        task = LayerLoadTask(None, [], 0, before_slot_write=lambda: order.append("open"))
        receiver.request_queue.put(task)
        receiver._handle_range_layer_task(task)
        self.assertEqual(order, ["external", "open"])
        task.slot_write_failed = MagicMock()
        task.before_slot_write = MagicMock(side_effect=RuntimeError("open failed"))
        receiver.request_queue.put(task)
        with self.assertRaisesRegex(RuntimeError, "open failed"):
            receiver._handle_range_layer_task(task)
        receiver.m_store.batch_copy_get.assert_not_called()
        task.slot_write_failed.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "asynchronous transfer"):
            receiver.raise_if_failed()

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.session_tracker import (
    LayerwiseSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVCacheStoreLayerSendingThread,
    KVTransferThread,
    LayerBatchBuilder,
    _build_range_debug_payload,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerBlockRange,
    LayerLoadTask,
    LayerRangeReqMeta,
    LayerTransferTask,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefill_reuse import SlotBarrier

# isort: on


def make_token_database() -> ChunkedTokenDatabase:
    database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None)
    database.set_group_buffers(
        {0: [1000, 2000, 3000]},
        {0: [10, 20, 30]},
        {0: [100, 200, 300]},
        group_num_layers={0: 2},
        group_layer_cache_entry_offsets={0: [0, 2, 3]},
    )
    return database


class TestMooncakeLayerBatchBuilder(unittest.TestCase):
    def test_prefill_tail_respects_segment_boundaries_and_keeps_object_offsets(self):
        database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [4], None)
        database.set_group_buffers(
            {0: [1000, 1100, 2000]},
            {0: [4, 4, 4]},
            {0: [256, 256, 4]},
            group_num_layers={0: 1},
            group_cache_entry_segments={0: [(0, 2), (1, 2), (0, 1)]},
        )
        builder = LayerBatchBuilder(database, page_size_bytes=12, num_layers=1)
        cut = ReqMeta("cut", block_ids=[1], load_block_keys=["public"])
        cut.prefill_history = 1
        cut.range_tokens = {"public": 1}
        full = ReqMeta("full", block_ids=[2], load_block_keys=["public"])
        full.prefill_history = 4
        task = LayerTransferTask(
            0,
            [LayerBlockRange(cut, 0, 1), LayerBlockRange(full, 0, 1)],
            use_key_major_ranges=True,
        )
        metadata = builder.build(task, is_save=False)
        self.assertIsInstance(metadata, LayerRangeReqMeta)
        self.assertEqual(metadata.all_buffers, [[1256, 2004], [1512, 1612, 2008]])
        self.assertEqual(metadata.all_sizes, [[2, 1], [4, 4, 4]])
        self.assertEqual(metadata.all_offsets, [[0, 8], [0, 4, 8]])

    def test_range_debug_payload_reports_per_key_bytes_and_offsets(self):
        payload = _build_range_debug_payload(
            "save",
            3,
            [[10, 20], [7]],
            [[30, 40], [50]],
            [30, -1],
        )

        self.assertEqual(payload["event"], "range")
        self.assertEqual(payload["layer_id"], 3)
        self.assertEqual(payload["requested_bytes"], [30, 7])
        self.assertEqual(payload["object_offsets"], [[30, 40], [50]])
        self.assertEqual(payload["results"], [30, -1])

    def test_key_major_ranges_use_full_object_offsets(self):
        request = ReqMeta("r1", block_ids=[2], block_hashes=[])
        request.save_block_keys = ["key"]
        task = LayerTransferTask(
            layer_id=1,
            layer_idx_in_group=1,
            block_ranges=[LayerBlockRange(request, 0, 1)],
            use_key_major_ranges=True,
        )
        builder = LayerBatchBuilder(make_token_database(), page_size_bytes=60, num_layers=2)

        result = builder.build(task)

        self.assertIsInstance(result, LayerRangeReqMeta)
        assert isinstance(result, LayerRangeReqMeta)
        self.assertEqual(result.keys, ["key"])
        self.assertEqual(result.block_ids, [2])
        self.assertEqual(result.all_buffers, [[3600]])
        self.assertEqual(result.all_sizes, [[30]])
        self.assertEqual(result.all_offsets, [[30]])

    def test_range_limits_split_rows_and_large_segments(self):
        batches = KVTransferThread._range_transfer_batches(
            ["k0", "k1"],
            [[100], [200]],
            [[25], [5]],
            [[1000], [2000]],
            max_transfer_blocks=1,
            max_transfer_bytes=10,
        )

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], (["k0"], [[100, 110, 120]], [[10, 10, 5]], [[1000, 1010, 1020]]))
        self.assertEqual(batches[1], (["k1"], [[200]], [[5]], [[2000]]))


class TestMooncakeLayerSaveSession(unittest.TestCase):
    def test_final_layer_commits_after_all_ranges(self):
        store = MagicMock()
        # Range APIs may return the positive number of bytes moved on success.
        store.batch_copy_put.return_value = [30]
        store.batch_commit.return_value = [0]
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("key", 0)])
        save_finished = [threading.Event(), threading.Event()]
        builder = LayerBatchBuilder(make_token_database(), page_size_bytes=60, num_layers=2)
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=make_token_database(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=save_finished,
            sync_save_events=[MagicMock(), MagicMock()],
            group_builders=[builder],
            put_started_keys={"key"},
            session_tracker=tracker,
        )
        request = ReqMeta("r1", block_ids=[2], block_hashes=[], is_last_chunk=True)
        request.save_block_keys = ["key"]

        for layer_id in range(2):
            task = LayerTransferTask(
                layer_id=layer_id,
                layer_idx_in_group=layer_id,
                block_ranges=[LayerBlockRange(request, 0, 1)],
                shared_block_data=builder.build_shared(
                    LayerTransferTask(
                        layer_id=layer_id,
                        block_ranges=[LayerBlockRange(request, 0, 1)],
                        use_key_major_ranges=True,
                    )
                ),
                use_key_major_ranges=True,
            )
            thread.add_stored_request("r1")
            thread.request_queue.put([task])
            thread._handle_request([task])

        self.assertEqual(store.batch_copy_put.call_count, 2)
        store.batch_commit.assert_called_once_with(["key"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("key", 0)])


class TestMooncakeLayerLoad(unittest.TestCase):
    def test_prepare_addresses_before_gate_and_load_each_layers_selection(self):
        db = make_token_database()
        builder = LayerBatchBuilder(db, page_size_bytes=60, num_layers=2)
        build_addrs = MagicMock(wraps=builder.build_addrs)
        builder.build_addrs = build_addrs
        store = MagicMock()
        slots = SlotBarrier(((0,), (1,)))
        recv = KVCacheStoreLayerRecvingThread(
            store,
            db,
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            ready_event=threading.Event(),
            get_event=threading.Event(),
            layer_load_finished_events=[threading.Event(), threading.Event()],
            layer_save_finished_events=[],
            sync_save_events=[],
            num_layers=2,
            group_builders=[builder],
        )
        recv.prefill_slot_provider = lambda: slots
        request = ReqMeta("r1", block_ids=[2, 3], load_block_keys=["k0", "k1"], is_last_chunk=True)
        store.batch_copy_get.side_effect = lambda keys, *_: [0] * len(keys)

        for layer, count in enumerate((1, 2)):
            def wait_for_attention(layer_id=layer, **_):
                self.assertEqual(build_addrs.call_count, layer_id + 1)
                self.assertEqual(store.batch_copy_get.call_count, layer_id)
                return True

            gate = MagicMock()
            gate.wait.side_effect = wait_for_attention
            task = LayerTransferTask(
                layer, [LayerBlockRange(request, 0, count)], layer_idx_in_group=layer, use_key_major_ranges=True
            )
            task.shared_block_data = builder.build_shared(task, is_save=False)
            load = LayerLoadTask(None, [task], layer, gate)
            recv.request_queue.put(load)
            recv._handle_request(load)
            self.assertEqual(store.batch_copy_get.call_args.args[0], request.load_block_keys[:count])


class TestMooncakeWorkerSessionPreparation(unittest.TestCase):
    @staticmethod
    def _make_worker() -> KVPoolWorker:
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.dcp_size = 1
        worker.tp_rank = 0
        worker.pp_rank = 0
        worker.put_step = 1
        worker.dcp_size = 1
        worker.group_uses_align_state = [False]
        worker.block_size = 16
        worker.grouped_block_size = [16]
        worker.num_layers = 1
        worker.hash_block_size = 16
        worker.model_name = "model"
        worker.head_or_tp_rank = 0
        worker.metadata = [KeyMetadata("model", 0, 0, 0)]
        worker.backend_name = "mooncake"
        worker.layerwise_protocol = mooncake_layerwise
        worker.layerwise_keys = mooncake_layerwise.bind_layerwise_keys(
            vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1)),
            kv_cache_config=None,
            model_name=worker.model_name,
            use_hybrid=False,
            grouped_block_size=worker.grouped_block_size,
        )
        worker.use_block_key_layerwise = True
        worker.layerwise_offload = False
        worker.independent_layers = []
        worker.page_size_bytes = 60
        worker.group_block_len = {0: [10, 20, 30]}
        worker.layerwise_max_transfer_blocks = 0
        worker.use_eagle = False
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._layerwise_session_tracker = LayerwiseSessionTracker()
        worker.m_store = MagicMock()
        return worker

    def _make_prefill_snapshot_worker(self):
        worker = self._make_worker()
        worker.prefill_slot_provider = MagicMock()
        worker.block_key_hybrid = False
        worker.num_kv_cache_groups = 1
        worker.cacheable_group_ids = [0]
        worker.grouped_block_size = [4]
        worker.hash_block_size = 4
        worker.group_block_len = {0: [8, 4]}
        worker._layer_load_aborted = threading.Event()
        worker._load_session_lock = threading.Lock()
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker.m_store.requires_exists_before_put = True
        worker.m_store.exists.side_effect = lambda keys: [0] * len(keys)
        worker.m_store.batch_put_start.side_effect = lambda keys, sizes: [0] * len(keys)
        worker.m_store.batch_get_end.return_value = 0
        readable = set()
        worker.m_store.batch_get_start.side_effect = lambda keys: [0 if key in readable else -1 for key in keys]
        return worker, readable

    def test_prefill_tail_requires_commit_then_promotes_to_public_block(self):
        worker, readable = self._make_prefill_snapshot_worker()
        tracker = worker._layerwise_session_tracker

        def prepare(history, extent, *, final=False):
            request = ReqMeta(
                "r1",
                block_ids=[7, 8],
                block_hashes=[b"complete"],
                target_token_len=extent,
                can_save=True,
                is_last_chunk=final,
            )
            request.prefill_history = history
            request.prefill_snapshot_id = str(extent)
            return mooncake_layerwise.prepare_layerwise_sessions(worker, [request])[0][0]

        first = prepare(0, 2)
        private = first.save_block_keys[0]
        self.assertIn("prefill_private_", private)
        self.assertEqual(first.range_tokens, {private: 2})
        self.assertEqual(tracker.private_blocks("r1", 0), {})
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            prepare(2, 5)
        readable.add(private)
        tracker.commit_put_keys([private])
        second = prepare(2, 5)
        self.assertEqual(second.load_block_keys, [private])
        public, next_private = second.save_block_keys
        self.assertEqual(public, worker._make_layerwise_full_key(0, b"complete".hex()))
        self.assertNotEqual(private, next_private)
        self.assertEqual(second.range_tokens, {private: 2, next_private: 1})
        readable.update([public, next_private])
        tracker.commit_put_keys([public, next_private])
        worker._finish_current_layerwise_load_sessions()
        final = prepare(5, 6, final=True)
        self.assertEqual(final.load_block_keys, [public, next_private])
        self.assertEqual(final.range_tokens, {next_private: 1})
        self.assertEqual(final.save_block_keys, [None])
        self.assertEqual(tracker.collect_private_removals(), [private])

    def test_replicated_private_snapshot_has_one_writer_and_store_gates_followers(self):
        owner, readable = self._make_prefill_snapshot_worker()
        follower, _ = self._make_prefill_snapshot_worker()
        owner.put_step = follower.put_step = 2
        follower.tp_rank = 1
        follower.m_store.batch_get_start.side_effect = owner.m_store.batch_get_start.side_effect

        def prepare(worker, history, extent, version, *, final=False):
            request = ReqMeta(
                "r1", block_ids=[7], block_hashes=[], target_token_len=extent, can_save=True, is_last_chunk=final
            )
            request.prefill_history = history
            request.prefill_snapshot_id = version
            return mooncake_layerwise.prepare_layerwise_sessions(worker, [request])[0][0]

        first = prepare(owner, 0, 2, "first")
        key = first.save_block_keys[0]
        self.assertIn("prefill_private_first:0", key)
        follower_view = prepare(follower, 0, 2, "first")
        self.assertEqual(follower_view.save_block_keys, [None])
        follower.m_store.batch_put_start.assert_not_called()
        for worker, view in ((owner, first), (follower, follower_view)):
            worker.layer_save_tasks = [[]]
            worker._process_save_for_layer_batch([view], 0)
            self.assertEqual(bool(worker.layer_save_tasks[0]), worker is owner)
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            prepare(follower, 2, 3, None, final=True)

        owner._layerwise_session_tracker.commit_put_keys([key])
        readable.add(key)
        for worker in (owner, follower):
            view = prepare(worker, 2, 3, None, final=True)
            self.assertEqual(view.load_block_keys, [key])
            self.assertEqual(view.range_tokens, {key: 2})
            worker._finish_current_layerwise_load_sessions()

        follower._retire_prefill_snapshots()
        follower.m_store.retire_private_keys.assert_not_called()
        self.assertEqual(follower._layerwise_session_tracker.private_blocks("r1", 0), {})
        owner.m_store.retire_private_keys.side_effect = [[], [key]]
        owner._retire_prefill_snapshots()
        self.assertEqual(owner._layerwise_session_tracker.collect_private_removals(), [key])
        owner._retire_prefill_snapshots()
        self.assertEqual(owner.m_store.retire_private_keys.call_count, 2)
        self.assertEqual(owner._layerwise_session_tracker.collect_private_removals(), [])

    def test_mtp_snapshot_uses_successor_hash_and_keeps_request_tail_private(self):
        worker, readable = self._make_prefill_snapshot_worker()
        worker.use_eagle = True

        def prepare(req_id, history, extent, hashes, *, final=False):
            request = ReqMeta(
                req_id,
                block_ids=[7, 8, 9],
                block_hashes=hashes,
                target_token_len=extent,
                num_prompt_tokens=8,
                can_save=True,
                is_last_chunk=final,
            )
            request.prefill_history = history
            request.prefill_snapshot_id = f"{req_id}:{extent}"
            return mooncake_layerwise.prepare_layerwise_sessions(worker, [request])[0][0]

        # Generated token hashes cannot make the final prompt page public.
        hashes = [b"first", b"second", b"generated"]
        first = prepare("r1", 0, 8, hashes)
        public, private = first.save_block_keys
        self.assertEqual(public, worker._make_layerwise_full_key(0, f"prefill_mtp1:0:{b'second'.hex()}"))
        self.assertIn("prefill_private_", private)
        self.assertEqual(first.range_tokens, {private: 4})
        readable.update([public, private])
        worker._layerwise_session_tracker.commit_put_keys([public, private])
        later = prepare("r1", 8, 9, hashes, final=True)
        self.assertEqual(later.load_block_keys, [public, private])
        self.assertEqual(later.save_block_keys, [None])
        worker._finish_current_layerwise_load_sessions()

        replay = prepare("r2", 4, 8, hashes, final=True)
        self.assertEqual(replay.load_block_keys, [public])
        worker._finish_current_layerwise_load_sessions()
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            prepare("r3", 4, 8, [b"first", b"different"], final=True)

    def test_prefill_store_writers_partition_whole_keys_and_restore_on_every_rank(self):
        owner, readable = self._make_prefill_snapshot_worker()
        second, _ = self._make_prefill_snapshot_worker()
        follower, _ = self._make_prefill_snapshot_worker()
        workers = (owner, second, follower)
        for rank, worker in enumerate(workers):
            worker.put_step = 3
            worker.tp_rank = rank
            worker.layerwise_prefill_put_writers = 2
            worker.m_store.batch_get_start.side_effect = lambda keys: [0 if key in readable else -704 for key in keys]

        def prepare(worker, history, extent, *, final=False):
            request = ReqMeta(
                "r1",
                block_ids=[7, 8, 9, 10, 11],
                block_hashes=[b"first", b"second", b"third"],
                target_token_len=extent,
                can_save=True,
                is_last_chunk=final,
            )
            request.prefill_history = history
            request.prefill_snapshot_id = str(extent)
            return mooncake_layerwise.prepare_layerwise_sessions(worker, [request])[0][0]

        first_views = [prepare(worker, 0, 15) for worker in workers]
        first, _, third, _ = first_views[0].save_block_keys
        _, middle, _, tail = first_views[1].save_block_keys
        self.assertEqual(first_views[0].save_block_keys, [first, None, third, None])
        self.assertEqual(first_views[1].save_block_keys, [None, middle, None, tail])
        self.assertEqual(first_views[2].save_block_keys, [None, None, None, None])
        self.assertEqual(len({first, middle, third, tail, None}), 5)
        follower.m_store.batch_put_start.assert_not_called()
        for worker, view in zip(workers, first_views, strict=True):
            worker.layer_save_tasks = [[]]
            worker._process_save_for_layer_batch([view], 0)
            self.assertEqual(bool(worker.layer_save_tasks[0]), worker is not follower)
            keys = [key for key in view.save_block_keys if key is not None]
            worker._layerwise_session_tracker.commit_put_keys(keys)
            readable.update(keys)
            if worker is owner:
                with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
                    prepare(follower, 15, 17, final=True)
        for worker in workers:
            view = prepare(worker, 15, 17, final=True)
            self.assertEqual(view.load_block_keys, [first, middle, third, tail])
            self.assertEqual(view.range_tokens, {tail: 3})
            worker._finish_current_layerwise_load_sessions()
        self.assertEqual(second._layerwise_session_tracker.collect_private_removals(), [tail])
        for worker in (owner, follower):
            self.assertEqual(worker._layerwise_session_tracker.collect_private_removals(), [])

    def test_prefill_concurrent_public_put_uses_committed_request_private_full_block(self):
        owner, readable = self._make_prefill_snapshot_worker()
        follower, _ = self._make_prefill_snapshot_worker()
        owner.put_step = follower.put_step = 2
        follower.tp_rank = 1
        public = owner._make_layerwise_full_key(0, b"shared".hex())
        stable = owner._make_layerwise_full_key(0, b"stable".hex())
        owner.m_store.batch_put_start.side_effect = lambda keys, sizes: [-705 if key == public else 0 for key in keys]
        for worker in (owner, follower):
            worker.m_store.exists.side_effect = lambda keys: [int(key in readable) for key in keys]
            worker.m_store.batch_get_start.side_effect = lambda keys: [
                0 if key in readable else -703 if key == public else -704 for key in keys
            ]

        def prepare(worker, history, extent, *, instance="first", final=False):
            request = ReqMeta(
                "r1",
                block_ids=[7, 8, 9],
                block_hashes=[b"shared", b"stable"],
                target_token_len=extent,
                can_save=True,
                is_last_chunk=final,
            )
            request.prefill_history = history
            request.prefill_request_instance = instance
            return mooncake_layerwise.prepare_layerwise_sessions(worker, [request])[0][0]

        first = prepare(owner, 0, 8)
        private, saved_public = first.save_block_keys
        self.assertNotEqual(private, public)
        self.assertEqual(saved_public, stable)
        self.assertEqual(prepare(follower, 0, 8).save_block_keys, [None, None])
        follower.m_store.batch_put_start.assert_not_called()
        self.assertEqual(owner._layerwise_session_tracker.private_blocks("r1", 0), {})
        readable.add(stable)
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            prepare(follower, 8, 9, final=True)

        readable.add(private)
        owner._layerwise_session_tracker.commit_put_keys([private])
        for worker in (owner, follower):
            restored = prepare(worker, 8, 9, final=True)
            self.assertEqual(restored.load_block_keys, [private, stable])
            self.assertEqual(restored.load_keys, [private, stable])
            worker._finish_current_layerwise_load_sessions()
        self.assertEqual(owner._layerwise_session_tracker.collect_private_removals(), [private])
        self.assertEqual(follower._layerwise_session_tracker.collect_private_removals(), [])

        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            prepare(follower, 8, 9, instance="second", final=True)
        readable.add(public)
        self.assertEqual(prepare(follower, 8, 9, instance="second", final=True).load_block_keys, [public, stable])

    def test_prefill_public_put_conflict_in_last_chunk_does_not_need_fallback(self):
        owner, _ = self._make_prefill_snapshot_worker()
        owner.m_store.batch_put_start.side_effect = lambda keys, sizes: [-705] * len(keys)
        request = ReqMeta(
            "r1", block_ids=[7], block_hashes=[b"shared"], target_token_len=4, can_save=True, is_last_chunk=True
        )
        request.prefill_history = 0
        self.assertEqual(mooncake_layerwise.prepare_layerwise_sessions(owner, [request])[0][0].save_block_keys, [None])
        owner.m_store.batch_put_start.assert_called_once()
        owner.m_store.batch_put_start.side_effect = lambda keys, sizes: [-800] * len(keys)
        with self.assertRaisesRegex(RuntimeError, "could not allocate Prefill snapshot"):
            mooncake_layerwise.prepare_layerwise_sessions(owner, [request])

    def test_put_start_uses_full_current_layout_size_and_skips_hits(self):
        worker = self._make_worker()
        worker.m_store.batch_put_start.return_value = [0]
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids=[1, 2],
            block_hashes=[b"h0", b"h1"],
            can_save=True,
            load_spec=LoadSpec(0, 16, can_load=True),
        )

        worker._prepare_layerwise_put_session(request)

        worker.m_store.batch_put_start.assert_called_once_with(["model@6831@0"], [60])
        self.assertEqual(request.save_key_block_offset, 1)
        self.assertEqual(request.save_block_keys, ["model@6831@0"])

    def test_full_remote_hit_loads_last_block_even_when_vllm_keeps_one_token(self):
        worker = self._make_worker()
        request = ReqMeta(
            "r1",
            block_ids=[1, 2, 3, 4],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],
            load_spec=LoadSpec(0, 63, can_load=True, kvpool_store_skip_tokens=64),
        )

        slots = worker._prepare_layerwise_get_session(request)

        self.assertEqual(len(slots), 4)
        self.assertEqual(request.load_block_keys[-1], "model@6833@0")
        self.assertIsNone(request.load_last_block_key)

    def test_next_chunk_reloads_committed_prefix_without_new_load_spec(self):
        worker = self._make_worker()
        worker._layerwise_session_tracker.register_put_keys(
            "r1",
            [("model@6830@0", 0)],
        )
        worker._layerwise_session_tracker.commit_put_keys(["model@6830@0"])
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0", b"h1"],
            load_spec=None,
            is_last_chunk=False,
        )

        slots = worker._prepare_layerwise_get_session(request)
        worker.layer_load_tasks = [[]]
        worker._process_load_for_layer_batch([request], 0)

        self.assertEqual(slots, [("model@6830@0", 10, 0)])
        self.assertEqual(request.load_block_keys, ["model@6830@0"])
        self.assertEqual(len(worker.layer_load_tasks[0]), 1)
        block_range = worker.layer_load_tasks[0][0].block_ranges[0]
        self.assertEqual((block_range.start_block, block_range.end_block), (0, 1))

    def test_hashless_boundary_key_uses_the_matching_block_slot(self):
        worker = self._make_worker()
        request = ReqMeta(
            "r1",
            token_len_chunk=32,
            block_ids=[10, 11],
            block_hashes=[b"h0"],
            load_spec=LoadSpec(0, 32, can_load=True),
        )

        slots = worker._prepare_layerwise_get_session(request)

        self.assertEqual(
            request.load_block_keys,
            ["model@6830@0", "model@r1_lastblock@0"],
        )
        self.assertIsNone(request.load_last_block_key)
        self.assertEqual(slots[-1], ("model@r1_lastblock@0", 11, 1))


class TestLayerwiseSessionTracker(unittest.TestCase):
    def test_commit_promotes_shared_put_key_to_every_request_owner(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("shared", 0)])
        tracker.register_put_keys("r2", [("shared", 1)])

        tracker.commit_put_keys(["shared"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("shared", 0)])
        self.assertEqual(tracker.prepare_load_entries("r2", []), [("shared", 1)])

    def test_complete_key_replaces_partial_key_for_the_same_block(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("partial", 1)])
        tracker.commit_put_keys(["partial"])
        tracker.register_put_keys("r1", [("complete", 1)])
        tracker.commit_put_keys(["complete"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [("complete", 1)])

    def test_shared_get_ends_only_after_the_last_owner_releases_it(self):
        tracker = LayerwiseSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.prepare_load_entries("r2", [("shared", 0)])
        tracker.record_get_result("shared", {"r1", "r2"}, succeeded=True)

        self.assertEqual(tracker.release_terminal({"r1"}), [])
        self.assertEqual(tracker.release_terminal({"r2"}), ["shared"])
        self.assertEqual(tracker.release_terminal({"r2"}), [])

    def test_failed_renewal_retains_desired_keys_for_retry(self):
        tracker = LayerwiseSessionTracker()
        tracker.prepare_load_entries("r1", [("shared", 0)])
        tracker.register_put_keys("r1", [("pending", 1)])
        tracker.record_get_result("shared", {"r1"}, succeeded=True)

        tracker.record_get_result("shared", {"r1"}, succeeded=False)
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.release_for_retry({"r1"}), [])
        self.assertEqual(
            tracker.prepare_load_entries("r1", []),
            [("shared", 0), ("pending", 1)],
        )

    def test_failed_get_attempt_preserves_unrelated_shared_owner(self):
        tracker = LayerwiseSessionTracker()
        tracker.prepare_load_entries("old-owner", [("shared", 0)])
        tracker.prepare_load_entries(
            "new-owner",
            [("shared", 0), ("new-key", 1)],
        )
        tracker.record_get_result(
            "shared",
            {"old-owner", "new-owner"},
            succeeded=True,
        )

        keys_to_end = tracker.release_failed_get_attempts(
            {
                "shared": {"new-owner"},
                "new-key": {"new-owner"},
            }
        )

        self.assertEqual(keys_to_end, ["new-key"])
        self.assertEqual(tracker.release_terminal({"old-owner"}), ["shared"])
        self.assertEqual(
            tracker.prepare_load_entries("new-owner", []),
            [("shared", 0), ("new-key", 1)],
        )

    def test_terminal_request_loses_pending_put_ownership(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("pending", 0)])

        tracker.release_terminal({"r1"})
        tracker.commit_put_keys(["pending"])

        self.assertEqual(tracker.prepare_load_entries("r1", []), [])

    def test_chunk_commit_retry_and_terminal_cleanup(self):
        tracker = LayerwiseSessionTracker()
        tracker.register_put_keys("r1", [("k0", 0)])
        tracker.commit_put_keys(["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_for_retry({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [("k0", 0)])

        tracker.record_get_result("k0", ["r1"], succeeded=True)
        self.assertEqual(tracker.release_terminal({"r1"}), ["k0"])
        self.assertEqual(tracker.prepare_load_entries("r1", []), [])


if __name__ == "__main__":
    unittest.main()

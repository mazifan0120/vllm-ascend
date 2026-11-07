# SPDX-License-Identifier: Apache-2.0
"""Request-private versions, prefix coverage and lease-safe retirement."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec,
    ReqMeta,
    RequestTracker,
    mtp_prefix_block_hash,
)


class TestMainSnapshotVersions(unittest.TestCase):
    def make_worker(self, published=None, blocked_public=None):
        from tests.ut.distributed.ascend_store.test_mooncake_layerwise import TestMooncakeWorkerSessionPreparation

        worker = TestMooncakeWorkerSessionPreparation._make_worker()
        worker._main_buffer_layer_slots = {0: 0}
        worker.layerwise_offload = True
        worker._load_session_lock = threading.RLock()
        worker._layer_load_aborted = threading.Event()
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._current_layerwise_request_ids = set()
        worker._current_layerwise_last_chunk_req_ids = set()
        worker._private_snapshot_cleanup_retry_at = 0.0
        worker.kv_recv_thread = worker.kv_send_thread = None
        published = published if published is not None else set()
        blocked_public = blocked_public or set()
        worker.m_store.batch_is_exist.side_effect = lambda keys: [int(key in published) for key in keys]
        worker.m_store.batch_put_start.side_effect = lambda keys, sizes: [int(key in blocked_public) for key in keys]
        worker.m_store.batch_get_start.side_effect = lambda keys: [0 if key in published else -1 for key in keys]
        worker.m_store.batch_get_end.return_value = 0

        def remove(keys):
            self.assertTrue(all("@private_" in key for key in keys))
            codes = [0 if key in published else -1 for key in keys]
            published.difference_update(keys)
            return codes

        worker.m_store.remove_private_snapshots.side_effect = remove
        return worker, published

    @staticmethod
    def request(end, generation, *, previous=0, instance="instance-A", req_id="r1", previous_generation=None):
        previous_partial = previous % 16 != 0
        load = (
            LoadSpec(
                previous,
                previous,
                can_load=True,
                layerwise_previous_step=True,
                layerwise_snapshot_end_token=previous if previous_partial else None,
                layerwise_snapshot_block_index=previous // 16 if previous_partial else None,
                layerwise_snapshot_generation=previous_generation
                if previous_generation is not None
                else generation - 1,
            )
            if previous
            else None
        )
        return ReqMeta(
            req_id,
            request_instance=instance,
            snapshot_generation=generation,
            target_token_len=end,
            num_prompt_tokens=end,
            save_end_token=end // 16 * 16,
            save_start_token=previous // 16 * 16,
            restore_end_token=previous,
            block_ids=list(range(1, (end + 15) // 16 + 1)),
            block_hashes=[f"h{index}".encode() for index in range(end // 16)],
            partial_block_index=end // 16 if end % 16 else None,
            load_spec=load,
            can_save=True,
            is_last_chunk=False,
        )

    @staticmethod
    def commit(worker, published, request):
        keys = [key for key in request.save_block_keys if key is not None]
        if request.save_last_block_key is not None:
            keys.append(request.save_last_block_key)
        published.update(keys)
        worker._layerwise_session_tracker.commit_put_keys(keys)
        worker._put_started_keys.difference_update(keys)
        return keys

    @staticmethod
    def open(worker, request):
        slots = worker._prepare_layerwise_get_session(request)
        worker._open_layerwise_get_sessions([(request, *slot) for slot in slots])
        return slots

    def test_three_partial_chunks_load_old_versions_and_write_distinct_new_versions(self):
        worker, published = self.make_worker()
        partials = []
        for generation, end in enumerate((10, 20, 30), start=1):
            previous = (0, 10, 20)[generation - 1]
            request = self.request(end, generation, previous=previous)
            self.open(worker, request)
            if partials:
                self.assertIn(partials[-1], request.load_keys)
            worker._prepare_layerwise_put_session(request)
            self.assertNotIn(request.save_last_block_key, request.load_keys)
            partials.append(request.save_last_block_key)
            self.commit(worker, published, request)
        self.assertEqual(len(set(partials)), 3)
        self.assertIn("generation:3", partials[-1])
        self.assertEqual(worker._layerwise_session_tracker.private_snapshot_end(partials[-1]), 30)

    def test_failed_new_version_does_not_replace_a_committed_old_version(self):
        worker, published = self.make_worker()
        first = self.request(5, 1)
        self.open(worker, first)
        worker._prepare_layerwise_put_session(first)
        self.commit(worker, published, first)
        second = self.request(10, 2, previous=5)
        self.open(worker, second)
        worker._prepare_layerwise_put_session(second)
        worker._layerwise_session_tracker.revoke_put_keys([second.save_last_block_key])
        third = self.request(12, 3, previous=10)
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            self.open(worker, third)
        self.assertEqual(
            worker._layerwise_session_tracker.prepare_load_entries("r1", []), [(first.save_last_block_key, 0)]
        )
        self.assertTrue(worker._layer_load_aborted.is_set())

    def test_reused_external_request_id_cannot_load_an_old_instance(self):
        worker, published = self.make_worker()
        old = self.request(10, 1)
        self.open(worker, old)
        worker._prepare_layerwise_put_session(old)
        self.commit(worker, published, old)
        new = self.request(10, 1, instance="instance-B")
        self.assertEqual(self.open(worker, new), [])
        worker._prepare_layerwise_put_session(new)
        self.assertNotEqual(new.save_last_block_key, old.save_last_block_key)
        worker._layerwise_session_tracker.commit_put_keys([old.save_last_block_key])
        self.assertEqual(worker._layerwise_session_tracker.prepare_load_entries("r1", []), [])

    def test_non_saving_rank_uses_scheduler_snapshot_identity(self):
        worker, published = self.make_worker()
        worker.tp_rank = 1
        worker.put_step = 16
        first = self.request(10, 1)
        key = worker._main_snapshot_key(first, 0, generation=1, end_token=10)
        published.add(key)
        second = self.request(20, 2, previous=10)
        self.open(worker, second)
        self.assertEqual(second.load_keys, [key])
        self.assertEqual(worker._layerwise_session_tracker.private_snapshot_end(key), 10)

    def test_non_saving_rank_promotes_an_old_partial_when_that_block_becomes_full(self):
        for contention in (False, True):
            with self.subTest(contention=contention):
                published = set()
                blocked = {"model@6830@0"} if contention else set()
                owner, _ = self.make_worker(published=published, blocked_public=blocked)
                reader, _ = self.make_worker(published=published)
                reader.tp_rank, reader.put_step = 1, 16
                old_partial = None
                for generation, end in enumerate((10, 20, 30, 40, 50), start=1):
                    previous = (generation - 1) * 10
                    save = self.request(end, generation, previous=previous)
                    self.open(owner, save)
                    owner._prepare_layerwise_put_session(save)
                    self.commit(owner, published, save)
                    if generation == 1:
                        old_partial = save.save_last_block_key
                    restore = self.request(end, generation, previous=previous)
                    self.open(reader, restore)
                    if generation >= 3:
                        self.assertNotIn(old_partial, restore.load_keys)
                        for index, key in enumerate(restore.load_block_keys):
                            valid_end = reader._layerwise_session_tracker.private_snapshot_end(key)
                            self.assertTrue(valid_end is None or valid_end >= min(previous, (index + 1) * 16))

    def test_failed_full_block_promotion_preserves_the_previously_committed_partial(self):
        published = set()
        owner, _ = self.make_worker(published=published)
        reader, _ = self.make_worker(published=published)
        reader.tp_rank, reader.put_step = 1, 16
        first = self.request(10, 1)
        owner._prepare_layerwise_put_session(first)
        self.commit(owner, published, first)
        self.open(reader, self.request(20, 2, previous=10))
        second = self.request(20, 2, previous=10)
        self.open(owner, second)
        owner._prepare_layerwise_put_session(second)
        self.commit(owner, published, second)
        published.remove(second.save_block_keys[0])
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            self.open(reader, self.request(30, 3, previous=20))
        entries = reader._layerwise_session_tracker.prepare_load_entries("r1", [])
        self.assertEqual({index: key for key, index in entries}[0], first.save_last_block_key)

    def test_public_pending_writer_uses_private_full_fallback_without_waiting(self):
        worker, published = self.make_worker(blocked_public={"model@6830@0"})
        first = self.request(20, 1)
        self.open(worker, first)
        worker._prepare_layerwise_put_session(first)
        self.assertIn("@private_full_v1:", first.save_block_keys[0])
        self.commit(worker, published, first)
        # Another rank has no sender-local commit tracker, but can resolve the
        # same immutable fallback from the request instance and block index.
        peer, _ = self.make_worker(published=published)
        second = self.request(24, 2, previous=20)
        self.open(peer, second)
        self.assertEqual(second.load_block_keys[0], first.save_block_keys[0])
        self.assertIn(first.save_last_block_key, second.load_keys)

    def test_mtp_full_snapshot_is_request_private_and_restored_by_every_rank(self):
        owner, published = self.make_worker(published={"model@6830@0"})
        reader, _ = self.make_worker(published=published)
        owner.use_eagle = reader.use_eagle = True
        reader.tp_rank, reader.put_step = 1, 16
        first = self.request(20, 1)
        owner._prepare_layerwise_put_session(first)
        self.assertIn("@private_full_v1:", first.save_block_keys[0])
        self.commit(owner, published, first)
        second = self.request(32, 2, previous=20)
        self.open(reader, second)
        self.assertEqual(second.load_keys, [first.save_block_keys[0], first.save_last_block_key])
        other = self.request(20, 1, instance="instance-B", req_id="r2")
        owner._prepare_layerwise_put_session(other)
        self.assertNotEqual(other.save_block_keys[0], first.save_block_keys[0])

    def test_mtp_prefix_hash_requires_successor_and_isolates_different_lookahead(self):
        hashes = [b"prefix", b"successor", b"tail"]
        self.assertEqual(mtp_prefix_block_hash(hashes, 0), "mtp_prefix_v1:0:737563636573736f72")
        self.assertIsNone(mtp_prefix_block_hash(hashes, 2))
        self.assertIsNone(mtp_prefix_block_hash(hashes, -1))
        changed = [b"prefix", b"different successor", b"tail"]
        self.assertNotEqual(mtp_prefix_block_hash(hashes, 0), mtp_prefix_block_hash(changed, 0))

    def test_mtp_prefix_is_public_but_prompt_tail_and_generated_blocks_stay_private(self):
        worker, published = self.make_worker()
        worker.use_eagle = worker.main_reuse_mtp_prefix = True
        request = self.request(64, 1)
        request.num_prompt_tokens = 48
        worker._prepare_layerwise_put_session(request)
        keys = self.commit(worker, published, request)
        self.assertTrue(all("@mtp_prefix_v1:" in key for key in keys[:2]))
        self.assertTrue(all("@private_full_v1:" in key for key in keys[2:]))
        peer, _ = self.make_worker(published=published)
        peer.use_eagle = peer.main_reuse_mtp_prefix = True
        replay = self.request(64, 1, instance="instance-B", req_id="r2")
        replay.num_prompt_tokens = 48
        replay.restore_end_token = 32
        replay.load_spec = LoadSpec(0, 32, can_load=True)
        self.open(peer, replay)
        self.assertEqual(replay.load_keys, keys[:2])
        worker._layerwise_session_tracker.release_terminal({"r1"})
        worker._cleanup_main_private_snapshots()
        self.assertTrue(set(keys[:2]).issubset(published))

    def test_mtp_public_contention_falls_back_privately_and_other_rank_can_restore(self):
        public = "model@mtp_prefix_v1:0:6831@0"
        owner, published = self.make_worker(blocked_public={public})
        owner.use_eagle = owner.main_reuse_mtp_prefix = True
        first = self.request(32, 1)
        owner._prepare_layerwise_put_session(first)
        self.assertIn("@private_full_v1:", first.save_block_keys[0])
        self.commit(owner, published, first)
        reader, _ = self.make_worker(published=published)
        reader.use_eagle = reader.main_reuse_mtp_prefix = True
        reader.tp_rank, reader.put_step = 1, 16
        second = self.request(40, 2, previous=32)
        self.open(reader, second)
        self.assertEqual(second.load_block_keys, first.save_block_keys)

    def test_mtp_public_eviction_fails_before_attention_without_borrowing_private_versions(self):
        worker, _ = self.make_worker()
        worker.use_eagle = worker.main_reuse_mtp_prefix = True
        request = self.request(48, 1, instance="instance-B", req_id="r2")
        request.restore_end_token = 16
        request.load_spec = LoadSpec(0, 16, can_load=True)
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            self.open(worker, request)
        self.assertTrue(worker._layer_load_aborted.is_set())

    def test_missing_restore_plan_fails_even_when_load_spec_is_absent(self):
        worker, _ = self.make_worker()
        request = self.request(20, 1)
        request.restore_end_token = 16
        with self.assertRaisesRegex(RuntimeError, "no Store restore plan"):
            worker._prepare_layerwise_get_session(request)

    def test_query_then_eviction_is_a_failure_before_attention(self):
        worker, _ = self.make_worker()
        request = self.request(40, 1)
        request.restore_end_token = 16
        request.load_spec = LoadSpec(16, 16, can_load=True)
        with self.assertRaisesRegex(RuntimeError, "complete committed Store prefix"):
            self.open(worker, request)
        self.assertTrue(worker._invalid_block_ids)

    def test_cleanup_waits_for_transfer_quiescence_and_never_removes_public_keys(self):
        worker, published = self.make_worker()
        first = self.request(5, 1)
        self.open(worker, first)
        worker._prepare_layerwise_put_session(first)
        self.commit(worker, published, first)
        second = self.request(10, 2, previous=5)
        self.open(worker, second)
        worker._prepare_layerwise_put_session(second)
        self.commit(worker, published, second)
        worker.kv_recv_thread = SimpleNamespace(
            raise_if_failed=lambda: None, request_queue=SimpleNamespace(unfinished_tasks=1)
        )
        worker._cleanup_main_private_snapshots()
        worker.m_store.remove_private_snapshots.assert_not_called()
        worker.kv_recv_thread.request_queue.unfinished_tasks = 0
        worker._cleanup_main_private_snapshots()
        self.assertNotIn(first.save_last_block_key, published)
        self.assertIn(second.save_last_block_key, published)

    def test_cleanup_failure_retains_retry_record(self):
        worker, published = self.make_worker()
        request = self.request(5, 1)
        self.open(worker, request)
        worker._prepare_layerwise_put_session(request)
        self.commit(worker, published, request)
        worker._release_layerwise_requests_terminal({"r1"})
        worker.m_store.remove_private_snapshots.side_effect = None
        worker.m_store.remove_private_snapshots.return_value = [-706]
        clock = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.monotonic"
        with patch(clock, return_value=10.0) as monotonic:
            worker._cleanup_main_private_snapshots()
            self.assertEqual(
                worker._layerwise_session_tracker.private_cleanup_candidates(), [request.save_last_block_key]
            )
            worker.m_store.remove_private_snapshots.return_value = [0]
            worker._cleanup_main_private_snapshots()
            self.assertEqual(worker.m_store.remove_private_snapshots.call_count, 1)
            monotonic.return_value = 11.0
            worker._cleanup_main_private_snapshots()
        self.assertEqual(worker.m_store.remove_private_snapshots.call_count, 2)
        self.assertEqual(worker._layerwise_session_tracker.private_cleanup_candidates(), [])


class TestMainSnapshotMetadata(unittest.TestCase):
    def test_metadata_versions_partial_blocks_without_a_gva(self):
        tracker = RequestTracker("r1", token_len=5, allocated_block_ids=[1, 2])
        first = ReqMeta.from_request_tracker(tracker, 16, save_partial_block=True, block_hashes=[])
        self.assertEqual(first.partial_block_index, 0)
        self.assertEqual(first.snapshot_generation, 1)
        self.assertEqual(first.request_instance, tracker.request_instance)
        tracker.token_len = 10
        second = ReqMeta.from_request_tracker(tracker, 16, save_partial_block=True, block_hashes=[])
        self.assertEqual(second.snapshot_generation, 2)
        self.assertEqual(second.request_instance, first.request_instance)
        self.assertNotEqual(RequestTracker("r1", token_len=0).request_instance, tracker.request_instance)

    def test_last_partial_prompt_tokens_are_not_treated_as_already_finished(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.layerwise_offload = True
        scheduler._discard_partial_chunks = True
        self.assertEqual(scheduler._get_last_chunk_tokens_num(list(range(50))), 50)

    def test_scheduler_does_not_apply_worker_physical_plan_to_merged_specs(self):
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

        from tests.ut.distributed.ascend_store.test_pool_scheduler import make_config
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import mooncake_layerwise
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

        config = make_config(
            extra_config={
                "backend": "mooncake",
                "use_layerwise": True,
                "layerwise_num_shared_buffers": 1,
                "layerwise_independent_layers": [],
            }
        )
        cache = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["model.layers.0.attn", "model.layers.1.attn"],
                    FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype="float32"),
                )
            ]
        )
        with (
            patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as imports,
            patch.object(
                mooncake_layerwise, "build_cache_plan", side_effect=AssertionError("worker allocation only")
            ) as planner,
        ):
            imports.import_module.return_value = MagicMock()
            scheduler = KVPoolScheduler(config, True, cache)
        self.assertTrue(scheduler.layerwise_offload)
        planner.assert_not_called()

    def test_scheduler_rejects_a_restore_without_load_spec(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.layerwise_offload = True
        with self.assertRaisesRegex(RuntimeError, "requires a Store restore plan"):
            scheduler._build_req_meta(RequestTracker("r1", 20), [], None, list(range(30)), False, restore_end_token=16)

    def test_scheduler_rejects_store_miss_or_short_hit_below_local_prefix(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

        for hit in (0, 16):
            with self.subTest(hit=hit):
                scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
                scheduler.layerwise_offload = scheduler.use_block_key_layerwise = scheduler.use_layerwise = True
                scheduler.use_eagle = False
                scheduler.kv_role = "kv_producer"
                scheduler.load_specs = {}
                scheduler.cache_transfer_granularity = 16
                scheduler._floor_to_cache_transfer_granularity = lambda length: length // 16 * 16
                scheduler._get_block_key_layerwise_hit_tokens = MagicMock(return_value=hit)
                request = SimpleNamespace(request_id="r1", prompt_token_ids=list(range(48)), num_tokens=48)
                with self.assertRaisesRegex(RuntimeError, "not fully recoverable"):
                    scheduler.get_num_new_matched_tokens(request, 32)

    def test_main_only_mtp_skips_cross_request_lookup(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.layerwise_offload = scheduler.use_block_key_layerwise = scheduler.use_eagle = True
        scheduler.kv_role = "kv_producer"
        scheduler.load_specs = {}
        scheduler._get_block_key_layerwise_hit_tokens = MagicMock()
        request = SimpleNamespace(request_id="r1")
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (0, False))
        with self.assertRaisesRegex(RuntimeError, "cannot restore a cross-request local prefix"):
            scheduler.get_num_new_matched_tokens(request, 16)
        scheduler._get_block_key_layerwise_hit_tokens.assert_not_called()

    def test_main_only_mtp_prefix_lookup_loads_matching_successors_only(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

        scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
        scheduler.layerwise_offload = scheduler.use_block_key_layerwise = scheduler.use_layerwise = True
        scheduler.use_eagle = scheduler.main_reuse_mtp_prefix = True
        scheduler.block_key_hybrid = scheduler.load_async = False
        scheduler.kv_role = "kv_producer"
        scheduler.load_specs = {}
        scheduler.cache_transfer_granularity = scheduler.lcm_block_size = scheduler._block_size = 16
        scheduler.hash_block_size = 16
        scheduler.original_block_size = [16]
        scheduler.cacheable_group_ids = [0]
        scheduler._make_layerwise_hit_check_keys = lambda group, identity: [identity]
        scheduler._query_layerwise_block_hits = MagicMock(return_value=[True, False])
        request = SimpleNamespace(
            request_id="r1", prompt_token_ids=list(range(48)), num_tokens=48, block_hashes=[b"h0", b"h1", b"h2"]
        )
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (16, False))
        scheduler._query_layerwise_block_hits.assert_called_once_with(
            [[mtp_prefix_block_hash(request.block_hashes, 0)], [mtp_prefix_block_hash(request.block_hashes, 1)]]
        )
        self.assertEqual(scheduler.load_specs["r1"].kvpool_cached_tokens, 16)
        # Even if the last query hits, the request-private final block is
        # recomputed so its hidden state and next-token draft stay current.
        scheduler._query_layerwise_block_hits.return_value = [True, True]
        self.assertEqual(scheduler.get_num_new_matched_tokens(request, 0), (32, False))
        with self.assertRaisesRegex(RuntimeError, "cannot restore a cross-request local prefix"):
            scheduler.get_num_new_matched_tokens(request, 16)

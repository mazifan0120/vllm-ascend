# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Task-01 Decode Scheduler admission path (spec §11.2).

Covers token accounting (E_DE = R - L_DE independent of the Store hit), the
pre-allocation lookup cache lifecycle, DecodeKVSnapshot creation/binding
invariants, duplicate scheduling and allocation-failure retry, parent
delegation for non-selected requests, and terminal cleanup.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (  # noqa: E402
    DecodeKVSnapshot,
    DualPathConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (  # noqa: E402
    LoadSpec,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"


def _make_vllm_config(kv_role="kv_consumer"):
    config = MagicMock()
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.is_kv_consumer = kv_role in {"kv_consumer", "kv_both"}
    config.kv_transfer_config.engine_id = "test_engine"
    config.kv_transfer_config.kv_port = 5000
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {"tls_config": {}}.get(
        key, default
    )
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.cache_config.block_size = 16
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    return config


def _make_kv_cache_config(block_size=16):
    spec = MagicMock()
    spec.block_size = block_size
    group = MagicMock()
    group.kv_cache_spec = spec
    group.layer_names = ["layer.0"]
    return SimpleNamespace(kv_cache_groups=[group], kv_cache_tensors=[], num_blocks=64)


def _make_request(request_id, num_tokens, kv_transfer_params=None):
    request = MagicMock()
    request.request_id = request_id
    request.num_tokens = num_tokens
    request.prompt_token_ids = list(range(num_tokens))
    request.kv_transfer_params = kv_transfer_params
    return request


def _make_blocks(block_ids_by_group):
    blocks = MagicMock()
    blocks.get_block_ids.return_value = tuple(list(group) for group in block_ids_by_group)
    return blocks


def _selected_params():
    return {"do_remote_prefill": True, "metaserver": "http://meta"}


class TestDecodeAdmission(unittest.TestCase):
    def setUp(self):
        self._adapter_patch = patch(f"{_CONNECTOR_NS}.KVPoolAdapter")
        self._worker_adapter_patch = patch(f"{_CONNECTOR_NS}.KVPoolWorkerAdapter")
        self._adapter_patch.start()
        self._worker_adapter_patch.start()
        self.addCleanup(self._adapter_patch.stop)
        self.addCleanup(self._worker_adapter_patch.stop)
        self.scheduler = DualPathConnectorScheduler(
            _make_vllm_config(),
            _make_kv_cache_config(),
            "test_engine",
            DualPathConfig(role="decode"),
        )
        self.addCleanup(self.scheduler.executor.shutdown, False)
        self.addCleanup(self.scheduler.metaserver_client.close)
        self.scheduler.executor = MagicMock(name="executor")

    def _admit(self, request, local_tokens, block_ids_by_group=((1, 2),), lookup_spec=None):
        """Run one lookup + successful allocation bind for a selected request."""
        self.scheduler._kvpool_adapter.lookup.return_value = lookup_spec
        matched = self.scheduler.get_num_new_matched_tokens(request, local_tokens)
        blocks = _make_blocks(block_ids_by_group)
        self.scheduler.update_state_after_alloc(request, blocks, matched[0])
        return matched, blocks

    def test_hbm_complete_returns_zero_false_no_adapter_call_no_state(self):
        request = _make_request("req-full-hbm", 48, _selected_params())
        self.scheduler._lookup_results["req-full-hbm"] = (10, None)  # stale entry
        result = self.scheduler.get_num_new_matched_tokens(request, 47)
        self.assertEqual(result, (0, False))
        self.scheduler._kvpool_adapter.lookup.assert_not_called()
        self.assertEqual(self.scheduler._lookup_results, {})
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})

    def test_full_partial_miss_all_return_same_external_delta(self):
        full_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
        partial_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
        for request_id, spec in (("req-f", full_spec), ("req-p", partial_spec), ("req-m", None)):
            with self.subTest(request_id=request_id):
                request = _make_request(request_id, 48, _selected_params())
                self.scheduler._kvpool_adapter.lookup.return_value = spec
                self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))

    def test_first_lookup_creates_only_lookup_results(self):
        request = _make_request("req-first", 48, _selected_params())
        spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
        self.scheduler._kvpool_adapter.lookup.return_value = spec
        self.scheduler.get_num_new_matched_tokens(request, 16)
        self.assertEqual(self.scheduler._lookup_results, {"req-first": (31, spec)})
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})

    def test_alloc_consumes_lookup_and_creates_one_snapshot_with_frozen_blocks(self):
        request = _make_request("req-bind", 48, _selected_params())
        spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
        matched, _ = self._admit(request, 16, block_ids_by_group=([7, 8], [9]), lookup_spec=spec)
        self.assertEqual(matched, (31, True))
        self.assertEqual(self.scheduler._lookup_results, {})
        snapshot = self.scheduler._decode_kv_snapshots["req-bind"]
        self.assertEqual(snapshot.target_tokens, 47)
        self.assertEqual(snapshot.external_tokens, 31)
        self.assertIs(snapshot.store_load_spec, spec)
        self.assertEqual(snapshot.final_block_ids, ((7, 8), (9,)))
        self.assertEqual(len(self.scheduler._decode_kv_snapshots), 1)

    def test_snapshot_derives_local_and_store_tokens(self):
        spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
        snapshot = DecodeKVSnapshot(
            target_tokens=47,
            external_tokens=31,
            store_load_spec=spec,
            final_block_ids=((1,),),
        )
        self.assertEqual(snapshot.local_tokens, 16)
        self.assertEqual(snapshot.store_tokens, 32)
        miss_snapshot = DecodeKVSnapshot(
            target_tokens=47,
            external_tokens=31,
            store_load_spec=None,
            final_block_ids=((1,),),
        )
        self.assertEqual(miss_snapshot.store_tokens, miss_snapshot.local_tokens)

    def test_identical_duplicate_lookup_performs_one_adapter_call(self):
        request = _make_request("req-dup", 48, _selected_params())
        self.scheduler._kvpool_adapter.lookup.return_value = None
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        self.scheduler._kvpool_adapter.lookup.assert_called_once()

    def test_changed_external_delta_replaces_unbound_result(self):
        request = _make_request("req-change", 48, _selected_params())
        spec_a = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
        spec_b = LoadSpec(vllm_cached_tokens=32, kvpool_cached_tokens=40, can_load=False)
        self.scheduler._kvpool_adapter.lookup.side_effect = [spec_a, spec_b]
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 32), (15, True))
        self.assertEqual(self.scheduler._lookup_results, {"req-change": (15, spec_b)})
        self.assertEqual(self.scheduler._kvpool_adapter.lookup.call_count, 2)

    def test_changed_external_delta_discards_before_reprobe_when_lookup_raises(self):
        request = _make_request("req-discard", 48, _selected_params())
        self.scheduler._kvpool_adapter.lookup.return_value = None
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        self.scheduler._kvpool_adapter.lookup.side_effect = RuntimeError("spec invariant violated")
        with self.assertRaisesRegex(RuntimeError, "spec invariant violated"):
            self.scheduler.get_num_new_matched_tokens(request, 32)
        self.assertEqual(self.scheduler._lookup_results, {})

    def test_external_token_mismatch_at_bind_raises(self):
        request = _make_request("req-ext-mismatch", 48, _selected_params())
        self.scheduler._kvpool_adapter.lookup.return_value = None
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        with self.assertRaisesRegex(RuntimeError, "external-token mismatch"):
            self.scheduler.update_state_after_alloc(request, _make_blocks(((1,),)), 30)
        self.assertEqual(self.scheduler._lookup_results, {})
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})

    def test_store_range_outside_target_at_bind_raises(self):
        request = _make_request("req-range", 48, _selected_params())
        out_of_range = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
        self.scheduler._kvpool_adapter.lookup.return_value = out_of_range
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        with self.assertRaisesRegex(RuntimeError, "outside"):
            self.scheduler.update_state_after_alloc(request, _make_blocks(((1,),)), 31)
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})

    def test_alloc_failure_retry_reuses_unbound_result(self):
        request = _make_request("req-retry", 48, _selected_params())
        spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
        self.scheduler._kvpool_adapter.lookup.return_value = spec
        self.scheduler.get_num_new_matched_tokens(request, 16)
        # allocate_slots returned None: update_state_after_alloc never runs.
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 16), (31, True))
        self.scheduler._kvpool_adapter.lookup.assert_called_once()
        self.scheduler.update_state_after_alloc(request, _make_blocks(((5, 6),)), 31)
        self.assertIn("req-retry", self.scheduler._decode_kv_snapshots)

    def test_identical_duplicate_bind_is_idempotent(self):
        request = _make_request("req-idem", 48, _selected_params())
        self._admit(request, 16, block_ids_by_group=((7, 8),))
        first = self.scheduler._decode_kv_snapshots["req-idem"]
        self.scheduler.update_state_after_alloc(request, _make_blocks(((7, 8),)), 31)
        self.assertEqual(len(self.scheduler._decode_kv_snapshots), 1)
        self.assertIs(self.scheduler._decode_kv_snapshots["req-idem"], first)

    def test_conflicting_duplicate_bind_raises_and_preserves_original(self):
        request = _make_request("req-conflict", 48, _selected_params())
        self._admit(request, 16, block_ids_by_group=((7, 8),))
        first = self.scheduler._decode_kv_snapshots["req-conflict"]
        with self.assertRaisesRegex(RuntimeError, "conflicting duplicate"):
            self.scheduler.update_state_after_alloc(request, _make_blocks(((9, 9),)), 31)
        self.assertIs(self.scheduler._decode_kv_snapshots["req-conflict"], first)

    def test_missing_lookup_result_at_bind_raises(self):
        request = _make_request("req-orphan", 48, _selected_params())
        with self.assertRaisesRegex(RuntimeError, "no Task-01 lookup result"):
            self.scheduler.update_state_after_alloc(request, _make_blocks(((1,),)), 31)
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})

    def test_reprobe_of_admitted_request_raises(self):
        request = _make_request("req-reprobe", 48, _selected_params())
        self._admit(request, 16)
        with self.assertRaisesRegex(RuntimeError, "already admitted"):
            self.scheduler.get_num_new_matched_tokens(request, 16)

    def test_selected_request_skips_parent_recv_queue_and_metaserver(self):
        request = _make_request("req-skip", 48, _selected_params())
        self._admit(request, 16)
        self.assertEqual(self.scheduler._reqs_need_recv, {})
        self.assertTrue(request.kv_transfer_params["do_remote_prefill"])
        self.scheduler.executor.submit.assert_not_called()

    def test_prefill_role_and_plain_decode_delegate_to_parent(self):
        prefill_scheduler = DualPathConnectorScheduler(
            _make_vllm_config(kv_role="kv_producer"),
            _make_kv_cache_config(),
            "test_engine",
            DualPathConfig(role="prefill"),
        )
        self.addCleanup(prefill_scheduler.executor.shutdown, False)
        self.addCleanup(prefill_scheduler.metaserver_client.close)
        self.assertIsNone(prefill_scheduler._kvpool_adapter)
        selected = _make_request("req-pf", 4, _selected_params())
        self.assertEqual(prefill_scheduler.get_num_new_matched_tokens(selected, 0), (4, True))

        plain_decode = _make_request("req-plain", 48, {"metaserver": "http://meta"})
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(plain_decode, 0), (0, False))
        no_params = _make_request("req-none", 48, None)
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(no_params, 0), (0, False))

    def test_finish_callbacks_and_shutdown_clear_all_task01_state(self):
        request = _make_request("req-fin", 48, _selected_params())
        self._admit(request, 16)
        other = _make_request("req-pending", 48, _selected_params())
        self.scheduler._kvpool_adapter.lookup.return_value = None
        self.scheduler.get_num_new_matched_tokens(other, 16)

        result = self.scheduler.request_finished(request, [7, 8])
        self.assertEqual(result, (False, None))
        self.assertNotIn("req-fin", self.scheduler._decode_kv_snapshots)
        self.assertEqual(self.scheduler._lookup_results, {"req-pending": (31, None)})

        result = self.scheduler.request_finished_all_groups(other, ([5, 6],))
        self.assertEqual(result, (False, None))
        self.assertEqual(self.scheduler._lookup_results, {})

        self._admit(_make_request("req-shutdown", 48, _selected_params()), 16)
        self.scheduler.shutdown()
        self.assertEqual(self.scheduler._lookup_results, {})
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})
        self.scheduler._kvpool_adapter.close.assert_called_once()

    def test_hbm_complete_alloc_callback_creates_no_state_and_skips_parent(self):
        request = _make_request("req-zero-ext", 48, _selected_params())
        self.assertEqual(self.scheduler.get_num_new_matched_tokens(request, 47), (0, False))
        self.scheduler.update_state_after_alloc(request, _make_blocks(((1, 2, 3),)), 0)
        self.assertEqual(self.scheduler._decode_kv_snapshots, {})
        self.assertEqual(self.scheduler._lookup_results, {})
        self.assertEqual(self.scheduler._reqs_need_recv, {})
        self.assertTrue(request.kv_transfer_params["do_remote_prefill"])
        self.scheduler.executor.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()

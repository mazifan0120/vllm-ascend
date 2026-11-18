# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import deque
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.layerwise_decode_scheduler import AsyncLayerwiseDecodeScheduler, LayerwiseDecodeScheduler
from vllm_ascend.core.layerwise_offload_scheduler import DecodeWorkerEvent, DecodeWorkerLink
from vllm_ascend.core.prefill_restore_scheduler import PrefillRestoreScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_plan import CatalogRef, TargetAllocation, TargetId


class _Request:
    def __init__(self, identifier="engine-req", status=RequestStatus.WAITING_FOR_REMOTE_KVS):
        self.request_id = identifier
        self.num_prompt_tokens = 6
        self.num_tokens = 6
        self.num_computed_tokens = 6
        self.num_preemptions = 0
        self.last_sched_seq = 0
        self.status = status

    def is_finished(self):
        return RequestStatus.is_finished(self.status)


@dataclass
class _Block:
    block_id: int
    ref_cnt: int = 1
    is_null: bool = False
    block_hash: str | None = None


class _Cache:
    def __init__(self, request):
        self.groups = {request.request_id: ([_Block(3), _Block(5)], [_Block(7), _Block(9)])}
        self.released = []
        self.cache_blocks = Mock()
        self.reset_prefix_cache = Mock(return_value=True)
        self.block_pool = SimpleNamespace(free_blocks=self._release)

    def get_blocks(self, request_id):
        return SimpleNamespace(blocks=self.groups[request_id])

    def pop_blocks_for_free(self, request):
        return [block for group in self.groups.pop(request.request_id) for block in group]

    def free(self, request):
        self._release(reversed(self.pop_blocks_for_free(request)))

    def _release(self, blocks):
        for block in blocks:
            assert block.ref_cnt == 1
            block.ref_cnt -= 1
            self.released.append(block.block_id)


@dataclass
class _RunnerOutput:
    kv_connector_output: KVConnectorOutput | None = None


class TestPrefillRestoreScheduler(unittest.TestCase):
    def test_finishes_only_confirmed_restore_requests_in_a_mixed_step(self):
        scheduler = PrefillRestoreScheduler.__new__(PrefillRestoreScheduler)
        scheduler.finished_recving_kv_req_ids = {"restore", "failed", "ordinary"}
        scheduler.finished_req_ids_dict = {0: {"restore"}}
        request = SimpleNamespace(
            client_index=0,
            request_id="restore",
            trace_headers=None,
            get_finished_reason=lambda: "stop",
            take_events=lambda: None,
            take_prefill_stats=lambda: None,
        )
        scheduler.finish_requests = Mock(return_value=[request])
        step = SimpleNamespace(
            kv_connector_metadata=SimpleNamespace(metadata=[SimpleNamespace(prefill_restores={"restore", "failed"})])
        )
        result = _RunnerOutput(
            KVConnectorOutput(finished_recving={"restore", "ordinary", "failed"}, failed_recving={"failed"})
        )
        with patch.object(Scheduler, "update_from_output", return_value={}):
            outputs = scheduler.update_from_output(step, result)
        scheduler.finish_requests.assert_called_once_with({"restore"}, RequestStatus.FINISHED_STOPPED)
        self.assertEqual([output.request_id for output in outputs[0].outputs], ["restore"])
        self.assertEqual(outputs[0].outputs[0].new_token_ids, [])
        self.assertEqual(outputs[0].finished_requests, {"restore"})


class TestLayerwiseDecodeScheduler(unittest.TestCase):
    def setUp(self):
        self.request = _Request()
        self.cache = _Cache(self.request)
        self.refs = tuple(CatalogRef(f"worker-{rank}", 3, rank) for rank in range(2))
        self.allocation = TargetAllocation(TargetId("transfer-instance", 1, 2), 6, ((3, 5), (7, 9)))
        self.close_messages = []
        self.link = DecodeWorkerLink(Mock(), lambda target, ranks: self.close_messages.append((target, ranks)), Mock())

    def _make_scheduler(self, cls=LayerwiseDecodeScheduler):
        def init(scheduler):
            scheduler.num_spec_tokens = 0
            scheduler.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

        with patch.object(Scheduler, "__init__", init):
            scheduler = cls()
        scheduler.connector = Mock()
        scheduler.ec_connector = None
        scheduler.kv_cache_manager = self.cache
        scheduler.encoder_cache_manager = Mock()
        scheduler._inflight_prefills = set()
        scheduler.finished_req_ids = set()
        scheduler.finished_req_ids_dict = None
        scheduler.finished_recving_kv_req_ids = set()
        scheduler.failed_recving_kv_req_ids = set()
        scheduler.recompute_kv_load_failures = False
        scheduler.prev_step_scheduled_req_ids = set()
        scheduler.requests = {self.request.request_id: self.request}
        scheduler.running = []
        scheduler.waiting = Mock()
        scheduler.skipped_waiting = Mock()
        scheduler.defer_block_free = False
        scheduler.processed_step_seq = 0
        scheduler.sched_step_seq = 0
        scheduler.deferred_frees = deque()
        return scheduler

    def _bound(self, cls=LayerwiseDecodeScheduler):
        scheduler = self._make_scheduler(cls)
        admission = scheduler.bind_layerwise_workers(SimpleNamespace(refs=self.refs), self.link)
        self.assertEqual(scheduler.retain_layerwise_target(self.request, self.allocation), self.refs)
        return scheduler, admission

    def test_abort_before_publication_preserves_actual_blocks_until_all_close(self):
        scheduler, admission = self._bound()
        target = self.allocation.target
        admission.prepared(target, self.refs[0])
        scheduler.finish_requests(self.request.request_id, RequestStatus.FINISHED_ABORTED)
        self.assertEqual(self.close_messages, [(target, self.refs)])
        self.assertIsNone(admission.publish(target))
        self.assertEqual(self.cache.released, [])
        self.assertIn(self.request.request_id, scheduler.requests)
        # A stock completion would enter _free_blocks for an already-finished request.
        scheduler._update_from_kv_xfer_finished(KVConnectorOutput(finished_recving={self.request.request_id}))
        self.assertEqual(self.cache.released, [])
        self.assertFalse(admission.closed(target, self.refs[0]))
        self.assertEqual([b.ref_cnt for group in self.cache.groups[self.request.request_id] for b in group], [1] * 4)
        admission.prepared(target, self.refs[1])
        self.assertTrue(admission.closed(target, self.refs[1]))
        self.assertNotIn(self.request.request_id, scheduler.requests)
        self.assertEqual(self.cache.released, [9, 7, 5, 3])
        self.assertTrue(admission.closed(target, self.refs[0]))
        self.assertEqual(self.cache.released, [9, 7, 5, 3])

    def test_real_ready_gate_and_upstream_async_free_fence(self):
        scheduler, admission = self._bound(AsyncLayerwiseDecodeScheduler)
        target = self.allocation.target
        for ref in self.refs:
            admission.prepared(target, ref)
        admission.publish(target)
        admission.ready(target, self.refs[0])
        self.assertFalse(scheduler._try_promote_blocked_waiting_request(self.request))
        self.cache.cache_blocks.assert_not_called()
        admission.ready(target, self.refs[1])
        self.assertTrue(scheduler._try_promote_blocked_waiting_request(self.request))
        self.cache.cache_blocks.assert_called_once_with(self.request, 6)
        self.request.status = RequestStatus.RUNNING
        self.request.last_sched_seq = 4
        scheduler.defer_block_free = True
        scheduler.sched_step_seq = 4
        scheduler.finish_requests(self.request.request_id, RequestStatus.FINISHED_ABORTED)
        self.assertEqual(self.close_messages, [])
        self.assertIn(self.request.request_id, self.cache.groups)
        self.assertEqual(self.cache.released, [])
        self.assertFalse(scheduler.reset_prefix_cache(reset_running_requests=True))
        scheduler.processed_step_seq = 4
        with patch.object(Scheduler, "schedule", return_value=None):
            scheduler.schedule()
        self.assertEqual(self.close_messages, [(target, self.refs)])
        for ref in self.refs:
            admission.closed(target, ref)
        self.assertNotIn(self.request.request_id, self.cache.groups)
        self.assertEqual(self.cache.released, [9, 7, 5, 3])
        self.assertTrue(scheduler.reset_prefix_cache(reset_running_requests=True))

    def test_cancel_holds_upstream_pool_blocks_until_last_close(self):
        spec = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.float16)
        config = KVCacheConfig(8, [], [KVCacheGroupSpec(["layer.0"], spec)])
        cache = KVCacheManager(config, 16, scheduler_block_size=4, hash_block_size=4, enable_caching=False)
        free_before = cache.block_pool.get_num_free_blocks()
        request = Request("real-request", [1] * 6, SamplingParams(max_tokens=1), None)
        allocated = cache.allocate_slots(request, request.num_prompt_tokens)
        self.assertIsNotNone(allocated)
        (blocks,) = cache.get_blocks(request.request_id).blocks
        target = TargetId("real-transfer", 1, 2)
        allocation = TargetAllocation(target, request.num_prompt_tokens, (tuple(block.block_id for block in blocks),))
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler = self._make_scheduler()
        scheduler.kv_cache_manager = cache
        scheduler.requests = {request.request_id: request}
        admission = scheduler.bind_layerwise_workers(SimpleNamespace(refs=self.refs), self.link)
        scheduler.retain_layerwise_target(request, allocation)
        for ref in self.refs:
            admission.prepared(target, ref)
        admission.publish(target)

        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        self.assertEqual(self.close_messages, [(target, self.refs)])
        admission.closed(target, self.refs[0])
        self.assertEqual(cache.block_pool.get_num_free_blocks(), free_before - 2)
        self.assertEqual([block.ref_cnt for block in blocks], [1, 1])
        old_target = replace(target, allocation_generation=1)
        scheduler.post_layerwise_event(DecodeWorkerEvent("closed", old_target, self.refs[1]))
        with patch.object(Scheduler, "schedule", return_value=None):
            scheduler.schedule()
        self.assertEqual(cache.block_pool.get_num_free_blocks(), free_before - 2)

        admission.closed(target, self.refs[1])
        self.assertEqual(cache.block_pool.get_num_free_blocks(), free_before)
        self.assertEqual([block.ref_cnt for block in blocks], [0, 0])
        self.assertNotIn(request.request_id, scheduler.requests)

    def test_dispatch_carries_target_identity_and_validates_new_block_ids(self):
        scheduler, _ = self._bound()
        output = SimpleNamespace(
            num_scheduled_tokens={self.request.request_id: 1, "ordinary-request": 1},
            scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
            scheduled_new_reqs=[SimpleNamespace(req_id=self.request.request_id, block_ids=([3, 5, 6], [7, 9, 8]))],
        )
        self.assertEqual(scheduler.layerwise_scheduled_targets(output), (self.allocation.target,))
        output.scheduled_new_reqs[0].block_ids[0][0] = 11
        with self.assertRaisesRegex(RuntimeError, "different Host target"):
            scheduler.layerwise_scheduled_targets(output)
        output.scheduled_new_reqs.clear()
        output.scheduled_cached_reqs.resumed_req_ids.add(self.request.request_id)
        with self.assertRaisesRegex(RuntimeError, "cannot resume"):
            scheduler.layerwise_scheduled_targets(output)

    def test_preempt_pressure_and_reset_dispatch_a_request_failure(self):
        scheduler, admission = self._bound()
        self.request.status = RequestStatus.RUNNING
        scheduler.running.append(self.request)
        self.assertFalse(scheduler._request_blocks_can_be_freed(self.request))
        self.assertFalse(scheduler.reset_prefix_cache(reset_running_requests=True))
        self.cache.reset_prefix_cache.assert_not_called()

        def process(base, step, output):
            failed = base._handle_failed_recving(output.kv_connector_output.failed_recving)
            base.finish_requests(failed, RequestStatus.FINISHED_ERROR)
            return failed

        with patch.object(Scheduler, "update_from_output", process):
            self.assertEqual(scheduler.update_from_output(None, _RunnerOutput()), {self.request.request_id})
        self.assertEqual(self.request.status, RequestStatus.FINISHED_ERROR)
        self.assertEqual(self.close_messages, [(self.allocation.target, self.refs)])
        self.assertEqual(self.cache.released, [])
        for ref in self.refs:
            admission.closed(self.allocation.target, ref)
        self.assertTrue(scheduler.reset_prefix_cache(reset_running_requests=True))
        self.cache.reset_prefix_cache.assert_called_once_with()

    def test_admission_checks_real_allocation_and_leaves_refcount_alone(self):
        scheduler = self._make_scheduler()
        scheduler.bind_layerwise_workers(SimpleNamespace(refs=self.refs), self.link)
        block = self.cache.groups[self.request.request_id][0][0]
        block.ref_cnt = 2
        with self.assertRaisesRegex(ValueError, "private"):
            scheduler.retain_layerwise_target(self.request, self.allocation)
        block.ref_cnt = 1
        self.assertEqual(scheduler.retain_layerwise_target(self.request, self.allocation), self.refs)
        self.assertEqual(block.ref_cnt, 1)

    def test_failed_remote_kv_does_not_enter_upstream_direct_free(self):
        scheduler, admission = self._bound()
        target = self.allocation.target
        for ref in self.refs:
            admission.prepared(target, ref)
        admission.publish(target)
        for ref in self.refs:
            admission.ready(target, ref)
        scheduler.failed_recving_kv_req_ids.add(self.request.request_id)
        self.request.num_computed_tokens = 0
        self.assertFalse(scheduler._try_promote_blocked_waiting_request(self.request))
        self.assertEqual(self.cache.released, [])

    def test_prepare_ready_and_close_events_progress_without_model_forward(self):
        scheduler, admission = self._bound()
        self.link.prepare.assert_called_once_with(self.allocation, self.refs)
        target = self.allocation.target
        with patch.object(Scheduler, "schedule", return_value=None):
            scheduler.post_layerwise_event(DecodeWorkerEvent("prepared", target, self.refs[0]))
            scheduler.schedule()
            self.link.publish.assert_not_called()
            for _ in range(2):
                scheduler.post_layerwise_event(DecodeWorkerEvent("prepared", target, self.refs[1]))
            scheduler.schedule()
            self.link.publish.assert_called_once_with(self.request, self.allocation, self.refs)
            for ref in self.refs:
                scheduler.post_layerwise_event(DecodeWorkerEvent("ready", target, ref))
            scheduler.schedule()
            self.assertTrue(admission.may_decode(target))
            scheduler.finish_requests(self.request.request_id, RequestStatus.FINISHED_ABORTED)
            for ref in self.refs:
                scheduler.post_layerwise_event(DecodeWorkerEvent("closed", target, ref))
            scheduler.schedule()
        self.assertEqual(self.cache.released, [9, 7, 5, 3])

    def test_ordinary_request_uses_upstream_finish_and_free(self):
        self.request.status = RequestStatus.RUNNING
        scheduler = self._make_scheduler()
        scheduler.connector = None
        scheduler.running.append(self.request)
        scheduler.finish_requests(self.request.request_id, RequestStatus.FINISHED_ABORTED)
        self.assertEqual(self.cache.released, [9, 7, 5, 3])
        self.assertEqual(self.close_messages, [])


if __name__ == "__main__":
    unittest.main()

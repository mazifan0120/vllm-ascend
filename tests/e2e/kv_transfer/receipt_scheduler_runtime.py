# SPDX-License-Identifier: Apache-2.0
"""Real vLLM scheduler/block-manager control paths; no tensor DMA/model claim.

Run inside the supported native vLLM/Ascend environment. Requests, block pools,
queues and base scheduler methods are real. The external proof producer is a
CPU test control: no remote transfer is submitted by this test.
"""

import json
import threading
from types import SimpleNamespace

import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.receipt_scheduler import ReceiptScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    ContractError,
    ReleaseAck,
    RequestKey,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import (
    SchedulerRetentionRegistry,
    SerializedKVCacheManager,
)


def _fixture(name):
    # Only scheduler metadata is constructed; no worker or KV tensor exists.
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn.attn"],
                kv_cache_spec=FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16),
            )
        ],
    )
    raw = KVCacheManager(config, max_model_len=128, scheduler_block_size=16, hash_block_size=16, enable_caching=False)
    lock = threading.RLock()
    manager = SerializedKVCacheManager(raw, lock)
    registry = SchedulerRetentionRegistry(manager, allocation_lock=lock)
    manager.bind_retention_registry(registry)
    request = Request(name, list(range(23)), SamplingParams(max_tokens=32), pooling_params=None)
    assert manager.allocate_slots(request, num_new_tokens=23) is not None
    allocation = registry.issue(
        name, BlockAllocation(RequestKey(name, 0), 0, tuple(tuple(g) for g in manager.get_block_ids(name)), 16, 23)
    )
    before = tuple(block.ref_cnt for block in raw.block_pool.blocks)
    guard = registry.retain(allocation)
    assert tuple(block.ref_cnt for block in raw.block_pool.blocks) == before
    proof = threading.Event()
    calls = []

    def prepare_free(req):
        calls.append(req.request_id)
        if not proof.is_set():
            return False
        guard.release()
        registry.retire(allocation, ReleaseAck(True, allocation.retention_token, "CPU no-DMA proof fixture"))
        return True

    scheduler = object.__new__(ReceiptScheduler)
    scheduler.kv_cache_manager = manager
    scheduler._target_registry = registry
    scheduler._retained_finished = set()
    scheduler.requests = {name: request}
    scheduler.running = [request]
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.connector = SimpleNamespace(prepare_target_free=prepare_free)
    scheduler.defer_block_free = False
    scheduler._inflight_prefills = set()
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda req: None)
    scheduler.log_stats = False
    scheduler.reset_preempted_req_ids = set()
    scheduler.prev_step_scheduled_req_ids = {name}
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = set()
    request.status = RequestStatus.RUNNING
    return scheduler, request, allocation, proof, calls, raw


def exercise_scheduler_lifetime():
    # Proofs below authorize CPU-only unused blocks, not real remote transfers.
    passed = []
    scheduler, request, allocation, proof, calls, raw = _fixture("retained-preempt")
    original_ids = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    for method in (scheduler.kv_cache_manager.free, scheduler.kv_cache_manager.pop_blocks_for_free):
        try:
            method(request)
        except ContractError:
            pass
        else:
            raise AssertionError("retained block-map entered the real free path")
    scheduler.defer_block_free = True
    scheduler.processed_step_seq = 0
    request.last_sched_seq = 1
    assert not scheduler._request_blocks_can_be_freed(request)
    assert not calls, "GPU completion must precede target-owner release"
    scheduler.processed_step_seq = 1
    assert not scheduler._request_blocks_can_be_freed(request)
    assert request.status == RequestStatus.RUNNING
    assert scheduler.kv_cache_manager.get_block_ids(request.request_id) == original_ids
    proof.set()
    assert scheduler._request_blocks_can_be_freed(request)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, 0.0)
    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.waiting.peek_request() is request
    assert all(not group for group in raw.get_block_ids(request.request_id))
    assert raw.block_pool.get_num_free_blocks() == 15
    passed.append("real_preemption_and_gpu_fence")

    scheduler, request, _, proof, _, raw = _fixture("retained-finished")
    scheduler.running.clear()
    request.status = RequestStatus.FINISHED_ABORTED
    scheduler._connector_finished = lambda req: (True, None)
    scheduler.ec_connector = None
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._free_request(request)
    assert request.request_id in scheduler._retained_finished
    scheduler._drain_retained_finished()
    assert request.request_id in scheduler.requests
    assert raw.get_block_ids(request.request_id)[0]
    proof.set()
    scheduler._drain_retained_finished()
    assert request.request_id not in scheduler.requests
    assert not scheduler._retained_finished
    assert raw.block_pool.get_num_free_blocks() == 15
    passed.append("finished_cancel_deferred_then_retired")

    scheduler, request, _, proof, _, raw = _fixture("retained-reset")
    assert not scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.running == [request]
    assert scheduler.prev_step_scheduled_req_ids == {request.request_id}
    assert request.status == RequestStatus.RUNNING
    proof.set()
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert not scheduler.running
    assert not scheduler.prev_step_scheduled_req_ids
    assert request.status == RequestStatus.PREEMPTED
    assert raw.block_pool.get_num_free_blocks() == 15
    passed.append("forced_reset_does_not_pop_pending_victim")

    scheduler, request, _, proof, _, raw = _fixture("retained-load-failure")
    scheduler.running.clear()
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    request.num_computed_tokens = 0
    scheduler.failed_recving_kv_req_ids.add(request.request_id)
    scheduler.finished_recving_kv_req_ids.add(request.request_id)
    scheduler.needs_kv_cache_zeroing = False
    assert not scheduler._try_promote_blocked_waiting_request(request)
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert raw.get_block_ids(request.request_id)[0]
    proof.set()
    assert scheduler._try_promote_blocked_waiting_request(request)
    assert request.status == RequestStatus.WAITING
    assert raw.block_pool.get_num_free_blocks() == 15
    passed.append("failed_receive_cannot_bypass_retention")
    print(json.dumps({"status": "PASS", "real_scheduler_cases": passed, "model_or_dma_test": False}), flush=True)


if __name__ == "__main__":
    exercise_scheduler_lifetime()

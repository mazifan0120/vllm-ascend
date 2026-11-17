# SPDX-License-Identifier: Apache-2.0
"""Synthetic A1/A2 native resource test; no model/PD receipt end-to-end claim."""

import argparse
import ctypes
import json
import pickle
import socket
import threading
import uuid
from types import SimpleNamespace

import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    Component,
    ContractError,
    ReceiveReceipt,
    ReleaseAck,
    RequestKey,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostPoolTopology,
    MooncakeHostPool,
    allocate_mooncake_host_region,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import SparseKVOffloadManager
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import DeferredTargetProvider
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SchedulerRetentionRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_runtime import ReceivedTargetWrites
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE


def reject(action, expected=(RuntimeError, ContractError)):
    try:
        action()
    except expected:
        return
    raise AssertionError("operation incorrectly succeeded")


def exercise_real_scheduler():
    """Verify authoritative guards against actual framework block ownership."""
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request

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
    manager = KVCacheManager(
        config, max_model_len=128, scheduler_block_size=16, hash_block_size=16, enable_caching=False
    )
    request = Request(
        request_id="target-retention-real",
        prompt_token_ids=list(range(23)),
        sampling_params=SamplingParams(max_tokens=32),
        pooling_params=None,
    )
    assert manager.allocate_slots(request, num_new_tokens=23) is not None
    ids = tuple(tuple(group) for group in manager.get_block_ids(request.request_id))
    registry = SchedulerRetentionRegistry(manager)
    allocation = registry.issue(
        request.request_id, BlockAllocation(RequestKey("actual-vllm-manager", 1), 1, ids, 16, 23)
    )
    before = tuple(block.ref_cnt for block in manager.block_pool.blocks)
    guard = registry.retain(allocation)
    assert tuple(block.ref_cnt for block in manager.block_pool.blocks) == before, "retention must not trigger COW"
    request.num_computed_tokens = 23
    request.append_output_token_ids(list(range(16)))
    assert manager.allocate_slots(request, num_new_tokens=16) is not None
    guard.validate()
    assert len(manager.get_block_ids(request.request_id)[0]) == 3
    try:
        registry.retire(allocation, ReleaseAck(True, allocation.retention_token))
    except ContractError:
        pass
    else:
        raise AssertionError("live owner guard must prohibit retirement")
    guard.release()
    registry.retire(allocation, ReleaseAck(True, allocation.retention_token))
    manager.free(request)
    assert all(not group for group in manager.get_block_ids(request.request_id))
    assert manager.block_pool.get_num_free_blocks() == 15
    print(
        json.dumps(
            {
                "result": "PASS",
                "actual_framework_scheduler_allocation": True,
                "refcounts_unchanged_by_retain": True,
                "decode_append_preserves_prompt_retention": True,
                "retire_before_free": True,
            }
        ),
        flush=True,
    )


def worker(rank, tp, port, run, hostname):
    def mark(step):
        print(json.dumps({"rank": rank, "step": step}), flush=True)

    mark("set-device")
    torch.npu.set_device(rank)
    group = None
    if tp > 1:
        from vllm.distributed import get_tp_group, init_distributed_environment, initialize_model_parallel

        init_distributed_environment(
            world_size=tp, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=rank, backend="hccl"
        )
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(tensor_model_parallel_size=tp)
        group = get_tp_group()
    topology = HostPoolTopology(rank, tp, device_id=rank, tp_group=group)
    region = allocate_mooncake_host_region(
        size_bytes=2 * 1024 * 1024, alignment=2 * 1024 * 1024, topology=topology, name="a1a2_" + run
    )
    pool = MooncakeHostPool(region, topology)
    # HostRegister exposes a separate device VA. Read the same shm through its
    # real CPU VA; generic Tensor.cpu() cannot classify that device alias here.
    host_buffer = (ctypes.c_ubyte * pool.nbytes).from_address(region.handle.base_addr() + region.segment_offset)
    host_bytes = torch.frombuffer(host_buffer, dtype=torch.int8)

    def host_cpu(tensor):
        offset = tensor.data_ptr() - pool.data_ptr
        size = tensor.numel() * tensor.element_size()
        return host_bytes[offset : offset + size].view(tensor.dtype).view(tensor.shape)

    manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
    manager._host_kv_allocator = pool
    manager._target_owner_lock = threading.RLock()
    manager._target_owner_facade = None
    manager.fused_overlap_membership_region = None
    manager.num_target_layers = 2
    manager.current_kv_save_stream = torch.npu.Stream()
    manager.fused_plan_stream = torch.npu.Stream()
    consumption = torch.npu.Stream()
    manager.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    caches, main_names, index_names, refs = {}, [], [], []
    manager.k_caches_cpu, manager.v_caches_cpu = [], []
    for layer in range(2):
        name = f"model.layers.{layer}.self_attn.attn"
        main_names.append(name)
        raw_k, raw_v = pool.allocate_tensors([256, 256], alignment=32)
        k = raw_k.view(torch.bfloat16).view(4, 16, 1, 2)
        v = raw_v.view(torch.bfloat16).view(4, 16, 1, 2)
        manager.k_caches_cpu.append(k)
        manager.v_caches_cpu.append(v)
        caches[name] = (None, None, k, v, None, None)
        index = name + ".indexer"
        index_names.append(index)
        raw_data = torch.empty(2 * 1024 * 1024, dtype=torch.int8, device=f"npu:{rank}")
        raw_scale = torch.empty(2 * 1024 * 1024, dtype=torch.int8, device=f"npu:{rank}")
        refs.extend((raw_data, raw_scale))
        caches[index] = (raw_data[:512].view(4, 16, 1, 8), raw_scale[:128].view(torch.float16).view(4, 16, 1, 1))
    manager.offload_layer_names = main_names
    manager.layer_name_to_offload_id = {n: i for i, n in enumerate(main_names)}
    main_spec = AscendMLAAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=2, head_size_v=2, dtype=torch.bfloat16, store_on_host=True
    )
    index_spec = AscendSFAIndexerCacheSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.int8, scale_dim=1, scale_dtype=torch.float16
    )
    manager.kv_cache_config = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[
            SimpleNamespace(layer_names=main_names, kv_cache_spec=main_spec),
            SimpleNamespace(layer_names=index_names, kv_cache_spec=index_spec),
        ],
    )
    te = GlobalTE()
    te.get_transfer_engine(hostname, None)

    def register_existing_allocations(kv_caches):
        assert kv_caches is caches
        te.register_buffer(
            [pool.data_ptr, *[t.data_ptr() for t in refs]],
            [pool.nbytes, *[t.numel() for t in refs]],
            [f"npu:{rank}", *["*" for _ in refs]],
        )

    scheduler = SimpleNamespace(get_block_ids=lambda request: ((1, 3), (1, 3)))
    retention = SchedulerRetentionRegistry(scheduler)
    facade = DeferredTargetProvider()
    reject(facade.catalog)
    # Exercise real A2 connector methods without creating model/transport
    # threads. B's registration output is supplied above using native TE.
    worker_connector = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    worker_connector.engine = te.transfer_engine
    worker_connector.kv_recv_layer_thread = None
    worker_connector.target_provider = facade
    worker_connector.register_kv_caches = register_existing_allocations
    connector = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    connector.connector_worker = worker_connector
    connector.connector_scheduler = None
    connector.target_provider = facade
    connector._target_owner_binding = None
    connector.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config={"layerwise_receipt_mode": True})
    )
    multi = AscendMultiConnector.__new__(AscendMultiConnector)
    multi._connectors = [connector]
    multi._configure_target_provider()
    assert connector._target_provider is facade and worker_connector._target_provider is facade
    reject(lambda: connector.register_kv_caches(caches))
    reject(te.registration_snapshot)
    reject(
        lambda: connector.configure_target_owner(
            manager, retention=retention, consumption_stream=consumption, timeout=20.0, te=GlobalTE()
        )
    )
    connector.configure_target_owner(manager, retention=retention, consumption_stream=consumption, timeout=20.0, te=te)
    mark("bind-start")
    connector.register_kv_caches(caches)
    bound = manager._target_owner_facade
    mark("bind-complete")
    assert bound is facade and facade.ready
    reject(
        lambda: manager.bind_registered_target_owner(
            caches, te, retention=retention, consumption_stream=consumption, timeout=20.0
        )
    )
    catalog = facade.catalog()
    assert pickle.loads(pickle.dumps(catalog)) == catalog
    assert len(catalog.entries) == 8 and catalog.required_layers == {Component.MAIN: (0, 1), Component.INDEXER: (0, 1)}
    assert {e.group for e in catalog.entries if e.component is Component.MAIN} == {0}
    assert {e.group for e in catalog.entries if e.component is Component.INDEXER} == {1}
    host_ids = {r.backing_id for r in catalog.regions if r.memory_kind == "host"}
    if tp > 1:
        peers = [None] * tp
        torch.distributed.all_gather_object(
            peers, (tuple(host_ids), tuple(r.region_id for r in catalog.regions)), group=group.cpu_group
        )
        assert all(p[0] == peers[0][0] for p in peers)
        assert all(set(peers[0][1]).isdisjoint(p[1]) for p in peers[1:])
    request = RequestKey("npu-" + run, 1)
    allocation = retention.issue("owned-request", BlockAllocation(request, 1, ((1, 3), (1, 3)), 16, 23))
    lease = facade.acquire_target(request, allocation)
    mark("lease-acquired")
    for action in (te.unregister_buffer, pool.close, manager.close):
        reject(action)
    reject(lambda: facade.prepare_decode(lease, ReceiveReceipt(request, 1, True, True, True)))
    # A failed prepare is terminal in the provider. Release safely, then use a
    # distinct transfer generation for the successful real-dependency path.
    mark("release-start")
    facade.release_target(lease)
    mark("release-complete")
    retention.retire(allocation, ReleaseAck(True, allocation.retention_token))
    request = RequestKey("npu-" + run, 2)
    allocation = retention.issue("owned-request", BlockAllocation(request, 2, ((1, 3), (1, 3)), 16, 23))
    lease = facade.acquire_target(request, allocation)
    mark("lease-acquired")
    mark("receive-writes-start")
    producer = torch.npu.Stream()
    with torch.npu.stream(producer):
        for layer in range(2):
            for tensor in (manager.k_caches_cpu[layer], manager.v_caches_cpu[layer]):
                tensor.fill_(layer + 3)
            caches[index_names[layer]][0].fill_(rank + 5)
            caches[index_names[layer]][1].fill_(rank + 7)
        device_done = torch.npu.Event()
        device_done.record()
    # This local completion proves Host-memory visibility for the harness's
    # local writes only. B must supply remote Host-visible completion in PD.
    host_visible = threading.Event()
    device_done.synchronize()
    host_visible.set()
    if tp > 1:
        torch.distributed.barrier(group=group.cpu_group)
    mark("receive-writes-complete")
    reject(
        lambda: facade._owner.runtime._streams.install_visibility(
            ReceivedTargetWrites(lease, host_visible, torch.npu.Event())
        )
    )
    facade.publish_received(lease, host_visible, device_done)
    facade.prepare_decode(lease, ReceiveReceipt(request, 2, True, True, True))
    mark("use-start")
    with facade.use(lease):
        mark("native-use-entered")
        for layer in range(2):
            assert bool(torch.all(host_cpu(manager.k_caches_cpu[layer]) == layer + 3))
            assert bool(torch.all(host_cpu(manager.v_caches_cpu[layer]) == layer + 3))
            assert bool(torch.all(caches[index_names[layer]][0].cpu() == rank + 5))
            assert bool(torch.all(caches[index_names[layer]][1].cpu() == rank + 7))
        if tp > 1:
            torch.distributed.barrier(group=group.cpu_group)
        # Native writeback-stream work must outlive the Python use context.
        with torch.npu.stream(manager.current_kv_save_stream):
            a = torch.ones((4096, 4096), dtype=torch.float16, device=f"npu:{rank}")
            b = torch.ones_like(a)
            product = torch.empty_like(a)
            for _ in range(64):
                torch.mm(a, b, out=product)
            if rank == 0:
                manager.k_caches_cpu[0].fill_(19)
            writeback_done = torch.npu.Event()
            writeback_done.record()
    runtime = facade._owner.runtime
    pending = not writeback_done.query()
    assert pending, "test needs genuinely pending native writeback"
    runtime._timeout = 0.000001
    reject(lambda: facade.release_target(lease), expected=(TimeoutError,))
    reject(manager.close)
    reject(lambda: retention.retire(allocation, ReleaseAck(True, allocation.retention_token)))
    writeback_done.synchronize()
    runtime._timeout = 20.0
    mark("release-start")
    facade.release_target(lease)
    mark("release-complete")
    retention.retire(allocation, ReleaseAck(True, allocation.retention_token))
    if tp > 1:
        torch.distributed.barrier(group=group.cpu_group)
    assert bool(torch.all(host_cpu(manager.k_caches_cpu[0]) == 19))
    # Catalog's permanent pins still protect registrations until owner close.
    reject(te.unregister_buffer)
    facade.close()
    te.unregister_buffer()
    manager.close()
    assert region._released and not te.is_register_buffer
    print(
        json.dumps(
            {
                "result": "PASS",
                "rank": rank,
                "tp_size": tp,
                "entries": len(catalog.entries),
                "host_backing_ids": sorted(host_ids),
                "real_registration_generation": lease.regions[0].registration_generation,
                "retention_and_explicit_close": True,
                "native_visibility_and_content": True,
                "pending_native_writeback_blocked_release": pending,
                "shared_identity_verified": tp > 1,
                "scope": "synthetic native A1/A2, not model PD transfer",
            }
        ),
        flush=True,
    )
    if tp > 1:
        from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument("--hostname", required=True)
    args = parser.parse_args()
    exercise_real_scheduler()
    run = uuid.uuid4().hex
    if args.tp == 1:
        worker(0, 1, 0, run, args.hostname)
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        torch.multiprocessing.spawn(worker, args=(args.tp, port, run, args.hostname), nprocs=args.tp, join=True)

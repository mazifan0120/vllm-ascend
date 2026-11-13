# SPDX-License-Identifier: Apache-2.0
"""Production bounded MR registration and real native WRITE; no model claim."""

import argparse
import ctypes
import json
import socket
import threading
import time
import uuid
from collections import defaultdict
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
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import GlobalTE, global_te


def native_producer(manifest, hostname, side_port, connection):
    """A distinct process/NPU uses the production writer and real peer proof path."""
    import queue
    import traceback

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_peer import (
        ManifestPeerProofs,
        resolve_manifest_addresses,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_writer import ManifestSourceWriter
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import (
        LayerSource,
        RequestSource,
        SlotEpoch,
        SourceView,
    )
    from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
    from vllm_ascend.distributed.kv_transfer.utils.utils import align_memory

    writer = None
    try:
        import faulthandler
        import gc

        faulthandler.enable(all_threads=True)
        torch.npu.set_device(1)
        engine = global_te.get_transfer_engine(hostname, None)
        metadata = SimpleNamespace(
            remote_target_manifest=manifest,
            remote_host=hostname,
            remote_port=side_port,
            request_instance=manifest.lease.request.request_instance,
        )
        work = queue.Queue()
        writer = ManifestSourceWriter(0, work, resolve_manifest_addresses, ManifestPeerProofs(0))
        writer.register_request("native-producer", metadata)
        entries = manifest.targets[0].catalog.entries
        raw_storage, tensors = {}, {}
        alignment = 2 * 1024 * 1024
        # Main buffers are reused across layers, Indexer allocations remain distinct.
        for entry in entries:
            slot = (entry.component, entry.entry, None if entry.component is Component.MAIN else entry.logical_layer)
            if slot not in tensors:
                raw = torch.empty(alignment * 2, dtype=torch.int8, device="npu:1")
                raw_storage[slot] = raw
                tensors[slot] = align_memory(raw, alignment)[:alignment]
        global_te.register_buffer([t.data_ptr() for t in tensors.values()], [alignment] * len(tensors))
        completed = []
        native_calls = 0
        for layer in range(2):
            views = []
            for entry in entries:
                if entry.logical_layer != layer:
                    continue
                view = entry.binding.view
                slot = (entry.component, entry.entry, None if entry.component is Component.MAIN else layer)
                tensor = tensors[slot]
                span = (entry.shape[0] - 1) * entry.strides_bytes[0] + entry.binding.page_bytes
                data = torch.full((span,), -91, dtype=torch.int8)
                token_bytes = entry.binding.page_bytes // view.page_size_tokens
                pages = view.manager_block_size_tokens // view.page_size_tokens
                for token in range(23):
                    ordinal, offset = divmod(token, 16)
                    physical = (2, 0)[ordinal]
                    page, within = divmod(offset, view.page_size_tokens)
                    at = (physical * pages + page) * entry.strides_bytes[0] + within * token_bytes
                    data[at : at + token_bytes] = layer * 40 + entry.entry * 5 + token + 1
                tensor[:span].copy_(data)
                views.append(
                    SourceView(
                        layer,
                        entry.component.value.upper(),
                        entry.group,
                        str(slot),
                        tensor.data_ptr(),
                        span,
                        entry.shape,
                        entry.strides_bytes,
                        view.dtype,
                        view.manager_block_size_tokens,
                        view.page_size_tokens,
                        entry.entry,
                    )
                )
            ready = torch.npu.Event()
            ready.record()
            source = LayerSource(
                layer, tuple(SlotEpoch("native-P", v.slot, layer + 1, layer) for v in views), tuple(views)
            )
            request = RequestSource("native-producer", "store-private", 0, 23, ((2, 0), (2, 0)))
            job = writer.prepare_job(source, request, ready)
            job.publish(lambda error, quiet: completed.append((None if error is None else str(error), quiet)))
            work.get_nowait().execute(engine)
            native_calls += 2  # Per layer: Host Main and device Indexer batches.
            assert completed[-1] == (None, True), completed
        writer._proof_queue.join()
        assert writer._requests["native-producer"].delivery_error is None
        writer.finish_requests({"native-producer"})
        writer.close()
        print(json.dumps({"source_step": "writes-and-proofs-complete"}), flush=True)
        connection.send(
            {
                "status": "WRITES_COMPLETE",
                "source_device": 1,
                "source_completions": completed,
                "expected_native_batches": native_calls,
                "real_mooncake_write": True,
            }
        )
        assert connection.poll(90), "D did not acknowledge target byte verification"
        assert connection.recv() == "TARGET_BYTES_VERIFIED"
        print(json.dumps({"source_step": "target-bytes-verified"}), flush=True)
        torch.npu.synchronize()
        global_te.unregister_buffer()
        print(json.dumps({"source_step": "unregister-complete"}), flush=True)
        # Destroy the TE while its local tensors and NPU context still live.
        # This is a diagnostic ordering experiment, not a production fix.
        global_te.transfer_engine = None
        faulthandler.disable()
        faulthandler.enable(all_threads=True)
        del engine
        gc.collect()
        print(json.dumps({"source_step": "engine-release-complete"}), flush=True)
        torch.npu.synchronize()
        connection.send({"status": "CLEANUP_COMPLETE"})
    except Exception:
        connection.send({"status": "FAIL", "traceback": traceback.format_exc()})
        raise
    finally:
        connection.close()


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
        model_config=SimpleNamespace(enforce_eager=True, hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
        kv_transfer_config=SimpleNamespace(is_kv_producer=False, is_kv_consumer=True),
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
        kv_cache_tensors=[],
        kv_cache_groups=[
            SimpleNamespace(layer_names=main_names, kv_cache_spec=main_spec),
            SimpleNamespace(layer_names=index_names, kv_cache_spec=index_spec),
        ],
    )
    te = global_te
    te.get_transfer_engine(hostname, None)
    with socket.socket() as sock:
        sock.bind((hostname, 0))
        side_port = sock.getsockname()[1]

    scheduler = SimpleNamespace(get_block_ids=lambda request: ((1, 3), (1, 3)))
    retention = SchedulerRetentionRegistry(scheduler)
    facade = DeferredTargetProvider()
    reject(facade.catalog)
    # Exercise production worker registration and its actual receive thread,
    # over deliberately tiny real allocations, without constructing a model.
    worker_connector = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    worker_connector.engine = te.transfer_engine
    worker_connector.kv_recv_layer_thread = None
    worker_connector.target_provider = facade
    worker_connector.tp_rank, worker_connector.tp_size = rank, tp
    worker_connector.pd_head_ratio = 1
    worker_connector.engine_id = "native-target-" + run
    worker_connector.side_channel_port = side_port
    worker_connector.te_rpc_port = te.transfer_engine.get_rpc_port()
    worker_connector.vllm_config = manager.vllm_config
    worker_connector.kv_cache_config = manager.kv_cache_config
    worker_connector.kv_cache_specs = [main_spec, index_spec]
    worker_connector.use_attn_mamba_hybrid = False
    worker_connector.enable_kv_quant = False
    worker_connector.enable_c8_quant = False
    worker_connector.sparse_sfa_c8_layers = set()
    worker_connector.attn_resharding_group_idx = set()
    worker_connector.kernel_block_size_scale = [1, 1]
    worker_connector.layer_metadata = {}
    worker_connector.index_to_name = defaultdict(list)
    worker_connector.total_layers = 2
    worker_connector.request_map = {}
    worker_connector._recving_metadata = {}
    worker_connector._failed_receives = set()
    worker_connector._invalid_block_ids = set()
    worker_connector.virtual_request = set()
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
    mark("remote-receive-start")
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import (
        RankTarget,
        TargetManifest,
        aggregate_rank_targets,
    )

    targets = (RankTarget(catalog, lease),)
    manifest = TargetManifest(aggregate_rank_targets(targets), targets)
    internal_id = "native-target-12345678"
    delivery = worker_connector._target_delivery
    delivery.admit_target(manifest, internal_id)
    # Initialize the real destinations before the P process can submit any writes.
    for layer in range(2):
        host_cpu(manager.k_caches_cpu[layer]).view(torch.int8).fill_(-91)
        host_cpu(manager.v_caches_cpu[layer]).view(torch.int8).fill_(-91)
        for tensor in caches[index_names[layer]]:
            tensor.view(torch.int8).fill_(-91)
    torch.npu.synchronize()
    context = torch.multiprocessing.get_context("spawn")
    recv, send = context.Pipe(duplex=True)
    producer = context.Process(target=native_producer, args=(manifest, hostname, side_port, send))
    producer.start()
    send.close()
    assert recv.poll(180), "Native producer did not report before the bounded deadline"
    result = recv.recv()
    print(json.dumps(result), flush=True)
    assert result["status"] == "WRITES_COMPLETE", result
    deadline = time.monotonic() + 30
    while not delivery.delivery_status(lease)["ready"]:
        assert time.monotonic() < deadline, "D did not prepare actual native receive dependencies"
        time.sleep(0.01)
    assert delivery.delivery_status(lease)["remote_quiesced"]

    def verify_entry(entry, actual):
        view = entry.binding.view
        expected = torch.full((actual.numel(),), -91, dtype=torch.int8)
        token_bytes = entry.binding.page_bytes // view.page_size_tokens
        pages = view.manager_block_size_tokens // view.page_size_tokens
        for token in range(23):
            ordinal, offset = divmod(token, 16)
            physical = (1, 3)[ordinal]
            page, within = divmod(offset, view.page_size_tokens)
            at = (physical * pages + page) * entry.strides_bytes[0] + within * token_bytes
            expected[at : at + token_bytes] = entry.logical_layer * 40 + entry.entry * 5 + token + 1
        assert torch.equal(actual.reshape(-1), expected), (entry.component, entry.logical_layer, entry.entry)

    mark("remote-receive-ready")
    mark("use-start")
    with delivery.use_context((internal_id,)):
        mark("native-use-entered")
        for entry in catalog.entries:
            if entry.component is Component.MAIN:
                tensor = (manager.k_caches_cpu, manager.v_caches_cpu)[entry.entry][entry.logical_layer]
                actual = host_cpu(tensor).view(torch.int8).reshape(-1)
            else:
                tensor = caches[index_names[entry.logical_layer]][entry.entry]
                actual = tensor.view(torch.int8).cpu().reshape(-1)
            verify_entry(entry, actual)
        mark("all-target-bytes-verified")
        recv.send("TARGET_BYTES_VERIFIED")
        assert recv.poll(90), "Source did not complete explicit native cleanup"
        source_cleanup = recv.recv()
        print(json.dumps({"source_cleanup": source_cleanup}), flush=True)
        producer.join(30)
        assert source_cleanup["status"] == "CLEANUP_COMPLETE" and producer.exitcode == 0, source_cleanup
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
    reject(lambda: delivery.release_target(lease), expected=(TimeoutError,))
    reject(manager.close)
    reject(lambda: retention.retire(allocation, ReleaseAck(True, allocation.retention_token)))
    writeback_done.synchronize()
    runtime._timeout = 20.0
    mark("release-start")
    delivery.release_target(lease)
    mark("release-complete")
    retention.retire(allocation, ReleaseAck(True, allocation.retention_token))
    if tp > 1:
        torch.distributed.barrier(group=group.cpu_group)
    assert bool(torch.all(host_cpu(manager.k_caches_cpu[0]) == 19))
    # Catalog's permanent pins still protect registrations until owner close.
    reject(te.unregister_buffer)
    registration_count = len(te._registered_regions)
    facade.close()
    te.unregister_buffer()
    worker_connector.engine = None
    te.transfer_engine = None
    import gc

    gc.collect()
    mark("target-engine-release-complete")
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
                "scope": "production bounded registration, real cross-process Mooncake writer/read, not model PD",
                "real_remote_write_read": True,
                "target_registrations": registration_count,
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
    parser.add_argument("--tp", type=int, choices=(1,), default=1)
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

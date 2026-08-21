# SPDX-License-Identifier: Apache-2.0
"""Shared stubs for DualPath unit tests.

Mirrors the module-stub preamble of ``test_dual_path_connector.py``: the
optional Mooncake engine, torch_npu, and uvloop extensions are stubbed before
any vllm_ascend import so the suites run on a plain CPU checkout.
"""

import contextlib
import importlib.util
import sys
import threading
import types
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool

_fake_engine = types.ModuleType("mooncake.engine")
_fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("mooncake.engine", _fake_engine)

_fake_torch_npu = types.ModuleType("torch_npu")
_fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
_fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
_fake_torch_npu.npu.current_device = MagicMock(return_value=0)  # type: ignore[attr-defined]
_fake_torch_npu.npu.Stream = MagicMock  # type: ignore[attr-defined]
_fake_torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", _fake_torch_npu)
torch.npu = _fake_torch_npu.npu  # type: ignore[attr-defined]

_fake_uvloop = types.ModuleType("uvloop")
_fake_uvloop.__spec__ = importlib.util.spec_from_loader("uvloop", loader=None)
sys.modules.setdefault("uvloop", _fake_uvloop)

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import metadata as metadata_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorWorker  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (  # noqa: E402
    DualPathRequestKey,
    PathDecisionRequest,
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (  # noqa: E402
    DecodeControlEndpoint,
)

_SCHEDULER_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler"
PE_TEST_BLOCK_SIZE = 16
PE_TEST_POOL_BLOCKS = 128
DECODE_TEST_INSTANCE_ID = "decode-engine:2:boot-7"
DECODE_TEST_CONTROL_ENDPOINT = DecodeControlEndpoint(host="192.0.2.44", port=24001)


def init_dual_path_worker_state(worker: DualPathConnectorWorker, role: str = "decode") -> DualPathConnectorWorker:
    """Initialize every DualPath-owned field of a bare worker instance.

    ``DualPathConnectorWorker.__init__`` initializes these fields
    unconditionally, so production code reads them via direct attribute
    access. Tests that build a worker with ``object.__new__`` must route
    through this helper to stay pinned to that field set. The KVPool worker
    adapter is left as None; tests override it with their own stub.
    """
    worker.dual_path_cfg = DualPathConfig(role=role)
    worker._kvpool_worker_adapter = None
    worker._registered_kv_caches = None
    worker._registered_layer_order = ()
    worker._accepting_split_requests = True
    worker._split_trackers = {}
    worker._reverse_terminal_lock = threading.Lock()
    worker._pending_local_reverse_terminals = {}
    worker._control_failed_recving = set()
    worker._forward_receive_bindings = {}
    worker._pending_forward_done_wire_ids = set()
    worker._pending_forward_failed_wire_ids = set()
    worker._consumed_forward_terminal_wire_ids = {}
    worker._reverse_receive_bindings = {}
    worker._reverse_request_map = {}
    worker._pending_reverse_done_wire_ids = set()
    worker._pending_reverse_failed_wire_ids = set()
    worker._consumed_reverse_terminal_wire_ids = {}
    worker._completion_reports_lock = threading.Lock()
    worker._pending_completion_reports = {}
    worker._pending_failure_reports = {}
    worker._completion_report_outcomes = {}
    return worker


@contextlib.contextmanager
def worker_environment():
    transfer_engine = MagicMock(name="transfer_engine")
    transfer_engine.get_rpc_port.return_value = 9090
    transfer_engine.initialize.return_value = 0
    transfer_engine.register_memory.return_value = 0
    send_threads = []
    recv_threads = []

    def make_send_thread(*_args, **_kwargs):
        thread = MagicMock(name=f"send_thread_{len(send_threads)}")
        send_threads.append(thread)
        return thread

    def make_recv_thread(*_args, **_kwargs):
        thread = MagicMock(name=f"recv_thread_{len(recv_threads)}")
        recv_threads.append(thread)
        return thread

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.Tensor.size", return_value=(10, 16, 8, 16)))
        stack.enter_context(patch("torch.Tensor.element_size", return_value=4))
        stack.enter_context(patch("torch.Tensor.data_ptr", return_value=0x1000))
        stack.enter_context(patch("math.prod", return_value=128))
        stack.enter_context(patch("random.Random"))
        stack.enter_context(patch.object(layerwise_module, "get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(layerwise_module, "get_tp_group", return_value=None))
        stack.enter_context(patch.object(layerwise_module, "get_ip", return_value="127.0.0.1"))
        stack.enter_context(
            patch.object(layerwise_module, "string_to_int64_hash", side_effect=lambda value: hash(value))
        )
        get_transfer_engine = stack.enter_context(
            patch.object(layerwise_module.global_te, "get_transfer_engine", return_value=transfer_engine)
        )
        register_buffer = stack.enter_context(
            patch.object(layerwise_module.global_te, "register_buffer", return_value=None)
        )
        send_factory = stack.enter_context(
            patch.object(layerwise_module, "KVCacheSendingLayerThread", side_effect=make_send_thread)
        )
        recv_factory = stack.enter_context(
            patch.object(layerwise_module, "KVCacheRecvingLayerThread", side_effect=make_recv_thread)
        )
        stack.enter_context(patch.object(layerwise_module, "logger", MagicMock()))
        stack.enter_context(patch.object(layerwise_module.threading, "Event", MagicMock()))
        stack.enter_context(
            patch.object(
                layerwise_module,
                "get_ascend_config",
                return_value=SimpleNamespace(pd_tp_ratio=1, num_head_replica=1, pd_head_ratio=1),
            )
        )
        stack.enter_context(
            patch.object(
                layerwise_module,
                "get_pcp_group",
                return_value=SimpleNamespace(world_size=1, rank_in_group=0),
            )
        )
        stack.enter_context(
            patch.object(layerwise_module, "get_decode_context_model_parallel_world_size", return_value=1, create=True)
        )
        stack.enter_context(patch.object(layerwise_module, "get_decode_context_model_parallel_rank", return_value=0))
        stack.enter_context(
            patch.object(
                layerwise_module,
                "npu_stream_switch",
                side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
            )
        )
        yield SimpleNamespace(
            transfer_engine=transfer_engine,
            get_transfer_engine=get_transfer_engine,
            register_buffer=register_buffer,
            send_factory=send_factory,
            recv_factory=recv_factory,
            send_threads=send_threads,
            recv_threads=recv_threads,
        )


class FixedPathPolicy:
    def __init__(self, path: PathKind) -> None:
        self.path = path
        self.calls = 0

    def choose(self, request: PathDecisionRequest) -> PathKind:
        self.calls += 1
        return self.path


def make_block_pool(num_blocks: int = PE_TEST_POOL_BLOCKS) -> BlockPool:
    return BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=PE_TEST_BLOCK_SIZE)


def make_prefill_vllm_config(world_size: int = 1) -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.is_kv_consumer = False
    config.kv_transfer_config.is_kv_producer = True
    config.kv_transfer_config.engine_id = "prefill-engine"
    config.kv_transfer_config.kv_port = 5000
    config.kv_transfer_config.kv_load_failure_policy = "fail"
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {"tls_config": {}}.get(
        key, default
    )
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = world_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.world_size = world_size
    config.cache_config.block_size = PE_TEST_BLOCK_SIZE
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    return config


def make_prefill_kv_cache_config() -> SimpleNamespace:
    spec = MagicMock()
    spec.block_size = PE_TEST_BLOCK_SIZE
    group = MagicMock()
    group.kv_cache_spec = spec
    group.layer_names = ["layer.0"]
    return SimpleNamespace(kv_cache_groups=[group], kv_cache_tensors=[], num_blocks=64)


def make_completed_future() -> Future[None]:
    future: Future[None] = Future()
    future.set_result(None)
    return future


def make_empty_scheduler_output(preempted_req_ids: set[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
        num_scheduled_tokens={},
        preempted_req_ids=preempted_req_ids,
    )


def make_worker_metadata(
    completion_reports: dict[int, int] | None = None, failure_reports: dict[int, int] | None = None
):
    return metadata_module.DualPathWorkerMetadata(
        completion_reports=completion_reports or {},
        failure_reports=failure_reports or {},
    )


def make_sender_req_meta() -> layerwise_module.ReqMeta:
    return layerwise_module.ReqMeta(
        local_block_ids=[[5]],
        token_ids=[1],
        remote_block_ids=[[10]],
        remote_block_size=[[16]],
        remote_engine_id="remote_engine",
        remote_host="127.0.0.1",
        remote_port=7777,
        remote_te_rpc_port=6000,
        remote_layer_metadata={},
        metaserver=None,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        chunk_finish=True,
        trans_count=[1],
    )


def make_sending_layer_thread() -> layerwise_module.KVCacheSendingLayerThread:
    engine = MagicMock()
    engine.batch_transfer_sync_write.return_value = 1
    vllm_config = MagicMock()
    vllm_config.cache_config.mamba_cache_mode = None
    vllm_config.speculative_config = None
    kv_cache_config = MagicMock()
    group_spec = MagicMock()
    group_spec.kv_cache_spec = MagicMock(block_size=PE_TEST_BLOCK_SIZE)
    kv_cache_config.kv_cache_groups = [group_spec]
    layer_metadata = {
        "layer0": layerwise_module.LayerMetadata(
            tensor_group_idx=[0],
            kv_caches_base_addr=[1000, 2000],
            block_len=[1024],
            block_size_scale=[1],
        ),
    }
    return layerwise_module.KVCacheSendingLayerThread(
        engine=engine,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        kv_cache_specs=[MagicMock(block_size=PE_TEST_BLOCK_SIZE)],
        attn_resharding_group_idx=set(),
        total_layers=1,
        ready_event=threading.Event(),
        tp_size=1,
        tp_rank=0,
        pd_head_ratio=1,
        num_head_replica=1,
        layer_metadata=layer_metadata,
        use_mla=True,
        use_attn_mamba_hybrid=False,
        k_buffer=MagicMock(),
        v_buffer=MagicMock(),
        enable_kv_quant=False,
        enable_c8_quant=False,
        resharding_stream=MagicMock(),
        callback_func=MagicMock(),
    )


@contextlib.contextmanager
def successful_terminal_ack_zmq_ctx(_socket_type, _addr):
    sock = MagicMock(name="ack_sock")
    sock.poll.return_value = True
    sock.recv.return_value = b"ACK"
    yield sock


@pytest.fixture()
def pe_scheduler_factory():
    schedulers = []
    with (
        patch(f"{_SCHEDULER_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_SCHEDULER_NS}.get_ip", return_value="192.0.2.44"),
    ):
        coordinator = MagicMock(name="prefill_coordinator")
        coordinator.submit.return_value = make_completed_future()
        coordinator.take_received_aborts.return_value = []
        coordinator_cls.for_prefill.return_value = coordinator

        def make(
            path: PathKind = PathKind.PE_READ,
            *,
            world_size: int = 1,
            pool: BlockPool | None = None,
            policy=None,
            prefill_control_port: int | None = None,
        ):
            scheduler = connector_module.DualPathConnectorScheduler(
                make_prefill_vllm_config(world_size),
                make_prefill_kv_cache_config(),
                "prefill-engine",
                DualPathConfig(
                    role="prefill",
                    prefill_control_port=prefill_control_port,
                ),
                path_policy=policy if policy is not None else FixedPathPolicy(path),
            )
            scheduler.executor.shutdown(wait=False)
            scheduler.metaserver_client.close()
            scheduler.executor = MagicMock(name="prefill_executor")
            scheduler.side_channel_host = "198.51.100.20"
            if pool is not None:
                scheduler.bind_gpu_block_pool(pool)
            schedulers.append(scheduler)
            return scheduler, coordinator

        yield make

    for scheduler in schedulers:
        scheduler.shutdown()


def make_decode_vllm_config(world_size: int = 1) -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.is_kv_consumer = True
    config.kv_transfer_config.engine_id = "decode-engine"
    config.kv_transfer_config.kv_port = 5000
    config.kv_transfer_config.kv_load_failure_policy = "fail"
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {"tls_config": {}}.get(
        key, default
    )
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = world_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.world_size = world_size
    config.cache_config.block_size = PE_TEST_BLOCK_SIZE
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    return config


@pytest.fixture()
def decode_control_seams():
    with (
        patch(f"{_SCHEDULER_NS}.KVPoolSchedulerAdapter") as adapter_cls,
        patch(f"{_SCHEDULER_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_SCHEDULER_NS}.get_ip", return_value="192.0.2.44"),
    ):
        decode_coordinator = MagicMock(name="decode_coordinator")
        decode_coordinator.decode_engine_instance_id = DECODE_TEST_INSTANCE_ID
        decode_coordinator.decode_control_endpoint = DECODE_TEST_CONTROL_ENDPOINT
        admission_ids = iter(range(1_000_000))
        decode_coordinator.new_request_key.side_effect = lambda request_id: DualPathRequestKey(
            DECODE_TEST_INSTANCE_ID,
            request_id,
            next(admission_ids),
        )
        prefill_coordinator = MagicMock(name="prefill_coordinator")
        coordinator_cls.for_decode.return_value = decode_coordinator
        coordinator_cls.for_prefill.return_value = prefill_coordinator
        yield SimpleNamespace(
            adapter_cls=adapter_cls,
            coordinator_cls=coordinator_cls,
            decode_coordinator=decode_coordinator,
            prefill_coordinator=prefill_coordinator,
        )


@pytest.fixture()
def decode_scheduler_factory(decode_control_seams):
    schedulers = []

    def make(*, world_size: int = 1, pool: BlockPool | None = None):
        scheduler = connector_module.DualPathConnectorScheduler(
            make_decode_vllm_config(world_size),
            make_prefill_kv_cache_config(),
            "decode-engine",
            DualPathConfig(role="decode", dual_path_control_port=7100),
        )
        scheduler.executor.shutdown(wait=False)
        scheduler.metaserver_client.close()
        scheduler.executor = MagicMock(name="decode_executor")
        scheduler.executor.submit.return_value = make_completed_future()
        scheduler.side_channel_host = "198.51.100.20"
        if pool is not None:
            scheduler.bind_gpu_block_pool(pool)
        schedulers.append(scheduler)
        return scheduler

    yield make
    for scheduler in schedulers:
        scheduler.shutdown()

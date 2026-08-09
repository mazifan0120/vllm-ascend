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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

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
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorWorker  # noqa: E402


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

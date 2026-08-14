# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Layerwise sender failure facts.

Covers the failed-session wrong-attribution fix, the guarantee that a failed
request never gets a success callback, and terminal-ACK failure surfacing as a
worker-visible fact.
"""

import importlib.util
import sys
import threading
import types
import unittest
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

# Clean up stale mock modules installed by other test files
# (e.g., ascend_store/_mock_deps.py) that replace real kv_transfer
# subpackages with MagicMock/fake modules, breaking our imports.
# We save the removed modules so we can restore them after our imports
# complete, so other test files (ascend_store) still see their mocks.
_kv_xfer = "vllm_ascend.distributed.kv_transfer"
_vllm_kv_xfer = "vllm.distributed.kv_transfer"
_saved_modules: dict[str, types.ModuleType] = {}
_to_remove = []
for _k in list(sys.modules):
    if _k.startswith(_kv_xfer):
        _suffix = _k[len(_kv_xfer) :]
        if _suffix == "" or _suffix.startswith(".utils") or _suffix.startswith(".kv_p2p"):
            _to_remove.append(_k)
    elif _k.startswith(_vllm_kv_xfer):
        _to_remove.append(_k)
for _m in _to_remove:
    _saved_modules[_m] = sys.modules.pop(_m)

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: E402
    KVCacheSendingLayerThread,
    LayerMetadata,
    MooncakeLayerwiseConnectorWorker,
    ReqMeta,
    SendTask,
)

# Restore the mocked modules so other test files still work correctly.
for _k, _v in _saved_modules.items():
    sys.modules[_k] = _v


def _make_layer_metadata(**overrides):
    defaults = dict(
        tensor_group_idx=[0],
        kv_caches_base_addr=[1000, 2000],
        block_len=[1024],
        block_size_scale=[1],
    )
    defaults.update(overrides)
    return LayerMetadata(**defaults)


def _make_sending_thread(engine=None, total_layers=2, callback_func=None) -> KVCacheSendingLayerThread:
    if engine is None:
        engine = MagicMock()
        engine.batch_transfer_sync_write.return_value = 1

    vllm_config = MagicMock()
    vllm_config.cache_config.mamba_cache_mode = None
    vllm_config.speculative_config = None

    kv_cache_config = MagicMock()
    group_spec = MagicMock()
    group_spec.kv_cache_spec = MagicMock(block_size=16)
    kv_cache_config.kv_cache_groups = [group_spec]

    layer_metadata = {
        "layer0": _make_layer_metadata(kv_caches_base_addr=[1000, 2000]),
        "layer1": _make_layer_metadata(kv_caches_base_addr=[3000, 4000]),
    }

    return KVCacheSendingLayerThread(
        engine=engine,
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        kv_cache_specs=[MagicMock(block_size=16)],
        attn_resharding_group_idx=set(),
        total_layers=total_layers,
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
        callback_func=callback_func if callback_func is not None else MagicMock(),
    )


def _make_req_meta(chunk_finish=True, remote_host="127.0.0.1", remote_te_rpc_port=6000) -> ReqMeta:
    return ReqMeta(
        local_block_ids=[[5]],
        token_ids=[1],
        remote_block_ids=[[10]],
        remote_block_size=[[16]],
        remote_engine_id="remote_engine",
        remote_host=remote_host,
        remote_port=7777,
        remote_te_rpc_port=remote_te_rpc_port,
        remote_layer_metadata={},
        metaserver=None,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        chunk_finish=chunk_finish,
        trans_count=[1],
    )


class TestLayerwiseSenderFailureFacts(unittest.TestCase):
    def test_failed_session_marks_every_member_req(self):
        engine = MagicMock()
        engine.batch_transfer_sync_write.return_value = -1
        thread = _make_sending_thread(engine=engine)
        thread.get_transfer_meta = MagicMock(return_value=([1], [2], [64]))

        req_meta_r1 = _make_req_meta()
        req_meta_r2 = _make_req_meta()
        send_task = SendTask(
            send_request={"r1": req_meta_r1, "r2": req_meta_r2},
            wait_event=MagicMock(),
            layer_idx=0,
            layer_name="layer0",
        )

        thread._transfer_kv_cache(send_task)

        self.assertEqual(thread.failed_reqs, {"r1", "r2"})

    def test_final_layer_callback_never_publishes_success_for_failed_req(self):
        callback_func = MagicMock()
        engine = MagicMock()
        engine.batch_transfer_sync_write.return_value = 1
        thread = _make_sending_thread(engine=engine, callback_func=callback_func)
        thread.get_transfer_meta = MagicMock(return_value=([1], [2], [64]))
        thread.failed_reqs.add("r2")

        req_meta_r1 = _make_req_meta()
        req_meta_r2 = _make_req_meta()
        send_task = SendTask(
            send_request={"r1": req_meta_r1, "r2": req_meta_r2},
            wait_event=MagicMock(),
            layer_idx=thread.total_layers - 1,
            layer_name="layer0",
        )

        thread._transfer_kv_cache(send_task)

        callback_func.assert_any_call("r1", req_meta_r1, 0, trans_flag=True)
        callback_func.assert_any_call("r2", req_meta_r2, 0, trans_flag=False)
        success_calls_for_r2 = [
            call
            for call in callback_func.call_args_list
            if call.args[0] == "r2" and call.kwargs.get("trans_flag") is True
        ]
        self.assertEqual(success_calls_for_r2, [])
        self.assertNotIn("r2", thread.failed_reqs)

    def test_terminal_ack_failure_is_drained_once(self):
        thread = _make_sending_thread()
        worker = object.__new__(MooncakeLayerwiseConnectorWorker)
        worker.kv_send_layer_thread = thread
        worker.side_channel_host = "127.0.0.1"
        worker.handshake_port = 9999
        worker.timeout = 0.01

        req_id = "req_terminal_ack_failure_000000000"
        req_meta = _make_req_meta()
        with patch.object(layerwise_module, "zmq_ctx", side_effect=RuntimeError("no route to host")):
            worker.send_done_send_signal(req_id, req_meta, 0, trans_flag=True)

        self.assertIn(req_id, thread.terminal_ack_failed)

        self.assertEqual(worker.drain_terminal_ack_failures(), {req_id})
        # A second drain returns nothing; the set was popped.
        self.assertEqual(worker.drain_terminal_ack_failures(), set())

        idle_worker = object.__new__(MooncakeLayerwiseConnectorWorker)
        idle_worker.kv_send_layer_thread = None
        self.assertEqual(idle_worker.drain_terminal_ack_failures(), set())


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Executable PR-00 contract for DualPath stage 1.

This suite implements DETAILED-SPEC section 11: the foundation config matrix,
call-through construction parity, ten Mooncake Layerwise behavior-parity
scenarios, section 9 inheritance/drift guards, and connector registration.
It is deterministic and stubs optional Mooncake/NPU extensions before imports.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import sys
import threading
import types
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import regex as re
import torch

fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine
fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
fake_torch_npu.npu.current_device = MagicMock(return_value=0)  # type: ignore[attr-defined]
fake_torch_npu.npu.Stream = MagicMock  # type: ignore[attr-defined]
fake_torch_npu.npu_fusion_attention = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", fake_torch_npu)
torch.npu = fake_torch_npu.npu  # type: ignore[attr-defined]
fake_uvloop = types.ModuleType("uvloop")
fake_uvloop.__spec__ = importlib.util.spec_from_loader("uvloop", loader=None)
sys.modules.setdefault("uvloop", fake_uvloop)

_ASCEND_KV_TRANSFER = "vllm_ascend.distributed.kv_transfer"
_VLLM_KV_TRANSFER = "vllm.distributed.kv_transfer"
_saved_modules: dict[str, types.ModuleType] = {}
_modules_to_remove: list[str] = []
for _module_name in list(sys.modules):
    if _module_name.startswith(_ASCEND_KV_TRANSFER):
        _suffix = _module_name[len(_ASCEND_KV_TRANSFER) :]
        if _suffix == "" or _suffix.startswith(".utils") or _suffix.startswith(".kv_p2p"):
            _modules_to_remove.append(_module_name)
    elif _module_name.startswith(_VLLM_KV_TRANSFER):
        _modules_to_remove.append(_module_name)
# Only purge stub modules installed by other suites (MagicMock/fake modules
# have no real __file__); real package modules are reused so class identity
# holds across test files within one pytest session.
_modules_to_remove = [
    _m for _m in _modules_to_remove if not isinstance(getattr(sys.modules[_m], "__file__", None), str)
]
for _module_name in _modules_to_remove:
    _saved_modules[_module_name] = sys.modules.pop(_module_name)

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)

from vllm_ascend.distributed.kv_transfer import register_connector as register_kv_connectors  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import (  # noqa: E402
    ALLOWED_EXTRA_CONFIG_KEYS,
    DualPathConfig,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (  # noqa: E402
    DualPathConnector,
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (  # noqa: E402
    DecodeControlEndpoint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: E402
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    SendTask,
    get_external_request_id,
)

for _module_name, _module in _saved_modules.items():
    sys.modules[_module_name] = _module

import pytest  # noqa: E402


# Task-01: Decode-role DualPath components compose KVPool lookup adapters.
# The parity/construction suites in this file target parent-behavior parity,
# so the adapters are replaced with mocks; the Task-01 admission contract
# itself is covered by test_decode_scheduler.py with the same seam.
@pytest.fixture(autouse=True)
def _patch_task01_adapters():
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.KVPoolAdapter"),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.KVPoolWorkerAdapter"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.PathDecisionCoordinator"
        ) as coordinator_cls,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.get_ip",
            return_value="127.0.0.1",
        ),
    ):
        decode_coordinator = MagicMock(name="decode_coordinator")
        decode_coordinator.decode_engine_instance_id = "test_engine:0:test-boot"
        decode_coordinator.decode_control_endpoint = DecodeControlEndpoint(host="127.0.0.1", port=7100)
        coordinator_cls.for_decode.return_value = decode_coordinator
        coordinator_cls.for_prefill.return_value = MagicMock(name="prefill_coordinator")
        yield


class MockVllmConfig:
    def __init__(self, dual_role="prefill", kv_role="kv_producer"):
        self.model_config = MagicMock()
        self.parallel_config = MagicMock()
        self.cache_config = MagicMock()
        self.scheduler_config = MagicMock()
        self.kv_transfer_config = MagicMock()
        self.speculative_config = None
        self.quant_config = None

        self.model_config.use_mla = True
        self.model_config.is_deepseek_mla = True
        self.model_config.hf_config.num_key_value_heads = 1
        self.model_config.hf_text_config = MagicMock()
        self.model_config.hf_text_config.model_type = "default"
        self.model_config.hf_text_config.num_key_value_heads = 1
        self.model_config.get_num_layers = MagicMock(return_value=1)
        self.model_config.get_total_num_hidden_layers = MagicMock(return_value=1)
        self.model_config.get_total_num_kv_heads = MagicMock(return_value=1)

        self.parallel_config.tensor_parallel_size = 2
        self.parallel_config.pipeline_parallel_size = 1
        self.parallel_config.data_parallel_rank_local = 0
        self.parallel_config.data_parallel_size_local = 1
        self.parallel_config.data_parallel_size = 1
        self.parallel_config.data_parallel_rank = 0
        self.parallel_config.prefill_context_parallel_size = 1
        self.parallel_config.decode_context_parallel_size = 1
        self.cache_config.block_size = 16
        self.cache_config.mamba_cache_mode = None
        self.scheduler_config.disable_hybrid_kv_cache_manager = True

        self.kv_transfer_config.engine_id = "test_engine"
        self.kv_transfer_config.kv_port = 5000
        self.kv_transfer_config.kv_load_failure_policy = "fail"
        self.kv_transfer_config.kv_role = kv_role
        self.kv_transfer_config.is_kv_producer = kv_role in {"kv_producer", "kv_both"}
        self.kv_transfer_config.is_kv_consumer = kv_role in {"kv_consumer", "kv_both"}
        self.kv_transfer_config.kv_connector_extra_config = {"role": dual_role}
        if dual_role == "decode":
            self.kv_transfer_config.kv_connector_extra_config["dual_path_control_port"] = 7100
        self.kv_transfer_config.get_from_extra_config = MagicMock()
        self.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {
            "tls_config": {},
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 2, "dp_size": 1},
        }.get(key, default)


class MockKVCacheConfig:
    def __init__(self, block_size=16):
        kv_cache_spec = MagicMock()
        kv_cache_spec.block_size = block_size
        group_spec = MagicMock()
        group_spec.kv_cache_spec = kv_cache_spec
        group_spec.layer_names = ["encoder.layer.0"]
        self.kv_cache_groups = [group_spec]
        self.kv_cache_tensors = []
        self.num_blocks = 10


class MockRequest:
    def __init__(self, request_id, prompt_token_ids=None, kv_transfer_params=None, status=None):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.prompt_embeds = None
        self.kv_transfer_params = kv_transfer_params or {}
        self.status = status or "running"
        self.output_token_ids = [101, 102]
        self.num_computed_tokens = 0
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self.num_tokens = len(self.prompt_token_ids)
        self.max_tokens = 16
        self.all_token_ids = list(self.prompt_token_ids)
        self._all_token_ids = list(self.prompt_token_ids)


class MockBlocks:
    def __init__(self, unhashed, block_ids_tuple=None):
        self._unhashed = list(unhashed)
        self._block_ids_tuple = block_ids_tuple if block_ids_tuple is not None else ([1, 2],)

    def get_unhashed_block_ids(self):
        return list(self._unhashed)

    def get_block_ids(self):
        return self._block_ids_tuple


class MockSchedulerOutput:
    def __init__(
        self,
        cached_req_ids=None,
        cached_new_block_ids=None,
        cached_num_computed=None,
        new_reqs=None,
        num_sched=None,
        scheduled_spec_decode_tokens=None,
    ):
        self.scheduled_cached_reqs = SimpleNamespace(
            req_ids=cached_req_ids or [],
            new_block_ids=cached_new_block_ids or [],
            num_computed_tokens=cached_num_computed or [],
        )
        self.scheduled_spec_decode_tokens = scheduled_spec_decode_tokens or {}
        self.scheduled_new_reqs = new_reqs or []
        self.num_scheduled_tokens = num_sched or {}


def make_kv_transfer_config(kv_role):
    return SimpleNamespace(
        kv_role=kv_role,
        is_kv_producer=kv_role in {"kv_producer", "kv_both"},
        is_kv_consumer=kv_role in {"kv_consumer", "kv_both"},
    )


def make_kv_caches():
    key_cache = MagicMock(name="key_cache")
    key_cache.shape = (10, 16, 8, 16)
    key_cache.data_ptr.return_value = 0x1000
    key_cache.element_size.return_value = 4
    value_cache = MagicMock(name="value_cache")
    value_cache.shape = (10, 16, 8, 16)
    value_cache.data_ptr.return_value = 0x2000
    value_cache.element_size.return_value = 4
    return {"encoder.layer.0": (key_cache, value_cache)}


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


def metadata_snapshot(metadata):
    return {
        "requests": {request_id: vars(request_meta).copy() for request_id, request_meta in metadata.requests.items()},
        "send_task": {
            key: value
            for key, value in vars(metadata.send_task).items()
            if key not in {"k_cache", "v_cache", "wait_event"}
        },
    }


class TestDualPathConfig(unittest.TestCase):
    def test_valid_prefill_with_kv_producer(self):
        config = DualPathConfig.from_extra_config({"role": "prefill"}, make_kv_transfer_config("kv_producer"))
        self.assertEqual(config, DualPathConfig(role="prefill"))

    def test_valid_decode_with_kv_consumer(self):
        config = DualPathConfig.from_extra_config(
            {"role": "decode", "dual_path_control_port": 7100},
            make_kv_transfer_config("kv_consumer"),
        )
        self.assertEqual(config, DualPathConfig(role="decode", dual_path_control_port=7100))

    def test_valid_prefill_with_kv_both(self):
        config = DualPathConfig.from_extra_config({"role": "prefill"}, make_kv_transfer_config("kv_both"))
        self.assertEqual(config.role, "prefill")

    def test_valid_decode_with_kv_both(self):
        config = DualPathConfig.from_extra_config(
            {"role": "decode", "dual_path_control_port": 7100},
            make_kv_transfer_config("kv_both"),
        )
        self.assertEqual(config, DualPathConfig(role="decode", dual_path_control_port=7100))

    def test_missing_role_rejected(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*requires 'role')(?=.*prefill)(?=.*decode)"):
            DualPathConfig.from_extra_config({}, make_kv_transfer_config("kv_both"))

    def test_unknown_role_value_rejected(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*other)(?=.*not supported)"):
            DualPathConfig.from_extra_config({"role": "other"}, make_kv_transfer_config("kv_both"))

    def test_role_pe_legacy_alias_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*pe)(?=.*legacy alias)(?=.*prefill)(?=.*decode)",
        ):
            DualPathConfig.from_extra_config({"role": "pe"}, make_kv_transfer_config("kv_both"))

    def test_role_de_legacy_alias_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*de)(?=.*legacy alias)(?=.*prefill)(?=.*decode)",
        ):
            DualPathConfig.from_extra_config({"role": "de"}, make_kv_transfer_config("kv_both"))

    def test_prefill_with_kv_consumer_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*role='prefill' requires kv_role)(?=.*kv_consumer)",
        ):
            DualPathConfig.from_extra_config({"role": "prefill"}, make_kv_transfer_config("kv_consumer"))

    def test_decode_with_kv_producer_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*role='decode' requires kv_role)(?=.*kv_producer)",
        ):
            DualPathConfig.from_extra_config({"role": "decode"}, make_kv_transfer_config("kv_producer"))

    def test_tls_prefill_decode_keys_are_accepted_without_storage(self):
        config = DualPathConfig.from_extra_config(
            {
                "role": "prefill",
                "tls_config": {"ssl_enable": False},
                "prefill": {"tp_size": 2, "dp_size": 1},
                "decode": {"tp_size": 2, "dp_size": 1},
            },
            make_kv_transfer_config("kv_producer"),
        )
        self.assertEqual(vars(config), {"role": "prefill", "dual_path_control_port": None})
        self.assertEqual(
            ALLOWED_EXTRA_CONFIG_KEYS,
            frozenset(
                {
                    "role",
                    "tls_config",
                    "prefill",
                    "decode",
                    "consumer_is_to_load",
                    "load_async",
                    "backend",
                    "lookup_rpc_port",
                    "mooncake_rpc_port",
                    "discard_partial_chunks",
                    "dual_path_control_port",
                }
            ),
        )

    def assert_removed_field_rejected(self, field_name):
        pattern = rf"(?=.*unsupported kv_connector_extra_config key\(s\))(?=.*{field_name})(?=.*fail fast)"
        with self.assertRaisesRegex(ValueError, pattern):
            DualPathConfig.from_extra_config(
                {"role": "prefill", field_name: True},
                make_kv_transfer_config("kv_producer"),
            )

    def test_removed_path_strategy_rejected(self):
        self.assert_removed_field_rejected("path_strategy")

    def test_removed_relay_rejected(self):
        self.assert_removed_field_rejected("relay")

    def test_removed_path_planner_rejected(self):
        self.assert_removed_field_rejected("path_planner")

    def test_removed_monitor_rejected(self):
        self.assert_removed_field_rejected("monitor")

    def test_removed_topology_rejected(self):
        self.assert_removed_field_rejected("topology")

    def test_removed_enable_value_function_shadow_rejected(self):
        self.assert_removed_field_rejected("enable_value_function_shadow")

    def test_removed_enable_link_monitor_shadow_rejected(self):
        self.assert_removed_field_rejected("enable_link_monitor_shadow")

    def test_arbitrary_unknown_key_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*unsupported kv_connector_extra_config key\(s\))(?=.*arbitrary_key)",
        ):
            DualPathConfig.from_extra_config(
                {"role": "prefill", "arbitrary_key": 1},
                make_kv_transfer_config("kv_producer"),
            )

    def test_unknown_keys_are_reported_in_sorted_order(self):
        with self.assertRaises(ValueError) as raised:
            DualPathConfig.from_extra_config(
                {"role": "prefill", "zeta": 1, "alpha": 2},
                make_kv_transfer_config("kv_producer"),
            )
        self.assertLess(str(raised.exception).index("alpha"), str(raised.exception).index("zeta"))

    def test_config_is_frozen(self):
        config = DualPathConfig(role="prefill")
        with self.assertRaises(FrozenInstanceError):
            config.role = "decode"

    def test_none_extra_config_behaves_like_empty_config(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*requires 'role')(?=.*prefill)(?=.*decode)"):
            DualPathConfig.from_extra_config(None, make_kv_transfer_config("kv_both"))

    def test_config_accepts_kvpool_passthrough_keys(self):
        passthrough_extra = {
            "consumer_is_to_load": True,
            "backend": "kvpool",
            "lookup_rpc_port": 18080,
            "mooncake_rpc_port": 18081,
            "discard_partial_chunks": False,
        }
        for key, value in passthrough_extra.items():
            extra = {"role": "decode", "dual_path_control_port": 7100, key: value}
            if key == "consumer_is_to_load":
                # Decode consumer load requires async Store I/O (Task-06).
                extra["load_async"] = True
            config = DualPathConfig.from_extra_config(extra, make_kv_transfer_config("kv_both"))
            self.assertEqual(config, DualPathConfig(role="decode", dual_path_control_port=7100))
        combined = DualPathConfig.from_extra_config(
            {"role": "decode", "dual_path_control_port": 7100, **passthrough_extra, "load_async": True},
            make_kv_transfer_config("kv_both"),
        )
        self.assertEqual(combined, DualPathConfig(role="decode", dual_path_control_port=7100))
        self.assertEqual(vars(combined), {"role": "decode", "dual_path_control_port": 7100})

    def test_config_rejects_invalid_consumer_is_to_load_type(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*consumer_is_to_load)(?=.*boolean)"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "consumer_is_to_load": "true"},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_rejects_empty_backend(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*backend)(?=.*non-empty)"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "backend": ""},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_rejects_negative_lookup_rpc_port(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*lookup_rpc_port)(?=.*non-negative)"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "lookup_rpc_port": -1},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_rejects_non_int_mooncake_rpc_port(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*mooncake_rpc_port)(?=.*non-negative)"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "mooncake_rpc_port": "18081"},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_rejects_non_bool_discard_partial_chunks(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*discard_partial_chunks)(?=.*boolean)"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "discard_partial_chunks": 1},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_still_rejects_consumer_is_to_put(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*unsupported kv_connector_extra_config key\(s\))(?=.*consumer_is_to_put)",
        ):
            DualPathConfig.from_extra_config(
                {"role": "decode", "dual_path_control_port": 7100, "consumer_is_to_put": False},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_accepts_load_async(self):
        config = DualPathConfig.from_extra_config(
            {"role": "decode", "dual_path_control_port": 7100, "load_async": True},
            make_kv_transfer_config("kv_both"),
        )
        self.assertEqual(config, DualPathConfig(role="decode", dual_path_control_port=7100))

    def test_config_accepts_load_async_for_each_role(self):
        for role, ktc in (
            ("prefill", make_kv_transfer_config("kv_producer")),
            ("decode", make_kv_transfer_config("kv_both")),
        ):
            extra = {"role": role, "load_async": True}
            if role == "decode":
                extra["dual_path_control_port"] = 7100
            config = DualPathConfig.from_extra_config(extra, ktc)
            self.assertEqual(config.role, role)

    def test_config_rejects_non_bool_load_async(self):
        with self.assertRaisesRegex(ValueError, r"(?=.*load_async)(?=.*boolean)"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "dual_path_control_port": 7100, "load_async": "true"},
                make_kv_transfer_config("kv_both"),
            )

    def test_config_rejects_decode_consumer_load_without_load_async(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*consumer_is_to_load)(?=.*load_async)(?=.*synchronous)",
        ):
            DualPathConfig.from_extra_config(
                {
                    "role": "decode",
                    "dual_path_control_port": 7100,
                    "consumer_is_to_load": True,
                },
                make_kv_transfer_config("kv_both"),
            )

    def test_config_rejects_decode_consumer_load_with_load_async_false(self):
        with self.assertRaisesRegex(
            ValueError,
            r"(?=.*consumer_is_to_load)(?=.*load_async)(?=.*synchronous)",
        ):
            DualPathConfig.from_extra_config(
                {
                    "role": "decode",
                    "dual_path_control_port": 7100,
                    "consumer_is_to_load": True,
                    "load_async": False,
                },
                make_kv_transfer_config("kv_both"),
            )

    def test_config_accepts_decode_consumer_load_with_load_async_true(self):
        config = DualPathConfig.from_extra_config(
            {
                "role": "decode",
                "dual_path_control_port": 7100,
                "consumer_is_to_load": True,
                "load_async": True,
            },
            make_kv_transfer_config("kv_both"),
        )
        self.assertEqual(config, DualPathConfig(role="decode", dual_path_control_port=7100))

    def test_config_accepts_decode_without_consumer_load_any_load_async(self):
        for load_async in (True, False):
            config = DualPathConfig.from_extra_config(
                {
                    "role": "decode",
                    "dual_path_control_port": 7100,
                    "consumer_is_to_load": False,
                    "load_async": load_async,
                },
                make_kv_transfer_config("kv_both"),
            )
            self.assertEqual(config, DualPathConfig(role="decode", dual_path_control_port=7100))

    def test_config_accepts_prefill_consumer_load_without_load_async(self):
        config = DualPathConfig.from_extra_config(
            {"role": "prefill", "consumer_is_to_load": True},
            make_kv_transfer_config("kv_producer"),
        )
        self.assertEqual(config, DualPathConfig(role="prefill"))


class TestDualPathConstructionParity(unittest.TestCase):
    def test_kv_connector_base_init_runs_exactly_once(self):
        config = MockVllmConfig("prefill", "kv_producer")
        kv_cache_config = MockKVCacheConfig()
        original_init = KVConnectorBase_V1.__init__
        calls = []

        def init_spy(instance, *args, **kwargs):
            calls.append((args, kwargs))
            return original_init(instance, *args, **kwargs)

        with patch.object(KVConnectorBase_V1, "__init__", new=init_spy):
            DualPathConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        self.assertEqual(len(calls), 1)

    def test_scheduler_construction_calls_real_parent_once_and_never_worker(self):
        config = MockVllmConfig("prefill", "kv_producer")
        kv_cache_config = MockKVCacheConfig()
        scheduler_init = MooncakeLayerwiseConnectorScheduler.__init__
        worker_init = MooncakeLayerwiseConnectorWorker.__init__
        scheduler_calls = []
        worker_calls = []

        def scheduler_spy(instance, *args, **kwargs):
            scheduler_calls.append((args, kwargs))
            return scheduler_init(instance, *args, **kwargs)

        def worker_spy(instance, *args, **kwargs):
            worker_calls.append((args, kwargs))
            return worker_init(instance, *args, **kwargs)

        with (
            patch.object(MooncakeLayerwiseConnectorScheduler, "__init__", new=scheduler_spy),
            patch.object(MooncakeLayerwiseConnectorWorker, "__init__", new=worker_spy),
        ):
            connector = DualPathConnector(config, KVConnectorRole.SCHEDULER, kv_cache_config)
        self.assertEqual(len(scheduler_calls), 1)
        self.assertEqual(worker_calls, [])
        self.assertIsInstance(connector.connector_scheduler, DualPathConnectorScheduler)
        self.assertIsNone(connector.connector_worker)

    def test_worker_construction_calls_real_parent_once_and_never_scheduler(self):
        config = MockVllmConfig("prefill", "kv_producer")
        config.parallel_config.tensor_parallel_size = 1
        kv_cache_config = MockKVCacheConfig()
        scheduler_init = MooncakeLayerwiseConnectorScheduler.__init__
        worker_init = MooncakeLayerwiseConnectorWorker.__init__
        scheduler_calls = []
        worker_calls = []

        def scheduler_spy(instance, *args, **kwargs):
            scheduler_calls.append((args, kwargs))
            return scheduler_init(instance, *args, **kwargs)

        def worker_spy(instance, *args, **kwargs):
            worker_calls.append((args, kwargs))
            return worker_init(instance, *args, **kwargs)

        with (
            worker_environment(),
            patch.object(MooncakeLayerwiseConnectorScheduler, "__init__", new=scheduler_spy),
            patch.object(MooncakeLayerwiseConnectorWorker, "__init__", new=worker_spy),
        ):
            connector = DualPathConnector(config, KVConnectorRole.WORKER, kv_cache_config)
        self.assertEqual(scheduler_calls, [])
        self.assertEqual(len(worker_calls), 1)
        self.assertIsNone(connector.connector_scheduler)
        self.assertIsInstance(connector.connector_worker, DualPathConnectorWorker)

    def test_scheduler_facade_state_matches_parent_reference(self):
        parent_config = MockVllmConfig("prefill", "kv_producer")
        dual_config = MockVllmConfig("prefill", "kv_producer")
        parent = MooncakeLayerwiseConnector(parent_config, KVConnectorRole.SCHEDULER, MockKVCacheConfig())
        dual = DualPathConnector(dual_config, KVConnectorRole.SCHEDULER, MockKVCacheConfig())
        self.assertEqual(parent._is_kv_producer, dual._is_kv_producer)
        self.assertEqual(parent.engine_id, dual.engine_id)
        self.assertIs(type(parent._connector_metadata), MooncakeLayerwiseConnectorMetadata)
        self.assertIs(type(dual._connector_metadata), MooncakeLayerwiseConnectorMetadata)
        self.assertEqual(parent._connector_metadata.requests, dual._connector_metadata.requests)
        self.assertIsInstance(parent._connector_metadata.send_task, SendTask)
        self.assertIsInstance(dual._connector_metadata.send_task, SendTask)
        self.assertIsNot(parent._connector_metadata.send_task, dual._connector_metadata.send_task)
        self.assertEqual(vars(parent._connector_metadata.send_task), vars(dual._connector_metadata.send_task))
        self.assertEqual(set(vars(parent)), set(vars(dual)))
        self.assertNotIn("dual_path_cfg", vars(dual))

    def test_worker_facade_state_matches_parent_reference(self):
        parent_config = MockVllmConfig("prefill", "kv_producer")
        dual_config = MockVllmConfig("prefill", "kv_producer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        with worker_environment():
            parent = MooncakeLayerwiseConnector(parent_config, KVConnectorRole.WORKER, MockKVCacheConfig())
            dual = DualPathConnector(dual_config, KVConnectorRole.WORKER, MockKVCacheConfig())
        self.assertEqual(parent._is_kv_producer, dual._is_kv_producer)
        self.assertEqual(parent.engine_id, dual.engine_id)
        self.assertEqual(metadata_snapshot(parent._connector_metadata), metadata_snapshot(dual._connector_metadata))
        self.assertEqual(set(vars(parent)), set(vars(dual)))
        self.assertNotIn("dual_path_cfg", vars(dual))

    def test_subclasses_store_same_frozen_config_object(self):
        dual_path_config = DualPathConfig(role="prefill")
        parent_scheduler = MooncakeLayerwiseConnectorScheduler(
            MockVllmConfig("prefill", "kv_producer"),
            MockKVCacheConfig(),
            "test_engine",
        )
        scheduler = DualPathConnectorScheduler(
            MockVllmConfig("prefill", "kv_producer"),
            MockKVCacheConfig(),
            "test_engine",
            dual_path_config,
        )
        parent_worker_config = MockVllmConfig("prefill", "kv_producer")
        worker_config = MockVllmConfig("prefill", "kv_producer")
        parent_worker_config.parallel_config.tensor_parallel_size = 1
        worker_config.parallel_config.tensor_parallel_size = 1
        with worker_environment():
            parent_worker = MooncakeLayerwiseConnectorWorker(
                parent_worker_config,
                MockKVCacheConfig(),
                "test_engine",
            )
            worker = DualPathConnectorWorker(
                worker_config,
                MockKVCacheConfig(),
                "test_engine",
                dual_path_config,
            )
        self.assertIs(scheduler.dual_path_cfg, dual_path_config)
        self.assertIs(worker.dual_path_cfg, dual_path_config)
        dual_path_scheduler_fields = {
            "_kvpool_adapter",
            "_lookup_results",
            "_decode_kv_snapshots",
            "_decode_decision_states",
            "_decision_timeout_seconds",
            "_accepting_task01",
            "_accepting_pe_decisions",
            "_path_decision_coordinator",
            "_path_decider",
            "_pe_request_keys",
            "_pe_path_results",
            "_pe_forward_plans",
            "_pe_forward_send_infos",
            "_pe_delivery_futures",
            "_pe_invalid_request_ids",
        }
        self.assertEqual(
            set(vars(scheduler)), set(vars(parent_scheduler)) | {"dual_path_cfg"} | dual_path_scheduler_fields
        )
        self.assertEqual(
            set(vars(worker)),
            set(vars(parent_worker))
            | {
                "dual_path_cfg",
                "_kvpool_worker_adapter",
                "_control_failed_recving",
                "_forward_receive_bindings",
                "_pending_forward_done",
                "_pending_forward_failed",
                "_consumed_forward_terminals",
            },
        )

    def test_scheduler_has_no_req_path(self):
        scheduler = DualPathConnectorScheduler(
            MockVllmConfig("prefill", "kv_producer"),
            MockKVCacheConfig(),
            "test_engine",
            DualPathConfig(role="prefill"),
        )
        self.assertFalse(hasattr(scheduler, "_req_path"))

    def test_scheduler_shutdown_closes_inherited_resources_idempotently(self):
        scheduler = DualPathConnectorScheduler(
            MockVllmConfig("prefill", "kv_producer"),
            MockKVCacheConfig(),
            "test_engine",
            DualPathConfig(role="prefill"),
        )
        executor = scheduler.executor
        metaserver_client = scheduler.metaserver_client

        try:
            scheduler.shutdown()
            scheduler.shutdown()

            self.assertTrue(executor._shutdown)
            self.assertTrue(metaserver_client.is_closed)
        finally:
            executor.shutdown(wait=False)
            metaserver_client.close()

    def test_worker_construction_adds_no_threads_beyond_parent(self):
        parent_config = MockVllmConfig("prefill", "kv_producer")
        dual_config = MockVllmConfig("prefill", "kv_producer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        with worker_environment():
            before_parent = set(threading.enumerate())
            MooncakeLayerwiseConnectorWorker(parent_config, MockKVCacheConfig(), "test_engine")
            parent_delta = set(threading.enumerate()) - before_parent
            before_dual = set(threading.enumerate())
            DualPathConnectorWorker(
                dual_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="prefill"),
            )
            dual_delta = set(threading.enumerate()) - before_dual
        self.assertEqual(
            {(thread.name, thread.daemon) for thread in parent_delta},
            {(thread.name, thread.daemon) for thread in dual_delta},
        )

    def test_transfer_engine_and_buffer_registration_match_parent(self):
        parent_config = MockVllmConfig("prefill", "kv_producer")
        dual_config = MockVllmConfig("prefill", "kv_producer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        with worker_environment() as runtime:
            parent = MooncakeLayerwiseConnectorWorker(parent_config, MockKVCacheConfig(), "test_engine")
            parent.register_kv_caches(make_kv_caches())
            parent_engine_calls = runtime.get_transfer_engine.call_count
            parent_buffer_calls = runtime.register_buffer.call_count
            runtime.get_transfer_engine.reset_mock()
            runtime.register_buffer.reset_mock()
            dual = DualPathConnectorWorker(
                dual_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="prefill"),
            )
            dual.register_kv_caches(make_kv_caches())
            dual_engine_calls = runtime.get_transfer_engine.call_count
            dual_buffer_calls = runtime.register_buffer.call_count
        self.assertEqual(parent_engine_calls, 1)
        self.assertEqual(dual_engine_calls, parent_engine_calls)
        self.assertEqual(parent_buffer_calls, 1)
        self.assertEqual(dual_buffer_calls, parent_buffer_calls)

    def test_unsupported_connector_role_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Unsupported KVConnectorRole"):
            DualPathConnector(MockVllmConfig(), "unsupported", MockKVCacheConfig())


class TestDualPathBehaviorParity(unittest.TestCase):
    def setUp(self):
        self.schedulers = []

    def tearDown(self):
        for scheduler in self.schedulers:
            scheduler.executor.shutdown(wait=False)
            scheduler.metaserver_client.close()

    def make_scheduler_pair(self, dual_role, kv_role):
        parent = MooncakeLayerwiseConnectorScheduler(
            MockVllmConfig(dual_role, kv_role),
            MockKVCacheConfig(),
            "test_engine",
        )
        dual = DualPathConnectorScheduler(
            MockVllmConfig(dual_role, kv_role),
            MockKVCacheConfig(),
            "test_engine",
            DualPathConfig(
                role=dual_role,
                dual_path_control_port=7100 if dual_role == "decode" else None,
            ),
        )
        self.schedulers.extend([parent, dual])
        return parent, dual

    def replace_executors_with_mocks(self, parent, dual):
        parent.executor.shutdown(wait=False)
        dual.executor.shutdown(wait=False)
        parent.executor = MagicMock(name="parent_executor")
        dual.executor = MagicMock(name="dual_executor")
        parent_future = MagicMock(name="parent_future")
        dual_future = MagicMock(name="dual_future")
        parent_future.exception.return_value = None
        dual_future.exception.return_value = None
        parent.executor.submit.return_value = parent_future
        dual.executor.submit.return_value = dual_future
        return parent_future, dual_future

    def test_no_remote_transfer_matches_parent(self):
        parent, dual = self.make_scheduler_pair("prefill", "kv_producer")
        self.replace_executors_with_mocks(parent, dual)
        parent_request = MockRequest("req-none")
        dual_request = MockRequest("req-none")

        parent_result = parent.get_num_new_matched_tokens(parent_request, 0)
        dual_result = dual.get_num_new_matched_tokens(dual_request, 0)

        self.assertEqual(parent_result, (0, False))
        self.assertEqual(dual_result, parent_result)
        self.assertEqual(parent._reqs_need_recv, dual._reqs_need_recv)
        self.assertEqual(parent._reqs_need_send_layerwise, dual._reqs_need_send_layerwise)
        self.assertEqual(parent.executor.submit.call_args_list, dual.executor.submit.call_args_list)

    def test_no_remote_transfer_allocation_matches_parent_block_access(self):
        parent, dual = self.make_scheduler_pair("prefill", "kv_producer")
        parent_request = MockRequest("req-none")
        dual_request = MockRequest("req-none")
        parent_blocks = MagicMock(name="parent_blocks")
        dual_blocks = MagicMock(name="dual_blocks")
        parent_method = MooncakeLayerwiseConnectorScheduler.update_state_after_alloc

        parent.update_state_after_alloc(parent_request, parent_blocks, 0)
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "update_state_after_alloc",
            autospec=True,
            side_effect=parent_method,
        ) as parent_spy:
            dual.update_state_after_alloc(dual_request, dual_blocks, 0)

        parent_spy.assert_called_once_with(dual, dual_request, dual_blocks, 0)
        self.assertEqual(parent_blocks.get_block_ids.call_count, 0)
        self.assertEqual(dual_blocks.get_block_ids.call_count, parent_blocks.get_block_ids.call_count)

    def test_decode_remote_prefill_routes_to_task01_admission(self):
        parent, dual = self.make_scheduler_pair("decode", "kv_consumer")
        parent_future, dual_future = self.replace_executors_with_mocks(parent, dual)
        params = {"do_remote_prefill": True, "metaserver": "http://meta"}
        parent_request = MockRequest("req-load", kv_transfer_params=dict(params))
        dual_request = MockRequest("req-load", kv_transfer_params=dict(params))
        parent_blocks = MockBlocks(unhashed=[], block_ids_tuple=([4, 5, 6],))
        dual_blocks = MockBlocks(unhashed=[], block_ids_tuple=([4, 5, 6],))
        dual._kvpool_adapter.lookup.return_value = None

        parent_match = parent.get_num_new_matched_tokens(parent_request, 0)
        dual_match = dual.get_num_new_matched_tokens(dual_request, 0)
        parent.update_state_after_alloc(parent_request, parent_blocks, num_external_tokens=4)
        dual.update_state_after_alloc(dual_request, dual_blocks, num_external_tokens=4)

        # The parent keeps the legacy full-prompt remote-prefill flow.
        self.assertEqual(parent_match, (4, True))
        self.assertIn("req-load", parent._reqs_need_recv)
        self.assertFalse(parent_request.kv_transfer_params["do_remote_prefill"])
        self.assertEqual(parent_future.add_done_callback.call_count, 1)

        # Task-01: ordinary Attention preserves the Layerwise target T = P = 4
        # and binds a DecodeKVSnapshot without touching parent machinery.
        self.assertEqual(dual_match, (4, True))
        self.assertEqual(dual._reqs_need_recv, {})
        self.assertFalse(dual_request.kv_transfer_params["do_remote_prefill"])
        self.assertEqual(dual.executor.submit.call_count, 1)
        self.assertEqual(dual_future.add_done_callback.call_count, 1)
        dual._path_decision_coordinator.register_pending.assert_called_once()
        self.assertEqual(dual._lookup_results, {})
        snapshot = dual._decode_kv_snapshots["req-load"]
        self.assertEqual(snapshot.target_tokens, 3)
        self.assertEqual(snapshot.transfer_tokens, 4)
        self.assertEqual(snapshot.local_tokens, 0)
        self.assertEqual(snapshot.external_tokens, 4)
        self.assertIsNone(snapshot.store_load_spec)
        self.assertEqual(snapshot.final_block_ids, ((4, 5, 6),))
        self.assertIn("req-load", dual._decode_decision_states)
        self.assertNotIn("req-load", dual.build_connector_meta(MockSchedulerOutput()).requests)

        # Parent metadata still carries the legacy recv entry; Task-01 metadata does not.
        parent_meta = parent.build_connector_meta(MockSchedulerOutput())
        self.assertEqual(parent_meta.requests["req-load"].local_block_ids, ([4, 5, 6],))
        self.assertEqual(parent._reqs_need_recv, {})
        self.assertEqual(dual._reqs_need_recv, {})

    def test_prefill_remote_decode_matches_parent(self):
        parent, dual = self.make_scheduler_pair("prefill", "kv_producer")
        params = {
            "do_remote_decode": True,
            "remote_block_ids": [[4, 5]],
            "remote_block_size": [[16]],
            "remote_cached_tokens": 0,
            "remote_engine_id": "decode-engine",
            "remote_host": "127.0.0.2",
            "remote_port": 6000,
        }
        parent_request = MockRequest(
            "req-save",
            prompt_token_ids=list(range(10)),
            kv_transfer_params=dict(params),
        )
        dual_request = MockRequest(
            "req-save",
            prompt_token_ids=list(range(10)),
            kv_transfer_params=dict(params),
        )
        parent.update_state_after_alloc(
            parent_request,
            MockBlocks(unhashed=[], block_ids_tuple=([[7, 8, 9]],)),
            num_external_tokens=0,
        )
        dual.update_state_after_alloc(
            dual_request,
            MockBlocks(unhashed=[], block_ids_tuple=([[7, 8, 9]],)),
            num_external_tokens=0,
        )
        parent_info = parent._reqs_need_send_layerwise["req-save"]
        dual_info = dual._reqs_need_send_layerwise["req-save"]
        self.assertEqual(parent_info.local_block_ids, dual_info.local_block_ids)
        self.assertEqual(parent_info.local_block_ids, [[[7, 8, 9]]])
        self.assertEqual(parent_info.local_transferred_tokens, dual_info.local_transferred_tokens)
        self.assertEqual(parent_info.local_computed_tokens, dual_info.local_computed_tokens)

        parent_meta = parent.build_connector_meta(
            MockSchedulerOutput(
                cached_req_ids=["req-save"],
                cached_new_block_ids=[None],
                cached_num_computed=[0],
                num_sched={"req-save": 10},
            )
        )
        dual_meta = dual.build_connector_meta(
            MockSchedulerOutput(
                cached_req_ids=["req-save"],
                cached_new_block_ids=[None],
                cached_num_computed=[0],
                num_sched={"req-save": 10},
            )
        )
        self.assertEqual(metadata_snapshot(parent_meta), metadata_snapshot(dual_meta))
        self.assertTrue(parent_meta.requests["req-save"].chunk_finish)
        self.assertEqual(parent_meta.requests["req-save"].prompt_len, 10)
        self.assertEqual(parent_meta.requests["req-save"].local_block_ids, [[[7, 8, 9]]])
        self.assertNotIn("req-save", parent._reqs_need_send_layerwise)
        self.assertNotIn("req-save", dual._reqs_need_send_layerwise)

    def test_attention_mamba_hybrid_matches_parent(self):
        parent, dual = self.make_scheduler_pair("decode", "kv_both")
        parent.need_truncate = True
        dual.need_truncate = True
        parent_decode = MockRequest(
            "req-hybrid-save",
            prompt_token_ids=list(range(4)),
            kv_transfer_params={"do_remote_decode": True},
        )
        dual_decode = MockRequest(
            "req-hybrid-save",
            prompt_token_ids=list(range(4)),
            kv_transfer_params={"do_remote_decode": True},
        )
        self.assertEqual(
            parent.get_num_new_matched_tokens(parent_decode, 0),
            dual.get_num_new_matched_tokens(dual_decode, 0),
        )
        self.assertEqual(parent_decode.prompt_token_ids, [0, 1, 2])
        self.assertEqual(vars(parent_decode), vars(dual_decode))

        self.replace_executors_with_mocks(parent, dual)
        parent_load = MockRequest(
            "req-hybrid-load",
            prompt_token_ids=list(range(17)),
            kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://meta"},
        )
        dual_load = MockRequest(
            "req-hybrid-load",
            prompt_token_ids=list(range(17)),
            kv_transfer_params={"do_remote_prefill": True, "metaserver": "http://meta"},
        )
        dual._kvpool_adapter.lookup.return_value = None
        parent_match = parent.get_num_new_matched_tokens(parent_load, 0)
        dual_match = dual.get_num_new_matched_tokens(dual_load, 0)
        parent.update_state_after_alloc(
            parent_load,
            MockBlocks(unhashed=[], block_ids_tuple=([4, 5],)),
            num_external_tokens=16,
        )
        dual.update_state_after_alloc(
            dual_load,
            MockBlocks(unhashed=[], block_ids_tuple=([4, 5],)),
            num_external_tokens=16,
        )
        # R = P - 1 = 16 coincides with the parent's hybrid-truncated count, but
        # the dual side reaches it through the Task-01 admission path.
        self.assertEqual(parent_match, (16, True))
        self.assertEqual(dual_match, parent_match)
        self.assertEqual(parent.executor.submit.call_count, 1)
        self.assertEqual(dual.executor.submit.call_count, 1)
        self.assertEqual(parent.executor.submit.call_args.kwargs["message"]["remote_block_ids"], ([4],))
        self.assertEqual(dual.executor.submit.call_args.kwargs["message"]["remote_block_ids"], ([4],))
        self.assertEqual(dual._reqs_need_recv, {})
        self.assertFalse(dual_load.kv_transfer_params["do_remote_prefill"])
        snapshot = dual._decode_kv_snapshots["req-hybrid-load"]
        self.assertEqual(snapshot.target_tokens, 16)
        self.assertEqual(snapshot.transfer_tokens, 16)
        self.assertEqual(snapshot.local_tokens, 0)
        self.assertEqual(snapshot.external_tokens, 16)
        self.assertEqual(snapshot.final_block_ids, ((4, 5),))

    def test_allocation_task01_snapshot_replaces_parent_bookkeeping(self):
        parent, dual = self.make_scheduler_pair("decode", "kv_consumer")
        self.replace_executors_with_mocks(parent, dual)
        params = {"do_remote_prefill": True, "do_virtual": True, "metaserver": "http://meta"}
        parent_request = MockRequest(
            "req-alloc",
            prompt_token_ids=list(range(24)),
            kv_transfer_params=dict(params),
        )
        dual_request = MockRequest(
            "req-alloc",
            prompt_token_ids=list(range(24)),
            kv_transfer_params=dict(params),
        )
        dual._kvpool_adapter.lookup.return_value = None
        parent.update_state_after_alloc(
            parent_request,
            MockBlocks(unhashed=[4, 5, 6], block_ids_tuple=([[4, 5, 6]],)),
            num_external_tokens=8,
        )
        dual_match = dual.get_num_new_matched_tokens(dual_request, 0)
        dual.update_state_after_alloc(
            dual_request,
            MockBlocks(unhashed=[4, 5, 6], block_ids_tuple=([[4, 5, 6]],)),
            num_external_tokens=24,
        )
        # Parent: legacy bookkeeping tracks the request for receive and clears
        # the remote-prefill flag; DualPath Task-01: no parent bookkeeping at all.
        parent_state = parent._reqs_need_recv["req-alloc"]
        self.assertIs(parent_state[0], parent_request)
        self.assertEqual(parent_state[2], ([[4, 5, 6]],))
        self.assertFalse(parent_request.kv_transfer_params["do_remote_prefill"])

        self.assertEqual(dual_match, (24, True))
        self.assertEqual(dual._reqs_need_recv, {})
        self.assertFalse(dual_request.kv_transfer_params["do_remote_prefill"])
        self.assertEqual(dual.executor.submit.call_count, 0)
        self.assertIn("req-alloc", dual._decode_decision_states)
        dual._path_decision_coordinator.register_pending.assert_called_once()
        snapshot = dual._decode_kv_snapshots["req-alloc"]
        self.assertEqual(snapshot.target_tokens, 23)
        self.assertEqual(snapshot.transfer_tokens, 24)
        self.assertEqual(snapshot.external_tokens, 24)
        self.assertEqual(snapshot.local_tokens, 0)
        self.assertEqual(snapshot.store_tokens, 0)
        self.assertEqual(snapshot.final_block_ids, (([4, 5, 6],),))

    def test_worker_load_matches_parent(self):
        parent_config = MockVllmConfig("decode", "kv_consumer")
        dual_config = MockVllmConfig("decode", "kv_consumer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        params = {
            "remote_block_ids": [[1, 2]],
            "remote_block_size": [[16]],
            "remote_engine_id": "prefill-engine",
            "remote_host": "127.0.0.2",
            "remote_port": 6000,
            "remote_te_rpc_port": 9090,
        }
        normal_request_id = "reqA-load-00000001"
        virtual_request_id = "reqB-virt-00000001"
        parent_meta = MooncakeLayerwiseConnectorMetadata()
        dual_meta = MooncakeLayerwiseConnectorMetadata()
        parent_meta.add_new_req(normal_request_id, [[7, 8]], dict(params))
        dual_meta.add_new_req(normal_request_id, [[7, 8]], dict(params))
        parent_meta.add_new_req(virtual_request_id, [[9]], {"do_virtual": True})
        dual_meta.add_new_req(virtual_request_id, [[9]], {"do_virtual": True})

        with worker_environment():
            parent = MooncakeLayerwiseConnectorWorker(parent_config, MockKVCacheConfig(), "test_engine")
            dual = DualPathConnectorWorker(
                dual_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="decode"),
            )
            parent.register_kv_caches(make_kv_caches())
            dual.register_kv_caches(make_kv_caches())
            self.assertIsNone(parent.kv_send_layer_thread)
            self.assertIsNone(dual.kv_send_layer_thread)
            self.assertIsNotNone(parent.kv_recv_layer_thread)
            self.assertIsNotNone(dual.kv_recv_layer_thread)
            self.assertEqual(parent.kv_recv_layer_thread.start.call_count, dual.kv_recv_layer_thread.start.call_count)

            parent.start_load_kv(parent_meta)
            dual.start_load_kv(dual_meta)
            external_request_id = get_external_request_id(normal_request_id)
            expected_request_map = {external_request_id: normal_request_id}
            self.assertEqual(parent.current_layer, 0)
            self.assertEqual(dual.current_layer, parent.current_layer)
            self.assertEqual(parent.request_map, expected_request_map)
            self.assertEqual(dual.request_map, parent.request_map)
            self.assertEqual(set(parent._recving_metadata), {normal_request_id})
            self.assertEqual(set(dual._recving_metadata), set(parent._recving_metadata))
            self.assertEqual(parent.virtual_request, {virtual_request_id})
            self.assertEqual(dual.virtual_request, parent.virtual_request)
            self.assertIsNone(parent.wait_for_layer_load("encoder.layer.0"))
            self.assertIsNone(dual.wait_for_layer_load("encoder.layer.0"))

    def test_worker_save_matches_parent(self):
        parent_config = MockVllmConfig("prefill", "kv_producer")
        dual_config = MockVllmConfig("prefill", "kv_producer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        params = {
            "remote_block_ids": [[4, 5]],
            "remote_block_size": [[16]],
            "remote_engine_id": "decode-engine",
            "remote_host": "127.0.0.2",
            "remote_port": 6000,
            "remote_te_rpc_port": 9090,
        }
        parent_meta = MooncakeLayerwiseConnectorMetadata()
        dual_meta = MooncakeLayerwiseConnectorMetadata()
        parent_meta.add_new_req("req-save", [[7, 8]], dict(params), chunk_finish=True)
        dual_meta.add_new_req("req-save", [[7, 8]], dict(params), chunk_finish=True)

        with worker_environment() as runtime:
            parent = MooncakeLayerwiseConnectorWorker(parent_config, MockKVCacheConfig(), "test_engine")
            dual = DualPathConnectorWorker(
                dual_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="prefill"),
            )
            parent.register_kv_caches(make_kv_caches())
            dual.register_kv_caches(make_kv_caches())
            self.assertIsNotNone(parent.kv_send_layer_thread)
            self.assertIsNotNone(dual.kv_send_layer_thread)
            self.assertIsNone(parent.kv_recv_layer_thread)
            self.assertIsNone(dual.kv_recv_layer_thread)
            parent.kv_send_layer_thread.reset_mock()
            dual.kv_send_layer_thread.reset_mock()
            parent.current_layer = 0
            dual.current_layer = 0
            parent_key = MagicMock(name="parent_key")
            parent_value = MagicMock(name="parent_value")
            dual_key = MagicMock(name="dual_key")
            dual_value = MagicMock(name="dual_value")
            parent_attention = MagicMock(name="parent_attention")
            dual_attention = MagicMock(name="dual_attention")

            with (
                patch.object(parent, "update_decoder_info", side_effect=lambda req_id, req_meta: req_meta),
                patch.object(dual, "update_decoder_info", side_effect=lambda req_id, req_meta: req_meta),
            ):
                parent.save_kv_layer(
                    "encoder.layer.0",
                    [parent_key, parent_value],
                    parent_attention,
                    parent_meta,
                )
                dual.save_kv_layer(
                    "encoder.layer.0",
                    [dual_key, dual_value],
                    dual_attention,
                    dual_meta,
                )

            self.assertEqual(parent.kv_send_layer_thread.send_queue.put.call_count, 1)
            self.assertEqual(dual.kv_send_layer_thread.send_queue.put.call_count, 1)
            parent_task = parent.kv_send_layer_thread.send_queue.put.call_args.args[0]
            dual_task = dual.kv_send_layer_thread.send_queue.put.call_args.args[0]
            self.assertEqual(parent_task.layer_idx, 0)
            self.assertEqual(dual_task.layer_idx, parent_task.layer_idx)
            self.assertEqual(parent_task.layer_name, "encoder.layer.0")
            self.assertEqual(dual_task.layer_name, parent_task.layer_name)
            self.assertEqual(set(parent_task.send_request), {"req-save"})
            self.assertEqual(set(dual_task.send_request), set(parent_task.send_request))
            self.assertTrue(parent_task.send_request["req-save"].chunk_finish)
            self.assertTrue(dual_task.send_request["req-save"].chunk_finish)
            self.assertIs(parent_task.wait_event, parent_attention.reshape_cache_event)
            self.assertIs(dual_task.wait_event, dual_attention.reshape_cache_event)
            self.assertEqual(parent.current_layer, 1)
            self.assertEqual(dual.current_layer, parent.current_layer)
            self.assertEqual(runtime.send_factory.call_count, 2)
            parent_callback = runtime.send_factory.call_args_list[0].kwargs["callback_func"]
            dual_callback = runtime.send_factory.call_args_list[1].kwargs["callback_func"]
            self.assertIs(parent_callback.__func__, dual_callback.__func__)

    def test_completion_matches_parent(self):
        parent_consumer_config = MockVllmConfig("decode", "kv_consumer")
        dual_consumer_config = MockVllmConfig("decode", "kv_consumer")
        parent_producer_config = MockVllmConfig("prefill", "kv_producer")
        dual_producer_config = MockVllmConfig("prefill", "kv_producer")
        for config in (
            parent_consumer_config,
            dual_consumer_config,
            parent_producer_config,
            dual_producer_config,
        ):
            config.parallel_config.tensor_parallel_size = 1
        normal_request_id = "reqC-load-00000001"
        virtual_request_id = "reqD-virt-00000001"
        normal_params = {
            "remote_block_ids": [[1, 2]],
            "remote_block_size": [[16]],
            "remote_engine_id": "prefill-engine",
            "remote_host": "127.0.0.2",
            "remote_port": 6000,
            "remote_te_rpc_port": 9090,
        }
        parent_meta = MooncakeLayerwiseConnectorMetadata()
        dual_meta = MooncakeLayerwiseConnectorMetadata()
        parent_meta.add_new_req(normal_request_id, [[7, 8]], dict(normal_params))
        dual_meta.add_new_req(normal_request_id, [[7, 8]], dict(normal_params))
        parent_meta.add_new_req(virtual_request_id, [[9]], {"do_virtual": True})
        dual_meta.add_new_req(virtual_request_id, [[9]], {"do_virtual": True})
        with worker_environment():
            parent_consumer = MooncakeLayerwiseConnectorWorker(
                parent_consumer_config,
                MockKVCacheConfig(),
                "test_engine",
            )
            dual_consumer = DualPathConnectorWorker(
                dual_consumer_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="decode"),
            )
            parent_producer = MooncakeLayerwiseConnectorWorker(
                parent_producer_config,
                MockKVCacheConfig(),
                "test_engine",
            )
            dual_producer = DualPathConnectorWorker(
                dual_producer_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="prefill"),
            )
            for worker in (parent_consumer, dual_consumer, parent_producer, dual_producer):
                worker.register_kv_caches(make_kv_caches())
            parent_consumer.start_load_kv(parent_meta)
            dual_consumer.start_load_kv(dual_meta)
            external_request_id = get_external_request_id(normal_request_id)
            parent_consumer.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {external_request_id}
            dual_consumer.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {external_request_id}
            parent_consumer.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
            dual_consumer.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()

            parent_consumer_result = parent_consumer.get_finished()
            dual_consumer_result = dual_consumer.get_finished(set())
            parent_producer_result = parent_producer.get_finished()
            dual_producer_result = dual_producer.get_finished(set())

            expected_finished = {normal_request_id, virtual_request_id}
            self.assertEqual(parent_consumer_result, (set(), expected_finished))
            self.assertEqual(dual_consumer_result, parent_consumer_result)
            self.assertEqual(parent_producer_result, (set(), set()))
            self.assertEqual(dual_producer_result, parent_producer_result)
            self.assertEqual(parent_consumer.virtual_request, set())
            self.assertEqual(dual_consumer.virtual_request, parent_consumer.virtual_request)
            self.assertNotIn(normal_request_id, parent_consumer._recving_metadata)
            self.assertEqual(dual_consumer._recving_metadata, parent_consumer._recving_metadata)
            self.assertNotIn(external_request_id, parent_consumer.request_map)
            self.assertEqual(dual_consumer.request_map, parent_consumer.request_map)

    def test_ordinary_done_and_failed_completion_matches_parent(self):
        parent_config = MockVllmConfig("decode", "kv_consumer")
        dual_config = MockVllmConfig("decode", "kv_consumer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        request_id = "reqC-load-00000001"
        params = {
            "remote_block_ids": [[1, 2]],
            "remote_block_size": [[16]],
            "remote_engine_id": "prefill-engine",
            "remote_host": "127.0.0.2",
            "remote_port": 6000,
            "remote_te_rpc_port": 9090,
        }
        parent_meta = MooncakeLayerwiseConnectorMetadata()
        dual_meta = MooncakeLayerwiseConnectorMetadata()
        parent_meta.add_new_req(request_id, [[7, 8]], dict(params))
        dual_meta.add_new_req(request_id, [[7, 8]], dict(params))

        with worker_environment():
            parent = MooncakeLayerwiseConnectorWorker(parent_config, MockKVCacheConfig(), "test_engine")
            dual = DualPathConnectorWorker(
                dual_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="decode"),
            )
            parent.register_kv_caches(make_kv_caches())
            dual.register_kv_caches(make_kv_caches())
            parent.start_load_kv(parent_meta)
            dual.start_load_kv(dual_meta)
            external_request_id = get_external_request_id(request_id)
            parent.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {external_request_id}
            dual.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {external_request_id}
            parent.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {external_request_id}
            dual.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {external_request_id}

            parent_result = parent.get_finished()
            dual_result = dual.get_finished(set())
            parent_invalid = parent.get_block_ids_with_load_errors()
            dual_invalid = dual.get_block_ids_with_load_errors()

        self.assertEqual(parent_result, (set(), {request_id}))
        self.assertEqual(dual_result, parent_result)
        self.assertEqual(dual_invalid, parent_invalid)
        self.assertEqual(dual.request_map, parent.request_map)
        self.assertEqual(dual._recving_metadata, parent._recving_metadata)

    def test_cleanup_matches_parent(self):
        parent, dual = self.make_scheduler_pair("prefill", "kv_producer")
        params = {
            "do_remote_decode": True,
            "remote_block_ids": [[4, 5]],
            "remote_block_size": [[16]],
            "remote_cached_tokens": 0,
        }
        parent_request = MockRequest(
            "req-cleanup",
            prompt_token_ids=list(range(4)),
            kv_transfer_params=dict(params),
        )
        dual_request = MockRequest(
            "req-cleanup",
            prompt_token_ids=list(range(4)),
            kv_transfer_params=dict(params),
        )
        parent.update_state_after_alloc(parent_request, MockBlocks([], ([[1, 2]],)), 0)
        dual.update_state_after_alloc(dual_request, MockBlocks([], ([[1, 2]],)), 0)

        parent_finished = parent.request_finished(parent_request, [1, 2])
        dual_finished = dual.request_finished(dual_request, [1, 2])

        self.assertEqual(parent_finished, (False, None))
        self.assertEqual(dual_finished, parent_finished)
        self.assertIn("req-cleanup", parent._reqs_need_send_layerwise)
        self.assertIn("req-cleanup", dual._reqs_need_send_layerwise)
        parent.build_connector_meta(
            MockSchedulerOutput(
                cached_req_ids=["req-cleanup"],
                cached_new_block_ids=[None],
                cached_num_computed=[0],
                num_sched={"req-cleanup": 4},
            )
        )
        dual.build_connector_meta(
            MockSchedulerOutput(
                cached_req_ids=["req-cleanup"],
                cached_new_block_ids=[None],
                cached_num_computed=[0],
                num_sched={"req-cleanup": 4},
            )
        )
        self.assertNotIn("req-cleanup", parent._reqs_need_send_layerwise)
        self.assertNotIn("req-cleanup", dual._reqs_need_send_layerwise)

    def test_failure_matches_parent(self):
        class ExpectedMetaserverError(RuntimeError):
            pass

        parent, dual = self.make_scheduler_pair("decode", "kv_consumer")
        parent_error = ExpectedMetaserverError("parent failure")
        dual_error = ExpectedMetaserverError("dual failure")
        parent.metaserver_client = MagicMock()
        dual.metaserver_client = MagicMock()
        parent.metaserver_client.post.side_effect = parent_error
        dual.metaserver_client.post.side_effect = dual_error
        with self.assertRaises(ExpectedMetaserverError) as parent_raised:
            parent._access_metaserver("http://meta", {"request_id": "bad"})
        with self.assertRaises(ExpectedMetaserverError) as dual_raised:
            dual._access_metaserver("http://meta", {"request_id": "bad"})
        self.assertIs(parent_raised.exception, parent_error)
        self.assertIs(dual_raised.exception, dual_error)
        self.assertEqual(
            parent.metaserver_client.post.call_args_list,
            dual.metaserver_client.post.call_args_list,
        )
        self.assertEqual(parent.metaserver_client.post.call_count, 3)

        parent_config = MockVllmConfig("decode", "kv_consumer")
        dual_config = MockVllmConfig("decode", "kv_consumer")
        parent_config.parallel_config.tensor_parallel_size = 1
        dual_config.parallel_config.tensor_parallel_size = 1
        params = {
            "remote_block_ids": [[1, 2]],
            "remote_block_size": [[16]],
            "remote_engine_id": "prefill-engine",
            "remote_host": "127.0.0.2",
            "remote_port": 6000,
            "remote_te_rpc_port": 9090,
        }
        parent_meta = MooncakeLayerwiseConnectorMetadata()
        dual_meta = MooncakeLayerwiseConnectorMetadata()
        failed_request_id = "reqE-bad-00000001"
        parent_meta.add_new_req(failed_request_id, [[7, 8]], dict(params))
        dual_meta.add_new_req(failed_request_id, [[7, 8]], dict(params))
        with worker_environment():
            parent_worker = MooncakeLayerwiseConnectorWorker(
                parent_config,
                MockKVCacheConfig(),
                "test_engine",
            )
            dual_worker = DualPathConnectorWorker(
                dual_config,
                MockKVCacheConfig(),
                "test_engine",
                DualPathConfig(role="decode"),
            )
            parent_worker.register_kv_caches(make_kv_caches())
            dual_worker.register_kv_caches(make_kv_caches())
            parent_worker.start_load_kv(parent_meta)
            dual_worker.start_load_kv(dual_meta)
            external_request_id = get_external_request_id(failed_request_id)
            parent_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
            dual_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
            parent_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {external_request_id}
            dual_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {external_request_id}
            self.assertEqual(parent_worker.get_finished(), (set(), set()))
            self.assertEqual(dual_worker.get_finished(set()), (set(), set()))
            parent_invalid = parent_worker.get_block_ids_with_load_errors()
            dual_invalid = dual_worker.get_block_ids_with_load_errors()
            parent_cleared = parent_worker.get_block_ids_with_load_errors()
            dual_cleared = dual_worker.get_block_ids_with_load_errors()
        self.assertEqual(parent_invalid, {7, 8})
        self.assertEqual(dual_invalid, parent_invalid)
        self.assertEqual(parent_cleared, set())
        self.assertEqual(dual_cleared, parent_cleared)


class TestDualPathFoundationGuards(unittest.TestCase):
    LIFECYCLE_METHODS = frozenset(
        {
            "get_num_new_matched_tokens",
            "update_state_after_alloc",
            "build_connector_meta",
            "request_finished",
            "request_finished_all_groups",
            "register_kv_caches",
            "get_finished",
            "get_block_ids_with_load_errors",
            "start_load_kv",
            "wait_for_layer_load",
            "save_kv_layer",
            "wait_for_save",
        }
    )
    FACADE_METHOD_EXEMPTIONS = frozenset({"get_finished"})

    def test_facade_lifecycle_methods_are_identical_to_parent(self):
        for method_name in self.LIFECYCLE_METHODS - self.FACADE_METHOD_EXEMPTIONS:
            with self.subTest(method=method_name):
                self.assertIs(
                    getattr(DualPathConnector, method_name),
                    getattr(MooncakeLayerwiseConnector, method_name),
                )

    def test_dual_path_classes_define_only_contracted_methods(self):
        # Task-00 contracted __init__ only; Task-01 adds exactly the Decode
        # admission surface below. Any further method must update this guard
        # together with its owning Task spec.
        expected_methods = {
            "DualPathConnector": {"__init__", "get_finished", "shutdown"},
            "DualPathConnectorScheduler": {
                "__init__",
                "_is_task01_decode_request",
                "_handle_prefill_decision",
                "_try_install_forward_plan",
                "_sweep_pe_delivery",
                "get_num_new_matched_tokens",
                "update_state_after_alloc",
                "build_connector_meta",
                "request_finished",
                "request_finished_all_groups",
                "shutdown",
            },
            "DualPathConnectorWorker": {
                "__init__",
                "_install_forward_receive_binding",
                "_release_finished_forward_terminals",
                "_consume_forward_receive_binding",
                "start_load_kv",
                "get_finished",
                "shutdown",
            },
        }
        for connector_class in (
            DualPathConnector,
            DualPathConnectorScheduler,
            DualPathConnectorWorker,
        ):
            with self.subTest(connector_class=connector_class.__name__):
                own_methods = {name for name, value in connector_class.__dict__.items() if inspect.isfunction(value)}
                self.assertEqual(own_methods, expected_methods[connector_class.__name__])
        self.assertEqual(self.LIFECYCLE_METHODS.intersection(DualPathConnector.__dict__), self.FACADE_METHOD_EXEMPTIONS)

    def test_facade_get_finished_forwards_core_finished_ids(self):
        # DualPath exempts get_finished from parent identity because the parent
        # facade drops the Core-finished IDs before calling its worker.
        connector = object.__new__(DualPathConnector)
        worker = object.__new__(DualPathConnectorWorker)
        worker.get_finished = MagicMock(return_value=({"sent"}, {"received"}))
        connector.connector_worker = worker
        finished_req_ids = {"decode-request-00000001"}

        result = connector.get_finished(finished_req_ids)

        self.assertEqual(result, ({"sent"}, {"received"}))
        worker.get_finished.assert_called_once_with(finished_req_ids)

    def test_parent_constructor_signatures_are_pinned(self):
        expected = ["self", "vllm_config", "kv_cache_config", "engine_id"]
        self.assertEqual(
            list(inspect.signature(MooncakeLayerwiseConnectorScheduler.__init__).parameters),
            expected,
        )
        self.assertEqual(
            list(inspect.signature(MooncakeLayerwiseConnectorWorker.__init__).parameters),
            expected,
        )

    def test_parent_worker_get_finished_source_is_pinned(self):
        # DualPath reimplements this method to retain early Forward terminals
        # while mirroring every ordinary completion, failure, and cleanup rule.
        source = inspect.getsource(MooncakeLayerwiseConnectorWorker.get_finished)
        self.assertEqual(
            hashlib.sha256(source.encode()).hexdigest(),
            "3e430f6cb2b4f6dd23e9da6fc6fed45bfa61c433350f3928b9cba040cdc4f399",
        )

    def test_parent_facade_constructor_still_builds_parent_classes(self):
        source = inspect.getsource(MooncakeLayerwiseConnector.__init__)
        self.assertIn("MooncakeLayerwiseConnectorScheduler(", source)
        self.assertIn("MooncakeLayerwiseConnectorWorker(", source)

    def test_parent_extra_config_keys_are_compatible(self):
        source = inspect.getsource(layerwise_module)
        keys = set(
            re.findall(
                r"get_from_extra_config\(\s*['\"]([^'\"]+)['\"]",
                source,
            )
        )
        self.assertLessEqual(keys, {"tls_config", "prefill", "decode"})


class TestDualPathRegistration(unittest.TestCase):
    MODULE_PATH = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"

    def test_register_connector_registers_dual_path(self):
        with (
            patch.object(KVConnectorFactory, "_registry", {}),
            patch.object(KVConnectorFactory, "register_connector") as register,
        ):
            register_kv_connectors()
        register.assert_any_call("DualPathConnector", self.MODULE_PATH, "DualPathConnector")
        matching_calls = [
            connector_call
            for connector_call in register.call_args_list
            if connector_call.args and connector_call.args[0] == "DualPathConnector"
        ]
        self.assertEqual(len(matching_calls), 1)

    def test_registered_module_path_resolves_to_dual_path_connector(self):
        module = importlib.import_module(self.MODULE_PATH)
        self.assertIs(module.DualPathConnector, DualPathConnector)

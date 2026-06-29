# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DualPathConnector Stage 1 foundation.

Covers: DualPathConfig validation matrix, connector construction under both
KVConnectorRole, inherited-behavior parity with MooncakeLayerwiseConnector,
parent-signature snapshot guards (doc R5/R9), and registration.

Mirrors the mocking conventions of
``tests/ut/kv_offload/test_mooncake_layerwise_connector.py``: ``mooncake.engine``
(and ``torch_npu`` if absent) are stubbed before importing the connector chain.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Stub C-extension / Ascend deps before importing the connector chain.
for _m in ("mooncake", "mooncake.engine"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules["mooncake.engine"].TransferEngine = MagicMock()  # type: ignore[attr-defined]
try:
    import torch_npu  # noqa: F401
except ImportError:
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")

from vllm.config import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
    SupportsHMA,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import (  # noqa: E402
    DEFAULT_WEIGHTS,
    DualPathConfig,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (  # noqa: E402
    DualPathConnector,
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: E402
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
)


def _make_vllm_config(role: str = "pe", kv_role: str = "kv_producer",
                      extra: dict | None = None) -> SimpleNamespace:
    """A minimal VllmConfig stand-in: only .kv_transfer_config is accessed."""
    merged = {"role": role}
    if extra:
        merged.update(extra)
    ktc = KVTransferConfig(
        kv_connector="DualPathConnector",
        kv_role=kv_role,
        kv_connector_extra_config=merged,
    )
    return SimpleNamespace(kv_transfer_config=ktc)


def _mock_kv_cache_config() -> MagicMock:
    cfg = MagicMock()
    group_spec = MagicMock()
    group_spec.kv_cache_spec.block_size = 16
    cfg.kv_cache_groups = [group_spec]
    return cfg


class TestDualPathConfig(unittest.TestCase):
    def test_valid_pe_with_producer(self):
        cfg = DualPathConfig.from_extra_config(
            {"role": "pe"}, _make_vllm_config(role="pe", kv_role="kv_producer").kv_transfer_config)
        self.assertEqual(cfg.role, "pe")
        self.assertFalse(cfg.path_planner.enabled)
        self.assertEqual(cfg.relay.control_plane, "zmq")

    def test_valid_de_with_consumer(self):
        cfg = DualPathConfig.from_extra_config(
            {"role": "de"}, _make_vllm_config(role="de", kv_role="kv_consumer").kv_transfer_config)
        self.assertEqual(cfg.role, "de")

    def test_kv_both_satisfies_either_role(self):
        for role in ("pe", "de"):
            cfg = DualPathConfig.from_extra_config(
                {"role": role},
                _make_vllm_config(role=role, kv_role="kv_both").kv_transfer_config)
            self.assertEqual(cfg.role, role)

    def test_role_missing_raises(self):
        with self.assertRaisesRegex(ValueError, "requires 'role'"):
            DualPathConfig.from_extra_config({}, _make_vllm_config().kv_transfer_config)

    def test_role_invalid_raises(self):
        with self.assertRaisesRegex(ValueError, "requires 'role'"):
            DualPathConfig.from_extra_config(
                {"role": "middle"}, _make_vllm_config().kv_transfer_config)

    def test_role_pe_requires_producer(self):
        with self.assertRaisesRegex(ValueError, "role='pe' requires kv_role"):
            DualPathConfig.from_extra_config(
                {"role": "pe"},
                _make_vllm_config(role="pe", kv_role="kv_consumer").kv_transfer_config)

    def test_role_de_requires_consumer(self):
        with self.assertRaisesRegex(ValueError, "role='de' requires kv_role"):
            DualPathConfig.from_extra_config(
                {"role": "de"},
                _make_vllm_config(role="de", kv_role="kv_producer").kv_transfer_config)

    def test_default_weights_and_params(self):
        cfg = DualPathConfig.from_extra_config(
            {"role": "pe"}, _make_vllm_config().kv_transfer_config)
        self.assertEqual(cfg.path_strategy.value_function.weights, dict(DEFAULT_WEIGHTS))
        self.assertEqual(cfg.path_strategy.value_function.decision_window_ms, 50)
        self.assertEqual(cfg.path_strategy.value_function.params["N_fail"], 5)

    def test_unknown_weight_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown path_strategy weights"):
            DualPathConfig.from_extra_config(
                {"role": "pe", "path_strategy": {"value_function": {
                    "weights": {"f_pe_io_headroom": 1.0, "bogus": 2.0}}}},
                _make_vllm_config().kv_transfer_config)

    def test_zero_weights_warns_but_ok(self):
        zero = {k: 0.0 for k in DEFAULT_WEIGHTS}
        with self.assertLogs(level="WARNING") as cm:
            cfg = DualPathConfig.from_extra_config(
                {"role": "pe", "path_strategy": {"value_function": {"weights": zero}}},
                _make_vllm_config().kv_transfer_config)
        self.assertTrue(any("degenerate" in m for m in cm.output))
        self.assertEqual(cfg.path_strategy.value_function.weights, zero)

    def test_decision_window_non_positive_rejected(self):
        with self.assertRaisesRegex(ValueError, "decision_window_ms"):
            DualPathConfig.from_extra_config(
                {"role": "pe", "path_strategy": {"value_function": {
                    "decision_window_ms": 0}}},
                _make_vllm_config().kv_transfer_config)

    def test_path_planner_enabled_rejected(self):
        with self.assertRaises(NotImplementedError):
            DualPathConfig.from_extra_config(
                {"role": "pe", "path_planner": {"enabled": True}},
                _make_vllm_config().kv_transfer_config)

    def test_unsupported_relay_rejected(self):
        with self.assertRaises(NotImplementedError):
            DualPathConfig.from_extra_config(
                {"role": "pe", "relay": {"control_plane": "urpc"}},
                _make_vllm_config().kv_transfer_config)

    def test_static_yaml_missing_file_raises(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            DualPathConfig.from_extra_config(
                {"role": "pe", "topology": {"static_yaml": "/no/such/topo.yaml"}},
                _make_vllm_config().kv_transfer_config)

    def test_static_yaml_present_ok(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("version: 1\n")
            path = f.name
        try:
            cfg = DualPathConfig.from_extra_config(
                {"role": "pe", "topology": {"static_yaml": path}},
                _make_vllm_config().kv_transfer_config)
            self.assertEqual(cfg.topology.static_yaml, path)
        finally:
            os.unlink(path)

    def test_monitor_sources_override(self):
        cfg = DualPathConfig.from_extra_config(
            {"role": "pe", "monitor": {"sources": {
                "mooncake": {"period_ms": 250},
                "extra": {"enabled": True}}}},
            _make_vllm_config().kv_transfer_config)
        self.assertEqual(cfg.monitor.sources["mooncake"]["period_ms"], 250)
        self.assertTrue(cfg.monitor.sources["prometheus"]["enabled"] is False)
        self.assertTrue(cfg.monitor.sources["extra"]["enabled"])


class TestDualPathConnectorConstruction(unittest.TestCase):
    def _build(self, role: KVConnectorRole, cfg_role: str = "pe",
               kv_role: str = "kv_producer"):
        cfg = _make_vllm_config(role=cfg_role, kv_role=kv_role)
        # Avoid the parent scheduler/worker heavy __init__ (TE engine, parallel
        # state, httpx client): patch them to no-ops so we test DualPath's wiring.
        with patch.object(MooncakeLayerwiseConnectorScheduler, "__init__",
                          lambda self, *a, **k: None), \
             patch.object(MooncakeLayerwiseConnectorWorker, "__init__",
                          lambda self, *a, **k: None):
            return DualPathConnector(cfg, role, _mock_kv_cache_config())

    def test_is_mooncake_layerwise_subclass(self):
        self.assertTrue(issubclass(DualPathConnector, MooncakeLayerwiseConnector))
        self.assertTrue(issubclass(DualPathConnector, SupportsHMA))
        self.assertTrue(issubclass(DualPathConnectorScheduler,
                                   MooncakeLayerwiseConnectorScheduler))
        self.assertTrue(issubclass(DualPathConnectorWorker,
                                   MooncakeLayerwiseConnectorWorker))

    def test_construct_scheduler_role(self):
        conn = self._build(KVConnectorRole.SCHEDULER, cfg_role="pe", kv_role="kv_producer")
        self.assertIsInstance(conn.connector_scheduler, DualPathConnectorScheduler)
        self.assertIsNone(conn.connector_worker)
        self.assertEqual(conn.dual_path_cfg.role, "pe")
        self.assertEqual(conn.connector_scheduler._req_path, {})
        self.assertIs(conn.connector_scheduler.dual_path_cfg, conn.dual_path_cfg)

    def test_construct_worker_role(self):
        conn = self._build(KVConnectorRole.WORKER, cfg_role="de", kv_role="kv_consumer")
        self.assertIsInstance(conn.connector_worker, DualPathConnectorWorker)
        self.assertIsNone(conn.connector_scheduler)
        self.assertEqual(conn.dual_path_cfg.role, "de")
        self.assertIs(conn.connector_worker.dual_path_cfg, conn.dual_path_cfg)

    def test_construct_unknown_role_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported KVConnectorRole"):
            DualPathConnector(_make_vllm_config(), "not_a_role", _mock_kv_cache_config())

    def test_get_num_new_matched_tokens_is_inherited(self):
        # Stage 1 does not override the matched-tokens contract; behavior is
        # identical to MooncakeLayerwiseConnector.
        self.assertIs(DualPathConnector.get_num_new_matched_tokens,
                      MooncakeLayerwiseConnector.get_num_new_matched_tokens)


class TestParentSignatureSnapshot(unittest.TestCase):
    """Guard against upstream drift (doc R5/R9). DualPathConnector's __init__
    override relies on the parent hard-naming its scheduler/worker."""

    def test_parent_scheduler_init_signature(self):
        params = list(inspect.signature(
            MooncakeLayerwiseConnectorScheduler.__init__).parameters)
        self.assertEqual(params, ["self", "vllm_config", "kv_cache_config", "engine_id"])

    def test_parent_worker_init_signature(self):
        params = list(inspect.signature(
            MooncakeLayerwiseConnectorWorker.__init__).parameters)
        self.assertEqual(params, ["self", "vllm_config", "kv_cache_config", "engine_id"])

    def test_parent_connector_init_still_hard_names_subclasses(self):
        src = inspect.getsource(MooncakeLayerwiseConnector.__init__)
        self.assertIn("MooncakeLayerwiseConnectorScheduler(", src)
        self.assertIn("MooncakeLayerwiseConnectorWorker(", src)


class TestDualPathRegistration(unittest.TestCase):
    def test_register_connector_registers_dual_path(self):
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        from vllm_ascend.distributed.kv_transfer import register_connector

        recorded: dict[str, tuple[str, str]] = {}

        def fake_register(name, module_path, class_name):
            recorded[name] = (module_path, class_name)

        # Seed the two names register_connector() pops directly off _registry,
        # so the real factory state does not matter.
        seeded = {"MultiConnector": None, "SimpleCPUOffloadConnector": None}
        with patch.object(KVConnectorFactory, "register_connector",
                          side_effect=fake_register), \
             patch.dict(KVConnectorFactory._registry, seeded, clear=False):
            register_connector()

        self.assertIn("DualPathConnector", recorded)
        self.assertEqual(
            recorded["DualPathConnector"],
            ("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector",
             "DualPathConnector"))

    def test_registered_module_path_resolves_to_class(self):
        mod = importlib.import_module(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector")
        self.assertIs(mod.DualPathConnector, DualPathConnector)


if __name__ == "__main__":
    unittest.main()

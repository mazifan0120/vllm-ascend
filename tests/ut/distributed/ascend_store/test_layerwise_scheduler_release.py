# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual scheduler release method without importing NPU modules.

This is a CPU failure-path check using the real retention registry. Extracting
the method avoids pretending that the CPU dependency shim imports the complete
production connector; native module import and model PD remain separate gates.
"""

import ast
import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ReleaseAck


def _release_method():
    source = (
        Path(__file__).resolve().parents[4]
        / "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py"
    )
    tree = ast.parse(source.read_text())
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MooncakeLayerwiseConnectorScheduler"
    )
    method = next(
        node for node in scheduler.body if isinstance(node, ast.FunctionDef) and node.name == "_release_target_if_any"
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {"ReleaseAck": ReleaseAck, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[method.name]


class TestSchedulerTargetRelease(unittest.TestCase):
    def setUp(self):
        self.fixture = OwnerFixture()
        self.request = SimpleNamespace(
            request_id="internal", kv_transfer_params={"request_instance": self.fixture.request.request_instance}
        )
        self.lease = self.fixture.acquire()
        self.scheduler = SimpleNamespace(
            _target_provider=self.fixture.facade,
            _request_leases={self.fixture.request.request_instance: self.lease},
            _retention_registry=self.fixture.retention,
        )
        self.release = _release_method()

    def tearDown(self):
        # Retrying release is idempotent and allowed for this CPU target on
        # which no remote transfer or compute was submitted.
        self.fixture.facade.release_target(self.lease)
        if self.fixture.allocation.retention_token in self.fixture.retention._records:
            self.fixture.retention.retire(
                self.fixture.allocation, ReleaseAck(True, self.fixture.allocation.retention_token)
            )
        self.fixture.facade.close()
        self.fixture.te.unregister_buffer()
        self.fixture.pool.close()

    def test_retire_failure_preserves_lease_and_defers_free_until_successful_retry(self):
        guard = self.fixture.retention.retain(self.fixture.allocation)
        self.assertFalse(self.release(self.scheduler, self.request))
        self.assertIn(self.fixture.request.request_instance, self.scheduler._request_leases)
        self.fixture.retention.validate(self.fixture.allocation)
        # A late counted owner guard prevents retirement, even though the
        # first owner release returned normally. The retry must check again.
        self.assertFalse(self.release(self.scheduler, self.request))
        guard.release()
        self.assertTrue(self.release(self.scheduler, self.request))
        self.assertEqual(self.scheduler._request_leases, {})

    def test_missing_registry_cannot_authorize_free(self):
        self.scheduler._retention_registry = None
        self.assertFalse(self.release(self.scheduler, self.request))
        self.assertIn(self.fixture.request.request_instance, self.scheduler._request_leases)
        self.fixture.retention.validate(self.fixture.allocation)

    def test_owner_failure_preserves_lease_and_registry(self):
        self.scheduler._target_provider = SimpleNamespace(
            release_target=Mock(side_effect=TimeoutError("writeback pending"))
        )
        self.assertFalse(self.release(self.scheduler, self.request))
        self.assertIn(self.fixture.request.request_instance, self.scheduler._request_leases)
        self.fixture.retention.validate(self.fixture.allocation)

    def test_plain_mode_has_no_receipt_retention(self):
        self.assertTrue(self.release(SimpleNamespace(_target_provider=None, _request_leases={}), self.request))

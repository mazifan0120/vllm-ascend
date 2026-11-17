# SPDX-License-Identifier: Apache-2.0
"""Explicit instance-owned local transport and actual connector binding tests."""

import importlib.util
import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store._mock_deps import _MOCK_VLLM_DEPS
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    ReleaseAck,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerDispatcher,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import DispatcherOwnerTransport

_HAS_CONNECTOR_DEPS = not _MOCK_VLLM_DEPS and importlib.util.find_spec("msgspec") is not None


class TestLocalOwnerTransport(unittest.TestCase):
    @staticmethod
    def client(fixture):
        return TargetProviderClient(DispatcherOwnerTransport(TargetOwnerDispatcher(fixture.facade)))

    @staticmethod
    def close(fixture, lease):
        fixture.ready(lease)
        TestLocalOwnerTransport.client(fixture).release_target(lease)
        fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
        fixture.facade.close()
        fixture.te.unregister_buffer()
        fixture.pool.close()

    def test_two_explicit_channels_keep_live_owners_independent(self):
        first, second = OwnerFixture(), OwnerFixture()
        client, peer = self.client(first), self.client(second)
        lease = client.acquire_target(first.request, first.allocation)
        other = peer.acquire_target(second.request, second.allocation)
        self.assertEqual(client.get_target_views(lease), lease.views)
        self.close(second, other)
        # Retiring the other owner cannot reset a process-global rendezvous.
        self.assertEqual(client.get_target_views(lease), lease.views)
        self.close(first, lease)

    def test_unbound_client_and_closed_owner_fail_closed(self):
        with self.assertRaises(ContractError):
            TargetProviderClient().catalog()
        fixture = OwnerFixture()
        client = self.client(fixture)
        fixture.facade.close()
        with self.assertRaises(ContractError):
            client.catalog()
        fixture.te.unregister_buffer()
        fixture.pool.close()

    @unittest.skipUnless(_HAS_CONNECTOR_DEPS, "real vLLM and msgspec needed for connector integration")
    def test_connector_requires_explicit_authority_and_transport(self):
        """Production initialization cannot silently use a module rendezvous."""
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector
        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
            DeferredTargetProvider,
            TargetOwnerDispatcher,
        )
        from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import DispatcherOwnerTransport

        fixture = OwnerFixture()
        scheduler = object.__new__(MooncakeLayerwiseConnector)
        scheduler.connector_scheduler = SimpleNamespace()
        scheduler.connector_worker = None
        scheduler.target_provider = TargetProviderClient()
        scheduler._target_rpc_config = None
        worker = object.__new__(MooncakeLayerwiseConnector)
        worker.connector_scheduler = None
        worker.connector_worker = SimpleNamespace(engine=fixture.te.transfer_engine)
        worker.target_provider = DeferredTargetProvider()
        worker._target_owner_binding = None
        worker._target_rpc_config = None
        with self.assertRaisesRegex(ContractError, "explicit RPC ports"):
            scheduler.bind_target_owner_transport()
        with self.assertRaisesRegex(ContractError, "explicit RPC ports"):
            worker.configure_target_owner(
                object(), retention=None, consumption_stream=fixture.streams, timeout=1, te=fixture.te
            )
        scheduler.bind_target_owner_transport(DispatcherOwnerTransport(TargetOwnerDispatcher(fixture.facade)))
        worker.configure_target_owner(
            object(), retention=fixture.retention, consumption_stream=fixture.streams, timeout=1, te=fixture.te
        )
        self.assertIs(worker._target_owner_binding[2], fixture.retention)
        lease = scheduler.target_provider.acquire_target(fixture.request, fixture.allocation)
        self.assertEqual(scheduler.target_provider.get_target_views(lease), lease.views)
        scheduler.target_provider.release_target(lease)
        fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
        fixture.facade.close()
        fixture.te.unregister_buffer()
        fixture.pool.close()


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Co-located target owner channel tests.

The scheduler ``TargetProviderClient`` reaches the worker owner through the
module-level rendezvous, while the scheduler retention registry is shared by
reference so the owner runtime's nested ``retain()`` during ``acquire_target``
calls back into the scheduler process synchronously.
"""

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
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import target_owner_channel
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetProviderClient

_HAS_CONNECTOR_DEPS = not _MOCK_VLLM_DEPS and importlib.util.find_spec("msgspec") is not None


class TestCoLocatedOwnerChannel(unittest.TestCase):
    def setUp(self):
        target_owner_channel.reset_active_channel()

    def tearDown(self):
        target_owner_channel.reset_active_channel()

    def test_colocated_transport_drives_full_lifecycle_with_nested_retain(self):
        f = OwnerFixture()
        target_owner_channel.bind_worker_dispatcher(f.facade)
        target_owner_channel.bind_scheduler_retention_authority(f.retention)
        client = TargetProviderClient(target_owner_channel.LocalOwnerTransport())

        # acquire nests a reverse retain on the shared scheduler registry; the
        # co-located channel runs it on the calling thread without deadlock.
        catalog = client.catalog()
        self.assertEqual(len(catalog.entries), 8)
        lease = client.acquire_target(f.request, f.allocation)
        self.assertEqual(client.get_target_views(lease), lease.views)

        f.ready(lease)
        client.release_target(lease)
        # The shared registry still holds the authoritative record; retiring it
        # through the scheduler's own free path must succeed after release.
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_scheduler_client_fails_closed_when_dispatcher_unbound(self):
        f = OwnerFixture()
        # Registry is bound but no dispatcher published: admission must fail
        # loudly rather than silently wait forever.
        target_owner_channel.bind_scheduler_retention_authority(f.retention)
        client = TargetProviderClient(target_owner_channel.LocalOwnerTransport())
        with self.assertRaises(ContractError):
            client.catalog()
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_dispatcher_cannot_be_rebound_across_owner_generations(self):
        f = OwnerFixture()
        target_owner_channel.bind_worker_dispatcher(f.facade)
        with self.assertRaises(ContractError):
            target_owner_channel.bind_worker_dispatcher(f.facade)
        target_owner_channel.reset_active_channel()
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_registry_cannot_be_rebound(self):
        f = OwnerFixture()
        target_owner_channel.bind_scheduler_retention_authority(f.retention)
        with self.assertRaises(ContractError):
            target_owner_channel.bind_scheduler_retention_authority(f.retention)
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_reset_channel_allows_a_new_owner_generation(self):
        f = OwnerFixture()
        target_owner_channel.bind_worker_dispatcher(f.facade)
        target_owner_channel.bind_scheduler_retention_authority(f.retention)
        client = TargetProviderClient(target_owner_channel.LocalOwnerTransport())
        lease = client.acquire_target(f.request, f.allocation)
        f.ready(lease)
        client.release_target(lease)
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

        # A fresh owner generation on a reset channel re-binds cleanly.
        target_owner_channel.reset_active_channel()
        g = OwnerFixture()
        target_owner_channel.bind_worker_dispatcher(g.facade)
        target_owner_channel.bind_scheduler_retention_authority(g.retention)
        client = TargetProviderClient(target_owner_channel.LocalOwnerTransport())
        lease = client.acquire_target(g.request, g.allocation)
        self.assertEqual(client.get_target_views(lease), lease.views)
        g.ready(lease)
        client.release_target(lease)
        g.retention.retire(g.allocation, ReleaseAck(True, g.allocation.retention_token))
        g.facade.close()
        g.te.unregister_buffer()
        g.pool.close()

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

# SPDX-License-Identifier: Apache-2.0
"""Co-located target owner channel tests.

The scheduler ``TargetProviderClient`` reaches the worker owner through the
module-level rendezvous, while the scheduler retention registry is shared by
reference so the owner runtime's nested ``retain()`` during ``acquire_target``
calls back into the scheduler process synchronously.
"""

import importlib.util
import unittest

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
    def test_connector_colocated_auto_wiring_through_channel(self):
        """The connector's configure/bind methods auto-wire through the
        module-level channel when the caller omits the transport and retention
        arguments, exactly as a co-located deployment would.

        The scheduler wrapper publishes its retention registry to the channel
        (the deployment's responsibility, mirroring bind_scheduler_retention);
        the connector facade methods then resolve the missing transport and
        retention from that same channel.
        """
        from types import SimpleNamespace

        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
            MooncakeLayerwiseConnector,
            MooncakeLayerwiseConnectorWorker,
        )

        f = OwnerFixture()

        # A stub sparse manager whose bind_registered_target_owner binds the
        # pre-built owner into the facade, mirroring the real A2 path without
        # re-implementing cache spec inspection (covered by A's own tests).
        class _StubManager:
            def __init__(self, fixture):
                self._fixture = fixture

            def bind_registered_target_owner(
                self, kv_caches, te, *, retention, consumption_stream, timeout, facade=None
            ):
                facade.bind(self._fixture.owner)

        worker = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
        worker.engine = f.te.transfer_engine
        worker.kv_recv_layer_thread = None
        worker.target_provider = f.facade

        scheduler_connector = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
        scheduler_connector.connector_scheduler = SimpleNamespace()
        scheduler_connector.connector_worker = None
        scheduler_connector.target_provider = TargetProviderClient()
        scheduler_connector._target_owner_binding = None
        scheduler_connector._target_provider = None

        worker_connector = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
        worker_connector.connector_scheduler = None
        worker_connector.connector_worker = worker
        worker_connector.target_provider = f.facade
        worker_connector._target_owner_binding = None
        worker_connector._target_provider = None

        # Scheduler wrapper publishes its retention registry to the channel;
        # the scheduler facade then auto-wires its transport from the channel.
        target_owner_channel.bind_scheduler_retention_authority(f.retention)
        scheduler_connector.bind_target_owner_transport(None)
        # Worker configure auto-wires retention from the channel when omitted.
        worker_connector.configure_target_owner(
            _StubManager(f), retention=None, consumption_stream=f.streams, timeout=0.02, te=f.te
        )
        # register_kv_caches binds the owner and publishes the dispatcher.
        worker_connector.register_kv_caches({})

        client = scheduler_connector.target_provider
        catalog = client.catalog()
        self.assertEqual(len(catalog.entries), 8)
        lease = client.acquire_target(f.request, f.allocation)
        self.assertEqual(client.get_target_views(lease), lease.views)
        f.ready(lease)
        client.release_target(lease)
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()


if __name__ == "__main__":
    unittest.main()

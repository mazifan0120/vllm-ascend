# SPDX-License-Identifier: Apache-2.0
"""Production role factories over real loopback sockets; CPU descriptors only."""

import socket
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_coordinator import DescriptorOwner, coordinated
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError, ReleaseAck
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerDispatcher,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import (
    SchedulerTargetRpc,
    TargetOwnerRpcConfig,
    WorkerTargetRpc,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import (
    SchedulerRetentionRegistry,
    SerializedKVCacheManager,
)


def _endpoints(count):
    sockets = [socket.socket() for _ in range(count)]
    try:
        for stream in sockets:
            stream.bind(("127.0.0.1", 0))
        return tuple(f"tcp://127.0.0.1:{stream.getsockname()[1]}" for stream in sockets)
    finally:
        for stream in sockets:
            stream.close()


class MutableManager:
    def __init__(self):
        self.ids = {"internal": ((1, 3),)}
        self.frees = []
        self.entered = threading.Event()
        self.resume = threading.Event()

    def get_block_ids(self, request_id):
        return self.ids[request_id]

    def free(self, request):
        self.frees.append(request.request_id)
        self.ids[request.request_id] = ((),)

    def pop_blocks_for_free(self, request):
        return self.free(request)

    def allocate_slots(self, request, num_new_tokens):
        self.ids[request.request_id] = ((99,),)
        self.entered.set()
        if not self.resume.wait(3):
            raise TimeoutError("CPU allocation fixture was not released")
        self.ids[request.request_id] = ((1, 3, 7),)


def _registry():
    raw = MutableManager()
    lock = threading.RLock()
    manager = SerializedKVCacheManager(raw, lock)
    registry = SchedulerRetentionRegistry(manager, allocation_lock=lock)
    manager.bind_retention_registry(registry)
    _, _, request, allocation = coordinated()
    allocation = registry.issue("internal", replace(allocation, retention_token=None))
    return raw, manager, registry, request, allocation


class TestSerializedAllocationAuthority(unittest.TestCase):
    def test_free_and_pop_require_retirement_even_after_last_guard_release(self):
        raw, manager, registry, _, allocation = _registry()
        request = SimpleNamespace(request_id="internal")
        guard = registry.retain(allocation)
        for free in (manager.free, manager.pop_blocks_for_free):
            with self.assertRaises(ContractError):
                free(request)
        guard.release()
        with self.assertRaises(ContractError):
            manager.free(request)
        self.assertEqual(raw.frees, [])
        registry.retire(allocation, ReleaseAck(True, allocation.retention_token))
        manager.free(request)
        self.assertEqual(raw.frees, ["internal"])

    def test_reverse_authority_cannot_observe_intermediate_allocation_state(self):
        raw, manager, registry, _, allocation = _registry()
        request = SimpleNamespace(request_id="internal")
        started, validated = threading.Event(), threading.Event()
        failures = []

        def validate():
            started.set()
            try:
                registry.validate(allocation)
                validated.set()
            except Exception as error:
                failures.append(error)

        writer = threading.Thread(target=manager.allocate_slots, args=(request, 16))
        writer.start()
        self.assertTrue(raw.entered.wait(1))
        reader = threading.Thread(target=validate)
        reader.start()
        try:
            self.assertTrue(started.wait(1))
            self.assertFalse(validated.wait(0.05))
            self.assertTrue(reader.is_alive())
        finally:
            raw.resume.set()
            writer.join(3)
            reader.join(3)
        self.assertFalse(writer.is_alive())
        self.assertFalse(reader.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(validated.is_set())
        self.assertEqual(manager.get_block_ids("internal"), ((1, 3, 7),))

    def test_second_active_generation_for_same_internal_request_is_rejected(self):
        _, _, registry, request, allocation = _registry()
        with self.assertRaises(ContractError):
            registry.issue(
                "internal", replace(allocation, request=replace(request, transfer_generation=1), retention_token=None)
            )


class TestTargetRoleFactories(unittest.TestCase):
    def test_explicit_port_configuration_covers_all_ranks_and_rejects_overlap(self):
        config = TargetOwnerRpcConfig.from_extra(
            {"target_owner_rpc_base_port": 24000, "target_retention_rpc_port": 24016}, 16
        )
        self.assertEqual(len(config.owner_endpoints), 16)
        self.assertEqual(config.owner_endpoints[-1], "tcp://127.0.0.1:24015")
        self.assertIsNone(TargetOwnerRpcConfig.from_extra({}, 16))
        for extra in (
            {"target_owner_rpc_base_port": True, "target_retention_rpc_port": 24016},
            {"target_owner_rpc_base_port": 24000, "target_retention_rpc_port": 24001},
            {"target_owner_rpc_base_port": 65535, "target_retention_rpc_port": 1},
            {"target_owner_rpc_base_port": 24000},
            {
                "target_owner_rpc_base_port": 24000,
                "target_retention_rpc_port": 24016,
                "target_owner_rpc_timeout_s": float("inf"),
            },
        ):
            with self.subTest(extra=extra), self.assertRaises(ContractError):
                TargetOwnerRpcConfig.from_extra(extra, 16)

    def test_real_two_rank_role_factories_use_reverse_retention_and_preserve_local_leases(self):
        raw, _, registry, request, allocation = _registry()
        ports = _endpoints(3)
        config = TargetOwnerRpcConfig(ports[:2], ports[2], 3)
        workers = []
        owners = []
        scheduler = None

        class Facade:
            ready = True

            def __init__(self, owner):
                self.owner = owner

            def close(self):
                if self.owner.provider._states:
                    raise ContractError("CPU owner target remains active")

        try:
            # Workers register before the scheduler exists. Client construction
            # and binding the owner server must not wait for reverse retention.
            for rank in range(2):
                owner = DescriptorOwner(rank, 2)
                facade = Facade(owner)
                worker = WorkerTargetRpc(config, rank, facade)
                guards = {}

                def retain(lease, authority=worker.retention, guards=guards):
                    guards[lease.request] = authority.retain(lease.allocation)

                def release(lease, guards=guards):
                    guards.pop(lease.request).release()

                owner.retain = retain
                owner.release = release
                worker.serve_registered_owner(TargetOwnerDispatcher(owner))
                workers.append(worker)
                owners.append(owner)
            scheduler = SchedulerTargetRpc(config, registry)
            client = TargetProviderClient(scheduler.transport)
            lease = client.acquire_target(request, allocation)
            manifest = client.get_target_manifest(lease)
            self.assertEqual(len(manifest.targets), 2)
            self.assertEqual(registry._records[allocation.retention_token].guards, 2)
            with self.assertRaises(ContractError):
                scheduler.close()
            for worker in workers:
                with self.assertRaises(ContractError):
                    worker.close()
            client.release_target(lease)
            self.assertEqual(registry._records[allocation.retention_token].guards, 0)
            self.assertEqual(raw.ids["internal"], allocation.group_block_ids)
            registry.retire(allocation, ReleaseAck(True, allocation.retention_token))
        finally:
            if scheduler is not None:
                scheduler.close()
            for worker in workers:
                worker.close()

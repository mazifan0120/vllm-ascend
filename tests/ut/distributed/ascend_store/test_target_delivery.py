# SPDX-License-Identifier: Apache-2.0
"""Admission/retirement proofs over owner RPC; CPU descriptors, no DMA."""

import unittest
from dataclasses import replace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_coordinator import coordinated
from tests.ut.distributed.ascend_store.test_target_owner_factory import _endpoints
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    Status,
    TargetClosedError,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_coordinator import MultiRankTargetProvider
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_delivery import WorkerTargetDelivery
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_rpc import OwnerRpcClient, OwnerRpcServer


class TestTargetDelivery(unittest.TestCase):
    def test_compact_admission_rejects_missing_or_foreign_entries(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        local = manifest.for_rank(1).lease
        plan = ManifestWritePlan(manifest)
        entries = plan.receive_keys(1)
        installed = []
        control = WorkerTargetDelivery(1, owners[1], lambda *args: installed.append(args))
        try:
            with self.assertRaisesRegex(ContractError, "cover the local registered views"):
                control.admit_target(local, "internal", entries[:-1])
            with self.assertRaisesRegex(ContractError, "cover the local registered views"):
                control.admit_target(local, "internal", plan.receive_keys(0))
            self.assertEqual(installed, [])
            control.admit_target(local, "internal", entries)
            self.assertEqual(len(installed), 1)
        finally:
            provider.release_target(lease)

    def test_abort_after_address_lookup_retains_target_until_all_writers_quiesce(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        local = manifest.for_rank(0).lease
        ledgers = []
        control = WorkerTargetDelivery(0, owners[0], lambda _, __, ledger: ledgers.append(ledger))
        try:
            control.admit_target(manifest, "internal")
            with control.source_lookup(local):
                pass
            control.abort_requests({"internal"})
            with self.assertRaises(TargetClosedError), control.source_lookup(local):
                self.fail("An aborted target exposed native addresses again")
            self.assertEqual(control.delivery_status(local), {"ready": False, "remote_quiesced": False})
            with self.assertRaises(ContractError):
                control.release_target(local)
            ledgers[0].finish_rank(request, 0, Status.SUCCESS, True)
            ledgers[0].finish_rank(request, 1, Status.CANCELLED, False)
            with self.assertRaises(ContractError):
                control.release_target(local)
            self.assertEqual(owners[0].releases, [])
            ledgers[0].finish_rank(request, 1, Status.CANCELLED, True)
            self.assertEqual(control.delivery_status(local), {"ready": False, "remote_quiesced": True})
            control.release_target(local)
            self.assertEqual(owners[0].releases, [local])
        finally:
            provider.release_target(lease)

    def test_real_two_owner_rpc_ack_waits_for_all_receive_maps_and_retains_exact_proofs(self):
        _, owners, request, allocation = coordinated()
        maps = ({}, {})
        controls = []
        servers, transports = [], []
        provider = None
        lease = None
        try:
            for rank, endpoint in enumerate(_endpoints(2)):

                def install(internal, local, ledger, rank=rank):
                    maps[rank][internal] = (local, ledger)

                control = WorkerTargetDelivery(rank, owners[rank], install)
                controls.append(control)
                servers.append(
                    OwnerRpcServer(
                        endpoint, TargetOwnerDispatcher(owners[rank], delivery=control), command_type=TargetOwnerCall
                    )
                )
                transports.append(OwnerRpcClient(endpoint, timeout=3))
            provider = MultiRankTargetProvider(tuple(TargetProviderClient(t) for t in transports))
            lease = provider.acquire_target(request, allocation)
            manifest = provider.manifest(lease)
            self.assertEqual(provider.delivery_status(lease), {"ready": False, "remote_quiesced": True})
            provider.admit_target(manifest, "internal-request")
            with self.assertRaises(ContractError):
                provider.admit_target(manifest, "changed-internal-request")
            self.assertTrue(all("internal-request" in mapping for mapping in maps))
            # Scheduler metadata can be delivered later, without replacing a
            # ledger that already contains fast peer completions.
            provider.admit_target(manifest, "internal-request")
            self.assertEqual(provider.delivery_status(lease), {"ready": False, "remote_quiesced": False})
            plan = ManifestWritePlan(manifest)
            for rank in (0, 1):
                local, ledger = maps[rank]["internal-request"]
                self.assertEqual(local, manifest.for_rank(rank).lease)
                for key in plan.receive_keys(rank):
                    ledger.record(TransferResult(key, Status.SUCCESS, True))
                for writer in (0, 1):
                    ledger.finish_rank(request, writer, Status.SUCCESS, True)
                self.assertFalse(provider.delivery_status(lease)["ready"])
                ledger.mark_visible(rank)
            self.assertEqual(provider.delivery_status(lease), {"ready": True, "remote_quiesced": True})
        finally:
            if lease is not None:
                provider.release_target(lease)
            for transport in transports:
                transport.close()
            for server in servers:
                server.close()

    def test_owner_rejects_mismatched_lease_or_request_mapping_and_preserves_first_ledger(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        installed = []
        control = WorkerTargetDelivery(0, owners[0], lambda *args: installed.append(args))
        control.admit_target(manifest, "internal")
        control.admit_target(manifest, "internal")
        self.assertEqual(len(installed), 1)
        with self.assertRaises(ContractError):
            control.admit_target(manifest, "another-internal")
        local = manifest.for_rank(0).lease
        with self.assertRaises(ContractError):
            control.delivery_status(replace(local, allocation=replace(local.allocation, retention_token="changed")))
        provider.release_target(lease)

    def test_uncertain_installer_is_never_acknowledged_or_retried_as_clean_admission(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        calls = []

        def install(*args):
            calls.append(args)
            raise RuntimeError("injected partial receive installation")

        control = WorkerTargetDelivery(0, owners[0], install)
        with self.assertRaises(RuntimeError):
            control.admit_target(manifest, "internal")
        with self.assertRaises(ContractError):
            control.admit_target(manifest, "internal")
        self.assertEqual(control.delivery_status(manifest.for_rank(0).lease), {"ready": False, "remote_quiesced": True})
        self.assertEqual(len(calls), 1)
        provider.release_target(lease)

    def test_partial_rank_admission_quarantines_target_and_cannot_publish(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        calls = []
        owners[0].admit_target = lambda *_: calls.append(0)

        def fail(*args):
            raise TimeoutError("rank 1 did not acknowledge installation")

        owners[1].admit_target = fail
        with self.assertRaisesRegex(ContractError, "producer must not be notified"):
            provider.admit_target(manifest, "internal")
        self.assertEqual(calls, [0])
        with self.assertRaises(ContractError):
            provider.manifest(lease)
        with self.assertRaises(ContractError):
            provider.admit_target(manifest, "internal")
        provider.release_target(lease)

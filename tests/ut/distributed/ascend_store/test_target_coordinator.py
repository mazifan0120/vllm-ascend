# SPDX-License-Identifier: Apache-2.0
"""Rank/entry coordination on explicit descriptors, not physical memory tests."""

import unittest
from copy import deepcopy
from dataclasses import replace
from threading import Barrier

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    BlockAllocation,
    CacheView,
    Component,
    ContractError,
    ReceiveReceipt,
    RegisteredRegion,
    RequestKey,
    TokenRange,
    WriterPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import (
    target_manifest_from_wire,
    target_manifest_to_wire,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import TargetCatalog, TargetEntry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_coordinator import MultiRankTargetProvider
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import TargetManifest
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerDispatcher,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import (
    decode_owner_value,
    encode_owner_value,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_factory import DispatcherOwnerTransport
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import (
    RegisteredTargetProvider,
    TargetBinding,
)


class DescriptorOwner:
    """Real provider state machine over declared, non-allocated test geometry."""

    def __init__(self, rank, tp_size, layers=2, *, block_capacity=4, block_size=16):
        self.rank = rank
        self.retains, self.prepares, self.releases = [], [], []
        self.fail_release = False
        main_entry_bytes = block_capacity * 64
        main_layer_bytes = 2 * main_entry_bytes
        host = RegisteredRegion(
            f"owner-{rank}",
            1,
            f"host-{rank}",
            "shared-host",
            3,
            0,
            layers * main_layer_bytes,
            "host",
            f"node:{10000 + rank}",
        )
        regions = [host]
        entries = []
        for layer in range(layers):
            for component in Component:
                for entry in range(2):
                    if component is Component.MAIN:
                        region = host
                        dtype, shape, strides, page_bytes = "bfloat16", (block_capacity, 16, 2), (64, 4, 2), 64
                        offset, page_tokens = layer * main_layer_bytes + entry * main_entry_bytes, block_size
                    else:
                        stride = 32 if entry == 0 else 16
                        indexer_pages = 2 * block_capacity
                        region = RegisteredRegion(
                            f"owner-{rank}",
                            1,
                            f"index-{rank}-{layer}-{entry}",
                            f"device-{rank}-{layer}-{entry}",
                            1,
                            0,
                            indexer_pages * stride,
                            "device",
                            f"node:{10000 + rank}",
                        )
                        regions.append(region)
                        dtype = "int8" if entry == 0 else "float16"
                        shape = (indexer_pages, 8, 4 if entry == 0 else 1)
                        strides = (stride, 4, 1) if entry == 0 else (stride, 2, 2)
                        page_bytes, offset, page_tokens = stride, 0, block_size // 2
                    view = CacheView(
                        component,
                        layer,
                        region.region_id,
                        dtype,
                        (strides[0],),
                        page_tokens,
                        block_size,
                        block_capacity,
                        offset,
                        (),
                    )
                    binding = TargetBinding(view, 0, page_bytes)
                    entries.append(
                        TargetEntry(
                            f"layer.{layer}.{component.value}", component, layer, 0, entry, shape, strides, binding
                        )
                    )
        plan = WriterPlan({"shared-host": (0,)}, {"shared-host": (TokenRange(0, block_capacity * block_size),)})
        required = {component: tuple(range(layers)) for component in Component}
        self.descriptor = TargetCatalog(CONTRACT_VERSION, rank, tp_size, tuple(regions), tuple(entries), required, plan)
        self.provider = RegisteredTargetProvider(
            rank=rank,
            tp_size=tp_size,
            regions=tuple(regions),
            bindings=tuple(e.binding for e in entries),
            required_layers=required,
            writer_plan=plan,
            runtime=self,
        )

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def catalog(self):
        return deepcopy(self.descriptor)

    def retain(self, lease):
        self.retains.append(lease)

    def make_visible(self, lease, receipt):
        self.prepares.append(lease)

    def release(self, lease):
        if self.fail_release:
            raise TimeoutError("CPU descriptor fixture: pending writeback")
        self.releases.append(lease)


def coordinated(tp=2, layers=2):
    owners = tuple(DescriptorOwner(rank, tp, layers) for rank in range(tp))
    request = RequestKey("descriptor-request", 0)
    allocation = BlockAllocation(request, 9, ((1, 3),), 16, 23, "descriptor-fixture-retention")
    return MultiRankTargetProvider(owners), owners, request, allocation


class TestMultiRankTargets(unittest.TestCase):
    def test_aggregate_has_single_main_and_every_rank_indexer_with_original_owner_leases(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        self.assertEqual(len([v for v in lease.views if v.component is Component.MAIN]), 4)
        self.assertEqual(len([v for v in lease.views if v.component is Component.INDEXER]), 8)
        self.assertEqual(len([r for r in lease.regions if r.memory_kind == "host"]), 1)
        self.assertEqual(lease.writer_plan.ranges_by_backing, {"shared-host": (TokenRange(0, 23),)})
        self.assertEqual({rank for rank, e in manifest.transfer_entries() if e.component is Component.INDEXER}, {0, 1})
        for rank, owner in enumerate(owners):
            self.assertEqual(manifest.for_rank(rank).lease, owner.retains[0])
            self.assertNotEqual(manifest.for_rank(rank).lease, lease)
        provider.prepare_decode(lease, ReceiveReceipt(request, 9, True, True, True))
        for rank, owner in enumerate(owners):
            self.assertEqual(owner.prepares, [manifest.for_rank(rank).lease])
        provider.release_target(lease)
        provider.release_target(lease)
        for rank, owner in enumerate(owners):
            self.assertEqual(owner.releases, [manifest.for_rank(rank).lease])

    def test_aggregate_and_manifest_cross_the_real_value_codec_and_dispatch_boundary(self):
        provider, _, request, allocation = coordinated()
        dispatcher = TargetOwnerDispatcher(provider)

        class Transport:
            def call(self, command):
                decoded = decode_owner_value(encode_owner_value(command))
                return decode_owner_value(encode_owner_value(dispatcher.dispatch(decoded)))

        client = TargetProviderClient(Transport())
        lease = client.acquire_target(request, allocation)
        manifest = client.get_target_manifest(lease)
        self.assertEqual(manifest, provider.manifest(lease))
        self.assertEqual(target_manifest_from_wire(target_manifest_to_wire(manifest)), manifest)
        self.assertIsInstance(manifest, TargetManifest)
        self.assertEqual(client.get_target_views(lease), lease.views)
        client.release_target(lease)

    def test_scheduler_local_manifest_isolated_after_removing_duplicate_copy(self):
        provider, _, request, allocation = coordinated()
        exported_catalogs = provider.catalogs()
        exported_catalogs[0].required_layers.clear()
        client = TargetProviderClient(DispatcherOwnerTransport(TargetOwnerDispatcher(provider, copy_reply=False)))
        lease = client.acquire_target(request, allocation)
        try:
            expected = client.get_target_manifest(lease)
            published = client.get_target_manifest(lease)
            published.lease.writer_plan.writers_by_backing.clear()
            published.for_rank(0).catalog.required_layers.clear()

            self.assertNotEqual(published, expected)
            self.assertEqual(client.get_target_manifest(lease), expected)
        finally:
            client.release_target(lease)

    def test_local_publication_checks_sender_count_and_admits_each_exact_owner(self):
        provider, owners, request, allocation = coordinated()
        admissions = {}
        for rank, owner in enumerate(owners):
            owner.admit_target = lambda *args, rank=rank: admissions.setdefault(rank, args)
        lease = provider.acquire_target(request, allocation)
        try:
            manifest = provider.manifest(lease)
            with self.assertRaisesRegex(ContractError, "Producer and consumer TP sizes"):
                provider.publish_target(lease, "engine-request", 1)
            self.assertFalse(admissions)
            wire = provider.publish_target(lease, "engine-request", 2)
            self.assertEqual(wire, target_manifest_to_wire(manifest))
            plan = ManifestWritePlan(manifest)
            self.assertEqual(
                admissions,
                {rank: (manifest.for_rank(rank).lease, "engine-request", plan.receive_keys(rank)) for rank in range(2)},
            )
            wire.clear()
            self.assertEqual(provider.manifest(lease), manifest)
        finally:
            provider.release_target(lease)

    def test_parallel_delivery_poll_requires_every_owner_proof(self):
        _, owners, request, allocation = coordinated()
        provider = MultiRankTargetProvider(owners, parallel_owner_calls=True)
        lease = provider.acquire_target(request, allocation)
        barrier = Barrier(len(owners), timeout=5)

        def ready(_):
            barrier.wait()
            return {"ready": True, "remote_quiesced": True}

        def pending(_):
            barrier.wait()
            return {"ready": False, "remote_quiesced": True}

        owners[0].delivery_status = ready
        owners[1].delivery_status = pending
        self.assertEqual(provider.delivery_status(lease), {"ready": False, "remote_quiesced": True})
        owners[1].delivery_status = ready
        self.assertEqual(provider.delivery_status(lease), {"ready": True, "remote_quiesced": True})
        provider.release_target(lease)

    def test_tp16_model_sized_manifest_round_trip_fits_transport_limits(self):
        owners = tuple(DescriptorOwner(rank, 16, 78, block_capacity=513, block_size=128) for rank in range(16))
        provider = MultiRankTargetProvider(owners)
        request = RequestKey("model-sized", 0)
        allocation = BlockAllocation(request, 9, (tuple(range(1, 490)),), 128, 62474, "descriptor-fixture-retention")
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        frame = encode_owner_value(manifest)
        self.assertEqual(decode_owner_value(frame), manifest)
        self.assertEqual(len(manifest.targets), 16)
        self.assertEqual(len(manifest.transfer_entries()), 78 * 2 * 17)
        provider.release_target(lease)

    def test_idempotent_acquire_does_not_retain_twice(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        self.assertEqual(provider.acquire_target(request, allocation), lease)
        self.assertEqual([len(owner.retains) for owner in owners], [1, 1])
        with self.assertRaises(ContractError):
            provider.acquire_target(request, replace(allocation, prompt_tokens=22))
        provider.release_target(lease)
        with self.assertRaises(ContractError):
            provider.acquire_target(request, allocation)

    def test_failed_acquire_preserves_successful_and_uncertain_owner_records(self):
        provider, owners, request, allocation = coordinated()

        def unresolved(*args):
            raise TimeoutError("RPC reply not observed")

        owners[1].acquire_target = unresolved
        with self.assertRaisesRegex(ContractError, "remain retained"):
            provider.acquire_target(request, allocation)
        self.assertEqual(provider._targets[request].acquired, {0: owners[0].retains[0]})
        self.assertEqual(owners[0].releases, [])
        with self.assertRaisesRegex(ContractError, "unresolved"):
            provider.acquire_target(request, allocation)

    def test_parallel_acquire_publishes_all_owner_leases(self):
        _, owners, request, allocation = coordinated()
        provider = MultiRankTargetProvider(owners, parallel_owner_calls=True)
        barrier = Barrier(len(owners), timeout=5)

        def synchronized_acquire(acquire):
            def acquire_after_barrier(request, allocation):
                barrier.wait()
                return acquire(request, allocation)

            return acquire_after_barrier

        for owner in owners:
            owner.acquire_target = synchronized_acquire(owner.acquire_target)

        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        self.assertEqual(tuple(target.lease for target in manifest.targets), tuple(o.retains[0] for o in owners))
        provider.release_target(lease)

    def test_parallel_acquire_records_other_leases_after_an_unresolved_rank(self):
        _, owners, request, allocation = coordinated()
        provider = MultiRankTargetProvider(owners, parallel_owner_calls=True)

        def unresolved(*args):
            raise TimeoutError("RPC reply not observed")

        owners[0].acquire_target = unresolved
        with self.assertRaisesRegex(ContractError, "remain retained"):
            provider.acquire_target(request, allocation)
        target = provider._targets[request]
        self.assertIsNone(target.manifest)
        self.assertEqual(target.acquired, {1: owners[1].retains[0]})
        self.assertEqual(owners[1].releases, [])

    def test_partial_release_does_not_ack_or_lose_unreleased_rank(self):
        provider, owners, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        owners[1].fail_release = True
        with self.assertRaisesRegex(ContractError, "stay retained"):
            provider.release_target(lease)
        self.assertIn(request, provider._targets)
        self.assertEqual(provider._targets[request].released, {0})
        with self.assertRaises(ContractError):
            provider.manifest(lease)
        owners[1].fail_release = False
        provider.release_target(lease)
        self.assertEqual([len(owner.releases) for owner in owners], [1, 1])

    def test_parallel_release_retries_only_unresolved_ranks(self):
        _, owners, request, allocation = coordinated()
        provider = MultiRankTargetProvider(owners, parallel_owner_calls=True)
        lease = provider.acquire_target(request, allocation)
        owners[0].fail_release = True

        with self.assertRaisesRegex(ContractError, "stay retained"):
            provider.release_target(lease)
        self.assertEqual(provider._targets[request].released, {1})
        with self.assertRaisesRegex(ContractError, "active, partial or uncertain"):
            provider.assert_can_close()

        owners[0].fail_release = False
        provider.release_target(lease)
        self.assertEqual([len(owner.releases) for owner in owners], [1, 1])
        provider.assert_can_close()

    def test_shared_main_identity_and_offsets_must_match_before_publish(self):
        for changed in ("backing", "offset", "entry", "consumer"):
            provider, owners, request, allocation = coordinated()
            owner = owners[1]
            catalog = owner.descriptor
            if changed == "backing":
                regions = (replace(catalog.regions[0], backing_id="different-host"), *catalog.regions[1:])
                catalog = replace(catalog, regions=regions)
            elif changed == "offset":
                entry = catalog.entries[0]
                entry = replace(
                    entry, binding=replace(entry.binding, view=replace(entry.binding.view, region_offset_bytes=64))
                )
                catalog = replace(catalog, entries=(entry, *catalog.entries[1:]))
            elif changed == "entry":
                catalog = replace(catalog, entries=catalog.entries[:-1])
            else:
                catalog = replace(catalog, rank=0)
            owner.descriptor = catalog
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                provider.acquire_target(request, allocation)

    def test_changed_manifest_aggregate_or_lease_fails_validation(self):
        provider, _, request, allocation = coordinated()
        lease = provider.acquire_target(request, allocation)
        manifest = provider.manifest(lease)
        with self.assertRaises(ContractError):
            replace(manifest, lease=replace(lease, views=lease.views[:-1]))
        with self.assertRaises(ContractError):
            replace(manifest, targets=tuple(reversed(manifest.targets)))
        with self.assertRaises(ContractError):
            manifest.for_rank(True)
        with self.assertRaises(ContractError):
            provider.get_target_views(replace(lease, allocation=replace(allocation, allocation_generation=10)))
        provider.release_target(lease)

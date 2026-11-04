# SPDX-License-Identifier: Apache-2.0
"""Rank/entry coordination on explicit descriptors, not physical memory tests."""

import unittest
from copy import deepcopy
from dataclasses import replace

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
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import (
    RegisteredTargetProvider,
    TargetBinding,
)


class DescriptorOwner:
    """Real provider state machine over declared, non-allocated test geometry."""

    def __init__(self, rank, tp_size, layers=2):
        self.rank = rank
        self.retains, self.prepares, self.releases = [], [], []
        self.fail_release = False
        host = RegisteredRegion(
            f"owner-{rank}", 1, f"host-{rank}", "shared-host", 3, 0, layers * 512, "host", f"node:{10000 + rank}"
        )
        regions = [host]
        entries = []
        for layer in range(layers):
            for component in Component:
                for entry in range(2):
                    if component is Component.MAIN:
                        region = host
                        dtype, shape, strides, page_bytes = "bfloat16", (4, 16, 2), (64, 4, 2), 64
                        offset, page_tokens = layer * 512 + entry * 256, 16
                    else:
                        stride = 32 if entry == 0 else 16
                        region = RegisteredRegion(
                            f"owner-{rank}",
                            1,
                            f"index-{rank}-{layer}-{entry}",
                            f"device-{rank}-{layer}-{entry}",
                            1,
                            0,
                            8 * stride,
                            "device",
                            f"node:{10000 + rank}",
                        )
                        regions.append(region)
                        dtype = "int8" if entry == 0 else "float16"
                        shape = (8, 8, 4 if entry == 0 else 1)
                        strides = (stride, 4, 1) if entry == 0 else (stride, 2, 2)
                        page_bytes, offset, page_tokens = stride, 0, 8
                    view = CacheView(
                        component, layer, region.region_id, dtype, (strides[0],), page_tokens, 16, 4, offset, ()
                    )
                    binding = TargetBinding(view, 0, page_bytes)
                    entries.append(
                        TargetEntry(
                            f"layer.{layer}.{component.value}", component, layer, 0, entry, shape, strides, binding
                        )
                    )
        plan = WriterPlan({"shared-host": (0,)}, {"shared-host": (TokenRange(0, 64),)})
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

    def test_tp16_model_sized_manifest_round_trip_fits_transport_limits(self):
        provider, _, request, allocation = coordinated(tp=16, layers=78)
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

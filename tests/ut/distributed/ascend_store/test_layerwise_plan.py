# SPDX-License-Identifier: Apache-2.0
"""Run planned writes against actual CPU storage, including page/tail bounds."""

import ctypes
import math
import unittest
from dataclasses import replace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_coordinator import coordinated
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import Component, ContractError
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import (
    ManifestWritePlan,
    RegionAddress,
    manifest_requirements,
    plan_layer_writes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import (
    LayerSource,
    RequestSource,
    SlotEpoch,
    SourceView,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import encode_owner_value


class MemoryFixture:
    def __init__(self, *, layers=2):
        self.provider, _, request, allocation = coordinated(layers=layers)
        self.manifest = self.provider.manifest(self.provider.acquire_target(request, allocation))
        self.destinations = {}
        self.handles = []
        for target in self.manifest.targets:
            for region in target.lease.regions:
                raw = self.destinations.setdefault(region.backing_id, ctypes.create_string_buffer(region.length_bytes))
                ctypes.memset(ctypes.addressof(raw), 0xA5, len(raw))
                self.handles.append(
                    RegionAddress(
                        request,
                        allocation.allocation_generation,
                        allocation.retention_token,
                        region,
                        ctypes.addressof(raw),
                    )
                )
        self.sources = {}

    def source(self, rank, *, start=0, end=23, padding=0):
        views = []
        for entry in self.manifest.targets[rank].catalog.entries:
            if entry.logical_layer != 0:
                continue
            page_tokens = 4 if entry.component is Component.INDEXER else 16
            dtype_bytes = 1 if entry.binding.view.dtype == "int8" else 2
            shape = (4 * 16 // page_tokens, page_tokens, *entry.shape[2:])
            page_bytes = math.prod(shape[1:]) * dtype_bytes
            stride = page_bytes + padding
            length = (shape[0] - 1) * stride + page_bytes
            raw = ctypes.create_string_buffer(length)
            identity = (rank, entry.component, entry.entry)
            self.sources[identity] = raw
            token_bytes = page_bytes // page_tokens
            for token in range(32):
                block, offset = divmod(token, 16)
                physical = (2, 0)[block]
                page, within = divmod(offset, page_tokens)
                address = (
                    ctypes.addressof(raw) + (physical * (16 // page_tokens) + page) * stride + within * token_bytes
                )
                ctypes.memset(address, rank * 40 + entry.entry * 5 + token + 1, token_bytes)
            inner_strides = tuple(math.prod(shape[i + 1 :]) * dtype_bytes for i in range(1, len(shape)))
            views.append(
                SourceView(
                    0,
                    entry.component.value.upper(),
                    1 if entry.component is Component.MAIN else 0,
                    f"slot-{entry.component.value}-{entry.entry}",
                    ctypes.addressof(raw),
                    length,
                    shape,
                    (stride, *inner_strides),
                    entry.binding.view.dtype,
                    16,
                    page_tokens,
                    entry.entry,
                )
            )
        epochs = tuple(SlotEpoch(f"worker-{rank}", v.slot, 1, 0) for v in views)
        source = LayerSource(0, epochs, tuple(views))
        # Store-private identity is intentionally unrelated to the D lease.
        request = RequestSource("internal-source", "private-store-instance", start, end, ((2, 0), (2, 0)))
        return source, request

    def apply(self, plans):
        for plan in plans:
            for src, dst, length in zip(plan.source_addresses, plan.target_addresses, plan.lengths):
                ctypes.memmove(dst, src, length)

    def verify(self, start, end):
        for target in self.manifest.targets:
            for entry in target.catalog.entries:
                view = entry.binding.view
                region = next(r for r in target.lease.regions if r.region_id == view.region_binding)
                base = ctypes.addressof(self.destinations[region.backing_id]) + view.region_offset_bytes
                token_bytes = entry.binding.page_bytes // view.page_size_tokens
                writer = 0 if entry.component is Component.MAIN else target.rank
                expected = bytearray(
                    [0xA5] * ((entry.shape[0] - 1) * entry.strides_bytes[0] + entry.binding.page_bytes)
                )
                if entry.logical_layer == 0:
                    for token in range(start, end):
                        ordinal, offset = divmod(token, 16)
                        physical = (1, 3)[ordinal]
                        page, within = divmod(offset, view.page_size_tokens)
                        at = (physical * (16 // view.page_size_tokens) + page) * view.strides_bytes[
                            0
                        ] + within * token_bytes
                        expected[at : at + token_bytes] = (
                            bytes([writer * 40 + entry.entry * 5 + token + 1]) * token_bytes
                        )
                self_data = ctypes.string_at(base, len(expected))
                if self_data != bytes(expected):
                    raise AssertionError((target.rank, entry.component, entry.logical_layer, entry.entry))


class TestLayerwiseTransferPlan(unittest.TestCase):
    def test_compiled_plan_indexes_each_rank_layer_and_receiver_once(self):
        fixture = MemoryFixture()
        plan = ManifestWritePlan(fixture.manifest)
        for rank in range(2):
            source, request = fixture.source(rank)
            direct = plan_layer_writes(
                source, request, fixture.manifest, producer_rank=rank, addresses=tuple(fixture.handles)
            )
            compiled = plan_layer_writes(source, request, plan, producer_rank=rank, addresses=tuple(fixture.handles))
            self.assertEqual(compiled, direct)
            self.assertEqual(len(plan.receive_keys(rank)), 8)
        self.assertEqual(plan.for_layer(0, 99), ())
        with self.assertRaises(ContractError):
            plan.receive_keys(3)

    def test_all_entry_receipt_requirements_are_explicit_not_cartesian_backings(self):
        fixture = MemoryFixture()
        requirements = manifest_requirements(fixture.manifest)
        self.assertEqual(len(requirements), 12)
        self.assertEqual(len({r.key for r in requirements}), 12)
        for requirement in requirements:
            if requirement.key.component is Component.MAIN:
                self.assertEqual(requirement.key.rank, 0)
                self.assertEqual(requirement.receiver_ranks, (0, 1))
            else:
                self.assertEqual(requirement.receiver_ranks, (requirement.owner_rank,))
                self.assertEqual(requirement.key.target_scope.consumer_rank, requirement.owner_rank)

    def test_native_cpu_memmove_matches_logical_tokens_and_preserves_padding_and_other_layers(self):
        fixture = MemoryFixture()
        counts = []
        for rank in range(2):
            source, request = fixture.source(rank, padding=16)
            plans = plan_layer_writes(
                source, request, fixture.manifest, producer_rank=rank, addresses=tuple(fixture.handles)
            )
            counts.append(len(plans))
            self.assertTrue(all(plan.key.request == fixture.manifest.lease.request for plan in plans))
            fixture.apply(plans)
        self.assertEqual(counts, [4, 2])
        fixture.verify(0, 23)

    def test_unaligned_subrange_copies_only_required_tail_bytes(self):
        fixture = MemoryFixture()
        for rank in range(2):
            source, request = fixture.source(rank, start=17, end=23)
            fixture.apply(
                plan_layer_writes(
                    source, request, fixture.manifest, producer_rank=rank, addresses=tuple(fixture.handles)
                )
            )
        fixture.verify(17, 23)

    def test_empty_range_submits_no_bytes(self):
        fixture = MemoryFixture()
        source, request = fixture.source(0, start=0, end=0)
        plans = plan_layer_writes(source, request, fixture.manifest, producer_rank=0, addresses=tuple(fixture.handles))
        self.assertTrue(plans)
        self.assertTrue(all(not plan.lengths for plan in plans))
        fixture.verify(0, 0)

    def test_stale_or_changed_region_handles_are_rejected(self):
        fixture = MemoryFixture()
        source, request = fixture.source(0)
        for changed in (
            replace(fixture.handles[0], retention_token="retired-token"),
            replace(fixture.handles[0], allocation_generation=10),
            replace(fixture.handles[0], region=replace(fixture.handles[0].region, registration_generation=2)),
        ):
            with self.subTest(handle=changed), self.assertRaises(ContractError):
                plan_layer_writes(
                    source, request, fixture.manifest, producer_rank=0, addresses=(changed, *fixture.handles[1:])
                )

    def test_source_dtype_capacity_stride_and_missing_entry_fail_before_submission(self):
        fixture = MemoryFixture()
        source, request = fixture.source(0)
        first = source.views[0]
        for invalid in (
            replace(source, views=(replace(first, dtype="float16"), *source.views[1:])),
            replace(source, views=source.views[1:]),
            replace(source, views=(replace(first, length_bytes=16), *source.views[1:])),
            replace(source, views=(replace(first, group=5), *source.views[1:])),
        ):
            with self.subTest(source=invalid), self.assertRaises(ContractError):
                plan_layer_writes(invalid, request, fixture.manifest, producer_rank=0, addresses=tuple(fixture.handles))
        with self.assertRaises(ContractError):
            plan_layer_writes(
                source,
                replace(request, block_ids_by_group=((-1, 0), (-1, 0))),
                fixture.manifest,
                producer_rank=0,
                addresses=tuple(fixture.handles),
            )

    def test_address_handles_are_never_scheduler_owner_ipc_values(self):
        fixture = MemoryFixture()
        with self.assertRaises(ContractError):
            encode_owner_value(fixture.handles[0])

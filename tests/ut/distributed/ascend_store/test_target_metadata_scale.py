# SPDX-License-Identifier: Apache-2.0
"""Real-size TP/layer/block-map descriptors; no device storage or DMA."""

import json
import unittest
from dataclasses import replace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_coordinator import DescriptorOwner
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    RequestKey,
    TokenRange,
    WriterPlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import (
    target_lease_from_wire,
    target_lease_to_wire,
    target_manifest_from_wire,
    target_manifest_to_wire,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_coordinator import MultiRankTargetProvider
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import (
    MAX_OWNER_FRAME_BYTES,
    decode_owner_value,
    encode_owner_value,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import RegisteredTargetProvider


def large_manifest(prompt, *, tp=16, layers=78, capacity=2048, manager_tokens=128):
    """Resize the descriptor fixture, retaining explicit per-entry geometry."""
    factor = capacity * manager_tokens // (4 * 16)
    owners = []
    for rank in range(tp):
        owner = DescriptorOwner(rank, tp, layers)
        old = owner.descriptor
        regions = tuple(replace(r, length_bytes=r.length_bytes * factor) for r in old.regions)
        entries = []
        for entry in old.entries:
            view = entry.binding.view
            view = replace(
                view,
                manager_block_size_tokens=manager_tokens,
                block_capacity=capacity,
                region_offset_bytes=view.region_offset_bytes * factor,
            )
            shape = (capacity * manager_tokens // view.page_size_tokens, *entry.shape[1:])
            entries.append(replace(entry, shape=shape, binding=replace(entry.binding, view=view)))
        writers = WriterPlan({"shared-host": (0,)}, {"shared-host": (TokenRange(0, capacity * manager_tokens),)})
        owner.descriptor = replace(old, regions=regions, entries=tuple(entries), writer_plan=writers)
        owner.provider = RegisteredTargetProvider(
            rank=rank,
            tp_size=tp,
            regions=regions,
            bindings=tuple(e.binding for e in entries),
            required_layers=old.required_layers,
            writer_plan=writers,
            runtime=owner,
        )
        owners.append(owner)
    provider = MultiRankTargetProvider(tuple(owners))
    request = RequestKey(f"large-metadata-{prompt}", 0)
    allocation = BlockAllocation(
        request,
        0,
        (tuple(range((prompt + manager_tokens - 1) // manager_tokens)),),
        manager_tokens,
        prompt,
        "metadata-only-no-native",
    )
    lease = provider.acquire_target(request, allocation)
    return provider, provider.manifest(lease)


class TestTargetMetadataScale(unittest.TestCase):
    def test_tp16_layers78_32k_64k_128k_fit_unchanged_limits_and_both_wire_round_trips(self):
        sizes = []
        for prompt in (32768, 65536, 131072):
            with self.subTest(prompt=prompt):
                provider, manifest = large_manifest(prompt)
                try:
                    frame = encode_owner_value(manifest)
                    self.assertLess(len(frame), MAX_OWNER_FRAME_BYTES)
                    self.assertEqual(decode_owner_value(frame), manifest)
                    wire = target_manifest_to_wire(manifest)
                    self.assertEqual(target_manifest_from_wire(json.loads(json.dumps(wire))), manifest)
                    lease_wire = target_lease_to_wire(manifest.lease)
                    self.assertEqual(target_lease_from_wire(json.loads(json.dumps(lease_wire))), manifest.lease)
                    self.assertTrue(all(type(view["block_ids"]) is dict for view in lease_wire["views"]))
                    sizes.append(len(frame))
                finally:
                    provider.release_target(manifest.lease)
        # Increasing only prompt blocks must not multiply every entry's map.
        self.assertLess(sizes[-1] - sizes[0], 65536)

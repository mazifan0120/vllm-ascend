# SPDX-License-Identifier: Apache-2.0
"""Rank-explicit transfer targets over the existing registered owner leases.

The public aggregate has one canonical registration of shared Host Main and
every rank's device Indexer. Owners keep their original leases: an aggregate
must never be passed to a local owner as if that owner registered other ranks'
device allocations. No address, backing ID or registration is invented here.
"""

import math
from dataclasses import dataclass, replace

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    Component,
    ContractError,
    TargetLease,
    TokenRange,
    WriterPlan,
    assert_lease_consistent,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import TargetCatalog, TargetEntry

CANONICAL_MAIN_RANK = 0
TARGET_DTYPE_BYTES = (("int8", 1), ("float16", 2), ("bfloat16", 2), ("float32", 4))


@dataclass(frozen=True)
class RankTarget:
    catalog: TargetCatalog
    lease: TargetLease

    @property
    def rank(self) -> int:
        return self.catalog.rank


def _validate_entry_geometry(entry, view, region, block_ids, prompt_tokens) -> None:
    itemsize = next((size for name, size in TARGET_DTYPE_BYTES if view.dtype.removeprefix("torch.") == name), None)
    if (
        itemsize is None
        or not entry.layer_name
        or type(entry.entry) is not int
        or entry.entry < 0
        or type(entry.group) is not int
        or entry.group < 0
        or len(entry.shape) < 2
        or len(entry.strides_bytes) != len(entry.shape)
        or any(type(v) is not int or v < 1 for v in (*entry.shape, *entry.strides_bytes))
        or view.block_capacity < 1
        or not view.strides_bytes
        or entry.strides_bytes[0] != view.strides_bytes[0]
    ):
        raise ContractError("Invalid target entry type, shape, dtype or stride")
    pages_per_block = view.manager_block_size_tokens // view.page_size_tokens
    page_bytes = math.prod(entry.shape[1:]) * itemsize
    end = view.region_offset_bytes + (entry.shape[0] - 1) * entry.strides_bytes[0] + page_bytes
    if (
        entry.shape[0] != view.block_capacity * pages_per_block
        or entry.binding.page_bytes != page_bytes
        or entry.strides_bytes[0] < page_bytes
        or end > region.length_bytes
        or any(entry.strides_bytes[i] != math.prod(entry.shape[i + 1 :]) * itemsize for i in range(1, len(entry.shape)))
        or len(block_ids) * view.manager_block_size_tokens < prompt_tokens
        or len(set(block_ids)) != len(block_ids)
        or any(type(block) is not int or not 0 <= block < view.block_capacity for block in block_ids)
    ):
        raise ContractError("Target entry footprint or request blocks exceed registered capacity")
    if view.component is Component.MAIN and view.dtype.removeprefix("torch.") != "bfloat16":
        raise ContractError("Shared Host Main target requires BF16")


def _validate_rank(target: RankTarget, tp_size: int) -> None:
    if (
        type(target) is not RankTarget
        or type(target.catalog) is not TargetCatalog
        or type(target.lease) is not TargetLease
    ):
        raise ContractError("Manifest requires an explicit catalog and lease for every rank")
    catalog, lease = target.catalog, target.lease
    if (
        type(catalog.rank) is not int
        or not 0 <= catalog.rank < tp_size
        or type(catalog.tp_size) is not int
        or catalog.tp_size != tp_size
        or type(catalog.contract_version) is not int
        or catalog.contract_version != CONTRACT_VERSION
        or lease.rank != catalog.rank
        or lease.regions != catalog.regions
        or lease.writer_plan != catalog.writer_plan
    ):
        raise ContractError("Manifest owner rank, registration or protocol mismatch")
    assert_lease_consistent(lease, lease.request)
    if not lease.allocation.retention_token:
        raise ContractError("Manifest targets must retain the actual scheduler allocation")
    expected = {}
    layers = {Component.MAIN: set(), Component.INDEXER: set()}
    regions = {region.region_id: region for region in catalog.regions}
    entry_ids = set()
    for entry in catalog.entries:
        if type(entry) is not TargetEntry or entry.component not in layers:
            raise ContractError("Manifest catalog contains an unsupported component")
        view = entry.binding.view
        identity = (entry.component, entry.logical_layer, entry.entry)
        if identity in entry_ids:
            raise ContractError("Manifest catalog repeats a component/layer/entry")
        entry_ids.add(identity)
        if (
            entry.component is not view.component
            or entry.logical_layer != view.logical_layer
            or entry.group != entry.binding.group
            or not 0 <= entry.group < len(lease.allocation.group_block_ids)
            or view.region_binding not in regions
            or view.block_ids
            or view.manager_block_size_tokens != lease.allocation.manager_block_size_tokens
        ):
            raise ContractError("Manifest entry does not match its registered view or allocation group")
        kind = "host" if entry.component is Component.MAIN else "device"
        if regions[view.region_binding].memory_kind != kind:
            raise ContractError("Manifest component binds the wrong physical memory kind")
        _validate_entry_geometry(
            entry,
            view,
            regions[view.region_binding],
            lease.allocation.group_block_ids[entry.group],
            lease.allocation.prompt_tokens,
        )
        expected_view = replace(view, block_ids=lease.allocation.group_block_ids[entry.group])
        if expected_view in expected:
            raise ContractError("Manifest catalog ambiguously aliases two entries")
        expected[expected_view] = entry
        layers[entry.component].add(entry.logical_layer)
    if (
        not layers[Component.MAIN]
        or not layers[Component.INDEXER]
        or {c: tuple(sorted(v)) for c, v in layers.items()} != catalog.required_layers
        or len(lease.views) != len(expected)
        or set(lease.views) != set(expected)
    ):
        raise ContractError("Manifest lease does not cover the complete registered catalog")


def _component_geometry(target: RankTarget, component: Component):
    regions = {r.region_id: r for r in target.catalog.regions}
    result = {}
    for entry in target.catalog.entries:
        if entry.component is not component:
            continue
        view = entry.binding.view
        region = regions[view.region_binding]
        location = (
            (
                region.backing_id,
                region.backing_generation,
                region.backing_offset_bytes + view.region_offset_bytes,
            )
            if component is Component.MAIN
            else ()
        )
        result[(entry.logical_layer, entry.entry)] = (
            entry.layer_name,
            entry.group,
            entry.shape,
            entry.strides_bytes,
            view.dtype,
            view.strides_bytes,
            view.page_size_tokens,
            view.manager_block_size_tokens,
            view.block_capacity,
            entry.binding.page_bytes,
            location,
        )
    return result


def aggregate_rank_targets(targets: tuple[RankTarget, ...]) -> TargetLease:
    """Validate all owner facts before publishing any P-side target."""
    if type(targets) is not tuple or not targets:
        raise ContractError("Manifest requires all TP owner targets")
    tp_size = len(targets)
    for target in targets:
        _validate_rank(target, tp_size)
    if tuple(target.rank for target in targets) != tuple(range(tp_size)):
        raise ContractError("Manifest must contain every TP rank exactly once in rank order")
    canonical = targets[CANONICAL_MAIN_RANK]
    main_geometry = _component_geometry(canonical, Component.MAIN)
    indexer_geometry = _component_geometry(canonical, Component.INDEXER)
    request, allocation = canonical.lease.request, canonical.lease.allocation
    main_backings = {r.backing_id for r in canonical.catalog.regions if r.memory_kind == "host"}
    for target in targets:
        if target.lease.request != request or target.lease.allocation != allocation:
            raise ContractError("TP owners acquired different request/allocation generations")
        if _component_geometry(target, Component.MAIN) != main_geometry:
            raise ContractError("TP owners do not share the same physical Main layout")
        if _component_geometry(target, Component.INDEXER) != indexer_geometry:
            raise ContractError("TP Indexer layer/entry geometry is inconsistent")
        if set(target.catalog.writer_plan.writers_by_backing) != main_backings or any(
            ranks != (CANONICAL_MAIN_RANK,) for ranks in target.catalog.writer_plan.writers_by_backing.values()
        ):
            raise ContractError("Manifest requires one canonical writer for each shared Main backing")
    selected_views, selected_regions = [], []
    seen_regions, seen_backings = set(), set()
    for target in targets:
        views = tuple(
            v for v in target.lease.views if v.component is Component.INDEXER or target.rank == CANONICAL_MAIN_RANK
        )
        selected_views.extend(views)
        needed_regions = {v.region_binding for v in views}
        for region in target.lease.regions:
            if region.region_id not in needed_regions:
                continue
            backing = (region.backing_id, region.backing_generation)
            if region.region_id in seen_regions or backing in seen_backings:
                raise ContractError("Manifest repeats a region or aliases device backings across owners")
            seen_regions.add(region.region_id)
            seen_backings.add(backing)
            selected_regions.append(region)
    aggregate = TargetLease(
        CONTRACT_VERSION,
        request,
        CANONICAL_MAIN_RANK,
        allocation,
        tuple(selected_regions),
        tuple(selected_views),
        WriterPlan(
            {backing: (CANONICAL_MAIN_RANK,) for backing in sorted(main_backings)},
            {backing: (TokenRange(0, allocation.prompt_tokens),) for backing in sorted(main_backings)},
        ),
    )
    assert_lease_consistent(aggregate, request)
    return aggregate


@dataclass(frozen=True)
class TargetManifest:
    lease: TargetLease
    targets: tuple[RankTarget, ...]

    def __post_init__(self) -> None:
        if aggregate_rank_targets(self.targets) != self.lease:
            raise ContractError("Manifest aggregate differs from the actual TP owner targets")

    def for_rank(self, rank: int) -> RankTarget:
        if type(rank) is not int or not 0 <= rank < len(self.targets):
            raise ContractError("Manifest consumer rank is out of range")
        return self.targets[rank]

    def transfer_entries(self) -> tuple[tuple[int, TargetEntry], ...]:
        """Explicit consumer ownership; Main has one canonical registration."""
        return tuple(
            (target.rank, entry)
            for target in self.targets
            for entry in target.catalog.entries
            if entry.component is Component.INDEXER or target.rank == CANONICAL_MAIN_RANK
        )

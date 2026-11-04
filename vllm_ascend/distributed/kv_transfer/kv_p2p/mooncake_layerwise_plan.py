# SPDX-License-Identifier: Apache-2.0
"""Manifest-bound layerwise writes using the existing DSA page mapper.

Address handles belong only to the peer data plane. They are deliberately not
owner-IPC values: the scheduler carries registered identities and leases, never
worker-local addresses. Every write binds an actual request, entry and region.
"""

import math
from dataclasses import dataclass, replace
from types import MappingProxyType

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import DsaCacheLayout, build_component_read
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    Component,
    ContractError,
    RegisteredRegion,
    RequestKey,
    TargetScope,
    TokenRange,
    TransferKey,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import LayerSource, RequestSource, SourceView
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import TargetEntry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import (
    CANONICAL_MAIN_RANK,
    TARGET_DTYPE_BYTES,
    TargetManifest,
)


@dataclass(frozen=True)
class RegionAddress:
    request: RequestKey
    allocation_generation: int
    retention_token: str
    region: RegisteredRegion
    base_address: int

    def __post_init__(self) -> None:
        if (
            type(self.request) is not RequestKey
            or type(self.allocation_generation) is not int
            or self.allocation_generation < 0
            or type(self.retention_token) is not str
            or not self.retention_token
            or type(self.region) is not RegisteredRegion
            or type(self.base_address) is not int
            or self.base_address <= 0
        ):
            raise ContractError("Invalid registered peer data-plane address handle")


@dataclass(frozen=True)
class TransferRequirement:
    key: TransferKey
    owner_rank: int
    entry: TargetEntry
    region: RegisteredRegion
    receiver_ranks: tuple[int, ...]


@dataclass(frozen=True)
class PlannedWrite:
    key: TransferKey
    endpoint: str
    region_id: str
    memory_kind: str
    source_addresses: tuple[int, ...]
    target_addresses: tuple[int, ...]
    lengths: tuple[int, ...]
    receiver_ranks: tuple[int, ...]


def manifest_requirements(manifest: TargetManifest) -> tuple[TransferRequirement, ...]:
    requirements = []
    for operation_id, (rank, entry) in enumerate(manifest.transfer_entries()):
        lease = manifest.for_rank(rank).lease
        region = next(r for r in lease.regions if r.region_id == entry.binding.view.region_binding)
        writer = CANONICAL_MAIN_RANK if entry.component is Component.MAIN else rank
        scope = TargetScope(
            region.backing_id, region.backing_generation, None if entry.component is Component.MAIN else rank
        )
        key = TransferKey(
            manifest.lease.request,
            writer,
            entry.logical_layer,
            operation_id,
            entry.component,
            TokenRange(0, lease.allocation.prompt_tokens),
            scope,
        )
        receivers = tuple(range(len(manifest.targets))) if entry.component is Component.MAIN else (rank,)
        requirements.append(TransferRequirement(key, rank, entry, region, receivers))
    return tuple(requirements)


class ManifestWritePlan:
    """Compile the admitted request once, outside the layer/attention hot path."""

    def __init__(self, manifest: TargetManifest) -> None:
        if type(manifest) is not TargetManifest:
            raise ContractError("Write plan requires a validated rank-explicit target manifest")
        self.manifest = manifest
        self.requirements = manifest_requirements(manifest)
        by_layer = {}
        by_receiver = {rank: [] for rank in range(len(manifest.targets))}
        for requirement in self.requirements:
            identity = (requirement.key.rank, requirement.key.logical_layer)
            by_layer.setdefault(identity, []).append(requirement)
            for rank in requirement.receiver_ranks:
                by_receiver[rank].append(requirement.key)
        self._by_layer = MappingProxyType({key: tuple(values) for key, values in by_layer.items()})
        self._by_receiver = MappingProxyType({rank: tuple(keys) for rank, keys in by_receiver.items()})

    def for_layer(self, producer_rank: int, layer: int) -> tuple[TransferRequirement, ...]:
        return self._by_layer.get((producer_rank, layer), ())

    def receive_keys(self, rank: int) -> tuple[TransferKey, ...]:
        if type(rank) is not int or rank not in self._by_receiver:
            raise ContractError("Receiver rank is outside the admitted target plan")
        return self._by_receiver[rank]


def _source_layout(view: SourceView) -> DsaCacheLayout:
    dtype = view.dtype.removeprefix("torch.")
    itemsize = next((size for name, size in TARGET_DTYPE_BYTES if name == dtype), None)
    if (
        itemsize is None
        or type(view.address) is not int
        or view.address <= 0
        or type(view.manager_block_tokens) is not int
        or view.manager_block_tokens < 1
        or type(view.page_tokens) is not int
        or view.page_tokens < 1
        or view.manager_block_tokens % view.page_tokens
        or len(view.shape) < 2
        or len(view.shape) != len(view.strides_bytes)
        or any(type(size) is not int or size < 1 for size in (*view.shape, *view.strides_bytes))
        or view.shape[1] != view.page_tokens
    ):
        raise ContractError("Source view is not an explicit token-contiguous component page")
    scale = view.manager_block_tokens // view.page_tokens
    page_bytes = math.prod(view.shape[1:]) * itemsize
    if (
        view.shape[0] % scale
        or view.strides_bytes[0] < page_bytes
        or any(view.strides_bytes[i] != math.prod(view.shape[i + 1 :]) * itemsize for i in range(1, len(view.shape)))
        or (view.shape[0] - 1) * view.strides_bytes[0] + page_bytes > view.length_bytes
    ):
        raise ContractError("Source page footprint exceeds its actual view")
    return DsaCacheLayout(
        str(view.layer),
        view.entry,
        view.address,
        page_bytes,
        view.strides_bytes[0],
        scale,
        view.manager_block_tokens,
        dtype,
        view.shape[0] // scale,
    )


def plan_layer_writes(
    source: LayerSource,
    request: RequestSource,
    manifest: TargetManifest | ManifestWritePlan,
    *,
    producer_rank: int,
    addresses: tuple[RegionAddress, ...],
) -> tuple[PlannedWrite, ...]:
    prepared = manifest if type(manifest) is ManifestWritePlan else ManifestWritePlan(manifest)
    manifest = prepared.manifest
    if type(producer_rank) is not int or not 0 <= producer_rank < len(manifest.targets):
        raise ContractError("Producer rank is outside the target TP manifest")
    end = min(request.end_token, manifest.lease.allocation.prompt_tokens)
    if request.start_token < 0 or end < request.start_token:
        raise ContractError("Source transfer range is outside the target prompt")
    span = TokenRange(request.start_token, end)
    handles = {}
    for handle in addresses:
        if type(handle) is not RegionAddress or handle.region.region_id in handles:
            raise ContractError("Duplicate or malformed peer region handle")
        if (
            handle.request != manifest.lease.request
            or handle.allocation_generation != manifest.lease.allocation.allocation_generation
            or handle.retention_token != manifest.lease.allocation.retention_token
        ):
            raise ContractError("Peer address belongs to another retained request generation")
        handles[handle.region.region_id] = handle
    views = {}
    for view in source.views:
        identity = (view.component.lower(), view.layer, view.entry)
        if view.layer != source.layer or identity in views:
            raise ContractError("Source layer repeats or misidentifies a component entry")
        views[identity] = view
    plans = []
    mapping_cache = {}
    for requirement in prepared.for_layer(producer_rank, source.layer):
        identity = (requirement.key.component.value, source.layer, requirement.entry.entry)
        view = views.get(identity)
        if view is None or not 0 <= view.group < len(request.block_ids_by_group):
            raise ContractError("Producer source is missing an actual component/entry/block group")
        handle = handles.get(requirement.region.region_id)
        if handle is None or handle.region != requirement.region:
            raise ContractError("Required registered region has no matching peer address handle")
        local = _source_layout(view)
        source_ids = request.block_ids_by_group[view.group]
        if len(source_ids) * local.block_tokens < end or any(
            type(block) is not int or not 0 <= block < local.capacity for block in source_ids
        ):
            raise ContractError("Source request blocks exceed actual component capacity")
        target = requirement.entry.binding.view
        if local.block_tokens != target.manager_block_size_tokens:
            raise ContractError("P/D manager block token sizes differ")
        destination_ids = manifest.for_rank(requirement.owner_rank).lease.allocation.group_block_ids[
            requirement.entry.group
        ]
        remote = DsaCacheLayout(
            requirement.entry.layer_name,
            requirement.entry.entry,
            handle.base_address + target.region_offset_bytes,
            requirement.entry.binding.page_bytes,
            target.strides_bytes[0],
            target.manager_block_size_tokens // target.page_size_tokens,
            target.manager_block_size_tokens,
            target.dtype.removeprefix("torch."),
            target.block_capacity,
        )
        try:
            # The established helper describes a read into its 'local' side.
            # D is that destination; reverse its lists for a P-origin WRITE.
            destinations, sources, lengths = build_component_read(
                remote,
                local,
                source_ids,
                destination_ids,
                span.start,
                span.end,
                cp_size=1,
                cp_rank=0,
                writer_rank=0,
                writer_size=1,
                indexer=requirement.key.component is Component.INDEXER,
                mapping_cache=mapping_cache,
            )
        except ValueError as error:
            raise ContractError(f"Invalid component transfer geometry: {error}") from error
        for address, length in zip(destinations, lengths):
            if (
                not handle.base_address
                <= address
                < address + length
                <= handle.base_address + handle.region.length_bytes
            ):
                raise ContractError("Transfer exceeds the actual registered target region")
        plans.append(
            PlannedWrite(
                replace(requirement.key, token_range=span),
                handle.region.transport_endpoint,
                handle.region.region_id,
                handle.region.memory_kind,
                tuple(sources),
                tuple(destinations),
                tuple(lengths),
                requirement.receiver_ranks,
            )
        )
    return tuple(plans)

# SPDX-License-Identifier: Apache-2.0
"""Fixed layerwise offload descriptors and the common producer/receiver plan."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, NamedTuple

from .mooncake_dsa_transfer import DsaCacheLayout, build_component_read, layout_span_bytes

PROTOCOL_VERSION = 1
Component = Literal["main", "indexer"]


class Span(NamedTuple):
    start: int
    end: int


@dataclass(frozen=True)
class TargetId:
    request: str
    transfer_generation: int
    allocation_generation: int

    def __post_init__(self) -> None:
        if not self.request or min(self.transfer_generation, self.allocation_generation) < 0:
            raise ValueError("Invalid layerwise target identity")


@dataclass(frozen=True)
class TargetAllocation:
    target: TargetId
    prompt_tokens: int
    blocks: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or not self.blocks:
            raise ValueError("Layerwise target requires a nonempty prompt and block map")
        for group in self.blocks:
            if not group or min(group) < 0 or len(set(group)) != len(group):
                raise ValueError("A target group has missing, negative or repeated blocks")


@dataclass(frozen=True)
class CatalogRef:
    engine: str
    registration: int
    rank: int


@dataclass(frozen=True)
class RegisteredRegion:
    name: str
    backing: str
    backing_generation: int
    backing_offset: int
    length: int
    kind: Component


class EntryId(NamedTuple):
    layer: int
    component: Component
    index: int


@dataclass(frozen=True)
class Destination:
    entry: EntryId
    group: int
    region: str
    offset: int
    page_bytes: int
    page_stride: int
    page_tokens: int
    block_tokens: int
    capacity: int
    dtype: str

    def as_layout(self, region_base: int) -> DsaCacheLayout:
        return DsaCacheLayout(
            str(self.entry.layer),
            self.entry.index,
            region_base + self.offset,
            self.page_bytes,
            self.page_stride,
            self.block_tokens // self.page_tokens,
            self.block_tokens,
            self.dtype,
            self.capacity,
        )


@dataclass(frozen=True)
class RankCatalog:
    ref: CatalogRef
    endpoint: str
    regions: tuple[RegisteredRegion, ...]
    entries: tuple[Destination, ...]


def validate_catalog(catalog: RankCatalog) -> dict[EntryId, tuple[Destination, RegisteredRegion]]:
    if not catalog.ref.engine or not catalog.endpoint or min(catalog.ref.registration, catalog.ref.rank) < 0:
        raise ValueError("Invalid registration identity or transport endpoint")
    regions = {region.name: region for region in catalog.regions}
    if len(regions) != len(catalog.regions):
        raise ValueError("Duplicate registered region")
    for region in regions.values():
        if (
            not region.name
            or not region.backing
            or region.kind not in ("main", "indexer")
            or min(region.backing_generation, region.backing_offset) < 0
            or region.length <= 0
        ):
            raise ValueError("Invalid registered region geometry")
    bound = {}
    extents: dict[tuple[str, int], list[Span]] = {}
    for entry in catalog.entries:
        region = regions.get(entry.region)
        if (
            region is None
            or entry.entry.component != region.kind
            or entry.entry in bound
            or min(entry.group, entry.entry.layer, entry.entry.index, entry.offset) < 0
            or min(entry.page_tokens, entry.block_tokens, entry.page_bytes, entry.capacity) <= 0
            or entry.block_tokens % entry.page_tokens
            or entry.page_bytes % entry.page_tokens
            or not entry.dtype
        ):
            raise ValueError("Invalid component mapping in registered catalog")
        length = layout_span_bytes(entry.as_layout(0))
        if entry.offset + length > region.length:
            raise ValueError("Destination extends outside its registration")
        offset = region.backing_offset + entry.offset
        extents.setdefault((region.backing, region.backing_generation), []).append(Span(offset, offset + length))
        bound[entry.entry] = (entry, region)
    for spans in extents.values():
        if any(a.end > b.start for a, b in pairwise(sorted(spans))):
            raise ValueError("Destination entries alias the same physical bytes")
    if {key.component for key in bound} != {"main", "indexer"}:
        raise ValueError("Both Main and Indexer destinations are required")
    return bound


@dataclass(frozen=True)
class Route:
    writer: int
    receiver: int
    destination_rank: int
    writer_index: int
    writer_count: int
    entry: Destination

    def required(self, prompt: int) -> tuple[Span, ...]:
        if self.writer_count == 1:
            return (Span(0, prompt),)
        width = self.entry.block_tokens
        return tuple(
            Span(start, min(prompt, start + width))
            for start in range(self.writer_index * width, prompt, self.writer_count * width)
        )


class RegisteredLayout:
    """Static routes shared by P and D. Holds descriptors but no request-local views or addresses."""

    @staticmethod
    def validate_tp(producer_tp: int, decode_tp: int, *, replicated: bool) -> None:
        if decode_tp <= 0 or producer_tp < decode_tp or producer_tp % decode_tp:
            raise ValueError("Layerwise offload requires Prefill TP divisible by Decode TP")
        if producer_tp != decode_tp and not replicated:
            raise ValueError("Unequal layerwise TP requires replicated GLM DSA Main and Indexer KV")

    def __init__(self, catalogs: tuple[RankCatalog, ...], producer_tp: int, *, parallel_main: bool = False):
        decode_tp = len(catalogs)
        # The producer validates model replication before accepting the target.
        self.validate_tp(producer_tp, decode_tp, replicated=True)
        if tuple(catalog.ref.rank for catalog in catalogs) != tuple(range(decode_tp)):
            raise ValueError("Registration must cover every D rank in order")
        per_rank = tuple(validate_catalog(catalog) for catalog in catalogs)
        main_writers = tuple(range(decode_tp)) if parallel_main else (0,)
        indexer_backings = set()
        routes = []
        for rank, bound in enumerate(per_rank):
            if bound.keys() != per_rank[0].keys():
                raise ValueError("D ranks must expose identical layer/component entries")
            backing_ids = set()
            for key, (entry, region) in sorted(bound.items()):
                first, first_region = per_rank[0][key]
                shape = (
                    entry.group,
                    entry.page_bytes,
                    entry.page_stride,
                    entry.page_tokens,
                    entry.block_tokens,
                    entry.capacity,
                    entry.dtype,
                )
                reference = (
                    first.group,
                    first.page_bytes,
                    first.page_stride,
                    first.page_tokens,
                    first.block_tokens,
                    first.capacity,
                    first.dtype,
                )
                if shape != reference:
                    raise ValueError("D ranks disagree on destination geometry")
                if key.component == "main":
                    if (region.backing, region.backing_generation, region.backing_offset + entry.offset) != (
                        first_region.backing,
                        first_region.backing_generation,
                        first_region.backing_offset + first.offset,
                    ):
                        raise ValueError("D ranks disagree on the shared Main backing")
                    for index, writer in enumerate(main_writers):
                        routes.append(Route(writer, rank, writer, index, len(main_writers), per_rank[writer][key][0]))
                else:
                    backing_ids.add((region.backing, region.backing_generation))
                    routes.append(Route(rank, rank, rank, 0, 1, entry))
            if backing_ids & indexer_backings:
                raise ValueError("Indexer backings must be private to each D rank")
            indexer_backings.update(backing_ids)
        self.catalogs = catalogs
        self.routes = tuple(routes)
        # Indexer writer r feeds consumer r. Surplus producer ranks keep their
        # local Store work and never acquire remote addresses.
        self.writers = tuple(range(decode_tp))

    @property
    def refs(self) -> tuple[CatalogRef, ...]:
        return tuple(catalog.ref for catalog in self.catalogs)

    def validate_allocation(self, allocation: TargetAllocation) -> None:
        sizes = {(route.entry.group, route.entry.block_tokens, route.entry.capacity) for route in self.routes}
        for group, width, capacity in sizes:
            if group >= len(allocation.blocks):
                raise ValueError("Allocation is missing a destination group")
            blocks = allocation.blocks[group]
            if len(blocks) * width < allocation.prompt_tokens or max(blocks) >= capacity:
                raise ValueError("Allocation cannot hold the complete prompt")

    def receive_plan(self, rank: int, prompt: int) -> dict[tuple[int, EntryId], tuple[Span, ...]]:
        return {
            (route.writer, route.entry.entry): route.required(prompt) for route in self.routes if route.receiver == rank
        }

    def writer_routes(self, rank: int, layer: int) -> tuple[Route, ...]:
        return tuple(route for route in self.routes if route.writer == rank and route.entry.entry.layer == layer)


class PeerAddress(NamedTuple):
    catalog: CatalogRef
    region: str
    base: int
    length: int


class WriteResult(NamedTuple):
    writer: int
    entry: EntryId
    span: Span


@dataclass(frozen=True)
class Write:
    endpoint: str
    component: Component
    sources: list[int]
    targets: list[int]
    lengths: list[int]
    results: tuple[WriteResult, ...]
    receivers: tuple[int, ...]


def plan_writes(
    layout: RegisteredLayout,
    allocation: TargetAllocation,
    sources: Mapping[EntryId, tuple[DsaCacheLayout, tuple[int, ...]]],
    regions: Mapping[tuple[int, str], PeerAddress],
    writer: int,
    layer: int,
    span: Span,
) -> tuple[Write, ...]:
    if not 0 <= span.start <= span.end <= allocation.prompt_tokens:
        raise ValueError("Source chunk exceeds the admitted prompt")
    writes: dict[tuple[str, Component, tuple[int, ...]], Write] = {}
    mapping_cache = {}
    by_entry: dict[EntryId, list[Route]] = {}
    for route in layout.writer_routes(writer, layer):
        by_entry.setdefault(route.entry.entry, []).append(route)
    for key, routes in by_entry.items():
        route = routes[0]
        entry = route.entry
        catalog = layout.catalogs[route.destination_rank]
        address = regions[(route.destination_rank, entry.region)]
        registered = next(region for region in catalog.regions if region.name == entry.region)
        if address.catalog != catalog.ref or address.region != entry.region or address.length != registered.length:
            raise ValueError("Target address does not match the admitted registration")
        if address.base <= 0:
            raise ValueError("Nonpositive target registration address")
        source, source_blocks = sources[key]
        if source.base <= 0 or source.capacity <= 0 or any(not 0 <= block < source.capacity for block in source_blocks):
            raise ValueError("Source chunk references invalid local storage")
        dest = entry.as_layout(address.base)
        dst, src, lengths = build_component_read(
            dest,
            source,
            source_blocks,
            allocation.blocks[entry.group],
            span.start,
            span.end,
            cp_size=1,
            cp_rank=0,
            writer_rank=route.writer_index,
            writer_size=route.writer_count,
            indexer=key.component == "indexer",
            mapping_cache=mapping_cache,
        )
        results = tuple(
            WriteResult(writer, key, Span(max(expected.start, span.start), min(expected.end, span.end)))
            for expected in route.required(allocation.prompt_tokens)
            if max(expected.start, span.start) < min(expected.end, span.end)
        )
        expected_bytes = sum(result.span.end - result.span.start for result in results) * (
            entry.page_bytes // entry.page_tokens
        )
        if sum(lengths) != expected_bytes or any(
            not address.base <= ptr < ptr + length <= address.base + address.length
            for ptr, length in zip(dst, lengths, strict=True)
        ):
            raise ValueError("Native write plan does not match the admitted target ranges")
        if results:
            receivers = tuple(route.receiver for route in routes)
            batch_key = catalog.endpoint, key.component, receivers
            previous = writes.get(batch_key)
            if previous is None:
                writes[batch_key] = Write(catalog.endpoint, key.component, src, dst, lengths, results, receivers)
            else:
                previous.sources.extend(src)
                previous.targets.extend(dst)
                previous.lengths.extend(lengths)
                writes[batch_key] = replace(previous, results=previous.results + results)
    return tuple(writes.values())

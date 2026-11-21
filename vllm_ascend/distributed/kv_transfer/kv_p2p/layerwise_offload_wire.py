# SPDX-License-Identifier: Apache-2.0
"""Fixed value-only frames for rendezvous, directory exchange and receive proofs."""

from typing import Any

from .layerwise_offload_plan import (
    PROTOCOL_VERSION,
    CatalogRef,
    Destination,
    EntryId,
    PeerAddress,
    RankCatalog,
    RegisteredRegion,
    Span,
    TargetAllocation,
    TargetId,
    WriteResult,
)


def _row(value: Any, length: int | None = None) -> tuple:
    if type(value) not in (list, tuple) or (length is not None and len(value) != length):
        raise ValueError("Malformed layerwise protocol row")
    return tuple(value)


def _integer(value: Any, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid layerwise protocol integer")
    return value


def _string(value: Any) -> str:
    if type(value) is not str or not value:
        raise ValueError("Invalid layerwise protocol string")
    return value


def _component(value: Any):
    if value not in ("main", "indexer"):
        raise ValueError("Unknown layerwise protocol component")
    return value


def _frame(value: Any, name: str, length: int) -> tuple:
    op, version, *data = _row(value, length + 2)
    if op != name or _integer(version) != PROTOCOL_VERSION:
        raise ValueError("Wrong layerwise operation or protocol version")
    return tuple(data)


def _target_id(value: Any) -> TargetId:
    request, transfer, allocation = _row(value, 3)
    return TargetId(_string(request), _integer(transfer), _integer(allocation))


def _catalog_ref(value: Any) -> CatalogRef:
    engine, generation, rank = _row(value, 3)
    return CatalogRef(_string(engine), _integer(generation), _integer(rank))


def _entry_id(value: Any) -> EntryId:
    layer, component, position = _row(value, 3)
    return EntryId(_integer(layer), _component(component), _integer(position))


def encode_allocation(allocation: TargetAllocation, refs: tuple[CatalogRef, ...]) -> tuple:
    target = allocation.target
    identities = tuple((ref.engine, ref.registration, ref.rank) for ref in refs)
    return (
        "target",
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        allocation.prompt_tokens,
        allocation.blocks,
        identities,
    )


def decode_allocation(value: Any) -> tuple[TargetAllocation, tuple[CatalogRef, ...]]:
    identity, prompt, blocks, refs = _frame(value, "target", 4)
    groups = tuple(tuple(_integer(block) for block in _row(group)) for group in _row(blocks))
    return TargetAllocation(_target_id(identity), _integer(prompt, 1), groups), tuple(
        _catalog_ref(ref) for ref in _row(refs)
    )


def encode_catalog(catalog: RankCatalog) -> tuple:
    ref = catalog.ref
    return (
        "catalog",
        PROTOCOL_VERSION,
        (ref.engine, ref.registration, ref.rank),
        catalog.endpoint,
        tuple((r.name, r.backing, r.backing_generation, r.backing_offset, r.length, r.kind) for r in catalog.regions),
        tuple(
            (
                tuple(d.entry),
                d.group,
                d.region,
                d.offset,
                d.page_bytes,
                d.page_stride,
                d.page_tokens,
                d.block_tokens,
                d.capacity,
                d.dtype,
            )
            for d in catalog.entries
        ),
    )


def encode_catalog_query(ref: CatalogRef) -> tuple:
    return "get-catalog", PROTOCOL_VERSION, (ref.engine, ref.registration, ref.rank)


def decode_catalog_query(value: Any) -> CatalogRef:
    (ref,) = _frame(value, "get-catalog", 1)
    return _catalog_ref(ref)


def decode_catalog(value: Any) -> RankCatalog:
    ref, endpoint, regions, entries = _frame(value, "catalog", 4)
    parsed_regions = []
    for item in _row(regions):
        name, backing, gen, offset, length, kind = _row(item, 6)
        parsed_regions.append(
            RegisteredRegion(
                _string(name), _string(backing), _integer(gen), _integer(offset), _integer(length, 1), _component(kind)
            )
        )
    parsed_entries = []
    for item in _row(entries):
        entry, group, region, offset, size, stride, page, block, capacity, dtype = _row(item, 10)
        parsed_entries.append(
            Destination(
                _entry_id(entry),
                _integer(group),
                _string(region),
                _integer(offset),
                _integer(size, 1),
                _integer(stride, 1),
                _integer(page, 1),
                _integer(block, 1),
                _integer(capacity, 1),
                _string(dtype),
            )
        )
    return RankCatalog(_catalog_ref(ref), _string(endpoint), tuple(parsed_regions), tuple(parsed_entries))


def encode_results(target: TargetId, results: tuple[WriteResult, ...]) -> tuple:
    return (
        "results",
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        results,
    )


def decode_results(value: Any) -> tuple[TargetId, tuple[WriteResult, ...]]:
    identity, rows = _frame(value, "results", 2)
    results = []
    for row in _row(rows):
        writer, entry, span = _row(row, 3)
        start, end = _row(span, 2)
        results.append(WriteResult(_integer(writer), _entry_id(entry), Span(_integer(start), _integer(end, 1))))
    return _target_id(identity), tuple(results)


def encode_addresses(target: TargetId, rows: tuple[PeerAddress, ...]) -> tuple:
    return (
        "addresses",
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        tuple(((r.catalog.engine, r.catalog.registration, r.catalog.rank), r.region, r.base, r.length) for r in rows),
    )


def decode_addresses(value: Any, target: TargetId) -> tuple[PeerAddress, ...]:
    identity, rows = _frame(value, "addresses", 2)
    if _target_id(identity) != target:
        raise ValueError("Layerwise address response belongs to a different target")
    output = []
    for row in _row(rows):
        ref, region, base, length = _row(row, 4)
        output.append(PeerAddress(_catalog_ref(ref), _string(region), _integer(base, 1), _integer(length, 1)))
    return tuple(output)


def encode_writer_query(op: str, target: TargetId, writer: int, ref: CatalogRef) -> tuple:
    if op not in ("get-addresses", "poll-writer"):
        raise ValueError("Unknown writer query")
    return (
        op,
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        writer,
        (ref.engine, ref.registration, ref.rank),
    )


def decode_writer_query(value: Any, op: str) -> tuple[TargetId, int, CatalogRef]:
    identity, writer, ref = _frame(value, op, 3)
    return _target_id(identity), _integer(writer), _catalog_ref(ref)


def encode_terminal(target: TargetId, writer: int, error: str | None, quiesced: bool) -> tuple:
    return (
        "terminal",
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        writer,
        error,
        quiesced,
    )


def decode_terminal(value: Any) -> tuple[TargetId, int, str | None, bool]:
    identity, writer, error, quiesced = _frame(value, "terminal", 4)
    if type(quiesced) is not bool:
        raise ValueError("Writer terminal must include an explicit native quiescence verdict")
    return _target_id(identity), _integer(writer), None if error is None else _string(error), quiesced


def encode_worker_query(op: str, target: TargetId, ref: CatalogRef) -> tuple:
    if op not in ("poll-target", "close-target"):
        raise ValueError("Unknown worker query")
    return (
        op,
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        (ref.engine, ref.registration, ref.rank),
    )


def decode_worker_query(value: Any, op: str) -> tuple[TargetId, CatalogRef]:
    identity, ref = _frame(value, op, 2)
    return _target_id(identity), _catalog_ref(ref)


def encode_worker_status(state: str, target: TargetId, ref: CatalogRef, writers: tuple[int, ...] = ()) -> tuple:
    return (
        state,
        PROTOCOL_VERSION,
        (target.request, target.transfer_generation, target.allocation_generation),
        (ref.engine, ref.registration, ref.rank),
        writers,
    )


def decode_worker_status(value: Any, target: TargetId, ref: CatalogRef) -> tuple[str, tuple[int, ...]]:
    state, version, identity, worker, writers = _row(value, 5)
    if state not in ("prepared", "waiting", "ready", "failed", "closing", "closed"):
        raise ValueError("Unknown D worker target status")
    if _integer(version) != PROTOCOL_VERSION or _target_id(identity) != target or _catalog_ref(worker) != ref:
        raise ValueError("D worker status belongs to a different target or registration")
    ranks = tuple(_integer(rank) for rank in _row(writers))
    if ranks and (state != "closing" or tuple(sorted(set(ranks))) != ranks):
        raise ValueError("Invalid outstanding writer set in a D worker status")
    return state, ranks

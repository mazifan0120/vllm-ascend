# SPDX-License-Identifier: Apache-2.0
"""The D worker's local target binding and receive lifecycle."""

from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from threading import RLock

from ..kv_p2p.layerwise_offload_plan import (
    CatalogRef,
    Destination,
    EntryId,
    PeerAddress,
    RankCatalog,
    RegisteredLayout,
    RegisteredRegion,
    Span,
    TargetAllocation,
    TargetId,
    WriteResult,
    validate_catalog,
)
from ..kv_p2p.layerwise_offload_wire import (
    decode_allocation,
    decode_catalog_query,
    decode_results,
    decode_terminal,
    decode_worker_query,
    decode_writer_query,
    encode_addresses,
    encode_catalog,
    encode_worker_status,
)
from ..kv_p2p.mooncake_dsa_transfer import (
    DsaCacheLayout,
    DsaRegisterAtom,
    collect_bounded_register_regions,
    layout_span_bytes,
    tensor_cache_layout,
)


def register_model_targets(
    ref: CatalogRef,
    endpoint: str,
    host_backing: str,
    manager,
    kv_caches,
    layers: Mapping[str, tuple[int, int, int]],
    num_blocks: int,
    register: Callable[[list[int], list[int], list[str]], None],
) -> tuple[RankCatalog, dict[str, int]]:
    """Register actual Host Main and device Indexer storage, excluding selection buffers."""
    pool = manager.get_mooncake_host_pool()
    host_layers = set(manager.offload_layer_names)
    if num_blocks <= 0 or not host_layers or set(layers) != kv_caches.keys() or not host_layers <= kv_caches.keys():
        raise ValueError("Model target layers do not match the registered Host/Indexer layout")
    host_start = pool.data_ptr
    allocations = {}
    views = {}
    atoms = []
    positions: dict[tuple[int, str], int] = {}
    for name, (layer, group, block_tokens) in sorted(layers.items()):
        component = "main" if name in host_layers else "indexer"
        if component == "main":
            tensors = manager.get_local_host_kv_views(name)
        else:
            if not name.endswith(".indexer.k_cache"):
                raise ValueError(f"Unsupported D target cache {name}")
            tensors = kv_caches[name]
        tensors = tensors if isinstance(tensors, (tuple, list)) else (tensors,)
        if not tensors:
            raise ValueError(f"D target cache has no tensors for {name}")
        for tensor in tensors:
            position = positions.get((layer, component), 0)
            positions[layer, component] = position + 1
            layout = tensor_cache_layout(name, position, tensor, num_blocks, block_tokens)
            end = layout.base + layout_span_bytes(layout)
            if component == "main":
                if (
                    layout.scale != 1
                    or layout.dtype != "bfloat16"
                    or not host_start <= layout.base < end <= host_start + pool.nbytes
                ):
                    raise ValueError("Main target must use the manager's BF16 Host allocation")
                allocation = (component, host_start)
                backing = (host_backing, host_start - pool.region.segment_offset)
                location = f"npu:{pool.topology.device_id}"
            else:
                storage = tensor.untyped_storage()
                start = int(storage.data_ptr())
                if (
                    (tensor.device.type, tensor.device.index) != ("npu", pool.topology.device_id)
                    or not (start <= layout.base < end <= start + storage.nbytes())
                    or not (end <= host_start or layout.base >= host_start + pool.nbytes)
                ):
                    raise ValueError("Indexer target must use a separate local NPU allocation")
                allocation = (component, start)
                backing = allocations.get(allocation, (f"{ref.engine}:indexer:{len(allocations)}", start))
                location = "*"
            allocations[allocation] = backing
            views[EntryId(layer, component, position)] = (group, layout)
            atoms.append(DsaRegisterAtom(layout.base, end, location, allocation))

    bounded = collect_bounded_register_regions(atoms)
    ordered = sorted(atoms, key=lambda atom: atom.start)
    atom_starts = [atom.start for atom in ordered]
    regions = []
    for index, (base, length) in enumerate(zip(bounded.ptrs, bounded.lengths, strict=True)):
        # The packer preserves allocation boundaries. Region bases are the
        # start of an atom or a native-size split inside that atom.
        atom_index = bisect_right(atom_starts, base) - 1
        if atom_index < 0 or base >= ordered[atom_index].end:
            raise ValueError("Packed region does not start in a model target allocation")
        allocation = ordered[atom_index].allocation
        backing, origin = allocations[allocation]
        name = f"{ref.rank}:{index}"
        regions.append((RegisteredRegion(name, backing, ref.registration, base - origin, length, allocation[0]), base))
    catalog = register_rank_catalog(
        ref, endpoint, views, tuple(regions), lambda bases, lengths: register(bases, lengths, bounded.locations)
    )
    return catalog, {region.name: base for region, base in regions}


def register_rank_catalog(
    ref: CatalogRef,
    endpoint: str,
    views: Mapping[EntryId, tuple[int, DsaCacheLayout]],
    regions: tuple[tuple[RegisteredRegion, int], ...],
    register: Callable[[list[int], list[int]], None],
) -> RankCatalog:
    """Bind local model views to actual transfer-engine registrations."""
    entries = []
    for entry, (group, view) in views.items():
        if view.scale <= 0 or view.block_tokens % view.scale:
            raise ValueError("Local model view has invalid page geometry")
        end = view.base + layout_span_bytes(view)
        matches = [
            (region, base)
            for region, base in regions
            if region.kind == entry.component and base <= view.base < end <= base + region.length
        ]
        if len(matches) != 1:
            raise ValueError("Local model view must belong to exactly one registered region")
        region, base = matches[0]
        entries.append(
            Destination(
                entry,
                group,
                region.name,
                view.base - base,
                view.block_bytes,
                view.stride,
                view.block_tokens // view.scale,
                view.block_tokens,
                view.capacity,
                view.dtype,
            )
        )
    address_spans = sorted((base, base + region.length) for region, base in regions)
    if any(base <= 0 for base, _ in address_spans) or any(
        left[1] > right[0] for left, right in pairwise(address_spans)
    ):
        raise ValueError("Transfer-engine registrations must have positive, disjoint address ranges")
    catalog = RankCatalog(ref, endpoint, tuple(region for region, _ in regions), tuple(entries))
    validate_catalog(catalog)
    register([base for _, base in regions], [region.length for region, _ in regions])
    return catalog


@dataclass
class _Target:
    allocation: TargetAllocation
    expected: dict[tuple[int, EntryId], tuple[Span, ...]]
    addresses_issued: set[int] = field(default_factory=set)
    coverage: dict[tuple[int, EntryId], list[Span]] = field(default_factory=dict)
    terminal: dict[int, tuple[str | None, bool]] = field(default_factory=dict)
    local_uses: set[int] = field(default_factory=set)
    closing: bool = False
    failed: bool = False
    visibility_pending: bool = False
    visible: bool = False
    wait_visible: Callable[[], None] | None = None


class DecodeTargets:
    """Owns worker-local state only; scheduler retention outlives this record."""

    def __init__(
        self,
        layout: RegisteredLayout,
        rank: int,
        region_bases: Mapping[str, int],
        validate_resources: Callable[[], None],
        install_visibility: Callable[[TargetAllocation], Callable[[], None] | None],
    ) -> None:
        self._layout = layout
        self._catalog = layout.catalogs[rank]
        self._catalog_frame = encode_catalog(self._catalog)
        self._addresses = tuple(
            PeerAddress(self._catalog.ref, region.name, region_bases[region.name], region.length)
            for region in self._catalog.regions
        )
        if any(address.base <= 0 for address in self._addresses):
            raise ValueError("D worker requires real addresses for all registered regions")
        self._validate_resources = validate_resources
        self._install_visibility = install_visibility
        self._lock = RLock()
        self._targets: dict[TargetId, _Target] = {}
        self._closed: set[TargetId] = set()
        self._next_use = 0
        self._decode_completions: deque[tuple[tuple[tuple[TargetId, int], ...], Callable[[], bool]]] = deque()

    def handle_side_channel(self, frame: tuple) -> tuple:
        try:
            if frame and frame[0] in ("target", "poll-target", "close-target"):
                return self.handle_scheduler(frame)
            return self.handle_peer(frame)
        except (KeyError, ValueError, RuntimeError):
            return ("error",)

    def handle_peer(self, frame: tuple | list) -> tuple:
        """Data-plane side-channel handler, independent of the D forward loop."""
        op = frame[0] if frame else None
        if op == "get-catalog":
            if decode_catalog_query(frame) != self._catalog.ref:
                raise ValueError("Catalog query belongs to a stale D registration")
            self._validate_resources()
            return self._catalog_frame
        if op in ("get-addresses", "poll-writer"):
            target, writer, ref = decode_writer_query(frame, op)
            if op == "get-addresses":
                return encode_addresses(target, self.addresses(target, writer, ref))
            with self._lock:
                if ref != self._catalog.ref or writer != ref.rank:
                    raise ValueError("Writer query references the wrong registration")
                record = self._targets.get(target)
                if record is None or record.closing or record.failed:
                    return encode_worker_status("closing", target, ref)
                if writer not in record.addresses_issued:
                    raise ValueError("Writer queried a target without an address grant")
                return encode_worker_status("waiting", target, ref)
        if op == "results":
            self.apply(*decode_results(frame))
        elif op == "terminal":
            self.terminal(*decode_terminal(frame))
        else:
            raise ValueError("Unexpected layerwise peer operation")
        return ("ack",)

    def handle_scheduler(self, frame: tuple | list) -> tuple:
        """Control-side requests also run when the engine has no active decode forward."""
        op = frame[0] if frame else None
        if op == "target":
            allocation, refs = decode_allocation(frame)
            worker = self.prepare(allocation, refs)
            return encode_worker_status("prepared", allocation.target, worker)
        if op not in ("poll-target", "close-target"):
            raise ValueError("Unknown layerwise scheduler operation")
        target, worker = decode_worker_query(frame, op)
        if worker != self._catalog.ref:
            raise ValueError("Scheduler operation references a stale worker registration")
        if op == "close-target":
            self.close(target)
        state, writers = self._target_status(target)
        return encode_worker_status(state, target, worker, writers)

    def _target_status(self, target: TargetId) -> tuple[str, tuple[int, ...]]:
        if self.try_closed(target):
            return "closed", ()
        with self._lock:
            record = self._targets.get(target)
            if record is None:
                return "waiting", ()
            if record.closing:
                return "closing", self.close(target)
            if record.failed:
                return "failed", ()
        return ("ready" if self.try_ready(target) else "waiting"), ()

    def prepare(self, allocation: TargetAllocation, refs: tuple[CatalogRef, ...]) -> CatalogRef:
        if refs != self._layout.refs:
            raise ValueError("Prepare references a different registered layout")
        self._layout.validate_allocation(allocation)
        self._validate_resources()
        target = allocation.target
        with self._lock:
            if target in self._closed:
                raise RuntimeError("Target generation is closed")
            current = self._targets.get(target)
            if current is None:
                self._targets[target] = _Target(
                    allocation, self._layout.receive_plan(self._catalog.ref.rank, allocation.prompt_tokens)
                )
            elif current.closing or current.allocation != allocation:
                raise RuntimeError("Cannot change or reopen an admitted allocation")
        return self._catalog.ref

    def addresses(self, target: TargetId, writer: int, ref: CatalogRef) -> tuple[PeerAddress, ...]:
        self._validate_resources()
        with self._lock:
            record = self._targets[target]
            if (
                record.closing
                or record.failed
                or writer in record.terminal
                or writer != self._catalog.ref.rank
                or ref != self._catalog.ref
                or writer not in self._layout.writers
            ):
                raise RuntimeError("Writer cannot obtain addresses for this target")
            record.addresses_issued.add(writer)
            return self._addresses

    def apply(self, target: TargetId, results: tuple[WriteResult, ...]) -> None:
        with self._lock:
            record = self._targets.get(target)
            if record is None or record.closing or record.failed:
                return
            try:
                for result in results:
                    self._record_range(record, result)
            except (KeyError, ValueError):
                record.failed = True
                raise

    @staticmethod
    def _record_range(record: _Target, result: WriteResult) -> None:
        key = (result.writer, result.entry)
        start, end = result.span
        expected = record.expected[key]
        if not any(owned.start <= start < end <= owned.end for owned in expected):
            raise ValueError("Writer reported an unassigned destination range")
        existing = record.coverage.setdefault(key, [])
        pos = bisect_left(existing, result.span)
        if pos < len(existing) and existing[pos] == result.span:
            return
        if (pos and existing[pos - 1].end > start) or (pos < len(existing) and existing[pos].start < end):
            raise ValueError("Writer reported overlapping destination operations")
        if record.visible:
            raise ValueError("Coverage cannot change after decode admission")
        existing.insert(pos, result.span)

    def terminal(self, target: TargetId, writer: int, error: str | None, quiesced: bool) -> None:
        if writer not in self._layout.writers or (error is None and not quiesced):
            raise ValueError("Invalid writer terminal")
        with self._lock:
            record = self._targets.get(target)
            if record is None:
                return
            previous = record.terminal.get(writer)
            if previous is not None and (previous[0] != error or (previous[1] and not quiesced)):
                record.failed = True
                raise ValueError("Conflicting writer terminal")
            record.terminal[writer] = (error, quiesced)
            if error is not None:
                record.failed = True

    def try_ready(self, target: TargetId) -> bool:
        with self._lock:
            record = self._targets.get(target)
            if record is None or record.closing or record.failed:
                return False
            if record.visible:
                return True
            if record.visibility_pending or any(
                record.terminal.get(rank) != (None, True) for rank in self._layout.writers
            ):
                return False
            for key, expected in record.expected.items():
                received = record.coverage.get(key, ())
                expected_tokens = sum(span.end - span.start for span in expected)
                if sum(span.end - span.start for span in received) != expected_tokens:
                    return False
            record.visibility_pending = True
        wait_visible = None
        try:
            wait_visible = self._install_visibility(record.allocation)
        except Exception:
            with self._lock:
                record.failed = True
            raise
        finally:
            with self._lock:
                record.visibility_pending = False
                record.visible = not record.failed and not record.closing
                if record.visible:
                    record.wait_visible = wait_visible
        return record.visible

    def acquire_decode(self, scheduled: TargetAllocation | TargetId) -> int:
        target = scheduled.target if isinstance(scheduled, TargetAllocation) else scheduled
        with self._lock:
            record = self._targets[target]
            admitted = record.allocation
            if record.failed or record.closing or not record.visible:
                raise RuntimeError("Decode reached an unavailable or different allocation")
            if isinstance(scheduled, TargetAllocation) and (
                scheduled.prompt_tokens != admitted.prompt_tokens
                or len(admitted.blocks) != len(scheduled.blocks)
                or any(
                    live[: len(prompt)] != prompt
                    for live, prompt in zip(scheduled.blocks, admitted.blocks, strict=True)
                )
            ):
                raise RuntimeError("Decode reached an unavailable or different allocation")
            self._next_use += 1
            use = self._next_use
            record.local_uses.add(use)
            wait_visible = record.wait_visible
        try:
            if wait_visible is not None:
                wait_visible()
        except Exception:
            with self._lock:
                record.failed = True
                record.local_uses.remove(use)
            raise
        return use

    def finish_decode(self, target: TargetId, use: int) -> None:
        """Only called when every stream of this use has finished its accesses."""
        with self._lock:
            self._targets[target].local_uses.remove(use)

    def defer_decode_completion(self, uses: tuple[tuple[TargetId, int], ...], completed: Callable[[], bool]) -> None:
        with self._lock:
            self._decode_completions.append((uses, completed))
        self.reap_decode_completions()

    def reap_decode_completions(self) -> None:
        while True:
            with self._lock:
                if not self._decode_completions:
                    return
                pending = self._decode_completions[0]
            uses, completed = pending
            if not completed():
                return
            with self._lock:
                if not self._decode_completions or self._decode_completions[0] is not pending:
                    continue
                self._decode_completions.popleft()
                for target, use in uses:
                    self._targets[target].local_uses.remove(use)

    def close(self, target: TargetId) -> tuple[int, ...]:
        with self._lock:
            record = self._targets.get(target)
            if record is None:
                self._closed.add(target)  # Reject a Prepare already queued for this generation.
                return ()
            record.closing = True
            return tuple(
                sorted(
                    writer for writer in record.addresses_issued if not record.terminal.get(writer, (None, False))[1]
                )
            )

    def try_closed(self, target: TargetId) -> bool:
        self.reap_decode_completions()
        with self._lock:
            record = self._targets.get(target)
            if record is None:
                return target in self._closed
            if not record.closing or record.local_uses or record.visibility_pending:
                return False
            if any(not record.terminal.get(writer, (None, False))[1] for writer in record.addresses_issued):
                return False
            del self._targets[target]
            self._closed.add(target)
            return True

# SPDX-License-Identifier: Apache-2.0
"""Catalog and resource owner over existing D Host/Main and device/Indexer.

No allocation or TE registration occurs here. Input tensor roles/layer/group
identities are explicit, not guessed from names or numeric group IDs.
"""

import math
import threading
from copy import deepcopy
from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import DsaCacheLayout
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    Component,
    ContractError,
    RegisteredRegion,
    TargetLease,
    TokenRange,
    WriterPlan,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import (
    TargetBinding,
    binding_from_dsa_layout,
    provider_from_registered_dsa_layouts,
)


@dataclass(frozen=True)
class TargetEntry:
    layer_name: str
    component: Component
    logical_layer: int
    group: int
    entry: int
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    binding: TargetBinding


@dataclass(frozen=True)
class TargetCatalog:
    contract_version: int
    rank: int
    tp_size: int
    regions: tuple[RegisteredRegion, ...]
    entries: tuple[TargetEntry, ...]
    required_layers: dict[Component, tuple[int, ...]]
    writer_plan: WriterPlan


class _ResourcePins:
    def __init__(self, pins: tuple, on_release=None) -> None:
        self._pins = pins
        self._on_release = on_release
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            for pin in reversed(self._pins):
                pin.release()
            if self._on_release is not None:
                self._on_release()
            self._released = True

    def __getstate__(self):
        raise TypeError("Target resource pins remain local")


class RegisteredTargetResources:
    """Actual registered allocation refs, including explicit close protection."""

    def __init__(self, manager, kv_caches, te, *, components, layer_groups, block_tokens, num_blocks: int) -> None:
        if type(num_blocks) is not int or num_blocks < 1:
            raise ContractError("Target catalog needs the actual manager block capacity")
        if set(kv_caches) != set(components) or set(kv_caches) != set(layer_groups):
            raise ContractError("Every target cache needs an explicit component, layer and group identity")
        if any(not isinstance(c, Component) for c in components.values()) or set(components.values()) != {
            Component.MAIN,
            Component.INDEXER,
        }:
            raise ContractError("Target catalog requires both Main and Indexer")
        if (
            set(block_tokens) != set(kv_caches)
            or len(set(block_tokens.values())) != 1
            or any(type(v) is not int or v < 1 for v in block_tokens.values())
        ):
            raise ContractError("Target components require one explicit manager block token size")
        self._active_pins = 0
        self._lock = threading.RLock()
        self._closed = False
        self._manager = manager
        self._te = te
        self._pool = manager.get_mooncake_host_pool()
        self._snapshot = te.registration_snapshot()
        self._permanent_pins = self._pin()
        self._tensors = []
        self._kv_caches = kv_caches
        self._cache_tensors = {}
        self._storage_refs = []
        self._tensor_signatures = []
        self._base_addresses = {}
        self._layouts = ([], [])
        self._layer_groups = dict(layer_groups)
        self._catalog = None
        try:
            self._build(kv_caches, components, layer_groups, block_tokens, num_blocks)
            self._view_signatures = self._view_signature(entry.binding.view for entry in self._catalog.entries)
        except Exception:
            self._permanent_pins.release()
            raise

    def _pin(self, on_release=None) -> _ResourcePins:
        registration = self._te.retain_registration(self._snapshot)
        try:
            pool = self._pool.retain_target_allocation()
        except Exception:
            registration.release()
            raise
        return _ResourcePins((registration, pool), on_release)

    def _build(self, kv_caches, components, layer_groups, block_tokens, num_blocks) -> None:
        pool_start, pool_end = self._pool.data_ptr, self._pool.data_ptr + self._pool.nbytes
        host_identity = self._pool.target_backing_identity()
        # Device backing names are private to this registration owner. Address
        # comparisons are only local checks of actual storage aliasing.
        device_storages = {}
        raw_entries = []
        for name, caches in kv_caches.items():
            component = components[name]
            tensors = self._manager.get_local_host_kv_views(name) if component is Component.MAIN else caches
            tensors = tuple(tensors) if isinstance(tensors, (tuple, list)) else (tensors,)
            if not tensors:
                raise ContractError("Target cache has no physical entries")
            self._cache_tensors[name] = (component, tensors)
            for position, tensor in enumerate(tensors):
                shape = tuple(tensor.shape)
                strides = tuple(tensor.stride())
                itemsize = tensor.element_size()
                if len(shape) < 2 or shape[0] % num_blocks or any(size < 1 for size in shape):
                    raise ContractError("Invalid target physical page shape/capacity")
                scale = shape[0] // num_blocks
                page_bytes = math.prod(shape[1:]) * itemsize
                if any(strides[i] != math.prod(shape[i + 1 :]) for i in range(1, len(shape))):
                    raise ContractError("Target pages must have a contiguous inner footprint")
                if strides[0] * itemsize < page_bytes or block_tokens[name] % scale:
                    raise ContractError("Invalid target page stride or manager/page token geometry")
                address = int(tensor.data_ptr())
                end = address + (shape[0] - 1) * strides[0] * itemsize + page_bytes
                if component is Component.MAIN:
                    if scale != 1 or str(tensor.dtype) not in ("torch.bfloat16", "bfloat16"):
                        raise ContractError("Existing D Host Main requires BF16 and one page per manager block")
                    if not pool_start <= address < end <= pool_end:
                        raise ContractError("Main target is not part of the existing Host pool")
                    backing = (host_identity[0], host_identity[1], pool_start - self._pool.region.segment_offset)
                else:
                    storage = tensor.untyped_storage()
                    self._storage_refs.append(storage)
                    storage_start, storage_end = int(storage.data_ptr()), int(storage.data_ptr()) + storage.nbytes()
                    if not storage_start <= address < end <= storage_end or not (
                        end <= pool_start or address >= pool_end
                    ):
                        raise ContractError("Indexer target must be a distinct live device allocation")
                    device = getattr(tensor, "device", None)
                    if getattr(device, "type", None) != "npu" or device.index != self._pool.topology.device_id:
                        raise ContractError("Indexer target is not on this owner's NPU")
                    key = (storage_start, storage_end)
                    backing = device_storages.setdefault(
                        key,
                        (
                            f"{self._snapshot.engine_instance}:device:{len(device_storages)}",
                            self._snapshot.generation,
                            storage_start,
                        ),
                    )
                matches = [r for r in self._snapshot.regions if r[0] <= address and end <= r[0] + r[1]]
                if len(matches) != 1:
                    raise ContractError("Target entry must fit one actually registered TE region")
                region_address, region_size, _ = matches[0]
                if component is Component.MAIN:
                    if not pool_start <= region_address < region_address + region_size <= pool_end:
                        raise ContractError("Host target region extends outside its actual pool")
                elif not storage_start <= region_address < region_address + region_size <= storage_end:
                    raise ContractError("Device target region mixes physical allocations")
                layout = DsaCacheLayout(
                    name,
                    position,
                    address,
                    page_bytes,
                    strides[0] * itemsize,
                    scale,
                    block_tokens[name],
                    str(tensor.dtype),
                    num_blocks,
                )
                self._layouts[0 if component is Component.INDEXER else 1].append(layout)
                self._tensors.append(tensor)
                self._tensor_signatures.append(self._tensor_signature(tensor))
                raw_entries.append(
                    (
                        name,
                        position,
                        component,
                        shape,
                        tuple(v * itemsize for v in strides),
                        layout,
                        matches[0],
                        backing,
                    )
                )
        regions = {}
        for _, _, component, _, _, _, registration, backing in raw_entries:
            address, size, _ = registration
            identity = (address, size)
            candidate = RegisteredRegion(
                self._snapshot.engine_instance,
                self._snapshot.generation,
                f"{self._snapshot.engine_instance}:{self._snapshot.generation}:region:{self._snapshot.regions.index(registration)}",
                backing[0],
                backing[1],
                address - backing[2],
                size,
                "host" if component is Component.MAIN else "device",
                self._snapshot.transport_endpoint,
            )
            if identity in regions and regions[identity] != candidate:
                raise ContractError("TE region ambiguously mixes target backing identities")
            regions[identity] = candidate
        entries = []
        required = {Component.MAIN: set(), Component.INDEXER: set()}
        for name, position, component, shape, strides, layout, registration, _ in raw_entries:
            region = regions[registration[:2]]
            layer, group = layer_groups[name]
            binding = binding_from_dsa_layout(
                layout,
                component=component,
                logical_layer=layer,
                group=group,
                region=region,
                region_base_address=registration[0],
            )
            self._base_addresses[region.region_id] = registration[0]
            entries.append(TargetEntry(name, component, layer, group, position, shape, strides, binding))
            required[component].add(layer)
        main_backings = {r.backing_id for r in regions.values() if r.memory_kind == "host"}
        topology = self._pool.topology
        capacity_tokens = num_blocks * min(block_tokens.values())
        # v1: one writer rank per Main backing. The design selects a unique
        # writer per backing via destination_ordinal % len(writer_ranks); v1
        # has a single Main backing and every P rank holds the full Main
        # content, so rank 0 is the canonical writer and other ranks skip Main
        # to avoid duplicate sends. A later version that shards Main across
        # ranks must switch to the ordinal-based selection in the producer.
        writers = WriterPlan(
            {b: (0,) for b in main_backings},
            {b: (TokenRange(0, capacity_tokens),) for b in main_backings},
        )
        self._catalog = TargetCatalog(
            CONTRACT_VERSION,
            topology.tp_rank,
            topology.tp_size,
            tuple(regions.values()),
            tuple(entries),
            {c: tuple(sorted(layers)) for c, layers in required.items()},
            writers,
        )

    def catalog(self) -> TargetCatalog:
        with self._lock:
            if self._closed:
                raise ContractError("Target catalog is closed")
            return deepcopy(self._catalog)

    @staticmethod
    def _tensor_signature(tensor) -> tuple:
        storage = tensor.untyped_storage()
        return (
            int(tensor.data_ptr()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            int(storage.data_ptr()),
            storage.nbytes(),
        )

    def build_provider(self, runtime):
        catalog = self.catalog()
        return provider_from_registered_dsa_layouts(
            rank=catalog.rank,
            tp_size=catalog.tp_size,
            regions=catalog.regions,
            region_base_addresses=self._base_addresses,
            layouts=self._layouts,
            layer_groups=self._layer_groups,
            required_layers=catalog.required_layers,
            writer_plan=catalog.writer_plan,
            runtime=runtime,
        )

    def validate_allocation(self) -> None:
        """Check live registration and tensor layout once for the entire batch."""
        with self._lock:
            if self._closed or self._te.registration_snapshot() != self._snapshot:
                raise ContractError("Target registration is no longer the retained owner generation")
            for name, (component, expected) in self._cache_tensors.items():
                actual = (
                    self._manager.get_local_host_kv_views(name)
                    if component is Component.MAIN
                    else self._kv_caches.get(name)
                )
                actual = tuple(actual) if isinstance(actual, (tuple, list)) else (actual,)
                if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
                    raise ContractError("Target tensor allocation or layout changed after registration")
            if [self._tensor_signature(tensor) for tensor in self._tensors] != self._tensor_signatures:
                raise ContractError("Target tensor allocation or layout changed after registration")

    @staticmethod
    def _view_signature(views):
        return sorted(
            (
                v.component.value,
                v.logical_layer,
                v.region_binding,
                v.region_offset_bytes,
                v.dtype,
                v.strides_bytes,
                v.page_size_tokens,
                v.manager_block_size_tokens,
                v.block_capacity,
            )
            for v in views
        )

    def validate(self, lease: TargetLease) -> None:
        """Full descriptor validation at acquisition/admission boundaries."""
        with self._lock:
            self.validate_allocation()
            catalog = self._catalog
            if lease.rank != catalog.rank or lease.regions != catalog.regions:
                raise ContractError("Lease regions do not belong to this owner catalog")
            # The existing factory orders Indexer then Main; compare value
            # identities rather than order and ignore only request block IDs.
            if self._view_signature(lease.views) != self._view_signatures:
                raise ContractError("Lease views changed after target registration")

    def retain(self, lease: TargetLease):
        with self._lock:
            self.validate(lease)
            pin = self._pin(self._drop_request_pin)
            self._active_pins += 1
            return pin

    def _drop_request_pin(self) -> None:
        with self._lock:
            if self._active_pins < 1:
                raise ContractError("Target allocation pin underflow")
            self._active_pins -= 1

    def close(self) -> None:
        with self._lock:
            if self._active_pins:
                raise ContractError("Registered target resources still have active request pins")
            if not self._closed:
                self._permanent_pins.release()
                self._closed = True
                self._tensors.clear()
                self._cache_tensors.clear()
                self._kv_caches = None
                self._storage_refs.clear()
                self._tensor_signatures.clear()

    def __getstate__(self):
        raise TypeError("Registered target resources and local base addresses are owner-local")

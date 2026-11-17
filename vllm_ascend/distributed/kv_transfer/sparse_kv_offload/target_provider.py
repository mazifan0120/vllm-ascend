# SPDX-License-Identifier: Apache-2.0
"""Request leases over registered D Host/Main and device/Indexer allocations.

The shared contract is owned by the receipt connector. This provider allocates
no KV tensors and owns no scheduler blocks. Runtime callbacks execute in the
allocation owner's process (or a real transport to it), never on pickled tensor
references. Missing retention, visibility or release guarantees fail closed.
"""

import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Protocol

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    VISIBILITY_CONTRACT,
    BlockAllocation,
    CacheView,
    Component,
    ContractError,
    ReceiveReceipt,
    RegisteredRegion,
    RequestKey,
    TargetLease,
    TokenRange,
    WriterPlan,
    assert_lease_consistent,
    assert_registered_regions,
    block_ids_in_capacity,
    copy_target_lease,
)


@dataclass(frozen=True)
class TargetBinding:
    """Registered component geometry, independent of any request block IDs.

    stride[0] is the physical page stride. page_bytes is the actual page
    footprint including inner strides, not manager-block bytes. These facts
    must come from the registered allocation, not from group numbering.
    """

    view: CacheView
    group: int
    page_bytes: int


def binding_from_dsa_layout(
    layout,
    *,
    component: Component,
    logical_layer: int,
    group: int,
    region: RegisteredRegion,
    region_base_address: int,
) -> TargetBinding:
    """Adapt an EXISTING registered DsaCacheLayout, without allocating memory.

    The region/backing identity comes from the registration owner, never a
    cross-process pointer comparison. layout.block_bytes is one physical page;
    scale maps a manager block to pages. Registration must already be complete.
    """
    if type(layout.scale) is not int or layout.scale < 1 or layout.block_tokens % layout.scale:
        raise ContractError("DSA manager block must contain integral physical pages")
    offset = layout.base - region_base_address
    if type(region_base_address) is not int or region_base_address <= 0 or offset < 0:
        raise ContractError("DSA view is outside its registered region")
    return TargetBinding(
        CacheView(
            component,
            logical_layer,
            region.region_id,
            layout.dtype,
            (layout.stride,),
            layout.block_tokens // layout.scale,
            layout.block_tokens,
            layout.capacity,
            offset,
            (),
        ),
        group,
        layout.block_bytes,
    )


class TargetRuntime(Protocol):
    """Owner-side operations. No default implementation reports success.

    retain validates live scheduler retention and pins existing allocations;
    it must be atomic or unwind its own partial failure. make_visible wires
    the real Host/device visibility dependency onto the consumption stream.
    release rejects active runtime/writeback readers or waits until quiescent;
    failure must preserve the lease so it can be retried safely.
    """

    def retain(self, lease: TargetLease) -> None: ...
    def make_visible(self, lease: TargetLease, receipt: ReceiveReceipt) -> None: ...
    def release(self, lease: TargetLease) -> None: ...


@dataclass
class _LeaseState:
    lease: TargetLease
    phase: str


class RegisteredTargetProvider:
    """Four-method receipt adapter over an immutable registered target catalog.

    Catalog replacement requires a new provider/registration generation.
    Callbacks are made without holding the state lock. A single request cannot
    acquire/prepare/release concurrently; those races fail explicitly.
    Released identities remain tombstoned to reject stale callbacks. This
    descriptor/block ownership machine deliberately remains distinct from the
    runtime's active readers, visibility dependencies and native completion.
    Callback arguments are copied: injected runtimes cannot mutate this state.
    """

    contract_version = CONTRACT_VERSION
    visibility_contract = VISIBILITY_CONTRACT
    supports_layerwise_buffer_reuse = True

    def __init__(
        self,
        *,
        rank: int,
        tp_size: int,
        regions: tuple[RegisteredRegion, ...],
        bindings: tuple[TargetBinding, ...],
        required_layers: dict[Component, tuple[int, ...]],
        writer_plan: WriterPlan,
        runtime: TargetRuntime,
    ) -> None:
        if type(tp_size) is not int or tp_size < 1 or type(rank) is not int or not 0 <= rank < tp_size:
            raise ContractError("Target rank must belong to the configured TP group")
        for method in ("retain", "make_visible", "release"):
            if not callable(getattr(runtime, method, None)):
                raise ContractError(f"Target runtime is missing {method}")
        if not regions or not bindings:
            raise ContractError("Target allocations must be registered before provider construction")
        by_region = {region.region_id: region for region in regions}
        if len(by_region) != len(regions):
            raise ContractError("Registered target region IDs must be unique")
        assert_registered_regions(regions)
        if set(required_layers) != {Component.MAIN, Component.INDEXER} or any(
            not layers
            or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer < 0 for layer in layers)
            for layers in required_layers.values()
        ):
            raise ContractError("Explicit Main and Indexer layer coverage is required")
        actual = {component: set() for component in Component}
        for binding in bindings:
            view = binding.view
            if type(binding.group) is not int or binding.group < 0:
                raise ContractError("Invalid component group")
            if type(binding.page_bytes) is not int or binding.page_bytes < 1:
                raise ContractError("Invalid physical page footprint")
            if view.block_ids or not view.strides_bytes or view.block_capacity < 1:
                raise ContractError("Target catalog must contain positive capacity and no request blocks")
            region = by_region.get(view.region_binding)
            expected_kind = "host" if view.component is Component.MAIN else "device"
            if view.component is Component.MAIN and view.dtype not in ("bfloat16", "torch.bfloat16"):
                raise ContractError("Existing sparse offload requires BF16 Main targets")
            if region is None or region.memory_kind != expected_kind:
                raise ContractError("Main must bind Host and Indexer must bind device memory")
            scale = view.manager_block_size_tokens // view.page_size_tokens
            stride = view.strides_bytes[0]
            end = view.region_offset_bytes + (view.block_capacity * scale - 1) * stride + binding.page_bytes
            if stride < binding.page_bytes or end > region.length_bytes:
                raise ContractError("Component physical pages exceed the registered region")
            actual[view.component].add(view.logical_layer)
        if actual != {component: set(layers) for component, layers in required_layers.items()}:
            raise ContractError("Registered component layers do not match the model requirements")
        self._rank = rank
        self._tp_size = tp_size
        self._regions = tuple(regions)
        self._bindings = tuple(bindings)
        # Copy mutable dictionaries: external mutation must not change leases.
        self._writer_plan = WriterPlan(
            {key: tuple(value) for key, value in writer_plan.writers_by_backing.items()},
            {key: tuple(value) for key, value in writer_plan.ranges_by_backing.items()},
        )
        self._runtime = runtime
        self._lock = threading.RLock()
        self._states: dict[RequestKey, _LeaseState] = {}
        self._block_owners: dict[tuple[int, int], RequestKey] = {}
        self._retired: dict[RequestKey, str] = {}

    def _build_lease(self, request: RequestKey, allocation: BlockAllocation) -> TargetLease:
        if allocation.request != request or not allocation.retention_token:
            raise ContractError("Target acquisition requires matching identity and live scheduler retention")
        needed = (allocation.prompt_tokens + allocation.manager_block_size_tokens - 1) // (
            allocation.manager_block_size_tokens
        )
        checked_groups = set()
        views = []
        for binding in self._bindings:
            view = binding.view
            if view.manager_block_size_tokens != allocation.manager_block_size_tokens:
                raise ContractError("Target and scheduler manager block sizes differ")
            if binding.group >= len(allocation.group_block_ids):
                raise ContractError("Allocation is missing a component group")
            ids = allocation.group_block_ids[binding.group]
            group = (binding.group, view.block_capacity)
            if group not in checked_groups:
                if (
                    len(ids) < needed
                    or len(set(ids)) != len(ids)
                    or not block_ids_in_capacity(ids, view.block_capacity)
                ):
                    raise ContractError("Target block capacity cannot cover the complete prompt")
                checked_groups.add(group)
            views.append(replace(view, block_ids=ids))
        main_backings = {
            r.backing_id
            for v in views
            if v.component is Component.MAIN
            for r in self._regions
            if r.region_id == v.region_binding
        }
        if (
            set(self._writer_plan.writers_by_backing) != main_backings
            or set(self._writer_plan.ranges_by_backing) != main_backings
        ):
            raise ContractError("Writer plan must describe exactly the registered Main backings")
        for backing in main_backings:
            writers = self._writer_plan.writers_by_backing.get(backing, ())
            ranges = self._writer_plan.ranges_by_backing.get(backing, ())
            if (
                not writers
                or len(set(writers)) != len(writers)
                or any(type(rank) is not int or not 0 <= rank < self._tp_size for rank in writers)
            ):
                raise ContractError("Main backing has no valid writer plan")
            if any(not isinstance(span, TokenRange) for span in ranges):
                raise ContractError("Main writer ranges must be explicit TokenRange values")
            covered = 0
            for span in sorted(ranges, key=lambda span: span.start):
                if span.start > covered:
                    break
                covered = max(covered, span.end)
            if covered < allocation.prompt_tokens:
                raise ContractError("Main writer plan does not cover the prompt")
        lease = TargetLease(
            CONTRACT_VERSION,
            request,
            self._rank,
            allocation,
            self._regions,
            tuple(views),
            WriterPlan(dict(self._writer_plan.writers_by_backing), dict(self._writer_plan.ranges_by_backing)),
        )
        assert_lease_consistent(lease, request)
        return lease

    def _blocks(self, lease: TargetLease) -> set[tuple[int, int]]:
        groups = {binding.group for binding in self._bindings}
        return {(group, block) for group in groups for block in lease.allocation.group_block_ids[group]}

    def acquire_target(self, request: RequestKey, allocation: BlockAllocation) -> TargetLease:
        lease = self._build_lease(request, allocation)
        with self._lock:
            if request in self._retired:
                raise ContractError("Request identity has already been retired")
            previous = self._states.get(request)
            if previous is not None:
                if previous.phase in ("ACQUIRED", "READY") and previous.lease == lease:
                    return copy_target_lease(previous.lease)
                raise ContractError("Request identity already acquired, changed or retired")
            blocks = self._blocks(lease)
            if any(block in self._block_owners for block in blocks):
                raise ContractError("Target blocks still belong to an earlier lease")
            state = _LeaseState(copy_target_lease(lease), "ACQUIRING")
            self._states[request] = state
            self._block_owners.update({block: request for block in blocks})
        try:
            self._runtime.retain(copy_target_lease(lease))
        except Exception:
            with self._lock:
                self._states.pop(request)
                self._retired[request] = self._fingerprint(lease)
                # Runtime retain is required to unwind partial failure itself.
                for block in blocks:
                    self._block_owners.pop(block)
            raise
        with self._lock:
            state.phase = "ACQUIRED"
        return lease

    def _state(self, lease: TargetLease) -> _LeaseState:
        state = self._states.get(lease.request)
        if state is None or state.lease != lease:
            raise ContractError("Unknown or stale target lease")
        return state

    def get_target_views(self, lease: TargetLease) -> tuple[CacheView, ...]:
        with self._lock:
            state = self._state(lease)
            if state.phase not in ("ACQUIRED", "READY"):
                raise ContractError("Target views are not available in this lease state")
            return state.lease.views

    @contextmanager
    def inspect_target(self, lease: TargetLease):
        """Worker-local atomic inspection, excluding concurrent lease release.

        This only protects a control-plane lookup. Actual model operations
        still require TargetRuntime.use(), including asynchronous writeback.
        """
        with self._lock:
            state = self._state(lease)
            if state.phase not in ("ACQUIRED", "READY"):
                raise ContractError("Target registration lookup is not available in this lease state")
            yield state.lease

    def prepare_decode(self, lease: TargetLease, receipt: ReceiveReceipt) -> None:
        if receipt.request != lease.request or receipt.target_generation != lease.allocation.allocation_generation:
            raise ContractError("Receipt identity or target generation mismatch")
        if not (receipt.main_union_ok and receipt.indexer_ranks_ok and receipt.terminals_ok):
            raise ContractError("Incomplete receipt cannot prepare decode")
        with self._lock:
            state = self._state(lease)
            if state.phase == "READY":
                return
            if state.phase != "ACQUIRED" or receipt.visible:
                raise ContractError("Target cannot install this receipt visibility dependency")
            state.phase = "PREPARING"
        try:
            self._runtime.make_visible(copy_target_lease(lease), deepcopy(receipt))
        except Exception:
            with self._lock:
                state.phase = "FAILED"
            raise
        with self._lock:
            state.phase = "READY"
        # The receipt connector, not this provider, publishes visible=True.

    @staticmethod
    def _fingerprint(lease: TargetLease) -> str:
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import lease_fingerprint

        return lease_fingerprint(lease)

    def release_target(self, lease: TargetLease) -> None:
        with self._lock:
            if lease.request in self._retired:
                if self._retired[lease.request] != self._fingerprint(lease):
                    raise ContractError("Retired target lease identity has changed")
                return
            state = self._state(lease)
            if state.phase not in ("ACQUIRED", "READY", "FAILED"):
                raise ContractError("Target has an in-flight lifecycle operation")
            previous_phase = state.phase
            state.phase = "RELEASING"
        try:
            self._runtime.release(copy_target_lease(lease))
        except Exception:
            with self._lock:
                state.phase = previous_phase
            raise
        with self._lock:
            for block in self._blocks(lease):
                if self._block_owners.get(block) == lease.request:
                    del self._block_owners[block]
            self._retired[lease.request] = self._fingerprint(lease)
            del self._states[lease.request]
        # Scheduler retention and remote DMA quiescence belong to the connector.


def provider_from_registered_dsa_layouts(
    *,
    rank: int,
    tp_size: int,
    regions: tuple[RegisteredRegion, ...],
    region_base_addresses: dict[str, int],
    layouts,
    layer_groups: dict[str, tuple[int, int]],
    required_layers: dict[Component, tuple[int, ...]],
    writer_plan: WriterPlan,
    runtime: TargetRuntime,
) -> RegisteredTargetProvider:
    """Consume existing ``_build_dsa_local_layouts`` output after TE registration.

    No pool, tensor or registration is created here. Local base addresses never
    enter TargetLease. A layout spanning registration regions is rejected;
    receipt v1 cannot represent multiple regions of the same physical backing.
    """
    if set(region_base_addresses) != {region.region_id for region in regions}:
        raise ContractError("Every registered region needs an explicit local address binding")
    if any(type(base) is not int or base <= 0 for base in region_base_addresses.values()):
        raise ContractError("Invalid local registration address")
    if len(layouts) != 2:
        raise ContractError("Expected existing DSA (Indexer layouts, Main layouts)")
    bindings = []
    for component, entries in ((Component.INDEXER, layouts[0]), (Component.MAIN, layouts[1])):
        kind = "host" if component is Component.MAIN else "device"
        for layout in entries:
            if layout.layer_name not in layer_groups:
                raise ContractError("DSA layer lacks explicit logical-layer/group identity")
            physical_pages = layout.capacity * layout.scale
            if physical_pages < 1:
                raise ContractError("DSA component has no allocated physical pages")
            end = layout.base + (physical_pages - 1) * layout.stride + layout.block_bytes
            matches = [
                region
                for region in regions
                if region.memory_kind == kind
                and region_base_addresses[region.region_id] <= layout.base
                and end <= region_base_addresses[region.region_id] + region.length_bytes
            ]
            if len(matches) != 1:
                raise ContractError("DSA component must bind exactly one complete registered region")
            region = matches[0]
            layer, group = layer_groups[layout.layer_name]
            bindings.append(
                binding_from_dsa_layout(
                    layout,
                    component=component,
                    logical_layer=layer,
                    group=group,
                    region=region,
                    region_base_address=region_base_addresses[region.region_id],
                )
            )
    return RegisteredTargetProvider(
        rank=rank,
        tp_size=tp_size,
        regions=regions,
        bindings=tuple(bindings),
        required_layers=required_layers,
        writer_plan=writer_plan,
        runtime=runtime,
    )

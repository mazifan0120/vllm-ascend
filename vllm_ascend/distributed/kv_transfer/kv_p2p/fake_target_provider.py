# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Test double for the target-side ``TargetProvider``.

Same interface as the production implementation, pure stdlib, referenced only
by unit tests and the contract tests. Production code must never import this
module (there is exactly one allocator).

Behavior is scriptable per test:
- ``lease_factory`` builds the TargetLease for ``acquire_target``;
- ``prepare_error`` / ``release_error`` inject exceptions (failure paths);
- every call is recorded for assertions (``acquire_calls`` etc.).
"""

import logging
from collections.abc import Callable

try:  # in-repo import (real deployment and stub-bootstrapped unit tests)
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
        CONTRACT_VERSION,
        VISIBILITY_CONTRACT,
        BlockAllocation,
        CacheView,
        Component,
        RegisteredRegion,
        RequestKey,
        TargetLease,
        TokenRange,
        WriterPlan,
    )
except ModuleNotFoundError:  # direct top-level loading (bare test harnesses)
    from mooncake_layerwise_contract import (  # type: ignore[no-redef]
        CONTRACT_VERSION,
        VISIBILITY_CONTRACT,
        BlockAllocation,
        CacheView,
        Component,
        RegisteredRegion,
        RequestKey,
        TargetLease,
        TokenRange,
        WriterPlan,
    )

logger = logging.getLogger(__name__)


def build_simple_lease(
    request: RequestKey,
    allocation: BlockAllocation,
    *,
    components: tuple[Component, ...] = (Component.MAIN, Component.INDEXER),
    layers: tuple[int, ...] = (0,),
    page_size_tokens: int = 16,
) -> TargetLease:
    """Synthesize a consistent lease: one backing per allocation group, one
    region + view per (component, layer), writer = rank 0 covering the prompt."""
    backings = [f"backing-{i}" for i in range(max(len(allocation.group_block_ids), 1))]
    max_region_bytes = max(len(g) for g in allocation.group_block_ids) * 4096 if allocation.group_block_ids else 4096
    regions = []
    for i, backing in enumerate(backings):
        regions.append(
            RegisteredRegion(
                engine_instance="fake-engine",
                registration_generation=1,
                region_id=f"region-{backing}",
                backing_id=backing,
                backing_generation=1,
                backing_offset_bytes=0,
                length_bytes=max_region_bytes,
                memory_kind="host" if i == 0 else "device",
                transport_endpoint="fake-host:0",
            )
        )
    views: list[CacheView] = []
    for i, backing in enumerate(backings):
        for component in components:
            for layer in layers:
                views.append(
                    CacheView(
                        component=component,
                        logical_layer=layer,
                        region_binding=f"region-{backing}",
                        dtype="bfloat16",
                        strides_bytes=(1,),
                        page_size_tokens=page_size_tokens,
                        manager_block_size_tokens=page_size_tokens,  # keep manager % page == 0
                        block_capacity=len(allocation.group_block_ids[i]) if allocation.group_block_ids else 1,
                        region_offset_bytes=0,
                        block_ids=(allocation.group_block_ids[i] if allocation.group_block_ids else (0,)),
                    )
                )
    writer_plan = WriterPlan(
        writers_by_backing={backing: (0,) for backing in backings},
        ranges_by_backing={backing: (TokenRange(0, allocation.prompt_tokens),) for backing in backings},
    )
    return TargetLease(
        contract_version=CONTRACT_VERSION,
        request=request,
        rank=0,
        allocation=allocation,
        regions=tuple(regions),
        views=tuple(views),
        writer_plan=writer_plan,
    )


class FakeTargetProvider:
    """Scriptable TargetProvider stand-in. Satisfies
    ``assert_provider_capabilities`` in the default configuration."""

    contract_version: int = CONTRACT_VERSION
    visibility_contract: str = VISIBILITY_CONTRACT
    supports_layerwise_buffer_reuse: bool = True

    def __init__(
        self,
        lease_factory: Callable[[RequestKey, BlockAllocation], TargetLease] | None = None,
    ) -> None:
        self._lease_factory = lease_factory or build_simple_lease
        self.acquire_calls: list[tuple[RequestKey, BlockAllocation]] = []
        self.view_overrides: dict[RequestKey, tuple[CacheView, ...]] = {}
        self.prepare_calls: list[tuple[TargetLease, object]] = []
        self.release_calls: list[TargetLease] = []
        self.prepare_error: BaseException | None = None
        self.release_error: BaseException | None = None

    def acquire_target(self, request: RequestKey, allocation: BlockAllocation) -> TargetLease:
        self.acquire_calls.append((request, allocation))
        return self._lease_factory(request, allocation)

    def get_target_views(self, lease: TargetLease) -> tuple[CacheView, ...]:
        # Flat tuple; consumers group by view.component.
        return tuple(self.view_overrides.get(lease.request, lease.views))

    def prepare_decode(self, lease: TargetLease, receipt) -> None:
        self.prepare_calls.append((lease, receipt))
        if self.prepare_error is not None:
            raise self.prepare_error

    def release_target(self, lease: TargetLease) -> None:
        self.release_calls.append(lease)
        logger.debug("FakeTargetProvider released lease for %s", lease.request)
        if self.release_error is not None:
            raise self.release_error


__all__ = ["FakeTargetProvider", "build_simple_lease"]

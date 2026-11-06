# SPDX-License-Identifier: Apache-2.0
"""Scheduler-local coordinator for every TP worker's registered target owner.

Only pure catalog/lease values live here. Owner-local receive fences, usage
contexts and allocation refs stay in their worker. Partial or uncertain owner
operations remain recorded; they cannot turn into an acknowledgement to free.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    VISIBILITY_CONTRACT,
    BlockAllocation,
    ContractError,
    RequestKey,
    TargetLease,
    assert_provider_capabilities,
    copy_target_lease,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import TargetCatalog
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import (
    RankTarget,
    TargetManifest,
    aggregate_rank_targets,
)


@dataclass
class _CoordinatedTarget:
    allocation: BlockAllocation
    acquired: dict[int, TargetLease] = field(default_factory=dict)
    manifest: TargetManifest | None = None
    prepared: set[int] = field(default_factory=set)
    admitted: set[int] = field(default_factory=set)
    released: set[int] = field(default_factory=set)
    error: Exception | None = None
    request_id: str | None = None


def _fingerprint(lease: TargetLease) -> str:
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import lease_fingerprint

    return lease_fingerprint(lease)


class MultiRankTargetProvider:
    contract_version = CONTRACT_VERSION
    visibility_contract = VISIBILITY_CONTRACT
    supports_layerwise_buffer_reuse = True

    def __init__(self, owners: tuple, *, parallel_owner_calls: bool = False) -> None:
        if type(owners) is not tuple or not owners:
            raise ContractError("Target coordination requires an explicit owner for every TP rank")
        for owner in owners:
            assert_provider_capabilities(owner)
            if not callable(getattr(owner, "catalog", None)):
                raise ContractError("TP target owner must export its actual registered catalog")
        self._owners = owners
        self._parallel_owner_calls = parallel_owner_calls
        self._lock = threading.RLock()
        self._catalogs: tuple[TargetCatalog, ...] | None = None
        self._targets: dict[RequestKey, _CoordinatedTarget] = {}
        self._retired: dict[RequestKey, str] = {}

    def catalog(self):
        raise ContractError("A TP target has per-rank catalogs; use catalogs(), not an invented rank-0 catalog")

    def catalogs(self) -> tuple[TargetCatalog, ...]:
        with self._lock:
            return deepcopy(self._owner_catalogs_locked())

    def _owner_catalogs_locked(self) -> tuple[TargetCatalog, ...]:
        if self._catalogs is None:
            catalogs = tuple(owner.catalog() for owner in self._owners)
            if any(
                type(c) is not TargetCatalog
                or c.contract_version != CONTRACT_VERSION
                or type(c.rank) is not int
                or c.rank != rank
                or type(c.tp_size) is not int
                or c.tp_size != len(self._owners)
                for rank, c in enumerate(catalogs)
            ):
                raise ContractError("Target catalogs must cover every TP owner exactly once")
            self._catalogs = deepcopy(catalogs)
        return self._catalogs

    def _target(self, lease: TargetLease) -> _CoordinatedTarget:
        target = self._targets.get(lease.request)
        if target is None or target.manifest is None or target.manifest.lease != lease:
            raise ContractError("Unknown, changed or unpublished multi-rank target")
        return target

    def _collect_owner_results(self, ranks, invoke, record) -> None:
        if not self._parallel_owner_calls or len(ranks) < 2:
            for rank in ranks:
                record(rank, invoke(rank))
            return

        # Each RPC goes to a distinct worker. Wait for every result so a
        # failed rank cannot hide another rank's successful operation.
        first_failure = None
        with ThreadPoolExecutor(max_workers=len(ranks)) as pool:
            pending = {rank: pool.submit(invoke, rank) for rank in ranks}
            for rank, result in pending.items():
                try:
                    record(rank, result.result())
                except Exception as error:
                    if first_failure is None:
                        first_failure = error
        if first_failure is not None:
            raise first_failure

    def acquire_target(self, request: RequestKey, allocation: BlockAllocation) -> TargetLease:
        if allocation.request != request or not allocation.retention_token:
            raise ContractError("TP acquire requires a matching scheduler-retained allocation")
        with self._lock:
            if request in self._retired:
                raise ContractError("Multi-rank target identity was already retired")
            previous = self._targets.get(request)
            if previous is not None:
                if previous.error is None and previous.allocation == allocation and previous.manifest is not None:
                    return copy_target_lease(previous.manifest.lease)
                raise ContractError("Multi-rank target is changed, partial or has an unresolved owner operation")
            catalogs = self._owner_catalogs_locked()
            target = _CoordinatedTarget(deepcopy(allocation))
            self._targets[request] = target
            try:
                # Record every lease before validating the aggregate. A
                # malformed reply can still correspond to retained memory.
                self._collect_owner_results(
                    range(len(self._owners)),
                    lambda rank: self._owners[rank].acquire_target(request, allocation),
                    target.acquired.__setitem__,
                )
                ranks = tuple(RankTarget(catalogs[rank], target.acquired[rank]) for rank in range(len(self._owners)))
                aggregate = aggregate_rank_targets(ranks)
                target.manifest = TargetManifest(aggregate, ranks)
                return copy_target_lease(aggregate)
            except Exception as error:
                target.error = error
                raise ContractError("TP target acquisition failed; partial/uncertain owners remain retained") from error

    def get_target_views(self, lease):
        with self._lock:
            return deepcopy(self._target(lease).manifest.lease.views)

    def manifest(self, lease) -> TargetManifest:
        with self._lock:
            target = self._target(lease)
            if target.error is not None or target.released:
                raise ContractError("Cannot publish a failed or partially released TP target")
            # Each layer view can reference the same long block map. Share
            # its immutable integers while still isolating mutable records.
            memo = {}
            for owner in target.manifest.targets:
                for blocks in owner.lease.allocation.group_block_ids:
                    if id(blocks) not in memo and all(type(block) is int for block in blocks):
                        memo[id(blocks)] = blocks
            return deepcopy(target.manifest, memo)

    def prepare_decode(self, lease, receipt) -> None:
        with self._lock:
            target = self._target(lease)
            if target.error is not None or target.released:
                raise ContractError("Failed/releasing TP target cannot become visible")
            try:
                for rank, owner in enumerate(self._owners):
                    if rank not in target.prepared:
                        owner.prepare_decode(target.acquired[rank], deepcopy(receipt))
                        target.prepared.add(rank)
            except Exception as error:
                target.error = error
                raise ContractError("TP decode visibility is incomplete; target remains retained") from error

    def admit_target(self, manifest, request_id) -> None:
        with self._lock:
            if type(manifest) is not TargetManifest or type(request_id) is not str or not request_id:
                raise ContractError("TP admission requires a manifest and internal request identity")
            target = self._target(manifest.lease)
            if (
                target.manifest != manifest
                or target.error is not None
                or target.released
                or target.request_id not in (None, request_id)
            ):
                raise ContractError("Cannot admit a changed, failed or releasing TP target")
            target.request_id = request_id
            try:
                plan = ManifestWritePlan(target.manifest)
                pending = tuple(rank for rank in range(len(self._owners)) if rank not in target.admitted)
                self._collect_owner_results(
                    pending,
                    lambda rank: self._owners[rank].admit_target(
                        target.acquired[rank], request_id, plan.receive_keys(rank)
                    ),
                    lambda rank, _: target.admitted.add(rank),
                )
            except Exception as error:
                target.error = error
                raise ContractError("TP receive admission is incomplete; producer must not be notified") from error

    def delivery_status(self, lease) -> dict:
        with self._lock:
            target = self._target(lease)
            statuses = {}
            self._collect_owner_results(
                range(len(self._owners)),
                lambda rank: self._owners[rank].delivery_status(target.acquired[rank]),
                statuses.__setitem__,
            )
            if any(
                type(status) is not dict
                or set(status) != {"ready", "remote_quiesced"}
                or any(type(value) is not bool for value in status.values())
                for status in statuses.values()
            ):
                raise ContractError("TP owner returned malformed delivery proofs")
            return {key: all(status[key] for status in statuses.values()) for key in ("ready", "remote_quiesced")}

    def release_target(self, lease) -> None:
        # Caller still owns remote-DMA proof. This method aggregates only the
        # owner's local-reference/writeback release promises for all ranks.
        with self._lock:
            retired = self._retired.get(lease.request)
            if retired is not None:
                if retired != _fingerprint(lease):
                    raise ContractError("Retired TP target lease has changed")
                return
            target = self._target(lease)
            pending = tuple(rank for rank in range(len(self._owners)) if rank not in target.released)
            try:
                self._collect_owner_results(
                    pending,
                    lambda rank: self._owners[rank].release_target(target.acquired[rank]),
                    lambda rank, _: target.released.add(rank),
                )
            except Exception as error:
                target.error = error
                raise ContractError("TP owner release is incomplete; scheduler blocks must stay retained") from error
            self._retired[lease.request] = _fingerprint(lease)
            del self._targets[lease.request]

    def __getstate__(self):
        raise TypeError("Multi-rank target coordinator and transports stay in the scheduler process")

    def assert_can_close(self) -> None:
        with self._lock:
            if self._targets:
                raise ContractError("TP targets still include active, partial or uncertain owner operations")

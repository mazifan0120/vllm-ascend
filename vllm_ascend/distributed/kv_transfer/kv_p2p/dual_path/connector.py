# SPDX-License-Identifier: Apache-2.0
"""DualPathConnector — behavior-preserving alias of MooncakeLayerwiseConnector.

PR-00 foundation scope: this connector inherits ``MooncakeLayerwiseConnector``
unchanged in behavior, so an ordinary Layerwise workload runs exactly as it
does today. It exists to create safe Scheduler and Worker subclass seams for
later PRs; it adds no DualPath decision or data path.

Why inheritance + a custom ``__init__``:
    ``MooncakeLayerwiseConnector.__init__`` hard-instantiates its own
    ``MooncakeLayerwiseConnectorScheduler`` / ``MooncakeLayerwiseConnectorWorker``
    (see ``mooncake_layerwise_connector.py``). To give DualPath its own
    scheduler/worker subclasses, we cannot call ``super().__init__``; instead we
    replicate the parent's facade setup and build OUR subclasses, calling
    ``KVConnectorBase_V1.__init__`` directly. The copied facade state is exactly
    ``_is_kv_producer``, ``engine_id``, ``_connector_metadata``,
    ``connector_scheduler``, and ``connector_worker``; a drift guard in the
    unit tests fails if the parent facade grows additional state.

Block forwarding is free:
    Because ``DualPathConnector`` IS-A ``MooncakeLayerwiseConnector``,
    ``AscendMultiConnector.update_state_after_alloc`` already forwards it the
    real blocks whenever it wins the first-wins routing (its
    ``isinstance(c, MooncakeLayerwiseConnector)`` check covers subclasses). No
    change to ``AscendMultiConnector`` is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (
    KVPoolAdapter,
    KVPoolWorkerAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
    PathDecisionCoordinator,
    derive_decode_control_port,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass(frozen=True)
class DecodeKVSnapshot:
    """Complete Task-01 admission record, created only after real allocation.

    ``local_tokens``/``store_tokens`` are derived rather than duplicated from
    the detached ``LoadSpec`` fields; ``final_block_ids`` is mandatory because
    the record cannot exist before vLLM allocates the final Decode blocks.
    """

    target_tokens: int
    external_tokens: int
    store_load_spec: LoadSpec | None
    final_block_ids: tuple[tuple[int, ...], ...]

    @property
    def local_tokens(self) -> int:
        return self.target_tokens - self.external_tokens

    @property
    def store_tokens(self) -> int:
        if self.store_load_spec is None:
            return self.local_tokens
        return self.store_load_spec.kvpool_cached_tokens


class DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler):
    """Scheduler side of DualPathConnector.

    Task-01: for a Decode-role request carrying ``do_remote_prefill=True``,
    owns the admission path from initial HBM lookup through final slot
    allocation: ``get_num_new_matched_tokens`` returns the Decode-ready
    external delta ``E_DE = R - L_DE`` (never the Store hit), and
    ``update_state_after_alloc`` binds the frozen final blocks into a
    ``DecodeKVSnapshot``. Every other request delegates to the parent
    Mooncake Layerwise implementation unchanged.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        engine_id: str,
        dual_path_cfg: DualPathConfig,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.dual_path_cfg = dual_path_cfg
        self._lookup_results: dict[str, tuple[int, LoadSpec | None]] = {}
        self._decode_kv_snapshots: dict[str, DecodeKVSnapshot] = {}
        self._kvpool_adapter: KVPoolAdapter | None = None
        self._accepting_task01 = True
        if dual_path_cfg.role == "decode":
            self._kvpool_adapter = KVPoolAdapter(vllm_config, kv_cache_config)
            data_parallel_rank = vllm_config.parallel_config.data_parallel_rank
            control_port = derive_decode_control_port(
                dual_path_control_port=dual_path_cfg.dual_path_control_port,
                data_parallel_rank=data_parallel_rank,
                kv_port=vllm_config.kv_transfer_config.kv_port,
                worker_port_span=(
                    vllm_config.parallel_config.data_parallel_size * vllm_config.parallel_config.tensor_parallel_size
                ),
            )
            self._path_decision_coordinator = PathDecisionCoordinator.for_decode(
                engine_id=engine_id,
                data_parallel_rank=data_parallel_rank,
                control_endpoint=DecodeControlEndpoint(host=get_ip(), port=control_port),
            )
        else:
            self._path_decision_coordinator = PathDecisionCoordinator.for_prefill()
        logger.info(
            "Initializing DualPath Scheduler %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )

    def _is_task01_decode_request(self, request: Request) -> bool:
        """Task-01 admission applies only to Decode-role requests that arrived
        with ``do_remote_prefill is True``; everything else keeps parent behavior."""
        if not self._accepting_task01 or self.dual_path_cfg.role != "decode":
            return False
        params = request.kv_transfer_params
        return params is not None and params.get("do_remote_prefill") is True

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        if not self._is_task01_decode_request(request):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)

        request_id = request.request_id
        if request_id in self._decode_kv_snapshots:
            raise RuntimeError(
                f"DualPath request {request_id} is already admitted; a new initial lookup is a lifecycle error"
            )

        target_tokens = max(request.num_tokens - 1, 0)
        local_tokens = num_computed_tokens
        if not 0 <= local_tokens <= target_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} initial admission requires "
                f"0 <= local_tokens ({local_tokens}) <= target_tokens ({target_tokens})"
            )
        if local_tokens >= target_tokens:
            # HBM-complete: no KVPool lookup, no Task-01 state.
            self._lookup_results.pop(request_id, None)
            return 0, False

        external_tokens = target_tokens - local_tokens
        cached = self._lookup_results.get(request_id)
        if cached is not None:
            if cached[0] == external_tokens:
                # Identical duplicate lookup (e.g. allocation-failure retry):
                # reuse the detached result instead of re-probing the KV pool.
                return external_tokens, True
            # A changed E_DE invalidates the unbound result; discard it before
            # the fresh lookup so a failing re-probe cannot leave it behind.
            del self._lookup_results[request_id]

        assert self._kvpool_adapter is not None
        detached_spec = self._kvpool_adapter.lookup(request, local_tokens)
        self._lookup_results[request_id] = (external_tokens, detached_spec)
        return external_tokens, True

    def update_state_after_alloc(self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int) -> None:
        if not self._is_task01_decode_request(request):
            return super().update_state_after_alloc(request, blocks, num_external_tokens)

        request_id = request.request_id
        frozen_block_ids = tuple(tuple(group) for group in blocks.get_block_ids())
        target_tokens = max(request.num_tokens - 1, 0)

        existing = self._decode_kv_snapshots.get(request_id)
        if existing is not None:
            if (
                existing.target_tokens == target_tokens
                and existing.external_tokens == num_external_tokens
                and existing.final_block_ids == frozen_block_ids
            ):
                return
            raise RuntimeError(
                f"DualPath request {request_id} got a conflicting duplicate admission bind; "
                "the original admission is preserved"
            )

        if num_external_tokens == 0:
            # HBM-complete admission returned (0, False): no Task-01 state, and
            # the parent must not fire its remote-prefill/metaserver flow.
            self._lookup_results.pop(request_id, None)
            return

        entry = self._lookup_results.pop(request_id, None)
        if entry is None:
            raise RuntimeError(f"DualPath request {request_id} has no Task-01 lookup result to bind after allocation")
        cached_external_tokens, detached_spec = entry
        local_tokens = target_tokens - cached_external_tokens
        if num_external_tokens != cached_external_tokens:
            raise RuntimeError(
                f"DualPath request {request_id} external-token mismatch: vLLM allocated "
                f"{num_external_tokens} but the cached lookup expected {cached_external_tokens}"
            )
        if detached_spec is not None:
            if detached_spec.vllm_cached_tokens != local_tokens:
                raise RuntimeError(
                    f"DualPath request {request_id} detached LoadSpec local tokens "
                    f"{detached_spec.vllm_cached_tokens} != {local_tokens}"
                )
            if not local_tokens < detached_spec.kvpool_cached_tokens <= target_tokens:
                raise RuntimeError(
                    f"DualPath request {request_id} detached LoadSpec store tokens "
                    f"{detached_spec.kvpool_cached_tokens} outside ({local_tokens}, {target_tokens}]"
                )

        self._decode_kv_snapshots[request_id] = DecodeKVSnapshot(
            target_tokens=target_tokens,
            external_tokens=num_external_tokens,
            store_load_spec=detached_spec,
            final_block_ids=frozen_block_ids,
        )

    def request_finished(self, request: Request, block_ids: list[int]) -> tuple[bool, dict | None]:
        self._lookup_results.pop(request.request_id, None)
        self._decode_kv_snapshots.pop(request.request_id, None)
        return super().request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict | None]:
        self._lookup_results.pop(request.request_id, None)
        self._decode_kv_snapshots.pop(request.request_id, None)
        return super().request_finished_all_groups(request, block_ids)

    def shutdown(self) -> None:
        """Stop Task-01 admission and release all owned records and clients."""
        self._accepting_task01 = False
        self._lookup_results.clear()
        self._decode_kv_snapshots.clear()
        if self._kvpool_adapter is not None:
            self._kvpool_adapter.close()
        self._path_decision_coordinator.close()


class DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker):
    """Worker side of DualPathConnector.

    Task-01: a Decode-role worker additionally owns a lookup-only
    ``KVPoolWorkerAdapter`` (non-layerwise ``KVPoolWorker`` + the existing
    ``LookupKeyServer`` on the owning rank). No transfer threads, KV cache
    registration, or Store load metadata are started by this adapter.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        engine_id: str,
        dual_path_cfg: DualPathConfig,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.dual_path_cfg = dual_path_cfg
        self._kvpool_worker_adapter: KVPoolWorkerAdapter | None = None
        if dual_path_cfg.role == "decode":
            self._kvpool_worker_adapter = KVPoolWorkerAdapter(vllm_config, kv_cache_config)
        logger.info(
            "Initializing DualPath Worker %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )

    def shutdown(self) -> None:
        if self._kvpool_worker_adapter is not None:
            self._kvpool_worker_adapter.close()


class DualPathConnector(MooncakeLayerwiseConnector, SupportsHMA):
    """Behavior-preserving alias of ``MooncakeLayerwiseConnector``.

    A selectable connector name that constructs DualPath Scheduler/Worker
    subclasses while preserving the parent's accounting, metadata, transfer,
    completion, invalid-block, cleanup, and failure behavior for ordinary
    Layerwise workloads.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        # NOTE: do NOT call MooncakeLayerwiseConnector.__init__ — it hard-builds
        # the parent scheduler/worker. Call KVConnectorBase_V1.__init__ exactly
        # once and replicate only the facade state listed in the PR-00
        # construction contract.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()
        dual_path_cfg = DualPathConfig.from_extra_config(
            vllm_config.kv_transfer_config.kv_connector_extra_config,
            vllm_config.kv_transfer_config,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = DualPathConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id), dual_path_cfg
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = DualPathConnectorWorker(
                vllm_config, kv_cache_config, str(self.engine_id), dual_path_cfg
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role!r}")

    def shutdown(self):
        """Release Task-01-owned state and adapters, then defer to the base."""
        if isinstance(self.connector_scheduler, DualPathConnectorScheduler):
            self.connector_scheduler.shutdown()
        if isinstance(self.connector_worker, DualPathConnectorWorker):
            self.connector_worker.shutdown()
        super().shutdown()

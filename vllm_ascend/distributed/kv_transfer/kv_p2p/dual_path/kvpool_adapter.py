# SPDX-License-Identifier: Apache-2.0
"""Decode-side KVPool adapters for ``DualPathConnector``.

``KVPoolAdapter`` gives the Decode Scheduler a private, non-layerwise
``KVPoolScheduler``. Lookup detaches candidate ``LoadSpec`` records; an
explicit post-allocation commit authorizes the existing async load lifecycle
using a copy so the detached admission fact remains unchanged.

``KVPoolWorkerAdapter`` delegates that load lifecycle to one private
``KVPoolWorker`` and retains ownership of the existing ``LookupKeyServer``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    LookupKeyServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class KVPoolAdapter:
    """Private Decode Scheduler adapter over one dedicated KVPoolScheduler."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        page_size_bytes = kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes
        self._pool_scheduler = KVPoolScheduler(
            vllm_config,
            use_layerwise=False,
            kv_cache_config=kv_cache_config,
            page_size_bytes=page_size_bytes,
        )

    def lookup(self, request: Request, local_tokens: int) -> LoadSpec | None:
        """Probe the KV pool and return a detached LoadSpec candidate.

        Returns ``None`` for a miss, a hit that does not extend the local HBM
        prefix, or a lookup failure (logged once per failure). The owned
        scheduler never retains a ``load_specs`` entry for the request on any
        outcome, and the returned object is exclusively owned by the caller.
        """
        request_id = request.request_id
        try:
            store_delta, _ = self._pool_scheduler.get_num_new_matched_tokens(request, local_tokens)
        except Exception:
            self._pool_scheduler.load_specs.pop(request_id, None)
            logger.exception("DualPath KVPool lookup failed for request %s; treating as Store miss", request_id)
            return None

        spec = self._pool_scheduler.load_specs.pop(request_id, None)
        if spec is None:
            return None

        original_delta = spec.kvpool_cached_tokens - spec.vllm_cached_tokens
        if store_delta != original_delta:
            raise RuntimeError(
                f"DualPath KVPool lookup delta mismatch for request {request_id}: "
                f"scheduler returned {store_delta} but detached LoadSpec implies {original_delta}"
            )
        if spec.vllm_cached_tokens != local_tokens:
            raise RuntimeError(
                f"DualPath KVPool lookup local-token mismatch for request {request_id}: "
                f"expected {local_tokens}, detached LoadSpec has {spec.vllm_cached_tokens}"
            )

        target_tokens = max(request.num_tokens - 1, 0)
        usable_store_tokens = min(spec.kvpool_cached_tokens, target_tokens)
        if usable_store_tokens != spec.kvpool_cached_tokens:
            spec = dataclasses.replace(spec, kvpool_cached_tokens=usable_store_tokens)
        if usable_store_tokens <= local_tokens:
            return None
        return spec

    def commit_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        load_spec: LoadSpec,
    ) -> None:
        pool = self._pool_scheduler
        request_id = request.request_id
        ready_tokens = max(request.num_tokens - 1, 0)
        ready_delta = ready_tokens - load_spec.vllm_cached_tokens

        if pool.kv_role != "kv_consumer" or pool.use_layerwise:
            raise RuntimeError("DualPath KVPool commit requires a non-layerwise Decode-owned scheduler")
        if load_spec.kvpool_cached_tokens != ready_tokens:
            raise RuntimeError(
                f"DualPath KVPool commit ready-token mismatch for request {request_id}: "
                f"expected {ready_tokens}, detached LoadSpec has {load_spec.kvpool_cached_tokens}"
            )
        if ready_delta <= 0:
            raise RuntimeError(
                f"DualPath KVPool commit requires a positive ready-token delta for request {request_id}: "
                f"ready={ready_tokens}, local={load_spec.vllm_cached_tokens}"
            )
        if request_id in pool.load_specs:
            raise RuntimeError(f"DualPath KVPool request {request_id} is already committed")

        had_unfinished_request = request_id in pool._unfinished_requests
        previous_unfinished_request = pool._unfinished_requests.get(request_id)
        had_unfinished_request_id = request_id in pool._unfinished_request_ids
        had_loading_request_id = request_id in pool._loading_req_ids

        pool.load_specs[request_id] = dataclasses.replace(load_spec)
        try:
            pool.update_state_after_alloc(request, blocks, ready_delta)
        except Exception:
            pool.load_specs.pop(request_id, None)
            if had_unfinished_request:
                assert previous_unfinished_request is not None
                pool._unfinished_requests[request_id] = previous_unfinished_request
            else:
                pool._unfinished_requests.pop(request_id, None)
            if had_unfinished_request_id:
                pool._unfinished_request_ids.add(request_id)
            else:
                pool._unfinished_request_ids.discard(request_id)
            if had_loading_request_id:
                pool._loading_req_ids.add(request_id)
            else:
                pool._loading_req_ids.discard(request_id)
            raise

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendConnectorMetadata:
        return self._pool_scheduler.build_connector_meta(scheduler_output)

    def close(self) -> None:
        """Close the lazily created LookupKeyClient, if any. Idempotent."""
        client = self._pool_scheduler.client
        if client is not None:
            client.close()
            self._pool_scheduler.client = None


class KVPoolWorkerAdapter:
    """Decode Worker adapter over a KVPoolWorker and LookupKeyServer."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self._pool_worker = KVPoolWorker(
            vllm_config,
            use_layerwise=False,
            kv_cache_config=kv_cache_config,
        )
        self._lookup_server: LookupKeyServer | None = None
        # Same ownership rule as AscendStoreConnector: the non-layerwise lookup
        # endpoint is bound by rank 0 only.
        if vllm_config.parallel_config.rank == 0:
            self._lookup_server = LookupKeyServer(self._pool_worker, vllm_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._pool_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, metadata: AscendConnectorMetadata) -> None:
        self._pool_worker.start_load_kv(metadata)

    def get_finished(
        self,
        finished_req_ids: set[str],
        metadata: AscendConnectorMetadata,
    ) -> tuple[set[str], set[str]]:
        return self._pool_worker.get_finished(finished_req_ids, metadata)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self._pool_worker.get_block_ids_with_load_errors()

    def close(self) -> None:
        """Terminate the lookup server if bound. Idempotent."""
        if self._lookup_server is not None:
            self._lookup_server.close()
            self._lookup_server = None

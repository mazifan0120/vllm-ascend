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
        store_delta = load_spec.kvpool_cached_tokens - load_spec.vllm_cached_tokens

        if pool.kv_role not in {"kv_consumer", "kv_both"} or pool.use_layerwise:
            raise RuntimeError("DualPath KVPool commit requires a non-layerwise Decode-owned scheduler")
        if not 0 <= load_spec.vllm_cached_tokens < load_spec.kvpool_cached_tokens <= ready_tokens:
            raise RuntimeError(
                f"DualPath KVPool commit token range is invalid for request {request_id}: "
                f"local={load_spec.vllm_cached_tokens}, store={load_spec.kvpool_cached_tokens}, "
                f"ready={ready_tokens}"
            )
        if (
            request_id in pool.load_specs
            or request_id in pool._request_trackers
            or request_id in pool._unfinished_requests
            or request_id in pool._unfinished_request_ids
            or request_id in pool._loading_req_ids
        ):
            raise RuntimeError(f"DualPath KVPool request {request_id} is already committed")

        pool.load_specs[request_id] = dataclasses.replace(load_spec)
        try:
            pool.update_state_after_alloc(request, blocks, store_delta)
        except Exception:
            pool.load_specs.pop(request_id, None)
            pool._request_trackers.pop(request_id, None)
            pool._unfinished_requests.pop(request_id, None)
            pool._unfinished_request_ids.discard(request_id)
            pool._loading_req_ids.discard(request_id)
            raise

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> AscendConnectorMetadata:
        pool = self._pool_scheduler
        owned_request_ids = set(pool._unfinished_requests)
        owned_request_ids.update(pool._request_trackers)
        owned_request_ids.update(pool.load_specs)
        owned_request_ids.update(pool._unfinished_request_ids)
        owned_request_ids.update(pool._loading_req_ids)
        owned_request_ids.update(pool._delayed_free_req_ids)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        owned_cached_indices = [
            index for index, request_id in enumerate(cached_reqs.req_ids) if request_id in owned_request_ids
        ]
        filtered_cached_reqs = dataclasses.replace(
            cached_reqs,
            req_ids=[cached_reqs.req_ids[index] for index in owned_cached_indices],
            resumed_req_ids=cached_reqs.resumed_req_ids.intersection(owned_request_ids),
            new_token_ids=[cached_reqs.new_token_ids[index] for index in owned_cached_indices],
            all_token_ids={
                request_id: token_ids
                for request_id, token_ids in cached_reqs.all_token_ids.items()
                if request_id in owned_request_ids
            },
            new_block_ids=[cached_reqs.new_block_ids[index] for index in owned_cached_indices],
            num_computed_tokens=[cached_reqs.num_computed_tokens[index] for index in owned_cached_indices],
            num_output_tokens=[cached_reqs.num_output_tokens[index] for index in owned_cached_indices],
        )
        filtered_num_scheduled_tokens = {
            request_id: token_count
            for request_id, token_count in scheduler_output.num_scheduled_tokens.items()
            if request_id in owned_request_ids
        }
        pool_scheduler_output = dataclasses.replace(
            scheduler_output,
            scheduled_new_reqs=[
                request for request in scheduler_output.scheduled_new_reqs if request.req_id in owned_request_ids
            ],
            scheduled_cached_reqs=filtered_cached_reqs,
            num_scheduled_tokens=filtered_num_scheduled_tokens,
            total_num_scheduled_tokens=sum(filtered_num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={
                request_id: token_ids
                for request_id, token_ids in scheduler_output.scheduled_spec_decode_tokens.items()
                if request_id in owned_request_ids
            },
            scheduled_encoder_inputs={
                request_id: encoder_inputs
                for request_id, encoder_inputs in scheduler_output.scheduled_encoder_inputs.items()
                if request_id in owned_request_ids
            },
            preempted_req_ids=(scheduler_output.preempted_req_ids or set()).intersection(owned_request_ids),
        )
        return pool.build_connector_meta(pool_scheduler_output)

    def close(self) -> None:
        """Close the lazily created LookupKeyClient, if any. Idempotent."""
        pool = self._pool_scheduler
        pool.load_specs.clear()
        pool._request_trackers.clear()
        pool._preempted_req_ids.clear()
        pool._unfinished_requests.clear()
        pool._unfinished_request_ids.clear()
        pool._loading_req_ids.clear()
        pool._delayed_free_req_ids.clear()
        pool.sending_blocks.clear()
        pool.sending_events.clear()
        client = pool.client
        if client is not None:
            client.close()
            pool.client = None


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

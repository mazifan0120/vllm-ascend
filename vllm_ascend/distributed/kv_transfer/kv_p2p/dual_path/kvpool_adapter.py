# SPDX-License-Identifier: Apache-2.0
"""Decode-side KVPool lookup adapters for ``DualPathConnector`` (Task-01).

``KVPoolAdapter`` gives the Decode Scheduler a private, non-layerwise
``KVPoolScheduler`` whose only job is to answer "how much of this request is
already in the KV pool". Each lookup immediately detaches the resulting
``LoadSpec`` from the owned scheduler's ``load_specs`` so no shared mutable
record survives: the detached spec is a candidate fact owned by the DualPath
admission path, and ``can_load`` never authorizes I/O in Task-01.

``KVPoolWorkerAdapter`` gives the Decode Worker the lookup-only half of the
AscendStore runtime: a ``KVPoolWorker`` plus the existing ``LookupKeyServer``,
bound only on the owning rank. It starts no transfer threads and registers no
KV caches.
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
    LoadSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

if TYPE_CHECKING:
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

    def close(self) -> None:
        """Close the lazily created LookupKeyClient, if any. Idempotent."""
        client = self._pool_scheduler.client
        if client is not None:
            client.close()
            self._pool_scheduler.client = None


class KVPoolWorkerAdapter:
    """Lookup-only Decode Worker adapter: KVPoolWorker + LookupKeyServer."""

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

    def close(self) -> None:
        """Terminate the lookup server if bound. Idempotent."""
        if self._lookup_server is not None:
            self._lookup_server.close()
            self._lookup_server = None

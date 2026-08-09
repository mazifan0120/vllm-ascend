# SPDX-License-Identifier: Apache-2.0
"""Dual-path KV transfer built on ``MooncakeLayerwiseConnector``.

Extends ordinary Layerwise with Decode Store admission, Prefill-owned path
selection, and ``PE_READ`` / ``DE_READ`` routes. Store-full stays on Decode;
non-full uses the selected forward or split Reverse/Forward path.

``MooncakeLayerwiseConnector.__init__`` hard-builds the parent scheduler and
worker, so DualPath calls ``KVConnectorBase_V1.__init__`` and constructs its
own subclasses instead of ``super().__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler import (
    DecodeKVSnapshot,
    DualPathConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker import (
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


__all__ = [
    "DecodeKVSnapshot",
    "DualPathConnector",
    "DualPathConnectorScheduler",
    "DualPathConnectorWorker",
]


class DualPathConnector(MooncakeLayerwiseConnector, SupportsHMA):
    """Layerwise connector with DualPath admission and transfer routing.

    The connector constructs DualPath Scheduler and Worker subclasses while
    preserving the parent's accounting, metadata, transfer, completion,
    invalid-block, cleanup, and failure behavior for ordinary Layerwise
    workloads.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        # Do not call MooncakeLayerwiseConnector.__init__; replicate only the
        # parent facade: _is_kv_producer, engine_id, _connector_metadata,
        # connector_scheduler, connector_worker.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = DualPathConnectorMetadata()
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

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # Deliberately bypasses super().get_finished(): the parent facade drops Core-finished ids.
        assert isinstance(self.connector_worker, DualPathConnectorWorker)
        return self.connector_worker.get_finished(finished_req_ids, self._connector_metadata)

    def shutdown(self) -> None:
        """Release DualPath-owned state and adapters, then defer to the base."""
        if isinstance(self.connector_scheduler, DualPathConnectorScheduler):
            self.connector_scheduler.shutdown()
        if isinstance(self.connector_worker, DualPathConnectorWorker):
            self.connector_worker.shutdown()
        super().shutdown()

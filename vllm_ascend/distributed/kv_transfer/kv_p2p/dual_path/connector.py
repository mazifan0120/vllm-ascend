# SPDX-License-Identifier: Apache-2.0
"""DualPathConnector — unified KV-transfer connector for Ascend (Stage 1).

Stage 1 foundation scope: this connector inherits ``MooncakeLayerwiseConnector``
unchanged in behavior, so KV transfer works exactly as it does today. The
dual-path decision-making subsystem (``PathStrategy`` / ``LinkMonitor`` /
``Topology``)
is added later and runs in *shadow mode* (compute + observe, no
execution change). The PE-Read / DE-Read data-plane execution arrives in a
later spike.

Why inheritance + a custom ``__init__``:
    ``MooncakeLayerwiseConnector.__init__`` hard-instantiates its own
    ``MooncakeLayerwiseConnectorScheduler`` / ``MooncakeLayerwiseConnectorWorker``
    (see ``mooncake_layerwise_connector.py``). To give DualPath its own
    scheduler/worker subclasses, we cannot call ``super().__init__``; instead we
    replicate the parent's small amount of setup and build OUR subclasses,
    calling ``KVConnectorBase_V1.__init__`` directly.

Block forwarding is free:
    Because ``DualPathConnector`` IS-A ``MooncakeLayerwiseConnector``,
    ``AscendMultiConnector.update_state_after_alloc`` already forwards it the
    real blocks whenever it wins the first-wins routing (its
    ``isinstance(c, MooncakeLayerwiseConnector)`` check covers subclasses). No
    change to ``AscendMultiConnector`` is required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler):
    """Scheduler side of DualPathConnector.

    Stage 1: inherits all Mooncake layerwise behavior; holds the parsed
    ``DualPathConfig`` for the shadow-mode decision logic. ``_req_path`` is
    the per-request decision side-table (decision logic writes here, execution
    reads it); it lives on the scheduler, not on ``ReqMeta``, so it survives the
    parent's ``copy.deepcopy(req_meta)``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
        engine_id: str,
        dual_path_cfg: DualPathConfig,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.dual_path_cfg = dual_path_cfg
        # request_id -> "pe_read" | "de_read" (shadow wiring now; execution later).
        self._req_path: dict[str, str] = {}
        logger.info(
            "Initializing DualPath Scheduler %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )


class DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker):
    """Worker side of DualPathConnector.

    Stage 1: inherits all Mooncake layerwise behavior; holds the parsed
    ``DualPathConfig``. The ``LinkMonitor`` starts here; execution dispatches
    PE-Read / DE-Read here.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig",
        engine_id: str,
        dual_path_cfg: DualPathConfig,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, engine_id)
        self.dual_path_cfg = dual_path_cfg
        logger.info(
            "Initializing DualPath Worker %s (role=%s)",
            engine_id,
            dual_path_cfg.role,
        )


class DualPathConnector(MooncakeLayerwiseConnector, SupportsHMA):
    """Unified KV-transfer connector combining Store + P2P with path selection.

    Configured as a sibling sub-connector inside ``AscendMultiConnector``
    alongside ``AscendStoreConnector``. Stage 1: behavior identical to
    ``MooncakeLayerwiseConnector``; the decision subsystem is added later
    in shadow mode.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        # NOTE: do NOT call MooncakeLayerwiseConnector.__init__ — it hard-builds
        # the parent scheduler/worker. Replicate its setup with our subclasses.
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()
        self.dual_path_cfg = DualPathConfig.from_extra_config(
            vllm_config.kv_transfer_config.kv_connector_extra_config,
            vllm_config.kv_transfer_config,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: (
                MooncakeLayerwiseConnectorScheduler | None
            ) = DualPathConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id), self.dual_path_cfg
            )
            self.connector_worker: (
                MooncakeLayerwiseConnectorWorker | None
            ) = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = DualPathConnectorWorker(
                vllm_config, kv_cache_config, str(self.engine_id), self.dual_path_cfg
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role!r}")

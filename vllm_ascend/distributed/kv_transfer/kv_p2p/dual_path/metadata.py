# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
)


@dataclass(frozen=True)
class DecisionTimeoutMetadata:
    request_id: str
    external_block_ids: tuple[int, ...]


class DualPathConnectorMetadata(MooncakeLayerwiseConnectorMetadata):
    decision_timeouts: list[DecisionTimeoutMetadata]

    def __init__(self) -> None:
        super().__init__()
        self.decision_timeouts = []

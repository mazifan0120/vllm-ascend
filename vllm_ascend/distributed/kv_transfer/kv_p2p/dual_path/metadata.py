# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionValidationError,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
)

BlockTable = tuple[tuple[int, ...], ...]


def _freeze_block_table(blocks: BlockTable) -> BlockTable:
    return tuple(tuple(group) for group in blocks)


def _validate_token_range(token_start: int, token_end: int) -> None:
    for name, value in (("token_start", token_start), ("token_end", token_end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PathDecisionValidationError(f"{name} must be an integer and must not be a boolean")
    if not 0 <= token_start < token_end:
        raise PathDecisionValidationError("token range must satisfy 0 <= token_start < token_end")


def _validate_block_table(blocks: BlockTable, name: str) -> None:
    if not blocks:
        raise PathDecisionValidationError(f"{name} must contain at least one group")
    for group in blocks:
        if not group:
            raise PathDecisionValidationError(f"{name} groups must be non-empty")
        for block_id in group:
            if isinstance(block_id, bool) or not isinstance(block_id, int):
                raise PathDecisionValidationError(f"{name} block ids must be integers and must not be booleans")


@dataclass(frozen=True)
class ForwardPlan:
    request_key: DualPathRequestKey
    token_start: int
    token_end: int
    source_block_ids: BlockTable
    destination_block_ids: BlockTable

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        _validate_token_range(self.token_start, self.token_end)
        _validate_block_table(self.source_block_ids, "source_block_ids")
        _validate_block_table(self.destination_block_ids, "destination_block_ids")
        object.__setattr__(self, "source_block_ids", _freeze_block_table(self.source_block_ids))
        object.__setattr__(
            self,
            "destination_block_ids",
            _freeze_block_table(self.destination_block_ids),
        )


@dataclass(frozen=True)
class ForwardReceiveBinding:
    request_key: DualPathRequestKey
    wire_request_id: str
    decode_request_id: str
    destination_block_ids: BlockTable
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        for name in ("wire_request_id", "decode_request_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PathDecisionValidationError(f"{name} must be a non-empty string")
        _validate_token_range(self.token_start, self.token_end)
        _validate_block_table(self.destination_block_ids, "destination_block_ids")
        object.__setattr__(
            self,
            "destination_block_ids",
            _freeze_block_table(self.destination_block_ids),
        )


@dataclass(frozen=True)
class DecisionTimeoutMetadata:
    request_id: str
    external_block_ids: tuple[int, ...]


class DualPathConnectorMetadata(MooncakeLayerwiseConnectorMetadata):
    decision_timeouts: list[DecisionTimeoutMetadata]
    forward_receive_bindings: list[ForwardReceiveBinding]

    def __init__(self) -> None:
        super().__init__()
        self.decision_timeouts = []
        self.forward_receive_bindings = []
        self.decode_store_metadata: AscendConnectorMetadata | None = None

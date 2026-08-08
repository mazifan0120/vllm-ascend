# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    Path,
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


def _validate_non_empty_strings(values: tuple[tuple[str, str], ...]) -> None:
    for name, value in values:
        if not isinstance(value, str) or not value:
            raise PathDecisionValidationError(f"{name} must be a non-empty string")


def _validate_positive_integers(values: tuple[tuple[str, int], ...]) -> None:
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PathDecisionValidationError(f"{name} must be a positive integer and must not be a boolean")


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
    path: Path
    wire_request_id: str
    decode_request_id: str
    destination_block_ids: BlockTable
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.path, Path):
            raise PathDecisionValidationError("path must be a Path")
        _validate_non_empty_strings(
            (
                ("wire_request_id", self.wire_request_id),
                ("decode_request_id", self.decode_request_id),
            )
        )
        _validate_token_range(self.token_start, self.token_end)
        _validate_block_table(self.destination_block_ids, "destination_block_ids")
        object.__setattr__(
            self,
            "destination_block_ids",
            _freeze_block_table(self.destination_block_ids),
        )


@dataclass(frozen=True)
class ReversePlan:
    request_key: DualPathRequestKey
    wire_request_id: str
    token_start: int
    token_end: int
    source_block_ids: BlockTable
    destination_block_ids: BlockTable
    remote_engine_id: str
    remote_host: str
    remote_port: int
    remote_block_sizes: tuple[int, ...]
    remote_tp_size: int
    remote_pcp_size: int
    remote_dcp_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        _validate_non_empty_strings(
            (
                ("wire_request_id", self.wire_request_id),
                ("remote_engine_id", self.remote_engine_id),
                ("remote_host", self.remote_host),
            )
        )
        _validate_token_range(self.token_start, self.token_end)

        source_block_ids = _freeze_block_table(self.source_block_ids)
        destination_block_ids = _freeze_block_table(self.destination_block_ids)
        remote_block_sizes = tuple(self.remote_block_sizes)
        _validate_block_table(source_block_ids, "source_block_ids")
        _validate_block_table(destination_block_ids, "destination_block_ids")
        _validate_positive_integers(
            (
                ("remote_port", self.remote_port),
                ("remote_tp_size", self.remote_tp_size),
                ("remote_pcp_size", self.remote_pcp_size),
                ("remote_dcp_size", self.remote_dcp_size),
                *((f"remote_block_sizes[{index}]", size) for index, size in enumerate(remote_block_sizes)),
            )
        )

        group_count = len(remote_block_sizes)
        if group_count == 0 or len(source_block_ids) != group_count or len(destination_block_ids) != group_count:
            raise PathDecisionValidationError(
                "source, destination, and remote block-size group counts must match and be non-empty"
            )
        if any(self.token_start % block_size != 0 for block_size in remote_block_sizes):
            raise PathDecisionValidationError("token_start must align with every remote block size")
        for name, block_table in (
            ("source_block_ids", source_block_ids),
            ("destination_block_ids", destination_block_ids),
        ):
            if any(
                len(group) * block_size < self.token_end
                for group, block_size in zip(block_table, remote_block_sizes, strict=True)
            ):
                raise PathDecisionValidationError(f"{name} must cover token_end under its declared block sizes")

        object.__setattr__(self, "source_block_ids", source_block_ids)
        object.__setattr__(self, "destination_block_ids", destination_block_ids)
        object.__setattr__(self, "remote_block_sizes", remote_block_sizes)


@dataclass(frozen=True)
class ReverseReceiveBinding:
    request_key: DualPathRequestKey
    wire_request_id: str
    prefill_request_id: str
    destination_block_ids: BlockTable
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        _validate_non_empty_strings(
            (
                ("wire_request_id", self.wire_request_id),
                ("prefill_request_id", self.prefill_request_id),
            )
        )
        _validate_token_range(self.token_start, self.token_end)
        destination_block_ids = _freeze_block_table(self.destination_block_ids)
        _validate_block_table(destination_block_ids, "destination_block_ids")
        object.__setattr__(self, "destination_block_ids", destination_block_ids)


class DualPathControlFailureReason(str, Enum):
    DECISION_TIMEOUT = "DECISION_TIMEOUT"
    ACTIVATION_FAILED = "ACTIVATION_FAILED"


@dataclass(frozen=True)
class DualPathControlFailureMetadata:
    request_id: str
    invalid_block_ids: tuple[int, ...]
    reason: DualPathControlFailureReason

    def __post_init__(self) -> None:
        if not isinstance(self.reason, DualPathControlFailureReason):
            raise PathDecisionValidationError("reason must be a DualPathControlFailureReason")
        object.__setattr__(self, "invalid_block_ids", tuple(self.invalid_block_ids))


class DualPathConnectorMetadata(MooncakeLayerwiseConnectorMetadata):
    control_failures: list[DualPathControlFailureMetadata]
    forward_receive_bindings: list[ForwardReceiveBinding]
    reverse_plans: list[ReversePlan]
    reverse_receive_bindings: list[ReverseReceiveBinding]

    def __init__(self) -> None:
        super().__init__()
        self.control_failures = []
        self.forward_receive_bindings = []
        self.reverse_plans = []
        self.reverse_receive_bindings = []
        self.decode_store_metadata: AscendConnectorMetadata | None = None

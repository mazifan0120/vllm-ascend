# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    JsonObject,
    JsonValue,
    PathDecisionValidationError,
    PathKind,
    ReverseAttemptKey,
    require_exact_payload,
    reverse_wire_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
)

BlockIdGroups = tuple[tuple[int, ...], ...]


def _freeze_block_table(blocks: BlockIdGroups) -> BlockIdGroups:
    return tuple(tuple(group) for group in blocks)


def _validate_token_range(token_start: int, token_end: int) -> None:
    for name, value in (("token_start", token_start), ("token_end", token_end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise PathDecisionValidationError(f"{name} must be an integer and must not be a boolean")
    if not 0 <= token_start < token_end:
        raise PathDecisionValidationError("token range must satisfy 0 <= token_start < token_end")


def _validate_block_table(blocks: BlockIdGroups, name: str) -> None:
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
    source_block_ids: BlockIdGroups
    destination_block_ids: BlockIdGroups

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_block_ids", _freeze_block_table(self.source_block_ids))
        object.__setattr__(
            self,
            "destination_block_ids",
            _freeze_block_table(self.destination_block_ids),
        )
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        _validate_token_range(self.token_start, self.token_end)
        _validate_block_table(self.source_block_ids, "source_block_ids")
        _validate_block_table(self.destination_block_ids, "destination_block_ids")


@dataclass(frozen=True)
class ForwardReceiveBinding:
    request_key: DualPathRequestKey
    path: PathKind
    wire_request_id: str
    decode_request_id: str
    destination_block_ids: BlockIdGroups
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "destination_block_ids",
            _freeze_block_table(self.destination_block_ids),
        )
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.path, PathKind):
            raise PathDecisionValidationError("path must be a PathKind")
        _validate_non_empty_strings(
            (
                ("wire_request_id", self.wire_request_id),
                ("decode_request_id", self.decode_request_id),
            )
        )
        if self.decode_request_id != self.request_key.decode_request_id:
            raise PathDecisionValidationError("decode_request_id must match request_key.decode_request_id")
        _validate_token_range(self.token_start, self.token_end)
        _validate_block_table(self.destination_block_ids, "destination_block_ids")


def _validate_non_negative_integers(values: tuple[tuple[str, int], ...]) -> None:
    for name, value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise PathDecisionValidationError(f"{name} must be a non-negative integer and must not be a boolean")


@dataclass(frozen=True)
class ReversePlan:
    request_key: DualPathRequestKey
    wire_request_id: str
    token_start: int
    token_end: int
    source_block_ids: BlockIdGroups
    destination_block_ids: BlockIdGroups
    remote_engine_id: str
    remote_host: str
    remote_port: int
    remote_block_sizes: tuple[int, ...]
    remote_tp_size: int
    remote_pcp_size: int
    remote_dcp_size: int
    reverse_attempt_id: int
    prefill_local_tokens: int
    reverse_send_completion_id: int | None

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
        _validate_non_negative_integers(
            (
                ("reverse_attempt_id", self.reverse_attempt_id),
                ("prefill_local_tokens", self.prefill_local_tokens),
            )
        )
        if self.reverse_send_completion_id is not None:
            _validate_non_negative_integers((("reverse_send_completion_id", self.reverse_send_completion_id),))

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
        if self.remote_port > 65535:
            raise PathDecisionValidationError("remote_port must be in the range 1..65535")

        group_count = len(remote_block_sizes)
        if group_count == 0 or len(source_block_ids) != group_count or len(destination_block_ids) != group_count:
            raise PathDecisionValidationError(
                "source, destination, and remote block-size group counts must match and be non-empty"
            )
        if any(self.token_start % block_size != 0 for block_size in remote_block_sizes):
            raise PathDecisionValidationError("token_start must align with every remote block size")
        if any(self.token_end % block_size != 0 for block_size in remote_block_sizes):
            # The Reverse terminal signal only fires when the transferred range
            # covers whole blocks; an unaligned token_end would hang the pull
            # silently, so reject it before constructing the block mapping.
            raise PathDecisionValidationError("token_end must align with every remote block size")
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

    def to_dict(self) -> JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "wire_request_id": self.wire_request_id,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "source_block_ids": [list(group) for group in self.source_block_ids],
            "destination_block_ids": [list(group) for group in self.destination_block_ids],
            "remote_engine_id": self.remote_engine_id,
            "remote_host": self.remote_host,
            "remote_port": self.remote_port,
            "remote_block_sizes": list(self.remote_block_sizes),
            "remote_tp_size": self.remote_tp_size,
            "remote_pcp_size": self.remote_pcp_size,
            "remote_dcp_size": self.remote_dcp_size,
            "reverse_attempt_id": self.reverse_attempt_id,
            "prefill_local_tokens": self.prefill_local_tokens,
            "reverse_send_completion_id": self.reverse_send_completion_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> ReversePlan:
        data = require_exact_payload(
            payload,
            frozenset(
                {
                    "request_key",
                    "wire_request_id",
                    "token_start",
                    "token_end",
                    "source_block_ids",
                    "destination_block_ids",
                    "remote_engine_id",
                    "remote_host",
                    "remote_port",
                    "remote_block_sizes",
                    "remote_tp_size",
                    "remote_pcp_size",
                    "remote_dcp_size",
                    "reverse_attempt_id",
                    "prefill_local_tokens",
                    "reverse_send_completion_id",
                }
            ),
        )
        try:
            return cls(
                request_key=DualPathRequestKey.from_dict(data["request_key"]),
                wire_request_id=data["wire_request_id"],
                token_start=data["token_start"],
                token_end=data["token_end"],
                source_block_ids=data["source_block_ids"],
                destination_block_ids=data["destination_block_ids"],
                remote_engine_id=data["remote_engine_id"],
                remote_host=data["remote_host"],
                remote_port=data["remote_port"],
                remote_block_sizes=data["remote_block_sizes"],
                remote_tp_size=data["remote_tp_size"],
                remote_pcp_size=data["remote_pcp_size"],
                remote_dcp_size=data["remote_dcp_size"],
                reverse_attempt_id=data["reverse_attempt_id"],
                prefill_local_tokens=data["prefill_local_tokens"],
                reverse_send_completion_id=data["reverse_send_completion_id"],
            )
        except TypeError as error:
            raise PathDecisionValidationError("serialized ReversePlan fields have invalid types") from error


@dataclass(frozen=True)
class ReverseReceiveBinding:
    request_key: DualPathRequestKey
    wire_request_id: str
    prefill_request_id: str
    destination_block_ids: BlockIdGroups
    token_start: int
    token_end: int
    reverse_attempt_id: int
    prefill_local_tokens: int
    reverse_receive_completion_id: int

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
        _validate_non_negative_integers(
            (
                ("reverse_attempt_id", self.reverse_attempt_id),
                ("prefill_local_tokens", self.prefill_local_tokens),
                ("reverse_receive_completion_id", self.reverse_receive_completion_id),
            )
        )
        destination_block_ids = _freeze_block_table(self.destination_block_ids)
        _validate_block_table(destination_block_ids, "destination_block_ids")
        object.__setattr__(self, "destination_block_ids", destination_block_ids)


@dataclass(frozen=True)
class ReverseReceiveFailureTerminal:
    request_key: DualPathRequestKey
    reverse_attempt_id: int
    reverse_receive_completion_id: int
    wire_request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        _validate_non_negative_integers(
            (
                ("reverse_attempt_id", self.reverse_attempt_id),
                ("reverse_receive_completion_id", self.reverse_receive_completion_id),
            )
        )
        _validate_non_empty_strings((("wire_request_id", self.wire_request_id),))
        expected_wire_request_id = reverse_wire_id(
            ReverseAttemptKey(
                request_key=self.request_key,
                reverse_attempt_id=self.reverse_attempt_id,
            )
        )
        if self.wire_request_id != expected_wire_request_id:
            raise PathDecisionValidationError("wire_request_id must match the Reverse attempt identity")


class DualPathControlFailureReason(str, Enum):
    ACTIVATION_FAILED = "ACTIVATION_FAILED"
    REVERSE_JOB_FAILED = "REVERSE_JOB_FAILED"
    PEER_ABORT = "PEER_ABORT"


@dataclass(frozen=True)
class DualPathControlFailureMetadata:
    request_id: str
    invalid_block_ids: tuple[int, ...]
    reason: DualPathControlFailureReason

    def __post_init__(self) -> None:
        object.__setattr__(self, "invalid_block_ids", tuple(self.invalid_block_ids))
        _validate_non_empty_strings((("request_id", self.request_id),))
        if not self.invalid_block_ids:
            raise PathDecisionValidationError("invalid_block_ids must contain at least one block id")
        for block_id in self.invalid_block_ids:
            if isinstance(block_id, bool) or not isinstance(block_id, int):
                raise PathDecisionValidationError(
                    "invalid_block_ids block ids must be integers and must not be booleans"
                )
        if not isinstance(self.reason, DualPathControlFailureReason):
            raise PathDecisionValidationError("reason must be a DualPathControlFailureReason")


@dataclass
class DualPathWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler completion reports.

    Each worker emits ``{completion_id: 1}`` at most once per completion.
    ``aggregate`` validates the concrete metadata type before merging counts.
    """

    completion_reports: dict[int, int] = field(default_factory=dict)
    failure_reports: dict[int, int] = field(default_factory=dict)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> DualPathWorkerMetadata:
        assert isinstance(other, DualPathWorkerMetadata)
        merged = DualPathWorkerMetadata(
            completion_reports=dict(self.completion_reports),
            failure_reports=dict(self.failure_reports),
        )
        for completion_id, count in other.completion_reports.items():
            merged.completion_reports[completion_id] = merged.completion_reports.get(completion_id, 0) + count
        for completion_id, count in other.failure_reports.items():
            merged.failure_reports[completion_id] = merged.failure_reports.get(completion_id, 0) + count
        return merged


class DualPathConnectorMetadata(MooncakeLayerwiseConnectorMetadata):
    control_failures: list[DualPathControlFailureMetadata]
    forward_receive_bindings: list[ForwardReceiveBinding]
    reverse_plans: list[ReversePlan]
    reverse_receive_bindings: list[ReverseReceiveBinding]
    reverse_receive_failure_terminals: list[ReverseReceiveFailureTerminal]
    decode_store_metadata: AscendConnectorMetadata | None

    def __init__(self) -> None:
        super().__init__()
        self.control_failures = []
        self.forward_receive_bindings = []
        self.reverse_plans = []
        self.reverse_receive_bindings = []
        self.reverse_receive_failure_terminals = []
        self.decode_store_metadata = None

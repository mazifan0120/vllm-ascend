# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class PathDecisionValidationError(ValueError):
    pass


def require_exact_payload(payload: JsonValue, expected_keys: frozenset[str]) -> JsonObject:
    if not isinstance(payload, dict):
        raise PathDecisionValidationError("serialized payload must be a dictionary")
    if set(payload) != expected_keys:
        raise PathDecisionValidationError(f"serialized payload fields must be exactly {sorted(expected_keys)}")
    return payload


class PathKind(str, Enum):
    """Transfer route chosen by the Prefill-owned policy: ``PE_READ`` serves
    the request from the Prefill engine (PE); ``DE_READ`` serves it on the
    Decode engine (DE) from Store plus a Forward suffix."""

    PE_READ = "PE_READ"
    DE_READ = "DE_READ"


@dataclass(frozen=True)
class DualPathRequestKey:
    decode_engine_instance_id: str
    decode_request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.decode_engine_instance_id, str) or not self.decode_engine_instance_id:
            raise PathDecisionValidationError("decode_engine_instance_id must be a non-empty string")
        if not isinstance(self.decode_request_id, str) or not self.decode_request_id:
            raise PathDecisionValidationError("decode_request_id must be a non-empty string")

    def to_dict(self) -> JsonObject:
        return {
            "decode_engine_instance_id": self.decode_engine_instance_id,
            "decode_request_id": self.decode_request_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> DualPathRequestKey:
        data = require_exact_payload(
            payload,
            frozenset({"decode_engine_instance_id", "decode_request_id"}),
        )
        return cls(
            decode_engine_instance_id=data["decode_engine_instance_id"],
            decode_request_id=data["decode_request_id"],
        )


@dataclass(frozen=True)
class PathDecisionRequest:
    request_key: DualPathRequestKey
    target_tokens: int
    decode_local_tokens: int
    decode_store_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")

        token_counts = (
            ("target_tokens", self.target_tokens),
            ("decode_local_tokens", self.decode_local_tokens),
            ("decode_store_tokens", self.decode_store_tokens),
        )
        for name, value in token_counts:
            if isinstance(value, bool) or not isinstance(value, int):
                raise PathDecisionValidationError(f"{name} must be an integer and must not be a boolean")
        if not 0 <= self.decode_local_tokens <= self.decode_store_tokens < self.target_tokens:
            raise PathDecisionValidationError(
                "token counts must satisfy 0 <= decode_local_tokens <= decode_store_tokens < target_tokens"
            )

    def to_dict(self) -> JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "target_tokens": self.target_tokens,
            "decode_local_tokens": self.decode_local_tokens,
            "decode_store_tokens": self.decode_store_tokens,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> PathDecisionRequest:
        data = require_exact_payload(
            payload,
            frozenset(
                {
                    "request_key",
                    "target_tokens",
                    "decode_local_tokens",
                    "decode_store_tokens",
                }
            ),
        )
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            target_tokens=data["target_tokens"],
            decode_local_tokens=data["decode_local_tokens"],
            decode_store_tokens=data["decode_store_tokens"],
        )


@dataclass(frozen=True)
class PathDecisionResult:
    request_key: DualPathRequestKey
    path: PathKind

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.path, PathKind):
            raise PathDecisionValidationError("path must be a PathKind")

    def to_dict(self) -> JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "path": self.path.value,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> PathDecisionResult:
        data = require_exact_payload(
            payload,
            frozenset({"request_key", "path"}),
        )
        try:
            path = PathKind(data["path"])
        except (TypeError, ValueError) as error:
            raise PathDecisionValidationError("serialized path is not valid") from error
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            path=path,
        )


class PathPolicy(Protocol):
    def choose(self, request: PathDecisionRequest) -> PathKind: ...


class RoundRobinPathPolicy:
    """Round-robin whose initial phase is randomized so co-located Prefill instances do not
    rotate in lockstep. Eligibility-forced decisions bypass this policy and do
    not advance the rotation phase."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._next_path = (rng or random.Random()).choice((PathKind.PE_READ, PathKind.DE_READ))

    def choose(self, request: PathDecisionRequest) -> PathKind:
        selected = self._next_path
        self._next_path = PathKind.DE_READ if selected is PathKind.PE_READ else PathKind.PE_READ
        return selected


@dataclass(frozen=True)
class _DecisionRecord:
    request: PathDecisionRequest
    prefill_local_tokens: int
    result: PathDecisionResult


class PathDecisionDecider:
    def __init__(self, policy: PathPolicy) -> None:
        self._policy = policy
        self._decision_records: dict[DualPathRequestKey, _DecisionRecord] = {}

    def decide(self, request: PathDecisionRequest, prefill_local_tokens: int) -> PathDecisionResult:
        if not isinstance(request, PathDecisionRequest):
            raise PathDecisionValidationError("decision input must be a PathDecisionRequest")

        existing = self._decision_records.get(request.request_key)
        if existing is not None:
            if request != existing.request or prefill_local_tokens != existing.prefill_local_tokens:
                raise PathDecisionValidationError(
                    "request key is already associated with different token facts or prefill_local_tokens"
                )
            return existing.result

        path = PathKind.PE_READ if prefill_local_tokens >= request.decode_store_tokens else self._policy.choose(request)

        result = PathDecisionResult(request_key=request.request_key, path=path)
        self._decision_records[request.request_key] = _DecisionRecord(
            request=request,
            prefill_local_tokens=prefill_local_tokens,
            result=result,
        )
        return result

    def discard(self, request_key: DualPathRequestKey) -> None:
        self._decision_records.pop(request_key, None)

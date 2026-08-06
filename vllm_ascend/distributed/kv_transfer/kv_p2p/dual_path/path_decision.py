# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

_JsonValue: TypeAlias = str | int | float | bool | None | list["_JsonValue"] | dict[str, "_JsonValue"]
_JsonObject: TypeAlias = dict[str, _JsonValue]


class PathDecisionValidationError(ValueError):
    pass


def _require_exact_payload(payload: _JsonValue, expected_keys: frozenset[str]) -> _JsonObject:
    if not isinstance(payload, dict):
        raise PathDecisionValidationError("serialized payload must be a dictionary")
    if set(payload) != expected_keys:
        raise PathDecisionValidationError(f"serialized payload fields must be exactly {sorted(expected_keys)}")
    return payload


class Path(str, Enum):
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

    def to_dict(self) -> _JsonObject:
        return {
            "decode_engine_instance_id": self.decode_engine_instance_id,
            "decode_request_id": self.decode_request_id,
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> DualPathRequestKey:
        data = _require_exact_payload(
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
            self.target_tokens,
            self.decode_local_tokens,
            self.decode_store_tokens,
        )
        if any(isinstance(token_count, bool) or not isinstance(token_count, int) for token_count in token_counts):
            raise PathDecisionValidationError("token counts must be integers and must not be booleans")
        if not 0 <= self.decode_local_tokens <= self.decode_store_tokens < self.target_tokens:
            raise PathDecisionValidationError(
                "token counts must satisfy 0 <= decode_local_tokens <= decode_store_tokens < target_tokens"
            )

    def to_dict(self) -> _JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "target_tokens": self.target_tokens,
            "decode_local_tokens": self.decode_local_tokens,
            "decode_store_tokens": self.decode_store_tokens,
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> PathDecisionRequest:
        data = _require_exact_payload(
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
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.path, Path):
            raise PathDecisionValidationError("path must be a Path")

    def to_dict(self) -> _JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "path": self.path.value,
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> PathDecisionResult:
        data = _require_exact_payload(
            payload,
            frozenset({"request_key", "path"}),
        )
        try:
            path = Path(data["path"])
        except (TypeError, ValueError) as error:
            raise PathDecisionValidationError("serialized path is not valid") from error
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            path=path,
        )


class PathPolicy(Protocol):
    def choose(self, request: PathDecisionRequest) -> Path: ...


class RoundRobinPathPolicy:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._next = (rng or random.Random()).choice((Path.PE_READ, Path.DE_READ))

    def choose(self, request: PathDecisionRequest) -> Path:
        selected = self._next
        self._next = Path.DE_READ if selected is Path.PE_READ else Path.PE_READ
        return selected


@dataclass(frozen=True)
class _DecisionRecord:
    request: PathDecisionRequest
    result: PathDecisionResult | None


class PathDecisionDecider:
    def __init__(self, policy: PathPolicy) -> None:
        self._policy = policy
        self._decision_records: dict[DualPathRequestKey, _DecisionRecord] = {}

    def decide(self, request: PathDecisionRequest) -> PathDecisionResult:
        if not isinstance(request, PathDecisionRequest):
            raise PathDecisionValidationError("decision input must be a PathDecisionRequest")

        existing = self._decision_records.get(request.request_key)
        if existing is not None:
            if request != existing.request:
                raise PathDecisionValidationError("request key is already associated with different token facts")
            if existing.result is None:
                raise PathDecisionValidationError("decision previously failed locally")
            return existing.result

        try:
            path = self._policy.choose(request)
        except Exception as error:  # noqa: BLE001
            self._decision_records[request.request_key] = _DecisionRecord(request=request, result=None)
            raise PathDecisionValidationError("path policy raised an exception") from error
        if not isinstance(path, Path):
            self._decision_records[request.request_key] = _DecisionRecord(request=request, result=None)
            raise PathDecisionValidationError(f"policy returned an invalid path: {path!r}")

        result = PathDecisionResult(request_key=request.request_key, path=path)
        self._decision_records[request.request_key] = _DecisionRecord(request=request, result=result)
        return result

    def discard(self, request_key: DualPathRequestKey) -> None:
        self._decision_records.pop(request_key, None)

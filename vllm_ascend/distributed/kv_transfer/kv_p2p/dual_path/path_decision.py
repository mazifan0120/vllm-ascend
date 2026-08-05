# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol, TypeAlias

ERROR_INVALID_REQUEST: Final[str] = "INVALID_REQUEST"
ERROR_CONFLICTING_REQUEST: Final[str] = "CONFLICTING_REQUEST"
ERROR_INVALID_POLICY_RESULT: Final[str] = "INVALID_POLICY_RESULT"

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
        if not 0 <= self.decode_local_tokens <= self.decode_store_tokens <= self.target_tokens:
            raise PathDecisionValidationError(
                "token counts must satisfy 0 <= decode_local_tokens <= decode_store_tokens <= target_tokens"
            )
        if self.decode_local_tokens >= self.target_tokens:
            raise PathDecisionValidationError("decode_local_tokens must be less than target_tokens")

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
class PathDecisionCommit:
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
    def from_dict(cls, payload: _JsonValue) -> PathDecisionCommit:
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


@dataclass(frozen=True)
class PathDecisionError:
    request_key: DualPathRequestKey
    error_code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.error_code, str):
            raise PathDecisionValidationError("error_code must be a string")
        if not isinstance(self.message, str):
            raise PathDecisionValidationError("message must be a string")

    def to_dict(self) -> _JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "error_code": self.error_code,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> PathDecisionError:
        data = _require_exact_payload(
            payload,
            frozenset({"request_key", "error_code", "message"}),
        )
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            error_code=data["error_code"],
            message=data["message"],
        )


PathDecisionResult = PathDecisionCommit | PathDecisionError


class PathPolicy(Protocol):
    def choose(self, request: PathDecisionRequest) -> Path: ...


class RoundRobinPathPolicy:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._next = (rng or random.Random()).choice((Path.PE_READ, Path.DE_READ))

    def choose(self, request: PathDecisionRequest) -> Path:
        selected = self._next
        self._next = Path.DE_READ if selected is Path.PE_READ else Path.PE_READ
        return selected


class PathDecisionDecider:
    def __init__(self, policy: PathPolicy) -> None:
        self._policy = policy
        self._path_decisions: dict[
            DualPathRequestKey,
            tuple[PathDecisionRequest, PathDecisionCommit],
        ] = {}

    def decide(self, request: PathDecisionRequest) -> PathDecisionResult:
        if not isinstance(request, PathDecisionRequest):
            request_key = getattr(request, "request_key", None)
            if isinstance(request_key, DualPathRequestKey):
                return PathDecisionError(
                    request_key=request_key,
                    error_code=ERROR_INVALID_REQUEST,
                    message="decision input must be a PathDecisionRequest",
                )
            raise PathDecisionValidationError("decision input must include a valid DualPathRequestKey")

        existing = self._path_decisions.get(request.request_key)
        if existing is not None:
            retained_request, retained_commit = existing
            if request == retained_request:
                return retained_commit
            return PathDecisionError(
                request_key=request.request_key,
                error_code=ERROR_CONFLICTING_REQUEST,
                message="request key is already associated with different token facts",
            )

        if request.decode_store_tokens == request.target_tokens:
            path = Path.DE_READ
        else:
            path = self._policy.choose(request)
            if not isinstance(path, Path):
                return PathDecisionError(
                    request_key=request.request_key,
                    error_code=ERROR_INVALID_POLICY_RESULT,
                    message=f"policy returned an invalid path: {path!r}",
                )

        commit = PathDecisionCommit(request_key=request.request_key, path=path)
        self._path_decisions[request.request_key] = (request, commit)
        return commit

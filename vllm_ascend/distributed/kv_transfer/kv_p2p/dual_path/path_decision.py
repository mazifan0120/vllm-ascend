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
    admission_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.decode_engine_instance_id, str) or not self.decode_engine_instance_id:
            raise PathDecisionValidationError("decode_engine_instance_id must be a non-empty string")
        if not isinstance(self.decode_request_id, str) or not self.decode_request_id:
            raise PathDecisionValidationError("decode_request_id must be a non-empty string")
        if isinstance(self.admission_id, bool) or not isinstance(self.admission_id, int) or self.admission_id < 0:
            raise PathDecisionValidationError("admission_id must be a non-negative integer")

    def to_dict(self) -> JsonObject:
        return {
            "decode_engine_instance_id": self.decode_engine_instance_id,
            "decode_request_id": self.decode_request_id,
            "admission_id": self.admission_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> DualPathRequestKey:
        data = require_exact_payload(
            payload,
            frozenset({"decode_engine_instance_id", "decode_request_id", "admission_id"}),
        )
        return cls(
            decode_engine_instance_id=data["decode_engine_instance_id"],
            decode_request_id=data["decode_request_id"],
            admission_id=data["admission_id"],
        )


class PathAbortReason(str, Enum):
    DECISION_FAILED = "DECISION_FAILED"
    DELIVERY_EXHAUSTED = "DELIVERY_EXHAUSTED"
    ACTIVATION_FAILED = "ACTIVATION_FAILED"
    REQUEST_ABORTED = "REQUEST_ABORTED"


class ReverseTerminalState(str, Enum):
    TERMINALIZED = "TERMINALIZED"


@dataclass(frozen=True)
class ReverseTerminalNotice:
    reverse_attempt_id: int
    state: ReverseTerminalState

    def __post_init__(self) -> None:
        if (
            isinstance(self.reverse_attempt_id, bool)
            or not isinstance(self.reverse_attempt_id, int)
            or self.reverse_attempt_id < 0
        ):
            raise PathDecisionValidationError("reverse_attempt_id must be a non-negative integer")
        if not isinstance(self.state, ReverseTerminalState):
            raise PathDecisionValidationError("state must be a ReverseTerminalState")

    def to_dict(self) -> JsonObject:
        return {
            "reverse_attempt_id": self.reverse_attempt_id,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> ReverseTerminalNotice:
        data = require_exact_payload(
            payload,
            frozenset({"reverse_attempt_id", "state"}),
        )
        try:
            state = ReverseTerminalState(data["state"])
        except (TypeError, ValueError) as error:
            raise PathDecisionValidationError("serialized reverse terminal state is not valid") from error
        return cls(
            reverse_attempt_id=data["reverse_attempt_id"],
            state=state,
        )


@dataclass(frozen=True)
class ReverseAdmissionTerminalNotice:
    may_have_started_through_attempt_id: int | None

    def __post_init__(self) -> None:
        value = self.may_have_started_through_attempt_id
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise PathDecisionValidationError(
                "may_have_started_through_attempt_id must be a non-negative integer or None"
            )

    def to_dict(self) -> JsonObject:
        return {
            "may_have_started_through_attempt_id": self.may_have_started_through_attempt_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> ReverseAdmissionTerminalNotice:
        data = require_exact_payload(
            payload,
            frozenset({"may_have_started_through_attempt_id"}),
        )
        return cls(data["may_have_started_through_attempt_id"])


@dataclass(frozen=True)
class PathAbortNotice:
    request_key: DualPathRequestKey
    reason: PathAbortReason
    reverse_terminal: ReverseTerminalNotice | None = None
    reverse_admission_terminal: ReverseAdmissionTerminalNotice | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.reason, PathAbortReason):
            raise PathDecisionValidationError("reason must be a PathAbortReason")
        if self.reverse_terminal is not None and not isinstance(self.reverse_terminal, ReverseTerminalNotice):
            raise PathDecisionValidationError("reverse_terminal must be a ReverseTerminalNotice or None")
        if self.reverse_admission_terminal is not None and not isinstance(
            self.reverse_admission_terminal, ReverseAdmissionTerminalNotice
        ):
            raise PathDecisionValidationError(
                "reverse_admission_terminal must be a ReverseAdmissionTerminalNotice or None"
            )

    def to_dict(self) -> JsonObject:
        data: JsonObject = {
            "request_key": self.request_key.to_dict(),
            "reason": self.reason.value,
        }
        if self.reverse_terminal is not None:
            data["reverse_terminal"] = self.reverse_terminal.to_dict()
        if self.reverse_admission_terminal is not None:
            data["reverse_admission_terminal"] = self.reverse_admission_terminal.to_dict()
        return data

    @classmethod
    def from_dict(cls, payload: JsonValue) -> PathAbortNotice:
        if not isinstance(payload, dict):
            raise PathDecisionValidationError("serialized payload must be a dictionary")
        expected_keys = frozenset({"request_key", "reason"})
        if "reverse_terminal" in payload:
            expected_keys = expected_keys | {"reverse_terminal"}
        if "reverse_admission_terminal" in payload:
            expected_keys = expected_keys | {"reverse_admission_terminal"}
        data = require_exact_payload(payload, expected_keys)
        try:
            reason = PathAbortReason(data["reason"])
        except (TypeError, ValueError) as error:
            raise PathDecisionValidationError("serialized abort reason is not valid") from error
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            reason=reason,
            reverse_terminal=(
                None
                if "reverse_terminal" not in data
                else ReverseTerminalNotice.from_dict(data["reverse_terminal"])
            ),
            reverse_admission_terminal=(
                None
                if "reverse_admission_terminal" not in data
                else ReverseAdmissionTerminalNotice.from_dict(data["reverse_admission_terminal"])
            ),
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
    reverse_attempt_id: int | None = None
    prefill_local_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if not isinstance(self.path, PathKind):
            raise PathDecisionValidationError("path must be a PathKind")
        if self.path is PathKind.PE_READ:
            if self.reverse_attempt_id is not None or self.prefill_local_tokens is not None:
                raise PathDecisionValidationError("PE_READ result must not carry Reverse attempt fields")
        else:
            for name, value in (
                ("reverse_attempt_id", self.reverse_attempt_id),
                ("prefill_local_tokens", self.prefill_local_tokens),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise PathDecisionValidationError(f"DE_READ result requires a non-negative integer {name}")

    def to_dict(self) -> JsonObject:
        data: JsonObject = {
            "request_key": self.request_key.to_dict(),
            "path": self.path.value,
        }
        if self.path is PathKind.DE_READ:
            data["reverse_attempt_id"] = self.reverse_attempt_id
            data["prefill_local_tokens"] = self.prefill_local_tokens
        return data

    @classmethod
    def from_dict(cls, payload: JsonValue) -> PathDecisionResult:
        if not isinstance(payload, dict):
            raise PathDecisionValidationError("serialized payload must be a dictionary")
        if "request_key" not in payload or "path" not in payload:
            raise PathDecisionValidationError("serialized payload must contain request_key and path")
        try:
            path = PathKind(payload["path"])
        except (TypeError, ValueError) as error:
            raise PathDecisionValidationError("serialized path is not valid") from error
        expected_keys = frozenset({"request_key", "path"})
        if path is PathKind.DE_READ:
            expected_keys = expected_keys | {"reverse_attempt_id", "prefill_local_tokens"}
        data = require_exact_payload(payload, expected_keys)
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            path=path,
            reverse_attempt_id=data.get("reverse_attempt_id"),
            prefill_local_tokens=data.get("prefill_local_tokens"),
        )


@dataclass(frozen=True)
class ReverseAttemptKey:
    """Complete identity of one DE_READ Reverse attempt."""

    request_key: DualPathRequestKey
    reverse_attempt_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_key, DualPathRequestKey):
            raise PathDecisionValidationError("request_key must be a DualPathRequestKey")
        if (
            isinstance(self.reverse_attempt_id, bool)
            or not isinstance(self.reverse_attempt_id, int)
            or self.reverse_attempt_id < 0
        ):
            raise PathDecisionValidationError("reverse_attempt_id must be a non-negative integer")

    def to_dict(self) -> JsonObject:
        return {
            "request_key": self.request_key.to_dict(),
            "reverse_attempt_id": self.reverse_attempt_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> ReverseAttemptKey:
        data = require_exact_payload(
            payload,
            frozenset({"request_key", "reverse_attempt_id"}),
        )
        return cls(
            request_key=DualPathRequestKey.from_dict(data["request_key"]),
            reverse_attempt_id=data["reverse_attempt_id"],
        )


def reverse_wire_id(attempt_key: ReverseAttemptKey) -> str:
    """Return an opaque, attempt-unique Reverse wire id that callers must not parse."""
    if not isinstance(attempt_key, ReverseAttemptKey):
        raise PathDecisionValidationError("attempt_key must be a ReverseAttemptKey")
    return (
        f"ra:{attempt_key.request_key.decode_engine_instance_id}:"
        f"{attempt_key.request_key.decode_request_id}:"
        f"{attempt_key.request_key.admission_id}:{attempt_key.reverse_attempt_id}"
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

    def decide(
        self,
        request: PathDecisionRequest,
        prefill_local_tokens: int,
        num_preemptions: int = 0,
    ) -> PathDecisionResult:
        if not isinstance(request, PathDecisionRequest):
            raise PathDecisionValidationError("decision input must be a PathDecisionRequest")

        existing = self._decision_records.get(request.request_key)
        if existing is not None:
            if request != existing.request or prefill_local_tokens != existing.prefill_local_tokens:
                raise PathDecisionValidationError(
                    "request key is already associated with different token counts or prefill_local_tokens"
                )
            return existing.result

        path = PathKind.PE_READ if prefill_local_tokens >= request.decode_store_tokens else self._policy.choose(request)

        result = PathDecisionResult(
            request_key=request.request_key,
            path=path,
            reverse_attempt_id=num_preemptions if path is PathKind.DE_READ else None,
            prefill_local_tokens=prefill_local_tokens if path is PathKind.DE_READ else None,
        )
        self._decision_records[request.request_key] = _DecisionRecord(
            request=request,
            prefill_local_tokens=prefill_local_tokens,
            result=result,
        )
        return result

    def discard(self, request_key: DualPathRequestKey) -> None:
        self._decision_records.pop(request_key, None)

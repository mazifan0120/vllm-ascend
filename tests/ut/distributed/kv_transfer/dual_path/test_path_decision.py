# SPDX-License-Identifier: Apache-2.0

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionDecider,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
    PathPolicy,
    RoundRobinPathPolicy,
)


def _key(request_id: str = "request-1") -> DualPathRequestKey:
    return DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1",
        decode_request_id=request_id,
    )


def _make_request(
    *,
    request_id: str = "request-1",
    target_tokens: int = 32,
    decode_local_tokens: int = 0,
    decode_store_tokens: int = 16,
) -> PathDecisionRequest:
    return PathDecisionRequest(
        request_key=_key(request_id),
        target_tokens=target_tokens,
        decode_local_tokens=decode_local_tokens,
        decode_store_tokens=decode_store_tokens,
    )


class SpyPolicy:
    def __init__(self, path: PathKind) -> None:
        self._path = path
        self.choose_count = 0
        self.requests: list[PathDecisionRequest] = []

    def choose(self, request: PathDecisionRequest) -> PathKind:
        self.choose_count += 1
        self.requests.append(request)
        return self._path


class AlwaysPePolicy:
    def choose(self, request: PathDecisionRequest) -> PathKind:
        return PathKind.PE_READ


class AlwaysDePolicy:
    def choose(self, request: PathDecisionRequest) -> PathKind:
        return PathKind.DE_READ


def test_request_key_equality_and_hash() -> None:
    first = _key()
    equal_but_distinct = _key()
    other = _key("request-2")

    assert first == equal_but_distinct
    assert first is not equal_but_distinct
    assert hash(first) == hash(equal_but_distinct)
    assert len({first, equal_but_distinct, other}) == 2


@pytest.mark.parametrize(
    ("engine_id", "request_id"),
    [
        ("", "request-1"),
        ("decode-engine-1", ""),
        (1, "request-1"),
        ("decode-engine-1", 1),
        (None, "request-1"),
        ("decode-engine-1", None),
    ],
)
def test_request_key_rejects_empty_identity_fields(engine_id, request_id) -> None:
    with pytest.raises(PathDecisionValidationError):
        DualPathRequestKey(
            decode_engine_instance_id=engine_id,
            decode_request_id=request_id,
        )


@pytest.mark.parametrize(
    ("target_tokens", "decode_local_tokens", "decode_store_tokens"),
    [
        (32, 8, 16),
        (32, 8, 8),
    ],
)
def test_request_construction_accepts_partial_and_miss(
    target_tokens: int,
    decode_local_tokens: int,
    decode_store_tokens: int,
) -> None:
    request = _make_request(
        target_tokens=target_tokens,
        decode_local_tokens=decode_local_tokens,
        decode_store_tokens=decode_store_tokens,
    )

    assert request.request_key == _key()
    assert request.target_tokens == target_tokens
    assert request.decode_local_tokens == decode_local_tokens
    assert request.decode_store_tokens == decode_store_tokens


@pytest.mark.parametrize(
    ("target_tokens", "decode_local_tokens", "decode_store_tokens"),
    [
        (32, -1, 0),
        (32, 0, -1),
        (32, 8, 7),
        (32, 0, 33),
        (32, 8, 32),
        (32, 0, 32),
        (-1, 0, 0),
        (True, 0, 0),
        (32, False, 0),
        (32, 0, False),
        ("32", 0, 0),
        (32, 0.0, 0),
        (32, 0, None),
    ],
)
def test_request_rejects_invalid_token_ordering(
    target_tokens,
    decode_local_tokens,
    decode_store_tokens,
) -> None:
    with pytest.raises(PathDecisionValidationError):
        _make_request(
            target_tokens=target_tokens,
            decode_local_tokens=decode_local_tokens,
            decode_store_tokens=decode_store_tokens,
        )


def test_request_rejects_hbm_complete_local_equals_target() -> None:
    with pytest.raises(PathDecisionValidationError):
        _make_request(
            target_tokens=32,
            decode_local_tokens=32,
            decode_store_tokens=32,
        )


def test_eligibility_forces_pe_read_when_l_pe_equals_k_de() -> None:
    request = _make_request(decode_store_tokens=16)
    policy = MagicMock(spec=PathPolicy)
    policy.choose.return_value = PathKind.DE_READ

    result = PathDecisionDecider(policy).decide(request, 16)

    assert result == PathDecisionResult(request_key=request.request_key, path=PathKind.PE_READ)
    policy.choose.assert_not_called()


def test_eligibility_forces_pe_read_when_l_pe_exceeds_k_de() -> None:
    request = _make_request(decode_store_tokens=16)
    policy = MagicMock(spec=PathPolicy)
    policy.choose.return_value = PathKind.DE_READ

    result = PathDecisionDecider(policy).decide(request, 24)

    assert result == PathDecisionResult(request_key=request.request_key, path=PathKind.PE_READ)
    policy.choose.assert_not_called()


@pytest.mark.parametrize("selected_path", [PathKind.PE_READ, PathKind.DE_READ])
def test_eligibility_invokes_policy_once_when_l_pe_below_k_de(selected_path: PathKind) -> None:
    request = _make_request(decode_store_tokens=16)
    policy = MagicMock(spec=PathPolicy)
    policy.choose.return_value = selected_path

    result = PathDecisionDecider(policy).decide(request, 8)

    assert result == PathDecisionResult(request_key=request.request_key, path=selected_path)
    policy.choose.assert_called_once_with(request)


def test_forced_pe_read_does_not_advance_seeded_round_robin() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    singleton = _make_request(request_id="singleton", decode_store_tokens=16)
    ambiguous = _make_request(request_id="ambiguous", decode_store_tokens=16)

    singleton_result = decider.decide(singleton, 16)
    ambiguous_result = decider.decide(ambiguous, 0)

    assert singleton_result.path is PathKind.PE_READ
    assert ambiguous_result.path is PathKind.PE_READ


def test_round_robin_initial_choice_seeded_pe_read() -> None:
    request = _make_request()
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))

    result = decider.decide(request, 0)

    assert result == PathDecisionResult(request_key=request.request_key, path=PathKind.PE_READ)


def test_round_robin_initial_choice_seeded_de_read() -> None:
    request = _make_request()
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(0)))

    result = decider.decide(request, 0)

    assert result == PathDecisionResult(request_key=request.request_key, path=PathKind.DE_READ)


def test_ambiguous_requests_follow_seeded_round_robin() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    requests = [_make_request(request_id=f"request-{index}") for index in range(4)]

    results = [decider.decide(request, 0) for request in requests]

    assert results == [
        PathDecisionResult(request_key=requests[0].request_key, path=PathKind.PE_READ),
        PathDecisionResult(request_key=requests[1].request_key, path=PathKind.DE_READ),
        PathDecisionResult(request_key=requests[2].request_key, path=PathKind.PE_READ),
        PathDecisionResult(request_key=requests[3].request_key, path=PathKind.DE_READ),
    ]


def test_identical_replay_reuses_result_without_policy_turn() -> None:
    request = _make_request()
    equal_but_distinct = _make_request()
    policy = SpyPolicy(PathKind.PE_READ)
    decider = PathDecisionDecider(policy)

    first_result = decider.decide(request, 8)
    duplicate_result = decider.decide(equal_but_distinct, 8)

    assert equal_but_distinct == request
    assert equal_but_distinct is not request
    assert duplicate_result is first_result
    assert policy.choose_count == 1
    assert policy.requests == [request]


@pytest.mark.parametrize("conflict", ["prefill-local", "request-facts"])
def test_conflicting_l_pe_or_facts_retain_first_record(conflict: str) -> None:
    original = _make_request()
    conflicting = _make_request(decode_store_tokens=24) if conflict == "request-facts" else _make_request()
    conflicting_prefill_local_tokens = 9 if conflict == "prefill-local" else 8
    policy = SpyPolicy(PathKind.PE_READ)
    decider = PathDecisionDecider(policy)

    retained_result = decider.decide(original, 8)
    with pytest.raises(PathDecisionValidationError):
        decider.decide(conflicting, conflicting_prefill_local_tokens)
    retried_result = decider.decide(_make_request(), 8)

    assert retried_result is retained_result
    assert decider._decision_records[original.request_key].prefill_local_tokens == 8
    assert policy.choose_count == 1
    assert policy.requests == [original]


def test_decide_with_invalid_protocol_input_raises_without_retaining_record() -> None:
    invalid_request = SimpleNamespace(request_key=_key())
    policy = SpyPolicy(PathKind.PE_READ)
    decider = PathDecisionDecider(policy)

    with pytest.raises(PathDecisionValidationError):
        decider.decide(invalid_request, 0)

    assert decider._decision_records == {}
    assert policy.choose_count == 0


def test_decide_with_invalid_protocol_input_without_key_raises_validation_error() -> None:
    policy = SpyPolicy(PathKind.PE_READ)
    decider = PathDecisionDecider(policy)

    with pytest.raises(PathDecisionValidationError):
        decider.decide(SimpleNamespace(), 0)

    assert decider._decision_records == {}
    assert policy.choose_count == 0


def test_discard_removes_retained_result_record() -> None:
    request = _make_request()
    policy = SpyPolicy(PathKind.PE_READ)
    decider = PathDecisionDecider(policy)
    decider.decide(request, 0)

    decider.discard(request.request_key)
    decider.decide(request, 0)

    assert policy.choose_count == 2


def test_discard_is_idempotent_and_does_not_rewind_policy() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    first = _make_request(request_id="request-1")
    second = _make_request(request_id="request-2")
    assert decider.decide(first, 0).path is PathKind.PE_READ

    decider.discard(first.request_key)
    decider.discard(first.request_key)

    assert decider.decide(second, 0).path is PathKind.DE_READ


def test_second_policy_satisfies_path_policy_without_caller_change() -> None:
    policies: tuple[tuple[PathPolicy, PathKind], ...] = (
        (AlwaysPePolicy(), PathKind.PE_READ),
        (AlwaysDePolicy(), PathKind.DE_READ),
    )

    def decide(policy: PathPolicy, request: PathDecisionRequest) -> PathDecisionResult:
        return PathDecisionDecider(policy).decide(request, 0)

    results = [
        decide(policy, _make_request(request_id=f"request-{index}")) for index, (policy, _) in enumerate(policies)
    ]

    assert results == [
        PathDecisionResult(request_key=_key(f"request-{index}"), path=expected_path)
        for index, (_, expected_path) in enumerate(policies)
    ]


def test_request_key_serialization_round_trips_both_directions() -> None:
    request_key = _key()
    payload = {
        "decode_engine_instance_id": "decode-engine-1",
        "decode_request_id": "request-1",
    }

    assert DualPathRequestKey.from_dict(request_key.to_dict()) == request_key
    assert DualPathRequestKey.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize(
    ("decode_local_tokens", "decode_store_tokens"),
    [
        pytest.param(8, 16, id="partial"),
        pytest.param(8, 8, id="miss"),
    ],
)
def test_request_serialization_round_trips_both_directions(
    decode_local_tokens: int,
    decode_store_tokens: int,
) -> None:
    request = _make_request(
        decode_local_tokens=decode_local_tokens,
        decode_store_tokens=decode_store_tokens,
    )
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "request-1",
        },
        "target_tokens": 32,
        "decode_local_tokens": decode_local_tokens,
        "decode_store_tokens": decode_store_tokens,
    }

    assert PathDecisionRequest.from_dict(request.to_dict()) == request
    assert PathDecisionRequest.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize("path", [PathKind.PE_READ, PathKind.DE_READ])
def test_result_serialization_round_trips_both_directions(path: PathKind) -> None:
    result = PathDecisionResult(request_key=_key(), path=path)
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "request-1",
        },
        "path": path.value,
    }

    assert PathDecisionResult.from_dict(result.to_dict()) == result
    assert PathDecisionResult.from_dict(payload).to_dict() == payload


def test_from_dict_rejects_unknown_path_value() -> None:
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "request-1",
        },
        "path": "UNKNOWN",
    }

    with pytest.raises(PathDecisionValidationError):
        PathDecisionResult.from_dict(payload)


@pytest.mark.parametrize(
    ("protocol_type", "payload"),
    [
        (
            DualPathRequestKey,
            {"decode_engine_instance_id": "decode-engine-1"},
        ),
        (
            PathDecisionRequest,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "target_tokens": 32,
                "decode_local_tokens": 8,
            },
        ),
        (
            PathDecisionResult,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                }
            },
        ),
    ],
)
def test_from_dict_rejects_missing_fields(protocol_type, payload) -> None:
    with pytest.raises(PathDecisionValidationError):
        protocol_type.from_dict(payload)


@pytest.mark.parametrize(
    ("protocol_type", "payload"),
    [
        (
            DualPathRequestKey,
            {
                "decode_engine_instance_id": "decode-engine-1",
                "decode_request_id": "request-1",
                "extra": "value",
            },
        ),
        (
            PathDecisionRequest,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "target_tokens": 32,
                "decode_local_tokens": 8,
                "decode_store_tokens": 16,
                "extra": "value",
            },
        ),
        (
            PathDecisionResult,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "path": "PE_READ",
                "extra": "value",
            },
        ),
    ],
)
def test_from_dict_rejects_extra_fields(protocol_type, payload) -> None:
    with pytest.raises(PathDecisionValidationError):
        protocol_type.from_dict(payload)


@pytest.mark.parametrize(
    ("engine_id", "request_id"),
    [
        ("", "request-1"),
        ("decode-engine-1", ""),
    ],
)
def test_from_dict_rejects_empty_identity(engine_id: str, request_id: str) -> None:
    with pytest.raises(PathDecisionValidationError):
        DualPathRequestKey.from_dict(
            {
                "decode_engine_instance_id": engine_id,
                "decode_request_id": request_id,
            }
        )


@pytest.mark.parametrize(
    ("target_tokens", "decode_local_tokens", "decode_store_tokens"),
    [
        (True, 0, 0),
        (32, False, 0),
        (32, 0, False),
    ],
)
def test_from_dict_rejects_bool_for_int(
    target_tokens,
    decode_local_tokens,
    decode_store_tokens,
) -> None:
    with pytest.raises(PathDecisionValidationError):
        PathDecisionRequest.from_dict(
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "target_tokens": target_tokens,
                "decode_local_tokens": decode_local_tokens,
                "decode_store_tokens": decode_store_tokens,
            }
        )


@pytest.mark.parametrize(
    ("target_tokens", "decode_local_tokens", "decode_store_tokens"),
    [
        (32, -1, 0),
        (32, 0, -1),
        (32, 8, 7),
        (32, 0, 33),
        (32, 8, 32),
        (32, 0, 32),
        (-1, 0, 0),
        (32, 32, 32),
    ],
)
def test_from_dict_rejects_invalid_token_ordering(
    target_tokens: int,
    decode_local_tokens: int,
    decode_store_tokens: int,
) -> None:
    with pytest.raises(PathDecisionValidationError):
        PathDecisionRequest.from_dict(
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "target_tokens": target_tokens,
                "decode_local_tokens": decode_local_tokens,
                "decode_store_tokens": decode_store_tokens,
            }
        )


@pytest.mark.parametrize(
    ("protocol_type", "payload"),
    [
        (DualPathRequestKey, None),
        (PathDecisionRequest, []),
        (PathDecisionResult, "payload"),
    ],
)
def test_from_dict_rejects_non_dict_payload(protocol_type, payload) -> None:
    with pytest.raises(PathDecisionValidationError):
        protocol_type.from_dict(payload)


@pytest.mark.parametrize(
    ("protocol_type", "payload"),
    [
        (
            DualPathRequestKey,
            {
                "decode_engine_instance_id": 1,
                "decode_request_id": "request-1",
            },
        ),
        (
            DualPathRequestKey,
            {
                "decode_engine_instance_id": "decode-engine-1",
                "decode_request_id": 1,
            },
        ),
        (
            PathDecisionRequest,
            {
                "request_key": _key(),
                "target_tokens": 32,
                "decode_local_tokens": 8,
                "decode_store_tokens": 16,
            },
        ),
        (
            PathDecisionRequest,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "target_tokens": "32",
                "decode_local_tokens": 8,
                "decode_store_tokens": 16,
            },
        ),
        (
            PathDecisionRequest,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "target_tokens": 32,
                "decode_local_tokens": 8.0,
                "decode_store_tokens": 16,
            },
        ),
        (
            PathDecisionResult,
            {
                "request_key": _key(),
                "path": "PE_READ",
            },
        ),
        (
            PathDecisionResult,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "path": 1,
            },
        ),
    ],
)
def test_from_dict_rejects_wrong_field_types(protocol_type, payload) -> None:
    with pytest.raises(PathDecisionValidationError):
        protocol_type.from_dict(payload)

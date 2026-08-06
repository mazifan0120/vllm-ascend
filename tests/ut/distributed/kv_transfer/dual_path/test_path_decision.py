# SPDX-License-Identifier: Apache-2.0

import random
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    Path,
    PathDecisionDecider,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
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
    decode_store_tokens: int = 0,
) -> PathDecisionRequest:
    return PathDecisionRequest(
        request_key=_key(request_id),
        target_tokens=target_tokens,
        decode_local_tokens=decode_local_tokens,
        decode_store_tokens=decode_store_tokens,
    )


class SpyPolicy:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.choose_count = 0
        self.requests: list[PathDecisionRequest] = []

    def choose(self, request: PathDecisionRequest) -> Path:
        self.choose_count += 1
        self.requests.append(request)
        return self._path


class InvalidResultPolicy:
    def __init__(self, result: str | None) -> None:
        self._result = result
        self.choose_count = 0

    def choose(self, request: PathDecisionRequest) -> str | None:
        self.choose_count += 1
        return self._result


class AlwaysPePolicy:
    def choose(self, request: PathDecisionRequest) -> Path:
        return Path.PE_READ


class AlwaysDePolicy:
    def choose(self, request: PathDecisionRequest) -> Path:
        return Path.DE_READ


class ExplodingPolicy:
    def __init__(self) -> None:
        self.choose_count = 0

    def choose(self, request: PathDecisionRequest) -> Path:
        self.choose_count += 1
        raise RuntimeError("policy failed")


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


@pytest.mark.parametrize(
    ("decode_local_tokens", "decode_store_tokens"),
    [
        pytest.param(8, 16, id="partial"),
        pytest.param(8, 8, id="miss"),
        pytest.param(0, 0, id="unavailable-equivalent"),
    ],
)
def test_non_full_requests_invoke_policy(
    decode_local_tokens: int,
    decode_store_tokens: int,
) -> None:
    request = _make_request(
        decode_local_tokens=decode_local_tokens,
        decode_store_tokens=decode_store_tokens,
    )
    policy = SpyPolicy(Path.PE_READ)

    result = PathDecisionDecider(policy).decide(request)

    assert result == PathDecisionResult(request_key=request.request_key, path=Path.PE_READ)
    assert policy.choose_count == 1
    assert policy.requests == [request]


def test_round_robin_initial_choice_seeded_pe_read() -> None:
    request = _make_request()
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))

    result = decider.decide(request)

    assert result == PathDecisionResult(request_key=request.request_key, path=Path.PE_READ)


def test_round_robin_initial_choice_seeded_de_read() -> None:
    request = _make_request()
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(0)))

    result = decider.decide(request)

    assert result == PathDecisionResult(request_key=request.request_key, path=Path.DE_READ)


def test_round_robin_alternates_across_unique_requests() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    requests = [_make_request(request_id=f"request-{index}") for index in range(4)]

    results = [decider.decide(request) for request in requests]

    assert results == [
        PathDecisionResult(request_key=requests[0].request_key, path=Path.PE_READ),
        PathDecisionResult(request_key=requests[1].request_key, path=Path.DE_READ),
        PathDecisionResult(request_key=requests[2].request_key, path=Path.PE_READ),
        PathDecisionResult(request_key=requests[3].request_key, path=Path.DE_READ),
    ]


def test_identical_duplicate_returns_retained_result_without_choose() -> None:
    request = _make_request()
    equal_but_distinct = _make_request()
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)

    first_result = decider.decide(request)
    duplicate_result = decider.decide(equal_but_distinct)

    assert equal_but_distinct == request
    assert equal_but_distinct is not request
    assert duplicate_result is first_result
    assert policy.choose_count == 1
    assert policy.requests == [request]


def test_conflicting_duplicate_raises_and_preserves_original_result_without_choose() -> None:
    original = _make_request()
    conflicting = _make_request(decode_store_tokens=8)
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)

    retained_result = decider.decide(original)
    with pytest.raises(PathDecisionValidationError):
        decider.decide(conflicting)
    retried_result = decider.decide(_make_request())

    assert retried_result is retained_result
    assert policy.choose_count == 1
    assert policy.requests == [original]


def test_decide_with_invalid_protocol_input_raises_without_retaining_record() -> None:
    invalid_request = SimpleNamespace(request_key=_key())
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)

    with pytest.raises(PathDecisionValidationError):
        decider.decide(invalid_request)

    assert decider._decision_records == {}
    assert policy.choose_count == 0


def test_decide_with_invalid_protocol_input_without_key_raises_validation_error() -> None:
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)

    with pytest.raises(PathDecisionValidationError):
        decider.decide(SimpleNamespace())

    assert decider._decision_records == {}
    assert policy.choose_count == 0


@pytest.mark.parametrize("invalid_result", ["PE_READ", None])
def test_policy_result_not_a_path_raises_and_retains_failure(invalid_result: str | None) -> None:
    request = _make_request()
    policy = InvalidResultPolicy(invalid_result)
    decider = PathDecisionDecider(policy)

    with pytest.raises(PathDecisionValidationError):
        decider.decide(request)
    retained = decider._decision_records[request.request_key]
    with pytest.raises(PathDecisionValidationError):
        decider.decide(_make_request())

    assert retained.request == request
    assert retained.result is None
    assert policy.choose_count == 1


def test_policy_exception_is_chained_retained_and_invoked_once() -> None:
    request = _make_request()
    policy = ExplodingPolicy()
    decider = PathDecisionDecider(policy)

    with pytest.raises(PathDecisionValidationError) as raised:
        decider.decide(request)
    with pytest.raises(PathDecisionValidationError):
        decider.decide(_make_request())

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "policy failed"
    assert decider._decision_records[request.request_key].result is None
    assert policy.choose_count == 1


def test_discard_removes_retained_result_record() -> None:
    request = _make_request()
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)
    decider.decide(request)

    decider.discard(request.request_key)
    decider.decide(request)

    assert policy.choose_count == 2


def test_discard_removes_failure_record() -> None:
    request = _make_request()
    policy = InvalidResultPolicy("PE_READ")
    decider = PathDecisionDecider(policy)
    with pytest.raises(PathDecisionValidationError):
        decider.decide(request)

    decider.discard(request.request_key)
    with pytest.raises(PathDecisionValidationError):
        decider.decide(request)

    assert policy.choose_count == 2


def test_discard_is_idempotent_and_does_not_rewind_policy() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    first = _make_request(request_id="request-1")
    second = _make_request(request_id="request-2")
    assert decider.decide(first).path is Path.PE_READ

    decider.discard(first.request_key)
    decider.discard(first.request_key)

    assert decider.decide(second).path is Path.DE_READ


def test_retained_failure_replay_raises_without_policy() -> None:
    request = _make_request()
    policy = InvalidResultPolicy(None)
    decider = PathDecisionDecider(policy)
    with pytest.raises(PathDecisionValidationError):
        decider.decide(request)

    with pytest.raises(PathDecisionValidationError):
        decider.decide(_make_request())

    assert policy.choose_count == 1


def test_second_policy_satisfies_path_policy_without_caller_change() -> None:
    policies: tuple[tuple[PathPolicy, Path], ...] = (
        (AlwaysPePolicy(), Path.PE_READ),
        (AlwaysDePolicy(), Path.DE_READ),
    )

    def decide(policy: PathPolicy, request: PathDecisionRequest) -> PathDecisionResult:
        return PathDecisionDecider(policy).decide(request)

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


@pytest.mark.parametrize("path", [Path.PE_READ, Path.DE_READ])
def test_result_serialization_round_trips_both_directions(path: Path) -> None:
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

# SPDX-License-Identifier: Apache-2.0

import random
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    ERROR_CONFLICTING_REQUEST,
    ERROR_INVALID_POLICY_RESULT,
    ERROR_INVALID_REQUEST,
    DualPathRequestKey,
    Path,
    PathDecisionCommit,
    PathDecisionDecider,
    PathDecisionError,
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
        (32, 8, 32),
        (32, 8, 8),
    ],
)
def test_request_construction_accepts_partial_full_and_miss(
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


def test_full_store_hit_returns_de_read_without_invoking_policy() -> None:
    request = _make_request(decode_local_tokens=8, decode_store_tokens=32)
    policy = SpyPolicy(Path.PE_READ)

    result = PathDecisionDecider(policy).decide(request)

    assert result == PathDecisionCommit(request_key=request.request_key, path=Path.DE_READ)
    assert policy.choose_count == 0
    assert policy.requests == []


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

    assert result == PathDecisionCommit(request_key=request.request_key, path=Path.PE_READ)
    assert policy.choose_count == 1
    assert policy.requests == [request]


def test_round_robin_initial_choice_seeded_pe_read() -> None:
    request = _make_request()
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))

    result = decider.decide(request)

    assert result == PathDecisionCommit(request_key=request.request_key, path=Path.PE_READ)


def test_round_robin_initial_choice_seeded_de_read() -> None:
    request = _make_request()
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(0)))

    result = decider.decide(request)

    assert result == PathDecisionCommit(request_key=request.request_key, path=Path.DE_READ)


def test_round_robin_alternates_across_unique_requests() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    requests = [_make_request(request_id=f"request-{index}") for index in range(4)]

    results = [decider.decide(request) for request in requests]

    assert results == [
        PathDecisionCommit(request_key=requests[0].request_key, path=Path.PE_READ),
        PathDecisionCommit(request_key=requests[1].request_key, path=Path.DE_READ),
        PathDecisionCommit(request_key=requests[2].request_key, path=Path.PE_READ),
        PathDecisionCommit(request_key=requests[3].request_key, path=Path.DE_READ),
    ]


def test_full_requests_do_not_advance_round_robin_state() -> None:
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))
    full_request = _make_request(
        request_id="full-request",
        decode_local_tokens=8,
        decode_store_tokens=32,
    )
    first_non_full = _make_request(request_id="first-non-full")
    second_non_full = _make_request(request_id="second-non-full")

    results = [
        decider.decide(full_request),
        decider.decide(first_non_full),
        decider.decide(second_non_full),
    ]

    assert results == [
        PathDecisionCommit(request_key=full_request.request_key, path=Path.DE_READ),
        PathDecisionCommit(request_key=first_non_full.request_key, path=Path.PE_READ),
        PathDecisionCommit(request_key=second_non_full.request_key, path=Path.DE_READ),
    ]


def test_identical_duplicate_returns_retained_commit_without_choose() -> None:
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


def test_identical_full_hit_duplicate_returns_retained_commit_without_choose() -> None:
    full_request = _make_request(decode_store_tokens=32)
    equal_but_distinct = _make_request(decode_store_tokens=32)
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)

    first_result = decider.decide(full_request)
    duplicate_result = decider.decide(equal_but_distinct)

    assert first_result.path is Path.DE_READ
    assert duplicate_result is first_result
    assert policy.choose_count == 0


def test_conflicting_duplicate_returns_typed_error_without_choose() -> None:
    original = _make_request()
    conflicting = _make_request(decode_store_tokens=8)
    policy = SpyPolicy(Path.PE_READ)
    decider = PathDecisionDecider(policy)

    retained_commit = decider.decide(original)
    conflict = decider.decide(conflicting)
    retried_commit = decider.decide(_make_request())

    assert isinstance(conflict, PathDecisionError)
    assert conflict.request_key == original.request_key
    assert conflict.error_code == ERROR_CONFLICTING_REQUEST
    assert conflict.message
    assert retried_commit is retained_commit
    assert policy.choose_count == 1
    assert policy.requests == [original]


def test_decide_with_invalid_protocol_input_returns_typed_error() -> None:
    invalid_request = SimpleNamespace(request_key=_key())
    policy = SpyPolicy(Path.PE_READ)

    result = PathDecisionDecider(policy).decide(invalid_request)

    assert isinstance(result, PathDecisionError)
    assert result.request_key == _key()
    assert result.error_code == ERROR_INVALID_REQUEST
    assert result.message
    assert policy.choose_count == 0


def test_decide_with_invalid_protocol_input_without_key_raises_validation_error() -> None:
    policy = SpyPolicy(Path.PE_READ)

    with pytest.raises(PathDecisionValidationError):
        PathDecisionDecider(policy).decide(SimpleNamespace())

    assert policy.choose_count == 0


@pytest.mark.parametrize("invalid_result", ["PE_READ", None])
def test_policy_result_not_a_path_returns_typed_error(invalid_result: str | None) -> None:
    request = _make_request()
    policy = InvalidResultPolicy(invalid_result)
    decider = PathDecisionDecider(policy)

    first_result = decider.decide(request)
    retry_result = decider.decide(_make_request())

    assert isinstance(first_result, PathDecisionError)
    assert first_result.request_key == request.request_key
    assert first_result.error_code == ERROR_INVALID_POLICY_RESULT
    assert first_result.message
    assert retry_result == first_result
    assert policy.choose_count == 2


def test_policy_exception_propagates_and_is_not_retained() -> None:
    policy = ExplodingPolicy()
    decider = PathDecisionDecider(policy)

    with pytest.raises(RuntimeError, match="policy failed"):
        decider.decide(_make_request())
    with pytest.raises(RuntimeError, match="policy failed"):
        decider.decide(_make_request())

    assert policy.choose_count == 2


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
        PathDecisionCommit(request_key=_key(f"request-{index}"), path=expected_path)
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
        pytest.param(8, 32, id="full"),
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
def test_commit_serialization_round_trips_both_directions(path: Path) -> None:
    commit = PathDecisionCommit(request_key=_key(), path=path)
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "request-1",
        },
        "path": path.value,
    }

    assert PathDecisionCommit.from_dict(commit.to_dict()) == commit
    assert PathDecisionCommit.from_dict(payload).to_dict() == payload


def test_error_serialization_round_trips_both_directions() -> None:
    error = PathDecisionError(
        request_key=_key(),
        error_code=ERROR_INVALID_REQUEST,
        message="invalid decision input",
    )
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "request-1",
        },
        "error_code": "INVALID_REQUEST",
        "message": "invalid decision input",
    }

    assert PathDecisionError.from_dict(error.to_dict()) == error
    assert PathDecisionError.from_dict(payload).to_dict() == payload


def test_from_dict_rejects_unknown_path_value() -> None:
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "request-1",
        },
        "path": "UNKNOWN",
    }

    with pytest.raises(PathDecisionValidationError):
        PathDecisionCommit.from_dict(payload)


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
            PathDecisionCommit,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                }
            },
        ),
        (
            PathDecisionError,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "error_code": "INVALID_REQUEST",
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
            PathDecisionCommit,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "path": "PE_READ",
                "extra": "value",
            },
        ),
        (
            PathDecisionError,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "error_code": "INVALID_REQUEST",
                "message": "invalid decision input",
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
        (PathDecisionCommit, "payload"),
        (PathDecisionError, _key()),
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
            PathDecisionCommit,
            {
                "request_key": _key(),
                "path": "PE_READ",
            },
        ),
        (
            PathDecisionCommit,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "path": 1,
            },
        ),
        (
            PathDecisionError,
            {
                "request_key": [],
                "error_code": "INVALID_REQUEST",
                "message": "invalid decision input",
            },
        ),
        (
            PathDecisionError,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "error_code": 1,
                "message": "invalid decision input",
            },
        ),
        (
            PathDecisionError,
            {
                "request_key": {
                    "decode_engine_instance_id": "decode-engine-1",
                    "decode_request_id": "request-1",
                },
                "error_code": "INVALID_REQUEST",
                "message": None,
            },
        ),
    ],
)
def test_from_dict_rejects_wrong_field_types(protocol_type, payload) -> None:
    with pytest.raises(PathDecisionValidationError):
        protocol_type.from_dict(payload)

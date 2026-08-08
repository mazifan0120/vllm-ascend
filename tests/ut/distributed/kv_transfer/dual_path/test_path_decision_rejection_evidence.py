import random

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    Path,
    PathDecisionDecider,
    PathDecisionRequest,
    PathDecisionValidationError,
    RoundRobinPathPolicy,
)


class _SpyPolicy:
    def __init__(self) -> None:
        self.choose_count = 0

    def choose(self, request: PathDecisionRequest) -> Path:
        self.choose_count += 1
        return Path.PE_READ


def test_store_full_request_construction_rejects_before_decider_policy_choice() -> None:
    # Given
    policy = _SpyPolicy()
    decider = PathDecisionDecider(policy)
    request_key = DualPathRequestKey("decode-engine", "store-full")

    # When
    with pytest.raises(PathDecisionValidationError):
        decider.decide(
            PathDecisionRequest(
                request_key=request_key,
                target_tokens=32,
                decode_local_tokens=16,
                decode_store_tokens=32,
            ),
            0,
        )

    # Then
    assert policy.choose_count == 0
    assert decider._decision_records == {}


def test_store_full_rejection_does_not_advance_seeded_round_robin_first_choice() -> None:
    # Given
    decider = PathDecisionDecider(RoundRobinPathPolicy(random.Random(1)))

    # When
    with pytest.raises(PathDecisionValidationError):
        decider.decide(
            PathDecisionRequest(
                request_key=DualPathRequestKey("decode-engine", "store-full"),
                target_tokens=32,
                decode_local_tokens=16,
                decode_store_tokens=32,
            ),
            0,
        )
    result = decider.decide(
        PathDecisionRequest(
            request_key=DualPathRequestKey("decode-engine", "non-full"),
            target_tokens=32,
            decode_local_tokens=16,
            decode_store_tokens=24,
        ),
        0,
    )

    # Then
    assert result.path is Path.PE_READ

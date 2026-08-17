# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import FrozenInstanceError, fields

import pytest

import vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata as metadata_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
    ForwardPlan,
    ForwardReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionValidationError,
    PathKind,
)


def _key(request_id: str = "request-1") -> DualPathRequestKey:
    return DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1",
        decode_request_id=request_id,
        admission_id=0,
    )


def _make_plan(
    *,
    token_start: int = 32,
    token_end: int = 51,
    source_block_ids: tuple[tuple[int, ...], ...] = ((10, 11, 12, 13),),
    destination_block_ids: tuple[tuple[int, ...], ...] = ((20, 21, 22, 23),),
) -> ForwardPlan:
    return ForwardPlan(
        request_key=_key(),
        token_start=token_start,
        token_end=token_end,
        source_block_ids=source_block_ids,
        destination_block_ids=destination_block_ids,
    )


def _make_binding(
    *,
    token_start: int = 32,
    token_end: int = 51,
    destination_block_ids: tuple[tuple[int, ...], ...] = ((20, 21, 22, 23),),
) -> ForwardReceiveBinding:
    return ForwardReceiveBinding(
        request_key=_key(),
        path=PathKind.PE_READ,
        wire_request_id="wire-request-1",
        decode_request_id="request-1",
        destination_block_ids=destination_block_ids,
        token_start=token_start,
        token_end=token_end,
    )


def test_forward_plan_deep_freezes_both_block_tables():
    plan = _make_plan()

    assert plan.source_block_ids == ((10, 11, 12, 13),)
    assert plan.destination_block_ids == ((20, 21, 22, 23),)
    assert isinstance(plan.source_block_ids, tuple)
    assert isinstance(plan.destination_block_ids, tuple)
    assert all(isinstance(group, tuple) for group in plan.source_block_ids)
    assert all(isinstance(group, tuple) for group in plan.destination_block_ids)

    with pytest.raises(FrozenInstanceError):
        plan.source_block_ids = ((1,),)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.destination_block_ids = ((1,),)  # type: ignore[misc]


def test_forward_receive_binding_deep_freezes_destination_blocks():
    binding = _make_binding()

    assert binding.destination_block_ids == ((20, 21, 22, 23),)
    assert isinstance(binding.destination_block_ids, tuple)
    assert all(isinstance(group, tuple) for group in binding.destination_block_ids)

    with pytest.raises(FrozenInstanceError):
        binding.destination_block_ids = ((1,),)  # type: ignore[misc]


def test_mutating_constructor_inputs_does_not_leak_into_retained_objects():
    source = [[10, 11, 12, 13]]
    destination = [[20, 21, 22, 23]]
    plan = ForwardPlan(
        request_key=_key(),
        token_start=32,
        token_end=51,
        source_block_ids=source,
        destination_block_ids=destination,
    )

    binding_destination = [[5, 6, 7]]
    binding = ForwardReceiveBinding(
        request_key=_key(),
        path=PathKind.PE_READ,
        wire_request_id="wire-request-1",
        decode_request_id="request-1",
        destination_block_ids=binding_destination,
        token_start=32,
        token_end=51,
    )

    source[0].append(99)
    source.append([1, 2])
    destination[0][0] = -1
    destination.append([8, 9])
    binding_destination[0].append(999)
    binding_destination[0][0] = -5

    assert plan.source_block_ids == ((10, 11, 12, 13),)
    assert plan.destination_block_ids == ((20, 21, 22, 23),)
    assert binding.destination_block_ids == ((5, 6, 7),)


def test_plan_rejects_invalid_range_misaligned_start_and_empty_groups():
    with pytest.raises(PathDecisionValidationError):
        _make_plan(token_start=-1)
    with pytest.raises(PathDecisionValidationError):
        _make_plan(token_start=51, token_end=51)
    with pytest.raises(PathDecisionValidationError):
        _make_plan(token_start=52, token_end=51)
    with pytest.raises(PathDecisionValidationError):
        _make_plan(token_start=True)
    with pytest.raises(PathDecisionValidationError):
        _make_plan(source_block_ids=())
    with pytest.raises(PathDecisionValidationError):
        _make_plan(destination_block_ids=())
    with pytest.raises(PathDecisionValidationError):
        _make_plan(source_block_ids=((),))
    with pytest.raises(PathDecisionValidationError):
        _make_plan(destination_block_ids=((),))
    with pytest.raises(PathDecisionValidationError):
        _make_plan(source_block_ids=((10, "not-an-int"),))

    with pytest.raises(PathDecisionValidationError):
        _make_binding(token_start=-1)
    with pytest.raises(PathDecisionValidationError):
        _make_binding(token_start=51, token_end=51)
    with pytest.raises(PathDecisionValidationError):
        _make_binding(destination_block_ids=())
    with pytest.raises(PathDecisionValidationError):
        _make_binding(destination_block_ids=((),))


def test_plan_and_binding_carry_no_layer_rank_endpoint_or_engine_objects():
    forbidden_substrings = (
        "layer",
        "rank",
        "address",
        "endpoint",
        "engine",
        "session",
        "transferengine",
    )
    for cls in (ForwardPlan, ForwardReceiveBinding):
        for name in fields(cls):
            lowered = name.name.lower()
            assert not any(needle in lowered for needle in forbidden_substrings), (
                f"{cls.__name__} field {name.name!r} references a forbidden runtime object"
            )


def test_forward_plan_methods_have_no_synchronous_waits():
    source = inspect.getsource(metadata_module)
    for forbidden in (".result(", ".wait(", "time.sleep"):
        assert forbidden not in source, f"metadata.py contains forbidden synchronous wait {forbidden!r}"


def test_dual_path_connector_metadata_initializes_forward_receive_bindings():
    metadata = DualPathConnectorMetadata()

    assert metadata.forward_receive_bindings == []
    assert metadata.control_failures == []


@pytest.mark.parametrize("reason", list(DualPathControlFailureReason))
def test_control_failure_metadata_carries_reason(reason):
    invalid_block_ids = [41, 42, 43]
    metadata = DualPathControlFailureMetadata(
        request_id="request-1",
        invalid_block_ids=invalid_block_ids,
        reason=reason,
    )

    invalid_block_ids.append(44)

    assert metadata.invalid_block_ids == (41, 42, 43)
    assert metadata.reason is reason
    with pytest.raises(FrozenInstanceError):
        metadata.reason = DualPathControlFailureReason.ACTIVATION_FAILED


def test_control_failure_metadata_rejects_bad_reason():
    with pytest.raises(PathDecisionValidationError, match="reason must be"):
        DualPathControlFailureMetadata(
            request_id="request-1",
            invalid_block_ids=(41,),
            reason="NOT_A_REASON",
        )


def test_dual_path_connector_metadata_decode_store_metadata_defaults_none():
    metadata = DualPathConnectorMetadata()

    assert metadata.decode_store_metadata is None

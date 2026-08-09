# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from typing import TypeAlias

import pytest

import vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata as metadata_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ForwardReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionValidationError,
    PathKind,
)

BlockTableInput: TypeAlias = tuple[tuple[int, ...], ...] | list[list[int]]
BlockSizesInput: TypeAlias = tuple[int, ...] | list[int]


def _key() -> DualPathRequestKey:
    return DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1",
        decode_request_id="decode-request-1",
    )


@dataclass(frozen=True, slots=True)
class _ReversePlanInput:
    request_key: DualPathRequestKey
    wire_request_id: str
    token_start: int
    token_end: int
    source_block_ids: BlockTableInput
    destination_block_ids: BlockTableInput
    remote_engine_id: str
    remote_host: str
    remote_port: int
    remote_block_sizes: BlockSizesInput
    remote_tp_size: int
    remote_pcp_size: int
    remote_dcp_size: int

    def build(self) -> metadata_module.ReversePlan:
        return metadata_module.ReversePlan(
            request_key=self.request_key,
            wire_request_id=self.wire_request_id,
            token_start=self.token_start,
            token_end=self.token_end,
            source_block_ids=self.source_block_ids,
            destination_block_ids=self.destination_block_ids,
            remote_engine_id=self.remote_engine_id,
            remote_host=self.remote_host,
            remote_port=self.remote_port,
            remote_block_sizes=self.remote_block_sizes,
            remote_tp_size=self.remote_tp_size,
            remote_pcp_size=self.remote_pcp_size,
            remote_dcp_size=self.remote_dcp_size,
        )


def _valid_reverse_plan_input() -> _ReversePlanInput:
    return _ReversePlanInput(
        request_key=_key(),
        wire_request_id="wire-request-1",
        token_start=16,
        token_end=64,
        source_block_ids=((10, 11, 12, 13),),
        destination_block_ids=((20, 21, 22, 23),),
        remote_engine_id="prefill-engine-1",
        remote_host="192.0.2.10",
        remote_port=5000,
        remote_block_sizes=(16,),
        remote_tp_size=2,
        remote_pcp_size=1,
        remote_dcp_size=1,
    )


def _make_reverse_binding(
    destination_block_ids: BlockTableInput = ((20, 21, 22, 23),),
    token_start: int = 16,
    token_end: int = 64,
) -> metadata_module.ReverseReceiveBinding:
    return metadata_module.ReverseReceiveBinding(
        request_key=_key(),
        wire_request_id="wire-request-1",
        prefill_request_id="prefill-request-1",
        destination_block_ids=destination_block_ids,
        token_start=token_start,
        token_end=token_end,
    )


def test_forward_receive_binding_path_is_required_and_typed() -> None:
    with pytest.raises(TypeError):
        ForwardReceiveBinding(
            request_key=_key(),
            wire_request_id="wire-request-1",
            decode_request_id="decode-request-1",
            destination_block_ids=((20, 21, 22, 23),),
            token_start=32,
            token_end=64,
        )

    for path in (PathKind.PE_READ, PathKind.DE_READ):
        binding = ForwardReceiveBinding(
            request_key=_key(),
            path=path,
            wire_request_id="wire-request-1",
            decode_request_id="decode-request-1",
            destination_block_ids=((20, 21, 22, 23),),
            token_start=32,
            token_end=64,
        )
        assert binding.path is path

    for invalid_path in (PathKind.PE_READ.value, None):
        with pytest.raises(PathDecisionValidationError):
            ForwardReceiveBinding(
                request_key=_key(),
                path=invalid_path,
                wire_request_id="wire-request-1",
                decode_request_id="decode-request-1",
                destination_block_ids=((20, 21, 22, 23),),
                token_start=32,
                token_end=64,
            )


def test_reverse_plan_deep_freezes_tables_and_peer_facts() -> None:
    source = [[10, 11, 12, 13]]
    destination = [[20, 21, 22, 23]]
    remote_block_sizes = [16]
    plan = replace(
        _valid_reverse_plan_input(),
        source_block_ids=source,
        destination_block_ids=destination,
        remote_block_sizes=remote_block_sizes,
    ).build()

    source[0].append(99)
    destination[0][0] = -1
    remote_block_sizes[0] = 32

    assert plan.source_block_ids == ((10, 11, 12, 13),)
    assert plan.destination_block_ids == ((20, 21, 22, 23),)
    assert plan.remote_block_sizes == (16,)
    assert plan.remote_engine_id == "prefill-engine-1"
    assert plan.remote_host == "192.0.2.10"
    assert plan.remote_port == 5000
    with pytest.raises(FrozenInstanceError):
        plan.remote_port = 5001


def test_reverse_plan_wire_round_trip() -> None:
    payload = {
        "request_key": {
            "decode_engine_instance_id": "decode-engine-1",
            "decode_request_id": "decode-request-1",
        },
        "wire_request_id": "wire-request-1",
        "token_start": 16,
        "token_end": 64,
        "source_block_ids": [[10, 11, 12, 13]],
        "destination_block_ids": [[20, 21, 22, 23]],
        "remote_engine_id": "prefill-engine-1",
        "remote_host": "192.0.2.10",
        "remote_port": 5000,
        "remote_block_sizes": [16],
        "remote_tp_size": 2,
        "remote_pcp_size": 1,
        "remote_dcp_size": 1,
    }
    expected = _valid_reverse_plan_input().build()

    reconstructed = metadata_module.ReversePlan.from_dict(payload)

    assert expected.to_dict() == payload
    assert reconstructed == expected
    payload["source_block_ids"][0].append(99)
    payload["destination_block_ids"][0][0] = -1
    payload["remote_block_sizes"][0] = 32
    assert reconstructed == expected

    invalid_payloads = (
        {key: value for key, value in expected.to_dict().items() if key != "remote_host"},
        {**expected.to_dict(), "unexpected": None},
        {**expected.to_dict(), "token_start": 64},
        {**expected.to_dict(), "remote_block_sizes": [0]},
    )
    for invalid_payload in invalid_payloads:
        with pytest.raises(PathDecisionValidationError):
            metadata_module.ReversePlan.from_dict(invalid_payload)


def test_reverse_receive_binding_deep_freezes_destination_ownership() -> None:
    destination = [[20, 21, 22, 23]]
    binding = _make_reverse_binding(destination)

    destination[0].append(99)
    destination.append([30])

    assert binding.destination_block_ids == ((20, 21, 22, 23),)
    with pytest.raises(FrozenInstanceError):
        binding.token_end = 80


def test_reverse_plan_rejects_invalid_token_range_and_boolean_integers() -> None:
    valid = _valid_reverse_plan_input()
    invalid_inputs = (
        replace(valid, token_start=-1),
        replace(valid, token_start=64),
        replace(valid, token_end=16),
        replace(valid, token_start=True),
        replace(valid, token_end=False),
        replace(valid, token_start=17),
        replace(valid, source_block_ids=((10, True, 12, 13),)),
    )

    for invalid_input in invalid_inputs:
        with pytest.raises(PathDecisionValidationError):
            invalid_input.build()


def test_reverse_plan_rejects_empty_groups_and_insufficient_coverage() -> None:
    valid = _valid_reverse_plan_input()
    invalid_inputs = (
        replace(valid, source_block_ids=()),
        replace(valid, destination_block_ids=()),
        replace(valid, source_block_ids=((),)),
        replace(valid, destination_block_ids=((),)),
        replace(valid, source_block_ids=((10, 11, 12),)),
        replace(valid, destination_block_ids=((20, 21, 22),)),
    )

    for invalid_input in invalid_inputs:
        with pytest.raises(PathDecisionValidationError):
            invalid_input.build()


def test_reverse_plan_rejects_invalid_topology_and_group_count_mismatch() -> None:
    valid = _valid_reverse_plan_input()
    invalid_inputs = (
        replace(valid, wire_request_id=""),
        replace(valid, remote_engine_id=""),
        replace(valid, remote_host=""),
        replace(valid, remote_port=0),
        replace(valid, remote_port=True),
        replace(valid, remote_block_sizes=()),
        replace(valid, remote_block_sizes=(0,)),
        replace(valid, remote_block_sizes=(True,)),
        replace(valid, remote_tp_size=0),
        replace(valid, remote_pcp_size=False),
        replace(valid, remote_dcp_size=-1),
        replace(valid, source_block_ids=((10, 11, 12, 13), (30, 31))),
        replace(valid, destination_block_ids=((20, 21, 22, 23), (40, 41))),
        replace(valid, remote_block_sizes=(16, 16)),
    )

    for invalid_input in invalid_inputs:
        with pytest.raises(PathDecisionValidationError):
            invalid_input.build()


def test_empty_reverse_interval_creates_no_plan_or_binding() -> None:
    metadata = DualPathConnectorMetadata()

    assert metadata.reverse_plans == []
    assert metadata.reverse_receive_bindings == []
    with pytest.raises(PathDecisionValidationError):
        replace(_valid_reverse_plan_input(), token_start=64).build()
    with pytest.raises(PathDecisionValidationError):
        _make_reverse_binding(token_start=64, token_end=64)


def test_reverse_contracts_carry_no_runtime_or_mutable_state() -> None:
    expected_fields = {
        "ReversePlan": {
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
        },
        "ReverseReceiveBinding": {
            "request_key",
            "wire_request_id",
            "prefill_request_id",
            "destination_block_ids",
            "token_start",
            "token_end",
        },
    }
    forbidden_type_fragments = (
        "layeraddress",
        "layermetadata",
        "transferengine",
        "tensor",
        "requeststate",
        "storehandle",
    )

    for contract in (metadata_module.ReversePlan, metadata_module.ReverseReceiveBinding):
        contract_fields = fields(contract)
        assert {contract_field.name for contract_field in contract_fields} == expected_fields[contract.__name__]
        for contract_field in contract_fields:
            field_contract = f"{contract_field.name}:{contract_field.type!r}".lower().replace("_", "")
            assert not any(fragment in field_contract for fragment in forbidden_type_fragments)

    for contract in (_valid_reverse_plan_input().build(), _make_reverse_binding()):
        for contract_field in fields(contract):
            assert not isinstance(getattr(contract, contract_field.name), (list, dict, set, bytearray))

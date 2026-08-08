# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    DECODE_REQUEST_ID,
    WIRE_REQUEST_ID,
    _make_prefill_worker,
    _make_reverse_plan,
    _make_reverse_receive_binding,
    _make_split_metadata,
    _make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import DualPathConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorWorker,
    get_external_request_id,
)


def test_identical_duplicate_plans_are_idempotent_and_conflicts_preserve_first() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    plan = _make_reverse_plan()

    worker._install_reverse_plan(plan)
    worker._install_reverse_plan(plan)
    with pytest.raises(RuntimeError, match="conflicting duplicate Reverse plan"):
        worker._install_reverse_plan(replace(plan, remote_port=plan.remote_port + 1))

    assert worker._reverse_plans == {DECODE_REQUEST_ID: plan}
    assert worker._reverse_plans[DECODE_REQUEST_ID] is plan
    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    assert tracker.plan is plan
    assert tracker.reverse_phase.value == "PENDING"


def test_reverse_plan_install_rejects_wire_and_split_boundary_mismatch() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    plan = _make_reverse_plan()

    with pytest.raises(RuntimeError, match="wire request"):
        worker._install_reverse_plan(replace(plan, wire_request_id=f"{WIRE_REQUEST_ID}-other"))
    with pytest.raises(RuntimeError, match="split boundary"):
        worker._install_reverse_plan(replace(plan, token_end=48))

    assert worker._reverse_plans == {}
    assert worker._split_trackers[DECODE_REQUEST_ID].plan is None


def test_decode_send_callback_records_failed_wins_and_always_delegates_parent_signal() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted = True
    req_meta = MagicMock()

    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal") as parent_signal:
        worker.send_done_send_signal(DECODE_REQUEST_ID, req_meta, 0, True)
        worker.send_done_send_signal(DECODE_REQUEST_ID, req_meta, 0, False)
        worker.send_done_send_signal(DECODE_REQUEST_ID, req_meta, 0, True)
        worker.send_done_send_signal("unknown-request", req_meta, 0, False)

    assert worker._pending_local_reverse_terminals == {DECODE_REQUEST_ID: False}
    assert parent_signal.call_count == 4


def test_reverse_done_before_binding_is_retained_and_reconciled() -> None:
    worker = _make_prefill_worker()
    binding = _make_reverse_receive_binding()
    empty_metadata = DualPathConnectorMetadata()
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}

    assert worker.get_finished(set(), empty_metadata) == (set(), set())
    assert worker._pending_forward_done == {binding.wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    metadata = DualPathConnectorMetadata()
    metadata.reverse_receive_bindings.append(binding)
    worker.start_load_kv(metadata)

    assert worker.get_finished(set(), metadata) == (set(), {binding.prefill_request_id})
    assert worker.get_finished(set(), metadata) == (set(), set())
    assert worker._pending_reverse_done == set()
    assert worker._consumed_reverse_terminals == {binding.wire_request_id: True}


def test_reverse_failed_before_binding_is_retained_and_reconciled() -> None:
    worker = _make_prefill_worker()
    binding = _make_reverse_receive_binding()
    empty_metadata = DualPathConnectorMetadata()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}

    assert worker.get_finished(set(), empty_metadata) == (set(), set())
    assert worker._pending_forward_failed == {binding.wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    metadata = DualPathConnectorMetadata()
    metadata.reverse_receive_bindings.append(binding)
    worker.start_load_kv(metadata)

    assert worker.get_finished(set(), metadata) == (set(), {binding.prefill_request_id})
    assert worker.get_block_ids_with_load_errors() == {71, 80, 81}
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    assert worker.get_finished(set(), metadata) == (set(), set())
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker._consumed_reverse_terminals == {binding.wire_request_id: False}


def test_unknown_and_ordinary_parent_terminals_are_never_attributed_to_split_requests() -> None:
    worker = _make_prefill_worker()
    metadata = DualPathConnectorMetadata()
    ordinary_request_id = "ordinary-request-00000001"
    ordinary_wire_request_id = get_external_request_id(ordinary_request_id)
    worker.request_map[ordinary_wire_request_id] = ordinary_request_id
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {ordinary_wire_request_id, "unknown-wire"}

    assert worker.get_finished(set(), metadata) == (set(), {ordinary_request_id})
    assert worker._pending_forward_done == {"unknown-wire"}
    assert worker._pending_reverse_done == set()
    assert worker._pending_reverse_failed == set()
    assert worker._reverse_receive_bindings == {}


def test_identical_duplicate_bindings_are_idempotent_and_conflicts_preserve_first() -> None:
    worker = _make_prefill_worker()
    binding = _make_reverse_receive_binding()

    worker._install_reverse_receive_binding(binding)
    worker._install_reverse_receive_binding(binding)
    with pytest.raises(RuntimeError, match="conflicting duplicate Reverse receive binding"):
        worker._install_reverse_receive_binding(replace(binding, destination_block_ids=((170, 171, 180, 181),)))
    with pytest.raises(RuntimeError, match="conflicting duplicate Reverse receive binding"):
        worker._install_reverse_receive_binding(
            _make_reverse_receive_binding(prefill_request_id=f"{WIRE_REQUEST_ID}other-prefill")
        )

    assert worker._reverse_receive_bindings == {binding.prefill_request_id: binding}
    assert worker._reverse_request_map == {binding.wire_request_id: binding.prefill_request_id}

    ordinary_owner_worker = _make_prefill_worker()
    ordinary_owner_worker.request_map[binding.wire_request_id] = "ordinary-prefill-request"
    with pytest.raises(RuntimeError, match="conflicting duplicate Reverse receive binding"):
        ordinary_owner_worker._install_reverse_receive_binding(binding)
    assert ordinary_owner_worker.request_map == {binding.wire_request_id: "ordinary-prefill-request"}
    assert ordinary_owner_worker._reverse_receive_bindings == {}

    consumed_forward_worker = _make_prefill_worker()
    consumed_forward_worker._consumed_forward_terminals[binding.wire_request_id] = "forward-request"
    with pytest.raises(RuntimeError, match="conflicting duplicate Reverse receive binding"):
        consumed_forward_worker._install_reverse_receive_binding(binding)
    assert consumed_forward_worker._reverse_receive_bindings == {}

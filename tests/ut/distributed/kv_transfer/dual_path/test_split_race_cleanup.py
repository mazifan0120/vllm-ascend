# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    DECODE_REQUEST_ID,
    WIRE_REQUEST_ID,
    _make_reverse_plan,
    _make_split_metadata,
    _make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorWorker,
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

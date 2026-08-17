# SPDX-License-Identifier: Apache-2.0

import inspect
import threading
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path.test_forward_receive_binding import (
    _admit_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_forward_receive_binding import (
    scheduler_factory as _scheduler_factory_fixture,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    DECODE_REQUEST_ID,
    REVERSE_ATTEMPT_KEY,
    WIRE_REQUEST_ID,
    _make_prefill_worker,
    _make_reverse_plan,
    _make_reverse_receive_binding,
    _make_split_metadata,
    _make_store_metadata,
    _make_worker,
    _set_forward_terminal,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorScheduler
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import DualPathConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind, ReverseAttemptKey
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorWorker,
    get_external_request_id,
)


class _RecordingLock:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.enter_count = 0

    @property
    def locked(self) -> bool:
        return self._lock.locked()

    def __enter__(self):
        self._lock.acquire()
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


class _LockObservedDict(dict):
    def __init__(self, lock: _RecordingLock, values: dict) -> None:
        super().__init__(values)
        self._lock = lock
        self.accesses: list[tuple[str, bool]] = []

    def _record(self, operation: str) -> None:
        self.accesses.append((operation, self._lock.locked))

    def get(self, key, default=None):
        self._record("get")
        return super().get(key, default)

    def pop(self, key, default=None):
        self._record("pop")
        return super().pop(key, default)

    def clear(self) -> None:
        self._record("clear")
        super().clear()

    def __setitem__(self, key, value) -> None:
        self._record("set")
        super().__setitem__(key, value)


def test_identical_duplicate_plans_are_idempotent_and_conflicts_preserve_first() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    plan = _make_reverse_plan()

    worker._install_reverse_plan(plan)
    worker._install_reverse_plan(plan)
    with pytest.raises(RuntimeError, match="conflicting duplicate Reverse plan"):
        worker._install_reverse_plan(replace(plan, remote_port=plan.remote_port + 1))

    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    assert tracker.reverse_plan is plan
    assert tracker.reverse_phase.value == "PENDING"


def test_reverse_plan_install_rejects_wire_and_split_boundary_mismatch() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    plan = _make_reverse_plan()

    with pytest.raises(RuntimeError, match="wire request"):
        worker._install_reverse_plan(replace(plan, wire_request_id=f"{WIRE_REQUEST_ID}-other"))
    with pytest.raises(RuntimeError, match="split boundary"):
        worker._install_reverse_plan(replace(plan, token_end=48))

    assert worker._split_trackers[DECODE_REQUEST_ID].reverse_plan is None


def test_decode_send_callback_records_failed_wins_and_always_delegates_parent_signal() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    req_meta = MagicMock()

    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal") as parent_signal:
        worker.send_done_send_signal(DECODE_REQUEST_ID, req_meta, 0, True)
        worker.send_done_send_signal(DECODE_REQUEST_ID, req_meta, 0, False)
        worker.send_done_send_signal(DECODE_REQUEST_ID, req_meta, 0, True)
        worker.send_done_send_signal("unknown-request", req_meta, 0, False)

    assert worker._pending_local_reverse_terminals == {REVERSE_ATTEMPT_KEY: False}
    assert parent_signal.call_count == 4


def test_reverse_done_before_binding_is_retained_and_reconciled() -> None:
    worker = _make_prefill_worker()
    binding = _make_reverse_receive_binding()
    empty_metadata = DualPathConnectorMetadata()
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}

    assert worker.get_finished(set(), empty_metadata) == (set(), set())
    assert worker._pending_forward_done_wire_ids == {binding.wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    metadata = DualPathConnectorMetadata()
    metadata.reverse_receive_bindings.append(binding)
    worker.start_load_kv(metadata)

    assert worker.get_finished(set(), metadata) == (set(), set())
    assert worker.build_connector_worker_meta().completed_jobs == {binding.reverse_completion_job_id: 1}
    assert worker.get_finished(set(), metadata) == (set(), set())
    assert worker._pending_reverse_done_wire_ids == set()
    assert worker._consumed_reverse_terminal_wire_ids == {binding.wire_request_id: REVERSE_ATTEMPT_KEY}


def test_reverse_failed_before_binding_is_retained_and_reconciled() -> None:
    worker = _make_prefill_worker()
    binding = _make_reverse_receive_binding()
    empty_metadata = DualPathConnectorMetadata()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}

    assert worker.get_finished(set(), empty_metadata) == (set(), set())
    assert worker._pending_forward_failed_wire_ids == {binding.wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    metadata = DualPathConnectorMetadata()
    metadata.reverse_receive_bindings.append(binding)
    worker.start_load_kv(metadata)

    assert worker.get_finished(set(), metadata) == (set(), set())
    assert worker.build_connector_worker_meta().failed_jobs == {binding.reverse_completion_job_id: 1}
    assert worker.get_block_ids_with_load_errors() == {71, 80, 81}
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    assert worker.get_finished(set(), metadata) == (set(), set())
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker._consumed_reverse_terminal_wire_ids == {binding.wire_request_id: REVERSE_ATTEMPT_KEY}


def test_unknown_and_ordinary_parent_terminals_are_never_attributed_to_split_requests() -> None:
    worker = _make_prefill_worker()
    metadata = DualPathConnectorMetadata()
    ordinary_request_id = "ordinary-request-00000001"
    ordinary_wire_request_id = get_external_request_id(ordinary_request_id)
    worker.request_map[ordinary_wire_request_id] = ordinary_request_id
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {ordinary_wire_request_id, "unknown-wire"}

    assert worker.get_finished(set(), metadata) == (set(), {ordinary_request_id})
    assert worker._pending_forward_done_wire_ids == {"unknown-wire"}
    assert worker._pending_reverse_done_wire_ids == set()
    assert worker._pending_reverse_failed_wire_ids == set()
    assert worker._reverse_receive_bindings == {}


def test_identical_duplicate_bindings_are_idempotent_and_conflicts_preserve_first() -> None:
    worker = _make_prefill_worker()
    binding = _make_reverse_receive_binding()

    worker._install_reverse_receive_binding(binding)
    worker._install_reverse_receive_binding(binding)
    with pytest.raises(RuntimeError, match="duplicate Reverse receive binding"):
        worker._install_reverse_receive_binding(replace(binding, destination_block_ids=((170, 171, 180, 181),)))
    with pytest.raises(RuntimeError, match="duplicate Reverse receive binding"):
        worker._install_reverse_receive_binding(
            _make_reverse_receive_binding(prefill_request_id=f"{WIRE_REQUEST_ID}other-prefill")
        )

    assert worker._reverse_receive_bindings == {REVERSE_ATTEMPT_KEY: binding}
    assert worker._reverse_request_map == {binding.wire_request_id: REVERSE_ATTEMPT_KEY}

    ordinary_owner_worker = _make_prefill_worker()
    ordinary_owner_worker.request_map[binding.wire_request_id] = "ordinary-prefill-request"
    with pytest.raises(RuntimeError, match="duplicate Reverse receive binding"):
        ordinary_owner_worker._install_reverse_receive_binding(binding)
    assert ordinary_owner_worker.request_map == {binding.wire_request_id: "ordinary-prefill-request"}
    assert ordinary_owner_worker._reverse_receive_bindings == {}

    consumed_forward_worker = _make_prefill_worker()
    consumed_forward_worker._consumed_forward_terminal_wire_ids[binding.wire_request_id] = "forward-request"
    with pytest.raises(RuntimeError, match="duplicate Reverse receive binding"):
        consumed_forward_worker._install_reverse_receive_binding(binding)
    assert consumed_forward_worker._reverse_receive_bindings == {}


def test_failed_wins_over_duplicate_and_late_done_for_every_source() -> None:
    store_worker = _make_worker()
    store_metadata = _make_split_metadata()
    store_worker._kvpool_worker_adapter.get_finished.side_effect = [
        (set(), {DECODE_REQUEST_ID}),
        (set(), {DECODE_REQUEST_ID}),
    ]
    store_worker._kvpool_worker_adapter.get_block_ids_with_load_errors.side_effect = [{20}, set(), set(), set()]
    store_worker.start_load_kv(store_metadata)
    assert store_worker.get_finished(set(), store_metadata) == (set(), {DECODE_REQUEST_ID})
    assert store_worker.get_finished(set(), store_metadata) == (set(), set())

    reverse_worker = _make_worker()
    reverse_metadata = _make_split_metadata(include_store=False, include_reverse=True)
    reverse_worker.start_load_kv(_make_split_metadata(include_store=False))
    reverse_worker._install_reverse_plan(reverse_metadata.reverse_plans[0])
    reverse_worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        reverse_worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, False)
    assert reverse_worker.get_finished(set(), reverse_metadata) == (set(), {DECODE_REQUEST_ID})
    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        reverse_worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, True)
    assert reverse_worker.get_finished(set(), reverse_metadata) == (set(), set())

    forward_worker = _make_worker()
    forward_metadata = _make_split_metadata(include_store=False)
    forward_worker.start_load_kv(forward_metadata)
    forward_worker.kv_recv_layer_thread = MagicMock()
    forward_worker.kv_recv_layer_thread.get_and_clear_failed_requests.side_effect = [{WIRE_REQUEST_ID}, set()]
    forward_worker.kv_recv_layer_thread.get_and_clear_done_requests.side_effect = [set(), {WIRE_REQUEST_ID}]
    assert forward_worker.get_finished(set(), forward_metadata) == (set(), {DECODE_REQUEST_ID})
    assert forward_worker.get_finished(set(), forward_metadata) == (set(), set())

    assert store_worker._split_trackers[DECODE_REQUEST_ID].store_phase.value == "FAILED"
    assert reverse_worker._split_trackers[DECODE_REQUEST_ID].reverse_phase.value == "FAILED"
    assert forward_worker._split_trackers[DECODE_REQUEST_ID].forward_phase.value == "FAILED"


@pytest.mark.parametrize("source", ["store", "reverse", "forward"])
def test_each_failure_publishes_exactly_one_local_terminal(source: str) -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_reverse=source == "reverse")
    worker.start_load_kv(metadata)

    if source == "store":
        worker._kvpool_worker_adapter.get_finished.return_value = (set(), {DECODE_REQUEST_ID})
        worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {20}
    elif source == "reverse":
        tracker = worker._split_trackers[DECODE_REQUEST_ID]
        tracker.reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
        with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
            worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, False)
            worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, False)
    else:
        _set_forward_terminal(worker, failed=True)

    first = worker.get_finished(set(), metadata)
    second = worker.get_finished(set(), metadata)

    assert first == (set(), {DECODE_REQUEST_ID})
    assert second == (set(), set())


@pytest.mark.parametrize("path", [PathKind.PE_READ, PathKind.DE_READ])
def test_forward_early_terminal_reconciliation_passes_for_both_paths(path: PathKind) -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_store=False)
    binding = replace(metadata.forward_receive_bindings[0], path=path)
    metadata.forward_receive_bindings[:] = [binding]
    worker.kv_recv_layer_thread = MagicMock()
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {WIRE_REQUEST_ID}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    assert worker.get_finished(set(), DualPathConnectorMetadata()) == (set(), set())
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()

    worker.start_load_kv(metadata)
    finished = worker.get_finished(set(), metadata)

    assert finished == (set(), {DECODE_REQUEST_ID})
    assert worker._pending_forward_done_wire_ids == set()


def test_pe_read_forward_done_still_completes_immediately() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata()
    binding = replace(metadata.forward_receive_bindings[0], path=PathKind.PE_READ)
    metadata.forward_receive_bindings[:] = [binding]
    metadata.decode_store_metadata = None
    worker.start_load_kv(metadata)
    _set_forward_terminal(worker)

    finished = worker.get_finished(set(), metadata)

    assert finished == (set(), {DECODE_REQUEST_ID})
    assert worker._split_trackers == {}


def test_unconsumed_pe_read_forward_binding_is_released_on_request_finish() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_store=False)
    binding = replace(metadata.forward_receive_bindings[0], path=PathKind.PE_READ)
    metadata.forward_receive_bindings[:] = [binding]
    worker.start_load_kv(metadata)

    finished = worker.get_finished({DECODE_REQUEST_ID}, metadata)

    assert finished == (set(), set())
    assert DECODE_REQUEST_ID not in worker._forward_receive_bindings
    assert WIRE_REQUEST_ID not in worker.request_map


def test_finished_req_ids_and_shutdown_release_all_task07_state_idempotently() -> None:
    decode_worker = _make_worker()
    decode_metadata = _make_split_metadata(include_store=False)
    decode_worker.start_load_kv(decode_metadata)
    plan = _make_reverse_plan()
    decode_worker._install_reverse_plan(plan)
    decode_worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    decode_worker._pending_local_reverse_terminals[REVERSE_ATTEMPT_KEY] = True
    decode_worker._pending_forward_done_wire_ids.add(WIRE_REQUEST_ID)
    decode_worker._consumed_forward_terminal_wire_ids[WIRE_REQUEST_ID] = DECODE_REQUEST_ID

    unrelated_decode_id = "unrelated-wire-id123456789"
    unrelated_wire_id = get_external_request_id(unrelated_decode_id)
    decode_worker._split_trackers[unrelated_decode_id] = decode_worker._split_trackers[DECODE_REQUEST_ID]
    decode_worker._forward_receive_bindings[unrelated_decode_id] = replace(
        decode_metadata.forward_receive_bindings[0],
        request_key=replace(
            decode_metadata.forward_receive_bindings[0].request_key,
            decode_request_id=unrelated_decode_id,
        ),
        wire_request_id=unrelated_wire_id,
        decode_request_id=unrelated_decode_id,
    )
    decode_worker.request_map[unrelated_wire_id] = unrelated_decode_id

    prefill_worker = _make_prefill_worker()
    reverse_binding = _make_reverse_receive_binding()
    prefill_worker._install_reverse_receive_binding(reverse_binding)
    prefill_worker._pending_reverse_done_wire_ids.add(reverse_binding.wire_request_id)
    prefill_worker._consumed_reverse_terminal_wire_ids[reverse_binding.wire_request_id] = REVERSE_ATTEMPT_KEY
    unrelated_prefill_id = "unrelated-prefill-local"
    unrelated_reverse_binding = replace(
        reverse_binding,
        request_key=replace(reverse_binding.request_key, decode_request_id="unrelated-decode-request"),
        wire_request_id="unrelated-reverse-wire",
        prefill_request_id=unrelated_prefill_id,
    )
    prefill_worker._install_reverse_receive_binding(unrelated_reverse_binding)

    assert decode_worker.get_finished({DECODE_REQUEST_ID}, DualPathConnectorMetadata()) == (set(), set())
    assert prefill_worker.get_finished({reverse_binding.prefill_request_id}, DualPathConnectorMetadata()) == (
        set(),
        set(),
    )

    # Section-5 removal rule: the local reverse terminal is drained during the
    # first finish pass, so the tracker is removed on the next one.
    assert DECODE_REQUEST_ID in decode_worker._split_trackers
    assert DECODE_REQUEST_ID not in decode_worker._forward_receive_bindings
    assert WIRE_REQUEST_ID not in decode_worker.request_map
    assert REVERSE_ATTEMPT_KEY not in decode_worker._pending_local_reverse_terminals
    assert WIRE_REQUEST_ID not in decode_worker._pending_forward_done_wire_ids
    assert WIRE_REQUEST_ID not in decode_worker._consumed_forward_terminal_wire_ids
    assert unrelated_decode_id in decode_worker._split_trackers
    assert decode_worker._split_trackers[unrelated_decode_id].reverse_plan is plan
    assert unrelated_decode_id in decode_worker._forward_receive_bindings
    assert decode_worker.request_map[unrelated_wire_id] == unrelated_decode_id

    assert REVERSE_ATTEMPT_KEY not in prefill_worker._reverse_receive_bindings
    assert reverse_binding.wire_request_id not in prefill_worker._reverse_request_map
    assert reverse_binding.wire_request_id not in prefill_worker._pending_reverse_done_wire_ids
    assert reverse_binding.wire_request_id not in prefill_worker._consumed_reverse_terminal_wire_ids
    unrelated_attempt_key = ReverseAttemptKey(unrelated_reverse_binding.request_key, 0)
    assert prefill_worker._reverse_receive_bindings[unrelated_attempt_key] == unrelated_reverse_binding
    assert prefill_worker._reverse_request_map[unrelated_reverse_binding.wire_request_id] == unrelated_attempt_key

    assert decode_worker.get_finished({DECODE_REQUEST_ID}, DualPathConnectorMetadata()) == (set(), set())
    assert prefill_worker.get_finished({reverse_binding.prefill_request_id}, DualPathConnectorMetadata()) == (
        set(),
        set(),
    )
    assert DECODE_REQUEST_ID not in decode_worker._split_trackers

    decode_state = (
        dict(decode_worker._split_trackers),
        {request_id: tracker.reverse_plan for request_id, tracker in decode_worker._split_trackers.items()},
        dict(decode_worker._forward_receive_bindings),
        dict(decode_worker.request_map),
    )
    prefill_state = (
        dict(prefill_worker._reverse_receive_bindings),
        dict(prefill_worker._reverse_request_map),
    )
    assert decode_worker.get_finished({DECODE_REQUEST_ID}, DualPathConnectorMetadata()) == (set(), set())
    assert prefill_worker.get_finished({reverse_binding.prefill_request_id}, DualPathConnectorMetadata()) == (
        set(),
        set(),
    )
    assert decode_state == (
        decode_worker._split_trackers,
        {request_id: tracker.reverse_plan for request_id, tracker in decode_worker._split_trackers.items()},
        decode_worker._forward_receive_bindings,
        decode_worker.request_map,
    )
    assert prefill_state == (
        prefill_worker._reverse_receive_bindings,
        prefill_worker._reverse_request_map,
    )

    with patch.object(decode_worker, "_enqueue_kv_layer_send") as enqueue:
        decode_worker.shutdown()
        decode_worker.shutdown()
        decode_worker._install_forward_receive_binding(decode_metadata.forward_receive_bindings[0])
        decode_worker._install_split_tracker(decode_metadata.forward_receive_bindings[0], None)
        decode_worker._install_reverse_plan(plan)
        decode_worker._submit_reverse(DECODE_REQUEST_ID)

    prefill_worker.shutdown()
    prefill_worker.shutdown()
    prefill_worker._install_reverse_receive_binding(reverse_binding)

    assert decode_worker._accepting_split_requests is False
    assert prefill_worker._accepting_split_requests is False
    assert decode_worker._split_trackers == {}
    assert decode_worker._forward_receive_bindings == {}
    assert decode_worker._pending_local_reverse_terminals == {}
    assert decode_worker._pending_forward_done_wire_ids == set()
    assert decode_worker._pending_forward_failed_wire_ids == set()
    assert decode_worker._consumed_forward_terminal_wire_ids == {}
    assert prefill_worker._reverse_receive_bindings == {}
    assert prefill_worker._reverse_request_map == {}
    assert prefill_worker._pending_reverse_done_wire_ids == set()
    assert prefill_worker._pending_reverse_failed_wire_ids == set()
    assert prefill_worker._consumed_reverse_terminal_wire_ids == {}
    decode_worker._kvpool_worker_adapter.close.assert_not_called()
    enqueue.assert_not_called()


def test_release_prevents_late_reverse_callback_from_recreating_terminal_state() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    worker._pending_local_reverse_terminals[REVERSE_ATTEMPT_KEY] = True

    worker._drain_local_reverse_terminals()
    worker._release_split_request_state({DECODE_REQUEST_ID})
    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, trans_flag=False)

    assert worker._pending_local_reverse_terminals == {}


def test_shutdown_prevents_late_reverse_callback_from_recreating_terminal_state() -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    worker._pending_local_reverse_terminals[REVERSE_ATTEMPT_KEY] = True

    with patch.object(MooncakeLayerwiseConnectorWorker, "shutdown", create=True):
        worker.shutdown()
    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, trans_flag=False)

    assert worker._pending_local_reverse_terminals == {}


@pytest.mark.parametrize("cleanup", ["release", "shutdown"])
def test_reverse_callback_and_cleanup_share_one_lock_for_tracker_and_terminal_state(cleanup: str) -> None:
    worker = _make_worker()
    worker.start_load_kv(_make_split_metadata(include_store=False))
    worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    recording_lock = _RecordingLock()
    tracked_trackers = _LockObservedDict(recording_lock, worker._split_trackers)
    tracked_terminals = _LockObservedDict(recording_lock, worker._pending_local_reverse_terminals)
    worker._reverse_terminal_lock = recording_lock
    worker._split_trackers = tracked_trackers
    worker._pending_local_reverse_terminals = tracked_terminals

    with patch.object(MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, trans_flag=False)
    worker._drain_local_reverse_terminals()
    if cleanup == "release":
        worker._release_split_request_state({DECODE_REQUEST_ID})
    else:
        with patch.object(MooncakeLayerwiseConnectorWorker, "shutdown", create=True):
            worker.shutdown()

    accesses = tracked_trackers.accesses + tracked_terminals.accesses
    assert recording_lock.enter_count >= 2
    assert accesses
    assert all(lock_held for _, lock_held in accesses), accesses
    assert tracked_terminals == {}


def test_shutdown_makes_public_split_start_load_inert_but_preserves_pre_shutdown_store_delegation() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_reverse=True)
    store_metadata = metadata.decode_store_metadata

    worker.start_load_kv(metadata)

    worker._kvpool_worker_adapter.start_load_kv.assert_called_once_with(store_metadata)
    assert DECODE_REQUEST_ID in worker._split_trackers
    assert worker._split_trackers[DECODE_REQUEST_ID].reverse_plan is not None

    with (
        patch.object(MooncakeLayerwiseConnectorWorker, "shutdown", create=True),
        patch.object(worker, "_enqueue_kv_layer_send") as enqueue,
    ):
        worker.shutdown()
        worker._kvpool_worker_adapter.start_load_kv.reset_mock()
        worker.start_load_kv(metadata)

    worker._kvpool_worker_adapter.start_load_kv.assert_not_called()
    assert worker._split_trackers == {}
    assert worker._forward_receive_bindings == {}
    enqueue.assert_not_called()

    ordinary_store_metadata = DualPathConnectorMetadata()
    ordinary_store_metadata.decode_store_metadata = store_metadata
    worker.start_load_kv(ordinary_store_metadata)
    worker._kvpool_worker_adapter.start_load_kv.assert_called_once_with(store_metadata)


def test_production_de_read_result_creates_plan_binding_store_without_scheduler_worker_io(monkeypatch) -> None:
    fixture = _scheduler_factory_fixture.__wrapped__(monkeypatch)
    scheduler_factory = next(fixture)
    try:
        scheduler, coordinator = scheduler_factory()
        request, snapshot, state = _admit_request(scheduler)
        decision = _de_read_decision(state, snapshot)
        store_metadata = _make_store_metadata(request.request_id)
        scheduler._kvpool_adapter.build_connector_meta.return_value = store_metadata
        coordinator.take_received_decisions.return_value = [decision]
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        assert len(metadata.forward_receive_bindings) == 1
        assert metadata.forward_receive_bindings[0].path is PathKind.DE_READ
        reverse_send_job_id = scheduler._reverse_send_job_ids[ReverseAttemptKey(state.request_key, 0)]
        assert metadata.reverse_plans == [replace(decision.reverse_plan, reverse_send_job_id=reverse_send_job_id)]
        assert metadata.reverse_receive_bindings == []
        assert metadata.decode_store_metadata is store_metadata
        assert scheduler._reqs_need_recv == {}
        assert scheduler._reqs_need_send_layerwise == {}
    finally:
        with pytest.raises(StopIteration):
            next(fixture)


def test_scheduler_method_set_is_pinned_and_has_no_blocking_hooks() -> None:
    expected_scheduler_methods = {
        "__init__",
        "_is_dual_path_decode_admission",
        "_stage_prefill_activation_failure",
        "_reverse_destination_slice",
        "_send_abort_notice",
        "_retain_prefill_admission_failure",
        "_decide_prefill_path_for_admission",
        "_discard_undelivered_prefill_decision",
        "_log_prefill_decision",
        "_prepare_forward_plan",
        "_try_install_forward_plan",
        "_activate_de_read_path",
        "_activate_received_decision",
        "_validate_committed_decision",
        "_log_decision_activation",
        "_build_decode_control_failure",
        "_build_remote_decode_message",
        "_reconcile_prefill_deliveries",
        "_update_prefill_state_after_alloc",
        "_invalidate_prefill_activation",
        "_bind_decode_admission_after_alloc",
        "_is_identical_duplicate_admission",
        "_register_pending_decode_decision",
        "_release_scheduler_request_state",
        "_may_install_forward_plan",
        "_delay_free_for_connector",
        "_has_open_reverse_send_job",
        "_resume_delivered_prefill_decision",
        "_deliver_prefill_decision",
        "_request_for_failed_job",
        "_handle_received_abort",
        "_recovery_invalid_block_ids",
        "_aggregate_worker_job_facts",
        "_run_job_close_action",
        "_close_reverse_completion_job",
        "bind_gpu_block_pool",
        "update_connector_output",
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "build_connector_meta",
        "request_finished",
        "request_finished_all_groups",
        "_send_prefill_client_abort",
        "shutdown",
    }
    scheduler_methods = {
        name for name, value in DualPathConnectorScheduler.__dict__.items() if inspect.isfunction(value)
    }
    source = inspect.getsource(connector_module)

    assert scheduler_methods == expected_scheduler_methods
    for blocking_call in (".result(", ".wait(", "time.sleep"):
        assert blocking_call not in source, f"dual_path/connector.py contains {blocking_call}"

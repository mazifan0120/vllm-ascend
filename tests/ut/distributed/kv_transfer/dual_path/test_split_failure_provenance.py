# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    DECODE_REQUEST_ID,
    REVERSE_ATTEMPT_KEY,
    _make_prefill_worker,
    _make_reverse_receive_binding,
    _make_split_metadata,
    _make_worker,
    _set_forward_terminal,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import DualPathConnectorMetadata


def test_store_failure_starts_no_reverse_and_invalidates_only_store_destination_slice() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata()
    worker._kvpool_worker_adapter.get_finished.side_effect = [
        (set(), {DECODE_REQUEST_ID}),
        (set(), {DECODE_REQUEST_ID}),
    ]
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.side_effect = [
        {20},
        set(),
        {20},
        set(),
    ]

    worker.start_load_kv(metadata)
    first_finished = worker.get_finished(set(), metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    first_invalid = worker.get_block_ids_with_load_errors()
    second_finished = worker.get_finished(set(), metadata)
    second_invalid = worker.get_block_ids_with_load_errors()

    assert tracker.store_phase.value == "FAILED"
    assert tracker.reverse_submitted_attempt is None
    assert tracker.terminal_published is True
    assert first_finished == (set(), {DECODE_REQUEST_ID})
    assert first_invalid == {20, 21}
    assert second_finished == (set(), set())
    assert second_invalid == {20}


def test_store_failure_seen_before_done_stays_failed_when_done_arrives_later() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata()
    worker._kvpool_worker_adapter.get_finished.side_effect = [
        (set(), set()),
        (set(), {DECODE_REQUEST_ID}),
    ]
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.side_effect = [
        {20},
        set(),
    ]

    worker.start_load_kv(metadata)
    with patch.object(worker, "_submit_reverse") as submit_reverse:
        first_finished = worker.get_finished(set(), metadata)
        tracker = worker._split_trackers[DECODE_REQUEST_ID]

        assert first_finished == (set(), set())
        assert tracker.store_load_failed is True
        assert tracker.store_phase.value == "PENDING"
        assert tracker.terminal_published is False
        submit_reverse.assert_not_called()

        second_finished = worker.get_finished(set(), metadata)

    assert second_finished == (set(), {DECODE_REQUEST_ID})
    assert tracker.store_phase.value == "FAILED"
    assert tracker.terminal_published is True
    submit_reverse.assert_not_called()


def test_unrelated_store_invalid_block_does_not_fail_split_request() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata()
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {DECODE_REQUEST_ID})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {999}

    worker.start_load_kv(metadata)
    with patch.object(worker, "_submit_reverse") as submit_reverse:
        finished = worker.get_finished(set(), metadata)

    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    assert finished == (set(), set())
    assert tracker.store_load_failed is False
    assert tracker.store_phase.value == "DONE"
    submit_reverse.assert_called_once_with(DECODE_REQUEST_ID)


def test_reverse_failure_records_local_terminal_invalidates_pe_destinations_and_never_arriving_forward_suffix() -> None:
    decode_worker = _make_worker()
    decode_metadata = _make_split_metadata(include_store=False, include_reverse=True)
    with (
        patch.object(decode_worker, "_build_reverse_send_metadata", return_value=MagicMock()),
        patch.object(decode_worker, "_enqueue_kv_layer_send"),
        patch.object(layerwise_module.torch.npu, "Event", return_value=MagicMock()),
    ):
        decode_worker._registered_kv_caches = {}
        decode_worker.start_load_kv(decode_metadata)
    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        decode_worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, False)

    decode_first = decode_worker.get_finished(set(), decode_metadata)
    decode_invalid = decode_worker.get_block_ids_with_load_errors()
    _set_forward_terminal(decode_worker)
    decode_late = decode_worker.get_finished(set(), decode_metadata)

    prefill_worker = _make_prefill_worker()
    reverse_binding = _make_reverse_receive_binding()
    prefill_metadata = DualPathConnectorMetadata()
    prefill_metadata.reverse_receive_bindings.append(reverse_binding)
    prefill_worker.start_load_kv(prefill_metadata)
    prefill_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {reverse_binding.wire_request_id}
    prefill_finished = prefill_worker.get_finished(set(), prefill_metadata)

    assert decode_first == (set(), {DECODE_REQUEST_ID})
    assert decode_invalid == {30, 31, 40, 41}
    assert decode_late == (set(), set())
    assert prefill_finished == (set(), set())
    assert prefill_worker.build_connector_worker_meta().failure_reports == {
        reverse_binding.reverse_receive_completion_id: 1
    }
    assert prefill_worker.get_block_ids_with_load_errors() == {71, 80, 81}


def test_forward_failure_invalidates_only_decode_forward_suffix() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_reverse=True)
    worker.start_load_kv(metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    with patch.object(worker, "_submit_reverse"):
        worker._consume_store_completions({DECODE_REQUEST_ID}, set())
    tracker.reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, True)
    assert worker.get_finished(set(), metadata) == (set(), set())

    _set_forward_terminal(worker, failed=True)
    finished = worker.get_finished(set(), metadata)

    assert tracker.forward_phase.value == "FAILED"
    assert finished == (set(), {DECODE_REQUEST_ID})
    assert worker.get_block_ids_with_load_errors() == {30, 31, 40, 41}


def test_failure_provenance_block_math_matches_spec_example() -> None:
    store_worker = _make_worker()
    store_metadata = _make_split_metadata()
    store_worker._kvpool_worker_adapter.get_finished.return_value = (set(), {DECODE_REQUEST_ID})
    store_worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {20}
    store_worker.start_load_kv(store_metadata)
    assert store_worker.get_finished(set(), store_metadata) == (set(), {DECODE_REQUEST_ID})

    forward_worker = _make_worker()
    forward_metadata = _make_split_metadata(include_store=False)
    forward_worker.start_load_kv(forward_metadata)
    _set_forward_terminal(forward_worker, failed=True)
    assert forward_worker.get_finished(set(), forward_metadata) == (set(), {DECODE_REQUEST_ID})

    reverse_decode_worker = _make_worker()
    reverse_decode_metadata = _make_split_metadata(include_store=False, include_reverse=True)
    reverse_decode_worker.start_load_kv(_make_split_metadata(include_store=False))
    reverse_decode_worker._install_reverse_plan(reverse_decode_metadata.reverse_plans[0])
    reverse_decode_worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt = REVERSE_ATTEMPT_KEY
    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        reverse_decode_worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, False)
    assert reverse_decode_worker.get_finished(set(), reverse_decode_metadata) == (set(), {DECODE_REQUEST_ID})

    prefill_worker = _make_prefill_worker()
    reverse_binding = _make_reverse_receive_binding()
    reverse_metadata = DualPathConnectorMetadata()
    reverse_metadata.reverse_receive_bindings.append(reverse_binding)
    prefill_worker.start_load_kv(reverse_metadata)
    prefill_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {reverse_binding.wire_request_id}
    assert prefill_worker.get_finished(set(), reverse_metadata) == (set(), set())
    assert prefill_worker.build_connector_worker_meta().failure_reports == {
        reverse_binding.reverse_receive_completion_id: 1
    }

    assert store_worker.get_block_ids_with_load_errors() == {20, 21}
    assert prefill_worker.get_block_ids_with_load_errors() == {71, 80, 81}
    assert reverse_decode_worker.get_block_ids_with_load_errors() == {30, 31, 40, 41}
    assert forward_worker.get_block_ids_with_load_errors() == {30, 31, 40, 41}

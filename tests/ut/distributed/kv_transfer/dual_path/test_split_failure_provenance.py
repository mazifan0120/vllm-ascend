# SPDX-License-Identifier: Apache-2.0

from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    DECODE_REQUEST_ID,
    _make_split_metadata,
    _make_worker,
)


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
    assert tracker.reverse_submitted is False
    assert tracker.terminal_published is True
    assert first_finished == (set(), {DECODE_REQUEST_ID})
    assert first_invalid == {20, 21}
    assert second_finished == (set(), set())
    assert second_invalid == {20}

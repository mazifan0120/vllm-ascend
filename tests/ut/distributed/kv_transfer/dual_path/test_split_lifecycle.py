# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from tests.ut.distributed.kv_transfer.dual_path.conftest import worker_environment
from tests.ut.distributed.kv_transfer.dual_path.test_decode_scheduler import (
    _make_kv_cache_config,
    _make_vllm_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ForwardReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    Path,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
)

DECODE_REQUEST_ID = "decode-split-request"
DESTINATION_BLOCKS = ((10, 11, 20, 21, 30, 31, 40, 41),)


def _make_worker() -> DualPathConnectorWorker:
    with (
        worker_environment(),
        patch.object(connector_module, "KVPoolWorkerAdapter"),
    ):
        return DualPathConnectorWorker(
            _make_vllm_config(),
            _make_kv_cache_config(),
            "decode-engine",
            DualPathConfig(role="decode"),
        )


def _make_store_metadata(request_id: str = DECODE_REQUEST_ID) -> AscendConnectorMetadata:
    store_metadata = AscendConnectorMetadata(set(), set(), loading_req_ids={request_id})
    store_metadata.add_request(
        ReqMeta(
            req_id=request_id,
            token_len_chunk=128,
            block_ids=list(DESTINATION_BLOCKS[0]),
            block_hashes=[bytes([block_id]) for block_id in range(8)],
            load_spec=LoadSpec(
                vllm_cached_tokens=32,
                kvpool_cached_tokens=64,
                can_load=True,
                token_len=128,
            ),
        )
    )
    return store_metadata


def _make_split_metadata() -> DualPathConnectorMetadata:
    metadata = DualPathConnectorMetadata()
    metadata.forward_receive_bindings.append(
        ForwardReceiveBinding(
            request_key=DualPathRequestKey("decode-instance", DECODE_REQUEST_ID),
            path=Path.DE_READ,
            wire_request_id="wire-split-request",
            decode_request_id=DECODE_REQUEST_ID,
            destination_block_ids=DESTINATION_BLOCKS,
            token_start=64,
            token_end=128,
        )
    )
    metadata.decode_store_metadata = _make_store_metadata()
    return metadata


def test_nonempty_store_does_not_submit_reverse_before_store_done() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata()
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), set())
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = set()

    worker.start_load_kv(metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    before_poll = (
        tracker.store_phase,
        tracker.reverse_phase,
        tracker.forward_phase,
        tracker.store_destination_slice,
        tracker.forward_destination_slice,
        tracker.plan,
        tracker.reverse_submitted,
        tracker.terminal_published,
    )
    finished = worker.get_finished(set(), metadata)

    assert tracker.store_phase.value == "PENDING"
    assert tracker.reverse_phase.value == "SKIPPED"
    assert tracker.forward_phase.value == "PENDING"
    assert tracker.store_destination_slice == (20, 21)
    assert tracker.forward_destination_slice == (30, 31, 40, 41)
    assert tracker.plan is None
    assert tracker.reverse_submitted is False
    assert tracker.terminal_published is False
    assert finished == (set(), set())
    assert (
        tracker.store_phase,
        tracker.reverse_phase,
        tracker.forward_phase,
        tracker.store_destination_slice,
        tracker.forward_destination_slice,
        tracker.plan,
        tracker.reverse_submitted,
        tracker.terminal_published,
    ) == before_poll


def test_store_full_creates_no_split_tracker_reverse_or_pe_work() -> None:
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    metadata.decode_store_metadata = _make_store_metadata("store-full-request")
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {"store-full-request"})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = set()
    engine_calls_before_start = list(worker.engine.method_calls)

    worker.start_load_kv(metadata)
    finished = worker.get_finished(set(), metadata)

    assert worker._split_trackers == {}
    assert finished == (set(), {"store-full-request"})
    assert worker.kv_recv_layer_thread is None
    assert worker.engine.method_calls == engine_calls_before_start


def test_store_done_marks_phase_without_outer_completion() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata()
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {DECODE_REQUEST_ID})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = set()

    worker.start_load_kv(metadata)
    finished = worker.get_finished(set(), metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]

    assert tracker.store_phase.value == "DONE"
    assert tracker.reverse_submitted is False
    assert tracker.terminal_published is False
    assert finished == (set(), set())
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker._split_trackers[DECODE_REQUEST_ID] is tracker

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.distributed.kv_transfer.dual_path.conftest import worker_environment
from tests.ut.distributed.kv_transfer.dual_path.test_decode_scheduler import (
    _make_kv_cache_config,
    _make_vllm_config,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ForwardReceiveBinding,
    ReversePlan,
    ReverseReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathKind,
    ReverseAttemptKey,
    reverse_wire_id,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
    ReqMeta,
)

WIRE_REQUEST_ID = "wire-split-request"
DECODE_REQUEST_ID = f"{WIRE_REQUEST_ID}123456789"
REVERSE_ATTEMPT_KEY = ReverseAttemptKey(DualPathRequestKey("decode-instance", DECODE_REQUEST_ID, 0), 0)
REVERSE_WIRE_REQUEST_ID = reverse_wire_id(REVERSE_ATTEMPT_KEY)
DESTINATION_BLOCKS = ((10, 11, 20, 21, 30, 31, 40, 41),)
REVERSE_DESTINATION_BLOCKS = ((70, 71, 80, 81),)


def _make_worker() -> DualPathConnectorWorker:
    with (
        worker_environment(),
        patch.object(worker_module, "KVPoolWorkerAdapter"),
    ):
        worker = DualPathConnectorWorker(
            _make_vllm_config(),
            _make_kv_cache_config(),
            "decode-engine",
            DualPathConfig(role="decode"),
        )
    worker.pd_head_ratio = 1
    worker.enable_kv_quant = False
    worker.enable_c8_quant = False
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), set())
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = set()
    return worker


def _make_prefill_worker() -> DualPathConnectorWorker:
    with (
        worker_environment(),
        patch.object(worker_module, "KVPoolWorkerAdapter"),
    ):
        worker = DualPathConnectorWorker(
            _make_vllm_config(),
            _make_kv_cache_config(),
            "prefill-engine",
            DualPathConfig(role="prefill"),
        )
    worker.kv_recv_layer_thread = MagicMock(name="kv_recv_layer_thread")
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    return worker


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


def _make_reverse_plan(reverse_send_job_id: int | None = None) -> ReversePlan:
    return ReversePlan(
        request_key=DualPathRequestKey("decode-instance", DECODE_REQUEST_ID, 0),
        wire_request_id=REVERSE_WIRE_REQUEST_ID,
        token_start=16,
        token_end=64,
        source_block_ids=((10, 11, 20, 21),),
        destination_block_ids=REVERSE_DESTINATION_BLOCKS,
        remote_engine_id="prefill-engine",
        remote_host="198.51.100.10",
        remote_port=6000,
        remote_block_sizes=(16,),
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        reverse_attempt_id=0,
        prefill_local_tokens=16,
        reverse_send_job_id=reverse_send_job_id,
    )


def _make_reverse_receive_binding(
    *,
    prefill_request_id: str = f"{WIRE_REQUEST_ID}prefill-local",
    destination_block_ids: tuple[tuple[int, ...], ...] = REVERSE_DESTINATION_BLOCKS,
    reverse_completion_job_id: int = 0,
) -> ReverseReceiveBinding:
    return ReverseReceiveBinding(
        request_key=DualPathRequestKey("decode-instance", DECODE_REQUEST_ID, 0),
        wire_request_id=REVERSE_WIRE_REQUEST_ID,
        prefill_request_id=prefill_request_id,
        destination_block_ids=destination_block_ids,
        token_start=16,
        token_end=64,
        reverse_attempt_id=0,
        prefill_local_tokens=16,
        reverse_completion_job_id=reverse_completion_job_id,
    )


def _make_split_metadata(*, include_store: bool = True, include_reverse: bool = False) -> DualPathConnectorMetadata:
    metadata = DualPathConnectorMetadata()
    metadata.forward_receive_bindings.append(
        ForwardReceiveBinding(
            request_key=DualPathRequestKey("decode-instance", DECODE_REQUEST_ID, 0),
            path=PathKind.DE_READ,
            wire_request_id=WIRE_REQUEST_ID,
            decode_request_id=DECODE_REQUEST_ID,
            destination_block_ids=DESTINATION_BLOCKS,
            token_start=64,
            token_end=128,
        )
    )
    if include_store:
        metadata.decode_store_metadata = _make_store_metadata()
    if include_reverse:
        metadata.reverse_plans.append(_make_reverse_plan())
    return metadata


def _register_two_layers(worker: DualPathConnectorWorker) -> dict[str, list[MagicMock]]:
    registered = {
        "model.layer.0": [MagicMock(name="layer_0_key"), MagicMock(name="layer_0_value")],
        "model.layer.1": [MagicMock(name="layer_1_key"), MagicMock(name="layer_1_value")],
    }
    worker._registered_kv_caches = registered
    worker._registered_layer_order = ((0, "model.layer.0"), (1, "model.layer.1"))
    return registered


def _set_forward_terminal(worker: DualPathConnectorWorker, *, failed: bool = False) -> None:
    worker.kv_recv_layer_thread = MagicMock(name="kv_recv_layer_thread")
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set() if failed else {WIRE_REQUEST_ID}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {WIRE_REQUEST_ID} if failed else set()


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
        tracker.reverse_plan,
        tracker.reverse_submitted_attempt,
        tracker.terminal_published,
    )
    finished = worker.get_finished(set(), metadata)

    assert tracker.store_phase.value == "PENDING"
    assert tracker.reverse_phase.value == "SKIPPED"
    assert tracker.forward_phase.value == "PENDING"
    assert tracker.store_destination_slice == (20, 21)
    assert tracker.forward_destination_slice == (30, 31, 40, 41)
    assert tracker.reverse_plan is None
    assert tracker.reverse_submitted_attempt is None
    assert tracker.terminal_published is False
    assert finished == (set(), set())
    assert (
        tracker.store_phase,
        tracker.reverse_phase,
        tracker.forward_phase,
        tracker.store_destination_slice,
        tracker.forward_destination_slice,
        tracker.reverse_plan,
        tracker.reverse_submitted_attempt,
        tracker.terminal_published,
    ) == before_poll


def test_prefill_publishes_no_completion_before_final_reverse_done() -> None:
    worker = _make_prefill_worker()
    metadata = DualPathConnectorMetadata()
    binding = _make_reverse_receive_binding()
    metadata.reverse_receive_bindings.append(binding)

    worker.start_load_kv(metadata)
    finished = worker.get_finished(set(), metadata)

    assert finished == (set(), set())
    assert worker._reverse_receive_bindings == {REVERSE_ATTEMPT_KEY: binding}
    assert worker._reverse_request_map == {binding.wire_request_id: REVERSE_ATTEMPT_KEY}


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
    assert tracker.reverse_submitted_attempt is None
    assert tracker.terminal_published is False
    assert finished == (set(), set())
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker._split_trackers[DECODE_REQUEST_ID] is tracker


def test_reverse_done_does_not_complete_decode_before_forward() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_reverse=True)
    worker.start_load_kv(metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    with patch.object(worker, "_submit_reverse"):
        worker._consume_store_completions({DECODE_REQUEST_ID}, set())
    tracker.reverse_submitted_attempt = REVERSE_ATTEMPT_KEY

    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, True)
    finished = worker.get_finished(set(), metadata)

    assert tracker.store_phase.value == "DONE"
    assert tracker.reverse_phase.value == "DONE"
    assert tracker.forward_phase.value == "PENDING"
    assert finished == (set(), set())


def test_decode_publishes_completion_only_when_full_predicate_satisfied() -> None:
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

    _set_forward_terminal(worker)
    first_finished = worker.get_finished(set(), metadata)
    second_finished = worker.get_finished(set(), metadata)

    assert tracker.store_phase.value == "DONE"
    assert tracker.reverse_phase.value == "DONE"
    assert tracker.forward_phase.value == "DONE"
    assert tracker.terminal_published is True
    assert first_finished == (set(), {DECODE_REQUEST_ID})
    assert second_finished == (set(), set())
    assert worker.get_block_ids_with_load_errors() == set()


def test_empty_reverse_creates_no_p2p_task_and_prefill_gate_starts_satisfied() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_store=False)
    with patch.object(worker, "_enqueue_kv_layer_send") as enqueue:
        worker.start_load_kv(metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]

    _set_forward_terminal(worker)
    finished = worker.get_finished(set(), metadata)

    enqueue.assert_not_called()
    assert tracker.reverse_phase.value == "SKIPPED"
    assert finished == (set(), {DECODE_REQUEST_ID})


def test_empty_store_and_reverse_still_waits_for_forward() -> None:
    worker = _make_worker()
    metadata = _make_split_metadata(include_store=False)
    worker.start_load_kv(metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]

    before_forward = worker.get_finished(set(), metadata)
    _set_forward_terminal(worker)
    after_forward = worker.get_finished(set(), metadata)

    assert tracker.store_phase.value == "SKIPPED"
    assert tracker.reverse_phase.value == "SKIPPED"
    assert before_forward == (set(), set())
    assert after_forward == (set(), {DECODE_REQUEST_ID})


def test_after_reverse_done_prefill_executes_inherited_layerwise_forward() -> None:
    worker = _make_prefill_worker()
    reverse_metadata = DualPathConnectorMetadata()
    binding = _make_reverse_receive_binding()
    reverse_metadata.reverse_receive_bindings.append(binding)
    worker.start_load_kv(reverse_metadata)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    # The Reverse terminal leaves the worker only as a completion completion id (I4).
    assert worker.get_finished(set(), reverse_metadata) == (set(), set())
    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.completed_jobs == {binding.reverse_completion_job_id: 1}

    forward_metadata = layerwise_module.MooncakeLayerwiseConnectorMetadata()
    forward_metadata.add_new_req(
        binding.prefill_request_id,
        [[90, 91, 92, 93, 94, 95, 96, 97]],
        {},
        prompt_len=128,
        local_computed_tokens=128,
        local_transed_tokens=64,
    )
    worker.current_layer = 0
    worker.total_layers = 2
    worker.index_to_name = {0: ["model.layer.0"], 1: ["model.layer.1"]}
    events = {
        "model.layer.0": SimpleNamespace(reshape_cache_event=MagicMock()),
        "model.layer.1": SimpleNamespace(reshape_cache_event=MagicMock()),
    }

    with patch.object(worker, "_enqueue_kv_layer_send") as enqueue:
        worker.save_kv_layer("", [MagicMock(), MagicMock()], events, forward_metadata)
        worker.save_kv_layer("", [MagicMock(), MagicMock()], events, forward_metadata)

    assert forward_metadata.requests[binding.prefill_request_id].local_transed_tokens == 64
    assert [call.kwargs["layer_name"] for call in enqueue.call_args_list] == ["model.layer.0", "model.layer.1"]


def test_reverse_submission_iterates_registered_layer_order_through_shared_enqueue_helper() -> None:
    worker = _make_worker()
    registered = _register_two_layers(worker)
    metadata = _make_split_metadata(include_store=False)
    reverse_metadata = layerwise_module.MooncakeLayerwiseConnectorMetadata()
    ready_event = MagicMock(name="reverse_ready_event")

    worker.start_load_kv(metadata)
    worker._install_reverse_plan(_make_reverse_plan())
    with (
        patch.object(worker, "_build_reverse_send_metadata", return_value=reverse_metadata),
        patch.object(worker, "_enqueue_kv_layer_send") as enqueue,
        patch.object(torch.npu, "Event", return_value=ready_event),
    ):
        worker._submit_reverse(DECODE_REQUEST_ID)

    assert ready_event.record.call_count == 1
    assert enqueue.call_count == 2
    for call_args, (layer_index, layer_name) in zip(
        enqueue.call_args_list,
        worker._registered_layer_order,
        strict=True,
    ):
        assert call_args.kwargs == {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "kv_layer": registered[layer_name],
            "ready_event": ready_event,
            "metadata": reverse_metadata,
        }


def test_store_done_submits_every_reverse_layer_exactly_once() -> None:
    worker = _make_worker()
    _register_two_layers(worker)
    metadata = _make_split_metadata(include_reverse=True)

    with (
        patch.object(worker, "_build_reverse_send_metadata", return_value=MagicMock()),
        patch.object(worker, "_enqueue_kv_layer_send") as enqueue,
        patch.object(torch.npu, "Event", return_value=MagicMock()),
    ):
        worker.start_load_kv(metadata)
        assert enqueue.call_count == 0
        worker._consume_store_completions({DECODE_REQUEST_ID}, set())

    assert enqueue.call_count == 2
    assert worker._split_trackers[DECODE_REQUEST_ID].store_phase.value == "DONE"
    assert worker._split_trackers[DECODE_REQUEST_ID].reverse_submitted_attempt is not None


def test_empty_store_submits_reverse_immediately_after_installation() -> None:
    worker = _make_worker()
    _register_two_layers(worker)
    metadata = _make_split_metadata(include_store=False, include_reverse=True)

    with (
        patch.object(worker, "_build_reverse_send_metadata", return_value=MagicMock()),
        patch.object(worker, "_enqueue_kv_layer_send") as enqueue,
        patch.object(torch.npu, "Event", return_value=MagicMock()),
    ):
        worker.start_load_kv(metadata)

    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    assert enqueue.call_count == 2
    assert tracker.store_phase.value == "SKIPPED"
    assert tracker.reverse_phase.value == "PENDING"
    assert tracker.reverse_plan is metadata.reverse_plans[0]
    assert tracker.reverse_submitted_attempt is not None


def test_duplicate_store_done_never_enqueues_reverse_twice() -> None:
    worker = _make_worker()
    _register_two_layers(worker)
    metadata = _make_split_metadata(include_reverse=True)

    with (
        patch.object(worker, "_build_reverse_send_metadata", return_value=MagicMock()),
        patch.object(worker, "_enqueue_kv_layer_send") as enqueue,
        patch.object(torch.npu, "Event", return_value=MagicMock()),
    ):
        worker.start_load_kv(metadata)
        worker._consume_store_completions({DECODE_REQUEST_ID}, set())
        calls_after_first_done = enqueue.call_count
        worker._consume_store_completions({DECODE_REQUEST_ID}, set())

    assert calls_after_first_done == 2
    assert enqueue.call_count == calls_after_first_done


def test_reverse_metadata_adapts_plan_through_parent_split_primitives() -> None:
    worker = _make_worker()
    plan = _make_reverse_plan()

    def transfer_mappings(*args):
        req_meta = args[4]
        assert req_meta.local_block_ids == [list(plan.source_block_ids[0])]
        assert req_meta.remote_block_ids == [list(plan.destination_block_ids[0])]
        assert req_meta.remote_block_size == list(plan.remote_block_sizes)
        assert req_meta.chunk_finish is True
        return {
            (plan.remote_host, plan.remote_port): {
                "local_block_ids": [11, 20, 21],
                "remote_block_ids": [71, 80, 81],
                "trans_count": 1,
            }
        }

    with (
        patch.object(layerwise_module, "get_cp_group", return_value=[[0]]),
        patch.object(layerwise_module, "context_parallel_parameters_check"),
        patch.object(
            layerwise_module,
            "get_local_remote_block_port_mappings",
            return_value=({}, {}, {}, {}),
        ),
        patch.object(layerwise_module, "get_transfer_mappings", side_effect=transfer_mappings),
    ):
        metadata = worker._build_reverse_send_metadata(plan, DECODE_REQUEST_ID)

    assert set(metadata.requests) == {DECODE_REQUEST_ID}
    req_meta = metadata.requests[DECODE_REQUEST_ID]
    assert req_meta.local_block_ids == [[11, 20, 21]]
    assert req_meta.remote_block_ids == [[71, 80, 81]]
    assert req_meta.remote_block_size == [16]
    assert req_meta.remote_engine_id == plan.remote_engine_id
    assert req_meta.remote_host == plan.remote_host
    assert req_meta.remote_port == plan.remote_port
    assert req_meta.remote_te_rpc_port is None
    assert req_meta.remote_layer_metadata is None
    assert req_meta.token_ids is None
    assert req_meta.metaserver is None
    assert req_meta.prompt_len == plan.token_end
    assert req_meta.local_computed_tokens == plan.token_end
    assert req_meta.local_transed_tokens == plan.token_start
    assert req_meta.remote_cache_tokens == 0
    assert req_meta.chunk_finish is True
    assert req_meta.trans_count == [1]


def test_reverse_metadata_final_registered_layer_is_the_only_callback_eligible_task() -> None:
    worker = _make_worker()
    worker.total_layers = 2
    worker.layer_metadata = {
        "model.layer.0": SimpleNamespace(tensor_group_idx=[0]),
        "model.layer.1": SimpleNamespace(tensor_group_idx=[0]),
    }
    worker.kv_send_layer_thread = MagicMock()
    metadata = layerwise_module.MooncakeLayerwiseConnectorMetadata()
    req_meta = MagicMock(chunk_finish=True, local_block_ids=[[11]], remote_layer_metadata={})
    metadata.requests[DECODE_REQUEST_ID] = req_meta

    with patch.object(worker, "update_decoder_info", side_effect=lambda _req_id, meta: meta):
        for layer_index, layer_name in ((0, "model.layer.0"), (1, "model.layer.1")):
            worker._enqueue_kv_layer_send(
                layer_index=layer_index,
                layer_name=layer_name,
                kv_layer=[MagicMock(), MagicMock()],
                ready_event=MagicMock(),
                metadata=metadata,
            )

    send_tasks = [put_call.args[0] for put_call in worker.kv_send_layer_thread.send_queue.put.call_args_list]
    assert [task.layer_idx for task in send_tasks] == [0, worker.total_layers - 1]
    assert sum(task.layer_idx == worker.total_layers - 1 for task in send_tasks) == 1
    assert send_tasks[-1].send_request[DECODE_REQUEST_ID].chunk_finish is True

# SPDX-License-Identifier: Apache-2.0
"""Scheduler integration acceptance for DualPath Decode admission and completion.

Uses the real vLLM v1 Scheduler on CPU with the real ``DualPathConnector``;
only the KVPool backend module and the ``LookupKeyClient`` transport are
constrained at their existing seams. Proves that for ``L_DE < R`` one
``schedule()`` call admits the request into ``WAITING_FOR_REMOTE_KVS`` with
final blocks bound in a ``DecodeKVSnapshot``, and that an HBM-complete
request takes the normal local path with no Task-01 state. It also proves that
an explicit PE ABORT reaches ``FINISHED_ERROR`` and releases delayed blocks
through the Worker/Core relay. Task-06 coverage drives the complete
DE-local Store-full success and probe/load-race failure lifecycles through the
same real Scheduler, including final-token recomputation and delayed-block
release without Proxy, PE, Decision, Forward, or Reverse activity.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.v1.structured_output import StructuredOutputManager  # noqa: E402

from tests.ut.distributed.kv_transfer.dual_path.conftest import init_dual_path_worker_state  # noqa: E402
from tests.ut.distributed.kv_transfer.dual_path.test_de_local_store_full import (  # noqa: E402
    _make_real_store_worker_adapter,
)
from vllm_ascend.distributed.kv_transfer import register_connector  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (  # noqa: E402
    DualPathConnector,
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (  # noqa: E402
    KVPoolWorkerAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (  # noqa: E402
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (  # noqa: E402
    DualPathRequestKey,
    PathAbortNotice,
    PathAbortReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (  # noqa: E402
    DecodeControlEndpoint,
)

_BLOCK_SIZE = 16
_NONE_HASH_INITIALIZED = False
_CONNECTOR_REGISTERED = False


def _ensure_connector_registered() -> None:
    global _CONNECTOR_REGISTERED
    if not _CONNECTOR_REGISTERED:
        register_connector()
        _CONNECTOR_REGISTERED = True


def _make_vllm_config(kv_role: str = "kv_consumer") -> VllmConfig:
    fake_weight_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "_fake_weight")
    model_config = ModelConfig(model=fake_weight_path, skip_tokenizer_init=True)
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=_BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="DualPathConnector",
        kv_role=kv_role,
        kv_connector_extra_config={
            "role": "decode",
            "consumer_is_to_load": True,
            "load_async": True,
            "backend": "mooncake",
            "lookup_rpc_port": 18883,
            "dual_path_control_port": 24001,
        },
    )
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def _make_scheduler(vllm_config: VllmConfig, num_blocks: int = 1000) -> Scheduler:
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(block_size=_BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float16),
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        block_size=_BLOCK_SIZE,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _make_request(request_id: str, prompt_token_ids: list[int], kv_transfer_params: dict | None) -> Request:
    global _NONE_HASH_INITIALIZED
    if not _NONE_HASH_INITIALIZED:
        init_none_hash(sha256)
        _NONE_HASH_INITIALIZED = True
    sampling_params = SamplingParams(max_tokens=8)
    request = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(_BLOCK_SIZE, sha256),
    )
    request.kv_transfer_params = kv_transfer_params
    return request


def _runner_output_for(requests: list[Request]) -> ModelRunnerOutput:
    req_ids = [request.request_id for request in requests]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=KVConnectorOutput(finished_sending=set(), finished_recving=set()),
    )


def _make_bare_worker(
    load_result: list[int] | None,
) -> tuple[DualPathConnectorWorker, KVPoolWorkerAdapter, MagicMock]:
    worker = init_dual_path_worker_state(object.__new__(DualPathConnectorWorker))
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False))
    worker.kv_recv_layer_thread = MagicMock(name="kv_recv_layer_thread")
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    worker.request_map = {}
    worker.virtual_request = set()
    worker._recving_metadata = {}
    worker._invalid_block_ids = set()
    adapter, backend = _make_real_store_worker_adapter(load_result)
    worker._kvpool_worker_adapter = adapter
    worker.engine = MagicMock(name="transfer_engine")
    worker.block_size = [_BLOCK_SIZE]
    return worker, adapter, backend


def test_integration_worker_completion_source_is_real_kvpool_adapter() -> None:
    worker, adapter, _ = _make_bare_worker([0, 0])

    assert worker._kvpool_worker_adapter is adapter
    assert isinstance(adapter, KVPoolWorkerAdapter)


@pytest.fixture(autouse=True)
def _constrain_kvpool_seams():
    """Constrain the KVPool backend resolution and lookup transport at their
    existing seams; everything between vLLM core and the adapter stays real."""
    _ensure_connector_registered()
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
        ) as mock_lookup_client_cls,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.PathDecisionCoordinator"
        ) as coordinator_cls,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.get_ip",
            return_value="127.0.0.1",
        ),
    ):
        decode_coordinator = MagicMock(name="decode_coordinator")
        decode_coordinator.decode_engine_instance_id = "integration-engine:0:test-boot"
        decode_coordinator.decode_control_endpoint = DecodeControlEndpoint(host="127.0.0.1", port=24001)
        admission_ids = iter(range(1_000_000))
        decode_coordinator.new_request_key.side_effect = lambda request_id: DualPathRequestKey(
            "integration-engine:0:test-boot",
            request_id,
            next(admission_ids),
        )
        coordinator_cls.for_decode.return_value = decode_coordinator
        mock_importlib.import_module.return_value = MagicMock()
        yield mock_lookup_client_cls


@pytest.fixture()
def scheduler():
    instance = _make_scheduler(_make_vllm_config())
    yield instance
    instance.shutdown()


def _dual_scheduler(scheduler: Scheduler) -> DualPathConnectorScheduler:
    connector = scheduler.connector
    assert isinstance(connector, DualPathConnector)
    dual = connector.connector_scheduler
    assert isinstance(dual, DualPathConnectorScheduler)
    return dual


def _admit_one_request(scheduler: Scheduler):
    """Schedule one DualPath Decode request (P=33, R=32, L_DE=0) and return the
    request, the SchedulerOutput, the matched-token return capture, and the
    lookup/allocate spies of the admission step."""
    dual = _dual_scheduler(scheduler)
    matched_returns: list[tuple[int, bool]] = []
    real_matched = scheduler.connector.get_num_new_matched_tokens

    def capture_matched(request, num_computed_tokens):
        result = real_matched(request, num_computed_tokens)
        matched_returns.append(result)
        return result

    request = _make_request(
        "req-de",
        list(range(33)),
        {"do_remote_prefill": True, "do_virtual": True},
    )
    scheduler.add_request(request)
    with (
        patch.object(dual._kvpool_adapter, "lookup", wraps=dual._kvpool_adapter.lookup) as lookup_mock,
        patch.object(scheduler.connector, "get_num_new_matched_tokens", side_effect=capture_matched) as matched_mock,
        patch.object(
            scheduler.kv_cache_manager, "allocate_slots", wraps=scheduler.kv_cache_manager.allocate_slots
        ) as alloc_mock,
    ):
        scheduler_output = scheduler.schedule()
    return request, scheduler_output, matched_returns, lookup_mock, matched_mock, alloc_mock


def _assert_admission_invariants(
    scheduler,
    request,
    scheduler_output,
    matched_returns,
    alloc_mock,
    *,
    expect_store_spec,
    store_full,
):
    dual = _dual_scheduler(scheduler)
    expected_external_tokens = 32 if store_full else 33
    assert matched_returns == [(expected_external_tokens, True)]
    # 2. allocate_slots received the external delta with delayed caching
    assert alloc_mock.call_args.kwargs["num_external_computed_tokens"] == expected_external_tokens
    assert alloc_mock.call_args.kwargs["delay_cache_blocks"] is True
    # 3. final block IDs exist and equal the snapshot's frozen IDs
    final_block_ids = tuple(
        tuple(group) for group in scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    )
    snapshot = dual._decode_kv_snapshots[request.request_id]
    assert snapshot.final_block_ids == final_block_ids
    assert snapshot.transfer_tokens == 33
    assert snapshot.local_tokens == 0
    assert snapshot.external_tokens == expected_external_tokens
    if expect_store_spec:
        assert snapshot.store_load_spec is not None
    else:
        assert snapshot.store_load_spec is None
    # 4. the request waits for remote KVs
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # 5. vLLM recorded the Layerwise transfer target for the pending receive
    assert request.num_computed_tokens == expected_external_tokens
    # 6. no model tokens were scheduled for the request in this step
    assert scheduler_output.num_scheduled_tokens.get(request.request_id, 0) == 0
    # 7. connector metadata contains no Store or P2P work for the request
    metadata = scheduler_output.kv_connector_metadata
    assert metadata is None or request.request_id not in metadata.requests
    assert dual._reqs_need_recv == {}
    if store_full:
        dual._path_decision_coordinator.register_pending.assert_not_called()
    else:
        dual._path_decision_coordinator.register_pending.assert_called_once()
    # 8. no finished_recving completion is published
    scheduler.update_from_output(scheduler_output, _runner_output_for([]))
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert scheduler.finished_recving_kv_req_ids == set()


def test_admission_full_store_hit(_constrain_kvpool_seams, scheduler):
    _constrain_kvpool_seams.return_value.lookup.return_value = 32
    request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)
    lookup_mock.assert_called_once()
    _assert_admission_invariants(
        scheduler,
        request,
        scheduler_output,
        matched_returns,
        alloc_mock,
        expect_store_spec=True,
        store_full=True,
    )
    snapshot = _dual_scheduler(scheduler)._decode_kv_snapshots[request.request_id]
    assert snapshot.store_load_spec.kvpool_cached_tokens == 32
    assert snapshot.store_tokens == 32


def test_kv_both_store_full_admission_commits_through_adapter_metadata_without_partial_state(
    _constrain_kvpool_seams,
):
    # Given
    scheduler = _make_scheduler(_make_vllm_config(kv_role="kv_both"))
    _constrain_kvpool_seams.return_value.lookup.return_value = 32
    dual = _dual_scheduler(scheduler)
    adapter = dual._kvpool_adapter
    assert adapter is not None

    try:
        # When
        with patch.object(adapter, "commit_after_alloc", wraps=adapter.commit_after_alloc) as commit_after_alloc:
            request, scheduler_output, _, _, _, _ = _admit_one_request(scheduler)

        # Then
        commit_after_alloc.assert_called_once()
        metadata = scheduler_output.kv_connector_metadata
        assert isinstance(metadata, DualPathConnectorMetadata)
        store_metadata = metadata.decode_store_metadata
        assert store_metadata is not None
        assert [store_request.req_id for store_request in store_metadata.requests] == [request.request_id]
        assert dual._lookup_results == {}
        assert set(dual._decode_kv_snapshots) == {request.request_id}
        assert dual._decode_decision_states == {}
        pool = adapter._pool_scheduler
        assert pool.load_specs == {}
        assert set(pool._unfinished_requests) == {request.request_id}
        assert pool._unfinished_request_ids == {request.request_id}
        assert pool._loading_req_ids == {request.request_id}
    finally:
        scheduler.shutdown()


def test_admission_partial_store_hit(_constrain_kvpool_seams, scheduler):
    _constrain_kvpool_seams.return_value.lookup.return_value = 16
    request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)
    lookup_mock.assert_called_once()
    _assert_admission_invariants(
        scheduler,
        request,
        scheduler_output,
        matched_returns,
        alloc_mock,
        expect_store_spec=True,
        store_full=False,
    )
    snapshot = _dual_scheduler(scheduler)._decode_kv_snapshots[request.request_id]
    # Partial hit does not change the Core-facing external delta.
    assert snapshot.store_load_spec.kvpool_cached_tokens == 16
    assert snapshot.external_tokens == 33


def test_admission_store_miss(_constrain_kvpool_seams, scheduler):
    _constrain_kvpool_seams.return_value.lookup.return_value = 0
    request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)
    lookup_mock.assert_called_once()
    _assert_admission_invariants(
        scheduler,
        request,
        scheduler_output,
        matched_returns,
        alloc_mock,
        expect_store_spec=False,
        store_full=False,
    )


def test_hbm_complete_schedules_normally_without_task01_state(_constrain_kvpool_seams, scheduler):
    dual = _dual_scheduler(scheduler)
    prompt = list(range(33))

    # Prime the prefix cache with an identical ordinary request.
    primer = _make_request("req-primer", prompt, None)
    scheduler.add_request(primer)
    primer_output = scheduler.schedule()
    assert primer_output.num_scheduled_tokens[primer.request_id] == 33
    scheduler.update_from_output(primer_output, _runner_output_for([primer]))
    scheduler.finish_requests([primer.request_id], RequestStatus.FINISHED_STOPPED)

    lookup_spy = patch.object(dual._kvpool_adapter, "lookup", wraps=dual._kvpool_adapter.lookup)
    request = _make_request(
        "req-de-hbm",
        prompt,
        {"do_remote_prefill": True, "do_virtual": True},
    )
    scheduler.add_request(request)
    with lookup_spy as lookup_mock:
        scheduler_output = scheduler.schedule()

    # L_DE = 32 = R: no KVPool lookup, no Task-01 state, normal local scheduling
    # with last-token recomputation.
    lookup_mock.assert_not_called()
    assert dual._lookup_results == {}
    assert dual._decode_kv_snapshots == {}
    assert scheduler_output.num_scheduled_tokens[request.request_id] == 1
    assert request.status == RequestStatus.RUNNING


def test_store_full_success_completes_locally_and_recomputes_last_token(
    _constrain_kvpool_seams,
    scheduler,
):
    # Given
    _constrain_kvpool_seams.return_value.lookup.return_value = 32
    dual = _dual_scheduler(scheduler)
    coordinator = dual._path_decision_coordinator
    pool_scheduler = dual._kvpool_adapter._pool_scheduler
    committed_specs = []
    real_build_connector_meta = dual._kvpool_adapter.build_connector_meta

    def capture_committed_spec(scheduler_output):
        committed_specs.append(pool_scheduler.load_specs["req-de"])
        return real_build_connector_meta(scheduler_output)

    with (
        patch.object(
            dual._kvpool_adapter,
            "build_connector_meta",
            side_effect=capture_committed_spec,
        ),
        patch.object(dual, "_access_metaserver") as proxy_http,
    ):
        request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)

    snapshot = dual._decode_kv_snapshots[request.request_id]
    metadata = scheduler_output.kv_connector_metadata
    assert isinstance(metadata, DualPathConnectorMetadata)
    store_metadata = metadata.decode_store_metadata
    assert store_metadata is not None
    assert matched_returns == [(32, True)]
    assert alloc_mock.call_args.kwargs["num_external_computed_tokens"] == 32
    assert alloc_mock.call_args.kwargs["delay_cache_blocks"] is True
    lookup_mock.assert_called_once()
    assert snapshot.final_block_ids == tuple(
        tuple(group) for group in scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    )
    assert snapshot.local_tokens == 0
    assert snapshot.external_tokens == 32
    assert snapshot.store_load_spec is not None
    assert snapshot.store_load_spec.can_load is False
    assert committed_specs[0] is not snapshot.store_load_spec
    assert committed_specs[0].can_load is True
    assert pool_scheduler._loading_req_ids == {request.request_id}
    assert pool_scheduler.load_specs == {}
    assert metadata.requests == {}
    assert len(store_metadata.requests) == 1
    store_request = store_metadata.requests[0]
    assert store_request.req_id == request.request_id
    assert store_request.target_token_len == 32
    assert store_request.load_spec is committed_specs[0]
    assert store_metadata.loading_req_ids == {request.request_id}
    assert scheduler_output.num_scheduled_tokens.get(request.request_id, 0) == 0
    assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    assert dual._decode_decision_states == {}
    coordinator.register_pending.assert_not_called()
    coordinator.submit.assert_not_called()
    coordinator.unregister.assert_not_called()
    proxy_http.assert_not_called()

    worker, adapter, backend = _make_bare_worker([0, 0])

    # When
    worker.start_load_kv(metadata)
    recv_thread = adapter._pool_worker.kv_recv_thread
    assert recv_thread is not None
    recv_thread.request_queue.join()
    finished_sending, finished_recving = worker.get_finished(set(), metadata)
    connector_output = KVConnectorOutput(
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        invalid_block_ids=worker.get_block_ids_with_load_errors(),
    )
    runner_output = _runner_output_for([])
    runner_output.kv_connector_output = connector_output
    scheduler.update_from_output(scheduler_output, runner_output)
    allocation_state = []
    real_allocate_slots = scheduler.kv_cache_manager.allocate_slots

    def capture_resumed_allocation(resumed_request, *args, **kwargs):
        allocation_state.append((resumed_request.status, resumed_request.num_computed_tokens))
        return real_allocate_slots(resumed_request, *args, **kwargs)

    with patch.object(
        scheduler.kv_cache_manager,
        "allocate_slots",
        side_effect=capture_resumed_allocation,
    ):
        resumed_output = scheduler.schedule()

    # Then
    assert connector_output.finished_recving == {request.request_id}
    assert connector_output.invalid_block_ids == set()
    assert allocation_state == [(RequestStatus.WAITING, 32)]
    assert resumed_output.num_scheduled_tokens[request.request_id] == 1
    assert request.status is RequestStatus.RUNNING
    backend.get.assert_called_once()
    worker.kv_recv_layer_thread.get_and_clear_done_requests.assert_called_once_with()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.assert_called_once_with()
    assert worker.engine.method_calls == []
    assert worker._forward_receive_bindings == {}
    assert dual._prefill_request_keys == {}
    assert dual._prefill_path_results == {}
    assert dual._prefill_forward_plans == {}
    assert dual._prefill_delivery_futures == {}
    assert dual._reqs_need_recv == {}


def test_store_full_failure_fails_closed_and_releases_delayed_blocks(
    _constrain_kvpool_seams,
    scheduler,
):
    # Given
    prompt = list(range(33))
    primer = _make_request("req-primer-store-failure", prompt[:17], None)
    scheduler.add_request(primer)
    primer_output = scheduler.schedule()
    scheduler.update_from_output(primer_output, _runner_output_for([primer]))
    scheduler.finish_requests([primer.request_id], RequestStatus.FINISHED_STOPPED)
    block_pool = scheduler.kv_cache_manager.block_pool
    baseline_free_blocks = block_pool.free_block_queue.num_free_blocks
    _constrain_kvpool_seams.return_value.lookup.return_value = 32
    dual = _dual_scheduler(scheduler)
    coordinator = dual._path_decision_coordinator

    request = _make_request(
        "req-de-store-failure",
        prompt,
        {"do_remote_prefill": True, "do_virtual": True},
    )
    scheduler.add_request(request)
    matched_returns = []
    real_matched = scheduler.connector.get_num_new_matched_tokens

    def capture_matched(admitted_request, num_computed_tokens):
        result = real_matched(admitted_request, num_computed_tokens)
        matched_returns.append(result)
        return result

    with (
        patch.object(scheduler.connector, "get_num_new_matched_tokens", side_effect=capture_matched),
        patch.object(
            scheduler.kv_cache_manager,
            "allocate_slots",
            wraps=scheduler.kv_cache_manager.allocate_slots,
        ) as alloc_mock,
        patch.object(dual, "_access_metaserver") as proxy_http,
    ):
        scheduler_output = scheduler.schedule()

    snapshot = dual._decode_kv_snapshots[request.request_id]
    destination_blocks = snapshot.final_block_ids[0]
    hbm_prefix_blocks = set(destination_blocks[:1])
    failed_suffix_blocks = set(destination_blocks[1:2])
    metadata = scheduler_output.kv_connector_metadata
    assert isinstance(metadata, DualPathConnectorMetadata)
    store_metadata = metadata.decode_store_metadata
    assert store_metadata is not None
    assert matched_returns == [(16, True)]
    assert alloc_mock.call_args.kwargs["num_external_computed_tokens"] == 16
    assert alloc_mock.call_args.kwargs["delay_cache_blocks"] is True
    assert snapshot.local_tokens == 16
    assert snapshot.external_tokens == 16
    assert len(destination_blocks) == 2
    assert failed_suffix_blocks
    assert block_pool.free_block_queue.num_free_blocks < baseline_free_blocks
    assert scheduler_output.num_scheduled_tokens.get(request.request_id, 0) == 0
    assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    assert dual._decode_decision_states == {}
    coordinator.register_pending.assert_not_called()
    coordinator.submit.assert_not_called()
    coordinator.unregister.assert_not_called()
    proxy_http.assert_not_called()

    worker, adapter, backend = _make_bare_worker(None)

    # When
    worker.start_load_kv(metadata)
    recv_thread = adapter._pool_worker.kv_recv_thread
    assert recv_thread is not None
    recv_thread.request_queue.join()
    finished_sending, finished_recving = worker.get_finished(set(), metadata)
    connector_output = KVConnectorOutput(
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        invalid_block_ids=worker.get_block_ids_with_load_errors(),
    )
    runner_output = _runner_output_for([])
    runner_output.kv_connector_output = connector_output
    scheduler.update_from_output(scheduler_output, runner_output)

    # Then
    assert connector_output.finished_recving == {request.request_id}
    assert connector_output.invalid_block_ids == failed_suffix_blocks
    assert connector_output.invalid_block_ids.isdisjoint(hbm_prefix_blocks)
    assert request.status is RequestStatus.FINISHED_ERROR
    assert request.request_id not in scheduler.requests
    assert block_pool.free_block_queue.num_free_blocks == baseline_free_blocks
    backend.get.assert_called_once()
    assert worker.engine.method_calls == []
    assert worker._forward_receive_bindings == {}
    assert dual._decode_decision_states == {}
    assert dual._prefill_request_keys == {}
    assert dual._prefill_path_results == {}
    assert dual._prefill_forward_plans == {}
    assert dual._prefill_delivery_futures == {}
    assert dual._reqs_need_recv == {}


class TestAbortIntegration:
    def test_pending_request_waits_until_abort_then_finishes_and_releases_delayed_blocks(
        self,
        _constrain_kvpool_seams,
    ):
        # Given
        _constrain_kvpool_seams.return_value.lookup.return_value = 0
        scheduler = _make_scheduler(_make_vllm_config())
        dual = _dual_scheduler(scheduler)
        coordinator = dual._path_decision_coordinator
        coordinator.take_received_decisions.return_value = []
        coordinator.take_received_aborts.return_value = []
        block_pool = scheduler.kv_cache_manager.block_pool
        baseline_free_blocks = block_pool.free_block_queue.num_free_blocks

        try:
            with (
                patch.object(
                    scheduler_module,
                    "time",
                    SimpleNamespace(monotonic=lambda: 100.0),
                    create=True,
                ),
                patch.object(dual, "_access_metaserver") as proxy_http,
            ):
                request, scheduler_output, matched_returns, lookup_mock, _, alloc_mock = _admit_one_request(scheduler)

            state = dual._decode_decision_states[request.request_id]
            snapshot = dual._decode_kv_snapshots[request.request_id]
            external_block_ids = snapshot.final_block_ids[0]

            assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
            assert matched_returns == [(33, True)]
            assert alloc_mock.call_args.kwargs["num_external_computed_tokens"] == 33
            assert alloc_mock.call_args.kwargs["delay_cache_blocks"] is True
            lookup_mock.assert_called_once()
            coordinator.register_pending.assert_called_once_with(state.request_key)
            proxy_http.assert_not_called()
            coordinator.submit.assert_not_called()
            assert state.status is scheduler_module._DecodeDecisionStatus.PENDING
            assert dual._reqs_need_recv == {}
            assert block_pool.free_block_queue.num_free_blocks < baseline_free_blocks

            scheduler.update_from_output(scheduler_output, _runner_output_for([]))

            # The request remains parked without an explicit outcome, even after
            # the former Decision timeout window would have elapsed.
            with patch.object(
                scheduler_module,
                "time",
                SimpleNamespace(monotonic=lambda: 10_000.0),
                create=True,
            ):
                waiting_scheduler_output = scheduler.schedule()
            waiting_metadata = waiting_scheduler_output.kv_connector_metadata
            assert isinstance(waiting_metadata, DualPathConnectorMetadata)
            assert waiting_metadata.control_failures == []
            assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
            assert block_pool.free_block_queue.num_free_blocks < baseline_free_blocks

            coordinator.take_received_aborts.return_value = [
                PathAbortNotice(
                    request_key=state.request_key,
                    reason=PathAbortReason.ACTIVATION_FAILED,
                )
            ]

            # When
            abort_scheduler_output = scheduler.schedule()

            metadata = abort_scheduler_output.kv_connector_metadata
            assert isinstance(metadata, DualPathConnectorMetadata)
            assert metadata.requests == {}
            assert metadata.control_failures == [
                DualPathControlFailureMetadata(
                    request_id=request.request_id,
                    invalid_block_ids=external_block_ids,
                    reason=DualPathControlFailureReason.PEER_ABORT,
                )
            ]

            worker, _, backend = _make_bare_worker([0, 0])

            worker.start_load_kv(metadata)
            assert worker.kv_recv_layer_thread.method_calls == []
            backend.get.assert_not_called()
            assert worker.engine.method_calls == []
            finished_sending, finished_recving = worker.get_finished(set(), metadata)
            invalid_block_ids = worker.get_block_ids_with_load_errors()
            connector_output = KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                invalid_block_ids=invalid_block_ids,
            )
            model_runner_output = _runner_output_for([])
            model_runner_output.kv_connector_output = connector_output
            scheduler.update_from_output(abort_scheduler_output, model_runner_output)

            # Then
            assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
            assert connector_output.finished_recving == {request.request_id}
            assert connector_output.invalid_block_ids == set(external_block_ids)
            assert request.status is RequestStatus.FINISHED_ERROR
            assert request.request_id not in scheduler.requests
            assert block_pool.free_block_queue.num_free_blocks == baseline_free_blocks
            backend.get.assert_not_called()
            assert worker.engine.method_calls == []
            assert dual._reqs_need_recv == {}
        finally:
            scheduler.shutdown()

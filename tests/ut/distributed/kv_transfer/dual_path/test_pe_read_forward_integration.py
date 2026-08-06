# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle integration coverage for the DualPath PE_READ route."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from tests.ut.distributed.kv_transfer.dual_path import test_decode_admission_integration as admission_harness
from tests.ut.distributed.kv_transfer.dual_path import test_pe_read_forward as forward_harness
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ForwardPlan,
    ForwardReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    Path,
    PathDecisionRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorWorker,
    get_external_request_id,
)

_BLOCK_SIZE = 16
_PROMPT_TOKENS = 33
_DECODE_LOCAL_TOKENS = 16
_SOURCE_BLOCK_IDS = ([101, 102, 103],)


class AlwaysPEReadPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def choose(self, request: PathDecisionRequest) -> Path:
        self.calls += 1
        return Path.PE_READ


@dataclass(frozen=True, slots=True)
class LifecycleHarness:
    scheduler: Scheduler
    decode_request: Request
    decode_scheduler: DualPathConnectorScheduler
    pe_scheduler: DualPathConnectorScheduler
    pe_request: SimpleNamespace
    pe_policy: AlwaysPEReadPolicy
    pe_metadata: SimpleNamespace
    binding_output: SimpleNamespace
    binding: ForwardReceiveBinding
    worker: DualPathConnectorWorker
    forward_plan: ForwardPlan
    proxy_http: MagicMock
    backend: MagicMock
    baseline_free_blocks: int


def _completed_future() -> Future[None]:
    future: Future[None] = Future()
    future.set_result(None)
    return future


def _make_worker() -> DualPathConnectorWorker:
    worker = object.__new__(DualPathConnectorWorker)
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False))
    worker.kv_recv_layer_thread = MagicMock(name="kv_recv_layer_thread")
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    worker.request_map = {}
    worker.virtual_request = set()
    worker._recving_metadata = {}
    worker._invalid_block_ids = set()
    worker._control_failed_recving = set()
    worker._forward_receive_bindings = {}
    worker._pending_forward_done = set()
    worker._pending_forward_failed = set()
    worker._consumed_forward_terminals = {}
    worker._kvpool_worker_adapter = MagicMock(name="kvpool_worker_adapter")
    worker.engine = MagicMock(name="transfer_engine")
    worker.block_size = [_BLOCK_SIZE]
    return worker


def _prime_one_hbm_block(scheduler: Scheduler) -> None:
    primer = admission_harness._make_request("req-primer", list(range(17)), None)
    scheduler.add_request(primer)
    primer_output = scheduler.schedule()
    assert primer_output.num_scheduled_tokens[primer.request_id] == 17
    scheduler.update_from_output(primer_output, admission_harness._runner_output_for([primer]))
    scheduler.finish_requests([primer.request_id], RequestStatus.FINISHED_STOPPED)


def _make_pe_request(message: dict) -> SimpleNamespace:
    prompt_token_ids = list(range(_PROMPT_TOKENS))
    return SimpleNamespace(
        request_id="req-pe",
        num_tokens=_PROMPT_TOKENS,
        num_prompt_tokens=_PROMPT_TOKENS,
        num_computed_tokens=0,
        max_tokens=8,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=None,
        all_token_ids=list(prompt_token_ids),
        _all_token_ids=list(prompt_token_ids),
        kv_transfer_params=dict(message),
    )


def _make_pe_scheduler() -> tuple[DualPathConnectorScheduler, AlwaysPEReadPolicy]:
    coordinator = connector_module.PathDecisionCoordinator.for_prefill.return_value
    coordinator.submit.return_value = _completed_future()
    policy = AlwaysPEReadPolicy()
    scheduler = DualPathConnectorScheduler(
        forward_harness._make_vllm_config(),
        forward_harness._make_kv_cache_config(),
        "prefill-engine",
        DualPathConfig(role="prefill"),
        path_policy=policy,
    )
    scheduler.executor.shutdown(wait=False)
    scheduler.metaserver_client.close()
    scheduler.executor = MagicMock(name="prefill_executor")
    return scheduler, policy


def _build_parent_metadata(
    scheduler: DualPathConnectorScheduler,
    request: SimpleNamespace,
) -> SimpleNamespace:
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id, num_computed_tokens=0)],
        num_scheduled_tokens={request.request_id: request.num_prompt_tokens},
    )
    metadata = scheduler.build_connector_meta(scheduler_output)
    request_metadata = metadata.requests[request.request_id]
    with patch.object(MooncakeLayerwiseConnectorWorker, "save_kv_layer", autospec=True) as layer_send:
        layer_send(MagicMock(), "layer.0", [], MagicMock(), metadata)
        layer_send.assert_called_once()
    return request_metadata


def _build_lifecycle(
    scheduler: Scheduler,
    backend: MagicMock,
    *,
    defer_first_allocation: bool,
) -> LifecycleHarness:
    _prime_one_hbm_block(scheduler)
    decode_scheduler = admission_harness._dual_scheduler(scheduler)
    baseline_free_blocks = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    captured_messages: list[dict] = []
    real_message_builder = connector_module.build_remote_decode_message

    def capture_message(*args, **kwargs) -> dict:
        message = real_message_builder(*args, **kwargs)
        captured_messages.append(message)
        return message

    matched_returns: list[tuple[int, bool]] = []
    real_matched = scheduler.connector.get_num_new_matched_tokens

    def capture_matched(request: Request, num_computed_tokens: int) -> tuple[int, bool]:
        result = real_matched(request, num_computed_tokens)
        matched_returns.append(result)
        return result

    decode_request = admission_harness._make_request(
        "req-de-00000001",
        list(range(_PROMPT_TOKENS)),
        {"do_remote_prefill": True, "do_virtual": True},
    )
    scheduler.add_request(decode_request)
    with (
        patch.object(connector_module, "build_remote_decode_message", side_effect=capture_message),
        patch.object(decode_scheduler, "_access_metaserver") as proxy_http,
        patch.object(scheduler.connector, "get_num_new_matched_tokens", side_effect=capture_matched),
        patch.object(
            scheduler.kv_cache_manager,
            "allocate_slots",
            wraps=scheduler.kv_cache_manager.allocate_slots,
        ) as allocate_slots,
    ):
        admission_output = scheduler.schedule()

    assert matched_returns == [(_PROMPT_TOKENS - _DECODE_LOCAL_TOKENS, True)]
    assert allocate_slots.call_args.kwargs["num_external_computed_tokens"] == 17
    assert allocate_slots.call_args.kwargs["delay_cache_blocks"] is True
    assert len(captured_messages) == 1
    assert decode_request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    assert decode_request.num_computed_tokens == _PROMPT_TOKENS
    scheduler.update_from_output(admission_output, admission_harness._runner_output_for([]))

    pe_scheduler, pe_policy = _make_pe_scheduler()
    pe_request = _make_pe_request(captured_messages[0])
    assert pe_scheduler.get_num_new_matched_tokens(pe_request, 0) == (0, False)
    pe_result = pe_scheduler._pe_path_results[pe_request.request_id]
    full_blocks = forward_harness._blocks(_SOURCE_BLOCK_IDS)
    if defer_first_allocation:
        pe_scheduler.update_state_after_alloc(
            pe_request,
            forward_harness._blocks((_SOURCE_BLOCK_IDS[0][:-1],)),
            0,
        )
        assert pe_scheduler._pe_forward_plans == {}
        assert pe_scheduler._reqs_need_send_layerwise == {}
    pe_scheduler.update_state_after_alloc(pe_request, full_blocks, 0)
    forward_plan = pe_scheduler._pe_forward_plans[pe_request.request_id]
    pe_scheduler.update_state_after_alloc(pe_request, full_blocks, 0)
    assert pe_scheduler._pe_forward_plans == {pe_request.request_id: forward_plan}

    decode_scheduler._path_decision_coordinator.take_received_results.return_value = [pe_result]
    binding_output = scheduler.schedule()
    metadata = binding_output.kv_connector_metadata
    assert isinstance(metadata, DualPathConnectorMetadata)
    assert len(metadata.forward_receive_bindings) == 1
    binding = metadata.forward_receive_bindings[0]
    decode_scheduler._path_decision_coordinator.take_received_results.return_value = [pe_result]
    duplicate_metadata = decode_scheduler.build_connector_meta(MagicMock(name="duplicate_output"))
    assert duplicate_metadata.forward_receive_bindings == []

    pe_metadata = _build_parent_metadata(pe_scheduler, pe_request)
    worker = _make_worker()
    worker.start_load_kv(metadata)
    backend.reset_mock()
    return LifecycleHarness(
        scheduler=scheduler,
        decode_request=decode_request,
        decode_scheduler=decode_scheduler,
        pe_scheduler=pe_scheduler,
        pe_request=pe_request,
        pe_policy=pe_policy,
        pe_metadata=pe_metadata,
        binding_output=binding_output,
        binding=binding,
        worker=worker,
        forward_plan=forward_plan,
        proxy_http=proxy_http,
        backend=backend,
        baseline_free_blocks=baseline_free_blocks,
    )


def _worker_connector_output(harness: LifecycleHarness, *, failed: bool) -> KVConnectorOutput:
    wire_request_id = get_external_request_id(harness.decode_request.request_id)
    harness.worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
        set() if failed else {wire_request_id}
    )
    harness.worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
        {wire_request_id} if failed else set()
    )
    finished_sending, finished_recving = harness.worker.get_finished(set())
    return KVConnectorOutput(
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        invalid_block_ids=harness.worker.get_block_ids_with_load_errors(),
    )


def _assert_chain_invariants(harness: LifecycleHarness) -> None:
    plan = harness.forward_plan
    assert harness.decode_request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    assert plan.source_block_ids == tuple(tuple(group) for group in _SOURCE_BLOCK_IDS)
    assert plan.destination_block_ids == harness.binding.destination_block_ids
    assert (plan.token_start, plan.token_end) == (_DECODE_LOCAL_TOKENS, _PROMPT_TOKENS)
    assert harness.pe_metadata.local_block_ids == [list(group) for group in plan.source_block_ids]
    assert harness.pe_metadata.remote_cache_tokens == _DECODE_LOCAL_TOKENS
    assert harness.pe_policy.calls == 1
    assert harness.decode_scheduler._reqs_need_recv == {}
    assert harness.worker._recving_metadata == {}
    assert harness.worker._kvpool_worker_adapter.method_calls == []
    assert harness.worker.engine.method_calls == []
    assert harness.backend.method_calls == []
    harness.proxy_http.assert_not_called()


def _feed_worker_output(harness: LifecycleHarness, connector_output: KVConnectorOutput) -> None:
    runner_output = admission_harness._runner_output_for([])
    runner_output.kv_connector_output = connector_output
    harness.scheduler.update_from_output(harness.binding_output, runner_output)


@pytest.fixture(autouse=True)
def _constrain_external_seams():
    if "DualPathConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "DualPathConnector",
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector",
            "DualPathConnector",
        )
    with (
        patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib,
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
        ) as lookup_client_cls,
        patch.object(connector_module, "PathDecisionCoordinator") as coordinator_cls,
        patch.object(connector_module, "get_ip", return_value="127.0.0.1"),
    ):
        backend = MagicMock(name="kvpool_backend")
        mock_importlib.import_module.return_value = backend
        lookup_client_cls.return_value.lookup.return_value = 0
        decode_coordinator = MagicMock(name="decode_coordinator")
        decode_coordinator.decode_engine_instance_id = "integration-engine:0:test-boot"
        decode_coordinator.decode_control_endpoint = DecodeControlEndpoint(host="127.0.0.1", port=24001)
        coordinator_cls.for_decode.return_value = decode_coordinator
        yield backend


@pytest.fixture()
def scheduler() -> Scheduler:
    instance = admission_harness._make_scheduler(admission_harness._make_vllm_config())
    yield instance
    instance.shutdown()


@pytest.fixture()
def lifecycle_factory(scheduler: Scheduler, _constrain_external_seams, request: pytest.FixtureRequest):
    def make(*, deferred: bool = False) -> LifecycleHarness:
        harness = _build_lifecycle(
            scheduler,
            _constrain_external_seams,
            defer_first_allocation=deferred,
        )
        request.addfinalizer(harness.pe_scheduler.shutdown)
        return harness

    return make


def _assert_next_schedule_recomputes_last_token(harness: LifecycleHarness) -> None:
    allocation_state: list[tuple[RequestStatus, int]] = []
    real_allocate_slots = harness.scheduler.kv_cache_manager.allocate_slots

    def capture_allocation(request: Request, *args, **kwargs):
        allocation_state.append((request.status, request.num_computed_tokens))
        return real_allocate_slots(request, *args, **kwargs)

    with patch.object(harness.scheduler.kv_cache_manager, "allocate_slots", side_effect=capture_allocation):
        resumed_output = harness.scheduler.schedule()
    assert allocation_state == [(RequestStatus.WAITING, _PROMPT_TOKENS - 1)]
    assert resumed_output.num_scheduled_tokens[harness.decode_request.request_id] == 1
    assert harness.decode_request.status is RequestStatus.RUNNING


def test_pe_read_success_returns_request_to_waiting(lifecycle_factory) -> None:
    harness = lifecycle_factory()
    _assert_chain_invariants(harness)
    connector_output = _worker_connector_output(harness, failed=False)

    _feed_worker_output(harness, connector_output)

    assert connector_output.finished_recving == {harness.decode_request.request_id}
    assert connector_output.invalid_block_ids == set()
    assert harness.decode_request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    _assert_next_schedule_recomputes_last_token(harness)


def test_pe_read_failure_terminates_with_invalid_destination_suffix(lifecycle_factory) -> None:
    harness = lifecycle_factory()
    _assert_chain_invariants(harness)
    connector_output = _worker_connector_output(harness, failed=True)
    destination = harness.binding.destination_block_ids[0]
    expected_invalid = set(destination[1:3])

    _feed_worker_output(harness, connector_output)

    assert connector_output.finished_recving == {harness.decode_request.request_id}
    assert connector_output.invalid_block_ids == expected_invalid
    assert destination[0] not in connector_output.invalid_block_ids
    assert harness.decode_request.status is RequestStatus.FINISHED_ERROR
    assert harness.decode_request.request_id not in harness.scheduler.requests
    assert (
        harness.scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks == harness.baseline_free_blocks
    )


def test_pe_read_deferred_plan_completes_after_second_allocation(lifecycle_factory) -> None:
    harness = lifecycle_factory(deferred=True)
    _assert_chain_invariants(harness)
    connector_output = _worker_connector_output(harness, failed=False)

    _feed_worker_output(harness, connector_output)

    assert connector_output.finished_recving == {harness.decode_request.request_id}
    assert list(harness.pe_scheduler._pe_forward_plans) == [harness.pe_request.request_id]
    assert harness.pe_policy.calls == 1
    _assert_next_schedule_recomputes_last_token(harness)

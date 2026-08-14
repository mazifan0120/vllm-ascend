import inspect
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
    ReverseReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionRequest,
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    PathDecision,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    SendReqInfo,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
_SCHEDULER_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler"
_DECODE_INSTANCE_ID = "decode-engine:2:boot-7"
_BLOCK_SIZE = 16


class FixedPathPolicy:
    def __init__(self, path: PathKind) -> None:
        self.path = path
        self.calls = 0

    def choose(self, request: PathDecisionRequest) -> PathKind:
        self.calls += 1
        return self.path


def _make_vllm_config() -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.is_kv_consumer = False
    config.kv_transfer_config.is_kv_producer = True
    config.kv_transfer_config.engine_id = "prefill-engine"
    config.kv_transfer_config.kv_port = 5000
    config.kv_transfer_config.kv_load_failure_policy = "fail"
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {"tls_config": {}}.get(
        key, default
    )
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.block_size = _BLOCK_SIZE
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    return config


def _make_kv_cache_config() -> SimpleNamespace:
    spec = MagicMock()
    spec.block_size = _BLOCK_SIZE
    group = MagicMock()
    group.kv_cache_spec = spec
    group.layer_names = ["layer.0"]
    return SimpleNamespace(kv_cache_groups=[group], kv_cache_tensors=[], num_blocks=64)


def _completed_future() -> Future[None]:
    future: Future[None] = Future()
    future.set_result(None)
    return future


@pytest.fixture()
def scheduler_factory():
    schedulers = []
    with (
        patch(f"{_SCHEDULER_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_SCHEDULER_NS}.get_ip", return_value="192.0.2.44"),
    ):
        coordinator = MagicMock(name="prefill_coordinator")
        coordinator.submit.return_value = _completed_future()
        coordinator_cls.for_prefill.return_value = coordinator

        def make(path: PathKind = PathKind.PE_READ, *, need_truncate: bool = False):
            policy = FixedPathPolicy(path)
            scheduler = connector_module.DualPathConnectorScheduler(
                _make_vllm_config(),
                _make_kv_cache_config(),
                "prefill-engine",
                DualPathConfig(role="prefill"),
                path_policy=policy,
            )
            scheduler.executor.shutdown(wait=False)
            scheduler.metaserver_client.close()
            scheduler.executor = MagicMock(name="prefill_executor")
            scheduler.need_truncate = need_truncate
            schedulers.append(scheduler)
            return scheduler, policy, coordinator

        yield make

    for scheduler in schedulers:
        scheduler.shutdown()


def _decision_payload(*, target_tokens: int, local_tokens: int, store_tokens: int) -> dict:
    return {
        "decision_request": {
            "request_key": {
                "decode_engine_instance_id": _DECODE_INSTANCE_ID,
                "decode_request_id": "decode-request-7",
            },
            "target_tokens": target_tokens,
            "decode_local_tokens": local_tokens,
            "decode_store_tokens": store_tokens,
        },
        "decode_control_endpoint": {"host": "192.0.2.44", "port": 24001},
    }


def _make_request(
    *,
    request_id: str = "prefill-request-7",
    target_tokens: int = 32,
    prompt_tokens: int = 33,
    local_tokens: int = 16,
    store_tokens: int = 16,
    destination_block_ids: list[list[int]] | None = None,
) -> SimpleNamespace:
    prompt_token_ids = list(range(prompt_tokens))
    params = {
        "do_remote_decode": True,
        "remote_block_ids": destination_block_ids or [[20, 21, 22]],
        "remote_block_size": [_BLOCK_SIZE],
        "remote_cached_tokens": local_tokens,
        "remote_engine_id": "decode-engine",
        "remote_host": "198.51.100.20",
        "remote_port": 6000,
        "remote_tp_size": 1,
        "remote_pcp_size": 1,
        "remote_dcp_size": 1,
        "remote_pp_size": 1,
        "remote_dp_size": 1,
        "dual_path": _decision_payload(
            target_tokens=target_tokens,
            local_tokens=local_tokens,
            store_tokens=store_tokens,
        ),
    }
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=prompt_tokens,
        num_prompt_tokens=prompt_tokens,
        num_computed_tokens=0,
        num_preemptions=0,
        max_tokens=16,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=None,
        all_token_ids=list(prompt_token_ids),
        _all_token_ids=list(prompt_token_ids),
        kv_transfer_params=params,
    )


def _blocks(block_ids: tuple[list[int], ...]) -> MagicMock:
    blocks = MagicMock(name="blocks")
    blocks.get_block_ids.return_value = block_ids
    blocks.new_empty.return_value = MagicMock(name="empty_blocks")
    return blocks


def _decide(scheduler, request) -> None:
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)


@pytest.fixture()
def parent_forward_metadata_pair(scheduler_factory):
    dual_scheduler, _, _ = scheduler_factory()
    request = _make_request()
    _decide(dual_scheduler, request)
    dual_scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    parent_scheduler = MooncakeLayerwiseConnectorScheduler(
        _make_vllm_config(),
        _make_kv_cache_config(),
        "prefill-engine",
    )
    try:
        installed_send_info = dual_scheduler._reqs_need_send_layerwise[request.request_id]
        parent_scheduler._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
            local_block_ids=[list(group) for group in installed_send_info.local_block_ids],
            local_transferred_tokens=installed_send_info.local_transferred_tokens,
            local_computed_tokens=installed_send_info.local_computed_tokens,
            request=request,
        )
        scheduler_output = SimpleNamespace(
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[request.request_id],
                new_block_ids=[[[13]]],
                num_computed_tokens=[16],
            ),
            scheduled_spec_decode_tokens={},
            scheduled_new_reqs=[],
            num_scheduled_tokens={request.request_id: 17},
        )

        dual_metadata = dual_scheduler.build_connector_meta(scheduler_output)
        parent_metadata = parent_scheduler.build_connector_meta(scheduler_output)
        return dual_metadata.requests[request.request_id], parent_metadata.requests[request.request_id]
    finally:
        parent_scheduler.executor.shutdown(wait=False)
        parent_scheduler.metaserver_client.close()


@pytest.mark.parametrize(
    ("local_tokens", "path", "expected_policy_calls"),
    [
        pytest.param(16, PathKind.DE_READ, 0, id="forced"),
        pytest.param(0, PathKind.PE_READ, 1, id="policy"),
    ],
)
def test_pe_read_returns_zero_false_for_forced_and_policy_paths(
    scheduler_factory,
    local_tokens,
    path,
    expected_policy_calls,
):
    scheduler, policy, coordinator = scheduler_factory(path)
    request = _make_request()

    result = scheduler.get_num_new_matched_tokens(request, local_tokens)

    assert result == (0, False)
    assert scheduler._prefill_path_results[request.request_id].path is PathKind.PE_READ
    assert policy.calls == expected_policy_calls
    coordinator.submit.assert_not_called()


def test_de_read_returns_exact_reverse_budget_and_wins_first_positive(scheduler_factory):
    scheduler, policy, _ = scheduler_factory(PathKind.DE_READ)
    dual_path = connector_module.DualPathConnector.__new__(connector_module.DualPathConnector)
    dual_path.connector_scheduler = scheduler
    store = MagicMock(name="ascend_store")
    store.has_preempted_request = None
    store.get_num_new_matched_tokens.return_value = (25, True)
    multi = AscendMultiConnector.__new__(AscendMultiConnector)
    multi._connectors = [dual_path, store]
    multi._requests_to_connector = {}
    request = _make_request(store_tokens=24)
    request.num_computed_tokens = 8

    result = multi.get_num_new_matched_tokens(request, 8)

    assert result == (16, True)
    assert policy.calls == 1
    assert multi._requests_to_connector[request.request_id] == 0
    store.get_num_new_matched_tokens.assert_called_once_with(request, 8)


def test_no_sender_future_after_lookup_alone(scheduler_factory):
    scheduler, _, coordinator = scheduler_factory()
    request = _make_request()

    _decide(scheduler, request)

    assert scheduler._prefill_delivery_futures == {}
    coordinator.submit.assert_not_called()


def test_successful_allocation_installs_before_creating_one_future(scheduler_factory):
    scheduler, policy, coordinator = scheduler_factory()
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))

    def submit_after_install(_endpoint, _decision):
        assert request.request_id in scheduler._prefill_forward_plans
        assert request.request_id in scheduler._reqs_need_send_layerwise
        return _completed_future()

    coordinator.submit.side_effect = submit_after_install

    _decide(scheduler, request)
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)
    scheduler.update_state_after_alloc(request, blocks, 0)

    assert list(scheduler._prefill_path_results) == [request.request_id]
    assert list(scheduler._prefill_forward_plans) == [request.request_id]
    assert policy.calls == 1
    coordinator.submit.assert_called_once()
    assert list(scheduler._prefill_delivery_futures) == [request.request_id]


def test_pe_read_freezes_forward_range_with_final_tables(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    destination = [[20, 21, 22]]
    source = [10, 11, 12]
    request = _make_request(destination_block_ids=destination)
    blocks = _blocks((source,))

    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)
    source.append(13)
    destination[0].append(23)

    plan = scheduler._prefill_forward_plans[request.request_id]
    assert (plan.token_start, plan.token_end) == (16, 33)
    assert plan.source_block_ids == ((10, 11, 12),)
    assert plan.destination_block_ids == ((20, 21, 22),)


def test_hybrid_transfer_target_applied_exactly_once(scheduler_factory):
    scheduler, policy, _ = scheduler_factory(need_truncate=True)
    request = _make_request(target_tokens=32, prompt_tokens=33)

    _decide(scheduler, request)
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11],)), 0)

    plan = scheduler._prefill_forward_plans[request.request_id]
    assert request.num_prompt_tokens == 32
    assert len(request.prompt_token_ids) == 32
    assert request.kv_transfer_params["_p_side_truncated"] is True
    assert plan.token_end == 32
    assert policy.calls == 1


def test_installation_creates_send_req_info_with_L_DE_transferred(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()

    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    plan = scheduler._prefill_forward_plans[request.request_id]
    assert scheduler._reqs_need_send_layerwise[request.request_id] == SendReqInfo(
        local_block_ids=[list(group) for group in plan.source_block_ids],
        local_transferred_tokens=plan.token_start,
        local_computed_tokens=0,
        request=request,
    )


def test_identical_duplicate_alloc_is_idempotent(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)
    first_plan = scheduler._prefill_forward_plans[request.request_id]
    first_send = scheduler._reqs_need_send_layerwise[request.request_id]

    scheduler.update_state_after_alloc(request, blocks, 0)

    assert scheduler._prefill_forward_plans[request.request_id] is first_plan
    assert scheduler._reqs_need_send_layerwise[request.request_id] is first_send


def test_conflicting_duplicate_alloc_fails_locally_preserving_first_plan(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)
    first_plan = scheduler._prefill_forward_plans[request.request_id]

    scheduler.update_state_after_alloc(request, _blocks(([13, 14, 15],)), 0)

    assert scheduler._prefill_forward_plans[request.request_id] is first_plan
    assert request.request_id in scheduler._prefill_invalid_request_ids
    assert request.request_id not in scheduler._prefill_path_results


def test_conflicting_forward_plan_raises_within_epoch_and_replaces_after_preemption(pe_scheduler_factory):
    # Two disagreeing allocations within one scheduling epoch are a protocol
    # violation; after a preemption the same reallocation is the legal replay.
    from tests.ut.distributed.kv_transfer.dual_path.conftest import make_block_pool

    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
    request = _make_request()
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)
    first_plan = scheduler._prefill_forward_plans[request.request_id]

    # Same epoch: the conflicting reallocation raises and preserves the plan.
    scheduler.update_state_after_alloc(request, _blocks(([13, 14, 15],)), 0)
    assert scheduler._prefill_forward_plans[request.request_id] is first_plan
    assert request.request_id in scheduler._prefill_invalid_request_ids

    # After a preemption a fresh request replaces its plan legally.
    resumed_request = _make_request(request_id="prefill-request-resumed")
    _decide(scheduler, resumed_request)
    scheduler.update_state_after_alloc(resumed_request, _blocks(([10, 11, 12],)), 0)
    first_resumed_plan = scheduler._prefill_forward_plans[resumed_request.request_id]

    resumed_request.num_preemptions += 1
    scheduler.update_state_after_alloc(resumed_request, _blocks(([13, 14, 15],)), 0)
    replaced_plan = scheduler._prefill_forward_plans[resumed_request.request_id]
    assert replaced_plan is not first_resumed_plan
    assert replaced_plan.source_block_ids == ((13, 14, 15),)


def test_post_install_validation_failure_preserves_first_plan_and_send_state(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)
    first_plan = scheduler._prefill_forward_plans[request.request_id]
    first_send = scheduler._reqs_need_send_layerwise[request.request_id]
    request.kv_transfer_params.pop("remote_host")

    scheduler.update_state_after_alloc(request, blocks, 0)

    assert scheduler._prefill_forward_plans[request.request_id] is first_plan
    assert scheduler._reqs_need_send_layerwise[request.request_id] is first_send
    assert request.request_id in scheduler._prefill_invalid_request_ids
    assert request.request_id not in scheduler._prefill_path_results


def test_pe_ascendstore_may_win_after_pe_read_forward_still_installs(scheduler_factory):
    scheduler, policy, _ = scheduler_factory()
    dual_path = connector_module.DualPathConnector.__new__(connector_module.DualPathConnector)
    dual_path.connector_scheduler = scheduler
    store = MagicMock(name="ascend_store")
    store.has_preempted_request = None
    store.get_num_new_matched_tokens.return_value = (17, True)
    multi = AscendMultiConnector.__new__(AscendMultiConnector)
    multi._connectors = [dual_path, store]
    multi._requests_to_connector = {}
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))

    assert multi.get_num_new_matched_tokens(request, 0) == (17, True)
    multi.update_state_after_alloc(request, blocks, 17)

    assert policy.calls == 1
    assert multi._requests_to_connector[request.request_id] == 1
    assert scheduler._prefill_forward_plans[request.request_id].source_block_ids == ((10, 11, 12),)
    store.update_state_after_alloc.assert_called_once_with(request, blocks, 17)


def test_all_zero_sibling_still_installs_forward(scheduler_factory):
    scheduler, policy, _ = scheduler_factory()
    dual_path = connector_module.DualPathConnector.__new__(connector_module.DualPathConnector)
    dual_path.connector_scheduler = scheduler
    store = MagicMock(name="ascend_store")
    store.has_preempted_request = None
    store.get_num_new_matched_tokens.return_value = (0, False)
    multi = AscendMultiConnector.__new__(AscendMultiConnector)
    multi._connectors = [dual_path, store]
    multi._requests_to_connector = {}
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))

    assert multi.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.request_id not in multi._requests_to_connector
    multi.update_state_after_alloc(request, blocks, 0)

    assert policy.calls == 1
    assert scheduler._prefill_forward_plans[request.request_id].source_block_ids == ((10, 11, 12),)
    store.update_state_after_alloc.assert_called_once()


def test_no_multiconnector_change_guard():
    source = inspect.getsource(AscendMultiConnector)

    assert "dual_path" not in source.lower()


@pytest.mark.parametrize(
    "pe_state,num_external_tokens", [("hbm", 0), ("store_full", 16), ("store_partial", 8), ("store_miss", 33)]
)
def test_pe_hbm_store_full_store_partial_store_miss_converge_through_same_send_seam(
    scheduler_factory, pe_state, num_external_tokens
):
    scheduler, _, _ = scheduler_factory()
    request = _make_request(request_id=f"prefill-{pe_state}")

    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), num_external_tokens)

    send_info = scheduler._reqs_need_send_layerwise[request.request_id]
    assert (send_info.local_block_ids, send_info.local_transferred_tokens) == ([[10, 11, 12]], 16)


def test_scheduler_defers_plan_while_source_table_short_of_T(scheduler_factory):
    scheduler, policy, _ = scheduler_factory()
    request = _make_request(target_tokens=16, prompt_tokens=17, local_tokens=0, store_tokens=0)
    _decide(scheduler, request)

    scheduler.update_state_after_alloc(request, _blocks(([10],)), 0)

    assert request.request_id in scheduler._prefill_path_results
    assert request.request_id not in scheduler._prefill_forward_plans
    assert request.request_id not in scheduler._reqs_need_send_layerwise
    assert request.request_id not in scheduler._prefill_invalid_request_ids

    scheduler.update_state_after_alloc(request, _blocks(([10, 11],)), 0)

    assert list(scheduler._prefill_forward_plans) == [request.request_id]
    assert policy.calls == 0


def test_allocation_retry_before_bind_sends_nothing_and_does_not_redecide(scheduler_factory):
    scheduler, policy, coordinator = scheduler_factory()
    request = _make_request()

    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11],)), 0)
    _decide(scheduler, request)

    assert policy.calls == 1
    assert scheduler._prefill_forward_plans == {}
    assert scheduler._prefill_delivery_futures == {}
    coordinator.submit.assert_not_called()


def test_lookup_or_install_alone_invokes_no_worker_p2p(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()

    with patch.object(MooncakeLayerwiseConnectorWorker, "save_kv_layer", autospec=True) as p2p_spy:
        _decide(scheduler, request)
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    p2p_spy.assert_not_called()


def test_partial_de_read_freezes_exact_ranges_with_distinct_tables(scheduler_factory):
    scheduler, _, _ = scheduler_factory(PathKind.DE_READ)
    request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
    )
    blocks = _blocks(([70, 71, 72, 73],))

    assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
    scheduler.update_state_after_alloc(request, blocks, 0)

    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    forward = scheduler._prefill_forward_plans[request.request_id]
    decision = scheduler._path_decision_coordinator.submit.call_args.args[1]
    reverse = decision.reverse_plan
    assert reverse is not None
    assert (binding.token_start, binding.token_end) == (16, 32)
    assert (reverse.token_start, reverse.token_end) == (16, 32)
    assert (forward.token_start, forward.token_end) == (32, 49)
    assert binding.destination_block_ids == reverse.destination_block_ids == ((70, 71, 72, 73),)
    assert reverse.source_block_ids == forward.destination_block_ids == ((20, 21, 22, 23),)
    assert forward.source_block_ids == ((70, 71, 72, 73),)
    assert (reverse.remote_engine_id, reverse.remote_host, reverse.remote_port) == (
        scheduler.engine_id,
        scheduler.side_channel_host,
        scheduler.side_channel_port,
    )
    assert (reverse.remote_tp_size, reverse.remote_pcp_size, reverse.remote_dcp_size) == (1, 1, 1)


def test_de_read_installs_binding_plan_and_forward_before_submit(scheduler_factory):
    scheduler, policy, coordinator = scheduler_factory(PathKind.DE_READ)
    request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
    )
    blocks = _blocks(([70, 71, 72, 73],))
    events = []

    class RecordingDict(dict):
        def __init__(self, event_name):
            super().__init__()
            self.event_name = event_name

        def __setitem__(self, key, value):
            events.append(self.event_name)
            return super().__setitem__(key, value)

    scheduler._prefill_pending_reverse_receive_bindings = RecordingDict("binding")
    scheduler._prefill_forward_plans = RecordingDict("forward")

    def submit_after_install(_endpoint, decision):
        events.append("submit")
        assert isinstance(decision, PathDecision)
        assert decision.reverse_plan is not None
        assert request.request_id in scheduler._prefill_pending_reverse_receive_bindings
        assert request.request_id in scheduler._prefill_forward_plans
        return _completed_future()

    coordinator.submit.side_effect = submit_after_install

    assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
    scheduler.update_state_after_alloc(request, blocks, 0)
    scheduler.update_state_after_alloc(request, blocks, 0)

    assert events[:3] == ["binding", "forward", "submit"]
    assert policy.calls == 1
    coordinator.submit.assert_called_once()


def test_de_read_delivers_with_admission_table_and_defers_forward_plan(scheduler_factory):
    """Real vLLM admission allocates only the external-token blocks (the
    Reverse destination) and parks the request in WAITING_FOR_REMOTE_KVS, so
    the T-covering table only appears on the post-Reverse allocation. The
    Decision must be delivered from the admission table; the Forward plan
    installs on the later allocation without a second delivery."""
    scheduler, _, coordinator = scheduler_factory(PathKind.DE_READ)
    request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
    )
    assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)

    # Admission allocation covers only the Reverse destination [L_PE, K_DE).
    scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)

    assert coordinator.submit.call_count == 1
    decision = coordinator.submit.call_args.args[1]
    assert decision.reverse_plan is not None
    assert (decision.reverse_plan.token_start, decision.reverse_plan.token_end) == (16, 32)
    assert decision.reverse_plan.destination_block_ids == ((70, 71),)
    assert request.request_id in scheduler._prefill_pending_reverse_receive_bindings
    assert request.request_id not in scheduler._prefill_forward_plans

    # Post-Reverse allocation covers T: Forward plan installs, no re-delivery.
    scheduler.update_state_after_alloc(request, _blocks(([70, 71, 72, 73],)), 0)

    assert coordinator.submit.call_count == 1
    plan = scheduler._prefill_forward_plans[request.request_id]
    assert (plan.token_start, plan.token_end) == (32, 49)
    assert scheduler._reqs_need_send_layerwise[request.request_id].local_block_ids == [[70, 71, 72, 73]]


def test_miss_de_read_freezes_reverse_hbm_range_and_no_store(scheduler_factory):
    scheduler, _, _ = scheduler_factory(PathKind.DE_READ)
    request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=32,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
    )

    assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
    scheduler.update_state_after_alloc(request, _blocks(([70, 71, 72, 73],)), 0)

    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    forward = scheduler._prefill_forward_plans[request.request_id]
    decision = scheduler._path_decision_coordinator.submit.call_args.args[1]
    assert decision.reverse_plan is not None
    assert (binding.token_start, binding.token_end) == (16, 32)
    assert (decision.reverse_plan.token_start, decision.reverse_plan.token_end) == (16, 32)
    assert (forward.token_start, forward.token_end) == (32, 49)
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
    assert isinstance(metadata, DualPathConnectorMetadata)
    assert metadata.decode_store_metadata is None


@pytest.mark.parametrize(
    "mismatch",
    ["group-count", "alignment", "coverage", "endpoint", "topology", "wire-id"],
)
def test_activation_fact_mismatch_fails_with_local_control_failure(scheduler_factory, mismatch):
    scheduler, _, coordinator = scheduler_factory(PathKind.DE_READ)
    request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=16,
        store_tokens=32,
        destination_block_ids=[[20, 21, 22, 23]],
    )
    blocks = _blocks(([70, 71, 72, 73],))
    prefill_local_tokens = 8 if mismatch == "alignment" else 16

    assert scheduler.get_num_new_matched_tokens(request, prefill_local_tokens) == (
        32 - prefill_local_tokens,
        True,
    )
    if mismatch == "group-count":
        request.kv_transfer_params["remote_block_size"] = [16, 16]
    elif mismatch == "coverage":
        request.kv_transfer_params["remote_block_ids"] = [[20]]
    elif mismatch == "endpoint":
        request.kv_transfer_params["remote_host"] = ""
    elif mismatch == "topology":
        request.kv_transfer_params["remote_tp_size"] = 0
    wire_context = (
        patch.object(scheduler_module, "reverse_wire_id", return_value="") if mismatch == "wire-id" else nullcontext()
    )

    with wire_context:
        scheduler.update_state_after_alloc(request, blocks, 0)

    coordinator.submit.assert_not_called()
    assert scheduler._prefill_forward_plans == {}
    assert scheduler._prefill_pending_reverse_receive_bindings == {}
    expected_failure = DualPathControlFailureMetadata(
        request_id=request.request_id,
        invalid_block_ids=(70, 71) if mismatch == "alignment" else (71,),
        reason=DualPathControlFailureReason.ACTIVATION_FAILED,
    )
    assert scheduler._prefill_control_failures == {request.request_id: expected_failure}

    first = scheduler.build_connector_meta(MagicMock(name="first_scheduler_output"))
    second = scheduler.build_connector_meta(MagicMock(name="second_scheduler_output"))
    assert first.control_failures == [expected_failure]
    assert isinstance(second, DualPathConnectorMetadata)
    assert second.control_failures == []


def test_pe_read_local_plan_failure_sends_no_decision(scheduler_factory):
    scheduler, _, coordinator = scheduler_factory(PathKind.PE_READ)
    request = _make_request()
    _decide(scheduler, request)

    with patch.object(scheduler, "_try_install_forward_plan", side_effect=RuntimeError("local plan failed")):
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    coordinator.submit.assert_not_called()
    assert scheduler._prefill_delivery_futures == {}


def test_pe_metadata_emits_binding_and_control_failure_once(scheduler_factory):
    scheduler, _, _ = scheduler_factory(PathKind.DE_READ)
    request_key = DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7")
    binding = ReverseReceiveBinding(
        request_key=request_key,
        wire_request_id="wire-request-7",
        prefill_request_id="prefill-request-7",
        destination_block_ids=((70, 71, 72, 73),),
        token_start=16,
        token_end=32,
        reverse_attempt_id=0,
        prefill_local_tokens=16,
        reverse_completion_job_id=0,
    )
    failure = DualPathControlFailureMetadata(
        request_id="prefill-failed",
        invalid_block_ids=(81,),
        reason=DualPathControlFailureReason.ACTIVATION_FAILED,
    )
    scheduler._prefill_pending_reverse_receive_bindings[binding.prefill_request_id] = binding
    scheduler._prefill_control_failures[failure.request_id] = failure
    parent_metadata = MooncakeLayerwiseConnectorMetadata()

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        "build_connector_meta",
        autospec=True,
        return_value=parent_metadata,
    ):
        first = scheduler.build_connector_meta(MagicMock(name="first_scheduler_output"))
        second = scheduler.build_connector_meta(MagicMock(name="second_scheduler_output"))

    assert isinstance(first, DualPathConnectorMetadata)
    assert first.reverse_receive_bindings == [binding]
    assert first.control_failures == [failure]
    assert scheduler._prefill_pending_reverse_receive_bindings == {}
    assert scheduler._prefill_control_failures == {}
    assert isinstance(second, DualPathConnectorMetadata)
    assert second.requests is parent_metadata.requests
    assert second.reverse_receive_bindings == []
    assert second.control_failures == []


@pytest.mark.parametrize(
    ("method_name", "block_ids"),
    [
        ("request_finished", [10, 11, 12]),
        ("request_finished_all_groups", ([10, 11, 12],)),
    ],
)
def test_pe_finish_and_shutdown_remove_task05_records_idempotently(
    scheduler_factory,
    method_name,
    block_ids,
):
    parent_result = (True, {"owner": "parent"})
    scheduler, _, _ = scheduler_factory()
    request = _make_request(request_id="prefill-finish-unconsumed")
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        method_name,
        autospec=True,
        return_value=parent_result,
    ) as parent_finish:
        assert getattr(scheduler, method_name)(request, block_ids) == parent_result
        assert scheduler._prefill_path_results == {}
        assert scheduler._prefill_forward_plans == {}
        assert request.request_id not in scheduler._reqs_need_send_layerwise

        replacement_request = _make_request(request_id=request.request_id)
        replacement_send_info = SendReqInfo(
            local_block_ids=[[90, 91, 92]],
            local_transferred_tokens=16,
            local_computed_tokens=0,
            request=replacement_request,
        )
        scheduler._reqs_need_send_layerwise[request.request_id] = replacement_send_info
        assert getattr(scheduler, method_name)(request, block_ids) == parent_result
        assert scheduler._reqs_need_send_layerwise[request.request_id] is replacement_send_info

        consumed_scheduler, _, _ = scheduler_factory()
        consumed_request = _make_request(request_id="prefill-finish-consumed")
        _decide(consumed_scheduler, consumed_request)
        consumed_scheduler.update_state_after_alloc(consumed_request, _blocks(([10, 11, 12],)), 0)
        consumed_scheduler.build_connector_meta(
            SimpleNamespace(
                scheduled_cached_reqs=SimpleNamespace(
                    req_ids=[consumed_request.request_id],
                    new_block_ids=[None],
                    num_computed_tokens=[0],
                ),
                scheduled_spec_decode_tokens={},
                scheduled_new_reqs=[],
                num_scheduled_tokens={consumed_request.request_id: consumed_request.num_prompt_tokens},
            )
        )
        assert consumed_request.request_id not in consumed_scheduler._reqs_need_send_layerwise
        assert getattr(consumed_scheduler, method_name)(consumed_request, block_ids) == parent_result
        assert consumed_scheduler._prefill_path_results == {}
        assert consumed_scheduler._prefill_forward_plans == {}

    assert parent_finish.call_count == 3

    shutdown_scheduler, _, _ = scheduler_factory()
    shutdown_request = _make_request(request_id="prefill-shutdown-records")
    _decide(shutdown_scheduler, shutdown_request)
    shutdown_scheduler.update_state_after_alloc(shutdown_request, _blocks(([10, 11, 12],)), 0)
    owned_send_info = shutdown_scheduler._reqs_need_send_layerwise[shutdown_request.request_id]
    owned_send_info.local_transferred_tokens = 0
    owned_send_info.local_block_ids[0].append(13)
    ordinary_request = _make_request(request_id="ordinary-parent-send-state")
    ordinary_send_info = SendReqInfo(
        local_block_ids=[[70, 71, 72]],
        local_transferred_tokens=0,
        local_computed_tokens=0,
        request=ordinary_request,
    )
    shutdown_scheduler._reqs_need_send_layerwise[ordinary_request.request_id] = ordinary_send_info

    shutdown_scheduler.shutdown()
    shutdown_scheduler.shutdown()

    assert shutdown_scheduler._prefill_path_results == {}
    assert shutdown_scheduler._prefill_forward_plans == {}
    assert shutdown_request.request_id not in shutdown_scheduler._reqs_need_send_layerwise
    assert shutdown_scheduler._reqs_need_send_layerwise[ordinary_request.request_id] is ordinary_send_info


@pytest.mark.parametrize(
    "invalid_case",
    [
        "misaligned_token_start",
        "group_count_mismatch",
        "insufficient_destination_coverage",
        "missing_remote_block_size",
        "missing_remote_engine_id",
        "missing_remote_host",
        "missing_remote_port",
    ],
)
def test_scheduler_rejects_invalid_forward_plan_without_send_state(scheduler_factory, invalid_case):
    scheduler, _, _ = scheduler_factory()
    request = _make_request(local_tokens=8 if invalid_case == "misaligned_token_start" else 16)
    blocks = _blocks(([10, 11, 12],))
    if invalid_case == "group_count_mismatch":
        blocks.get_block_ids.return_value = ([10, 11, 12], [30, 31, 32])
    elif invalid_case == "insufficient_destination_coverage":
        request.kv_transfer_params["remote_block_ids"] = [[20, 21]]
    elif invalid_case.startswith("missing_"):
        request.kv_transfer_params.pop(invalid_case.removeprefix("missing_"))
    _decide(scheduler, request)

    scheduler.update_state_after_alloc(request, blocks, 0)

    assert request.request_id in scheduler._prefill_invalid_request_ids
    assert request.request_id not in scheduler._prefill_path_results
    assert request.request_id not in scheduler._prefill_forward_plans
    assert request.request_id not in scheduler._reqs_need_send_layerwise


@pytest.mark.parametrize("topology_field", ["remote_tp_size", "remote_pcp_size", "remote_dcp_size"])
def test_scheduler_rejects_missing_forward_topology_without_send_state(scheduler_factory, topology_field):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    request.kv_transfer_params.pop(topology_field)
    _decide(scheduler, request)

    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    assert request.request_id in scheduler._prefill_invalid_request_ids
    assert request.request_id not in scheduler._prefill_path_results
    assert request.request_id not in scheduler._prefill_forward_plans
    assert request.request_id not in scheduler._reqs_need_send_layerwise


@pytest.mark.parametrize("topology_field", ["remote_tp_size", "remote_pcp_size", "remote_dcp_size"])
@pytest.mark.parametrize("invalid_value", [None, 0, -1, True, "1"])
def test_scheduler_rejects_invalid_forward_topology_without_send_state(
    scheduler_factory,
    topology_field,
    invalid_value,
):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    request.kv_transfer_params[topology_field] = invalid_value
    _decide(scheduler, request)

    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    assert request.request_id in scheduler._prefill_invalid_request_ids
    assert request.request_id not in scheduler._prefill_path_results
    assert request.request_id not in scheduler._prefill_forward_plans
    assert request.request_id not in scheduler._reqs_need_send_layerwise


def test_build_connector_meta_frontier_matches_parent(parent_forward_metadata_pair):
    dual_req_meta, parent_req_meta = parent_forward_metadata_pair

    assert (
        (
            dual_req_meta.local_transed_tokens,
            dual_req_meta.local_computed_tokens,
            dual_req_meta.chunk_finish,
        )
        == (
            parent_req_meta.local_transed_tokens,
            parent_req_meta.local_computed_tokens,
            parent_req_meta.chunk_finish,
        )
        == (0, 33, True)
    )


def test_remote_cache_tokens_skips_decode_ready_prefix_exactly_once(parent_forward_metadata_pair):
    dual_req_meta, parent_req_meta = parent_forward_metadata_pair

    assert dual_req_meta.remote_cache_tokens == parent_req_meta.remote_cache_tokens == 16
    assert max(dual_req_meta.remote_cache_tokens, dual_req_meta.local_transed_tokens) == 16


def test_non_block_aligned_T_selects_containing_final_block(parent_forward_metadata_pair):
    dual_req_meta, parent_req_meta = parent_forward_metadata_pair
    token_end = 33
    containing_block_index = token_end // _BLOCK_SIZE

    assert token_end % _BLOCK_SIZE != 0
    assert (
        dual_req_meta.local_block_ids[0][containing_block_index],
        parent_req_meta.local_block_ids[0][containing_block_index],
        dual_req_meta.prompt_len,
        parent_req_meta.prompt_len,
        dual_req_meta.local_computed_tokens,
    ) == (12, 12, token_end, token_end, token_end)


def test_task05_methods_have_no_synchronous_waits():
    task05_methods = (
        connector_module.DualPathConnectorScheduler._try_install_forward_plan,
        connector_module.DualPathConnectorScheduler.build_connector_meta,
        connector_module.DualPathConnectorWorker._install_forward_receive_binding,
        connector_module.DualPathConnectorWorker.start_load_kv,
        connector_module.DualPathConnectorWorker.get_finished,
    )
    blocking_calls = (".result(", ".wait(", "time.sleep(", ".recv(")

    for method in task05_methods:
        source = inspect.getsource(method)
        for blocking_call in blocking_calls:
            assert blocking_call not in source, f"{method.__qualname__} contains {blocking_call}"

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    Path,
    PathDecisionRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    SendReqInfo,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
_DECODE_INSTANCE_ID = "decode-engine:2:boot-7"
_BLOCK_SIZE = 16


class FixedPathPolicy:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.calls = 0

    def choose(self, request: PathDecisionRequest) -> Path:
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
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
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
        patch(f"{_CONNECTOR_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_CONNECTOR_NS}.get_ip", return_value="192.0.2.44"),
    ):
        coordinator = MagicMock(name="prefill_coordinator")
        coordinator.submit.return_value = _completed_future()
        coordinator_cls.for_prefill.return_value = coordinator

        def make(path: Path = Path.PE_READ, *, need_truncate: bool = False):
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
        "protocol_version": 1,
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


def test_pe_read_result_retained_once_and_single_plan_after_alloc_covers_T(scheduler_factory):
    scheduler, policy, coordinator = scheduler_factory()
    request = _make_request()

    _decide(scheduler, request)
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    assert list(scheduler._pe_path_results) == [request.request_id]
    assert list(scheduler._pe_forward_plans) == [request.request_id]
    assert policy.calls == 1
    coordinator.submit.assert_called_once()


def test_plan_freezes_complete_tables_and_logical_range_ordinary_T_equals_P(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    destination = [[20, 21, 22]]
    source = [10, 11, 12]
    request = _make_request(destination_block_ids=destination)
    blocks = _blocks((source,))

    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)
    source.append(13)
    destination[0].append(23)

    plan = scheduler._pe_forward_plans[request.request_id]
    assert (plan.token_start, plan.token_end) == (16, 33)
    assert plan.source_block_ids == ((10, 11, 12),)
    assert plan.destination_block_ids == ((20, 21, 22),)


def test_plan_hybrid_truncates_once_T_equals_R_and_replay_never_double_truncates(scheduler_factory):
    scheduler, policy, _ = scheduler_factory(need_truncate=True)
    request = _make_request(target_tokens=32, prompt_tokens=33)

    _decide(scheduler, request)
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11],)), 0)

    plan = scheduler._pe_forward_plans[request.request_id]
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

    plan = scheduler._pe_forward_plans[request.request_id]
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
    first_plan = scheduler._pe_forward_plans[request.request_id]
    first_send = scheduler._reqs_need_send_layerwise[request.request_id]

    scheduler.update_state_after_alloc(request, blocks, 0)

    assert scheduler._pe_forward_plans[request.request_id] is first_plan
    assert scheduler._reqs_need_send_layerwise[request.request_id] is first_send


def test_conflicting_duplicate_alloc_raises_preserving_first_plan(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)
    first_plan = scheduler._pe_forward_plans[request.request_id]

    with pytest.raises(RuntimeError, match="conflicting duplicate Forward plan"):
        scheduler.update_state_after_alloc(request, _blocks(([13, 14, 15],)), 0)

    assert scheduler._pe_forward_plans[request.request_id] is first_plan


def test_post_install_validation_failure_preserves_first_plan_and_send_state(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))
    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)
    first_plan = scheduler._pe_forward_plans[request.request_id]
    first_send = scheduler._reqs_need_send_layerwise[request.request_id]
    request.kv_transfer_params.pop("remote_host")

    scheduler.update_state_after_alloc(request, blocks, 0)

    assert scheduler._pe_forward_plans[request.request_id] is first_plan
    assert scheduler._reqs_need_send_layerwise[request.request_id] is first_send
    assert request.request_id in scheduler._pe_invalid_request_ids
    assert request.request_id not in scheduler._pe_path_results


def test_multi_connector_order_dual_path_decides_first_and_still_receives_real_blocks(scheduler_factory):
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
    assert scheduler._pe_forward_plans[request.request_id].source_block_ids == ((10, 11, 12),)
    store.update_state_after_alloc.assert_called_once_with(request, blocks, 17)


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

    assert request.request_id in scheduler._pe_path_results
    assert request.request_id not in scheduler._pe_forward_plans
    assert request.request_id not in scheduler._reqs_need_send_layerwise
    assert request.request_id not in scheduler._pe_invalid_request_ids

    scheduler.update_state_after_alloc(request, _blocks(([10, 11],)), 0)

    assert list(scheduler._pe_forward_plans) == [request.request_id]
    assert policy.calls == 1


def test_lookup_or_install_alone_invokes_no_worker_p2p(scheduler_factory):
    scheduler, _, _ = scheduler_factory()
    request = _make_request()

    with patch.object(MooncakeLayerwiseConnectorWorker, "save_kv_layer", autospec=True) as p2p_spy:
        _decide(scheduler, request)
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    p2p_spy.assert_not_called()


def test_de_read_result_creates_no_plan_no_send_queue_no_error(scheduler_factory):
    scheduler, _, _ = scheduler_factory(Path.DE_READ)
    request = _make_request()
    blocks = _blocks(([10, 11, 12],))

    _decide(scheduler, request)
    scheduler.update_state_after_alloc(request, blocks, 0)

    assert scheduler._pe_path_results[request.request_id].path is Path.DE_READ
    assert scheduler._pe_forward_plans == {}
    assert scheduler._reqs_need_send_layerwise == {}
    assert scheduler._pe_invalid_request_ids == set()
    blocks.get_block_ids.assert_not_called()


@pytest.mark.parametrize(
    "invalid_case",
    [
        "misaligned_token_start",
        "group_count_mismatch",
        "insufficient_destination_coverage",
        "destination_table_drift",
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
    elif invalid_case == "destination_table_drift":
        blocks.get_block_ids.side_effect = lambda: (
            request.kv_transfer_params.__setitem__("remote_block_ids", [[90, 91, 92]]) or ([10, 11, 12],)
        )
    elif invalid_case.startswith("missing_"):
        request.kv_transfer_params.pop(invalid_case.removeprefix("missing_"))
    _decide(scheduler, request)

    scheduler.update_state_after_alloc(request, blocks, 0)

    assert request.request_id in scheduler._pe_invalid_request_ids
    assert request.request_id not in scheduler._pe_path_results
    assert request.request_id not in scheduler._pe_forward_plans
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

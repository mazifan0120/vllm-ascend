from __future__ import annotations

import json
import random
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    init_dual_path_worker_state,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_prefill_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
    ReversePlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathAbortNotice,
    PathAbortReason,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
    ReverseAttemptKey,
    RoundRobinPathPolicy,
    reverse_wire_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
    PathDecision,
    PathDecisionDeliveryError,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
_SCHEDULER_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler"
_DECODE_INSTANCE_ID = "decode-engine:2:boot-7"
_CONTROL_ENDPOINT = DecodeControlEndpoint(host="192.0.2.44", port=24001)


def _make_vllm_config(*, kv_role: str = "kv_consumer", failure_policy: str = "fail") -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.is_kv_consumer = kv_role in {"kv_consumer", "kv_both"}
    config.kv_transfer_config.engine_id = "decode-engine"
    config.kv_transfer_config.kv_port = 5000
    config.kv_transfer_config.kv_load_failure_policy = failure_policy
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {"tls_config": {}}.get(
        key, default
    )
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.world_size = 2
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.cache_config.block_size = 16
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    return config


def _make_kv_cache_config(*, group_count: int = 1) -> SimpleNamespace:
    groups = []
    for index in range(group_count):
        spec = MagicMock(name=f"kv_cache_spec_{index}")
        spec.block_size = 16
        group = MagicMock(name=f"kv_cache_group_{index}")
        group.kv_cache_spec = spec
        group.layer_names = [f"layer.{index}"]
        groups.append(group)
    return SimpleNamespace(kv_cache_groups=groups, kv_cache_tensors=[], num_blocks=64)


def _make_request(
    params: dict | None = None,
    *,
    request_id: str = "request-local-7",
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=49,
        num_preemptions=0,
        prompt_token_ids=list(range(49)),
        kv_transfer_params=params or {"do_remote_prefill": True, "metaserver": "http://proxy.example/v1/kv"},
    )


def _completed_future(error: RuntimeError | None = None) -> Future[None]:
    future: Future[None] = Future()
    if error is None:
        future.set_result(None)
    else:
        future.set_exception(error)
    return future


@pytest.fixture()
def task04_seams():
    with (
        patch(f"{_SCHEDULER_NS}.KVPoolSchedulerAdapter") as adapter_cls,
        patch(f"{_SCHEDULER_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_SCHEDULER_NS}.get_ip", return_value="192.0.2.44"),
    ):
        decode_coordinator = MagicMock(name="decode_coordinator")
        decode_coordinator.decode_engine_instance_id = _DECODE_INSTANCE_ID
        decode_coordinator.decode_control_endpoint = _CONTROL_ENDPOINT
        admission_ids = iter(range(1_000_000))
        decode_coordinator.new_request_key.side_effect = lambda request_id: DualPathRequestKey(
            _DECODE_INSTANCE_ID,
            request_id,
            next(admission_ids),
        )
        prefill_coordinator = MagicMock(name="prefill_coordinator")
        prefill_delivery_future = MagicMock(spec=Future, name="prefill_delivery_future")
        prefill_delivery_future.done.return_value = False
        prefill_delivery_future.cancelled.return_value = False
        prefill_delivery_future.exception.return_value = None
        prefill_coordinator.submit.return_value = prefill_delivery_future
        coordinator_cls.for_decode.return_value = decode_coordinator
        coordinator_cls.for_prefill.return_value = prefill_coordinator
        yield SimpleNamespace(
            adapter_cls=adapter_cls,
            coordinator_cls=coordinator_cls,
            decode_coordinator=decode_coordinator,
            prefill_coordinator=prefill_coordinator,
            prefill_delivery_future=prefill_delivery_future,
        )


@pytest.fixture()
def scheduler_factory(task04_seams):
    schedulers = []

    def make(
        *,
        role: str = "decode",
        config: MagicMock | None = None,
        kv_cache_config: SimpleNamespace | None = None,
        path_policy=None,
    ):
        scheduler_kwargs = {}
        if path_policy is not None:
            scheduler_kwargs["path_policy"] = path_policy
        scheduler = connector_module.DualPathConnectorScheduler(
            config or _make_vllm_config(kv_role="kv_consumer" if role == "decode" else "kv_producer"),
            kv_cache_config or _make_kv_cache_config(),
            "decode-engine",
            DualPathConfig(role=role, dual_path_control_port=7100 if role == "decode" else None),
            **scheduler_kwargs,
        )
        scheduler.executor.shutdown(wait=False)
        scheduler.metaserver_client.close()
        scheduler.executor = MagicMock(name=f"{role}_executor")
        scheduler.executor.submit.return_value = _completed_future()
        scheduler.side_channel_host = "198.51.100.20"
        schedulers.append(scheduler)
        return scheduler

    yield make
    for scheduler in schedulers:
        scheduler.shutdown()


@pytest.fixture()
def decode_scheduler(scheduler_factory):
    return scheduler_factory()


@pytest.fixture()
def proxy_echo():
    def echo(message):
        return json.loads(json.dumps({"kv_transfer_params": message}))["kv_transfer_params"]

    return echo


def _admit_request(scheduler, request: SimpleNamespace, block_ids: tuple[int, ...]):
    load_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
    scheduler._kvpool_adapter.lookup.return_value = load_spec
    assert scheduler.get_num_new_matched_tokens(request, 16) == (33, True)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(block_ids),)
    scheduler.update_state_after_alloc(request, blocks, 33)
    return request, blocks


def _admit(scheduler, params: dict | None = None):
    request = _make_request(params)
    return _admit_request(scheduler, request, (41, 42, 43, 44))


def _result(
    request_id: str = "request-local-7",
    reverse_attempt_id: int = 0,
    admission_id: int = 0,
) -> PathDecisionResult:
    return PathDecisionResult(
        request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, request_id, admission_id),
        path=PathKind.DE_READ,
        reverse_attempt_id=reverse_attempt_id,
        prefill_local_tokens=16,
    )


def _decision(result: PathDecisionResult | None = None) -> PathDecision:
    retained_result = result or _result()
    return PathDecision(
        result=retained_result,
        reverse_plan=ReversePlan(
            request_key=retained_result.request_key,
            wire_request_id=reverse_wire_id(
                ReverseAttemptKey(retained_result.request_key, retained_result.reverse_attempt_id)
            ),
            token_start=16,
            token_end=32,
            source_block_ids=((41, 42, 43, 44),),
            destination_block_ids=((71, 72, 73, 74),),
            remote_engine_id="prefill-engine",
            remote_host="198.51.100.10",
            remote_port=6000,
            remote_block_sizes=(16,),
            remote_tp_size=2,
            remote_pcp_size=1,
            remote_dcp_size=1,
            reverse_attempt_id=retained_result.reverse_attempt_id,
            prefill_local_tokens=16,
            reverse_send_job_id=None,
        ),
    )


def _control_only_worker():
    worker = init_dual_path_worker_state(object.__new__(connector_module.DualPathConnectorWorker))
    worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False))
    worker.kv_recv_layer_thread = MagicMock(name="kv_recv_layer_thread")
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    worker.request_map = {}
    worker.virtual_request = set()
    worker._recving_metadata = {}
    worker._invalid_block_ids = set()
    worker._kvpool_worker_adapter = MagicMock(name="kvpool_worker_adapter")
    worker.engine = MagicMock(name="transfer_engine")
    worker.block_size = [16]
    return worker


def _expected_dual_path_payload() -> dict:
    return {
        "decision_request": {
            "request_key": {
                "decode_engine_instance_id": _DECODE_INSTANCE_ID,
                "decode_request_id": "request-local-7",
                "admission_id": 0,
            },
            "target_tokens": 48,
            "decode_local_tokens": 16,
            "decode_store_tokens": 32,
        },
        "decode_control_endpoint": {"host": "192.0.2.44", "port": 24001},
    }


def _prefill_decision_payload(
    *,
    decode_request_id: str = "decode-request-7",
    admission_id: int = 0,
    target_tokens: int = 48,
    decode_local_tokens: int = 16,
    decode_store_tokens: int = 32,
) -> dict:
    return {
        "decision_request": {
            "request_key": {
                "decode_engine_instance_id": _DECODE_INSTANCE_ID,
                "decode_request_id": decode_request_id,
                "admission_id": admission_id,
            },
            "target_tokens": target_tokens,
            "decode_local_tokens": decode_local_tokens,
            "decode_store_tokens": decode_store_tokens,
        },
        "decode_control_endpoint": {
            "host": _CONTROL_ENDPOINT.host,
            "port": _CONTROL_ENDPOINT.port,
        },
    }


def _remote_decode_params(*, dual_path: dict | None = None, include_dual_path: bool = True) -> dict:
    params = {
        "do_remote_decode": True,
        "remote_block_ids": [[4, 5]],
        "remote_block_size": [16],
        "remote_cached_tokens": 0,
        "remote_engine_id": "decode-engine",
        "remote_host": "198.51.100.20",
        "remote_port": 6000,
    }
    if include_dual_path:
        params["dual_path"] = dual_path if dual_path is not None else _prefill_decision_payload()
    return params


def _make_prefill_request(request_id: str, params: dict) -> SimpleNamespace:
    prompt_token_ids = list(range(49))
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=len(prompt_token_ids),
        num_prompt_tokens=len(prompt_token_ids),
        num_computed_tokens=0,
        num_preemptions=0,
        max_tokens=16,
        prompt_token_ids=prompt_token_ids,
        prompt_embeds=None,
        _all_token_ids=list(prompt_token_ids),
        kv_transfer_params=params,
    )


def _bind_prefill(
    scheduler,
    request: SimpleNamespace,
    *,
    local_block_ids: tuple[int, ...] = (10, 11, 12, 13),
) -> MagicMock:
    decision_request = request.kv_transfer_params["dual_path"]["decision_request"]
    request.kv_transfer_params.update(
        {
            "remote_block_ids": [[4, 5, 6, 7]],
            "remote_cached_tokens": decision_request["decode_local_tokens"],
            "remote_tp_size": 2,
            "remote_pcp_size": 1,
            "remote_dcp_size": 1,
            "remote_pp_size": 1,
            "remote_dp_size": 1,
        }
    )
    blocks = MagicMock(name=f"{request.request_id}_blocks")
    blocks.get_block_ids.return_value = (list(local_block_ids),)
    scheduler.update_state_after_alloc(request, blocks, 0)
    return blocks


class TestDecodeConstructionGuards:
    def test_decode_rejects_multiple_kv_cache_groups(self, scheduler_factory, task04_seams):
        with pytest.raises(ValueError, match="exactly one KV cache group"):
            scheduler_factory(kv_cache_config=_make_kv_cache_config(group_count=2))
        task04_seams.adapter_cls.assert_not_called()

    def test_decode_rejects_non_fail_load_policy(self, scheduler_factory, task04_seams):
        config = _make_vllm_config(failure_policy="recompute")
        with pytest.raises(ValueError, match="kv_load_failure_policy.*fail"):
            scheduler_factory(config=config)
        task04_seams.adapter_cls.assert_not_called()

    def test_prefill_skips_decode_only_guards(self, scheduler_factory, task04_seams):
        config = _make_vllm_config(kv_role="kv_producer", failure_policy="recompute")
        scheduler = scheduler_factory(
            role="prefill",
            config=config,
            kv_cache_config=_make_kv_cache_config(group_count=2),
        )
        assert scheduler._kvpool_adapter is None
        task04_seams.adapter_cls.assert_not_called()


class TestDecodeAdmissionControl:
    def test_hbm_complete_bypass_has_no_task04_side_effects(self, decode_scheduler):
        request = _make_request()
        assert decode_scheduler.get_num_new_matched_tokens(request, 48) == (0, False)
        blocks = MagicMock()
        blocks.get_block_ids.return_value = ([1, 2, 3],)
        decode_scheduler.update_state_after_alloc(request, blocks, 0)
        assert decode_scheduler._decode_decision_states == {}
        decode_scheduler._path_decision_coordinator.register_pending.assert_not_called()
        decode_scheduler.executor.submit.assert_not_called()

    def test_hbm_complete_and_store_full_never_reach_decider(self, decode_scheduler):
        hbm_complete = _make_request(request_id="hbm-complete")
        store_full = _make_request(request_id="store-full")
        store_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
        store_blocks = MagicMock(name="store_full_blocks")
        store_blocks.get_block_ids.return_value = ([11, 12, 13],)

        with patch.object(scheduler_module.PathDecisionDecider, "decide", autospec=True) as decide:
            assert decode_scheduler.get_num_new_matched_tokens(hbm_complete, 48) == (0, False)
            decode_scheduler.update_state_after_alloc(hbm_complete, MagicMock(name="hbm_blocks"), 0)
            decode_scheduler._kvpool_adapter.lookup.return_value = store_spec
            assert decode_scheduler.get_num_new_matched_tokens(store_full, 16) == (32, True)
            decode_scheduler.update_state_after_alloc(store_full, store_blocks, 32)

        decide.assert_not_called()
        decode_scheduler._path_decision_coordinator.register_pending.assert_not_called()
        decode_scheduler.executor.submit.assert_not_called()

    def test_snapshot_builds_exact_key_request_and_metadata(self, decode_scheduler):
        request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        snapshot = decode_scheduler._decode_kv_snapshots[request.request_id]
        expected_key = DualPathRequestKey(_DECODE_INSTANCE_ID, request.request_id, 0)
        assert snapshot.transfer_tokens == 49
        assert snapshot.local_tokens == 16
        assert snapshot.external_tokens == 33
        assert state.request_key == expected_key
        assert state.decision_request == PathDecisionRequest(expected_key, 48, 16, 32)
        assert state.status is scheduler_module._DecodeDecisionStatus.PENDING
        assert (
            decode_scheduler.executor.submit.call_args.kwargs["message"]["dual_path"] == _expected_dual_path_payload()
        )

    def test_reused_local_request_id_gets_distinct_decode_admission_key(self, decode_scheduler):
        first_request, _ = _admit(decode_scheduler)
        first_key = decode_scheduler._decode_decision_states[first_request.request_id].request_key
        decode_scheduler.request_finished(first_request, [41, 42, 43, 44])

        second_request, _ = _admit(decode_scheduler)
        second_key = decode_scheduler._decode_decision_states[second_request.request_id].request_key

        assert first_key.decode_engine_instance_id == second_key.decode_engine_instance_id
        assert first_key.decode_request_id == second_key.decode_request_id == first_request.request_id
        assert (first_key.admission_id, second_key.admission_id) == (0, 1)
        assert first_key != second_key

    def test_consumed_decode_admission_does_not_mint_key_on_retry(self, decode_scheduler):
        request, blocks = _admit(decode_scheduler)
        coordinator = decode_scheduler._path_decision_coordinator
        assert request.kv_transfer_params["do_remote_prefill"] is False
        assert coordinator.new_request_key.call_count == 1

        request.num_preemptions += 1
        decode_scheduler.update_state_after_alloc(request, blocks, 0)

        assert coordinator.new_request_key.call_count == 1

    def test_register_pending_precedes_executor_submission(self, decode_scheduler):
        ordered = MagicMock()
        ordered.attach_mock(decode_scheduler._path_decision_coordinator.register_pending, "register_pending")
        ordered.attach_mock(decode_scheduler.executor.submit, "submit")
        _admit(decode_scheduler)
        assert ordered.method_calls[:2] == [
            call.register_pending(DualPathRequestKey(_DECODE_INSTANCE_ID, "request-local-7", 0)),
            call.submit(
                decode_scheduler._access_metaserver,
                url="http://proxy.example/v1/kv",
                message=decode_scheduler.executor.submit.call_args.kwargs["message"],
            ),
        ]

    def test_outgoing_message_preserves_mooncake_fields_and_hybrid_trim(self, decode_scheduler):
        decode_scheduler.need_truncate = True
        with patch.object(
            scheduler_module,
            "get_external_request_id",
            return_value="external-request-7",
        ) as external_id:
            request = _make_request()
            load_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
            decode_scheduler._kvpool_adapter.lookup.return_value = load_spec
            assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
            blocks = MagicMock()
            blocks.get_block_ids.return_value = ([41, 42, 43, 44],)
            decode_scheduler.update_state_after_alloc(request, blocks, 32)
        external_id.assert_called_once_with("request-local-7")
        message = decode_scheduler.executor.submit.call_args.kwargs["message"]
        assert message == {
            "token_ids": [],
            "request_id": "external-request-7",
            "do_remote_prefill": False,
            "do_remote_decode": True,
            "remote_block_ids": ([41, 42, 43],),
            "remote_block_size": [16],
            "remote_engine_id": "decode-engine",
            "remote_host": "198.51.100.20",
            "remote_port": 5000,
            "remote_tp_size": 2,
            "remote_pcp_size": 1,
            "remote_dcp_size": 1,
            "remote_pp_size": 1,
            "remote_dp_size": 1,
            "remote_cached_tokens": 16,
            "dual_path": _expected_dual_path_payload(),
        }

    def test_proxy_echo_preserves_nested_mapping_at_json_value_level(self, decode_scheduler, proxy_echo):
        _admit(decode_scheduler)
        message = decode_scheduler.executor.submit.call_args.kwargs["message"]
        echoed = proxy_echo(message)
        assert echoed["dual_path"] == _expected_dual_path_payload()

    def test_initial_submission_never_enters_parent_receive_queue(self, decode_scheduler):
        request, _ = _admit(decode_scheduler)
        assert decode_scheduler._reqs_need_recv == {}
        assert request.kv_transfer_params["do_remote_prefill"] is False
        decode_scheduler._path_decision_coordinator.register_pending.assert_called_once()
        decode_scheduler.executor.submit.assert_called_once()

    def test_identical_duplicate_does_not_register_or_notify_again(self, decode_scheduler):
        request, blocks = _admit(decode_scheduler)
        first_state = decode_scheduler._decode_decision_states[request.request_id]
        decode_scheduler._path_decision_coordinator.register_pending.reset_mock()
        decode_scheduler.executor.submit.reset_mock()
        decode_scheduler.update_state_after_alloc(request, blocks, 33)
        assert decode_scheduler._decode_decision_states[request.request_id] is first_state
        decode_scheduler._path_decision_coordinator.register_pending.assert_not_called()
        decode_scheduler.executor.submit.assert_not_called()

    def test_http_success_remains_pending(self, decode_scheduler):
        future = _completed_future()
        decode_scheduler.executor.submit.return_value = future
        request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is scheduler_module._DecodeDecisionStatus.PENDING
        decode_scheduler.executor.submit.assert_called_once()

    def test_synchronous_executor_failure_remains_pending_without_retry(self, decode_scheduler):
        decode_scheduler.executor.submit.side_effect = RuntimeError("executor closed")
        with patch.object(scheduler_module.logger, "error") as log_error:
            request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is scheduler_module._DecodeDecisionStatus.PENDING
        decode_scheduler.executor.submit.assert_called_once()
        log_error.assert_called_once()

    def test_async_future_failure_is_logged_and_remains_pending(self, decode_scheduler):
        future = _completed_future(RuntimeError("proxy unavailable"))
        decode_scheduler.executor.submit.return_value = future
        with patch.object(scheduler_module.logger, "error") as log_error:
            request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is scheduler_module._DecodeDecisionStatus.PENDING
        decode_scheduler.executor.submit.assert_called_once()
        log_error.assert_called_once()

    def test_virtual_request_skips_http_and_remains_pending(self, decode_scheduler):
        params = {
            "do_remote_prefill": True,
            "do_virtual": True,
            "metaserver": "http://proxy.example/v1/kv",
        }
        request, _ = _admit(decode_scheduler, params)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is scheduler_module._DecodeDecisionStatus.PENDING
        decode_scheduler.executor.submit.assert_not_called()


class TestPrefillDecisionHook:
    @staticmethod
    def _assert_parent_accounting_once(scheduler, request, expected_result=(0, False)) -> None:
        scheduler.need_truncate = True
        parent_method = MooncakeLayerwiseConnectorScheduler.get_num_new_matched_tokens
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            side_effect=parent_method,
        ) as parent_spy:
            result = scheduler.get_num_new_matched_tokens(request, 0)

        assert result == expected_result
        parent_spy.assert_called_once_with(scheduler, request, 0)
        assert request.kv_transfer_params["_p_side_truncated"] is True
        assert len(request.prompt_token_ids) == 48

    def test_parent_accounting_runs_exactly_once_for_valid_dual_path(self, scheduler_factory):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.DE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-valid", _remote_decode_params())

        self._assert_parent_accounting_once(scheduler, request, (32, True))

        policy.choose.assert_called_once()
        assert scheduler._prefill_local_tokens == {request.request_id: 0}

    def test_invalid_token_bounds_fence_same_id_reuse_by_request_identity(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-invalid-local", _remote_decode_params())
        replacement = _make_prefill_request(request.request_id, _remote_decode_params())
        assert (
            request.kv_transfer_params["dual_path"]["decision_request"]
            == replacement.kv_transfer_params["dual_path"]["decision_request"]
        )

        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=(0, False),
        ):
            result = scheduler.get_num_new_matched_tokens(request, 50)
            with pytest.raises(RuntimeError, match="request-id reuse"):
                scheduler.get_num_new_matched_tokens(replacement, 0)

        assert result == (0, False)
        policy.choose.assert_not_called()
        task04_seams.prefill_coordinator.submit.assert_not_called()
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            PathAbortNotice(
                request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0),
                reason=PathAbortReason.DECISION_FAILED,
            ),
        )
        assert scheduler._prefill_invalid_request_ids == {request.request_id}
        assert scheduler._prefill_local_tokens == {}

    def test_prefill_hook_passes_incoming_l_pe_to_forced_eligibility(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.DE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-forced", _remote_decode_params())

        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=(0, False),
        ):
            result = scheduler.get_num_new_matched_tokens(request, 32)

        assert result == (0, False)
        policy.choose.assert_not_called()
        assert scheduler._prefill_path_results[request.request_id].path is PathKind.PE_READ
        task04_seams.prefill_coordinator.submit.assert_not_called()
        assert scheduler._prefill_local_tokens == {request.request_id: 32}

    def test_parent_accounting_runs_exactly_once_for_ordinary_request(self, scheduler_factory, task04_seams):
        scheduler = scheduler_factory(role="prefill")
        request = _make_prefill_request(
            "prefill-ordinary",
            _remote_decode_params(include_dual_path=False),
        )

        self._assert_parent_accounting_once(scheduler, request)

        task04_seams.prefill_coordinator.submit.assert_not_called()

    @pytest.mark.parametrize(
        "params",
        [
            pytest.param(
                {"dual_path": _prefill_decision_payload()},
                id="missing-do-remote-decode",
            ),
            pytest.param(
                {
                    "do_remote_decode": False,
                    "dual_path": _prefill_decision_payload(),
                },
                id="false-do-remote-decode",
            ),
        ],
    )
    def test_valid_envelope_without_true_remote_decode_does_not_activate_decision(
        self,
        params,
        scheduler_factory,
        task04_seams,
    ):
        # Given
        policy = MagicMock(name="path_policy")
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-not-remote-decode", params)
        parent_result = (7, True)

        # When
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=parent_result,
        ):
            result = scheduler.get_num_new_matched_tokens(request, 0)

        # Then
        assert result == parent_result
        policy.choose.assert_not_called()
        task04_seams.prefill_coordinator.submit.assert_not_called()
        assert scheduler._prefill_request_keys == {}
        assert scheduler._prefill_delivery_futures == {}

    def test_seeded_non_full_requests_invoke_policy_once_and_alternate(self, scheduler_factory, task04_seams):
        policy = MagicMock(spec=RoundRobinPathPolicy, wraps=RoundRobinPathPolicy(random.Random(1)))
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        requests = [
            _make_prefill_request(
                f"prefill-{index}",
                _remote_decode_params(dual_path=_prefill_decision_payload(decode_request_id=f"decode-request-{index}")),
            )
            for index in range(2)
        ]

        results = [scheduler.get_num_new_matched_tokens(request, 0) for request in requests]

        assert results == [(0, False), (32, True)]
        assert policy.choose.call_count == 2
        assert [scheduler._prefill_path_results[request.request_id].path for request in requests] == [
            PathKind.PE_READ,
            PathKind.DE_READ,
        ]
        task04_seams.prefill_coordinator.submit.assert_not_called()

    def test_identical_replay_neither_advances_policy_nor_submits_second_future(self, scheduler_factory, task04_seams):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-replay", _remote_decode_params())

        first = scheduler.get_num_new_matched_tokens(request, 0)
        second = scheduler.get_num_new_matched_tokens(request, 0)
        _bind_prefill(scheduler, request)
        _bind_prefill(scheduler, request)

        assert first == second == (0, False)
        policy.choose.assert_called_once()
        task04_seams.prefill_coordinator.submit.assert_called_once()
        assert list(scheduler._prefill_forward_plans) == [request.request_id]
        assert list(scheduler._prefill_delivery_futures) == [request.request_id]

    def test_undelivered_conflicting_facts_discard_and_redecide(self, scheduler_factory, task04_seams):
        # An admission retry carrying changed facts before any delivery discards
        # the uncommitted decision and decides fresh from the new facts.
        policy = MagicMock(name="path_policy")
        policy.choose.side_effect = [PathKind.PE_READ, PathKind.DE_READ]
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        original = _make_prefill_request(
            "prefill-conflict",
            _remote_decode_params(dual_path=_prefill_decision_payload(decode_store_tokens=24)),
        )
        conflicting = _make_prefill_request(
            original.request_id,
            _remote_decode_params(dual_path=_prefill_decision_payload(decode_store_tokens=32)),
        )
        first = scheduler.get_num_new_matched_tokens(original, 0)
        second = scheduler.get_num_new_matched_tokens(conflicting, 0)

        assert first == (0, False)
        assert second == (32, True)
        assert policy.choose.call_count == 2
        task04_seams.prefill_coordinator.submit.assert_not_called()
        assert scheduler._prefill_invalid_request_ids == set()
        assert scheduler._prefill_forward_plans == {}
        assert scheduler._prefill_pending_reverse_receive_bindings == {}
        assert scheduler._reqs_need_send_layerwise == {}
        retained = scheduler._prefill_decision_metadata[original.request_id].decision_request
        assert retained.decode_store_tokens == 32
        assert scheduler._prefill_path_results[original.request_id].path is PathKind.DE_READ

    def test_undelivered_prefill_prefix_change_discards_and_redecides(
        self,
        scheduler_factory,
        task04_seams,
    ):
        # The same request re-admitted with a different local prefix (e.g. an
        # allocation-failure retry re-probing a grown prefix cache) re-decides
        # from the new prefix instead of raising.
        policy = MagicMock(name="path_policy")
        policy.choose.side_effect = [PathKind.PE_READ, PathKind.DE_READ]
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-prefix-conflict", _remote_decode_params())

        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=(0, False),
        ):
            first = scheduler.get_num_new_matched_tokens(request, 0)
            second = scheduler.get_num_new_matched_tokens(request, 16)

        assert first == (0, False)
        assert second == (16, True)
        assert policy.choose.call_count == 2
        task04_seams.prefill_coordinator.submit.assert_not_called()
        assert scheduler._prefill_invalid_request_ids == set()
        assert scheduler._prefill_forward_plans == {}
        assert scheduler._prefill_pending_reverse_receive_bindings == {}
        assert scheduler._reqs_need_send_layerwise == {}
        assert scheduler._prefill_local_tokens[request.request_id] == 16
        assert scheduler._prefill_path_results[request.request_id].path is PathKind.DE_READ

    def test_delivered_conflicting_facts_resume_with_frozen_path(self, scheduler_factory, task04_seams):
        # Once the decision has been delivered, re-deciding would fork the
        # protocol (e.g. preemption resume after delivery): the resume branch
        # reuses the frozen PathKind and never touches the invalid set.
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-delivered-conflict", _remote_decode_params())

        parent_result = (7, True)
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=parent_result,
        ):
            first = scheduler.get_num_new_matched_tokens(request, 0)
            _bind_prefill(scheduler, request)
            second = scheduler.get_num_new_matched_tokens(request, 16)

        assert first == (0, False)
        assert second == (0, False)
        policy.choose.assert_called_once()
        task04_seams.prefill_coordinator.submit.assert_called_once()
        assert scheduler._prefill_invalid_request_ids == set()
        assert scheduler._prefill_path_results[request.request_id].path is PathKind.PE_READ
        assert list(scheduler._prefill_forward_plans) == [request.request_id]

    def test_same_key_request_id_reuse_is_fenced_by_request_identity(
        self,
        scheduler_factory,
        task04_seams,
    ):
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        old_request = _make_prefill_request("prefill-reused", _remote_decode_params())
        new_request = _make_prefill_request("prefill-reused", _remote_decode_params())

        assert scheduler.get_num_new_matched_tokens(old_request, 0) == (0, False)
        _bind_prefill(scheduler, old_request)

        with pytest.raises(RuntimeError, match="request-id reuse"):
            scheduler.get_num_new_matched_tokens(new_request, 0)

        policy.choose.assert_called_once()
        task04_seams.prefill_coordinator.submit.assert_called_once()

    def test_malformed_decision_envelope_converges_to_parent_result(self, scheduler_factory, task04_seams):
        policy = MagicMock(name="path_policy")
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request(
            "prefill-malformed",
            _remote_decode_params(dual_path={"unexpected": "shape"}),
        )

        parent_result = (7, True)
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=parent_result,
        ):
            first = scheduler.get_num_new_matched_tokens(request, 0)
            second = scheduler.get_num_new_matched_tokens(request, 0)

        assert first == second == parent_result
        policy.choose.assert_not_called()
        task04_seams.prefill_coordinator.submit.assert_not_called()
        task04_seams.prefill_coordinator.submit_abort.assert_not_called()
        assert scheduler._prefill_invalid_request_ids == {request.request_id}

    def test_malformed_first_contact_fences_same_id_reuse_by_request_identity(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request(
            "prefill-malformed-reused",
            _remote_decode_params(dual_path={"unexpected": "shape"}),
        )
        replacement = _make_prefill_request(request.request_id, _remote_decode_params())

        parent_result = (7, True)
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "get_num_new_matched_tokens",
            autospec=True,
            return_value=parent_result,
        ):
            assert scheduler.get_num_new_matched_tokens(request, 0) == parent_result
            with pytest.raises(RuntimeError, match="request-id reuse"):
                scheduler.get_num_new_matched_tokens(replacement, 0)

        policy.choose.assert_not_called()
        task04_seams.prefill_coordinator.submit.assert_not_called()

    def test_active_owner_alone_fences_request_id_reuse_after_delivery_reconciles(
        self,
        scheduler_factory,
        task04_seams,
    ):
        task04_seams.prefill_coordinator.submit.return_value = _completed_future()
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-reconciled-owner", _remote_decode_params())
        replacement = _make_prefill_request(request.request_id, _remote_decode_params())

        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        _bind_prefill(scheduler, request)
        scheduler.build_connector_meta(MagicMock(name="reconcile_scheduler_output"))

        assert request.request_id not in scheduler._prefill_delivery_records
        assert scheduler._prefill_admission_owners[request.request_id] is request
        with pytest.raises(RuntimeError, match="request-id reuse"):
            scheduler.get_num_new_matched_tokens(replacement, 0)

        policy.choose.assert_called_once()
        task04_seams.prefill_coordinator.submit.assert_called_once()

    def test_update_state_after_alloc_suppresses_send_queue_for_dual_path(self, scheduler_factory):
        scheduler = scheduler_factory(role="prefill")
        request = _make_prefill_request("prefill-valid", _remote_decode_params())
        blocks = MagicMock(name="blocks")

        scheduler.update_state_after_alloc(request, blocks, 0)

        assert scheduler._reqs_need_send_layerwise == {}
        blocks.get_block_ids.assert_not_called()

    def test_ordinary_remote_decode_alloc_matches_parent(self, scheduler_factory):
        parent = MooncakeLayerwiseConnectorScheduler(
            _make_vllm_config(kv_role="kv_producer"),
            _make_kv_cache_config(),
            "prefill-engine",
        )
        scheduler = scheduler_factory(role="prefill")
        parent_request = _make_prefill_request(
            "prefill-ordinary",
            _remote_decode_params(include_dual_path=False),
        )
        dual_request = _make_prefill_request(
            "prefill-ordinary",
            _remote_decode_params(include_dual_path=False),
        )
        parent_blocks = MagicMock(name="parent_blocks")
        parent_blocks.get_block_ids.return_value = ([7, 8, 9],)
        dual_blocks = MagicMock(name="dual_blocks")
        dual_blocks.get_block_ids.return_value = ([7, 8, 9],)
        parent_method = MooncakeLayerwiseConnectorScheduler.update_state_after_alloc

        try:
            parent.update_state_after_alloc(parent_request, parent_blocks, 0)
            with patch.object(
                MooncakeLayerwiseConnectorScheduler,
                "update_state_after_alloc",
                autospec=True,
                side_effect=parent_method,
            ) as parent_spy:
                scheduler.update_state_after_alloc(dual_request, dual_blocks, 0)

            parent_spy.assert_called_once_with(scheduler, dual_request, dual_blocks, 0)
            parent_info = parent._reqs_need_send_layerwise[parent_request.request_id]
            dual_info = scheduler._reqs_need_send_layerwise[dual_request.request_id]
            assert dual_info.local_block_ids == parent_info.local_block_ids
            assert dual_info.local_transferred_tokens == parent_info.local_transferred_tokens
            assert dual_info.local_computed_tokens == parent_info.local_computed_tokens
        finally:
            parent.executor.shutdown(wait=False)
            parent.metaserver_client.close()


class TestPrefillAbortTriggers:
    @staticmethod
    def _expected_notice(reason: PathAbortReason) -> PathAbortNotice:
        return PathAbortNotice(
            request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0),
            reason=reason,
        )

    def test_initial_decision_failure_fences_same_id_reuse_by_request_identity(
        self,
        scheduler_factory,
        task04_seams,
    ):
        scheduler = scheduler_factory(role="prefill")
        request = _make_prefill_request("prefill-initial-decision-failure", _remote_decode_params())
        replacement = _make_prefill_request(request.request_id, _remote_decode_params())
        assert (
            request.kv_transfer_params["dual_path"]["decision_request"]
            == replacement.kv_transfer_params["dual_path"]["decision_request"]
        )
        assert scheduler._path_decider is not None

        with patch.object(
            scheduler._path_decider,
            "decide",
            side_effect=PathDecisionValidationError("decision failed"),
        ):
            scheduler.get_num_new_matched_tokens(request, 0)

        with pytest.raises(RuntimeError, match="request-id reuse"):
            scheduler.get_num_new_matched_tokens(replacement, 0)

        assert scheduler._prefill_invalid_request_ids == {request.request_id}
        assert scheduler._prefill_request_keys == {}
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            self._expected_notice(PathAbortReason.DECISION_FAILED),
        )

    def test_fresh_redecision_failure_emits_abort_from_current_metadata(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        original = _make_prefill_request(
            "prefill-fresh-decision-failure",
            _remote_decode_params(dual_path=_prefill_decision_payload(decode_store_tokens=24)),
        )
        conflicting = _make_prefill_request(
            original.request_id,
            _remote_decode_params(dual_path=_prefill_decision_payload(decode_store_tokens=32)),
        )
        scheduler.get_num_new_matched_tokens(original, 0)
        assert scheduler._path_decider is not None

        with patch.object(
            scheduler._path_decider,
            "decide",
            side_effect=[
                PathDecisionValidationError("facts changed"),
                PathDecisionValidationError("fresh decision failed"),
            ],
        ):
            scheduler.get_num_new_matched_tokens(conflicting, 0)

        assert scheduler._prefill_invalid_request_ids == {original.request_id}
        assert scheduler._prefill_admission_owners[original.request_id] is conflicting
        assert scheduler._prefill_request_keys == {}
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            self._expected_notice(PathAbortReason.DECISION_FAILED),
        )

    def test_local_activation_exception_emits_activation_failed_abort(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-activation-failure", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(request, 0)

        with patch.object(
            scheduler,
            "_try_install_forward_plan",
            side_effect=PathDecisionValidationError("invalid allocation"),
        ):
            _bind_prefill(scheduler, request)

        assert scheduler._prefill_invalid_request_ids == {request.request_id}
        assert scheduler._prefill_admission_owners[request.request_id] is request
        replacement = _make_prefill_request(request.request_id, _remote_decode_params())
        with pytest.raises(RuntimeError, match="request-id reuse"):
            scheduler.get_num_new_matched_tokens(replacement, 0)
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            self._expected_notice(PathAbortReason.ACTIVATION_FAILED),
        )

    def test_real_malformed_forward_plan_emits_activation_failed_abort(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-malformed-forward", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(request, 0)
        decision_request = request.kv_transfer_params["dual_path"]["decision_request"]
        request.kv_transfer_params.update(
            {
                "remote_block_ids": [[4, 5, 6, 7]],
                "remote_cached_tokens": decision_request["decode_local_tokens"],
                "remote_tp_size": 2,
                "remote_pcp_size": 1,
                "remote_dcp_size": 1,
                "remote_pp_size": 1,
            }
        )
        blocks = MagicMock(name="malformed_forward_blocks")
        blocks.get_block_ids.return_value = ([10, 11, 12, 13],)

        scheduler.update_state_after_alloc(request, blocks, 0)

        assert scheduler._prefill_invalid_request_ids == {request.request_id}
        assert scheduler._prefill_path_results == {}
        assert scheduler._prefill_forward_plans == {}
        task04_seams.prefill_coordinator.submit.assert_not_called()
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            self._expected_notice(PathAbortReason.ACTIVATION_FAILED),
        )

    def test_closed_coordinator_submit_failure_still_attempts_abort_without_escaping(
        self,
        scheduler_factory,
        task04_seams,
    ):
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-submit-failure", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(request, 0)
        task04_seams.prefill_coordinator.submit.side_effect = RuntimeError("coordinator closed")
        task04_seams.prefill_coordinator.submit_abort.side_effect = RuntimeError("coordinator closed")

        with patch.object(scheduler_module.logger, "error") as log_error:
            _bind_prefill(scheduler, request)

        assert scheduler._prefill_invalid_request_ids == {request.request_id}
        assert scheduler._prefill_forward_plans == {}
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            self._expected_notice(PathAbortReason.ACTIVATION_FAILED),
        )
        assert log_error.call_count == 2

    def test_exhausted_abort_delivery_is_logged_asynchronously(self, scheduler_factory, task04_seams):
        scheduler = scheduler_factory(role="prefill")
        abort_error = PathDecisionDeliveryError("abort delivery exhausted")
        task04_seams.prefill_coordinator.submit_abort.return_value = _completed_future(abort_error)
        notice = self._expected_notice(PathAbortReason.DELIVERY_EXHAUSTED)

        with patch.object(scheduler_module.logger, "error") as log_error:
            scheduler._send_abort_notice(notice.request_key, _CONTROL_ENDPOINT, notice.reason)

        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(_CONTROL_ENDPOINT, notice)
        assert abort_error in log_error.call_args.args
        assert (log_error.call_args.args[0] % log_error.call_args.args[1:]).endswith("admission_id=0")

    @pytest.mark.parametrize(
        ("method_name", "block_ids"),
        [
            pytest.param("request_finished", [4, 5], id="single-group"),
            pytest.param("request_finished_all_groups", ([4, 5],), id="all-groups"),
        ],
    )
    def test_predecision_client_abort_emits_request_aborted(
        self,
        method_name,
        block_ids,
        scheduler_factory,
        task04_seams,
    ):
        scheduler = scheduler_factory(role="prefill")
        request = _make_prefill_request("prefill-client-abort", _remote_decode_params())
        request.status = RequestStatus.FINISHED_ABORTED
        assert scheduler._prefill_request_keys == {}

        assert getattr(scheduler, method_name)(request, block_ids) == (False, None)

        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            self._expected_notice(PathAbortReason.REQUEST_ABORTED),
        )

    @pytest.mark.parametrize(
        "status",
        [RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED],
    )
    def test_normal_prefill_completion_does_not_emit_abort(
        self,
        status,
        scheduler_factory,
        task04_seams,
    ):
        scheduler = scheduler_factory(role="prefill")
        request = _make_prefill_request("prefill-normal-finish", _remote_decode_params())
        request.status = status

        scheduler.request_finished(request, [4, 5])

        task04_seams.prefill_coordinator.submit_abort.assert_not_called()


class TestDecodeResultConsumption:
    def test_received_result_commits_via_build_connector_meta_without_worker_metadata(
        self,
        decode_scheduler,
        task04_seams,
    ):
        # Given
        request, _ = _admit(decode_scheduler)
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        result = _result()
        task04_seams.decode_coordinator.take_received_decisions.return_value = [_decision(result)]

        # When
        metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        # Then
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is scheduler_module._DecodeDecisionStatus.COMMITTED
        assert metadata.control_failures == []
        assert request.request_id not in metadata.requests
        assert request.status is RequestStatus.WAITING_FOR_REMOTE_KVS
        assert decode_scheduler._reqs_need_recv == {}
        task04_seams.decode_coordinator.unregister.assert_not_called()

    def test_late_decision_after_activation_failure_is_stale(self, decode_scheduler, task04_seams):
        request, snapshot = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        invalid_decision = PathDecision(
            result=_result(),
            reverse_plan=None,
        )
        task04_seams.decode_coordinator.take_received_decisions.return_value = [invalid_decision]
        first_metadata = decode_scheduler.build_connector_meta(MagicMock(name="failure_scheduler_output"))
        task04_seams.decode_coordinator.take_received_decisions.return_value = [_decision()]

        late_metadata = decode_scheduler.build_connector_meta(MagicMock(name="late_scheduler_output"))

        assert snapshot.get_block_ids.return_value == ([41, 42, 43, 44],)
        assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
        assert len(first_metadata.control_failures) == 1
        assert late_metadata.control_failures == []
        assert late_metadata.reverse_plans == []


class TestWorkerFailureRelay:
    def test_control_failure_metadata_starts_no_store_or_p2p_operation(self):
        # Given
        worker = _control_only_worker()
        metadata = DualPathConnectorMetadata()
        metadata.control_failures.append(
            DualPathControlFailureMetadata(
                "request-local-7",
                (42, 43, 44),
                DualPathControlFailureReason.ACTIVATION_FAILED,
            )
        )
        assert metadata.requests == {}

        # When
        worker.start_load_kv(metadata)

        # Then
        assert worker._control_failed_recving == {"request-local-7"}
        assert worker._invalid_block_ids == {42, 43, 44}
        assert worker.kv_recv_layer_thread.method_calls == []
        assert worker._kvpool_worker_adapter.method_calls == []
        assert worker.engine.method_calls == []

    def test_single_kv_connector_output_carries_finished_recving_and_invalid_blocks(self):
        # Given
        worker = _control_only_worker()
        metadata = DualPathConnectorMetadata()
        metadata.control_failures.append(
            DualPathControlFailureMetadata(
                "request-local-7",
                (42, 43, 44),
                DualPathControlFailureReason.ACTIVATION_FAILED,
            )
        )
        worker.start_load_kv(metadata)

        # When
        done_sending, done_recving = worker.get_finished(set(), metadata)
        invalid_block_ids = worker.get_block_ids_with_load_errors()

        # Then
        assert done_sending == set()
        assert done_recving == {"request-local-7"}
        assert invalid_block_ids == {42, 43, 44}
        assert worker.get_finished(set(), metadata) == (set(), set())
        assert worker.get_block_ids_with_load_errors() == set()

    def test_finished_request_removes_pending_control_failure_without_publication(self):
        # Given
        worker = _control_only_worker()
        metadata = DualPathConnectorMetadata()
        metadata.control_failures.append(
            DualPathControlFailureMetadata(
                "request-local-7",
                (42, 43, 44),
                DualPathControlFailureReason.ACTIVATION_FAILED,
            )
        )
        worker.start_load_kv(metadata)

        # When
        result = worker.get_finished({"request-local-7"}, metadata)

        # Then
        assert result == (set(), set())
        assert worker._control_failed_recving == set()


class TestCleanupAndShutdown:
    @pytest.mark.parametrize(
        ("method_name", "block_ids"),
        [
            ("request_finished", [41, 42, 43, 44]),
            ("request_finished_all_groups", ([41, 42, 43, 44],)),
        ],
    )
    def test_cancellation_before_result_cleans_all_state(
        self,
        method_name,
        block_ids,
        decode_scheduler,
        task04_seams,
    ):
        # Given
        request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        parent_result = (True, {"owner": "parent"})

        # When
        with patch.object(
            MooncakeLayerwiseConnectorScheduler,
            method_name,
            autospec=True,
            return_value=parent_result,
        ) as parent_finish:
            result = getattr(decode_scheduler, method_name)(request, block_ids)

        # Then
        assert result == parent_result
        assert decode_scheduler._lookup_results == {}
        assert decode_scheduler._decode_kv_snapshots == {}
        assert decode_scheduler._decode_decision_states == {}
        task04_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)
        parent_finish.assert_called_once_with(decode_scheduler, request, block_ids)

    def test_cancellation_after_commit_unregisters_first_time(self, decode_scheduler, task04_seams):
        # Given
        request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        task04_seams.decode_coordinator.take_received_decisions.return_value = [_decision()]
        commit_metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        assert state.status is scheduler_module._DecodeDecisionStatus.COMMITTED
        task04_seams.decode_coordinator.unregister.assert_not_called()

        # When: the reverse-send job is still open, so the finish is delayed
        # (engine progress) until the job closes.
        result = decode_scheduler.request_finished(request, [41, 42, 43, 44])

        # Then
        assert result == (True, None)
        reverse_send_job_id = commit_metadata.reverse_plans[0].reverse_send_job_id
        decode_scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={reverse_send_job_id: 2}))
        )
        result_after = decode_scheduler.request_finished(request, [41, 42, 43, 44])
        assert result_after == (False, None)
        assert request.request_id not in decode_scheduler._decode_decision_states
        task04_seams.decode_coordinator.unregister.assert_called_once_with(state.request_key)

    def test_cancellation_after_peer_abort_unregisters_idempotently(self, decode_scheduler, task04_seams):
        # Given
        request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        task04_seams.decode_coordinator.take_received_decisions.return_value = []
        task04_seams.decode_coordinator.take_received_aborts.return_value = [
            PathAbortNotice(
                request_key=state.request_key,
                reason=PathAbortReason.REQUEST_ABORTED,
            )
        ]
        decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED

        # When
        result = decode_scheduler.request_finished(request, [41, 42, 43, 44])

        # Then
        assert result == (False, None)
        assert request.request_id not in decode_scheduler._decode_decision_states
        assert task04_seams.decode_coordinator.unregister.call_args_list == [
            call(state.request_key),
            call(state.request_key),
        ]

    @pytest.mark.parametrize(
        ("method_name", "block_ids"),
        [
            ("request_finished", [4, 5]),
            ("request_finished_all_groups", ([4, 5],)),
        ],
    )
    def test_pe_finish_during_inflight_delivery_defers_then_sweeps(
        self,
        method_name,
        block_ids,
        scheduler_factory,
        task04_seams,
    ):
        # Given
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-inflight", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(request, 0)
        _bind_prefill(scheduler, request)
        request_key = DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0)
        assert scheduler._path_decider is not None

        # When
        with patch.object(
            scheduler._path_decider,
            "discard",
            wraps=scheduler._path_decider.discard,
        ) as discard:
            getattr(scheduler, method_name)(request, block_ids)
            assert request.request_id not in scheduler._prefill_request_keys
            assert request.request_id not in scheduler._prefill_local_tokens
            assert scheduler._prefill_delivery_futures == {request.request_id: delivery_future}
            discard.assert_called_once_with(request_key)

            delivery_future.set_result(None)
            scheduler.build_connector_meta(MagicMock(name="idle_scheduler_output"))

        # Then
        discard.assert_called_once_with(request_key)
        assert scheduler._prefill_delivery_futures == {}
        assert request_key not in scheduler._path_decider._decision_records

    @pytest.mark.parametrize(
        ("method_name", "block_ids"),
        [
            ("request_finished", [4, 5]),
            ("request_finished_all_groups", ([4, 5],)),
        ],
    )
    def test_pe_finish_clears_invalid_request_marker(
        self,
        method_name,
        block_ids,
        scheduler_factory,
    ):
        # Given
        scheduler = scheduler_factory(role="prefill")
        request = _make_prefill_request(
            "prefill-invalid-cleanup",
            _remote_decode_params(),
        )
        scheduler._prefill_invalid_request_ids.add(request.request_id)
        assert scheduler._prefill_invalid_request_ids == {request.request_id}

        # When
        result = getattr(scheduler, method_name)(request, block_ids)

        # Then
        assert result == (False, None)
        assert scheduler._prefill_invalid_request_ids == set()

    def test_pe_read_delivery_exhaustion_emits_peer_abort_on_next_decode_build(
        self,
        scheduler_factory,
        task04_seams,
    ):
        # Given
        decode_scheduler = scheduler_factory(role="decode")
        decode_request, _ = _admit(decode_scheduler)
        decode_state = decode_scheduler._decode_decision_states[decode_request.request_id]
        task04_seams.decode_coordinator.take_received_decisions.return_value = []
        received_aborts = []
        task04_seams.decode_coordinator.take_received_aborts.side_effect = lambda: (
            [received_aborts.pop(0)] if received_aborts else []
        )

        delivery_error = PathDecisionDeliveryError("delivery exhausted")
        delivery_future: Future[None] = Future()
        delivery_future.set_exception(delivery_error)
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        task04_seams.prefill_coordinator.submit_abort.side_effect = lambda endpoint, notice: (
            received_aborts.append(notice) or _completed_future()
        )
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        prefill_scheduler = scheduler_factory(role="prefill", path_policy=policy)
        prefill_request = _make_prefill_request(
            "prefill-delivery-failure",
            _remote_decode_params(
                dual_path=_prefill_decision_payload(decode_request_id=decode_request.request_id),
            ),
        )

        # When
        with patch.object(scheduler_module.logger, "error") as log_error:
            prefill_scheduler.get_num_new_matched_tokens(prefill_request, 0)
            prefill_scheduler.get_num_new_matched_tokens(prefill_request, 0)
            _bind_prefill(prefill_scheduler, prefill_request)
            prefill_scheduler.build_connector_meta(MagicMock(name="prefill_scheduler_output"))
            metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        # Then
        log_error.assert_called_once()
        assert delivery_future.exception() is delivery_error
        policy.choose.assert_called_once()
        task04_seams.prefill_coordinator.submit.assert_called_once()
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            PathAbortNotice(
                request_key=decode_state.request_key,
                reason=PathAbortReason.DELIVERY_EXHAUSTED,
            ),
        )
        decode_scheduler.executor.submit.assert_called_once()
        assert decode_state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
        assert metadata.control_failures == [
            DualPathControlFailureMetadata(
                request_id=decode_request.request_id,
                invalid_block_ids=(42, 43, 44),
                reason=DualPathControlFailureReason.PEER_ABORT,
            )
        ]
        assert prefill_request.request_id in prefill_scheduler._prefill_invalid_request_ids

    @pytest.mark.parametrize("cancelled", [False, True])
    def test_de_read_delivery_exhaustion_fails_pe_reverse_destination_once(
        self,
        cancelled,
        scheduler_factory,
        task04_seams,
    ):
        # Given
        delivery_error = PathDecisionDeliveryError("delivery exhausted")
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="de_read_path_policy")
        policy.choose.return_value = PathKind.DE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request(
            "prefill-de-read-delivery-failure",
            _remote_decode_params(),
        )
        scheduler.get_num_new_matched_tokens(request, 0)
        _bind_prefill(scheduler, request)
        binding_metadata = scheduler.build_connector_meta(MagicMock(name="binding_scheduler_output"))
        assert len(binding_metadata.reverse_receive_bindings) == 1
        assert binding_metadata.control_failures == []

        # When
        if cancelled:
            delivery_future.cancel()
        else:
            delivery_future.set_exception(delivery_error)
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        # Then
        expected_failure = DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(10, 11),
            reason=DualPathControlFailureReason.ACTIVATION_FAILED,
        )
        assert metadata.reverse_receive_bindings == []
        assert metadata.control_failures == [expected_failure]
        assert scheduler._prefill_control_failures == {}
        assert request.request_id in scheduler._prefill_invalid_request_ids
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            PathAbortNotice(
                request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0),
                reason=PathAbortReason.DELIVERY_EXHAUSTED,
            ),
        )

        worker = _make_prefill_worker()
        worker.start_load_kv(binding_metadata)
        worker.start_load_kv(metadata)
        assert worker.get_finished(set(), metadata) == (set(), {request.request_id})
        assert worker.get_block_ids_with_load_errors() == {10, 11}
        assert worker.get_finished(set(), metadata) == (set(), set())

        repeated_metadata = scheduler.build_connector_meta(MagicMock(name="repeated_scheduler_output"))
        assert getattr(repeated_metadata, "control_failures", []) == []
        scheduler.request_finished(request, [10, 11, 12, 13])
        assert worker.get_finished({request.request_id}, metadata) == (set(), set())
        assert worker._control_failed_recving == set()
        # §5 removal rule: the binding's terminal was never consumed, so the
        # attempt-keyed binding is retained past the request finish.
        installed_binding = binding_metadata.reverse_receive_bindings[0]
        assert set(worker._reverse_receive_bindings) == {
            ReverseAttemptKey(installed_binding.request_key, installed_binding.reverse_attempt_id)
        }
        assert scheduler._prefill_delivery_futures == {}
        assert scheduler._prefill_pending_reverse_receive_bindings == {}
        assert scheduler._prefill_control_failures == {}

    def test_de_read_delivery_failure_before_first_build_emits_control_only(
        self,
        scheduler_factory,
        task04_seams,
    ):
        # Given
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="de_read_path_policy")
        policy.choose.return_value = PathKind.DE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request(
            "prefill-de-read-pre-drain-failure",
            _remote_decode_params(),
        )
        scheduler.get_num_new_matched_tokens(request, 0)
        _bind_prefill(scheduler, request)
        assert request.request_id in scheduler._prefill_pending_reverse_receive_bindings
        delivery_future.set_exception(PathDecisionDeliveryError("delivery exhausted"))

        # When
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        # Then
        assert metadata.reverse_receive_bindings == []
        assert metadata.control_failures == [
            DualPathControlFailureMetadata(
                request_id=request.request_id,
                invalid_block_ids=(10, 11),
                reason=DualPathControlFailureReason.ACTIVATION_FAILED,
            )
        ]
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            PathAbortNotice(
                request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0),
                reason=PathAbortReason.DELIVERY_EXHAUSTED,
            ),
        )
        worker = _make_prefill_worker()
        worker.start_load_kv(metadata)
        assert worker.request_map == {}
        assert worker._reverse_receive_bindings == {}
        assert worker.get_finished(set(), metadata) == (set(), {request.request_id})

    def test_de_read_delivery_failure_after_binding_drain_without_forward_plan_uses_delivery_context(
        self,
        scheduler_factory,
        task04_seams,
    ):
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="de_read_path_policy")
        policy.choose.return_value = PathKind.DE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        request = _make_prefill_request("prefill-de-read-drained", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(request, 0)
        _bind_prefill(scheduler, request, local_block_ids=(10, 11))
        assert scheduler._prefill_forward_plans == {}

        binding_metadata = scheduler.build_connector_meta(MagicMock(name="binding_scheduler_output"))
        assert len(binding_metadata.reverse_receive_bindings) == 1
        assert scheduler._prefill_pending_reverse_receive_bindings == {}
        delivery_future.set_exception(PathDecisionDeliveryError("delivery exhausted"))

        metadata = scheduler.build_connector_meta(MagicMock(name="failure_scheduler_output"))

        assert metadata.control_failures == [
            DualPathControlFailureMetadata(
                request_id=request.request_id,
                invalid_block_ids=(10, 11),
                reason=DualPathControlFailureReason.ACTIVATION_FAILED,
            )
        ]
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            PathAbortNotice(
                request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0),
                reason=PathAbortReason.DELIVERY_EXHAUSTED,
            ),
        )

    def test_released_failed_delivery_aborts_without_staging_or_invalidating_reused_request(
        self,
        scheduler_factory,
        task04_seams,
    ):
        delivery_future: Future[None] = Future()
        new_delivery_future = _completed_future()
        task04_seams.prefill_coordinator.submit.side_effect = [delivery_future, new_delivery_future]
        task04_seams.prefill_coordinator.submit_abort.return_value = _completed_future()
        policy = MagicMock(name="de_read_path_policy")
        policy.choose.side_effect = [PathKind.DE_READ, PathKind.PE_READ]
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        old_request = _make_prefill_request("prefill-de-read-finished", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(old_request, 0)
        _bind_prefill(scheduler, old_request, local_block_ids=(10, 11))
        scheduler.build_connector_meta(MagicMock(name="binding_scheduler_output"))
        old_request.status = RequestStatus.FINISHED_STOPPED
        scheduler.request_finished(old_request, [10, 11])
        assert scheduler._prefill_request_keys == {}
        assert scheduler._prefill_path_results == {}

        delivery_future.set_exception(PathDecisionDeliveryError("delivery exhausted"))
        new_request = _make_prefill_request(old_request.request_id, _remote_decode_params())
        assert scheduler.get_num_new_matched_tokens(new_request, 0) == (0, False)
        _bind_prefill(scheduler, new_request, local_block_ids=(30, 31, 32, 33))
        metadata = scheduler.build_connector_meta(MagicMock(name="new_scheduler_output"))

        assert metadata.control_failures == []
        task04_seams.prefill_coordinator.submit_abort.assert_called_once_with(
            _CONTROL_ENDPOINT,
            PathAbortNotice(
                request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, "decode-request-7", 0),
                reason=PathAbortReason.DELIVERY_EXHAUSTED,
            ),
        )
        assert new_request.request_id not in scheduler._prefill_invalid_request_ids
        assert scheduler._prefill_admission_owners[new_request.request_id] is new_request
        assert scheduler._prefill_path_results[new_request.request_id].path is PathKind.PE_READ
        assert policy.choose.call_count == 2
        assert scheduler._prefill_delivery_futures == {new_request.request_id: new_delivery_future}

    def test_active_failed_delivery_fences_reuse_until_failure_is_published(
        self,
        scheduler_factory,
        task04_seams,
    ):
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        task04_seams.prefill_coordinator.submit_abort.return_value = _completed_future()
        policy = MagicMock(name="de_read_path_policy")
        policy.choose.return_value = PathKind.DE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        old_request = _make_prefill_request("prefill-active-failure", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(old_request, 0)
        _bind_prefill(scheduler, old_request, local_block_ids=(10, 11))
        scheduler.build_connector_meta(MagicMock(name="binding_scheduler_output"))

        delivery_future.set_exception(PathDecisionDeliveryError("delivery exhausted"))
        scheduler._reconcile_prefill_deliveries()

        assert old_request.request_id not in scheduler._prefill_delivery_futures
        assert old_request.request_id in scheduler._prefill_delivery_records
        new_request = _make_prefill_request(old_request.request_id, _remote_decode_params())
        with pytest.raises(RuntimeError, match="request-id reuse"):
            scheduler.get_num_new_matched_tokens(new_request, 0)

        metadata = scheduler.build_connector_meta(MagicMock(name="failure_scheduler_output"))
        assert metadata.control_failures == [
            DualPathControlFailureMetadata(
                request_id=old_request.request_id,
                invalid_block_ids=(10, 11),
                reason=DualPathControlFailureReason.ACTIVATION_FAILED,
            )
        ]
        assert old_request.request_id not in scheduler._prefill_delivery_records
        repeated_metadata = scheduler.build_connector_meta(MagicMock(name="repeated_scheduler_output"))
        assert repeated_metadata.control_failures == []

        old_request.status = RequestStatus.FINISHED_STOPPED
        scheduler.request_finished(old_request, [10, 11])
        assert scheduler.get_num_new_matched_tokens(new_request, 0) == (32, True)

    def test_request_id_reuse_is_fenced_until_prior_delivery_resolves(
        self,
        scheduler_factory,
        task04_seams,
    ):
        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        scheduler = scheduler_factory(role="prefill", path_policy=policy)
        old_request = _make_prefill_request("prefill-reused", _remote_decode_params())
        scheduler.get_num_new_matched_tokens(old_request, 0)
        _bind_prefill(scheduler, old_request)
        old_request.status = RequestStatus.FINISHED_STOPPED
        scheduler.request_finished(old_request, [10, 11, 12, 13])
        new_request = _make_prefill_request(
            old_request.request_id,
            _remote_decode_params(
                dual_path=_prefill_decision_payload(decode_request_id="decode-request-new"),
            ),
        )

        with pytest.raises(RuntimeError, match="request-id reuse"):
            scheduler.get_num_new_matched_tokens(new_request, 0)

        assert scheduler._prefill_request_keys == {}
        assert policy.choose.call_count == 1
        delivery_future.set_result(None)
        scheduler.build_connector_meta(MagicMock(name="idle_scheduler_output"))

        assert scheduler.get_num_new_matched_tokens(new_request, 0) == (0, False)
        assert scheduler._prefill_request_keys[new_request.request_id] == DualPathRequestKey(
            _DECODE_INSTANCE_ID,
            "decode-request-new",
            0,
        )
        assert policy.choose.call_count == 2

    def test_concurrent_requests_stay_isolated_across_outcomes(
        self,
        decode_scheduler,
        scheduler_factory,
        task04_seams,
    ):
        # Given
        decode_scheduler.executor.submit.side_effect = [
            _completed_future(),
            _completed_future(),
            _completed_future(),
        ]
        committed_request, _ = _admit_request(
            decode_scheduler,
            _make_request(request_id="request-committed"),
            (41, 42, 43, 44),
        )
        aborted_request, _ = _admit_request(
            decode_scheduler,
            _make_request(request_id="request-aborted"),
            (51, 52, 53, 54),
        )
        cancelled_request, _ = _admit_request(
            decode_scheduler,
            _make_request(request_id="request-cancelled"),
            (61, 62, 63, 64),
        )
        committed_state = decode_scheduler._decode_decision_states[committed_request.request_id]
        aborted_state = decode_scheduler._decode_decision_states[aborted_request.request_id]
        cancelled_state = decode_scheduler._decode_decision_states[cancelled_request.request_id]

        completed_delivery_future = _completed_future()
        inflight_delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.side_effect = [
            completed_delivery_future,
            inflight_delivery_future,
        ]
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        prefill_scheduler = scheduler_factory(role="prefill", path_policy=policy)
        committed_prefill_request = _make_prefill_request(
            "prefill-committed",
            _remote_decode_params(
                dual_path=_prefill_decision_payload(
                    decode_request_id=committed_request.request_id,
                    admission_id=committed_state.request_key.admission_id,
                ),
            ),
        )
        inflight_prefill_request = _make_prefill_request(
            "prefill-inflight",
            _remote_decode_params(
                dual_path=_prefill_decision_payload(
                    decode_request_id=aborted_request.request_id,
                    admission_id=aborted_state.request_key.admission_id,
                ),
            ),
        )
        prefill_scheduler.get_num_new_matched_tokens(committed_prefill_request, 0)
        prefill_scheduler.get_num_new_matched_tokens(inflight_prefill_request, 0)
        _bind_prefill(prefill_scheduler, committed_prefill_request)
        _bind_prefill(prefill_scheduler, inflight_prefill_request)
        prefill_scheduler.request_finished(committed_prefill_request, [4, 5])

        task04_seams.decode_coordinator.take_received_decisions.return_value = [
            _decision(
                _result(
                    committed_request.request_id,
                    admission_id=committed_state.request_key.admission_id,
                )
            )
        ]
        commit_metadata = decode_scheduler.build_connector_meta(MagicMock(name="commit_scheduler_output"))
        reverse_send_job_id = commit_metadata.reverse_plans[0].reverse_send_job_id
        decode_scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completed_jobs={reverse_send_job_id: 2}))
        )

        # When
        decode_scheduler.request_finished(cancelled_request, [61, 62, 63, 64])
        task04_seams.decode_coordinator.take_received_decisions.return_value = []
        task04_seams.decode_coordinator.take_received_aborts.return_value = [
            PathAbortNotice(
                request_key=aborted_state.request_key,
                reason=PathAbortReason.REQUEST_ABORTED,
            )
        ]
        failure_metadata = decode_scheduler.build_connector_meta(MagicMock(name="abort_scheduler_output"))

        # Then
        assert committed_state.status is scheduler_module._DecodeDecisionStatus.COMMITTED
        assert aborted_state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
        assert cancelled_request.request_id not in decode_scheduler._decode_decision_states
        assert set(decode_scheduler._decode_decision_states) == {
            committed_request.request_id,
            aborted_request.request_id,
        }
        assert set(decode_scheduler._decode_kv_snapshots) == {
            committed_request.request_id,
            aborted_request.request_id,
        }
        assert failure_metadata.control_failures == [
            DualPathControlFailureMetadata(
                request_id=aborted_request.request_id,
                invalid_block_ids=(52, 53, 54),
                reason=DualPathControlFailureReason.PEER_ABORT,
            )
        ]
        assert task04_seams.decode_coordinator.register_pending.call_args_list == [
            call(committed_state.request_key),
            call(aborted_state.request_key),
            call(cancelled_state.request_key),
        ]
        assert task04_seams.decode_coordinator.unregister.call_args_list == [
            call(cancelled_state.request_key),
            call(aborted_state.request_key),
        ]
        assert [
            submitted.args[1].result.request_key for submitted in task04_seams.prefill_coordinator.submit.call_args_list
        ] == [committed_state.request_key, aborted_state.request_key]
        assert completed_delivery_future is not inflight_delivery_future
        assert prefill_scheduler._prefill_request_keys == {
            inflight_prefill_request.request_id: aborted_state.request_key,
        }
        assert prefill_scheduler._prefill_delivery_futures == {
            inflight_prefill_request.request_id: inflight_delivery_future,
        }
        assert not inflight_delivery_future.done()
        assert prefill_scheduler._path_decider is not None
        assert set(prefill_scheduler._path_decider._decision_records) == {aborted_state.request_key}
        assert policy.choose.call_count == 2

        decode_scheduler.request_finished(committed_request, [41, 42, 43, 44])
        decode_scheduler.request_finished(aborted_request, [51, 52, 53, 54])
        assert decode_scheduler._decode_decision_states == {}
        assert decode_scheduler._decode_kv_snapshots == {}
        assert task04_seams.decode_coordinator.unregister.call_args_list == [
            call(cancelled_state.request_key),
            call(aborted_state.request_key),
            call(committed_state.request_key),
            call(aborted_state.request_key),
        ]
        assert decode_scheduler.executor.submit.call_count == 3

    def test_shutdown_is_idempotent_and_leaves_no_owned_state(self, scheduler_factory, task04_seams):
        # Given
        decode_scheduler = scheduler_factory(role="decode")
        decode_request, _ = _admit(decode_scheduler)

        delivery_future: Future[None] = Future()
        task04_seams.prefill_coordinator.submit.return_value = delivery_future
        policy = MagicMock(name="path_policy")
        policy.choose.return_value = PathKind.PE_READ
        prefill_scheduler = scheduler_factory(role="prefill", path_policy=policy)
        prefill_request = _make_prefill_request("prefill-shutdown", _remote_decode_params())
        prefill_scheduler.get_num_new_matched_tokens(prefill_request, 0)
        _bind_prefill(prefill_scheduler, prefill_request)
        invalid_request = _make_prefill_request(
            "prefill-invalid-shutdown",
            _remote_decode_params(),
        )
        prefill_scheduler._prefill_invalid_request_ids.add(invalid_request.request_id)
        worker = _control_only_worker()
        worker._control_failed_recving.add("request-control-failure")
        events = []

        def record_decode_close():
            events.append(
                (
                    "decode-coordinator",
                    decode_scheduler._accepting_decode_admission,
                    getattr(decode_scheduler, "_accepting_prefill_decisions", None),
                    decode_request.request_id in decode_scheduler._decode_decision_states,
                )
            )

        def record_prefill_close():
            events.append(
                (
                    "prefill-coordinator",
                    prefill_scheduler._accepting_decode_admission,
                    getattr(prefill_scheduler, "_accepting_prefill_decisions", None),
                    prefill_request.request_id in prefill_scheduler._prefill_request_keys,
                    prefill_request.request_id in prefill_scheduler._prefill_delivery_futures,
                )
            )
            delivery_future.cancel()

        def record_decode_adapter_close():
            events.append(
                (
                    "decode-adapter",
                    task04_seams.decode_coordinator.close.call_count,
                    not decode_scheduler._decode_decision_states,
                )
            )

        task04_seams.decode_coordinator.close.side_effect = record_decode_close
        task04_seams.prefill_coordinator.close.side_effect = record_prefill_close
        decode_scheduler._kvpool_adapter.close.side_effect = record_decode_adapter_close

        # When
        decode_scheduler.shutdown()
        prefill_scheduler.shutdown()
        worker.shutdown()
        decode_scheduler.shutdown()
        prefill_scheduler.shutdown()
        worker.shutdown()

        post_shutdown_request = _make_prefill_request("prefill-after-shutdown", _remote_decode_params())
        post_shutdown_result = prefill_scheduler.get_num_new_matched_tokens(post_shutdown_request, 0)

        # Then
        assert events[:3] == [
            ("decode-coordinator", False, False, True),
            ("decode-adapter", 1, True),
            ("prefill-coordinator", False, False, True, True),
        ]
        assert post_shutdown_result == (0, False)
        policy.choose.assert_called_once()
        task04_seams.prefill_coordinator.submit.assert_called_once()
        assert delivery_future.cancelled()
        assert decode_scheduler._lookup_results == {}
        assert decode_scheduler._decode_kv_snapshots == {}
        assert decode_scheduler._decode_decision_states == {}
        assert prefill_scheduler._prefill_request_keys == {}
        assert prefill_scheduler._prefill_local_tokens == {}
        assert prefill_scheduler._prefill_delivery_futures == {}
        assert prefill_scheduler._prefill_invalid_request_ids == set()
        assert prefill_scheduler._path_decider is not None
        assert prefill_scheduler._path_decider._decision_records == {}
        assert worker._control_failed_recving == set()
        assert task04_seams.decode_coordinator.close.call_count == 2
        assert task04_seams.prefill_coordinator.close.call_count == 2
        assert decode_scheduler._kvpool_adapter.close.call_count == 2
        worker._kvpool_worker_adapter.close.assert_not_called()

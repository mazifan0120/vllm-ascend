from __future__ import annotations

import json
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionRequest,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DUAL_PATH_PROTOCOL_VERSION,
    DecodeControlEndpoint,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
_TIMEOUT_ENV = "VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT"
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
    config.parallel_config.data_parallel_rank = 2
    config.parallel_config.data_parallel_size = 4
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 3
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


def _make_request(params: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="request-local-7",
        num_tokens=49,
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
        patch(f"{_CONNECTOR_NS}.KVPoolAdapter") as adapter_cls,
        patch(f"{_CONNECTOR_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_CONNECTOR_NS}.get_ip", return_value="192.0.2.44"),
    ):
        decode_coordinator = MagicMock(name="decode_coordinator")
        decode_coordinator.decode_engine_instance_id = _DECODE_INSTANCE_ID
        decode_coordinator.decode_control_endpoint = _CONTROL_ENDPOINT
        prefill_coordinator = MagicMock(name="prefill_coordinator")
        coordinator_cls.for_decode.return_value = decode_coordinator
        coordinator_cls.for_prefill.return_value = prefill_coordinator
        yield SimpleNamespace(
            adapter_cls=adapter_cls,
            coordinator_cls=coordinator_cls,
            decode_coordinator=decode_coordinator,
            prefill_coordinator=prefill_coordinator,
        )


@pytest.fixture()
def scheduler_factory(task04_seams):
    schedulers = []

    def make(
        *,
        role: str = "decode",
        config: MagicMock | None = None,
        kv_cache_config: SimpleNamespace | None = None,
    ):
        scheduler = connector_module.DualPathConnectorScheduler(
            config or _make_vllm_config(kv_role="kv_consumer" if role == "decode" else "kv_producer"),
            kv_cache_config or _make_kv_cache_config(),
            "decode-engine",
            DualPathConfig(role=role, dual_path_control_port=7100 if role == "decode" else None),
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
def decode_scheduler(monkeypatch, scheduler_factory):
    monkeypatch.delenv(_TIMEOUT_ENV, raising=False)
    return scheduler_factory()


@pytest.fixture()
def proxy_echo():
    def echo(message):
        return json.loads(json.dumps({"kv_transfer_params": message}))["kv_transfer_params"]

    return echo


def _admit(scheduler, params: dict | None = None):
    request = _make_request(params)
    load_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
    scheduler._kvpool_adapter.lookup.return_value = load_spec
    assert scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([41, 42, 43, 44],)
    scheduler.update_state_after_alloc(request, blocks, 32)
    return request, blocks


def _expected_dual_path_payload() -> dict:
    return {
        "protocol_version": DUAL_PATH_PROTOCOL_VERSION,
        "decision_request": {
            "request_key": {
                "decode_engine_instance_id": _DECODE_INSTANCE_ID,
                "decode_request_id": "request-local-7",
            },
            "target_tokens": 48,
            "decode_local_tokens": 16,
            "decode_store_tokens": 32,
        },
        "decode_control_endpoint": {"host": "192.0.2.44", "port": 24001},
    }


class TestDecisionTimeoutEnv:
    def test_default_is_60_seconds(self, monkeypatch, scheduler_factory):
        monkeypatch.delenv(_TIMEOUT_ENV, raising=False)
        scheduler = scheduler_factory()
        assert scheduler._decision_timeout_seconds == 60

    def test_valid_override_is_read_once_and_does_not_rearm(self, monkeypatch, scheduler_factory):
        monkeypatch.setenv(_TIMEOUT_ENV, "7")
        scheduler = scheduler_factory()
        monkeypatch.setenv(_TIMEOUT_ENV, "99")
        with patch.object(connector_module.time, "monotonic", return_value=100.0):
            request, _ = _admit(scheduler)
        assert scheduler._decision_timeout_seconds == 7
        assert scheduler._decode_decision_states[request.request_id].deadline == 107.0

    @pytest.mark.parametrize("raw", ["not-an-int", "True", "False", "0", "-3"])
    def test_invalid_string_zero_and_negative_values_fail_fast(self, raw, monkeypatch, scheduler_factory):
        monkeypatch.setenv(_TIMEOUT_ENV, raw)
        with pytest.raises(ValueError):
            scheduler_factory()

    def test_boolean_runtime_value_fails_fast(self, monkeypatch, scheduler_factory):
        monkeypatch.setitem(ascend_envs.env_variables, _TIMEOUT_ENV, lambda: True)
        with pytest.raises(ValueError):
            scheduler_factory()

    def test_prefill_role_never_reads_timeout_env(self, monkeypatch, scheduler_factory):
        reader = MagicMock(side_effect=AssertionError("prefill read decode timeout"))
        monkeypatch.setitem(ascend_envs.env_variables, _TIMEOUT_ENV, reader)
        scheduler = scheduler_factory(role="prefill")
        reader.assert_not_called()
        assert scheduler._decision_timeout_seconds is None


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

    def test_snapshot_builds_exact_key_request_and_metadata(self, decode_scheduler):
        with patch.object(connector_module.time, "monotonic", return_value=10.0):
            request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        expected_key = DualPathRequestKey(_DECODE_INSTANCE_ID, request.request_id)
        assert state.request_key == expected_key
        assert state.decision_request == PathDecisionRequest(expected_key, 48, 16, 32)
        assert state.status is connector_module.DecodeDecisionStatus.PENDING
        assert state.deadline == 70.0
        assert (
            decode_scheduler.executor.submit.call_args.kwargs["message"]["dual_path"] == _expected_dual_path_payload()
        )

    def test_register_pending_precedes_executor_submission(self, decode_scheduler):
        ordered = MagicMock()
        ordered.attach_mock(decode_scheduler._path_decision_coordinator.register_pending, "register_pending")
        ordered.attach_mock(decode_scheduler.executor.submit, "submit")
        _admit(decode_scheduler)
        assert ordered.method_calls[:2] == [
            call.register_pending(DualPathRequestKey(_DECODE_INSTANCE_ID, "request-local-7")),
            call.submit(
                decode_scheduler._access_metaserver,
                url="http://proxy.example/v1/kv",
                message=decode_scheduler.executor.submit.call_args.kwargs["message"],
            ),
        ]

    def test_outgoing_message_preserves_mooncake_fields_and_hybrid_trim(self, decode_scheduler):
        decode_scheduler.need_truncate = True
        with patch.object(
            connector_module,
            "get_external_request_id",
            return_value="external-request-7",
        ) as external_id:
            _admit(decode_scheduler)
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
            "remote_port": 5004,
            "remote_tp_size": 2,
            "remote_pcp_size": 1,
            "remote_dcp_size": 3,
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

    def test_identical_duplicate_does_not_register_notify_or_reset_deadline(self, decode_scheduler):
        request, blocks = _admit(decode_scheduler)
        first_state = decode_scheduler._decode_decision_states[request.request_id]
        decode_scheduler._path_decision_coordinator.register_pending.reset_mock()
        decode_scheduler.executor.submit.reset_mock()
        decode_scheduler.update_state_after_alloc(request, blocks, 32)
        assert decode_scheduler._decode_decision_states[request.request_id] is first_state
        assert decode_scheduler._decode_decision_states[request.request_id].deadline == first_state.deadline
        decode_scheduler._path_decision_coordinator.register_pending.assert_not_called()
        decode_scheduler.executor.submit.assert_not_called()

    def test_http_success_remains_pending(self, decode_scheduler):
        future = _completed_future()
        decode_scheduler.executor.submit.return_value = future
        request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is connector_module.DecodeDecisionStatus.PENDING
        assert state.proxy_future is future

    def test_synchronous_executor_failure_remains_pending_without_retry(self, decode_scheduler):
        decode_scheduler.executor.submit.side_effect = RuntimeError("executor closed")
        with patch.object(connector_module.logger, "error") as log_error:
            request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is connector_module.DecodeDecisionStatus.PENDING
        assert state.proxy_future is None
        decode_scheduler.executor.submit.assert_called_once()
        log_error.assert_called_once()

    def test_async_future_failure_is_logged_and_remains_pending(self, decode_scheduler):
        future = _completed_future(RuntimeError("proxy unavailable"))
        decode_scheduler.executor.submit.return_value = future
        with patch.object(connector_module.logger, "error") as log_error:
            request, _ = _admit(decode_scheduler)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is connector_module.DecodeDecisionStatus.PENDING
        assert state.proxy_future is future
        decode_scheduler.executor.submit.assert_called_once()
        log_error.assert_called_once()

    def test_virtual_request_skips_http_and_keeps_deadline(self, decode_scheduler):
        params = {
            "do_remote_prefill": True,
            "do_virtual": True,
            "metaserver": "http://proxy.example/v1/kv",
        }
        with patch.object(connector_module.time, "monotonic", return_value=30.0):
            request, _ = _admit(decode_scheduler, params)
        state = decode_scheduler._decode_decision_states[request.request_id]
        assert state.status is connector_module.DecodeDecisionStatus.PENDING
        assert state.deadline == 90.0
        assert state.proxy_future is None
        decode_scheduler.executor.submit.assert_not_called()

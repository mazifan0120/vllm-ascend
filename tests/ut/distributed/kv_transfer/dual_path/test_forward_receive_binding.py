from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    ForwardReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    Path,
    PathDecisionResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    LoadSpec,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
_DECODE_INSTANCE_ID = "decode-engine:2:boot-7"
_CONTROL_ENDPOINT = DecodeControlEndpoint(host="192.0.2.44", port=24001)


def _make_vllm_config() -> MagicMock:
    config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.is_kv_consumer = True
    config.kv_transfer_config.engine_id = "decode-engine"
    config.kv_transfer_config.kv_port = 5000
    config.kv_transfer_config.kv_load_failure_policy = "fail"
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


def _make_kv_cache_config() -> SimpleNamespace:
    spec = MagicMock(name="kv_cache_spec")
    spec.block_size = 16
    group = MagicMock(name="kv_cache_group")
    group.kv_cache_spec = spec
    group.layer_names = ["layer.0"]
    return SimpleNamespace(kv_cache_groups=[group], kv_cache_tensors=[], num_blocks=64)


@pytest.fixture()
def scheduler_factory(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT", raising=False)
    schedulers = []
    with (
        patch(f"{_CONNECTOR_NS}.KVPoolAdapter"),
        patch(f"{_CONNECTOR_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_CONNECTOR_NS}.get_ip", return_value="192.0.2.44"),
    ):

        def make(*, need_truncate: bool = False):
            coordinator = MagicMock(name="decode_coordinator")
            coordinator.decode_engine_instance_id = _DECODE_INSTANCE_ID
            coordinator.decode_control_endpoint = _CONTROL_ENDPOINT
            coordinator.take_received_results.return_value = []
            coordinator_cls.for_decode.return_value = coordinator

            scheduler = connector_module.DualPathConnectorScheduler(
                _make_vllm_config(),
                _make_kv_cache_config(),
                "decode-engine",
                DualPathConfig(role="decode", dual_path_control_port=7100),
            )
            scheduler.executor.shutdown(wait=False)
            scheduler.metaserver_client.close()
            scheduler.executor = MagicMock(name="decode_executor")
            scheduler.side_channel_host = "198.51.100.20"
            scheduler.need_truncate = need_truncate
            schedulers.append(scheduler)
            return scheduler, coordinator

        yield make

    for scheduler in schedulers:
        scheduler.shutdown()


def _admit_request(scheduler):
    request = SimpleNamespace(
        request_id="request-local-7",
        num_tokens=49,
        prompt_token_ids=list(range(49)),
        kv_transfer_params={
            "do_remote_prefill": True,
            "metaserver": "http://proxy.example/v1/kv",
        },
    )
    scheduler._kvpool_adapter.lookup.return_value = LoadSpec(
        vllm_cached_tokens=16,
        kvpool_cached_tokens=32,
        can_load=False,
    )
    external_tokens = 32 if scheduler.need_truncate else 33
    assert scheduler.get_num_new_matched_tokens(request, 16) == (external_tokens, True)
    blocks = MagicMock(name="decode_blocks")
    blocks.get_block_ids.return_value = ([41, 42, 43, 44],)
    scheduler.update_state_after_alloc(request, blocks, external_tokens)
    return (
        request,
        scheduler._decode_kv_snapshots[request.request_id],
        scheduler._decode_decision_states[request.request_id],
    )


def _destination_from_snapshot(scheduler, snapshot) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(group)
        for group in scheduler._trim_hybrid_remote_block_ids(
            snapshot.final_block_ids,
            snapshot.target_tokens + 1,
        )
    )


def test_commit_pe_read_emits_exactly_one_control_only_binding_with_advertised_table(
    scheduler_factory,
):
    # Given
    scheduler, coordinator = scheduler_factory()
    request, snapshot, state = _admit_request(scheduler)
    result = PathDecisionResult(request_key=state.request_key, path=Path.PE_READ)
    coordinator.take_received_results.return_value = [result]
    advertised_destination = tuple(
        tuple(group) for group in scheduler.executor.submit.call_args.kwargs["message"]["remote_block_ids"]
    )
    derived_destination = _destination_from_snapshot(scheduler, snapshot)
    assert derived_destination == ((41, 42, 43, 44),)
    assert advertised_destination == derived_destination
    receive_queue_before = scheduler._reqs_need_recv.copy()

    # When
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    # Then
    assert isinstance(metadata, DualPathConnectorMetadata)
    assert metadata.forward_receive_bindings == [
        ForwardReceiveBinding(
            request_key=state.request_key,
            wire_request_id=get_external_request_id(request.request_id),
            decode_request_id=request.request_id,
            destination_block_ids=derived_destination,
            token_start=snapshot.local_tokens,
            token_end=snapshot.transfer_tokens,
        )
    ]
    assert request.request_id not in metadata.requests
    assert scheduler._reqs_need_recv == receive_queue_before == {}
    assert metadata.decision_timeouts == []
    assert state.status is connector_module.DecodeDecisionStatus.COMMITTED


def test_binding_destination_table_includes_hybrid_trimming(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory(need_truncate=True)
    _, snapshot, state = _admit_request(scheduler)
    coordinator.take_received_results.return_value = [
        PathDecisionResult(request_key=state.request_key, path=Path.PE_READ)
    ]
    advertised_destination = tuple(
        tuple(group) for group in scheduler.executor.submit.call_args.kwargs["message"]["remote_block_ids"]
    )

    # When
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    # Then
    derived_destination = _destination_from_snapshot(scheduler, snapshot)
    assert derived_destination == ((41, 42, 43),)
    assert advertised_destination == derived_destination
    assert metadata.forward_receive_bindings[0].destination_block_ids == derived_destination


def test_duplicate_pe_read_result_does_not_emit_second_binding(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory()
    _, _, state = _admit_request(scheduler)
    result = PathDecisionResult(request_key=state.request_key, path=Path.PE_READ)
    coordinator.take_received_results.return_value = [result]
    first_metadata = scheduler.build_connector_meta(MagicMock(name="first_scheduler_output"))
    coordinator.take_received_results.return_value = [result]

    # When
    second_metadata = scheduler.build_connector_meta(MagicMock(name="second_scheduler_output"))

    # Then
    assert len(first_metadata.forward_receive_bindings) == 1
    assert second_metadata.forward_receive_bindings == []
    assert state.status is connector_module.DecodeDecisionStatus.COMMITTED


def test_committed_de_read_result_emits_no_binding(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory()
    _, _, state = _admit_request(scheduler)
    result = PathDecisionResult(request_key=state.request_key, path=Path.DE_READ)
    coordinator.take_received_results.return_value = [result]

    # When
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    # Then
    assert metadata.forward_receive_bindings == []
    assert state.status is connector_module.DecodeDecisionStatus.COMMITTED
    assert state.result is result

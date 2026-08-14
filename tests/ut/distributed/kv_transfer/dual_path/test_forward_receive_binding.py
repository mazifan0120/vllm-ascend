from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path.conftest import init_dual_path_worker_state
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    DualPathControlFailureMetadata,
    DualPathControlFailureReason,
    ForwardReceiveBinding,
    ReversePlan,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionResult,
    PathKind,
    ReverseAttemptKey,
    reverse_wire_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
    PathDecision,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
)

_CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
_SCHEDULER_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler"
_DECODE_INSTANCE_ID = "decode-engine:2:boot-7"
_CONTROL_ENDPOINT = DecodeControlEndpoint(host="192.0.2.44", port=24001)


def _decision(result: PathDecisionResult) -> PathDecision:
    return PathDecision(
        result=result,
        reverse_plan=None,
    )


def _de_read_decision(state, snapshot, **plan_overrides) -> PathDecision:
    destination_block_ids = ((71, 72, 73, 74),)
    token_start = plan_overrides.pop("token_start", 16)
    token_end = plan_overrides.pop("token_end", snapshot.store_tokens)
    plan = ReversePlan(
        request_key=state.request_key,
        wire_request_id=reverse_wire_id(ReverseAttemptKey(state.request_key, 0)),
        token_start=token_start,
        token_end=token_end,
        source_block_ids=tuple(tuple(group) for group in snapshot.final_block_ids),
        destination_block_ids=destination_block_ids,
        remote_engine_id="prefill-engine",
        remote_host="198.51.100.10",
        remote_port=6000,
        remote_block_sizes=(16,),
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        reverse_attempt_id=0,
        prefill_local_tokens=token_start,
        reverse_send_job_id=None,
    )
    if plan_overrides:
        plan = dataclasses.replace(plan, **plan_overrides)
    return PathDecision(
        result=PathDecisionResult(
            request_key=state.request_key,
            path=PathKind.DE_READ,
            reverse_attempt_id=0,
            prefill_local_tokens=token_start,
        ),
        reverse_plan=plan,
    )


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
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.world_size = 2
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
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
        patch(f"{_SCHEDULER_NS}.KVPoolSchedulerAdapter"),
        patch(f"{_SCHEDULER_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_SCHEDULER_NS}.get_ip", return_value="192.0.2.44"),
    ):

        def make(*, need_truncate: bool = False):
            coordinator = MagicMock(name="decode_coordinator")
            coordinator.decode_engine_instance_id = _DECODE_INSTANCE_ID
            coordinator.decode_control_endpoint = _CONTROL_ENDPOINT
            coordinator.take_received_decisions.return_value = []
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
            scheduler._kvpool_adapter.build_connector_meta.return_value = AscendConnectorMetadata(set(), set())
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


def _destination_from_snapshot(scheduler, snapshot, decision_request) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(group)
        for group in scheduler._trim_hybrid_remote_block_ids(
            snapshot.final_block_ids,
            decision_request.target_tokens + 1,
        )
    )


def _make_binding(
    decode_request_id: str = "decode-request-00000001",
    destination_block_ids: tuple[tuple[int, ...], ...] = ((101, 102, 103, 104),),
) -> ForwardReceiveBinding:
    return ForwardReceiveBinding(
        request_key=DualPathRequestKey(_DECODE_INSTANCE_ID, decode_request_id),
        path=PathKind.PE_READ,
        wire_request_id=get_external_request_id(decode_request_id),
        decode_request_id=decode_request_id,
        destination_block_ids=destination_block_ids,
        token_start=16,
        token_end=49,
    )


def _make_worker():
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


def _binding_metadata(binding: ForwardReceiveBinding) -> DualPathConnectorMetadata:
    metadata = DualPathConnectorMetadata()
    metadata.forward_receive_bindings.append(binding)
    return metadata


def test_commit_pe_read_emits_exactly_one_control_only_binding_with_advertised_table(
    scheduler_factory,
):
    # Given
    scheduler, coordinator = scheduler_factory()
    request, snapshot, state = _admit_request(scheduler)
    result = PathDecisionResult(request_key=state.request_key, path=PathKind.PE_READ)
    coordinator.take_received_decisions.return_value = [_decision(result)]
    advertised_destination = tuple(
        tuple(group) for group in scheduler.executor.submit.call_args.kwargs["message"]["remote_block_ids"]
    )
    derived_destination = _destination_from_snapshot(scheduler, snapshot, state.decision_request)
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
            path=PathKind.PE_READ,
            wire_request_id=get_external_request_id(request.request_id),
            decode_request_id=request.request_id,
            destination_block_ids=derived_destination,
            token_start=snapshot.local_tokens,
            token_end=snapshot.transfer_tokens,
        )
    ]
    assert request.request_id not in metadata.requests
    assert scheduler._reqs_need_recv == receive_queue_before == {}
    assert metadata.control_failures == []
    assert state.status is scheduler_module._DecodeDecisionStatus.COMMITTED


def test_binding_destination_table_includes_hybrid_trimming(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory(need_truncate=True)
    _, snapshot, state = _admit_request(scheduler)
    coordinator.take_received_decisions.return_value = [
        _decision(PathDecisionResult(request_key=state.request_key, path=PathKind.PE_READ))
    ]
    advertised_destination = tuple(
        tuple(group) for group in scheduler.executor.submit.call_args.kwargs["message"]["remote_block_ids"]
    )

    # When
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    # Then
    derived_destination = _destination_from_snapshot(scheduler, snapshot, state.decision_request)
    assert derived_destination == ((41, 42, 43),)
    assert advertised_destination == derived_destination
    assert metadata.forward_receive_bindings[0].destination_block_ids == derived_destination


def test_hybrid_timeout_uses_literal_frozen_table_suffix_without_changing_message_trim(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory(need_truncate=True)
    request, snapshot, state = _admit_request(scheduler)
    assert snapshot.final_block_ids == ((41, 42, 43, 44),)
    assert scheduler.executor.submit.call_args.kwargs["message"]["remote_block_ids"] == ([41, 42, 43],)
    coordinator.take_received_decisions.return_value = []

    # When
    with patch.object(scheduler_module.time, "monotonic", return_value=state.deadline):
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    # Then
    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.DECISION_TIMEOUT,
        )
    ]


def test_duplicate_pe_read_result_does_not_emit_second_binding(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory()
    _, _, state = _admit_request(scheduler)
    result = PathDecisionResult(request_key=state.request_key, path=PathKind.PE_READ)
    coordinator.take_received_decisions.return_value = [_decision(result)]
    first_metadata = scheduler.build_connector_meta(MagicMock(name="first_scheduler_output"))
    coordinator.take_received_decisions.return_value = [_decision(result)]

    # When
    second_metadata = scheduler.build_connector_meta(MagicMock(name="second_scheduler_output"))

    # Then
    assert len(first_metadata.forward_receive_bindings) == 1
    assert second_metadata.forward_receive_bindings == []
    assert state.status is scheduler_module._DecodeDecisionStatus.COMMITTED


def test_de_read_decision_requires_non_empty_reverse_plan(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory()
    _, _, state = _admit_request(scheduler)
    coordinator.take_received_decisions.return_value = [
        _decision(
            PathDecisionResult(
                request_key=state.request_key,
                path=PathKind.DE_READ,
                reverse_attempt_id=0,
                prefill_local_tokens=16,
            )
        )
    ]

    # When
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    # Then
    assert metadata.forward_receive_bindings == []
    assert metadata.reverse_plans == []
    assert len(metadata.control_failures) == 1
    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED


def test_partial_commit_metadata_targets_k_de_only(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    request, snapshot, state = _admit_request(scheduler)
    store_metadata = AscendConnectorMetadata(set(), set(), loading_req_ids={request.request_id})
    store_request = SimpleNamespace(req_id=request.request_id, target_token_len=32)
    store_metadata.requests.append(store_request)
    scheduler._kvpool_adapter.build_connector_meta.return_value = store_metadata
    coordinator.take_received_decisions.return_value = [_de_read_decision(state, snapshot)]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert metadata.decode_store_metadata is store_metadata
    assert metadata.decode_store_metadata.requests[0].target_token_len == snapshot.store_tokens == 32
    scheduler._kvpool_adapter.commit_after_alloc.assert_called_once_with(
        request,
        snapshot.allocated_blocks,
        snapshot.store_load_spec,
    )


def test_store_miss_commits_nothing_and_enters_skipped(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    _, snapshot, state = _admit_request(scheduler)
    miss_snapshot = dataclasses.replace(snapshot, store_load_spec=None)
    scheduler._decode_kv_snapshots[state.request_key.decode_request_id] = miss_snapshot
    state.decision_request = dataclasses.replace(state.decision_request, decode_store_tokens=16)
    scheduler._kvpool_adapter.reset_mock()
    decision = _de_read_decision(
        state,
        miss_snapshot,
        token_start=0,
        token_end=16,
    )
    coordinator.take_received_decisions.return_value = [decision]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    scheduler._kvpool_adapter.commit_after_alloc.assert_not_called()
    assert metadata.decode_store_metadata is None
    reverse_send_job_id = scheduler._reverse_send_job_ids[ReverseAttemptKey(state.request_key, 0)]
    assert metadata.reverse_plans == [
        dataclasses.replace(decision.reverse_plan, reverse_send_job_id=reverse_send_job_id)
    ]
    assert state.status is scheduler_module._DecodeDecisionStatus.COMMITTED


def test_commit_failure_rolls_back_and_emits_one_activation_failure(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    _, snapshot, state = _admit_request(scheduler)
    scheduler._kvpool_adapter.commit_after_alloc.side_effect = RuntimeError("delegated commit failed")
    coordinator.take_received_decisions.return_value = [_de_read_decision(state, snapshot)]

    first_metadata = scheduler.build_connector_meta(MagicMock(name="first_scheduler_output"))
    second_metadata = scheduler.build_connector_meta(MagicMock(name="second_scheduler_output"))

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert len(first_metadata.control_failures) == 1
    assert second_metadata.control_failures == []
    assert first_metadata.reverse_plans == []
    assert first_metadata.forward_receive_bindings == []
    scheduler._kvpool_adapter.commit_after_alloc.assert_called_once()


def test_snapshot_plan_mismatch_uses_activation_failure_not_timeout(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    _, snapshot, state = _admit_request(scheduler)
    coordinator.take_received_decisions.return_value = [
        _de_read_decision(state, snapshot, source_block_ids=((51, 52, 53, 54),))
    ]

    with patch.object(scheduler_module.time, "monotonic", return_value=state.deadline + 1):
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED
    assert metadata.control_failures[0].reason is DualPathControlFailureReason.ACTIVATION_FAILED


def test_activation_failure_invalidates_all_external_destinations(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    request, snapshot, state = _admit_request(scheduler)
    coordinator.take_received_decisions.return_value = [
        _decision(
            PathDecisionResult(
                request_key=state.request_key,
                path=PathKind.DE_READ,
                reverse_attempt_id=0,
                prefill_local_tokens=16,
            )
        )
    ]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert metadata.control_failures == [
        DualPathControlFailureMetadata(
            request_id=request.request_id,
            invalid_block_ids=(42, 43, 44),
            reason=DualPathControlFailureReason.ACTIVATION_FAILED,
        )
    ]
    coordinator.unregister.assert_called_once_with(state.request_key)


def test_de_read_activation_emits_plan_binding_store_in_one_lifecycle(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    request, snapshot, state = _admit_request(scheduler)
    decision = _de_read_decision(state, snapshot)
    store_metadata = AscendConnectorMetadata(set(), set(), loading_req_ids={request.request_id})
    scheduler._kvpool_adapter.build_connector_meta.return_value = store_metadata
    coordinator.take_received_decisions.return_value = [decision]

    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    reverse_send_job_id = scheduler._reverse_send_job_ids[ReverseAttemptKey(state.request_key, 0)]
    assert metadata.reverse_plans == [
        dataclasses.replace(decision.reverse_plan, reverse_send_job_id=reverse_send_job_id)
    ]
    assert metadata.forward_receive_bindings == [
        ForwardReceiveBinding(
            request_key=state.request_key,
            path=PathKind.DE_READ,
            wire_request_id=get_external_request_id(request.request_id),
            decode_request_id=request.request_id,
            destination_block_ids=snapshot.final_block_ids,
            token_start=snapshot.store_tokens,
            token_end=snapshot.transfer_tokens,
        )
    ]
    assert metadata.decode_store_metadata is store_metadata
    assert state.status is scheduler_module._DecodeDecisionStatus.COMMITTED


def test_decisions_processed_before_store_metadata_build(scheduler_factory):
    scheduler, coordinator = scheduler_factory()
    _, snapshot, state = _admit_request(scheduler)
    events = []
    scheduler._kvpool_adapter.commit_after_alloc.side_effect = lambda *args: events.append("commit")
    scheduler._kvpool_adapter.build_connector_meta.side_effect = lambda output: (
        events.append("build") or AscendConnectorMetadata(set(), set())
    )
    coordinator.take_received_decisions.return_value = [_de_read_decision(state, snapshot)]

    scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert events == ["commit", "build"]


@pytest.mark.parametrize("failure_stage", ["plan", "commit"])
def test_de_read_committed_only_after_all_steps_succeed(scheduler_factory, failure_stage):
    scheduler, coordinator = scheduler_factory()
    _, snapshot, state = _admit_request(scheduler)
    decision = _de_read_decision(state, snapshot)
    if failure_stage == "plan":
        decision = _decision(
            PathDecisionResult(
                request_key=state.request_key,
                path=PathKind.DE_READ,
                reverse_attempt_id=0,
                prefill_local_tokens=16,
            )
        )
    else:
        scheduler._kvpool_adapter.commit_after_alloc.side_effect = RuntimeError("commit failed")
    coordinator.take_received_decisions.return_value = [decision]

    scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert state.status is scheduler_module._DecodeDecisionStatus.ACTIVATION_FAILED


def test_pe_read_never_calls_decode_kvpool_or_store_commit_surfaces(scheduler_factory):
    # Given
    scheduler, coordinator = scheduler_factory()
    request, _, state = _admit_request(scheduler)
    scheduler._kvpool_adapter.reset_mock()
    coordinator.take_received_decisions.return_value = [
        _decision(PathDecisionResult(request_key=state.request_key, path=PathKind.PE_READ))
    ]
    worker = _make_worker()

    # When
    scheduler_output = MagicMock(name="scheduler_output")
    metadata = scheduler.build_connector_meta(scheduler_output)
    binding = metadata.forward_receive_bindings[0]
    worker.start_load_kv(metadata)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    finished = worker.get_finished(set(), metadata)

    # Then
    assert finished == (set(), {request.request_id})
    scheduler._kvpool_adapter.lookup.assert_not_called()
    scheduler._kvpool_adapter.commit_after_alloc.assert_not_called()
    scheduler._kvpool_adapter.build_connector_meta.assert_called_once_with(scheduler_output)
    assert worker._kvpool_worker_adapter.method_calls == []


def test_binding_install_starts_no_receive_or_load_request():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    metadata = _binding_metadata(binding)

    # When
    worker.start_load_kv(metadata)

    # Then
    assert worker.request_map == {binding.wire_request_id: binding.decode_request_id}
    assert worker._forward_receive_bindings == {binding.decode_request_id: binding}
    assert worker._recving_metadata == {}
    assert worker.kv_recv_layer_thread.method_calls == []
    assert worker._kvpool_worker_adapter.method_calls == []
    assert worker.engine.method_calls == []


def test_identical_binding_install_is_idempotent():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    metadata = _binding_metadata(binding)
    worker.start_load_kv(metadata)

    # When
    worker.start_load_kv(metadata)

    # Then
    assert worker.request_map == {binding.wire_request_id: binding.decode_request_id}
    assert worker._forward_receive_bindings == {binding.decode_request_id: binding}


def test_conflicting_binding_install_preserves_first_binding():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    conflicting = _make_binding(destination_block_ids=((201, 202, 203, 204),))
    worker.start_load_kv(_binding_metadata(binding))

    # When
    with pytest.raises(RuntimeError):
        worker.start_load_kv(_binding_metadata(conflicting))

    # Then
    assert worker.request_map == {binding.wire_request_id: binding.decode_request_id}
    assert worker._forward_receive_bindings == {binding.decode_request_id: binding}


def test_done_after_binding_publishes_finished_recving_only():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    metadata = _binding_metadata(binding)
    worker.start_load_kv(metadata)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}

    # When
    finished = worker.get_finished(set(), metadata)

    # Then
    assert finished == (set(), {binding.decode_request_id})
    assert worker.get_block_ids_with_load_errors() == set()
    assert worker._forward_receive_bindings == {}
    assert worker.request_map == {}
    assert worker._consumed_forward_terminal_wire_ids == {binding.wire_request_id: binding.decode_request_id}


def test_failed_after_binding_publishes_exact_forward_suffix_and_finished_recving_prefix_preserved():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    metadata = _binding_metadata(binding)
    worker.start_load_kv(metadata)
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}

    # When
    finished = worker.get_finished(set(), metadata)
    invalid_block_ids = worker.get_block_ids_with_load_errors()

    # Then
    assert finished == (set(), {binding.decode_request_id})
    assert invalid_block_ids == {102, 103, 104}
    assert 101 not in invalid_block_ids


def test_done_before_binding_is_retained_and_reconciled_after_install():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    empty_metadata = DualPathConnectorMetadata()
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    assert worker.get_finished(set(), empty_metadata) == (set(), set())
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
    binding_metadata = _binding_metadata(binding)

    # When
    worker.start_load_kv(binding_metadata)
    finished = worker.get_finished(set(), binding_metadata)

    # Then
    assert finished == (set(), {binding.decode_request_id})
    assert worker._pending_forward_done_wire_ids == set()


def test_failed_before_binding_is_retained_and_reconciled_after_install():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    empty_metadata = DualPathConnectorMetadata()
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}
    assert worker.get_finished(set(), empty_metadata) == (set(), set())
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
    binding_metadata = _binding_metadata(binding)

    # When
    worker.start_load_kv(binding_metadata)
    finished = worker.get_finished(set(), binding_metadata)

    # Then
    assert finished == (set(), {binding.decode_request_id})
    assert worker.get_block_ids_with_load_errors() == {102, 103, 104}
    assert worker._pending_forward_failed_wire_ids == set()


def test_conflicting_done_and_failed_resolves_as_failed():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    metadata = _binding_metadata(binding)
    worker.start_load_kv(metadata)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}

    # When
    finished = worker.get_finished(set(), metadata)

    # Then
    assert finished == (set(), {binding.decode_request_id})
    assert worker.get_block_ids_with_load_errors() == {102, 103, 104}


def test_unknown_wire_ids_are_retained_without_attribution():
    # Given
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    worker.request_map["known-wire"] = "known-request-00000001"
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {"unknown-done"}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {"unknown-failed"}

    # When
    finished = worker.get_finished(set(), metadata)

    # Then
    assert finished == (set(), set())
    assert worker._pending_forward_done_wire_ids == {"unknown-done"}
    assert worker._pending_forward_failed_wire_ids == {"unknown-failed"}
    assert worker.request_map == {"known-wire": "known-request-00000001"}
    assert worker.get_block_ids_with_load_errors() == set()


def test_finished_req_ids_do_not_suppress_ordinary_parent_terminals():
    # Given
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    done_request_id = "ordinary-done-00000001"
    failed_request_id = "ordinary-failed-00000001"
    done_wire_request_id = get_external_request_id(done_request_id)
    failed_wire_request_id = get_external_request_id(failed_request_id)
    worker.request_map = {
        done_wire_request_id: done_request_id,
        failed_wire_request_id: failed_request_id,
    }
    worker._recving_metadata = {
        done_request_id: SimpleNamespace(local_block_ids=((201, 202),)),
        failed_request_id: SimpleNamespace(local_block_ids=((301, 302),)),
    }
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {done_wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {failed_wire_request_id}

    # When
    finished = worker.get_finished({done_request_id, failed_request_id}, metadata)

    # Then
    assert finished == (set(), {done_request_id})
    assert worker.get_block_ids_with_load_errors() == {301, 302}
    assert worker.request_map == {}
    assert worker._recving_metadata == {}


def test_duplicate_terminals_are_idempotent_and_consumed_record_releases_on_finished_req_ids():
    # Given
    worker = _make_worker()
    binding = _make_binding()
    metadata = _binding_metadata(binding)
    worker.start_load_kv(metadata)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    assert worker.get_finished(set(), metadata) == (set(), {binding.decode_request_id})
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}
    assert worker.get_finished(set(), metadata) == (set(), set())
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}
    worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = {binding.wire_request_id}

    # When
    released = worker.get_finished({binding.decode_request_id}, metadata)

    # Then
    assert released == (set(), set())
    assert worker._consumed_forward_terminal_wire_ids == {}
    assert worker._pending_forward_done_wire_ids == set()
    assert worker._pending_forward_failed_wire_ids == set()


def test_shutdown_clears_task05_worker_state_and_active_wire_mapping():
    # Given
    worker = _make_worker()
    consumed_binding = _make_binding()
    active_binding = _make_binding("active-request-00000001")
    consumed_metadata = _binding_metadata(consumed_binding)
    worker.start_load_kv(consumed_metadata)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {consumed_binding.wire_request_id}
    assert worker.get_finished(set(), consumed_metadata) == (set(), {consumed_binding.decode_request_id})
    worker.start_load_kv(_binding_metadata(active_binding))
    worker._pending_forward_done_wire_ids.add("unknown-done")
    worker._pending_forward_failed_wire_ids.add("unknown-failed")

    # When
    worker.shutdown()
    worker.shutdown()

    # Then
    assert active_binding.wire_request_id not in worker.request_map
    assert worker._forward_receive_bindings == {}
    assert worker._pending_forward_done_wire_ids == set()
    assert worker._pending_forward_failed_wire_ids == set()
    assert worker._consumed_forward_terminal_wire_ids == {}
    worker._kvpool_worker_adapter.close.assert_not_called()

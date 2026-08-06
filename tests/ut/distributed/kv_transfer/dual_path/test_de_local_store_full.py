from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from vllm.v1.outputs import KVConnectorOutput

from tests.ut.distributed.kv_transfer.dual_path.test_decode_scheduler import (
    _CONNECTOR_NS,
    _make_blocks,
    _make_kv_cache_config,
    _make_request,
    _make_vllm_config,
    _selected_params,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DecisionTimeoutMetadata,
    DualPathConnectorMetadata,
    ForwardReceiveBinding,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    Path,
    PathDecisionResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import DecodeControlEndpoint
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
    MooncakeLayerwiseConnectorWorker,
    get_external_request_id,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
)


def _make_store_metadata() -> AscendConnectorMetadata:
    return AscendConnectorMetadata(set(), set())


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
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), set())
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = set()
    worker.engine = MagicMock(name="transfer_engine")
    worker.block_size = [16]
    return worker


@pytest.fixture()
def decode_scheduler():
    with (
        patch(f"{_CONNECTOR_NS}.KVPoolAdapter"),
        patch(f"{_CONNECTOR_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_CONNECTOR_NS}.get_ip", return_value="127.0.0.1"),
        patch(f"{_CONNECTOR_NS}.derive_decode_control_port", return_value=7100),
    ):
        coordinator = MagicMock(name="decode_coordinator")
        coordinator.decode_engine_instance_id = "test_engine:0:test-boot"
        coordinator.decode_control_endpoint = DecodeControlEndpoint(host="127.0.0.1", port=7100)
        coordinator_cls.for_decode.return_value = coordinator
        scheduler = DualPathConnectorScheduler(
            _make_vllm_config(),
            _make_kv_cache_config(),
            "test_engine",
            DualPathConfig(role="decode"),
        )
        scheduler.executor.shutdown(wait=False)
        scheduler.metaserver_client.close()
        scheduler.executor = MagicMock(name="executor")
        scheduler._path_decision_coordinator.take_received_results.return_value = []
        scheduler._kvpool_adapter.build_connector_meta.return_value = _make_store_metadata()
        yield scheduler
        scheduler.shutdown()


def test_full_uses_ready_delta_partial_and_miss_use_transfer_delta(decode_scheduler) -> None:
    full = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    partial = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)

    for request_id, spec, expected in (
        ("full", full, (31, True)),
        ("partial", partial, (32, True)),
        ("miss", None, (32, True)),
    ):
        request = _make_request(request_id, 48, _selected_params())
        decode_scheduler._kvpool_adapter.lookup.return_value = spec

        assert decode_scheduler.get_num_new_matched_tokens(request, 16) == expected


def test_hybrid_full_and_non_full_share_boundary_but_not_route() -> None:
    with (
        patch(f"{_CONNECTOR_NS}.KVPoolAdapter"),
        patch(f"{_CONNECTOR_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_CONNECTOR_NS}.get_ip", return_value="127.0.0.1"),
        patch(f"{_CONNECTOR_NS}.derive_decode_control_port", return_value=7100),
    ):
        coordinator_cls.for_decode.return_value.decode_engine_instance_id = "test_engine:0:test-boot"
        coordinator_cls.for_decode.return_value.decode_control_endpoint = DecodeControlEndpoint(
            host="127.0.0.1", port=7100
        )
        scheduler = DualPathConnectorScheduler(
            _make_vllm_config(),
            _make_kv_cache_config(need_truncate=True),
            "test_engine",
            DualPathConfig(role="decode"),
        )
        scheduler.executor.shutdown(wait=False)
        scheduler.metaserver_client.close()
        scheduler.executor = MagicMock(name="executor")
        try:
            full = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
            partial = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
            full_request = _make_request("hybrid-full", 48, _selected_params())
            partial_request = _make_request("hybrid-partial", 48, _selected_params())
            scheduler._kvpool_adapter.lookup.side_effect = [full, partial]

            assert scheduler.get_num_new_matched_tokens(full_request, 16) == (31, True)
            assert scheduler.get_num_new_matched_tokens(partial_request, 16) == (31, True)
            scheduler.update_state_after_alloc(full_request, _make_blocks(((1, 2, 3),)), 31)
            scheduler.update_state_after_alloc(partial_request, _make_blocks(((4, 5, 6),)), 31)

            scheduler._kvpool_adapter.commit_after_alloc.assert_called_once_with(
                full_request, scheduler._kvpool_adapter.commit_after_alloc.call_args.args[1], full
            )
            assert "hybrid-full" not in scheduler._decode_decision_states
            assert "hybrid-partial" in scheduler._decode_decision_states
        finally:
            scheduler.shutdown()


def test_alignment_clamped_store_hit_stays_on_remote_route(decode_scheduler) -> None:
    request = _make_request("clamped-partial", 48, _selected_params())
    clamped_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=46, can_load=False)
    decode_scheduler._kvpool_adapter.lookup.return_value = clamped_spec

    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    decode_scheduler.update_state_after_alloc(request, _make_blocks(((1, 2, 3),)), 32)

    decode_scheduler._kvpool_adapter.commit_after_alloc.assert_not_called()
    assert request.request_id in decode_scheduler._decode_decision_states


def test_store_full_snapshot_binds_explicit_fields_and_detached_spec_stays_untouched(decode_scheduler) -> None:
    request = _make_request("snapshot", 48, _selected_params())
    spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    original_spec = deepcopy(spec)
    blocks = _make_blocks(((7, 8), (9,)))
    decode_scheduler._kvpool_adapter.lookup.return_value = spec

    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
    decode_scheduler.update_state_after_alloc(request, blocks, 31)

    snapshot = decode_scheduler._decode_kv_snapshots[request.request_id]
    assert snapshot.transfer_tokens == 48
    assert snapshot.local_tokens == 16
    assert snapshot.external_tokens == 31
    assert snapshot.store_load_spec is spec
    assert snapshot.final_block_ids == ((7, 8), (9,))
    assert spec == original_spec
    decode_scheduler._kvpool_adapter.commit_after_alloc.assert_called_once_with(request, blocks, spec)


def test_store_full_identical_retry_reuses_lookup_and_duplicate_bind_never_commits_twice(decode_scheduler) -> None:
    request = _make_request("retry", 48, _selected_params())
    spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    blocks = _make_blocks(((7, 8, 9),))
    decode_scheduler._kvpool_adapter.lookup.return_value = spec

    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
    decode_scheduler.update_state_after_alloc(request, blocks, 31)
    first_snapshot = decode_scheduler._decode_kv_snapshots[request.request_id]
    request.kv_transfer_params["do_remote_prefill"] = True
    decode_scheduler.update_state_after_alloc(request, blocks, 31)

    decode_scheduler._kvpool_adapter.lookup.assert_called_once_with(request, 16)
    decode_scheduler._kvpool_adapter.commit_after_alloc.assert_called_once_with(request, blocks, spec)
    assert decode_scheduler._decode_kv_snapshots[request.request_id] is first_snapshot

    with pytest.raises(RuntimeError, match="conflicting duplicate"):
        decode_scheduler.update_state_after_alloc(request, _make_blocks(((10, 11, 12),)), 31)
    assert decode_scheduler._decode_kv_snapshots[request.request_id] is first_snapshot
    decode_scheduler._kvpool_adapter.commit_after_alloc.assert_called_once()


def test_store_full_alloc_creates_no_decision_side_effects(decode_scheduler) -> None:
    request = _make_request("side-effects", 48, _selected_params())
    spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    decode_scheduler._kvpool_adapter.lookup.return_value = spec
    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)

    with (
        patch(f"{_CONNECTOR_NS}.DualPathRequestKey", side_effect=AssertionError("request key constructed")),
        patch(f"{_CONNECTOR_NS}.build_remote_decode_message", side_effect=AssertionError("envelope built")),
    ):
        decode_scheduler.update_state_after_alloc(request, _make_blocks(((1, 2, 3),)), 31)

    assert decode_scheduler._decode_decision_states == {}
    assert decode_scheduler._reqs_need_recv == {}
    decode_scheduler._path_decision_coordinator.register_pending.assert_not_called()
    decode_scheduler.executor.submit.assert_not_called()
    assert request.kv_transfer_params["do_remote_prefill"] is False


def test_store_full_never_touches_path_policy() -> None:
    policy = MagicMock(name="path_policy")
    with (
        patch(f"{_CONNECTOR_NS}.KVPoolAdapter"),
        patch(f"{_CONNECTOR_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{_CONNECTOR_NS}.get_ip", return_value="127.0.0.1"),
        patch(f"{_CONNECTOR_NS}.derive_decode_control_port", return_value=7100),
    ):
        coordinator_cls.for_decode.return_value.decode_engine_instance_id = "test_engine:0:test-boot"
        coordinator_cls.for_decode.return_value.decode_control_endpoint = DecodeControlEndpoint(
            host="127.0.0.1", port=7100
        )
        scheduler = DualPathConnectorScheduler(
            _make_vllm_config(),
            _make_kv_cache_config(),
            "test_engine",
            DualPathConfig(role="decode"),
            path_policy=policy,
        )
        scheduler.executor.shutdown(wait=False)
        scheduler.metaserver_client.close()
        scheduler.executor = MagicMock(name="executor")
        try:
            request = _make_request("policy", 48, _selected_params())
            spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
            scheduler._kvpool_adapter.lookup.return_value = spec

            assert scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
            scheduler.update_state_after_alloc(request, _make_blocks(((1, 2, 3),)), 31)

            policy.choose.assert_not_called()
        finally:
            scheduler.shutdown()


def test_decode_register_kv_caches_delegates_to_parent_and_store_worker_exactly_once() -> None:
    decode_config = MagicMock(name="decode_config")
    prefill_config = MagicMock(name="prefill_config")
    kv_cache_config = MagicMock(name="kv_cache_config")
    kv_caches = {"layer.0": MagicMock(name="kv_cache")}

    with (
        patch.object(MooncakeLayerwiseConnectorWorker, "__init__", autospec=True, return_value=None) as parent_init,
        patch.object(MooncakeLayerwiseConnectorWorker, "register_kv_caches", autospec=True) as parent_register,
        patch.object(connector_module, "KVPoolWorkerAdapter") as adapter_cls,
    ):
        decode_worker = DualPathConnectorWorker(
            decode_config,
            kv_cache_config,
            "decode-engine",
            DualPathConfig(role="decode"),
        )
        prefill_worker = DualPathConnectorWorker(
            prefill_config,
            kv_cache_config,
            "prefill-engine",
            DualPathConfig(role="prefill"),
        )

        decode_worker.register_kv_caches(kv_caches)
        prefill_worker.register_kv_caches(kv_caches)

    assert parent_init.call_count == 2
    assert parent_register.call_args_list == [
        call(decode_worker, kv_caches),
        call(prefill_worker, kv_caches),
    ]
    adapter_cls.assert_called_once_with(decode_config, kv_cache_config)
    adapter_cls.return_value.register_kv_caches.assert_called_once_with(kv_caches)
    assert prefill_worker._kvpool_worker_adapter is None


def test_start_load_kv_runs_store_adapter_once_between_control_and_parent() -> None:
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    binding = MagicMock(name="forward_binding")
    metadata.forward_receive_bindings.append(binding)
    metadata.decision_timeouts.append(DecisionTimeoutMetadata("timed-out-request", (91, 92)))
    store_metadata = _make_store_metadata()
    store_metadata.loading_req_ids.add("store-request")
    metadata.decode_store_metadata = store_metadata
    order: list[str] = []

    def start_store(received_metadata: AscendConnectorMetadata) -> None:
        assert worker._control_failed_recving == {"timed-out-request"}
        assert worker._invalid_block_ids == {91, 92}
        assert received_metadata is store_metadata
        order.append("store")

    worker._kvpool_worker_adapter.start_load_kv.side_effect = start_store
    with (
        patch.object(worker, "_install_forward_receive_binding", side_effect=lambda _: order.append("binding")),
        patch.object(
            MooncakeLayerwiseConnectorWorker,
            "start_load_kv",
            autospec=True,
            side_effect=lambda *_: order.append("parent"),
        ) as parent_start,
    ):
        worker.start_load_kv(metadata)

    assert order == ["binding", "store", "parent"]
    worker._kvpool_worker_adapter.start_load_kv.assert_called_once_with(store_metadata)
    parent_start.assert_called_once_with(worker, metadata)
    assert worker.kv_recv_layer_thread.method_calls == []


def test_store_done_publishes_finished_recving_exactly_once() -> None:
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    store_metadata = _make_store_metadata()
    store_metadata.loading_req_ids.add("store-request")
    metadata.decode_store_metadata = store_metadata
    worker._kvpool_worker_adapter.get_finished.side_effect = [
        (set(), {"store-request"}),
        (set(), set()),
    ]

    first = worker.get_finished(set(), metadata)
    second = worker.get_finished(set(), metadata)

    assert first == (set(), {"store-request"})
    assert second == (set(), set())
    assert worker._kvpool_worker_adapter.get_finished.call_args_list == [
        call(set(), store_metadata),
        call(set(), store_metadata),
    ]


def test_store_failed_publishes_destination_invalid_blocks_and_finished_recving_together() -> None:
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    store_metadata = _make_store_metadata()
    store_metadata.loading_req_ids.add("store-request")
    metadata.decode_store_metadata = store_metadata
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {"store-request"})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {102, 103}

    finished_sending, finished_recving = worker.get_finished(set(), metadata)
    connector_output = KVConnectorOutput(
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        invalid_block_ids=worker.get_block_ids_with_load_errors(),
    )

    assert connector_output.finished_sending == set()
    assert connector_output.finished_recving == {"store-request"}
    assert connector_output.invalid_block_ids == {102, 103}


def test_store_failure_never_invalidates_hbm_prefix_blocks() -> None:
    worker = _make_worker()
    metadata = DualPathConnectorMetadata()
    store_metadata = _make_store_metadata()
    store_metadata.loading_req_ids.add("store-request")
    metadata.decode_store_metadata = store_metadata
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {"store-request"})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {102, 103}

    worker.get_finished(set(), metadata)
    invalid_block_ids = worker.get_block_ids_with_load_errors()

    assert invalid_block_ids == {102, 103}
    assert 101 not in invalid_block_ids


def test_full_probe_then_worker_miss_fails_closed_without_proxy_fallback(decode_scheduler) -> None:
    request = _make_request("probe-load-race", 48, _selected_params())
    spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    blocks = _make_blocks(((101, 102, 103),))
    decode_scheduler._kvpool_adapter.lookup.return_value = spec
    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
    decode_scheduler.update_state_after_alloc(request, blocks, 31)
    store_metadata = _make_store_metadata()
    store_metadata.requests.append(SimpleNamespace(req_id=request.request_id))
    store_metadata.loading_req_ids.add(request.request_id)
    decode_scheduler._kvpool_adapter.build_connector_meta.return_value = store_metadata
    parent_metadata = MooncakeLayerwiseConnectorMetadata()

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        "build_connector_meta",
        autospec=True,
        return_value=parent_metadata,
    ):
        metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    worker = _make_worker()
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {request.request_id})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {102, 103}
    with patch.object(MooncakeLayerwiseConnectorWorker, "start_load_kv", autospec=True):
        worker.start_load_kv(metadata)
    finished_sending, finished_recving = worker.get_finished(set(), metadata)
    connector_output = KVConnectorOutput(
        finished_sending=finished_sending,
        finished_recving=finished_recving,
        invalid_block_ids=worker.get_block_ids_with_load_errors(),
    )

    assert connector_output.finished_recving == {request.request_id}
    assert connector_output.invalid_block_ids == {102, 103}
    decode_scheduler._kvpool_adapter.lookup.assert_called_once_with(request, 16)
    decode_scheduler._kvpool_adapter.commit_after_alloc.assert_called_once_with(request, blocks, spec)
    decode_scheduler._path_decision_coordinator.register_pending.assert_not_called()
    decode_scheduler.executor.submit.assert_not_called()
    assert request.kv_transfer_params["do_remote_prefill"] is False


def test_parent_timeout_forward_and_store_completions_stay_isolated() -> None:
    worker = _make_worker()
    ordinary_request_id = "ordinary-request"
    forward_request_id = "forward-request"
    store_request_id = "store-request"
    timeout_request_id = "timeout-request"
    ordinary_wire_id = get_external_request_id(ordinary_request_id)
    binding = ForwardReceiveBinding(
        request_key=DualPathRequestKey("decode-instance", forward_request_id),
        wire_request_id=get_external_request_id(forward_request_id),
        decode_request_id=forward_request_id,
        destination_block_ids=((201, 202),),
        token_start=16,
        token_end=32,
    )
    metadata = DualPathConnectorMetadata()
    metadata.forward_receive_bindings.append(binding)
    metadata.decision_timeouts.append(DecisionTimeoutMetadata(timeout_request_id, (401, 402)))
    store_metadata = _make_store_metadata()
    store_metadata.loading_req_ids.add(store_request_id)
    metadata.decode_store_metadata = store_metadata
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {store_request_id})

    with patch.object(MooncakeLayerwiseConnectorWorker, "start_load_kv", autospec=True):
        worker.start_load_kv(metadata)
    worker.request_map[ordinary_wire_id] = ordinary_request_id
    worker._recving_metadata[ordinary_request_id] = SimpleNamespace(local_block_ids=((301, 302),))
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {
        ordinary_wire_id,
        binding.wire_request_id,
    }

    finished = worker.get_finished(set(), metadata)
    invalid_block_ids = worker.get_block_ids_with_load_errors()

    assert finished == (
        set(),
        {ordinary_request_id, forward_request_id, store_request_id, timeout_request_id},
    )
    assert invalid_block_ids == {401, 402}
    worker._kvpool_worker_adapter.get_finished.assert_called_once_with(set(), store_metadata)
    assert worker._forward_receive_bindings == {}
    assert worker._control_failed_recving == set()


@pytest.mark.parametrize(
    "populate_store_metadata",
    [
        pytest.param(None, id="empty"),
        pytest.param(
            lambda metadata: metadata.requests.append(SimpleNamespace(req_id="store-request")),
            id="requests",
        ),
        pytest.param(
            lambda metadata: metadata.unfinished_request_ids.add("store-request"),
            id="unfinished-request-ids",
        ),
        pytest.param(
            lambda metadata: metadata.preempted_req_ids.add("store-request"),
            id="preempted-request-ids",
        ),
        pytest.param(
            lambda metadata: metadata.loading_req_ids.add("store-request"),
            id="loading-request-ids",
        ),
        pytest.param(
            lambda metadata: metadata.delayed_free_req_ids.add("store-request"),
            id="delayed-free-request-ids",
        ),
    ],
)
def test_decode_store_metadata_attaches_iff_lifecycle_collection_is_nonempty(
    decode_scheduler,
    populate_store_metadata,
) -> None:
    store_metadata = _make_store_metadata()
    if populate_store_metadata is not None:
        populate_store_metadata(store_metadata)
    decode_scheduler._kvpool_adapter.build_connector_meta.return_value = store_metadata
    scheduler_output = MagicMock(name="scheduler_output")

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        "build_connector_meta",
        autospec=True,
        return_value=MooncakeLayerwiseConnectorMetadata(),
    ):
        metadata = decode_scheduler.build_connector_meta(scheduler_output)

    decode_scheduler._kvpool_adapter.build_connector_meta.assert_called_once_with(scheduler_output)
    expected = store_metadata if populate_store_metadata is not None else None
    assert metadata.decode_store_metadata is expected


def test_parent_requests_map_never_holds_store_full_request(decode_scheduler) -> None:
    request = _make_request("store-only", 48, _selected_params())
    spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    decode_scheduler._kvpool_adapter.lookup.return_value = spec
    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
    decode_scheduler.update_state_after_alloc(request, _make_blocks(((101, 102, 103),)), 31)
    store_metadata = _make_store_metadata()
    store_metadata.requests.append(SimpleNamespace(req_id=request.request_id))
    decode_scheduler._kvpool_adapter.build_connector_meta.return_value = store_metadata

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        "build_connector_meta",
        autospec=True,
        return_value=MooncakeLayerwiseConnectorMetadata(),
    ):
        metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert metadata.requests == {}
    assert [store_request.req_id for store_request in metadata.decode_store_metadata.requests] == [request.request_id]
    assert decode_scheduler._reqs_need_recv == {}


def test_decode_metadata_composition_builds_store_after_results_bindings_and_deadlines(decode_scheduler) -> None:
    request = _make_request("composition-order", 48, _selected_params())
    partial_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
    decode_scheduler._kvpool_adapter.lookup.return_value = partial_spec
    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    decode_scheduler.update_state_after_alloc(request, _make_blocks(((101, 102, 103),)), 32)
    state = decode_scheduler._decode_decision_states[request.request_id]
    result = PathDecisionResult(request_key=state.request_key, path=Path.PE_READ)
    store_metadata = _make_store_metadata()
    order: list[str] = []
    original_binding_type = ForwardReceiveBinding

    def build_parent(*_args) -> MooncakeLayerwiseConnectorMetadata:
        order.append("parent")
        return MooncakeLayerwiseConnectorMetadata()

    def take_results() -> list[PathDecisionResult]:
        order.append("results")
        return [result]

    def build_binding(**kwargs) -> ForwardReceiveBinding:
        order.append("bindings")
        return original_binding_type(**kwargs)

    def read_deadline_clock() -> float:
        order.append("deadlines")
        return state.deadline - 1

    def build_store(_scheduler_output) -> AscendConnectorMetadata:
        order.append("store")
        return store_metadata

    decode_scheduler._path_decision_coordinator.take_received_results.side_effect = take_results
    decode_scheduler._kvpool_adapter.build_connector_meta.side_effect = build_store
    with (
        patch.object(
            MooncakeLayerwiseConnectorScheduler,
            "build_connector_meta",
            autospec=True,
            side_effect=build_parent,
        ),
        patch.object(connector_module, "ForwardReceiveBinding", side_effect=build_binding),
        patch.object(connector_module.time, "monotonic", side_effect=read_deadline_clock),
    ):
        metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

    assert order == ["parent", "results", "bindings", "deadlines", "store"]
    assert len(metadata.forward_receive_bindings) == 1


@pytest.mark.parametrize(
    ("method_name", "block_ids"),
    [
        ("request_finished", [101, 102, 103]),
        ("request_finished_all_groups", ([101, 102, 103],)),
    ],
)
def test_terminal_hooks_and_shutdown_release_all_store_records(
    decode_scheduler,
    method_name,
    block_ids,
) -> None:
    request = _make_request(f"terminal-{method_name}", 48, _selected_params())
    spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=47, can_load=False)
    decode_scheduler._kvpool_adapter.lookup.return_value = spec
    assert decode_scheduler.get_num_new_matched_tokens(request, 16) == (31, True)
    decode_scheduler.update_state_after_alloc(request, _make_blocks(((101, 102, 103),)), 31)
    parent_result = (False, None)

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        method_name,
        autospec=True,
        return_value=parent_result,
    ) as parent_finish:
        result = getattr(decode_scheduler, method_name)(request, block_ids)

    scheduler_output = SimpleNamespace(
        finished_req_ids={request.request_id},
        preempted_req_ids=set(),
    )
    decode_scheduler._kvpool_adapter.build_connector_meta.return_value = _make_store_metadata()
    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        "build_connector_meta",
        autospec=True,
        return_value=MooncakeLayerwiseConnectorMetadata(),
    ):
        decode_scheduler.build_connector_meta(scheduler_output)

    worker = _make_worker()
    worker_metadata = DualPathConnectorMetadata()
    store_metadata = _make_store_metadata()
    store_metadata.loading_req_ids.add(request.request_id)
    worker_metadata.decode_store_metadata = store_metadata
    worker.get_finished({request.request_id}, worker_metadata)
    decode_scheduler.shutdown()
    worker.shutdown()

    assert result == parent_result
    parent_finish.assert_called_once_with(decode_scheduler, request, block_ids)
    assert decode_scheduler._lookup_results == {}
    assert decode_scheduler._decode_kv_snapshots == {}
    decode_scheduler._kvpool_adapter.build_connector_meta.assert_called_once_with(scheduler_output)
    worker._kvpool_worker_adapter.get_finished.assert_called_once_with({request.request_id}, store_metadata)
    assert decode_scheduler._kvpool_adapter.close.call_count == 1
    assert worker._kvpool_worker_adapter.close.call_count == 1

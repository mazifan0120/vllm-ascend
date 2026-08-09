# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Task-01 Decode ``KVPoolAdapter`` (spec §11.1).

The adapter owns a dedicated non-layerwise ``KVPoolScheduler``; these tests
prove lookup results are always detached from the owned ``load_specs`` map,
token invariants are enforced, the Decode-ready clamp preserves the raw
``LoadSpec`` fields, and ``close()`` is idempotent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (  # noqa: E402
    KVPoolAdapter,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (  # noqa: E402
    LoadSpec,
)

_BLOCK_SIZE = 16
_PAGE_SIZE_BYTES = 12345


@pytest.fixture(autouse=True)
def _patch_pool_scheduler_importlib():
    """Same seam as the ascend_store scheduler tests: the KVPool backend is
    resolved dynamically via importlib; point it at a MagicMock."""
    with patch("vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.importlib") as mock_importlib:
        mock_importlib.import_module.return_value = MagicMock()
        yield


@pytest.fixture()
def mock_lookup_client_cls():
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler.LookupKeyClient"
    ) as mock_client_cls:
        yield mock_client_cls


def _make_vllm_config(extra_config=None, from_extra=None, disable_hybrid=True):
    config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_consumer"
    config.kv_transfer_config.kv_connector_extra_config = extra_config or {"consumer_is_to_load": True}
    if from_extra is None:
        config.kv_transfer_config.get_from_extra_config.return_value = True
    else:
        config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: from_extra.get(key, default)
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.rank = 0
    config.parallel_config.world_size = 1
    config.cache_config.block_size = _BLOCK_SIZE
    config.cache_config.hash_block_size = _BLOCK_SIZE
    config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hybrid
    config.model_config.model = "org/llama-7b"
    config.model_config.use_mla = False
    config.model_config.hf_text_config = MagicMock(spec=[])
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.model_config.get_num_layers.return_value = 2
    return config


def _make_kv_cache_config():
    spec = SimpleNamespace(page_size_bytes=_PAGE_SIZE_BYTES, block_size=_BLOCK_SIZE)
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=["layer.0"])])


def _make_adapter(extra_config=None):
    adapter = KVPoolAdapter(_make_vllm_config(extra_config), _make_kv_cache_config())
    return adapter


def _make_request(request_id: str, num_tokens: int) -> MagicMock:
    request = MagicMock()
    request.request_id = request_id
    request.num_tokens = num_tokens
    request.prompt_token_ids = list(range(num_tokens))
    request.block_hashes = [MagicMock() for _ in range(num_tokens // _BLOCK_SIZE + 1)]
    return request


def test_constructs_dedicated_non_layerwise_scheduler_with_group0_page_size(mock_lookup_client_cls):
    adapter = _make_adapter()
    pool = adapter._pool_scheduler
    assert pool.use_layerwise is False
    assert pool.page_size_bytes == _PAGE_SIZE_BYTES


def test_lookup_full_hit_clamped_to_decode_ready_boundary(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.return_value = 48
    spec = adapter.lookup(_make_request("req-full", 48), 0)
    assert spec is not None
    assert spec.kvpool_cached_tokens == 47  # full-prompt hit reduced to R = P - 1
    assert spec.vllm_cached_tokens == 0
    assert "req-full" not in adapter._pool_scheduler.load_specs


def test_lookup_partial_hit(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.return_value = 32
    spec = adapter.lookup(_make_request("req-partial", 48), 16)
    assert spec is not None
    assert spec.vllm_cached_tokens == 16
    assert spec.kvpool_cached_tokens == 32  # L_DE < K_DE < R
    assert "req-partial" not in adapter._pool_scheduler.load_specs


def test_lookup_miss_returns_none(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.return_value = 0
    assert adapter.lookup(_make_request("req-miss", 48), 0) is None
    assert "req-miss" not in adapter._pool_scheduler.load_specs


def test_lookup_hit_not_extending_hbm_returns_none(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.return_value = 16
    assert adapter.lookup(_make_request("req-no-extend", 48), 16) is None
    assert "req-no-extend" not in adapter._pool_scheduler.load_specs


def test_lookup_inherits_alignment_and_discard_partial_chunks(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.return_value = 48
    spec = adapter.lookup(_make_request("req-align", 50), 0)
    # 50-token prompt is floored to the 16-token cache-transfer granularity.
    called_token_len = mock_lookup_client_cls.return_value.lookup.call_args[0][0]
    assert called_token_len == 48
    assert spec is not None
    assert spec.kvpool_cached_tokens == 48


def test_lookup_multi_group_forwards_all_group_ids_and_returns_common_prefix(mock_lookup_client_cls):
    # Two-group hybrid cache: group 1 uses a non-FullAttention placeholder spec
    # so the owned scheduler participates with both group ids; the worker-side
    # lookup contract returns the minimum/common usable prefix across groups.
    spec0 = SimpleNamespace(page_size_bytes=_PAGE_SIZE_BYTES, block_size=16)
    spec1 = SimpleNamespace(page_size_bytes=_PAGE_SIZE_BYTES, block_size=32)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=spec0, layer_names=["layer.0"]),
            SimpleNamespace(kv_cache_spec=spec1, layer_names=["layer.1"]),
        ]
    )
    config = _make_vllm_config(disable_hybrid=False)
    adapter = KVPoolAdapter(config, kv_cache_config)
    assert adapter._pool_scheduler.kv_cache_group_ids == [0, 1]

    mock_lookup_client_cls.return_value.lookup.return_value = 48
    spec = adapter.lookup(_make_request("req-multi", 64), 0)
    call_args = mock_lookup_client_cls.return_value.lookup.call_args[0]
    assert call_args[2] == [0, 1]
    assert spec is not None
    assert spec.kvpool_cached_tokens == 48  # common prefix value is passed through
    assert "req-multi" not in adapter._pool_scheduler.load_specs


def test_lookup_with_discard_partial_chunks_disabled_uses_unfloored_length(mock_lookup_client_cls):
    config = _make_vllm_config(from_extra={"discard_partial_chunks": False})
    adapter = KVPoolAdapter(config, _make_kv_cache_config())
    mock_lookup_client_cls.return_value.lookup.return_value = 50
    spec = adapter.lookup(_make_request("req-nofloor", 50), 0)
    called_token_len = mock_lookup_client_cls.return_value.lookup.call_args[0][0]
    assert called_token_len == 50  # unfloored prompt length reaches the lookup path
    assert spec is not None
    assert spec.kvpool_cached_tokens == 49  # full-prompt hit reduced to R = P - 1


def test_lookup_ignores_kvpool_async_flag(mock_lookup_client_cls):
    adapter = _make_adapter(extra_config={"consumer_is_to_load": True, "load_async": True})
    mock_lookup_client_cls.return_value.lookup.return_value = 48
    spec = adapter.lookup(_make_request("req-async", 48), 0)
    # KVPoolScheduler reports (delta, True) for load_async; the adapter returns
    # the same detached LoadSpec regardless and owns the Core-facing async bit.
    assert spec is not None
    assert spec.kvpool_cached_tokens == 47


def _stub_scheduler_lookup(adapter, spec: LoadSpec | None, delta: int):
    def fake_get_num_new_matched_tokens(request, local_tokens):
        if spec is not None:
            adapter._pool_scheduler.load_specs[request.request_id] = spec
        return delta, False

    return patch.object(
        adapter._pool_scheduler,
        "get_num_new_matched_tokens",
        side_effect=fake_get_num_new_matched_tokens,
    )


@pytest.mark.parametrize(
    ("request_id", "local_tokens", "spec", "delta"),
    [
        pytest.param(
            "req-delta",
            0,
            LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=False),
            10,
            id="lookup-delta-mismatch",
        ),
        pytest.param(
            "req-local",
            16,
            LoadSpec(vllm_cached_tokens=8, kvpool_cached_tokens=32, can_load=False),
            24,
            id="local-token-mismatch",
        ),
    ],
)
def test_lookup_accounting_divergence_degrades_to_store_miss(
    mock_lookup_client_cls,
    request_id,
    local_tokens,
    spec,
    delta,
):
    # Given
    adapter = _make_adapter()
    request = _make_request(request_id, 48)

    # When
    with (
        _stub_scheduler_lookup(adapter, spec, delta),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter.logger.exception") as log_exception,
    ):
        result = adapter.lookup(request, local_tokens)

    # Then
    assert result is None
    assert request_id not in adapter._pool_scheduler.load_specs
    log_exception.assert_called_once()


def test_lookup_exception_returns_none_and_detaches(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.side_effect = RuntimeError("store unreachable")
    assert adapter.lookup(_make_request("req-error", 48), 0) is None
    assert "req-error" not in adapter._pool_scheduler.load_specs


def test_lookup_leaves_private_load_specs_empty(mock_lookup_client_cls):
    adapter = _make_adapter()
    client = mock_lookup_client_cls.return_value
    client.lookup.return_value = 48
    adapter.lookup(_make_request("req-h", 48), 0)
    client.lookup.return_value = 0
    adapter.lookup(_make_request("req-m", 48), 0)
    client.lookup.side_effect = RuntimeError("boom")
    adapter.lookup(_make_request("req-e", 48), 0)
    assert adapter._pool_scheduler.load_specs == {}


def _make_commit_adapter() -> KVPoolAdapter:
    return _make_adapter(extra_config={"consumer_is_to_load": True, "load_async": True})


def _make_blocks(block_ids: list[list[int]]) -> MagicMock:
    blocks = MagicMock()
    blocks.get_block_ids.return_value = block_ids
    return blocks


def test_commit_after_alloc_inserts_copy_not_detached_spec(mock_lookup_client_cls):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-copy", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)

    # When
    adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)

    # Then
    committed_spec = adapter._pool_scheduler.load_specs[request.request_id]
    assert committed_spec is not detached_spec
    assert committed_spec.vllm_cached_tokens == detached_spec.vllm_cached_tokens
    assert committed_spec.kvpool_cached_tokens == detached_spec.kvpool_cached_tokens


def test_commit_after_alloc_sets_can_load_only_on_the_copy(mock_lookup_client_cls):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-can-load", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)

    # When
    adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)

    # Then
    assert adapter._pool_scheduler.load_specs[request.request_id].can_load is True
    assert detached_spec.can_load is False


@pytest.mark.parametrize("kvpool_cached_tokens", [32, 48])
def test_unified_commit_accepts_full_and_partial_specs(mock_lookup_client_cls, kvpool_cached_tokens):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request(f"req-unified-{kvpool_cached_tokens}", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=kvpool_cached_tokens, can_load=False)

    # When
    adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)

    # Then
    committed_spec = adapter._pool_scheduler.load_specs[request.request_id]
    assert committed_spec.kvpool_cached_tokens == kvpool_cached_tokens


def test_partial_commit_delegates_exact_delta_and_blocks(mock_lookup_client_cls):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-partial-delegate", 49)
    blocks = _make_blocks([[7, 8, 9]])
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)

    # When
    with patch.object(adapter._pool_scheduler, "update_state_after_alloc") as update_state_after_alloc:
        adapter.commit_after_alloc(request, blocks, detached_spec)

    # Then
    update_state_after_alloc.assert_called_once_with(request, blocks, 16)


def test_terminal_metadata_cleanup_releases_commit_ownership(mock_lookup_client_cls):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-terminal-release", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
    blocks = _make_blocks([[7, 8, 9]])
    adapter.commit_after_alloc(request, blocks, detached_spec)
    active_output = SchedulerOutput.make_empty()
    active_output.preempted_req_ids = set()
    adapter.build_connector_meta(active_output)

    # When
    finished_output = SchedulerOutput.make_empty()
    finished_output.finished_req_ids = {request.request_id}
    finished_output.preempted_req_ids = set()
    adapter.build_connector_meta(finished_output)

    # Then
    pool = adapter._pool_scheduler
    assert request.request_id not in pool.load_specs
    assert request.request_id not in pool._unfinished_requests
    assert request.request_id not in pool._unfinished_request_ids
    assert request.request_id not in pool._loading_req_ids
    adapter.commit_after_alloc(request, blocks, detached_spec)
    assert request.request_id in pool.load_specs


@pytest.mark.parametrize(
    "detached_spec",
    [
        LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False),
        LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False),
    ],
)
def test_commit_after_alloc_rolls_back_adapter_created_state_on_delegated_failure(
    mock_lookup_client_cls, detached_spec
):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-rollback", 49)
    original_update = adapter._pool_scheduler.update_state_after_alloc

    def fail_after_delegated_mutation(request_arg, blocks_arg, ready_delta):
        original_update(request_arg, blocks_arg, ready_delta)
        adapter._pool_scheduler._request_trackers[request.request_id] = MagicMock(name="created_tracker")
        raise RuntimeError("delegated failure")

    # When
    with (
        patch.object(
            adapter._pool_scheduler,
            "update_state_after_alloc",
            side_effect=fail_after_delegated_mutation,
        ),
        pytest.raises(RuntimeError, match="delegated failure"),
    ):
        adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)

    # Then
    assert request.request_id not in adapter._pool_scheduler.load_specs
    assert request.request_id not in adapter._pool_scheduler._request_trackers
    assert request.request_id not in adapter._pool_scheduler._unfinished_requests
    assert request.request_id not in adapter._pool_scheduler._unfinished_request_ids
    assert request.request_id not in adapter._pool_scheduler._loading_req_ids
    assert detached_spec.can_load is False


def test_build_connector_meta_emits_one_async_load_reqmeta_with_ready_boundary(mock_lookup_client_cls):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-metadata", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
    final_block_ids = [[7, 8, 9]]
    adapter.commit_after_alloc(request, _make_blocks(final_block_ids), detached_spec)
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData([], set(), [], {}, [], [], []),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
    )

    # When
    metadata = adapter.build_connector_meta(scheduler_output)

    # Then
    assert len(metadata.requests) == 1
    request_metadata = metadata.requests[0]
    assert request_metadata.req_id == request.request_id
    assert request_metadata.block_ids_by_group == final_block_ids
    assert request_metadata.block_hashes == request.block_hashes
    assert request_metadata.target_token_len == 48
    assert request_metadata.load_spec is not detached_spec
    assert request_metadata.load_spec is not None
    assert request_metadata.load_spec.vllm_cached_tokens == 16
    assert request_metadata.load_spec.kvpool_cached_tokens == 48
    assert request_metadata.load_spec.can_load is True
    assert metadata.loading_req_ids == {request.request_id}


def test_partial_commit_metadata_targets_k_de_only(mock_lookup_client_cls):
    # Given
    adapter = _make_commit_adapter()
    request = _make_request("req-partial-metadata", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=32, can_load=False)
    adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.preempted_req_ids = set()

    # When
    metadata = adapter.build_connector_meta(scheduler_output)

    # Then
    assert len(metadata.requests) == 1
    request_metadata = metadata.requests[0]
    assert request_metadata.target_token_len == 32
    assert request_metadata.load_spec is not None
    assert request_metadata.load_spec.kvpool_cached_tokens == 32


def test_build_connector_meta_filters_unowned_preemption_but_cleans_owned_tracker(mock_lookup_client_cls):
    adapter = _make_commit_adapter()
    request = _make_request("req-preempted", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
    adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.preempted_req_ids = {request.request_id, "ordinary-request"}

    metadata = adapter.build_connector_meta(scheduler_output)

    pool = adapter._pool_scheduler
    assert metadata.preempted_req_ids == {request.request_id}
    assert pool._preempted_req_ids == {request.request_id}
    assert pool._unfinished_requests == {}
    assert pool._request_trackers == {}
    assert pool._loading_req_ids == set()


def test_lookup_clamp_preserves_can_load_and_token_len_via_replace(mock_lookup_client_cls):
    adapter = _make_adapter()
    raw_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=60, can_load=False, token_len=7)
    with _stub_scheduler_lookup(adapter, raw_spec, delta=60):
        spec = adapter.lookup(_make_request("req-clamp", 48), 0)
    assert spec is not None
    assert spec is not raw_spec
    assert spec.kvpool_cached_tokens == 47  # clamped to R = P - 1
    assert spec.can_load is False
    assert spec.token_len == 7
    assert raw_spec.kvpool_cached_tokens == 60  # raw object never mutated


def test_close_idempotent_before_and_after_lazy_client_creation(mock_lookup_client_cls):
    adapter = _make_adapter()
    adapter.close()  # no client created yet
    mock_lookup_client_cls.return_value.lookup.return_value = 48
    adapter.lookup(_make_request("req-close", 48), 0)
    assert adapter._pool_scheduler.client is not None
    adapter.close()
    adapter.close()
    mock_lookup_client_cls.return_value.close.assert_called_once()
    assert adapter._pool_scheduler.client is None


def test_close_releases_all_private_scheduler_request_records(mock_lookup_client_cls):
    adapter = _make_commit_adapter()
    request = _make_request("req-shutdown", 49)
    detached_spec = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
    adapter.commit_after_alloc(request, _make_blocks([[7, 8, 9]]), detached_spec)
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.preempted_req_ids = set()
    adapter.build_connector_meta(scheduler_output)

    adapter.close()

    pool = adapter._pool_scheduler
    assert pool.load_specs == {}
    assert pool._request_trackers == {}
    assert pool._preempted_req_ids == set()
    assert pool._unfinished_requests == {}
    assert pool._unfinished_request_ids == set()
    assert pool._loading_req_ids == set()
    assert pool._delayed_free_req_ids == set()

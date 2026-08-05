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


def test_lookup_store_delta_mismatch_raises_and_detaches(mock_lookup_client_cls):
    adapter = _make_adapter()
    bad_spec = LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=False)
    with _stub_scheduler_lookup(adapter, bad_spec, delta=10), pytest.raises(RuntimeError, match="delta mismatch"):
        adapter.lookup(_make_request("req-delta", 48), 0)
    assert "req-delta" not in adapter._pool_scheduler.load_specs


def test_lookup_vllm_cached_mismatch_raises_and_detaches(mock_lookup_client_cls):
    adapter = _make_adapter()
    bad_spec = LoadSpec(vllm_cached_tokens=8, kvpool_cached_tokens=32, can_load=False)
    with (
        _stub_scheduler_lookup(adapter, bad_spec, delta=24),
        pytest.raises(RuntimeError, match="local-token mismatch"),
    ):
        adapter.lookup(_make_request("req-local", 48), 16)
    assert "req-local" not in adapter._pool_scheduler.load_specs


def test_lookup_exception_returns_none_and_detaches(mock_lookup_client_cls):
    adapter = _make_adapter()
    mock_lookup_client_cls.return_value.lookup.side_effect = RuntimeError("store unreachable")
    assert adapter.lookup(_make_request("req-error", 48), 0) is None
    assert "req-error" not in adapter._pool_scheduler.load_specs


def test_lookup_leaves_no_load_specs_entry_on_hit_miss_and_exception(mock_lookup_client_cls):
    adapter = _make_adapter()
    client = mock_lookup_client_cls.return_value
    client.lookup.return_value = 48
    adapter.lookup(_make_request("req-h", 48), 0)
    client.lookup.return_value = 0
    adapter.lookup(_make_request("req-m", 48), 0)
    client.lookup.side_effect = RuntimeError("boom")
    adapter.lookup(_make_request("req-e", 48), 0)
    assert adapter._pool_scheduler.load_specs == {}


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

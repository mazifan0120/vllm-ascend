# SPDX-License-Identifier: Apache-2.0
"""Chunked-get batching for the non-layerwise store receive thread."""

from unittest.mock import MagicMock

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (  # noqa: E402
    KVCacheStoreRecvingThread,
    _estimate_get_staging_bytes,
    _plan_get_batches,
)


def test_backend_staging_buffer_bytes_default_none():
    assert Backend.staging_buffer_bytes(MagicMock()) is None


def test_mooncake_backend_reports_local_buffer_size():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    backend = object.__new__(MooncakeBackend)
    backend.config = MagicMock(local_buffer_size=8 * 1024**3)
    backend._use_fabric_mem = False
    backend._contribute_memory = True
    assert backend.staging_buffer_bytes() == 8 * 1024**3


def test_mooncake_address_direct_backend_has_no_staging_constraint():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    backend = object.__new__(MooncakeBackend)
    backend.config = MagicMock(local_buffer_size=8 * 1024**3)
    backend._use_fabric_mem = True
    backend._contribute_memory = True
    assert backend.staging_buffer_bytes() is None


def test_plan_get_batches_packs_keys_under_usable_budget():
    sizes = [[1 * 1024**2], [2 * 1024**2], [3 * 1024**2], [4 * 1024**2], [512 * 1024]]
    usable = sum(_estimate_get_staging_bytes(key_sizes) for key_sizes in sizes[:3])
    batches, oversized = _plan_get_batches(
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=8 * 1024**2,
    )
    assert batches == [[0, 1, 2], [3, 4]]
    assert oversized == set()


def test_plan_get_batches_runs_key_between_usable_and_raw_alone():
    sizes = [[1 * 1024**2], [5 * 1024**2], [1 * 1024**2]]
    batches, oversized = _plan_get_batches(
        sizes,
        usable_budget_bytes=4 * 1024**2,
        raw_budget_bytes=6 * 1024**2,
    )
    assert batches == [[0], [1], [2]]
    assert oversized == set()


def test_plan_get_batches_marks_only_key_over_raw_and_continues():
    sizes = [[1 * 1024**2], [7 * 1024**2], [1 * 1024**2]]
    batches, oversized = _plan_get_batches(
        sizes,
        usable_budget_bytes=4 * 1024**2,
        raw_budget_bytes=6 * 1024**2,
    )
    assert batches == [[0, 2]]
    assert oversized == {1}


def test_estimate_get_staging_bytes_includes_alignment_and_padding():
    assert _estimate_get_staging_bytes([4097]) == 16 * 1024


def _make_thread(backend):
    thread = object.__new__(KVCacheStoreRecvingThread)
    thread.m_store = backend
    return thread


def test_chunked_store_get_merges_per_key_results():
    backend = MagicMock()
    backend.get.side_effect = [[0, 0, 0], None, [0]]
    thread = _make_thread(backend)
    keys = ["k0", "k1", "k2", "k3", "k4"]
    sizes = [[1 * 1024**2], [2 * 1024**2], [3 * 1024**2], [7 * 1024**2], [512 * 1024]]
    usable = sum(_estimate_get_staging_bytes(key_sizes) for key_sizes in sizes[:3])
    raw = _estimate_get_staging_bytes(sizes[3])
    ret = thread._chunked_store_get(
        keys,
        [[0]] * 5,
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=raw,
    )
    assert ret == [0, 0, 0, 1, 0]
    assert backend.get.call_count == 3


def test_chunked_store_get_oversized_key_failed_without_issuing():
    backend = MagicMock()
    backend.get.return_value = [0, 0]
    thread = _make_thread(backend)
    sizes = [[1 * 1024**2], [7 * 1024**2], [1 * 1024**2]]
    ret = thread._chunked_store_get(
        ["k0", "big", "k2"],
        [[0]] * 3,
        sizes,
        usable_budget_bytes=4 * 1024**2,
        raw_budget_bytes=6 * 1024**2,
    )
    assert ret == [0, 1, 0]
    assert "big" not in backend.get.call_args.args[0]


def test_chunked_store_get_marks_short_status_batch_failed_and_continues():
    backend = MagicMock()
    backend.get.side_effect = [[0], [0]]
    thread = _make_thread(backend)
    sizes = [[1 * 1024**2]] * 3
    usable = 2 * _estimate_get_staging_bytes(sizes[0])

    ret = thread._chunked_store_get(
        ["k0", "k1", "k2"],
        [[0]] * 3,
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=4 * 1024**2,
    )

    assert ret == [1, 1, 0]
    assert backend.get.call_count == 2


def test_chunked_store_get_marks_long_status_batch_failed_and_continues():
    backend = MagicMock()
    backend.get.side_effect = [[0, 0, 0], [0]]
    thread = _make_thread(backend)
    sizes = [[1 * 1024**2]] * 3
    usable = 2 * _estimate_get_staging_bytes(sizes[0])

    ret = thread._chunked_store_get(
        ["k0", "k1", "k2"],
        [[0]] * 3,
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=4 * 1024**2,
    )

    assert ret == [1, 1, 0]
    assert backend.get.call_count == 2


def test_chunked_store_get_marks_non_vector_status_batch_failed_and_continues():
    backend = MagicMock()
    backend.get.side_effect = [0, [0]]
    thread = _make_thread(backend)
    sizes = [[1 * 1024**2]] * 3
    usable = 2 * _estimate_get_staging_bytes(sizes[0])

    ret = thread._chunked_store_get(
        ["k0", "k1", "k2"],
        [[0]] * 3,
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=4 * 1024**2,
    )

    assert ret == [1, 1, 0]
    assert backend.get.call_count == 2


def test_staging_get_budgets_uses_fixed_reserve_fraction():
    backend = MagicMock()
    backend.staging_buffer_bytes.return_value = 10_000
    assert _make_thread(backend)._staging_get_budgets() == (10_000, 9_000)


def test_staging_get_budgets_none_for_unbounded_backend():
    backend = MagicMock()
    backend.staging_buffer_bytes.return_value = None
    assert _make_thread(backend)._staging_get_budgets() is None


def test_staging_get_budgets_rejects_non_positive_backend_budget():
    backend = MagicMock()
    backend.staging_buffer_bytes.return_value = 0
    with pytest.raises(ValueError, match="raw staging budget must be positive"):
        _make_thread(backend)._staging_get_budgets()

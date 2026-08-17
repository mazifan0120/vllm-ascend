# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Decode ``KVPoolWorkerAdapter``.

The worker adapter is a lookup-only composition: a non-layerwise
``KVPoolWorker`` plus the existing ``LookupKeyServer`` bound only on the
owning rank. These tests patch both collaborators at the adapter namespace;
the server lifetime follows the owning Worker process.
"""

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter import (  # noqa: E402
    KVPoolWorkerAdapter,
)

_ADAPTER_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.kvpool_adapter"


def _make_vllm_config(rank: int) -> MagicMock:
    config = MagicMock()
    config.parallel_config.rank = rank
    return config


@pytest.fixture()
def collaborators():
    with (
        patch(f"{_ADAPTER_NS}.KVPoolWorker") as mock_worker_cls,
        patch(f"{_ADAPTER_NS}.LookupKeyServer") as mock_server_cls,
    ):
        yield mock_worker_cls, mock_server_cls


def test_decode_rank0_binds_lookup_server(collaborators):
    mock_worker_cls, mock_server_cls = collaborators
    config = _make_vllm_config(rank=0)
    adapter = KVPoolWorkerAdapter(config, MagicMock())
    mock_worker_cls.assert_called_once()
    assert mock_worker_cls.call_args.kwargs["use_layerwise"] is False
    mock_server_cls.assert_called_once_with(mock_worker_cls.return_value, config)
    assert adapter._lookup_server is mock_server_cls.return_value


def test_nonzero_rank_does_not_bind(collaborators):
    mock_worker_cls, mock_server_cls = collaborators
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    mock_worker_cls.assert_called_once()
    mock_server_cls.assert_not_called()
    assert adapter._lookup_server is None


def test_worker_adapter_constructs_non_layerwise_pool_worker_only(collaborators):
    mock_worker_cls, _ = collaborators
    KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    worker = mock_worker_cls.return_value
    worker.register_kv_caches.assert_not_called()
    worker._start_kv_transfer_threads.assert_not_called()
    worker.start_load_kv.assert_not_called()


def test_register_kv_caches_delegates_exact_argument(collaborators):
    # Given
    mock_worker_cls, _ = collaborators
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    kv_caches = {"layer.0": MagicMock()}

    # When
    adapter.register_kv_caches(kv_caches)

    # Then
    delegated_kv_caches = mock_worker_cls.return_value.register_kv_caches.call_args.args[0]
    assert delegated_kv_caches is kv_caches


def test_start_load_kv_delegates_exact_metadata(collaborators):
    # Given
    mock_worker_cls, _ = collaborators
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    metadata = MagicMock()

    # When
    adapter.start_load_kv(metadata)

    # Then
    delegated_metadata = mock_worker_cls.return_value.start_load_kv.call_args.args[0]
    assert delegated_metadata is metadata


def test_get_finished_delegates_arguments_and_returns_exact_result(collaborators):
    # Given
    mock_worker_cls, _ = collaborators
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    finished_req_ids = {"req-finished"}
    metadata = MagicMock()
    expected = ({"req-sent"}, {"req-received"})
    mock_worker_cls.return_value.get_finished.return_value = expected

    # When
    result = adapter.get_finished(finished_req_ids, metadata)

    # Then
    assert result is expected
    mock_worker_cls.return_value.get_finished.assert_called_once_with(finished_req_ids, metadata)


def test_get_block_ids_with_load_errors_delegates_and_returns_exact_result(collaborators):
    # Given
    mock_worker_cls, _ = collaborators
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    expected = {7, 8}
    mock_worker_cls.return_value.get_block_ids_with_load_errors.return_value = expected

    # When
    result = adapter.get_block_ids_with_load_errors()

    # Then
    assert result is expected
    mock_worker_cls.return_value.get_block_ids_with_load_errors.assert_called_once_with()

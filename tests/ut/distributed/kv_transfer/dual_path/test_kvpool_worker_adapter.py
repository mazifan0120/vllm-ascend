# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Task-01 Decode ``KVPoolWorkerAdapter`` (spec §11.3).

The worker adapter is a lookup-only composition: a non-layerwise
``KVPoolWorker`` plus the existing ``LookupKeyServer`` bound only on the
owning rank. These tests patch both collaborators at the adapter namespace;
the lifecycle of a real server is covered by ``test_lookup_key_server_close``.
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


def test_worker_close_idempotent_and_delegates_to_server_close(collaborators):
    _, mock_server_cls = collaborators
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=0), MagicMock())
    adapter.close()
    adapter.close()
    mock_server_cls.return_value.close.assert_called_once()
    assert adapter._lookup_server is None


def test_worker_close_without_server_is_noop(collaborators):
    adapter = KVPoolWorkerAdapter(_make_vllm_config(rank=1), MagicMock())
    adapter.close()
    adapter.close()
    assert adapter._lookup_server is None

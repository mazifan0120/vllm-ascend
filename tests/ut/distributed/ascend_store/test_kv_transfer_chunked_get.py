# SPDX-License-Identifier: Apache-2.0
"""Chunked-get batching for the non-layerwise store receive thread."""

from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend  # noqa: E402


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

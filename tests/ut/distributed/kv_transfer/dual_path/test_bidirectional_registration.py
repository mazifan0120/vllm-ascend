from unittest.mock import patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path.conftest import worker_environment
from tests.ut.distributed.kv_transfer.dual_path.test_dual_path_connector import (
    MockKVCacheConfig,
    MockVllmConfig,
    make_kv_caches,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorWorker


def _make_worker(role: str) -> DualPathConnectorWorker:
    config = MockVllmConfig(role, "kv_producer" if role == "prefill" else "kv_consumer")
    config.parallel_config.tensor_parallel_size = 1
    return DualPathConnectorWorker(config, MockKVCacheConfig(), "test_engine", DualPathConfig(role=role))


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_dual_path_worker_repeated_runtime_ensure_is_idempotent(role: str) -> None:
    with (
        worker_environment() as runtime,
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker.KVPoolWorkerAdapter"),
    ):
        worker = _make_worker(role)
        worker.register_kv_caches(make_kv_caches())
        register_buffer_calls = runtime.register_buffer.call_count

        worker._ensure_send_layer_runtime()
        worker._ensure_receive_layer_runtime()
        worker._ensure_send_layer_runtime()
        worker._ensure_receive_layer_runtime()

    assert runtime.send_factory.call_count == 1
    assert runtime.recv_factory.call_count == 1
    assert runtime.register_buffer.call_count == register_buffer_calls == 1


def test_prefill_dual_path_worker_owns_exactly_one_send_and_one_receive_runtime() -> None:
    kv_caches = make_kv_caches()

    with worker_environment() as runtime:
        worker = _make_worker("prefill")
        worker.register_kv_caches(kv_caches)

    assert worker.kv_send_layer_thread is not None
    assert worker.kv_recv_layer_thread is not None
    assert runtime.send_factory.call_count == 1
    assert runtime.recv_factory.call_count == 1
    assert worker.kv_send_layer_thread.start.call_count == 1
    assert worker.kv_recv_layer_thread.start.call_count == 1
    assert worker._registered_layer_order == ((0, "encoder.layer.0"),)
    assert worker._registered_kv_caches is not None
    assert worker._registered_kv_caches["encoder.layer.0"][0] is kv_caches["encoder.layer.0"][0]
    assert worker._registered_kv_caches["encoder.layer.0"][1] is kv_caches["encoder.layer.0"][1]


def test_decode_dual_path_worker_owns_exactly_one_receive_and_one_send_runtime() -> None:
    kv_caches = make_kv_caches()

    with (
        worker_environment() as runtime,
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker.KVPoolWorkerAdapter"),
    ):
        worker = _make_worker("decode")
        worker.register_kv_caches(kv_caches)

    assert worker.kv_recv_layer_thread is not None
    assert worker.kv_send_layer_thread is not None
    assert runtime.recv_factory.call_count == 1
    assert runtime.send_factory.call_count == 1
    assert worker.kv_recv_layer_thread.start.call_count == 1
    assert worker.kv_send_layer_thread.start.call_count == 1
    assert worker._registered_layer_order == ((0, "encoder.layer.0"),)
    assert worker._registered_kv_caches is not None
    assert worker._registered_kv_caches["encoder.layer.0"][0] is kv_caches["encoder.layer.0"][0]
    assert worker._registered_kv_caches["encoder.layer.0"][1] is kv_caches["encoder.layer.0"][1]

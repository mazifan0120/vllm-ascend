# SPDX-License-Identifier: Apache-2.0
"""Lifecycle-safety tests for ``LookupKeyServer.close()`` (Task-01 §5.4).

The lookup protocol and response format are unchanged; these tests prove the
first explicit owner can stop the server cleanly: the daemon recv loop exits,
the endpoint is released, and repeated close() is a no-op. Real ZMQ sockets on
a pytest-tmp IPC base path are used throughout.
"""

import shutil
import tempfile
import threading
import time
from unittest.mock import MagicMock

import pytest
import zmq  # noqa: E402
from vllm.v1.serial_utils import MsgpackEncoder  # noqa: E402

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (  # noqa: E402
    LookupKeyServer,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (  # noqa: E402
    get_zmq_rpc_path_lookup,
)

_LOOKUP_HIT_TOKENS = 48


def _make_vllm_config() -> MagicMock:
    config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.kv_transfer_config.kv_connector_extra_config = {"lookup_rpc_port": 18881}
    return config


@pytest.fixture()
def rpc_base_path(monkeypatch):
    # macOS limits ipc paths to 103 chars; pytest's tmp_path is far deeper.
    base = tempfile.mkdtemp(prefix="lk", dir="/tmp")
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_RPC_BASE_PATH", base)
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture()
def lookup_server(rpc_base_path):
    pool_worker = MagicMock()
    pool_worker.lookup_scheduler.return_value = _LOOKUP_HIT_TOKENS
    server = LookupKeyServer(pool_worker, _make_vllm_config())
    yield server
    server.close()


def _client_lookup_once(socket_path: str, token_len: int, hash_strs: list[str], group_ids: list[int]) -> int:
    ctx = zmq.Context()
    try:
        socket = ctx.socket(zmq.REQ)
        socket.connect(socket_path)
        encoder = MsgpackEncoder()
        token_len_bytes = token_len.to_bytes(4, byteorder="big")
        group_frames = encoder.encode(group_ids)
        hash_frames = encoder.encode(hash_strs)
        socket.send_multipart([token_len_bytes, *group_frames, *hash_frames], copy=False)
        response = socket.recv()
        socket.close(linger=0)
        return int.from_bytes(response, "big")
    finally:
        ctx.term()


def test_close_joins_lookup_thread(lookup_server):
    assert lookup_server.thread.is_alive()
    lookup_server.close()
    assert not lookup_server.thread.is_alive()


def test_close_is_idempotent(lookup_server):
    lookup_server.close()
    lookup_server.close()
    assert not lookup_server.thread.is_alive()


def test_close_releases_endpoint(lookup_server):
    path = get_zmq_rpc_path_lookup(_make_vllm_config())
    lookup_server.close()
    ctx = zmq.Context()
    try:
        socket = ctx.socket(zmq.REP)
        socket.bind(path)  # raises ZMQError if the endpoint is still held
        socket.close(linger=0)
    finally:
        ctx.term()


def test_lookup_roundtrip_protocol_unchanged(lookup_server):
    path = get_zmq_rpc_path_lookup(_make_vllm_config())
    result = _client_lookup_once(path, 64, ["aa" * 32], [0])
    assert result == _LOOKUP_HIT_TOKENS
    pool_worker = lookup_server.pool_worker
    pool_worker.lookup_scheduler.assert_called_once()
    args, kwargs = pool_worker.lookup_scheduler.call_args
    assert args[0] == 64
    assert args[1] == ["aa" * 32]
    assert args[2] == [0]
    assert kwargs.get("use_layerwise") is False


def test_close_while_blocked_in_recv_returns_promptly(lookup_server):
    start = time.monotonic()
    lookup_server.close()
    assert time.monotonic() - start < 5.0
    assert not lookup_server.thread.is_alive()


def test_recv_loop_exits_without_raising_on_close(lookup_server):
    thread_errors: list[BaseException] = []
    original_excepthook = threading.excepthook

    def capture(args):
        if args.thread is lookup_server.thread:
            thread_errors.append(args.exc_value)
        original_excepthook(args)

    threading.excepthook = capture
    try:
        lookup_server.close()
    finally:
        threading.excepthook = original_excepthook
    assert not lookup_server.thread.is_alive()
    assert thread_errors == []


def test_close_during_in_flight_request_is_clean(rpc_base_path):
    lookup_started = threading.Event()
    release_lookup = threading.Event()

    def blocking_lookup(*args, **kwargs):
        lookup_started.set()
        release_lookup.wait(timeout=5)
        return _LOOKUP_HIT_TOKENS

    pool_worker = MagicMock()
    pool_worker.lookup_scheduler.side_effect = blocking_lookup
    server = LookupKeyServer(pool_worker, _make_vllm_config())
    thread_errors: list[BaseException] = []
    original_excepthook = threading.excepthook

    def capture(args):
        if args.thread is server.thread:
            thread_errors.append(args.exc_value)
        original_excepthook(args)

    threading.excepthook = capture
    try:
        client_ctx = zmq.Context()
        client = client_ctx.socket(zmq.REQ)
        client.setsockopt(zmq.RCVTIMEO, 2000)
        client.connect(get_zmq_rpc_path_lookup(_make_vllm_config()))
        encoder = MsgpackEncoder()
        client.send_multipart(
            [(64).to_bytes(4, "big"), *encoder.encode([0]), *encoder.encode(["aa" * 32])],
            copy=False,
        )
        assert lookup_started.wait(timeout=5)

        close_errors: list[BaseException] = []

        def do_close():
            try:
                server.close()
            except BaseException as exc:  # noqa: BLE001 - captured for assertion
                close_errors.append(exc)

        closer = threading.Thread(target=do_close)
        closer.start()
        time.sleep(0.2)  # let close() reach ctx.term() while the lookup is in flight
        release_lookup.set()
        closer.join(timeout=5)
        assert not closer.is_alive()
        assert close_errors == []
        assert not server.thread.is_alive()
        # The server thread must exit without an escaped exception regardless
        # of whether the in-flight reply raced the shutdown.
        assert thread_errors == []
        client.close(linger=0)
        client_ctx.term()
    finally:
        threading.excepthook = original_excepthook
        server.close()

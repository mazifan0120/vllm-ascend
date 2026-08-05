# SPDX-License-Identifier: Apache-2.0

import threading
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import TypeAlias
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (
    DualPathConnector,
    DualPathConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    Path,
    PathDecisionCommit,
    PathDecisionError,
    PathDecisionRequest,
    PathDecisionValidationError,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DUAL_PATH_PROTOCOL_VERSION,
    DecodeControlEndpoint,
    DualPathDecisionMetadata,
    PathDecision,
    PathDecisionCoordinator,
    PathDecisionDeliveryError,
    _deliver_decision,
    decode_path_decision,
    encode_path_decision,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)

_JsonValue: TypeAlias = str | int | float | bool | None | list["_JsonValue"] | dict[str, "_JsonValue"]
_JsonObject: TypeAlias = dict[str, _JsonValue]


def _request() -> PathDecisionRequest:
    return PathDecisionRequest(
        request_key=DualPathRequestKey(
            decode_engine_instance_id="decode-engine-1:0:boot-1",
            decode_request_id="request-1",
        ),
        target_tokens=32,
        decode_local_tokens=8,
        decode_store_tokens=16,
    )


def _key_payload() -> _JsonObject:
    return {
        "decode_engine_instance_id": "decode-engine-1:0:boot-1",
        "decode_request_id": "request-1",
    }


def _request_payload() -> _JsonObject:
    return {
        "request_key": _key_payload(),
        "target_tokens": 32,
        "decode_local_tokens": 8,
        "decode_store_tokens": 16,
    }


def _metadata() -> DualPathDecisionMetadata:
    return DualPathDecisionMetadata(
        protocol_version=DUAL_PATH_PROTOCOL_VERSION,
        decision_request=_request(),
        decode_control_endpoint=DecodeControlEndpoint(host="192.0.2.10", port=24001),
    )


def _metadata_payload() -> _JsonObject:
    return {
        "protocol_version": 1,
        "decision_request": _request_payload(),
        "decode_control_endpoint": {"host": "192.0.2.10", "port": 24001},
    }


def _commit_payload() -> _JsonObject:
    return {"request_key": _key_payload(), "path": "PE_READ"}


@pytest.mark.parametrize(
    ("host", "port"),
    [
        pytest.param("", 24001, id="empty-host"),
        pytest.param(123, 24001, id="non-string-host"),
        pytest.param("192.0.2.10", True, id="boolean-port"),
        pytest.param("192.0.2.10", 0, id="zero-port"),
        pytest.param("192.0.2.10", -1, id="negative-port"),
        pytest.param("192.0.2.10", 65536, id="port-above-maximum"),
        pytest.param("192.0.2.10", "24001", id="non-integer-port"),
    ],
)
def test_decode_control_endpoint_validates_host_and_port(host, port) -> None:
    with pytest.raises(PathDecisionValidationError):
        DecodeControlEndpoint(host=host, port=port)


def test_decode_control_endpoint_round_trip() -> None:
    endpoint = DecodeControlEndpoint(host="192.0.2.10", port=24001)
    payload = {"host": "192.0.2.10", "port": 24001}

    assert endpoint.to_dict() == payload
    assert DecodeControlEndpoint.from_dict(payload) == endpoint


def test_dual_path_decision_metadata_round_trip() -> None:
    metadata = _metadata()
    payload = _metadata_payload()

    assert metadata.to_dict() == payload
    assert DualPathDecisionMetadata.from_dict(payload) == metadata


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "protocol_version": 1,
                "decision_request": _request_payload(),
            },
            id="missing-outer-field",
        ),
        pytest.param(
            {**_metadata_payload(), "unexpected": None},
            id="extra-outer-field",
        ),
        pytest.param(
            {
                **_metadata_payload(),
                "decode_control_endpoint": {"host": "192.0.2.10"},
            },
            id="missing-nested-field",
        ),
        pytest.param(
            {
                **_metadata_payload(),
                "decision_request": {**_request_payload(), "unexpected": None},
            },
            id="extra-nested-field",
        ),
    ],
)
def test_dual_path_decision_metadata_rejects_non_exact_keys(payload: _JsonValue) -> None:
    with pytest.raises(PathDecisionValidationError):
        DualPathDecisionMetadata.from_dict(payload)


def test_path_decision_msgpack_round_trip_commit() -> None:
    decision = PathDecision(
        protocol_version=DUAL_PATH_PROTOCOL_VERSION,
        result=PathDecisionCommit(request_key=_request().request_key, path=Path.PE_READ),
    )
    expected_bytes = msgspec.msgpack.encode(
        {
            "protocol_version": 1,
            "result_type": "commit",
            "result": _commit_payload(),
        }
    )

    assert encode_path_decision(decision) == expected_bytes
    assert decode_path_decision(expected_bytes) == decision


def test_path_decision_msgpack_round_trip_error() -> None:
    decision = PathDecision(
        protocol_version=DUAL_PATH_PROTOCOL_VERSION,
        result=PathDecisionError(
            request_key=_request().request_key,
            error_code="INVALID_REQUEST",
            message="invalid decision input",
        ),
    )
    expected_bytes = msgspec.msgpack.encode(
        {
            "protocol_version": 1,
            "result_type": "error",
            "result": {
                "request_key": _key_payload(),
                "error_code": "INVALID_REQUEST",
                "message": "invalid decision input",
            },
        }
    )

    assert encode_path_decision(decision) == expected_bytes
    assert decode_path_decision(expected_bytes) == decision


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "protocol_version": 1,
                "result_type": "unknown",
                "result": {},
            },
            id="unknown-result-type",
        ),
        pytest.param(
            {
                "protocol_version": True,
                "result_type": "commit",
                "result": _commit_payload(),
            },
            id="boolean-version",
        ),
        pytest.param(
            {
                "protocol_version": "1",
                "result_type": "commit",
                "result": _commit_payload(),
            },
            id="non-integer-version",
        ),
    ],
)
def test_path_decision_from_dict_rejects_unknown_result_type_and_bad_version(payload: _JsonValue) -> None:
    with pytest.raises(PathDecisionValidationError):
        PathDecision.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"protocol_version": 1, "result_type": "commit"},
            id="missing-field",
        ),
        pytest.param(
            {
                "protocol_version": 1,
                "result_type": "commit",
                "result": _commit_payload(),
                "unexpected": None,
            },
            id="extra-field",
        ),
    ],
)
def test_path_decision_from_dict_rejects_non_exact_keys(payload: _JsonValue) -> None:
    with pytest.raises(PathDecisionValidationError):
        PathDecision.from_dict(payload)


def test_decode_path_decision_rejects_malformed_msgpack() -> None:
    with pytest.raises(PathDecisionValidationError):
        decode_path_decision(b"\x81")


def test_nested_dual_path_envelope_matches_kv_transfer_params_shape() -> None:
    kv_transfer_params = {
        "do_remote_decode": True,
        "remote_block_ids": [11, 12],
        "remote_host": "198.51.100.20",
        "remote_port": 25001,
        "dual_path": _metadata().to_dict(),
    }

    assert kv_transfer_params == {
        "do_remote_decode": True,
        "remote_block_ids": [11, 12],
        "remote_host": "198.51.100.20",
        "remote_port": 25001,
        "dual_path": {
            "protocol_version": 1,
            "decision_request": {
                "request_key": _key_payload(),
                "target_tokens": 32,
                "decode_local_tokens": 8,
                "decode_store_tokens": 16,
            },
            "decode_control_endpoint": {"host": "192.0.2.10", "port": 24001},
        },
    }


def test_fake_proxy_forwards_dual_path_envelope_transparently() -> None:
    kv_transfer_params: _JsonObject = {
        "do_remote_decode": True,
        "remote_block_ids": [11, 12],
        "remote_host": "198.51.100.20",
        "remote_port": 25001,
        "dual_path": _metadata().to_dict(),
    }

    def fake_proxy(params: _JsonObject) -> _JsonObject:
        forwarded_params = deepcopy(params)
        envelope = forwarded_params["dual_path"]
        assert DualPathDecisionMetadata.from_dict(envelope).to_dict() == envelope
        return forwarded_params

    forwarded_params = fake_proxy(kv_transfer_params)

    assert forwarded_params == kv_transfer_params
    assert forwarded_params["dual_path"] == _metadata_payload()


def _free_control_endpoint() -> DecodeControlEndpoint:
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    socket.close(linger=0)
    context.term()
    return DecodeControlEndpoint(host="127.0.0.1", port=port)


def _coordinator(endpoint: DecodeControlEndpoint, *, boot_id: str | None = "boot-1") -> PathDecisionCoordinator:
    return PathDecisionCoordinator.for_decode(
        engine_id="decode-engine-1",
        data_parallel_rank=0,
        control_endpoint=endpoint,
        boot_id=boot_id,
    )


def _decision(key: DualPathRequestKey, *, path: Path = Path.PE_READ) -> PathDecision:
    return PathDecision(
        protocol_version=DUAL_PATH_PROTOCOL_VERSION,
        result=PathDecisionCommit(request_key=key, path=path),
    )


def _error_decision(key: DualPathRequestKey) -> PathDecision:
    return PathDecision(
        protocol_version=DUAL_PATH_PROTOCOL_VERSION,
        result=PathDecisionError(
            request_key=key,
            error_code="INVALID_REQUEST",
            message="invalid decision input",
        ),
    )


def _raw_request(endpoint: DecodeControlEndpoint, payload: bytes, *, timeout_ms: int = 100) -> bytes | None:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{endpoint.host}:{endpoint.port}")
    try:
        socket.send(payload)
        if socket.poll(timeout_ms) == 0:
            return None
        return socket.recv()
    finally:
        socket.close(linger=0)
        context.term()


class _FakeDeliverySocket:
    def __init__(self, *, ack: bytes | None = None, blocked: threading.Event | None = None) -> None:
        self.ack = ack
        self.blocked = blocked
        self.send_timeouts: list[int] = []
        self.poll_timeouts: list[int] = []
        self.sent: list[bytes] = []
        self.closed = False

    def set_send_timeout(self, timeout_ms: int) -> None:
        self.send_timeouts.append(timeout_ms)

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)
        if self.blocked is not None:
            assert self.blocked.wait(timeout=5)

    def poll(self, timeout_ms: int) -> bool:
        self.poll_timeouts.append(timeout_ms)
        return self.ack is not None

    def recv(self) -> bytes:
        assert self.ack is not None
        return self.ack

    def close(self) -> None:
        self.closed = True


def test_boot_id_differs_across_coordinator_constructions() -> None:
    first = _coordinator(_free_control_endpoint(), boot_id=None)
    second = _coordinator(_free_control_endpoint(), boot_id=None)
    try:
        assert first.decode_engine_instance_id != second.decode_engine_instance_id
    finally:
        first.close()
        second.close()


def test_decode_engine_instance_id_is_stable_within_one_coordinator() -> None:
    coordinator = _coordinator(_free_control_endpoint())
    try:
        first = coordinator.decode_engine_instance_id
        assert coordinator.decode_engine_instance_id == first
    finally:
        coordinator.close()


def test_injected_boot_id_produces_deterministic_instance_id() -> None:
    coordinator = _coordinator(_free_control_endpoint(), boot_id="known-boot")
    try:
        assert coordinator.decode_engine_instance_id == "decode-engine-1:0:known-boot"
    finally:
        coordinator.close()


def test_instance_ids_separate_data_parallel_ranks() -> None:
    endpoint_a = _free_control_endpoint()
    endpoint_b = _free_control_endpoint()
    first = PathDecisionCoordinator.for_decode(
        engine_id="decode-engine-1",
        data_parallel_rank=0,
        control_endpoint=endpoint_a,
        boot_id="same-boot",
    )
    second = PathDecisionCoordinator.for_decode(
        engine_id="decode-engine-1",
        data_parallel_rank=1,
        control_endpoint=endpoint_b,
        boot_id="same-boot",
    )
    try:
        assert first.decode_engine_instance_id != second.decode_engine_instance_id
    finally:
        first.close()
        second.close()


def test_prefill_coordinator_construction_creates_no_sockets() -> None:
    calls = 0

    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        nonlocal calls
        calls += 1
        yield _FakeDeliverySocket(ack=b"ACK")

    coordinator = PathDecisionCoordinator.for_prefill(_socket_opener=opener)
    try:
        assert calls == 0
        with pytest.raises(RuntimeError):
            _ = coordinator.decode_control_endpoint
    finally:
        coordinator.close()


def test_prefill_submit_does_no_socket_io_on_caller_thread() -> None:
    entered = threading.Event()
    release = threading.Event()
    opener_thread_ids: list[int] = []

    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        opener_thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(timeout=5)
        yield _FakeDeliverySocket(ack=b"ACK")

    coordinator = PathDecisionCoordinator.for_prefill(_socket_opener=opener)
    caller_thread_id = threading.get_ident()
    try:
        future = coordinator.submit(_free_control_endpoint(), _decision(_request().request_key))
        assert entered.wait(timeout=5)
        assert not future.done()
        assert opener_thread_ids == [opener_thread_ids[0]]
        assert opener_thread_ids[0] != caller_thread_id
        release.set()
        assert future.result(timeout=5) is None
    finally:
        release.set()
        coordinator.close()


def test_commit_delivery_round_trip_enqueues_once_and_returns_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    sender = PathDecisionCoordinator.for_prefill()
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert sender.submit(endpoint, _decision(key)).result(timeout=5) is None
        assert receiver.take_decisions() == [_decision(key).result]
        assert receiver.take_decisions() == []
    finally:
        sender.close()
        receiver.close()


def test_receiver_ack_frame_is_exactly_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == b"ACK"
    finally:
        receiver.close()


def test_error_decision_delivery_is_accepted_enqueued_and_acked() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    sender = PathDecisionCoordinator.for_prefill()
    key = _request().request_key
    try:
        receiver.register_pending(key)
        decision = _error_decision(key)
        assert sender.submit(endpoint, decision).result(timeout=5) is None
        assert receiver.take_decisions() == [decision.result]
    finally:
        sender.close()
        receiver.close()


def test_identical_redelivery_is_acked_without_duplicate_inbox() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    encoded = encode_path_decision(_decision(key))
    context = zmq.Context()
    first = context.socket(zmq.REQ)
    first.setsockopt(zmq.LINGER, 100)
    first.connect(f"tcp://{endpoint.host}:{endpoint.port}")
    try:
        receiver.register_pending(key)
        first.send(encoded)
        assert first.poll(1000) != 0
        first.close(linger=0)
        assert _raw_request(endpoint, encoded) == b"ACK"
        assert receiver.take_decisions() == [_decision(key).result]
        assert receiver.take_decisions() == []
    finally:
        first.close(linger=0)
        context.term()
        receiver.close()


def test_submit_retries_identical_bytes_until_ack() -> None:
    endpoint = _free_control_endpoint()
    captured: list[bytes] = []
    ready = threading.Event()

    def scripted_router() -> None:
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(f"tcp://{endpoint.host}:{endpoint.port}")
        ready.set()
        try:
            for attempt in range(2):
                identity, delimiter, payload = socket.recv_multipart()
                assert delimiter == b""
                captured.append(payload)
                if attempt == 1:
                    socket.send_multipart([identity, b"", b"ACK"])
        finally:
            socket.close(linger=0)
            context.term()

    thread = threading.Thread(target=scripted_router, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    sender = PathDecisionCoordinator.for_prefill(
        _send_timeout_ms=50,
        _poll_timeout_ms=50,
        _retry_spacing_s=0.01,
    )
    decision = _decision(_request().request_key)
    try:
        assert sender.submit(endpoint, decision).result(timeout=5) is None
        assert captured == [encode_path_decision(decision), encode_path_decision(decision)]
    finally:
        sender.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_conflicting_duplicate_gets_no_ack_and_no_inbox_growth() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == b"ACK"
        assert _raw_request(endpoint, encode_path_decision(_decision(key, path=Path.DE_READ))) is None
        assert receiver.take_decisions() == [_decision(key).result]
        assert receiver.take_decisions() == []
    finally:
        receiver.close()


def test_unknown_key_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    try:
        assert _raw_request(endpoint, encode_path_decision(_decision(_request().request_key))) is None
        assert receiver.take_decisions() == []
    finally:
        receiver.close()


def test_wrong_incarnation_key_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    pending = _request().request_key
    wrong = DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1:0:other-boot",
        decode_request_id=pending.decode_request_id,
    )
    try:
        receiver.register_pending(pending)
        assert _raw_request(endpoint, encode_path_decision(_decision(wrong))) is None
        assert receiver.take_decisions() == []
    finally:
        receiver.close()


def test_registered_wrong_incarnation_key_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint, boot_id="correct")
    wrong = DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1:0:wrong",
        decode_request_id="request-1",
    )
    try:
        receiver.register_pending(wrong)
        assert _raw_request(endpoint, encode_path_decision(_decision(wrong))) is None
        assert receiver.take_decisions() == []
        assert wrong not in receiver._accepted_results
    finally:
        receiver.close()


def test_unsupported_protocol_version_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    payload = _decision(key).to_dict()
    payload["protocol_version"] = 2
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, msgspec.msgpack.encode(payload)) is None
        assert receiver.take_decisions() == []
    finally:
        receiver.close()


def test_malformed_payload_gets_no_ack_and_receiver_survives() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, b"\x81") is None
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == b"ACK"
        assert receiver.take_decisions() == [_decision(key).result]
    finally:
        receiver.close()


def test_delivery_makes_exactly_three_attempts_with_fresh_req_sockets() -> None:
    sockets: list[_FakeDeliverySocket] = []

    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        socket = _FakeDeliverySocket()
        sockets.append(socket)
        yield socket

    with pytest.raises(PathDecisionDeliveryError):
        _deliver_decision(
            b"encoded",
            _free_control_endpoint(),
            opener=opener,
            sleep=lambda _: None,
            should_stop=lambda: False,
            send_timeout_ms=1000,
            poll_timeout_ms=1000,
            retry_spacing_s=0.1,
        )

    assert len(sockets) == 3
    assert len({id(socket) for socket in sockets}) == 3
    assert all(socket.sent == [b"encoded"] for socket in sockets)


def test_delivery_uses_spec_send_timeout_poll_bound_and_retry_spacing() -> None:
    sockets: list[_FakeDeliverySocket] = []
    sleeps: list[float] = []

    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        socket = _FakeDeliverySocket()
        sockets.append(socket)
        yield socket

    coordinator = PathDecisionCoordinator.for_prefill(_socket_opener=opener, _sleep=sleeps.append)
    try:
        future = coordinator.submit(_free_control_endpoint(), _decision(_request().request_key))
        with pytest.raises(PathDecisionDeliveryError):
            future.result(timeout=5)
        assert [socket.send_timeouts for socket in sockets] == [[1000], [1000], [1000]]
        assert [socket.poll_timeouts for socket in sockets] == [[1000], [1000], [1000]]
        assert sleeps == [0.1, 0.1]
    finally:
        coordinator.close()


def test_delivery_exhaustion_raises_typed_transport_error() -> None:
    calls = 0

    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        nonlocal calls
        calls += 1
        yield _FakeDeliverySocket()

    coordinator = PathDecisionCoordinator.for_prefill(_socket_opener=opener, _sleep=lambda _: None)
    try:
        with pytest.raises(PathDecisionDeliveryError):
            coordinator.submit(_free_control_endpoint(), _decision(_request().request_key)).result(timeout=5)
        assert calls == 3
    finally:
        coordinator.close()


def test_successful_delivery_completes_future_with_none() -> None:
    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        yield _FakeDeliverySocket(ack=b"ACK")

    coordinator = PathDecisionCoordinator.for_prefill(_socket_opener=opener)
    try:
        assert coordinator.submit(_free_control_endpoint(), _decision(_request().request_key)).result(timeout=5) is None
    finally:
        coordinator.close()


def test_concurrent_submissions_stay_isolated() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    sender = PathDecisionCoordinator.for_prefill()
    keys = [DualPathRequestKey(receiver.decode_engine_instance_id, f"request-{index}") for index in range(8)]
    decisions = [_decision(key) if index % 2 == 0 else _error_decision(key) for index, key in enumerate(keys)]
    try:
        for key in keys:
            receiver.register_pending(key)
        futures = [sender.submit(endpoint, decision) for decision in decisions]
        assert [future.result(timeout=5) for future in futures] == [None] * 8
        assert set(receiver.take_decisions()) == {decision.result for decision in decisions}
        assert receiver.take_decisions() == []
    finally:
        sender.close()
        receiver.close()


def test_register_pending_is_idempotent() -> None:
    coordinator = _coordinator(_free_control_endpoint())
    key = _request().request_key
    try:
        coordinator.register_pending(key)
        coordinator.register_pending(key)
        assert coordinator._pending_keys == {key}
    finally:
        coordinator.close()


def test_unregister_removes_key_from_pending_and_accepted_registries() -> None:
    endpoint = _free_control_endpoint()
    coordinator = _coordinator(endpoint)
    key = _request().request_key
    try:
        coordinator.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == b"ACK"
        coordinator.unregister(key)
        assert key not in coordinator._pending_keys
        assert key not in coordinator._accepted_results
    finally:
        coordinator.close()


def test_unregister_is_idempotent_for_unknown_keys() -> None:
    coordinator = _coordinator(_free_control_endpoint())
    key = _request().request_key
    try:
        coordinator.unregister(key)
        coordinator.unregister(key)
        assert coordinator.take_decisions() == []
    finally:
        coordinator.close()


def test_register_and_submit_after_close_are_rejected() -> None:
    receiver = _coordinator(_free_control_endpoint())
    sender = PathDecisionCoordinator.for_prefill()
    receiver.close()
    sender.close()

    with pytest.raises(RuntimeError):
        receiver.register_pending(_request().request_key)
    with pytest.raises(RuntimeError):
        sender.submit(_free_control_endpoint(), _decision(_request().request_key))
    receiver.unregister(_request().request_key)
    assert receiver.take_decisions() == []


def test_close_is_idempotent() -> None:
    receiver = _coordinator(_free_control_endpoint())
    sender = PathDecisionCoordinator.for_prefill()
    receiver.close()
    receiver.close()
    sender.close()
    sender.close()


def test_close_leaves_no_threads_sockets_futures_or_retained_state() -> None:
    endpoint = _free_control_endpoint()
    baseline_threads = {thread.ident for thread in threading.enumerate()}
    coordinator = _coordinator(endpoint)
    key = _request().request_key
    coordinator.register_pending(key)
    assert _raw_request(endpoint, encode_path_decision(_decision(key))) == b"ACK"

    coordinator.close()

    assert not coordinator._receiver_thread.is_alive()
    assert coordinator._pending_keys == set()
    assert coordinator._accepted_results == {}
    assert coordinator.take_decisions() == []
    assert {thread.ident for thread in threading.enumerate()} == baseline_threads
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(f"tcp://{endpoint.host}:{endpoint.port}")
    finally:
        socket.close(linger=0)
        context.term()


def test_prefill_close_cancels_outstanding_futures_and_stops_executor() -> None:
    entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    entered_count = 0

    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        nonlocal entered_count
        with count_lock:
            entered_count += 1
            if entered_count == 32:
                entered.set()
        yield _FakeDeliverySocket(blocked=release)

    coordinator = PathDecisionCoordinator.for_prefill(_socket_opener=opener, _sleep=lambda _: None)
    futures = [coordinator.submit(_free_control_endpoint(), _decision(_request().request_key)) for _ in range(40)]
    assert entered.wait(timeout=5)
    close_thread = threading.Thread(target=coordinator.close)
    close_thread.start()
    deadline = time.monotonic() + 5
    while not coordinator._closed and time.monotonic() < deadline:
        time.sleep(0.001)
    release.set()
    close_thread.join(timeout=5)

    assert not close_thread.is_alive()
    assert any(future.cancelled() for future in futures)
    for future in futures:
        if future.cancelled():
            with pytest.raises(CancelledError):
                future.result()
        else:
            with pytest.raises(PathDecisionDeliveryError):
                future.result(timeout=5)
    with pytest.raises(RuntimeError):
        coordinator.submit(_free_control_endpoint(), _decision(_request().request_key))


def _make_kv_transfer_config(kv_role: str) -> SimpleNamespace:
    return SimpleNamespace(
        kv_role=kv_role,
        is_kv_producer=kv_role in {"kv_producer", "kv_both"},
        is_kv_consumer=kv_role in {"kv_consumer", "kv_both"},
    )


class TestDualPathControlPortConfig:
    def test_decode_role_requires_dual_path_control_port(self) -> None:
        with pytest.raises(ValueError, match=r"dual_path_control_port"):
            DualPathConfig.from_extra_config({"role": "decode"}, _make_kv_transfer_config("kv_consumer"))

    @pytest.mark.parametrize("port", [True, "7100"])
    def test_dual_path_control_port_rejects_bool_and_non_int_values(self, port) -> None:
        with pytest.raises(ValueError, match=r"dual_path_control_port"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "dual_path_control_port": port},
                _make_kv_transfer_config("kv_consumer"),
            )

    @pytest.mark.parametrize("port", [0, -1, 65536])
    def test_dual_path_control_port_rejects_out_of_range_values(self, port) -> None:
        with pytest.raises(ValueError, match=r"dual_path_control_port"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "dual_path_control_port": port},
                _make_kv_transfer_config("kv_consumer"),
            )

    def test_prefill_role_does_not_require_dual_path_control_port(self) -> None:
        config = DualPathConfig.from_extra_config({"role": "prefill"}, _make_kv_transfer_config("kv_producer"))
        assert config.dual_path_control_port is None

    def test_decode_role_accepts_valid_dual_path_control_port(self) -> None:
        config = DualPathConfig.from_extra_config(
            {"role": "decode", "dual_path_control_port": 7100},
            _make_kv_transfer_config("kv_consumer"),
        )
        assert config.dual_path_control_port == 7100


def _make_scheduler_vllm_config(
    *,
    dual_role: str,
    kv_role: str,
    dual_path_control_port: int | None = None,
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    kv_port: int = 5000,
):
    config = MagicMock()
    config.speculative_config = None
    config.quant_config = None
    config.model_config.use_mla = True
    config.model_config.is_deepseek_mla = True
    config.model_config.hf_config.num_key_value_heads = 1
    config.model_config.hf_text_config.model_type = "default"
    config.model_config.hf_text_config.num_key_value_heads = 1
    config.model_config.get_num_layers.return_value = 1
    config.model_config.get_total_num_hidden_layers.return_value = 1
    config.model_config.get_total_num_kv_heads.return_value = 1
    config.parallel_config.tensor_parallel_size = tensor_parallel_size
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.data_parallel_rank_local = data_parallel_rank
    config.parallel_config.data_parallel_size_local = data_parallel_size
    config.parallel_config.data_parallel_size = data_parallel_size
    config.parallel_config.data_parallel_rank = data_parallel_rank
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1
    config.cache_config.block_size = 16
    config.cache_config.mamba_cache_mode = None
    config.scheduler_config.disable_hybrid_kv_cache_manager = True
    config.kv_transfer_config.engine_id = "test_engine"
    config.kv_transfer_config.kv_port = kv_port
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.is_kv_producer = kv_role in {"kv_producer", "kv_both"}
    config.kv_transfer_config.is_kv_consumer = kv_role in {"kv_consumer", "kv_both"}
    extra_config = {"role": dual_role}
    if dual_path_control_port is not None:
        extra_config["dual_path_control_port"] = dual_path_control_port
    config.kv_transfer_config.kv_connector_extra_config = extra_config
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {
        "tls_config": {},
        "prefill": {"tp_size": tensor_parallel_size, "dp_size": data_parallel_size},
        "decode": {"tp_size": tensor_parallel_size, "dp_size": data_parallel_size},
    }.get(key, default)
    return config


def _make_scheduler_kv_cache_config():
    kv_cache_spec = MagicMock()
    kv_cache_spec.block_size = 16
    group_spec = MagicMock()
    group_spec.kv_cache_spec = kv_cache_spec
    group_spec.layer_names = ["encoder.layer.0"]
    return SimpleNamespace(kv_cache_groups=[group_spec], kv_cache_tensors=[], num_blocks=10)


def test_derive_decode_control_port_adds_data_parallel_rank() -> None:
    assert (
        path_decision_channel.derive_decode_control_port(
            dual_path_control_port=7100,
            data_parallel_rank=3,
            kv_port=5000,
            worker_port_span=8,
        )
        == 7103
    )


@pytest.mark.parametrize(
    ("dual_path_control_port", "data_parallel_rank"),
    [(0, 0), (65535, 1)],
)
def test_derive_decode_control_port_rejects_out_of_range_derivation(
    dual_path_control_port: int,
    data_parallel_rank: int,
) -> None:
    with pytest.raises(ValueError):
        path_decision_channel.derive_decode_control_port(
            dual_path_control_port=dual_path_control_port,
            data_parallel_rank=data_parallel_rank,
            kv_port=5000,
            worker_port_span=8,
        )


def test_derive_decode_control_port_rejects_worker_kv_port_range_overlap() -> None:
    with pytest.raises(ValueError):
        path_decision_channel.derive_decode_control_port(
            dual_path_control_port=5000,
            data_parallel_rank=2,
            kv_port=5000,
            worker_port_span=4,
        )


def test_scheduler_constructs_role_specific_coordinator() -> None:
    data_parallel_rank = 1
    derived_port = _free_control_endpoint().port
    control_port = derived_port - data_parallel_rank
    decode_config = _make_scheduler_vllm_config(
        dual_role="decode",
        kv_role="kv_consumer",
        dual_path_control_port=control_port,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=2,
        tensor_parallel_size=2,
        kv_port=1,
    )
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.get_ip",
            return_value="127.0.0.1",
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.KVPoolAdapter"),
    ):
        decode_scheduler = DualPathConnectorScheduler(
            decode_config,
            _make_scheduler_kv_cache_config(),
            "test_engine",
            DualPathConfig(role="decode", dual_path_control_port=control_port),
        )
    try:
        decode_coordinator = decode_scheduler._path_decision_coordinator
        assert isinstance(decode_coordinator, PathDecisionCoordinator)
        assert decode_coordinator.decode_control_endpoint == DecodeControlEndpoint(
            host="127.0.0.1",
            port=derived_port,
        )
        assert decode_coordinator.decode_engine_instance_id.startswith(f"test_engine:{data_parallel_rank}:")
    finally:
        decode_scheduler.shutdown()
        decode_scheduler.executor.shutdown(wait=False)
        decode_scheduler.metaserver_client.close()

    prefill_scheduler = DualPathConnectorScheduler(
        _make_scheduler_vllm_config(dual_role="prefill", kv_role="kv_producer"),
        _make_scheduler_kv_cache_config(),
        "test_engine",
        DualPathConfig(role="prefill"),
    )
    try:
        prefill_coordinator = prefill_scheduler._path_decision_coordinator
        assert isinstance(prefill_coordinator, PathDecisionCoordinator)
        assert callable(prefill_coordinator.submit)
        assert prefill_coordinator._receiver_thread is None
        with pytest.raises(RuntimeError):
            _ = prefill_coordinator.decode_control_endpoint
    finally:
        prefill_scheduler.shutdown()
        prefill_scheduler.executor.shutdown(wait=False)
        prefill_scheduler.metaserver_client.close()


def test_scheduler_shutdown_closes_coordinator_idempotently() -> None:
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.PathDecisionCoordinator"
    ) as coordinator_cls:
        scheduler = DualPathConnectorScheduler(
            _make_scheduler_vllm_config(dual_role="prefill", kv_role="kv_producer"),
            _make_scheduler_kv_cache_config(),
            "test_engine",
            DualPathConfig(role="prefill"),
        )
    coordinator = coordinator_cls.for_prefill.return_value
    try:
        scheduler.shutdown()
        scheduler.shutdown()
        assert coordinator.close.call_count == 2
    finally:
        scheduler.executor.shutdown(wait=False)
        scheduler.metaserver_client.close()


def test_facade_shutdown_delegates_coordinator_close() -> None:
    config = _make_scheduler_vllm_config(
        dual_role="decode",
        kv_role="kv_consumer",
        dual_path_control_port=7100,
        kv_port=5000,
    )
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.get_ip",
            return_value="127.0.0.1",
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.KVPoolAdapter"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector.PathDecisionCoordinator"
        ) as coordinator_cls,
        patch.object(MooncakeLayerwiseConnector, "shutdown", autospec=True) as base_shutdown,
    ):
        connector = DualPathConnector(config, KVConnectorRole.SCHEDULER, _make_scheduler_kv_cache_config())
        try:
            connector.shutdown()
            coordinator_cls.for_decode.return_value.close.assert_called_once_with()
            base_shutdown.assert_called_once_with(connector)
        finally:
            scheduler = connector.connector_scheduler
            assert scheduler is not None
            scheduler.executor.shutdown(wait=False)
            scheduler.metaserver_client.close()

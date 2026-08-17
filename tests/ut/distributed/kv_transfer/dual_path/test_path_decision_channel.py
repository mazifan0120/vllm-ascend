# SPDX-License-Identifier: Apache-2.0

import threading
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
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
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import ReversePlan
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    JsonObject,
    PathAbortNotice,
    PathAbortReason,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecisionReplyStatus,
    DecodeControlEndpoint,
    DualPathDecisionMetadata,
    PathDecision,
    PathDecisionCoordinator,
    PathDecisionDeliveryError,
    _deliver_decision,
    decode_path_decision,
    encode_decision_reply,
    encode_path_abort,
    encode_path_decision,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)


def _request() -> PathDecisionRequest:
    return PathDecisionRequest(
        request_key=DualPathRequestKey(
            decode_engine_instance_id="decode-engine-1:0:boot-1",
            decode_request_id="request-1",
            admission_id=0,
        ),
        target_tokens=32,
        decode_local_tokens=8,
        decode_store_tokens=16,
    )


def _key_payload() -> JsonObject:
    return {
        "decode_engine_instance_id": "decode-engine-1:0:boot-1",
        "decode_request_id": "request-1",
        "admission_id": 0,
    }


def _request_payload() -> JsonObject:
    return {
        "request_key": _key_payload(),
        "target_tokens": 32,
        "decode_local_tokens": 8,
        "decode_store_tokens": 16,
    }


def _metadata() -> DualPathDecisionMetadata:
    return DualPathDecisionMetadata(
        decision_request=_request(),
        decode_control_endpoint=DecodeControlEndpoint(host="192.0.2.10", port=24001),
    )


def _metadata_payload() -> JsonObject:
    return {
        "decision_request": _request_payload(),
        "decode_control_endpoint": {"host": "192.0.2.10", "port": 24001},
    }


_ACK_BYTES = encode_decision_reply(DecisionReplyStatus.ACK)
_UNKNOWN_REQUEST_BYTES = encode_decision_reply(DecisionReplyStatus.UNKNOWN_REQUEST)
_PROTOCOL_ERROR_BYTES = encode_decision_reply(DecisionReplyStatus.PROTOCOL_ERROR)


def _result_payload() -> JsonObject:
    return {"request_key": _key_payload(), "path": "PE_READ"}


def _reverse_plan(key: DualPathRequestKey) -> ReversePlan:
    return ReversePlan(
        request_key=key,
        wire_request_id="wire-request-1",
        token_start=0,
        token_end=16,
        source_block_ids=((10,),),
        destination_block_ids=((20,),),
        remote_engine_id="prefill-engine-1",
        remote_host="192.0.2.10",
        remote_port=5000,
        remote_block_sizes=(16,),
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        reverse_attempt_id=0,
        prefill_local_tokens=0,
        reverse_send_completion_id=None,
    )


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


def test_path_decision_msgpack_round_trip_result() -> None:
    decision = PathDecision(
        result=PathDecisionResult(request_key=_request().request_key, path=PathKind.PE_READ),
        reverse_plan=None,
    )
    expected_bytes = msgspec.msgpack.encode(
        {
            "kind": "Decision",
            "payload": {
                "result": _result_payload(),
                "reverse_plan": None,
            },
        }
    )

    assert encode_path_decision(decision) == expected_bytes
    assert decode_path_decision(expected_bytes) == decision


def test_decision_request_and_result_round_trip() -> None:
    metadata = _metadata()
    key = metadata.decision_request.request_key
    decision = PathDecision(
        result=PathDecisionResult(
            request_key=key,
            path=PathKind.DE_READ,
            reverse_attempt_id=0,
            prefill_local_tokens=0,
        ),
        reverse_plan=_reverse_plan(key),
    )

    assert set(metadata.to_dict()) == {
        "decision_request",
        "decode_control_endpoint",
    }
    assert set(decision.to_dict()) == {"result", "reverse_plan"}
    assert decode_path_decision(encode_path_decision(decision)) == decision


def test_pe_read_serializes_none_reverse_plan() -> None:
    decision = PathDecision(
        result=PathDecisionResult(request_key=_request().request_key, path=PathKind.PE_READ),
        reverse_plan=None,
    )

    assert decision.to_dict()["reverse_plan"] is None


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
    kv_transfer_params: JsonObject = {
        "do_remote_decode": True,
        "remote_block_ids": [11, 12],
        "remote_host": "198.51.100.20",
        "remote_port": 25001,
        "dual_path": _metadata().to_dict(),
    }

    def fake_proxy(params: JsonObject) -> JsonObject:
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


def _decision(key: DualPathRequestKey, *, path: PathKind = PathKind.PE_READ) -> PathDecision:
    return PathDecision(
        result=(
            PathDecisionResult(request_key=key, path=path)
            if path is PathKind.PE_READ
            else PathDecisionResult(
                request_key=key,
                path=path,
                reverse_attempt_id=0,
                prefill_local_tokens=0,
            )
        ),
        reverse_plan=_reverse_plan(key) if path is PathKind.DE_READ else None,
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


def test_decode_coordinator_mints_monotonic_admission_keys() -> None:
    coordinator = _coordinator(_free_control_endpoint())
    try:
        first = coordinator.new_request_key("decode-request-7")
        second = coordinator.new_request_key("decode-request-7")

        assert first.decode_engine_instance_id == second.decode_engine_instance_id
        assert first.decode_request_id == second.decode_request_id == "decode-request-7"
        assert (first.admission_id, second.admission_id) == (0, 1)
        assert first != second
    finally:
        coordinator.close()


def test_prefill_coordinator_cannot_mint_decode_admission_key() -> None:
    coordinator = PathDecisionCoordinator.for_prefill()
    try:
        with pytest.raises(RuntimeError, match="Decode coordinator"):
            coordinator.new_request_key("decode-request-7")
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
        yield _FakeDeliverySocket(ack=_ACK_BYTES)

    coordinator = PathDecisionCoordinator.for_prefill(socket_opener=opener)
    try:
        assert calls == 0
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
        yield _FakeDeliverySocket(ack=_ACK_BYTES)

    coordinator = PathDecisionCoordinator.for_prefill(socket_opener=opener)
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


def test_result_delivery_round_trip_enqueues_once_and_returns_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    sender = PathDecisionCoordinator.for_prefill()
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert sender.submit(endpoint, _decision(key)).result(timeout=5) is None
        assert receiver.take_received_decisions() == [_decision(key)]
        assert receiver.take_received_decisions() == []
    finally:
        sender.close()
        receiver.close()


def test_queue_returns_complete_path_decision() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    decision = _decision(key)
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(decision)) == _ACK_BYTES
        assert receiver.take_received_decisions() == [decision]
    finally:
        receiver.close()


def test_receiver_ack_frame_is_exactly_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == _ACK_BYTES
    finally:
        receiver.close()


def test_identical_redelivery_is_acked_without_duplicate_received_result() -> None:
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
        assert _raw_request(endpoint, encoded) == _ACK_BYTES
        assert receiver.take_received_decisions() == [_decision(key)]
        assert receiver.take_received_decisions() == []
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
                    socket.send_multipart([identity, b"", _ACK_BYTES])
        finally:
            socket.close(linger=0)
            context.term()

    thread = threading.Thread(target=scripted_router, daemon=True)
    thread.start()
    assert ready.wait(timeout=5)
    sender = PathDecisionCoordinator.for_prefill(
        send_timeout_ms=50,
        poll_timeout_ms=50,
        retry_spacing_s=0.01,
    )
    decision = _decision(_request().request_key)
    try:
        assert sender.submit(endpoint, decision).result(timeout=5) is None
        assert captured == [encode_path_decision(decision), encode_path_decision(decision)]
    finally:
        sender.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_conflicting_duplicate_gets_no_ack_and_no_received_result_growth() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == _ACK_BYTES
        assert (
            _raw_request(endpoint, encode_path_decision(_decision(key, path=PathKind.DE_READ))) == _PROTOCOL_ERROR_BYTES
        )
        assert receiver.take_received_decisions() == [_decision(key)]
        assert receiver.take_received_decisions() == []
    finally:
        receiver.close()


def test_identical_duplicate_accepted_once_conflict_rejected() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    decision = _decision(key, path=PathKind.DE_READ)
    conflicting = PathDecision(
        result=decision.result,
        reverse_plan=replace(decision.reverse_plan, remote_port=5001),
    )
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(decision)) == _ACK_BYTES
        assert _raw_request(endpoint, encode_path_decision(decision)) == _ACK_BYTES
        assert _raw_request(endpoint, encode_path_decision(conflicting)) == _PROTOCOL_ERROR_BYTES
        assert receiver.take_received_decisions() == [decision]
        assert receiver.take_received_decisions() == []
    finally:
        receiver.close()


def test_de_read_requires_serialized_reverse_plan() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    without_plan = PathDecision(
        result=PathDecisionResult(
            request_key=key,
            path=PathKind.DE_READ,
            reverse_attempt_id=0,
            prefill_local_tokens=0,
        ),
        reverse_plan=None,
    )
    other_key = DualPathRequestKey(key.decode_engine_instance_id, "request-2", 0)
    mismatched_plan = PathDecision(
        result=PathDecisionResult(
            request_key=key,
            path=PathKind.DE_READ,
            reverse_attempt_id=0,
            prefill_local_tokens=0,
        ),
        reverse_plan=_reverse_plan(other_key),
    )
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, encode_path_decision(without_plan)) == _PROTOCOL_ERROR_BYTES
        assert _raw_request(endpoint, encode_path_decision(mismatched_plan)) == _PROTOCOL_ERROR_BYTES
        assert receiver.take_received_decisions() == []
    finally:
        receiver.close()


def test_unknown_key_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    try:
        assert _raw_request(endpoint, encode_path_decision(_decision(_request().request_key))) == _UNKNOWN_REQUEST_BYTES
        assert receiver.take_received_decisions() == []
    finally:
        receiver.close()


def test_wrong_incarnation_key_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    pending = _request().request_key
    wrong = DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1:0:other-boot",
        decode_request_id=pending.decode_request_id,
        admission_id=pending.admission_id,
    )
    try:
        receiver.register_pending(pending)
        assert _raw_request(endpoint, encode_path_decision(_decision(wrong))) == _UNKNOWN_REQUEST_BYTES
        assert receiver.take_received_decisions() == []
    finally:
        receiver.close()


def test_registered_wrong_incarnation_key_gets_no_ack() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint, boot_id="correct")
    wrong = DualPathRequestKey(
        decode_engine_instance_id="decode-engine-1:0:wrong",
        decode_request_id="request-1",
        admission_id=0,
    )
    try:
        receiver.register_pending(wrong)
        assert _raw_request(endpoint, encode_path_decision(_decision(wrong))) == _UNKNOWN_REQUEST_BYTES
        assert receiver.take_received_decisions() == []
        assert wrong not in receiver._accepted_decisions
    finally:
        receiver.close()


def test_malformed_payload_gets_no_ack_and_receiver_survives() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    key = _request().request_key
    try:
        receiver.register_pending(key)
        assert _raw_request(endpoint, b"\x81") is None
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == _ACK_BYTES
        assert receiver.take_received_decisions() == [_decision(key)]
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

    coordinator = PathDecisionCoordinator.for_prefill(socket_opener=opener, sleep=sleeps.append)
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

    coordinator = PathDecisionCoordinator.for_prefill(socket_opener=opener, sleep=lambda _: None)
    try:
        with pytest.raises(PathDecisionDeliveryError):
            coordinator.submit(_free_control_endpoint(), _decision(_request().request_key)).result(timeout=5)
        assert calls == 3
    finally:
        coordinator.close()


def test_successful_delivery_completes_future_with_none() -> None:
    @contextmanager
    def opener(endpoint: DecodeControlEndpoint):
        yield _FakeDeliverySocket(ack=_ACK_BYTES)

    coordinator = PathDecisionCoordinator.for_prefill(socket_opener=opener)
    try:
        assert coordinator.submit(_free_control_endpoint(), _decision(_request().request_key)).result(timeout=5) is None
    finally:
        coordinator.close()


def test_concurrent_submissions_stay_isolated() -> None:
    endpoint = _free_control_endpoint()
    receiver = _coordinator(endpoint)
    sender = PathDecisionCoordinator.for_prefill()
    keys = [DualPathRequestKey(receiver.decode_engine_instance_id, f"request-{index}", 0) for index in range(8)]
    decisions = [
        _decision(key, path=PathKind.PE_READ if index % 2 == 0 else PathKind.DE_READ) for index, key in enumerate(keys)
    ]
    try:
        for key in keys:
            receiver.register_pending(key)
        futures = [sender.submit(endpoint, decision) for decision in decisions]
        assert [future.result(timeout=5) for future in futures] == [None] * 8
        assert set(receiver.take_received_decisions()) == set(decisions)
        assert receiver.take_received_decisions() == []
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
        assert _raw_request(endpoint, encode_path_decision(_decision(key))) == _ACK_BYTES
        coordinator.unregister(key)
        assert key not in coordinator._pending_keys
        assert key not in coordinator._accepted_decisions
    finally:
        coordinator.close()


def test_unregister_is_idempotent_for_unknown_keys() -> None:
    coordinator = _coordinator(_free_control_endpoint())
    key = _request().request_key
    try:
        coordinator.unregister(key)
        coordinator.unregister(key)
        assert coordinator.take_received_decisions() == []
    finally:
        coordinator.close()


def test_submit_after_close_is_rejected_and_decode_close_is_inert() -> None:
    receiver = _coordinator(_free_control_endpoint())
    sender = PathDecisionCoordinator.for_prefill()
    receiver.close()
    sender.close()

    with pytest.raises(RuntimeError):
        sender.submit(_free_control_endpoint(), _decision(_request().request_key))
    receiver.unregister(_request().request_key)
    assert receiver.take_received_decisions() == []


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
    assert _raw_request(endpoint, encode_path_decision(_decision(key))) == _ACK_BYTES
    assert (
        _raw_request(
            endpoint,
            encode_path_abort(
                PathAbortNotice(
                    request_key=key,
                    reason=PathAbortReason.REQUEST_ABORTED,
                )
            ),
        )
        == _ACK_BYTES
    )

    coordinator.close()

    assert not coordinator._receiver_thread.is_alive()
    assert coordinator._pending_keys == set()
    assert coordinator._accepted_decisions == {}
    assert coordinator.take_received_decisions() == []
    assert coordinator.take_received_aborts() == []
    assert coordinator._received_aborts.empty()
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

    coordinator = PathDecisionCoordinator.for_prefill(socket_opener=opener, sleep=lambda _: None)
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
    config.kv_transfer_config.kv_load_failure_policy = "fail"
    config.kv_transfer_config.kv_role = kv_role
    config.kv_transfer_config.is_kv_producer = kv_role in {"kv_producer", "kv_both"}
    config.kv_transfer_config.is_kv_consumer = kv_role in {"kv_consumer", "kv_both"}
    extra_config = {"role": dual_role}
    if dual_path_control_port is not None:
        extra_config["dual_path_control_port"] = dual_path_control_port
    config.kv_transfer_config.kv_connector_extra_config = extra_config
    config.kv_transfer_config.get_from_extra_config.side_effect = lambda key, default: {
        "tls_config": {},
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
    # Stage-2 topology guard restricts DualPath to data_parallel_size == 1, so
    # the per-rank control port derivation degenerates to the base port.
    data_parallel_rank = 0
    derived_port = _free_control_endpoint().port
    control_port = derived_port - data_parallel_rank
    decode_config = _make_scheduler_vllm_config(
        dual_role="decode",
        kv_role="kv_consumer",
        dual_path_control_port=control_port,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=1,
        tensor_parallel_size=2,
        kv_port=1,
    )
    with (
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.get_ip",
            return_value="127.0.0.1",
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.KVPoolSchedulerAdapter"),
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
    finally:
        prefill_scheduler.shutdown()
        prefill_scheduler.executor.shutdown(wait=False)
        prefill_scheduler.metaserver_client.close()


def test_scheduler_shutdown_closes_coordinator_idempotently() -> None:
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.PathDecisionCoordinator"
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
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.get_ip",
            return_value="127.0.0.1",
        ),
        patch("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.KVPoolSchedulerAdapter"),
        patch(
            "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler.PathDecisionCoordinator"
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

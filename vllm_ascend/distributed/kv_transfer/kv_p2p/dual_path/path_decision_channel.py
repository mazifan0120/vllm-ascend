# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias, assert_never

import msgspec
import zmq
from vllm.logger import logger

from .metadata import ReversePlan
from .path_decision import (
    DualPathRequestKey,
    Path,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
)

DUAL_PATH_PROTOCOL_VERSION: Final[int] = 2
PATH_DECISION_SEND_WORKERS: Final[int] = 32

_MAX_DELIVERY_ATTEMPTS: Final[int] = 3
_SEND_TIMEOUT_MS: Final[int] = 1000
_POLL_TIMEOUT_MS: Final[int] = 1000
_RETRY_SPACING_S: Final[float] = 0.1
_ACK: Final[bytes] = b"ACK"
_RECEIVER_READY_TIMEOUT_S: Final[float] = 5.0

_JsonValue: TypeAlias = str | int | float | bool | None | list["_JsonValue"] | dict[str, "_JsonValue"]
_JsonObject: TypeAlias = dict[str, _JsonValue]


def derive_decode_control_port(
    *,
    dual_path_control_port: int,
    data_parallel_rank: int,
    kv_port: int,
    worker_port_span: int,
) -> int:
    derived_port = dual_path_control_port + data_parallel_rank
    if not 1 <= derived_port <= 65535:
        raise ValueError(f"derived DualPath control port {derived_port} is outside 1..65535")
    if kv_port <= derived_port < kv_port + worker_port_span:
        raise ValueError(
            f"derived DualPath control port {derived_port} overlaps the worker KV port range "
            f"[{kv_port}, {kv_port + worker_port_span})"
        )
    return derived_port


def _require_exact_payload(payload: _JsonValue, expected_keys: frozenset[str]) -> _JsonObject:
    if not isinstance(payload, dict):
        raise PathDecisionValidationError("serialized payload must be a dictionary")
    if set(payload) != expected_keys:
        raise PathDecisionValidationError(f"serialized payload fields must be exactly {sorted(expected_keys)}")
    return payload


def _require_protocol_version(protocol_version: _JsonValue) -> int:
    if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
        raise PathDecisionValidationError("protocol_version must be an integer and must not be a boolean")
    return protocol_version


@dataclass(frozen=True)
class DecodeControlEndpoint:
    host: str
    port: int

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise PathDecisionValidationError("host must be a non-empty string")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise PathDecisionValidationError("port must be an integer in the range 1..65535 and must not be a boolean")

    def to_dict(self) -> _JsonObject:
        return {"host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> DecodeControlEndpoint:
        data = _require_exact_payload(payload, frozenset({"host", "port"}))
        return cls(host=data["host"], port=data["port"])


@dataclass(frozen=True)
class DualPathDecisionMetadata:
    protocol_version: int
    decision_request: PathDecisionRequest
    decode_control_endpoint: DecodeControlEndpoint

    def __post_init__(self) -> None:
        _require_protocol_version(self.protocol_version)
        if not isinstance(self.decision_request, PathDecisionRequest):
            raise PathDecisionValidationError("decision_request must be a PathDecisionRequest")
        if not isinstance(self.decode_control_endpoint, DecodeControlEndpoint):
            raise PathDecisionValidationError("decode_control_endpoint must be a DecodeControlEndpoint")

    def to_dict(self) -> _JsonObject:
        return {
            "protocol_version": self.protocol_version,
            "decision_request": self.decision_request.to_dict(),
            "decode_control_endpoint": self.decode_control_endpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> DualPathDecisionMetadata:
        data = _require_exact_payload(
            payload,
            frozenset({"protocol_version", "decision_request", "decode_control_endpoint"}),
        )
        return cls(
            protocol_version=_require_protocol_version(data["protocol_version"]),
            decision_request=PathDecisionRequest.from_dict(data["decision_request"]),
            decode_control_endpoint=DecodeControlEndpoint.from_dict(data["decode_control_endpoint"]),
        )


@dataclass(frozen=True)
class PathDecision:
    protocol_version: int
    result: PathDecisionResult
    reverse_plan: ReversePlan | None

    def __post_init__(self) -> None:
        _require_protocol_version(self.protocol_version)
        if not isinstance(self.result, PathDecisionResult):
            raise PathDecisionValidationError("result must be a PathDecisionResult")
        if self.reverse_plan is not None and not isinstance(self.reverse_plan, ReversePlan):
            raise PathDecisionValidationError("reverse_plan must be a ReversePlan or None")

    def to_dict(self) -> _JsonObject:
        return {
            "protocol_version": self.protocol_version,
            "result": self.result.to_dict(),
            "reverse_plan": None if self.reverse_plan is None else self.reverse_plan.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> PathDecision:
        data = _require_exact_payload(payload, frozenset({"protocol_version", "result", "reverse_plan"}))
        reverse_plan_payload = data["reverse_plan"]
        return cls(
            protocol_version=_require_protocol_version(data["protocol_version"]),
            result=PathDecisionResult.from_dict(data["result"]),
            reverse_plan=None if reverse_plan_payload is None else ReversePlan.from_dict(reverse_plan_payload),
        )


def encode_path_decision(decision: PathDecision) -> bytes:
    return msgspec.msgpack.encode(decision.to_dict())


def decode_path_decision(payload: bytes) -> PathDecision:
    try:
        decoded = msgspec.msgpack.decode(payload)
    except msgspec.DecodeError as error:
        raise PathDecisionValidationError("path decision payload is not valid MessagePack") from error
    return PathDecision.from_dict(decoded)


class PathDecisionDeliveryError(Exception):
    pass


class _DeliverySocket(Protocol):
    def set_send_timeout(self, timeout_ms: int) -> None: ...

    def send(self, payload: bytes) -> None: ...

    def poll(self, timeout_ms: int) -> bool: ...

    def recv(self) -> bytes: ...

    def close(self) -> None: ...


_SocketOpener: TypeAlias = Callable[[DecodeControlEndpoint], AbstractContextManager[_DeliverySocket]]
_Sleep: TypeAlias = Callable[[float], None]
_ShouldStop: TypeAlias = Callable[[], bool]


class _ZmqReqSocket:
    def __init__(self, socket: zmq.Socket) -> None:
        self._socket = socket

    def set_send_timeout(self, timeout_ms: int) -> None:
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)

    def send(self, payload: bytes) -> None:
        self._socket.send(payload)

    def poll(self, timeout_ms: int) -> bool:
        return self._socket.poll(timeout_ms) != 0

    def recv(self) -> bytes:
        return self._socket.recv()

    def close(self) -> None:
        self._socket.close(linger=0)


@contextmanager
def _zmq_req_opener(endpoint: DecodeControlEndpoint):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    adapter = _ZmqReqSocket(socket)
    try:
        socket.connect(f"tcp://{endpoint.host}:{endpoint.port}")
        yield adapter
    finally:
        adapter.close()
        context.destroy(linger=0)


def _deliver_decision(
    encoded: bytes,
    endpoint: DecodeControlEndpoint,
    *,
    opener: _SocketOpener,
    sleep: _Sleep,
    should_stop: _ShouldStop,
    send_timeout_ms: int,
    poll_timeout_ms: int,
    retry_spacing_s: float,
) -> None:
    last_error: Exception | None = None
    for attempt in range(_MAX_DELIVERY_ATTEMPTS):
        try:
            with opener(endpoint) as socket:
                try:
                    socket.set_send_timeout(send_timeout_ms)
                    socket.send(encoded)
                    if not socket.poll(poll_timeout_ms):
                        raise PathDecisionDeliveryError("path decision acknowledgement timed out")
                    if socket.recv() != _ACK:
                        raise PathDecisionDeliveryError("path decision acknowledgement was invalid")
                    return
                finally:
                    socket.close()
        except Exception as error:  # noqa: BLE001
            last_error = error

        if attempt == _MAX_DELIVERY_ATTEMPTS - 1 or should_stop():
            break
        sleep(retry_spacing_s)

    raise PathDecisionDeliveryError("path decision delivery failed") from last_error


class PathDecisionCoordinator:
    def __init__(self) -> None:
        self._role = ""
        self._closed = False
        self._running = False
        self._decode_engine_instance_id: str | None = None
        self._decode_control_endpoint: DecodeControlEndpoint | None = None
        self._pending_keys: set[DualPathRequestKey] = set()
        self._accepted_decisions: dict[DualPathRequestKey, PathDecision] = {}
        self._received_decisions: queue.SimpleQueue[PathDecision] = queue.SimpleQueue()
        self._registry_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._context: zmq.Context | None = None
        self._receiver_thread: threading.Thread | None = None
        self._receiver_error: BaseException | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._socket_opener: _SocketOpener = _zmq_req_opener
        self._sleep: _Sleep = time.sleep
        self._send_timeout_ms = _SEND_TIMEOUT_MS
        self._poll_timeout_ms = _POLL_TIMEOUT_MS
        self._retry_spacing_s = _RETRY_SPACING_S

    @classmethod
    def for_decode(
        cls,
        *,
        engine_id: str,
        data_parallel_rank: int,
        control_endpoint: DecodeControlEndpoint,
        boot_id: str | None = None,
    ) -> PathDecisionCoordinator:
        coordinator = cls()
        coordinator._role = "decode"
        coordinator._running = True
        incarnation = boot_id if boot_id is not None else uuid.uuid4().hex
        coordinator._decode_engine_instance_id = f"{engine_id}:{data_parallel_rank}:{incarnation}"
        coordinator._decode_control_endpoint = control_endpoint
        coordinator._context = zmq.Context()
        ready_event = threading.Event()
        coordinator._receiver_thread = threading.Thread(
            target=coordinator._receive_results,
            args=(ready_event,),
            name=f"path-decision-result-receiver-{data_parallel_rank}",
            daemon=True,
        )
        coordinator._receiver_thread.start()
        if not ready_event.wait(timeout=_RECEIVER_READY_TIMEOUT_S):
            coordinator.close()
            raise RuntimeError("path decision result receiver did not become ready")
        if coordinator._receiver_error is not None:
            error = coordinator._receiver_error
            coordinator.close()
            raise error
        return coordinator

    @classmethod
    def for_prefill(
        cls,
        *,
        _socket_opener: _SocketOpener | None = None,
        _sleep: _Sleep | None = None,
        _send_timeout_ms: int = _SEND_TIMEOUT_MS,
        _poll_timeout_ms: int = _POLL_TIMEOUT_MS,
        _retry_spacing_s: float = _RETRY_SPACING_S,
    ) -> PathDecisionCoordinator:
        coordinator = cls()
        coordinator._role = "prefill"
        coordinator._socket_opener = _socket_opener or _zmq_req_opener
        coordinator._sleep = _sleep or time.sleep
        coordinator._send_timeout_ms = _send_timeout_ms
        coordinator._poll_timeout_ms = _poll_timeout_ms
        coordinator._retry_spacing_s = _retry_spacing_s
        coordinator._executor = ThreadPoolExecutor(
            max_workers=PATH_DECISION_SEND_WORKERS,
            thread_name_prefix="path-decision-sender",
        )
        return coordinator

    @property
    def decode_engine_instance_id(self) -> str:
        self._require_role("decode")
        assert self._decode_engine_instance_id is not None
        return self._decode_engine_instance_id

    @property
    def decode_control_endpoint(self) -> DecodeControlEndpoint:
        self._require_role("decode")
        assert self._decode_control_endpoint is not None
        return self._decode_control_endpoint

    def register_pending(self, key: DualPathRequestKey) -> None:
        self._require_role("decode")
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("path decision coordinator is closed")
            with self._registry_lock:
                self._pending_keys.add(key)

    def unregister(self, key: DualPathRequestKey) -> None:
        self._require_role("decode")
        with self._registry_lock:
            self._pending_keys.discard(key)
            self._accepted_decisions.pop(key, None)

    def take_received_decisions(self) -> list[PathDecision]:
        self._require_role("decode")
        if self._closed:
            return []
        decisions: list[PathDecision] = []
        with self._registry_lock:
            while True:
                try:
                    decisions.append(self._received_decisions.get_nowait())
                except queue.Empty:
                    return decisions

    def submit(
        self,
        endpoint: DecodeControlEndpoint,
        decision: PathDecision,
    ) -> Future[None]:
        self._require_role("prefill")
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("path decision coordinator is closed")
            encoded = encode_path_decision(decision)
            assert self._executor is not None
            return self._executor.submit(
                _deliver_decision,
                encoded,
                endpoint,
                opener=self._socket_opener,
                sleep=self._sleep,
                should_stop=lambda: self._closed,
                send_timeout_ms=self._send_timeout_ms,
                poll_timeout_ms=self._poll_timeout_ms,
                retry_spacing_s=self._retry_spacing_s,
            )

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._running = False
        if self._role == "decode":
            assert self._context is not None
            assert self._receiver_thread is not None
            self._context.term()
            self._receiver_thread.join()
            with self._registry_lock:
                self._pending_keys.clear()
                self._accepted_decisions.clear()
            self._drain_received_decisions()
        elif self._role == "prefill":
            assert self._executor is not None
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _require_role(self, role: str) -> None:
        if self._role != role:
            raise RuntimeError(f"path decision coordinator does not support {role} operations")

    def _receive_results(self, ready_event: threading.Event) -> None:
        assert self._context is not None
        assert self._decode_control_endpoint is not None
        socket: zmq.Socket | None = None
        try:
            socket = self._context.socket(zmq.ROUTER)
            socket.setsockopt(zmq.LINGER, 0)
            endpoint = self._decode_control_endpoint
            socket.bind(f"tcp://{endpoint.host}:{endpoint.port}")
            ready_event.set()
            while self._running:
                try:
                    frames = socket.recv_multipart()
                except zmq.ContextTerminated:
                    break
                except zmq.ZMQError:
                    if not self._running:
                        break
                    logger.exception("path decision result receiver socket failure")
                    continue
                try:
                    self._handle_frames(socket, frames)
                except Exception:  # noqa: BLE001
                    logger.exception("path decision result receiver rejected an unexpected message failure")
        except BaseException as error:  # noqa: BLE001
            self._receiver_error = error
            ready_event.set()
        finally:
            if socket is not None:
                socket.close(linger=0)

    def _handle_frames(self, socket: zmq.Socket, frames: list[bytes]) -> None:
        if len(frames) != 3 or frames[1] != b"":
            logger.warning("path decision result receiver rejected invalid frame shape")
            return
        identity, _, payload = frames
        try:
            decision = decode_path_decision(payload)
        except PathDecisionValidationError:
            logger.warning("path decision result receiver rejected malformed payload")
            return
        if decision.protocol_version != DUAL_PATH_PROTOCOL_VERSION:
            logger.warning("path decision result receiver rejected unsupported protocol version")
            return

        result = decision.result
        key = result.request_key
        if key.decode_engine_instance_id != self.decode_engine_instance_id:
            logger.warning("path decision result receiver rejected wrong-incarnation key")
            return
        if result.path is Path.PE_READ:
            if decision.reverse_plan is not None:
                logger.warning("path decision result receiver rejected PE_READ Reverse plan")
                return
        elif result.path is Path.DE_READ:
            if decision.reverse_plan is None:
                logger.warning("path decision result receiver rejected DE_READ without Reverse plan")
                return
            if decision.reverse_plan.request_key != key:
                logger.warning("path decision result receiver rejected mismatched Reverse plan key")
                return
        else:
            assert_never(result.path)
        with self._registry_lock:
            if key not in self._pending_keys:
                logger.warning("path decision result receiver rejected unknown or stale key")
                return
            retained = self._accepted_decisions.get(key)
            if retained is not None:
                if retained != decision:
                    logger.warning("path decision result receiver rejected conflicting duplicate")
                    return
            else:
                self._accepted_decisions[key] = decision
                self._received_decisions.put(decision)

        socket.send_multipart([identity, b"", _ACK])

    def _drain_received_decisions(self) -> None:
        while True:
            try:
                self._received_decisions.get_nowait()
            except queue.Empty:
                return

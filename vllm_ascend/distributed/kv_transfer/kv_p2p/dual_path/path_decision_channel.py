# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, Protocol, TypeAlias

import msgspec
import zmq
from typing_extensions import assert_never
from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import MAX_TCP_PORT, MIN_TCP_PORT
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import ReversePlan
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    JsonObject,
    JsonValue,
    PathAbortNotice,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
    require_exact_payload,
)

_PATH_DECISION_SEND_WORKERS: Final[int] = 32

_MAX_DELIVERY_ATTEMPTS: Final[int] = 3
_SEND_TIMEOUT_MS: Final[int] = 1000
_POLL_TIMEOUT_MS: Final[int] = 1000
_RETRY_SPACING_S: Final[float] = 0.1
_RECEIVER_READY_TIMEOUT_S: Final[float] = 5.0


def derive_decode_control_port(
    *,
    dual_path_control_port: int,
    data_parallel_rank: int,
    kv_port: int,
    worker_port_span: int,
) -> int:
    derived_port = dual_path_control_port + data_parallel_rank
    if not MIN_TCP_PORT <= derived_port <= MAX_TCP_PORT:
        raise ValueError(f"derived DualPath control port {derived_port} is outside 1..65535")
    if kv_port <= derived_port < kv_port + worker_port_span:
        raise ValueError(
            f"derived DualPath control port {derived_port} overlaps the worker KV port range "
            f"[{kv_port}, {kv_port + worker_port_span})"
        )
    return derived_port


@dataclass(frozen=True)
class DecodeControlEndpoint:
    host: str
    port: int

    def __post_init__(self) -> None:
        if not isinstance(self.host, str) or not self.host:
            raise PathDecisionValidationError("host must be a non-empty string")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not MIN_TCP_PORT <= self.port <= MAX_TCP_PORT
        ):
            raise PathDecisionValidationError("port must be an integer in the range 1..65535 and must not be a boolean")

    def to_dict(self) -> JsonObject:
        return {"host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, payload: JsonValue) -> DecodeControlEndpoint:
        data = require_exact_payload(payload, frozenset({"host", "port"}))
        return cls(host=data["host"], port=data["port"])


@dataclass(frozen=True)
class DualPathDecisionMetadata:
    decision_request: PathDecisionRequest
    decode_control_endpoint: DecodeControlEndpoint

    def __post_init__(self) -> None:
        if not isinstance(self.decision_request, PathDecisionRequest):
            raise PathDecisionValidationError("decision_request must be a PathDecisionRequest")
        if not isinstance(self.decode_control_endpoint, DecodeControlEndpoint):
            raise PathDecisionValidationError("decode_control_endpoint must be a DecodeControlEndpoint")

    def to_dict(self) -> JsonObject:
        return {
            "decision_request": self.decision_request.to_dict(),
            "decode_control_endpoint": self.decode_control_endpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> DualPathDecisionMetadata:
        data = require_exact_payload(
            payload,
            frozenset({"decision_request", "decode_control_endpoint"}),
        )
        return cls(
            decision_request=PathDecisionRequest.from_dict(data["decision_request"]),
            decode_control_endpoint=DecodeControlEndpoint.from_dict(data["decode_control_endpoint"]),
        )


@dataclass(frozen=True)
class PathDecision:
    result: PathDecisionResult
    reverse_plan: ReversePlan | None
    # Optional reverse-direction control endpoint; the Decode side sends
    # terminal failure notices (ABORT) here when an activated request dies
    # before its reverse transfer could start.
    prefill_control_endpoint: DecodeControlEndpoint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, PathDecisionResult):
            raise PathDecisionValidationError("result must be a PathDecisionResult")
        if self.reverse_plan is not None and not isinstance(self.reverse_plan, ReversePlan):
            raise PathDecisionValidationError("reverse_plan must be a ReversePlan or None")
        if self.prefill_control_endpoint is not None and not isinstance(
            self.prefill_control_endpoint,
            DecodeControlEndpoint,
        ):
            raise PathDecisionValidationError(
                "prefill_control_endpoint must be a DecodeControlEndpoint or None"
            )

    def to_dict(self) -> JsonObject:
        return {
            "result": self.result.to_dict(),
            "reverse_plan": None if self.reverse_plan is None else self.reverse_plan.to_dict(),
            "prefill_control_endpoint": (
                None if self.prefill_control_endpoint is None else self.prefill_control_endpoint.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> PathDecision:
        data = require_exact_payload(
            payload,
            frozenset({"result", "reverse_plan", "prefill_control_endpoint"}),
        )
        reverse_plan_payload = data["reverse_plan"]
        endpoint_payload = data["prefill_control_endpoint"]
        return cls(
            result=PathDecisionResult.from_dict(data["result"]),
            reverse_plan=None if reverse_plan_payload is None else ReversePlan.from_dict(reverse_plan_payload),
            prefill_control_endpoint=(
                None if endpoint_payload is None else DecodeControlEndpoint.from_dict(endpoint_payload)
            ),
        )


class ControlMessageKind(str, Enum):
    DECISION = "Decision"
    ABORT = "Abort"


class DecisionReplyStatus(str, Enum):
    ACK = "ACK"
    STALE_CLOSED = "STALE_CLOSED"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    UNKNOWN_REQUEST = "UNKNOWN_REQUEST"


def encode_control_message(kind: ControlMessageKind, payload: JsonObject) -> bytes:
    return msgspec.msgpack.encode({"kind": kind.value, "payload": payload})


def decode_control_message(payload: bytes) -> tuple[ControlMessageKind, JsonObject]:
    try:
        decoded = msgspec.msgpack.decode(payload)
    except msgspec.DecodeError as error:
        raise PathDecisionValidationError("control message payload is not valid MessagePack") from error
    data = require_exact_payload(decoded, frozenset({"kind", "payload"}))
    try:
        kind = ControlMessageKind(data["kind"])
    except (TypeError, ValueError) as error:
        raise PathDecisionValidationError("control message kind is not valid") from error
    if not isinstance(data["payload"], dict):
        raise PathDecisionValidationError("control message payload must be a dictionary")
    return kind, data["payload"]


def _encode_status(status: DecisionReplyStatus) -> bytes:
    return msgspec.msgpack.encode({"status": status.value})


def _decode_status(payload: bytes, enum_cls: type) -> DecisionReplyStatus:
    try:
        decoded = msgspec.msgpack.decode(payload)
    except msgspec.DecodeError as error:
        raise PathDecisionValidationError("reply payload is not valid MessagePack") from error
    data = require_exact_payload(decoded, frozenset({"status"}))
    try:
        return enum_cls(data["status"])
    except (TypeError, ValueError) as error:
        raise PathDecisionValidationError("reply status is not valid") from error


def encode_decision_reply(status: DecisionReplyStatus) -> bytes:
    return _encode_status(status)


def decode_decision_reply(payload: bytes) -> DecisionReplyStatus:
    return _decode_status(payload, DecisionReplyStatus)


def encode_path_decision(decision: PathDecision) -> bytes:
    return encode_control_message(ControlMessageKind.DECISION, decision.to_dict())


def decode_path_decision(payload: bytes) -> PathDecision:
    kind, decision_payload = decode_control_message(payload)
    if kind is not ControlMessageKind.DECISION:
        raise PathDecisionValidationError(f"control message kind {kind.value} is not a Decision")
    return PathDecision.from_dict(decision_payload)


def encode_path_abort(notice: PathAbortNotice) -> bytes:
    return encode_control_message(ControlMessageKind.ABORT, notice.to_dict())


def decode_path_abort(payload: bytes) -> PathAbortNotice:
    kind, notice_payload = decode_control_message(payload)
    if kind is not ControlMessageKind.ABORT:
        raise PathDecisionValidationError(f"control message kind {kind.value} is not an Abort")
    return PathAbortNotice.from_dict(notice_payload)


class PathDecisionDeliveryError(Exception):
    pass


class PathDecisionRejectedError(PathDecisionDeliveryError):
    """A terminal registry rejection: no retry can change the answer."""

    def __init__(self, status: DecisionReplyStatus) -> None:
        super().__init__(f"path decision rejected with status {status.value}")
        self.status = status


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
def _zmq_req_opener(endpoint: DecodeControlEndpoint) -> Iterator[_DeliverySocket]:
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
                socket.set_send_timeout(send_timeout_ms)
                socket.send(encoded)
                if not socket.poll(poll_timeout_ms):
                    raise PathDecisionDeliveryError("path decision acknowledgement timed out")
                try:
                    status = decode_decision_reply(socket.recv())
                except PathDecisionValidationError as error:
                    raise PathDecisionDeliveryError("path decision acknowledgement was invalid") from error
                if status is DecisionReplyStatus.ACK:
                    return
                raise PathDecisionRejectedError(status)
        except PathDecisionRejectedError:
            raise
        except Exception as error:  # noqa: BLE001
            last_error = error

        if attempt == _MAX_DELIVERY_ATTEMPTS - 1 or should_stop():
            break
        sleep(retry_spacing_s)

    raise PathDecisionDeliveryError("path decision delivery failed") from last_error


class PathDecisionCoordinator:
    def __init__(self) -> None:
        self._role: Literal["prefill", "decode"] | None = None
        self._closed = False
        self._decode_engine_instance_id: str | None = None
        self._decode_control_endpoint: DecodeControlEndpoint | None = None
        self._bind_endpoint: DecodeControlEndpoint | None = None
        self._next_admission_id = 0
        self._pending_keys: set[DualPathRequestKey] = set()
        self._accepted_decisions: dict[DualPathRequestKey, PathDecision] = {}
        self._closed_through_attempt_ids: dict[DualPathRequestKey, int] = {}
        self._prefill_abort_keys: set[DualPathRequestKey] = set()
        self._prefill_received_abort_notices: dict[
            DualPathRequestKey, set[PathAbortNotice]
        ] = {}
        self._received_decisions: queue.SimpleQueue[PathDecision] = queue.SimpleQueue()
        self._received_aborts: queue.SimpleQueue[PathAbortNotice] = queue.SimpleQueue()
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
        incarnation = boot_id if boot_id is not None else uuid.uuid4().hex
        coordinator._decode_engine_instance_id = f"{engine_id}:{data_parallel_rank}:{incarnation}"
        coordinator._decode_control_endpoint = control_endpoint
        coordinator._bind_endpoint = control_endpoint
        coordinator._executor = ThreadPoolExecutor(
            max_workers=_PATH_DECISION_SEND_WORKERS,
            thread_name_prefix="path-abort-sender",
        )
        coordinator._start_receiver(f"path-decision-result-receiver-{data_parallel_rank}")
        return coordinator

    @classmethod
    def for_prefill(
        cls,
        *,
        control_endpoint: DecodeControlEndpoint | None = None,
        socket_opener: _SocketOpener | None = None,
        sleep: _Sleep | None = None,
        send_timeout_ms: int = _SEND_TIMEOUT_MS,
        poll_timeout_ms: int = _POLL_TIMEOUT_MS,
        retry_spacing_s: float = _RETRY_SPACING_S,
    ) -> PathDecisionCoordinator:
        coordinator = cls()
        coordinator._role = "prefill"
        coordinator._socket_opener = socket_opener or _zmq_req_opener
        coordinator._sleep = sleep or time.sleep
        coordinator._send_timeout_ms = send_timeout_ms
        coordinator._poll_timeout_ms = poll_timeout_ms
        coordinator._retry_spacing_s = retry_spacing_s
        coordinator._executor = ThreadPoolExecutor(
            max_workers=_PATH_DECISION_SEND_WORKERS,
            thread_name_prefix="path-decision-sender",
        )
        if control_endpoint is not None:
            coordinator._bind_endpoint = control_endpoint
            coordinator._start_receiver("path-abort-receiver")
        return coordinator

    def _start_receiver(self, name: str) -> None:
        self._context = zmq.Context()
        ready_event = threading.Event()
        self._receiver_thread = threading.Thread(
            target=self._receive_decisions,
            args=(ready_event,),
            name=name,
            daemon=True,
        )
        self._receiver_thread.start()
        if not ready_event.wait(timeout=_RECEIVER_READY_TIMEOUT_S):
            self.close()
            raise RuntimeError("path decision result receiver did not become ready")
        if self._receiver_error is not None:
            error = self._receiver_error
            self.close()
            raise error

    @property
    def decode_engine_instance_id(self) -> str:
        assert self._decode_engine_instance_id is not None
        return self._decode_engine_instance_id

    @property
    def decode_control_endpoint(self) -> DecodeControlEndpoint:
        assert self._decode_control_endpoint is not None
        return self._decode_control_endpoint

    def new_request_key(self, decode_request_id: str) -> DualPathRequestKey:
        if self._role != "decode":
            raise RuntimeError("only a Decode coordinator can mint admission keys")
        with self._registry_lock:
            admission_id = self._next_admission_id
            self._next_admission_id += 1
        return DualPathRequestKey(
            decode_engine_instance_id=self.decode_engine_instance_id,
            decode_request_id=decode_request_id,
            admission_id=admission_id,
        )

    def register_pending(self, key: DualPathRequestKey) -> None:
        # Unlike unregister, this also takes _lifecycle_lock so close() cannot
        # clear pending state and then observe a fresh registration.
        with self._lifecycle_lock, self._registry_lock:
            if self._closed:
                raise RuntimeError("path decision coordinator is closed")
            self._pending_keys.add(key)

    def unregister(self, key: DualPathRequestKey) -> None:
        with self._registry_lock:
            self._pending_keys.discard(key)
            self._accepted_decisions.pop(key, None)
            self._closed_through_attempt_ids.pop(key, None)

    def register_prefill_abort_key(self, key: DualPathRequestKey) -> None:
        if self._role != "prefill":
            raise RuntimeError("only a Prefill coordinator can register abort keys")
        with self._lifecycle_lock, self._registry_lock:
            if self._closed:
                raise RuntimeError("path decision coordinator is closed")
            self._prefill_abort_keys.add(key)

    def unregister_prefill_abort_key(self, key: DualPathRequestKey) -> None:
        if self._role != "prefill":
            raise RuntimeError("only a Prefill coordinator can unregister abort keys")
        with self._registry_lock:
            self._prefill_abort_keys.discard(key)
            self._prefill_received_abort_notices.pop(key, None)

    def take_received_decisions(self) -> list[PathDecision]:
        if self._closed:
            return []
        decisions: list[PathDecision] = []
        # Draining while holding _registry_lock defers decisions enqueued
        # mid-drain to the next batch instead of returning them here.
        with self._registry_lock:
            while True:
                try:
                    decisions.append(self._received_decisions.get_nowait())
                except queue.Empty:
                    return decisions

    def take_received_aborts(self) -> list[PathAbortNotice]:
        if self._closed:
            return []
        notices: list[PathAbortNotice] = []
        with self._registry_lock:
            while True:
                try:
                    notices.append(self._received_aborts.get_nowait())
                except queue.Empty:
                    return notices

    def submit(
        self,
        endpoint: DecodeControlEndpoint,
        decision: PathDecision,
    ) -> Future[None]:
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

    def submit_abort(
        self,
        endpoint: DecodeControlEndpoint,
        notice: PathAbortNotice,
    ) -> Future[None]:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("path decision coordinator is closed")
            encoded = encode_path_abort(notice)
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
        if self._receiver_thread is not None:
            assert self._context is not None
            self._context.term()
            self._receiver_thread.join()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        with self._registry_lock:
            self._pending_keys.clear()
            self._accepted_decisions.clear()
            self._closed_through_attempt_ids.clear()
            self._prefill_abort_keys.clear()
            self._prefill_received_abort_notices.clear()
        self._drain_received_decisions()
        self._drain_received_aborts()

    def _receive_decisions(self, ready_event: threading.Event) -> None:
        assert self._context is not None
        assert self._bind_endpoint is not None
        socket: zmq.Socket | None = None
        try:
            socket = self._context.socket(zmq.ROUTER)
            socket.setsockopt(zmq.LINGER, 0)
            endpoint = self._bind_endpoint
            socket.bind(f"tcp://{endpoint.host}:{endpoint.port}")
            ready_event.set()
            while not self._closed:
                try:
                    frames = socket.recv_multipart()
                except zmq.ContextTerminated:
                    break
                except zmq.ZMQError:
                    if self._closed:
                        break
                    logger.exception("path decision result receiver socket failure")
                    continue
                try:
                    self._handle_frames(socket, frames)
                except Exception:  # noqa: BLE001
                    logger.exception("path decision result receiver rejected an unexpected message failure")
        except BaseException as error:  # noqa: BLE001
            self._receiver_error = error
            logger.exception("path decision result receiver thread terminated")
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
            kind, message_payload = decode_control_message(payload)
        except PathDecisionValidationError:
            logger.warning("path decision result receiver rejected malformed payload")
            return
        if kind is ControlMessageKind.DECISION:
            self._handle_decision_frame(socket, identity, message_payload)
        elif kind is ControlMessageKind.ABORT:
            self._handle_abort_frame(socket, identity, message_payload)
        else:
            assert_never(kind)

    def _handle_decision_frame(self, socket: zmq.Socket, identity: bytes, payload: JsonObject) -> None:
        if self._role != "decode":
            logger.warning("Prefill control receiver rejected Decision payload")
            self._reply_decision(socket, identity, DecisionReplyStatus.PROTOCOL_ERROR)
            return
        try:
            decision = PathDecision.from_dict(payload)
        except PathDecisionValidationError:
            logger.warning("path decision result receiver rejected malformed Decision payload")
            return
        result = decision.result
        key = result.request_key
        if key.decode_engine_instance_id != self.decode_engine_instance_id:
            logger.warning("path decision result receiver rejected wrong-incarnation key")
            self._reply_decision(socket, identity, DecisionReplyStatus.UNKNOWN_REQUEST)
            return
        if result.path is PathKind.PE_READ:
            shape_valid = decision.reverse_plan is None
        elif result.path is PathKind.DE_READ:
            shape_valid = decision.reverse_plan is not None and decision.reverse_plan.request_key == key
        else:
            assert_never(result.path)
        if not shape_valid:
            logger.warning("path decision result receiver rejected invalid Decision shape")
            self._reply_decision(socket, identity, DecisionReplyStatus.PROTOCOL_ERROR)
            return
        with self._registry_lock:
            if key not in self._pending_keys:
                logger.warning("path decision result receiver rejected unknown or stale key")
                reply_status = DecisionReplyStatus.UNKNOWN_REQUEST
            else:
                reply_status = self._register_decision_locked(key, decision)
        self._reply_decision(socket, identity, reply_status)

    def _handle_abort_frame(self, socket: zmq.Socket, identity: bytes, payload: JsonObject) -> None:
        try:
            notice = PathAbortNotice.from_dict(payload)
        except PathDecisionValidationError:
            logger.warning("path decision result receiver rejected malformed Abort payload")
            self._reply_decision(socket, identity, DecisionReplyStatus.PROTOCOL_ERROR)
            return
        key = notice.request_key
        if self._role == "prefill":
            with self._registry_lock:
                if key in self._prefill_abort_keys:
                    received_notices = self._prefill_received_abort_notices.setdefault(
                        key,
                        set(),
                    )
                    if notice not in received_notices:
                        received_notices.add(notice)
                        self._received_aborts.put(notice)
            self._reply_decision(socket, identity, DecisionReplyStatus.ACK)
            return
        assert self._role == "decode"
        if key.decode_engine_instance_id != self.decode_engine_instance_id:
            logger.warning("path decision result receiver rejected wrong-incarnation Abort key")
            self._reply_decision(socket, identity, DecisionReplyStatus.UNKNOWN_REQUEST)
            return
        with self._registry_lock:
            if key not in self._pending_keys and key not in self._accepted_decisions:
                # ABORT removes any queued Decision for this exact request key.
                # If none exists, acknowledge immediately.
                reply_status = DecisionReplyStatus.ACK
            else:
                self._pending_keys.discard(key)
                self._accepted_decisions.pop(key, None)
                self._closed_through_attempt_ids.pop(key, None)
                self._received_aborts.put(notice)
                reply_status = DecisionReplyStatus.ACK
        self._reply_decision(socket, identity, reply_status)

    def _register_decision_locked(self, key: DualPathRequestKey, decision: PathDecision) -> DecisionReplyStatus:
        attempt_id = decision.result.reverse_attempt_id
        closed_through = self._closed_through_attempt_ids.get(key)
        retained = self._accepted_decisions.get(key)
        if retained is None:
            if closed_through is not None and attempt_id is not None and attempt_id <= closed_through:
                return DecisionReplyStatus.STALE_CLOSED
            self._close_skipped_lower_attempts_locked(key, attempt_id)
            self._accept_decision_locked(key, decision)
            return DecisionReplyStatus.ACK
        retained_attempt_id = retained.result.reverse_attempt_id
        if (retained_attempt_id is None) != (attempt_id is None):
            return DecisionReplyStatus.PROTOCOL_ERROR
        if retained_attempt_id == attempt_id:
            return DecisionReplyStatus.ACK if retained == decision else DecisionReplyStatus.PROTOCOL_ERROR
        if attempt_id is not None and retained_attempt_id is not None and attempt_id > retained_attempt_id:
            if closed_through is not None and attempt_id <= closed_through:
                return DecisionReplyStatus.STALE_CLOSED
            self._close_skipped_lower_attempts_locked(key, attempt_id)
            self._accept_decision_locked(key, decision)
            return DecisionReplyStatus.ACK
        return DecisionReplyStatus.STALE_CLOSED

    def _close_skipped_lower_attempts_locked(self, key: DualPathRequestKey, accepted_attempt_id: int | None) -> None:
        # Accepting attempt M permanently closes every skipped lower number.
        if accepted_attempt_id is None or accepted_attempt_id == 0:
            return
        self._closed_through_attempt_ids[key] = max(
            self._closed_through_attempt_ids.get(key, -1),
            accepted_attempt_id - 1,
        )

    def _accept_decision_locked(self, key: DualPathRequestKey, decision: PathDecision) -> None:
        self._accepted_decisions[key] = decision
        self._received_decisions.put(decision)

    @staticmethod
    def _reply_decision(socket: zmq.Socket, identity: bytes, status: DecisionReplyStatus) -> None:
        socket.send_multipart([identity, b"", encode_decision_reply(status)])

    def _drain_received_decisions(self) -> None:
        while True:
            try:
                self._received_decisions.get_nowait()
            except queue.Empty:
                return

    def _drain_received_aborts(self) -> None:
        while True:
            try:
                self._received_aborts.get_nowait()
            except queue.Empty:
                return

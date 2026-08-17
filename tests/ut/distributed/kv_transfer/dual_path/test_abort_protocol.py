# SPDX-License-Identifier: Apache-2.0
"""Request-terminal ABORT protocol and channel behavior."""

from __future__ import annotations

from contextlib import contextmanager

import msgspec
import pytest

from tests.ut.distributed.kv_transfer.dual_path.test_channel_registry import (
    _ENDPOINT,
    _KEY,
    _decision,
    _decision_reply,
    _deliver_frames,
    _make_receiver,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import (
    path_decision as decision_model,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import (
    path_decision_channel as channel,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionValidationError,
)


def _abort(reason: str = "DELIVERY_EXHAUSTED"):
    return decision_model.PathAbortNotice(
        request_key=_KEY,
        reason=decision_model.PathAbortReason(reason),
    )


def test_abort_notice_has_exact_reasons_and_round_trips() -> None:
    assert [reason.value for reason in decision_model.PathAbortReason] == [
        "DECISION_FAILED",
        "DELIVERY_EXHAUSTED",
        "ACTIVATION_FAILED",
        "REQUEST_ABORTED",
    ]
    notice = _abort()
    payload = {
        "request_key": {
            "decode_engine_instance_id": _KEY.decode_engine_instance_id,
            "decode_request_id": _KEY.decode_request_id,
            "admission_id": _KEY.admission_id,
        },
        "reason": "DELIVERY_EXHAUSTED",
    }

    assert notice.to_dict() == payload
    assert decision_model.PathAbortNotice.from_dict(payload) == notice


@pytest.mark.parametrize(
    ("request_key", "reason"),
    [
        pytest.param("not-a-key", "DELIVERY_EXHAUSTED", id="request-key-type"),
        pytest.param(_KEY, "DELIVERY_EXHAUSTED", id="reason-type"),
    ],
)
def test_abort_notice_constructor_rejects_non_value_objects(request_key, reason) -> None:
    with pytest.raises(PathDecisionValidationError):
        decision_model.PathAbortNotice(request_key=request_key, reason=reason)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([], id="non-dictionary"),
        pytest.param({"request_key": _KEY.to_dict()}, id="missing-reason"),
        pytest.param(
            {
                "request_key": _KEY.to_dict(),
                "reason": "DELIVERY_EXHAUSTED",
                "extra": True,
            },
            id="extra-field",
        ),
        pytest.param(
            {"request_key": _KEY.to_dict(), "reason": "BOGUS"},
            id="unknown-reason",
        ),
        pytest.param(
            {"request_key": _KEY.to_dict(), "reason": 1},
            id="non-string-reason",
        ),
    ],
)
def test_abort_notice_from_dict_rejects_malformed_payload(payload) -> None:
    with pytest.raises(PathDecisionValidationError):
        decision_model.PathAbortNotice.from_dict(payload)


def test_abort_envelope_round_trips_and_rejects_decision_kind() -> None:
    notice = _abort("ACTIVATION_FAILED")
    expected = msgspec.msgpack.encode(
        {
            "kind": "Abort",
            "payload": notice.to_dict(),
        }
    )

    assert channel.encode_path_abort(notice) == expected
    assert channel.decode_path_abort(expected) == notice
    with pytest.raises(PathDecisionValidationError, match="not an Abort"):
        channel.decode_path_abort(channel.encode_path_decision(_decision()))


def test_pending_abort_is_acked_enqueued_once_and_makes_late_decision_unknown() -> None:
    receiver = _make_receiver()
    receiver.register_pending(_KEY)
    notice = _abort()

    reply = _deliver_frames(receiver, channel.encode_path_abort(notice))

    assert channel.decode_decision_reply(reply) is channel.DecisionReplyStatus.ACK
    assert receiver.take_received_aborts() == [notice]
    assert receiver.take_received_aborts() == []
    assert _decision_reply(receiver, _decision()) is channel.DecisionReplyStatus.UNKNOWN_REQUEST


def test_abort_is_accepted_when_only_accepted_decision_registry_retains_key() -> None:
    receiver = _make_receiver()
    receiver.register_pending(_KEY)
    assert _decision_reply(receiver, _decision()) is channel.DecisionReplyStatus.ACK
    receiver._pending_keys.remove(_KEY)

    reply = _deliver_frames(receiver, channel.encode_path_abort(_abort("REQUEST_ABORTED")))

    assert channel.decode_decision_reply(reply) is channel.DecisionReplyStatus.ACK
    assert receiver.take_received_aborts() == [_abort("REQUEST_ABORTED")]


def test_duplicate_abort_is_acked_without_duplicate_notice() -> None:
    receiver = _make_receiver()
    receiver.register_pending(_KEY)
    notice = _abort("REQUEST_ABORTED")

    first_reply = _deliver_frames(receiver, channel.encode_path_abort(notice))
    duplicate_reply = _deliver_frames(receiver, channel.encode_path_abort(notice))

    assert channel.decode_decision_reply(first_reply) is channel.DecisionReplyStatus.ACK
    assert channel.decode_decision_reply(duplicate_reply) is channel.DecisionReplyStatus.ACK
    assert receiver.take_received_aborts() == [notice]
    assert receiver.take_received_aborts() == []


def test_never_registered_same_incarnation_abort_is_acked_without_notice() -> None:
    receiver = _make_receiver()
    notice = _abort("REQUEST_ABORTED")

    reply = _deliver_frames(receiver, channel.encode_path_abort(notice))

    assert channel.decode_decision_reply(reply) is channel.DecisionReplyStatus.ACK
    assert receiver.take_received_aborts() == []


def test_wrong_incarnation_abort_is_rejected() -> None:
    receiver = _make_receiver()
    receiver.register_pending(_KEY)
    key = DualPathRequestKey("decode-engine:2:wrong-boot", _KEY.decode_request_id, _KEY.admission_id)
    notice = decision_model.PathAbortNotice(
        request_key=key,
        reason=decision_model.PathAbortReason.REQUEST_ABORTED,
    )

    reply = _deliver_frames(receiver, channel.encode_path_abort(notice))

    assert channel.decode_decision_reply(reply) is channel.DecisionReplyStatus.UNKNOWN_REQUEST
    assert receiver.take_received_aborts() == []


def test_malformed_abort_payload_gets_protocol_error_and_is_not_enqueued() -> None:
    receiver = _make_receiver()
    receiver.register_pending(_KEY)
    encoded = channel.encode_control_message(
        channel.ControlMessageKind.ABORT,
        {"request_key": _KEY.to_dict(), "reason": "BOGUS"},
    )

    reply = _deliver_frames(receiver, encoded)

    assert channel.decode_decision_reply(reply) is channel.DecisionReplyStatus.PROTOCOL_ERROR
    assert receiver.take_received_aborts() == []


class _NoAckSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def set_send_timeout(self, timeout_ms: int) -> None:
        pass

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def poll(self, timeout_ms: int) -> bool:
        return False

    def recv(self) -> bytes:
        raise AssertionError("no reply is available")

    def close(self) -> None:
        pass


class _LoopbackAbortSocket:
    def __init__(self, receiver, *, drop_reply: bool) -> None:
        self._receiver = receiver
        self._drop_reply = drop_reply
        self._reply: bytes | None = None

    def set_send_timeout(self, timeout_ms: int) -> None:
        pass

    def send(self, payload: bytes) -> None:
        self._reply = _deliver_frames(self._receiver, payload)

    def poll(self, timeout_ms: int) -> bool:
        return not self._drop_reply and self._reply is not None

    def recv(self) -> bytes:
        assert self._reply is not None
        return self._reply

    def close(self) -> None:
        pass


def test_lost_first_abort_ack_retries_to_success_without_duplicate_notice() -> None:
    receiver = _make_receiver()
    receiver.register_pending(_KEY)
    notice = _abort("REQUEST_ABORTED")
    attempts = 0

    @contextmanager
    def opener(endpoint):
        nonlocal attempts
        attempts += 1
        yield _LoopbackAbortSocket(receiver, drop_reply=attempts == 1)

    sender = channel.PathDecisionCoordinator.for_prefill(
        socket_opener=opener,
        sleep=lambda _: None,
        poll_timeout_ms=1,
        retry_spacing_s=0,
    )
    try:
        future = sender.submit_abort(_ENDPOINT, notice)
        assert future.result(timeout=5) is None
    finally:
        sender.close()

    assert attempts == 2
    assert receiver.take_received_aborts() == [notice]
    assert receiver.take_received_aborts() == []


def test_abort_submission_uses_shared_three_attempt_delivery_pool() -> None:
    sockets: list[_NoAckSocket] = []

    @contextmanager
    def opener(endpoint):
        socket = _NoAckSocket()
        sockets.append(socket)
        yield socket

    coordinator = channel.PathDecisionCoordinator.for_prefill(
        socket_opener=opener,
        sleep=lambda _: None,
    )
    notice = _abort()
    try:
        with pytest.raises(channel.PathDecisionDeliveryError):
            coordinator.submit_abort(_ENDPOINT, notice).result(timeout=5)
    finally:
        coordinator.close()

    assert len(sockets) == 3
    assert [socket.sent for socket in sockets] == [[channel.encode_path_abort(notice)]] * 3

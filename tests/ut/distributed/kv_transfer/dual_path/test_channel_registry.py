# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W5: control-channel envelope, reply statuses, attempt-aware registry."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import msgspec
import pytest

from tests.ut.distributed.kv_transfer.dual_path.conftest import DECODE_TEST_INSTANCE_ID
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel as channel
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import ReversePlan
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
    ReverseAttemptKey,
    reverse_wire_id,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
    PathDecision,
    PathDecisionCoordinator,
    PathDecisionDeliveryError,
)

_KEY = DualPathRequestKey(DECODE_TEST_INSTANCE_ID, "decode-request-9", 0)
_ENDPOINT = DecodeControlEndpoint(host="192.0.2.10", port=24009)


def _decision(attempt_id: int = 0, *, remote_port: int = 6000) -> PathDecision:
    return PathDecision(
        result=PathDecisionResult(
            request_key=_KEY,
            path=PathKind.DE_READ,
            reverse_attempt_id=attempt_id,
            prefill_local_tokens=16,
        ),
        reverse_plan=ReversePlan(
            request_key=_KEY,
            wire_request_id=reverse_wire_id(ReverseAttemptKey(_KEY, attempt_id)),
            token_start=16,
            token_end=32,
            source_block_ids=((41, 42),),
            destination_block_ids=((71, 72),),
            remote_engine_id="prefill-engine",
            remote_host="198.51.100.10",
            remote_port=remote_port,
            remote_block_sizes=(16,),
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
            reverse_attempt_id=attempt_id,
            prefill_local_tokens=16,
            reverse_send_job_id=None,
        ),
    )


def _make_receiver() -> PathDecisionCoordinator:
    coordinator = PathDecisionCoordinator()
    coordinator._role = "decode"
    coordinator._decode_engine_instance_id = DECODE_TEST_INSTANCE_ID
    return coordinator


def _deliver_frames(receiver: PathDecisionCoordinator, encoded: bytes) -> bytes | None:
    socket = MagicMock(name="router_socket")
    receiver._handle_frames(socket, [b"identity", b"", encoded])
    if socket.send_multipart.call_count == 0:
        return None
    reply_frames = socket.send_multipart.call_args.args[0]
    assert reply_frames[0] == b"identity"
    assert reply_frames[1] == b""
    return reply_frames[2]


def _decision_reply(receiver: PathDecisionCoordinator, decision: PathDecision):
    reply = _deliver_frames(receiver, channel.encode_path_decision(decision))
    assert reply is not None
    return channel.decode_decision_reply(reply)


class _ScriptedSocket:
    def __init__(self, replies: list[bytes | None]) -> None:
        self.replies = list(replies)
        self.sent: list[bytes] = []

    def set_send_timeout(self, timeout_ms: int) -> None:
        pass

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def poll(self, timeout_ms: int) -> bool:
        return bool(self.replies)

    def recv(self) -> bytes:
        return self.replies.pop(0)

    def close(self) -> None:
        pass


def _deliver_with_replies(replies_per_attempt: list[list[bytes | None]], decision: PathDecision):
    sockets = [_ScriptedSocket(replies) for replies in replies_per_attempt]
    open_calls = 0

    @contextmanager
    def opener(endpoint):
        nonlocal open_calls
        socket = sockets[open_calls]
        open_calls += 1
        yield socket

    coordinator = PathDecisionCoordinator.for_prefill(
        socket_opener=opener,
        sleep=lambda _: None,
        poll_timeout_ms=1,
        retry_spacing_s=0,
    )
    try:
        future = coordinator.submit(_ENDPOINT, decision)
        return future, sockets
    finally:
        coordinator.close()


class TestMessageKindEnvelope:
    def test_decision_envelope_round_trips_and_unknown_kind_is_rejected(self):
        decision = _decision(attempt_id=0)
        kind, payload = channel.decode_control_message(channel.encode_path_decision(decision))
        assert kind is channel.ControlMessageKind.DECISION
        assert PathDecision.from_dict(payload) == decision

        foreign = msgspec.msgpack.encode({"kind": "CloseReverseAttempt", "payload": {}})
        with pytest.raises(PathDecisionValidationError):
            channel.decode_control_message(foreign)

    def test_control_message_kind_has_decision_and_abort_members(self):
        assert [member.value for member in channel.ControlMessageKind] == ["Decision", "Abort"]


class TestRegistryMatrix:
    def test_equal_attempt_identical_payload_ack_duplicate(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        decision = _decision(0)

        assert _decision_reply(receiver, decision) is channel.DecisionReplyStatus.ACK
        assert _decision_reply(receiver, decision) is channel.DecisionReplyStatus.ACK
        assert receiver.take_received_decisions() == [decision]
        assert receiver.take_received_decisions() == []

    def test_equal_attempt_conflicting_payload_protocol_error(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        decision = _decision(0)

        assert _decision_reply(receiver, decision) is channel.DecisionReplyStatus.ACK
        assert _decision_reply(receiver, _decision(0, remote_port=6001)) is channel.DecisionReplyStatus.PROTOCOL_ERROR
        assert receiver.take_received_decisions() == [decision]

    def test_greater_attempt_accepted_under_serialization_rule(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        first = _decision(0)
        greater = _decision(1)

        assert _decision_reply(receiver, first) is channel.DecisionReplyStatus.ACK
        assert _decision_reply(receiver, greater) is channel.DecisionReplyStatus.ACK
        assert receiver.take_received_decisions() == [first, greater]
        # Accepting 1 permanently closes the skipped lower numbers.
        assert receiver._closed_through_attempt_ids[_KEY] == 0

    def test_closed_or_lower_attempt_stale_closed_never_enqueued(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        assert _decision_reply(receiver, _decision(1)) is channel.DecisionReplyStatus.ACK

        # A delayed Decision at a closed number is stale, never enqueued.
        assert _decision_reply(receiver, _decision(0)) is channel.DecisionReplyStatus.STALE_CLOSED
        assert len(receiver.take_received_decisions()) == 1

    def test_unknown_logical_key_unknown_request_no_attempt_created(self):
        receiver = _make_receiver()
        assert _decision_reply(receiver, _decision(0)) is channel.DecisionReplyStatus.UNKNOWN_REQUEST
        assert receiver._accepted_decisions == {}
        assert receiver._closed_through_attempt_ids == {}
        assert receiver.take_received_decisions() == []

    def test_accepted_decisions_migration_preserves_legitimate_retries(self):
        receiver = _make_receiver()
        receiver.register_pending(_KEY)
        legacy = PathDecision(
            result=PathDecisionResult(request_key=_KEY, path=PathKind.PE_READ),
            reverse_plan=None,
        )
        # A pre-Stage-2 single-decision registry entry (bare PathDecision).
        receiver._accepted_decisions[_KEY] = legacy

        assert _decision_reply(receiver, legacy) is channel.DecisionReplyStatus.ACK
        assert receiver.take_received_decisions() == []

        conflicting = PathDecision(
            result=PathDecisionResult(
                request_key=_KEY,
                path=PathKind.DE_READ,
                reverse_attempt_id=0,
                prefill_local_tokens=0,
            ),
            reverse_plan=_decision(0).reverse_plan,
        )
        assert _decision_reply(receiver, conflicting) is channel.DecisionReplyStatus.PROTOCOL_ERROR


class TestReplyEncoding:
    def test_response_encoding_round_trip(self):
        for status in channel.DecisionReplyStatus:
            assert channel.decode_decision_reply(channel.encode_decision_reply(status)) is status
        with pytest.raises(PathDecisionValidationError):
            channel.decode_decision_reply(channel.msgspec.msgpack.encode({"status": "BOGUS"}))
        with pytest.raises(PathDecisionValidationError):
            channel.decode_decision_reply(b"\x81")

    def test_retry_idempotent_under_registry_semantics(self):
        decision = _decision(0)
        ack = channel.encode_decision_reply(channel.DecisionReplyStatus.ACK)

        # An uncertain first attempt (timeout) retries identical bytes and the
        # registry absorbs the duplicate.
        future, sockets = _deliver_with_replies([[], [ack]], decision)
        assert future.result(timeout=5) is None
        assert sockets[0].sent == sockets[1].sent

        # Terminal rejections are not retried.
        for status in (
            channel.DecisionReplyStatus.STALE_CLOSED,
            channel.DecisionReplyStatus.PROTOCOL_ERROR,
            channel.DecisionReplyStatus.UNKNOWN_REQUEST,
        ):
            future, sockets = _deliver_with_replies(
                [[channel.encode_decision_reply(status)]],
                decision,
            )
            with pytest.raises(channel.PathDecisionRejectedError) as exc_info:
                future.result(timeout=5)
            assert exc_info.value.status is status
            assert len(sockets) == 1
            assert isinstance(exc_info.value, PathDecisionDeliveryError)

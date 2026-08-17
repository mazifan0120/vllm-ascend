import queue
import threading
from unittest.mock import patch

import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision_channel
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionResult,
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import (
    DecodeControlEndpoint,
    PathDecision,
    PathDecisionCoordinator,
)


class _MidDrainInjectingQueue(queue.SimpleQueue[PathDecision]):
    def __init__(
        self,
        registry_lock: threading.Lock,
        initial_decision: PathDecision,
        late_decision: PathDecision,
    ) -> None:
        super().__init__()
        self._registry_lock = registry_lock
        self._late_decision = late_decision
        self._injection_started = False
        self._producer_started = threading.Event()
        self._producer_finished = threading.Event()
        self._producer_thread: threading.Thread | None = None
        self.put(initial_decision)

    def get_nowait(self) -> PathDecision:
        try:
            return super().get_nowait()
        except queue.Empty:
            if self._injection_started:
                raise
            self._injection_started = True
            drain_holds_registry_lock = self._registry_lock.locked()
            self._producer_thread = threading.Thread(target=self._enqueue_late_decision)
            self._producer_thread.start()
            assert self._producer_started.wait(timeout=5)
            if drain_holds_registry_lock:
                raise
            assert self._producer_finished.wait(timeout=5)
            return super().get_nowait()

    def wait_for_injection(self) -> None:
        assert self._producer_finished.wait(timeout=5)
        assert self._producer_thread is not None
        self._producer_thread.join(timeout=5)
        assert not self._producer_thread.is_alive()

    def _enqueue_late_decision(self) -> None:
        self._producer_started.set()
        with self._registry_lock:
            self.put(self._late_decision)
        self._producer_finished.set()


def test_take_received_decisions_defers_decision_enqueued_during_drain() -> None:
    # Given
    coordinator = PathDecisionCoordinator()
    coordinator._role = "decode"
    initial_decision = PathDecision(
        result=PathDecisionResult(
            request_key=DualPathRequestKey("decode-engine:0:boot", "request-initial", 0),
            path=PathKind.PE_READ,
        ),
        reverse_plan=None,
    )
    late_decision = PathDecision(
        result=PathDecisionResult(
            request_key=DualPathRequestKey("decode-engine:0:boot", "request-late", 0),
            path=PathKind.DE_READ,
            reverse_attempt_id=0,
            prefill_local_tokens=0,
        ),
        reverse_plan=None,
    )
    injecting_queue = _MidDrainInjectingQueue(coordinator._registry_lock, initial_decision, late_decision)
    coordinator._received_decisions = injecting_queue

    # When
    first_batch = coordinator.take_received_decisions()
    injecting_queue.wait_for_injection()
    second_batch = coordinator.take_received_decisions()

    # Then
    assert first_batch == [initial_decision]
    assert second_batch == [late_decision]


class _BlockingRecvSocket:
    def __init__(self, recv_entered: threading.Event, context_terminated: threading.Event) -> None:
        self._recv_entered = recv_entered
        self._context_terminated = context_terminated

    def setsockopt(self, option: int, value: int) -> None:
        return None

    def bind(self, endpoint: str) -> None:
        return None

    def recv_multipart(self) -> list[bytes]:
        self._recv_entered.set()
        assert self._context_terminated.wait(timeout=5)
        raise zmq.ContextTerminated()

    def close(self, linger: int = 0) -> None:
        return None


class _TerminatingContext:
    def __init__(self, socket: _BlockingRecvSocket, context_terminated: threading.Event) -> None:
        self._socket = socket
        self._context_terminated = context_terminated

    def socket(self, socket_type: int) -> _BlockingRecvSocket:
        return self._socket

    def term(self) -> None:
        self._context_terminated.set()


def test_close_waits_for_blocked_receiver_with_unbounded_join() -> None:
    # Given
    recv_entered = threading.Event()
    context_terminated = threading.Event()
    socket = _BlockingRecvSocket(recv_entered, context_terminated)
    context = _TerminatingContext(socket, context_terminated)
    endpoint = DecodeControlEndpoint(host="127.0.0.1", port=24001)
    with patch.object(path_decision_channel.zmq, "Context", return_value=context):
        coordinator = PathDecisionCoordinator.for_decode(
            engine_id="decode-engine",
            data_parallel_rank=0,
            control_endpoint=endpoint,
            boot_id="boot",
        )
    assert recv_entered.wait(timeout=5)
    receiver_thread = coordinator._receiver_thread
    assert receiver_thread is not None
    assert receiver_thread.is_alive()

    # When
    with patch.object(receiver_thread, "join", wraps=receiver_thread.join) as join_receiver:
        coordinator.close()

    # Then
    join_receiver.assert_called_once_with()
    assert not receiver_thread.is_alive()

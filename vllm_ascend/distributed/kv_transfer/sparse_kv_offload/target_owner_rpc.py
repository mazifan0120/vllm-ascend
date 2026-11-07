# SPDX-License-Identifier: Apache-2.0
"""Instance-owned loopback RPC for separately launched vLLM processes.

No inherited pipe descriptors, module rendezvous, pickle or new environment
variable is required. Workers can construct the reverse-retention client before
the scheduler exists. Binding a server is explicit and does not contact peers.

An uncertain call poisons its client: retrying acquire/release automatically
could duplicate a retain or free while a late command is still executing.
Transport timeout is an error, never a DMA-quiescence acknowledgement.
"""

import math
import threading
from uuid import uuid4

import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import ContractError
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetOwnerCall, TargetOwnerReply
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import (
    MAX_OWNER_FRAME_BYTES,
    decode_owner_value,
    encode_owner_value,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_wire import (
    RetentionDispatcher,
    RetentionOp,
    RetentionResult,
    WireRetentionAuthority,
)

RPC_POLL_MILLISECONDS = 100
RPC_BIND_TIMEOUT_SECONDS = 5.0


def _validate_endpoint(endpoint: str, *, bind: bool = False) -> None:
    prefix = "tcp://127.0.0.1:"
    if type(endpoint) is not str or not endpoint.startswith(prefix):
        raise ContractError("Target owner RPC must use an explicit loopback TCP endpoint")
    port = endpoint[len(prefix) :]
    if bind and port == "*":
        return
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ContractError("Target owner RPC requires a valid loopback port")


def _milliseconds(timeout: float) -> int:
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ContractError("Target owner RPC timeout must be positive and finite")
    return max(1, math.ceil(timeout * 1000))


class OwnerRpcClient:
    def __init__(self, endpoint: str, *, timeout: float, context=None) -> None:
        _validate_endpoint(endpoint)
        self._endpoint = endpoint
        self._timeout_ms = _milliseconds(timeout)
        self._lock = threading.Lock()
        self._closed = False
        self._uncertain = False
        self._owns_context = context is None
        self._context = zmq.Context() if context is None else context

    def call(self, command):
        # Reject live objects before creating a socket or poisoning the client.
        if type(command) not in (TargetOwnerCall, RetentionOp):
            raise ContractError("Unsupported target owner RPC command")
        nonce = uuid4().hex
        frame = encode_owner_value((nonce, command))
        with self._lock:
            if self._closed or self._uncertain:
                raise ContractError("Target owner RPC client is closed or has an unresolved call")
            socket = self._context.socket(zmq.REQ)
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
                socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_OWNER_FRAME_BYTES)
                socket.connect(self._endpoint)
                socket.send(frame)
                response = decode_owner_value(socket.recv())
                if type(response) is not tuple or len(response) != 2 or response[0] != nonce:
                    raise ContractError("Target owner RPC response nonce mismatch")
                reply = response[1]
                expected = TargetOwnerReply if type(command) is TargetOwnerCall else RetentionResult
                if type(reply) is not expected:
                    raise ContractError("Target owner RPC returned an unexpected reply type")
                command_id = command.call_id if type(command) is TargetOwnerCall else command.op_id
                reply_id = reply.call_id if type(reply) is TargetOwnerReply else reply.op_id
                if reply_id != command_id:
                    raise ContractError("Target owner RPC returned another command's reply")
                return reply
            except (zmq.ZMQError, ContractError) as error:
                self._uncertain = True
                raise ContractError(f"Target owner RPC outcome is unresolved; retain target blocks: {error}") from error
            finally:
                socket.close(linger=0)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._owns_context:
                    self._context.term()


class RpcRetentionAuthority(WireRetentionAuthority):
    """Same guard semantics as the pipe protocol, without inherited FDs."""

    def __init__(self, endpoint: str, *, timeout: float) -> None:
        # Guard implementation depends on _invoke(), not on a pipe endpoint.
        self._client = OwnerRpcClient(endpoint, timeout=timeout)

    def _invoke(self, operation, allocation) -> None:
        result = self._client.call(RetentionOp(uuid4().hex, operation, allocation))
        if not result.ok:
            error = ContractError if result.error_type == "ContractError" else RuntimeError
            raise error(f"Retention {result.error_type}: {result.error_message}")

    def close(self) -> None:
        self._client.close()

    def stop(self) -> None:
        # A worker cannot shut down the scheduler's authority for other ranks.
        self.close()


class OwnerRpcServer:
    """Own socket/thread/context locally; caller owns authoritative dispatcher.

    The handler runs on this server's thread. A retention dispatcher therefore
    needs an authority whose allocation checks are serialized with the real
    scheduler. Starting this server alone does not wire vLLM free/preempt paths.
    """

    def __init__(self, endpoint: str, dispatcher, *, command_type) -> None:
        _validate_endpoint(endpoint, bind=True)
        if command_type not in (TargetOwnerCall, RetentionOp):
            raise ContractError("Unsupported target owner RPC server command type")
        if not callable(getattr(dispatcher, "dispatch", None)):
            raise ContractError("Target owner RPC server requires an explicit dispatcher")
        self._endpoint = endpoint
        self._dispatcher = dispatcher
        self._command_type = command_type
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._serve, name="target-owner-rpc", daemon=True)
        self._thread.start()
        if not self._ready.wait(RPC_BIND_TIMEOUT_SECONDS):
            self._stop.set()
            raise ContractError("Target owner RPC server did not bind in time")
        if self._error is not None:
            raise ContractError(f"Target owner RPC bind failed: {self._error}") from self._error

    @classmethod
    def retention(cls, endpoint, registry):
        return cls(endpoint, RetentionDispatcher(registry), command_type=RetentionOp)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def _serve(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        try:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_OWNER_FRAME_BYTES)
            socket.setsockopt(zmq.SNDTIMEO, RPC_POLL_MILLISECONDS)
            socket.bind(self._endpoint)
            self._endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            self._ready.set()
            while not self._stop.is_set():
                if not socket.poll(RPC_POLL_MILLISECONDS):
                    continue
                frame = socket.recv()
                nonce, command = "", None
                try:
                    value = decode_owner_value(frame)
                    if type(value) is not tuple or len(value) != 2 or type(value[0]) is not str or not value[0]:
                        raise ContractError("Malformed target owner RPC envelope")
                    nonce, command = value
                    if type(command) is not self._command_type:
                        raise ContractError("Target owner RPC command belongs to another service")
                    reply = self._dispatcher.dispatch(command)
                    expected = TargetOwnerReply if self._command_type is TargetOwnerCall else RetentionResult
                    if type(reply) is not expected:
                        raise ContractError("Target owner RPC dispatcher returned a malformed reply")
                except Exception as error:
                    if self._command_type is TargetOwnerCall:
                        reply = TargetOwnerReply(
                            getattr(command, "call_id", ""), error_type=type(error).__name__, error_message=str(error)
                        )
                    else:
                        reply = RetentionResult(getattr(command, "op_id", ""), False, type(error).__name__, str(error))
                socket.send(encode_owner_value((nonce, reply)))
        except Exception as error:
            self._error = error
            self._ready.set()
        finally:
            socket.close(linger=0)
            context.term()

    def close(self, *, timeout: float = RPC_BIND_TIMEOUT_SECONDS) -> None:
        _milliseconds(timeout)
        self._stop.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise ContractError("Target owner RPC still has a running command; keep its allocations retained")

    def __getstate__(self):
        raise TypeError("Target owner RPC server is process-local")

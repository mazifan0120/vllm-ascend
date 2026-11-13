# SPDX-License-Identifier: Apache-2.0
"""Request-bound data-plane control frames on the existing D side channel."""

import math
import time

import msgspec
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    ContractError,
    TargetClosedError,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import RegionAddress
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_protocol import (
    GET_TARGET_REGIONS_MSG,
    LAYER_RESULTS_MSG,
    MAX_PROOF_SEND_ATTEMPTS,
    PROOF_ACK,
    PROOF_ACK_TARGET_CLOSED,
    TARGET_PEER_ERROR,
    TARGET_PEER_TIMEOUT_SECONDS,
    TARGET_REGIONS_REPLY,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_protocol import (
    MAX_PROOF_BATCH_FRAMES as MAX_PROOF_BATCH_FRAMES,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import target_lease_to_wire


def encode_target_addresses(lease, handles) -> tuple:
    expected = {r.region_id: r for r in lease.regions}
    rows = []
    seen = set()
    for handle in handles:
        if (
            type(handle) is not RegionAddress
            or handle.request != lease.request
            or handle.allocation_generation != lease.allocation.allocation_generation
            or handle.retention_token != lease.allocation.retention_token
            or expected.get(handle.region.region_id) != handle.region
            or handle.region.region_id in seen
        ):
            raise ContractError("Data-plane address response does not match the retained local lease")
        seen.add(handle.region.region_id)
        rows.append(
            (
                handle.region.region_id,
                handle.base_address,
                handle.region.length_bytes,
                handle.region.engine_instance,
                handle.region.registration_generation,
            )
        )
    if seen != set(expected):
        raise ContractError("Data-plane address response omits registered target regions")
    return (
        TARGET_REGIONS_REPLY,
        CONTRACT_VERSION,
        lease.request.request_instance,
        lease.request.transfer_generation,
        lease.allocation.allocation_generation,
        lease.allocation.retention_token,
        tuple(rows),
    )


def decode_target_addresses(lease, response) -> tuple[RegionAddress, ...]:
    if (
        type(response) is not tuple
        or len(response) != 7
        or response[:6]
        != (
            TARGET_REGIONS_REPLY,
            CONTRACT_VERSION,
            lease.request.request_instance,
            lease.request.transfer_generation,
            lease.allocation.allocation_generation,
            lease.allocation.retention_token,
        )
        or type(response[1]) is not int
        or type(response[3]) is not int
        or type(response[4]) is not int
        or type(response[6]) not in (tuple, list)
    ):
        raise ContractError("Peer address response belongs to another request or protocol generation")
    regions = {r.region_id: r for r in lease.regions}
    result = []
    seen = set()
    for row in response[6]:
        if type(row) not in (tuple, list) or len(row) != 5 or type(row[0]) is not str:
            raise ContractError("Malformed peer registered-region address row")
        region_id, base, length, engine, generation = row
        region = regions.get(region_id)
        if (
            region is None
            or region_id in seen
            or type(length) is not int
            or type(generation) is not int
            or (length, engine, generation)
            != (region.length_bytes, region.engine_instance, region.registration_generation)
        ):
            raise ContractError("Peer registered region identity or bounds have changed")
        seen.add(region_id)
        result.append(
            RegionAddress(
                lease.request, lease.allocation.allocation_generation, lease.allocation.retention_token, region, base
            )
        )
    if seen != set(regions):
        raise ContractError("Peer response is missing a registered target region")
    return tuple(result)


def target_peer_call(host: str, port: int, message: tuple, *, timeout: float) -> tuple:
    """One bounded, independently owned socket; no cross-thread socket reuse."""
    if type(host) is not str or not host or type(port) is not int or not 1 <= port <= 65535:
        raise ContractError("Target side-channel address is invalid")
    if type(timeout) not in (float, int) or not math.isfinite(timeout) or timeout <= 0:
        raise ContractError("Target side-channel timeout must be positive and finite")
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    try:
        milliseconds = max(1, math.ceil(timeout * 1000))
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, milliseconds)
        socket.setsockopt(zmq.RCVTIMEO, milliseconds)
        socket.connect(f"tcp://{host}:{port}")
        socket.send(msgspec.msgpack.encode(message))
        response = msgspec.msgpack.decode(socket.recv(), type=tuple)
        if response and response[0] == TARGET_PEER_ERROR:
            if len(response) > 1 and response[1] == "TargetClosedError":
                raise TargetClosedError("Target peer closed this exact delivery")
            raise ContractError(f"Target peer rejected the operation: {response[1:]}")
        return response
    except (zmq.ZMQError, msgspec.DecodeError) as error:
        raise ContractError(f"Target side-channel operation failed: {error}") from error
    finally:
        socket.close(linger=0)


def resolve_manifest_addresses(metadata, plan, producer_rank):
    # Every producer writes only its local Indexer, plus canonical P0 Main.
    # The original owner lease (not the aggregate) authenticates the lookup.
    lease = plan.manifest.for_rank(producer_rank).lease
    response = target_peer_call(
        metadata.remote_host,
        metadata.remote_port + producer_rank,
        (GET_TARGET_REGIONS_MSG, target_lease_to_wire(lease)),
        timeout=TARGET_PEER_TIMEOUT_SECONDS,
    )
    return decode_target_addresses(lease, response)


class ManifestPeerProofs:
    """Persistent REQ sockets exclusively owned by the writer's proof thread."""

    def __init__(self, producer_rank, *, timeout=TARGET_PEER_TIMEOUT_SECONDS):
        if type(producer_rank) is not int or producer_rank < 0:
            raise ContractError("Proof transport requires its explicit producer rank")
        self.producer_rank = producer_rank
        self._timeout_ms = max(1, math.ceil(timeout * 1000))
        self._sockets = {}

    def _socket(self, endpoint):
        socket = self._sockets.get(endpoint)
        if socket is None:
            socket = zmq.Context.instance().socket(zmq.REQ)
            self._sockets[endpoint] = socket
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            socket.connect(endpoint)
        return socket

    def _send_batches(self, messages) -> bool:
        pending = {endpoint: msgspec.msgpack.encode(message) for endpoint, message in messages.items()}
        cancelled = False
        last_error = None
        for _ in range(MAX_PROOF_SEND_ATTEMPTS):
            awaiting = {}
            # Dispatch to every receiver before waiting for ACKs. A D worker
            # may install its visibility dependency before acknowledging.
            for endpoint, encoded in pending.items():
                try:
                    socket = self._socket(endpoint)
                    socket.send(encoded)
                    awaiting[endpoint] = socket
                except zmq.ZMQError as error:
                    last_error = error
                    socket = self._sockets.pop(endpoint, None)
                    if socket is not None:
                        socket.close(linger=0)
            deadline = time.monotonic() + self._timeout_ms / 1000
            for endpoint, socket in awaiting.items():
                try:
                    remaining = max(0, math.ceil((deadline - time.monotonic()) * 1000))
                    if not socket.poll(remaining):
                        raise ContractError(f"Completion proof ACK timed out for {endpoint}")
                    response = socket.recv()
                    if response not in (PROOF_ACK, PROOF_ACK_TARGET_CLOSED):
                        raise ContractError(f"Peer {endpoint} did not acknowledge the exact completion proof")
                    cancelled |= response == PROOF_ACK_TARGET_CLOSED
                    pending.pop(endpoint)
                except (zmq.ZMQError, ContractError) as error:
                    last_error = error
                    self._sockets.pop(endpoint).close(linger=0)
            if not pending:
                return cancelled
        raise ContractError(f"Manifest completion proof was not acknowledged: {last_error}") from last_error

    def __call__(self, metadata, results, terminal) -> bool:
        batches = {}
        for result, receivers in results:
            key = result.key
            frame = (
                b"layer_result_msg",
                CONTRACT_VERSION,
                key.request.request_instance,
                key.request.transfer_generation,
                key.rank,
                key.logical_layer,
                key.operation_id,
                key.component.value,
                key.token_range.start,
                key.token_range.end,
                key.target_scope.backing_id,
                key.target_scope.backing_generation,
                result.status.value,
                result.source_quiesced,
            )
            for rank in receivers:
                batches.setdefault(rank, []).append(frame)
        if terminal is not None:
            status, quiesced = terminal
            request = metadata.remote_target_manifest.lease.request
            for rank in range(len(metadata.remote_target_manifest.targets)):
                batches.setdefault(rank, []).append(
                    (
                        b"rank_terminal_msg",
                        CONTRACT_VERSION,
                        request.request_instance,
                        request.transfer_generation,
                        # Set explicitly at construction by the producer worker.
                        self.producer_rank,
                        status.value,
                        quiesced,
                    )
                )
        return self._send_batches(
            {
                f"tcp://{metadata.remote_host}:{metadata.remote_port + rank}": (
                    LAYER_RESULTS_MSG,
                    CONTRACT_VERSION,
                    tuple(frames),
                )
                for rank, frames in batches.items()
            }
        )

    def close(self):
        for socket in self._sockets.values():
            socket.close(linger=0)
        self._sockets.clear()

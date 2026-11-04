# SPDX-License-Identifier: Apache-2.0
"""Request-bound data-plane control frames on the existing D side channel."""

import math

import msgspec
import zmq

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import CONTRACT_VERSION, ContractError
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import RegionAddress

GET_TARGET_REGIONS_MSG = b"get_target_regions_v1"
TARGET_REGIONS_REPLY = b"target_regions_v1"
TARGET_PEER_ERROR = b"target_peer_error_v1"


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
            raise ContractError(f"Target peer rejected the operation: {response[1:]}")
        return response
    except (zmq.ZMQError, msgspec.DecodeError) as error:
        raise ContractError(f"Target side-channel operation failed: {error}") from error
    finally:
        socket.close(linger=0)

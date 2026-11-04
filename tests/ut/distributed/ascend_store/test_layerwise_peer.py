# SPDX-License-Identifier: Apache-2.0
"""Request-authenticated address lookups on actual CPU allocations/real ZMQ."""

import threading
import unittest
from dataclasses import replace

import msgspec
import zmq

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    ReleaseAck,
    assert_pd_consistent,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_peer import (
    GET_TARGET_REGIONS_MSG,
    decode_target_addresses,
    encode_target_addresses,
    target_peer_call,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import (
    target_lease_from_wire,
    target_lease_to_wire,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetOwnerCall, TargetOwnerDispatcher
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import encode_owner_value


class TestTargetPeer(unittest.TestCase):
    def test_registered_address_lookup_uses_real_retained_owner_and_msgpack(self):
        fixture = OwnerFixture()
        lease = fixture.acquire()
        try:
            handles = fixture.facade.resolve_target_regions(lease)
            response = msgspec.msgpack.decode(
                msgspec.msgpack.encode(encode_target_addresses(lease, handles)), type=tuple
            )
            self.assertEqual(decode_target_addresses(lease, response), handles)
            self.assertEqual({h.region.region_id: h.base_address for h in handles}, fixture.resources._base_addresses)
            with self.assertRaises(ContractError):
                encode_owner_value(handles)
            reply = TargetOwnerDispatcher(fixture.facade).dispatch(
                TargetOwnerCall("not-owner-ipc", "regions", (lease,))
            )
            self.assertEqual(reply.error_type, "ContractError")
            with self.assertRaises(ContractError):
                fixture.facade.resolve_target_regions(
                    replace(lease, allocation=replace(lease.allocation, retention_token="old"))
                )
        finally:
            fixture.finish(lease)

    def test_fragmented_host_backing_round_trips_and_rejects_overlapping_registration(self):
        fixture = OwnerFixture(registration_chunk_bytes=512)
        lease = fixture.acquire()
        try:
            hosts = tuple(r for r in lease.regions if r.memory_kind == "host")
            self.assertEqual(len(hosts), 2)
            self.assertEqual(len({r.backing_id for r in hosts}), 1)
            self.assertEqual({r.backing_offset_bytes for r in hosts}, {0, 512})
            self.assertEqual(target_lease_from_wire(target_lease_to_wire(lease)), lease)
            assert_pd_consistent(lease, producer_tp_size=1, producer_block_sizes=(16,), producer_main_dtype="bfloat16")
            handles = fixture.facade.resolve_target_regions(lease)
            self.assertEqual(decode_target_addresses(lease, encode_target_addresses(lease, handles)), handles)
            bad = replace(hosts[1], backing_offset_bytes=256)
            with self.assertRaisesRegex(ContractError, "overlap"):
                replace(lease, regions=tuple(bad if r == hosts[1] else r for r in lease.regions))
            with self.assertRaises(ContractError):
                replace(
                    lease,
                    regions=tuple(replace(r, registration_generation=9) if r == hosts[1] else r for r in lease.regions),
                )
        finally:
            fixture.finish(lease)

    def test_retired_or_replaced_registration_cannot_resolve_addresses(self):
        fixture = OwnerFixture()
        lease = fixture.acquire()
        fixture.facade.release_target(lease)
        with self.assertRaises(ContractError):
            fixture.facade.resolve_target_regions(lease)
        fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
        fixture.facade.close()
        fixture.te.unregister_buffer()
        fixture.pool.close()

    def test_peer_reply_requires_exact_request_region_bounds_and_generation(self):
        fixture = OwnerFixture()
        lease = fixture.acquire()
        try:
            response = encode_target_addresses(lease, fixture.facade.resolve_target_regions(lease))
            row = response[6][0]
            for bad in (
                (*response[:5], "wrong-token", response[6]),
                (response[0], True, *response[2:]),
                (*response[:6], response[6][1:]),
                (*response[:6], (*response[6], row)),
                (*response[:6], ((row[0], row[1], row[2] + 1, *row[3:]), *response[6][1:])),
                (*response[:6], ((row[0], row[1], row[2], "old-engine", row[4]), *response[6][1:])),
            ):
                with self.subTest(response=bad), self.assertRaises(ContractError):
                    decode_target_addresses(lease, bad)
        finally:
            fixture.finish(lease)

    def test_real_side_channel_socket_resolves_matching_lease_without_owner_ipc_addresses(self):
        fixture = OwnerFixture()
        lease = fixture.acquire()
        ready = threading.Event()
        endpoint = []
        errors = []

        def serve():
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            try:
                socket.bind("tcp://127.0.0.1:*")
                endpoint.append(socket.getsockopt_string(zmq.LAST_ENDPOINT))
                ready.set()
                if not socket.poll(3000):
                    raise TimeoutError("CPU test peer did not receive a query")
                request = msgspec.msgpack.decode(socket.recv(), type=tuple)
                self.assertEqual(request[0], GET_TARGET_REGIONS_MSG)
                local = target_lease_from_wire(request[1])
                handles = fixture.facade.resolve_target_regions(local)
                socket.send(msgspec.msgpack.encode(encode_target_addresses(local, handles)))
            except Exception as error:
                errors.append(error)
            finally:
                socket.close(linger=0)
                context.term()

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            self.assertTrue(ready.wait(1))
            port = int(endpoint[0].rsplit(":", 1)[1])
            result = target_peer_call(
                "127.0.0.1", port, (GET_TARGET_REGIONS_MSG, target_lease_to_wire(lease)), timeout=2
            )
            self.assertEqual(decode_target_addresses(lease, result), fixture.facade.resolve_target_regions(lease))
        finally:
            thread.join(4)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            fixture.finish(lease)

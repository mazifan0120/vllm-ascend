# SPDX-License-Identifier: Apache-2.0
"""Request-authenticated address lookups on actual CPU allocations/real ZMQ."""

import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import zmq

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_coordinator import coordinated
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    ReleaseAck,
    Status,
    TransferResult,
    assert_pd_consistent,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_peer import (
    GET_TARGET_REGIONS_MSG,
    LAYER_RESULTS_MSG,
    PROOF_ACK,
    PROOF_ACK_TARGET_CLOSED,
    ManifestPeerProofs,
    decode_target_addresses,
    encode_target_addresses,
    target_peer_call,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import manifest_requirements
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import (
    target_lease_from_wire,
    target_lease_to_wire,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetOwnerCall, TargetOwnerDispatcher
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import encode_owner_value


class TestTargetPeer(unittest.TestCase):
    def test_real_proof_socket_batches_entries_and_reuses_connection_until_terminal(self):
        provider, _, request, allocation = coordinated(tp=1, layers=1)
        manifest = provider.manifest(provider.acquire_target(request, allocation))
        ready = threading.Event()
        endpoints, messages, errors = [], [], []

        def serve():
            context = zmq.Context()
            socket = context.socket(zmq.ROUTER)
            try:
                socket.bind("tcp://127.0.0.1:*")
                endpoints.append(socket.getsockopt_string(zmq.LAST_ENDPOINT))
                ready.set()
                for ack in (PROOF_ACK, PROOF_ACK_TARGET_CLOSED):
                    if not socket.poll(2000):
                        raise TimeoutError("CPU proof test did not receive frame")
                    identity, _, payload = socket.recv_multipart()
                    messages.append((identity, msgspec.msgpack.decode(payload, type=tuple)))
                    socket.send_multipart((identity, b"", ack))
            except Exception as error:
                errors.append(error)
            finally:
                socket.close(linger=0)
                context.term()

        thread = threading.Thread(target=serve)
        thread.start()
        proofs = ManifestPeerProofs(0, timeout=1)
        try:
            self.assertTrue(ready.wait(1))
            meta = SimpleNamespace(
                remote_target_manifest=manifest,
                remote_host="127.0.0.1",
                remote_port=int(endpoints[0].rsplit(":", 1)[1]),
            )
            key = manifest_requirements(manifest)[0].key
            self.assertFalse(proofs(meta, ((TransferResult(key, Status.SUCCESS, True), (0,)),), None))
            self.assertTrue(proofs(meta, (), (Status.SUCCESS, True)))
            self.assertEqual(messages[0][0], messages[1][0])
            self.assertTrue(all(message[1][0] == LAYER_RESULTS_MSG for message in messages))
            result = messages[0][1][2][0]
            self.assertEqual(
                (result[2], result[6], result[10]),
                (request.request_instance, key.operation_id, key.target_scope.backing_id),
            )
            self.assertEqual(messages[1][1][2][0][-2:], ["success", True])
        finally:
            proofs.close()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            provider.release_target(manifest.lease)

    def test_proof_fanout_sends_all_receivers_before_waiting_and_retries_only_failures(self):
        ready = threading.Event()
        endpoints, messages, errors = [], [], []

        def serve():
            context = zmq.Context()
            sockets = tuple(context.socket(zmq.ROUTER) for _ in range(2))
            try:
                for socket in sockets:
                    socket.bind("tcp://127.0.0.1:*")
                    endpoints.append(socket.getsockopt_string(zmq.LAST_ENDPOINT))
                ready.set()
                identities = []
                # Hold both ACKs until both receivers have the proof.
                for socket in sockets:
                    if not socket.poll(2000):
                        raise TimeoutError("Both D ranks must receive proofs before ACKs are awaited")
                    identity, _, payload = socket.recv_multipart()
                    identities.append(identity)
                    messages.append(msgspec.msgpack.decode(payload))
                sockets[0].send_multipart((identities[0], b"", PROOF_ACK))
                sockets[1].send_multipart((identities[1], b"", b"INVALID"))
                if not sockets[1].poll(2000):
                    raise TimeoutError("D rank with invalid ACK did not receive a retry")
                identity, _, payload = sockets[1].recv_multipart()
                messages.append(msgspec.msgpack.decode(payload))
                sockets[1].send_multipart((identity, b"", PROOF_ACK_TARGET_CLOSED))
                self.assertFalse(sockets[0].poll(50))
            except Exception as error:
                errors.append(error)
            finally:
                for socket in sockets:
                    socket.close(linger=0)
                context.term()

        thread = threading.Thread(target=serve)
        thread.start()
        proofs = ManifestPeerProofs(0, timeout=1)
        try:
            self.assertTrue(ready.wait(1))
            self.assertTrue(proofs._send_batches({endpoints[0]: "rank-0", endpoints[1]: "rank-1"}))
        finally:
            proofs.close()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(messages, ["rank-0", "rank-1", "rank-1"])

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

# SPDX-License-Identifier: Apache-2.0
"""Real process/socket owner IPC tests on CPU, without an NPU visibility claim."""

import json
import multiprocessing
import os
import socket
import threading
import time
import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    Component,
    ContractError,
    LocalRegion,
    ReleaseAck,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetOwnerReply,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_codec import (
    MAX_OWNER_WIRE_DEPTH,
    decode_owner_value,
    encode_owner_value,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_rpc import (
    OwnerRpcClient,
    OwnerRpcServer,
    RpcRetentionAuthority,
)


def _close_fixture(fixture):
    fixture.facade.close()
    fixture.te.unregister_buffer()
    fixture.pool.close()


def _rpc_owner_process(control):
    # Everything owning tensors, a registration or a ZMQ context is created
    # after fork in this process. Only bootstrap strings use the control pipe.
    fixture = OwnerFixture()
    authority = RpcRetentionAuthority(control.recv(), timeout=3)
    fixture.runtime._retention = authority
    server = OwnerRpcServer("tcp://127.0.0.1:*", TargetOwnerDispatcher(fixture.facade), command_type=TargetOwnerCall)
    control.send((os.getpid(), server.endpoint))
    try:
        if control.recv() != "close":
            raise AssertionError("Unexpected CPU test control command")
        server.close()
        _close_fixture(fixture)
        authority.close()
        control.send("closed")
    finally:
        control.close()


class TestOwnerValueCodec(unittest.TestCase):
    def test_actual_catalog_and_lease_round_trip_preserves_enum_keys_and_tuples(self):
        fixture = OwnerFixture()
        lease = fixture.acquire()
        try:
            for original in (fixture.facade.catalog(), lease, {Component.MAIN: (Component.INDEXER,)}):
                decoded = decode_owner_value(encode_owner_value(original))
                self.assertEqual(decoded, original)
                if isinstance(decoded, dict):
                    self.assertIs(next(iter(decoded)), Component.MAIN)
        finally:
            fixture.finish(lease)

    def test_live_region_and_arbitrary_runtime_objects_are_not_wire_values(self):
        for value in (LocalRegion(0, 0, 1234, 64), threading.Event(), object(), 1.5):
            with self.subTest(value=type(value).__name__), self.assertRaises(ContractError):
                encode_owner_value(value)

    def test_unknown_types_bad_versions_duplicates_and_malformed_records_fail_closed(self):
        frames = (
            [True, None],
            [2, None],
            [1, ["record", "LocalRegion", []]],
            [1, ["record", "RequestKey", []]],
            [1, ["enum", "Component", "typo"]],
            [1, ["dict", [["key", 1], ["key", 2]]]],
            [1, ["dict", [[[], 1]]]],
            [1, ["tuple", "not-a-list"]],
        )
        for frame in frames:
            with self.subTest(frame=frame), self.assertRaises(ContractError):
                decode_owner_value(json.dumps(frame).encode())

    def test_excessive_nesting_is_rejected(self):
        value = None
        for _ in range(MAX_OWNER_WIRE_DEPTH + 2):
            value = (value,)
        with self.assertRaises(ContractError):
            encode_owner_value(value)


class TestOwnerRpc(unittest.TestCase):
    def test_reverse_retention_and_owner_lifecycle_use_distinct_servers(self):
        fixture = OwnerFixture()
        retain_server = OwnerRpcServer.retention("tcp://127.0.0.1:*", fixture.retention)
        authority = RpcRetentionAuthority(retain_server.endpoint, timeout=2)
        fixture.runtime._retention = authority
        owner_server = OwnerRpcServer(
            "tcp://127.0.0.1:*", TargetOwnerDispatcher(fixture.facade), command_type=TargetOwnerCall
        )
        transport = OwnerRpcClient(owner_server.endpoint, timeout=3)
        client = TargetProviderClient(transport)
        try:
            self.assertEqual(client.catalog(), fixture.facade.catalog())
            lease = client.acquire_target(fixture.request, fixture.allocation)
            self.assertEqual(client.get_target_views(lease), lease.views)
            # CPU receive dependencies stay on the owner; they are never sent
            # over either RPC service.
            fixture.ready(lease)
            client.release_target(lease)
            fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
        finally:
            transport.close()
            owner_server.close()
            authority.close()
            retain_server.close()
            _close_fixture(fixture)

    def test_explicit_dispatch_failure_does_not_poison_next_command(self):
        fixture = OwnerFixture()
        server = OwnerRpcServer(
            "tcp://127.0.0.1:*", TargetOwnerDispatcher(fixture.facade), command_type=TargetOwnerCall
        )
        transport = OwnerRpcClient(server.endpoint, timeout=1)
        try:
            reply = transport.call(TargetOwnerCall("bad", "unsupported", ()))
            self.assertEqual(reply.error_type, "ContractError")
            self.assertEqual(TargetProviderClient(transport).catalog(), fixture.facade.catalog())
        finally:
            transport.close()
            server.close()
            _close_fixture(fixture)

    def test_timeout_quarantines_client_and_does_not_retry_late_command(self):
        entered, unblock = threading.Event(), threading.Event()
        calls = []

        class DelayedDispatcher:
            def dispatch(self, command):
                calls.append(command.call_id)
                entered.set()
                if not unblock.wait(2):
                    raise TimeoutError("CPU test dispatcher remained blocked")
                return TargetOwnerReply(command.call_id)

        server = OwnerRpcServer("tcp://127.0.0.1:*", DelayedDispatcher(), command_type=TargetOwnerCall)
        client = OwnerRpcClient(server.endpoint, timeout=0.15)
        try:
            with self.assertRaisesRegex(ContractError, "unresolved"):
                client.call(TargetOwnerCall("one", "release", ()))
            self.assertTrue(entered.is_set())
            with self.assertRaisesRegex(ContractError, "unresolved"):
                client.call(TargetOwnerCall("two", "release", ()))
            with self.assertRaisesRegex(ContractError, "running command"):
                server.close(timeout=0.01)
            self.assertEqual(calls, ["one"])
        finally:
            unblock.set()
            client.close()
            server.close()

    def test_bind_collision_and_nonlocal_endpoint_are_rejected(self):
        fixture = OwnerFixture()
        dispatcher = TargetOwnerDispatcher(fixture.facade)
        server = OwnerRpcServer("tcp://127.0.0.1:*", dispatcher, command_type=TargetOwnerCall)
        try:
            with self.assertRaisesRegex(ContractError, "bind failed"):
                OwnerRpcServer(server.endpoint, dispatcher, command_type=TargetOwnerCall)
            for endpoint in ("tcp://0.0.0.0:12345", "tcp://127.0.0.1:0", "ipc:///tmp/owner"):
                with self.subTest(endpoint=endpoint), self.assertRaises(ContractError):
                    OwnerRpcClient(endpoint, timeout=1)
        finally:
            server.close()
            _close_fixture(fixture)

    def test_client_can_be_constructed_before_scheduler_binds(self):
        # Model initialization precedes scheduler KVCacheManager construction.
        # Client construction must not block waiting for that reverse service.
        with socket.socket() as free_port:
            free_port.bind(("127.0.0.1", 0))
            endpoint = f"tcp://127.0.0.1:{free_port.getsockname()[1]}"
        start = time.monotonic()
        authority = RpcRetentionAuthority(endpoint, timeout=1)
        fixture = OwnerFixture()
        server = OwnerRpcServer.retention(endpoint, fixture.retention)
        try:
            self.assertLess(time.monotonic() - start, 1)
            guard = authority.retain(fixture.allocation)
            guard.validate()
            guard.release()
            guard.release()
            fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
        finally:
            authority.close()
            server.close()
            _close_fixture(fixture)

    @unittest.skipUnless(hasattr(os, "fork"), "CPU process test requires Linux fork")
    def test_independent_owner_process_nests_reverse_rpc_to_parent_scheduler(self):
        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe()
        process = context.Process(target=_rpc_owner_process, args=(child,))
        # Fork before constructing any socket/context/server on the parent.
        process.start()
        child.close()
        fixture = OwnerFixture()
        retain_server = OwnerRpcServer.retention("tcp://127.0.0.1:*", fixture.retention)
        transport = None
        try:
            parent.send(retain_server.endpoint)
            self.assertTrue(parent.poll(10))
            pid, endpoint = parent.recv()
            self.assertNotEqual(pid, os.getpid())
            transport = OwnerRpcClient(endpoint, timeout=4)
            client = TargetProviderClient(transport)
            self.assertEqual(client.catalog().tp_size, 1)
            lease = client.acquire_target(fixture.request, fixture.allocation)
            self.assertEqual(client.get_target_views(lease), lease.views)
            # CPU-only unused target: no transfer was submitted. This release
            # exercises reverse guard release, not remote DMA quiescence.
            client.release_target(lease)
            fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
            parent.send("close")
            self.assertTrue(parent.poll(10))
            self.assertEqual(parent.recv(), "closed")
            process.join(5)
            self.assertEqual(process.exitcode, 0)
        finally:
            if transport is not None:
                transport.close()
            if process.is_alive():
                process.terminate()
                process.join(5)
            parent.close()
            retain_server.close()
            _close_fixture(fixture)

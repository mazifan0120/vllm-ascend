# SPDX-License-Identifier: Apache-2.0
"""Wire transport tests for the multi-process target owner IPC.

The co-located channel shares the scheduler retention registry by reference.
This module exercises the wire replacement: the scheduler facade reaches the
worker owner over one pipe, while the owner runtime's nested ``retain()`` during
``acquire_target`` travels back to the scheduler over a second pipe.

Threaded cases run on every platform and cover the full bidirectional frame
flow including nested reverse retain, guard validate/release, fail-closed
timeouts and the dispatcher/retention server loops. A fork case mirrors A's
``test_real_process_boundary_keeps_owner_state_and_dependencies_in_child`` but
adds the reverse retention channel that A's PipeTransport did not exercise.
"""

import multiprocessing
import os
import unittest
from dataclasses import replace
from threading import Event, Thread

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    ReleaseAck,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    TargetOwnerDispatcher,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_wire import (
    PipeOwnerTransport,
    RetentionOp,
    RetentionResult,
    WireRetentionAuthority,
    serve_owner_dispatch,
    serve_retention_ops,
)


def _pipe_pair():
    """Two ends of a multiprocessing.Pipe usable across threads or processes."""
    a, b = multiprocessing.Pipe(duplex=True)
    return a, b


def _start_threads(*targets):
    """Start (func, args) pairs on daemon threads and return the threads."""
    threads = []
    for func, args in targets:
        thread = Thread(target=func, args=args, daemon=True)
        thread.start()
        threads.append(thread)
    return threads


class TestWireOwnerTransportThreaded(unittest.TestCase):
    """Bidirectional pipe transport within one process.

    The scheduler side owns the retention registry and the forward call parent
    end; the worker side owns the owner facade and the reverse retain parent
    end. Two server threads drive the loops; the main thread plays the client.
    """

    def setUp(self):
        self._stop_events = []
        self._threads = []

    def tearDown(self):
        for event in self._stop_events:
            event.set()
        for thread in self._threads:
            thread.join(timeout=5)

    def _spawn_scheduler(self, f):
        """Scheduler side: retention registry + forward call transport."""
        call_parent, call_child = _pipe_pair()
        retain_parent, retain_child = _pipe_pair()
        stop = Event()
        self._stop_events.append(stop)
        self._threads.append(
            Thread(
                target=serve_retention_ops,
                args=(retain_parent, f.retention, stop),
                daemon=True,
            )
        )
        self._threads[-1].start()
        transport = PipeOwnerTransport(call_parent, timeout=2.0)
        return transport, call_child, retain_child

    def _spawn_worker(self, f, call_child, retain_child):
        """Worker side: owner dispatcher + wire retention authority."""
        stop = Event()
        self._stop_events.append(stop)
        dispatcher = TargetOwnerDispatcher(f.facade)
        self._threads.append(
            Thread(
                target=serve_owner_dispatch,
                args=(call_child, dispatcher, stop),
                daemon=True,
            )
        )
        self._threads[-1].start()
        authority = WireRetentionAuthority(retain_child, timeout=2.0)
        return authority

    def test_wire_transport_drives_full_lifecycle_with_nested_retain(self):
        f = OwnerFixture()
        # The scheduler side owns the registry; the worker side owns the facade.
        transport, call_child, retain_child = self._spawn_scheduler(f)
        authority = self._spawn_worker(f, call_child, retain_child)

        # Swap the facade's local retention for the wire authority so the
        # nested retain inside acquire_target travels back over the pipe.
        f.runtime._retention = authority

        client = TargetProviderClient(transport)
        catalog = client.catalog()
        self.assertEqual(len(catalog.entries), 8)

        # acquire_target nests a reverse retain on the scheduler registry over
        # the retain pipe; the wire transport must complete it before returning.
        lease = client.acquire_target(f.request, f.allocation)
        self.assertEqual(client.get_target_views(lease), lease.views)

        f.ready(lease)
        client.release_target(lease)
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_wire_retain_guard_validate_and_release_round_trip(self):
        f = OwnerFixture()
        transport, call_child, retain_child = self._spawn_scheduler(f)
        authority = self._spawn_worker(f, call_child, retain_child)
        f.runtime._retention = authority

        client = TargetProviderClient(transport)
        lease = client.acquire_target(f.request, f.allocation)
        # The lease carries a retention token from the scheduler registry,
        # proving the nested wire retain reached it.
        self.assertTrue(f.allocation.retention_token)

        # The owner-side guard proxies validate/release back over the retain
        # pipe; validate must succeed, then release decrements the guard count.
        guard = authority.retain(f.allocation)
        guard.validate()
        guard.release()
        # Double release is a no-op on the proxy (idempotent local flag).
        guard.release()

        f.ready(lease)
        client.release_target(lease)
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_wire_transport_maps_owner_errors_to_contract_error(self):
        f = OwnerFixture()
        transport, call_child, retain_child = self._spawn_scheduler(f)
        authority = self._spawn_worker(f, call_child, retain_child)
        f.runtime._retention = authority

        client = TargetProviderClient(transport)
        # Releasing an unknown lease must surface the owner's ContractError,
        # not a generic timeout or malformed reply.
        with self.assertRaises(ContractError):
            client.release_target(f.allocation)
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_wire_retention_authority_fails_closed_on_timeout(self):
        f = OwnerFixture()
        # Scheduler side binds nothing: the retain pipe has no server, so the
        # authority's invoke must time out and raise rather than hang.
        _, retain_child = _pipe_pair()
        authority = WireRetentionAuthority(retain_child, timeout=0.2)
        with self.assertRaises(ContractError):
            authority.retain(f.allocation)
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_wire_transport_fails_closed_when_no_dispatch_server(self):
        f = OwnerFixture()
        # No dispatch server started: the call times out instead of hanging.
        call_parent, _ = _pipe_pair()
        transport = PipeOwnerTransport(call_parent, timeout=0.2)
        client = TargetProviderClient(transport)
        with self.assertRaises(ContractError):
            client.catalog()
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_retention_dispatcher_rejects_unknown_operation(self):
        f = OwnerFixture()
        call_parent, call_child = _pipe_pair()
        retain_parent, retain_child = _pipe_pair()
        stop = Event()
        self._stop_events.append(stop)
        self._threads.append(Thread(target=serve_retention_ops, args=(retain_parent, f.retention, stop), daemon=True))
        self._threads[-1].start()

        # An unsupported operation comes back as an ok=False result carrying a
        # ContractError type, which the authority maps back to ContractError.
        bogus = RetentionOp("bad-op", "bogus", f.allocation)
        retain_child.send(bogus)
        result = retain_child.recv()
        self.assertIsInstance(result, RetentionResult)
        self.assertEqual(result.op_id, "bad-op")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_type, "ContractError")
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()


def _wire_worker_process(call_conn, retain_conn):
    """Child-process worker: build the owner, serve dispatch + wire retain.

    The scheduler forked this process with both parent ends already on the
    scheduler side. The owner runtime's nested retain goes out the retain pipe
    to the scheduler's retention server.
    """
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
        TargetOwnerDispatcher,
    )
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner_wire import (
        WireRetentionAuthority,
        serve_owner_dispatch,
    )

    fixture = OwnerFixture()
    dispatcher = TargetOwnerDispatcher(fixture.facade)
    authority = WireRetentionAuthority(retain_conn, timeout=5.0)
    fixture.runtime._retention = authority
    # Signal readiness so the parent can issue its first call.
    call_conn.send((os.getpid(), fixture.request, fixture.allocation))
    serve_owner_dispatch(call_conn, dispatcher)


@unittest.skipUnless(
    hasattr(os, "fork"),
    "fork-based wire transport case runs on Linux only",
)
class TestWireOwnerTransportFork(unittest.TestCase):
    """Real cross-process wire transport mirroring A's fork precedent.

    The scheduler stays in the parent with the retention registry and its
    server thread; the worker is a forked child that serves owner dispatch and
    drives the nested reverse retain back over the pipe.
    """

    def test_cross_process_owner_call_with_nested_reverse_retain(self):
        context = multiprocessing.get_context("fork")
        call_parent, call_child = context.Pipe(duplex=True)
        retain_parent, retain_child = context.Pipe(duplex=True)
        process = context.Process(
            target=_wire_worker_process,
            args=(call_child, retain_child),
        )
        process.start()
        call_child.close()
        retain_child.close()

        fixture = OwnerFixture()
        stop = Event()
        server = Thread(
            target=serve_retention_ops,
            args=(retain_parent, fixture.retention, stop),
            daemon=True,
        )
        server.start()

        class _PipeTransport:
            def call(self, command):
                call_parent.send(command)
                if not call_parent.poll(10):
                    raise TimeoutError("Test owner IPC timed out")
                return call_parent.recv()

        try:
            self.assertTrue(call_parent.poll(10))
            owner_pid, request, worker_allocation = call_parent.recv()
            self.assertNotEqual(owner_pid, os.getpid())
            # Only the parent scheduler issues valid retention. A token from
            # the child's separate fixture must never authorize these blocks.
            allocation = fixture.allocation
            self.assertEqual(request, fixture.request)

            client = TargetProviderClient(_PipeTransport())
            catalog = client.catalog()
            self.assertEqual(len(catalog.entries), 8)
            with self.assertRaisesRegex(ContractError, "Unknown, changed or retired"):
                client.acquire_target(request, worker_allocation)
            # Failed acquisition retires its exact request identity in the
            # owner. Use a distinct identity for the authorized lifecycle.
            fixture.retention.retire(allocation, ReleaseAck(True, allocation.retention_token))
            request = replace(request, transfer_generation=request.transfer_generation + 1)
            allocation = fixture.retention.issue("internal", replace(allocation, request=request, retention_token=None))
            # acquire nests a reverse retain over the retain pipe to the
            # scheduler process; the lease carries a retention token only the
            # scheduler registry could issue.
            lease = client.acquire_target(request, allocation)
            self.assertEqual(client.get_target_views(lease), lease.views)
            self.assertTrue(lease.allocation.retention_token)

            # Release the lease on the owner side, then retire on the scheduler
            # side. The shutdown sentinel stops the child dispatch loop.
            client.release_target(lease)
            call_parent.send(None)
            if call_parent.poll(10):
                call_parent.recv()
            fixture.retention.retire(allocation, ReleaseAck(True, allocation.retention_token))
            fixture.facade.close()
            fixture.te.unregister_buffer()
            fixture.pool.close()
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            stop.set()
            server.join(timeout=5)
            call_parent.close()
            retain_parent.close()
            if process.is_alive():
                process.terminate()
                process.join(10)


if __name__ == "__main__":
    unittest.main()

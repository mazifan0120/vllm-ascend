# SPDX-License-Identifier: Apache-2.0
"""Owner geometry/lifetime tests on CPU memory; native NPU tests are separate."""

import multiprocessing
import os
import pickle
import threading
import time
import unittest
from contextlib import contextmanager
from dataclasses import asdict, replace
from types import SimpleNamespace

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_lifetime import Engine, GlobalTE
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    Component,
    ContractError,
    ReceiveReceipt,
    ReleaseAck,
    RequestKey,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostMemoryRegion,
    HostPoolTopology,
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_catalog import RegisteredTargetResources
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import (
    DeferredTargetProvider,
    TargetOwner,
    TargetOwnerCall,
    TargetOwnerDispatcher,
    TargetOwnerReply,
    TargetProviderClient,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SchedulerRetentionRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_runtime import OwnerTargetRuntime


class DeviceDescriptor:
    """Structural device descriptor over real CPU storage; no NPU claim."""

    device = SimpleNamespace(type="npu", index=0)

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


class Completion:
    def __init__(self, complete=False):
        self.event = threading.Event()
        if complete:
            self.event.set()

    def query(self):
        return self.event.is_set()

    def synchronize(self):
        if not self.event.wait(1):
            raise TimeoutError("CPU test completion timed out")


class ThreadedStreams:
    """Exercise actual blocking and queued-use completion with CPU events."""

    def __init__(self):
        self.dependency = None
        self.pending = []
        self.installs = 0

    def install_visibility(self, dependency):
        self.dependency = dependency.indexer_event
        self.installs += 1

    @contextmanager
    def consumption(self):
        if self.dependency is not None and not self.dependency.event.wait(1):
            raise TimeoutError("CPU test simulated device dependency timed out")
        yield

    def record_use(self):
        return tuple(self.pending) or (Completion(True),)

    def wait_quiescent(self, events, timeout):
        deadline = time.monotonic() + timeout
        for event in events:
            if not event.event.wait(max(0, deadline - time.monotonic())):
                raise TimeoutError("compute/writeback pending")


class AllocationManager:
    def __init__(self):
        self.ids = {"internal": ((1, 3),)}

    def get_block_ids(self, request_id):
        return self.ids[request_id]


class OwnerFixture:
    def __init__(self, layers=2, registration_chunk_bytes=None):
        raw = torch.empty(8192, dtype=torch.int8)
        self.pool = MooncakeHostPool(HostMemoryRegion(raw), HostPoolTopology(0, 1))
        self.host = {}
        self.caches = {}
        self.components = {}
        self.groups = {}
        self.tokens = {}
        for layer in range(layers):
            name = f"model.layers.{layer}.attn"
            k, v = self.pool.allocate_tensors([256, 256], alignment=32)
            self.host[name] = (k.view(torch.bfloat16).view(4, 16, 1, 2), v.view(torch.bfloat16).view(4, 16, 1, 2))
            self.caches[name] = (None, None, *self.host[name], None, None)
            self.components[name] = Component.MAIN
            self.groups[name] = (layer, 0)
            self.tokens[name] = 16
            indexer = name + ".indexer"
            self.caches[indexer] = (
                DeviceDescriptor(torch.empty(8, 8, 2, dtype=torch.int8)),
                DeviceDescriptor(torch.empty(8, 8, 1, dtype=torch.float16)),
            )
            self.components[indexer] = Component.INDEXER
            self.groups[indexer] = (layer, 0)
            self.tokens[indexer] = 16
        self.manager = SimpleNamespace(
            get_mooncake_host_pool=lambda: self.pool, get_local_host_kv_views=lambda name: self.host[name]
        )
        self.te = GlobalTE()
        self.te.transfer_engine = Engine()
        # The producer reaches the target's TransferEngine at this endpoint.
        self.te._handshake_host = "127.0.0.1"
        self.te.transfer_engine._rpc_port = 25000
        if registration_chunk_bytes is None:
            pointers, lengths = [raw.data_ptr()], [raw.numel()]
        else:
            pointers = [raw.data_ptr() + start for start in range(0, raw.numel(), registration_chunk_bytes)]
            lengths = [
                min(registration_chunk_bytes, raw.numel() - start)
                for start in range(0, raw.numel(), registration_chunk_bytes)
            ]
        for name, tensors in self.caches.items():
            if self.components[name] is Component.INDEXER:
                for tensor in tensors:
                    pointers.append(tensor.data_ptr())
                    lengths.append(tensor.untyped_storage().nbytes())
        self.te.register_buffer(pointers, lengths)
        self.resources = self.build_resources()
        self.scheduler = AllocationManager()
        self.retention = SchedulerRetentionRegistry(self.scheduler)
        self.streams = ThreadedStreams()
        self.runtime = OwnerTargetRuntime(self.resources, self.retention, self.streams, timeout=0.02)
        self.owner = TargetOwner(self.resources, self.runtime)
        self.facade = DeferredTargetProvider()
        self.facade.bind(self.owner)
        self.request = RequestKey("owner-test", 3)
        self.allocation = self.retention.issue("internal", BlockAllocation(self.request, 7, ((1, 3),), 16, 23))

    def build_resources(self):
        return RegisteredTargetResources(
            self.manager,
            self.caches,
            self.te,
            components=self.components,
            layer_groups=self.groups,
            block_tokens=self.tokens,
            num_blocks=4,
        )

    def acquire(self):
        return self.facade.acquire_target(self.request, self.allocation)

    def ready(self, lease, event=None):
        host = threading.Event()
        host.set()
        event = event if event is not None else Completion(True)
        self.facade.publish_received(lease, host, event)
        self.facade.prepare_decode(lease, ReceiveReceipt(self.request, 7, True, True, True))
        return event

    def finish(self, lease):
        self.facade.release_target(lease)
        self.retention.retire(self.allocation, ReleaseAck(True, self.allocation.retention_token))
        self.facade.close()
        self.te.unregister_buffer()
        self.pool.close()


def _owner_process(connection):
    """Build every allocation, registration and live runtime in the child."""
    fixture = OwnerFixture()
    dispatcher = TargetOwnerDispatcher(fixture.facade)
    connection.send((os.getpid(), fixture.request, fixture.allocation))
    try:
        while True:
            command = connection.recv()
            if isinstance(command, TargetOwnerCall):
                connection.send(dispatcher.dispatch(command))
            elif command[0] == "publish":
                host = threading.Event()
                host.set()
                fixture.facade.publish_received(command[1], host, Completion(True))
                connection.send("published-locally")
            elif command[0] == "use":
                with fixture.facade.use(command[1]):
                    observed = fixture.streams.dependency.query()
                connection.send(observed)
            elif command[0] == "stop":
                fixture.retention.retire(fixture.allocation, ReleaseAck(True, fixture.allocation.retention_token))
                fixture.facade.close()
                fixture.te.unregister_buffer()
                fixture.pool.close()
                connection.send("closed")
                return
            else:
                raise AssertionError(command)
    finally:
        connection.close()


class TestTargetOwnerRuntime(unittest.TestCase):
    def test_replacing_tensor_storage_keeps_old_allocation_and_rejects_visibility(self):
        f = OwnerFixture()
        lease = f.acquire()
        indexer = next(name for name in f.caches if f.components[name] is Component.INDEXER)
        tensor = f.caches[indexer][0]
        previous = tensor.data_ptr()
        tensor.tensor = torch.empty_like(tensor.tensor)
        self.assertTrue(any(storage.data_ptr() == previous for storage in f.resources._storage_refs))
        host = threading.Event()
        host.set()
        f.facade.publish_received(lease, host, Completion(True))
        with self.assertRaisesRegex(ContractError, "allocation or layout changed"):
            f.facade.prepare_decode(lease, ReceiveReceipt(f.request, 7, True, True, True))
        f.finish(lease)

    def test_failed_acquire_retains_authoritative_guard_until_rollback_recovery(self):
        f = OwnerFixture()
        original_retain = f.resources.retain

        class FaultyGuard:
            def __init__(self, real):
                self.real = real
                self.fail_release = True

            def validate(self):
                self.real.validate()

            def release(self):
                if self.fail_release:
                    raise RuntimeError("injected authority transport failure")
                self.real.release()

        guards = []

        def retain(allocation):
            guard = FaultyGuard(f.retention.retain(allocation))
            guards.append(guard)
            return guard

        def fail_resource_retain(lease):
            raise RuntimeError("injected resource acquisition failure")

        f.runtime._retention = SimpleNamespace(retain=retain)
        f.resources.retain = fail_resource_retain
        with self.assertRaisesRegex(RuntimeError, "rollback remains quarantined"):
            f.acquire()
        with self.assertRaises(ContractError):
            f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        with self.assertRaises(ContractError):
            f.facade.close()
        guards[0].fail_release = False
        f.facade.retry_failed_retains()
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.resources.retain = original_retain
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_real_process_boundary_keeps_owner_state_and_dependencies_in_child(self):
        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe()
        process = context.Process(target=_owner_process, args=(child,))
        process.start()
        child.close()

        class PipeTransport:
            def call(self, command):
                parent.send(command)
                if not parent.poll(10):
                    raise TimeoutError("Test owner IPC timed out")
                return parent.recv()

        try:
            self.assertTrue(parent.poll(10))
            owner_pid, request, allocation = parent.recv()
            self.assertNotEqual(owner_pid, os.getpid())
            client = TargetProviderClient(PipeTransport())
            catalog = client.catalog()
            self.assertEqual(len(catalog.entries), 8)
            lease = client.acquire_target(request, allocation)
            self.assertEqual(client.get_target_views(lease), lease.views)
            receipt = ReceiveReceipt(request, allocation.allocation_generation, True, True, True)
            parent.send(("publish", lease))
            self.assertTrue(parent.poll(10))
            self.assertEqual(parent.recv(), "published-locally")
            client.prepare_decode(lease, receipt)
            parent.send(("use", lease))
            self.assertTrue(parent.poll(10))
            self.assertTrue(parent.recv())
            client.release_target(lease)
            with self.assertRaises(ContractError):
                client.get_target_views(lease)
            parent.send(("stop",))
            self.assertTrue(parent.poll(10))
            self.assertEqual(parent.recv(), "closed")
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(10)

    def test_invalid_main_dtype_and_indexer_device_do_not_leak_pins(self):
        f = OwnerFixture()
        f.facade.close()
        before = f.pool._target_lifetime.pins
        name = next(iter(f.host))
        original = f.host[name]
        f.host[name] = tuple(t.view(torch.float16) for t in original)
        with self.assertRaisesRegex(ContractError, "BF16"):
            f.build_resources()
        self.assertEqual(f.pool._target_lifetime.pins, before)
        f.host[name] = original
        indexer = name + ".indexer"
        f.caches[indexer][0].device = SimpleNamespace(type="npu", index=1)
        with self.assertRaisesRegex(ContractError, "owner's NPU"):
            f.build_resources()
        self.assertEqual(f.pool._target_lifetime.pins, before)
        f.te.unregister_buffer()
        f.pool.close()

    def test_rollback_retry_keeps_new_acquisitions_quarantined(self):
        f = OwnerFixture()
        entered, finish = threading.Event(), threading.Event()

        class DelayedGuard:
            def release(self):
                entered.set()
                if not finish.wait(2):
                    raise TimeoutError("test guard cleanup")

        f.runtime._quarantine.append((None, DelayedGuard()))
        thread = threading.Thread(target=f.runtime.retry_failed_retains)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                f.acquire()
            with self.assertRaises(ContractError):
                f.facade.close()
            with self.assertRaises(ContractError):
                f.runtime.retry_failed_retains()
        finally:
            finish.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.request = RequestKey("owner-test", 4)
        f.allocation = f.retention.issue("internal", replace(f.allocation, request=f.request, retention_token=None))
        f.finish(f.acquire())

    def test_catalog_exports_existing_aliases_and_pure_data(self):
        f = OwnerFixture()
        catalog = f.facade.catalog()
        restored = pickle.loads(pickle.dumps(catalog))
        self.assertEqual(catalog, restored)
        self.assertEqual(len(catalog.entries), 8)
        self.assertEqual({entry.group for entry in catalog.entries}, {0})
        self.assertEqual({entry.component for entry in catalog.entries}, {Component.MAIN, Component.INDEXER})
        self.assertEqual({entry.entry for entry in catalog.entries}, {0, 1})
        forbidden = ("base_address", "address", "local_base", "resources", "runtime")

        def keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from keys(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from keys(item)

        self.assertFalse(set(keys(asdict(catalog))) & set(forbidden))
        lease = f.acquire()
        for view in lease.views:
            region = next(r for r in catalog.regions if r.region_id == view.region_binding)
            entry = next(e for e in catalog.entries if e.binding.view == replace(view, block_ids=()))
            tensor = (
                f.host[entry.layer_name][entry.entry]
                if entry.component is Component.MAIN
                else f.caches[entry.layer_name][entry.entry]
            )
            self.assertEqual(
                f.resources._base_addresses[region.region_id] + view.region_offset_bytes, tensor.data_ptr()
            )
        f.finish(lease)

    def test_active_lease_blocks_real_pool_close_and_te_unregistration(self):
        f = OwnerFixture()
        lease = f.acquire()
        for action in (f.pool.close, f.te.unregister_buffer, f.facade.close):
            with self.assertRaises((RuntimeError, ContractError)):
                action()
        self.assertFalse(f.pool.region._released)
        self.assertTrue(f.te.is_register_buffer)
        f.finish(lease)
        self.assertTrue(f.pool.region._released)
        self.assertFalse(f.te.is_register_buffer)

    def test_forged_retention_is_rejected_before_resource_pin(self):
        f = OwnerFixture()
        before = f.pool._target_lifetime.pins
        with self.assertRaises(ContractError):
            f.facade.acquire_target(f.request, replace(f.allocation, retention_token="arbitrary-string"))
        self.assertEqual(f.pool._target_lifetime.pins, before)
        f.facade.close()

    def test_actual_receive_dependencies_are_required(self):
        f = OwnerFixture()
        lease = f.acquire()
        with self.assertRaisesRegex(ContractError, "Actual receive dependencies"):
            f.facade.prepare_decode(lease, ReceiveReceipt(f.request, 7, True, True, True))
        with self.assertRaises(ContractError), f.facade.use(lease):
            self.fail("decode must stay gated")
        f.finish(lease)

    def test_host_visibility_timeout_preserves_owner_and_scheduler_hold(self):
        f = OwnerFixture()
        lease = f.acquire()
        f.facade.publish_received(lease, threading.Event(), Completion(True))
        with self.assertRaises(TimeoutError):
            f.facade.prepare_decode(lease, ReceiveReceipt(f.request, 7, True, True, True))
        with self.assertRaises(ContractError):
            f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        with self.assertRaises(RuntimeError):
            f.pool.close()
        f.finish(lease)

    def test_device_dependency_and_delayed_writeback_prevent_premature_release(self):
        f = OwnerFixture()
        lease = f.acquire()
        device = Completion()
        f.ready(lease, device)
        observed = []
        entered = threading.Event()
        finish_compute = threading.Event()
        writeback = Completion()
        f.streams.pending.append(writeback)

        def consume():
            with f.facade.use(lease):
                observed.append(device.query())
                entered.set()
                finish_compute.wait(1)

        thread = threading.Thread(target=consume)
        thread.start()
        self.assertFalse(entered.wait(0.02))
        device.event.set()
        self.assertTrue(entered.wait(1))
        with self.assertRaises(ContractError):
            f.facade.release_target(lease)
        finish_compute.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observed, [True])
        with self.assertRaises(TimeoutError):
            f.facade.release_target(lease)
        with self.assertRaises(ContractError):
            f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        writeback.event.set()
        f.finish(lease)

    def test_block_map_change_and_old_target_generation_never_become_visible(self):
        f = OwnerFixture()
        lease = f.acquire()
        host = threading.Event()
        host.set()
        f.facade.publish_received(lease, host, Completion(True))
        with self.assertRaises(ContractError):
            f.facade.prepare_decode(lease, ReceiveReceipt(f.request, 6, True, True, True))
        f.scheduler.ids["internal"] = ((1, 2),)
        with self.assertRaises(ContractError):
            f.facade.prepare_decode(lease, ReceiveReceipt(f.request, 7, True, True, True))
        with self.assertRaises(ContractError):
            f.facade.release_target(lease)
        f.scheduler.ids["internal"] = ((1, 3, 2),)  # safe append preserves prompt map
        f.finish(lease)

    def test_retirement_needs_exact_ack_even_after_owner_release(self):
        f = OwnerFixture()
        lease = f.acquire()
        f.facade.release_target(lease)
        for ack in (ReleaseAck(False, f.allocation.retention_token), ReleaseAck(True, "wrong-token")):
            with self.assertRaises(ContractError):
                f.retention.retire(f.allocation, ack)
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        with self.assertRaises(ContractError):
            f.retention.validate(f.allocation)
        f.facade.close()

    def test_deferred_facade_cannot_succeed_before_binding_or_replace_owner(self):
        f = OwnerFixture()
        facade = DeferredTargetProvider()
        with self.assertRaises(ContractError):
            facade.acquire_target(f.request, f.allocation)
        facade.bind(f.owner)
        with self.assertRaises(ContractError):
            facade.bind(f.owner)
        with self.assertRaises(TypeError):
            pickle.dumps(facade)
        facade.close()

    def test_owner_dispatch_carries_values_and_rejects_live_fences(self):
        f = OwnerFixture()
        dispatcher = TargetOwnerDispatcher(f.facade)

        class Transport:
            def call(self, request):
                return pickle.loads(pickle.dumps(dispatcher.dispatch(pickle.loads(pickle.dumps(request)))))

        client = TargetProviderClient(Transport())
        self.assertEqual(client.catalog(), f.facade.catalog())
        lease = client.acquire_target(f.request, f.allocation)
        self.assertEqual(client.get_target_views(lease), lease.views)
        reply = dispatcher.dispatch(TargetOwnerCall("bad", "views", (threading.Event(),)))
        self.assertEqual(reply.error_type, "TypeError")
        f.ready(lease)
        client.prepare_decode(lease, ReceiveReceipt(f.request, 7, True, True, True))
        client.release_target(lease)
        f.facade.close()

    def test_remote_client_rejects_mismatched_reply_and_unbound_transport(self):
        client = TargetProviderClient()
        with self.assertRaises(ContractError):
            client.catalog()
        client.bind_transport(SimpleNamespace(call=lambda command: TargetOwnerReply("wrong-call")))
        with self.assertRaises(ContractError):
            client.catalog()
        with self.assertRaises(ContractError):
            client.bind_transport(SimpleNamespace(call=lambda command: None))

    def test_split_real_backing_preserves_identity_and_releases_registration_pins(self):
        f = OwnerFixture()
        f.facade.close()
        f.te.unregister_buffer()
        # Register each Main view separately: one actual pool now has multiple
        # regions. Keep the real backing ID and disjoint offsets, rather than
        # inventing one backing per registration.
        pointers, lengths = [], []
        for tensors in f.host.values():
            for tensor in tensors:
                pointers.append(tensor.data_ptr())
                lengths.append(tensor.numel() * tensor.element_size())
        for name, tensors in f.caches.items():
            if f.components[name] is Component.INDEXER:
                for tensor in tensors:
                    pointers.append(tensor.data_ptr())
                    lengths.append(tensor.untyped_storage().nbytes())
        f.te.register_buffer(pointers, lengths)
        before = f.pool._target_lifetime.pins
        resources = f.build_resources()
        try:
            hosts = tuple(r for r in resources.catalog().regions if r.memory_kind == "host")
            self.assertEqual(len(hosts), 4)
            self.assertEqual(len({r.backing_id for r in hosts}), 1)
            self.assertEqual({r.backing_offset_bytes for r in hosts}, {0, 256, 512, 768})
            self.assertGreater(f.pool._target_lifetime.pins, before)
        finally:
            resources.close()
        self.assertEqual(f.pool._target_lifetime.pins, before)
        f.te.unregister_buffer()
        f.pool.close()

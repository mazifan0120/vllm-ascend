# SPDX-License-Identifier: Apache-2.0
"""Stream ordering and failure recovery, without pretending CPU events are DMA."""

import pickle
import threading
import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture, ThreadedStreams
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_completion import CompletionFrontier, StreamCompletion
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_runtime import NpuTargetStreams


class FakeNpu:
    def __init__(self):
        self.current = None
        self.fail_stream = None
        outer = self

        class Event:
            complete = False

            def record(self):
                self.stream = outer.current
                if self.stream is outer.fail_stream:
                    outer.fail_stream = None
                    raise RuntimeError("record failed")

            def query(self):
                return self.complete

            def synchronize(self):
                assert self.complete

        self.Event = Event

    @contextmanager
    def device(self, ordinal):
        yield

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class TestCompletionFrontier(unittest.TestCase):
    def test_superseded_native_event_destructor_runs_outside_runtime_state_lock(self):
        fixture = OwnerFixture()
        lock = fixture.runtime._lock
        acquired = []

        class Event:
            def query(self):
                return True

            def synchronize(self):
                pass

            def __del__(self):
                def inspect_lock():
                    available = lock.acquire(timeout=0.1)
                    acquired.append(available)
                    if available:
                        lock.release()

                thread = threading.Thread(target=inspect_lock)
                thread.start()
                thread.join(0.2)

        class Streams(ThreadedStreams):
            def __init__(self):
                super().__init__()
                self.owner = object()
                self.stream = SimpleNamespace(npu_stream=1)
                self.sequence = 0

            def record_use(self):
                self.sequence += 1
                return (StreamCompletion(self.owner, self.stream, self.sequence, Event()),)

            def wait_quiescent(self, events, timeout):
                for event in events:
                    event.synchronize()

        fixture.streams = fixture.runtime._streams = Streams()
        lease = fixture.acquire()
        fixture.ready(lease)
        binding = fixture.runtime.bind_use(lease)
        for _ in range(2):
            with fixture.runtime.use_batch(((binding, fixture.allocation),)):
                pass
        self.assertEqual(acquired, [True])
        fixture.finish(lease)

    def test_out_of_order_context_returns_cannot_replace_newer_stream_tail(self):
        owner, stream = object(), SimpleNamespace(npu_stream=7)
        older = StreamCompletion(owner, stream, 1, object())
        newer = StreamCompletion(owner, stream, 2, object())
        frontier = CompletionFrontier()
        frontier.extend((newer,))
        frontier.extend((older,))
        self.assertEqual(tuple(frontier), (newer,))
        with self.assertRaises(TypeError):
            pickle.dumps(newer)

    def test_distinct_owners_streams_and_opaque_dependencies_are_not_dropped(self):
        owner = object()
        stream = SimpleNamespace(npu_stream=7)
        one = StreamCompletion(owner, stream, 1, object())
        peer = StreamCompletion(object(), stream, 100, object())
        dynamic = StreamCompletion(owner, SimpleNamespace(npu_stream=8), 101, object())
        opaque = [object(), object()]
        frontier = CompletionFrontier()
        frontier.extend((one, peer, dynamic, *opaque))
        self.assertEqual(len(frontier), 5)
        self.assertEqual(set(frontier), {one, peer, dynamic, *opaque})

    def test_actual_runtime_retains_latest_tail_after_thousand_uses_and_waits_on_release(self):
        progress = SimpleNamespace(value=0)
        release_recorded = threading.Event()

        class OrderedStreams(ThreadedStreams):
            def __init__(self):
                super().__init__()
                self.owner = object()
                self.stream = SimpleNamespace(npu_stream=7)
                self.sequence = 0

            def record_use(self):
                self.sequence += 1
                sequence = self.sequence
                event = SimpleNamespace(query=lambda: progress.value >= sequence, synchronize=lambda: None)
                if sequence > 1000:
                    release_recorded.set()
                return (StreamCompletion(self.owner, self.stream, sequence, event),)

            def wait_quiescent(self, events, timeout):
                deadline = time.monotonic() + timeout
                for event in events:
                    while not event.query():
                        if time.monotonic() > deadline:
                            raise TimeoutError("Simulated stream is still pending")
                        time.sleep(0.001)
                    event.synchronize()

        fixture = OwnerFixture()
        fixture.streams = fixture.runtime._streams = OrderedStreams()
        fixture.runtime._timeout = 1
        lease = fixture.acquire()
        fixture.ready(lease)
        binding = fixture.runtime.bind_use(lease)
        for _ in range(1000):
            with fixture.runtime.use_batch(((binding, fixture.allocation),)):
                pass
        self.assertEqual(len(binding.state.events), 1)
        errors = []

        def release():
            try:
                fixture.facade.release_target(lease)
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=release)
        thread.start()
        self.assertTrue(release_recorded.wait(1))
        self.assertTrue(thread.is_alive())
        self.assertIn(fixture.request, fixture.runtime._states)
        progress.value = fixture.streams.sequence
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        fixture.finish(lease)


class TestNativeRecorderRecovery(unittest.TestCase):
    def fixture(self):
        npu = FakeNpu()
        consume, update = [SimpleNamespace(device=SimpleNamespace(index=0), npu_stream=i) for i in range(2)]
        manager = SimpleNamespace(streams=(update,))
        manager.target_runtime_streams = lambda: manager.streams
        with patch.object(torch, "npu", npu):
            recorder = NpuTargetStreams(manager, 0, consume)
        return npu, manager, recorder, consume, update

    def test_partial_failure_fences_removed_stream_before_forgetting_it(self):
        npu, manager, recorder, consume, update = self.fixture()
        npu.fail_stream = update
        with self.assertRaisesRegex(RuntimeError, "record failed"):
            recorder.record_use()
        manager.streams = ()  # Disappeared after the failed call, still owns pending work.
        recovered = recorder.record_use()
        self.assertEqual({f.stream.npu_stream for f in recovered}, {consume.npu_stream, update.npu_stream})
        for fence in recovered:
            fence.event.complete = True
        final = recorder.record_use()
        self.assertEqual([f.stream.npu_stream for f in final], [consume.npu_stream])

    def test_foreign_device_or_unenumerable_streams_quarantine_the_owner(self):
        _, manager, recorder, _, _ = self.fixture()
        manager.streams = (SimpleNamespace(device=SimpleNamespace(index=1), npu_stream=5),)
        with self.assertRaisesRegex(RuntimeError, "another NPU"):
            recorder.record_use()
        manager.streams = ()
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            recorder.record_use()
        _, manager, recorder, _, _ = self.fixture()

        def unavailable():
            raise RuntimeError("stream set unavailable")

        manager.target_runtime_streams = unavailable
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            recorder.record_use()
        manager.target_runtime_streams = lambda: ()
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            recorder.record_use()

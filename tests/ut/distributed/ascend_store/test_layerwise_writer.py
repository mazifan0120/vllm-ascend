# SPDX-License-Identifier: Apache-2.0
"""Actual CPU copies through production SourceReady planning/queue machinery."""

import ctypes
import queue
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_layerwise_plan import MemoryFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    Status,
    TargetClosedError,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_source_consumer import LayerwiseSourceConsumer
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_writer import ManifestSourceWriter


class CpuEngine:
    def __init__(self, result=0):
        self.calls = []
        self.result = result

    def batch_transfer_sync_write(self, endpoint, sources, destinations, lengths):
        self.calls.append((endpoint, tuple(sources), tuple(destinations), tuple(lengths)))
        if isinstance(self.result, Exception):
            raise self.result
        if self.result == 0:
            for src, dst, length in zip(sources, destinations, lengths):
                ctypes.memmove(dst, src, length)
        return self.result


def metadata(fixture):
    return SimpleNamespace(
        remote_target_manifest=fixture.manifest,
        remote_host="cpu-descriptor",
        remote_port=1,
        request_instance=fixture.manifest.lease.request.request_instance,
    )


class TestSourceWriter(unittest.TestCase):
    def test_closed_remote_target_before_address_lookup_cancels_without_native_reads(self):
        def resolve(*args):
            raise TargetClosedError("D cancelled before exposing addresses")

        fixture, writer, work, proofs = self.make(resolve=resolve)
        source, request = fixture.source(0)
        done = []
        engine = CpuEngine()
        try:
            job = writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
            job.publish(lambda *args: done.append(args))
            work.get_nowait().execute(engine)
            writer._proof_queue.join()
            self.assertEqual(engine.calls, [])
            self.assertEqual(done, [(None, True)])
            self.assertEqual(proofs[-1][2], (Status.CANCELLED, True))
        finally:
            self.stop(writer)

    def test_failed_proof_delivery_does_not_repeat_timeout_for_every_layer_or_hold_source(self):
        calls = []

        def fail(*args):
            calls.append(args)
            raise TimeoutError("injected exhausted peer ACK retries")

        fixture, writer, work, _ = self.make(proofs=fail)
        source, request = fixture.source(0)
        done = []
        try:
            for start, end in ((0, 11), (11, 23)):
                job = writer.prepare_job(
                    source,
                    replace(request, start_token=start, end_token=end),
                    SimpleNamespace(synchronize=lambda: None),
                )
                job.publish(lambda *args: done.append(args))
                work.get_nowait().execute(CpuEngine())
            writer._proof_queue.join()
            self.assertEqual(len(calls), 1)
            self.assertEqual(done, [(None, True), (None, True)])
            self.assertIsInstance(writer._requests[request.request_id].delivery_error, TimeoutError)
        finally:
            self.stop(writer)

    def make(self, rank=0, *, resolve=None, proofs=None):
        fixture = MemoryFixture(layers=1)
        native_queue = queue.Queue()
        received = []
        writer = ManifestSourceWriter(
            rank,
            native_queue,
            resolve or (lambda *_: tuple(fixture.handles)),
            proofs or (lambda *args: received.append(args)),
        )
        writer.register_request("internal-source", metadata(fixture))
        return fixture, writer, native_queue, received

    def stop(self, writer):
        try:
            writer.close()
        except ContractError:
            # These tests inject unknown native outcomes on a CPU engine.
            # Stop only its daemon proof transport, without certifying release
            # or clearing its retained unknown-reader records.
            writer._proof_queue.put(None)
            writer._proof_thread.join(1)

    def test_all_rank_cpu_bytes_match_and_main_is_written_only_once_in_separate_batches(self):
        fixture, writer, work, received = self.make()
        peer = ManifestSourceWriter(1, work, lambda *_: tuple(fixture.handles), lambda *args: received.append(args))
        peer.register_request("internal-source", metadata(fixture))
        engine = CpuEngine()
        completed = []
        try:
            for rank, producer in enumerate((writer, peer)):
                source, request = fixture.source(rank, padding=16)
                job = producer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
                job.publish(lambda error, q: completed.append((error, q)))
                work.get_nowait().execute(engine)
            writer._proof_queue.join()
            peer._proof_queue.join()
            self.assertEqual(completed, [(None, True), (None, True)])
            self.assertEqual(len(engine.calls), 3)  # Main host, P0 Indexer, P1 Indexer
            self.assertEqual(sorted(len(item[1]) for item in received), [2, 4])
            self.assertTrue(all(item[2] == (Status.SUCCESS, True) for item in received))
            fixture.verify(0, 23)
        finally:
            self.stop(writer)
            self.stop(peer)

    def test_native_completion_releases_exact_epoch_while_peer_ack_is_still_blocked(self):
        entered = threading.Event()
        release = threading.Event()

        def proof(*args):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("CPU test peer ACK was not released")

        fixture, writer, work, _ = self.make(proofs=proof)
        source, request = fixture.source(0)
        worker = SimpleNamespace(
            prepare_source_batch=lambda *args: ((writer.prepare_job(*args[:1], request, args[2]),), False)
        )
        consumer = LayerwiseSourceConsumer(worker, timeout=0.1)
        try:
            consumer.open_source(source)
            consumer.source_ready(source, (request,), SimpleNamespace(synchronize=lambda: None))
            consumer.seal_source(source)
            work.get_nowait().execute(CpuEngine())
            self.assertTrue(entered.wait(1))
            consumer.wait_source(source)
            self.assertTrue(writer._proof_thread.is_alive())
            self.assertFalse(release.is_set())
        finally:
            release.set()
            self.stop(writer)

    def test_ready_fence_precedes_any_native_source_read(self):
        fixture, writer, work, _ = self.make()
        source, request = fixture.source(0)
        entered, release = threading.Event(), threading.Event()
        engine = CpuEngine()
        done = []

        def fence():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("CPU source fence")

        job = writer.prepare_job(source, request, SimpleNamespace(synchronize=fence))
        job.publish(lambda *args: done.append(args))
        thread = threading.Thread(target=lambda: work.get_nowait().execute(engine))
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(engine.calls, [])
            self.assertEqual(done, [])
            release.set()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(done, [(None, True)])
        finally:
            release.set()
            thread.join(2)
            self.stop(writer)

    def test_native_failure_or_exception_has_unknown_quiescence_and_no_successful_terminal(self):
        for result in (-1, 1, RuntimeError("unknown native failure"), TargetClosedError("error during native call")):
            fixture, writer, work, proofs = self.make()
            source, request = fixture.source(0)
            done = []
            try:
                job = writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
                job.publish(lambda *args, done=done: done.append(args))
                work.get_nowait().execute(CpuEngine(result))
                writer._proof_queue.join()
                self.assertIsNotNone(done[0][0])
                self.assertFalse(done[0][1])
                self.assertEqual(proofs[0][2], (Status.FAILED, False))
                with self.assertRaises(ContractError):
                    writer.close()
                with self.assertRaises(ContractError):
                    writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
            finally:
                self.stop(writer)

    def test_lookup_failure_submits_no_native_job_and_can_prove_no_source_reader(self):
        def resolve(*args):
            raise ContractError("target address generation changed")

        fixture, writer, work, proofs = self.make(resolve=resolve)
        source, request = fixture.source(0)
        done = []
        engine = CpuEngine()
        try:
            job = writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
            job.publish(lambda *args: done.append(args))
            work.get_nowait().execute(engine)
            writer._proof_queue.join()
            self.assertEqual(engine.calls, [])
            self.assertIsNotNone(done[0][0])
            self.assertTrue(done[0][1])
            self.assertEqual(proofs[0][2], (Status.FAILED, True))
        finally:
            self.stop(writer)

    def test_terminal_requires_full_entry_coverage_and_all_prepared_jobs_to_end(self):
        fixture, writer, work, proofs = self.make()
        source, request = fixture.source(0)
        done = []
        try:
            first = writer.prepare_job(
                source, replace(request, end_token=11), SimpleNamespace(synchronize=lambda: None)
            )
            last = writer.prepare_job(
                source, replace(request, start_token=11), SimpleNamespace(synchronize=lambda: None)
            )
            for job in (first, last):
                job.publish(lambda *args: done.append(args))
            work.get_nowait().execute(CpuEngine())
            writer._proof_queue.join()
            self.assertIsNone(proofs[-1][2])
            work.get_nowait().execute(CpuEngine())
            writer._proof_queue.join()
            self.assertEqual(proofs[-1][2], (Status.SUCCESS, True))
            self.assertEqual(len(done), 2)
            with self.assertRaises(ContractError):
                last.publish(lambda *args: None)
            with self.assertRaises(ContractError):
                last.execute(CpuEngine())
        finally:
            self.stop(writer)

    def test_finished_request_drains_prepared_native_readers_then_drops_heavy_plan_state(self):
        fixture, writer, work, proofs = self.make()
        source, request = fixture.source(0)
        done = []
        try:
            job = writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
            job.publish(lambda *args: done.append(args))
            writer.finish_requests((request.request_id,))
            self.assertIn(request.request_id, writer._requests)
            with self.assertRaises(ContractError):
                writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
            work.get_nowait().execute(CpuEngine())
            writer._proof_queue.join()
            self.assertEqual(proofs[-1][2], (Status.SUCCESS, True))
            self.assertNotIn(request.request_id, writer._requests)
            # A transport retry can get the same or another vLLM request ID.
            # Neither may reopen native writes after the quiescent terminal.
            for request_id in (request.request_id, "replayed-internal-id"):
                with self.assertRaises(ContractError):
                    writer.register_request(request_id, metadata(fixture))
            self.assertEqual(writer._target_owners, {fixture.manifest.lease.request: request.request_id})
        finally:
            self.stop(writer)

    def test_cancelled_partial_plan_never_executes_queued_native_reads_or_reports_success(self):
        fixture, writer, work, proofs = self.make()
        source, request = fixture.source(0, end=11)
        done = []
        engine = CpuEngine()
        try:
            job = writer.prepare_job(source, request, SimpleNamespace(synchronize=lambda: None))
            job.publish(lambda *args: done.append(args))
            writer.finish_requests((request.request_id,))
            work.get_nowait().execute(engine)
            writer._proof_queue.join()
            self.assertEqual(engine.calls, [])
            self.assertEqual(proofs[-1][2], (Status.CANCELLED, True))
            self.assertNotIn(request.request_id, writer._requests)
            self.assertEqual(done, [(None, True)])
        finally:
            self.stop(writer)

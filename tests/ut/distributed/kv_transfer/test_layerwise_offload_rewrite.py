# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import unittest
from dataclasses import replace
from threading import Event, Thread
from time import monotonic
from types import SimpleNamespace

import msgspec
import zmq

from vllm_ascend.core.layerwise_offload_scheduler import DecodeAdmission
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_control import (
    DecodeControl,
    PrefillControl,
    SideChannelClient,
    WriterControl,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_plan import (
    CatalogRef,
    EntryId,
    RegisteredLayout,
    RegisteredRegion,
    Span,
    TargetAllocation,
    TargetId,
    WriteResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_sender import (
    LayerwiseSender,
    NativeResult,
    PrefillChunk,
    mooncake_sync_write,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.layerwise_offload_wire import (
    decode_addresses,
    decode_allocation,
    decode_catalog,
    decode_worker_status,
    encode_allocation,
    encode_catalog,
    encode_catalog_query,
    encode_results,
    encode_terminal,
    encode_worker_query,
    encode_writer_query,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import (
    DsaCacheLayout,
    DsaRegisterAtom,
    prefill_source_atoms,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.prefill_reuse import (
    ReadOutcome,
    SlotBarrier,
    restore_into_slot,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.layerwise_receive import (
    DecodeTargets,
    register_model_targets,
    register_rank_catalog,
)


def _roundtrip(message):
    return json.loads(json.dumps(message))


def _catalog(rank):
    regions = (
        (RegisteredRegion("host", "shared-main", 1, 0, 64, "main"), 1000),
        (RegisteredRegion("device", f"indexer-{rank}", 1, 0, 64, "indexer"), 2000 + 100 * rank),
    )
    views = {
        EntryId(layer, region.kind, 0): (group, DsaCacheLayout(str(layer), 0, base + layer * 32, 4, 4, 2, 4, "bf16", 4))
        for layer in range(2)
        for group, (region, base) in enumerate(regions)
    }

    def register(bases, lengths):
        if bases != [1000, 2000 + 100 * rank] or lengths != [64, 64]:
            raise ValueError("Wrong registration passed to the transfer engine")

    return register_rank_catalog(CatalogRef(f"engine-{rank}", 1, rank), f"test-{rank}", views, regions, register)


def _serve_rank(worker, endpoint, stop):
    with zmq.Context() as context, context.socket(zmq.ROUTER) as sock:
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        endpoint.send(f"tcp://127.0.0.1:{port}")
        endpoint.close()
        while not stop.is_set():
            if sock.poll(20):
                identity, delimiter, payload = sock.recv_multipart()
                frame = msgspec.msgpack.decode(payload, type=tuple)
                response = msgspec.msgpack.encode(worker.handle_side_channel(frame))
                sock.send_multipart((identity, delimiter, response))


class _Fixture:
    def __init__(self, *, multi_writer=True, producer_tp=4, decode_tp=2):
        catalogs = tuple(decode_catalog(_roundtrip(encode_catalog(_catalog(rank)))) for rank in range(decode_tp))
        self.layout = RegisteredLayout(catalogs, producer_tp, parallel_main=multi_writer)
        self.allocation = TargetAllocation(TargetId("request", 1, 7), 7, ((1, 0), (2, 1)))
        self.target = self.allocation.target
        self.bytes = {}
        self.visibility = []
        self.d = tuple(
            DecodeTargets(
                self.layout,
                rank,
                {"host": 1000, "device": 2000 + 100 * rank},
                lambda: None,
                lambda allocation, rank=rank: self.visibility.append((rank, allocation.target)),
            )
            for rank in range(decode_tp)
        )
        self.retained = []
        self.released = []
        self.block_owners = {}
        self.scheduler = DecodeAdmission(self.layout, self.hold_blocks, self.release_blocks)
        self.barriers = tuple(SlotBarrier(((0, 1),)) for _ in range(producer_tp))
        self.p = tuple(
            LayerwiseSender(rank, self.layout, self.barriers[rank], self.addresses, self.write, self.report)
            for rank in range(decode_tp)
        )

    def hold_blocks(self, allocation):
        keys = [(group, block) for group, blocks in enumerate(allocation.blocks) for block in blocks]
        if any(key in self.block_owners for key in keys):
            raise RuntimeError("Request tries to allocate a retained block")
        self.block_owners.update(dict.fromkeys(keys, allocation.target))
        self.retained.append(allocation)

    def release_blocks(self, allocation):
        for group, blocks in enumerate(allocation.blocks):
            for block in blocks:
                assert self.block_owners.pop((group, block)) == allocation.target
        self.released.append(allocation)

    def publish(self):
        refs = self.scheduler.retain(self.allocation)
        command = _roundtrip(encode_allocation(self.allocation, refs))
        for rank, worker in enumerate(self.d):
            assert self.scheduler.publish(self.target) is None
            status = _roundtrip(worker.handle_scheduler(command))
            assert decode_worker_status(status, self.target, refs[rank]) == ("prepared", ())
            self.scheduler.prepared(self.target, refs[rank])
        allocation, refs = self.scheduler.publish(self.target)
        return decode_allocation(_roundtrip(encode_allocation(allocation, refs)))

    def addresses(self, rank, target, writer, ref):
        query = _roundtrip(encode_writer_query("get-addresses", target, writer, ref))
        return decode_addresses(_roundtrip(self.d[rank].handle_peer(query)), target)

    def write(self, write):
        return mooncake_sync_write(self, write)

    def batch_transfer_sync_write(self, endpoint, sources, targets, lengths):
        assert endpoint in {catalog.endpoint for catalog in self.layout.catalogs}
        for src, dst, count in zip(sources, targets, lengths, strict=True):
            for offset in range(count):
                self.bytes[dst + offset] = self.bytes[src + offset]
        return 0

    def report(self, rank, target, results):
        assert self.d[rank].handle_peer(_roundtrip(encode_results(target, results))) == ("ack",)

    def worker_status(self, rank, op="poll-target"):
        ref = self.layout.refs[rank]
        request = _roundtrip(encode_worker_query(op, self.target, ref))
        return decode_worker_status(_roundtrip(self.d[rank].handle_scheduler(request)), self.target, ref)

    def sources(self, rank, layer):
        result = {}
        for component, base in (("main", 4000 + rank * 100), ("indexer", 5000 + rank * 100)):
            layout = DsaCacheLayout(str(layer), 0, base, 4, 4, 2, 4, "bf16", 4)
            for offset in range(16):
                self.bytes[base + offset] = 10 * (layer + 1) + offset + (rank if component == "indexer" else 0)
            result[EntryId(layer, component, 0)] = (layout, (0, 1))
        return result

    def collect_terminal(self, rank, outcome):
        frame = _roundtrip(encode_terminal(self.target, rank, outcome.error, outcome.quiesced))
        for worker in self.d:
            assert worker.handle_peer(frame) == ("ack",)


class TestLayerwiseRewrite(unittest.TestCase):
    def test_unequal_tp_requires_replicated_model_and_divisible_ratio(self):
        RegisteredLayout.validate_tp(8, 8, replicated=False)
        for producer_tp, decode_tp in ((8, 0), (2, 8), (8, 3)):
            with (
                self.subTest(producer_tp=producer_tp, decode_tp=decode_tp),
                self.assertRaisesRegex(ValueError, "divisible"),
            ):
                RegisteredLayout.validate_tp(producer_tp, decode_tp, replicated=True)
        with self.assertRaisesRegex(ValueError, "replicated GLM DSA"):
            RegisteredLayout.validate_tp(8, 2, replicated=False)

    def test_only_exact_main_aliases_are_registered_once(self):
        source = DsaCacheLayout("layer.0", 0, 100, 8, 8, 1, 4, "int8", 2)
        atom = DsaRegisterAtom(100, 116, "*", 92)
        indexer = DsaRegisterAtom(200, 216, "*", 192)
        main = {0: [(source, atom)], 1: [(replace(source, layer_name="layer.1"), atom)]}
        self.assertEqual(prefill_source_atoms(main, [indexer], ((0, 1),)), [atom, indexer])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prefill_source_atoms(main, [indexer, indexer], ((0, 1),))
        with self.assertRaisesRegex(ValueError, "does not alias"):
            prefill_source_atoms(
                {0: main[0], 1: [(replace(source, stride=16), atom)]}, [indexer], ((0, 1),)
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prefill_source_atoms(main, [indexer], ((0,), (1,)))

    def test_native_catalog_binds_manager_host_views_across_distinct_virtual_addresses(self):
        main, indexer = "model.layers.0.attn", "model.layers.0.attn.indexer.k_cache"
        catalogs = []

        def tensor(base, rank):
            storage = SimpleNamespace(data_ptr=lambda: base, nbytes=lambda: 64)
            return SimpleNamespace(
                shape=(4, 4, 2),
                stride=lambda: (8, 2, 1),
                dtype="torch.bfloat16",
                data_ptr=lambda: base,
                element_size=lambda: 2,
                untyped_storage=lambda: storage,
                device=SimpleNamespace(type="npu", index=rank),
            )

        for rank in range(2):
            host, device = 4096 + 65536 * rank, 20480 + 65536 * rank
            pool = SimpleNamespace(
                data_ptr=host,
                nbytes=4096,
                region=SimpleNamespace(segment_offset=32),
                topology=SimpleNamespace(device_id=rank),
            )
            manager = SimpleNamespace(
                offload_layer_names=(main,),
                get_mooncake_host_pool=lambda pool=pool: pool,
                get_local_host_kv_views=lambda name, host=host, rank=rank: (
                    tensor(host + 64, rank),
                    tensor(host + 128, rank),
                ),
            )
            registered = []
            inputs = {main: object(), indexer: tensor(device, rank)}  # Selection buffers must never be registered.
            layers = {main: (0, 0, 4), indexer: (0, 1, 4)}
            catalog, bases = register_model_targets(
                CatalogRef(f"engine-{rank}", 0, rank),
                "transport",
                "shared-host",
                manager,
                inputs,
                layers,
                4,
                lambda ptrs, lengths, locations, registered=registered: registered.extend(
                    zip(ptrs, lengths, locations, strict=True)
                ),
            )
            self.assertEqual(registered, [(host + 64, 128, f"npu:{rank}"), (device, 64, "*")])
            self.assertEqual(list(bases.values()), [host + 64, device])
            self.assertEqual(catalog.regions[0].backing_offset, 96)
            self.assertNotIn(str(device), catalog.regions[1].backing)
            catalogs.append(decode_catalog(encode_catalog(catalog)))
        layout = RegisteredLayout(tuple(catalogs), 2, parallel_main=True)
        self.assertEqual(layout.refs[1].rank, 1)

    def test_ready_target_installs_visibility_on_the_actual_consumer(self):
        case = _Fixture()
        calls = []
        worker = DecodeTargets(
            case.layout,
            0,
            {"host": 1000, "device": 2000},
            lambda: None,
            lambda allocation: lambda: calls.append(allocation.target),
        )
        worker.prepare(case.allocation, case.layout.refs)
        for (writer, entry), spans in case.layout.receive_plan(0, case.allocation.prompt_tokens).items():
            worker.apply(case.target, tuple(WriteResult(writer, entry, span) for span in spans))
        for writer in case.layout.writers:
            worker.terminal(case.target, writer, None, True)
        self.assertTrue(worker.try_ready(case.target))
        self.assertEqual(calls, [])
        use = worker.acquire_decode(case.allocation)
        self.assertEqual(calls, [case.target])
        worker.finish_decode(case.target, use)

    def test_close_waits_for_compute_and_host_writeback_completion(self):
        case = _Fixture()
        case.publish()
        worker = case.d[0]
        for (writer, entry), spans in case.layout.receive_plan(0, case.allocation.prompt_tokens).items():
            worker.apply(case.target, tuple(WriteResult(writer, entry, span) for span in spans))
        for writer in case.layout.writers:
            worker.terminal(case.target, writer, None, True)
        self.assertTrue(worker.try_ready(case.target))
        use = worker.acquire_decode(case.target)
        compute, writeback = Event(), Event()
        worker.defer_decode_completion(((case.target, use),), lambda: compute.is_set() and writeback.is_set())
        worker.close(case.target)
        self.assertFalse(worker.try_closed(case.target))
        compute.set()
        self.assertFalse(worker.try_closed(case.target))
        writeback.set()
        self.assertTrue(worker.try_closed(case.target))

    def _side_channels(self, case):
        context = multiprocessing.get_context("fork")
        stop = context.Event()
        workers = []

        def shutdown():
            stop.set()
            for worker in workers:
                worker.join(5)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(5)
                self.assertEqual(worker.exitcode, 0)

        self.addCleanup(shutdown)
        endpoints = []
        for target in case.d:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_serve_rank, args=(target, child, stop))
            process.start()
            child.close()
            workers.append(process)
            with parent:
                self.assertTrue(parent.poll(5))
                endpoints.append(parent.recv())
        return tuple(endpoints)

    def test_side_channel_close_keeps_uncertain_native_writes_pinned(self):
        case = _Fixture(multi_writer=False)
        endpoints = self._side_channels(case)
        events = []
        d_client = SideChannelClient(endpoints, 1000)
        p_client = SideChannelClient(endpoints, 1000)
        self.addCleanup(p_client.close)
        self.addCleanup(d_client.close)
        control = DecodeControl(d_client, events.append, timeout_s=10)
        writer = WriterControl(p_client, case.layout.refs)
        refs = case.scheduler.retain(case.allocation)
        control.prepare(case.allocation, refs)
        control.step()
        self.assertEqual([event.kind for event in events], ["prepared", "prepared"])
        for event in events:
            case.scheduler.prepared(event.target, event.worker)
        events.clear()
        barrier = case.barriers[0]
        sender = LayerwiseSender(
            0,
            case.layout,
            barrier,
            writer.read_addresses,
            lambda write: NativeResult("native status unknown", False),
            writer.report,
        )
        sender.attach(*case.scheduler.publish(case.target))
        epoch = restore_into_slot(barrier, 0, None)
        job = sender.reserve_layer(case.target, epoch, Span(0, 4), case.sources(0, 0), lambda: None)
        barrier.seal(epoch)
        sender.execute(job)
        control.close_target(case.target, case.scheduler.begin_close(case.target))
        control.step()
        self.assertEqual([(event.kind, event.worker.rank) for event in events], [("closed", 1)])
        self.assertFalse(case.scheduler.closed(events[0].target, events[0].worker))
        events.clear()
        self.assertFalse(writer.progress(sender, case.target))
        control.step()
        self.assertEqual(events, [])
        self.assertEqual(case.released, [])
        sender.confirm_native_quiescence(case.target)
        self.assertTrue(writer.progress(sender, case.target))
        control.step()
        self.assertEqual([(event.kind, event.worker.rank) for event in events], [("closed", 0)])
        self.assertTrue(case.scheduler.closed(events[0].target, events[0].worker))
        self.assertEqual(case.released, [case.allocation])
        sender.forget(case.target)

    def test_side_channel_delivers_ready_only_after_peer_proofs(self):
        case = _Fixture()
        endpoints = self._side_channels(case)
        client = SideChannelClient(endpoints, 1000)
        self.addCleanup(client.close)
        events = []
        control = DecodeControl(client, events.append, timeout_s=10)
        refs = case.scheduler.retain(case.allocation)
        control.prepare(case.allocation, refs)
        control.step()
        for event in events:
            self.assertEqual(event.kind, "prepared")
            case.scheduler.prepared(event.target, event.worker)
        published = case.scheduler.publish(case.target)
        events.clear()
        control.step()
        self.assertEqual(events, [])
        writer = WriterControl(client, refs)
        for catalog in case.layout.catalogs:
            self.assertEqual(
                decode_catalog(client.request(catalog.ref.rank, encode_catalog_query(catalog.ref))), catalog
            )
        for rank in case.layout.writers:
            barrier = case.barriers[rank]
            sender = LayerwiseSender(rank, case.layout, barrier, writer.read_addresses, case.write, writer.report)
            sender.attach(*published)
            for layer in range(2):
                epoch = restore_into_slot(barrier, layer, None)
                job = sender.reserve_layer(case.target, epoch, Span(0, 7), case.sources(rank, layer), lambda: None)
                barrier.seal(epoch)
                sender.execute(job)
            self.assertTrue(writer.progress(sender, case.target, finished=True))
            sender.forget(case.target)
            control.step()
            if rank == 0:
                self.assertEqual(events, [])
        self.assertEqual([(event.kind, event.worker.rank) for event in events], [("ready", 0), ("ready", 1)])
        for event in events:
            case.scheduler.ready(event.target, event.worker)
        self.assertTrue(case.scheduler.may_decode(case.target))

    def test_prefill_control_reuses_catalog_and_completes_multiple_chunks(self):
        for decode_tp, multi_writer in ((2, False), (4, True)):
            with self.subTest(decode_tp=decode_tp, multi_writer=multi_writer):
                self._check_prefill_control(decode_tp, multi_writer)

    def _check_prefill_control(self, decode_tp, multi_writer):
        producer_tp = 8
        case = _Fixture(producer_tp=producer_tp, decode_tp=decode_tp, multi_writer=multi_writer)
        endpoints = self._side_channels(case)
        client = SideChannelClient(endpoints, 1000)
        self.addCleanup(client.close)
        events = []
        control = DecodeControl(client, events.append, timeout_s=10)
        refs = case.scheduler.retain(case.allocation)
        control.prepare(case.allocation, refs)
        control.step()
        for event in events:
            self.assertEqual(event.kind, "prepared")
            case.scheduler.prepared(event.target, event.worker)
        case.scheduler.publish(case.target)
        events.clear()

        p_controls = tuple(
            PrefillControl(rank, producer_tp, barrier, case.write) for rank, barrier in enumerate(case.barriers)
        )
        stop = Event()
        threads = tuple(Thread(target=p.run, args=(stop,)) for p in p_controls)
        for thread in threads:
            thread.start()
        senders = {}
        try:
            for span in (Span(0, 5), Span(5, 7)):
                chunk = PrefillChunk(case.allocation, refs, endpoints, multi_writer, ((0, 1), (0, 1)), span)
                for rank, p in enumerate(p_controls):
                    assignments = p.prepare((chunk,))
                    if rank >= decode_tp:
                        self.assertEqual(assignments, ())
                        self.assertEqual(p._sessions, {})
                        continue
                    ((sender, attached),) = assignments
                    self.assertIs(attached, chunk)
                    self.assertIs(senders.setdefault(rank, sender), sender)
                    barrier = case.barriers[rank]
                    for layer in range(2):
                        epoch = restore_into_slot(barrier, layer, None)
                        job = sender.reserve_layer(case.target, epoch, span, case.sources(rank, layer), lambda: None)
                        barrier.seal(epoch)
                        sender.execute(job)
                    if chunk.finished:
                        p.finish((case.target,))
                control.step()
                if not chunk.finished:
                    self.assertEqual(events, [])
            deadline = monotonic() + 5
            while len(events) < decode_tp and monotonic() < deadline:
                control.step()
                stop.wait(0.01)
            expected_ranks = set(range(decode_tp))
            self.assertEqual(
                {(event.kind, event.worker.rank) for event in events}, {("ready", rank) for rank in expected_ranks}
            )
            for event in events:
                self.assertFalse(case.scheduler.may_decode(case.target))
                case.scheduler.ready(event.target, event.worker)
            self.assertTrue(case.scheduler.may_decode(case.target))
            self.assertEqual(case.bytes[1000 + 32 + 5], 33)
            for rank in expected_ranks:
                self.assertEqual(case.bytes[2000 + 100 * rank + 32 + 8 + 5], 33 + rank)

            events.clear()
            control.close_target(case.target, case.scheduler.begin_close(case.target))
            control.step()
            self.assertEqual(
                {(event.kind, event.worker.rank) for event in events}, {("closed", rank) for rank in expected_ranks}
            )
            for event in events:
                self.assertEqual(case.released, [])
                case.scheduler.closed(event.target, event.worker)
            self.assertEqual(case.released, [case.allocation])
        finally:
            stop.set()
            for thread in threads:
                thread.join(5)
                self.assertFalse(thread.is_alive())

    def test_d_first_multi_chunk_partial_main_and_indexer(self):
        case = _Fixture()
        published = case.publish()
        for sender in case.p:
            sender.attach(*published)
        for span in (Span(0, 5), Span(5, 7)):
            for layer in range(2):
                for rank, sender in enumerate(case.p):
                    barrier = case.barriers[rank]
                    epoch = restore_into_slot(barrier, layer, None, timeout=0)
                    attn = barrier.register(epoch, "attention", span)
                    store = barrier.register(epoch, "store", span)
                    sources = case.sources(rank, layer)
                    job = sender.reserve_layer(case.target, epoch, span, sources, lambda: None)
                    self.assertIsNone(sender.reserve_layer(case.target, epoch, span, sources, lambda: None))
                    barrier.seal(epoch)
                    if job is not None:
                        sender.execute(job)
                        sender.flush_proofs(case.target)
                    barrier.complete(store, ReadOutcome(True))
                    barrier.complete(attn, ReadOutcome(True))
            self.assertFalse(any(worker.try_ready(case.target) for worker in case.d))
        for rank, sender in enumerate(case.p):
            case.collect_terminal(rank, sender.finish(case.target))
        for rank, worker in enumerate(case.d):
            self.assertTrue(worker.try_ready(case.target))
            case.scheduler.ready(case.target, case.layout.refs[rank])
        self.assertTrue(case.scheduler.may_decode(case.target))
        self.assertEqual(len(case.visibility), 2)
        for layer in range(2):
            for token in range(7):
                block, offset = divmod(token, 4)
                for byte in range(2):
                    value = 10 * (layer + 1) + 2 * token + byte
                    main_ptr = 1000 + 32 * layer + 8 * case.allocation.blocks[0][block] + 2 * offset + byte
                    self.assertEqual(case.bytes[main_ptr], value)
                    for rank in range(2):
                        ptr = 2000 + 100 * rank + 32 * layer + 8 * case.allocation.blocks[1][block] + 2 * offset + byte
                        self.assertEqual(case.bytes[ptr], value + rank)
            self.assertNotIn(1000 + 32 * layer + 7, case.bytes)
        extended = replace(case.allocation, blocks=((1, 0, 3), (2, 1, 0)))
        with self.assertRaisesRegex(RuntimeError, "different allocation"):
            case.d[0].acquire_decode(replace(extended, blocks=((3, 0, 1), (2, 1, 0))))
        use = case.d[0].acquire_decode(extended)
        overlapping = replace(case.allocation, target=TargetId("new-request", 1, 8))
        case.scheduler.begin_close(case.target)
        for worker in case.d:
            self.assertEqual(worker.close(case.target), ())
        self.assertTrue(case.d[1].try_closed(case.target))
        self.assertFalse(case.scheduler.closed(case.target, case.layout.refs[1]))
        self.assertFalse(case.d[0].try_closed(case.target))
        self.assertEqual(case.released, [])
        with self.assertRaisesRegex(RuntimeError, "retained block"):
            case.scheduler.retain(overlapping)
        case.d[0].finish_decode(case.target, use)
        self.assertTrue(case.d[0].try_closed(case.target))
        self.assertTrue(case.scheduler.closed(case.target, case.layout.refs[0]))
        self.assertEqual(case.released, [case.allocation])
        self.assertEqual(case.scheduler.retain(overlapping), case.layout.refs)

    def test_empty_get_cannot_overwrite_until_all_reads_stop(self):
        barrier = SlotBarrier(((0, 1),))
        epoch = barrier.begin_write(0)
        tickets = [barrier.register(epoch, name, 1) for name in ("attention", "store", "transfer")]
        with self.assertRaises(TimeoutError):
            restore_into_slot(barrier, 1, None, timeout=0)
        barrier.seal(epoch)
        for ticket in tickets:
            with self.assertRaises(TimeoutError):
                restore_into_slot(barrier, 1, None, timeout=0)
            barrier.complete(ticket, ReadOutcome(True))
        self.assertEqual(barrier.begin_write(1, timeout=0).generation, 2)
        barrier.complete(tickets[-1], ReadOutcome(True))

    def test_get_error_cannot_admit_new_compute_or_reuse(self):
        barrier = SlotBarrier(((0, 1),))
        with self.assertRaisesRegex(RuntimeError, "Store GET failed"):
            restore_into_slot(barrier, 0, lambda: ReadOutcome(False, "Store GET failed"))
        with self.assertRaisesRegex(RuntimeError, "Store GET failed"):
            restore_into_slot(barrier, 1, None, timeout=0)
        self.assertFalse(barrier.may_dispose())

    def test_source_can_rotate_while_d_proof_is_still_in_flight(self):
        case = _Fixture()
        allocation, refs = case.publish()
        sent, acknowledged = Event(), Event()

        def report(rank, target, results):
            sent.set()
            if not acknowledged.wait(5):
                raise TimeoutError("test proof acknowledgement was not signaled")
            case.report(rank, target, results)

        barrier = case.barriers[0]
        sender = LayerwiseSender(0, case.layout, barrier, case.addresses, case.write, report)
        sender.attach(allocation, refs)
        epoch = restore_into_slot(barrier, 0, None)
        job = sender.reserve_layer(case.target, epoch, Span(0, 4), case.sources(0, 0), lambda: None)
        barrier.seal(epoch)
        sender.execute(job)
        thread = Thread(target=sender.flush_proofs, args=(case.target,))
        thread.start()
        try:
            self.assertTrue(sent.wait(5))
            next_epoch = restore_into_slot(barrier, 1, None, timeout=0)
            self.assertEqual(next_epoch.generation, 2)
            other_job = sender.reserve_layer(case.target, next_epoch, Span(0, 4), case.sources(0, 1), lambda: None)
            barrier.seal(next_epoch)
            sender.execute(other_job)
            self.assertIsNone(sender.finish(case.target))
            self.assertFalse(case.d[0].try_ready(case.target))
        finally:
            acknowledged.set()
            thread.join(5)
        self.assertEqual(sender.finish(case.target), NativeResult(None, True))
        self.assertEqual(sender.finish(case.target, cancel=True), NativeResult(None, True))

    def test_cancel_between_publish_and_native_completion_retains_blocks(self):
        case = _Fixture(multi_writer=False)
        allocation, refs = case.publish()
        entered, complete = Event(), Event()

        def failed_write(write):
            entered.set()
            if not complete.wait(5):
                raise TimeoutError("test release was not signaled")
            return NativeResult("DMA result unknown", False)

        sender = LayerwiseSender(0, case.layout, case.barriers[0], case.addresses, failed_write, case.report)
        sender.attach(allocation, refs)
        epoch = case.barriers[0].begin_write(0)
        job = sender.reserve_layer(case.target, epoch, Span(0, 4), case.sources(0, 0), lambda: None)
        case.barriers[0].seal(epoch)
        thread = Thread(target=sender.execute, args=(job,))
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            case.scheduler.begin_close(case.target)
            self.assertEqual(case.worker_status(0, "close-target"), ("closing", (0,)))
            self.assertEqual(case.worker_status(1, "close-target"), ("closed", ()))
            self.assertIsNone(sender.finish(case.target, cancel=True))
            self.assertTrue(case.d[1].try_closed(case.target))
            self.assertFalse(case.scheduler.closed(case.target, refs[1]))
            self.assertEqual(case.released, [])
        finally:
            complete.set()
            thread.join(5)
        outcome = sender.finish(case.target, cancel=True)
        self.assertEqual(outcome, NativeResult("DMA result unknown", False))
        case.collect_terminal(0, outcome)
        self.assertEqual(case.worker_status(0), ("closing", (0,)))
        self.assertFalse(case.barriers[0].may_dispose())
        sender.confirm_native_quiescence(case.target)
        case.collect_terminal(0, sender.finish(case.target, cancel=True))
        self.assertTrue(case.barriers[0].may_dispose())
        self.assertEqual(case.worker_status(0), ("closed", ()))
        self.assertTrue(case.scheduler.closed(case.target, refs[0]))
        sender.forget(case.target)

    def test_late_prepare_cannot_reopen_allocation(self):
        case = _Fixture()
        refs = case.scheduler.retain(case.allocation)
        case.scheduler.prepared(case.target, case.d[0].prepare(case.allocation, refs))
        self.assertIsNone(case.scheduler.publish(case.target))
        for ref in case.scheduler.begin_close(case.target):
            case.d[ref.rank].close(case.target)
            self.assertTrue(case.d[ref.rank].try_closed(case.target))
            case.scheduler.closed(case.target, ref)
        with self.assertRaisesRegex(RuntimeError, "generation is closed"):
            case.d[1].prepare(case.allocation, refs)
        new = replace(case.allocation, target=replace(case.target, allocation_generation=8))
        case.scheduler.retain(new)
        self.assertTrue(case.scheduler.closed(case.target, refs[0]))
        self.assertEqual(case.released, [case.allocation])

    def test_spoofed_writer_and_terminal_only_fail_admission(self):
        case = _Fixture()
        case.publish()
        case.collect_terminal(0, NativeResult(None, True))
        case.collect_terminal(1, NativeResult(None, True))
        self.assertFalse(case.d[0].try_ready(case.target))
        with self.assertRaises(ValueError):
            case.d[0].apply(case.target, (WriteResult(1, EntryId(0, "main", 0), Span(0, 1)),))
        self.assertFalse(case.d[0].try_ready(case.target))
        self.assertEqual(case.visibility, [])

    def test_wire_rejects_wrong_protocol_and_boolean_block(self):
        case = _Fixture()
        frame = _roundtrip(encode_allocation(case.allocation, case.layout.refs))
        frame[1] = 9
        with self.assertRaisesRegex(ValueError, "protocol version"):
            decode_allocation(frame)
        frame[1] = 1
        frame[4][0][0] = True
        with self.assertRaisesRegex(ValueError, "integer"):
            decode_allocation(frame)


if __name__ == "__main__":
    unittest.main()

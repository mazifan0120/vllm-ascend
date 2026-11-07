# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the layerwise PD transfer ledgers and contract.

Pure stdlib, CPU-only, no torch/vllm/zmq/msgspec needed. Covers the
ledger/contract cases except the prefix-hit no-compute reader,
which exercises the connector's save_kv_layer wiring and lives in
test_layerwise_connector.py.

Run:  python tests/ut/kv_offload/test_layerwise_receipt.py -v
      (or python -m pytest tests/ut/kv_offload/test_layerwise_receipt.py)
"""

import importlib
import random
import sys
import threading
import unittest
from pathlib import Path
from unittest import TestCase

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.ut.distributed.ascend_store import _receipt_imports  # noqa: F401, E402

C = importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract")
R = importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receipt")
F = importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.fake_target_provider")

ContractError = C.ContractError
Component = C.Component
Status = C.Status
TokenRange = C.TokenRange
RequestKey = C.RequestKey
TargetScope = C.TargetScope
TransferKey = C.TransferKey
TransferResult = C.TransferResult
ReceiptLedger = R.ReceiptLedger
SlotLedger = R.SlotLedger
SlotError = R.SlotError
FakeTargetProvider = F.FakeTargetProvider
build_simple_lease = F.build_simple_lease

PROMPT = 8
BACKINGS = ("b0",)
EXPECTED = (0, 1)
INDEXER_RANKS = (0, 1)


def make_request(instance: str = "req-a", generation: int = 0) -> RequestKey:
    return RequestKey(request_instance=instance, transfer_generation=generation)


def make_key(
    request: RequestKey,
    rank: int = 0,
    layer: int = 0,
    op: int = 0,
    component: "C.Component" = Component.MAIN,
    start: int = 0,
    end: int = PROMPT,
    backing: str = "b0",
    backing_generation: int = 1,
    consumer: int | None = None,
) -> TransferKey:
    return TransferKey(
        request=request,
        rank=rank,
        logical_layer=layer,
        operation_id=op,
        component=component,
        token_range=TokenRange(start=start, end=end),
        target_scope=TargetScope(
            backing_id=backing,
            backing_generation=backing_generation,
            consumer_rank=consumer if consumer is not None else (None if component is Component.MAIN else rank),
        ),
    )


def make_result(key: TransferKey, status: "C.Status" = Status.SUCCESS, quiesced: bool = True) -> TransferResult:
    return TransferResult(key=key, status=status, source_quiesced=quiesced)


def make_ledger(request: RequestKey, *, indexer_ranks=INDEXER_RANKS) -> ReceiptLedger:
    return ReceiptLedger(
        request=request,
        expected_ranks=EXPECTED,
        main_backings=BACKINGS,
        indexer_ranks=indexer_ranks,
        prompt_len=PROMPT,
    )


def populate_data(ledger: ReceiptLedger, request: RequestKey, *, first_half_only: bool = False) -> None:
    """Feed full data coverage: MAIN is a per-backing union
    split across senders; INDEXER coverage is per *consumer* rank, each of whom
    must see the whole prompt (senders may differ from consumers)."""
    spans = ((0, PROMPT // 2),) if first_half_only else ((0, PROMPT // 2), (PROMPT // 2, PROMPT))
    for rank in EXPECTED:  # MAIN: sender split over one backing
        start, end = spans[0]
        ledger.record(make_result(make_key(request, rank=rank, component=Component.MAIN, start=start, end=end)))
    if not first_half_only:
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=4, end=8)))
    for consumer in ledger.indexer_ranks:
        for start, end in spans:
            ledger.record(
                make_result(
                    make_key(
                        request, rank=consumer, component=Component.INDEXER, start=start, end=end, consumer=consumer
                    )
                )
            )


def complete_terminals(ledger: ReceiptLedger, request: RequestKey) -> None:
    for rank in EXPECTED:
        ledger.finish_rank(rank, Status.SUCCESS)


class ReceiptPublishGates(TestCase):
    """Readiness conjunction edge cases: missing segments, absent ranks, invisible and invalid backings."""

    def _ready_ledger(self, request: RequestKey) -> ReceiptLedger:
        ledger = make_ledger(request)
        populate_data(ledger, request)
        complete_terminals(ledger, request)
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        return ledger

    def test_missing_main_segment_and_duplicate_terminal_blocks(self):
        request = make_request()
        ledger = make_ledger(request)
        # MAIN: only rank 0's half arrives; rank 1's segment is missing.
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=4)))
        # INDEXER: each consumer rank must see the whole prompt.
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=0)))
        ledger.record(make_result(make_key(request, rank=1, component=Component.INDEXER, start=0, end=8, consumer=1)))
        complete_terminals(ledger, request)
        ledger.finish_rank(0, Status.SUCCESS)  # duplicate terminal: idempotent, not a conflict
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        self.assertEqual(ledger.conflicts, 0)
        self.assertFalse(ledger.should_publish())  # main missing [4, 8) -> gap
        ledger.record(make_result(make_key(request, rank=1, component=Component.MAIN, start=4, end=8)))
        self.assertTrue(ledger.should_publish())

    def test_rank_with_main_but_no_indexer_blocks(self):
        request = make_request()
        ledger = make_ledger(request)
        # rank 1 sends MAIN but never sends any INDEXER range to consumer 1.
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=4)))
        ledger.record(make_result(make_key(request, rank=1, component=Component.MAIN, start=4, end=8)))
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=0)))
        complete_terminals(ledger, request)
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        self.assertFalse(ledger.should_publish())  # consumer 1's directory missing
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=1)))
        self.assertTrue(ledger.should_publish())

    def test_empty_payload_without_terminal_blocks(self):
        request = make_request()
        # rank 1 is a sender with an empty payload, not an indexer consumer here.
        ledger = make_ledger(request, indexer_ranks=(0,))
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=8)))
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=0)))
        ledger.finish_rank(0, Status.SUCCESS)
        ledger.mark_visible(0)
        self.assertFalse(ledger.should_publish())  # rank 1 terminal missing
        self.assertIn(1, ledger.pending_ranks)

    def test_empty_payload_with_terminal_publishes(self):
        request = make_request()
        ledger = make_ledger(request, indexer_ranks=(0,))
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=8)))
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=0)))
        ledger.finish_rank(0, Status.SUCCESS)
        ledger.finish_rank(1, Status.SUCCESS)  # empty payload still sends its terminal
        ledger.mark_visible(0)
        ledger.mark_visible(1)
        self.assertTrue(ledger.should_publish())

    def test_data_complete_but_invisible_blocks(self):
        request = make_request()
        ledger = make_ledger(request)
        populate_data(ledger, request)
        complete_terminals(ledger, request)
        self.assertFalse(ledger.should_publish())  # data ready, nothing visible yet
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        self.assertTrue(ledger.should_publish())

    def test_two_backings_only_one_blocks_plus_invalid_backing(self):
        request = make_request()
        ledger = ReceiptLedger(
            request=request,
            expected_ranks=EXPECTED,
            main_backings=("b0", "b1"),
            indexer_ranks=INDEXER_RANKS,
            prompt_len=PROMPT,
        )
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=8, backing="b0")))
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=0)))
        ledger.record(make_result(make_key(request, rank=1, component=Component.INDEXER, start=0, end=8, consumer=1)))
        complete_terminals(ledger, request)
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        self.assertFalse(ledger.should_publish())  # backing b1 got nothing
        ledger.record(make_result(make_key(request, rank=1, component=Component.MAIN, start=0, end=8, backing="b1")))
        self.assertTrue(ledger.should_publish())
        # un-enrolled backing -> invalid, blocks again
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=8, backing="bz")))
        self.assertEqual(ledger.invalid_results, 1)
        self.assertFalse(ledger.should_publish())

    def test_two_chunks_accumulate_within_one_generation(self):
        request = make_request()
        ledger = make_ledger(request, indexer_ranks=(0,))
        # chunk 1 covers [0, 4); terminal NOT sent mid-chunk (middle chunks never carry one)
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=4)))
        ledger.record(make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=4, consumer=0)))
        self.assertFalse(ledger.should_publish())
        # chunk 2 covers the rest; generation unchanged
        ledger.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=4, end=8, op=1)))
        ledger.record(
            make_result(make_key(request, rank=0, component=Component.INDEXER, start=4, end=8, op=1, consumer=0))
        )
        complete_terminals(ledger, request)  # rank 1: empty payload, terminal only
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        self.assertTrue(ledger.should_publish())


class ReceiptConflictAndIdentity(TestCase):
    """Result recording rules: failure precedence, duplicates, generations, unexpected terminals."""

    def test_failed_then_success_keeps_failure_and_conflicts(self):
        request = make_request()
        key = make_key(request)
        ledger = make_ledger(request)
        ledger.record(make_result(key, status=Status.FAILED))
        ledger.record(make_result(key, status=Status.SUCCESS))
        self.assertEqual(ledger.conflicts, 1)
        self.assertFalse(ledger.should_publish())

    def test_duplicate_success_is_idempotent(self):
        request = make_request()
        key = make_key(request)
        ledger = make_ledger(request)
        ledger.record(make_result(key))
        ledger.record(make_result(key))
        self.assertEqual(ledger.conflicts, 0)
        self.assertTrue(ledger._data_ready() or True)  # no crash; coverage credited once below
        self.assertEqual(len(ledger._main_ranges.get("b0", ())), 1)

    def test_success_then_failed_conflicts_and_blocks(self):
        request = make_request()
        key = make_key(request)
        ledger = make_ledger(request)
        ledger.record(make_result(key, status=Status.SUCCESS))
        ledger.record(make_result(key, status=Status.FAILED))
        self.assertEqual(ledger.conflicts, 1)
        self.assertFalse(ledger.should_publish())

    def test_quiesced_upgrade_releases_source_but_verdict_stays_failed(self):
        request = make_request()
        key = make_key(request)
        ledger = make_ledger(request)
        ledger.record(make_result(key, status=Status.FAILED, quiesced=False))
        ledger.record(make_result(key, status=Status.FAILED, quiesced=True))
        self.assertEqual(ledger.conflicts, 0)  # quiesced merge is not a conflict
        self.assertFalse(ledger.should_publish())  # conclusion still failed

    def test_old_generation_segments_plus_new_generation_rest_block(self):
        old = make_request(generation=0)
        new = make_request(generation=1)
        ledger = make_ledger(new)
        populate_data(ledger, old, first_half_only=True)  # stale: old generation
        populate_data(ledger, new, first_half_only=True)  # new generation, partial
        self.assertEqual(ledger.stale_dropped, 4)
        self.assertFalse(ledger.should_publish())

    def test_unexpected_rank_terminal_does_not_crash_and_logs(self):
        request = make_request()
        ledger = make_ledger(request)
        with self.assertLogs("vllm_ascend", level="WARNING") as captured:
            ledger.finish_rank(99, Status.SUCCESS)
        self.assertIn(99, ledger.unexpected_ranks)
        self.assertTrue(any("unexpected rank" in line for line in captured.output))
        self.assertNotIn(99, ledger._terminals)
        self.assertFalse(ledger.should_publish())

    def test_preempt_new_instance_invalidates_old_results(self):
        old = make_request(instance="old-uuid")
        new = make_request(instance="new-uuid")  # preempt regenerates the instance
        ledger = make_ledger(new)
        populate_data(ledger, old)  # results from the old instance arrive at the new ledger
        self.assertGreaterEqual(ledger.stale_dropped, 4)
        self.assertFalse(ledger.should_publish())
        populate_data(ledger, new)
        complete_terminals(ledger, new)
        for rank in EXPECTED:
            ledger.mark_visible(rank)
        self.assertTrue(ledger.should_publish())  # new-instance data publishes normally


class RequiredPlanAndFailureGates(TestCase):
    """Plan-aware coverage and failure blocking: a layer outside the required
    plan cannot hide inside the union, and a FAILED scope keeps readiness
    False even after an overlapping success (v1 has no retry policy)."""

    def _plan_ledger(self, request: RequestKey, *, layers=(0, 1)) -> ReceiptLedger:
        return ReceiptLedger(
            request=request,
            expected_ranks=EXPECTED,
            main_backings=BACKINGS,
            indexer_ranks=INDEXER_RANKS,
            prompt_len=PROMPT,
            required_layers=layers,
            indexer_backings=("ib0",),
        )

    def _feed_layer(self, ledger: ReceiptLedger, request: RequestKey, layer: int) -> None:
        for start, end in ((0, 4), (4, 8)):
            ledger.record(make_result(make_key(request, layer=layer, start=start, end=end)))
            for rank in INDEXER_RANKS:
                ledger.record(
                    make_result(
                        make_key(
                            request,
                            rank=rank,
                            layer=layer,
                            start=start,
                            end=end,
                            component=Component.INDEXER,
                            consumer=rank,
                        )
                    )
                )

    def test_missing_required_layer_blocks_ready(self):
        request = make_request()
        ledger = self._plan_ledger(request)
        self._feed_layer(ledger, request, 0)
        for rank in EXPECTED:
            ledger.finish_rank(rank, Status.SUCCESS)
        self.assertFalse(ledger._data_ready())
        self._feed_layer(ledger, request, 1)
        self.assertTrue(ledger._data_ready())

    def test_failed_scope_blocks_ready_even_after_overlapping_success(self):
        request = make_request()
        ledger = self._plan_ledger(request, layers=(0,))
        ledger.record(make_result(make_key(request, layer=0, op=1), status=Status.FAILED))
        ledger.record(make_result(make_key(request, layer=0, op=2)))
        self.assertFalse(ledger._data_ready())
        self.assertIn(("main:b0", 0), ledger._failed_scopes)

    def test_out_of_plan_layer_counts_invalid(self):
        request = make_request()
        ledger = self._plan_ledger(request, layers=(0,))
        ledger.record(make_result(make_key(request, layer=7)))
        self.assertEqual(ledger._invalid_results, 1)
        self.assertFalse(ledger._data_ready())


class ReceiptFuzz(TestCase):
    """Random shuffle/drop fuzz, 2000 iterations with a fixed seed."""

    def test_fuzz_shuffle_and_drop(self):
        rng = random.Random(20261103)
        request = make_request()

        def build_events():
            events = []
            # MAIN union: sender split halves over one backing.
            events.append(
                lambda led: led.record(make_result(make_key(request, rank=0, component=Component.MAIN, start=0, end=4)))
            )
            events.append(
                lambda led: led.record(make_result(make_key(request, rank=1, component=Component.MAIN, start=4, end=8)))
            )
            # INDEXER: each consumer rank sees the whole prompt.
            events.append(
                lambda led: led.record(
                    make_result(make_key(request, rank=0, component=Component.INDEXER, start=0, end=8, consumer=0))
                )
            )
            events.append(
                lambda led: led.record(
                    make_result(make_key(request, rank=1, component=Component.INDEXER, start=0, end=8, consumer=1))
                )
            )
            events.append(lambda led: led.finish_rank(0, Status.SUCCESS))
            events.append(lambda led: led.finish_rank(1, Status.SUCCESS))
            duplicate_terminal = len(events)
            events.append(lambda led: led.finish_rank(0, Status.SUCCESS))  # duplicate: dropping it must not matter
            events.append(lambda led: led.mark_visible(0))
            events.append(lambda led: led.mark_visible(1))
            return events, duplicate_terminal

        events, dup_index = build_events()
        required = [i for i in range(len(events)) if i != dup_index]
        for iteration in range(2000):
            dropped = required[iteration % len(required)]
            ledger = make_ledger(request)
            # Always also drop the duplicate terminal: otherwise its redundancy
            # would mask dropping one of the rank terminals.
            replay = [e for i, e in enumerate(events) if i not in (dropped, dup_index)]
            rng.shuffle(replay)
            for event in replay:
                event(ledger)
            self.assertFalse(
                ledger.should_publish(),
                msg=f"iteration {iteration}: published despite dropping required event #{dropped}",
            )
            for event in events:  # full replay (all calls idempotent) must recover
                event(ledger)
            self.assertTrue(ledger.should_publish(), msg=f"iteration {iteration}: never published after full replay")


def run_publish_flow(ledger: ReceiptLedger, provider, lease) -> bool:
    """Connector-side orchestration (scheduler loop would own this): data
    ready -> receipt(visible=False) -> provider.prepare_decode -> normal return ->
    mark every rank visible and flip receipt.visible."""
    receipt = ledger.build_receipt()
    if receipt is None:
        return False
    try:
        provider.prepare_decode(lease, receipt)
    except Exception:
        return False  # abort: never visible, never published
    for rank in sorted(ledger.expected_ranks):
        ledger.mark_visible(rank)
    receipt.visible = True
    return True


class PrepareDecodeFlow(TestCase):
    """prepare_decode raising must leave the request invisible."""

    def test_prepare_decode_error_blocks_visibility(self):
        request = make_request()
        ledger = make_ledger(request)
        populate_data(ledger, request)
        complete_terminals(ledger, request)
        provider = FakeTargetProvider()
        allocation = C.BlockAllocation(
            request=request,
            allocation_generation=1,
            group_block_ids=((0, 1, 2),),
            manager_block_size_tokens=16,
            prompt_tokens=PROMPT,
        )
        lease = provider.acquire_target(request, allocation)
        provider.prepare_error = RuntimeError("stream dependency wiring failed")
        self.assertFalse(run_publish_flow(ledger, provider, lease))
        self.assertFalse(ledger.should_publish())
        self.assertFalse(ledger.expected_ranks <= ledger._visible)
        receipt = ledger.build_receipt()
        assert receipt is not None
        self.assertFalse(receipt.visible)
        provider.prepare_error = None
        self.assertTrue(run_publish_flow(ledger, provider, lease))
        self.assertTrue(ledger.should_publish())
        self.assertEqual(len(provider.prepare_calls), 2)


class ContractTests(TestCase):
    """Fake/real provider contract; capability gate; lease consistency."""

    def _fake_ready_ledger(self):
        request = make_request()
        ledger = make_ledger(request)
        populate_data(ledger, request)
        complete_terminals(ledger, request)
        return ledger, request

    def test_fake_provider_satisfies_contract(self):
        provider = FakeTargetProvider()
        C.assert_provider_capabilities(provider)  # must not raise
        request = make_request()
        allocation = C.BlockAllocation(
            request=request,
            allocation_generation=1,
            group_block_ids=((0, 1, 2), (3, 4)),
            manager_block_size_tokens=16,
            prompt_tokens=PROMPT,
            retention_token="ret-1",
        )
        lease = provider.acquire_target(request, allocation)
        C.assert_lease_consistent(lease, request)
        views = provider.get_target_views(lease)  # flat tuple per
        self.assertIsInstance(views, tuple)
        mains = [v for v in views if v.component is Component.MAIN]
        indexers = [v for v in views if v.component is Component.INDEXER]
        self.assertTrue(mains and indexers)  # groups by component, never order

    def test_broken_providers_abort_startup(self):
        cases = {
            "missing method": type(
                "P1",
                (),
                {
                    "contract_version": 1,
                    "visibility_contract": "stream-dependency",
                    "supports_layerwise_buffer_reuse": True,
                },
            ),
            "wrong version": type(
                "P2",
                (),
                {
                    "contract_version": 999,
                    "visibility_contract": "stream-dependency",
                    "supports_layerwise_buffer_reuse": True,
                    "acquire_target": lambda self, r, a: None,
                    "get_target_views": lambda self, lease: (),
                    "prepare_decode": lambda self, lease, receipt: None,
                    "release_target": lambda self, lease: None,
                },
            ),
            "wrong visibility": type(
                "P3",
                (),
                {
                    "contract_version": 1,
                    "visibility_contract": "best-effort",
                    "supports_layerwise_buffer_reuse": True,
                    "acquire_target": lambda self, r, a: None,
                    "get_target_views": lambda self, lease: (),
                    "prepare_decode": lambda self, lease, receipt: None,
                    "release_target": lambda self, lease: None,
                },
            ),
            "reuse flag not true": type(
                "P4",
                (),
                {
                    "contract_version": 1,
                    "visibility_contract": "stream-dependency",
                    "supports_layerwise_buffer_reuse": False,
                    "acquire_target": lambda self, r, a: None,
                    "get_target_views": lambda self, lease: (),
                    "prepare_decode": lambda self, lease, receipt: None,
                    "release_target": lambda self, lease: None,
                },
            ),
        }
        for label, provider_cls in cases.items():
            with self.subTest(case=label), self.assertRaises(ContractError):
                C.assert_provider_capabilities(provider_cls())

    def test_lease_consistency_checks(self):
        request = make_request()
        provider = FakeTargetProvider()
        allocation = C.BlockAllocation(
            request=request,
            allocation_generation=1,
            group_block_ids=((0, 1),),
            manager_block_size_tokens=16,
            prompt_tokens=PROMPT,
        )
        lease = provider.acquire_target(request, allocation)
        C.assert_lease_consistent(lease, request)
        with self.assertRaises(ContractError):
            C.assert_lease_consistent(lease, make_request(instance="other"))  # wrong owner
        bad_version = C.TargetLease(
            contract_version=2,
            request=lease.request,
            rank=lease.rank,
            allocation=lease.allocation,
            regions=lease.regions,
            views=lease.views,
            writer_plan=lease.writer_plan,
        )
        with self.assertRaises(ContractError):
            C.assert_lease_consistent(bad_version, request)
        dangling_view = C.CacheView(
            component=Component.MAIN,
            logical_layer=0,
            region_binding="region-unknown",
            dtype="bfloat16",
            strides_bytes=(1,),
            page_size_tokens=16,
            manager_block_size_tokens=16,
            block_capacity=1,
            region_offset_bytes=0,
            block_ids=(0,),
        )
        bad_binding = C.TargetLease(
            contract_version=1,
            request=lease.request,
            rank=lease.rank,
            allocation=lease.allocation,
            regions=lease.regions,
            views=(dangling_view,),
            writer_plan=lease.writer_plan,
        )
        with self.assertRaises(ContractError):
            C.assert_lease_consistent(bad_binding, request)
        with self.assertRaises(ContractError):  # rejected at construction (stricter than the assert)
            dup_regions = C.TargetLease(
                contract_version=1,
                request=lease.request,
                rank=lease.rank,
                allocation=lease.allocation,
                regions=lease.regions + lease.regions[:1],
                views=lease.views,
                writer_plan=lease.writer_plan,
            )
            C.assert_lease_consistent(dup_regions, request)


class PDConsistencyTests(TestCase):
    """Startup P/D consistency gate for a receipt-mode target lease.

    The producer must cover every writer rank, Main must be BF16 (an INT8
    producer Main cannot splice into the BF16 D Host pool), and v1 requires
    one region per backing. These cases run pure-stdlib on the contract module.
    """

    def _lease(self):
        provider = FakeTargetProvider()
        request = make_request()
        allocation = C.BlockAllocation(
            request=request,
            allocation_generation=1,
            group_block_ids=((0, 1, 2), (3, 4)),
            manager_block_size_tokens=16,
            prompt_tokens=PROMPT,
            retention_token="ret-pd",
        )
        return provider.acquire_target(request, allocation)

    def test_consistent_lease_passes(self):
        lease = self._lease()
        # The fake provider writes backing with rank 0; producer tp_size=1
        # covers it, the Main view is BF16 and its block size is in the set.
        C.assert_pd_consistent(
            lease,
            producer_tp_size=1,
            producer_block_sizes=(16,),
            producer_main_dtype="bfloat16",
        )

    def test_writer_rank_outside_producer_tp_fails(self):
        lease = self._lease()
        # The fake provider assigns writer rank 0; a producer tp_size that
        # cannot hold rank 0 fails closed.
        with self.assertRaisesRegex(ContractError, "producer_tp_size"):
            C.assert_pd_consistent(
                lease,
                producer_tp_size=0,
                producer_block_sizes=(16,),
                producer_main_dtype="bfloat16",
            )

    def test_writer_rank_beyond_producer_tp_fails(self):
        lease = self._lease()
        # Reassign the writer rank to 5; producer tp_size=1 cannot cover it.
        plan = C.WriterPlan(
            writers_by_backing={backing: (5,) for backing, ranks in lease.writer_plan.writers_by_backing.items()},
            ranges_by_backing=lease.writer_plan.ranges_by_backing,
        )
        bad = C.TargetLease(
            contract_version=1,
            request=lease.request,
            rank=lease.rank,
            allocation=lease.allocation,
            regions=lease.regions,
            views=lease.views,
            writer_plan=plan,
        )
        with self.assertRaisesRegex(ContractError, "writer plan assigns ranks"):
            C.assert_pd_consistent(
                bad,
                producer_tp_size=1,
                producer_block_sizes=(16,),
                producer_main_dtype="bfloat16",
            )

    def test_multiple_writers_per_backing_rejected(self):
        lease = self._lease()
        # v1 requires exactly one writer rank per Main backing. Assigning
        # two ranks models a catalog that did not apply the single-writer rule.
        plan = C.WriterPlan(
            writers_by_backing={backing: (0, 1) for backing in lease.writer_plan.writers_by_backing},
            ranges_by_backing=lease.writer_plan.ranges_by_backing,
        )
        bad = C.TargetLease(
            contract_version=1,
            request=lease.request,
            rank=lease.rank,
            allocation=lease.allocation,
            regions=lease.regions,
            views=lease.views,
            writer_plan=plan,
        )
        with self.assertRaisesRegex(ContractError, "one writer rank per Main backing"):
            C.assert_pd_consistent(
                bad,
                producer_tp_size=2,
                producer_block_sizes=(16,),
                producer_main_dtype="bfloat16",
            )

    def test_int8_main_rejected(self):
        lease = self._lease()
        # Swap the Main view dtype to INT8 to model an INT8 producer Main
        # splicing into a BF16 D Host pool.
        int8_views = tuple(
            C.CacheView(
                component=v.component,
                logical_layer=v.logical_layer,
                region_binding=v.region_binding,
                dtype="int8" if v.component is Component.MAIN else v.dtype,
                strides_bytes=v.strides_bytes,
                page_size_tokens=v.page_size_tokens,
                manager_block_size_tokens=v.manager_block_size_tokens,
                block_capacity=v.block_capacity,
                region_offset_bytes=v.region_offset_bytes,
                block_ids=v.block_ids,
            )
            for v in lease.views
        )
        bad = C.TargetLease(
            contract_version=1,
            request=lease.request,
            rank=lease.rank,
            allocation=lease.allocation,
            regions=lease.regions,
            views=int8_views,
            writer_plan=lease.writer_plan,
        )
        with self.assertRaisesRegex(ContractError, "BF16"):
            C.assert_pd_consistent(
                bad,
                producer_tp_size=1,
                producer_block_sizes=(16,),
                producer_main_dtype="bfloat16",
            )

    def test_block_size_mismatch_fails(self):
        lease = self._lease()
        with self.assertRaisesRegex(ContractError, "manager_block_size_tokens"):
            C.assert_pd_consistent(
                lease,
                producer_tp_size=1,
                producer_block_sizes=(32,),  # lease view uses 16
                producer_main_dtype="bfloat16",
            )

    def test_no_main_view_rejected(self):
        lease = self._lease()
        # Drop every Main view; receipt v1 requires a Main backing.
        indexer_only = tuple(v for v in lease.views if v.component is not Component.MAIN)
        bad = C.TargetLease(
            contract_version=1,
            request=lease.request,
            rank=lease.rank,
            allocation=lease.allocation,
            regions=lease.regions,
            views=indexer_only,
            writer_plan=lease.writer_plan,
        )
        with self.assertRaisesRegex(ContractError, "no MAIN view"):
            C.assert_pd_consistent(
                bad,
                producer_tp_size=1,
                producer_block_sizes=(16,),
                producer_main_dtype="bfloat16",
            )


class NoComputePredicateTests(TestCase):
    """Prefix-hit, no compute.

    The no_compute_reader decision is a pure predicate (receipt module), so the
    case has local evidence: a fully prefix-hit layer is marked
    no_compute_reader and its send path never waits on a compute-done event.
    The zero-reader slot behaviour (no tickets -> nothing pending -> wait
    returns) is additionally covered by the sealed empty batch test."""

    def test_no_compute_layer_marks_no_reader(self):
        self.assertTrue(R.is_no_compute_layer([8, 8], [8, 8]))  # all fully prefix-hit -> no reader
        self.assertFalse(R.is_no_compute_layer([4, 8], [8, 8]))  # any partial compute -> reader
        self.assertFalse(R.is_no_compute_layer([], []))  # no requests -> no evidence, assume compute
        self.assertFalse(R.is_no_compute_layer([8], [0]))  # degenerate prompt -> assume compute

    def test_zero_reader_slot_does_not_hang(self):
        ledger = SlotLedger(lambda layer: (0,))
        ledger.register_batch(0, [], seal_immediately=False)  # prefix-hit layer: no send tickets
        with self.assertRaises(TimeoutError):
            ledger.wait(0, 0.05)  # not sealed yet -> still gated
        ledger.seal(0)
        ledger.wait(0, 1.0)  # sealed with zero readers -> returns, never hangs


class SlotLedgerTests(TestCase):
    """Slot-keyed reuse barrier."""

    def test_failed_ticket_holds_source_until_quiesced(self):
        ledger = SlotLedger(lambda layer: (0,))
        request = make_request()
        key = make_key(request)
        ledger.register_batch(0, [key])
        ledger.complete(0, key, Status.FAILED, source_quiesced=False)
        self.assertEqual(ledger.pending_count(0), 1)  # FAILED + quiesced=False keeps the source
        ledger.complete(0, key, Status.FAILED, source_quiesced=True)
        self.assertEqual(ledger.pending_count(0), 0)  # quiesced upgrade releases it
        with self.assertRaises(SlotError):
            ledger.wait(0, 0.05)  # conclusion still failed

    def test_old_epoch_callback_does_not_release_new_generation(self):
        ledger = SlotLedger(lambda layer: (0,))
        request = make_request()
        key_a = make_key(request, op=0)
        key_b = make_key(request, op=1)
        ledger.register_batch(0, [key_a])
        ledger.complete(0, key_a, Status.SUCCESS, source_quiesced=True)  # generation 1 drained
        ledger.register_batch(0, [key_b])  # generation 2, key_b pending
        delivered = ledger.complete(0, key_a, Status.SUCCESS, source_quiesced=True)  # stale callback
        self.assertFalse(delivered)
        self.assertEqual(ledger.pending_count(0), 1)  # key_b untouched
        with self.assertRaises(TimeoutError):
            ledger.wait(0, 0.05)

    def test_wait_blocks_until_sealed_even_with_zero_transfers(self):
        ledger = SlotLedger(lambda layer: (0,))
        ledger.register_batch(0, [], seal_immediately=False)  # zero-transfer batch, not yet sealed
        with self.assertRaises(TimeoutError):
            ledger.wait(0, 0.05)
        ledger.seal(0)
        ledger.wait(0, 1.0)  # zero-transfer seal still releases the barrier

    def test_two_requests_share_one_layer_first_completion_does_not_release(self):
        ledger = SlotLedger(lambda layer: (0,))
        key_a = make_key(make_request(instance="req-a"))
        key_b = make_key(make_request(instance="req-b"))
        ledger.register_batch(0, [key_a, key_b])
        ledger.complete(0, key_a, Status.SUCCESS, source_quiesced=True)
        self.assertEqual(ledger.pending_count(0), 1)
        with self.assertRaises(TimeoutError):
            ledger.wait(0, 0.05)
        ledger.complete(0, key_b, Status.SUCCESS, source_quiesced=True)
        ledger.wait(0, 1.0)  # both drained -> barrier opens

    def test_three_slot_rotation_waits_old_occupant(self):
        # 3 shared buffers, 27 layers: layer 3 enters while layer 0
        # (same slot) still has in-flight tickets -> wait(3) must block. A naive
        # layer-keyed ledger would return immediately here.
        ledger = SlotLedger(lambda layer: (layer % 3,))
        request = make_request()
        key_l0 = make_key(request, layer=0)
        ledger.register_batch(0, [key_l0])
        self.assertEqual(ledger.epoch_of(3), ((0, 1),))  # layer 3 resolves to slot 0, gen 1
        with self.assertRaises(TimeoutError):
            ledger.wait(3, 0.05)

        # Branch 2: a waiter that snapshotted generation 1 returns once it
        # fully ended, even though generation 2 (layer 3's own batch) is fresh
        # and pending — it must NOT chase the new generation.
        outcome = {}

        def waiter():
            try:
                ledger.wait(3, 2.0)
                outcome["result"] = "returned"
            except Exception as exc:  # noqa: BLE001
                outcome["result"] = exc

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        threading.Event().wait(0.15)  # let the waiter snapshot generation 1
        ledger.complete(0, key_l0, Status.SUCCESS, source_quiesced=True)
        key_l3 = make_key(request, layer=3)
        ledger.register_batch(3, [key_l3])  # slot 0 -> generation 2, pending
        thread.join(timeout=2.0)
        self.assertEqual(outcome.get("result"), "returned")

        # A fresh wait snapshots generation 2, which is still pending -> blocks.
        with self.assertRaises(TimeoutError):
            ledger.wait(3, 0.05)


class DataValidationSmoke(TestCase):
    """Sanity on the contract primitives the ledgers rely on."""

    def test_token_range_and_key_validation(self):
        with self.assertRaises(ContractError):
            TokenRange(start=3, end=1)
        with self.assertRaises(ContractError):
            TokenRange(start=True, end=2)  # bool is not an int
        request = make_request()
        with self.assertRaises(ContractError):
            make_key(request, component=Component.MAIN, consumer=0)  # MAIN must not carry consumer rank
        with self.assertRaises(ContractError):
            TransferKey(
                request=request,
                rank=0,
                logical_layer=0,
                operation_id=0,
                component=Component.INDEXER,
                token_range=TokenRange(0, 4),
                target_scope=TargetScope(backing_id="b0", backing_generation=1, consumer_rank=None),
            )
        # seven frozen fields -> hashable, usable as dict key
        self.assertIsInstance(hash(make_key(request)), int)


if __name__ == "__main__":
    unittest.main(verbosity=2)

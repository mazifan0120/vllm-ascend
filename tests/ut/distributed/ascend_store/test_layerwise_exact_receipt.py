# SPDX-License-Identifier: Apache-2.0
"""Entry/range completion, stale messages and bounded retirement proofs."""

import unittest
from dataclasses import replace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_coordinator import coordinated
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    RequestKey,
    Status,
    TokenRange,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan


def ledger_fixture(rank=0):
    provider, _, request, allocation = coordinated()
    manifest = provider.manifest(provider.acquire_target(request, allocation))
    plan = ManifestWritePlan(manifest)
    keys = plan.receive_keys(rank)
    ledger = ExactReceiptLedger(request, keys, expected_ranks=(0, 1), target_generation=7, visibility_rank=rank)
    return ledger, keys


def terminals(ledger, status=Status.SUCCESS, quiesced=True):
    for rank in (0, 1):
        ledger.finish_rank(ledger.request, rank, status, quiesced)


def results(ledger, keys, status=Status.SUCCESS, quiesced=True):
    for key in keys:
        ledger.record(TransferResult(key, status, quiesced))


class TestExactReceiptLedger(unittest.TestCase):
    def test_missing_sibling_entry_cannot_be_credited_by_other_view_or_terminal(self):
        ledger, keys = ledger_fixture()
        results(ledger, keys[1:])
        terminals(ledger)
        self.assertTrue(ledger.remote_quiesced())
        self.assertIsNone(ledger.build_receipt())
        ledger.record(TransferResult(keys[0], Status.SUCCESS, True))
        receipt = ledger.build_receipt()
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.target_generation, 7)
        self.assertFalse(ledger.should_publish())
        ledger.mark_visible(0)
        self.assertTrue(ledger.should_publish())

    def test_both_d_owners_need_shared_main_and_only_own_indexer(self):
        for rank in (0, 1):
            ledger, keys = ledger_fixture(rank)
            self.assertEqual(len(keys), 8)
            self.assertTrue(all(k.target_scope.consumer_rank in (None, rank) for k in keys))
            results(ledger, keys)
            terminals(ledger)
            ledger.mark_visible(rank)
            self.assertTrue(ledger.should_publish())
            with self.assertRaises(ContractError):
                ledger.mark_visible(1 - rank)

    def test_out_of_order_overlapping_chunks_require_full_exact_entry_coverage(self):
        ledger, keys = ledger_fixture()
        results(ledger, keys[1:])
        terminals(ledger)
        for start, end in ((16, 23), (0, 8), (4, 12)):
            ledger.record(TransferResult(replace(keys[0], token_range=TokenRange(start, end)), Status.SUCCESS, True))
        self.assertIsNone(ledger.build_receipt())
        ledger.record(TransferResult(replace(keys[0], token_range=TokenRange(12, 16)), Status.SUCCESS, True))
        self.assertIsNotNone(ledger.build_receipt())

    def test_unquiesced_success_is_neither_coverage_nor_release_proof(self):
        ledger, keys = ledger_fixture()
        results(ledger, keys, quiesced=False)
        terminals(ledger)
        self.assertIsNone(ledger.build_receipt())
        self.assertFalse(ledger.remote_quiesced())
        results(ledger, keys)
        results(ledger, keys, quiesced=False)  # delayed messages cannot revoke completion
        self.assertIsNotNone(ledger.build_receipt())
        self.assertTrue(ledger.remote_quiesced())

    def test_closed_failed_or_cancelled_writer_allows_release_but_never_decode(self):
        for status in (Status.FAILED, Status.CANCELLED):
            ledger, keys = ledger_fixture()
            results(ledger, keys, status, False)
            terminals(ledger, status, False)
            self.assertFalse(ledger.remote_quiesced())
            results(ledger, keys, status, True)
            terminals(ledger, status, True)
            self.assertTrue(ledger.remote_quiesced())
            self.assertIsNone(ledger.build_receipt())
        ledger, _ = ledger_fixture()
        terminals(ledger, Status.CANCELLED, True)  # no jobs ever admitted by P
        self.assertTrue(ledger.remote_quiesced())
        self.assertIsNone(ledger.build_receipt())

    def test_terminal_quiescence_must_be_explicit_and_includes_empty_ranks(self):
        ledger, keys = ledger_fixture()
        results(ledger, keys)
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        self.assertFalse(ledger.remote_quiesced())
        ledger.finish_rank(ledger.request, 1, Status.SUCCESS, False)
        self.assertFalse(ledger.remote_quiesced())
        ledger.finish_rank(ledger.request, 1, Status.SUCCESS, True)
        self.assertTrue(ledger.remote_quiesced())
        with self.assertRaises(ContractError):
            ledger.finish_rank(ledger.request, 1, Status.SUCCESS, 1)

    def test_result_or_terminal_conflict_keeps_allocation_retained(self):
        for conflict in ("result", "terminal"):
            ledger, keys = ledger_fixture()
            results(ledger, keys)
            terminals(ledger)
            if conflict == "result":
                ledger.record(TransferResult(keys[0], Status.FAILED, True))
            else:
                ledger.finish_rank(ledger.request, 0, Status.CANCELLED, True)
            self.assertEqual(ledger.protocol_errors, 1)
            self.assertFalse(ledger.remote_quiesced())
            self.assertIsNone(ledger.build_receipt())

    def test_stale_request_generation_is_dropped_without_touching_current_proof(self):
        ledger, keys = ledger_fixture()
        old = RequestKey(ledger.request.request_instance, ledger.request.transfer_generation + 1)
        ledger.record(TransferResult(replace(keys[0], request=old), Status.FAILED, False))
        ledger.finish_rank(old, 0, Status.FAILED, False)
        self.assertEqual(ledger.stale_dropped, 2)
        self.assertEqual(ledger.protocol_errors, 0)
        results(ledger, keys)
        terminals(ledger)
        self.assertIsNotNone(ledger.build_receipt())

    def test_unknown_scope_range_or_writer_is_a_protocol_error(self):
        _, keys = ledger_fixture()
        for bad in (
            replace(keys[0], operation_id=100),
            replace(keys[0], target_scope=replace(keys[0].target_scope, backing_generation=99)),
            replace(keys[0], token_range=TokenRange(0, 24)),
            replace(keys[0], token_range=TokenRange(0, 0)),
        ):
            ledger, _ = ledger_fixture()
            ledger.record(TransferResult(bad, Status.SUCCESS, True))
            results(ledger, keys)
            terminals(ledger)
            self.assertEqual(ledger.protocol_errors, 1)
            self.assertFalse(ledger.remote_quiesced())
        ledger, _ = ledger_fixture()
        ledger.finish_rank(ledger.request, 5, Status.SUCCESS, True)
        self.assertEqual(ledger.protocol_errors, 1)

    def test_constructor_rejects_repeated_or_missing_component_plan(self):
        ledger, keys = ledger_fixture()
        for bad in ((), (keys[0], keys[0]), tuple(k for k in keys if k.component.value == "main")):
            with self.assertRaises(ContractError):
                ExactReceiptLedger(ledger.request, bad, expected_ranks=(0, 1), target_generation=0, visibility_rank=0)

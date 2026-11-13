# SPDX-License-Identifier: Apache-2.0
"""Actual owner use/closure state with CPU fences, including model exceptions."""

import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.core.receipt_execution import use_receipt_targets
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    ReceiveReceipt,
    Status,
    TargetClosedError,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_delivery import WorkerTargetDelivery
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_manifest import (
    RankTarget,
    TargetManifest,
    aggregate_rank_targets,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_owner import TargetOwnerCall, TargetOwnerDispatcher


def delivery_fixture():
    fixture = OwnerFixture(layers=1)
    lease = fixture.acquire()
    target = RankTarget(fixture.facade.catalog(), lease)
    manifest = TargetManifest(aggregate_rank_targets((target,)), (target,))
    records = []
    retired = []
    control = WorkerTargetDelivery(
        0, fixture.facade, lambda *args: records.append(args), retire=lambda *args: retired.append(args)
    )
    control.admit_target(manifest, "internal")
    ledger = records[0][2]
    return fixture, lease, manifest, control, ledger, retired


def complete(fixture, lease, manifest, ledger):
    for key in ManifestWritePlan(manifest).receive_keys(0):
        ledger.record(TransferResult(key, Status.SUCCESS, True))
    ledger.finish_rank(lease.request, 0, Status.SUCCESS, True)
    receipt = ledger.build_receipt()
    fixture.facade.confirm_native_received(lease, receipt)
    fixture.facade.prepare_decode(lease, receipt)
    ledger.mark_visible(0)


class TestReceiptExecution(unittest.TestCase):
    def test_native_receive_confirmation_requires_exact_complete_receipt_and_is_owner_local(self):
        fixture = OwnerFixture(layers=1)
        lease = fixture.acquire()
        receipt = ReceiveReceipt(lease.request, lease.allocation.allocation_generation, True, True, True)
        try:
            for bad in (
                replace(receipt, main_union_ok=False),
                replace(receipt, target_generation=99),
                replace(receipt, visible=True),
            ):
                with self.assertRaises(ContractError):
                    fixture.facade.confirm_native_received(lease, bad)
            self.assertEqual(fixture.streams.receive_confirms, 0)
            fixture.facade.confirm_native_received(lease, receipt)
            fixture.facade.confirm_native_received(lease, receipt)
            self.assertEqual(fixture.streams.receive_confirms, 1)
            with self.assertRaises(ContractError), fixture.facade.use(lease):
                pass  # A receive fence alone is not prepare_decode.
            reply = TargetOwnerDispatcher(fixture.facade).dispatch(
                TargetOwnerCall("not-remote-receive", "confirm_native_received", (lease, receipt))
            )
            self.assertEqual(reply.error_type, "ContractError")
            fixture.facade.prepare_decode(lease, receipt)
        finally:
            fixture.finish(lease)

    def test_delivery_covers_actual_owner_use_and_retains_compact_closed_identity(self):
        fixture, lease, manifest, control, ledger, retired = delivery_fixture()
        try:
            with control.source_lookup(lease):
                self.assertTrue(fixture.facade.resolve_target_regions(lease))
            with self.assertRaises(ContractError), control.use_context(("internal",)):
                pass
            complete(fixture, lease, manifest, ledger)
            with control.use_context(("internal", "ordinary-local")):
                self.assertEqual(fixture.runtime._states[lease.request].users, 1)
                with self.assertRaises(ContractError):
                    control.release_target(lease)
            self.assertEqual(fixture.runtime._states[lease.request].users, 0)
            self.assertEqual(control.delivery_status(lease), {"ready": True, "remote_quiesced": True})
            with self.assertRaises(TargetClosedError), control.source_lookup(lease):
                pass  # All future writes were already closed by the terminal.
            control.release_target(lease)
            self.assertEqual(len(retired), 1)
            self.assertFalse(control._deliveries)
            self.assertEqual(control.delivery_status(lease), {"ready": False, "remote_quiesced": True})
            with self.assertRaises(ContractError), control.use_context(("internal",)):
                pass  # A retired request is never treated as an ordinary one.
        finally:
            fixture.finish(lease)

    def test_aborted_delivery_rejects_new_addresses_and_late_results_never_enable_decode(self):
        fixture, lease, manifest, control, ledger, _ = delivery_fixture()
        try:
            with control.source_lookup(lease):
                pass
            control.abort_requests(("internal",))
            with self.assertRaises(TargetClosedError), control.source_lookup(lease):
                pass
            self.assertFalse(control.delivery_status(lease)["remote_quiesced"])
            for key in ManifestWritePlan(manifest).receive_keys(0):
                ledger.record(TransferResult(key, Status.SUCCESS, True))
            ledger.finish_rank(lease.request, 0, Status.SUCCESS, True)
            self.assertIsNone(ledger.build_receipt())
            self.assertEqual(control.delivery_status(lease), {"ready": False, "remote_quiesced": True})
            control.release_target(lease)
        finally:
            fixture.finish(lease)

    def test_scheduled_proof_avoids_per_rank_rpc_and_preserves_the_runtime_use_gate(self):
        fixture, lease, manifest, control, ledger, _ = delivery_fixture()
        try:
            complete(fixture, lease, manifest, ledger)
            allocations = fixture.retention.validate_scheduled(("internal", "ordinary"))
            with (
                self.assertRaisesRegex(ContractError, "no validated allocation"),
                control.use_context(("internal",), {}),
            ):
                pass
            changed = replace(allocations["internal"], allocation_generation=99)
            with (
                self.assertRaisesRegex(ContractError, "does not match the retained target"),
                control.use_context(("internal",), {"internal": changed}),
            ):
                pass
            state = fixture.runtime._states[lease.request]
            with patch.object(state.retention, "validate", wraps=state.retention.validate) as validate:
                with control.use_context(("internal", "ordinary"), allocations):
                    self.assertEqual(state.users, 1)
                validate.assert_not_called()
                with control.use_context(("internal",)):
                    self.assertEqual(state.users, 1)
                validate.assert_called_once_with()
            self.assertTrue(state.events)
            control.release_target(lease)
        finally:
            fixture.finish(lease)

    def test_cancel_before_any_address_lookup_can_retire_without_inventing_rank_terminals(self):
        fixture, lease, _, control, ledger, _ = delivery_fixture()
        try:
            control.abort_requests(("internal",))
            self.assertFalse(ledger.remote_quiesced())
            self.assertTrue(control.delivery_status(lease)["remote_quiesced"])
            with self.assertRaises(TargetClosedError), control.source_lookup(lease):
                pass
            control.release_target(lease)
        finally:
            fixture.finish(lease)

    def test_execution_decorator_records_queued_use_on_none_return_and_on_exception(self):
        events = []

        class Runner:
            @contextmanager
            def receipt_target_use_context(self, output):
                events.append(("enter", tuple(output.num_scheduled_tokens)))
                try:
                    yield
                finally:
                    events.append("record-queued-stream-tails")

            @use_receipt_targets
            def execute(self, output, *, fail=False):
                events.append("enqueue-model-and-writeback")
                if fail:
                    raise RuntimeError("injected model exception")
                return None

        output = SimpleNamespace(num_scheduled_tokens={"internal": 1})
        self.assertIsNone(Runner().execute(output))
        with self.assertRaises(RuntimeError):
            Runner().execute(output, fail=True)
        self.assertEqual(
            events, [("enter", ("internal",)), "enqueue-model-and-writeback", "record-queued-stream-tails"] * 2
        )

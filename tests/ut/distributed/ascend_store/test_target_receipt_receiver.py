# SPDX-License-Identifier: Apache-2.0
"""Real receipt parsing and lifecycle tests, runnable without the NPU runner."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    CONTRACT_VERSION,
    Component,
    ContractError,
    RequestKey,
    Status,
    TargetScope,
    TokenRange,
    TransferKey,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_protocol import (
    LAYER_RESULT_MSG,
    LAYER_RESULTS_MSG,
    PROOF_ACK,
    PROOF_ACK_TARGET_CLOSED,
    RANK_TERMINAL_MSG,
    RECEIPT_WATCHDOG_SECONDS,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receipt import ReceiptLedger
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_receiver import ReceiptReceiver


def _encode_decode(msg):
    return msgspec.msgpack.Decoder(type=tuple).decode(msgspec.msgpack.Encoder().encode(msg))


class TestExactReceiverProtocol(unittest.TestCase):
    def publisher(self):
        receiver, ledger, required = self.fixture()
        order = []
        receiver.status.receipt_provider = SimpleNamespace(
            confirm_native_received=lambda *args: order.append("native-receive-fence"),
            prepare_decode=lambda *args: order.append("install-consumer-dependency"),
        )
        ledger.lease = object()
        del receiver._maybe_publish
        return receiver, ledger, required, order

    def test_actual_publisher_confirms_receive_before_visibility_and_is_idempotent(self):
        receiver, ledger, required, order = self.publisher()
        ledger.record(TransferResult(required[0], Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(order, [])
        ledger.record(TransferResult(required[1], Status.SUCCESS, True))
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(order, ["native-receive-fence", "install-consumer-dependency"])
        self.assertEqual(receiver.status.done_requests, {"external"})
        receiver.status.done_requests.clear()
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(len(order), 2)

    def test_actual_watchdog_keeps_late_quiescence_without_late_decode(self):
        receiver, ledger, required, order = self.publisher()
        instance = ledger.request.request_instance
        receiver._sweep_receipt_deadlines()
        receiver.forget_request("external", failed=True)
        self.assertIs(receiver.requests.get(instance).ledger, ledger)
        for key in required:
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        receiver._maybe_publish(instance)
        self.assertTrue(ledger.remote_quiesced())
        self.assertFalse(receiver.status.done_requests)
        self.assertEqual(order, [])

    def test_actual_terminal_failure_fails_bounded_before_watchdog(self):
        receiver, ledger, _, order = self.publisher()
        ledger.finish_rank(ledger.request, 0, Status.FAILED, False)
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(receiver.status.failed_requests, {"external"})
        self.assertFalse(ledger.remote_quiesced())
        self.assertEqual(order, [])

    def fixture(self, instance="wire-entry-request"):
        request = RequestKey(instance, 5)
        required = tuple(
            TransferKey(
                request,
                0,
                0,
                entry,
                component,
                TokenRange(0, 23),
                TargetScope(
                    "shared-main" if component is Component.MAIN else "indexer",
                    7,
                    None if component is Component.MAIN else 0,
                ),
            )
            for entry, component in enumerate(Component)
        )
        ledger = ExactReceiptLedger(request, required, expected_ranks=(0,), target_generation=8, visibility_rank=0)
        receiver = ReceiptReceiver(RECEIPT_WATCHDOG_SECONDS, Mock())
        receiver.requests.install("external", "external", ledger, 0)
        receiver._maybe_publish = Mock()
        return receiver, ledger, required

    def test_actual_message_handlers_preserve_generation_entries_and_explicit_quiescence(self):
        receiver, ledger, required = self.fixture()
        for key in required:
            message = (
                LAYER_RESULT_MSG,
                CONTRACT_VERSION,
                key.request.request_instance,
                5,
                0,
                0,
                key.operation_id,
                key.component.value,
                0,
                23,
                key.target_scope.backing_id,
                7,
                "success",
                True,
            )
            receiver.handle_layer_result(_encode_decode(message))
        terminal = (RANK_TERMINAL_MSG, CONTRACT_VERSION, ledger.request.request_instance, 5, 0, "success", True)
        receiver.handle_rank_terminal(_encode_decode((*terminal[:3], 4, *terminal[4:])))
        self.assertFalse(ledger.remote_quiesced())
        receiver.handle_rank_terminal(_encode_decode(terminal[:-1]))
        self.assertFalse(ledger.remote_quiesced())
        receiver.handle_rank_terminal(_encode_decode(terminal))
        self.assertTrue(ledger.remote_quiesced())
        self.assertIsNotNone(ledger.build_receipt())

    def test_truthy_untyped_quiescence_and_empty_backing_cannot_credit_a_manifest_entry(self):
        receiver, ledger, required = self.fixture()
        key = required[0]
        message = (
            LAYER_RESULT_MSG,
            CONTRACT_VERSION,
            ledger.request.request_instance,
            5,
            0,
            0,
            key.operation_id,
            key.component.value,
            0,
            23,
            key.target_scope.backing_id,
            7,
            "success",
            1,
        )
        receiver.handle_layer_result(_encode_decode(message))
        receiver.handle_layer_result(_encode_decode((message[0], True, *message[2:13], True)))
        receiver.handle_layer_result(_encode_decode((*message[:10], "", *message[11:13], True)))
        for key in required[1:]:
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        self.assertIsNone(ledger.build_receipt())

    def test_actual_batch_handler_accepts_entry_results_and_explicit_cancel_terminal(self):
        receiver, ledger, required = self.fixture()
        frames = []
        for key in required:
            frames.append(
                (
                    LAYER_RESULT_MSG,
                    CONTRACT_VERSION,
                    key.request.request_instance,
                    5,
                    0,
                    0,
                    key.operation_id,
                    key.component.value,
                    0,
                    23,
                    key.target_scope.backing_id,
                    7,
                    "success",
                    True,
                )
            )
        frames.append((RANK_TERMINAL_MSG, CONTRACT_VERSION, ledger.request.request_instance, 5, 0, "cancelled", True))
        batch = _encode_decode((LAYER_RESULTS_MSG, CONTRACT_VERSION, tuple(frames)))
        self.assertEqual(receiver.handle_proof_batch(batch), PROOF_ACK)
        self.assertTrue(ledger.remote_quiesced())
        self.assertIsNone(ledger.build_receipt())
        ledger.abort_delivery()
        self.assertEqual(receiver.handle_proof_batch(batch), PROOF_ACK_TARGET_CLOSED)

    def test_actual_batch_handler_rejects_invalid_envelope_before_crediting_any_frame(self):
        receiver, ledger, _ = self.fixture()
        terminal = (
            RANK_TERMINAL_MSG,
            RECEIPT_WATCHDOG_SECONDS,
            CONTRACT_VERSION,
            ledger.request.request_instance,
            ledger.request.transfer_generation,
            0,
            "success",
            True,
        )
        stale_terminal = (*terminal[:3], ledger.request.transfer_generation + 1, *terminal[4:])
        for message in (
            (LAYER_RESULTS_MSG, True, ()),
            (LAYER_RESULTS_MSG, CONTRACT_VERSION, ()),
            (LAYER_RESULTS_MSG, CONTRACT_VERSION, ((b"unsupported",),)),
            (LAYER_RESULTS_MSG, CONTRACT_VERSION, (terminal, stale_terminal)),
        ):
            with self.assertRaises(ContractError):
                receiver.handle_proof_batch(_encode_decode(message))
        self.assertFalse(ledger.remote_quiesced())

    def test_provider_callback_runs_without_index_lock_and_cannot_publish_retired_record(self):
        receiver, ledger, required, _ = self.publisher()
        for key in required:
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        observed = []

        def confirm(*_):
            def retire():
                observed.append(receiver.requests.retire(ledger.request.request_instance, ledger))

            thread = threading.Thread(target=retire)
            thread.start()
            thread.join(1)
            self.assertFalse(thread.is_alive(), "Provider callback held the receive index lock")

        receiver.status.receipt_provider.confirm_native_received = confirm
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(observed, [True])
        self.assertFalse(receiver.status.done_requests)
        self.assertFalse(ledger.visibility_installed)

    def test_callback_abort_keeps_late_quiescence_without_publishing(self):
        receiver, ledger, required, _ = self.publisher()
        for key in required:
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        receiver.status.receipt_provider.prepare_decode = lambda *_: ledger.abort_delivery()
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertTrue(ledger.remote_quiesced())
        self.assertFalse(receiver.status.done_requests)

    def test_cancellation_winning_visibility_lock_cannot_escape_the_receive_loop(self):
        receiver, ledger, required, _ = self.publisher()
        for key in required:
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        mark = ledger.mark_visible

        def cancel_before_mark(rank):
            ledger.abort_delivery()
            mark(rank)

        ledger.mark_visible = cancel_before_mark
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertTrue(ledger.delivery_aborted)
        self.assertFalse(receiver.status.done_requests)

    def test_legacy_publish_keeps_mode_until_forget_and_uses_live_status_sets(self):
        status = SimpleNamespace(done_requests=set(), failed_requests=set(), receipt_provider=Mock())
        receiver = ReceiptReceiver(RECEIPT_WATCHDOG_SECONDS, Mock(), status=status)
        first_done = status.done_requests
        for i in range(2):
            _, source, keys = self.fixture(str(i))
            ledger = ReceiptLedger(
                request=source.request,
                expected_ranks=(0,),
                main_backings=("shared-main",),
                indexer_ranks=(0,),
                prompt_len=23,
                required_layers=(0,),
                indexer_backings=("indexer",),
            )
            ledger.lease = object()
            for key in keys:
                ledger.record(TransferResult(key, Status.SUCCESS, True))
            ledger.finish_rank(0, Status.SUCCESS)
            self.assertIsNotNone(ledger.build_receipt())
            request_id = "legacy-" + str(i)
            receiver.requests.install(request_id, request_id, ledger, 0)
            receiver._maybe_publish(str(i))
            self.assertEqual(status.done_requests, {request_id})
            self.assertIsNone(receiver.requests.get(str(i)))
            self.assertTrue(receiver.requests.in_receipt_mode(request_id))
            receiver.forget_request(request_id)
            self.assertFalse(receiver.requests.in_receipt_mode(request_id))
            status.done_requests = set()  # The socket adapter drains by replacing this set.
        self.assertEqual(first_done, {"legacy-0"})


class TestReceiveRequestIndex(unittest.TestCase):
    def test_malformed_unhashable_wire_identity_is_dropped(self):
        receiver, _, _ = TestExactReceiverProtocol().fixture()
        self.assertIsNone(receiver.requests.get({"not": "an identity"}))
        self.assertIsNone(receiver.requests.get(["not", "an identity"]))

    def test_active_alias_cannot_be_erased_by_another_request_retirement(self):
        receiver, first, _ = TestExactReceiverProtocol().fixture("first")
        _, second, _ = TestExactReceiverProtocol().fixture("second")
        index = receiver.requests
        index.retire("first", first)
        index.install("outer-inner", "outer", first, 10)
        index.install("outer-inner-tail", "outer-inner", second, 20)
        self.assertTrue(index.retire("first", first))
        self.assertTrue(index.in_receipt_mode("outer-inner"))
        index.retire("second", second)
        self.assertFalse(index.in_receipt_mode("outer-inner"))

    def test_shared_external_id_does_not_merge_worker_lifetimes_or_block_maps(self):
        receiver, first, _ = TestExactReceiverProtocol().fixture("first")
        _, second, _ = TestExactReceiverProtocol().fixture("second")
        first.lease = SimpleNamespace(allocation=SimpleNamespace(group_block_ids=((1, 3),)))
        second.lease = SimpleNamespace(allocation=SimpleNamespace(group_block_ids=((8, 9),)))
        index = receiver.requests
        index.retire("first", first)
        index.install("client-12345678", "client", first, 10)
        index.install("client-abcdefgh", "client", second, 20)
        self.assertEqual(index.retained_blocks("client-12345678"), ((1, 3),))
        self.assertEqual(index.retained_blocks("client-abcdefgh"), ((8, 9),))
        self.assertFalse(index.retire("first", second))
        self.assertTrue(index.retire("first", first))
        self.assertTrue(index.in_receipt_mode("client"))
        self.assertIs(index.get("second").ledger, second)
        self.assertTrue(index.retire("second", second))
        self.assertFalse(index.in_receipt_mode("client"))
        self.assertIsNone(index.retained_blocks("client-abcdefgh"))

    def test_duplicate_identity_and_internal_id_fail_without_replacing_live_ledger(self):
        receiver, first, _ = TestExactReceiverProtocol().fixture("first")
        _, other, _ = TestExactReceiverProtocol().fixture("other")
        with self.assertRaises(ContractError):
            receiver.requests.install("another-request", "another", first, 5)
        with self.assertRaises(ContractError):
            receiver.requests.install("external", "external", other, 5)
        self.assertIs(receiver.requests.get("first").ledger, first)
        self.assertIsNone(receiver.requests.get("other"))

    def test_forget_aborts_exact_target_but_retains_its_late_proof_identity(self):
        receiver, ledger, required = TestExactReceiverProtocol().fixture()
        receiver.forget_request("external", failed=True)
        self.assertTrue(ledger.delivery_aborted)
        self.assertIs(receiver.requests.get(ledger.request.request_instance).ledger, ledger)
        for key in required:
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, Status.SUCCESS, True)
        self.assertTrue(ledger.remote_quiesced())

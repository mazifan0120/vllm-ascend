# mypy: ignore-errors
# SPDX-License-Identifier: Apache-2.0
"""Connector-level tests for the layerwise receipt protocol.

These need the real connector module (torch / torch_npu / vllm / mooncake), so
they skip on CPU-only boxes — the ledger/contract logic they rely on is covered
by test_layerwise_receipt.py, which runs anywhere. The no-compute marking
case exercises save_kv_layer end to end and belongs on the NPU CI.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

npu = pytest.importorskip("torch_npu")
msgspec = pytest.importorskip("msgspec")

from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as lc  # noqa: E402
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_exact_receipt import ExactReceiptLedger  # noqa: E402


def _encode_decode(msg):
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(type=tuple)
    return decoder.decode(encoder.encode(msg))


def test_producer_decodes_the_remote_target_only_on_the_first_chunk():
    identity = lc.RequestKey("instance", 0)
    lease = SimpleNamespace(request=identity, writer_plan=None)
    manifest = SimpleNamespace(lease=lease)
    request = SimpleNamespace(
        all_token_ids=[1] * 384,
        kv_transfer_params={
            "do_remote_decode": True,
            "request_instance": identity.request_instance,
            "remote_target_lease": {"lease": True},
            "remote_target_manifest": {"manifest": True},
        },
    )
    scheduler = lc.MooncakeLayerwiseConnectorScheduler.__new__(lc.MooncakeLayerwiseConnectorScheduler)
    scheduler.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=False))
    scheduler.block_size = [128]
    scheduler._reqs_need_send_layerwise = {"req": lc.SendReqInfo([[0]], 0, 0, request)}
    cached = SimpleNamespace(req_ids=["req"], new_block_ids=[None], num_computed_tokens=[0])
    step = SimpleNamespace(
        scheduled_cached_reqs=cached,
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={"req": 128},
    )
    with (
        patch.object(lc, "target_lease_from_wire", return_value=lease) as parse_lease,
        patch.object(lc, "target_manifest_from_wire", return_value=manifest) as parse_manifest,
    ):
        first = scheduler.build_connector_meta(step).requests["req"]
        cached.num_computed_tokens = [128]
        second = scheduler.build_connector_meta(step).requests["req"]

    assert first.remote_target_manifest is second.remote_target_manifest is manifest
    assert first.local_computed_tokens == 128 and second.local_computed_tokens == 256
    assert second.local_transed_tokens == 128
    parse_lease.assert_called_once_with(request.kv_transfer_params["remote_target_lease"])
    parse_manifest.assert_called_once_with(request.kv_transfer_params["remote_target_manifest"])


def test_consumer_publishes_retention_proof_in_scheduler_step_metadata():
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_retention import SchedulerRetentionRegistry

    block_ids = ((1, 3),)
    registry = SchedulerRetentionRegistry(SimpleNamespace(get_block_ids=lambda _: block_ids))
    allocation = registry.issue("internal", lc.BlockAllocation(lc.RequestKey("instance", 0), 2, block_ids, 16, 23))
    scheduler = object.__new__(lc.MooncakeLayerwiseConnectorScheduler)
    scheduler.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_consumer=True))
    scheduler._retention_registry = registry
    scheduler._reqs_need_recv = {}
    step = SimpleNamespace(num_scheduled_tokens={"internal": 1, "ordinary": 1})
    guard = registry.retain(allocation)
    try:
        metadata = scheduler.build_connector_meta(step)
        assert metadata.scheduled_target_allocations == {"internal": allocation}
        connector = object.__new__(lc.MooncakeLayerwiseConnector)
        delivery = SimpleNamespace(use_context=Mock())
        connector.connector_worker = SimpleNamespace(_target_delivery=delivery)
        connector.target_use_context(tuple(step.num_scheduled_tokens), metadata)
        delivery.use_context.assert_called_once_with(("internal", "ordinary"), {"internal": allocation})
    finally:
        guard.release()


class TestExactReceiverProtocol(unittest.TestCase):
    def publisher(self):
        receiver, ledger, required = self.fixture()
        receiver.instance_to_req = {ledger.request.request_instance: "external"}
        receiver.done_requests = set()
        receiver.failed_requests = set()
        receiver.receipt_mode_reqs = set()
        receiver.receipt_deadlines = {}
        order = []
        receiver.receipt_provider = SimpleNamespace(
            confirm_native_received=lambda *args: order.append("native-receive-fence"),
            prepare_decode=lambda *args: order.append("install-consumer-dependency"),
        )
        ledger.lease = object()
        receiver._maybe_publish = lc.KVCacheRecvingLayerThread._maybe_publish.__get__(receiver)
        return receiver, ledger, required, order

    def test_actual_publisher_confirms_receive_before_visibility_and_is_idempotent(self):
        receiver, ledger, required, order = self.publisher()
        ledger.record(lc.TransferResult(required[0], lc.Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, lc.Status.SUCCESS, True)
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(order, [])
        ledger.record(lc.TransferResult(required[1], lc.Status.SUCCESS, True))
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(order, ["native-receive-fence", "install-consumer-dependency"])
        self.assertEqual(receiver.done_requests, {"external"})
        receiver.done_requests.clear()
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(len(order), 2)

    def test_actual_watchdog_keeps_late_quiescence_without_late_decode(self):
        receiver, ledger, required, order = self.publisher()
        instance = ledger.request.request_instance
        receiver.receipt_deadlines[instance] = 0
        receiver._sweep_receipt_deadlines()
        receiver.forget_request("external", failed=True)
        self.assertIs(receiver.ledger_by_req[instance], ledger)
        for key in required:
            ledger.record(lc.TransferResult(key, lc.Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, lc.Status.SUCCESS, True)
        receiver._maybe_publish(instance)
        self.assertTrue(ledger.remote_quiesced())
        self.assertFalse(receiver.done_requests)
        self.assertEqual(order, [])

    def test_actual_terminal_failure_fails_bounded_before_watchdog(self):
        receiver, ledger, _, order = self.publisher()
        ledger.finish_rank(ledger.request, 0, lc.Status.FAILED, False)
        receiver._maybe_publish(ledger.request.request_instance)
        self.assertEqual(receiver.failed_requests, {"external"})
        self.assertFalse(ledger.remote_quiesced())
        self.assertEqual(order, [])

    def fixture(self):
        request = lc.RequestKey("wire-entry-request", 5)
        required = tuple(
            lc.TransferKey(
                request,
                0,
                0,
                entry,
                component,
                lc.TokenRange(0, 23),
                lc.TargetScope(
                    "shared-main" if component is lc.Component.MAIN else "indexer",
                    7,
                    None if component is lc.Component.MAIN else 0,
                ),
            )
            for entry, component in enumerate(lc.Component)
        )
        ledger = ExactReceiptLedger(request, required, expected_ranks=(0,), target_generation=8, visibility_rank=0)
        receiver = object.__new__(lc.KVCacheRecvingLayerThread)
        receiver.lock = threading.RLock()
        receiver.ledger_by_req = {request.request_instance: ledger}
        receiver._maybe_publish = Mock()
        return receiver, ledger, required

    def test_actual_message_handlers_preserve_generation_entries_and_explicit_quiescence(self):
        receiver, ledger, required = self.fixture()
        for key in required:
            message = (
                lc.LAYER_RESULT_MSG,
                lc.CONTRACT_VERSION,
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
        terminal = (lc.RANK_TERMINAL_MSG, lc.CONTRACT_VERSION, ledger.request.request_instance, 5, 0, "success", True)
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
            lc.LAYER_RESULT_MSG,
            lc.CONTRACT_VERSION,
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
            ledger.record(lc.TransferResult(key, lc.Status.SUCCESS, True))
        ledger.finish_rank(ledger.request, 0, lc.Status.SUCCESS, True)
        self.assertIsNone(ledger.build_receipt())

    def test_actual_batch_handler_accepts_entry_results_and_explicit_cancel_terminal(self):
        receiver, ledger, required = self.fixture()
        frames = []
        for key in required:
            frames.append(
                (
                    lc.LAYER_RESULT_MSG,
                    lc.CONTRACT_VERSION,
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
        frames.append(
            (lc.RANK_TERMINAL_MSG, lc.CONTRACT_VERSION, ledger.request.request_instance, 5, 0, "cancelled", True)
        )
        batch = _encode_decode((lc.LAYER_RESULTS_MSG, lc.CONTRACT_VERSION, tuple(frames)))
        self.assertEqual(receiver.handle_proof_batch(batch), lc.PROOF_ACK)
        self.assertTrue(ledger.remote_quiesced())
        self.assertIsNone(ledger.build_receipt())
        ledger.abort_delivery()
        self.assertEqual(receiver.handle_proof_batch(batch), lc.PROOF_ACK_TARGET_CLOSED)

    def test_actual_batch_handler_rejects_invalid_envelope_before_crediting_any_frame(self):
        receiver, ledger, _ = self.fixture()
        terminal = (
            lc.RANK_TERMINAL_MSG,
            lc.CONTRACT_VERSION,
            ledger.request.request_instance,
            ledger.request.transfer_generation,
            0,
            "success",
            True,
        )
        stale_terminal = (*terminal[:3], ledger.request.transfer_generation + 1, *terminal[4:])
        for message in (
            (lc.LAYER_RESULTS_MSG, True, ()),
            (lc.LAYER_RESULTS_MSG, lc.CONTRACT_VERSION, ()),
            (lc.LAYER_RESULTS_MSG, lc.CONTRACT_VERSION, ((b"unsupported",),)),
            (lc.LAYER_RESULTS_MSG, lc.CONTRACT_VERSION, (terminal, stale_terminal)),
        ):
            with self.assertRaises(lc.ContractError):
                receiver.handle_proof_batch(_encode_decode(message))
        self.assertFalse(ledger.remote_quiesced())


class TestEarlyTargetAdmission(unittest.TestCase):
    def test_actual_install_maps_early_completions_before_scheduler_metadata(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.request_map = {}
        worker.kv_recv_layer_thread = SimpleNamespace(
            lock=threading.RLock(), ledger_by_req={}, instance_to_req={}, receipt_mode_reqs=set(), receipt_deadlines={}
        )
        _, ledger, _ = TestExactReceiverProtocol().fixture()
        request_id = "early-request-12345678"
        lease = SimpleNamespace(request=ledger.request, allocation=SimpleNamespace(group_block_ids=((1, 3),)))
        worker._install_admitted_target(request_id, lease, ledger)
        external = lc.get_external_request_id(request_id)
        self.assertEqual(worker.request_map[request_id], request_id)
        self.assertNotIn(external, worker.request_map)
        self.assertIs(worker.kv_recv_layer_thread.ledger_by_req[ledger.request.request_instance], ledger)
        self.assertEqual(worker._admitted_target_blocks[request_id], ((1, 3),))
        with self.assertRaises(lc.ContractError):
            worker._install_admitted_target(request_id, lease, ledger)

    def test_reused_external_id_keeps_both_engine_lifetimes_until_their_own_retirement(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.request_map = {}
        worker.kv_recv_layer_thread = SimpleNamespace(
            lock=threading.RLock(), ledger_by_req={}, instance_to_req={}, receipt_mode_reqs=set(), receipt_deadlines={}
        )
        instances = []
        for suffix in ("12345678", "abcdefgh"):
            _, ledger, _ = TestExactReceiverProtocol().fixture()
            ledger.request = lc.RequestKey(suffix, 0)
            request_id = "same-external-" + suffix
            lease = SimpleNamespace(request=ledger.request, allocation=SimpleNamespace(group_block_ids=((1, 3),)))
            worker._install_admitted_target(request_id, lease, ledger)
            instances.append((request_id, lease, ledger))
        self.assertEqual(len(worker.request_map), 2)
        worker._retire_admitted_target(*instances[0])
        request_id, lease, ledger = instances[1]
        self.assertEqual(worker.request_map, {request_id: request_id})
        self.assertIs(worker.kv_recv_layer_thread.ledger_by_req[lease.request.request_instance], ledger)
        self.assertIn("same-external", worker.kv_recv_layer_thread.receipt_mode_reqs)
        worker._retire_admitted_target(*instances[1])
        self.assertFalse(worker.request_map)
        self.assertFalse(worker.kv_recv_layer_thread.receipt_mode_reqs)


class TestReceiptFailureIdentity(unittest.TestCase):
    def test_actual_transfer_results_report_receipt_failure_by_request_without_numeric_block_invalidation(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False)
        )
        failed = "failed-request-12345678"
        other = "ordinary-request-abcdefgh"
        worker.request_map = {failed: failed, "ordinary-request": other}
        worker._admitted_target_blocks = {failed: ((1, 3), (1, 3))}
        worker._invalid_block_ids = set()
        worker._failed_receives = set()
        worker._recving_metadata = {other: SimpleNamespace(local_block_ids=[[1, 3], [1, 3]])}
        worker.virtual_request = set()
        worker.kv_recv_layer_thread = SimpleNamespace(
            get_and_clear_done_requests=lambda: set(),
            get_and_clear_failed_requests=lambda: {failed},
            forget_request=Mock(),
        )
        connector = object.__new__(lc.MooncakeLayerwiseConnector)
        connector.connector_worker = worker
        result = connector.get_transfer_results(set())
        self.assertEqual(result.failed_recving, {failed})
        self.assertEqual(result.finished_recving, {failed})
        self.assertFalse(worker.get_block_ids_with_load_errors())
        self.assertEqual(worker.request_map, {failed: failed, "ordinary-request": other})
        self.assertIn(failed, worker._admitted_target_blocks)
        worker.kv_recv_layer_thread.forget_request.assert_called_once_with(failed, failed=True)
        self.assertFalse(worker.take_failed_receives())


class TestSourceReadyPreparation(unittest.TestCase):
    def test_actual_runner_context_uses_registered_owner_and_preserves_ordinary_mode(self):
        from contextlib import nullcontext

        from vllm_ascend.worker import model_runner_v1 as mr

        metadata = object()
        output = SimpleNamespace(num_scheduled_tokens={"internal": 1}, kv_connector_metadata=metadata)
        context = nullcontext()
        group = SimpleNamespace(requires_target_owner=True, target_use_context=Mock(return_value=context))
        runner = object.__new__(mr.NPUModelRunner)
        with (
            patch.object(mr, "has_kv_transfer_group", return_value=True),
            patch.object(mr, "get_kv_transfer_group", return_value=group),
        ):
            self.assertIs(runner.receipt_target_use_context(output), context)
            group.target_use_context.assert_called_once_with(("internal",), metadata)
            group.requires_target_owner = False
            runner.receipt_target_use_context(output)
            self.assertEqual(group.target_use_context.call_count, 1)

    def worker(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace(is_kv_producer=True))
        worker.total_layers = 1
        worker.pcp_size = worker.dcp_size = worker.pd_head_ratio = 1
        worker.tp_size = 2
        worker.enable_c8_quant = worker.enable_kv_quant = False
        worker._source_contract_bound = True
        worker._manifest_source_writer = SimpleNamespace(register_request=Mock(), prepare_job=Mock(return_value="job"))
        worker.attn_resharding_group_idx = set()
        worker._align_remote_block_ids = Mock(side_effect=AssertionError("manifest must retain manager block IDs"))
        target = SimpleNamespace(lease=SimpleNamespace(allocation=SimpleNamespace(prompt_tokens=23)))
        meta = SimpleNamespace(
            remote_target_manifest=target,
            remote_pcp_size=1,
            remote_dcp_size=1,
            remote_tp_size=2,
            local_transed_tokens=16,
        )
        metadata = SimpleNamespace(requests={"internal-source": meta})
        return worker, metadata

    def test_real_prepare_preserves_manager_ids_and_registers_before_source_callbacks(self):
        worker, metadata = self.worker()
        before = metadata.requests["internal-source"]
        worker.prepare_layerwise_step(metadata)
        worker._manifest_source_writer.register_request.assert_called_once_with("internal-source", before)
        self.assertIs(metadata.requests["internal-source"], before)
        worker._align_remote_block_ids.assert_not_called()

    def test_real_prepare_sizes_compute_events_for_registered_draft_layer(self):
        worker, metadata = self.worker()
        worker._compute_done_events = [None] * 78
        worker._no_compute_layers = {0}
        worker.total_layers = 79
        worker.prepare_layerwise_step(metadata)
        self.assertEqual(worker._compute_done_events, [None] * 79)
        self.assertFalse(worker._no_compute_layers)

    def test_real_prepare_rejects_unwired_source_contract_before_any_transfer_job(self):
        worker, metadata = self.worker()
        worker._source_contract_bound = False
        with self.assertRaisesRegex(lc.ContractError, "wired Store"):
            worker.prepare_layerwise_step(metadata)
        worker._manifest_source_writer.register_request.assert_not_called()

    def test_real_source_batch_clamps_store_private_identity_to_actual_pd_tail(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.reuse_source import RequestSource

        worker, metadata = self.worker()
        worker.prepare_layerwise_step(metadata)
        source, event = object(), object()
        request = RequestSource("internal-source", "private-store-instance", 0, 24, ((2, 0), (3, 1)))
        jobs, legacy = worker.prepare_source_batch(source, (request,), event)
        self.assertEqual(jobs, ("job",))
        self.assertFalse(legacy)
        actual = worker._manifest_source_writer.prepare_job.call_args.args
        self.assertIs(actual[0], source)
        self.assertIs(actual[2], event)
        self.assertEqual((actual[1].start_token, actual[1].end_token), (16, 23))
        self.assertEqual(actual[1].request_instance, "private-store-instance")
        self.assertEqual(actual[1].block_ids_by_group, request.block_ids_by_group)

    def test_real_source_batch_rejects_missing_store_request(self):
        worker, metadata = self.worker()
        worker.prepare_layerwise_step(metadata)
        with self.assertRaisesRegex(lc.ContractError, "missing from the Store"):
            worker.prepare_source_batch(object(), (), object())
        worker._manifest_source_writer.prepare_job.assert_not_called()


class TestSchedulerRetentionBinding:
    def test_direct_connector_binds_its_stable_provider_without_multi_wrapper(self):
        for role, factory in (
            (lc.KVConnectorRole.SCHEDULER, "MooncakeLayerwiseConnectorScheduler"),
            (lc.KVConnectorRole.WORKER, "MooncakeLayerwiseConnectorWorker"),
        ):
            inner = SimpleNamespace(attach_target_provider=Mock())
            config = SimpleNamespace(
                kv_transfer_config=SimpleNamespace(
                    is_kv_producer=False,
                    is_kv_consumer=True,
                    engine_id="unit-target",
                    kv_connector_extra_config={"layerwise_receipt_mode": True},
                ),
                parallel_config=SimpleNamespace(tensor_parallel_size=2),
            )
            with patch.object(lc, factory, return_value=inner):
                connector = lc.MooncakeLayerwiseConnector(config, role, None)
            assert connector._target_provider is connector.target_provider
            inner.attach_target_provider.assert_called_once_with(connector.target_provider)
            connector.set_target_provider(connector.target_provider)
            inner.attach_target_provider.assert_called_once()

    def test_plain_direct_connector_does_not_install_receipt_provider(self):
        inner = SimpleNamespace(attach_target_provider=Mock())
        config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_producer=False,
                is_kv_consumer=True,
                engine_id="plain-target",
                kv_connector_extra_config={},
            )
        )
        with patch.object(lc, "MooncakeLayerwiseConnectorScheduler", return_value=inner):
            connector = lc.MooncakeLayerwiseConnector(config, lc.KVConnectorRole.SCHEDULER, None)
        assert connector.target_provider is None
        inner.attach_target_provider.assert_not_called()

    def test_standard_kv_cache_manager_hook_binds_receipt_authority(self):
        connector = object.__new__(lc.MooncakeLayerwiseConnector)
        bind = Mock()
        connector.connector_scheduler = SimpleNamespace(bind_scheduler_retention=bind)
        connector.target_provider = object()
        manager = SimpleNamespace(block_pool=object())
        connector.bind_kv_cache_manager(manager)
        assert connector._kv_cache_manager is manager
        bind.assert_called_once_with(manager)

    def test_plain_scheduler_retains_the_existing_binding_behavior(self):
        connector = object.__new__(lc.MooncakeLayerwiseConnector)
        bind = Mock()
        connector.connector_scheduler = SimpleNamespace(bind_scheduler_retention=bind)
        connector.target_provider = None
        manager = SimpleNamespace(block_pool=object())
        connector.bind_kv_cache_manager(manager)
        assert connector._kv_cache_manager is manager
        bind.assert_not_called()

    def test_finished_hook_defers_target_without_releasing_before_gpu_fence(self):
        scheduler = object.__new__(lc.MooncakeLayerwiseConnectorScheduler)
        scheduler._request_leases = {"instance": object()}
        scheduler._release_target_if_any = Mock(
            side_effect=AssertionError("retirement must follow scheduler GPU fence")
        )
        request = SimpleNamespace(kv_transfer_params={"request_instance": "instance"})
        assert scheduler.request_finished(request, []) == (True, None)
        assert scheduler.request_finished_all_groups(request, ([],)) == (True, None)
        scheduler._release_target_if_any.assert_not_called()

    def test_plain_finished_hook_preserves_immediate_release(self):
        scheduler = object.__new__(lc.MooncakeLayerwiseConnectorScheduler)
        scheduler._request_leases = {}
        request = SimpleNamespace(kv_transfer_params={})
        assert scheduler.request_finished(request, []) == (False, None)


class TestSourceDependencies(unittest.TestCase):
    """Use real sender code; the CPU engine records submission, never DMA."""

    def _sender_and_task(self, failed_fence=None):
        observed = []

        def fence(name):
            def synchronize():
                observed.append(name)
                if name == failed_fence:
                    raise TimeoutError(name)

            return SimpleNamespace(synchronize=synchronize)

        sender = object.__new__(lc.KVCacheSendingLayerThread)
        sender.pd_head_ratio = 1
        sender.layer_metadata = {"layer": lc.LayerMetadata([0], [4096], [16], [1])}
        sender.get_transfer_meta = lambda *args: ([4096], [8192], [16])

        def submit(*args):
            observed.append("submit")
            return 0

        sender.engine = SimpleNamespace(batch_transfer_sync_write=submit)
        metadata = lc.ReqMeta([[0]], None, [[0]], [[16]], None, None, None, None, None, None, 1, 1, 1)
        metadata.remote_te_rpc_endpoint = "127.0.0.1:1"
        task = lc.SendTask(
            send_request={"request": metadata},
            layer_name="layer",
            wait_event=fence("legacy"),
            source_ready_event=fence("source"),
        )
        return sender, task, observed

    def test_both_fences_precede_transfer_submission(self):
        sender, task, observed = self._sender_and_task()
        self.assertIsNone(sender._transfer_kv_cache(task))
        self.assertEqual(observed, ["source", "legacy", "submit"])

    def test_failed_source_fence_prevents_submission(self):
        sender, task, observed = self._sender_and_task("source")
        with self.assertRaises(TimeoutError):
            sender._transfer_kv_cache(task)
        self.assertEqual(observed, ["source"])

    def test_failed_legacy_fence_prevents_submission(self):
        sender, task, observed = self._sender_and_task("legacy")
        with self.assertRaises(TimeoutError):
            sender._transfer_kv_cache(task)
        self.assertEqual(observed, ["source", "legacy"])


class TestTargetHostRegistration(unittest.TestCase):
    def test_registers_existing_host_main_and_device_indexer_in_distinct_groups(self):
        # Real CPU storage verifies pointer selection; no NPU allocation or
        # native registration is performed by the recording TE below.
        host = lc.torch.empty(1024, dtype=lc.torch.int8, device="cpu")
        host_views = (
            host[:256].view(lc.torch.bfloat16).view(4, 16, 1, 2),
            host[256:512].view(lc.torch.bfloat16).view(4, 16, 1, 2),
        )
        main = "model.layers.0.self_attn.attn"
        indexer = "model.layers.0.self_attn.indexer"
        hbm_main = tuple(lc.torch.empty_like(tensor) for tensor in host_views)
        hbm_indexer = (
            lc.torch.empty(8, 8, 1, 2, dtype=lc.torch.int8, device="cpu"),
            lc.torch.empty(8, 8, 1, 1, dtype=lc.torch.float16, device="cpu"),
        )
        original = {main: hbm_main, indexer: hbm_indexer}
        effective = {main: host_views, indexer: hbm_indexer}
        pool = SimpleNamespace(data_ptr=host.data_ptr(), nbytes=host.numel(), topology=SimpleNamespace(device_id=0))
        manager = SimpleNamespace(
            get_target_transfer_caches=Mock(return_value=effective),
            get_target_cache_identities=lambda: ({main: lc.Component.MAIN, indexer: lc.Component.INDEXER}, {}, {}),
            get_mooncake_host_pool=lambda: pool,
        )
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.target_owner_manager = manager
        spec = SimpleNamespace()
        worker.kv_cache_config = SimpleNamespace(
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[
                SimpleNamespace(layer_names=[main], kv_cache_spec=spec),
                SimpleNamespace(layer_names=[indexer], kv_cache_spec=spec),
            ],
        )
        worker.kv_cache_specs = [spec, spec]
        worker.use_attn_mamba_hybrid = False
        worker.pd_head_ratio = 1
        worker.enable_kv_quant = False
        worker.enable_c8_quant = False
        worker.sparse_sfa_c8_layers = set()
        worker.attn_resharding_group_idx = set()
        worker.kernel_block_size_scale = [1, 1]
        worker.layer_metadata = {}
        worker.index_to_name = lc.defaultdict(list)
        worker.total_layers = 1
        worker.te_rpc_port = 1
        worker._init_layerwise_reuse_state = Mock()
        worker._run_startup_validations = Mock()
        worker.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
            kv_transfer_config=SimpleNamespace(is_kv_producer=False, is_kv_consumer=False),
        )
        te = SimpleNamespace(register_buffer=Mock())
        with patch.object(lc, "global_te", te):
            worker.register_kv_caches(original)
        pointers, lengths, locations = te.register_buffer.call_args.args
        self.assertEqual((pointers[0], lengths[0], locations[0]), (host.data_ptr(), 512, "npu:0"))
        self.assertEqual(set(pointers[1:]), {t.untyped_storage().data_ptr() for t in hbm_indexer})
        self.assertTrue(set(pointers).isdisjoint(t.data_ptr() for t in hbm_main))
        self.assertEqual(worker.layer_metadata[main].kv_caches_base_addr, [t.data_ptr() for t in host_views])
        self.assertEqual(worker.layer_metadata[indexer].block_size_scale, [2, 2])
        manager.get_target_transfer_caches.assert_called_once_with(original)


class TestWireFormat:
    """Positional arrays only; the decoder is Decoder(type=tuple),
    so a map-encoded message must fail loudly instead of binding incorrectly."""

    def test_layer_result_roundtrip(self):
        msg = (
            lc.LAYER_RESULT_MSG,
            lc.CONTRACT_VERSION,
            "instance-1",
            0,
            2,
            3,
            3,
            "main",
            0,
            8,
            "b0",
            1,
            "success",
            True,
        )
        assert _encode_decode(msg) == msg

    def test_rank_terminal_roundtrip(self):
        msg = (lc.RANK_TERMINAL_MSG, lc.CONTRACT_VERSION, "instance-1", 0, 2, "success")
        assert _encode_decode(msg) == msg

    def test_map_encoded_message_is_rejected(self):
        import msgspec as msgspec_mod

        encoder = msgspec_mod.msgpack.Encoder()
        decoder = msgspec.msgpack.Decoder(type=tuple)
        with pytest.raises(msgspec.ValidationError):
            decoder.decode(encoder.encode({"type": lc.LAYER_RESULT_MSG, "version": lc.CONTRACT_VERSION}))


class TestReceiptHandler:
    """Drive the receive-thread handlers directly (no zmq): malformed frames
    are logged and dropped, version mismatches never reach the ledger."""

    def _thread(self):
        thread = object.__new__(lc.KVCacheRecvingLayerThread)
        import threading

        thread.lock = threading.Lock()
        thread.done_requests = set()
        thread.failed_requests = set()
        thread.task_tracker = {}
        thread.ledger_by_req = {}
        thread.instance_to_req = {}
        thread.receipt_provider = None
        return thread

    def test_version_mismatch_is_dropped(self):
        thread = self._thread()
        msg = (lc.LAYER_RESULT_MSG, 999, "inst", 0, 0, 0, 0, "main", 0, 8, "b0", 1, "success", True)
        thread.handle_layer_result(msg)  # must not raise
        assert not thread.ledger_by_req

    def test_malformed_frame_is_dropped(self):
        thread = self._thread()
        thread.handle_layer_result((lc.LAYER_RESULT_MSG, lc.CONTRACT_VERSION, "inst"))  # too short
        thread.handle_rank_terminal((lc.RANK_TERMINAL_MSG, lc.CONTRACT_VERSION))  # too short

    def test_unknown_instance_is_dropped(self):
        thread = self._thread()
        msg = (lc.LAYER_RESULT_MSG, lc.CONTRACT_VERSION, "ghost", 0, 0, 0, 0, "main", 0, 8, "b0", 1, "success", True)
        thread.handle_layer_result(msg)
        assert not thread.done_requests


class TestPrefixHitNoCompute:
    """Prefix-hit a layer whose requests are all fully prefix-hit has
    no compute reader — it must be marked no_compute_reader and must not hang
    the send path waiting on a compute-done event that will never come."""

    def test_no_compute_layer_is_marked(self):
        pytest.importorskip("vllm")
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)

        worker.vllm_config = None  # set below via a stub config
        worker.current_layer = 0
        worker.total_layers = 4
        worker.index_to_name = {0: ["layer0"]}
        worker.layer_metadata = {}
        worker._slot_ledger = None
        worker._slot_ledger_active = False
        worker._layer_keys = {}
        worker._compute_done_events = [None] * 4
        worker._no_compute_layers = set()
        worker._cache_write_events = [None] * 4
        worker.kv_send_layer_thread = None
        worker._reqs_need_send_layerwise = {}
        worker.use_mla = False
        worker.kv_cache_specs = []
        worker.block_size = [16]
        worker.sparse_sfa_c8_layers = set()
        worker.enable_kv_quant = False
        worker.enable_c8_quant = False
        worker.pd_head_ratio = 1
        worker.resharding_stream = None

        class _Cfg:
            class kv_transfer_config:  # noqa: N801
                is_kv_producer = True
                is_kv_consumer = False

        class _Meta:
            pass

        req = lc.ReqMeta(
            local_block_ids=[[0, 1]],
            token_ids=[],
            remote_block_ids=[],
            remote_block_size=[],
            remote_engine_id=None,
            remote_host=None,
            remote_port=None,
            remote_te_rpc_port=None,
            remote_layer_metadata=None,
            metaserver=None,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
            prompt_len=8,
            local_computed_tokens=8,  # fully prefix-hit
        )
        metadata = lc.MooncakeLayerwiseConnectorMetadata()
        metadata.requests["req-1"] = req
        worker.vllm_config = _Cfg()
        worker.save_kv_layer("layer0", [], None, metadata)
        assert 0 in worker._no_compute_layers
        assert worker._compute_done_events[0] is None  # no compute-done event recorded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

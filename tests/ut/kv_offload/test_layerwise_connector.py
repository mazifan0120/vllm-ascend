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


def make_receiver_adapter():
    adapter = SimpleNamespace(receipt_timeout_s=lc.RECEIPT_WATCHDOG_SECONDS)
    adapter.receipts = lc.ReceiptReceiver(lc.RECEIPT_WATCHDOG_SECONDS, lc.logger, status=adapter)
    adapter.lock = adapter.receipts.lock
    return adapter


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


# Protocol parsing/lifecycle cases live in the CPU suite; use its fixture here
# for tests which specifically exercise the real NPU connector adapter.
from tests.ut.distributed.ascend_store.test_target_receipt_receiver import (  # noqa: E402
    TestExactReceiverProtocol as ReceiverProtocolFixture,
)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "7200", None])
def test_receipt_timeout_rejects_invalid_values(timeout):
    with pytest.raises(ValueError, match="layerwise_receipt_timeout_s"):
        lc.KVCacheRecvingLayerThread(
            0, 57000, 1, 1, "engine", SimpleNamespace(), threading.Event(), receipt_timeout_s=timeout
        )


@pytest.mark.parametrize("timeout", [lc.RECEIPT_WATCHDOG_SECONDS, 7200.0])
def test_admitted_target_uses_configured_receipt_deadline(timeout):
    with patch.object(lc, "get_ip", return_value="127.0.0.1"):
        receiver = lc.KVCacheRecvingLayerThread(
            0, 57000, 1, 1, "engine", SimpleNamespace(), threading.Event(), receipt_timeout_s=timeout
        )
    worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
    worker.request_map = {}
    worker.kv_recv_layer_thread = receiver
    _, ledger, _ = ReceiverProtocolFixture().fixture()
    request_id = "long-prefill-12345678"
    lease = SimpleNamespace(request=ledger.request, allocation=SimpleNamespace(group_block_ids=((1, 3),)))
    with patch.object(lc.time, "monotonic", return_value=100):
        worker._install_admitted_target(request_id, lease, ledger)
    assert receiver.receipts.requests.get(ledger.request.request_instance).deadline == 100 + timeout
    with patch.object(lc.time, "monotonic", return_value=401):
        receiver._sweep_receipt_deadlines()
    assert bool(receiver.failed_requests) == (timeout <= 301)
    with patch.object(lc.time, "monotonic", return_value=101 + timeout):
        receiver._sweep_receipt_deadlines()
    assert receiver.failed_requests == {request_id}
    assert not receiver.done_requests


class TestEarlyTargetAdmission(unittest.TestCase):
    def test_actual_install_maps_early_completions_before_scheduler_metadata(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.request_map = {}
        worker.kv_recv_layer_thread = make_receiver_adapter()
        _, ledger, _ = ReceiverProtocolFixture().fixture()
        request_id = "early-request-12345678"
        lease = SimpleNamespace(request=ledger.request, allocation=SimpleNamespace(group_block_ids=((1, 3),)))
        worker._install_admitted_target(request_id, lease, ledger)
        external = lc.get_external_request_id(request_id)
        self.assertNotIn(request_id, worker.request_map)
        self.assertNotIn(external, worker.request_map)
        self.assertIs(worker.kv_recv_layer_thread.receipts.requests.get(ledger.request.request_instance).ledger, ledger)
        self.assertEqual(worker.kv_recv_layer_thread.receipts.requests.retained_blocks(request_id), ((1, 3),))
        with self.assertRaises(lc.ContractError):
            worker._install_admitted_target(request_id, lease, ledger)

    def test_reused_external_id_keeps_both_engine_lifetimes_until_their_own_retirement(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.request_map = {}
        worker.kv_recv_layer_thread = make_receiver_adapter()
        instances = []
        for suffix in ("12345678", "abcdefgh"):
            _, ledger, _ = ReceiverProtocolFixture().fixture()
            ledger.request = lc.RequestKey(suffix, 0)
            request_id = "same-external-" + suffix
            lease = SimpleNamespace(request=ledger.request, allocation=SimpleNamespace(group_block_ids=((1, 3),)))
            worker._install_admitted_target(request_id, lease, ledger)
            instances.append((request_id, lease, ledger))
        self.assertFalse(worker.request_map)
        worker._retire_admitted_target(*instances[0])
        request_id, lease, ledger = instances[1]
        self.assertFalse(worker.request_map)
        self.assertIs(worker.kv_recv_layer_thread.receipts.requests.get(lease.request.request_instance).ledger, ledger)
        self.assertTrue(worker.kv_recv_layer_thread.receipts.requests.in_receipt_mode("same-external"))
        worker._retire_admitted_target(*instances[1])
        self.assertFalse(worker.request_map)
        self.assertFalse(worker.kv_recv_layer_thread.receipts.requests.in_receipt_mode("same-external"))


class TestReceiptFailureIdentity(unittest.TestCase):
    def test_actual_transfer_results_report_receipt_failure_by_request_without_numeric_block_invalidation(self):
        worker = object.__new__(lc.MooncakeLayerwiseConnectorWorker)
        worker.vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_consumer=True, is_kv_producer=False)
        )
        failed = "failed-request-12345678"
        other = "ordinary-request-abcdefgh"
        worker.request_map = {"ordinary-request": other}
        worker._invalid_block_ids = set()
        worker._failed_receives = set()
        worker._recving_metadata = {other: SimpleNamespace(local_block_ids=[[1, 3], [1, 3]])}
        worker.virtual_request = set()
        worker.kv_recv_layer_thread = make_receiver_adapter()
        worker.kv_recv_layer_thread.get_and_clear_done_requests = lambda: set()
        worker.kv_recv_layer_thread.get_and_clear_failed_requests = lambda: {failed}
        worker.kv_recv_layer_thread.forget_request = Mock()
        _, ledger, _ = ReceiverProtocolFixture().fixture()
        ledger.lease = SimpleNamespace(allocation=SimpleNamespace(group_block_ids=((1, 3), (1, 3))))
        worker.kv_recv_layer_thread.receipts.requests.install(failed, "failed-request", ledger, 0)
        connector = object.__new__(lc.MooncakeLayerwiseConnector)
        connector.connector_worker = worker
        result = connector.get_transfer_results(set())
        self.assertEqual(result.failed_recving, {failed})
        self.assertEqual(result.finished_recving, {failed})
        self.assertFalse(worker.get_block_ids_with_load_errors())
        self.assertEqual(worker.request_map, {"ordinary-request": other})
        self.assertTrue(worker._has_admitted_target(failed))
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
        worker.vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_producer=True),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
        )
        worker.use_mla = True
        worker.total_layers = 1
        worker.pcp_size = worker.dcp_size = worker.pd_head_ratio = 1
        worker.tp_size = 2
        worker.enable_c8_quant = worker.enable_kv_quant = False
        worker._source_contract_bound = True
        worker._manifest_source_writer = SimpleNamespace(register_request=Mock(), prepare_job=Mock(return_value="job"))
        worker.attn_resharding_group_idx = set()
        worker._align_remote_block_ids = Mock(side_effect=AssertionError("manifest must retain manager block IDs"))
        target = SimpleNamespace(
            targets=(object(), object()), lease=SimpleNamespace(allocation=SimpleNamespace(prompt_tokens=23))
        )
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

    def test_real_prepare_accepts_replicated_glm_p8_d4(self):
        worker, metadata = self.worker()
        worker.tp_size = 8
        meta = metadata.requests["internal-source"]
        meta.remote_tp_size = 4
        meta.remote_target_manifest.targets = tuple(range(4))
        worker.prepare_layerwise_step(metadata)
        worker._manifest_source_writer.register_request.assert_called_once_with("internal-source", meta)

    def test_real_prepare_rejects_unsupported_unequal_tp_before_registering_sources(self):
        for change in ("reverse", "nondivisible", "model", "mla", "pcp", "dcp", "quant", "manifest"):
            worker, metadata = self.worker()
            worker.tp_size = 8
            meta = metadata.requests["internal-source"]
            meta.remote_tp_size = 4
            meta.remote_target_manifest.targets = tuple(range(4))
            if change == "reverse":
                worker.tp_size = 2
            elif change == "nondivisible":
                worker.tp_size = 6
            elif change == "model":
                worker.vllm_config.model_config.hf_text_config.model_type = "deepseek_v3"
            elif change == "mla":
                worker.use_mla = False
            elif change == "pcp":
                worker.pcp_size = 2
            elif change == "dcp":
                meta.remote_dcp_size = 2
            elif change == "quant":
                worker.enable_c8_quant = True
            else:
                meta.remote_tp_size = 2
            with self.subTest(change=change), self.assertRaises(lc.ContractError):
                worker.prepare_layerwise_step(metadata)
            worker._manifest_source_writer.register_request.assert_not_called()

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
        thread.done_requests = set()
        thread.failed_requests = set()
        thread.task_tracker = {}
        thread.receipt_provider = None
        thread.receipts = lc.ReceiptReceiver(lc.RECEIPT_WATCHDOG_SECONDS, lc.logger, status=thread)
        thread.lock = thread.receipts.lock
        return thread

    def test_version_mismatch_is_dropped(self):
        thread = self._thread()
        msg = (lc.LAYER_RESULT_MSG, 999, "inst", 0, 0, 0, 0, "main", 0, 8, "b0", 1, "success", True)
        thread.handle_layer_result(msg)  # must not raise
        assert thread.receipts.requests.get("inst") is None

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
        worker.layer_metadata = {"layer0": SimpleNamespace(tensor_group_idx=(0,))}
        worker._slot_ledger = None
        worker._slot_ledger_active = False
        worker._layer_keys = {}
        worker._compute_done_events = [None] * 4
        worker._no_compute_layers = set()
        worker._cache_write_events = [None] * 4
        worker.kv_send_layer_thread = Mock()
        worker._mark_layer_reuse_pending = Mock()
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
        worker.update_decoder_info = Mock(return_value=req)
        worker.save_kv_layer("layer0", [], None, metadata)
        assert 0 in worker._no_compute_layers
        assert worker._compute_done_events[0] is None  # no compute-done event recorded
        worker.kv_send_layer_thread.send_queue.put.assert_called_once()
        worker._mark_layer_reuse_pending.assert_called_once_with(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

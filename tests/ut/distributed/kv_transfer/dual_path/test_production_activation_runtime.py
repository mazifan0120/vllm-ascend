# SPDX-License-Identifier: Apache-2.0
# allow: SIZE_OK - Production Scheduler-to-Worker lifecycle acceptance is one narrative.

from concurrent.futures import Future
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path import test_forward_receive_binding as decode_helpers
from tests.ut.distributed.kv_transfer.dual_path import test_pe_read_forward as prefill_helpers
from tests.ut.distributed.kv_transfer.dual_path import test_split_integration as split_helpers
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import DualPathConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionRequest,
    PathKind,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision_channel import DecodeControlEndpoint
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnectorMetadata,
    MooncakeLayerwiseConnectorScheduler,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    ReqMeta as LayerwiseReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    AscendConnectorMetadata,
    LoadSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    ReqMeta as StoreReqMeta,
)

CONNECTOR_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector"
SCHEDULER_NS = "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler"
DECODE_REQUEST_ID = "request-local-7"
PREFILL_REQUEST_ID = "reques123456789"
L_DE = 16
K_DE = 32
L_PE = 16
R = 48
T = 49
DE_BLOCKS = (41, 42, 43, 44)
PE_BLOCKS = (70, 71, 72, 73)


def _completed_future() -> Future[None]:
    future: Future[None] = Future()
    future.set_result(None)
    return future


def _empty_scheduler_output() -> SimpleNamespace:
    return SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[], num_computed_tokens=[]),
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
        num_scheduled_tokens={},
    )


def _store_metadata() -> AscendConnectorMetadata:
    metadata = AscendConnectorMetadata(set(), set(), loading_req_ids={DECODE_REQUEST_ID})
    metadata.add_request(
        StoreReqMeta(
            req_id=DECODE_REQUEST_ID,
            token_len_chunk=T,
            block_ids=list(DE_BLOCKS),
            block_hashes=[bytes([block_id]) for block_id in range(len(DE_BLOCKS))],
            load_spec=LoadSpec(
                vllm_cached_tokens=L_DE,
                kvpool_cached_tokens=K_DE,
                can_load=True,
                token_len=T,
            ),
        )
    )
    return metadata


@dataclass(slots=True)
class ProductionHarness:
    decode_scheduler: connector_module.DualPathConnectorScheduler
    prefill_scheduler: connector_module.DualPathConnectorScheduler
    decode_coordinator: MagicMock
    prefill_coordinator: MagicMock
    decode_request: SimpleNamespace
    prefill_request: SimpleNamespace
    de_worker: connector_module.DualPathConnectorWorker
    pe_worker: connector_module.DualPathConnectorWorker
    de_metadata: DualPathConnectorMetadata
    pe_metadata: MooncakeLayerwiseConnectorMetadata

    @property
    def wire_request_id(self) -> str:
        return self.de_metadata.forward_receive_bindings[0].wire_request_id

    def finish_store(self, *, failed: bool = False) -> tuple[set[str], set[str]]:
        self.de_worker._kvpool_worker_adapter.get_finished.return_value = (set(), {DECODE_REQUEST_ID})
        self.de_worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {42} if failed else set()
        result = self.de_worker.get_finished(set(), self.de_metadata)
        self.de_worker._kvpool_worker_adapter.get_finished.return_value = (set(), set())
        self.de_worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = set()
        return result

    def finish_reverse(self, *, failed: bool = False) -> tuple[tuple[set[str], set[str]], tuple[set[str], set[str]]]:
        with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
            self.de_worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, trans_flag=not failed)
        self.pe_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
            set() if failed else {self.wire_request_id}
        )
        self.pe_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
            {self.wire_request_id} if failed else set()
        )
        de_result = self.de_worker.get_finished(set(), self.de_metadata)
        pe_result = self.pe_worker.get_finished(set(), self.pe_metadata)
        self.pe_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
        self.pe_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
        return de_result, pe_result

    def run_forward(self) -> MooncakeLayerwiseConnectorMetadata:
        metadata = DualPathConnectorMetadata()
        metadata.requests[PREFILL_REQUEST_ID] = LayerwiseReqMeta(
            local_block_ids=[list(PE_BLOCKS)],
            token_ids=None,
            remote_block_ids=[list(DE_BLOCKS)],
            remote_block_size=[16],
            remote_engine_id="decode-engine",
            remote_host="198.51.100.20",
            remote_port=7000,
            remote_te_rpc_port=None,
            remote_layer_metadata=None,
            metaserver=None,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
            chunk_finish=True,
            prompt_len=T,
            trans_count=[],
            local_computed_tokens=T,
            local_transed_tokens=K_DE,
        )
        self.pe_worker.start_load_kv(metadata)
        events = {
            name: SimpleNamespace(reshape_cache_event=MagicMock(name=f"{name}_event"))
            for name in split_helpers.LAYER_NAMES
        }
        for _ in split_helpers.LAYER_NAMES:
            self.pe_worker.save_kv_layer("", [MagicMock(), MagicMock()], events, metadata)
        return metadata

    def finish_forward(self, *, failed: bool = False) -> tuple[set[str], set[str]]:
        self.de_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
            set() if failed else {self.wire_request_id}
        )
        self.de_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
            {self.wire_request_id} if failed else set()
        )
        result = self.de_worker.get_finished(set(), self.de_metadata)
        self.de_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
        self.de_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()
        return result


@pytest.fixture()
def production_harness_factory(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT", raising=False)
    schedulers = []
    with (
        patch(f"{SCHEDULER_NS}.KVPoolSchedulerAdapter"),
        patch(f"{SCHEDULER_NS}.PathDecisionCoordinator") as coordinator_cls,
        patch(f"{SCHEDULER_NS}.get_ip", return_value="192.0.2.44"),
    ):

        def make(*, include_store: bool = True, path: PathKind = PathKind.DE_READ) -> ProductionHarness:
            decode_coordinator = MagicMock(name="decode_coordinator")
            decode_coordinator.decode_engine_instance_id = decode_helpers._DECODE_INSTANCE_ID
            decode_coordinator.decode_control_endpoint = DecodeControlEndpoint(host="192.0.2.44", port=24001)
            decode_coordinator.take_received_decisions.return_value = []
            prefill_coordinator = MagicMock(name="prefill_coordinator")
            prefill_coordinator.submit.return_value = _completed_future()
            coordinator_cls.for_decode.return_value = decode_coordinator
            coordinator_cls.for_prefill.return_value = prefill_coordinator

            decode_scheduler = connector_module.DualPathConnectorScheduler(
                decode_helpers._make_vllm_config(),
                decode_helpers._make_kv_cache_config(),
                "decode-engine",
                DualPathConfig(role="decode", dual_path_control_port=7100),
            )
            policy = prefill_helpers.FixedPathPolicy(path)
            prefill_scheduler = connector_module.DualPathConnectorScheduler(
                prefill_helpers._make_vllm_config(),
                prefill_helpers._make_kv_cache_config(),
                "prefill-engine",
                DualPathConfig(role="prefill"),
                path_policy=policy,
            )
            for scheduler in (decode_scheduler, prefill_scheduler):
                scheduler.executor.shutdown(wait=False)
                scheduler.metaserver_client.close()
                scheduler.executor = MagicMock(name=f"{scheduler.dual_path_cfg.role}_executor")
                scheduler.side_channel_host = "198.51.100.20"
            decode_scheduler._kvpool_adapter.build_connector_meta.return_value = (
                _store_metadata()
                if include_store and path is PathKind.DE_READ
                else AscendConnectorMetadata(set(), set())
            )
            schedulers.extend((decode_scheduler, prefill_scheduler))

            decode_request, snapshot, state = decode_helpers._admit_request(decode_scheduler)
            if not include_store:
                snapshot = replace(snapshot, store_load_spec=None)
                decode_scheduler._decode_kv_snapshots[DECODE_REQUEST_ID] = snapshot
                state.decision_request = PathDecisionRequest(
                    request_key=state.request_key,
                    target_tokens=R,
                    decode_local_tokens=L_DE,
                    decode_store_tokens=L_DE,
                )

            prefill_request = prefill_helpers._make_request(
                request_id=PREFILL_REQUEST_ID,
                target_tokens=R,
                prompt_tokens=T,
                local_tokens=L_DE,
                store_tokens=K_DE if include_store else L_DE,
                destination_block_ids=[list(DE_BLOCKS)],
            )
            decision_request = prefill_request.kv_transfer_params["dual_path"]["decision_request"]
            decision_request["request_key"]["decode_request_id"] = DECODE_REQUEST_ID
            prefill_local_tokens = L_PE if include_store else 0
            expected_match = (
                ((K_DE if include_store else L_DE) - prefill_local_tokens, True)
                if path is PathKind.DE_READ
                else (0, False)
            )
            assert prefill_scheduler.get_num_new_matched_tokens(prefill_request, prefill_local_tokens) == expected_match
            assert policy.calls == 1
            prefill_scheduler.update_state_after_alloc(
                prefill_request,
                prefill_helpers._blocks((list(PE_BLOCKS),)),
                (K_DE if include_store else L_DE) - prefill_local_tokens,
            )
            decision = prefill_coordinator.submit.call_args.args[1]
            with patch.object(
                MooncakeLayerwiseConnectorScheduler,
                "build_connector_meta",
                autospec=True,
                return_value=MooncakeLayerwiseConnectorMetadata(),
            ):
                pe_metadata = prefill_scheduler.build_connector_meta(_empty_scheduler_output())

            decode_coordinator.take_received_decisions.return_value = [decision]
            de_metadata = decode_scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
            assert isinstance(de_metadata, DualPathConnectorMetadata)

            workers = split_helpers.SplitHarness.make()
            reverse_metadata = MooncakeLayerwiseConnectorMetadata()
            reverse_metadata.requests[DECODE_REQUEST_ID] = MagicMock(chunk_finish=True)
            workers.de_worker._build_reverse_send_metadata = MagicMock(return_value=reverse_metadata)
            workers.pe_worker.start_load_kv(pe_metadata)
            workers.de_worker.start_load_kv(de_metadata)
            return ProductionHarness(
                decode_scheduler=decode_scheduler,
                prefill_scheduler=prefill_scheduler,
                decode_coordinator=decode_coordinator,
                prefill_coordinator=prefill_coordinator,
                decode_request=decode_request,
                prefill_request=prefill_request,
                de_worker=workers.de_worker,
                pe_worker=workers.pe_worker,
                de_metadata=de_metadata,
                pe_metadata=pe_metadata,
            )

        yield make

    for scheduler in schedulers:
        scheduler.shutdown()


def test_partial_store_completes_before_reverse_submission(production_harness_factory) -> None:
    harness = production_harness_factory()
    tracker = harness.de_worker._split_trackers[DECODE_REQUEST_ID]
    assert tracker.store_phase.value == "PENDING"
    assert tracker.reverse_submitted is False

    assert harness.finish_store() == (set(), set())

    assert tracker.store_phase.value == "DONE"
    assert tracker.reverse_submitted is True
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == len(split_helpers.LAYER_NAMES)


def test_store_miss_submits_reverse_after_mapping_install(production_harness_factory) -> None:
    harness = production_harness_factory(include_store=False)
    tracker = harness.de_worker._split_trackers[DECODE_REQUEST_ID]

    assert harness.de_worker.request_map[harness.wire_request_id] == DECODE_REQUEST_ID
    assert tracker.store_phase.value == "SKIPPED"
    assert tracker.plan is harness.de_metadata.reverse_plans[0]
    assert tracker.reverse_submitted is True


def test_pe_execution_blocked_until_final_reverse_done(production_harness_factory) -> None:
    harness = production_harness_factory(include_store=False)

    assert harness.pe_worker.get_finished(set(), harness.pe_metadata) == (set(), set())
    assert harness.pe_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    _, pe_finished = harness.finish_reverse()

    assert pe_finished == (set(), {PREFILL_REQUEST_ID})


def test_pe_uses_inherited_layerwise_forward_after_reverse(production_harness_factory) -> None:
    harness = production_harness_factory(include_store=False)
    harness.finish_reverse()

    metadata = harness.run_forward()

    tasks = [call.args[0] for call in harness.pe_worker.kv_send_layer_thread.send_queue.put.call_args_list]
    assert metadata.requests[PREFILL_REQUEST_ID].local_transed_tokens == K_DE
    assert [task.layer_name for task in tasks] == list(split_helpers.LAYER_NAMES)


def test_de_does_not_complete_on_single_phase_terminals(production_harness_factory) -> None:
    harness = production_harness_factory()

    assert harness.finish_store() == (set(), set())
    assert harness.finish_reverse()[0] == (set(), set())
    assert harness.de_worker._split_trackers[DECODE_REQUEST_ID].terminal_published is False


def test_de_completes_exactly_once_after_full_predicate(production_harness_factory) -> None:
    harness = production_harness_factory()
    harness.finish_store()
    harness.finish_reverse()
    harness.run_forward()

    assert harness.finish_forward() == (set(), {DECODE_REQUEST_ID})
    assert harness.de_worker.get_finished(set(), harness.de_metadata) == (set(), set())


@pytest.mark.parametrize(
    ("source", "expected_invalid"),
    [("store", {42}), ("reverse", {43, 44}), ("forward", {43, 44})],
)
def test_production_metadata_failure_provenance_is_exact(
    production_harness_factory,
    source: str,
    expected_invalid: set[int],
) -> None:
    harness = production_harness_factory()
    if source == "store":
        assert harness.finish_store(failed=True) == (set(), {DECODE_REQUEST_ID})
    else:
        harness.finish_store()
        if source == "reverse":
            assert harness.finish_reverse(failed=True)[0] == (set(), {DECODE_REQUEST_ID})
        else:
            harness.finish_reverse()
            assert harness.finish_forward(failed=True) == (set(), {DECODE_REQUEST_ID})

    assert harness.de_worker.get_block_ids_with_load_errors() == expected_invalid


@pytest.mark.parametrize("failed", [False, True])
def test_production_metadata_reconciles_early_terminals(production_harness_factory, failed: bool) -> None:
    harness = production_harness_factory(include_store=False)
    early = production_harness_factory(include_store=False)
    early.pe_worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
        set() if failed else {harness.wire_request_id}
    )
    early.pe_worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
        {harness.wire_request_id} if failed else set()
    )
    early.pe_worker._reverse_receive_bindings.clear()
    early.pe_worker._reverse_request_map.clear()
    assert early.pe_worker.get_finished(set(), DualPathConnectorMetadata()) == (set(), set())

    early.pe_worker.start_load_kv(harness.pe_metadata)

    assert early.pe_worker.get_finished(set(), harness.pe_metadata) == (set(), {PREFILL_REQUEST_ID})


def test_production_metadata_failed_wins_over_late_done(production_harness_factory) -> None:
    harness = production_harness_factory()
    harness.finish_store()
    assert harness.finish_reverse(failed=True)[0] == (set(), {DECODE_REQUEST_ID})

    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        harness.de_worker.send_done_send_signal(DECODE_REQUEST_ID, MagicMock(), 0, trans_flag=True)

    assert harness.de_worker.get_finished(set(), harness.de_metadata) == (set(), set())
    assert harness.de_worker._split_trackers[DECODE_REQUEST_ID].reverse_phase.value == "FAILED"


def test_request_finish_releases_all_task08_state_idempotently(production_harness_factory) -> None:
    harness = production_harness_factory()
    unrelated_key = DualPathRequestKey("decode-instance", "unrelated")
    harness.prefill_scheduler._pe_prefill_local_tokens["unrelated"] = 7
    harness.prefill_scheduler._path_decider._decision_records[unrelated_key] = MagicMock()

    with patch.object(
        MooncakeLayerwiseConnectorScheduler,
        "request_finished",
        autospec=True,
        return_value=(True, None),
    ):
        harness.prefill_scheduler.request_finished(harness.prefill_request, [1])
        harness.prefill_scheduler.request_finished(harness.prefill_request, [1])
        harness.decode_scheduler.request_finished(harness.decode_request, [1])
        harness.decode_scheduler.request_finished(harness.decode_request, [1])
    assert harness.de_worker.get_finished({DECODE_REQUEST_ID}, harness.de_metadata) == (set(), set())
    assert harness.de_worker.get_finished({DECODE_REQUEST_ID}, harness.de_metadata) == (set(), set())
    assert harness.pe_worker.get_finished({PREFILL_REQUEST_ID}, harness.pe_metadata) == (set(), set())
    assert harness.pe_worker.get_finished({PREFILL_REQUEST_ID}, harness.pe_metadata) == (set(), set())

    assert harness.prefill_scheduler._pe_prefill_local_tokens == {"unrelated": 7}
    assert harness.prefill_scheduler._pe_pending_reverse_receive_bindings == {}
    assert harness.prefill_scheduler._pe_control_failures == {}
    assert harness.prefill_scheduler._pe_delivery_futures == {}
    assert set(harness.prefill_scheduler._path_decider._decision_records) == {unrelated_key}
    assert harness.decode_scheduler._decode_kv_snapshots == {}
    assert harness.decode_scheduler._decode_decision_states == {}
    assert harness.de_worker._split_trackers == {}
    assert harness.de_worker._forward_receive_bindings == {}
    assert harness.pe_worker._reverse_receive_bindings == {}
    assert harness.pe_worker._reverse_request_map == {}


def test_shutdown_leaves_no_task08_residue(production_harness_factory) -> None:
    harness = production_harness_factory()

    harness.prefill_scheduler.shutdown()
    harness.decode_scheduler.shutdown()
    harness.pe_worker.shutdown()
    harness.de_worker.shutdown()

    assert harness.prefill_scheduler._pe_prefill_local_tokens == {}
    assert harness.prefill_scheduler._pe_pending_reverse_receive_bindings == {}
    assert harness.prefill_scheduler._pe_control_failures == {}
    assert harness.prefill_scheduler._pe_delivery_futures == {}
    assert harness.prefill_scheduler._path_decider._decision_records == {}
    assert harness.decode_scheduler._decode_kv_snapshots == {}
    assert harness.decode_scheduler._decode_decision_states == {}
    assert harness.decode_coordinator.close.call_count == 1
    assert harness.prefill_coordinator.close.call_count == 1
    assert harness.de_worker._split_trackers == {}
    assert harness.pe_worker._reverse_receive_bindings == {}


def test_structured_logs_reconstruct_request_facts(production_harness_factory) -> None:
    with (
        patch.object(scheduler_module.logger, "info") as info,
        patch.object(scheduler_module.logger, "warning") as warning,
        patch.object(scheduler_module.logger, "error") as error,
    ):
        harness = production_harness_factory()
        harness.finish_store()
        harness.finish_reverse()
        harness.finish_forward()
    log_calls = (*info.call_args_list, *warning.call_args_list, *error.call_args_list)
    messages = "\n".join(call.args[0] % call.args[1:] for call in log_calls)

    assert "key=decode-engine:2:boot-7/request-local-7" in messages
    assert (
        "decode_local_tokens=16 decode_store_tokens=32 prefill_local_tokens=16 "
        "decode_ready_tokens=48 target_tokens=49" in messages
    )
    assert "eligibility=policy selected_path=DE_READ store=partial" in messages
    assert "store_range=[16,32) reverse_range=[16,32) forward_range=[32,49)" in messages
    assert "delivery_terminal=SUCCEEDED" in messages
    assert "final_predicate=SUCCESS" in messages
    assert "PathDecision" not in messages


def test_pe_read_logs_store_as_not_authorized(production_harness_factory) -> None:
    with patch.object(scheduler_module.logger, "info") as info:
        production_harness_factory(path=PathKind.PE_READ)
    messages = "\n".join(call.args[0] % call.args[1:] for call in info.call_args_list)

    assert "selected_path=PE_READ store=none" in messages
    assert "store_range=[16,16)" in messages
    assert "selected_path=PE_READ store=partial" not in messages

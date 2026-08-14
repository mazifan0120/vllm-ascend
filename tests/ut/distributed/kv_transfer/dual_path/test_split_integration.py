# SPDX-License-Identifier: Apache-2.0
# allow: SIZE_OK - The injected lifecycle belongs in one narrative test module.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path import test_split_lifecycle as lifecycle
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_layerwise_connector as layerwise_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import DualPathConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import DualPathConnectorMetadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    LayerMetadata,
    MooncakeLayerwiseConnectorMetadata,
    ReqMeta,
)

BLOCK_SIZE = 16
L_DE = 32
K_DE = 64
L_PE = 16
TARGET_TOKENS = 128
PREFILL_REQUEST_ID = f"{lifecycle.WIRE_REQUEST_ID}prefill-local"
LAYER_NAMES = ("model.layer.0", "model.layer.1")
PE_BLOCKS = (70, 71, 80, 81, 90, 91, 100, 101)
FORWARD_DE_BLOCKS = {30, 31, 40, 41}
REVERSE_PE_BLOCKS = {71, 80, 81}


@dataclass(frozen=True, slots=True)
class SplitHarness:
    de_worker: DualPathConnectorWorker
    pe_worker: DualPathConnectorWorker

    @classmethod
    def make(cls) -> "SplitHarness":
        de_worker = lifecycle._make_worker()
        pe_worker = lifecycle._make_prefill_worker()
        for worker in (de_worker, pe_worker):
            worker.pd_head_ratio = 1
            worker.enable_kv_quant = False
            worker.enable_c8_quant = False
            worker.total_layers = len(LAYER_NAMES)
            worker.current_layer = 0
            worker.index_to_name = {index: [name] for index, name in enumerate(LAYER_NAMES)}
            worker.layer_metadata = {name: LayerMetadata([0], [0], [BLOCK_SIZE], [1]) for name in LAYER_NAMES}
            worker.kv_send_layer_thread = MagicMock(name=f"{worker.dual_path_cfg.role}_send_thread")
            worker.update_decoder_info = MagicMock(side_effect=lambda _request_id, req_meta: req_meta)

        lifecycle._register_two_layers(de_worker)
        lifecycle._register_two_layers(pe_worker)
        de_worker.kv_recv_layer_thread = MagicMock(name="decode_recv_thread")
        cls._clear_receive(de_worker)
        cls._clear_receive(pe_worker)

        reverse_metadata = cls._reverse_send_metadata()
        de_worker._build_reverse_send_metadata = MagicMock(return_value=reverse_metadata)
        pe_worker._align_remote_block_ids = MagicMock()
        pe_worker._get_kv_split_metadata = MagicMock(
            return_value={
                ("198.51.100.20", 7000): {
                    "local_block_ids": list(PE_BLOCKS[K_DE // BLOCK_SIZE :]),
                    "remote_block_ids": list(lifecycle.DESTINATION_BLOCKS[0][K_DE // BLOCK_SIZE :]),
                    "trans_count": 1,
                }
            }
        )
        return cls(de_worker=de_worker, pe_worker=pe_worker)

    @staticmethod
    def _reverse_send_metadata() -> MooncakeLayerwiseConnectorMetadata:
        metadata = MooncakeLayerwiseConnectorMetadata()
        metadata.requests[lifecycle.DECODE_REQUEST_ID] = ReqMeta(
            local_block_ids=[[11, 20, 21]],
            token_ids=None,
            remote_block_ids=[[71, 80, 81]],
            remote_block_size=[BLOCK_SIZE],
            remote_engine_id="prefill-engine",
            remote_host="198.51.100.10",
            remote_port=6000,
            remote_te_rpc_port=None,
            remote_layer_metadata=None,
            metaserver=None,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
            chunk_finish=True,
            prompt_len=K_DE,
            trans_count=[1],
            local_computed_tokens=K_DE,
            local_transed_tokens=L_PE,
        )
        return metadata

    @staticmethod
    def _clear_receive(worker: DualPathConnectorWorker) -> None:
        worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = set()
        worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = set()

    @staticmethod
    def inject_receive(worker: DualPathConnectorWorker, *, failed: bool = False) -> None:
        worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
            set() if failed else {lifecycle.WIRE_REQUEST_ID}
        )
        worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
            {lifecycle.WIRE_REQUEST_ID} if failed else set()
        )

    @staticmethod
    def inject_reverse_receive(worker: DualPathConnectorWorker, *, failed: bool = False) -> None:
        worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = (
            set() if failed else {lifecycle.REVERSE_WIRE_REQUEST_ID}
        )
        worker.kv_recv_layer_thread.get_and_clear_failed_requests.return_value = (
            {lifecycle.REVERSE_WIRE_REQUEST_ID} if failed else set()
        )

    def poll_de(self, metadata: DualPathConnectorMetadata) -> tuple[set[str], set[str]]:
        result = self.de_worker.get_finished(set(), metadata)
        self._clear_receive(self.de_worker)
        return result

    def poll_pe(self, metadata: DualPathConnectorMetadata) -> tuple[set[str], set[str]]:
        result = self.pe_worker.get_finished(set(), metadata)
        self._clear_receive(self.pe_worker)
        return result

    def install(
        self,
        *,
        include_store: bool,
        include_reverse: bool,
    ) -> tuple[DualPathConnectorMetadata, DualPathConnectorMetadata]:
        pe_metadata = DualPathConnectorMetadata()
        if include_reverse:
            pe_metadata.reverse_receive_bindings.append(lifecycle._make_reverse_receive_binding())
        self.pe_worker.start_load_kv(pe_metadata)

        de_metadata = lifecycle._make_split_metadata(
            include_store=include_store,
            include_reverse=include_reverse,
        )
        self.de_worker.start_load_kv(de_metadata)
        return de_metadata, pe_metadata

    def finish_store(self, metadata: DualPathConnectorMetadata) -> tuple[set[str], set[str]]:
        self.de_worker._kvpool_worker_adapter.get_finished.return_value = (
            set(),
            {lifecycle.DECODE_REQUEST_ID},
        )
        result = self.poll_de(metadata)
        self.de_worker._kvpool_worker_adapter.get_finished.return_value = (set(), set())
        return result

    def finish_reverse(
        self,
        de_metadata: DualPathConnectorMetadata,
        pe_metadata: DualPathConnectorMetadata,
        *,
        failed: bool = False,
    ) -> tuple[tuple[set[str], set[str]], tuple[set[str], set[str]]]:
        with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
            self.de_worker.send_done_send_signal(
                lifecycle.DECODE_REQUEST_ID,
                self._reverse_send_metadata().requests[lifecycle.DECODE_REQUEST_ID],
                0,
                trans_flag=not failed,
            )
        self.inject_reverse_receive(self.pe_worker, failed=failed)
        return self.poll_de(de_metadata), self.poll_pe(pe_metadata)

    def run_forward(self) -> MooncakeLayerwiseConnectorMetadata:
        metadata = DualPathConnectorMetadata()
        metadata.requests[PREFILL_REQUEST_ID] = ReqMeta(
            local_block_ids=[list(PE_BLOCKS)],
            token_ids=None,
            remote_block_ids=[list(lifecycle.DESTINATION_BLOCKS[0])],
            remote_block_size=[BLOCK_SIZE],
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
            prompt_len=TARGET_TOKENS,
            trans_count=[],
            local_computed_tokens=TARGET_TOKENS,
            local_transed_tokens=K_DE,
        )
        self.pe_worker.start_load_kv(metadata)
        events = {name: SimpleNamespace(reshape_cache_event=MagicMock(name=f"{name}_event")) for name in LAYER_NAMES}
        for _ in LAYER_NAMES:
            self.pe_worker.save_kv_layer("", [MagicMock(), MagicMock()], events, metadata)
        return metadata


def _assert_forward_tasks(harness: SplitHarness, metadata: MooncakeLayerwiseConnectorMetadata) -> None:
    assert metadata.requests[PREFILL_REQUEST_ID].local_transed_tokens == K_DE
    tasks = [call.args[0] for call in harness.pe_worker.kv_send_layer_thread.send_queue.put.call_args_list]
    assert [task.layer_name for task in tasks] == list(LAYER_NAMES)
    assert [task.layer_idx for task in tasks] == [0, 1]
    assert all(set(task.send_request) == {PREFILL_REQUEST_ID} for task in tasks)


def test_injected_split_success_with_non_empty_store_and_reverse() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=True, include_reverse=True)

    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.finish_store(de_metadata) == (set(), set())
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == len(LAYER_NAMES)
    assert harness.poll_pe(pe_metadata) == (set(), set())

    de_reverse, pe_reverse = harness.finish_reverse(de_metadata, pe_metadata)
    assert de_reverse == (set(), set())
    assert pe_reverse == (set(), set())
    pe_jobs = harness.pe_worker.build_connector_worker_meta()
    assert pe_jobs.completed_jobs == {pe_metadata.reverse_receive_bindings[0].reverse_completion_job_id: 1}
    forward_metadata = harness.run_forward()
    _assert_forward_tasks(harness, forward_metadata)

    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.de_worker.get_block_ids_with_load_errors() == set()
    assert harness.pe_worker.get_block_ids_with_load_errors() == set()


def test_injected_split_success_with_empty_store() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=False, include_reverse=True)

    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == len(LAYER_NAMES)
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.finish_reverse(de_metadata, pe_metadata) == (
        (set(), set()),
        (set(), set()),
    )
    _assert_forward_tasks(harness, harness.run_forward())
    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.de_worker.get_block_ids_with_load_errors() == set()
    assert harness.pe_worker.get_block_ids_with_load_errors() == set()


def test_injected_split_success_with_empty_reverse() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=True, include_reverse=False)

    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.finish_store(de_metadata) == (set(), set())
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    _assert_forward_tasks(harness, harness.run_forward())
    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.de_worker.get_block_ids_with_load_errors() == set()
    assert harness.pe_worker.get_block_ids_with_load_errors() == set()


def test_injected_split_success_with_empty_store_and_reverse() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=False, include_reverse=False)

    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.poll_pe(pe_metadata) == (set(), set())
    _assert_forward_tasks(harness, harness.run_forward())
    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.de_worker.get_block_ids_with_load_errors() == set()
    assert harness.pe_worker.get_block_ids_with_load_errors() == set()


def test_injected_reverse_failure_end_to_end() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=True, include_reverse=True)
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.finish_store(de_metadata) == (set(), set())
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == len(LAYER_NAMES)

    de_failed, pe_failed = harness.finish_reverse(de_metadata, pe_metadata, failed=True)
    assert de_failed == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert pe_failed == (set(), set())
    pe_jobs = harness.pe_worker.build_connector_worker_meta()
    assert pe_jobs.failed_jobs == {pe_metadata.reverse_receive_bindings[0].reverse_completion_job_id: 1}
    assert harness.de_worker.get_block_ids_with_load_errors() == FORWARD_DE_BLOCKS
    assert harness.pe_worker.get_block_ids_with_load_errors() == REVERSE_PE_BLOCKS
    assert harness.pe_worker.kv_send_layer_thread.send_queue.put.call_count == 0

    harness.inject_receive(harness.de_worker)
    harness.inject_reverse_receive(harness.pe_worker)
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.poll_pe(pe_metadata) == (set(), set())


def test_injected_forward_failure_end_to_end() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=True, include_reverse=True)
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.finish_store(de_metadata) == (set(), set())
    assert harness.finish_reverse(de_metadata, pe_metadata) == (
        (set(), set()),
        (set(), set()),
    )
    _assert_forward_tasks(harness, harness.run_forward())

    harness.inject_receive(harness.de_worker, failed=True)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.de_worker.get_block_ids_with_load_errors() == FORWARD_DE_BLOCKS
    assert harness.pe_worker.get_block_ids_with_load_errors() == set()
    assert harness.poll_de(de_metadata) == (set(), set())


@pytest.mark.parametrize("failed", [False, True])
def test_injected_early_terminal_race(failed: bool) -> None:
    harness = SplitHarness.make()
    empty_metadata = DualPathConnectorMetadata()
    harness.inject_receive(harness.de_worker, failed=failed)
    harness.inject_reverse_receive(harness.pe_worker, failed=failed)
    assert harness.poll_de(empty_metadata) == (set(), set())
    assert harness.poll_pe(empty_metadata) == (set(), set())

    de_metadata, pe_metadata = harness.install(include_store=False, include_reverse=True)
    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        harness.de_worker.send_done_send_signal(
            lifecycle.DECODE_REQUEST_ID,
            harness._reverse_send_metadata().requests[lifecycle.DECODE_REQUEST_ID],
            0,
            trans_flag=not failed,
        )
    expected_de_invalid = FORWARD_DE_BLOCKS if failed else set()
    expected_pe_invalid = REVERSE_PE_BLOCKS if failed else set()
    assert harness.poll_pe(pe_metadata) == (set(), set())
    pe_jobs = harness.pe_worker.build_connector_worker_meta()
    pe_completion_job_id = pe_metadata.reverse_receive_bindings[0].reverse_completion_job_id
    if failed:
        assert pe_jobs.failed_jobs == {pe_completion_job_id: 1}
    else:
        assert pe_jobs.completed_jobs == {pe_completion_job_id: 1}
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.de_worker.get_block_ids_with_load_errors() == expected_de_invalid
    assert harness.pe_worker.get_block_ids_with_load_errors() == expected_pe_invalid
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.poll_pe(pe_metadata) == (set(), set())


def test_injected_duplicate_and_replay_safety() -> None:
    harness = SplitHarness.make()
    de_metadata = lifecycle._make_split_metadata(include_store=True, include_reverse=True)
    de_metadata.forward_receive_bindings.append(de_metadata.forward_receive_bindings[0])
    de_metadata.reverse_plans.append(de_metadata.reverse_plans[0])
    pe_metadata = DualPathConnectorMetadata()
    pe_binding = lifecycle._make_reverse_receive_binding()
    pe_metadata.reverse_receive_bindings.extend((pe_binding, pe_binding))

    harness.pe_worker.start_load_kv(pe_metadata)
    harness.de_worker.start_load_kv(de_metadata)
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == 0
    assert harness.poll_pe(pe_metadata) == (set(), set())
    harness.de_worker._kvpool_worker_adapter.get_finished.return_value = (
        set(),
        {lifecycle.DECODE_REQUEST_ID},
    )
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.de_worker.kv_send_layer_thread.send_queue.put.call_count == len(LAYER_NAMES)

    with patch.object(layerwise_module.MooncakeLayerwiseConnectorWorker, "send_done_send_signal"):
        reverse_req_meta = harness._reverse_send_metadata().requests[lifecycle.DECODE_REQUEST_ID]
        harness.de_worker.send_done_send_signal(lifecycle.DECODE_REQUEST_ID, reverse_req_meta, 0, True)
        harness.de_worker.send_done_send_signal(lifecycle.DECODE_REQUEST_ID, reverse_req_meta, 0, True)
    harness.inject_reverse_receive(harness.pe_worker)
    assert harness.poll_pe(pe_metadata) == (set(), set())
    pe_jobs = harness.pe_worker.build_connector_worker_meta()
    assert pe_jobs.completed_jobs == {pe_binding.reverse_completion_job_id: 1}
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.poll_de(de_metadata) == (set(), set())

    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), set())
    assert harness.poll_de(de_metadata) == (set(), set())


def test_after_reverse_done_prefill_executes_inherited_layerwise_forward() -> None:
    harness = SplitHarness.make()
    de_metadata, pe_metadata = harness.install(include_store=False, include_reverse=True)
    assert harness.poll_pe(pe_metadata) == (set(), set())
    assert harness.finish_reverse(de_metadata, pe_metadata) == (
        (set(), set()),
        (set(), set()),
    )

    forward_metadata = harness.run_forward()
    _assert_forward_tasks(harness, forward_metadata)
    reverse_tasks = [call.args[0] for call in harness.de_worker.kv_send_layer_thread.send_queue.put.call_args_list]
    assert [task.layer_name for task in reverse_tasks] == list(LAYER_NAMES)
    assert [task.layer_idx for task in reverse_tasks] == [0, 1]
    assert all(set(task.send_request) == {lifecycle.DECODE_REQUEST_ID} for task in reverse_tasks)

    harness.inject_receive(harness.de_worker)
    assert harness.poll_de(de_metadata) == (set(), {lifecycle.DECODE_REQUEST_ID})
    assert harness.poll_de(de_metadata) == (set(), set())

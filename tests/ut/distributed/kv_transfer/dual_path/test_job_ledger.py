# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W2: completion-job ledger accounting and the no-pinning contract."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_reverse_send_proof import (
    _activate_decision,
    _admit_decode_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    PathKind,
    ReverseAttemptKey,
)


def _ledgers():
    return importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.ledgers")


class TestJobLedger:
    def test_all_worker_rule_closes_only_on_the_last_report(self):
        ledgers = _ledgers()
        ledger = ledgers.JobLedger()
        job = ledger.create_job(ledgers.JobKind.REVERSE_COMPLETION, expected_worker_count=2)

        assert ledger.record_reports(job.job_id, 1) is False
        assert job.completed_worker_count == 1
        assert job.closed is False

        assert ledger.record_reports(job.job_id, 1) is True
        assert job.completed_worker_count == 2
        assert job.closed is True

    def test_duplicate_reports_are_capped_at_the_expected_count(self):
        ledgers = _ledgers()
        ledger = ledgers.JobLedger()
        job = ledger.create_job(ledgers.JobKind.REVERSE_COMPLETION, expected_worker_count=2)

        assert ledger.record_reports(job.job_id, 5) is True
        assert job.completed_worker_count == 2

        # Post-close reports are ignored entirely and never re-close the job.
        assert ledger.record_reports(job.job_id, 1) is False
        assert job.completed_worker_count == 2

    def test_failure_closes_the_job_as_failed(self):
        ledgers = _ledgers()
        ledger = ledgers.JobLedger()
        job = ledger.create_job(ledgers.JobKind.REVERSE_COMPLETION, expected_worker_count=2)

        assert ledger.record_failure(job.job_id) is True
        assert job.closed is True
        assert job.failed is True
        # A second failure report is absorbed.
        assert ledger.record_failure(job.job_id) is False

    def test_invalid_expected_worker_count_rejected(self):
        ledgers = _ledgers()
        ledger = ledgers.JobLedger()
        with pytest.raises(TypeError):
            ledger.create_job(ledgers.JobKind.REVERSE_SEND, expected_worker_count=True)
        with pytest.raises(ValueError):
            ledger.create_job(ledgers.JobKind.REVERSE_SEND, expected_worker_count=0)


class TestReverseDestinationNotPinned:
    def test_de_read_reverse_destination_blocks_are_never_pinned(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )

        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)

        # The Decision still ships, but the Reverse destination slice
        # [L_PE, K_DE) = [16, 32), which lives on block 71, is no longer pinned.
        assert coordinator.submit.call_count == 1
        assert pool.blocks[71].ref_cnt == 0
        assert not hasattr(scheduler, "_hold_ledger")
        assert not hasattr(scheduler, "_reverse_destination_holds")

    def test_forward_source_blocks_are_never_pinned(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _make_request()
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

        # The PE_READ Forward range [16, 33) covers source blocks 11 and 12;
        # the Forward direction follows the parent's immediate-free semantics,
        # so no hold and no gating job exist for them.
        assert not hasattr(scheduler, "_hold_ledger")
        assert scheduler._job_ledger.open_count() == 0
        assert pool.blocks[11].ref_cnt == 0
        assert pool.blocks[12].ref_cnt == 0

    def test_expected_worker_count_from_world_size(self, pe_scheduler_factory):
        _scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, world_size=2, pool=make_block_pool())
        assert _scheduler._expected_worker_count == 2

    def test_worker_metadata_aggregate_rejects_foreign_type(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata

        class ForeignWorkerMetadata(KVConnectorWorkerMetadata):
            def aggregate(self, other):
                return self

        metadata = make_worker_metadata(completed_jobs={1: 1})
        with pytest.raises(AssertionError):
            metadata.aggregate(ForeignWorkerMetadata())

        merged = metadata.aggregate(make_worker_metadata(completed_jobs={1: 1, 2: 1}, failed_jobs={3: 1}))
        assert merged.completed_jobs == {1: 2, 2: 1}
        assert merged.failed_jobs == {3: 1}


class TestJobRecordReclamation:
    @staticmethod
    def _admit_de_read(pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )
        assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
        scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)
        return scheduler, request

    def test_failed_completion_job_record_is_reclaimed_with_the_request(self, pe_scheduler_factory):
        scheduler, request = self._admit_de_read(pe_scheduler_factory)
        job_id = scheduler._prefill_pending_reverse_receive_bindings[request.request_id].reverse_completion_job_id
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        assert metadata.reverse_receive_bindings[0].reverse_completion_job_id == job_id
        assert scheduler._prefill_pending_reverse_receive_bindings == {}
        assert scheduler._job_ledger.record_failure(job_id) is True

        scheduler._release_scheduler_request_state(request)

        assert scheduler._job_ledger.get(job_id) is None

    def test_open_completion_job_record_survives_request_cleanup(self, pe_scheduler_factory):
        scheduler, request = self._admit_de_read(pe_scheduler_factory)
        job_id = scheduler._prefill_pending_reverse_receive_bindings[request.request_id].reverse_completion_job_id
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        assert metadata.reverse_receive_bindings[0].reverse_completion_job_id == job_id
        assert scheduler._prefill_pending_reverse_receive_bindings == {}

        scheduler._release_scheduler_request_state(request)

        assert scheduler._job_ledger.get(job_id) is not None

    def test_late_completion_failure_after_request_finished_discards_orphan_without_side_effects(
        self,
        pe_scheduler_factory,
    ):
        scheduler, request = self._admit_de_read(pe_scheduler_factory)
        job_id = scheduler._prefill_pending_reverse_receive_bindings[request.request_id].reverse_completion_job_id
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        assert metadata.reverse_receive_bindings[0].reverse_completion_job_id == job_id
        request.status = RequestStatus.FINISHED_STOPPED
        assert scheduler.request_finished(request, [70, 71]) == (False, None)
        assert scheduler._job_ledger.get(job_id) is not None

        output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(failed_jobs={job_id: 1}))
        scheduler.update_connector_output(output)

        assert scheduler._job_ledger.get(job_id) is None
        assert scheduler._prefill_control_failures == {}
        assert output.finished_recving is None

    def test_request_cleanup_reclaims_every_closed_completion_job_after_binding_delivery(self, pe_scheduler_factory):
        scheduler, request = self._admit_de_read(pe_scheduler_factory)
        binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
        successful_job_id = binding.reverse_completion_job_id
        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
        assert metadata.reverse_receive_bindings[0].reverse_completion_job_id == successful_job_id
        assert scheduler._prefill_pending_reverse_receive_bindings == {}

        ledgers = _ledgers()
        failed_job = scheduler._job_ledger.create_job(
            ledgers.JobKind.REVERSE_COMPLETION,
            expected_worker_count=1,
            reverse_attempt_key=ReverseAttemptKey(binding.request_key, 1),
        )
        open_job = scheduler._job_ledger.create_job(
            ledgers.JobKind.REVERSE_COMPLETION,
            expected_worker_count=1,
            reverse_attempt_key=ReverseAttemptKey(binding.request_key, 2),
        )
        assert scheduler._job_ledger.record_reports(successful_job_id, 1) is True
        assert scheduler._job_ledger.record_failure(failed_job.job_id) is True

        scheduler._release_scheduler_request_state(request)

        assert scheduler._job_ledger.get(successful_job_id) is None
        assert scheduler._job_ledger.get(failed_job.job_id) is None
        assert scheduler._job_ledger.get(open_job.job_id) is open_job

    @pytest.mark.parametrize("latest_closed", [False, True], ids=["latest-open", "latest-closed"])
    def test_request_cleanup_reclaims_closed_superseded_send_jobs_and_preserves_latest_state(
        self, decode_scheduler_factory, decode_task04_seams, latest_closed
    ):
        scheduler = decode_scheduler_factory()
        request = _admit_decode_request(scheduler)
        first_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
        second_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(1))
        old_job_id = first_metadata.reverse_plans[0].reverse_send_job_id
        latest_job_id = second_metadata.reverse_plans[0].reverse_send_job_id
        assert old_job_id is not None
        assert latest_job_id is not None
        state = scheduler._decode_decision_states[request.request_id]
        old_attempt = ReverseAttemptKey(state.request_key, 0)
        latest_attempt = ReverseAttemptKey(state.request_key, 1)
        assert scheduler._job_ledger.record_reports(old_job_id, 1) is True
        if latest_closed:
            assert scheduler._job_ledger.record_reports(latest_job_id, 1) is True

        scheduler._release_scheduler_request_state(request)

        assert old_attempt not in scheduler._reverse_send_job_ids
        assert scheduler._job_ledger.get(old_job_id) is None
        if latest_closed:
            assert latest_attempt not in scheduler._reverse_send_job_ids
            assert scheduler._job_ledger.get(latest_job_id) is None
            assert request.request_id not in scheduler._latest_reverse_attempt_ids
        else:
            assert scheduler._reverse_send_job_ids[latest_attempt] == latest_job_id
            assert scheduler._job_ledger.get(latest_job_id) is not None
            assert scheduler._latest_reverse_attempt_ids[request.request_id] == 1

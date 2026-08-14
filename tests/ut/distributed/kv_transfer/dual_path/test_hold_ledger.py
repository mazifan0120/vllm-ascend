# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W2: hold/job ledger accounting and Reverse destination hold lifecycle."""

from __future__ import annotations

import importlib

import pytest

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind


def _ledgers():
    return importlib.import_module("vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.ledgers")


class TestHoldLedger:
    def test_hold_accounting_acquire_release_balance(self):
        ledgers = _ledgers()
        pool = make_block_pool(num_blocks=8)
        ledger = ledgers.HoldLedger()
        free_at_start = pool.free_block_queue.num_free_blocks

        first = ledger.acquire(pool, (1, 2), ledgers.HoldKind.REVERSE_DESTINATION)
        second = ledger.acquire(pool, (2, 3), ledgers.HoldKind.REVERSE_DESTINATION)
        assert pool.blocks[1].ref_cnt == 1
        assert pool.blocks[2].ref_cnt == 2
        assert pool.blocks[3].ref_cnt == 1

        # Overlapping release: block 2 keeps one reference from the second hold.
        assert ledger.release(pool, first.hold_id)
        assert pool.blocks[1].ref_cnt == 0
        assert pool.blocks[2].ref_cnt == 1
        assert pool.blocks[3].ref_cnt == 1

        # Retry of the same release is a no-op and cannot unbalance refcounts.
        assert not ledger.release(pool, first.hold_id)
        assert pool.blocks[1].ref_cnt == 0
        assert pool.blocks[2].ref_cnt == 1

        assert ledger.release(pool, second.hold_id)
        assert all(pool.blocks[block_id].ref_cnt == 0 for block_id in (1, 2, 3))
        assert pool.free_block_queue.num_free_blocks == free_at_start

        # A repeated acquire/release cycle over the same ids stays balanced.
        retry = ledger.acquire(pool, (1, 2, 3), ledgers.HoldKind.REVERSE_DESTINATION)
        assert ledger.release(pool, retry.hold_id)
        assert all(pool.blocks[block_id].ref_cnt == 0 for block_id in (1, 2, 3))
        assert pool.free_block_queue.num_free_blocks == free_at_start

    def test_one_record_per_held_reference(self):
        ledgers = _ledgers()
        pool = make_block_pool(num_blocks=8)
        ledger = ledgers.HoldLedger()

        first = ledger.acquire(pool, (4,), ledgers.HoldKind.REVERSE_DESTINATION)
        second = ledger.acquire(pool, (4,), ledgers.HoldKind.REVERSE_DESTINATION)

        assert first.hold_id != second.hold_id
        assert pool.blocks[4].ref_cnt == 2

        assert ledger.release(pool, first.hold_id)
        assert pool.blocks[4].ref_cnt == 1
        assert ledger.get(second.hold_id).released is False

        assert ledger.release(pool, second.hold_id)
        assert pool.blocks[4].ref_cnt == 0


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

    def test_failure_closes_the_job_without_releasing_holds(self):
        ledgers = _ledgers()
        ledger = ledgers.JobLedger()
        job = ledger.create_job(
            ledgers.JobKind.REVERSE_COMPLETION,
            expected_worker_count=2,
            affected_hold_ids=(7,),
        )

        assert ledger.record_failure(job.job_id) is True
        assert job.closed is True
        assert job.failed is True
        assert job.affected_hold_ids == (7,)
        # A second failure report is absorbed.
        assert ledger.record_failure(job.job_id) is False

    def test_invalid_expected_worker_count_rejected(self):
        ledgers = _ledgers()
        ledger = ledgers.JobLedger()
        with pytest.raises(TypeError):
            ledger.create_job(ledgers.JobKind.REVERSE_SEND, expected_worker_count=True)
        with pytest.raises(ValueError):
            ledger.create_job(ledgers.JobKind.REVERSE_SEND, expected_worker_count=0)


class TestReverseDestinationHoldLifecycle:
    def test_de_read_holds_reverse_destination_before_delivery(self, pe_scheduler_factory):
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

        # The Reverse destination hold over [L_PE, K_DE) = [16, 32) exists
        # before the Decision is delivered.
        assert coordinator.submit.call_count == 1
        reverse_hold_id = scheduler._reverse_destination_holds[request.request_id]
        reverse_hold = scheduler._hold_ledger.get(reverse_hold_id)
        assert reverse_hold.hold_kind.name == "REVERSE_DESTINATION"
        assert reverse_hold.block_ids == (71,)
        assert pool.blocks[71].ref_cnt == 1

    def test_forward_source_blocks_are_never_pinned(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _make_request()
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

        # The PE_READ Forward range [16, 33) covers source blocks 11 and 12;
        # the Forward direction follows the parent's immediate-free semantics,
        # so no hold and no gating job exist for them.
        assert scheduler._hold_ledger.unreleased_count() == 0
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

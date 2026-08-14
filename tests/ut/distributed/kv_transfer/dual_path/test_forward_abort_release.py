# SPDX-License-Identifier: Apache-2.0
"""The Forward direction retains nothing on abort, normal finish, or preemption.

Forward is the ordinary Layerwise push: it pins no block and delays no free, so
none of these paths may leave a hold or a gating job behind.
"""

from __future__ import annotations

from vllm.v1.request import RequestStatus

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_empty_scheduler_output,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind


def _admit_pe_read(scheduler):
    request = _make_request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)
    return request


def _assert_nothing_retained(scheduler, pool):
    assert scheduler._hold_ledger.unreleased_count() == 0
    assert scheduler._job_ledger.open_count() == 0
    assert scheduler._pending_finished_sending == set()
    # The PE_READ Forward range [16, 33) covers source blocks 11 and 12.
    assert pool.blocks[11].ref_cnt == 0
    assert pool.blocks[12].ref_cnt == 0


class TestForwardRetainsNothing:
    def test_running_abort_frees_immediately(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _admit_pe_read(scheduler)

        request.status = RequestStatus.FINISHED_ABORTED
        assert scheduler.request_finished(request, []) == (False, None)
        _assert_nothing_retained(scheduler, pool)

    def test_normal_finish_frees_immediately(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _admit_pe_read(scheduler)

        request.status = RequestStatus.FINISHED_STOPPED
        assert scheduler.request_finished(request, []) == (False, None)
        _assert_nothing_retained(scheduler, pool)

    def test_preemption_creates_no_fence_work(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _admit_pe_read(scheduler)

        metadata = scheduler.build_connector_meta(make_empty_scheduler_output(preempted_req_ids={request.request_id}))

        # A preempting pass carries no fence job and pins nothing, so the
        # worker has no barrier to drain before the next forward pass.
        assert not hasattr(metadata, "barrier_jobs")
        _assert_nothing_retained(scheduler, pool)

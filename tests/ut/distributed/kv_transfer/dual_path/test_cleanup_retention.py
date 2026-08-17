# SPDX-License-Identifier: Apache-2.0
"""Cleanup retention for request-owned Scheduler state."""

from __future__ import annotations

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind


class TestCleanupRetention:
    def test_logical_cleanup_removes_forward_state_as_unit(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _make_request()
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

        scheduler.request_finished(request, [10, 11, 12])

        assert scheduler._waiting_reverse_attempt_ids == {}
        assert scheduler._prefill_forward_plans == {}
        assert scheduler._prefill_forward_plan_epochs == {}
        assert scheduler._prefill_request_keys == {}
        assert scheduler._reqs_need_send_layerwise == {}
        assert pool.blocks[11].ref_cnt == 0

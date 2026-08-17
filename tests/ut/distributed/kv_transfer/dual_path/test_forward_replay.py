# SPDX-License-Identifier: Apache-2.0
"""Replay oracles for the Forward plan across a preemption.

The Forward direction pins nothing, so a resume rebuilds the plan and the send
state directly from the new block table without waiting on a release signal.
"""

from __future__ import annotations

from tests.ut.distributed.kv_transfer.dual_path.conftest import make_block_pool
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind

_L_DE = 16
_T = 33


def _admit_pe_read(scheduler, request=None):
    if request is None:
        request = _make_request()
        if scheduler._expected_worker_count > 1:
            request.kv_transfer_params["remote_tp_size"] = scheduler._expected_worker_count
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)
    return request


def _resume_and_realloc(scheduler, request, block_ids):
    request.num_preemptions += 1
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, _blocks((list(block_ids),)), 0)


def test_no_de_binding_replacement_on_replay(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, coordinator = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
    request = _admit_pe_read(scheduler)

    _resume_and_realloc(scheduler, request, [14, 15, 16])

    # PE_READ recovery rebuilds only the local Forward source: no new
    # Decision, no DE reactivation, no Reverse artifacts.
    assert coordinator.submit.call_count == 1
    assert scheduler._prefill_reverse_plans == {}
    assert scheduler._prefill_pending_reverse_receive_bindings == {}


def test_source_plan_and_send_state_follow_the_new_block_table(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
    request = _admit_pe_read(scheduler)
    old_plan = scheduler._prefill_forward_plans[request.request_id]

    _resume_and_realloc(scheduler, request, [14, 15, 16])

    new_plan = scheduler._prefill_forward_plans[request.request_id]
    assert new_plan is not old_plan
    assert new_plan.source_block_ids == ((14, 15, 16),)
    assert scheduler._reqs_need_send_layerwise[request.request_id].local_block_ids == [[14, 15, 16]]
    # No Forward hold exists, so the superseded blocks keep the ordinary
    # request refcount only.
    assert pool.blocks[11].ref_cnt == 0
    assert pool.blocks[15].ref_cnt == 0


def test_coincidentally_equal_block_ids_reinstall_the_same_plan(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
    request = _admit_pe_read(scheduler)

    # The freed blocks are coincidentally reallocated: the replayed plan is
    # value-equal to the previous one and the send state stays consistent.
    _resume_and_realloc(scheduler, request, [10, 11, 12])

    plan = scheduler._prefill_forward_plans[request.request_id]
    assert plan.source_block_ids == ((10, 11, 12),)
    assert scheduler._reqs_need_send_layerwise[request.request_id].local_block_ids == [[10, 11, 12]]


def test_pe_read_forward_range_remains_l_de_to_t(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
    request = _admit_pe_read(scheduler)
    _resume_and_realloc(scheduler, request, [14, 15, 16])

    plan = scheduler._prefill_forward_plans[request.request_id]
    assert (plan.token_start, plan.token_end) == (_L_DE, _T)
    assert plan.destination_block_ids == ((20, 21, 22),)

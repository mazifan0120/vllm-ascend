# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W4: resume admission for delivered DualPath decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

from tests.ut.distributed.kv_transfer.dual_path.conftest import make_block_pool
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    PathDecisionRequest,
    PathKind,
)

_L_PE_OLD = 16
_K_DE = 32


class _ExplodingOnRerunPolicy:
    """Returns the frozen path once; any later policy turn is a test failure."""

    def __init__(self, path: PathKind) -> None:
        self.path = path
        self.calls = 0

    def choose(self, request: PathDecisionRequest) -> PathKind:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("PathPolicy re-run for a delivered decision")
        return self.path


def _make_de_read_request(remote_tp_size: int = 1):
    request = _make_request(
        target_tokens=48,
        prompt_tokens=49,
        local_tokens=_L_PE_OLD,
        store_tokens=_K_DE,
        destination_block_ids=[[20, 21, 22, 23]],
    )
    request.kv_transfer_params["remote_tp_size"] = remote_tp_size
    return request


def _admit_de_read(scheduler, request=None):
    if request is None:
        request = _make_de_read_request()
    assert scheduler.get_num_new_matched_tokens(request, _L_PE_OLD) == (_K_DE - _L_PE_OLD, True)
    scheduler.update_state_after_alloc(request, _blocks(([70, 71, 72, 73],)), _K_DE - _L_PE_OLD)
    return request


def test_delivered_state_detection_precedes_decider(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)

    scheduler._path_decider.decide = MagicMock(side_effect=AssertionError("decider consulted on resume"))
    assert scheduler.get_num_new_matched_tokens(request, 24) == (_K_DE - 24, True)
    scheduler._path_decider.decide.assert_not_called()


def test_frozen_path_kind_reused_policy_never_rerun(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(policy=_ExplodingOnRerunPolicy(PathKind.DE_READ), pool=pool)
    request = _admit_de_read(scheduler)

    assert scheduler.get_num_new_matched_tokens(request, 24) == (_K_DE - 24, True)
    assert scheduler._prefill_path_results[request.request_id].path is PathKind.DE_READ


def test_same_request_object_can_resume_after_delivery(pe_scheduler_factory):
    pool = make_block_pool()
    policy = _ExplodingOnRerunPolicy(PathKind.DE_READ)
    scheduler, _ = pe_scheduler_factory(policy=policy, pool=pool)
    request = _admit_de_read(scheduler)

    assert scheduler.get_num_new_matched_tokens(request, 24) == (_K_DE - 24, True)
    assert scheduler._prefill_path_results[request.request_id].path is PathKind.DE_READ
    assert policy.calls == 1


def test_de_read_resume_token_matrix(pe_scheduler_factory):
    cases = [
        (8, (24, True)),
        (16, (16, True)),
        (24, (8, True)),
        (32, (0, False)),
        (40, (0, False)),
    ]
    for new_local_tokens, expected in cases:
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read(scheduler)

        assert scheduler.get_num_new_matched_tokens(request, new_local_tokens) == expected


def test_de_read_vacuous_reverse_returns_zero_false_and_skips_reverse_machinery(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)
    job_records_before = len(scheduler._job_ledger._records)

    request.num_preemptions += 1
    assert scheduler.get_num_new_matched_tokens(request, 40) == (0, False)

    # A vacuous Reverse bypasses the Reverse machinery entirely: no new
    # completion job and no waiting-attempt entry (the request goes straight
    # back to RUNNING, never parks).
    assert len(scheduler._job_ledger._records) == job_records_before
    assert request.request_id not in scheduler._waiting_reverse_attempt_ids

    # The bypass survives the post-resume allocation: no waiting entry is
    # reinstalled from the retained old plan, no replacement Decision is
    # delivered, and only the local Forward source is rebuilt.
    scheduler.update_state_after_alloc(request, _blocks(([80, 81, 82, 83],)), 0)
    assert request.request_id not in scheduler._waiting_reverse_attempt_ids
    assert len(scheduler._job_ledger._records) == job_records_before
    assert scheduler._prefill_forward_plans[request.request_id].source_block_ids == ((80, 81, 82, 83),)
    assert coordinator.submit.call_count == 1
    # The Forward source is never pinned, so its blocks keep no extra reference.
    assert pool.blocks[82].ref_cnt == 0
    assert pool.blocks[83].ref_cnt == 0

    # Nothing gates the free: the request finishes with the parent's semantics.
    assert scheduler.request_finished(request, [80, 81, 82, 83]) == (False, None)


def test_pe_read_resume_returns_zero_false_rebuilds_local_source_only(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(policy=_ExplodingOnRerunPolicy(PathKind.PE_READ), pool=pool)
    request = _make_request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)
    plan_before = scheduler._prefill_forward_plans[request.request_id]
    job_records_before = len(scheduler._job_ledger._records)

    assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
    # No plan/job mutation and no re-delivery on the resume admission itself.
    assert scheduler._prefill_forward_plans[request.request_id] is plan_before
    assert len(scheduler._job_ledger._records) == job_records_before
    assert scheduler._reqs_need_send_layerwise[request.request_id].local_block_ids == [[10, 11, 12]]


def test_invalid_set_convergence_no_longer_fires_for_delivered_decisions(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)

    # A changed re-probed prefix must not converge into the invalid set: the
    # delivered decision resumes with its frozen path instead.
    assert scheduler.get_num_new_matched_tokens(request, 8) == (24, True)
    assert request.request_id not in scheduler._prefill_invalid_request_ids
    assert scheduler._prefill_control_failures == {}
    assert coordinator.submit.call_count == 1

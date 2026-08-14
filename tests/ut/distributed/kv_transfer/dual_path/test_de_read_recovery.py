# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W6: DE_READ route-preserving recovery after a normal preemption."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm.v1.outputs import KVConnectorOutput

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
from tests.ut.distributed.kv_transfer.dual_path.test_reverse_attempt_identity import (
    _make_reverse_plan,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathKind,
    ReverseAttemptKey,
)

_L_PE_OLD = 16
_K_DE = 48
_T = 65
_ADMISSION_BLOCKS = [70, 71, 72, 73, 74]
_RESUME_BLOCKS = [80, 81, 82, 83, 84]


def _make_de_read_request():
    return _make_request(
        target_tokens=_T - 1,
        prompt_tokens=_T,
        local_tokens=_L_PE_OLD,
        store_tokens=_K_DE,
        destination_block_ids=[[20, 21, 22, 23, 24]],
    )


def _admit_de_read(scheduler, request=None):
    if request is None:
        request = _make_de_read_request()
    assert scheduler.get_num_new_matched_tokens(request, _L_PE_OLD) == (_K_DE - _L_PE_OLD, True)
    scheduler.update_state_after_alloc(request, _blocks((list(_ADMISSION_BLOCKS),)), _K_DE - _L_PE_OLD)
    return request


def _complete_reverse_attempt_zero(scheduler, request) -> None:
    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    output = KVConnectorOutput(
        kv_connector_worker_meta=make_worker_metadata(completed_jobs={binding.reverse_completion_job_id: 1})
    )
    scheduler.update_connector_output(output)
    assert output.finished_recving == {request.request_id}


def _resume(scheduler, request, new_local_tokens: int, new_blocks=None):
    request.num_preemptions += 1
    result = scheduler.get_num_new_matched_tokens(request, new_local_tokens)
    if new_blocks is not None:
        scheduler.update_state_after_alloc(request, _blocks((list(new_blocks),)), max(_K_DE - new_local_tokens, 0))
    return result


def test_de_read_reverse_range_expands_shrinks_vacuous_forward_remains_k_de_to_t(pe_scheduler_factory):
    # (new L_PE, expected reverse range or None when vacuous)
    cases = [(0, (0, _K_DE)), (32, (32, _K_DE)), (64, None)]
    for new_local_tokens, expected_reverse_range in cases:
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _admit_de_read(scheduler)
        _complete_reverse_attempt_zero(scheduler, request)

        _resume(scheduler, request, new_local_tokens, _RESUME_BLOCKS)

        forward_plan = scheduler._prefill_forward_plans[request.request_id]
        assert (forward_plan.token_start, forward_plan.token_end) == (_K_DE, _T)
        if expected_reverse_range is None:
            reverse_plan = scheduler._prefill_reverse_plans[request.request_id]
            assert reverse_plan.reverse_attempt_id == 0
        else:
            reverse_plan = scheduler._prefill_reverse_plans[request.request_id]
            assert (reverse_plan.token_start, reverse_plan.token_end) == expected_reverse_range
            assert reverse_plan.reverse_attempt_id == 1


def test_normal_de_read_preemption_starts_from_reverse_done(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)
    _complete_reverse_attempt_zero(scheduler, request)
    assert request.request_id not in scheduler._waiting_reverse_attempt_ids

    assert _resume(scheduler, request, 32, _RESUME_BLOCKS) == (16, True)
    attempt_one = ReverseAttemptKey(
        scheduler._prefill_path_results[request.request_id].request_key,
        1,
    )
    assert scheduler._waiting_reverse_attempt_ids[request.request_id] == attempt_one


def test_normal_path_never_sends_close_reverse_attempt(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)
    _complete_reverse_attempt_zero(scheduler, request)
    _resume(scheduler, request, 32, _RESUME_BLOCKS)

    assert coordinator.submit.call_count == 2
    for submit_call in coordinator.submit.call_args_list:
        decision = submit_call.args[1]
        assert isinstance(decision.result.reverse_attempt_id, int)
    first_decision = coordinator.submit.call_args_list[0].args[1]
    second_decision = coordinator.submit.call_args_list[1].args[1]
    assert first_decision.result.reverse_attempt_id == 0
    assert second_decision.result.reverse_attempt_id == 1


def test_old_reverse_destination_hold_released_before_running(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)
    # The attempt-0 Reverse destination hold covers [16, 48) -> blocks 71, 72.
    assert pool.blocks[71].ref_cnt == 1
    assert pool.blocks[72].ref_cnt == 1
    _complete_reverse_attempt_zero(scheduler, request)
    # The attempt-0 destination hold is released when the I4 gate passes,
    # before the request can become RUNNING again.
    assert pool.blocks[71].ref_cnt == 0
    assert pool.blocks[72].ref_cnt == 0

    refcnt_at_delivery: list[tuple[int, int]] = []

    def record_submit(_endpoint, _decision):
        refcnt_at_delivery.append((pool.blocks[82].ref_cnt, pool.blocks[83].ref_cnt))
        future = MagicMock(name="delivery_future")
        future.done.return_value = False
        return future

    coordinator.submit.side_effect = record_submit
    _resume(scheduler, request, 32, _RESUME_BLOCKS)

    # The new Reverse destination hold over [32, 48) covers block 82 and exists
    # before delivery; block 83 carries the Forward source, which is never
    # pinned.
    assert refcnt_at_delivery == [(1, 0)]


def test_attempt_n_plus_1_created_on_resume_allocation(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, coordinator = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read(scheduler)
    _complete_reverse_attempt_zero(scheduler, request)
    old_reverse_plan = scheduler._prefill_reverse_plans[request.request_id]

    # The admission pass itself creates no attempt artifacts; they follow the
    # allocation.
    request.num_preemptions += 1
    assert scheduler.get_num_new_matched_tokens(request, 32) == (16, True)
    assert scheduler._prefill_reverse_plans[request.request_id] is old_reverse_plan
    assert coordinator.submit.call_count == 1

    scheduler.update_state_after_alloc(request, _blocks((list(_RESUME_BLOCKS),)), 16)

    new_reverse_plan = scheduler._prefill_reverse_plans[request.request_id]
    assert new_reverse_plan.reverse_attempt_id == 1
    assert new_reverse_plan is not old_reverse_plan
    second_decision = coordinator.submit.call_args_list[1].args[1]
    assert second_decision.result.reverse_attempt_id == 1
    assert second_decision.reverse_plan == new_reverse_plan


def test_fresh_tracker_and_latch_for_new_attempt_old_plan_never_resubmitted():
    from tests.ut.distributed.kv_transfer.dual_path.test_reverse_attempt_identity import _KEY

    worker = _make_worker()
    decode_request_id = _KEY.decode_request_id
    attempt_zero = ReverseAttemptKey(_KEY, 0)
    plan_zero = _make_reverse_plan(reverse_attempt_id=0, reverse_send_job_id=5)
    tracker = worker_module._SplitTracker(
        store_phase=worker_module._SplitPhase.DONE,
        reverse_phase=worker_module._SplitPhase.DONE,
        forward_phase=worker_module._SplitPhase.PENDING,
        store_destination_slice=(),
        forward_destination_slice=(70, 71),
        reverse_plan=plan_zero,
        reverse_submitted_attempt=attempt_zero,
        store_load_failed=False,
        terminal_published=False,
    )
    worker._split_trackers[decode_request_id] = tracker
    binding = MagicMock(name="forward_binding")
    binding.request_key = _KEY
    binding.token_start = 64
    worker._forward_receive_bindings[decode_request_id] = binding

    plan_one = _make_reverse_plan(reverse_attempt_id=1, reverse_send_job_id=6)
    worker._registered_kv_caches = {"model.layer.0": [MagicMock(), MagicMock()]}
    worker._registered_layer_order = ((0, "model.layer.0"),)
    with (
        patch.object(worker, "_build_reverse_send_metadata", return_value=MagicMock()),
        patch.object(worker, "_enqueue_kv_layer_send") as enqueue,
        patch("torch.npu.Event", MagicMock(return_value=MagicMock())),
    ):
        worker._install_reverse_plan(plan_one)

    assert tracker.reverse_plan is plan_one
    assert tracker.reverse_phase is worker_module._SplitPhase.PENDING
    assert tracker.reverse_submitted_attempt == ReverseAttemptKey(_KEY, 1)
    assert enqueue.call_count == 1

    # The old attempt's latch is replaced, and resubmitting is guarded by the
    # fresh latch: a duplicate submit is a no-op.
    worker._submit_reverse(decode_request_id)
    assert enqueue.call_count == 1


def test_de_validation_rejects_admission_drift(decode_scheduler_factory, decode_task04_seams):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    first_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
    assert len(first_metadata.reverse_plans) == 1

    # A greater attempt whose Store boundary drifted from the frozen admission
    # fails validation: no plan is installed and a control failure is staged.
    drifted = _de_read_decision(1)
    drifted_plan = drifted.reverse_plan
    object.__setattr__(drifted_plan, "token_end", 48)
    second_metadata = _activate_decision(scheduler, decode_task04_seams, drifted)

    assert second_metadata.reverse_plans == []
    assert len(second_metadata.control_failures) == 1
    assert second_metadata.control_failures[0].reason is DualPathControlFailureReason.ACTIVATION_FAILED


def test_de_read_refresh_keeps_forward_binding_and_installs_fresh_send_job(
    decode_scheduler_factory, decode_task04_seams
):
    scheduler = decode_scheduler_factory()
    _admit_decode_request(scheduler)
    first_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(0))
    assert len(first_metadata.forward_receive_bindings) == 1
    first_job_id = first_metadata.reverse_plans[0].reverse_send_job_id

    second_metadata = _activate_decision(scheduler, decode_task04_seams, _de_read_decision(1))

    # The logical Forward binding is neither replaced nor reinstalled; a fresh
    # Reverse plan with a new reverse_send_job_id is installed.
    assert second_metadata.forward_receive_bindings == []
    assert len(second_metadata.reverse_plans) == 1
    refreshed_plan = second_metadata.reverse_plans[0]
    assert refreshed_plan.reverse_attempt_id == 1
    assert refreshed_plan.reverse_send_job_id is not None
    assert refreshed_plan.reverse_send_job_id != first_job_id
    attempt_one = ReverseAttemptKey(
        DualPathRequestKey(
            first_metadata.forward_receive_bindings[0].request_key.decode_engine_instance_id, "decode-request-7"
        ),
        1,
    )
    assert scheduler._reverse_send_job_ids[attempt_one] == refreshed_plan.reverse_send_job_id

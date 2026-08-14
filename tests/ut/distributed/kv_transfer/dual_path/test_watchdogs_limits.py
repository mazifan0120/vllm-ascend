# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W8: watchdog expiry and cleanup retention."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_empty_scheduler_output,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_reverse_send_proof import (
    _admit_decode_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind


def test_env_vars_registered_with_defaults(monkeypatch):
    from vllm_ascend import envs as ascend_envs

    for name in (
        "VLLM_ASCEND_DUALPATH_RECOVERY_WATCHDOG_S",
        "VLLM_ASCEND_DUALPATH_DE_PROGRESS_WATCHDOG_S",
    ):
        monkeypatch.delenv(name, raising=False)
        value = getattr(ascend_envs, name)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert value > 0
        monkeypatch.setenv(name, "7")
        assert getattr(ascend_envs, name) == 7


class TestWatchdogs:
    def test_pe_recovery_watchdog_expiry_fails_request_through_control_failure(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )
        clock = MagicMock(name="monotonic_clock")
        clock.monotonic.return_value = 1000.0
        with patch.object(scheduler_module, "time", clock):
            assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
            scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)

            clock.monotonic.return_value = 1000.0 + scheduler._recovery_watchdog_s + 1.0
            expired_metadata = scheduler.build_connector_meta(make_empty_scheduler_output())

        # Expiry fails the request through the control-failure path.
        assert len(expired_metadata.control_failures) == 1
        failure = expired_metadata.control_failures[0]
        assert failure.request_id == request.request_id
        assert failure.reason is DualPathControlFailureReason.RECOVERY_TIMEOUT
        assert request.request_id in scheduler._prefill_invalid_request_ids

    def test_de_progress_watchdog_expiry_fails_request_without_closing_send_job(
        self, decode_scheduler_factory, decode_task04_seams
    ):
        scheduler = decode_scheduler_factory()
        clock = MagicMock(name="monotonic_clock")
        clock.monotonic.return_value = 2000.0
        with patch.object(scheduler_module, "time", clock):
            _admit_decode_request(scheduler)
            decode_task04_seams.decode_coordinator.take_received_decisions.return_value = [_de_read_decision(0)]
            metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
            job_id = metadata.reverse_plans[0].reverse_send_job_id

            clock.monotonic.return_value = 2000.0 + scheduler._de_progress_watchdog_s + 1.0
            decode_task04_seams.decode_coordinator.take_received_decisions.return_value = []
            expired_metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        assert len(expired_metadata.control_failures) == 1
        failure = expired_metadata.control_failures[0]
        assert failure.reason is DualPathControlFailureReason.RECOVERY_TIMEOUT
        job = scheduler._job_ledger.get(job_id)
        assert job.closed is False
        assert job.failed is False


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

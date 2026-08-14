# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W8: local and remote parallel-topology fail-fast validation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_prefill_kv_cache_config,
    make_prefill_vllm_config,
)
from tests.ut.distributed.kv_transfer.dual_path.test_de_reverse_send_proof import (
    _admit_decode_request,
    _de_read_decision,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector as connector_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.config import DualPathConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathControlFailureReason,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind


class TestLocalTopologyGuards:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("pipeline_parallel_size", 2),
            ("data_parallel_size", 2),
            ("prefill_context_parallel_size", 2),
            ("decode_context_parallel_size", 2),
        ],
    )
    def test_rejects_unsupported_local_topology(self, pe_scheduler_factory, field, value):
        config = make_prefill_vllm_config()
        setattr(config.parallel_config, field, value)
        with pytest.raises(ValueError, match="DualPath"):
            connector_module.DualPathConnectorScheduler(
                config,
                make_prefill_kv_cache_config(),
                "prefill-engine",
                DualPathConfig(role="prefill"),
            )


class TestRemoteTopologyValidation:
    def _admit(self, scheduler, request):
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        scheduler.update_state_after_alloc(request, _blocks(([10, 11, 12],)), 0)

    def test_remote_tp_mismatch_rejected_before_decision_and_holds(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _make_request()
        # Local TP is 1; a remote TP of 2 must never enter the protocol.
        request.kv_transfer_params["remote_tp_size"] = 2

        self._admit(scheduler, request)

        assert request.request_id in scheduler._prefill_invalid_request_ids
        assert coordinator.submit.call_count == 0
        assert scheduler._prefill_forward_plans == {}

    @pytest.mark.parametrize("field", ["remote_pcp_size", "remote_dcp_size", "remote_pp_size", "remote_dp_size"])
    def test_remote_unsupported_topology_rejected(self, pe_scheduler_factory, field):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _make_request()
        request.kv_transfer_params[field] = 2

        self._admit(scheduler, request)

        assert request.request_id in scheduler._prefill_invalid_request_ids
        assert coordinator.submit.call_count == 0

    def test_bootstrap_message_missing_topology_fields_rejected(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.PE_READ, pool=pool)
        request = _make_request()
        # A pre-Stage-2 bootstrap message lacks the new fields entirely.
        del request.kv_transfer_params["remote_pp_size"]
        del request.kv_transfer_params["remote_dp_size"]

        self._admit(scheduler, request)

        assert request.request_id in scheduler._prefill_invalid_request_ids
        assert coordinator.submit.call_count == 0

    def test_de_side_rejects_reverse_plan_topology_mismatch(self, decode_scheduler_factory, decode_task04_seams):
        scheduler = decode_scheduler_factory()
        _admit_decode_request(scheduler)
        decision = _de_read_decision(0)
        drifted_plan = replace(decision.reverse_plan, remote_tp_size=2)
        decision = replace(decision, reverse_plan=drifted_plan)
        decode_task04_seams.decode_coordinator.take_received_decisions.return_value = [decision]

        metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))

        assert metadata.reverse_plans == []
        assert len(metadata.control_failures) == 1
        assert metadata.control_failures[0].reason is DualPathControlFailureReason.ACTIVATION_FAILED

    def test_matching_tp2_follows_tp1_semantics(self, pe_scheduler_factory):
        pool = make_block_pool()
        scheduler, coordinator = pe_scheduler_factory(PathKind.PE_READ, world_size=2, pool=pool)
        request = _make_request()
        # Matching TP=2 on both sides is accepted and follows TP=1 semantics.
        request.kv_transfer_params["remote_tp_size"] = 2

        self._admit(scheduler, request)

        assert request.request_id not in scheduler._prefill_invalid_request_ids
        assert coordinator.submit.call_count == 1
        assert scheduler._prefill_forward_plans[request.request_id].source_block_ids == ((10, 11, 12),)
        assert scheduler._expected_worker_count == 2

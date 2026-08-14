# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W3: ReverseAttemptKey identity, wire ids, attempt-keyed installers."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest

from tests.ut.distributed.kv_transfer.dual_path.conftest import layerwise_module
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_prefill_worker,
    _make_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import metadata as metadata_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import path_decision as path_decision_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import worker as worker_module
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import (
    DualPathRequestKey,
    PathDecisionDecider,
    PathDecisionRequest,
    PathDecisionResult,
    PathDecisionValidationError,
    PathKind,
)

_KEY = DualPathRequestKey("decode-instance-1", "decode-request-1")
_ATTEMPT_TOKENS = 16
_COMPLETION_JOB_ID = 41


def _attempt_key(attempt_id: int, request_key: DualPathRequestKey = _KEY):
    return path_decision_module.ReverseAttemptKey(request_key=request_key, reverse_attempt_id=attempt_id)


def _reverse_wire_id(attempt_id: int, request_key: DualPathRequestKey = _KEY) -> str:
    return path_decision_module.reverse_wire_id(_attempt_key(attempt_id, request_key))


def _make_reverse_binding(
    *,
    reverse_attempt_id: int = 0,
    prefill_request_id: str = "prefill-request-1",
    token_start: int = _ATTEMPT_TOKENS,
    token_end: int = 64,
    reverse_completion_job_id: int = _COMPLETION_JOB_ID,
):
    return metadata_module.ReverseReceiveBinding(
        request_key=_KEY,
        wire_request_id=_reverse_wire_id(reverse_attempt_id),
        prefill_request_id=prefill_request_id,
        destination_block_ids=((70, 71, 80, 81),),
        token_start=token_start,
        token_end=token_end,
        reverse_attempt_id=reverse_attempt_id,
        prefill_local_tokens=token_start,
        reverse_completion_job_id=reverse_completion_job_id,
    )


def _make_reverse_plan(*, reverse_attempt_id: int = 0, reverse_send_job_id: int | None = None):
    return metadata_module.ReversePlan(
        request_key=_KEY,
        wire_request_id=_reverse_wire_id(reverse_attempt_id),
        token_start=_ATTEMPT_TOKENS,
        token_end=64,
        source_block_ids=((10, 11, 20, 21),),
        destination_block_ids=((70, 71, 80, 81),),
        remote_engine_id="prefill-engine",
        remote_host="198.51.100.10",
        remote_port=6000,
        remote_block_sizes=(16,),
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        reverse_attempt_id=reverse_attempt_id,
        prefill_local_tokens=_ATTEMPT_TOKENS,
        reverse_send_job_id=reverse_send_job_id,
    )


class TestAttemptKeySchema:
    def test_pe_read_result_schema_unchanged(self):
        result = PathDecisionResult(request_key=_KEY, path=PathKind.PE_READ)
        assert result.to_dict() == {"request_key": _KEY.to_dict(), "path": "PE_READ"}
        assert PathDecisionResult.from_dict(result.to_dict()) == result

        with pytest.raises(PathDecisionValidationError):
            PathDecisionResult(
                request_key=_KEY,
                path=PathKind.PE_READ,
                reverse_attempt_id=0,
                prefill_local_tokens=_ATTEMPT_TOKENS,
            )
        with pytest.raises(PathDecisionValidationError):
            PathDecisionResult.from_dict(
                {
                    "request_key": _KEY.to_dict(),
                    "path": "PE_READ",
                    "reverse_attempt_id": 0,
                    "prefill_local_tokens": _ATTEMPT_TOKENS,
                }
            )

    def test_de_read_result_carries_attempt_fields(self):
        result = PathDecisionResult(
            request_key=_KEY,
            path=PathKind.DE_READ,
            reverse_attempt_id=3,
            prefill_local_tokens=_ATTEMPT_TOKENS,
        )
        assert result.to_dict() == {
            "request_key": _KEY.to_dict(),
            "path": "DE_READ",
            "reverse_attempt_id": 3,
            "prefill_local_tokens": _ATTEMPT_TOKENS,
        }
        assert PathDecisionResult.from_dict(result.to_dict()) == result

        with pytest.raises(PathDecisionValidationError):
            PathDecisionResult(request_key=_KEY, path=PathKind.DE_READ)
        with pytest.raises(PathDecisionValidationError):
            PathDecisionResult.from_dict({"request_key": _KEY.to_dict(), "path": "DE_READ"})

    def test_reverse_wire_id_unique_per_attempt(self):
        assert _reverse_wire_id(0) != _reverse_wire_id(1)
        assert _reverse_wire_id(0) == _reverse_wire_id(0)

        # The Forward wire identity stays logical and stable.
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
            get_external_request_id,
        )

        binding = metadata_module.ForwardReceiveBinding(
            request_key=_KEY,
            path=PathKind.DE_READ,
            wire_request_id=get_external_request_id("decode-request-1123456789"),
            decode_request_id="decode-request-1",
            destination_block_ids=((20, 21, 22, 23),),
            token_start=32,
            token_end=64,
        )
        assert binding.wire_request_id == "decode-request-1"

    def test_decider_stamps_attempt_identity_from_num_preemptions(self):
        request = PathDecisionRequest(
            request_key=_KEY,
            target_tokens=49,
            decode_local_tokens=0,
            decode_store_tokens=32,
        )
        de_read_result = PathDecisionDecider(lambda_policy(PathKind.DE_READ)).decide(request, 16, num_preemptions=3)
        assert de_read_result.path is PathKind.DE_READ
        assert de_read_result.reverse_attempt_id == 3
        assert de_read_result.prefill_local_tokens == 16

        pe_read_result = PathDecisionDecider(lambda_policy(PathKind.PE_READ)).decide(request, 0, num_preemptions=3)
        assert pe_read_result.path is PathKind.PE_READ
        assert pe_read_result.reverse_attempt_id is None
        assert pe_read_result.prefill_local_tokens is None


def lambda_policy(path: PathKind):
    class _Policy:
        def choose(self, request: PathDecisionRequest) -> PathKind:
            return path

    return _Policy()


class TestAttemptKeyedInstallers:
    def test_installers_accept_attempt_m_while_retaining_attempt_n_tombstones(self):
        worker = _make_prefill_worker()
        binding_n = _make_reverse_binding(reverse_attempt_id=0)
        worker._install_reverse_receive_binding(binding_n)

        # Consume attempt 0's DONE terminal: tombstone recorded for its wire id.
        raw_done = {binding_n.wire_request_id}
        worker._consume_reverse_wire_terminals(raw_done, set(), set(), set())
        assert binding_n.wire_request_id in worker._consumed_reverse_terminal_wire_ids

        # A greater attempt installs alongside; attempt 0's tombstone survives.
        binding_m = _make_reverse_binding(reverse_attempt_id=1, reverse_completion_job_id=42)
        worker._install_reverse_receive_binding(binding_m)
        assert worker._reverse_receive_bindings[_attempt_key(1)] == binding_m
        assert worker._consumed_reverse_terminal_wire_ids[binding_n.wire_request_id] == _attempt_key(0)

    def test_equal_attempt_conflicting_binding_still_raises(self):
        worker = _make_prefill_worker()
        binding = _make_reverse_binding(reverse_attempt_id=0)
        worker._install_reverse_receive_binding(binding)

        # Identical retry is idempotent.
        worker._install_reverse_receive_binding(binding)

        conflicting = replace(binding, token_end=48)
        with pytest.raises(RuntimeError):
            worker._install_reverse_receive_binding(conflicting)

    def test_old_tracker_removed_only_under_section5_rule(self):
        worker = _make_worker()
        attempt_key = _attempt_key(0)
        plan = _make_reverse_plan(reverse_attempt_id=0)
        tracker = worker_module._SplitTracker(
            store_phase=worker_module._SplitPhase.SKIPPED,
            reverse_phase=worker_module._SplitPhase.PENDING,
            forward_phase=worker_module._SplitPhase.PENDING,
            store_destination_slice=(),
            forward_destination_slice=(70, 71),
            reverse_plan=plan,
            reverse_submitted_attempt=attempt_key,
            store_load_failed=False,
            terminal_published=False,
        )
        decode_request_id = _KEY.decode_request_id
        worker._split_trackers[decode_request_id] = tracker
        worker._pending_local_reverse_terminals[attempt_key] = True

        # Terminal not yet consumed and attempt not complete: the finish
        # retains the tracker and its tombstones.
        worker._release_split_request_state({decode_request_id})
        assert decode_request_id in worker._split_trackers
        assert attempt_key in worker._pending_local_reverse_terminals

        # Terminal consumed and the attempt normally complete: removal takes
        # the tracker and its tombstones together.
        worker._drain_local_reverse_terminals()
        assert tracker.reverse_phase is worker_module._SplitPhase.DONE
        worker._release_split_request_state({decode_request_id})
        assert decode_request_id not in worker._split_trackers
        assert attempt_key not in worker._pending_local_reverse_terminals

    def test_tp_gt1_reverse_mapping_assertion_holds_attempt_keyed(self):
        worker = _make_worker()
        plan = _make_reverse_plan(reverse_attempt_id=0, reverse_send_job_id=7)

        def two_mappings(*args):
            return {
                (plan.remote_host, plan.remote_port): {
                    "local_block_ids": [11],
                    "remote_block_ids": [71],
                    "trans_count": 1,
                },
                ("203.0.113.5", 6100): {
                    "local_block_ids": [12],
                    "remote_block_ids": [72],
                    "trans_count": 1,
                },
            }

        with (
            patch.object(layerwise_module, "get_cp_group", return_value=[[0]]),
            patch.object(layerwise_module, "context_parallel_parameters_check"),
            patch.object(
                layerwise_module,
                "get_local_remote_block_port_mappings",
                return_value=({}, {}, {}, {}),
            ),
            patch.object(layerwise_module, "get_transfer_mappings", side_effect=two_mappings),
            pytest.raises(RuntimeError, match="multiple transfer tasks"),
        ):
            worker._build_reverse_send_metadata(plan, _KEY.decode_request_id)

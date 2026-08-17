# SPDX-License-Identifier: Apache-2.0
"""Current-attempt filtering for Reverse completion reports."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm.v1.outputs import KVConnectorOutput

from tests.ut.distributed.kv_transfer.dual_path.conftest import (
    make_block_pool,
    make_worker_metadata,
)
from tests.ut.distributed.kv_transfer.dual_path.test_pe_read_forward import (
    _blocks,
    _make_request,
)
from tests.ut.distributed.kv_transfer.dual_path.test_reverse_attempt_identity import (
    _attempt_key,
    _make_reverse_binding,
)
from tests.ut.distributed.kv_transfer.dual_path.test_split_lifecycle import (
    _make_prefill_worker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import PathKind


def _admit_de_read_request(scheduler, request=None):
    if request is None:
        request = _make_request(
            target_tokens=48,
            prompt_tokens=49,
            local_tokens=16,
            store_tokens=32,
            destination_block_ids=[[20, 21, 22, 23]],
        )
    assert scheduler.get_num_new_matched_tokens(request, 16) == (16, True)
    scheduler.update_state_after_alloc(request, _blocks(([70, 71],)), 16)
    return request


def test_worker_emits_only_reverse_receive_completion_id():
    worker = _make_prefill_worker()
    binding = _make_reverse_binding(reverse_attempt_id=0)
    worker._install_reverse_receive_binding(binding)
    worker.kv_recv_layer_thread.get_and_clear_done_requests.return_value = {binding.wire_request_id}

    done_sending, done_recving = worker.get_finished(set(), DualPathConnectorMetadata())

    # No request id leaves the worker for a Reverse terminal.
    assert binding.prefill_request_id not in done_recving
    assert done_recving == set()
    worker_metadata = worker.build_connector_worker_meta()
    assert worker_metadata.completion_reports == {binding.reverse_receive_completion_id: 1}
    assert worker_metadata.failure_reports == {}


def test_stale_attempt_completion_absorbed_never_reaches_finished_recving(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read_request(scheduler)
    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    completion_id = binding.reverse_receive_completion_id
    metadata = scheduler.build_connector_meta(MagicMock(name="scheduler_output"))
    assert metadata.reverse_receive_bindings[0].reverse_receive_completion_id == completion_id
    assert scheduler._prefill_pending_reverse_receive_bindings == {}

    # The logical request has re-parked at a newer attempt: the old attempt's
    # completion report is stale and must be absorbed.
    scheduler._waiting_reverse_attempt_ids[request.request_id] = _attempt_key(1, binding.request_key)
    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
    scheduler.update_connector_output(output)

    assert output.finished_recving is None
    assert output.finished_sending is None
    assert scheduler._completion_tracker.get(completion_id) is None


def test_current_attempt_completion_inserts_req_id_only_while_waiting(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read_request(scheduler)
    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    completion_id = binding.reverse_receive_completion_id
    assert scheduler._waiting_reverse_attempt_ids[request.request_id] == _attempt_key(0, binding.request_key)

    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
    scheduler.update_connector_output(output)

    # The current attempt matches, so the request id enters finished_recving.
    assert output.finished_recving == {request.request_id}
    assert pool.blocks[71].ref_cnt == 0
    assert request.request_id not in scheduler._waiting_reverse_attempt_ids

    # A late duplicate report hits the closed completion and is ignored.
    late_output = KVConnectorOutput(
        kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1})
    )
    scheduler.update_connector_output(late_output)
    assert late_output.finished_recving is None


def test_current_attempt_completion_for_running_request_does_not_insert(pe_scheduler_factory):
    pool = make_block_pool()
    scheduler, _ = pe_scheduler_factory(PathKind.DE_READ, pool=pool)
    request = _admit_de_read_request(scheduler)
    binding = scheduler._prefill_pending_reverse_receive_bindings[request.request_id]
    completion_id = binding.reverse_receive_completion_id

    # The request already left WAITING_FOR_REMOTE_KVS (it is RUNNING): the
    # gate finds no waiting entry and absorbs the report, avoiding the
    # upstream finished-status assertion crash.
    del scheduler._waiting_reverse_attempt_ids[request.request_id]
    output = KVConnectorOutput(kv_connector_worker_meta=make_worker_metadata(completion_reports={completion_id: 1}))
    scheduler.update_connector_output(output)

    assert output.finished_recving is None

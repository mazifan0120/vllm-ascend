# SPDX-License-Identifier: Apache-2.0
"""Stage-2 W8: facade forwarding matrix and MultiConnector compatibility."""

from __future__ import annotations

from unittest.mock import MagicMock

from vllm.v1.outputs import KVConnectorOutput

from tests.ut.distributed.kv_transfer.dual_path.conftest import make_worker_metadata
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (
    DualPathConnector,
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.metadata import (
    DualPathConnectorMetadata,
    DualPathWorkerMetadata,
)


def _bare_facade():
    connector = object.__new__(DualPathConnector)
    scheduler = object.__new__(DualPathConnectorScheduler)
    worker = object.__new__(DualPathConnectorWorker)
    connector.connector_scheduler = scheduler
    connector.connector_worker = worker
    connector._connector_metadata = DualPathConnectorMetadata()
    return connector, scheduler, worker


def test_facade_forwards_every_safety_hook():
    connector, scheduler, worker = _bare_facade()
    scheduler.bind_gpu_block_pool = MagicMock(name="bind_gpu_block_pool")
    scheduler.update_connector_output = MagicMock(name="update_connector_output")
    scheduler.request_finished = MagicMock(name="request_finished", return_value=(True, {"k": "v"}))
    scheduler.shutdown = MagicMock(name="scheduler_shutdown")
    worker.build_connector_worker_meta = MagicMock(name="build_connector_worker_meta")
    worker.get_finished = MagicMock(name="get_finished", return_value=({"sent"}, {"received"}))
    worker.shutdown = MagicMock(name="worker_shutdown")

    pool = object()
    connector.bind_gpu_block_pool(pool)
    scheduler.bind_gpu_block_pool.assert_called_once_with(pool)

    output = KVConnectorOutput()
    connector.update_connector_output(output)
    scheduler.update_connector_output.assert_called_once_with(output)

    worker_metadata = DualPathWorkerMetadata(completion_reports={1: 1})
    worker.build_connector_worker_meta.return_value = worker_metadata
    assert connector.build_connector_worker_meta() is worker_metadata

    assert connector.get_finished(set()) == ({"sent"}, {"received"})
    worker.get_finished.assert_called_once()

    request = MagicMock(name="request")
    assert connector.request_finished(request, [1]) == (True, {"k": "v"})
    scheduler.request_finished.assert_called_once_with(request, [1])

    connector.shutdown()
    scheduler.shutdown.assert_called_once()
    worker.shutdown.assert_called_once()


def test_multiconnector_aggregation_preserves_dual_path_worker_metadata_type():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorWorkerMetadata,
    )

    first = MultiKVConnectorWorkerMetadata(metadata=(None, make_worker_metadata(completion_reports={1: 1})))
    second = MultiKVConnectorWorkerMetadata(metadata=(None, make_worker_metadata(completion_reports={1: 1, 2: 1})))

    merged = first.aggregate(second)

    dual_path_slot = merged.metadata[1]
    assert type(dual_path_slot) is DualPathWorkerMetadata
    assert dual_path_slot.completion_reports == {1: 2, 2: 1}

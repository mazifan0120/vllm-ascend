# SPDX-License-Identifier: Apache-2.0


def test_scheduler_symbols_have_one_definition_and_legacy_exports():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.scheduler import (
        DecodeKVSnapshot,
        DualPathConnectorScheduler,
    )

    assert connector.DecodeKVSnapshot is DecodeKVSnapshot
    assert connector.DualPathConnectorScheduler is DualPathConnectorScheduler


def test_worker_symbol_has_one_definition_and_legacy_export():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path import connector
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker import (
        DualPathConnectorWorker,
    )

    assert connector.DualPathConnectorWorker is DualPathConnectorWorker

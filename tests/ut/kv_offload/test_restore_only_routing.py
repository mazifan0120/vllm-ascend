# SPDX-License-Identifier: Apache-2.0
"""Restore-only scheduler routing tests for receipt-mode decode.

The D scheduler marks a request whose entire prompt is covered by the target
lease as restore-only. The multi-connector routes that request to the
AscendStore restore driver instead of a normal model forward, and the Store
scheduler fills the full-range restore bounds the driver requires.

These cases need vllm/torch/msgspec to import the connector modules, so they
run in the NPU/CI environment. The local CPU lane covers the receipt ledger
and wire transport regressions that do not depend on the connector imports.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("msgspec")

from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (  # noqa: E402
    AscendMultiConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (  # noqa: E402
    MooncakeLayerwiseConnectorMetadata,
    ReqMeta,
)


def _restore_only_req_meta(req_id="restore-1", prompt_len=128):
    return ReqMeta(
        local_block_ids=[[1, 2]],
        token_ids=[],
        remote_block_ids=[],
        remote_block_size=[],
        remote_engine_id=None,
        remote_host=None,
        remote_port=None,
        remote_te_rpc_port=None,
        remote_layer_metadata=None,
        metaserver=None,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        prompt_len=prompt_len,
        request_instance="restore-instance",
        restore_only=True,
        restore_end_token=prompt_len,
        token_len_chunk=prompt_len,
    )


def _plain_req_meta(req_id="plain-1", prompt_len=64):
    return ReqMeta(
        local_block_ids=[[1]],
        token_ids=[],
        remote_block_ids=[],
        remote_block_size=[],
        remote_engine_id=None,
        remote_host=None,
        remote_port=None,
        remote_te_rpc_port=None,
        remote_layer_metadata=None,
        metaserver=None,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        prompt_len=prompt_len,
        request_instance="",
    )


class _FakeStore:
    """AscendStore child stand-in exposing the restore driver entry."""

    def __init__(self):
        self.connector_worker = SimpleNamespace(restore_kv_for_transfer=MagicMock())
        self.restore_calls = []
        self._connector_metadata = None

    def restore_kv_for_transfer(self, metadata):
        self.restore_calls.append(metadata)

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeLayerwise:
    """MooncakeLayerwise child stand-in carrying receipt metadata."""

    def __init__(self, metadata):
        self._connector_metadata = metadata


def _multi_connector_with_children(children):
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    connector._connectors = children
    return connector


def test_restore_only_requests_collected_from_layerwise_metadata():
    metadata = MooncakeLayerwiseConnectorMetadata()
    metadata.requests["restore-1"] = _restore_only_req_meta("restore-1")
    metadata.requests["plain-1"] = _plain_req_meta("plain-1")
    connector = _multi_connector_with_children([_FakeLayerwise(metadata), _FakeStore()])

    marked = connector._receipt_restore_only_requests()
    assert marked == {"restore-1"}


def test_store_restore_connector_locates_child_with_driver():
    store = _FakeStore()
    layerwise = _FakeLayerwise(MooncakeLayerwiseConnectorMetadata())
    connector = _multi_connector_with_children([layerwise, store])

    assert connector._store_restore_connector() is store


def test_start_load_kv_routes_restore_only_to_store_driver():
    metadata = MooncakeLayerwiseConnectorMetadata()
    metadata.requests["restore-1"] = _restore_only_req_meta("restore-1")
    store = _FakeStore()
    store._connector_metadata = SimpleNamespace(requests={"restore-1": object()})
    connector = _multi_connector_with_children([_FakeLayerwise(metadata), store])

    # No super().start_load_kv call: the Store restore driver is the only path.
    with pytest.MonkeyPatch.context() as mp:
        super_start = MagicMock()
        mp.setattr(
            AscendMultiConnector.__mro__[1],
            "start_load_kv",
            super_start,
            raising=False,
        )
        connector.start_load_kv(forward_context=None)

    assert len(store.restore_calls) == 1
    assert store.restore_calls[0] is store._connector_metadata
    super_start.assert_not_called()


def test_start_load_kv_falls_through_when_no_restore_only_requests():
    metadata = MooncakeLayerwiseConnectorMetadata()
    metadata.requests["plain-1"] = _plain_req_meta("plain-1")
    store = _FakeStore()
    connector = _multi_connector_with_children([_FakeLayerwise(metadata), store])

    with pytest.MonkeyPatch.context() as mp:
        super_start = MagicMock()
        mp.setattr(
            AscendMultiConnector.__mro__[1],
            "start_load_kv",
            super_start,
            raising=False,
        )
        connector.start_load_kv(forward_context=None)

    assert store.restore_calls == []
    super_start.assert_called_once()


def test_start_load_kv_fails_closed_without_store_driver():
    metadata = MooncakeLayerwiseConnectorMetadata()
    metadata.requests["restore-1"] = _restore_only_req_meta("restore-1")
    # No store child present.
    connector = _multi_connector_with_children([_FakeLayerwise(metadata)])

    with pytest.raises(RuntimeError, match="AscendStore connector"):
        connector.start_load_kv(forward_context=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

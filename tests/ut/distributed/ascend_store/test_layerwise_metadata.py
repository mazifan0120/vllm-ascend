# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual connector metadata without importing NPU workers."""

import dataclasses
import json
import pickle
import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_metadata import LayerMetadata, ReqMeta


class TestLayerwiseMetadata(unittest.TestCase):
    def legacy_arguments(self):
        return ([[1, 3]], None, [[5, 7]], [[128]], "decode", "host", 31000, 32000, None, None, 2, 1, 1)

    def test_legacy_required_fields_keep_their_positional_order(self):
        metadata = ReqMeta(*self.legacy_arguments(), True, 129)
        self.assertEqual(metadata.local_block_ids, [[1, 3]])
        self.assertEqual(metadata.remote_tp_size, 2)
        self.assertTrue(metadata.chunk_finish)
        self.assertEqual(metadata.prompt_len, 129)
        self.assertIsNone(metadata.remote_te_rpc_endpoint)
        self.assertFalse(metadata.restore_only)

    def test_transport_extensions_follow_all_legacy_fields(self):
        names = [f.name for f in dataclasses.fields(ReqMeta)]
        self.assertGreater(names.index("remote_te_rpc_endpoint"), names.index("request_instance"))
        self.assertGreater(names.index("restore_only"), names.index("request_instance"))

    def test_metadata_round_trip_preserves_component_geometry_and_identity(self):
        metadata = ReqMeta(*self.legacy_arguments())
        metadata.remote_layer_metadata = {"main": LayerMetadata([0], [4096], [256], [1])}
        metadata.request_instance = "generation-a"
        metadata.remote_te_rpc_endpoint = "host:32000"
        self.assertEqual(pickle.loads(pickle.dumps(metadata)), metadata)
        encoded = json.loads(json.dumps(dataclasses.asdict(metadata)))
        self.assertEqual(encoded["request_instance"], "generation-a")
        self.assertEqual(encoded["remote_layer_metadata"]["main"]["block_len"], [256])


if __name__ == "__main__":
    unittest.main()

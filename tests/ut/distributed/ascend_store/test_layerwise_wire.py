# SPDX-License-Identifier: Apache-2.0
"""Round-trip actual target contracts through a JSON metaserver payload."""

import copy
import json
import unittest
from dataclasses import replace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_target_owner_runtime import OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    ContractError,
    assert_pd_consistent,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_wire import (
    TARGET_LEASE_WIRE_VERSION,
    target_lease_from_wire,
    target_lease_to_wire,
)


class TestLayerwiseTargetWire(unittest.TestCase):
    def setUp(self):
        self.fixture = OwnerFixture()
        self.lease = self.fixture.acquire()
        self.addCleanup(self.fixture.finish, self.lease)
        self.wire = target_lease_to_wire(self.lease)

    def test_http_json_round_trip_preserves_the_acquired_lease(self):
        body = {"request_instance": self.lease.request.request_instance, "remote_target_lease": self.wire}
        received = json.loads(json.dumps(body))
        restored = target_lease_from_wire(received["remote_target_lease"])
        self.assertEqual(restored, self.lease)
        self.assertEqual(restored.writer_plan, self.lease.writer_plan)
        self.assertEqual(restored.views, self.lease.views)
        self.assertTrue(all(region.transport_endpoint for region in restored.regions))

    def test_schema_versions_are_checked_without_bool_coercion(self):
        for field, invalid in (
            ("wire_version", (True, TARGET_LEASE_WIRE_VERSION + 1, "2")),
            ("contract_version", (True, 2, "1")),
        ):
            for wrong in invalid:
                with self.subTest(field=field, wrong=wrong):
                    wire = copy.deepcopy(self.wire)
                    wire[field] = wrong
                    with self.assertRaises(ContractError):
                        target_lease_from_wire(wire)

    def test_block_maps_are_shared_with_allocation_and_legacy_arrays_still_decode(self):
        self.assertTrue(all(type(view["block_ids"]) is dict for view in self.wire["views"]))
        old = copy.deepcopy(self.wire)
        old["wire_version"] = 1
        for view in old["views"]:
            group = view["block_ids"]["allocation_group"]
            view["block_ids"] = list(self.lease.allocation.group_block_ids[group])
        self.assertEqual(target_lease_from_wire(old), self.lease)
        for invalid in (True, -1, len(self.lease.allocation.group_block_ids), "0"):
            wire = copy.deepcopy(self.wire)
            wire["views"][0]["block_ids"] = {"allocation_group": invalid}
            with self.subTest(group=invalid), self.assertRaises(ContractError):
                target_lease_from_wire(wire)
        old["views"][0]["block_ids"] = {"allocation_group": 0}
        with self.assertRaises(ContractError):
            target_lease_from_wire(old)

    def test_shared_block_map_is_revalidated_at_each_distinct_capacity(self):
        wire = copy.deepcopy(self.wire)
        last_view = self.lease.views[-1]
        self.assertIs(self.lease.views[0].block_ids, last_view.block_ids)
        block_capacity = max(last_view.block_ids)
        wire["views"][-1]["block_capacity"] = block_capacity
        with self.assertRaisesRegex(ContractError, "invalid block ID"):
            target_lease_from_wire(wire)
        invalid = replace(self.lease, views=(*self.lease.views[:-1], replace(last_view, block_capacity=block_capacity)))
        with self.assertRaisesRegex(ContractError, "invalid block ID"):
            target_lease_to_wire(invalid)

    def test_mismatched_request_identity_is_rejected(self):
        wire = copy.deepcopy(self.wire)
        wire["allocation"]["request"]["request_instance"] = "another-request"
        with self.assertRaises(ContractError):
            target_lease_from_wire(wire)

    def test_owner_objects_and_local_addresses_are_not_accepted(self):
        wire = copy.deepcopy(self.wire)
        wire["owner"] = self.fixture.facade
        with self.assertRaises(ContractError):
            target_lease_from_wire(wire)
        wire = copy.deepcopy(self.wire)
        wire["regions"][0]["base_address"] = 4096
        with self.assertRaises(ContractError):
            target_lease_from_wire(wire)

    def test_invalid_component_blocks_and_byte_strides_are_rejected(self):
        for field, wrong in (("component", "guessed-main"), ("block_ids", [True]), ("strides_bytes", [0])):
            with self.subTest(field=field):
                wire = copy.deepcopy(self.wire)
                wire["views"][0][field] = wrong
                with self.assertRaises(ContractError):
                    target_lease_from_wire(wire)

    def test_writer_plan_requires_explicit_matching_backings(self):
        wire = copy.deepcopy(self.wire)
        wire["writer_plan"]["ranges_by_backing"] = {}
        with self.assertRaises(ContractError):
            target_lease_from_wire(wire)

    def test_pd_gate_checks_the_producer_dtype_not_only_the_destination(self):
        with self.assertRaisesRegex(ContractError, "Producer MAIN dtype"):
            assert_pd_consistent(self.lease, producer_tp_size=1, producer_block_sizes=(16,), producer_main_dtype="int8")
        with self.assertRaises(ContractError):
            assert_pd_consistent(
                self.lease, producer_tp_size=True, producer_block_sizes=(16,), producer_main_dtype="bfloat16"
            )


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
import pickle
import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    CacheView,
    Component,
    ContractError,
    ReceiveReceipt,
    RegisteredRegion,
    RequestKey,
    TokenRange,
    WriterPlan,
    assert_provider_capabilities,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_provider import (
    RegisteredTargetProvider,
    TargetBinding,
    binding_from_dsa_layout,
    provider_from_registered_dsa_layouts,
)


class TestRegisteredTargetProvider(unittest.TestCase):
    def make(self):
        main = RegisteredRegion("engine", 1, "main-region", "shared-host", 1, 0, 1024, "host", "host:1")
        index = RegisteredRegion("engine", 1, "index-region", "rank0-index", 1, 0, 512, "device", "host:1")
        self.regions = (main, index)
        self.bindings = (
            TargetBinding(CacheView(Component.MAIN, 0, main.region_id, "bfloat16", (64,), 16, 32, 8, 0, ()), 0, 48),
            TargetBinding(CacheView(Component.INDEXER, 0, index.region_id, "int8", (32,), 16, 32, 8, 0, ()), 1, 16),
        )
        self.runtime = SimpleNamespace(retain=MagicMock(), make_visible=MagicMock(), release=MagicMock())
        provider = self.construct()
        request = RequestKey("request-instance", 3)
        allocation = BlockAllocation(request, 7, ((1, 3), (2, 5)), 32, 47, "scheduler-retention")
        return provider, request, allocation

    def construct(self):
        return RegisteredTargetProvider(
            rank=0,
            tp_size=2,
            regions=self.regions,
            bindings=self.bindings,
            required_layers={Component.MAIN: (0,), Component.INDEXER: (0,)},
            writer_plan=WriterPlan({"shared-host": (0, 1)}, {"shared-host": (TokenRange(0, 256),)}),
            runtime=self.runtime,
        )

    def receipt(self, request):
        return ReceiveReceipt(request, 7, True, True, True)

    def test_binds_existing_component_blocks_and_roundtrips_as_pure_data(self):
        provider, request, allocation = self.make()
        assert_provider_capabilities(provider)
        lease = provider.acquire_target(request, allocation)
        self.assertEqual(lease.allocation, allocation)
        self.assertEqual(provider.get_target_views(lease)[0].block_ids, (1, 3))
        self.assertEqual(provider.get_target_views(lease)[1].block_ids, (2, 5))
        self.assertEqual(pickle.loads(pickle.dumps(lease)), lease)
        self.runtime.retain.assert_called_once_with(lease)

    def test_missing_retention_or_short_component_fails_before_runtime_retain(self):
        provider, request, allocation = self.make()
        for invalid in (
            replace(allocation, retention_token=None),
            replace(allocation, group_block_ids=((1, 3), (2,))),
            replace(allocation, group_block_ids=((1, 1), (2, 5))),
            replace(allocation, group_block_ids=((1, 8), (2, 5))),
            replace(allocation, manager_block_size_tokens=16),
        ):
            with self.assertRaises(ContractError):
                provider.acquire_target(request, invalid)
        self.runtime.retain.assert_not_called()

    def test_idempotent_acquire_cannot_change_allocation(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        self.assertEqual(provider.acquire_target(request, allocation), lease)
        self.runtime.retain.assert_called_once()
        with self.assertRaises(ContractError):
            provider.acquire_target(request, replace(allocation, allocation_generation=8))

    def test_active_blocks_are_not_reassigned_on_cancel_or_failed_release(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        new = RequestKey("next-instance", 0)
        self.runtime.release.side_effect = RuntimeError("writeback still active")
        with self.assertRaisesRegex(RuntimeError, "writeback"):
            provider.release_target(lease)
        with self.assertRaises(ContractError):
            provider.acquire_target(new, replace(allocation, request=new))
        self.runtime.release.side_effect = None
        provider.release_target(lease)
        newer = provider.acquire_target(new, replace(allocation, request=new))
        provider.release_target(lease)  # late duplicate is harmless
        self.assertEqual(provider.get_target_views(newer), newer.views)
        with self.assertRaises(ContractError):
            provider.acquire_target(request, allocation)

    def test_runtime_retain_failure_is_not_published(self):
        provider, request, allocation = self.make()
        self.runtime.retain.side_effect = RuntimeError("retention expired")
        with self.assertRaisesRegex(RuntimeError, "expired"):
            provider.acquire_target(request, allocation)
        with self.assertRaises(ContractError):
            provider.acquire_target(request, allocation)
        self.runtime.make_visible.assert_not_called()

    def test_prepare_validates_receipt_before_wiring_and_never_sets_visible(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        receipt = self.receipt(request)
        for invalid in (
            replace(receipt, main_union_ok=False),
            replace(receipt, indexer_ranks_ok=False),
            replace(receipt, terminals_ok=False),
            replace(receipt, target_generation=6),
            replace(receipt, request=RequestKey("old", 3)),
            replace(receipt, visible=True),
        ):
            with self.assertRaises(ContractError):
                provider.prepare_decode(lease, invalid)
        self.runtime.make_visible.assert_not_called()
        provider.prepare_decode(lease, receipt)
        provider.prepare_decode(lease, receipt)
        self.runtime.make_visible.assert_called_once_with(lease, receipt)
        self.assertFalse(receipt.visible)

    def test_visibility_failure_is_terminal_but_can_release_safely(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        self.runtime.make_visible.side_effect = RuntimeError("no receive fence")
        with self.assertRaisesRegex(RuntimeError, "fence"):
            provider.prepare_decode(lease, self.receipt(request))
        with self.assertRaises(ContractError):
            provider.prepare_decode(lease, self.receipt(request))
        provider.release_target(lease)
        self.runtime.release.assert_called_once_with(lease)

    def test_returned_mutable_writer_plan_cannot_change_registered_identity(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        lease.writer_plan.writers_by_backing.clear()
        with self.assertRaises(ContractError):
            provider.get_target_views(lease)
        good = provider.acquire_target(request, allocation)
        self.assertTrue(good.writer_plan.writers_by_backing)

    def test_layout_bounds_memory_kind_and_required_layers_are_checked(self):
        self.make()
        original = self.bindings
        for change in (
            replace(original[0], page_bytes=65),
            replace(original[0], view=replace(original[0].view, region_offset_bytes=100)),
            replace(original[0], view=replace(original[0].view, region_binding="index-region")),
            replace(original[0], view=replace(original[0].view, logical_layer=1)),
        ):
            self.bindings = (change, original[1])
            with self.assertRaises(ContractError):
                self.construct()

    def test_release_keeps_only_identity_tombstone_and_rejects_changed_old_lease(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        provider.release_target(lease)
        self.assertFalse(provider._states)
        self.assertIsInstance(provider._retired[request], str)
        changed = replace(lease, allocation=replace(allocation, allocation_generation=8))
        with self.assertRaises(ContractError):
            provider.release_target(changed)

    def test_visibility_callback_cannot_mark_a_failed_receipt_visible(self):
        provider, request, allocation = self.make()
        lease = provider.acquire_target(request, allocation)
        receipt = self.receipt(request)

        def fail(lease, local_receipt):
            local_receipt.visible = True
            raise RuntimeError("failed visibility dependency")

        self.runtime.make_visible.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "failed visibility"):
            provider.prepare_decode(lease, receipt)
        self.assertFalse(receipt.visible)

    def test_overlapping_same_backing_registration_is_rejected(self):
        self.make()
        self.regions = (*self.regions, replace(self.regions[0], region_id="second-host-region"))
        with self.assertRaisesRegex(ContractError, "overlap"):
            self.construct()

    def test_int8_main_target_is_not_claimed_as_supported_by_bf16_runtime(self):
        self.make()
        self.bindings = (replace(self.bindings[0], view=replace(self.bindings[0].view, dtype="int8")), self.bindings[1])
        with self.assertRaisesRegex(ContractError, "BF16"):
            self.construct()

    def test_missing_real_runtime_operations_fail_at_construction(self):
        self.make()
        self.runtime = SimpleNamespace(retain=lambda lease: None)
        with self.assertRaisesRegex(ContractError, "make_visible"):
            self.construct()

    def test_dsa_layout_adapter_uses_page_scale_and_existing_registration_offset(self):
        self.make()
        layout = SimpleNamespace(
            base=1100, stride=64, scale=2, block_tokens=32, dtype="bfloat16", capacity=4, block_bytes=48
        )
        binding = binding_from_dsa_layout(
            layout, component=Component.MAIN, logical_layer=0, group=0, region=self.regions[0], region_base_address=1000
        )
        self.assertEqual(binding.view.region_offset_bytes, 100)
        self.assertEqual(binding.view.page_size_tokens, 16)
        self.assertEqual(binding.page_bytes, 48)
        with self.assertRaises(ContractError):
            binding_from_dsa_layout(
                layout,
                component=Component.MAIN,
                logical_layer=0,
                group=0,
                region=self.regions[0],
                region_base_address=1200,
            )

    def test_catalog_factory_consumes_existing_dsa_layouts_without_allocating(self):
        from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_dsa_transfer import DsaCacheLayout

        self.make()
        layouts = (
            [DsaCacheLayout("index", 0, 2000, 16, 32, 2, 32, "int8", 8)],
            [DsaCacheLayout("main", 0, 1000, 48, 64, 2, 32, "bfloat16", 8)],
        )
        arguments = dict(
            rank=0,
            tp_size=2,
            regions=self.regions,
            region_base_addresses={"main-region": 1000, "index-region": 2000},
            layouts=layouts,
            layer_groups={"main": (0, 0), "index": (0, 1)},
            required_layers={Component.MAIN: (0,), Component.INDEXER: (0,)},
            writer_plan=WriterPlan({"shared-host": (0, 1)}, {"shared-host": (TokenRange(0, 256),)}),
            runtime=self.runtime,
        )
        provider = provider_from_registered_dsa_layouts(**arguments)
        self.assertEqual(
            {binding.view.component for binding in provider._bindings}, {Component.MAIN, Component.INDEXER}
        )
        self.runtime.retain.assert_not_called()
        arguments["region_base_addresses"] = {"main-region": 1100, "index-region": 2000}
        with self.assertRaisesRegex(ContractError, "complete registered region"):
            provider_from_registered_dsa_layouts(**arguments)

    def test_lifecycle_callbacks_do_not_hold_state_lock(self):
        provider, request, allocation = self.make()
        entered, release, probe_done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def retain(lease):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test callback timed out")

        self.runtime.retain.side_effect = retain

        def acquire():
            try:
                provider.acquire_target(request, allocation)
            except Exception as error:
                errors.append(error)

        def probe():
            try:
                provider.acquire_target(request, allocation)
            except ContractError:
                probe_done.set()

        first = threading.Thread(target=acquire)
        second = threading.Thread(target=probe)
        first.start()
        try:
            self.assertTrue(entered.wait(5))
            second.start()
            self.assertTrue(probe_done.wait(2))
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(errors)
        self.assertFalse(first.is_alive())


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Batch admission capabilities preserve resource and completion ownership."""

import pickle
import unittest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.distributed.ascend_store.test_target_owner_runtime import Completion, OwnerFixture
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_contract import (
    BlockAllocation,
    Component,
    ContractError,
    ReceiveReceipt,
    ReleaseAck,
    RequestKey,
    Status,
    TransferResult,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_plan import ManifestWritePlan
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_coordinator import MultiRankTargetProvider
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_delivery import WorkerTargetDelivery
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_runtime import NpuTargetStreams


class TestTargetBatchExecution(unittest.TestCase):
    def test_native_fence_adapter_covers_graph_update_and_deduplicates_streams(self):
        recorded = []

        class Npu:
            current = None

            @contextmanager
            def device(self, ordinal):
                yield

            @contextmanager
            def stream(self, stream):
                previous, self.current = self.current, stream
                try:
                    yield
                finally:
                    self.current = previous

            class Event:
                def record(self):
                    recorded.append(npu.current)

        npu = Npu()
        consume, update, writeback = [SimpleNamespace(device=SimpleNamespace(index=0)) for _ in range(3)]
        manager = SimpleNamespace(target_runtime_streams=lambda: (update, writeback, update))
        with patch.object(torch, "npu", npu):
            streams = NpuTargetStreams(manager, 0, consume)
            events = streams.record_use()
        self.assertEqual(len(events), 3)
        self.assertEqual([id(stream) for stream in recorded], [id(consume), id(update), id(writeback)])

    def test_delivery_binds_at_admission_and_uses_the_local_batch_path(self):
        f = OwnerFixture()
        provider = MultiRankTargetProvider((f.facade,))
        aggregate = provider.acquire_target(f.request, f.allocation)
        manifest = provider.manifest(aggregate)
        local = manifest.for_rank(0).lease
        installed = []
        delivery = WorkerTargetDelivery(0, f.facade, lambda *args: installed.append(args))
        delivery.admit_target(manifest, "internal")
        ledger = installed[0][2]
        for key in ManifestWritePlan(manifest).receive_keys(0):
            ledger.record(TransferResult(key, Status.SUCCESS, True))
        ledger.finish_rank(f.request, 0, Status.SUCCESS, True)
        f.ready(local)
        ledger.mark_visible(0)
        with (
            patch.object(f.runtime, "_state", side_effect=AssertionError("hot descriptor comparison")),
            delivery.use_context(("internal",), {"internal": f.allocation}),
        ):
            self.assertEqual(f.runtime._states[f.request].users, 1)
        f.finish(local)

    def pair(self):
        f = OwnerFixture()
        first = f.acquire()
        f.ready(first)
        request = RequestKey("second-owner-test", 3)
        f.scheduler.ids["second"] = ((0, 2),)
        allocation = f.retention.issue("second", BlockAllocation(request, 7, ((0, 2),), 16, 23))
        second = f.facade.acquire_target(request, allocation)
        f.runtime.confirm_native_received(second, ReceiveReceipt(request, 7, True, True, True))
        f.facade.prepare_decode(second, ReceiveReceipt(request, 7, True, True, True))
        bindings = (f.facade.bind_use(first), f.facade.bind_use(second))
        return f, first, second, allocation, bindings

    @staticmethod
    def finish_pair(f, first, second, allocation):
        f.facade.release_target(second)
        f.retention.retire(allocation, ReleaseAck(True, allocation.retention_token))
        f.finish(first)

    def test_batch_checks_shared_pool_and_records_one_fence_without_lease_comparisons(self):
        f, first, second, allocation, bindings = self.pair()
        entries = tuple(zip(bindings, (f.allocation, allocation)))
        with (
            patch.object(f.runtime, "_state", side_effect=AssertionError("hot full-lease check")),
            patch.object(f.resources, "_view_signature", side_effect=AssertionError("hot view sort")),
            patch.object(f.resources, "validate_allocation", wraps=f.resources.validate_allocation) as validate,
            patch.object(f.streams, "record_use", wraps=f.streams.record_use) as record,
        ):
            before = f.streams.installs
            with f.facade.use_batch(entries):
                self.assertEqual([binding.state.users for binding in bindings], [1, 1])
                self.assertEqual(f.streams.installs - before, 2)
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(record.call_count, 1)
        self.assertIs(bindings[0].state.events[-1], bindings[1].state.events[-1])
        self.assertEqual([binding.state.users for binding in bindings], [0, 0])
        self.finish_pair(f, first, second, allocation)

    def test_changed_allocation_or_duplicate_binding_fails_before_any_batch_user(self):
        f, first, second, allocation, bindings = self.pair()
        bad = replace(allocation, group_block_ids=((1, 2),))
        for entries in (
            ((bindings[0], f.allocation), (bindings[1], bad)),
            ((bindings[0], f.allocation), (bindings[0], f.allocation)),
        ):
            with self.subTest(entries=entries), self.assertRaises(ContractError), f.facade.use_batch(entries):
                self.fail("invalid batch executed")
            self.assertEqual([binding.state.users for binding in bindings], [0, 0])
        self.finish_pair(f, first, second, allocation)

    def test_binding_is_local_revocable_and_not_serializable(self):
        f = OwnerFixture()
        lease = f.acquire()
        f.ready(lease)
        binding = f.facade.bind_use(lease)
        with self.assertRaises(TypeError):
            pickle.dumps(binding)
        other = OwnerFixture()
        with self.assertRaisesRegex(ContractError, "another owner"), other.facade.use_batch(((binding, f.allocation),)):
            self.fail("foreign owner binding executed")
        other.facade.close()
        other.te.unregister_buffer()
        other.pool.close()
        f.facade.release_target(lease)
        with self.assertRaisesRegex(ContractError, "stale"), f.facade.use_batch(((binding, f.allocation),)):
            self.fail("retired binding executed")
        f.retention.retire(f.allocation, ReleaseAck(True, f.allocation.retention_token))
        f.facade.close()
        f.te.unregister_buffer()
        f.pool.close()

    def test_forged_lease_is_rejected_when_binding(self):
        f = OwnerFixture()
        lease = f.acquire()
        changed = replace(lease.views[0], region_offset_bytes=lease.views[0].region_offset_bytes + 1)
        with self.assertRaises(ContractError):
            f.facade.bind_use(replace(lease, views=(changed, *lease.views[1:])))
        f.finish(lease)

    def test_storage_or_cache_reference_change_after_admission_is_detected(self):
        for change in ("storage", "reference"):
            with self.subTest(change=change):
                f = OwnerFixture()
                lease = f.acquire()
                f.ready(lease)
                binding = f.facade.bind_use(lease)
                name = next(name for name in f.caches if f.components[name] is Component.INDEXER)
                old = f.caches[name]
                tensor = old[0].tensor
                if change == "storage":
                    old[0].tensor = torch.empty_like(tensor)
                else:
                    f.caches[name] = old[::-1]
                with (
                    self.assertRaisesRegex(ContractError, "allocation or layout changed"),
                    f.facade.use_batch(((binding, f.allocation),)),
                ):
                    self.fail("changed registration executed")
                old[0].tensor = tensor
                f.caches[name] = old
                f.finish(lease)

    def test_model_exception_retains_every_request_until_shared_writeback_completes(self):
        f, first, second, allocation, bindings = self.pair()
        completion = Completion()
        f.streams.pending.append(completion)
        with (
            self.assertRaisesRegex(ValueError, "model failed"),
            f.facade.use_batch(tuple(zip(bindings, (f.allocation, allocation)))),
        ):
            raise ValueError("model failed")
        for lease in (first, second):
            with self.assertRaises(TimeoutError):
                f.facade.release_target(lease)
        completion.event.set()
        self.finish_pair(f, first, second, allocation)

    def test_record_failure_requires_recovery_fences_for_every_request(self):
        f, first, second, allocation, bindings = self.pair()
        with (
            patch.object(f.streams, "record_use", side_effect=RuntimeError("fence failed")),
            self.assertRaisesRegex(RuntimeError, "fence failed"),
            f.facade.use_batch(tuple(zip(bindings, (f.allocation, allocation)))),
        ):
            pass
        self.assertTrue(all(binding.state.needs_recovery_fence for binding in bindings))
        self.assertEqual([binding.state.users for binding in bindings], [0, 0])
        self.finish_pair(f, first, second, allocation)

    def test_registration_generation_change_is_detected_after_binding(self):
        f = OwnerFixture()
        lease = f.acquire()
        f.ready(lease)
        binding = f.facade.bind_use(lease)
        f.te._registration_generation += 1
        with (
            self.assertRaisesRegex(ContractError, "registration"),
            f.facade.use_batch(((binding, f.allocation),)),
        ):
            self.fail("changed registration executed")
        f.te._registration_generation -= 1
        f.finish(lease)

    def test_active_batch_rejects_request_release_and_owner_teardown(self):
        f, first, second, allocation, bindings = self.pair()
        with f.facade.use_batch(tuple(zip(bindings, (f.allocation, allocation)))):
            for lease in (first, second):
                with self.assertRaises(ContractError):
                    f.facade.release_target(lease)
            with self.assertRaises(ContractError):
                f.facade.close()
            with self.assertRaises(RuntimeError):
                f.te.unregister_buffer()
            with self.assertRaises(RuntimeError):
                f.pool.close()
        self.finish_pair(f, first, second, allocation)


if __name__ == "__main__":
    unittest.main()

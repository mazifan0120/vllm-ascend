# SPDX-License-Identifier: Apache-2.0
import importlib.util
import pickle
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostMemoryRegion,
    HostPoolTopology,
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.utils.allocation_lifetime import AllocationLifetime

# ascend_store mocks the TE module globally. Load the real implementation under
# an isolated name so this test does not change the other Store tests' stubs.
_te_path = Path(__file__).resolve().parents[4] / "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py"
_te_spec = importlib.util.spec_from_file_location("_target_owner_actual_te", _te_path)
_te_module = importlib.util.module_from_spec(_te_spec)
sys.modules[_te_spec.name] = _te_module
_te_spec.loader.exec_module(_te_module)
GlobalTE = _te_module.GlobalTE


class Engine:
    def __init__(self):
        self.registered = []
        self.removed = []
        self.fail_remove = False
        self.fail_register = False
        self._rpc_port = 0

    def get_rpc_port(self):
        return self._rpc_port

    def register_memory(self, address, size, location):
        if self.fail_register:
            return -1
        self.registered.append((address, size, location))
        return 0

    def unregister_memory(self, address):
        if self.fail_remove:
            return -1
        self.removed.append(address)
        return 0


class TestTargetLifetime(unittest.TestCase):
    def test_pinned_registration_cannot_be_unregistered_and_stale_snapshot_is_rejected(self):
        te = GlobalTE()
        te.transfer_engine = Engine()
        te.register_buffer([4096], [1024])
        first = te.registration_snapshot()
        pin = te.retain_registration(first)
        with self.assertRaisesRegex(RuntimeError, "active ownership pins"):
            te.unregister_buffer()
        self.assertEqual(te.transfer_engine.removed, [])
        pin.release()
        pin.release()
        te.unregister_buffer()
        self.assertEqual(te.transfer_engine.removed, [4096])
        te.register_buffer([4096], [1024])
        self.assertGreater(te.registration_snapshot().generation, first.generation)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            te.retain_registration(first)

    def test_failed_registration_and_partial_unregistration_are_not_catalogs(self):
        te = GlobalTE()
        te.transfer_engine = Engine()
        te.transfer_engine.fail_register = True
        with self.assertRaises(RuntimeError):
            te.register_buffer([4096], [1024])
        with self.assertRaisesRegex(RuntimeError, "complete live"):
            te.registration_snapshot()
        te.transfer_engine.fail_register = False
        te.register_buffer([4096], [1024])
        snapshot = te.registration_snapshot()
        te.transfer_engine.fail_remove = True
        with self.assertRaises(RuntimeError):
            te.unregister_buffer()
        with self.assertRaises(RuntimeError):
            te.retain_registration(snapshot)
        te.transfer_engine.fail_remove = False
        te.unregister_buffer()

    def test_pool_close_obeys_real_owner_pin(self):
        import torch

        released = []
        tensor = SimpleNamespace(dtype=torch.int8, is_contiguous=lambda: True)
        region = HostMemoryRegion(tensor, handle="actual-pool-handle", release_callback=released.append)
        pool = MooncakeHostPool(region, HostPoolTopology(0, 1))
        pin = pool.retain_target_allocation()
        with self.assertRaisesRegex(RuntimeError, "active ownership pins"):
            pool.close()
        self.assertEqual(released, [])
        self.assertFalse(region._released)
        pin.release()
        pool.close()
        pool.close()
        self.assertEqual(released, ["actual-pool-handle"])
        with self.assertRaises(RuntimeError):
            pool.retain_target_allocation()

    def test_failed_destroy_does_not_admit_new_owners_and_can_retry(self):
        life = AllocationLifetime("allocation")
        with self.assertRaises(ValueError):
            life.close(lambda: (_ for _ in ()).throw(ValueError("destroy failed")))
        with self.assertRaises(RuntimeError):
            life.retain()
        completed = []
        life.close(lambda: completed.append(True))
        self.assertEqual(completed, [True])

    def test_pin_release_is_thread_safe_and_not_serializable(self):
        life = AllocationLifetime("allocation")
        pin = life.retain()
        with self.assertRaises(TypeError):
            pickle.dumps(pin)
        threads = [threading.Thread(target=pin.release) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(life.pins, 0)
        life.close(lambda: None)

    def test_registration_snapshot_keeps_local_addresses_out_of_wire(self):
        te = GlobalTE()
        te.transfer_engine = Engine()
        te.register_buffer([4096], [1024])
        with self.assertRaises(TypeError):
            pickle.dumps(te.registration_snapshot())

# SPDX-License-Identifier: Apache-2.0
"""Host diagnostics preserve model exceptions and owner completion paths."""

import threading
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.target_timing import (
    ReceiptHostTiming,
    install_receipt_host_timing,
)


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 100
        return self.value


class Resources:
    def __init__(self):
        self._te = SimpleNamespace(registration_snapshot=lambda: (1,))

    @staticmethod
    def _tensor_signature(tensor):
        return tensor

    def validate(self, lease):
        self._te.registration_snapshot()
        self._tensor_signature(1)


class Streams:
    def __init__(self):
        self.records = 0

    def install_visibility(self, dependency):
        return None

    def record_use(self):
        self.records += 1
        return (object(),)


class Runtime:
    def __init__(self):
        self._resources = Resources()
        self._streams = Streams()
        self.users = 0

    def _state(self, lease):
        if lease.bad:
            raise RuntimeError("bad lease")

    @contextmanager
    def use(self, lease):
        self._state(lease)
        self._resources.validate(lease)
        self._streams.install_visibility(None)
        self.users += 1
        try:
            yield
        finally:
            self._streams.record_use()
            self.users -= 1


class Delivery:
    def __init__(self, runtime, prompt=55000):
        self._lock = threading.RLock()
        lease = SimpleNamespace(allocation=SimpleNamespace(prompt_tokens=prompt), bad=False)
        self._by_request_id = {"r": "key"}
        self._deliveries = {"key": SimpleNamespace(lease=lease)}
        self.runtime = runtime

    @contextmanager
    def use_context(self, request_ids, scheduled_allocations=None):
        with ExitStack() as stack:
            for request_id in request_ids:
                stack.enter_context(self.runtime.use(self._deliveries[self._by_request_id[request_id]].lease))
            yield


class TestReceiptHostTiming(unittest.TestCase):
    def make(self, *, prompt=55000, interval=1):
        runtime = Runtime()
        delivery = Delivery(runtime, prompt)
        records = []
        timer = ReceiptHostTiming(2, interval=interval, emit=records.append, clock=Clock())
        timer.install(runtime, delivery)
        return runtime, delivery, records, timer

    def test_disabled_install_does_not_access_an_owner(self):
        install_receipt_host_timing(None, None, {})
        install_receipt_host_timing(None, None, {"receipt_timing_interval": 0})

    def test_success_reports_distinct_entry_body_exit_and_nested_metrics(self):
        runtime, delivery, records, timer = self.make()
        with delivery.use_context(("r",)):
            self.assertEqual(runtime.users, 1)
        self.assertEqual(runtime.users, 0)
        self.assertEqual(runtime._streams.records, 1)
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual((row["rank"], row["phase"], row["requests"], row["batches"]), (2, "target", 1, 1))
        for label in (
            "lease_state",
            "resources_validate",
            "tensor_signature",
            "registration_snapshot",
            "visibility",
            "record_use",
        ):
            self.assertEqual(row[label + "_calls"], 1)
            self.assertGreater(row[label + "_ns"], 0)
        self.assertGreater(row["enter_ns"], 0)
        self.assertGreater(row["body_wall_ns"], 0)
        self.assertGreater(row["exit_ns"], 0)
        self.assertIsNone(timer._frame())

    def test_body_failure_propagates_and_records_real_completion(self):
        runtime, delivery, records, timer = self.make()
        with self.assertRaisesRegex(ValueError, "model failed"), delivery.use_context(("r",)):
            raise ValueError("model failed")
        self.assertEqual(runtime.users, 0)
        self.assertEqual(runtime._streams.records, 1)
        self.assertEqual(records[0]["failures"], 1)
        self.assertIsNone(timer._frame())

    def test_owner_entry_failure_is_not_hidden_or_given_a_completion(self):
        runtime, delivery, records, timer = self.make()
        delivery._deliveries["key"].lease.bad = True
        with self.assertRaisesRegex(RuntimeError, "bad lease"), delivery.use_context(("r",)):
            pass
        self.assertEqual(runtime._streams.records, 0)
        self.assertEqual(records[0]["failures"], 1)
        self.assertIsNone(timer._frame())

    def test_short_functional_requests_are_not_included(self):
        runtime, delivery, records, timer = self.make(prompt=100)
        with delivery.use_context(("r",)):
            pass
        self.assertFalse(records)
        self.assertEqual(runtime._streams.records, 1)

    def test_windows_and_draft_phase_are_distinguished(self):
        runtime, delivery, records, timer = self.make(interval=2)

        def propose_draft_token_ids():
            with delivery.use_context(("r",)):
                pass

        propose_draft_token_ids()
        self.assertFalse(records)
        propose_draft_token_ids()
        self.assertEqual(records[0]["phase"], "draft")
        self.assertEqual(records[0]["batches"], 2)
        self.assertEqual(records[0]["interval_calls"], 1)
        self.assertEqual(runtime._streams.records, 2)

    def test_invalid_settings_are_rejected(self):
        for interval in (0, -1, True, 1.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                ReceiptHostTiming(0, interval=interval)

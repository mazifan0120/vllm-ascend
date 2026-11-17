# SPDX-License-Identifier: Apache-2.0
"""A failed listener must not leave its worker waiting forever."""

import threading
import unittest
from types import SimpleNamespace

import zmq

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from vllm_ascend.distributed.kv_transfer.utils.thread_ready import wait_for_thread_ready


class TestThreadReady(unittest.TestCase):
    def test_real_bind_collision_is_reported_instead_of_waiting_forever(self):
        context = zmq.Context()
        occupied = context.socket(zmq.ROUTER)
        port = occupied.bind_to_random_port("tcp://127.0.0.1")
        ready = threading.Event()
        errors = []

        def start_listener():
            socket = context.socket(zmq.ROUTER)
            try:
                socket.bind(f"tcp://127.0.0.1:{port}")
                ready.set()
            except zmq.ZMQError as error:
                errors.append(error.errno)
            finally:
                socket.close(linger=0)

        thread = threading.Thread(target=start_listener, name="conflicting receiver")
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "conflicting receiver.*exited"):
                wait_for_thread_ready(thread, ready, timeout=2)
            self.assertEqual(errors, [zmq.EADDRINUSE])
            self.assertFalse(ready.is_set())
        finally:
            thread.join(2)
            occupied.close(linger=0)
            context.term()

    def test_started_transport_signals_ready(self):
        ready, finish = threading.Event(), threading.Event()

        def serve():
            ready.set()
            finish.wait(1)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            wait_for_thread_ready(thread, ready, timeout=1)
        finally:
            finish.set()
            thread.join(1)

    def test_dead_listener_is_rejected_even_if_ready_was_signaled(self):
        thread = SimpleNamespace(name="receiver", is_alive=lambda: False)
        for signaled in (False, True):
            ready = threading.Event()
            if signaled:
                ready.set()
            with self.subTest(signaled=signaled), self.assertRaisesRegex(RuntimeError, "receiver.*exited"):
                wait_for_thread_ready(thread, ready, timeout=0.01)

    def test_live_but_unready_transport_times_out(self):
        thread = SimpleNamespace(name="receiver", is_alive=lambda: True)
        with self.assertRaisesRegex(TimeoutError, "receiver.*ready"):
            wait_for_thread_ready(thread, threading.Event(), timeout=0.01)

    def test_invalid_timeout_is_rejected(self):
        for timeout in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                wait_for_thread_ready(None, None, timeout=timeout)

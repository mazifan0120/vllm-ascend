# SPDX-License-Identifier: Apache-2.0
"""Bound transport-thread startup waits without treating failure as ready."""

import math
import time

DEFAULT_THREAD_READY_TIMEOUT_S = 60.0
_READY_POLL_INTERVAL_S = 0.1


def wait_for_thread_ready(thread, ready_event, *, timeout=DEFAULT_THREAD_READY_TIMEOUT_S):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Transport startup timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Transport thread {thread.name} did not become ready within {timeout}s")
        ready = ready_event.wait(min(_READY_POLL_INTERVAL_S, remaining))
        if not thread.is_alive():
            raise RuntimeError(f"Transport thread {thread.name} exited during startup; check its startup exception")
        if ready:
            return

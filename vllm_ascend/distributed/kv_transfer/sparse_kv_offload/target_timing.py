# SPDX-License-Identifier: Apache-2.0
"""Opt-in host timing of real receipt execution without device synchronization.

The diagnostic wraps only the bound worker's instances. With the connector
option absent there are no wrappers in the model execution path. Wall times
inside the model body include enqueue, planner and any existing waits; they
are not device kernel durations. Nested metrics must not be added together.
"""

import json
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps

from vllm.logger import logger


class ReceiptHostTiming:
    def __init__(self, rank, *, interval=64, min_prompt_tokens=50000, emit=None, clock=None):
        if type(interval) is not int or interval < 1:
            raise ValueError("receipt_timing_interval must be a positive integer")
        if type(min_prompt_tokens) is not int or min_prompt_tokens < 0:
            raise ValueError("receipt_timing_min_prompt_tokens must be a nonnegative integer")
        self.rank = rank
        self.interval = interval
        self.min_prompt_tokens = min_prompt_tokens
        self._clock = clock or time.perf_counter_ns
        self._emit = emit or (lambda value: logger.info("RECEIPT_HOST_TIMING %s", json.dumps(value)))
        self._local = threading.local()
        self._lock = threading.Lock()
        self._windows = {}
        self._previous_start = {}

    def _frame(self):
        return getattr(self._local, "frames", [None])[-1]

    def _wrap_method(self, owner, name, label):
        original = getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            frame = self._frame()
            if frame is None:
                return original(*args, **kwargs)
            start = self._clock()
            try:
                return original(*args, **kwargs)
            finally:
                frame[label + "_ns"] += self._clock() - start
                frame[label + "_calls"] += 1

        setattr(owner, name, timed)

    @staticmethod
    def _phase():
        frame = sys._getframe(1)
        for _ in range(16):
            if frame is None:
                break
            if frame.f_code.co_name == "propose_draft_token_ids":
                return "draft"
            frame = frame.f_back
        return "target"

    def _record(self, phase, frame):
        with self._lock:
            window = self._windows.setdefault(phase, defaultdict(int))
            for key, value in frame.items():
                window[key] += value
            window["batches"] += 1
            if window["batches"] < self.interval:
                return
            report = dict(window, rank=self.rank, phase=phase)
            self._windows[phase] = defaultdict(int)
        self._emit(report)

    def install(self, runtime, delivery):
        self._wrap_method(runtime, "_state", "lease_state")
        self._wrap_method(runtime._resources, "validate", "resources_validate")
        self._wrap_method(runtime._resources, "_tensor_signature", "tensor_signature")
        self._wrap_method(runtime._resources._te, "registration_snapshot", "registration_snapshot")
        self._wrap_method(runtime._streams, "install_visibility", "visibility")
        self._wrap_method(runtime._streams, "record_use", "record_use")
        original = delivery.use_context

        @contextmanager
        def timed_context(request_ids, scheduled_allocations=None):
            # The same request list is still validated by the original owner.
            # This read only chooses whether to collect diagnostic statistics.
            request_ids = tuple(request_ids)
            with delivery._lock:
                prompts = [
                    delivery._deliveries[key].lease.allocation.prompt_tokens
                    for request_id in request_ids
                    if (key := delivery._by_request_id.get(request_id)) in delivery._deliveries
                ]
            selected = bool(prompts) and min(prompts) >= self.min_prompt_tokens
            if not selected:
                with original(request_ids, scheduled_allocations):
                    yield
                return
            phase = self._phase()
            frame = defaultdict(int, requests=len(prompts))
            previous_frames = getattr(self._local, "frames", [None])
            self._local.frames = [*previous_frames, frame]
            start = self._clock()
            previous = self._previous_start.get(phase)
            if previous is not None:
                frame["interval_ns"] = start - previous
                frame["interval_calls"] = 1
            self._previous_start[phase] = start
            entered = exit_start = None
            try:
                with original(request_ids, scheduled_allocations):
                    entered = self._clock()
                    frame["enter_ns"] = entered - start
                    try:
                        yield
                    finally:
                        exit_start = self._clock()
                        frame["body_wall_ns"] = exit_start - entered
            except BaseException:
                frame["failures"] += 1
                raise
            finally:
                end = self._clock()
                if exit_start is not None:
                    frame["exit_ns"] = end - exit_start
                elif entered is None:
                    frame["enter_ns"] = end - start
                self._local.frames = previous_frames
                self._record(phase, frame)

        delivery.use_context = timed_context


def install_receipt_host_timing(facade, delivery, extra_config):
    interval = extra_config.get("receipt_timing_interval", 0)
    if interval == 0:
        return
    timing = ReceiptHostTiming(
        delivery._rank,
        interval=interval,
        min_prompt_tokens=extra_config.get("receipt_timing_min_prompt_tokens", 50000),
    )
    with facade._operation() as owner:
        timing.install(owner.runtime, delivery)

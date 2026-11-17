#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager

import torch

_lock = threading.RLock()
_attention_compute_start_gate: AttentionComputeStartGate | None = None


class AttentionComputeCompletion:
    """One layer execution's completion, never reset or reused by a new step.

    Unlike a KV-ready event, this event must be recorded after the last
    attention read is enqueued. Waiting for an unrecorded device event is not
    sufficient: first wait for the host to publish that record, then wait for
    the device work. A failure wakes waiters without reporting safe reuse.
    """

    def __init__(self, event: torch.npu.Event | None = None) -> None:
        self._event = event if event is not None else torch.npu.Event()
        self._condition = threading.Condition()
        self._record_started = False
        self._recorded = False
        self._error: Exception | None = None

    def record(self) -> None:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Attention completion failed") from self._error
            if self._record_started:
                return
            self._record_started = True
        try:
            self._event.record()
        except Exception as exc:
            self.fail(exc)
            raise
        with self._condition:
            self._recorded = True
            self._condition.notify_all()

    def fail(self, error: Exception) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            if not self._condition.wait_for(lambda: self._recorded or self._error is not None, timeout):
                return False
            if self._error is not None:
                raise RuntimeError("Attention completion failed") from self._error
        # Do not hold a lock needed by record/fail while synchronizing NPU work.
        # As with the existing attention start gate, timeout bounds the host
        # publication wait; device synchronization uses the runtime's guarantee.
        self._event.synchronize()
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Attention completion failed") from self._error
        return True


class AttentionComputeStartGate:
    """Gate that opens when the compute stream reaches attention.

    The attention worker records an NPU event immediately before submitting the
    attention op. MemCache worker threads wait for that event to complete before
    submitting H2D/L2G work, so transfer starts when the compute stream is
    actually at the attention boundary rather than merely after the Python call
    site was reached.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.on_start: Callable[[], None] | None = None
        self.on_finish: Callable[[], None] | None = None
        self._event: torch.npu.Event | None = None

    def record(
        self,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        stream = stream or torch.npu.current_stream()
        event = torch.npu.Event()
        event.record(stream)
        with self._condition:
            if self._event is None:
                self._event = event
                self._condition.notify_all()
                if self.on_start is not None:
                    self.on_start()

    def cancel(self) -> None:
        """Release gated tasks after their caller has set its transfer abort flag."""
        self.on_start = None
        self.on_finish = None
        self.record()

    def wait(self, timeout: float = 10.0) -> bool:
        with self._condition:
            while self._event is None:
                if not self._condition.wait(timeout=timeout):
                    return False
            event = self._event

        event.synchronize()
        return True


def reset_attention_compute_start_gate() -> AttentionComputeStartGate:
    """Create a new per-layer gate for MemCache work.

    Layerwise prefetch tasks keep a reference to the gate that was current when
    they were submitted. The attention path opens that same gate when attention
    compute is about to be launched.
    """
    global _attention_compute_start_gate
    gate = AttentionComputeStartGate()
    with _lock:
        _attention_compute_start_gate = gate
    return gate


def get_attention_compute_start_gate() -> AttentionComputeStartGate:
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        gate = reset_attention_compute_start_gate()
    return gate


def record_attention_compute_start() -> None:
    """Record the compute-stream boundary immediately before attention."""
    with _lock:
        gate = _attention_compute_start_gate
    if gate is not None:
        gate.record()


@contextmanager
def attention_transfer_window():
    """Run the configured transfer completion policy after attention submission.

    The event is recorded after cache writes and any preceding collectives. The
    host then submits the attention kernel. The registered connector callback
    decides whether to drain, bound, or retain its asynchronous transfer work.
    """
    with _lock:
        gate = _attention_compute_start_gate
    if gate is None:
        yield
        return
    try:
        gate.record()
        yield
    finally:
        if gate.on_finish is not None:
            gate.on_finish()

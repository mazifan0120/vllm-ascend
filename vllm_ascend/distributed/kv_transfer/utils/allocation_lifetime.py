# SPDX-License-Identifier: Apache-2.0
"""Local ownership pins for allocations and registrations.

Pins are local runtime objects, never wire values. Explicit close/unregister
must respect them; merely keeping a tensor or a Python handle is insufficient.
"""

import threading
from collections.abc import Callable


class AllocationPin:
    def __init__(self, owner: "AllocationLifetime") -> None:
        self._owner = owner
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if not self._released:
                self._owner._release()
                self._released = True

    def __getstate__(self):
        raise TypeError("Allocation pins must stay in their owner process")


class AllocationLifetime:
    """Serialize pin acquisition against actual destruction."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._lock = threading.RLock()
        self._pins = 0
        self._closed = False
        self._closing = False

    def retain(self) -> AllocationPin:
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError(f"{self._name} is closed or closing")
            self._pins += 1
            return AllocationPin(self)

    def _release(self) -> None:
        with self._lock:
            if self._pins <= 0:
                raise RuntimeError(f"{self._name} pin count underflow")
            self._pins -= 1

    def close(self, destroy: Callable[[], None]) -> None:
        with self._lock:
            if self._closed:
                return
            if self._pins:
                raise RuntimeError(f"{self._name} still has {self._pins} active ownership pins")
            self._closing = True
            destroy()
            self._closed = True

    @property
    def pins(self) -> int:
        with self._lock:
            return self._pins

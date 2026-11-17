# SPDX-License-Identifier: Apache-2.0
"""Owner-local completion frontiers; no stream ordering is inferred for opaque events."""

from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)
class StreamCompletion:
    owner: object
    stream: object
    sequence: int
    event: object
    key: tuple = field(init=False)

    def __post_init__(self):
        # Keep the actual stream alive. Native handles cannot be recycled while
        # this completion owns the stream; equivalent wrappers share a key.
        object.__setattr__(self, "key", (self.owner, getattr(self.stream, "npu_stream", id(self.stream))))

    def query(self):
        return self.event.query()

    def synchronize(self):
        self.event.synchronize()

    def __getstate__(self):
        raise TypeError("Stream completions must remain in the worker owner")


class CompletionFrontier:
    """Newest ordered fence per stream; preserve every unclassified dependency.

    The owner runtime serializes updates with its state lock. A newer event on
    the same ordered stream dominates an older one even if contexts return out
    of order. Distinct/dynamic streams stay retained until actual release.
    No native query, wait or user callback executes while updating this state.
    """

    def __init__(self):
        self._ordered = {}
        self._opaque = []

    def extend(self, events):
        # The caller releases these references after dropping its state lock:
        # decref of a native Event can enqueue driver-side destruction work.
        superseded = []
        for event in events:
            if type(event) is StreamCompletion:
                previous = self._ordered.get(event.key)
                if previous is None or event.sequence > previous.sequence:
                    if previous is not None:
                        superseded.append(previous)
                    self._ordered[event.key] = event
            else:
                self._opaque.append(event)
        return superseded

    def __iter__(self):
        yield from self._ordered.values()
        yield from self._opaque

    def __len__(self):
        return len(self._ordered) + len(self._opaque)

    def __getitem__(self, index):
        return tuple(self)[index]

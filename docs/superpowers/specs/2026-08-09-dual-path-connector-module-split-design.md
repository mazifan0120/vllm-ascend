# DualPath Connector Module Split Design

## Goal

Split the current DualPath connector implementation by runtime ownership so
that the Scheduler, Worker, and connector facade can be reviewed independently,
without changing routing, transfer, completion, failure, or shutdown behavior.

## Scope

The target package layout is:

```text
dual_path/
├── connector.py
├── scheduler.py
├── worker.py
├── config.py
├── metadata.py
├── kvpool_adapter.py
├── path_decision.py
└── path_decision_channel.py
```

This change is a source-layout refactor only. It does not change protocol
payloads, state transitions, Store admission, PE/DE route selection, transfer
ordering, invalid-block publication, or public connector configuration.

## Module Ownership

### `scheduler.py`

Owns Scheduler-process behavior and Scheduler-only mutable state:

- `DualPathConnectorScheduler`
- `DecodeKVSnapshot`
- `DecodePathDecisionState`
- `_DecodeDecisionStatus`
- `_AdmissionLookup`
- Scheduler-only admission and decision helper functions

It may depend on configuration, metadata, path-decision, control-channel,
KVPool Scheduler adapter, and the parent Layerwise Scheduler. It must not import
`connector.py` or `worker.py`.

### `worker.py`

Owns Worker-process behavior and split-transfer mutable state:

- `DualPathConnectorWorker`
- `_SplitPhase`
- `_SplitTracker`
- Forward/Reverse binding, Store completion, terminal, and cleanup logic

It may depend on configuration, metadata, path-decision types, KVPool Worker
adapter, and the parent Layerwise Worker. It must not import `connector.py` or
`scheduler.py`.

### `connector.py`

Owns only the vLLM connector facade:

- `DualPathConnector`
- Scheduler/Worker construction by `KVConnectorRole`
- Facade-level `get_finished()` and `shutdown()` delegation
- Compatibility re-exports

The facade imports the Scheduler and Worker modules. The dependency direction
is therefore one-way:

```text
connector.py -> scheduler.py
             -> worker.py
```

## Compatibility Contract

These existing imports remain valid and resolve to the exact classes defined in
the new owning modules:

```python
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector import (
    DecodeKVSnapshot,
    DualPathConnector,
    DualPathConnectorScheduler,
    DualPathConnectorWorker,
)
```

`connector.py` defines `__all__` for those four names. Private implementation
dependencies such as `KVPoolSchedulerAdapter`, `PathDecisionCoordinator`,
`get_ip`, `time`, and `logger` are not compatibility exports. Tests patch those
symbols in `scheduler.py` or `worker.py`, where the implementation looks them
up after the split.

Private Scheduler and Worker state types move with their owners. Tests that
inspect `_DecodeDecisionStatus` import the Scheduler module directly rather
than treating the facade as their implementation namespace.

## Behavior Preservation

The class bodies move without semantic edits. In particular:

- Scheduler allocation and metadata construction order is unchanged.
- Worker binding installation remains ahead of completion reconciliation.
- Forward and Reverse plans retain their direction-specific gates.
- Store completion remains the Reverse submission gate for split DE reads.
- Early raw completion and failure state remains durable until mapping.
- Cancellation and shutdown retain their current drain-first behavior.
- The facade continues bypassing the parent `get_finished()` implementation in
  the same way as before the split.

## Testing

The first test establishes the new consumer-visible module contract and fails
before the modules exist:

- Scheduler symbols import from `dual_path.scheduler`.
- Worker symbols import from `dual_path.worker`.
- Legacy facade imports are object-identical to the new definitions.

Existing tests are then updated only where their patch target or private module
reference changes. Runtime assertions remain unchanged. Verification covers:

- the module-boundary compatibility test;
- the complete DualPath unit-test directory;
- AscendStore unit tests because the Worker embeds the KVPool adapter;
- focused Ruff, spelling, Markdown, and diff checks.

## Non-Goals

- No further decomposition inside Scheduler or Worker.
- No common state/helper module.
- No thin compatibility subclasses in `connector.py`.
- No renaming of protocols, metadata fields, state variables, or methods.
- No changes outside DualPath tests except those required by import ownership.

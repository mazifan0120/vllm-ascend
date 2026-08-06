# DualPath Stage 1 Task Series

## Status and authority

This directory is the working authority for all DualPath Stage 1 work after
the completed foundation Task-00.

The term **Task** describes an independently reviewable, testable, and
revertible delivery unit. A Task may later be delivered as one PR or split
across multiple PRs when review size requires it; PR numbering does not define
the architecture boundary.

The earlier [`dual-path-stage1`](../dual-path-stage1/README.md) directory is
retained for history and for the completed PR-00 evidence. Its PR-01 through
PR-07 decomposition is superseded by this directory. It must not be used as
the implementation order for remaining work unless a requirement is explicitly
re-admitted into a new Task detailed spec.

## Current baseline

- Task-00 through Task-03 source is present: the connector foundation, Decode
  admission snapshot, PE-owned decision policy, and direct PE-to-DE result
  channel are independently implemented and tested.
- Task-04 remains a detailed design. Its protocol-reconciliation section owns
  the small `PathDecisionResult` and received-result naming changes required
  before the real Scheduler control loop is complete.
- No Store load, Forward, Reverse, or composite completion path is active.
- The latest checked source baseline for Task-04 is
  `vllm-ascend@6761bb9c3f179e2c56f636c47051a9c31a98383d` with sibling
  `vllm@0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`.

## Documents

- [`TASKS.md`](TASKS.md) owns the Stage 1 Task graph, merge-state contracts,
  scope boundaries, and acceptance endpoints.
- Detailed Task specs will live under `tasks/` and may refine interfaces and
  tests without changing another Task's merge-state contract silently.
- [`Task-01 detailed spec`](tasks/TASK-01-decode-admission.md) defines the
  Decode admission path through final slot allocation and
  `WAITING_FOR_REMOTE_KVS`.
- [`Task-02 detailed spec`](tasks/TASK-02-decision-protocol.md) defines the
  minimal decision protocol, `Path`, fixed full-hit rule, and replaceable
  round-robin policy.
- [`Task-03 detailed spec`](tasks/TASK-03-direct-decision-channel.md) defines
  the nested bootstrap metadata and minimal direct PE-to-DE ZMQ result channel.
- [`Task-04 detailed spec`](tasks/TASK-04-scheduler-decision-control-loop.md)
  connects the real DE/PE Connector Scheduler hooks, without activating a KV
  data path.

## Decomposition rules

Every Task must satisfy all of the following:

1. End at an observable merge-state, not merely add unused types or helpers.
2. State which component owns every retained request record, Future, block
   mapping, transfer, and terminal cleanup path it introduces.
3. Keep Store lookup, final Scheduler accounting, path commitment, and Worker
   I/O authorization as distinct operations.
4. Keep Scheduler hooks non-blocking; direct-channel and HTTP operations belong
   to a `PathDecisionCoordinator` or Proxy coroutine rather than a Scheduler
   thread.
5. Do not activate a route until all of its data producers, data consumers,
   completion predicates, failure paths, and cleanup paths exist.
6. Extract parent Layerwise helpers only when the Task contains their first
   concrete DualPath consumer.
7. Include focused CPU tests in the owning Task. NPU acceptance belongs to the
   first Task that activates the corresponding production route.
8. Preserve ordinary `MooncakeLayerwiseConnector` and non-DualPath Proxy
   behavior unless a Task explicitly owns and tests a compatible extension.

## Change control

A detailed Task spec may split a Task when reviewers could reasonably approve
one resulting merge-state and reject the other. It must not merge Tasks merely
because they touch the same file. Any dependency, activation, or merge-state
change must update `TASKS.md` before implementation starts.

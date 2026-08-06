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

- Task-00 through Task-05 are the implementation baseline: connector
  foundation, Decode admission, non-full PE-owned decision policy, direct
  PE-to-DE result channel, Scheduler control loop, and controlled `PE_READ`
  Forward are treated as implemented.
- Task-06 is the next delivery. It changes Store-full from a PE-decided
  `DE_READ` result into a DE-local admission and completion path.
- Because Task-01 through Task-05 are already implemented, the Task-06 spec
  contains a normative backward-delta section that must be applied together
  with its Store runtime. These are implementation changes, not code changes
  made by this documentation revision.
- Reverse and non-full split `DE_READ` remain unimplemented Task-07/08 work.

## Documents

- [`TASKS.md`](TASKS.md) owns the Stage 1 Task graph, merge-state contracts,
  scope boundaries, and acceptance endpoints.
- Detailed Task specs will live under `tasks/` and may refine interfaces and
  tests without changing another Task's merge-state contract silently.
- [`Task-01 detailed spec`](tasks/TASK-01-decode-admission.md) defines the
  Decode admission path through final slot allocation and
  `WAITING_FOR_REMOTE_KVS`.
- [`Task-02 detailed spec`](tasks/TASK-02-decision-protocol.md) defines the
  minimal Store-non-full decision protocol, `Path`, and replaceable
  round-robin policy.
- [`Task-03 detailed spec`](tasks/TASK-03-direct-decision-channel.md) defines
  the nested bootstrap metadata and minimal direct PE-to-DE ZMQ result channel.
- [`Task-04 detailed spec`](tasks/TASK-04-scheduler-decision-control-loop.md)
  connects Store-non-full DE/PE Connector Scheduler hooks, without itself
  activating a KV data path.
- [`Task-05 detailed spec`](tasks/TASK-05-pe-read-forward.md) defines the
  controlled `PE_READ` Forward route, explicit Scheduler plan, DE completion
  binding, and inherited Worker reuse boundary.
- [`Task-06 detailed spec`](tasks/TASK-06-de-local-store-full.md) is
  self-contained and defines DE-local Store-full admission, commit, Worker
  load, terminal output, and every required Task-01 through Task-05 delta.

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
6. Reuse parent Layerwise behavior directly where practical. Extract a helper
   only when an owning Task proves a concrete consumer cannot use the inherited
   seam safely.
7. Include focused CPU tests in the owning Task. The first Task implementing a
   data route owns its controlled NPU acceptance even when production policy
   activation remains deferred.
8. Preserve ordinary `MooncakeLayerwiseConnector` and non-DualPath Proxy
   behavior unless a Task explicitly owns and tests a compatible extension.
9. Branch DE Store-full before Decision/Proxy/PE work. Only Store-non-full
   requests may consume path-policy or direct-decision-channel state.

## Change control

A detailed Task spec may split a Task when reviewers could reasonably approve
one resulting merge-state and reject the other. It must not merge Tasks merely
because they touch the same file. Any dependency, activation, or merge-state
change must update `TASKS.md` before implementation starts.

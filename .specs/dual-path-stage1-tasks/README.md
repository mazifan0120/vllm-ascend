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

- Task-00 implements the behavior-preserving DualPath connector foundation.
- Task-01 implements Decode admission through final slot allocation and
  `WAITING_FOR_REMOTE_KVS`.
- Task-02 implements non-full decision schemas, identity, and PE-owned policy.
- Task-03 implements direct PE-to-DE decision delivery with ACK and retry.
- Task-04 implements the Scheduler decision control loop and fail-closed state.
- Task-05 implements the controlled `PE_READ` Forward route.
- Task-06 implements DE-local Store-full admission and completion.
- Task-07 implements the bidirectional runtime for injected split plans.
- Task-08 activates eligibility-forced and policy-selected `PE_READ`, plus
  split `DE_READ`, completing the Stage 1 production route set.
- Task-00 through Task-08 are all implemented. NPU acceptance executions
  remain deferred hardware runs; their skeletons exist.

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
- [`Task-07 detailed spec`](tasks/TASK-07-bidirectional-split-runtime.md)
  defines the bidirectional Worker runtime and injected split-plan execution.
- [`Task-08 detailed spec`](tasks/TASK-08-non-full-production-activation.md)
  defines final non-full policy activation, eligibility, and production
  `PE_READ` and split `DE_READ` composition.

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

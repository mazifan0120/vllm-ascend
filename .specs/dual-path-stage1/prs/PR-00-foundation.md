# PR-00 DualPathConnector Foundation

- Series position: prerequisite
- Spec status: `PLANNED`
- Depends on: none
- Blocks: PR-01, PR-02
- Existing implementation: commit `78451d8f6`
- User-visible behavior: no new routing behavior
- Activation after merge: DualPath remains behavior-compatible with
  `MooncakeLayerwiseConnector`

## Goal

Introduce the smallest reviewable DualPathConnector foundation without active
path selection or data-plane changes.

## Merge-state contract

After this PR merges, `DualPathConnector` can be selected in configuration and
constructs its own Scheduler or Worker subclass, but request execution remains
identical to the inherited Mooncake Layerwise path. No Store probe, local-full
admission, cross-engine decision, Reverse transfer, or new Forward semantics are
active.

## In scope

- Register `DualPathConnector` with the vLLM Ascend KV connector registry.
- Add `DualPathConnector`, `DualPathConnectorScheduler`, and
  `DualPathConnectorWorker` subclasses.
- Construct the DualPath subclasses without double-initializing the parent
  runtime.
- Parse only configuration required by the approved Stage 1 topology and
  fail-fast rules.
- Verify inherited behavior and parent interface compatibility.

Implementation-task mapping: pre-existing foundation work before `DP-01`.

## Out of scope

- Store adapter composition.
- Active static path decisions.
- Value Function or LinkMonitor affecting execution.
- Proxy protocol changes.
- Bidirectional Worker capability.
- Any new transfer plan or completion lifecycle.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/__init__.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/__init__.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`

Before upstream review, remove or explicitly demote foundation configuration
for active Value Function, adaptive strategy, relay data-plane, and multi-path
slicing that is not part of the approved Stage 1 contract.

## Required tests

- Connector registration resolves `DualPathConnector`.
- PE and DE roles build the correct Scheduler/Worker subclass.
- Invalid role, `kv_role`, topology, and unsupported Stage 1 config fail fast.
- Inherited Scheduler and Worker behavior matches
  `MooncakeLayerwiseConnector` for equivalent configuration.
- Parent method-signature guards detect incompatible upstream changes.

Focused command:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
```

## Acceptance gates

- No active decision or additional I/O is reachable.
- The parent runtime initializes exactly once.
- Configuration contains no active promise that this PR cannot execute.
- Focused UT and formatting checks pass.

## Rollback contract

Reverting this PR removes the DualPath connector registration and leaves all
existing connector behavior unchanged.

## Review focus

- Is direct `KVConnectorBase_V1` initialization necessary and safe?
- Is the configuration surface the minimum needed by later approved PRs?
- Do inherited-behavior tests protect the intended no-op execution contract?

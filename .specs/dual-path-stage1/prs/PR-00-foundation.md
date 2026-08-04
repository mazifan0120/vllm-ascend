# PR-00 DualPathConnector Foundation

- Series position: prerequisite
- Spec status: `LOCAL_READY`
- Depends on: none
- Blocks: PR-01, PR-02
- Verified implementation revision: `9a12de829ee5`
- Implementation-task mapping: completed prerequisite; semantic task IDs start
  with PR-01 in `../TASKS.md`
- User-visible behavior: a new connector name with no new routing semantics
- Activation after merge: `DualPathConnector` is a behavior-preserving alias of
  `MooncakeLayerwiseConnector`
- Detailed implementation contract:
  [DETAILED-SPEC.md](PR-00-foundation/DETAILED-SPEC.md)

## Goal

Deliver the smallest upstream-reviewable DualPath connector seam that can run an
ordinary Mooncake Layerwise workload without changing its behavior.

## Merge-state contract

After this PR merges, `DualPathConnector` can be selected in configuration and
constructs its own Scheduler or Worker subclass. Replacing
`MooncakeLayerwiseConnector` with `DualPathConnector` for the same ordinary
Layerwise workload preserves Scheduler accounting, metadata, Worker transfer,
completion, invalid-block, cleanup, and failure behavior.

The foundation adds no Store probe, Store coverage candidate, cross-engine
decision, round-robin path selection, Reverse transfer, new Forward semantics,
or new completion lifecycle. CPU parity tests and a real NPU Layerwise smoke
test are both required before merge. Those local gates are recorded as PASS in
`../TRACKING.md`; the PR itself has not been opened.

## In scope

- Register `DualPathConnector` with the vLLM Ascend KV connector registry.
- Add `DualPathConnector`, `DualPathConnectorScheduler`, and
  `DualPathConnectorWorker` subclasses.
- Construct the DualPath subclasses without double-initializing the parent
  Scheduler, Worker, Mooncake runtime, Transfer Engine, threads, or KV buffers.
- Parse the foundation role as exactly `prefill` or `decode` while preserving the
  inherited Mooncake extra-configuration keys.
- Reject removed foundation promises such as Value Function, LinkMonitor, relay,
  planner, topology, and shadow configuration.
- Preserve all ordinary Layerwise lifecycle methods through inheritance.
- Verify construction, interface, behavior, error, and NPU execution parity.

## Out of scope

- Store adapters, Store coverage, token accounting, and `DE_READ`.
- `PathDecisionRequest`, `PathDecisionCommit`, decision RPC, or Proxy changes.
- Round-robin, Value Function, LinkMonitor, adaptive, or shadow decisions.
- Active first-positive accounting or any `DE_READ` result.
- Bidirectional Worker capability, Reverse, or new Forward plans.
- DualPath metadata, completion reconciliation, tombstones, or timeouts.
- MultiConnector behavior changes.
- Topology restrictions that the inherited Layerwise path does not require.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/__init__.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/__init__.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`
- `tests/e2e/nightly/multi_node/dual_path/test_foundation_parity.py`

No Store, Proxy, MultiConnector, or parent Mooncake Layerwise implementation file
is modified by this PR.

## Required tests

- Connector registration resolves `DualPathConnector`.
- `role=prefill` and `role=decode` build the correct Scheduler or Worker
  subclass and validate against the configured KV capability.
- Legacy `pe`/`de`, role/capability mismatches, and removed DualPath foundation
  fields fail fast.
- The base connector and the selected parent Scheduler or Worker initialize
  exactly once.
- Facade state and parent lifecycle method ownership match
  `MooncakeLayerwiseConnector`.
- Equivalent Scheduler and Worker inputs produce equivalent structured outputs
  and collaborator call traces.
- Existing ordinary Mooncake Layerwise regression tests pass.
- A real NPU remote-prefill smoke test produces the same tokens and terminal
  behavior for the parent and DualPath connector configurations.

Focused CPU commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
pytest -sv tests/ut/kv_offload/test_mooncake_layerwise_connector.py
bash format.sh ci
```

The exact NPU command, image, model, topology, and result are recorded in
`TRACKING.md` as merge evidence.

## Acceptance gates

- No active decision, new request state, additional I/O, or additional thread is
  reachable.
- The parent runtime, Transfer Engine, threads, and KV buffers initialize once.
- The foundation configuration contains no capability without a consumer.
- CPU interface, construction, behavior, and regression tests pass.
- The ordinary Layerwise NPU parity smoke test passes.
- Formatting and lint checks pass.
- The PR diff contains no Store, decision, round-robin, or new data-plane code.

## Rollback contract

Reverting this PR removes the DualPath connector registration and foundation
subclasses. Existing `MooncakeLayerwiseConnector` behavior and configuration
remain unchanged.

## Review focus

- Is direct `KVConnectorBase_V1` initialization necessary and proven equivalent
  to the parent facade initialization?
- Is every copied facade field explicitly listed and covered by a drift guard?
- Are all request lifecycle methods still inherited from the parent?
- Does the NPU smoke test prove a real drop-in replacement rather than only
  mocked construction?

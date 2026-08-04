# PR-01 Decode Store Coverage Foundation

- Series position: 1 of 7
- Spec status: `PLANNED`
- Depends on: PR-00
- Blocks: PR-03
- Implementation tasks: `DP-COV-01` through `DP-COV-04`
- User-visible behavior: none
- Activation after merge: coverage/probe foundation is testable, but no Store
  or P2P data path is authorized

## Goal

Represent Decode HBM and AscendStore coverage as side-effect-free candidate
facts without selecting or executing a DualPath route.

## Merge-state contract

After this PR merges, Decode can first distinguish HBM-complete requests from
requests that need external KV, then probe its local AscendStore and classify
the aligned result as full, partial, or miss. The result is an immutable
`StoreCoverage` associated with a pending probe handle.

Probe performs no HBM load and no P2P transfer. The handle can be aborted on
request finish, cancellation, error, or ordinary fall-through. There is no
`commit_after_alloc()` data path, no PE decision request, and no change to the
existing remote-prefill behavior after this PR.

## In scope

- HBM-first readiness check using `R = max(P - 1, 0)`.
- `StoreCoverage` fields, alignment, clamp, and full/partial/miss
  classification, including a typed unavailable probe status.
- Scheduler-side Store lookup through a DualPath-owned adapter.
- A minimal pure-lookup extraction in the existing Store Scheduler if the
  current lookup API cannot detach its temporary `LoadSpec` into the handle
  without leaking shared pending state.
- `StoreProbeHandle` identity and pending/aborted lifecycle.
- Exactly-once abort, duplicate-abort idempotency, and conflicting lifecycle
  rejection.
- Request-finish, cancellation, shutdown, and probe-error cleanup.
- Focused unit tests proving probe has no HBM load or transport side effects.

## Out of scope

- `PathDecisionRequest`, `PathDecisionCommit`, Proxy Future, or PE request.
- Round-robin or any final `PE_READ`/`DE_READ` selection.
- `commit_after_alloc()` and Worker-side Store bulk load.
- Consuming `do_remote_prefill` or bypassing the existing Proxy flow.
- Reverse, new Forward plans, or completion reconciliation.
- User-facing configuration that activates the probe in production traffic.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- Conditional, minimal no-behavior-change lookup seam in
  `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- Focused files under `tests/ut/distributed/kv_transfer/dual_path/`

No Proxy, AscendMultiConnector, AscendStore Worker, or parent Mooncake
Layerwise implementation file is modified by this PR. The adapter must not
call the current mutating lookup and then reach into a private shared map merely
to `pop()` an orphaned `LoadSpec`; use an owned result or a reviewed pure seam.

## Required tests

- `L_DE == R` skips Store probe and remains the inherited HBM-ready baseline.
- Full, partial, and miss lookup results produce aligned, clamped coverage.
- Probe does not call Worker Store load, create transport metadata, or mutate
  `do_remote_prefill`.
- Probe leaves no active Store Scheduler `load_specs` entry outside its owned
  handle.
- A pending handle is abortable exactly once; a duplicate identical abort is
  idempotent.
- Conflicting lifecycle transitions fail fast.
- Probe error produces `MISS` plus a typed unavailable status/error code and
  cleans temporary Store state.
- Request finish, cancellation, and shutdown leave no pending handle or
  retained lookup state.
- Existing remote-prefill and PR-00 parity tests remain unchanged.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/ -k 'coverage or probe'
pytest -sv tests/ut/kv_offload/test_mooncake_layerwise_connector.py
bash format.sh ci
```

## Acceptance gates

- Coverage is a fact, never a final path selection.
- Store Full does not consume `do_remote_prefill` or bypass Prefill.
- No active configuration can cause Store or P2P I/O.
- Every opened probe is either retained as a pending candidate or aborted; no
  temporary Store state leaks.
- Existing Layerwise behavior and PR-00 parity remain valid.

## Rollback contract

Reverting this PR removes Store coverage types and the probe adapter. PR-00
continues to behave as a pure Mooncake Layerwise alias.

## Review focus

- Is HBM readiness distinct from Store coverage?
- Is probe free of HBM load and P2P side effects?
- Does the handle make cleanup and exactly-once ownership explicit without
  prematurely defining a data path?
- Can any error or cleanup path mutate ordinary remote-prefill behavior?

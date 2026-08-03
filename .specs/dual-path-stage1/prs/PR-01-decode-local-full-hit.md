# PR-01 Decode Local Full Hit

- Series position: 1 of 6
- Spec status: `PLANNED`
- Depends on: PR-00
- Blocks: PR-03
- Implementation tasks: `DP-01` through `DP-04`
- User-visible behavior: yes, opt-in through DualPathConnector configuration
- Activation after merge: `DE_LOCAL_FULL_HIT` active; remote partial hit absent

## Goal

Allow Decode to prepare a complete decode-ready prefix from local HBM and DE
Store without creating a Proxy decision rendezvous or PE request.

## Merge-state contract

After this PR merges, DE evaluates local HBM first and then probes its local
Store. When aligned Store coverage reaches `R = max(P - 1, 0)`, DE freezes
`DE_LOCAL_FULL_HIT`, consumes `do_remote_prefill`, allocates final HBM blocks,
loads Store KV into those blocks, recomputes the final prompt token, and
continues decode. Store miss, partial hit, or pre-freeze probe error falls
through to the existing remote-prefill path unchanged.

## In scope

- Scheduler-side Store probe with exactly-once commit or abort.
- Worker-side bulk Store load through composed `KVPoolWorker` functionality.
- `StoreCoverage`, alignment, clamp, and local external-token accounting.
- Local route freeze before allocation-dependent Store load submission.
- Allocation-deferred retry without starting I/O.
- Store DONE/FAILED handling, invalid blocks, terminal cleanup, and idempotent
  late completion.
- Final prompt-token recomputation.
- Local-full unit tests and one-card NPU E2E.

## Out of scope

- `PathDecisionRequest`, `PathDecisionCommit`, or `DualPathKind` creation.
- `/v1/path-decision` or a Proxy decision Future.
- PE model requests for local-full cases.
- Reverse or Forward commands created by the local-full path.
- `DE_PARTIAL_HIT` and partial-hit accounting.
- Post-freeze fallback after Store load failure.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py`
- Focused files under `tests/ut/distributed/kv_transfer/dual_path/`
- A focused scenario under `tests/e2e/` for local-full execution

No Proxy file or parent Mooncake Layerwise source file is modified by this PR.

## Required tests

- `L_DE == R` skips Store probe and Proxy access.
- Full aligned Store coverage freezes `DE_LOCAL_FULL_HIT` and returns correct
  external-token accounting.
- Full hit consumes `do_remote_prefill` before the parent remote-prefill branch.
- Store miss, partial hit, and probe error preserve existing remote-prefill
  behavior.
- Store load begins only after final blocks are allocated.
- Allocation deferral retains the frozen local plan without I/O.
- Store DONE makes the request ready and recomputes only the prompt tail.
- Store FAILED reports invalid blocks and finishes with error without remote
  fallback.
- Late duplicate Store completion is idempotent.
- Local-full NPU E2E verifies output correctness and zero calls to
  `/v1/metaserver`, `/v1/path-decision`, and PE model serving.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/
bash format.sh ci
```

The NPU E2E command and environment are recorded in `TRACKING.md` when the test
is runnable on the target hardware.

## Acceptance gates

- Local full hit is a complete working vertical slice, not Scheduler-only
  accounting.
- Local-full success depends only on `STORE_DONE`.
- Local-full failure is fail-closed and cannot access Proxy later.
- Existing remote-prefill regression tests pass.
- CPU-only and NPU evidence are reported separately.

## Rollback contract

Reverting this PR restores the foundation behavior in which every
`do_remote_prefill` request follows the inherited Mooncake Layerwise path.

## Review focus

- Are HBM-complete and Store-can-complete represented as distinct cases?
- Does probe remain free of HBM load side effects?
- Is the local route irreversible only after the correct freeze boundary?
- Can failure or late completion accidentally re-enter remote scheduling?

# PR-05 Bidirectional Runtime and Lifecycle

- Series position: 5 of 6
- Spec status: `PLANNED`
- Depends on: PR-03, PR-04
- Blocks: PR-06
- Implementation tasks: `DP-17`, `DP-22`, `DP-23`, the bidirectional portion
  of `DP-25`, and `DP-26` through `DP-28`
- User-visible behavior: incomplete partial path remains non-activatable
- Activation after merge: injected committed plans can execute both directions;
  production partial selection remains disabled

## Goal

Complete one safe Mooncake runtime capable of Forward and Reverse with explicit
start gates, request-local completion, and drain-first resource handling.

## Merge-state contract

After this PR merges, a DualPath Worker can start one send thread and one receive
thread from its single inherited runtime. PE can receive Reverse and send
Forward; DE can send Reverse and receive Forward. Tests and internal integration
can execute frozen committed plans, but configuration cannot yet turn a partial
candidate into an active Scheduler winner.

## In scope

- Direction-aware runtime using one parent Worker initialization and one set of
  registered KV buffers.
- Role-local endpoint and wire-ID separation for Forward and Reverse.
- Batch ordering: install all inbound mappings, register receive plans, submit
  outbound work, then merge pending completion.
- DE Store DONE gate before Reverse submission.
- PE request readiness only after request-level Reverse DONE.
- DE readiness facts for Forward and Store without activating the final partial
  route.
- Cross-direction raw completion reconciliation.
- Terminal/cancel behavior that stops new work and drains submitted DMA.
- Source/destination block retention until completion, failure, or timeout.
- Execution timeout, terminal tombstone, orphan cleanup, and observability
  snapshot.
- Injected-plan success and failure tests for both directions.

## Out of scope

- Positive first-winner accounting for `DE_PARTIAL_HIT`.
- Production policy activation.
- Claiming end-to-end Stage 1 support before the full NPU matrix passes.
- Post-commit path fallback or force-cancelling submitted DMA.
- Multiple transport attempts or PP/DP expansion.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py`
- Focused runtime, gate, cancellation, timeout, and reconciliation tests under
  the DualPath UT directory

## Required tests

- Worker parent initialization and KV buffer registration occur exactly once.
- Each Engine owns one send and one receive thread with correct direction.
- Four-request mixed batches bind every inbound mapping before any outbound
  dispatch.
- Early raw DONE and FAILED survive across overlapping asynchronous batches.
- DE cannot submit Reverse before Store DONE and submits it at most once.
- PE cannot compute the partial tail before Reverse request-level DONE.
- Cancel/terminal prevents new work but retains blocks until quiescence.
- Timeout produces invalid blocks before failed terminal publication.
- Tombstones absorb ACK-loss retries and duplicate late terminal events.
- Store, Reverse, and Forward failure facts remain source-specific.
- Configuration still cannot produce active partial first-winner accounting.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/ -k 'reverse or gate or runtime or cancel or timeout or tombstone'
pytest -sv tests/ut/distributed/kv_transfer/ -k mooncake_layerwise
bash format.sh ci
```

## Acceptance gates

- No second Mooncake Worker, duplicate buffer registration, or extra Worker
  decision channel exists.
- Mapping-first ordering is enforced in code and tests.
- Cancel and timeout cannot permit premature block reuse.
- All operation completions preserve source and direction provenance.
- Active partial selection remains impossible until PR-06.

## Rollback contract

Reverting this PR removes Reverse and bidirectional lifecycle support while
retaining the explicit Forward path and decision control-plane closure.

## Review focus

- Does one inherited runtime safely support both capabilities?
- Are send/receive gates direction-specific despite shared machinery?
- Can any terminal or cancel race release blocks while DMA can still write?
- Are raw and Engine-local request identities reconciled exactly once?

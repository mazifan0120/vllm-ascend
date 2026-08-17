# PR-06 Bidirectional Runtime and Lifecycle

> **Superseded on 2026-08-17:** The current lifecycle and identity contract is
> [DualPath ABORT Notification and Watchdog Removal](../../../docs/superpowers/specs/2026-08-16-dual-path-abort-notification-and-watchdog-removal.md).
> Elapsed-time failure, ABORT tombstones, and two-field request keys below are
> historical only.

- Series position: 6 of 7
- Spec status: `PLANNED`
- Depends on: PR-04, PR-05
- Blocks: PR-07
- Implementation tasks: `DP-BIDI-01` through `DP-BIDI-05`
- User-visible behavior: Store-full routing remains active; partial `DE_READ`
  remains non-activatable
- Activation after merge: injected committed partial plans can execute both
  directions; production partial selection remains disabled

## Goal

Complete one safe Mooncake runtime capable of Forward and Reverse with explicit
start gates, request-local completion, and drain-first resource handling.

## Merge-state contract

After this PR merges, a DualPath Worker can start one send thread and one
receive thread from its single inherited runtime. PE can receive Reverse and
send Forward; DE can send Reverse and receive Forward. Tests and internal
integration can execute frozen committed partial plans, but configuration
cannot yet make partial Store coverage eligible for `DE_READ`.

The active Store-full `PE_READ`/`DE_READ` behavior from PR-04 remains unchanged.

## In scope

- Direction-aware runtime using one parent Worker initialization and one set of
  registered KV buffers.
- Role-local endpoint and wire-ID separation for Forward and Reverse.
- Reverse plan conversion and dispatch in registered layer order.
- Batch ordering: install all inbound mappings, reconcile pending raw
  completion, register receive plans, and only then evaluate outbound gates.
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

- Positive first-winner accounting for partial `DE_READ`.
- Production partial-policy activation.
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
- Configuration still cannot make partial `DE_READ` a first-positive winner.
- Store-full `PE_READ`/`DE_READ` regression tests remain green.

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
- Active partial selection remains impossible until PR-07.
- PR-04 Store-full behavior remains a complete supported path.

## Rollback contract

Reverting this PR removes Reverse and bidirectional lifecycle support while
retaining the explicit Forward path, decision control plane, and Store-full
activation.

## Review focus

- Does one inherited runtime safely support both capabilities?
- Are send/receive gates direction-specific despite shared machinery?
- Can any terminal or cancel race release blocks while DMA can still write?
- Are raw and Engine-local request identities reconciled exactly once?
- Is partial selection still unreachable outside injected tests?

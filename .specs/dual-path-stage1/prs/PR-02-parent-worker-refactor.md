# PR-02 Parent Layerwise Worker Helper Extraction

- Series position: 2 of 7
- Spec status: `PLANNED`
- Depends on: PR-00
- Blocks: PR-05
- Implementation task: `DP-REF-01`
- User-visible behavior: none
- Activation after merge: ordinary Layerwise execution remains unchanged

## Goal

Extract the protected Mooncake Layerwise Worker extension points required by a
single bidirectional DualPath runtime without changing parent behavior.

## Merge-state contract

After this PR merges, ordinary producer and consumer configurations start the
same threads, bind the same request metadata, enqueue the same layer transfers,
and publish the same completion results as before. DualPath still does not
enable bidirectional execution.

## In scope

- Extract protected capability checks for send and receive threads.
- Extract idempotent send-thread and receive-thread startup helpers.
- Extract receive metadata binding and send metadata preparation.
- Extract the per-layer send-enqueue helper used by the existing
  `save_kv_layer()` implementation.
- Preserve quantization, reshard, event handling, address calculation, and
  final-chunk completion behavior.
- Add behavior-parity tests around every extracted seam.

## Out of scope

- DualPath plan types or direction dispatch.
- Starting both directions for any current connector.
- New completion reconciliation or tombstones.
- Store integration.
- Cleanup unrelated to the helper extraction.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
- Existing and focused regression tests for Mooncake Layerwise producer and
  consumer behavior

The DualPath connector implementation should not need functional changes in
this PR beyond compatibility checks required by the extracted signatures.

## Required tests

- Producer starts only its existing send capability.
- Consumer starts only its existing receive capability.
- Repeated ensure-start calls do not create duplicate threads.
- Receive request mapping and metadata remain byte-for-byte equivalent at the
  parent boundary.
- Per-layer enqueue preserves block expansion, quantization/reshard behavior,
  event recording, and request-level completion.
- Existing Mooncake Layerwise unit and relevant E2E regressions pass.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/ -k mooncake_layerwise
bash format.sh ci
```

## Acceptance gates

- No new execution branch is reachable from ordinary Layerwise configuration.
- The diff separates mechanical extraction from DualPath feature logic.
- Parent behavior has explicit before/after regression evidence.
- No complete parent method is copied into DualPath.

## Rollback contract

Reverting this PR restores the original parent method layout without affecting
the PR-00 foundation or PR-01 coverage-probe foundation.

## Review focus

- Are the extracted methods protected extension points with focused purposes?
- Does extraction preserve the parent's hidden ordering and completion rules?
- Is any unrelated cleanup obscuring the behavioral parity review?

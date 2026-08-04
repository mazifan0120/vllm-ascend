# PR-05 Explicit Forward Data Path

- Series position: 5 of 7
- Spec status: `PLANNED`
- Depends on: PR-02
- Blocks: PR-06
- Implementation tasks: `DP-FWD-01` through `DP-FWD-05`
- User-visible behavior: no intended semantic change to PE-to-DE transfer
- Activation after merge: committed `PE_READ` and ordinary PE preparation use
  an explicit, tested Forward plan; Reverse remains unavailable

## Goal

Introduce the frozen block-plan and completion machinery through a real
Forward consumer before adding bidirectional execution.

## Merge-state contract

After this PR merges, DualPath can represent and execute PE-to-DE Forward work
with explicit source/destination block pairs and request-level completion. PE
Store hit, PE Store miss plus compute, and ordinary PE compute still produce the
same DE-ready prefix behavior. Store-full `DE_READ` from PR-04 remains Store
only. Reverse and active partial `DE_READ` remain unavailable.

## In scope

- Frozen `BlockPair`, Forward direction plan, binding, and Worker plan types.
- Scheduler ownership of token-to-physical-block conversion.
- Plan conversion into parent `ReqMeta`/`SendTask` structures.
- Forward region chunking and monotonic `ForwardFrontier`.
- PE `save_kv_layer()` driving Forward through the extracted parent enqueue
  helper.
- Raw completion retention, mapping, duplicate handling, and bounded orphan
  expiry for Forward.
- Worker plan queue and request-level Forward DONE/FAILED publication.
- Forward-only invalid-block reporting.
- PE Store hit/miss and PE compute to DE NPU E2E coverage.

## Out of scope

- Reverse send or receive.
- Starting both parent thread capabilities.
- DE Store DONE gating Reverse.
- Positive partial `DE_READ` accounting.
- DE success predicates combining Store and Forward.
- Cross-direction terminal reconciliation.
- Changing the PR-04 Store-full decision or Store-load semantics.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- Focused plan, frontier, completion, and Forward tests under the DualPath UT
  directory
- Forward scenarios under `tests/e2e/`

## Required tests

- Scheduler freezes complete-block source/destination pairs with equal length.
- Worker does not re-zip, trim, reorder, or infer pairs from token ranges.
- Plan conversion preserves parent layer metadata and direction identity.
- Forward frontier never resends completed blocks.
- Raw DONE/FAILED arriving before mapping is retained and published after bind.
- Duplicate terminal is idempotent; DONE/FAILED conflict is a protocol error.
- Unknown raw terminal expires after the configured execution/retry window.
- PE Store hit plus tail and Forward produces correct output on NPU.
- PE Store miss plus compute and Forward produces correct output on NPU.
- Store-full `PE_READ` uses the explicit Forward plan; Store-full `DE_READ`
  creates none.
- Existing ordinary Mooncake Layerwise behavior still passes regression tests.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/ -k 'block_pair or forward or completion'
pytest -sv tests/ut/distributed/kv_transfer/ -k mooncake_layerwise
bash format.sh ci
```

## Acceptance gates

- Every new plan and completion type has a real Forward consumer.
- Physical block ownership is fixed by Scheduler data, not Worker inference.
- Forward completion is request-level even though transfer remains layerwise.
- Existing PE-to-DE output and completion behavior are preserved.
- Reverse cannot start through any reachable configuration.

## Rollback contract

Reverting this PR returns DualPath Forward behavior to the inherited parent path
while preserving PR-02 helper extraction, the PR-03 control plane, and PR-04
Store-full activation.

## Review focus

- Is the token-accounting boundary cleanly separated from physical block pairs?
- Does Forward reuse parent quantization, reshard, event, and completion logic?
- Can a completion be lost or attributed to the wrong local request?
- Does this PR avoid changing the already active Store-full path decision?

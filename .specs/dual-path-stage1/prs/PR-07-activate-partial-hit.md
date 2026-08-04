# PR-07 Activate Partial DE_READ and Complete Stage 1

- Series position: 7 of 7
- Spec status: `PLANNED`
- Depends on: PR-06
- Blocks: none in Stage 1
- Implementation tasks: `DP-ACT-01` through `DP-ACT-05`
- User-visible behavior: yes
- Activation after merge: partial Store coverage can become eligible for the
  same PE-owned `PE_READ`/`DE_READ` round-robin used by Store-full requests

## Goal

Connect the verified control plane and bidirectional runtime, activate partial
`DE_READ`, and validate the complete Stage 1 contract on NPU.

## Merge-state contract

After this PR merges, eligible partial Store coverage participates in the same
PE-owned round-robin policy as Store-full coverage:

- committed `PE_READ` returns `(0, False)` from the DualPath child, allowing the
  outer `AscendStoreConnector` and then ordinary PE compute to proceed before
  Forward fills Decode;
- committed `DE_READ` returns positive DualPath accounting, loads the DE Store
  prefix, sends Reverse to PE when required, lets PE compute the missing tail,
  and sends Forward to DE.

Store-full coverage continues to enter the PE decision rendezvous; it never
becomes a local bypass. HBM-complete requests continue to use the inherited
baseline fast path.

## In scope

- Make admitted partial coverage eligible for `DE_READ`.
- Connect committed decisions to positive/zero first-positive accounting.
- Bind committed PE and DE Worker plans into Scheduler metadata.
- Integrate with `AscendMultiConnector` without changing the parent connector
  contract.
- Enforce child order: DualPath before outer `AscendStoreConnector` on PE.
- Enforce exact Store, Reverse, PE compute, and Forward ranges.
- Enforce DE success as `STORE_DONE && FORWARD_DONE` for partial `DE_READ`.
- Enforce PE tail compute only after Reverse DONE.
- Complete failure, timeout, cancellation, invalid-block, and no-fallback paths.
- Publish the final Stage 1 observability fields and metrics.
- Run the complete NPU E2E matrix and performance/HBM audit.
- Document the supported configuration and topology limits.

## Out of scope

- Active Value Function or LinkMonitor decisions.
- Post-commit fallback or partial recovery.
- Token striping, Store layerwise read, or Reverse/compute layer overlap.
- Relay staging or secondary HBM buffers.
- PP, DP, TP mismatch, PCP, DCP, multi-group, or hybrid-layout support.
- Stage 2 cascade or adaptive topology expansion.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- MultiConnector-focused tests under the DualPath UT directory
- Full Stage 1 scenarios under `tests/e2e/`
- User configuration and support-limit documentation

This PR should not require semantic changes to existing
`AscendMultiConnector`, `AscendStoreConnector`, or the parent
`MooncakeLayerwiseConnector` facade. Any such requirement returns to design
review before implementation continues.

## Required tests

Unit and integration coverage:

- Eligible partial candidates alternate between `PE_READ` and `DE_READ` under
  a fixed initial counter.
- Miss, unavailable, or failed admission selects `PE_READ` without advancing
  the counter.
- Committed `DE_READ` returns positive DualPath accounting.
- Committed `PE_READ` allows outer Store to win; full PE Store miss allows PE
  compute.
- Non-winning sibling connectors cannot start unauthorized I/O.
- Accepted plans preserve exact Store, Reverse, compute, and Forward ranges.
- DE waits for both Store DONE and Forward DONE on partial `DE_READ`.
- Every committed failure is fail-closed with no path switch.

NPU E2E matrix:

1. Store-full coverage committed to `PE_READ`, with PE Store/compute + Forward.
2. Store-full coverage committed to `DE_READ`, with DE Store only.
3. Partial coverage committed to `PE_READ`.
4. Partial coverage committed to `DE_READ`, with Store, Reverse, PE tail, and
   Forward.
5. Store load failure.
6. Reverse failure.
7. Forward failure.
8. TP-rank or topology mismatch fail-fast.
9. Raw completion before mapping.
10. Previous-batch completion under asynchronous scheduling.
11. Duplicate terminal and orphan expiry.
12. Cancel/shutdown drain with no block reuse before transport quiescence.

For every success scenario, verify output tokens, decision, token accounting,
physical transfer ranges, terminal state, and absence of leaked blocks. For
every failure scenario, verify invalid blocks, terminal provenance, no
fallback, and resource drain.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/
bash format.sh ci
```

Exact NPU E2E and benchmark commands are recorded as evidence in `TRACKING.md`
because they depend on the target cluster and image.

## Acceptance gates

- All earlier PR merge contracts remain true.
- Active partial selection is enabled only after the complete NPU matrix passes.
- No default-path regression is observed for ordinary Layerwise or outer Store.
- Performance and HBM deltas are measured and reviewed, not inferred.
- Supported configuration, topology limits, and failure semantics are
  documented.
- The Stage 1 checklist in the architecture design is fully mapped to passing
  evidence.

## Rollback contract

Reverting this PR makes partial Store coverage `PE_READ`-only again. Store-full
round-robin, explicit Forward support, and the dormant bidirectional runtime
remain available.

## Review focus

- Is first-positive accounting consistent with the committed physical plan?
- Can a non-winning connector start Store or P2P work?
- Are success and failure predicates complete on both Engines?
- Does NPU evidence cover both decisions for both full and partial coverage?
- Does rollback preserve the already supported Store-full vertical slice?

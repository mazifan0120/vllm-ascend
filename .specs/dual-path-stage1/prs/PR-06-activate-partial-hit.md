# PR-06 Activate Partial Hit and Complete Stage 1

- Series position: 6 of 6
- Spec status: `PLANNED`
- Depends on: PR-05
- Blocks: none in Stage 1
- Implementation tasks: `DP-29` through `DP-31` plus activation of the
  previously shadowed/rejected partial decision
- User-visible behavior: yes
- Activation after merge: approved static `DE_PARTIAL_HIT` policy is active

## Goal

Connect the verified control plane and bidirectional runtime, activate static
partial-hit selection, and validate the complete Stage 1 contract on NPU.

## Merge-state contract

After this PR merges, the configured static policy may accept
`DE_PARTIAL_HIT`. The DualPath child then returns positive matched-token
accounting and wins outer first-positive selection. DE loads its Store prefix,
sends Reverse to PE, PE computes the missing tail, and sends Forward to DE. A
rejected DualPath decision returns `(0, False)`, allowing outer
`AscendStoreConnector` and then ordinary PE compute to proceed. Local full hit
continues to bypass the entire decision rendezvous.

## In scope

- Activate `decide_on_pe()` and `matched_tokens_from_commit()` for approved
  partial candidates.
- Bind committed PE and DE Worker plans into Scheduler metadata.
- Integrate with `AscendMultiConnector` first-positive accounting without
  changing the parent connector contract.
- Enforce child order: DualPath before outer `AscendStoreConnector` on PE.
- Enforce DE success as `STORE_DONE && FORWARD_DONE` for partial hit.
- Enforce PE tail compute only after Reverse DONE.
- Complete failure, timeout, cancellation, invalid-block, and no-fallback paths.
- Publish the final Stage 1 observability fields and metrics.
- Run the complete NPU E2E matrix.
- Audit throughput, latency, CPU/NPU synchronization, and HBM usage.
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

- Partial plus `prefer_de` returns positive DualPath accounting.
- Partial plus `prefer_pe`, miss, unknown, or failed admission returns
  `(0, False)`.
- Rejected DualPath allows outer Store to win; full miss allows PE compute.
- Non-winning sibling connectors cannot start unauthorized I/O.
- Accepted plans preserve exact Store, Reverse, compute, and Forward ranges.
- DE waits for both Store DONE and Forward DONE.
- Every committed failure is fail-closed with no path switch.

NPU E2E matrix:

1. PE Store hit plus PE tail plus Forward.
2. PE Store miss plus PE compute plus Forward.
3. DE local full hit with zero decision calls.
4. DE partial hit with Store, Reverse, PE tail, and Forward.
5. Store load failure.
6. Reverse failure.
7. Forward failure.
8. TP-rank or topology mismatch fail-fast.
9. Raw completion before mapping.
10. Previous-batch completion under asynchronous scheduling.
11. Duplicate terminal and orphan expiry.
12. Cancel/shutdown drain with no block reuse before transport quiescence.

For every success scenario, verify output tokens, token accounting, physical
transfer ranges, terminal state, and absence of leaked blocks. For every failure
scenario, verify invalid blocks, terminal provenance, no fallback, and resource
drain.

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

Reverting this PR disables active partial selection and returns remote
candidates to the rejected/shadow behavior from PR-05. Local full hit, explicit
Forward support, and the dormant bidirectional runtime remain available.

## Review focus

- Is first-positive accounting consistent with the committed physical plan?
- Can a non-winning connector start Store or P2P work?
- Are success and failure predicates complete on both Engines?
- Does NPU evidence cover every activation and resource-lifetime risk?

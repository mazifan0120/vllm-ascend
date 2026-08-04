# PR-04 Activate Store-Full DE_READ

- Series position: 4 of 7
- Spec status: `PLANNED`
- Depends on: PR-03
- Blocks: PR-06
- Implementation tasks: `DP-FULL-01` through `DP-FULL-05`
- User-visible behavior: yes, opt-in through a supported DualPath configuration
- Activation after merge: Store-full candidates alternate between `PE_READ` and
  `DE_READ`; partial Store coverage remains `PE_READ`-only

## Goal

Activate the first complete PE-decided DualPath vertical slice for requests
whose Decode AscendStore can prepare the entire decode-ready prefix.

## Merge-state contract

After this PR merges, a request with `L_DE < R` always enters the PE-owned
decision flow even when Decode Store coverage is full. The PE Scheduler first
computes eligibility and then applies round-robin:

- `PE_READ` aborts the DE probe, lets the PE-side AscendStore/compute path
  prepare KV, and uses the inherited Layerwise Forward path to fill Decode.
- `DE_READ` commits the DE probe after final block allocation, performs one
  bulk Store load into Decode blocks, and completes without Reverse, PE model
  compute, or Forward.

The PE request created for a committed Store-full `DE_READ` remains the
decision-lifecycle carrier but performs zero model computation and zero data
transfer. Decode succeeds on `STORE_DONE` and then recomputes the final prompt
token locally.

Store partial or miss cannot select `DE_READ` in this PR. Those requests select
`PE_READ` without advancing the round-robin counter when it is the only eligible
path.

## In scope

- Exactly-once probe-handle `commit_after_alloc()` or `abort_probe()`.
- Decode bulk `KVPoolWorker` composition and Store metadata bound to final HBM
  blocks.
- Store-full `DE_READ` Scheduler and Worker plans.
- Positive Scheduler accounting for the committed Store-full `DE_READ` path.
- `PE_READ` fall-through to the PE outer `AscendStoreConnector` and inherited
  Layerwise Forward behavior.
- Zero-compute/zero-transfer termination of the PE request after a Store-full
  `DE_READ` commit.
- Store DONE/FAILED, invalid blocks, terminal cleanup, timeout, and no fallback.
- A minimal existing-KVPool completion seam, if required, so async completion
  preserves request-level DONE versus FAILED and shares invalid block IDs with
  the polling Worker.
- Active configuration and topology fail-fast checks required by this vertical
  slice.

## Out of scope

- Partial `DE_READ` eligibility or positive partial accounting.
- Reverse transfer or bidirectional Worker capability.
- New explicit Forward block plans; `PE_READ` uses the inherited Layerwise
  implementation until PR-05.
- Post-commit path switching or Store-load fallback.
- Value Function, LinkMonitor, adaptive routing, or weighted scheduling.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py`
- Conditional, narrowly scoped completion/failure changes in
  `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py` and
  `kv_transfer.py`
- Focused files under `tests/ut/distributed/kv_transfer/dual_path/`
- A Store-full scenario under `tests/e2e/`

No Reverse implementation or semantic change to the existing
`AscendStoreConnector`, `AscendMultiConnector`, or parent
`MooncakeLayerwiseConnector` is expected. A generic async finished ID must not
be treated as `STORE_DONE` when the underlying load reported invalid blocks.

## Required tests

- Full Store coverage is sent to PE and never freezes a local route on Decode.
- Two consecutive full candidates select `PE_READ`, then `DE_READ`, under a
  fixed initial counter; the next pair repeats.
- A partial/miss request between eligible full candidates does not advance the
  round-robin counter.
- Full coverage with `A_DE == 0` leaves only `PE_READ` eligible and does not
  advance the round-robin counter.
- `PE_READ` aborts the DE probe and starts no DE Store load.
- `DE_READ` commits the handle only after final blocks are allocated.
- Store-full `DE_READ` creates no Reverse or Forward plan and schedules no PE
  model token.
- Store-full `DE_READ` starts no PE outer-Store Worker load even though sibling
  lookup may have been queried by `AscendMultiConnector`.
- `PE_READ` still executes PE AscendStore/compute plus inherited Forward even
  when Decode reported full Store coverage.
- `STORE_DONE` completes Decode and triggers only final-prompt-token
  recomputation.
- `STORE_FAILED`, decision timeout, PE request failure, and cancellation are
  fail-closed with no post-commit path switch.
- Async Store get failure is published as FAILED with all affected invalid
  blocks before the failed receive terminal; it cannot appear as DONE merely
  because the request ID reached a finished set.
- Duplicate commit, abort, Store terminal, and late completion are idempotent.
- NPU E2E validates output equality and the exact Store/Forward operation
  absence/presence for both committed paths.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/ -k 'full and (decision or store or accounting)'
bash format.sh ci
```

The exact NPU E2E command and environment are recorded in `TRACKING.md` when
run on the supported hardware.

## Acceptance gates

- Coverage and final path remain distinct facts.
- The PE Scheduler is the unique path committer for every Store-full request.
- No Store load starts before commit and final block allocation.
- Committed `DE_READ` full has only one required operation: DE Store load.
- Committed `PE_READ` never consumes the Decode Store candidate.
- Partial `DE_READ` remains unreachable through configuration.
- Existing ordinary Layerwise and PE Store behavior pass regression tests.

## Rollback contract

Reverting this PR disables active Store-full selection and Store load. PR-03
continues to support a closed-loop, non-I/O decision protocol, and PR-01 retains
coverage/probe facts.

## Review focus

- Can Store Full accidentally bypass the PE Scheduler decision?
- Does PE request termination avoid model execution without inventing a second
  decision API?
- Are commit/abort and Store terminal transitions exactly once?
- Can partial coverage or a non-winning path start any data operation?

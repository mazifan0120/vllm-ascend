# DualPath Stage-2: Preemption Safety (Pin + Fail-Closed + Epoch Re-execution)

Status: design — single-stage delivery (approved direction: epoch-based
re-execution; no 2a/2b split)
Date: 2026-08-11
Scope: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` on vllm 0.23.0
Supersedes: the Stage-1 known limitation recorded in
`.specs/dual-path-stage1-tasks/TASKS.md` §3 ("prefill dual_path request preempted
after decision delivery is unsafe").

## 1. Problem

A PE-side dual_path request whose path decision has been **delivered** to DE is
unsafe if the vLLM scheduler later preempts it:

- `_preempt_request` (vllm `v1/core/sched/scheduler.py:974-995`) frees the
  request's blocks at `:983` with **no connector callback**, and the blocks can
  be reused within the same `schedule()` pass.
- The PE request must be `RUNNING` while it computes `[K_DE, T)` and streams
  Forward data to DE; only RUNNING requests are preemptible
  (`scheduler.py:499-501`). The Reverse-receive window itself is safe by
  construction: admission allocates only the Reverse destination blocks and
  parks the request in `WAITING_FOR_REMOTE_KVS` until the Reverse terminal
  arrives (dual_path `scheduler.py:598-603`).

### Failure modes (all confirmed by code walk-through)

- **F1 — Forward read-after-free.** Forward sends are async; when preemption
  frees the blocks at a step boundary, in-flight DMA reads freed/reused blocks
  and pushes garbage to DE → silent wrong output. MooncakeLayerwise heals the
  equivalent race by idempotent re-send into the consumer's *stable* blocks;
  DualPath has no re-send path.
- **F2 — PE re-admission parks forever.** After preemption the request
  re-admits, but the Reverse terminal was already consumed, so nothing promotes
  it out of `WAITING_FOR_REMOTE_KVS` again; the PE side has no timeout
  (`scheduler.py:204-207` — DECISION_TIMEOUT is DE-only).
- **F3 — DE hangs in COMMITTED forever.** DE's split tracker only advances on
  wire terminals (worker.py:675-755); the `reverse_submitted` latch
  (worker.py:430-436) prevents any re-send; the DE timeout loop only scans
  PENDING decisions (`scheduler.py:1220-1235`). Request and DE blocks leak.
- **F4 — engine assert crash.** A Reverse terminal drained for a PREEMPTED
  request reaches `_update_from_kv_xfer_finished` and trips
  `assert RequestStatus.is_finished(...)` (vllm `scheduler.py:2243`) → engine
  core dies.
- **F5 — latent third-party write.** The retained ReversePlan keeps the *old*
  PE block ids and is deliberately not re-installed (`scheduler.py:613-614`).
  Any future re-trigger would DMA into blocks now owned by an unrelated
  request. Held shut today only by the `reverse_submitted` latch and the parked
  window.

### Why existing mechanisms do not cover this

- `delay_free_blocks` is consulted only at request finish
  (`scheduler.py:1875-1910`), never on the preemption path.
- The waiting-queue immunity protects DE-side allocations and our own Reverse
  window, but not the RUNNING compute+Forward window.
- MooncakeLayerwise's "recompute + idempotent re-send" recovery requires a
  stable anchor (consumer blocks, exchanged once). Our Reverse anchor is the
  PE request's own blocks — the very thing preemption invalidates.

## 2. Goals / non-goals

Goals:

- No memory corruption and no engine crash under PE preemption (F1, F4, F5).
- A preempted delivered request is **re-executed as a new life of the same
  request**: fresh blocks, fresh Reverse plan, fresh delivery; DE discards the
  previous life explicitly (F2, F3).
- Every in-flight window bounded by a watchdog; failures converge to a
  client-visible error, never a hang (TASK-04 §7.2 philosophy).

Non-goals (Stage-2):

- DE-side request preemption (decode blocks are DE-local; separate analysis).
- TP>1 / PCP/DCP epoch ordering details (single-rank Stage-1 topology first).
- Engine-core restart: already covered by `decode_engine_instance_id`
  incarnation checks (`metadata.py:272-273`,
  `path_decision_channel.py:434-436`).

## 3. Available engine primitives (vllm 0.23.0, no engine patch needed)

- `KVConnector.bind_gpu_block_pool(block_pool)` — supported scheduler-side
  hook (kv_connector `v1/base.py:443-451`), invoked by the scheduler after the
  KVCacheManager is built (`scheduler.py:245-248`), fanned out by
  MultiConnector (`multi_connector.py:249-251`). In-tree precedent:
  AscendStoreConnector pins sends via `touch` (`pool_scheduler.py:984`) and
  releases via `free_blocks` (`:1008`); upstream SimpleCPUOffloadConnector uses
  the same pattern (`simple_kv_offload/manager.py:509`, `:703-726`).
- `BlockPool.touch(blocks)` / `BlockPool.free_blocks(blocks)`
  (`v1/core/block_pool.py:402-415`, `:419-441`).
- `SchedulerOutput.preempted_req_ids` (`v1/core/sched/output.py:217-219`),
  filled in the same pass that preempts (`scheduler.py:940`) and visible to the
  connector in `build_connector_meta` (`scheduler.py:955`); the worker-side
  `handle_preemptions` hook (`base.py:285-290`, fed at
  `gpu_model_runner.py:4036-4039`) is how OffloadingConnector learns the same
  set on the worker.
- `update_connector_output` (`base.py:532-540`, called at
  `scheduler.py:2232-2233`, MultiConnector fanout `:432`) — currently
  unimplemented in DualPath; the channel through which the scheduler side can
  observe worker-reported completions.

## 4. Design

Three mechanisms — L0 pinning, L1 fail-closed handling, L2 epoch re-execution —
delivered as **one unit** (§7). They are designed together and only safe
together: L0 makes detection-after-free harmless, L1 makes preemption visible
and contained, L2 gives the request its second life. None is optional.

### L0 — PE block pinning (memory safety; fixes F1, enables F5 fix)

Owner: `DualPathConnectorScheduler` (PE role). Override
`bind_gpu_block_pool` and keep the pool.

Pin sets, resolved from already-retained plan state (no new wire data):

- DE_READ: at delivery time (`_update_prefill_state_after_alloc`,
  `scheduler.py:726-789`) pin the Reverse destination slice of the admission
  block table covering `[L_PE, K_DE)`; the slice math has an in-tree precedent
  (`_stage_prefill_activation_failure`, `scheduler.py:248-255`). At the
  post-Reverse allocation, when the ForwardPlan installs
  (`scheduler.py:649-663`), pin the Forward source blocks.
- PE_READ: delivery is deferred until the ForwardPlan installs
  (`scheduler.py:759-762`); pin the Forward source blocks then.

Pin by block object (`block_pool.blocks[block_id]`, per the
simple_kv_offload precedent). Every pin is recorded per `(request_id, epoch)`
so unpin paths are exhaustive:

| Exit | Unpin trigger |
|---|---|
| Normal completion | Reverse destination slice: when the Reverse terminal is consumed and observed scheduler-side via a new `update_connector_output` implementation. Forward source: when the PE request finishes **and** the worker send queue for the request has drained — requires the send-completion signal below. |
| Fail-closed abort (L1) | When DE ACKs the new epoch that replaces this one (L2); bounded by the PE backstop watchdog so a dead DE cannot leak pins. |
| Delivery failure / mark-invalid before delivery | Immediately (decision never left the process). |
| Shutdown | Best-effort; process exit makes it moot locally. |

Required new signal: the parent layerwise worker never reports
`finished_sending` (`mooncake_layerwise_connector.py:1451`). Stage-2 adds
per-request send-completion tracking on the PE worker, surfaced through
`get_finished` so `update_connector_output` can unpin Forward sources only
after the last byte left the NIC.

### L1 — fail-closed preemption handling (fixes F4, bounds F1)

- Scheduler side: in `build_connector_meta`, intersect
  `scheduler_output.preempted_req_ids` with delivered dual_path requests. For
  each hit: mark the decision invalid locally (reuse
  `_invalidate_prefill_activation`, `scheduler.py:816-844`), stop further
  Forward scheduling for the old epoch, and schedule unpin per the L0 table.
  Detection is same-pass but post-free; the L0 pin is what makes
  detection-after-free safe.
- Worker side: implement `handle_preemptions` to quiesce in-flight/old-epoch
  sends for the preempted request and to **purge pending Reverse terminals for
  it without publishing** them into `get_finished`.
- F4 residual race: a terminal drained in the same pass that preempts can still
  reach vllm `scheduler.py:2243`. Mitigation: defer publishing a request's
  Reverse terminal by one pass once any preemption has been observed, or carry
  a one-line upstream guard (skip `finished_recving` for PREEMPTED requests).
  The one-pass deferral is self-contained and is the chosen approach.

### L2 — epoch-based re-execution (fixes F2, F3, F5; completes F1)

Core rule (the approved route): **a re-admitted delivered request is a new
life; the new life itself is the invalidation of the old one.** No separate
INVALIDATE message — lazily, the next epoch's decision replaces the old state.
Between preemption and re-admission, DE's stale state is bounded by the
watchdog below.

- Epoch source: `request.num_preemptions` (maintained by the engine at
  `_preempt_request`) — free, monotonic per request, visible in every
  scheduler-side hook that receives the request. No new PE state.
- Wire identity becomes `(request_key, decision_epoch)`:
  `PathDecisionResult` gains `decision_epoch: int`
  (`path_decision.py:117-147`), serialized through
  `path_decision_channel.py:114-138`. Delivery futures, decider records, and
  all `_prefill_*` maps key on `(request_id, epoch)`.
- DE receiver semantics (`path_decision_channel.py:450-463`), per key:
  - epoch greater than the accepted one → **replace**: ACK, then tear down the
    old epoch (below) and enqueue the new decision;
  - equal epoch, identical decision → ACK only (dedup, unchanged);
  - equal epoch, conflicting → reject, no ACK (still a bug signal, unchanged);
  - lower epoch → drop as stale (new branch), no ACK.
- DE teardown on replace: drop the split tracker and `_split_trackers` entry,
  clear `_consumed_forward/reverse_terminal_wire_ids` for the old wire id,
  reset the `reverse_submitted` latch, and discard the old ReversePlan — its
  PE block ids must never be used again (old blocks' pins are released only
  after the new epoch is ACKed).
- PE re-admission after preemption: because all per-request state is keyed by
  epoch, the retained-plan skip at `scheduler.py:613-614` becomes
  "retained iff same epoch"; a new epoch installs a fresh ReversePlan with the
  **new** block table and re-delivers. The decider's idempotent replay
  (`path_decision.py:184-190`) is scoped per epoch: identical facts under a new
  epoch re-decide cleanly instead of conflicting.
- Watchdogs (both sides, post-COMMIT data plane):
  - DE: deadline on Reverse/Forward completion after COMMITTED; on expiry →
    control failure terminal, request errors out (no hang), state torn down.
  - PE: backstop timeout for a parked-after-re-admission request; on expiry →
    fail the request locally. (In the epoch design this should be unreachable;
    the watchdog is the fail-safe, and it also bounds the L0 abort-unpin wait.)
- Versioning: add a `protocol_version` field to the envelope header; receiver
  rejects unknown versions with a loud log. Stage-2 ships both sides in one
  image, so the field is for future mixed-version clusters, not for
  negotiation between Stage-1 and Stage-2 code.

## 5. Failure-mode → mechanism matrix

| Failure | L0 pin | L1 fail-closed | L2 epoch | Watchdog |
|---|---|---|---|---|
| F1 forward read-after-free | blocks unpinnable until send-done | quiesce old sends | — | — |
| F2 PE park-forever | — | — | new epoch re-delivers and re-promotes | PE backstop |
| F3 DE hang/leak | — | — | replace tears down; re-execution completes | DE deadline |
| F4 engine assert | — | purge + one-pass deferral | — | — |
| F5 stale block ids | old pins held until replace ACKed | abort old epoch | new epoch installs new table | — |

## 6. UT / test strategy

- Receiver epoch matrix (replace/dedup/stale/conflict) over the fake channel —
  extends existing `path_decision_channel` UTs.
- Pin/unpin pairing with a fake BlockPool: every exit in the L0 table has a
  test; leak test asserts pool ref counts return to baseline on every path.
- Scheduler-level preemption replay: drive the connector scheduler through the
  engine's exact call order (admission → delivery → `build_connector_meta`
  carrying `preempted_req_ids` → re-admission with `num_preemptions=1`) using
  the existing UT fakes; assert abort, unpin, re-delivery with epoch=1, and no
  publication of purged terminals.
- DE worker: latch reset and old-plan discard on epoch replace; stale wire
  terminals for old wire ids are dropped.
- Regression: existing `tests/ut/distributed/kv_transfer/dual_path/` suite
  (551 tests) must stay green; Stage-1 behavior with preemption never triggered
  is unchanged.
- E2E (part of this delivery, in `dualpath-npu-test`): new scenario S9 —
  constrain PE HBM (low `gpu_memory_utilization` + concurrent long prompts) to
  force real preemption during window ②; expect no crash, no hang, output
  parity with the MooncakeLayerwise baseline.

## 7. Implementation order and delivery

Single deliverable; the order below is only the local build sequence, not a
release split. The feature is considered done only when everything in this
section is green.

1. Pin plumbing: `bind_gpu_block_pool` wiring, pin/unpin ledger per
   `(request_id, epoch)`, PE worker send-completion signal, scheduler-side
   `update_connector_output`.
2. Epoch protocol: wire fields (`decision_epoch`, `protocol_version`), receiver
   replace semantics, DE teardown, PE per-epoch re-keying and re-delivery.
3. Fail-closed handling: `preempted_req_ids` intersection, worker
   `handle_preemptions` quiesce/purge, one-pass terminal deferral; both
   watchdogs.
4. UT gates after each step: full `dual_path` suite + ruff check/format.
5. Overlay e2e (`k8s/dualpath-run.yaml`, no image rebuild needed): existing
   regression `--run` / `--run-fault` / `--run-parity`, then the new S9
   high-pressure scenario.
6. Final clean-image regression (`k8s/*-clean.yaml`) after the image is
   rebuilt and pushed — requires the root-owned build step on n1.
7. On landing: remove the known-limitation note from
   `.specs/dual-path-stage1-tasks/TASKS.md` §3 and point it at this document's
   result.

## 8. Open questions

- Whether the one-pass terminal deferral (F4) measurably delays promotion
  (expected: one scheduler pass, negligible); confirm in UT timing.
- Watchdog deadlines (DE post-COMMIT completion, PE backstop) — start from the
  DECISION_TIMEOUT constant and tune on the NPU rig.
- Whether `update_connector_output` gives enough context for Reverse-slice
  unpin, or the Reverse terminal needs a dedicated scheduler-side report via
  the existing `_drain_local_reverse_terminals` channel.

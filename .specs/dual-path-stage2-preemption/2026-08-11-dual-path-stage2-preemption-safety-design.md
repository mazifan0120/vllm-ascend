# DualPath Stage-2: Preemption Safety (Fence + Stable Forward Replay + DE_READ Reverse Attempts)

Status: design — single-stage delivery (approved direction: route-specific
recovery; no 2a/2b split)
Date: 2026-08-11
Updated: 2026-08-14
Revision note: 2026-08-14 implementation sync retires the
`CloseReverseAttempt` protocol and Reverse destination hold while retaining the
Decision epoch, `STALE_CLOSED`, I4 gate, and `JobLedger`. Earlier review history
is preserved below. Round 5:
retained-proof lookup precedence in the close matrix and cleanup wording.
Round 4: durable safe-close proofs (persisted before every SAFE reply, retained
until PE acknowledgment or verified teardown), request_finished contract
correction, engine-progress tests. (Round 3: engine-progress rule, RUNNING-request abort fence, safe-close
retention, single-owner reverse_send counting, barrier-attempt association.
Round 2: DE Reverse sender completion proof, close matrix, barrier-gated
resume defer, decision-complete ledgers. Round 1: synchronous fencing,
simplified CloseReverseAttempt, single hold ledger, job-based attempt gate.)
Scope: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` on vLLM 0.23.0
Supersedes: the Stage-1 known limitation recorded in
`.specs/dual-path-stage1-tasks/TASKS.md` §3 ("prefill dual_path request
preempted after decision delivery is unsafe").

> **2026-08-14 修订：CloseReverseAttempt 协议与 Reverse destination hold 已下线。**
>
> 二者服务的都是同一个场景：PE 侧请求在 Reverse 完成前异常终止（abort 或
> recovery watchdog 超时），此时 DE 可能仍在写 PE 的 destination block。
> 现决定该场景与 main 分支保持一致的语义——立即释放，不等远程安全证明。
>
> 保留的部分：`reverse_attempt_id`（取值 `num_preemptions`）作为控制面
> epoch，`_register_decision_locked` 的去重与单调性，以及 I4 gate。控制面
> 投递是 at-least-once 且多个 attempt 并发投递，执行端仍须据此拒绝过期指令。

This revision deliberately accepts the same use-after-free risk already
present in the parent/main P-to-D direction: if DE is still sending when PE
aborts or its recovery watchdog expires, the late D-to-P write can target a
block already reallocated to another request. DualPath adds this symmetric
D-to-P exposure; the retired hold/close design below documents the stronger
alternative and why it was not retained.

## 1. Problem

A PE-side dual_path request whose path decision has been **delivered** to DE is
unsafe if the vLLM scheduler later preempts it:

- `_preempt_request` (vLLM `v1/core/sched/scheduler.py:974-995`) frees the
  request's ordinary block ownership at `:983` with **no connector callback**,
  and the blocks can be reused within the same `schedule()` pass.
- The PE request must be `RUNNING` while it computes and streams Forward data
  to DE; only `RUNNING` requests are preemptible: the victim is popped from
  `self.running` (`scheduler.py:499` in the FCFS branch; `:479-497` under
  priority scheduling) and `_preempt_request` asserts
  `request.status == RequestStatus.RUNNING` at `:980-982`.
- For `DE_READ`, PE admission first allocates the Reverse destination blocks and
  parks the request in `WAITING_FOR_REMOTE_KVS` until all workers report Reverse
  completion (dual_path `scheduler.py:598-603`). The Reverse-wait window is
  therefore not part of normal scheduler preemption: a normally preempted
  `DE_READ` request has already completed its current Reverse attempt.
  `WAITING_FOR_REMOTE_KVS` requests live in the waiting queue, not
  `self.running`, and the waiting pass is skipped entirely in any step where a
  preemption occurred (`scheduler.py:563`).

### Failure modes confirmed by code walk-through

- **F1 — Forward read-after-free.** A queued or executing Forward transfer can
  read PE blocks after preemption returns their ordinary ownership to the pool,
  sending unrelated data to DE.
- **F2 — DE_READ re-admission parks forever.** A preempted `DE_READ` request
  needs Reverse data in its new PE allocation, but the old Reverse terminal was
  already consumed and no new attempt exists to promote the new allocation.
  Current code reinforces this: `_decide_prefill_path_for_admission` treats a
  changed prefix after a delivered Decision as invalid and converges the request
  into `_prefill_invalid_request_ids` (dual_path `scheduler.py:321-322`,
  `:350-362`), and `PathDecisionDecider.decide` rejects changed retained facts
  (`path_decision.py:186-189`).
- **F3 — DE_READ recovery reuses an exhausted sender latch.** DE's retained
  `reverse_submitted` latch (dual_path `worker.py:443`) prevents a second
  Reverse submission unless recovery creates a fresh Reverse attempt.
- **F4 — stale Reverse terminal escapes into vLLM.** A delayed terminal from an
  old attempt can be attributed to the current logical request and reach
  `_update_from_kv_xfer_finished`, either resuming the wrong allocation or
  tripping its request-status assertion (`scheduler.py:2243`). Today the PE
  worker converts Reverse terminals directly into request ids in `done_recving`
  (dual_path `worker.py:571-581`, `:619-673`), discarding attempt identity
  before the scheduler can validate it.
- **F5 — stale Reverse destination.** Reusing an old `ReversePlan` after PE
  re-admission writes into the old PE block table, which may now belong to
  another request. Current worker installers reject changed bindings/plans
  (dual_path `worker.py:216-226`, `:349-360`), so no replacement path exists.
- **F6 — exceptional Reverse termination risk (accepted on 2026-08-14).** If a
  `DE_READ` Decision was accepted but the PE request aborts or its recovery
  watchdog expires while waiting for Reverse, DE may still start or continue
  writing the old destination. The retired close/hold design tried to prove
  that writer safe before reuse; the implemented contract now releases
  immediately, matching the parent/main behavior and accepting this risk.

F2-F6 require a Reverse direction and therefore apply only to `DE_READ`.
`PE_READ` has no DE-to-PE write, no PE-side remote-KV wait, and no Reverse
attempt. Both paths nevertheless share F1 and the same stable Forward-binding
replay semantics.

### Why existing request-finish mechanisms do not cover this

- `delay_free_blocks` is consulted at request finish, not on preemption.
- Generic `finished_sending` means vLLM may free the request's ordinary block
  ownership (upstream unconditionally calls `_free_blocks` at
  `scheduler.py:2248`). It cannot mean "one extra DualPath hold may be
  released".
- MooncakeLayerwise sends from a single FIFO thread
  (`mooncake_layerwise_connector.py:266-273`), but `save_kv_layer()` can enqueue
  every layer before the background sender finishes (queue put at `:1831`). With
  async scheduling, a later `schedule()` can preempt the still-`RUNNING` request
  while those sends are queued.
- Waiting-queue immunity proves normal `DE_READ` preemption happens after
  Reverse completion; it does not cover abort, uncertain Decision receipt, or
  receipt-to-activation races. DualPath no longer adds a PE-side delayed-free
  or safety-proof path for an aborted `WAITING_FOR_REMOTE_KVS` request; it
  delegates release to the parent/main lifecycle.

## 2. Goals / non-goals

Goals:

- Prevent hangs and incorrect resume under normal PE preemption (F1-F5), while
  documenting rather than masking the accepted exceptional-termination risk
  in F6.
- Preserve the original `PathKind`. PE prefix loss is normal and never causes a
  policy re-run or a route switch.
- Keep one stable logical Forward binding on DE for both paths. Recovery only
  replaces PE-local Forward source state; `DE_READ` additionally creates a new
  Reverse attempt for the current PE allocation.
- Keep request-visible waits bounded. Normal Reverse completion is still
  attempt-gated; exceptional abort/watchdog expiry fails the request and
  releases its destination without waiting for remote proof.
- Support matching `TP>1` within the Stage-1 topology contract through
  all-participating-worker aggregation.
- Reject unsupported local and PE/DE parallel topologies before a decision is
  committed.

Non-goals (Stage-2):

- DE-side request preemption (Decode blocks are DE-local; separate analysis).
- Expanding Stage-1 data-plane support to PP, DP, PCP, DCP, TP mismatch,
  multi-group, or hybrid-layout configurations.
- Engine-core restart. A changed `decode_engine_instance_id` remains the
  incarnation boundary; verified old-process/transport teardown is the final
  safety proof when the old control endpoint cannot answer.
- Post-commit fallback or path switching during recovery.
- Cancelling an in-flight synchronous Mooncake transfer. Stage-2 fences work
  already accepted by the relevant sender and prevents new old-attempt work.

## 3. Reused engine and Connector primitives

Stage-2 reuses four existing mechanisms rather than creating another
completion framework:

1. **Parent ownership/lifecycle.** The implemented DualPath scheduler does not
   add a `HoldLedger` or call `BlockPool.touch` for Reverse destinations.
   Normal parked `DE_READ` requests keep their ordinary vLLM allocation until
   I4 publishes completion. Exceptional abort/watchdog expiry follows the
   parent connector's immediate-free semantics.
2. **Preemption delivery.** `SchedulerOutput.preempted_req_ids` is populated in
   the preempting schedule pass (`scheduler.py:940`) and is visible to
   `build_connector_meta` in the same pass (`scheduler.py:954-956`).
   Worker-side `handle_preemptions` (base no-op at `base.py:285-290`) receives
   the resulting connector metadata before the model forward of every step
   (upstream `vllm/v1/worker/gpu/kv_connector.py:61-68`,
   `vllm/v1/worker/gpu_model_runner.py:4036-4039`; Ascend
   `vllm_ascend/worker/model_runner_v1.py:1967-1969`). This is the existing
   worker-side synchronization point used by the upstream Offloading connector,
   which blocks in `handle_preemptions` on the exact jobs the scheduler flagged
   (`offloading/worker.py:232-239`).
3. **Worker completion metadata.** `KVConnectorWorkerMetadata.aggregate()`
   (`base.py:150-168`) is invoked while vLLM combines every executor worker's
   `KVConnectorOutput` (`distributed/kv_transfer/kv_connector/utils.py:130-139`;
   that fold performs no type check, so each concrete `aggregate` must assert
   its own type). `offloading` reports an opaque `{job_id: 1}` per worker
   (`offloading/common.py:35-60`); `simple_kv_offload` and AscendStore report
   `{event_idx: 1}` / `{event_id: 1}`. The Scheduler maps the opaque id back to
   semantic state and completes at an expected worker count sourced from
   `parallel_config.world_size`.
4. **Layerwise topology mapping.** MooncakeLayerwise already maps TP/PCP/DCP
   ranks and physical ranges. Stage-2 adds direction-appropriate fencing around
   the existing per-rank work instead of duplicating that math.

The active completion shape is: Reverse completion is reported through
`kv_connector_worker_meta`, not `finished_sending`; scheduler-side
`update_connector_output` (invoked at `scheduler.py:2233`, before the generic
finished-set processing at `:2236-2248`) waits for all participating workers
and then applies the I4 or reverse-send job-completion action. `request_finished`
returns `True` only on Decode when an open reverse-send job must keep the
engine stepping; PE abort does not add a delayed-free path.

## 4. Safety model and identities

### 4.1 Logical request, Forward state, and Reverse attempt

- **Logical/admission identity:** `DualPathRequestKey` remains exactly
  `(decode_engine_instance_id, decode_request_id)`. DE creates and registers it
  before PE has seen the request.
- **Stable Forward binding:** both `PE_READ` and `DE_READ` keep one logical DE
  `ForwardReceiveBinding`, one stable Forward wire id, the frozen DE destination
  table, and one fixed Forward range for the lifetime of the request.
- **PE-local Forward source attempt:** PE block allocation and the source table
  may change after preemption. The source plan is local attempt state under the
  logical request; it is not part of the PE-to-DE decision identity.
- **DE_READ Reverse attempt:** only `DE_READ` carries a non-negative
  `reverse_attempt_id`, sourced from PE `request.num_preemptions`, plus the
  attempt-local `prefill_local_tokens = L_PE(attempt)`. Scheduler/Worker code
  composes them into `ReverseAttemptKey(request_key, reverse_attempt_id)`. The
  first delivered value need not be zero and later values need not be
  contiguous: PE preemptions that happen before a Decision leaves the process
  may consume numbers without creating remote attempts.
- **Reverse wire id:** a `DE_READ` Reverse attempt uses an opaque id derived
  from its complete attempt key. Code looks it up explicitly and never recovers
  identity by slicing a string suffix.
- **Completion job id:** a Scheduler-assigned opaque integer. Worker code does
  not interpret route or attempt. PE Scheduler maps it to a Reverse completion;
  DE Scheduler maps it to a Reverse-send completion. `JobLedger` remains the
  sole all-worker counting authority.

`PathDecisionRequest` and DE pending admission remain keyed only by
`DualPathRequestKey`. The serialized `PE_READ` result remains exactly
`{request_key, path}` with `reverse_plan=None`. A `DE_READ` result adds
`reverse_attempt_id` and `prefill_local_tokens`; only its `ReversePlan` and
`ReverseReceiveBinding` use the complete attempt identity. Path-specific exact
field validation rejects an attempt-bearing `PE_READ` or a `DE_READ` missing
its attempt fields.

`ForwardPlan` remains a logical-keyed value containing range and block tables.
Both paths retain the current source plan under their logical request and may
replace it only after the prior source attempt is fenced (I6).

### 4.2 Required invariants

- **I1 — normal-path ownership.** While a `DE_READ` request is parked normally,
  its ordinary vLLM allocation remains owned until I4 publishes the current
  Reverse completion. The 2026-08-14 abort/watchdog exception deliberately does
  not extend this invariant beyond request termination.
- **I2 — identity follows the changing address.** Admission and Forward receive
  state use `DualPathRequestKey`; only DE_READ Reverse plans, bindings, wire
  map, terminals, latches, and cleanup lookups use `ReverseAttemptKey`. An
  admission lookup is never replaced by attempt-key membership.
- **I3 — stable Forward completion and idempotent overwrite.** A successful
  terminal means one complete valid write of the stable logical Forward range.
  Partial work publishes no terminal; a later source attempt deterministically
  overwrites the same stable destination range, so duplicate complete writes
  are safe. A normal-completion report that arrives after its source attempt
  was fenced and released is dropped: the replay overwrite makes it harmless,
  and it must not mutate any later source attempt.
- **I4 — current Reverse-attempt gate.** PE Workers report Reverse completion
  only as an opaque `{reverse_completion_job_id: 1}` in
  `DualPathWorkerMetadata`. Only the PE Scheduler, inside
  `update_connector_output` (which upstream invokes at `scheduler.py:2233`,
  before the generic finished-set processing at `:2236-2248`), maps the job
  back to its `ReverseAttemptKey`, validates that its `reverse_attempt_id`
  equals the logical request's `waiting_reverse_attempt_id` and that the
  request is still waiting for remote KV, and only then inserts the request id
  into the mutable `KVConnectorOutput.finished_recving`. Jobs from stale
  attempts are absorbed and never reach the generic finished sets.
- **I5 — all-worker proof.** A job or generic terminal is complete only after
  every expected participating worker has reported it exactly once.
- **I6 — fence before Forward source replacement.** No new Forward source plan
  or enqueue may become visible to any PE worker until every PE worker has
  consumed the FIFO barrier item for the old source attempt and the Scheduler
  has observed the all-worker barrier-job completion.
- **I7 — route-preserving recovery.** PE_READ rebuilds only its local Forward
  source attempt. DE_READ rebuilds that source attempt plus a new Reverse
  attempt from current `L_PE`. Neither path calls `PathPolicy` again.
- **I8 — retired exceptional-close invariant.** The original design required a
  `SAFE` reply from `CloseReverseAttempt` or verified teardown before releasing
  an uncertain Reverse destination. This invariant is no longer implemented:
  abort and recovery-watchdog expiry release immediately and accept a possible
  late write into a reallocated block. I4 remains the normal-completion gate.

## 5. Design

The active design separates three concerns: `JobLedger` completion aggregation;
Forward replay; and DE_READ Reverse-attempt renewal. The former hold ledger and
exceptional close protocol are retained below only as rejected design history.

### L0 — `JobLedger` and the retired hold ledger

Owner: `DualPathConnectorScheduler`. Both roles maintain one `JobLedger`: PE
uses it for Reverse completion and I4; DE uses it for Reverse-send completion.
The live record is:

```text
job_id -> { job_kind, reverse_attempt_key | None,
            expected_worker_count, completed_worker_count, failed, closed }
```

Active ledger rules:

- Each worker contributes at most one report per job; the Scheduler accumulates
  `completed_worker_count` across steps.
- When `completed_worker_count` reaches `expected_worker_count`, the
  kind-specific action runs exactly once and the job becomes `closed`. Reports
  arriving for a `closed` job are ignored, so duplicate or late reports cannot
  mutate a later attempt.
- A Reverse completion job and a DE reverse-send job each map to exactly one
  `ReverseAttemptKey`.
- A failure report closes the job as `failed`; the owning request is failed
  through the existing control-failure/watchdog path.
- The job ledger is the sole authority for reverse-send counting:
  `completed_worker_count`, `failed`, and `closed` for a `reverse_send_job_id`
  live only here. No second structure keeps a completion counter.
- Cleanup discards closed records with their owning request, including closed
  superseded reverse-send jobs. Open records remain reportable because
  `JobLedger.discard()` refuses them.

#### Retired hold acquisition

The earlier design added `HoldKind`, `HoldRecord`, `HoldLedger`,
`affected_hold_ids`, and `_reverse_destination_holds`, pinning the PE slice
`[L_PE, K_DE)` before Decision delivery. It was retired for two reasons:

1. On the normal path, a parked `DE_READ` request already owns its destination
   allocation until I4 publishes Reverse completion; the extra reference was
   redundant.
2. Its only additional protection was exceptional termination. Retaining that
   protection also required the close protocol, durable safety proofs,
   potentially unbounded failed holds, and two hold-pressure limits. The
   2026-08-14 decision instead accepts the parent/main immediate-free behavior.

Consequently `ledgers.py` contains `JobLedger` only, no KV block is pinned by
DualPath, and `VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS` plus
`VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS` are removed. Setting either legacy
environment variable has no effect.

#### Worker-to-Scheduler completion

```python
@dataclass
class DualPathWorkerMetadata(KVConnectorWorkerMetadata):
    completed_jobs: dict[int, int]
    failed_jobs: dict[int, int]

    def aggregate(self, other):
        assert isinstance(other, DualPathWorkerMetadata)
        # Add counts for identical job ids.
        ...
```

The `isinstance` check is mandatory: the upstream executor-level fold
(`distributed/kv_transfer/kv_connector/utils.py:130-139`) performs no type
check on worker metadata, unlike the stats field.

The full path is:

```text
Worker build_connector_worker_meta()
  -> ModelRunner KVConnectorOutput.kv_connector_worker_meta
  -> vLLM aggregate() across executor workers
  -> Scheduler update_connector_output()
  -> JobLedger transition
```

`get_finished()` keeps the base meanings of its generic sets. A validated
current-attempt Reverse completion becomes `finished_recving` only through the
I4 gate. A Decode request that finished with an open reverse-send job stays
delayed until JobLedger closes that job and injects `finished_sending`.

#### Retired hold release predicates

The table below records the rejected stronger design; none of these rows is a
live `HoldLedger` contract after 2026-08-14.

| Hold class and exit | Former required proof | Former release owner |
|---|---|---|
| Reverse destination, normal DE_READ | Every PE worker reports the current-attempt Reverse completion job; I4 validation passes | PE Scheduler `update_connector_output` |
| Reverse destination, uncertain Decision/abort | DE returns `SAFE` for `CloseReverseAttempt` | PE Scheduler |
| Decision never left PE | No remote writer was authorized | PE Scheduler immediately |
| Old DE process/transport teardown | Verified old transport can no longer issue DMA | Connector teardown cleanup |

The original design treated a Decision receipt, replacement receipt, watchdog
expiry, or `NOT_SAFE` reply as insufficient. The active implementation has no
hold to release: normal I4 completion still gates request resume, while abort
and watchdog expiry release ordinary ownership immediately.

### L1 — common stable Forward replay via synchronous fencing

#### Sender prerequisites (Stage-2 blockers)

The existing MooncakeLayerwise sender invalidates naive barrier use; three
fixes land before the fence protocol:

1. **Barrier items are processed outside `_transfer_kv_cache`.** The sender run
   loop (`mooncake_layerwise_connector.py:266-273`) dispatches barrier items on
   a dedicated path with exception-safe publication: the local event and the
   `{job_id: 1}` completion record are set in a `finally`-style path, so an
   exception can never strand a blocked model thread. Today `_handle_request`
   catches and swallows all transfer exceptions (`:275-283`); a barrier routed
   through that path could disappear without reporting its job.
2. **Fix the failed-request wrong-attribution bug.** On transfer failure the
   sender marks `self.failed_reqs.add(req_id)` using the stale loop variable
   from the outer request loop (`:507`). It must mark **every** member of
   `transfer_meta.req_ids` for the failed session; otherwise the final-layer
   callback (`:519-527`) can publish success for the actually failed request.
3. **Terminal-ACK failure becomes a worker failure fact.** A failed
   `send_done_send_signal` is currently only logged; it must surface as a
   failure fact for the affected request/job so the Scheduler terminates
   recovery instead of waiting forever.

#### Fence protocol

The Layerwise sender is a single thread consuming one FIFO queue, and
`batch_transfer_sync_write` returns synchronously. Model execution is serial
per worker, and `handle_preemptions` runs before the next forward pass, so
every old-attempt enqueue strictly precedes the fence in queue order. A
synchronous fence is therefore constructible without TransferEngine
cancellation:

1. In PE `build_connector_meta`, intersect
   `scheduler_output.preempted_req_ids` with delivered DualPath requests. On
   any hit, the Scheduler allocates **one barrier job per preempting pass**
   (a single FIFO point fences every old source attempt enqueued before it),
   records every fenced `ForwardSourceAttempt` on the job, sets each fenced
   attempt's `pending_barrier_job_id`, and carries the job in the connector
   metadata.
2. Worker `handle_preemptions` enqueues one FIFO barrier item on the sender
   queue and blocks on a local event until the sender thread consumes it.
3. The sender consumes the barrier item on the dedicated exception-safe path,
   sets the local event (unblocking the model thread), and records
   `{barrier_job_id: 1}` for the next `build_connector_worker_meta`.
4. PE Scheduler `update_connector_output` aggregates the barrier job across
   workers; at `expected_worker_count` it releases the fenced Forward source
   holds.

**Deadlock freedom.** The sender thread's only blocking waits are (a)
`queue.get`, which makes progress because the barrier item is already
enqueued; (b) `wait_event.synchronize()` (`mooncake_layerwise_connector.py:490`)
on NPU-recorded events from already-enqueued prior forward work — those events
are recorded by kernels already submitted to the NPU stream, which drains
independently of the model thread; and (c) the synchronous DMA call, which has
a bounded engine-level timeout. Blocking the model thread inside
`handle_preemptions` does not stall the NPU stream, so the sender always
reaches the barrier item, sets the event, and the model thread unblocks. No
circular wait exists.

The current queue has no `task_done()/join()` or barrier item; Stage-2 adds
this explicit primitive. Observing an empty queue is racy and is not a barrier.
The fence is required even when every layer callback for the last chunk has
already enqueued before preemption, as can happen with async scheduling.

#### After the fence: always release and replay

After a preemption fence completes, PE **always** releases the old Forward
source holds and replays the stable Forward range from the new allocation on
re-admission. There is no SATISFIED-versus-DRAINING arbitration:

- A complete old write needs no invalidation message: replay deterministically
  overwrites the same stable destination range (I3).
- A normal-completion report for an already-fenced source attempt may arrive
  late (a final send completion includes the synchronous DMA plus terminal ACK,
  `mooncake_layerwise_connector.py:519-527`). It is dropped: the holds are
  already released, replay makes it harmless, and it must not release holds or
  mutate a later source attempt.
- A send/control failure fact terminates recovery for that request.

PE may compute admission facts for the replacement while the fence is pending,
but such computation is provisional only: it must not mutate plans, holds,
`_reqs_need_send_layerwise`, or control state, must not expose the new source
plan, and must not enqueue new Forward work until every participating PE worker
has reported the barrier job (I6). The resume-admission entry point defers
instead of returning partial results (§L2 "Resume admission").

For PE_READ, the post-fence replacement can immediately rebuild and replay its
Forward source. For DE_READ, the source attempt is rebuilt but cannot run until
the new Reverse attempt described below completes.

```mermaid
sequenceDiagram
    participant PS as PE Scheduler
    participant PW as PE Worker (model thread)
    participant ST as PE Sender thread
    participant DW as DE Workers

    PS->>PW: expose held Forward source attempt
    PW->>ST: enqueue per-layer Forward writes (FIFO)
    ST->>DW: synchronous Forward writes
    PS->>PS: observe preempted_req_id; create barrier job
    PS->>PW: metadata carries barrier job
    PW->>ST: enqueue FIFO barrier item
    PW->>PW: block on local event
    ST->>ST: finish all prior FIFO work, consume barrier
    ST-->>PW: set local event
    ST-->>PS: completed_jobs[barrier_job] per worker
    PS->>PS: all-worker barrier complete; release old source holds
    PS->>PS: re-admit; rebuild source attempt from current L_PE
    PS->>PW: expose newly held source attempt
    PW->>ST: replay the same stable Forward range
    ST->>DW: overwrite stable Forward range
```

### L2 — route-preserving recovery and DE_READ Reverse attempts

#### Token boundaries may change

Let the DE admission record freeze `L_DE`, `K_DE`, `T`, its Decode allocation,
and DE source coverage through `K_DE`. Let `L_PE(current)` be the prefix PE has
at each admission. Preemption may make it smaller or larger than before.

Recovery recomputes ranges without calling `PathPolicy`:

| Fixed path | Recovery identity | Reverse | PE recompute | Stable Forward |
|---|---|---|---|---|
| `DE_READ` | logical request + new `reverse_attempt_id` | `[L_PE(new), K_DE)`; empty when `L_PE(new) >= K_DE` | `[max(L_PE(new), K_DE), T)` | `[K_DE, T)` |
| `PE_READ` | logical request + new local source jobs | none | `[L_PE(new), T)` | `[L_DE, T)` |

For DE_READ, a smaller PE prefix asks DE to resend a larger Reverse range from
the same stable Decode table. DE already has, or its logical Store load
provides, `[0, K_DE)`; `K_DE` and the path do not change. When
`L_PE(new) >= K_DE`, the new Reverse attempt is vacuously complete.

For PE_READ, PE prefix loss only changes how PE produces `[L_DE,T)`. Its
Decision contains no PE block address or PE-derived boundary, so PE sends no
replacement Decision and DE does not reactivate anything.

For both paths, the stable Forward binding lets DE combine a valid completed
rank terminal with later duplicate overwrites. DE retains the binding while
completion is pending and retains its consumed-terminal tombstone through final
request cleanup so a late duplicate cannot be attributed to another request.

#### Resume admission (replaces the invalid-set convergence)

`_decide_prefill_path_for_admission` gains a concrete resume branch, evaluated
**before** `PathDecisionDecider`:

1. Detect delivered DualPath state for the logical request (a committed
   Decision exists). This check precedes any decider call; the current code
   instead treats the changed prefix as invalid and converges into
   `_prefill_invalid_request_ids` (dual_path `scheduler.py:321-322`,
   `:350-362`), which this branch replaces for delivered decisions.
2. **Barrier gate.** If the logical request's Forward barrier job is still
   pending (all-worker completion not yet observed in
   `update_connector_output`), return `(None, False)`. Upstream permits a
   connector to return `None` from `get_num_new_matched_tokens`
   (`vllm/distributed/kv_transfer/kv_connector/v1/base.py:453-477`); the
   scheduler then moves the request to `skipped_waiting` and retries it in a
   later pass (`vllm/v1/core/sched/scheduler.py:615-629`). Only after
   `update_connector_output` observes all-worker barrier completion may resume
   compute `L_PE(new)`, return route-specific results, allocate replacement
   blocks, or deliver a replacement Decision.
3. Reuse the frozen `PathKind`; never re-run eligibility or `PathPolicy`.
4. `DE_READ`: return `(max(K_DE - L_PE(new), 0), True)` — the Reverse token
   count as external tokens, parking the request in `WAITING_FOR_REMOTE_KVS`.
   When `L_PE(new) >= K_DE` the Reverse is vacuous and bypasses the Reverse
   machinery entirely: return `(0, False)`; no
   `reverse_completion_job_id` is allocated, no
   `waiting_reverse_attempt_id` is installed, the request never enters
   `WAITING_FOR_REMOTE_KVS`, and Forward replay proceeds once the barrier gate
   is open.
5. `PE_READ`: return `(0, False)`; recovery rebuilds only the local Forward
   source plan.

`PathDecisionDecider.decide` (`path_decision.py:186-189`) is never consulted on
this path, so its changed-facts rejection cannot fire.

#### Normal DE_READ preemption

Normal scheduler preemption starts from a completed Reverse attempt:

```text
ReverseAttempt(N): DONE
  -> PE RUNNING / Forward in flight
  -> preemption
  -> fence (all-worker barrier) -> old source work drained
  -> re-admission (resume branch)
  -> ReverseAttempt(N+1): submitted -> DONE      # skipped when vacuous
  -> Forward replay -> complete
```

No close message exists on this path. The all-worker Reverse completion that
promoted the request from `WAITING_FOR_REMOTE_KVS` is already the I4 terminal
for attempt N; normal request ownership remains valid through that transition.

PE creates a greater `reverse_attempt_id`, captures current `L_PE` and the new
PE block table, and delivers a new DE_READ Decision only after the old local
Forward fence completes (I6). The
already-preempted PE scheduler request still follows normal vLLM
re-admission/completion bookkeeping.

DE validates that the logical admission still owns the same path, `K_DE`, `T`,
Decode allocation, and source coverage. It then installs a fresh Reverse plan
and latch for the new attempt. The logical Forward binding is neither replaced
nor reinstalled.

```mermaid
sequenceDiagram
    participant PS as PE Scheduler
    participant PW as PE Workers
    participant DC as DE Control/Scheduler
    participant DW as DE Workers

    DW-->>PW: ReverseAttempt(N) completes on every worker
    PW-->>PS: {reverse_completion_job_N: 1} per worker
    PS->>PS: I4 gate: attempt == waiting, still waiting -> finished_recving
    PS->>PS: request RUNNING with normal ownership
    PS->>PW: compute and enqueue stable Forward range
    PS->>PS: preemption; create barrier job in metadata
    PW-->>PS: all-worker barrier job completion
    PS->>PS: old Forward source work drained
    PS->>PS: resume admission; recompute current L_PE; keep PathKind=DE_READ
    PS->>DC: Decision(request_key, attempt M, current L_PE, new PE blocks)
    DC->>DC: validate stable admission; install fresh Reverse plan/latch
    DC-->>PS: Decision ACK
    DC->>DW: submit ReverseAttempt(M)
    DW-->>PW: write [L_PE(M), K_DE) into new PE blocks
    PW-->>PS: {reverse_completion_job_M: 1} from every worker
    PS->>PS: I4 gate passes; resume request
    PS->>PW: compute remainder + replay stable Forward range
```

This normal sequence uses Decision messages only: N completed before the
request became preemptible, and M does not exist until the new Decision is
sent.

#### Decision channel framing and registry semantics

The wire envelope keeps an explicit kind field, but the only supported kind is
`Decision`. The Decode endpoint warns and drops any other kind without a
reply. In particular, an old `CloseReverseAttempt` sender retries until its
delivery budget is exhausted; mixed old/new deployments are unsupported.

Decision delivery remains at-least-once: an attempt can be retried after an
uncertain ACK, and delivery Futures for multiple attempts may overlap. The
execution-side registry therefore remains attempt-aware under one logical key:

- equal `reverse_attempt_id` and identical payload: acknowledge as an `ACK`
  duplicate;
- equal attempt and conflicting payload: reject as `PROTOCOL_ERROR`;
- greater attempt: accept and replace the retained Decision;
- lower attempt, or an attempt at/below `closed_through_attempt_id`: reply
  `STALE_CLOSED`, never enqueue or reopen it;
- logical key not registered: reply `UNKNOWN_REQUEST` without creating state.

`reverse_attempt_id` remains the control-plane epoch and is sourced from
`request.num_preemptions`. `_register_decision_locked` performs the duplicate,
conflict, and monotonicity checks. When attempt M is accepted,
`_close_skipped_lower_attempts_locked` advances
`_closed_through_attempt_ids` through M-1; delayed lower attempts then receive
`STALE_CLOSED`. This watermark rejects stale execution commands only. It is not
a remote-writer safety proof and no longer participates in abort-time release.

#### Attempt-keyed worker state

Stage-2 re-keys exactly the following DE/PE worker structures by
`ReverseAttemptKey`:

- Reverse receive bindings and the reverse wire map (currently installed and
  collision-rejected per logical request at dual_path `worker.py:216-226`,
  `_reverse_request_map` at `:228`);
- Reverse plans;
- the DE sender latch `reverse_submitted` (currently per logical tracker at
  `worker.py:443`);
- pending local reverse terminals (currently `worker.py:102-104`, latched at
  `:531-539`, drained at `:588-617`);
- consumed reverse terminal tombstones (currently
  `_consumed_reverse_terminal_wire_ids` at `:260-265`);
- reverse tracker cleanup (an old attempt's tracker is removed together with
  its tombstones after its terminal is consumed; a normally completed
  attempt's tracker may be removed when its replacement attempt is installed).

These remain logical-keyed (by `DualPathRequestKey` / request id): the stable
Forward receive binding and its completion state, Forward plans, and admission
state.

The PE-worker-side mapping chain that feeds the I4 gate is: the
`ReverseReceiveBinding`/`ReversePlan` metadata carries the full attempt
identity; the worker maintains
`wire_request_id -> ReverseAttemptKey -> reverse_completion_job_id`; and on
consuming a reverse wire terminal it reports `{reverse_completion_job_id: 1}`
in its worker metadata. Attempt identity never leaves the worker as a raw
request id.

#### DE-owned logical and attempt state

DE owns one logical admission record:

```text
DecodeAdmissionRecord
  - request / frozen admission snapshot
  - original PathKind
  - stable Decode block table, K_DE, T
  - logical Store state
  - stable Forward binding and completion state
  - current_reverse_attempt
  - latest_reverse_attempt_id
  - closed_through_attempt_id
```

A Reverse attempt contains its receipt/activation state, Decision, Reverse
plan/binding, tracker/latch references, jobs (including its
`reverse_send_job_id`), deadline, and terminal result.
Normal replacement discards the completed current attempt before installing the
next, so DE holds at most one live Reverse attempt per logical request plus the
monotonic `closed_through_attempt_id` watermark. No staged-attempt slot and no unbounded
`dict[ReverseAttemptKey, DecodeLifecycleState]` exist.

`closed_through_attempt_id` remains authoritative for as long as the logical
admission exists: a delayed Decision for a closed attempt is rejected
permanently with `STALE_CLOSED`.

### DE Reverse sender completion proof

Normal replacement and finishing-request cleanup need the fact that the DE
writer finished. Stage-2 obtains it from the same job mechanism as everything
else:

- When the DE Scheduler creates a `ReverseAttemptKey`, it allocates a
  `reverse_send_job_id`, carried unchanged on the `ReversePlan` to every DE
  worker, and registers the attempt as accepted **before** any layer enters the
  sender queue.
- Each DE worker emits `{reverse_send_job_id: 1}` exactly once, only after its
  final synchronous write AND a successful terminal ACK. A DMA failure or a
  terminal-ACK failure records the job as failed, never as success.
- DE Scheduler `update_connector_output` aggregates reports in the job ledger
  across `expected_worker_count`; on full success the reverse-send job becomes
  `closed` (not `failed`). The job ledger is the sole counting authority. A
  finished Decode request with an open reverse-send job remains delayed so
  zero-token engine steps can harvest the reports; when the job closes, the
  Scheduler injects `finished_sending` and reclaims the closed record.

### Retired exceptional cleanup — `CloseReverseAttempt`

> **Retired on 2026-08-14.** This section preserves the stronger design and its
> close matrix for review history; it is not an implemented protocol or release
> gate. The active PE path sends no close, retains no Reverse destination hold,
> and releases immediately on abort or recovery-watchdog expiry.

The retired `CloseReverseAttempt` asked one narrow question:

> Can DE still activate or issue a Reverse write for this attempt into its PE
> destination block table?

The old PE design would have used it only when normal Reverse-completion proof
was unavailable:

- DE accepted a Decision but its Scheduler may not yet have activated it;
- the PE request is aborted/cancelled while waiting for Reverse;
- Decision delivery ACK is uncertain; or
- a control/transport error requires cleanup while the old DE endpoint can
  still answer.

Request:

```text
CloseReverseAttempt(request_key, reverse_attempt_id)
```

The retired responses were:

| Result | Meaning | May PE release the Reverse destination hold? |
|---|---|---|
| `SAFE` | activation/submission is closed AND (no worker work was published OR `sender_complete`) | Yes |
| `NOT_SAFE` | a writer may still be live; PE retains the hold | No |

#### Retired lifecycle stages and close matrix

Four lifecycle stages, defined against the real channel code, determine the
reply:

1. **Logical admission existence** — the request key is registered as pending
   (`path_decision_channel.py:326-332`) or retained as accepted.
2. **Decision receipt** — the Decision is validated and retained in the
   accepted registry (`path_decision_channel.py:450-460`).
3. **Activation claim** — the Scheduler consumes the received Decision (queue
   handoff at `path_decision_channel.py:460-461`) and claims activation under
   the registry lock.
4. **Worker publication** — the ReversePlan is published to Workers and at
   least one layer enters the sender queue.

Under the retired registry lock,
`CloseReverseAttempt(request_key, reverse_attempt_id)` would have replied by
this exact matrix. **Every** successful close transition would atomically
persists a minimal safe-close proof (a `ClosedReverseAttemptRecord`, below)
**before** the response is sent, so a lost `SAFE` response followed by an
identical retry still returns `SAFE`:

| Logical admission / attempt record / activation-work state | Response |
|---|---|
| retained safe-close proof for the exact `ReverseAttemptKey` exists (checked first, including after admission teardown) | `SAFE` |
| otherwise, admission absent | `NOT_SAFE` (no record is created; nothing was ever provably safe) |
| admission present, attempt never received | close, advance `closed_through_attempt_id`, persist proof; `SAFE` |
| Decision received, activation unclaimed | close the attempt, suppress activation, persist proof; `SAFE` |
| activation claimed, publication cancelled successfully by the claim owner | close, persist the no-publication proof; `SAFE` |
| activation claimed or publication uncertain, accepted work incomplete | `NOT_SAFE`; record retained awaiting the send job |
| worker work published, all DE workers complete (`sender_complete`) | close, persist the completed-send proof; `SAFE` |

In short, `SAFE` requires exactly: activation/submission closed AND (no worker
work was published OR `sender_complete`).

Close-retry rule: an identical close for an attempt retained as safely closed
returns `SAFE`. A close covered only by the rejection watermark without a
retained safety proof returns `NOT_SAFE` — "cannot reopen" is not "writer
proven stopped". A close for an attempt at or below
`closed_through_attempt_id` therefore returns `SAFE` only when a retained safe
closure exists, never merely because the number is old.

#### Retired `ClosedReverseAttemptRecord`

Every close on an existing admission — `SAFE` or `NOT_SAFE` — would have
created (or retained) one internal record:

```text
ClosedReverseAttemptRecord {
    attempt_key, activation_claimed, worker_work_published,
    reverse_send_job_id, safe_close_proof }
```

The record holds only the attempt key, the publication/closure flags, the
`reverse_send_job_id` reference, and the safe-close proof bit. It keeps **no**
completion counter: the job ledger is the sole authority for reverse-send
counting, and the record's `sender_complete` is read through that reference. A
record created on a `NOT_SAFE` reply converts to the retained `SAFE` proof when
its referenced reverse-send job closes not-failed (or when publication is later
proven never to have happened).

This record is not externally visible and is neither a staged attempt, a
diagnostic tombstone, nor a watermark:

- worker-side request cleanup (cf. dual_path `worker.py:237-245`, `:541-546`)
  must not delete its pending sender-completion job;
- DE `update_connector_output` keeps aggregating `reverse_send_job_id` reports
  into the job ledger referenced by the record;
- the record becomes `SAFE`-answerable once no work was published or the
  referenced job closed without failure;
- a retained `SAFE` proof survives until **PE acknowledgment of the `SAFE`
  response** or verified old-process/transport teardown. Logical-admission
  teardown does not erase safe-close proofs: it retains an independent minimal
  proof (`attempt_key` + `SAFE`) for safely-closed attempts until verified
  teardown, so an admission cleanup racing a dropped `SAFE` cannot turn a
  proven-safe attempt back into `NOT_SAFE`. No TTL tombstones exist;
- it can never accept Decisions: `closed_through_attempt_id` keeps rejecting
  them with `STALE_CLOSED`.

This protocol was removed in full: no `ClosedReverseAttemptRecord` is retained,
no `SAFE` proof is produced, and no close retry can release memory. The
`closed_through_attempt_id` watermark remains only for stale Decision
rejection.

#### Retired single-ROUTER close concurrency contract

The rejected design had Decision delivery and `CloseReverseAttempt` share the
single receiver thread and ROUTER socket. Its intended `_handle_frames`
contract was bounded in-memory work:

- it never waits on a Worker, queue, Future, or condition variable;
- Decision receipt, close, and status lookup use the same registry lock, so
  their ordering matches ROUTER receive order;
- the close handler evaluates the close matrix under the registry lock,
  atomically closing the attempt to new activation/submission where the matrix
  requires it; every close creates or updates the `ClosedReverseAttemptRecord`
  and persists the safe-close proof before any `SAFE` reply is sent;
- Scheduler/Worker code closes the reverse-send job in the job ledger
  (`sender_complete`) as a short locked transition, converting the record to
  the retained `SAFE` proof, and a later close retry for that attempt then
  observes the proof and replies `SAFE`;
- every request receives its reply before the receiver reads the next request,
  so one closing attempt never blocks other decisions.

Receipt-to-activation uses an atomic claim: under the registry lock, Scheduler
may claim activation only while the attempt is open. If a close wins,
activation is suppressed; if activation already claimed, the close is recorded
and the activation path must either cancel before Worker publication or
register the worker work the close will wait for.

The following is the historical flow, not the active wire protocol:

```mermaid
sequenceDiagram
    participant PC as PE Control/Scheduler
    participant DR as DE ROUTER thread
    participant DS as DE Scheduler
    participant DW as DE Workers

    PC->>DR: CloseReverseAttempt(request_key, attempt N)
    alt close matrix proves SAFE (never received / activation suppressed / publication cancelled / sender_complete / retained proof)
        DR->>DR: close N; advance closed_through; persist retained SAFE proof
        DR-->>PC: SAFE
        PC->>PC: release N destination hold
    else writer may be live
        DR->>DR: close N; record ClosedReverseAttemptRecord
        DR-->>PC: NOT_SAFE (non-blocking)
        PC->>PC: retain N destination hold
        DS->>DW: accepted Reverse work completes on all workers
        DS->>DR: reverse-send job closed in job ledger (locked transition)
        DR->>DR: record converts to retained SAFE proof
        PC->>DR: retry CloseReverseAttempt(N)
        DR-->>PC: SAFE
        PC->>PC: release N destination hold
    end
```

The active endpoint accepts Decision messages only. A legacy close kind is
logged as a warning and dropped without a reply. An old PE therefore exhausts
its `_deliver_close` retries against a new DE. This is intentional evidence
that old/new mixed deployment is unsupported, not a compatibility fallback.

### Abort release (replaces the retired dual-release)

The former dual-release distinguished ordinary vLLM ownership from a
connector-owned Reverse destination hold. After 2026-08-14 there is only the
ordinary parent/main release:

1. PE `request_finished` adds no close request, pending-release injection, or
   Reverse hold. `_delay_free_for_connector` returns `False` on the Prefill
   side.
2. The aborted request and its ordinary destination ownership are released by
   the parent lifecycle immediately; recovery-watchdog expiry follows the same
   terminal path.
3. No remote safety proof is awaited. If DE has already published or is still
   executing Reverse work, a late DMA can write a block that PE has reallocated
   to another request. This is the accepted behavioral risk and is symmetric
   with the parent/main P-to-D direction, with DualPath adding D-to-P exposure.

`reverse_attempt_id`, `_closed_through_attempt_ids`, and `STALE_CLOSED` do not
make this release safe; they only prevent delayed control messages from
reopening a lower execution epoch.

### RUNNING-request abort fence

Upstream abort of a delivered **RUNNING** dual_path request never populates
`preempted_req_ids`: `finish_requests` (`scheduler.py:1825-1886`) removes the
request from the running queue (`:1859-1860`, `:1867-1868`), marks it finished,
and calls `_free_request` (`:1888-1905`) → `_connector_finished` → the
connector's `request_finished`. The preemption fence path therefore never
fires for it, and without a fence its queued Forward work could read blocks
whose ordinary ownership is being freed. Stage-2 adds the abort fence:

1. PE `request_finished` detects an unreleased `ForwardSourceAttempt` for the
   request and returns `True` (connector-owned delayed free), so upstream
   retains ordinary ownership (`scheduler.py:1901-1903` skips `_free_blocks`).
2. The PE Scheduler creates an abort-fence barrier job associated with the
   attempt (recorded on the job and as the attempt's
   `pending_barrier_job_id`), carried to workers in the next connector
   metadata. Delivery is guaranteed even with no runnable request: the
   no-forward output step still runs (`vllm/v1/worker/gpu/model_runner.py:1098-1101`
   → `vllm/v1/worker/gpu/kv_connector.py:98-105`); see "Engine progress" below.
3. Every worker's `handle_preemptions` enqueues one FIFO barrier item and
   blocks on its local event until the sender consumes it — the same mechanism
   as the preemption fence.
4. All-worker aggregation in `update_connector_output` releases the Forward
   source hold.
5. The Scheduler then injects the finished request id into
   `finished_sending`. This is the legal channel for a `request_finished=True`
   request: `_free_request` already supplied the id through `finished_req_ids`
   (`scheduler.py:1897`, carried into `SchedulerOutput` at `:945`), satisfying
   the `get_finished` contract (`base.py:357-373`), and the upstream
   `finished_sending` branch unconditionally frees the delayed blocks
   (`scheduler.py:2245-2248`). `finished_recving` would also pass the status
   assertion for a finished request (`:2242-2244`), but `finished_sending` is
   the canonical delayed-free completion channel and is used here; the
   `finished_recving` injection remains reserved for the abort-while-waiting
   path above, whose requests never returned `True` from `request_finished`.

### Engine progress while connector jobs are pending

An open DE reverse-send job may outlive the last ordinary Decode request; if no
further model steps occurred, no connector outputs would be produced and the
job would never aggregate. The rule that prevents this:

- While the connector owns an open reverse-send job for a finished Decode
  request, that request returns `True` from `request_finished` (delayed
  free). Such a request remains in the scheduler's `self.requests` after
  leaving the scheduling queues, so `has_finished_requests()`
  (`scheduler.py:1931-1941`) returns `True`, `has_requests()`
  (`v1/core/sched/interface.py:185-188`) stays `True`, and `EngineCore.step()`
  keeps running (its only early return is `not has_requests()`,
  `v1/engine/core.py:452`).
- Each such step calls `schedule()`; with zero schedulable tokens the worker
  takes the no-forward path (`v1/worker/gpu/model_runner.py:1098-1101`), whose
  `post_forward` still calls `get_finished` and `build_connector_worker_meta`
  (`v1/worker/gpu/kv_connector.py:98-105`), producing a `KVConnectorOutput`.
  The scheduler's `update_from_output` then invokes
  `update_connector_output` (`scheduler.py:2233`), so job aggregation and
  `finished_sending` injection keep making
  progress with no ordinary request running.
- A finished request with no pending connector work must return
  `(False, None)` from `request_finished` so the engine can go idle.
- The bounded-watchdog fallback fails the request. On PE it does not synthesize
  a remote proof or retain a destination hold.

### Cleanup and bounded waits

- Final logical cleanup removes admission, Forward, and current Reverse state;
  it does not retain a close record or destination hold.
- Scheduler `_release_scheduler_request_state` reclaims every closed JobLedger
  record owned by the request, including superseded reverse-send attempts. An
  open record remains reportable because `JobLedger.discard()` refuses it.
- `closed_through_attempt_id` lives only with the active logical admission and
  is used by `_register_decision_locked` to reject stale Decisions. It is not a
  post-cleanup safety proof.
- The PE recovery watchdog and DE progress watchdog remain. PE expiry fails the
  request and follows immediate-free semantics; DE expiry continues to bound
  Store, Forward, and Reverse progress.
- There are no hold-pressure limits. The removed
  `VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS` and
  `VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS` variables no longer have any
  effect.

## 6. Parallel topology and ordering contract

Stage-2 neither silently narrows matching `TP>1` nor silently expands Stage-1
to unaccepted topologies.

### 6.1 Supported now

- `pipeline_parallel_size == 1`;
- `data_parallel_size == 1`;
- PE and DE have equal `tensor_parallel_size`, using the existing one-to-one
  rank mapping; matching `TP>1` is supported;
- `prefill_context_parallel_size == 1` and
  `decode_context_parallel_size == 1`;
- the remaining Stage-1 model, KV-group, layout, dtype, and block-size
  restrictions remain unchanged.

**This validation is new Stage-2 work, not current behavior.** DualPath today
validates role and ports but does not reject all listed PP/DP/PCP/DCP
combinations and does not validate PE/DE TP equality (the parent Decode
scheduler rejects PCP, which is not the full matrix). Stage-2 adds:

- local checks during connector construction for the local-side restrictions;
- remote checks from bootstrap/plan metadata before Decision commit and before
  acquiring DualPath holds. The existing bootstrap/meta exchange already
  carries the incarnation-qualified identity `decode_engine_instance_id =
  f"{engine_id}:{data_parallel_rank}:{incarnation}"`
  (`path_decision_channel.py:272-273`); Stage-2 extends that bootstrap metadata
  with the parallel-topology fields (`tensor_parallel_size`,
  `pipeline_parallel_size`, `data_parallel_size`,
  `prefill/decode_context_parallel_size`) needed for the compatibility matrix.

Every violation raises an explicit configuration/control error; no request
enters the partial protocol.

### 6.2 Reused TP barrier

Every Scheduler-created completion/barrier job is sent unchanged to every
participating local executor worker. Each worker reports `{job_id: 1}` exactly
once only after all rank-local subtransfers represented by that job complete.
The Scheduler accumulates counts across steps and completes at the recorded
`expected_worker_count`.

`expected_worker_count` is sourced directly from
`vllm_config.parallel_config.world_size`, which equals TP under the supported
`PP == PCP == DP == 1` topology (`vllm/config/parallel.py:774-784`). A
rank-local terminal can never release a process-wide hold. A missing or failed
rank leaves the job incomplete and takes the watchdog/fail-closed path.

One barrier job per preempting pass is preferred over per-request barrier jobs:
the single FIFO barrier item on each worker's sender queue already fences every
old source attempt enqueued before it.

### 6.3 PCP/DCP reuse boundary

MooncakeLayerwise already contains PCP/DCP token/block mapping, and vLLM
defines executor `world_size = TP * PP * PCP`; DCP reuses TP workers and does
not add executor workers. A future topology-expansion stage can reuse the
protocol and job aggregation:

- PCP ranks naturally add participating Workers;
- DCP contributes rank-local subtransfers that finish before that Worker
  reports its one job completion; and
- the barrier counts participating Workers, never raw direction terminals.

That reuse is not sufficient evidence to enable PCP/DCP in Stage-2. Stage-1
excludes them and the Decode Layerwise scheduler currently rejects PCP.
Stage-2 therefore explicitly rejects PCP/DCP rather than accepting undefined
attempt ordering. Removing the rejection requires a separate topology
acceptance matrix, not a change to the route-specific recovery model.

## 7. Failure-mode to mechanism matrix

| Failure | Ownership / fence | Identity / plan handling | Completion / publication |
|---|---|---|---|
| F1 Forward read-after-free | Forward source hold plus all-worker synchronous sender fence (I6), on preemption or on RUNNING-request abort | replace only PE-local source attempt | all-PE-worker completion/barrier jobs |
| F2 DE_READ re-admission parks forever | ordinary request ownership while parked | resume-admission branch reuses frozen PathKind; create new `reverse_attempt_id` and plan | only the current attempt resumes PE (I4) |
| F3 exhausted Reverse latch | no old-plan reuse | fresh per-attempt tracker and latch keyed by `ReverseAttemptKey` | old DONE attempt retired on normal replacement |
| F4 stale Reverse terminal | ordinary ownership remains until current completion | Worker reports an opaque completion job; PE Scheduler maps job_id to `ReverseAttemptKey` | I4 job-based gate absorbs stale-attempt jobs; only PE Scheduler publishes `finished_recving` |
| F5 stale Reverse destination | new ordinary allocation is used for the new attempt | new plan uses current PE block table | old plan never resubmitted |
| F6 exceptional abort/watchdog expiry | immediate parent/main release; no Reverse hold | Decision epoch rejects delayed lower attempts but cannot stop an already published writer | accepted risk: late DE DMA may write a reallocated PE block |

## 8. Connector facade requirements

`DualPathConnector` explicitly delegates every safety hook to its Scheduler or
Worker implementation. Relying on base-class no-ops would silently disable the
design. Required forwarding includes:

- `bind_gpu_block_pool`;
- `handle_preemptions`;
- `build_connector_worker_meta`;
- `update_connector_output`;
- existing `get_finished`, `request_finished`, and `shutdown` behavior.

The same metadata remains composable under `MultiConnector`; aggregation must
preserve the concrete `DualPathWorkerMetadata` type (its `aggregate` performs
the `isinstance` assertion the upstream fold omits), and `MultiConnector`'s
`bind_gpu_block_pool` / `handle_preemptions` forwarding must reach the DualPath
sub-connector.

## 9. Test strategy for implementation

No test execution is part of this design-only update. Tests land alongside
each implementation step of §10. Coverage must include:

**Step 1 — sender fixes and barrier primitive:**

- Barrier items are processed outside `_transfer_kv_cache`; an injected
  exception on the barrier path still sets the local event and records the job
  (no stranded model thread).
- Multi-request/session failure attribution: a failed session marks every
  member of `transfer_meta.req_ids` failed (regression for
  `mooncake_layerwise_connector.py:507`), and the final-layer callback never
  publishes success for a failed request.
- Terminal-ACK failure produces a worker failure fact, not only a log.
- Barrier metadata handoff: scheduler-created barrier job reaches
  `handle_preemptions`; enqueue precedes the next forward; FIFO ordering of old
  work before the barrier item; duplicate barrier item consumption is
  idempotent.

**Step 2 — JobLedger and retired hold lifecycle:**

- Retired coverage recorded the former HoldLedger acquisition/exit contract;
  the active no-pinning regression instead asserts that Reverse destination
  refcounts are unchanged and hold symbols are absent.
- Job aggregation across steps and matching `TP>1`; no completion before every
  expected worker reports exactly once; `expected_worker_count` equals
  `parallel_config.world_size`.
- `DualPathWorkerMetadata.aggregate` rejects a foreign metadata type.
- Duplicate-report and late-report oracles: a second report from the same
  worker never counts twice, and a report for a `closed` job is ignored without
  mutating later attempts.
- RUNNING-request abort fence: (1) abort while RUNNING with queued Forward
  work — `request_finished` returns `True`, the abort fence is created, and the
  source hold is released only after all-worker barrier completion; (2) abort
  before the final layer; (3) abort after the DMA but before the terminal ACK;
  (4) sole-request RUNNING abort — the no-forward output step still delivers
  the barrier metadata, and `finished_sending` injection frees the delayed
  ordinary ownership; (5) TP=2 with a late worker barrier — the hold is
  released only after both workers report.
- Engine progress, PE: a sole normally finished PE request with a pending
  normal Forward completion job — `request_finished` delays the free, and
  zero-token steps harvest the all-worker result, release the source hold,
  inject `finished_sending`, free ordinary ownership, and let the engine go
  idle.
- Engine progress, DE: a sole DE request with an open reverse-send job and no
  runnable request remaining — zero-token steps harvest all worker reports,
  inject `finished_sending`, and reclaim the closed JobLedger record.

**Step 3 — attempt identity and I4 gate:**

- Path-specific schemas: PE_READ remains `{request_key, path}` with
  `reverse_plan=None`; DE_READ serializes `reverse_attempt_id`, current `L_PE`,
  and an attempt-unique Reverse wire id. Forward binding/wire identity remains
  logical and stable on both paths.
- Attempt-keyed worker state: bindings, wire map, plans, `reverse_submitted`
  latch, pending local terminals, and consumed tombstones are
  `ReverseAttemptKey`-keyed; installers accept attempt M while retaining
  attempt N tombstones; old trackers are removed only under the §5 rule.
- The I4 gate: workers emit only `{reverse_completion_job_id: 1}`; a
  stale-attempt job is absorbed and never reaches generic `finished_recving`;
  the current-attempt job inserts the req id only while the request is still in
  `WAITING_FOR_REMOTE_KVS`; PE workers no longer convert reverse terminals to
  req ids directly.
- TP>1 Reverse mapping assertion at dual_path `worker.py:412-415` holds under
  the attempt-keyed mapping.
- DE Reverse sender completion proof: `reverse_send_job_id` is allocated at
  attempt creation and carried unchanged on the `ReversePlan`; partial TP
  completion never closes the reverse-send job; a
  terminal-ACK failure records the job as failed, never success; abort before
  the final layer leaves the job incomplete; final DE worker completion closes
  the ledger record without a close-protocol side effect.

**Step 4 — resume admission:**

- Delivered-state detection precedes `PathDecisionDecider`; the frozen
  `PathKind` is reused; policy is never re-run.
- DE_READ resume returns `(max(K_DE - L_PE(new), 0), True)`; cover
  `L_PE(new) <`, `==`, and `>` both `L_PE(old)` and `K_DE`; the vacuous case
  returns `(0, False)` and completes immediately.
- PE_READ resume returns `(0, False)` and rebuilds only the local source plan.
- The old invalid-set convergence no longer fires for delivered decisions.
- Barrier-gated defer: multiple `schedule()` passes before the barrier output
  returns must all defer with `(None, False)` and perform no replacement
  allocation, plan mutation, or Decision delivery; the first pass after
  all-worker barrier completion resumes normally.

**Step 5 — Decision-only receiver schema and registry:**

- Message-kind field accepts `Decision`; a legacy `CloseReverseAttempt` kind is
  warned and dropped without a reply.
- Decision registry matrix: equal+identical ACK duplicate; equal+conflicting
  protocol error; greater attempt accepted under the serialization rule;
  closed/lower attempt `STALE_CLOSED`; unknown logical key `UNKNOWN_REQUEST`.
- `_accepted_decisions` migration: an existing single-decision registry entry
  upgrades to attempt-aware comparison without rejecting legitimate retries.
- Decision response encoding round-trip and at-least-once retry idempotence;
  overlapping attempt deliveries still reject lower epochs as `STALE_CLOSED`.

**Step 6 — replay wiring:**

- Common Forward replay: no DE binding replacement; source plan changes only
  after the all-worker fence; coincidentally equal block ids still replay;
  duplicate complete writes are idempotent (I3).
- A normal-completion report arriving after the fence for the released source
  attempt is dropped: it releases nothing and mutates no later attempt.
- Exact ranges: DE_READ Reverse expands, shrinks, or is vacuous while Forward
  remains `[K_DE,T)`; PE_READ Forward remains `[L_DE,T)`.
- Normal DE_READ preemption starts from Reverse `DONE`, uses no close or
  Reverse hold, and creates N+1 only after the fence.

**Step 7 — close/hold retirement and abort semantics:**

- No PE close-driver fields or methods, close retry backoff, close message/reply
  schema, close registry, `ClosedReverseAttemptRecord`, HoldLedger type, hold
  map, or hold-pressure limit remains.
- The Decision envelope accepts `Decision` only. A literal legacy
  `CloseReverseAttempt` message produces one warning and no reply; this pins the
  unsupported mixed-version behavior.
- Abort while waiting and PE recovery-watchdog expiry create no connector
  delayed-free state and retain no destination reference. Ordinary ownership is
  released immediately through the parent/main lifecycle.
- No-pinning regressions assert unchanged block refcounts and absence of the
  retired scheduler surface.
- The epoch contract remains independently covered: `reverse_attempt_id` comes
  from `num_preemptions`; duplicate/equal, conflicting/equal, greater, and
  stale/lower Decisions exercise `_register_decision_locked`,
  `_closed_through_attempt_ids`, and `STALE_CLOSED`.
- I4 and both JobLedger kinds remain covered, including stale Reverse terminal
  absorption, all-worker completion, failed jobs, and closed-record reclamation.

**Step 8 — cleanup, watchdogs, limits, facade:**

- Logical cleanup removes current Reverse/Forward state and reclaims every
  closed job record while leaving open JobLedger records reportable.
- PE watchdog expiry fails the request and releases without a destination hold;
  the retired hold-pressure limits are absent.
- Facade and MultiConnector forwarding for every new hook.
- Explicit rejection for PP>1, DP>1, PCP>1, DCP>1, and PE/DE TP mismatch via
  the new bootstrap fields; matching TP=2 follows the same semantics as TP=1.

**Step 9 — NPU E2E:** force PE preemption during compute/Forward, including the
final-layer queued race, and verify no crash, hang, reuse corruption, or output
mismatch; include matching TP>1. Exercise abort-while-waiting separately and
record the accepted immediate-free/late-DMA risk instead of claiming the
retired hold guarantee.

## 10. Implementation order and delivery

Single deliverable; the order below is a local build sequence, not a release
split. Each step lands with its §9 tests.

1. **Sender failure fixes + barrier primitive.** Exception-safe barrier
   dispatch outside `_transfer_kv_cache`; `failed_reqs` attribution fix; the
   terminal-ACK failure fact; the FIFO barrier item and worker-local event.
2. **JobLedger lifecycle.** Scheduler-side `update_connector_output`
   all-worker aggregation with `expected_worker_count` from
   `parallel_config.world_size`, duplicate/failure handling, and closed-record
   reclamation. No HoldLedger or KV block pinning remains.
3. **Attempt identity + I4 terminal gate.** `ReverseAttemptKey`, attempt-unique
   Reverse wire ids, attempt-keyed worker state, job-based Reverse completion
   reporting, the DE `reverse_send_job_id` aggregation, and removal of the
   worker-side direct req-id conversion for dual_path reverse terminals.
4. **Resume admission.** The delivered-state resume branch in
   `_decide_prefill_path_for_admission`, including the barrier-gated
   `(None, False)` defer, replacing the invalid-set convergence for delivered
   decisions.
5. **Decision-only receiver schema and registry.** Message-kind field,
   attempt-aware `_accepted_decisions` semantics, response encoding,
   at-least-once retry behavior, and stale-epoch rejection.
6. **Replay wiring.** Common fence-and-replay for PE_READ first (no new
   Decision, no DE reactivation), then DE_READ (new Reverse attempt from
   current `L_PE`, fresh tracker/latch, stable Forward binding unchanged).
7. **Retire close and Reverse hold.** Remove the PE close driver, Decode close
   protocol/registry, Reverse destination hold and budgets; preserve Decision
   epoch monotonicity, `STALE_CLOSED`, I4, and JobLedger.
8. **Cleanup, watchdogs, failure reporting, and facade/MultiConnector
   forwarding**, plus topology fail-fast validation.
9. Implement the §9 tests per step and the separate PE_READ/DE_READ NPU
   preemption and TP acceptance scenarios.
10. On landing, remove the known-limitation note from
    `.specs/dual-path-stage1-tasks/TASKS.md` §3 and link to the implemented
    result.

## 11. Tunable parameters (not correctness decisions)

- PE Reverse-completion recovery watchdog.
- DE Store/Forward/Reverse progress watchdog.

`VLLM_ASCEND_DUALPATH_CLOSE_RETRY_BACKOFF_S`,
`VLLM_ASCEND_DUALPATH_MAX_HELD_RECOVERY_BLOCKS`, and
`VLLM_ASCEND_DUALPATH_MAX_RECOVERY_RECORDS` are removed. Setting them no longer
changes runtime behavior.

Defaults require NPU workload measurement, but every correctness predicate in
this document is independent of those values.

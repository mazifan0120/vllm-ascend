# DualPath Stage-2: Preemption Safety (Pin + Stable Forward Replay + DE_READ Reverse Attempts)

Status: design — single-stage delivery (approved direction: route-specific
recovery; no 2a/2b split)
Date: 2026-08-11
Updated: 2026-08-12
Scope: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` on vLLM 0.23.0
Supersedes: the Stage-1 known limitation recorded in
`.specs/dual-path-stage1-tasks/TASKS.md` §3 ("prefill dual_path request
preempted after decision delivery is unsafe").

## 1. Problem

A PE-side dual_path request whose path decision has been **delivered** to DE is
unsafe if the vLLM scheduler later preempts it:

- `_preempt_request` (vLLM `v1/core/sched/scheduler.py:974-995`) frees the
  request's ordinary block ownership at `:983` with **no connector callback**,
  and the blocks can be reused within the same `schedule()` pass.
- The PE request must be `RUNNING` while it computes and streams Forward data
  to DE; only `RUNNING` requests are preemptible (`scheduler.py:499-501`).
- For `DE_READ`, PE admission first allocates the Reverse destination blocks and
  parks the request in `WAITING_FOR_REMOTE_KVS` until all workers report Reverse
  completion (dual_path `scheduler.py:598-603`). The Reverse-wait window is
  therefore not part of normal scheduler preemption: a normally preempted
  `DE_READ` request has already completed its current Reverse attempt.

### Failure modes confirmed by code walk-through

- **F1 — Forward read-after-free.** A queued or executing Forward transfer can
  read PE blocks after preemption returns their ordinary ownership to the pool,
  sending unrelated data to DE.
- **F2 — DE_READ re-admission parks forever.** A preempted `DE_READ` request
  needs Reverse data in its new PE allocation, but the old Reverse terminal was
  already consumed and no new attempt exists to promote the new allocation.
- **F3 — DE_READ recovery reuses an exhausted sender latch.** DE's retained
  `reverse_submitted` latch prevents a second Reverse submission unless recovery
  creates a fresh Reverse attempt.
- **F4 — stale Reverse terminal escapes into vLLM.** A delayed terminal from an
  old attempt can be attributed to the current logical request and reach
  `_update_from_kv_xfer_finished`, either resuming the wrong allocation or
  tripping its request-status assertion.
- **F5 — stale Reverse destination.** Reusing an old `ReversePlan` after PE
  re-admission writes into the old PE block table, which may now belong to
  another request.
- **F6 — uncertain control cleanup.** If a `DE_READ` Decision was accepted but
  not activated, its ACK is uncertain, or the PE request is aborted while still
  waiting for Reverse, PE cannot infer whether DE may later start or continue an
  old Reverse write.

F2-F6 require a Reverse direction and therefore apply only to `DE_READ`.
`PE_READ` has no DE-to-PE write, no PE-side remote-KV wait, and no Reverse
attempt. Both paths nevertheless share F1 and the same stable Forward-binding
replay semantics.

### Why existing request-finish mechanisms do not cover this

- `delay_free_blocks` is consulted at request finish, not on preemption.
- Generic `finished_sending` means vLLM may free the request's ordinary block
  ownership. It cannot mean "one extra DualPath pin may be released".
- MooncakeLayerwise sends from a single FIFO thread, but `save_kv_layer()` can
  enqueue every layer before the background sender finishes. With async
  scheduling, a later `schedule()` can preempt the still-`RUNNING` request while
  those sends are queued.
- Waiting-queue immunity proves normal `DE_READ` preemption happens after
  Reverse completion; it does not cover abort, uncertain Decision receipt, or
  receipt-to-activation races.

## 2. Goals / non-goals

Goals:

- Prevent memory corruption, hangs, and engine crashes under PE preemption
  (F1-F5), and make uncertain DE_READ cleanup fail closed (F6).
- Preserve the original `PathKind`. PE prefix loss is normal and never causes a
  policy re-run or a route switch.
- Keep one stable logical Forward binding on DE for both paths. Recovery only
  replaces PE-local Forward source state; `DE_READ` additionally creates a new
  Reverse attempt for the current PE allocation.
- Keep request-visible waits bounded. Timeout may fail a request, but timeout
  alone must never release memory still reachable by a transport.
- Support matching `TP>1` within the Stage-1 topology contract through
  all-participating-worker aggregation.
- Reject unsupported local and PE/DE parallel topologies before a decision is
  committed or a DualPath pin is acquired.

Non-goals (Stage-2):

- DE-side request preemption (Decode blocks are DE-local; separate analysis).
- Expanding Stage-1 data-plane support to PP, DP, PCP, DCP, TP mismatch,
  multi-group, or hybrid-layout configurations.
- Engine-core restart. A changed `decode_engine_instance_id` remains the
  incarnation boundary; verified old-process/transport teardown is the final
  safety proof when the old control endpoint cannot answer.
- Post-commit fallback or path switching during recovery.
- Cancelling an in-flight synchronous Mooncake transfer. Stage-2 drains work
  already accepted by the relevant sender and prevents new old-attempt work.

## 3. Reused engine and Connector primitives

Stage-2 reuses four existing mechanisms rather than creating another
completion framework:

1. **Block pinning.** `KVConnector.bind_gpu_block_pool(block_pool)`,
   `BlockPool.touch(blocks)`, and `BlockPool.free_blocks(blocks)`. In-tree
   precedents are AscendStore and upstream `simple_kv_offload`.
2. **Preemption delivery.** `SchedulerOutput.preempted_req_ids` is populated in
   the preempting schedule pass and is visible to `build_connector_meta`.
   Worker-side `handle_preemptions` receives the resulting connector metadata.
3. **Worker completion metadata.** `KVConnectorWorkerMetadata.aggregate()` is
   invoked while vLLM combines every executor worker's `KVConnectorOutput`.
   `simple_kv_offload` and `offloading` report an opaque `{job_id: 1}` per worker
   and let the Scheduler map the job back to semantic state.
4. **Layerwise topology mapping.** MooncakeLayerwise already maps TP/PCP/DCP
   ranks and physical ranges. Stage-2 adds direction-appropriate barriers around
   the existing per-rank work instead of duplicating that math.

The Store precedent is exact: Store completion is reported through
`kv_connector_worker_meta`, not `finished_sending`; scheduler-side
`update_connector_output` waits for all participating workers and then releases
the extra references. `request_finished` remains `(False, None)` because
connector-owned refcounts, not vLLM's request-finish delay line, own the safety
barrier.

## 4. Safety model and identities

### 4.1 Logical request, Forward state, and Reverse attempt

- **Logical/admission identity:** `DualPathRequestKey` remains exactly
  `(decode_engine_instance_id, decode_request_id)`. DE creates and registers it
  before PE has seen the request.
- **Stable Forward binding:** both `PE_READ` and `DE_READ` keep one logical DE
  `ForwardReceiveBinding`, one stable Forward wire id, the frozen DE destination
  table, and one fixed Forward range for the lifetime of the request.
- **PE-local Forward source attempt:** PE block allocation and the source table
  may change after preemption. The source plan, pins, and opaque completion jobs
  are local attempt state under the logical request; they are not part of the
  PE-to-DE decision identity.
- **DE_READ Reverse attempt:** only `DE_READ` carries a non-negative
  `reverse_attempt_id`, sourced from PE `request.num_preemptions`, plus the
  attempt-local `prefill_local_tokens = L_PE(attempt)`. Scheduler/Worker code
  composes them into `ReverseAttemptKey(request_key, reverse_attempt_id)`. The
  first delivered value need not be zero and later values need not be
  contiguous: PE preemptions that happen before a Decision leaves the process
  may consume numbers without creating remote attempts.
- **Reverse wire id:** a `DE_READ` Reverse attempt uses an opaque id derived from
  its complete attempt key. Code looks it up explicitly and never recovers
  identity by slicing a string suffix.
- **Completion job id:** a Scheduler-assigned opaque integer. Worker code does
  not interpret route or attempt. PE Scheduler maps it to a Forward source
  completion/drain or Reverse destination completion; DE Scheduler maps it to
  exceptional Reverse-attempt retirement work.

`PathDecisionRequest` and DE pending admission remain keyed only by
`DualPathRequestKey`. The serialized `PE_READ` result remains exactly
`{request_key, path}` with `reverse_plan=None`. A `DE_READ` result adds
`reverse_attempt_id` and `prefill_local_tokens`; only its `ReversePlan` and
`ReverseReceiveBinding` use the complete attempt identity. Path-specific exact
field validation rejects an attempt-bearing `PE_READ` or a `DE_READ` missing its
attempt fields.

`ForwardPlan` remains a logical-keyed value containing range and block tables.
Both paths retain the current source plan under their logical request and may
replace it only after the prior source work is proven complete or drained.

### 4.2 Required invariants

- **I1 — no reuse before proof.** A pinned block is not reusable until every
  transport that can read or write it is proven unable to touch it.
- **I2 — identity follows the changing address.** Admission and Forward receive
  state use `DualPathRequestKey`; only DE_READ Reverse plans, bindings,
  terminals, latches, and cleanup lookups use `ReverseAttemptKey`. An admission
  lookup is never replaced by attempt-key membership.
- **I3 — stable Forward completion.** A successful terminal means one complete
  valid write of the stable logical Forward range. Partial work publishes no
  terminal; a later source attempt may overwrite it. A successful old attempt
  wins over replay and marks the logical Forward `SATISFIED`.
- **I4 — current Reverse-attempt gate.** A Worker Reverse terminal is a fact.
  Only PE Scheduler may convert it into generic `finished_recving`, and only
  when its `reverse_attempt_id` equals the logical request's
  `waiting_reverse_attempt_id` and the request is still waiting for remote KV.
- **I5 — all-worker proof.** A job or generic terminal is complete only after
  every expected participating worker has reported it exactly once.
- **I6 — drain before Forward source replacement.** No new Forward source plan
  or enqueue may become visible until every PE worker rejects further old-plan
  enqueue and crosses its local sender barrier, unless normal Forward completion
  already won.
- **I7 — route-preserving recovery.** PE_READ rebuilds only its local Forward
  source attempt. DE_READ rebuilds that source attempt plus a new Reverse
  attempt from current `L_PE`. Neither path calls `PathPolicy` again.
- **I8 — exceptional retire is fail closed.** Only `SAFE_TO_RELEASE` or verified
  old-process/transport teardown proves an uncertain Reverse attempt can no
  longer write its PE destination. Timeout, `RETIRING`, `RETIRE_FAILED`, and
  `UNKNOWN_REQUEST` are not release proofs.

## 5. Design

The design separates four concerns: block pins and completion jobs; one common
stable Forward replay mechanism; DE_READ Reverse-attempt renewal; and a narrow
exceptional cleanup protocol.

### L0 — PE pin and completion-job ledger

Owner: `DualPathConnectorScheduler` on the PE role. It binds the block pool and
maintains reason-specific pins plus opaque job mappings.

Pin acquisition:

- Before a DE_READ Decision is delivered, pin the current Reverse destination
  slice covering `[L_PE, K_DE)`; a vacuous Reverse acquires no such pin.
- Before either path exposes a Forward source plan to Workers, pin the source
  blocks covering that path's stable Forward range.
- Pin by `block_pool.blocks[block_id]`. Record each reason-specific reference so
  overlapping physical sets, retries, and cleanup cannot unbalance refcounts.

Worker-to-Scheduler completion follows the upstream offloading shape:

```python
@dataclass
class DualPathWorkerMetadata(KVConnectorWorkerMetadata):
    completed_jobs: dict[int, int]  # each worker emits {job_id: 1} once

    def aggregate(self, other):
        # Add counts for identical job ids.
        ...
```

The full path is:

```text
Worker build_connector_worker_meta()
  -> ModelRunner KVConnectorOutput.kv_connector_worker_meta
  -> vLLM aggregate() across executor workers
  -> Scheduler update_connector_output()
  -> pin/job-ledger transition
```

`get_finished()` keeps the base meanings of its generic sets. DE-side Forward
receive completion still publishes generic `finished_recving` for the stable
logical binding. PE-side Forward completion/drain and Reverse pin-release facts
use completed jobs instead of `finished_sending`. A current Reverse completion
may additionally become `finished_recving` after Scheduler validation under I4.

#### Pin release predicates

| Pin class and exit | Required proof | Release owner |
|---|---|---|
| Forward source, normal | Every PE worker reports final Forward completion after its last synchronous write and terminal ACK | PE Scheduler `update_connector_output` |
| Forward source, preempt/abort | Every PE worker reports the sender barrier after rejecting further old-plan enqueue | PE Scheduler `update_connector_output` |
| Reverse destination, normal DE_READ | Every PE worker reports current-attempt Reverse receive completion | PE Scheduler `update_connector_output` |
| Reverse destination, uncertain Decision/abort | DE returns `SAFE_TO_RELEASE` for `RetireReverseAttempt` | PE Scheduler |
| Decision never left PE | No remote writer was authorized | PE Scheduler immediately |
| Old DE process/transport teardown | Verified old transport can no longer issue DMA | Connector teardown cleanup |

A Decision receipt, replacement receipt, watchdog expiry, `RETIRING`,
`RETIRE_FAILED`, or `UNKNOWN_REQUEST` is not an unpin predicate.

### L1 — common stable Forward replay

#### Detect preemption and close old enqueue

In PE `build_connector_meta`, intersect
`scheduler_output.preempted_req_ids` with delivered DualPath requests. Every hit
atomically closes the current Forward source attempt to new enqueue and
allocates a Forward-drain job.

The Layerwise sender is a single thread consuming one FIFO queue, and
`batch_transfer_sync_write` returns synchronously. A local barrier is therefore
constructible without TransferEngine cancellation:

1. add the logical request/current source attempt to a rejected-enqueue set;
2. ensure every enqueue path checks that set before `queue.put`;
3. enqueue a barrier item after the reject flag becomes visible;
4. finish the current synchronous write and all earlier queue items; and
5. report the drain job when the barrier item is consumed.

The current queue has no `task_done()/join()` or barrier item; Stage-2 must add
this explicit primitive. Observing an empty queue is racy and is not a barrier.
This ordering is required even when every layer callback for the last chunk has
already enqueued before preemption, as can happen with async scheduling.

PE may compute admission facts for the replacement while the barrier is
pending, but must not expose the new source plan or enqueue new Forward work
until every participating PE worker reports either normal completion or the
drain job.

#### Normal-completion versus replay arbitration

Both paths use the same logical Forward state:

```text
ACTIVE
  -> SATISFIED                              # all workers completed final Forward
  or
  -> DRAINING -> ACTIVE                     # partial old attempt; replay needed
  or
  -> FAILED                                 # any terminal send/control failure
```

The normal-send and drain jobs are distinct facts evaluated together:

- failure wins and terminates recovery;
- if every worker completed the final Forward and delivered its terminal,
  normal completion wins, the logical binding becomes `SATISFIED`, source pins
  are released, and a later PE re-admission does not replay;
- otherwise, after every worker crosses the drain barrier, source pins are
  released and recovery may install a new PE source attempt.

Partial old writes need no invalidation message. They produced no final terminal,
so DE remains waiting, and the new complete Forward deterministically overwrites
the same stable destination range. An old successful terminal is not stale: it
is proof that the stable Forward range is already complete.

For PE_READ, the post-drain replacement can immediately rebuild and replay its
Forward source. For DE_READ, the source attempt is rebuilt but cannot run until
the new Reverse attempt described below completes.

```mermaid
sequenceDiagram
    participant PS as PE Scheduler
    participant PW as PE Workers
    participant DW as DE Workers

    PS->>PW: expose pinned Forward source attempt
    PW->>DW: queued/synchronous Forward writes
    PS->>PS: observe preempted_req_id
    PS->>PW: reject old enqueue + install barrier job
    PW->>PW: finish accepted FIFO work
    PW-->>PS: completed_jobs[barrier_job] per worker
    alt complete Forward terminal won on every worker
        PS->>PS: Forward = SATISFIED; release old source pins
    else old attempt was partial
        PS->>PS: release old source pins after all-worker barrier
        PS->>PW: expose newly pinned source attempt
        PW->>DW: overwrite the same stable Forward range
    end
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
the same stable Decode table. DE already has, or its logical Store load provides,
`[0, K_DE)`; `K_DE` and the path do not change. When `L_PE(new) >= K_DE`, the
new Reverse attempt is vacuously complete.

For PE_READ, PE prefix loss only changes how PE produces `[L_DE,T)`. Its Decision
contains no PE block address or PE-derived boundary, so PE sends no replacement
Decision and DE does not reactivate anything.

For both paths, the stable Forward binding lets DE combine a valid completed
rank terminal with later duplicate overwrites. DE retains the binding while
completion is pending and retains its consumed-terminal tombstone through final
request cleanup so a late duplicate cannot be attributed to another request.

#### Normal DE_READ preemption

Normal scheduler preemption starts from a completed Reverse attempt:

```text
ReverseAttempt(N): DONE
  -> PE RUNNING / Forward ACTIVE
  -> preemption
  -> Forward SATISFIED
       or Forward DRAINING -> drained
          -> re-admission
          -> ReverseAttempt(N+1): ACTIVE -> DONE
          -> Forward ACTIVE -> SATISFIED
```

PE does **not** send `RetireReverseAttempt(N)` on this path. The all-worker
Reverse completion that promoted the request from `WAITING_FOR_REMOTE_KVS` is
already the release proof for attempt N, and its Reverse destination pin was
released then.

If old Forward completion wins, DE has the complete `[K_DE,T)` range and no
replacement Reverse attempt is needed. Otherwise PE creates a greater
`reverse_attempt_id`, captures current `L_PE` and the new PE block table, pins
the new Reverse destination, and delivers a new DE_READ Decision only after the
old local Forward barrier completes.

The already-preempted PE scheduler request still follows normal vLLM
re-admission/completion bookkeeping. When logical Forward is already
`SATISFIED`, the Connector authorizes no further remote exchange; any local
recomputation needed to retire that PE request is not another DualPath attempt.

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
    PW-->>PS: current Reverse completion jobs
    PS->>PS: release N destination pin; request RUNNING
    PS->>PW: compute and enqueue stable Forward range
    PS->>PS: preemption; close old Forward enqueue
    PW-->>PS: all-worker Forward completion or barrier
    alt logical Forward already SATISFIED
        PS->>PS: no new remote exchange
    else replay required
        PS->>PS: re-admit; recompute current L_PE; keep PathKind=DE_READ
        PS->>PS: pin new Reverse destination
        PS->>DC: Decision(request_key, attempt M, current L_PE, new PE blocks)
        DC->>DC: validate stable admission; install fresh Reverse plan/latch
        DC-->>PS: Decision ACK
        DC->>DW: submit ReverseAttempt(M)
        DW-->>PW: write [L_PE(M), K_DE) into new PE blocks
        PW-->>PS: M completion jobs from every worker
        PS->>PS: release M destination pin; resume request
        PS->>PW: compute remainder + replay stable Forward range
    end
```

This normal sequence never sends `RetireReverseAttempt`: N was already safe
before the request became preemptible, and M does not exist until the new
Decision is sent.

#### DE receiver rules

The first path selection invokes eligibility and `PathPolicy`; recovery never
does. DE handles results by path:

- `PE_READ`: accept and activate the first logical Decision; ACK an identical
  retry; reject a conflicting or attempt-bearing payload. No replacement
  Decision is expected.
- `DE_READ`: accept a `reverse_attempt_id` only under an existing logical
  admission and apply the following matrix.

DE_READ receiver rules for one logical request are:

- logical key not registered: reject as `UNKNOWN_REQUEST` without creating an
  attempt;
- attempt at or below `closed_through_attempt_id`: reply `STALE_CLOSED`, never
  enqueue or reopen it, and do not let PE treat the reply as a fresh activation;
- equal current attempt and identical payload: acknowledge as a duplicate;
- equal current attempt and conflicting payload: reject as a protocol error;
- greater attempt: retain it idempotently only after the previous normal
  Reverse is `DONE` and PE's Decision ordering proves the old Forward source
  barrier completed; if an exceptional retirement is still pending, stage at
  most one attempt and do not publish Worker plans until it is safe.

Normal recovery therefore needs one current Reverse attempt. A staged attempt
exists only for the bounded control-plane receipt/activation race below, not as
the steady-state recovery model.

PE serializes DE_READ Decision generations. It may deliver attempt M only after
every lower attempt either never left PE, completed normally, or returned
`SAFE_TO_RELEASE`; it must also have completed the old local Forward source
barrier. At most one Decision delivery Future is unresolved for a logical
request. Consequently a greater attempt is not permission to overtake a live
old Reverse writer. On accepting M, DE may permanently close skipped lower
numbers; a later delayed Decision at one of those numbers receives
`STALE_CLOSED`.

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
  - optional staged_reverse_attempt
  - latest_reverse_attempt_id
  - closed_through_attempt_id
  - safe_to_release_through_attempt_id
```

A Reverse attempt contains its receipt/activation/retirement state, Decision,
Reverse plan/binding, tracker/latch references, jobs, deadline, and terminal
result. Normal replacement retires the completed current attempt before
installing the next. The optional staged slot covers only serialized control
races, so DE does not require an unbounded
`dict[ReverseAttemptKey, DecodeLifecycleState]`.

`closed_through_attempt_id` remains authoritative for as long as the logical
admission exists: an expired diagnostic tombstone can never make an old
Decision acceptable again. A bounded retired cache may preserve exact terminal
reasons for retries and logging, but TTL is not part of the safety proof.

### Exceptional cleanup — `RetireReverseAttempt`

`RetireReverseAttempt` asks one narrow question:

> Can DE still activate or issue a Reverse write for this attempt into its PE
> destination block table?

PE uses it only when normal Reverse-completion proof is unavailable:

- DE accepted a Decision but its Scheduler may not yet have activated it;
- the PE request is aborted/cancelled while waiting for Reverse;
- Decision delivery ACK is uncertain; or
- a control/transport error requires cleanup while the old DE endpoint can
  still answer.

Request:

```text
RetireReverseAttempt(request_key, reverse_attempt_id)
```

Responses:

| Result | Meaning | May PE release the Reverse destination pin? |
|---|---|---|
| `RETIRING` | DE has closed the attempt to new activation/submission and is waiting for accepted work | No |
| `SAFE_TO_RELEASE(COMPLETED)` | accepted work completed on every DE worker | Yes |
| `SAFE_TO_RELEASE(NOT_STARTED)` | DE atomically closed an attempt that had never started | Yes |
| `SAFE_TO_RELEASE(ALREADY_RETIRED)` | retained state/watermark proves an earlier safe retirement | Yes |
| `RETIRE_FAILED` | DE cannot prove its writer stopped | No |
| `UNKNOWN_REQUEST` | the logical admission/incarnation cannot be established | No |

`NOT_STARTED` is stronger than a dictionary miss. Under the same registry lock,
DE advances `closed_through_attempt_id` before replying, so a delayed Decision
for that attempt is rejected permanently. A plain missing record is
`UNKNOWN_REQUEST` and is not a release proof.

The exceptional internal state is intentionally small:

```text
ACTIVE -> RETIRING -> RETIRED
```

`activation_claimed`, `reject_new_submission`, and `drain_job_installed` are
internal flags inside `RETIRING`, not externally visible protocol states.
Replacement is represented by `current_reverse_attempt_id`, not a dedicated
replacement state.

#### Single-ROUTER concurrency contract

Decision delivery and `RetireReverseAttempt` share the existing single receiver
thread and ROUTER socket. `_handle_frames` decodes a message kind and performs
only bounded in-memory work:

- it never waits on a Worker, queue, Future, condition variable, or retirement;
- Decision receipt, retirement registration/status lookup, and attempt mutation
  use the same registry lock (or one equivalently serialized control record),
  so their ordering matches ROUTER receive order;
- the first retirement request changes `ACTIVE -> RETIRING`, closes new
  activation/submission, and immediately returns `RETIRING` unless it can prove
  `NOT_STARTED`;
- Scheduler/Worker code publishes completion as a short locked transition, and
  later polls return `SAFE_TO_RELEASE`;
- every request receives its reply before the receiver reads the next request,
  preventing one retiring attempt from blocking other decisions.

Receipt-to-activation uses an atomic claim. Under the registry lock, Scheduler
may claim activation only while the attempt is `ACTIVE`. If retirement wins,
activation is suppressed. If activation already claimed, retirement sets
`reject_new_submission`, returns `RETIRING`, and the activation path must either
cancel before Worker publication or register the worker work that retirement
will wait for. The attempt cannot become `RETIRED` while a claim or worker job
remains unaccounted.

```mermaid
sequenceDiagram
    participant PC as PE Control/Scheduler
    participant DR as DE ROUTER thread
    participant DS as DE Scheduler
    participant DW as DE Workers

    PC->>DR: RetireReverseAttempt(request_key, attempt N)
    alt receipt exists but activation was not claimed
        DR->>DR: atomically close N to activation
        DR-->>PC: SAFE_TO_RELEASE(NOT_STARTED)
        PC->>PC: release N destination pin
    else activation/work may exist
        DR->>DR: ACTIVE -> RETIRING; reject new submission
        DR-->>PC: RETIRING (non-blocking)
        DS->>DW: drain already accepted Reverse work
        DW-->>DS: completed retirement job per worker
        DS->>DR: publish RETIRED
        PC->>DR: poll RetireReverseAttempt(N)
        DR-->>PC: SAFE_TO_RELEASE(COMPLETED)
        PC->>PC: release N destination pin
    end
```

### Cleanup and bounded waits

- Final logical cleanup first closes the admission to new Decisions, then
  releases the stable Forward binding only after its data-plane state is
  satisfied, failed, or safely drained.
- Normal completed Reverse attempts advance the closed and safe watermarks when
  all-worker completion is consumed; this does not wait for a later
  replacement. Exceptional attempts advance the safe watermark only on
  `SAFE_TO_RELEASE`; `RETIRE_FAILED` never does.
- Logical cleanup first closes the pending-admission registry, then removes all
  current/staged attempt state together. Before discarding the active logical
  record, DE may publish a bounded `RetiredAdmissionRecord` containing its
  closed/safe watermarks for control retries. Once that diagnostic record
  expires, queries return `UNKNOWN_REQUEST`, never a synthetic release proof;
  late Decisions still fail because the admission registry is closed.
- If a remote writer cannot be retired, its PE destination pin survives request
  failure until verified old-process/transport teardown.
- A PE watchdog bounds client-visible recovery and retirement polling. A DE
  watchdog bounds Store, Forward, Reverse, and exceptional retirement progress.
  Expiry fails the request but never synthesizes a safety proof.
- Configuration limits pinned recovery blocks/bytes and concurrent recovery
  records. New uncommitted admissions are rejected before pinning when the
  budget would be exceeded. Already committed pins are never evicted to make
  room.

## 6. Parallel topology and ordering contract

Stage-2 neither silently narrows matching `TP>1` nor silently expands Stage-1 to
unaccepted topologies.

### 6.1 Supported now

- `pipeline_parallel_size == 1`;
- `data_parallel_size == 1`;
- PE and DE have equal `tensor_parallel_size`, using the existing one-to-one
  rank mapping; matching `TP>1` is supported;
- `prefill_context_parallel_size == 1` and
  `decode_context_parallel_size == 1`;
- the remaining Stage-1 model, KV-group, layout, dtype, and block-size
  restrictions remain unchanged.

Local restrictions are checked during connector construction. PE/DE topology
compatibility is checked from bootstrap/plan metadata before Decision commit
and before acquiring DualPath pins. Every violation raises an explicit
configuration/control error; no request enters the partial protocol.

### 6.2 Reused TP barrier

Every Scheduler-created completion/drain job is sent unchanged to every
participating local executor worker. Each worker reports `{job_id: 1}` exactly
once only after all rank-local subtransfers represented by that job complete.
The Scheduler accumulates counts across steps and completes at the recorded
`expected_worker_count`.

Under the supported Stage-2 topology, `expected_worker_count == TP`. A rank-local
terminal can never release a process-wide pin. A missing or failed rank leaves
the job incomplete and takes the watchdog/fail-closed path.

### 6.3 PCP/DCP reuse boundary

MooncakeLayerwise already contains PCP/DCP token/block mapping, and vLLM defines
executor `world_size = TP * PP * PCP`; DCP reuses TP workers and does not add
executor workers. A future topology-expansion stage can reuse the protocol and
job aggregation:

- PCP ranks naturally add participating Workers;
- DCP contributes rank-local subtransfers that finish before that Worker reports
  its one job completion; and
- the barrier counts participating Workers, never raw direction terminals.

That reuse is not sufficient evidence to enable PCP/DCP in Stage-2. Stage-1
excludes them and the Decode Layerwise scheduler currently rejects PCP. Stage-2
therefore explicitly rejects PCP/DCP rather than accepting undefined attempt
ordering. Removing the rejection requires a separate topology acceptance
matrix, not a change to the route-specific recovery model.

## 7. Failure-mode to mechanism matrix

| Failure | Pin / barrier | Identity / plan handling | Completion / publication |
|---|---|---|---|
| F1 Forward read-after-free | Forward source pin plus PE sender barrier | replace only PE-local source attempt | all-PE-worker normal/drain jobs |
| F2 DE_READ re-admission parks forever | pin new Reverse destination | create new `reverse_attempt_id` and plan | only current attempt resumes PE |
| F3 exhausted Reverse latch | no old-plan reuse | fresh per-attempt tracker and latch | old DONE advances watermark |
| F4 stale Reverse terminal | old pin already released only on proof | terminal carries attempt identity | PE current-attempt gate absorbs stale terminal |
| F5 stale Reverse destination | new destination pinned before delivery | new plan uses current PE block table | old plan never resubmitted |
| F6 uncertain control cleanup | retain destination pin | `RetireReverseAttempt` closes activation/submission | only `SAFE_TO_RELEASE` or teardown unpins |

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
preserve the concrete `DualPathWorkerMetadata` type.

## 9. Test strategy for implementation

No test execution is part of this design-only update. Implementation coverage
must include:

- Path-specific schemas: PE_READ remains `{request_key, path}` with
  `reverse_plan=None`; DE_READ serializes `reverse_attempt_id`, current `L_PE`,
  and an attempt-unique Reverse wire id. Forward binding/wire identity remains
  logical and stable on both paths. DE admission can register before any PE
  attempt exists.
- Recovery preserves `PathKind`; changed `L_PE` never calls policy again. Cover
  `L_PE(new) <`, `==`, and `>` both `L_PE(old)` and the relevant split boundary.
- Exact ranges: DE_READ Reverse expands, shrinks, or is vacuous while Forward
  remains `[K_DE,T)`; PE_READ Forward remains `[L_DE,T)`.
- Common Forward replay: no DE binding replacement, source plan changes only
  after all-worker drain, coincidentally equal block ids still replay unless
  `SATISFIED`, and normal completion wins over preemption.
- Async sender ordering: a final layer already queued, in-progress synchronous
  write, concurrent late enqueue, reject flag, barrier, terminal-versus-barrier
  race, duplicate job, and failure-wins semantics.
- Normal DE_READ preemption starts from Reverse `DONE`, does not call
  `RetireReverseAttempt`, releases the old Reverse destination pin before
  RUNNING, and creates N+1 only when Forward was not already satisfied.
- Reverse receiver matrix: unknown logical request, greater attempt, identical
  duplicate, conflicting duplicate, skipped attempt ids, and lower/closed
  attempt. Cover `STALE_CLOSED` separately from a successful activation and
  verify that closed admission state prevents reopen after diagnostic tombstone
  expiry.
- Current-attempt gate: an old Reverse terminal never reaches generic
  `finished_recving`; the current terminal resumes only a request still in
  `WAITING_FOR_REMOTE_KVS`.
- Exceptional retirement race matrix: retire-before-Decision produces
  `SAFE_TO_RELEASE(NOT_STARTED)` and permanently closes the attempt;
  Decision-before-retire returns `RETIRING`; receipt-before-activation cannot
  produce a vacuous release proof; activation winner remains visible until all
  worker work completes. ROUTER handling never blocks.
- Pin accounting for every acquisition and exit. Receipt, timeout,
  `RETIRING`, `RETIRE_FAILED`, and `UNKNOWN_REQUEST` deliberately retain the
  Reverse destination reference.
- `DualPathWorkerMetadata.aggregate()` across steps and matching `TP>1`; no job
  completion before every expected worker reports exactly once.
- Logical cleanup removes current/staged Reverse attempts and Forward state as a
  unit; bounded retired-admission records do not leak or weaken rejection, and
  their expiry degrades to fail-closed `UNKNOWN_REQUEST`.
- Explicit rejection for PP>1, DP>1, PCP>1, DCP>1, and PE/DE TP mismatch;
  matching TP=2 follows the same semantics as TP=1.
- Facade and MultiConnector forwarding for every new hook.
- NPU E2E: force PE preemption during compute/Forward, including the final-layer
  queued race, and verify no crash, hang, reuse corruption, or output mismatch;
  include matching TP>1.

## 10. Implementation order and delivery

Single deliverable; the order below is a local build sequence, not a release
split:

1. Keep `DualPathRequestKey` as the logical admission/Forward identity. Add
   `ReverseAttemptKey`, current `L_PE`, and attempt-unique Reverse wire ids only
   for DE_READ.
2. Add topology fail-fast validation and all-participating-worker job
   aggregation.
3. Add PE block-pool binding, reason-specific pin ledger, job ledger, and
   Scheduler-side `update_connector_output`.
4. Add the common Forward rejected-enqueue state, sender barrier, stable logical
   binding, `SATISFIED` arbitration, and drain-gated source-plan replacement.
5. Add PE_READ local replay without another Decision or DE reactivation.
6. Make DE_READ Forward use the same stable logical binding; keep only Reverse
   plans, bindings, wire terminals, and latches attempt-keyed.
7. Add fixed-DE_READ re-admission, new Reverse-attempt construction, current
   `L_PE` range rebuilding, and fresh per-attempt tracker/latch installation.
8. Add the PE current-Reverse-attempt terminal gate and normal Reverse pin
   completion.
9. Add non-blocking idempotent exceptional `RetireReverseAttempt`, atomic
   receipt/activation closing, DE all-worker retirement, and safe/closed
   watermarks.
10. Add exhaustive logical cleanup, watchdogs, pin-pressure limits, failure
    reporting, and facade/MultiConnector forwarding.
11. Implement §9 tests and separate PE_READ/DE_READ NPU preemption and TP
    acceptance scenarios.
12. On landing, remove the known-limitation note from
    `.specs/dual-path-stage1-tasks/TASKS.md` §3 and link to the implemented
    result.

## 11. Tunable parameters (not correctness decisions)

- PE Forward recovery and exceptional Reverse-attempt retirement watchdogs.
- Maximum pinned recovery blocks/bytes and concurrent recovery records.
- `RetireReverseAttempt` poll backoff and diagnostic tombstone retention.

Defaults require NPU workload measurement, but every correctness predicate in
this document is independent of those values.

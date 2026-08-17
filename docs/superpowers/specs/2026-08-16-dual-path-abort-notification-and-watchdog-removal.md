# DualPath ABORT Notification and Watchdog Removal

Status: accepted and implemented

Date: 2026-08-16

Scope: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/`

## Decision

DualPath uses explicit protocol outcomes and direct runtime failure facts. It
does not infer request failure, transport quiescence, or block safety from
elapsed time.

The following environment variables are removed:

- `VLLM_ASCEND_DUALPATH_DECISION_TIMEOUT`
- `VLLM_ASCEND_DUALPATH_RECOVERY_WATCHDOG_S`
- `VLLM_ASCEND_DUALPATH_DE_PROGRESS_WATCHDOG_S`

The Decode decision deadline, PE Reverse-completion recovery watchdog, DE
Store/Forward/Reverse progress watchdog, their state and sweep loops, and the
`DECISION_TIMEOUT` and `RECOVERY_TIMEOUT` control-failure reasons are removed
with them.

This decision supersedes the 2026-08-14 Stage-2 choice to retain the PE
recovery and DE progress watchdogs after retiring `CloseReverseAttempt` and the
Reverse destination hold.

## Why timers are not failure or safety evidence

A deadline can establish only that an interval elapsed. It cannot establish
that a peer died, that a Decision or ABORT will never arrive, that a Worker
stopped making progress, that all ranks observed a terminal, or that a DMA
writer is quiescent. Failing and freeing a request from that observation can
turn a liveness incident into a correctness incident, including a late write
into reallocated blocks.

DualPath therefore fails closed only from an explicit remote outcome or a
direct local/Worker failure fact. The tradeoff is deliberate: missing facts
can leave requests waiting indefinitely.

## Request-terminal ABORT protocol

### Message and delivery

`PathAbortNotice` carries the exact `DualPathRequestKey` and one reason:

- `DECISION_FAILED`
- `DELIVERY_EXHAUSTED`
- `ACTIVATION_FAILED`
- `REQUEST_ABORTED`

ABORT uses the existing PE-to-DE control endpoint and the same bounded,
at-least-once three-attempt delivery pool as Decision. A delivery Future is
observed asynchronously. Exhausting ABORT delivery is logged; it does not
create a second failure channel or a timer fallback.

The Decode receiver treats ABORT as an ensure-absent terminal notification. For
the exact current engine incarnation, a key retained by the pending or
accepted-decision registry is removed, ACKed, and enqueued exactly once. A
duplicate or never-registered same-incarnation ABORT is also ACKed, but it is
not enqueued and creates no tombstone because the terminal postcondition
already holds. Wrong-incarnation ABORT remains `UNKNOWN_REQUEST`, malformed
ABORT remains `PROTOCOL_ERROR`, and an accepted ABORT makes a later Decision
for that key unknown. Unknown Decision remains `UNKNOWN_REQUEST`.

This idempotency rule belongs only to the receiver. The PE sender remains
strict: an `UNKNOWN_REQUEST` reply is a terminal rejection, not ABORT success.

### Admission identity

`DualPathRequestKey` is the single identity for one Decode admission:

```text
(decode_engine_instance_id, decode_request_id, admission_id)
```

The Decode `PathDecisionCoordinator` allocates `admission_id` from a
monotonically increasing integer counter. The counter is scoped to one Decode
engine incarnation, so the incarnation plus the integer is globally sufficient
for this protocol; no UUID, TTL, or tombstone is required. Decode mints the key
exactly once when it registers the pending decision. Allocation retry and
preemption resume reuse the same `Request` and its already registered key.

Every Decision, ABORT, `ReversePlan`, `ReverseAttemptKey`,
`REVERSE_COMPLETION` job, and `REVERSE_SEND` job already embeds the request key,
so the admission component propagates through those paths without a second job
identity or close epoch. `reverse_wire_id` also encodes `admission_id`; otherwise
two attempt-zero Reverse transfers for reused client request IDs would still
collide at the data-plane boundary.

`DualPathRequestKey.from_dict` continues to require the exact key set. There is
no mixed-schema compatibility layer: PE, DE, and their external ops log parser
must be upgraded together. Existing log field order remains stable and the
admission ID is appended as a new field so the full identity is observable.

The coordinator's attempt watermark remains admission-local and unchanged. It
orders preemption attempts within one admission; it is not a substitute for the
admission identity.

### PE trigger semantics

PE sends a request-terminal ABORT when any of these terminal facts occurs:

| Trigger | ABORT reason |
|---|---|
| Initial or refreshed path decision cannot be produced from valid request facts | `DECISION_FAILED` |
| Decision delivery exhausts its retries | `DELIVERY_EXHAUSTED` |
| PE cannot activate or install the selected route | `ACTIVATION_FAILED` |
| The Prefill request finishes as client-aborted before normal completion | `REQUEST_ABORTED` |

If PE cannot recover the request key or Decode control endpoint from corrupt
bootstrap metadata, it cannot address an ABORT. That case is one of the
accepted indefinite-wait risks below.

### Admission-scoped Decision delivery failure

Prefill delivery Futures, immutable delivery records, deferred replacements,
and invalid markers are keyed by `DualPathRequestKey`, not the local request ID
or Python `Request` object identity. An old in-flight delivery can therefore
coexist with a later admission that reuses the local ID, while allocation retry
and preemption within one admission continue to share the same key and frozen
path. The earlier `Request is` owner map, owner field, and request-ID reuse
crash fences are removed rather than layered under the new identity.

The request-ID-keyed Prefill maps continue to represent only the one currently
active local admission. Installing a second live admission under the same local
ID before the first is released remains an upstream duplicate-injection bug and
may retain one explicit tripwire. Once release removes the active admission,
its unresolved delivery state remains independently addressable by its full
request key.

Every terminally failed Decision delivery sends `DELIVERY_EXHAUSTED` ABORT for
the delivery record's exact key and endpoint. Local Prefill failure staging is
stricter: it occurs only while `_prefill_request_keys[request_id]` equals the
record's request key. A failure reconciled after release or against another
admission sends ABORT but stages no local failure and invalidates no blocks in
the reused request lifecycle.

For a live `DE_READ` admission, reconciliation consumes the terminal Future
exactly once and retains its immutable delivery record until the staged control
failure is copied into outgoing connector metadata. Request cleanup may instead
drop the staged failure and retire the record. This short-lived record is an
admission/publication fence: it has no expiry, timer, receiver state, or
late-ABORT semantics and is not an ABORT tombstone.

If I7 has already installed a replacement Reverse attempt when an older
Decision Future fails, local invalidation follows the current Reverse plan for
the same request key. Its destination slice is recomputed from
`floor(token_start / block_size)` through `ceil(token_end / block_size)`. The
older delivery record's block snapshot is only a fallback when no matching
current plan exists; a released or different-admission delivery never invokes
local invalidation.

### Decode receive semantics

Each Decode metadata build preserves this order:

1. consume received Decisions and activate their plans;
2. consume received ABORT notices;
3. publish direct failures already staged by prior Worker output;
4. compose Decode Store metadata.

This order is intentional. A Decision and ABORT drained together first create
the attempt-scoped route and job facts, then terminate the logical request
without rewriting or closing those job-ledger facts.

For a `PENDING` or `COMMITTED` request, ABORT changes the Decode decision state
to `ACTIVATION_FAILED`, unregisters the request key, and stages one
`PEER_ABORT` control failure covering the external destination suffix. Worker
handling starts no Store or P2P operation for that control failure; it reports
`finished_recving` plus the invalid block IDs. The vLLM core then transitions
the request to `FINISHED_ERROR` and returns its delayed blocks.

Receiver-unknown and duplicate same-incarnation ABORT never reaches the
Scheduler. Already-terminal Scheduler state ignores a notice already queued
for a formerly known key. Failure to construct valid control-failure metadata
keeps the terminal state and logs the local error; it does not invent a timeout
outcome.

Because queues may outlive a released admission, Decode consumers compare the
complete key before applying a message to request-ID-keyed active state. A
queued ABORT whose key differs from the current state's key is a no-op. A queued
Decision with that mismatch is also discarded before validation, rather than
falling through activation failure and terminating the newer admission. A
genuine malformed Decision for the current key remains fail-closed.

## Direct failure staging

Worker-reported failed Reverse-completion and Reverse-send jobs with an exact
current admission key are direct failure facts. Scheduler `update_connector_output`
records them in the role-local staging buffer:

- Prefill stages `REVERSE_JOB_FAILED` in `_prefill_control_failures` and marks
  the admission key invalid.
- Decode stages `REVERSE_JOB_FAILED` in `_decode_control_failures` and marks
  the decision state `ACTIVATION_FAILED`.

The next `build_connector_meta` publishes those records after received
Decisions and ABORT notices. Staging avoids dropping failures produced after
the current step's connector metadata was built. It does not alter attempt
identity, epoch monotonicity, or JobLedger close/discard ownership.

Request cleanup can retain an open `REVERSE_COMPLETION` ledger record while
removing its exact-attempt owner. A later failure closes and discards only that
newly orphaned record; it stages no request failure and produces no generic
`finished_recving` side effect. Live-current attempt facts remain untouched.

Local activation errors continue to use `ACTIVATION_FAILED`. No staging buffer
or explicit failure path is replaced by a timer.

## Accepted indefinite-wait risks

Removal of all watchdog fallbacks explicitly accepts these liveness outcomes:

| Missing fact | Accepted outcome |
|---|---|
| PE process death before Decision or ABORT delivery | Decode may remain `PENDING` in `WAITING_FOR_REMOTE_KVS` indefinitely. |
| ABORT delivery exhaustion | Decode may remain pending or committed indefinitely after PE has failed locally. |
| Corrupt decision/bootstrap metadata that prevents PE from identifying the exact request key or endpoint | PE cannot address ABORT; Decode may wait indefinitely. |
| DE process death during a `DE_READ` Reverse attempt | PE may remain parked behind the current-attempt I4 gate indefinitely. |
| Hung DE Worker sender that emits neither completion nor failure | The open Reverse-send job remains open; a finished Decode request may retain delayed-free engine progress indefinitely, and PE may continue waiting for Reverse completion. |

These are operational liveness risks, not implicit authorization to release
blocks, close jobs, advance epochs, or synthesize completion. Recovery requires
external process supervision, cancellation that can deliver ABORT, or a future
protocol carrying direct failure/quiescence evidence.

## Known remaining Forward wire-identity limitation

The inherited Forward data path still derives
`ForwardReceiveBinding.wire_request_id` from the external client request ID. It
does not carry the Decode incarnation or `admission_id`. This revision does not
change the parent Mooncake sender contract. Consequently, reusing a client
request ID while an older admission still has an in-flight Forward chunk or a
Worker terminal tombstone remains unsafe: an old chunk can target the new
binding, or an old tombstone can suppress its installation.

Admission identity therefore closes the control-channel and Reverse-job
collisions in this revision, but it does not claim end-to-end same-ID reuse
safety across the inherited Forward wire path. Until that parent interface can
accept a caller-supplied wire ID, operations must avoid such reuse during the
Forward in-flight/tombstone window. Clearing pending Worker terminal state at
binding installation would only mask part of the race and is not adopted.

## Preserved correctness contracts

This change does not alter:

- `reverse_attempt_id` ordering within one admission;
- accepted-decision epoch monotonicity and stale-attempt rejection;
- I4 current-attempt Reverse completion gating;
- `JobLedger` all-worker aggregation, failure staging, and close/discard rules;
- Decode delayed free while any exact-request `REVERSE_SEND` job remains open;
- Prefill immediate-free semantics for normal finish, preemption, and ABORT;
- request-scoped invalid-block calculation, including
  `_recovery_invalid_block_ids` for direct Prefill failure reporting;
- the Decode ordering Decision → ABORT → staged failure; or
- Store, Forward, and Reverse plan ownership.

The removed timer state must not be replaced with source-spelling assertions or
another local heuristic. Behavior is covered through Decision, ABORT,
activation-failure, Worker-failure, block-invalidation, request-terminal, and
job-lifecycle tests. Static scans confirm that production contains none of the
removed environment names, enum/reason values, deadline fields, sweep methods,
or watchdog state.

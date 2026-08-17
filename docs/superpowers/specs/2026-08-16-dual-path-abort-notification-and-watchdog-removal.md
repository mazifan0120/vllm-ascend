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

Each submitted Decision Future is owned by the stable Prefill `Request` object
that produced it. Ownership comparison uses object identity, not only the local
request ID, `DualPathRequestKey`, or a newly parsed `DualPathDecisionMetadata`
instance. The exact same `Request` may legitimately reparse its envelope during
an allocation retry or preemption resume and continues with the frozen path. A
different `Request` reusing the local ID is rejected while either the active
owner or an unresolved delivery record still belongs to the earlier admission.
A retained pre-submit invalid marker is installed atomically with the `Request`
that encountered the failure. A successful Decision that has not yet been
submitted installs no owner, so changed admission facts may still discard and
re-decide that uncommitted Decision from a distinct replacement `Request`.

Every terminally failed Decision delivery sends `DELIVERY_EXHAUSTED` ABORT for
the delivery record's exact key and endpoint. Local Prefill failure staging is
stricter: it occurs only while that record's `Request` remains the active
admission owner. A failure reconciled after release or against another admission
sends ABORT but stages no local failure and invalidates no blocks in the reused
request lifecycle.

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
current plan exists; a released or different-owner delivery never invokes local
invalidation.

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

## Direct failure staging

Worker-reported failed Reverse-completion and Reverse-send jobs with an exact
current owner are direct failure facts. Scheduler `update_connector_output`
records them in the role-local staging buffer:

- Prefill stages `REVERSE_JOB_FAILED` in `_prefill_control_failures` and marks
  the request invalid.
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

## Preserved correctness contracts

This change does not alter:

- `reverse_attempt_id` and `ReverseAttemptKey` identity;
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

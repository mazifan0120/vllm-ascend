# DualPath Admission Activation Ceiling Design

**Status:** Proposed, awaiting written-spec review before implementation.

**Scope:** Close the remaining Prefill `REVERSE_RECEIVE` availability hole when Decode terminalizes an admission after Prefill has already refreshed to a newer Reverse attempt.

## Problem

The existing ABORT protocol can carry an exact `ReverseTerminalNotice` for one `(request_key, reverse_attempt_id)`. That proof is deliberately attempt-scoped: it must never fall back to the current attempt for the same request id.

This leaves a real lifecycle window:

1. Decode publishes Reverse attempt 0.
2. Prefill refreshes the same exact admission to attempt 1 and opens a new `REVERSE_RECEIVE` completion.
3. Decode proves attempt 0 terminal, transitions the admission to `ACTIVATION_FAILED`, and sends an ABORT with exact proof for attempt 0.
4. Prefill may terminalize attempt 0, but exact identity correctly prevents that proof from closing attempt 1.
5. Decode will never activate attempt 1 after the admission becomes terminal, so attempt 1 has no future wire terminal and remains pinned.

The missing fact is not another exact-attempt terminal. It is an admission-level upper bound proving which Reverse attempts could ever have reached Decode workers before the admission became terminal.

## Goals

- Preserve exact-attempt safety for Reverse work that may have started.
- Close every newer Prefill attempt that Decode proves was never published to workers.
- Fail the current exact Prefill admission even when the exact proof names a stale attempt.
- Preserve the existing all-worker barrier for every dispatched Prefill binding.
- Keep admission, attempt, completion, and wire identities exact; never fall back to request id.
- Reuse the existing scheduler-to-worker terminal and completion close paths.

## Non-goals

- No new environment variables, connector extra-config, watchdog, receive timeout, or block-release channel.
- No inference of terminal safety from a Decision delivery timeout, ACK loss, cancellation, generic error, or typed rejection.
- No scheduler-side close of a dispatched completion.
- No general bidirectional per-attempt retirement protocol in this change.

## Protocol contract

Add a frozen admission-level proof:

```python
@dataclass(frozen=True)
class ReverseAdmissionTerminalNotice:
    may_have_started_through_attempt_id: int | None


@dataclass(frozen=True)
class PathAbortNotice:
    request_key: DualPathRequestKey
    reason: PathAbortReason
    reverse_terminal: ReverseTerminalNotice | None = None
    reverse_admission_terminal: ReverseAdmissionTerminalNotice | None = None
```

The three wire states are intentionally distinct:

| Wire state | Meaning |
|---|---|
| `reverse_admission_terminal` absent | No admission-level upper-bound proof is available. This includes legacy and proofless ABORTs. |
| Object present with `may_have_started_through_attempt_id: null` | The admission is terminal and no Reverse attempt could have reached any Decode worker. |
| Object present with integer `N` | The admission is terminal; attempts `<= N` may have reached a Decode worker, while attempts `> N` provably never did. |

The integer is a conservative publication watermark, not a statement that every attempt `<= N` actually started. It must be a non-negative integer and must reject booleans.

`reverse_terminal` and `reverse_admission_terminal` are orthogonal and may coexist:

- exact proof authorizes terminalization of exactly one attempt, including an attempt `<= N`;
- the admission ceiling authorizes terminalization of every open attempt `> N`;
- neither proof authorizes terminalization of another attempt `<= N`.

### Strict wire shapes

`PathAbortNotice.from_dict()` accepts exactly four top-level shapes:

1. legacy/proofless: `request_key`, `reason`;
2. exact-only: the legacy fields plus `reverse_terminal`;
3. ceiling-only: the legacy fields plus `reverse_admission_terminal`;
4. combined: all four fields.

An optional field is omitted when unavailable. A present `reverse_terminal` or `reverse_admission_terminal` must contain its exact nested object shape; a top-level JSON `null`, missing nested field, extra nested field, negative value, boolean, or invalid enum is rejected. This preserves the semantic difference between “no ceiling evidence” and “explicitly no attempt could have started.”

## Decode producer and linearization

Decode owns an exact-admission Reverse publication watermark. It starts as “no attempt published.” Attempt ids are non-negative and refreshes for one exact admission remain strictly increasing; the publication watermark is therefore monotonic.

For a DE_READ activation or refresh, Decode advances the watermark to the candidate attempt immediately before making that attempt's Reverse plan visible to worker metadata. The ordering must be:

1. validate the exact admission and monotonic attempt;
2. establish all scheduler state needed to own the attempt;
3. immediately before publication, re-check that the exact admission is still activatable and has no frozen terminal evidence;
4. advance the exact-admission publication watermark;
5. publish the Reverse plan to workers.

Advancing before visibility is safe: an exception after step 4 may over-approximate an attempt as “may have started,” but it cannot underestimate work that a worker may own. An over-approximation may retain a completion until exact terminal proof arrives; an under-approximation could release blocks while a worker is still writing and is forbidden.

Publication and terminal freeze share one serialization domain. In the current scheduler this is the scheduler thread: the final gate, watermark advance, metadata publication, terminal state transition, registry unregister, and evidence freeze must not interleave across threads or callbacks. If any of those mutations moves off the scheduler thread in the future, they require one common lock or generation/CAS discipline with the same ordering. Initial validation alone is not sufficient: a refresh that validated before failure but reaches the final gate after freeze must be rejected without advancing or publishing.

On the first terminal transition of the admission—failure or request-state release—Decode freezes a `ReverseAdmissionTerminalNotice` only after it has:

1. moved the exact admission to `ACTIVATION_FAILED` or otherwise made it terminal;
2. unregistered the exact admission from Decision activation;
3. made future activation/refresh impossible.

The frozen proof is then attached to every ABORT for that exact admission. It never increases, decreases, or changes from `null` to an integer after first publication.

Examples:

- No Reverse plan was ever made visible: ceiling is explicit `null`.
- Attempt 0 was visible and attempt 1 failed before the watermark publication point: ceiling is `0`; an exact proof for attempt 1 may accompany it if Decode also proves attempt 1 never submitted.
- Attempt 1 reached the watermark publication point: ceiling is at least `1`, even if later code failed before every worker observed the plan. Exact proof is required to close attempt 1 until its terminal safety is known.
- A failed completion for attempt `M` closes after the latest published attempt is `N`: ABORT may carry exact proof for `M` and the same frozen admission ceiling `N`.

### Late completion reports

The ceiling must survive scheduler request-state release. Replace or extend the endpoint-only late ABORT context with a frozen exact-admission context containing:

- `request_key`;
- Prefill control endpoint;
- `ReverseAdmissionTerminalNotice`.

When request state is released while `REVERSE_SEND` completions remain open, no further attempt can be activated, so the current publication watermark is frozen into this late context. A later failed completion sends its exact terminal proof together with that same frozen ceiling. The context retires only after the last open `REVERSE_SEND` completion for the exact admission closes. Same request id with a newer admission must not reuse it.

## Prefill consumer

ABORT processing has two independent effects: admission invalidation and completion terminalization.

### Admission invalidation

Every accepted ABORT first adds its exact key to `_prefill_invalid_request_keys`.

If that key is still the current exact admission and the entry snapshot of `current_waiting_attempt.request_key` equals the ABORT key, Prefill emits `control_failures` and `invalid_block_ids` for the current plan. This is admission-wide and does not require the exact `reverse_terminal` to name the current attempt.

Consequently, an ABORT with exact proof for stale attempt 0 still fails the current attempt 1 plan for the same exact admission. It must not read or modify the plan of a same-ID replacement admission.

The invalid key also prevents a later delivery reconciliation or refresh from installing a new attempt for the terminal admission.

### Completion terminalization authority

For every open `REVERSE_RECEIVE` completion belonging to the exact ABORT key, Prefill derives authority independently:

```text
exact_authority = exact proof names this attempt
ceiling_authority = explicit ceiling exists and (
    ceiling is null or attempt_id > ceiling
)
may_terminalize = exact_authority or ceiling_authority
```

If `may_terminalize` is false, the completion remains open and unmodified. In particular, an attempt `<= N` without exact proof remains open even if Prefill has not locally dispatched its binding. Local `dispatched=False` is a close mechanism after remote safety is established; it is not remote safety evidence.

If `may_terminalize` is true:

- undispatched completion: scheduler calls the existing exact `force_fail_completion()` path and runs the existing close action;
- dispatched completion: scheduler latches failure, stages one exact `ReverseReceiveFailureTerminal`, and waits for every real Prefill worker to report through the existing barrier;
- duplicate or overlapping exact/ceiling authority is idempotent and must not stage a second terminal;
- late DONE/FAILED wire terminals after worker consumption remain idempotent.

No request-id lookup may substitute a current admission or attempt. The terminal helper always receives a concrete `ReverseAttemptKey` and validates the completion's exact key.

### Safety matrix

| Attempt relative to ceiling | Exact proof | Local dispatch | Result |
|---|---:|---:|---|
| `> N`, or any attempt when ceiling is explicit `null` | no | no | Scheduler close is authorized. |
| `> N`, or any attempt when ceiling is explicit `null` | no | yes | Stage exact worker terminal; real all-worker barrier closes. |
| `<= N` | yes | no | Scheduler close is authorized for that exact attempt. |
| `<= N` | yes | yes | Stage exact worker terminal; real all-worker barrier closes. |
| `<= N` | no | either | Keep open; await exact/wire terminal. |
| no ceiling evidence | no | either | Keep open; admission invalidation only. |

## Evidence consistency and delivery

- A proofless or exact-only ABORT may be followed by the first ABORT carrying a frozen ceiling.
- Prefill keeps an observed-ceiling ledger keyed by exact `DualPathRequestKey`. The first explicit ceiling, including explicit `null`, is stored. The ledger survives Prefill request-state release and all request-id keyed cleanup; it retires only with scheduler-driven abort-key unregister, after the exact admission is no longer active and its last `REVERSE_RECEIVE` completion is closed. Same request id with a replacement admission has a different ledger entry.
- Once an explicit ceiling has been observed for an exact admission, every later explicit ceiling must be identical. A conflicting ceiling is a protocol error: retain the first ledger value, keep the admission invalid, and ignore the conflicting ceiling for all ceiling-derived terminalization.
- A valid exact `reverse_terminal` co-carried with a conflicting ceiling remains an independent authority and may terminalize only the exact attempt it names. The conflicting ceiling must neither suppress that exact proof nor authorize any other attempt.
- Different exact attempt proofs may accompany the same frozen ceiling and remain distinct notices.
- Channel retry deduplication remains based on the full `PathAbortNotice`; scheduler-side terminal deduplication remains based on exact `ReverseAttemptKey` until completion close.
- Decision delivery outcomes remain ambiguous. The admission ceiling is created only by Decode after the admission is locally terminal and activation is disabled; it is never inferred from a sender retry result.

## Required regressions

1. Attempt 0 exact proof plus ceiling 0 while Prefill's current attempt is 1: admission-wide invalid blocks are emitted; attempt 0 is terminalized by exact proof; attempt 1 is terminalized by the ceiling.
2. The dispatched branch uses two real Prefill workers at TP=2: first report keeps completion open, second report closes and injects `finished_recving`.
3. An attempt `<= N` without exact proof remains open for both dispatched and undispatched local states.
4. An attempt `> N` closes when undispatched and uses the worker terminal/barrier when dispatched.
5. Explicit `null` ceiling covers every open attempt; absent ceiling covers none.
6. Stale exact proof without a ceiling does not close the current attempt, but still emits invalid blocks for the current plan of the same exact admission.
7. Same request id with an old admission cannot invalidate or close the replacement admission; no request-id fallback is permitted.
8. After Decode freezes the ceiling, no later activation or refresh can publish another attempt, including a refresh that passed initial validation before freeze but reaches the final pre-publication gate afterward.
9. Request-state release with open sends retains endpoint plus frozen ceiling; a late failed completion emits the same ceiling and its own exact proof.
10. Wire round trips cover all four top-level shapes and reject absent/null ambiguity, malformed nested payloads, booleans, negatives, and unknown fields.
11. Proofless→ceiling, exact-only→combined, multiple exact proofs with one ceiling, duplicate retry, and conflicting-ceiling behavior are explicit channel/scheduler tests. The conflict case crosses Prefill request-state release: the first ledger value survives until abort-key unregister, the late conflicting ceiling closes nothing by ceiling authority, and an independently valid exact proof in that notice still closes only its named attempt.
12. Admission invalidation remains effective if terminalization closes a stale completion before current-attempt control failure metadata is built; current-attempt state is snapshotted first.

## Alternatives considered

1. **Admission activation ceiling — selected.** It adds one monotonic, immutable safety fact and reuses the existing worker barrier. It closes provably unstarted newer attempts without weakening exact-attempt isolation.
2. **Bidirectional per-attempt retire receipts.** This is more expressive but adds protocol states, acknowledgements, retry ownership, and another lifecycle to reconcile. It is unnecessary for this bounded hole.
3. **Keep the newer attempt pinned fail-closed.** This is memory-safe but preserves the production availability leak after the admission is already known dead.

## Acceptance criteria

- The attempt-0/attempt-1 window closes without request-id fallback or scheduler-side close of dispatched work.
- Admission invalidation and completion terminalization are tested as separate effects.
- The Decode ceiling is conservative, frozen, exact-admission scoped, and retained for late ABORTs.
- The Prefill observed-ceiling ledger is exact-admission scoped, survives request-state release, rejects late conflicts, and retires with abort-key unregister.
- No new runtime configuration or release mechanism is introduced.
- Focused CPU unit tests pass; NPU E2E remains a separate hardware validation item.

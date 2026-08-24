# DualPath Admission Activation Ceiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Prefill Reverse attempts that Decode proves were never published after an admission became terminal, while preserving exact-attempt proof, exact-admission isolation, and the all-worker release barrier.

**Architecture:** Extend `PathAbortNotice` with a strict, frozen admission publication ceiling. Decode advances an exact-admission watermark immediately before Reverse plan publication and freezes it only after activation is disabled; Prefill separates admission-wide invalidation from per-attempt terminal authority, retains the first observed ceiling through abort-key lifetime, and routes every authorized dispatched attempt through the existing worker terminal barrier.

**Tech Stack:** Python 3.12, dataclasses, msgspec strict payloads, pytest, ZMQ, existing DualPath completion tracker and scheduler/worker metadata.

## Global Constraints

- No new environment variables, connector extra-config, watchdog, receive timeout, or block-release channel.
- `reverse_terminal` authorizes only its exact `(DualPathRequestKey, reverse_attempt_id)`; never fall back to request id or current attempt.
- An absent `reverse_admission_terminal` means no ceiling evidence. A present nested object with `may_have_started_through_attempt_id: null` means no attempt may have started. Integer `N` means attempts `<= N` may have started and attempts `> N` did not.
- Decode publication final gate, watermark advance, Reverse metadata append, terminal state transition, registry unregister, and evidence freeze share the scheduler serialization domain.
- A locally undispatched completion may close only after exact or ceiling authority establishes remote safety. Every dispatched completion closes through real worker reports and the existing all-worker barrier.
- Every accepted ABORT invalidates its exact admission. Current-plan invalid blocks depend on current exact admission ownership, not on the exact proof naming the current attempt.
- Prefill retains the first explicit ceiling in an exact-key ledger until scheduler-driven abort-key unregister; a conflicting ceiling grants no ceiling authority, while an independently valid co-carried exact proof remains usable.
- Decision delivery timeout, rejection, cancellation, ACK loss, or generic failure remains ambiguous and must not create terminal proof or ceiling evidence.
- Use `/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly` for tests.
- Follow strict TDD: each production behavior must have a focused test that is observed failing for the expected missing behavior before production code is written.
- Commits use Conventional Commits and `git commit -s`.

---

### Task 1: Strict admission terminal wire contract

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py:76-160`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py:109-152,584-663`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py:65-110`

**Interfaces:**
- Consumes: existing `ReverseTerminalNotice` and `PathAbortNotice` strict serialization.
- Produces: `ReverseAdmissionTerminalNotice(may_have_started_through_attempt_id: int | None)` and `PathAbortNotice.reverse_admission_terminal`.

- [ ] **Step 1: Write failing literal wire-shape tests**

Add literal round trips for all four accepted outer shapes:

```python
legacy = {"request_key": _key_payload(), "reason": "ACTIVATION_FAILED"}
exact_only = {**legacy, "reverse_terminal": {"reverse_attempt_id": 3, "state": "TERMINALIZED"}}
ceiling_only = {
    **legacy,
    "reverse_admission_terminal": {"may_have_started_through_attempt_id": None},
}
combined = {
    **exact_only,
    "reverse_admission_terminal": {"may_have_started_through_attempt_id": 2},
}
```

Assert `to_dict()`, `from_dict()`, `encode_path_abort()`, and `decode_path_abort()` preserve each literal shape. The production mutation caught is conflating an absent outer field with explicit inner `null`.

- [ ] **Step 2: Write failing validation and channel-dedupe tests**

Reject:

```python
ReverseAdmissionTerminalNotice(may_have_started_through_attempt_id=True)
ReverseAdmissionTerminalNotice(may_have_started_through_attempt_id=-1)
PathAbortNotice.from_dict({**legacy, "reverse_admission_terminal": None})
PathAbortNotice.from_dict({**legacy, "reverse_admission_terminal": {}})
PathAbortNotice.from_dict({
    **legacy,
    "reverse_admission_terminal": {
        "may_have_started_through_attempt_id": 0,
        "unexpected": "field",
    },
})
```

Extend the real-ZMQ Prefill coordinator test so proofless, ceiling-only, and combined exact+ceiling notices for one registered key are each enqueued once, while an identical full-notice retry is deduplicated. Channel code must remain full-notice based; it does not decide whether two ceiling values conflict.

- [ ] **Step 3: Run the focused tests and observe RED**

Run:

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py
```

Expected: collection/import or assertion failures because the admission terminal type and field do not exist.

- [ ] **Step 4: Implement the minimal strict model**

Add:

```python
@dataclass(frozen=True)
class ReverseAdmissionTerminalNotice:
    may_have_started_through_attempt_id: int | None

    def __post_init__(self) -> None:
        value = self.may_have_started_through_attempt_id
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise PathDecisionValidationError(
                "may_have_started_through_attempt_id must be a non-negative integer or None"
            )

    def to_dict(self) -> JsonObject:
        return {
            "may_have_started_through_attempt_id": self.may_have_started_through_attempt_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> ReverseAdmissionTerminalNotice:
        data = require_exact_payload(
            payload,
            frozenset({"may_have_started_through_attempt_id"}),
        )
        return cls(data["may_have_started_through_attempt_id"])
```

Extend `PathAbortNotice` with `reverse_admission_terminal: ReverseAdmissionTerminalNotice | None = None`. `to_dict()` omits the outer field when absent. `from_dict()` dynamically accepts only the exact union of the two base keys and whichever optional proof keys are present; a present outer field always delegates to the strict nested parser.

- [ ] **Step 5: Run GREEN and commit**

Run the Step 3 command. Expected: all selected tests pass.

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py
git commit -s -m "feat(dual_path): add admission terminal ceiling proof"
```

---

### Task 2: Decode publication watermark and frozen late-ABORT evidence

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py:105-122,250-290,360-390,1525-1570,1670-1825,2072-2100,2254-2365,2420-2450`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py:90-510`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py:80-180`

**Interfaces:**
- Consumes: Task 1 `ReverseAdmissionTerminalNotice` and `PathAbortNotice.reverse_admission_terminal`.
- Produces: exact-admission Decode publication watermark, immutable terminal evidence, and `_DecodeLateAbortContext(endpoint, reverse_admission_terminal)`.

- [ ] **Step 1: Write failing Decode evidence tests**

Update `_expected_abort()` so callers state both exact and ceiling evidence explicitly. Add/adjust assertions for:

```python
# Initial DE_READ validation failure before publication.
PathAbortNotice(
    request_key=_REQUEST_KEY,
    reason=PathAbortReason.ACTIVATION_FAILED,
    reverse_terminal=ReverseTerminalNotice(0, ReverseTerminalState.TERMINALIZED),
    reverse_admission_terminal=ReverseAdmissionTerminalNotice(None),
)

# Worker failure after attempt 0 publication.
PathAbortNotice(
    request_key=_REQUEST_KEY,
    reason=PathAbortReason.ACTIVATION_FAILED,
    reverse_terminal=ReverseTerminalNotice(0, ReverseTerminalState.TERMINALIZED),
    reverse_admission_terminal=ReverseAdmissionTerminalNotice(0),
)

# Attempt 1 validation/endpoint failure after attempt 0 publication.
PathAbortNotice(
    request_key=_REQUEST_KEY,
    reason=PathAbortReason.ACTIVATION_FAILED,
    reverse_terminal=ReverseTerminalNotice(1, ReverseTerminalState.TERMINALIZED),
    reverse_admission_terminal=ReverseAdmissionTerminalNotice(0),
)
```

PE_READ activation failure carries no exact proof and an explicit `ReverseAdmissionTerminalNotice(None)`.

- [ ] **Step 2: Write the failing publication/freeze interleaving test**

Wrap the real `_validate_committed_decision()` so it first returns a valid DE_READ result but invokes `_fail_decode_admission()` before `_activate_received_decision()` reaches its final publication gate. Assert:

```python
assert state.status is _DecodeDecisionStatus.ACTIVATION_FAILED
assert state.reverse_admission_terminal == ReverseAdmissionTerminalNotice(None)
assert metadata.reverse_plans == []
assert scheduler._completion_tracker.open_count() == 0
assert scheduler._reverse_send_completion_ids == {}
```

This catches an activation that passed initial validation but publishes attempt 0 after the admission froze a lower ceiling.

- [ ] **Step 3: Write failing late-context tests**

Publish attempts 0 and 1, finish the request with both `REVERSE_SEND` completions open, and assert the endpoint-only map is replaced by:

```python
_DecodeLateAbortContext(
    endpoint=_PREFILL_ENDPOINT_A,
    reverse_admission_terminal=ReverseAdmissionTerminalNotice(1),
)
```

Close each attempt as failed. Both ABORTs must use the same frozen ceiling `1` and their own exact attempt proof. The context retires only after the final exact-admission send completion closes and must not be reused by a same-request-id replacement admission.

- [ ] **Step 4: Run focused Decode tests and observe RED**

Run:

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py \
  tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py
```

Expected: assertions fail because ABORTs lack admission evidence, publication has no final terminal gate, and late state stores only the endpoint.

- [ ] **Step 5: Add exact-admission live and frozen state**

Extend scheduler state with:

```python
@dataclass(slots=True)
class DecodePathDecisionState:
    decision_request: PathDecisionRequest
    request: Request
    status: _DecodeDecisionStatus
    prefill_control_endpoint: DecodeControlEndpoint | None = None
    reverse_publication_watermark: int | None = None
    reverse_admission_terminal: ReverseAdmissionTerminalNotice | None = None


@dataclass(frozen=True, slots=True)
class _DecodeLateAbortContext:
    endpoint: DecodeControlEndpoint
    reverse_admission_terminal: ReverseAdmissionTerminalNotice
```

Add `_freeze_decode_reverse_admission_terminal(state)` that creates the nested proof once from `state.reverse_publication_watermark` and returns the identical stored object on every later call.

- [ ] **Step 6: Gate publication and advance the watermark before visibility**

Immediately before opening/publishing a Reverse attempt, re-check the same state object:

```python
expected_status = (
    _DecodeDecisionStatus.COMMITTED
    if is_attempt_refresh
    else _DecodeDecisionStatus.PENDING
)
if (
    state.status is not expected_status
    or state.reverse_admission_terminal is not None
):
    return

if reverse_plan is not None:
    state.reverse_publication_watermark = result.reverse_attempt_id
    # Open the exact completion, attach its id, then append to metadata.
```

The final gate, watermark update, metadata append, `_fail_decode_admission()` terminal transition/unregister/freeze, and request-state release remain on the scheduler thread. `_fail_decode_admission()` must set `ACTIVATION_FAILED`, unregister, freeze, then send one ABORT containing both optional exact proof and the frozen admission proof.

- [ ] **Step 7: Preserve frozen evidence for late failures**

Replace `_decode_late_abort_endpoints` with `_decode_late_abort_contexts`. At request-state release, unregister/pop makes further activation impossible; freeze the current watermark and store endpoint plus evidence when exact send completions remain open. Late completion failure reads only the exact-key context and sends its own exact proof plus the stored ceiling. Last-completion close and `shutdown()` clear the context.

Extend `_send_abort_notice()` with:

```python
reverse_admission_terminal: ReverseAdmissionTerminalNotice | None = None
```

and pass it unchanged into `PathAbortNotice`. Existing Prefill-originated ABORTs do not synthesize admission evidence; only Decode failure/release paths call the freeze helper.

- [ ] **Step 8: Run GREEN and commit**

Run the Step 4 command. Expected: all selected tests pass.

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py \
  tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py
git commit -s -m "fix(dual_path): freeze reverse admission publication ceiling"
```

---

### Task 3: Prefill ceiling authority, barrier closure, and ledger lifetime

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py:55-230`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py:250-290,1900-2010,2072-2145,2335-2450`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py:210-780`

**Interfaces:**
- Consumes: Task 1 wire type and Task 2 Decode producer.
- Produces: `TransferCompletionTracker.open_attempt_keys()`, Prefill first-observed ceiling ledger, and exact/ceiling-derived attempt terminalization.

- [ ] **Step 1: Write a failing tracker enumeration test**

Open multiple `REVERSE_RECEIVE` and `REVERSE_SEND` completions across two admissions. Assert:

```python
assert tracker.open_attempt_keys(CompletionKind.REVERSE_RECEIVE, request_key) == (
    ReverseAttemptKey(request_key, 0),
    ReverseAttemptKey(request_key, 1),
)
```

Closed, other-kind, and same-request-id/different-admission records must be excluded. This accessor is read-only and returns a tuple ordered by completion id.

- [ ] **Step 2: Write failing admission invalidation and authority tests**

Split the current stale-attempt oracle into explicit behaviors:

1. Exact attempt-0 proof without a ceiling terminalizes only attempt 0, leaves current attempt 1 open, but emits current attempt-1 invalid blocks because the exact admission is dead.
2. Ceiling `0` without exact proof leaves attempt 0 open even when locally undispatched, while authorizing attempt 1.
3. Explicit nested `null` authorizes every open attempt; an absent outer ceiling authorizes none.
4. Same request id with a replacement admission is neither invalidated nor terminalized.

Assert real completion state, pending terminals, control failures, and `finished_recving`; do not assert only mock calls.

- [ ] **Step 3: Write the failing TP=2 production-window test**

Use `_prefill_de_read_scheduler(expected_worker_count=2)` and two real `_make_prefill_worker()` instances:

1. Dispatch/install attempt 0 on both workers.
2. Refresh and dispatch/install attempt 1 on both workers.
3. Deliver one ABORT carrying exact proof for attempt 0 and ceiling `0`.
4. Assert the scheduler emits current-plan invalid blocks and one exact worker terminal for each attempt.
5. Feed the same scheduler metadata to both real workers. Each produces one report for each completion.
6. After the first worker metadata, both completions remain open. After the second, stale attempt 0 closes without generic injection, current attempt 1 closes and injects `finished_recving`.

This is the production regression for “Decode terminalized through attempt 0 while Prefill had already refreshed to attempt 1.”

- [ ] **Step 4: Write the failing late conflicting-ceiling ledger test**

Observe ceiling `1`, release Prefill request state while attempts remain open, then deliver a combined notice with conflicting ceiling `0` and valid exact proof for attempt 0. Assert:

```python
assert scheduler._prefill_observed_reverse_admission_terminals[request_key] \
    == ReverseAdmissionTerminalNotice(1)
assert attempt_zero_completion.failed is True
assert attempt_one_completion.failed is False
```

The conflicting ceiling grants no authority over attempt 1; the exact proof still applies to attempt 0. Deliver the original ceiling plus exact proof for attempt 1, close the last barrier, and assert abort-key unregister also removes the ledger entry. Add a same-ID replacement assertion so the old ledger never applies to the new key.

- [ ] **Step 5: Run focused Prefill tests and observe RED**

Run:

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py \
  tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py
```

Expected: failures show missing enumeration/ledger, stale proof suppressing current invalid blocks, and no ceiling-derived terminalization.

- [ ] **Step 6: Implement the read-only tracker accessor**

Add:

```python
def open_attempt_keys(
    self,
    completion_kind: CompletionKind,
    request_key: DualPathRequestKey,
) -> tuple[ReverseAttemptKey, ...]:
    return tuple(
        record.reverse_attempt_key
        for record in self._records.values()
        if not record.closed
        and record.completion_kind is completion_kind
        and record.reverse_attempt_key is not None
        and record.reverse_attempt_key.request_key == request_key
    )
```

Insertion order is completion-id order because `_records` is populated by the monotonic allocator.

- [ ] **Step 7: Implement exact-key ceiling ledger and independent invalidation**

Add:

```python
self._prefill_observed_reverse_admission_terminals: dict[
    DualPathRequestKey, ReverseAdmissionTerminalNotice
] = {}
```

In `_handle_received_peer_abort()` snapshot `current_waiting_attempt` before any close action, mark the exact admission invalid, then:

```python
current_attempt_belongs_to_admission = (
    current_waiting_attempt is not None
    and current_waiting_attempt.request_key == notice.request_key
)
```

Use only `active_exact_admission and current_attempt_belongs_to_admission` to build current-plan invalid blocks. Do not require exact proof equality.

For the optional ceiling, store the first object. If a later object differs, log a protocol error and ignore only that ceiling authority. Keep processing a valid co-carried exact proof.

- [ ] **Step 8: Derive per-attempt close authority and preserve the barrier**

Build a set of exact attempts:

```python
terminal_attempts: set[ReverseAttemptKey] = set()
if notice.reverse_terminal is not None:
    terminal_attempts.add(
        ReverseAttemptKey(
            notice.request_key,
            notice.reverse_terminal.reverse_attempt_id,
        )
    )
if accepted_ceiling is not None:
    ceiling = accepted_ceiling.may_have_started_through_attempt_id
    terminal_attempts.update(
        attempt_key
        for attempt_key in self._completion_tracker.open_attempt_keys(
            CompletionKind.REVERSE_RECEIVE,
            notice.request_key,
        )
        if ceiling is None or attempt_key.reverse_attempt_id > ceiling
    )
```

Call `_terminalize_prefill_reverse_attempt()` once per exact key in attempt-id order. That existing helper remains the only close mechanism: undispatched records close in the scheduler; dispatched records stage exact `ReverseReceiveFailureTerminal` and wait for real all-worker reports.

`_unregister_prefill_abort_key()` removes the ceiling ledger entry in the same operation as the scheduler/coordinator registry. Request-state release must not clear it. `shutdown()` clears it.

- [ ] **Step 9: Run GREEN and commit**

Run the Step 5 command. Expected: all selected tests pass.

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py \
  tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py
git commit -s -m "fix(dual_path): close attempts above terminal admission ceiling"
```

---

## Final verification

After all three reviewed task commits:

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/

/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m compileall -q \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path \
  tests/ut/distributed/kv_transfer/dual_path

git diff --check 4b982c283c8f4d51c3a0777f544e0c042190d0b6..HEAD
```

Run an independent whole-branch review over `4b982c283c8f4d51c3a0777f544e0c042190d0b6..HEAD`. NPU E2E remains a separate hardware validation item; do not claim it was run in this CPU worktree.

# DualPath Admission Identity Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one immutable admission identity distinguish reused client
request IDs across Decision, ABORT, Reverse wire transfers, jobs, queues, and
Prefill delivery reconciliation, while deleting the temporary `Request is`
fence network.

**Architecture:** Extend `DualPathRequestKey` with a Decode-coordinator-minted
monotonic `admission_id`. Existing protocol and Reverse entities already embed
that key, so identity propagates transitively. Key the Prefill asynchronous
delivery lifecycle by the complete request key, compare queued control messages
against the current Decode key before consuming them, and retain only one
key-based tripwire for two simultaneously live local admissions with the same
request ID.

**Tech Stack:** Python 3.11, dataclasses, msgspec MessagePack, ZMQ REQ/ROUTER,
vLLM v1 Scheduler connector APIs, pytest, Ruff, Markdownlint.

## Global Constraints

- Do not add an ABORT tombstone, TTL, watchdog, close epoch, UUID, or a second
  job identity.
- Same-incarnation unknown-key ABORT remains ACK-without-enqueue; unknown
  Decision remains `UNKNOWN_REQUEST`; the PE sender remains strict.
- `admission_id` is a non-negative integer allocated monotonically within one
  Decode coordinator incarnation.
- Allocation retry and preemption resume of one admission retain the same
  request key; a distinct later admission receives a different key even when
  its local/client request ID is identical.
- `reverse_wire_id` must encode `admission_id` as well as incarnation, request
  ID, and Reverse attempt ID.
- Exact payload decoding is the compatibility gate. Do not add an old-schema
  fallback. PE, DE, and external log consumers upgrade together.
- Preserve existing log field order; append `admission_id=<int>` to the end of
  each stable DualPath decision/activation/delivery/ABORT log record.
- Do not modify the inherited Forward sender or external-request-ID derivation.
  Document its same-ID reuse race as an accepted residual limitation.
- Keep all changes inside DualPath production/tests and its tracked specs.
- Tests must demonstrate RED for each Critical path before production edits.
- Every implementation commit must use Conventional Commits and `git commit -s`.
- Do not push.

---

### Task 1: Make the request key admission-unique

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_reverse_attempt_identity.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py`
- Mechanically update: all DualPath tests constructing `DualPathRequestKey`

**Interfaces:**

- Produces:
  `DualPathRequestKey(decode_engine_instance_id: str,
  decode_request_id: str, admission_id: int)`.
- Produces:
  `PathDecisionCoordinator.new_request_key(decode_request_id: str)
  -> DualPathRequestKey`, valid only for a Decode coordinator.
- Preserves: `register_pending(key)`, `unregister(key)`, Decision/ABORT exact
  payload behavior, and attempt-watermark semantics.

- [ ] **Step 1: Add focused schema and allocator regression tests**

  Add tests which assert:

  ```python
  first = coordinator.new_request_key("decode-request-7")
  second = coordinator.new_request_key("decode-request-7")
  assert first.decode_engine_instance_id == second.decode_engine_instance_id
  assert first.decode_request_id == second.decode_request_id
  assert (first.admission_id, second.admission_id) == (0, 1)
  assert first != second
  ```

  Also assert `DualPathRequestKey.from_dict` rejects an old two-field payload,
  rejects boolean/negative admission IDs, and round-trips the exact three-field
  payload. Assert a Prefill coordinator cannot mint Decode keys.

- [ ] **Step 2: Add Reverse wire identity RED tests**

  Construct two keys with the same incarnation/request ID and distinct
  admission IDs. Assert attempt zero has different wire IDs and the exact
  format includes the admission component before the attempt component:

  ```text
  ra:<incarnation>:<decode_request_id>:<admission_id>:<attempt_id>
  ```

- [ ] **Step 3: Add Decode registration RED tests**

  Register and release one Decode admission, then register a distinct `Request`
  with the same local ID. Assert the two `DecodePathDecisionState.request_key`
  values have admission IDs 0 and 1. Assert allocation retry/preemption of the
  original `Request` does not invoke `new_request_key` again after
  `do_remote_prefill` has been consumed.

- [ ] **Step 4: Run the focused RED tests**

  Run:

  ```bash
  /Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -q \
    -p no:randomly \
    tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py \
    tests/ut/distributed/kv_transfer/dual_path/test_reverse_attempt_identity.py \
    tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
    tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py
  ```

  Expected: only the new admission-field/allocator/wire-ID assertions fail.

- [ ] **Step 5: Implement the minimal key and allocator changes**

  Add strict `admission_id` validation and serialization to
  `DualPathRequestKey`. Add a coordinator counter initialized to zero and a
  registry-lock-protected `new_request_key` method that asserts Decode role,
  returns the current counter, then increments it. Replace the sole production
  constructor in `_register_pending_decode_decision` with this method. Extend
  `reverse_wire_id` with the admission component.

- [ ] **Step 6: Update constructors and stable logs mechanically**

  Supply explicit admission IDs at every test constructor. Use zero where the
  test does not compare admissions. Append, without reordering earlier fields,
  `admission_id=%s` to Prefill decision, Decode activation, Decision delivery,
  and ABORT delivery logs.

- [ ] **Step 7: Run Task 1 GREEN tests**

  Re-run the Step 4 command. Expected: PASS.

- [ ] **Step 8: Commit Task 1**

  ```bash
  git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path \
    tests/ut/distributed/kv_transfer/dual_path
  git commit -s -m "refactor(dual-path): unify admission identity"
  ```

### Task 2: Discard stale control messages and isolate Reverse jobs

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_de_reverse_send_proof.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py`

**Interfaces:**

- Consumes: complete `DualPathRequestKey` from Task 1.
- Preserves: current-key malformed Decision fail-closed behavior.
- Produces: stale queued Decision/ABORT no-op behavior and exact-admission
  Reverse job matching.

- [ ] **Step 1: Add the late old-success RED test**

  Create Prefill admission A, activate `DE_READ` attempt zero, release A while
  its `REVERSE_COMPLETION` job remains open, then create admission B with the
  same local request ID and activate its attempt zero. Close A's old job and
  assert:

  ```python
  assert request_id not in finished_recving
  assert scheduler._waiting_reverse_attempt_ids[request_id] == new_attempt_key
  assert scheduler._job_ledger.get(new_job_id) is not None
  ```

  The old job must be discarded without touching B.

- [ ] **Step 2: Add the late old-failure RED test**

  Reproduce the same two admissions, report A's old job as failed, and assert B
  receives no `_prefill_control_failures` entry, no invalid-admission marker,
  and no generic completion. B's waiting mapping and job remain intact.

- [ ] **Step 3: Add queued old-message RED tests**

  On Decode, enqueue/retain an old admission A Decision or ABORT, release A,
  register admission B under the same request ID, and drain the old message.
  Assert:

    - old ABORT does not change B from `PENDING` or unregister B;
    - old Decision does not emit a control failure or change B's status;
    - a malformed/current-key Decision still emits `ACTIVATION_FAILED`;
    - an old Decision retransmission is `UNKNOWN_REQUEST`, while an old unknown
      ABORT is ACK-without-enqueue.

- [ ] **Step 4: Run Task 2 RED tests**

  Run the exact new node IDs with `-q -p no:randomly`. Expected: old success
  unparks B, old failure stages against B, and queued messages affect B before
  the production guards are added.

- [ ] **Step 5: Add complete-key consumer guards**

  In `_activate_received_decision`, return immediately after finding active
  state when `state.request_key != decision.result.request_key`, before status
  refresh logic and `_validate_committed_decision`. In `_handle_received_abort`,
  return when the state is absent or its complete key differs from the notice.
  Keep all current-key validation and fail-closed behavior unchanged.

- [ ] **Step 6: Verify job matching requires the full key**

  Do not add job IDs to Scheduler ownership maps. Confirm the existing
  `_request_for_failed_job` and `_close_reverse_completion_job` comparisons use
  `ReverseAttemptKey` equality; Task 1's admission component must make the new
  RED tests pass without another entity. If any helper slices back to request
  ID before equality, replace that comparison with complete-key equality.

- [ ] **Step 7: Run Task 2 GREEN tests and a channel regression group**

  Run the Task 2 node IDs, then:

  ```bash
  /Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -q \
    -p no:randomly \
    tests/ut/distributed/kv_transfer/dual_path/test_abort_protocol.py \
    tests/ut/distributed/kv_transfer/dual_path/test_channel_registry.py \
    tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py
  ```

  Expected: PASS. Socket tests may require the existing loopback permission
  outside the restricted sandbox.

- [ ] **Step 8: Commit Task 2**

  ```bash
  git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
    tests/ut/distributed/kv_transfer/dual_path
  git commit -s -m "fix(dual-path): ignore stale admission control"
  ```

### Task 3: Key Prefill delivery lifecycle by admission and delete fences

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_de_read_recovery.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_audit_fixes.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`

**Interfaces:**

- Consumes: complete request keys and stale-message guards from Tasks 1-2.
- Produces:
  `_prefill_delivery_futures: dict[DualPathRequestKey, Future[None]]`,
  `_prefill_delivery_records: dict[DualPathRequestKey,
  _PrefillDecisionDelivery]`,
  `_prefill_deferred_deliveries: set[DualPathRequestKey]`, and
  `_prefill_invalid_request_keys: set[DualPathRequestKey]`.
- Removes: `_prefill_admission_owners` and
  `_PrefillDecisionDelivery.admission_owner`.

- [ ] **Step 1: Rewrite crash-fence tests as coexistence RED tests**

  Replace assertions for `RuntimeError("request-id reuse is not safe yet")`
  with tests that release admission A while its delivery Future remains
  unresolved, admit B with the same local ID/different key, and assert both key
  entries coexist. Completing or failing A must retire/ABORT A only and must
  not invalidate, stage blocks for, or suppress B.

- [ ] **Step 2: Add deferred replacement isolation RED tests**

  For one admission, keep attempt zero delivery unresolved, install an I7
  replacement attempt, and assert the same key is deferred until the earlier
  Future resolves. While that happens, admit same-ID admission B and assert its
  different key can submit independently. Draining A's deferred key must never
  deliver B's current result under A's identity.

- [ ] **Step 3: Add invalid-marker isolation RED tests**

  Fail A before submission or during delivery, release it, then admit B with the
  same ID. Assert A's invalid key does not make B return its parent result. For
  malformed bootstrap metadata, assert no owner/request-object fence is
  retained; reparsing remains safely fail-closed on each call.

- [ ] **Step 4: Add the simultaneous-live tripwire test**

  Keep A's active `_prefill_request_keys[request_id]`, present B with the same
  local ID but a different full key, and assert exactly one RuntimeError whose
  message says two live DualPath admissions share the local request ID. The
  assertion must depend only on key inequality, never Python object identity.

- [ ] **Step 5: Run Task 3 RED tests**

  Run the exact new node IDs with `-q -p no:randomly`. Expected: the current
  request-ID delivery maps collide or raise the old owner-fence errors.

- [ ] **Step 6: Migrate the asynchronous delivery maps**

  Iterate reconciliation by `request_key`, retrieve `request_id` from the
  immutable record, and define the active admission as:

  ```python
  same_admission = self._prefill_request_keys.get(request_id) == request_key
  ```

  Stage local failure and current I7 invalid blocks only when this is true.
  Always send `DELIVERY_EXHAUSTED` ABORT for the record's own key. Retain a
  failed live `DE_READ` record only until its control failure is published.
  Released or replaced records retire without touching current local maps.

- [ ] **Step 7: Make deferred delivery key-addressable**

  Change `_deliver_prefill_decision` to accept a `DualPathRequestKey`. Derive
  the local ID from it and discard the deferred key when
  `_prefill_request_keys.get(request_id) != request_key`. Active call sites pass
  `result.request_key`; metadata-build iteration passes each deferred key.

- [ ] **Step 8: Remove the owner and request-ID invalid fence network**

  Delete `_prefill_admission_owners`, `admission_owner`,
  `_retain_prefill_admission_failure`, and all owner-identity RuntimeErrors.
  Replace `_prefill_invalid_request_ids` with key-scoped invalid state. Parse
  metadata before consulting that state. Malformed metadata cannot identify an
  admission key and therefore retains no cross-call fence; it simply returns
  the parent result and logs each invalid call.

- [ ] **Step 9: Keep only the two-live-key tripwire**

  After strict metadata parsing, if the active local map contains a different
  complete key, raise the Task 3 Step 4 error. Do not reject a released A merely
  because its Future/record remains under A's distinct key.

- [ ] **Step 10: Update release, publication, shutdown, and surface guards**

  Release discards deferred/invalid state for the released key, but does not
  delete an unresolved delivery for another or older key. Publication removes
  the retained record for the active failed key after metadata copy. Shutdown
  clears all key-based containers. Update construction-parity allowlists to
  remove the owner field and admit only the new key-scoped field names.

- [ ] **Step 11: Run Task 3 GREEN and focused lifecycle groups**

  Run:

  ```bash
  /Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -q \
    -p no:randomly \
    tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py \
    tests/ut/distributed/kv_transfer/dual_path/test_de_read_recovery.py \
    tests/ut/distributed/kv_transfer/dual_path/test_audit_fixes.py \
    tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward.py \
    tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
  ```

  Expected: PASS.

- [ ] **Step 12: Commit Task 3**

  ```bash
  git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
    tests/ut/distributed/kv_transfer/dual_path
  git commit -s -m "refactor(dual-path): key deliveries by admission"
  ```

### Task 4: Synchronize authoritative specs and verify the branch

**Files:**

- Modify:
  `docs/superpowers/specs/2026-08-16-dual-path-abort-notification-and-watchdog-removal.md`
- Modify: `.specs/dual-path-stage1/README.md`
- Modify: `.specs/dual-path-stage1/TASKS.md`
- Modify: `.specs/dual-path-stage1/TRACKING.md`
- Modify: `.specs/dual-path-stage1/prs/PR-06-bidirectional-runtime.md`
- Modify: `.specs/dual-path-stage1/prs/PR-07-activate-partial-hit.md`
- Update: ignored task report in the existing `.superpowers/sdd/` workspace

**Interfaces:**

- Consumes: implemented code and tests from Tasks 1-3.
- Produces: one unambiguous current identity/watchdog contract and an auditable
  verification report.

- [ ] **Step 1: Mark the current design implemented**

  Change the current spec status to accepted and implemented. Verify its key
  schema, stale-message behavior, key-scoped Prefill delivery lifecycle,
  no-tombstone ABORT rule, same-upgrade requirement, log-field rule, and Forward
  residual limitation match production exactly.

- [ ] **Step 2: Supersede active Stage-1 timeout/tombstone claims**

  Add a visible 2026-08-17 supersession notice near the top of each Stage-1
  source-of-truth/status file listed above. Link to the current 2026-08-16 spec
  and state that elapsed-time failure, ABORT tombstones, and two-field request
  keys are historical only. Preserve historical task text rather than silently
  rewriting its chronology.

- [ ] **Step 3: Run the complete DualPath suite**

  Run outside the macOS socket/sysctl-restricted sandbox if needed:

  ```bash
  /Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -q \
    -p no:randomly tests/ut/distributed/kv_transfer/dual_path
  ```

  Expected: at least the `674 passed, 14 warnings` baseline plus the new
  regressions, with zero failures.

- [ ] **Step 4: Run static checks**

  Run the repository-pinned Ruff check and format check on every changed Python
  file, `python -m compileall` on the DualPath package, Markdownlint v0.45.0 on
  every changed tracked Markdown file, and `git diff --check`.

- [ ] **Step 5: Run forbidden-symbol scans**

  Assert production has no removed watchdog environment names, timeout reasons,
  sweep methods, `_prefill_admission_owners`, `admission_owner`, or
  `request-id reuse is not safe yet`. Assert no ABORT tombstone collection or
  TTL was introduced.

- [ ] **Step 6: Write the verification report**

  Record baseline, exact RED failures, focused GREEN groups, full-suite result,
  static outputs, signed commit SHAs, and the explicit no-NPU/no-E2E limitation.
  Record the inherited Forward wire-ID race as accepted and untested, not fixed.

- [ ] **Step 7: Commit Task 4**

  ```bash
  git add .specs/dual-path-stage1 \
    docs/superpowers/specs/2026-08-16-dual-path-abort-notification-and-watchdog-removal.md
  git commit -s -m "docs(dual-path): supersede legacy identity contracts"
  ```

- [ ] **Step 8: Request one scoped re-review**

  Generate the diff from `b199cca4bf211ede28417219e630d92b8848f10f` through
  the final Task 4 commit. The independent reviewer must examine both Critical
  reproducers, the complete delivery-map migration and fence deletion, strict
  schema/log/docs behavior, and the Forward known limitation. No second fix wave
  is authorized; report any remaining load-bearing finding to the user.

# DualPath Prefill Reverse Terminal Delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close a dispatched Prefill `REVERSE_RECEIVE` completion after the Decode peer proves that the exact Reverse attempt is terminal, without scheduler-side early release and without same-ID or cross-attempt aliasing.

**Architecture:** Preserve the existing completion barrier and `dispatched` guard. Decode attaches an optional exact-attempt terminal proof to `PathAbortNotice`; Prefill converts that proof into an exact worker control terminal. Every Prefill worker consumes its matching binding as failed and reports through the existing `DualPathWorkerMetadata` barrier, after which the existing close action injects `finished_recving` and releases blocks.

**Tech Stack:** Python 3.12, msgspec strict wire payloads, pytest, existing DualPath scheduler/worker completion tracker.

## Global Constraints

- Do not add environment variables, connector extra-config, watchdogs, receive timeouts, or a new block-release channel.
- Keep `TransferCompletionTracker.force_fail_completion()` fail-closed for `record.dispatched is True`; never release a dispatched completion directly from the scheduler.
- A worker terminal must identify one exact `(DualPathRequestKey, reverse_attempt_id, completion_id, wire_request_id)` tuple. Never fall back to `request_id`.
- `PathAbortReason` continues to describe why an admission failed. Terminal safety is a separate optional proof, not a reason overload.
- `TERMINALIZED` is emitted only when Decode proves that the exact attempt cannot write again: every Decode worker either never submitted Reverse or reported only after a wire terminal ACK.
- ABORT without exact terminal proof records the exact admission as invalid. It emits invalid-block control failure only while that key is the current exact admission and its current waiting attempt belongs to the same admission; it does not change or close the completion, stage a worker terminal, or inject `finished_recving`.
- A Prefill abort-key registration accepts multiple distinct notices until scheduler-driven unregister. The channel deduplicates only an identical full `PathAbortNotice`; proofless then terminalized proof, and attempt 0 then attempt 1, are separate evidence. Unregister and close clear the associated notice-dedupe state.
- A typed terminal registry rejection (`PathDecisionRejectedError`) proves that the delivery record's decision was not accepted. Only that exact DE_READ attempt may use the same scheduler terminalization helper as an exact peer ABORT. Delivery timeout, ACK loss, cancellation, generic exceptions, and untyped `PathDecisionDeliveryError` remain ambiguous and must not close a completion or stage a worker terminal.
- Every Prefill worker contributes at most one report per completion; duplicate control terminals and late DONE/FAILED wire terminals are idempotent.
- Follow TDD: each production behavior begins with a focused failing test whose failure is observed before implementation.
- Use `/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly` for tests.
- Commits use Conventional Commits and `git commit -s`.

---

### Task 1: Exact terminal wire and connector metadata contracts

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`

**Interfaces:**
- Consumes: existing `DualPathRequestKey`, `ReverseAttemptKey`, `PathAbortNotice`, and `DualPathConnectorMetadata`.
- Produces: `ReverseTerminalState`, `ReverseTerminalNotice`, `ReverseReceiveFailureTerminal`, and `DualPathConnectorMetadata.reverse_receive_failure_terminals`.

- [ ] **Step 1: Write failing strict-wire tests**

Add literal round-trip cases for an ABORT with no terminal proof and an ABORT with:

```python
ReverseTerminalNotice(
    reverse_attempt_id=3,
    state=ReverseTerminalState.TERMINALIZED,
)
```

Assert serialized keys and values literally. Add validation tests rejecting a boolean or negative `reverse_attempt_id`, a non-enum state, unknown nested fields, and incomplete nested payloads. The production mutation caught is accepting an ambiguous or malformed safety proof.

- [ ] **Step 2: Run the wire tests and observe RED**

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py
```

Expected: collection/import or assertion failures because the terminal-proof types and payload field do not exist.

- [ ] **Step 3: Implement the exact wire proof**

Add `ReverseTerminalState.TERMINALIZED` and a frozen `ReverseTerminalNotice(reverse_attempt_id, state)`. Extend `PathAbortNotice` with `reverse_terminal: ReverseTerminalNotice | None = None`. `to_dict()` must omit `reverse_terminal` when it is `None`, preserving the existing two-field ABORT payload; `from_dict()` accepts exactly either the legacy two-field shape or the three-field shape with a strict nested proof. Never infer a proof from `reason`.

- [ ] **Step 4: Write and observe failing connector-metadata tests**

Exercise real validation for:

```python
ReverseReceiveFailureTerminal(
    request_key=DualPathRequestKey("decode", "request", 0),
    reverse_attempt_id=3,
    reverse_receive_completion_id=17,
    wire_request_id="ra:decode:request:0:3",
)
```

Reject boolean or negative attempt/completion ids, empty wire ids, and a wire id inconsistent with `reverse_wire_id(ReverseAttemptKey(...))`. Assert a fresh `DualPathConnectorMetadata` owns an independent empty list.

- [ ] **Step 5: Implement connector metadata and run GREEN**

Add frozen `ReverseReceiveFailureTerminal` and `reverse_receive_failure_terminals: list[...]` to `DualPathConnectorMetadata`. Validate all identifiers and the deterministic wire id.

Run the Task 1 wire tests plus `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`. Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py \
  tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
git commit -s -m "feat(dual_path): carry exact reverse terminal proof"
```

---

### Task 2: Prefill worker exact failure terminal

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/conftest.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_de_local_store_full.py`

**Interfaces:**
- Consumes: Task 1 `ReverseReceiveFailureTerminal` and `DualPathConnectorMetadata.reverse_receive_failure_terminals`.
- Produces: exact, idempotent Prefill worker failure reports through existing `build_connector_worker_meta()`.

- [ ] **Step 1: Write the failing production-sequence worker test**

Use a real test worker. Deliver a `ReverseReceiveBinding`, then its matching `ReverseReceiveFailureTerminal` without placing DONE/FAILED in `kv_recv_layer_thread`. Assert one failure report, removal of the live reverse wire mapping, and addition to the consumed-wire set. The production mutation caught is the current open-binding control failure path that emits no report.

- [ ] **Step 2: Run the focused test and observe RED**

Run the new test alone. Expected: no failure report because `start_load_kv()` ignores `reverse_receive_failure_terminals`.

- [ ] **Step 3: Implement exact consumption**

Add `_pending_reverse_receive_failure_terminals: dict[ReverseAttemptKey, ReverseReceiveFailureTerminal]` to production initialization and the bare-worker test fixture. Add a helper that:

1. Builds the exact `ReverseAttemptKey`.
2. Treats an already-consumed matching wire id as an idempotent no-op.
3. Stores the exact terminal if the binding is not installed yet.
4. Validates completion and wire ids when the binding exists.
5. Calls `_consume_reverse_receive_binding(binding, attempt_key, False)` and removes the pending entry.

Process failure terminals after binding installation in `start_load_kv()`. When `_install_reverse_receive_binding()` installs a binding, immediately consume a pending exact terminal. Never match by request id.

- [ ] **Step 4: Add order, duplicate, late-wire, and isolation tests**

Cover terminal-before-binding, terminal-after-binding, duplicate terminal, late DONE/FAILED, same request id with a different admission/attempt, and mismatched completion/wire ids. Assert real state and emitted reports rather than mock call counts.

- [ ] **Step 5: Run GREEN and regressions**

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py \
  tests/ut/distributed/kv_transfer/dual_path/test_de_local_store_full.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py \
  tests/ut/distributed/kv_transfer/dual_path/conftest.py \
  tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py \
  tests/ut/distributed/kv_transfer/dual_path/test_de_local_store_full.py
git commit -s -m "fix(dual_path): terminate exact prefill reverse bindings"
```

---

### Task 3: Scheduler proof propagation and barrier closure

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py`
- Modify: `.specs/2026-08-20-dualpath-kv-failure-cleanup.md`

**Interfaces:**
- Consumes: Task 1 wire/metadata types and Task 2 worker behavior.
- Produces: Decode exact terminalized ABORT and Prefill scheduler-to-worker-to-scheduler barrier closure.

- [ ] **Step 1: Replace the false zero-report oracle with a failing real-chain test**

Replace `test_prefill_peer_abort_after_binding_dispatch_waits_for_zero_report_barrier` with a test that dispatches and installs a real binding, supplies an exact terminalized ABORT, runs scheduler-produced failure metadata through a real worker, returns that worker's report, and observes `finished_recving` without constructing a report manually.

- [ ] **Step 2: Run the focused test and observe RED**

Expected: no scheduler-to-worker terminal bridge exists, so the completion remains open without a real report.

- [ ] **Step 3: Emit exact terminal proof from Decode**

When a failed `REVERSE_SEND` completion closes, send an ABORT carrying `completion.reverse_attempt_key` as terminalized. For DE_READ activation or refresh failure before a new plan reaches workers, pass the decision's exact attempt id into `_fail_decode_admission()` and emit a terminalized proof for that unstarted attempt. PE_READ and failures without an exact safe attempt send no proof. Never recover the attempt from current request-id state after the failing operation.

- [ ] **Step 4: Stage exact Prefill worker terminals**

In `_handle_received_peer_abort()` resolve the exact admission and snapshot the current waiting attempt before any close action. An ABORT without proof records the exact admission invalid; only a current exact admission whose current waiting attempt belongs to that admission emits invalid-block control failure. It does not mutate or close the completion, stage `ReverseReceiveFailureTerminal`, or inject `finished_recving`. For an exact proof, call `force_fail_completion()`: undispatched completions close immediately; dispatched open completions stage one `ReverseReceiveFailureTerminal` in scheduler-owned pending state. Track staged-or-delivered exact attempts until their completion closes so a duplicate ABORT cannot redispatch the terminal after `build_connector_meta()` drains the pending payload but before worker reports return. Retire that dedupe state from both current and stale completion close paths. Existing worker reports then close through `_aggregate_worker_completion_reports()` and `_run_completion_close_action()`.

Use one exact-attempt terminal helper for both peer ABORT proof and typed Decision registry rejection. `_reconcile_prefill_deliveries()` must take the attempt from the failing delivery record, after any deferred replacement rollback; it must not recover an attempt from current request-id state. A typed rejection may scheduler-close only an undispatched completion. If the binding was dispatched it stages the exact worker terminal and waits for real worker metadata. Ambiguous delivery failures may retain the binding and emit invalid-block control failure, but leave the completion open and stage no terminal.

- [ ] **Step 5: Add barrier and isolation tests**

Cover TP=2, ordinary ABORT without proof, old admission versus same-ID replacement, attempt 0 versus attempt 1, stale-attempt close without current request-id injection, and duplicate ABORT idempotence. Add real-ZMQ channel regressions for proofless→exact, attempt 0→attempt 1, duplicate exact retry, and unregister/drop/re-register. Add delivery reconciliation regressions for typed rejection before dispatch, typed rejection after dispatch through a real worker report, deferred refresh/rollback exact-attempt provenance, and ambiguous delivery remaining open with no worker terminal.

- [ ] **Step 6: Run scheduler/worker GREEN tests**

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m pytest -p no:randomly -q \
  tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py \
  tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Remove superseded spec oracles and commit**

Update original Task 4/7/8 prose so it no longer equates zero reports with unstarted work or asks scheduler to close a dispatched completion.

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py \
  tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py \
  .specs/2026-08-20-dualpath-kv-failure-cleanup.md \
  .specs/2026-08-24-dualpath-prefill-reverse-terminal-delta.md
git commit -s -m "fix(dual_path): close dispatched prefill reverse failures"
```

---

### Task 4: Scoped and aggregate verification

**Files:**
- Verify only; production/test modifications happen only through a reviewed fix round.

**Interfaces:**
- Consumes: Tasks 1-3 complete branch.
- Produces: merge-readiness evidence within CPU/unit-test scope.

- [ ] **Step 1: Run every test file changed in Tasks 1-3**

Use `-p no:randomly -q`. Expected: all pass.

- [ ] **Step 2: Run the selected DualPath aggregate**

Run the same 13-file aggregate recorded in the original cleanup report, including completion tracker, reverse send, channel, connector, recovery, decode scheduler, current attempt, scheduler control, abort scheduler, and DE_READ recovery. If ZMQ/sysctl sandbox restrictions recur, rerun the identical command outside the sandbox and record that distinction.

- [ ] **Step 3: Run non-test checks**

```bash
/Users/leqi/Documents/Code/vllm-ascend/.venv/bin/python -m compileall -q \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path \
  tests/ut/distributed/kv_transfer/dual_path
git diff --check 5c7959bf7..HEAD
git status --short --branch
```

Expected: compileall and diff check exit 0; worktree clean except the branch header. Ruff remains a reported environment limitation if unavailable.

- [ ] **Step 4: Final review**

Review `5c7959bf7..HEAD` for exact-attempt isolation, all-worker barrier ownership, late terminal idempotence, strict wire compatibility, and absence of new configuration. Any Critical or Important finding enters one reviewed fix wave before completion is claimed.

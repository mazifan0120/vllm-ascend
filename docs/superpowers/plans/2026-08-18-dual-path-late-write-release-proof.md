# DualPath Late-Write Release-Proof Implementation Plan

> **Execution contract:** implement sequentially with test-driven development. Keep all behavior changes inside `dual_path`; do not change the parent connector, vLLM core, or wire protocol.

**Goal:** Prevent control-plane failure and request-finish paths from releasing KV blocks while a Reverse data-plane transfer can still write them, while preserving real terminal completion and all-worker closure semantics.

**Design:** Treat an open `REVERSE_RECEIVE`/`REVERSE_SEND` completion as the release gate. Aggregate successful and failed worker reports into one terminal count, latch failure without early close, and run the completion kind's close action only after every worker reports. Cancel only a provably unstarted exact attempt, without any close action. On the worker, retain a core-finished split tracker until its Reverse terminal arrives, but suppress request-level receive completion so source blocks are released only through `finished_sending`.

**Tech stack:** Python, pytest, vLLM v1 KV connector scheduler/worker APIs.

---

## Task 1: Completion closure and exact unstarted cancellation

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py`

1. Add failing tests proving failed reports latch failure without early close; mixed results close once after all workers; over-reporting fails; and `discard_unstarted` is exact, idempotent, rejects mismatches/reported records, and never emits a close action.
2. Run:

   ```bash
   .venv/bin/python -m pytest -q -p no:randomly tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py
   ```

   Confirm failures are caused by the missing behavior.
3. Implement a unified success/failure tally, count both terminal classes, latch failure, close only at `expected_worker_count`, and add exact close-without-action cancellation.
4. Re-run the focused test and confirm green.
5. Commit with sign-off as `fix(dual_path): wait for all worker terminals`.

## Task 2: PE Reverse receive as the scheduler release gate

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py`

1. Add failing tests for PE delay-free with an open current Reverse receive; waiting-attempt retention; real `DONE` and `FAILED` closure after all workers; one-pass mixed aggregation; async delivery failure retaining the binding; synchronous submit rejection cancelling the exact unstarted attempt; and activation-exception rollback restoring prior request state.
2. Run:

   ```bash
   .venv/bin/python -m pytest -q -p no:randomly \
     tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py \
     tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py \
     tests/ut/distributed/kv_transfer/dual_path/test_abort_scheduler.py
   ```

   Confirm the expected lifecycle failures.
3. Implement scheduler behavior:
   - tally each completion ID once with both result counts and run its kind-specific close action after all workers, including failed completions;
   - delay PE free and retain the current waiting attempt while its `REVERSE_RECEIVE` completion is open;
   - preserve the pending binding after asynchronous decision failure so metadata installs it before control failure;
   - make DE_READ activation transactional and cancel only the exact newly opened, provably unstarted completion on activation exception;
   - cancel the exact unstarted attempt on synchronous coordinator submission rejection;
   - never synthesize data-plane completion from control-plane failure.
4. Re-run the focused tests and confirm green.
5. Commit with sign-off as `fix(dual_path): hold reverse receive blocks to terminal`.

## Task 3: Drain Reverse terminal without post-finish receive release

**Files:**

- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_split_lifecycle.py`
- Modify: `tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py`

1. Add failing tests proving a Reverse-`PENDING` tracker is marked and retained after core finish; later `DONE` or `FAILED` emits its attempt completion report and removes the tracker without request-level `done_recving`; all split terminal consumers honor this suppression; control failure is suppressed while an installed/open Reverse attempt exists; and the no-data-plane control-failure path remains unchanged.
2. Run:

   ```bash
   .venv/bin/python -m pytest -q -p no:randomly \
     tests/ut/distributed/kv_transfer/dual_path/test_reverse_send_completion.py \
     tests/ut/distributed/kv_transfer/dual_path/test_split_lifecycle.py \
     tests/ut/distributed/kv_transfer/dual_path/test_completion_and_delivery_recovery.py
   ```

   Confirm failures reproduce premature request-level completion.
3. Add `core_request_finished` to `_SplitTracker`; mark and retain pending Reverse trackers; suppress generic receive completion in store, local Reverse, and Forward consumers after core finish; preserve completion reports; delete retained trackers after either Reverse terminal; and gate synthetic control failure when the asynchronous Reverse attempt is the real release proof.
4. Re-run the focused tests and confirm green.
5. Commit with sign-off as `fix(dual_path): drain reverse terminal after request finish`.

## Task 4: Cross-layer verification

1. Run all touched test files together with random ordering disabled.
2. Run the complete directory:

   ```bash
   .venv/bin/python -m pytest -q -p no:randomly tests/ut/distributed/kv_transfer/dual_path
   ```

   If the broad macOS run reproduces the pre-existing hang/environmental failure, record the exact boundary and run affected files independently; do not classify it as introduced without a before/after reproducer.
3. Run scoped Ruff, format check, and `git diff --check`:

   ```bash
   .venv/bin/python -m ruff check \
     vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py \
     vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
     vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py \
     tests/ut/distributed/kv_transfer/dual_path
   .venv/bin/python -m ruff format --check \
     vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py \
     vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
     vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py \
     tests/ut/distributed/kv_transfer/dual_path
   git diff --check
   ```

4. Inspect the final diff and verify: no out-of-scope files changed; no timeout/watchdog was added; async delivery failure retains its binding; only exact unstarted attempts are silently discarded; all-worker closure applies to both terminal results; and a core-finished DE tracker can emit a completion report but never `done_recving`.
5. If final verification requires test-only adjustments, commit them with sign-off as `test(dual_path): cover late-write release ordering`.

# DualPath Completion Tracker Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the DualPath `JobLedger` vocabulary to `TransferCompletionTracker`/completion vocabulary with zero behavior change, per the approved spec `docs/superpowers/specs/2026-08-17-dual-path-completion-tracker-rename-design.md` (in the main tree).

**Architecture:** Four sequential mechanical substitution passes over the dual_path package and its unit tests (tracker API → metadata schema → worker fact publication → file moves), each ending with a green test suite and one signed commit. The module docstring is rewritten to carry the gating-semantics rationale from the spec.

**Tech Stack:** Python (vLLM plugin package), pytest, ruff, `format.sh ci`.

## Global Constraints

- **Working directory for ALL commands:** `/Users/leqi/Documents/Code/vllm-ascend/.worktrees/dualpath-abort-watchdog-removal` (branch `dualpath-abort-watchdog-removal`). The spec and this plan live in the main tree; do not edit files there.
- **Zero behavior change.** Only identifiers, file names, docstrings, and comments change. Logic, control flow, and structure are untouched.
- **Scope fence:** only these directories are touched — `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` and `tests/ut/distributed/kv_transfer/dual_path/`. Never rename outside them. Never touch `__pycache__`.
- **Log message strings stay verbatim**, including the word "job" inside logger f-strings (ops log parsing depends on layout; see spec §3). The enum value change `REVERSE_COMPLETION` → `REVERSE_RECEIVE` does change one logged kind value; this is spec-approved.
- **Substitution mechanism:** word-boundary regex, longest-token-first. Use this script shape, filling each task's `PAIRS` (run from the worktree root):

```bash
python3 - <<'EOF'
import pathlib, re
SCOPE = [
    *pathlib.Path("vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path").glob("*.py"),
    *pathlib.Path("tests/ut/distributed/kv_transfer/dual_path").glob("*.py"),
]
PAIRS = [
    # (old, new) — longest first; word-boundary regex applied
]
for path in SCOPE:
    text = path.read_text()
    new_text = text
    for old, new in PAIRS:
        new_text = re.sub(rf"\b{re.escape(old)}\b", new, new_text)
    if new_text != text:
        path.write_text(new_text)
        print("updated", path)
EOF
```

- **Commits:** Conventional Commits with scope `dual_path`, signed via `git commit -s` (git user is configured). Commit only the files this task touched.
- **Test gate per task:** `pytest tests/ut/distributed/kv_transfer/dual_path/` — all tests pass. If the baseline (Task 1) is not green, STOP and report; do not start substitutions on a red baseline.

---

### Task 1: Baseline verification

**Files:** none modified.

**Interfaces:**
- Consumes: existing worktree state.
- Produces: a recorded green baseline (required before Task 2).

- [ ] **Step 1: Confirm clean state and branch**

Run:
```bash
cd /Users/leqi/Documents/Code/vllm-ascend/.worktrees/dualpath-abort-watchdog-removal
git status --short
git log --oneline -1
```
Expected: no modified tracked files; HEAD is a `dual-path` commit on branch `dualpath-abort-watchdog-removal`.

- [ ] **Step 2: Run the full dual_path unit suite**

Run:
```bash
pytest tests/ut/distributed/kv_transfer/dual_path/
```
Expected: all tests pass. If anything fails, STOP and report the failures — the rename must not start on a red baseline.

- [ ] **Step 3: Record the passing count**

Run `pytest tests/ut/distributed/kv_transfer/dual_path/ -q 2>/dev/null | tail -1` and note the passed/total counts. Every later task must end with the same counts passing.

No commit in this task.

---

### Task 2: Tracker API vocabulary

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ledgers.py` (class/enum/method names + full module docstring rewrite)
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py` (references + scheduler-local names)
- Modify: all test files whose content matches the pairs (script-driven)

- **Interfaces:**
- Consumes: green baseline from Task 1.
- Produces: `TransferCompletionTracker` with methods `open_completion(completion_kind, *, expected_worker_count, reverse_attempt_key=None) -> CompletionRecord`, `tally_reports(completion_id, report_count) -> bool`, `fail_completion(completion_id) -> bool`, `discard(completion_id) -> bool`, `discard_closed_completions(completion_kind, request_key) -> None`, `get(completion_id)`, `open_count()`; `CompletionRecord(completion_id, completion_kind, expected_worker_count, reverse_attempt_key, completed_worker_count, failed, closed)`; `CompletionKind.REVERSE_RECEIVE`, `CompletionKind.REVERSE_SEND`; scheduler field `_completion_tracker`, `_reverse_send_completion_ids`, methods `_has_open_reverse_send_completion`, `_aggregate_worker_completion_facts`, `_run_completion_close_action`, `_close_reverse_receive_completion`, `_request_for_failed_completion`. Later tasks rely on these exact names.

- [ ] **Step 1: Apply the tracker-API substitution (two runs)**

Run 1 — the Global Constraints script with default SCOPE (all dual_path source + test files) and:

```python
PAIRS = [
    ("_close_reverse_completion_job", "_close_reverse_receive_completion"),
    ("_aggregate_worker_job_facts", "_aggregate_worker_completion_facts"),
    ("_request_for_failed_job", "_request_for_failed_completion"),
    ("_has_open_reverse_send_job", "_has_open_reverse_send_completion"),
    ("_run_job_close_action", "_run_completion_close_action"),
    ("_reverse_send_job_ids", "_reverse_send_completion_ids"),
    ("discard_closed_jobs", "discard_closed_completions"),
    ("_next_job_id", "_next_completion_id"),
    ("_job_ledger", "_completion_tracker"),
    ("JobLedger", "TransferCompletionTracker"),
    ("JobRecord", "CompletionRecord"),
    ("JobKind", "CompletionKind"),
    ("REVERSE_COMPLETION", "REVERSE_RECEIVE"),
    ("create_job", "open_completion"),
    ("record_reports", "tally_reports"),
    ("record_failure", "fail_completion"),
    ("job_kind", "completion_kind"),
    ("job_id", "completion_id"),
    ("send_job", "send_completion"),
]
```

Run 2 — bare-token pass, **SCOPE restricted to `ledgers.py`, `scheduler.py`, and the test files only** (NOT `worker.py`, whose prose is swept in Task 4):

```python
PAIRS = [
    ("send_job", "send_completion"),
    ("job", "completion"),
]
```

Caution: the bare `job` pair also rewrites the failure-log literal `"DualPath job %s (kind=%s) reported failed; ..."` in `scheduler.py`; Step 3 restores it verbatim.

- [ ] **Step 2: Rewrite the `ledgers.py` module docstring and class docstrings**

Replace the module docstring of `ledgers.py` with exactly:

```python
# SPDX-License-Identifier: Apache-2.0
"""Per-attempt transfer completion tracking for DualPath Stage-2.

Each reverse attempt opens one completion on the Scheduler; every
participating worker reports it exactly once through
``DualPathWorkerMetadata``; at ``expected_worker_count`` the close action
runs exactly once. Reports arriving for a closed completion are ignored,
and a failure report closes the completion as failed.

Why this lives on the Scheduler: only the Scheduler sees every worker's
reports (the per-step worker-metadata fold is the sole cross-worker
aggregation point), and only it owns the close actions — unparking a
parked request via ``finished_recving``, authorizing the delayed free of
source blocks via ``finished_sending``, and failing the owning request.

Phase coupling makes per-worker epoch serialization structural: a
worker's completion report and its local reverse DONE latch drain in the
same step boundary, while replacement plans install at step start, so a
worker cannot hold an unfinished attempt alongside a replacement (it
raises instead). Tolerating concurrently open completions on the
Scheduler is defense in depth, not a normal-path state.

In-tree precedent: ascend_store's ``sending_events`` is the same counting
pattern without attempt identity; the upstream ``KVOutputAggregator``
cannot express attempt-scoped identity, failure closure, or closed-report
dedup (its per-request counting reopens on late duplicates).
"""
```

Replace the `TransferCompletionTracker` class docstring with exactly:

```python
    """Monotonic completion-id allocator with the all-worker completion rule.

    Each worker contributes at most one report per completion; aggregated
    counts are capped at ``expected_worker_count`` so a duplicated report can
    never overshoot, and the kind-specific close action runs exactly once.
    Reports arriving for a ``closed`` completion are ignored. A failure
    report closes the completion as ``failed``.
    """
```

- [ ] **Step 3: Restore the protected log literal and sweep leftovers**

In `scheduler.py`, find the failure log (after Step 1 it reads `"DualPath completion %s ..."` with `completion_id`/`completion.completion_kind.value` arguments) and restore the message string verbatim, keeping the renamed variables:

```python
                logger.error(
                    "DualPath job %s (kind=%s) reported failed; the owning request is failed closed",
                    completion_id,
                    completion.completion_kind.value,
                )
```

Then run `grep -rn "completion completion\|\bjob\b\|jobs" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ledgers.py vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py` and hand-fix: any "completion completion" phrasing created by the bare-token pass in comments/docstrings, and any remaining identifier or prose `job`/`jobs` in these two files. Expected end state: only the restored logger literal in `scheduler.py` contains `job`.

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ut/distributed/kv_transfer/dual_path/`
Expected: pass, same counts as Task 1. (Test files were updated by the same substitution pass, so imports and attribute accesses stay consistent.)

- [ ] **Step 5: Commit**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ledgers.py \
        vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
        tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "refactor(dual_path): rename tracker API to CompletionTracker vocabulary" \
  -m "JobLedger/JobRecord/JobKind become TransferCompletionTracker/CompletionRecord/CompletionKind; ledger methods become open_completion/tally_reports/fail_completion; scheduler-side names follow. Zero behavior change; the module docstring now records why the all-worker completion gate lives on the scheduler."
```

---

### Task 3: Metadata schema vocabulary

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py` (fields + serialization/validation string literals)
- Modify: `scheduler.py`, `worker.py` (references)
- Modify: test files including `conftest.py` (script-driven)

**Interfaces:**
- Consumes: Task 2 names (`_completion_tracker`, `CompletionKind`, …).
- Produces: `DualPathWorkerMetadata(completion_reports: dict[int, int], failure_reports: dict[int, int])`; `ReversePlan.reverse_send_completion_id: int | None`; `ReverseReceiveBinding.reverse_receive_completion_id: int` (serialization dict keys and validation tuples renamed in lockstep); `conftest.make_worker_metadata(completion_reports=None, failure_reports=None)`. Task 4 and Task 5 rely on these exact names.

- [ ] **Step 1: Apply the metadata substitution**

Run the Global Constraints script with:

```python
PAIRS = [
    ("reverse_completion_job_id", "reverse_receive_completion_id"),
    ("reverse_send_job_id", "reverse_send_completion_id"),
    ("completed_jobs", "completion_reports"),
    ("failed_jobs", "failure_reports"),
]
```

Word boundaries make `reverse_send_completion_id`/`reverse_receive_completion_id` immune to the earlier bare-token renames, and the string literals inside `metadata.py` serialization (`to_dict` field lists, `from_dict` keys, validation tuples) match these exact tokens, so they are renamed in the same pass.

- [ ] **Step 2: Verify serialization literals moved in lockstep**

Run: `grep -n "completion_id\|completion_reports\|failure_reports" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
Expected: every hit uses the new names; `grep -n "job" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py` returns nothing.

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/ut/distributed/kv_transfer/dual_path/`
Expected: pass, same counts as Task 1.

- [ ] **Step 4: Commit**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ \
        tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "refactor(dual_path): rename completion metadata fields" \
  -m "DualPathWorkerMetadata carries completion_reports/failure_reports; ReversePlan.reverse_send_job_id becomes reverse_send_completion_id and ReverseReceiveBinding.reverse_completion_job_id becomes reverse_receive_completion_id, including the dict serialization and validation literals that name these fields."
```

---

### Task 4: Worker fact-publication vocabulary and prose sweep

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py` (any leftover prose)
- Modify: test files with worker-field references (`conftest.py` and others, script-driven)

**Interfaces:**
- Consumes: Task 2 and Task 3 names.
- Produces: worker methods/fields `_publish_completion_fact(completion_id, *, succeeded)`, `_completion_facts_lock`, `_pending_completion_reports`, `_pending_failure_reports`. Task 5 relies on these exact names.

- [ ] **Step 1: Apply the worker substitution**

Run the Global Constraints script with:

```python
PAIRS = [
    ("_record_sender_job", "_publish_completion_fact"),
    ("_sender_job_facts_lock", "_completion_facts_lock"),
    ("_completed_sender_jobs", "_pending_completion_reports"),
    ("_failed_sender_jobs", "_pending_failure_reports"),
]
```

- [ ] **Step 2: Prose sweep across the package**

Run: `grep -rni "job" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ | grep -v __pycache__`
Expected remaining hits: only the protected failure-log literal in `scheduler.py` and possibly docstrings in test files. Rewrite any test-file docstring occurrences (e.g. `test_job_ledger.py` module docstring `"Stage-2 W2: completion-job ledger accounting..."` → `"Stage-2 W2: completion tracker accounting and the no-pinning contract."`) by hand.

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/ut/distributed/kv_transfer/dual_path/`
Expected: pass, same counts as Task 1.

- [ ] **Step 4: Commit**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py \
        vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
        tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "refactor(dual_path): rename worker completion fact publication" \
  -m "_record_sender_job is used by both the DE send terminal and the PE receive terminal, so it becomes _publish_completion_fact; the fact lock and pending-report dicts follow, and the last job-vocabulary prose is swept."
```

---

### Task 5: File moves, import paths, and final gates

**Files:**
- Rename: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ledgers.py` → `completion_tracker.py`
- Rename: `tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py` → `test_completion_tracker.py`
- Modify: `scheduler.py` import line (currently line 23: `from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.ledgers import (`) and any test importing the `ledgers` module.

**Interfaces:**
- Consumes: Tasks 2–4 names.
- Produces: module path `...dual_path.completion_tracker` importable as the final public layout; the finished refactor.

- [ ] **Step 1: Move the files**

```bash
git mv vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ledgers.py \
       vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py
git mv tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py \
       tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py
```

- [ ] **Step 2: Update module-path imports and attribute-style references**

Run: `grep -rn "dual_path\.ledgers\|import ledgers\|ledgers\." vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ tests/ut/distributed/kv_transfer/dual_path/ | grep -v __pycache__`
For each hit, replace the module token `ledgers` with `completion_tracker` — both the import statements (e.g. `scheduler.py` line 23, `from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.ledgers import (`) and attribute-style usages such as `ledgers.CompletionKind.REVERSE_RECEIVE` in the moved test file. After editing, the scheduler import reads:

```python
from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.completion_tracker import (
```

- [ ] **Step 3: Residual-vocabulary gate**

Run:
```bash
grep -rn "JobLedger\|JobRecord\|JobKind\|job_id\|completed_jobs\|failed_jobs\|reverse_send_job_id\|reverse_completion_job_id\|_record_sender_job\|create_job\|record_reports\|record_failure\|discard_closed_jobs" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ tests/ut/distributed/kv_transfer/dual_path/ | grep -v __pycache__
```
Expected: zero hits. Then run `grep -rni "job" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ tests/ut/distributed/kv_transfer/dual_path/ | grep -v __pycache__` — the ONLY allowed remaining hits are (a) log f-string literals in `scheduler.py` (the protected failure-path messages) and (b) `DualPathControlFailureReason.REVERSE_JOB_FAILED` (explicitly parked for the final review: its enum value is log-exposed and it sits outside the approved rename map). Rename every other remaining `job` token — known compounds include `first_job_id`, `old_job_id`, `latest_job_id`, `successful_job_id`, `failed_job`, `open_job`, `completion_job`, `completion_job_id`, helpers `_close_send_job`/`_fail_send_job`, class names `TestJobLedger`/`TestJobRecordReclamation`.

- [ ] **Step 4: Full suite**

Run: `pytest tests/ut/distributed/kv_transfer/dual_path/`
Expected: pass, same counts as Task 1.

- [ ] **Step 5: Lint and format gates**

Run:
```bash
ruff check vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/
ruff format vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ tests/ut/distributed/kv_transfer/dual_path/
bash format.sh ci
```
Expected: ruff clean; if `format.sh ci` rewrites files, `git add` them and include in the commit.

- [ ] **Step 6: Commit**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ \
        tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "refactor(dual_path): move ledgers.py to completion_tracker.py" \
  -m "Final module and test-file renames closing the CompletionTracker vocabulary refactor; import paths updated, residual-vocabulary grep clean, suite and format gates green."
```

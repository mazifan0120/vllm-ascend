# DualPath Completion Tracker Rename Design

Status: design — approved direction (vocabulary: CompletionTracker; scope: dual_path only)
Date: 2026-08-17
Scope: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` and
`tests/ut/distributed/kv_transfer/dual_path/`

## 0. Baseline

Line-number references in this document cite `dev/dualpath` @ `8b0c33607`.
A second, newer line of development exists in the
`.worktrees/dualpath-abort-watchdog-removal` worktree (15 commits ahead:
admission-identity unification, abort/watchdog work). Its dual_path deltas
touch `config.py`, `metadata.py`, `path_decision.py`,
`path_decision_channel.py`, and `scheduler.py`; **`ledgers.py` and
`worker.py` are identical across both trees**.

The rename map was verified against both trees on 2026-08-17: every mapped
identifier exists unchanged on the newer branch, and no job-related
identifier outside this map appears on either tree. Implementation lands on
top of whichever branch merges last; line numbers will have shifted but the
substitution set does not change.

## 1. Motivation

The current `JobLedger` vocabulary obscures what the mechanism does. Three
specific naming debts, verified in code:

1. **"Ledger" is an accounting metaphor, not semantics.** The class performs
   per-attempt all-worker completion counting whose close action fires exactly
   once. Nothing about "ledger" conveys the all-worker gate, the
   close-exactly-once rule, or the failure path.
2. **`_record_sender_job` is a genuine misnomer.** It is called by the DE
   worker after its reverse-send terminal ACK (`worker.py:639-643`) *and* by
   the PE worker when consuming a received reverse terminal
   (`worker.py:289-307`). The "sender" half of the name is wrong for the
   receiving side.
3. **"job" has no anchor in the tree.** `job_id` → `completed_jobs` /
   `failed_jobs` → `reverse_send_job_id` → `_run_job_close_action` all lean on
   a word with no precedent elsewhere in the repository (ascend_store uses
   `event_id` / `completed_events`; the upstream req-level aggregator counts
   plain request ids).

The chosen vocabulary states the mechanism directly: each reverse attempt
opens a **completion** that every participating worker must **report** exactly
once before the scheduler runs its close action.

## 2. Goals / non-goals

Goals:

- Zero behavior change. Semantics are locked by
  `tests/ut/distributed/kv_transfer/dual_path/`; only identifiers, file names,
  docstrings, and one added comment block change.
- Exhaustive, mechanical rename map (below) so implementation is substitution,
  not judgment.
- Document the gating semantics at the point of use so the "why scheduler-side"
  question (the one this rename grew out of) is answered in the code.

Non-goals:

- No API reshaping (no splitting `_run_completion_close_action` by role).
- No shared utility extraction for ascend_store / cpu_offload counters; that
  is a separate follow-up with cross-connector equivalence proofs.
- No changes to historical design documents; the upcoming RFC will use the
  new names.

## 3. Rename map

### Module and class surface (`ledgers.py` → `completion_tracker.py`)

| Current | New |
|---|---|
| `ledgers.py` | `completion_tracker.py` |
| `JobLedger` | `TransferCompletionTracker` |
| `JobRecord` | `CompletionRecord` |
| `JobKind` | `CompletionKind` |
| `JobKind.REVERSE_COMPLETION` | `CompletionKind.REVERSE_RECEIVE` |
| `JobKind.REVERSE_SEND` | `CompletionKind.REVERSE_SEND` |
| `JobRecord.job_id` | `CompletionRecord.completion_id` |
| `JobRecord.job_kind` | `CompletionRecord.completion_kind` |
| `create_job(...)` | `open_completion(...)` |
| `record_reports(job_id, n)` | `tally_reports(completion_id, n)` |
| `record_failure(job_id)` | `fail_completion(completion_id)` |
| `discard_closed_jobs(kind, key)` | `discard_closed_completions(kind, key)` |
| `discard(job_id)` | `discard(completion_id)` (method name unchanged; parameter follows the id rename) |
| `open_count()` | `open_count()` (unchanged) |
| `_next_job_id` | `_next_completion_id` |

Unchanged: `expected_worker_count`, `completed_worker_count`, `failed`,
`closed`, `reverse_attempt_key`.

Rationale for `REVERSE_COMPLETION` → `REVERSE_RECEIVE`: the old value names
the generic concept where the kind is specifically the PE-side (receiving)
completion; `REVERSE_RECEIVE` pairs symmetrically with `REVERSE_SEND`. The
enum is scheduler-internal; worker metadata carries only integer ids.

### Scheduler (`scheduler.py`)

| Current | New |
|---|---|
| `_job_ledger` | `_completion_tracker` |
| `_reverse_send_job_ids` | `_reverse_send_completion_ids` |
| `_has_open_reverse_send_job` | `_has_open_reverse_send_completion` |
| `_aggregate_worker_job_facts` | `_aggregate_worker_completion_facts` |
| `_run_job_close_action` | `_run_completion_close_action` |
| `_close_reverse_completion_job` | `_close_reverse_receive_completion` |
| `_request_for_failed_job` | `_request_for_failed_completion` |
| local names (`send_job`, `job`, `job_id` parameters/bodies) | follow the same vocabulary (`send_completion`, `completion`, `completion_id`) |

Unchanged: `_expected_worker_count`, `_pending_finished_sending`, and all log
message strings (field layouts are consumed by ops log parsing; see the
log-layout stability note near the activation log — `_log_decision_activation`).

### Worker (`worker.py`)

| Current | New |
|---|---|
| `_record_sender_job(job_id, *, succeeded)` | `_publish_completion_fact(completion_id, *, succeeded)` |
| `_sender_job_facts_lock` | `_completion_facts_lock` |
| `_completed_sender_jobs` | `_pending_completion_reports` |
| `_failed_sender_jobs` | `_pending_failure_reports` |

### Metadata (`metadata.py`)

| Current | New |
|---|---|
| `DualPathWorkerMetadata.completed_jobs: dict[int, int]` | `completion_reports: dict[int, int]` |
| `DualPathWorkerMetadata.failed_jobs: dict[int, int]` | `failure_reports: dict[int, int]` |
| `ReversePlan.reverse_send_job_id` | `reverse_send_completion_id` |
| `ReverseReceiveBinding.reverse_completion_job_id` | `reverse_receive_completion_id` |

All four containers are plain dataclasses. On the newer worktree branch,
`metadata.py` adds explicit dict (de)serialization and field validation for
`ReversePlan` / `ReverseReceiveBinding`; the **string literals naming these
fields** (serialize field lists, `from_dict` keys, validation tuples) must be
renamed in lockstep — they are part of the substitution set, not an optional
cleanup. `ReversePlan` fields travel the PE→DE decision channel
serialization; both sides change together and mixed-version deployment is
already unsupported (Stage-2 §Decision channel framing), so there is no
compatibility surface.

### Tests

- `git mv tests/ut/distributed/kv_transfer/dual_path/test_job_ledger.py
  test_completion_tracker.py`; internal identifiers follow the map.
- All remaining dual_path test files: mechanical substitution of the map.
  Footprint verified by grep on **both** trees: 4 source files plus test
  files (25 files total on each tree; the composition differs — the worktree
  branch adds abort-protocol tests that also reference the job vocabulary,
  e.g. `test_abort_scheduler.py`).

## 4. Documentation additions

`completion_tracker.py` module docstring states, in order:

1. Purpose: per reverse attempt, count one completion report per
   participating worker; at `expected_worker_count` the close action runs
   exactly once; reports after closure are inert; a failure report closes
   the completion as failed.
2. Why scheduler-side: only the scheduler sees every worker's reports (the
   per-step worker-metadata fold is the sole cross-worker aggregation
   point), and only it owns the close actions — unpark via
   `finished_recving`, delayed-free release via `finished_sending`,
   fail-close.
3. Phase coupling: per worker, the completion report harvest and the local
   reverse DONE latch drain in the same step boundary, while replacement
   plans install at step start; per-worker epoch serialization is therefore
   structural (the worker raises on violation), and the tracker's tolerance
   of concurrently open attempts is defense-in-depth, not a normal-path
   state.
4. Precedents pointer: ascend_store `sending_events` is the same pattern
   without attempt identity; the upstream `KVOutputAggregator` cannot
   express attempt-scoped identity, failure closure, or closed-report dedup
   (its per-req counting reopens on late duplicates).

## 5. Invariants preserved

- Counting: at most one report per worker per completion, capped at
  `expected_worker_count`; close fires exactly once; `closed` absorbs late
  and duplicate reports.
- Failure: one failure report closes the completion; the owning request is
  failed through the control-failure/watchdog path.
- Lifecycle: closed records are discarded with their owning attempt/request;
  open records stay reportable.
- I4 gate, delay-free harvest, `_has_open_reverse_send_completion`
  all-attempts-closed release rule: logic untouched.

## 6. Validation

1. `pytest -sv tests/ut/distributed/kv_transfer/dual_path/` — full suite
   green before and after (run before as baseline).
2. `ruff check vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` and
   `ruff format` on touched files.
3. `bash format.sh ci` (markdownlint covers this document).
4. `grep -rn "JobLedger\|job_id\|completed_jobs\|failed_jobs\|reverse_send_job_id\|reverse_completion_job_id\|_record_sender_job" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/ tests/ut/distributed/kv_transfer/dual_path/` returns
   no hits (historical docs under `.specs/` and `docs/superpowers/` are
   excluded — they record history; other modules keep their own `job_id`
   vocabulary).

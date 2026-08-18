# DualPath Late-Write Release-Proof Design

## Goal

Prevent PE destination blocks and DE source blocks from being released while a
Reverse transfer may still write or read them. Control-plane failure may fail a
request, but only attempt-scoped data-plane terminal reports from every worker
may authorize release of an already-started Reverse transfer.

## Scope

This change is limited to the DualPath connector:

- `completion_tracker.py`: all-worker terminal aggregation and explicit removal
  of a proven-unstarted completion.
- `scheduler.py`: PE Reverse-receive hold ownership, uncertain-delivery
  retention, exact rollback of never-submitted attempts, and terminal close
  actions.
- `worker.py`: retain a finished Decode request's split tracker until the
  Reverse terminal arrives without emitting another request-level receive
  completion.
- DualPath unit and scheduler integration tests.

It does not change the parent Mooncake connector, vLLM core, control protocol,
ABORT cancellation, sender queues, terminal retry count, Forward protocol, or
Store protocol.

## Safety Rules

1. A control-plane terminal is never proof that an already-started data-plane
   operation has stopped.
2. A PE request with an open current `REVERSE_RECEIVE` completion holds its
   destination blocks after logical request finish.
3. A DE request with any open `REVERSE_SEND` completion holds its source blocks
   after logical request finish.
4. An asynchronous Decision failure is ambiguous: the DE may have accepted the
   Decision before its ACK was lost. PE therefore installs and retains the
   Reverse binding and waits for the real terminal.
5. A completion may be removed without a data-plane terminal only when local
   code proves its work was never submitted. This removal performs no close
   action and emits no finished request ID.
6. DONE and FAILED both count as worker terminals. A completion closes only
   after the total terminal count reaches `expected_worker_count`; any FAILED
   report makes the final outcome failed.
7. After vLLM Core has logically finished a split Decode request, later Store,
   Forward, or Reverse terminals update transfer state and completion reports
   but never emit generic `finished_recving` for that request.

## Completion Tracker

`CompletionRecord.completed_worker_count` represents all terminal workers,
whether successful or failed. `failed` latches when any failed report arrives.
Scheduler aggregation processes success and failure counts for a completion as
one operation so a success count cannot close and discard the record before a
failure count from the same worker batch is observed.

The tracker adds an attempt-checked operation equivalent to:

```python
discard_unstarted(
    completion_id: int,
    *,
    expected_kind: CompletionKind,
    expected_attempt_key: ReverseAttemptKey,
) -> bool
```

Its contract is:

- missing or already-closed record: idempotent no-op;
- open matching record with zero terminal reports: delete it directly;
- wrong kind/key or any terminal report already observed: raise an invariant
  error;
- never run a Scheduler close action and never inject `finished_recving` or
  `finished_sending`.

## PE Scheduler Lifecycle

### Normal and uncertain delivery

`_activate_de_read_path()` opens the current attempt's receive completion and
stages its binding and plan. `_update_prefill_state_after_alloc()` records that
attempt as the current waiting owner before submitting the Decision.

If the returned delivery Future later fails or is cancelled,
`_reconcile_prefill_deliveries()` marks the request invalid and stages control
failure metadata but keeps the pending binding. The next metadata build carries
the binding before the control failure; each PE Worker installs the wire mapping
before processing invalid blocks. The request remains held until the real
Reverse terminal closes the completion.

### Proven-never-submitted rollback

Two local paths prove that the Reverse writer could not have started:

1. `_activate_de_read_path()` opens a completion and then raises before Decision
   submission.
2. `PathDecisionCoordinator.submit()` raises synchronously and returns no
   Future.

Activation is handled transactionally. Only artifacts created for the current
attempt are removed, and previous-attempt state is restored. This is necessary
for recovery, where an older delivered attempt may coexist with a newly opened,
not-yet-submitted attempt. The exact new completion is removed with
`discard_unstarted()`.

Synchronous submit failure removes the exact current pending attempt and its
unstarted completion. Asynchronous Future failure never uses this operation.

### Finish and terminal

`_delay_free_for_connector()` returns true for a Prefill request whose current
waiting attempt still has an open `REVERSE_RECEIVE` completion. Request cleanup
keeps `_waiting_reverse_attempt_ids` while that completion is open.

When all PE workers report DONE/FAILED, the Scheduler:

1. latches failure and invalidation if needed;
2. removes the current waiting owner and completion record;
3. injects `finished_recving` unconditionally for the current attempt.

The Core then either resumes a waiting request or frees the blocks of an already
finished delayed request. A stale attempt closes without emitting a generic
request ID.

## DE Worker Lifecycle

`_SplitTracker` gains a concrete lifecycle flag indicating that vLLM Core has
already finished the request. `_release_split_request_state()` behaves as
follows:

- terminal Reverse phase (`SKIPPED`, `DONE`, or `FAILED`): remove the tracker;
- pending Reverse phase: mark the tracker Core-finished and retain it for the
  terminal/completion report.

All split request-level terminal emitters use the same gate:

- `_consume_store_completions()`;
- `_drain_local_reverse_terminals()`;
- `_consume_forward_wire_terminals()`.

For a Core-finished tracker they still update phases, invalid blocks, and
attempt-scoped completion reports, but they do not add the request to
`done_recving`. Once the retained Reverse becomes terminal, the tracker is
removed. The Scheduler releases the DE source blocks only through the existing
`finished_sending` close action.

## Failure and Liveness

An actual terminal failure is a release proof only for the worker that reported
it. Other workers may still be writing, so failure waits at the same all-worker
barrier as success.

If a peer or worker dies permanently and never supplies a terminal, the relevant
blocks remain held indefinitely. This is an accepted liveness risk of the
correctness-first design. No timeout-based release is added.

## Tests

Tests must cover:

- mixed worker success/failure closes only after all reports and ends failed;
- `discard_unstarted()` success, invariant rejection, and idempotency;
- initial and replacement activation exceptions cancel only the newly opened
  completion and restore prior attempt state;
- synchronous submit failure cancels the unstarted attempt;
- asynchronous failure before the first metadata build sends binding plus
  control failure and produces no synthetic receive completion;
- PE finish retains blocks and attempt ownership until real DONE/FAILED;
- Decode abort followed by Reverse success/failure produces a send completion
  report but no `done_recving`;
- retained failed trackers are removed;
- Store and Forward late terminals obey the same Core-finished gate;
- a combined worker output cannot free the same request through both receiving
  and sending paths in one Core pass.

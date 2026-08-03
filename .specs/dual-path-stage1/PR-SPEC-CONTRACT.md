# PR Spec Contract

Every file under `prs/` follows the same review contract so that reviewers can
compare PRs without learning a new document shape each time.

## Required metadata

Each spec starts with:

- series position;
- tracker status;
- dependencies and blocked successors;
- implementation-task mapping;
- user-visible behavior classification;
- activation state after merge.

The status in a PR spec describes the approved scope and normally remains
`PLANNED`; operational status is maintained only in `TRACKING.md`.

## Required sections

1. **Goal** — one outcome sentence.
2. **Merge-state contract** — exact behavior available immediately after merge.
3. **In scope** — responsibilities delivered by this PR.
4. **Out of scope** — responsibilities explicitly deferred.
5. **Expected code surface** — files or modules a reviewer should expect.
6. **Required tests** — focused tests and regression boundaries.
7. **Acceptance gates** — binary conditions for review readiness.
8. **Rollback contract** — behavior restored by reverting this PR.
9. **Review focus** — the small set of architectural questions reviewers must
   answer.

## Change rules

- Do not add implementation progress, code-review transcripts, or raw logs to a
  PR spec.
- Do not broaden a PR merely because a dependency is convenient to edit.
- If a change is required by more than one later PR, put it in the earliest PR
  that has a real consumer and complete tests.
- Refactoring that changes a shared parent implementation stays separate from
  feature activation.
- Protocol types and callbacks must have an exercised producer and consumer in
  the same PR.
- Intermediate PRs may carry dormant behavior only when configuration cannot
  activate incomplete I/O.
- If PR-03 exceeds a reviewable control-plane change, split it into `PR-03A`
  (schema and Proxy rendezvous) and `PR-03B` (Engine coordinators and Scheduler
  inbox), then update `TRACKING.md` before opening either PR.

## PR description order

The GitHub PR description should use this order:

1. `Series: [DualPath Stage 1][N/6]`
2. dependency and stacked-base information;
3. what the PR does and why;
4. behavior after merge;
5. non-goals;
6. test evidence, separating CPU-only and NPU evidence;
7. reviewer reading order;
8. rollback behavior;
9. preserved `- vLLM version:` and `- vLLM main:` lines from the repository PR
   template.

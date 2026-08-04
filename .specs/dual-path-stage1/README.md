# DualPathConnector Stage 1 Delivery

This directory is the delivery control center for DualPathConnector Stage 1.
It tracks the ordered pull-request series, the contract of each PR, and the
evidence required to advance a PR. It does not contain implementation code or
raw test logs.

## Start here

1. Read [TRACKING.md](TRACKING.md) for the current series status and dependency
   chain.
2. Read the corresponding file under [prs/](prs/) before starting or reviewing
   a PR.
3. Read [TASKS.md](TASKS.md) for the task definitions owned by that PR.
4. Apply [PR-SPEC-CONTRACT.md](PR-SPEC-CONTRACT.md) whenever a PR spec or the
   series structure changes.

## Source-of-truth rules

- `TRACKING.md` is the only mutable source for PR status, branch, PR URL, and
  validation evidence.
- `prs/PR-XX-*.md` defines the stable scope and merge contract of one PR.
- A PR spec changes only when its architectural boundary changes. Routine
  progress updates belong only in `TRACKING.md`.
- `TASKS.md` is the only source for implementation task definitions and PR
  ownership. Semantic IDs such as `DP-COV-01` remain stable if a later PR is
  inserted.
- Task IDs are navigation aids. When a task definition conflicts with a PR
  spec, stop and resolve the architecture and PR spec before coding.
- Raw CI output, benchmark logs, and review transcripts stay in their native
  systems. The tracker records concise evidence links or exact commands and
  results.

## Status model

Use exactly one of these values in `TRACKING.md`:

| Status | Meaning |
|---|---|
| `PLANNED` | Scope is approved; implementation has not started. |
| `LOCAL_READY` | Implementation exists locally and satisfies its local gates. |
| `IN_PROGRESS` | Implementation or required validation is active. |
| `IN_REVIEW` | A pull request is open and ready for reviewer attention. |
| `BLOCKED` | A named external dependency prevents useful progress. |
| `MERGED` | The PR is merged into its target branch and post-merge checks pass. |

## Series invariants

Every PR in this directory must satisfy all of the following:

1. It has one reviewable purpose and an explicit non-goal list.
2. The repository remains buildable and existing user paths remain valid after
   the PR is merged.
3. New behavior and its regression tests land together.
4. Dormant control-plane or data-plane pieces cannot authorize I/O before the
   PR that activates the corresponding path: PR-04 for Store-full `DE_READ`
   and PR-07 for partial `DE_READ`.
5. A revert restores the behavior described by the previous merged PR.
6. NPU-dependent claims are labelled separately from CPU-only unit-test
   evidence.
7. No new environment variable is introduced for this series; activation and
   policy remain in `kv_connector_extra_config`.

## Architecture baseline

The detailed architecture remains documented in
[2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md](2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md).
This directory is authoritative for delivery boundaries and PR progress; the
architecture document is authoritative for Stage 1 semantics. If they diverge,
pause implementation and resolve both documents in the same documentation
change.

Current verified checkout baseline:

- Branch: `dev/dualpath`
- Commit: `9a12de829ee5`
- Current implementation: PR-00 behavior-preserving foundation is
  `LOCAL_READY`; PR-01 through PR-07 are not implemented. No active
  cross-engine decision, Store read selection, or Forward/Reverse extension is
  implemented.

# DualPath Stage-2: Preemption Safety

Design for making delivered Prefill-side dual_path requests safe under vLLM
preemption: block pinning (L0) + fail-closed abort (L1) + epoch-based
re-execution (L2) — delivered as a single unit, no 2a/2b split.

- [2026-08-11-dual-path-stage2-preemption-safety-design.md](2026-08-11-dual-path-stage2-preemption-safety-design.md) —
  problem/failure-mode analysis, available engine primitives, the three-layer
  design, UT strategy, and the one-shot implementation/delivery plan.

Stage-1 context and the known-limitation note this supersedes:
`../dual-path-stage1-tasks/TASKS.md` §3.

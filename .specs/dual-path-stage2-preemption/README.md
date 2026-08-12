# DualPath Stage-2: Preemption Safety

Design for making delivered Prefill-side dual_path requests safe under vLLM
preemption: block pinning plus one stable logical Forward binding and local
source replay on both paths. `DE_READ` additionally renews only its Reverse
attempt for the current PE allocation; the cross-process retirement protocol is
reserved for uncertain receipt/activation, waiting-period abort, and control
failure cleanup. Delivered as a single unit, no 2a/2b split.

- [2026-08-11-dual-path-stage2-preemption-safety-design.md](2026-08-11-dual-path-stage2-preemption-safety-design.md) —
  problem/failure-mode analysis, logical Forward versus DE_READ Reverse-attempt
  identity, upstream worker-metadata reuse, TP all-worker barriers, stable
  Forward replay, exception-only non-blocking `RetireReverseAttempt`, explicit
  unsupported-topology rejection, test strategy, and the one-shot
  implementation/delivery plan.

Stage-1 context and the known-limitation note this supersedes:
`../dual-path-stage1-tasks/TASKS.md` §3.

# DualPath Stage-2: Preemption Safety

Design for making delivered Prefill-side dual_path requests safe under vLLM
preemption: a synchronous worker-side sender fence, one stable logical Forward
binding, local source replay on both paths, and attempt-keyed Reverse
completion. `DE_READ` renews only its Reverse attempt for the current PE
allocation. Delivered as a single unit, no 2a/2b split.

> **2026-08-14 revision:** The cross-process `CloseReverseAttempt` protocol and
> the Reverse destination hold are retired. An aborted or recovery-watchdog-
> expired PE request releases its destination immediately, matching the parent
> connector and main-branch risk level instead of waiting for a remote
> `SAFE` proof. The `reverse_attempt_id` epoch, `STALE_CLOSED`, I4 gate, and
> `JobLedger` completion aggregation remain active.

- [2026-08-11-dual-path-stage2-preemption-safety-design.md](2026-08-11-dual-path-stage2-preemption-safety-design.md) —
  problem/failure-mode analysis, logical Forward versus DE_READ Reverse-attempt
  identity, upstream worker-metadata reuse, `JobLedger` TP all-worker
  aggregation, synchronous fence-and-replay for the stable Forward range,
  job-based current-Reverse-attempt gate, the immediate-free abort semantics,
  the retired close/hold design and its rationale, explicit unsupported-
  topology rejection, and test strategy.

Implementation state: implemented in
`vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` per the design's §10
order — sender failure fixes and the FIFO barrier primitive; `ledgers.py` with
`JobLedger` only; attempt identity and the I4 gate; resume admission; the
attempt-aware Decision registry; fence-and-replay wiring; and cleanup,
watchdogs, and topology guards. The control endpoint accepts Decision messages
only; a legacy `CloseReverseAttempt` message is warned and dropped without a
reply, so mixed old/new deployments are unsupported. Unit coverage
lives in `tests/ut/distributed/kv_transfer/` (sender barrier) and
`tests/ut/distributed/kv_transfer/dual_path/` (all Stage-2 machinery). The §9
step-9 NPU end-to-end scenarios remain outstanding.

Stage-1 context: the known-limitation note this supersedes was removed from
`../dual-path-stage1-tasks/TASKS.md` §3.

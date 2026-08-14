# DualPath Stage-2: Preemption Safety

Current runtime contract for delivered Prefill-side dual_path requests under
vLLM preemption: Forward remains an ordinary Layerwise push with no DualPath
pin, gating job, fence, or delayed free; Reverse completion remains
attempt-keyed. `DE_READ` renews only its Reverse attempt for the current PE
allocation. Delivered as a single unit, no 2a/2b split.

> **2026-08-14 revision:** The cross-process `CloseReverseAttempt` protocol and
> the Reverse destination hold are retired. An aborted or recovery-watchdog-
> expired PE request releases its destination immediately, matching the parent
> connector and main-branch risk level instead of waiting for a remote
> `SAFE` proof. The `reverse_attempt_id` epoch, `STALE_CLOSED`, I4 gate, and
> `JobLedger` completion aggregation remain active.

- [2026-08-11-dual-path-stage2-preemption-safety-design.md](2026-08-11-dual-path-stage2-preemption-safety-design.md) —
  problem/failure-mode analysis, logical Forward versus DE_READ Reverse-attempt
  identity, upstream worker-metadata reuse, ordinary Forward Layerwise push,
  `JobLedger` TP all-worker aggregation for Reverse jobs, job-based current-
  Reverse-attempt gate, immediate-free Prefill finish/abort/preemption
  semantics, the retired close/hold/fence design and its rationale, explicit
  unsupported-topology rejection, and test strategy.

Implementation state: implemented in
`vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` per the design's §10
order — `ledgers.py` with `JobLedger` only; attempt identity and the I4 gate;
resume admission; the attempt-aware Decision registry; ordinary Forward
Layerwise push without connector gating; and cleanup, watchdogs, and topology
guards. The control endpoint accepts Decision messages only; a legacy
`CloseReverseAttempt` message is warned and dropped without a reply, so mixed
old/new deployments are unsupported. Unit coverage
lives in `tests/ut/distributed/kv_transfer/` (Layerwise sender machinery) and
`tests/ut/distributed/kv_transfer/dual_path/` (all Stage-2 machinery). The §9
step-9 NPU end-to-end scenarios remain outstanding.

Stage-1 context: the [known-limitation note](../dual-path-stage1-tasks/TASKS.md#known-limitations-stage-1)
now records the retained epoch/I4 recovery behavior, the retired hold/fence/
close design, and the accepted late-DMA risk of immediate-free semantics.

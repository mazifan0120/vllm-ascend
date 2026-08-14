# DualPath Stage-2: Preemption Safety

Design for making delivered Prefill-side dual_path requests safe under vLLM
preemption: block holds plus a synchronous worker-side sender fence, one stable
logical Forward binding, and local source replay on both paths. `DE_READ`
additionally renews only its Reverse attempt for the current PE allocation; the
cross-process `CloseReverseAttempt` protocol (`SAFE`/`NOT_SAFE`, fail closed) is
reserved for uncertain receipt/activation, waiting-period abort, and control
failure cleanup. Delivered as a single unit, no 2a/2b split.

- [2026-08-11-dual-path-stage2-preemption-safety-design.md](2026-08-11-dual-path-stage2-preemption-safety-design.md) —
  problem/failure-mode analysis, logical Forward versus DE_READ Reverse-attempt
  identity, upstream worker-metadata reuse, single hold ledger with TP
  all-worker job aggregation, synchronous fence-and-replay for the stable
  Forward range, job-based current-Reverse-attempt gate, abort dual-release,
  exception-only non-blocking `CloseReverseAttempt`, explicit
  unsupported-topology rejection, test strategy, and the one-shot
  implementation/delivery plan.

Implementation state: implemented in
`vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` per the design's §10
order — sender failure fixes and the FIFO barrier primitive; hold/job ledgers
with all-worker aggregation; attempt identity and the I4 gate; resume
admission; the attempt-aware channel registry; fence-and-replay wiring;
`CloseReverseAttempt`; and cleanup/watchdogs/hold-pressure limits/topology
guards (env-tunable per §11, defaults pending NPU measurement). Unit coverage
lives in `tests/ut/distributed/kv_transfer/` (sender barrier) and
`tests/ut/distributed/kv_transfer/dual_path/` (all Stage-2 machinery). The §9
step-9 NPU end-to-end scenarios remain outstanding.

Stage-1 context: the known-limitation note this supersedes was removed from
`../dual-path-stage1-tasks/TASKS.md` §3.

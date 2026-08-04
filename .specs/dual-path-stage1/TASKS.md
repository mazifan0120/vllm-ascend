# DualPathConnector Stage 1 Task Catalog

This file is the only source for Stage 1 implementation task definitions and
their PR ownership. The architecture spec owns semantics; each PR spec owns its
merge boundary; this catalog only decomposes that approved work into reviewable
implementation units.

PR-00 is already implemented and tracked as `LOCAL_READY`. Its completed work
is not renumbered into this catalog. The tasks below start with PR-01.

## PR-01 Decode Store Coverage Foundation

- `DP-COV-01` — Define `StoreCoverage`, HBM-versus-Store readiness, alignment,
  clamp, and coverage classification.
- `DP-COV-02` — Add a Scheduler-side Store probe adapter, extracting a minimal
  pure-lookup seam when needed, so lookup leaves no active `LoadSpec`, HBM load,
  or P2P side effect outside the owned handle.
- `DP-COV-03` — Add pending probe-handle ownership, abort, cancellation, and
  terminal cleanup without a Store commit path.
- `DP-COV-04` — Cover full, partial, miss, error, duplicate probe, alignment,
  and cleanup behavior with focused unit tests.

## PR-02 Parent Layerwise Worker Helper Extraction

- `DP-REF-01` — Extract protected send/receive startup, metadata binding, and
  per-layer enqueue helpers with behavior-parity regression tests.

## PR-03 Unified Path Decision Control Plane

- `DP-CTL-01` — Define request identity, coverage proposal, `PathKind`, decision
  commit, and decision error schemas.
- `DP-CTL-02` — Implement eligible-path calculation and the Prefill-owned
  round-robin policy. Single-path decisions do not advance the counter.
- `DP-CTL-03` — Add Proxy decision-Future registration, PE dispatch ordering,
  callback validation, and mixed ordinary/DualPath request isolation.
- `DP-CTL-04` — Add the Prefill Scheduler decision hook and PE
  `PathDecisionCoordinator` producer.
- `DP-CTL-05` — Add the Decode coordinator, decision inbox, and Scheduler-thread
  drain consumer.
- `DP-CTL-06` — Cover commit-once, timeout, PE failure, callback security,
  idempotency, and closed-loop control-plane behavior without data I/O.

## PR-04 Activate Store-Full DE_READ

- `DP-FULL-01` — Extend the probe handle with exactly-once
  `commit_after_alloc()` or `abort_probe()` terminal transitions.
- `DP-FULL-02` — Compose the existing bulk `KVPoolWorker` on Decode, bind
  committed Store metadata to final HBM blocks, and expose the minimal typed
  Store completion/invalid-block seam required by the consumer.
- `DP-FULL-03` — Build the Store-full `DE_READ` Worker plan and request-level
  Store DONE/FAILED handling; a generic finished ID alone is not success.
- `DP-FULL-04` — Activate Prefill round-robin for Store-full candidates,
  including `PE_READ` fall-through and zero-compute PE request completion for
  `DE_READ`, while preventing the non-winning PE Store sibling from starting
  load I/O.
- `DP-FULL-05` — Add CPU integration and NPU E2E for alternating full-coverage
  `PE_READ`/`DE_READ`, Store failure, and absence of Reverse/Forward on the
  committed `DE_READ` path.

## PR-05 Explicit Forward Data Path

- `DP-FWD-01` — Define immutable `BlockPair`, Forward direction plans, bindings,
  and validation.
- `DP-FWD-02` — Convert frozen Forward plans into parent `ReqMeta`/`SendTask`
  structures without re-deriving physical blocks.
- `DP-FWD-03` — Add the monotonic Forward frontier driven by the inherited
  per-layer save callback.
- `DP-FWD-04` — Add Forward raw-completion retention, binding, duplicate
  handling, failure provenance, and request-level terminal publication.
- `DP-FWD-05` — Cover PE Store hit, PE Store miss plus compute, ordinary PE
  compute, and NPU Forward parity.

## PR-06 Bidirectional Runtime and Lifecycle

- `DP-BIDI-01` — Enable one inherited Worker runtime to own one send and one
  receive capability with one KV-buffer registration.
- `DP-BIDI-02` — Convert and dispatch Reverse plans in registered layer order.
- `DP-BIDI-03` — Enforce mapping-first batch ordering plus Store DONE, Reverse
  DONE, and Forward start gates.
- `DP-BIDI-04` — Add cross-direction completion reconciliation, timeout,
  cancellation, tombstones, block retention, and drain-first cleanup.
- `DP-BIDI-05` — Execute injected partial plans in success, failure, race, and
  cancellation tests while production partial selection remains unreachable.

## PR-07 Activate Partial DE_READ and Complete Stage 1

- `DP-ACT-01` — Make partial Store coverage eligible for `DE_READ` and connect
  committed decisions to first-positive Scheduler accounting.
- `DP-ACT-02` — Build exact Store, Reverse, PE compute, and Forward ranges and
  their frozen physical block plans.
- `DP-ACT-03` — Integrate MultiConnector sibling ordering and prohibit
  non-winning children from starting I/O.
- `DP-ACT-04` — Complete failure, cancellation, invalid-block, observability,
  timeout, and no-fallback behavior for active partial requests.
- `DP-ACT-05` — Run the complete NPU E2E, performance, HBM, topology, and
  supported-configuration acceptance matrix.

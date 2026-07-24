# DualPathConnector Option A Detailed Design Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the complete Option A Stage 1 detailed design so every chapter uses the agreed lightweight path-decision protocol, one shared Mooncake worker runtime, two logical transfer directions, and request-level completion semantics.

**Architecture:** Preserve the outer `AscendMultiConnector([DualPathConnector, AscendStoreConnector])` topology on PE and direct `DualPathConnector` topology on DE. Keep Store probe/allocation/commit/data-start as call ordering rather than a distributed state machine. Reuse one inherited `MooncakeLayerwiseConnectorWorker` send/recv runtime and existing framework completion channels instead of adding custom per-layer ACK, event-bus, fence, ownership, or quarantine subsystems.

**Tech Stack:** Markdown, Mermaid, Python interface sketches, vLLM v1 KV connector contracts, vLLM Ascend `AscendMultiConnector`, `MooncakeLayerwiseConnector`, and AscendStore `KVPoolScheduler`/`KVPoolWorker`.

## Global Constraints

- Modify `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md`; do not modify implementation code.
- Preserve the document status as an unimplemented design review draft.
- `PathDecisionRequest` contains only `request_key` and `de_store_coverage`.
- `DualPathKind` contains only `DE_FULL_HIT` and `DE_PARTIAL_HIT`.
- PE is the sole final decision committer.
- Every result uses the same Proxy model-request envelope; full hit is consumed at PE ingress before Scheduler/model execution.
- Path-decision control stays on Proxy/RPC and never reuses Worker ZMQ.
- No `decision_version`, transfer epoch, business ACK, application retry protocol, or explicit lifecycle enum.
- PE and DE each use one `DualPathConnectorWorker` with one inherited Mooncake send thread and one receive thread.
- Forward and Reverse are logical directions over that shared runtime, with distinct wire IDs and role-local receiver endpoints.
- Mooncake remains layerwise for data transfer and request-level for final DONE/FAILED completion.
- Preserve pending raw completion buffering when terminal notification arrives before local request mapping.
- Internal Store save is disabled in Stage 1; PE Store load remains owned by the outer `AscendStoreConnector`.
- Do not create git commits as part of this document refresh.

---

### Task 1: Normalize Chapters 1–8 Around the Approved Protocol

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md:1`

**Interfaces:**
- Consumes: Approved `DualPathRequestKey`, `StoreCoverage`, `DualPathKind`, `PathDecisionRequest`, and `PathDecisionCommit`.
- Produces: Canonical terminology and invariants consumed by Chapters 9–26.

- [ ] **Step 1: Re-read Chapters 1–8 and remove remaining terminology that treats outer AscendStore selection as a DualPath path.**

- [ ] **Step 2: Ensure the ID section reuses the DE request identity, derives direction-specific wire IDs, and defines no route ID or transfer epoch.**

- [ ] **Step 3: Ensure the decision section describes one Proxy request envelope, one PE commit, no ACK/retry state machine, and no explicit lifecycle enum.**

- [ ] **Step 4: Ensure data start is guarded only by local commit, local plan readiness, and non-terminal request status.**

- [ ] **Step 5: Check Chapters 1–8 contain no types or fields removed by the approved review.**

### Task 2: Replace Chapters 9–10 With the Shared-Worker Runtime Design

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md:751`

**Interfaces:**
- Consumes: Existing `DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker)`, inherited send/recv threads, `PathDecisionCommit`, and AscendStore load metadata.
- Produces: Minimal Connector/Scheduler/Worker APIs and request-level transfer plans.

- [ ] **Step 1: Redraw the class diagram without `SharedMooncakeTransferRuntime`, `TransferFenceRegistry`, `SchedulerReleaseLedger`, or `BlockOwnershipLedger`.**

- [ ] **Step 2: Define `DualPathConnectorWorker` as the sole owner of the inherited Mooncake runtime; retain Forward/Reverse only as lightweight private direction helpers if needed.**

- [ ] **Step 3: Reduce facade, Scheduler, and Worker interfaces to methods actually required by vLLM connector hooks and the agreed protocol.**

- [ ] **Step 4: Restrict the internal Store adapter to DE probe/load and remove Stage 1 Store-save metadata/event responsibilities.**

- [ ] **Step 5: Replace region/command/event abstractions with a minimal request plan containing commit, token boundaries, existing block mappings, direction wire IDs, and optional Store metadata.**

- [ ] **Step 6: Define only request-level Store, Reverse, Forward, and failure completion facts; keep pending raw DONE/FAILED buffering.**

### Task 3: Replace Chapters 11–13 With Request-Level Dependency and Completion Rules

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md:1500`

**Interfaces:**
- Consumes: Existing Mooncake layerwise send ordering, request-level DONE/FAILED notification, Store `loading_req_ids`, and framework `finished_recving`/`finished_sending`.
- Produces: Minimal dependencies, success predicates, failure publication, and block-lifetime rules.

- [ ] **Step 1: Replace per-layer remote fence events with `STORE_DONE -> Reverse layer writes -> REVERSE_DONE -> PE model -> Forward layer writes -> FORWARD_DONE`.**

- [ ] **Step 2: Replace `BlockOwnershipLedger` with explicit write-range ordering and request-owned block lifetime rules.**

- [ ] **Step 3: Retain the three DE success predicates and define PE Reverse completion separately.**

- [ ] **Step 4: Remove `AtomicRequestOutcome`, custom rank fences, custom worker-metadata aggregation, and `SchedulerReleaseLedger`.**

- [ ] **Step 5: Map request-level completion to framework `finished_recving`; reserve `finished_sending` for the existing delayed-free contract.**

- [ ] **Step 6: State that failed block IDs are published before or with the failed receive terminal, without introducing quarantine state.**

### Task 4: Align Chapters 14–20 With the Simplified Runtime and Control Plane

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md:1860`

**Interfaces:**
- Consumes: Canonical request/commit protocol, first-positive accounting, shared-worker directions, and request-level completion.
- Produces: End-to-end sequences, configuration contract, startup behavior, failures, and idempotency.

- [ ] **Step 1: Update all three path sequences to show layerwise data writes and request-level final completion only.**

- [ ] **Step 2: Split PE decision formation from matched-token accounting so the commit is not an input to the function that creates it.**

- [ ] **Step 3: Remove references to atomic sibling outcomes and custom fences from the A″ first-winner section.**

- [ ] **Step 4: Rewrite configuration around role, static partial policy, role-local Mooncake receive endpoint, DE Store settings, and Proxy/RPC integration; remove Worker-ZMQ decision configuration.**

- [ ] **Step 5: Rewrite initialization to construct one inherited worker runtime, register buffers once, start one send/recv pair, and reuse lazy Mooncake GET_META behavior.**

- [ ] **Step 6: Keep the Stage 1 topology restrictions but remove justifications based on deleted per-layer fences.**

- [ ] **Step 7: Simplify failure and concurrency rules to request-level maps/sets, commit-once, start-once, completion-once, and framework drain behavior.**

### Task 5: Synchronize Chapters 21–26 With the Implementable Design

**Files:**
- Modify: `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md:2411`

**Interfaces:**
- Consumes: Final architecture and protocol from Chapters 1–20.
- Produces: File layout, implementation gates, tests, observability, acceptance criteria, and development order.

- [ ] **Step 1: Remove `transfer_fence.py` and `block_ownership.py` from the proposed layout and mark existing `connector.py`/`config.py` as files to update.**

- [ ] **Step 2: Add implementation gates for Proxy commit return, full-hit ingress short-circuit, Store probe cleanup, and single-worker bidirectional reuse.**

- [ ] **Step 3: Remove tests for per-layer ACK, custom event bus, quarantine, ownership ledger, and worker-metadata release.**

- [ ] **Step 4: Add tests for one runtime, request-level DONE/FAILED, pending raw completion, first-positive accounting, Store probe commit/abort, and unified Proxy dispatch.**

- [ ] **Step 5: Reduce observability to request-level identity, decision, coverage, selected outer connector, direction completion, latency, and failure fields.**

- [ ] **Step 6: Rewrite acceptance criteria around one worker runtime, two logical directions, request-level completion, and no Worker-ZMQ decision path.**

- [ ] **Step 7: Reorder development to validate Proxy ingress/return first, then configuration/decision, Store adapter, shared-worker dispatch, completion mapping, Scheduler integration, and tests.**

### Task 6: Full-Document Consistency Validation

**Files:**
- Verify: `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md`

**Interfaces:**
- Consumes: Updated Chapters 1–26.
- Produces: A structurally valid and internally consistent design document.

- [ ] **Step 1: Search for removed terms and fields.**

Run:

```bash
rg -n "PE_READ|DE_PARTIAL_READ|CoverageProposal|DecisionAck|decision_version|DecisionPhase|PathKind|transfer_epoch|RankArmAck|TransferFenceRegistry|SchedulerReleaseLedger|BlockOwnershipLedger|AtomicRequestOutcome|REVERSE_LAYER_DONE|FORWARD_LAYER_DONE" docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
```

Expected: no active design definitions or requirements using these names.

- [ ] **Step 2: Verify required canonical types and flows are present.**

Run:

```bash
rg -n "DualPathKind|PathDecisionRequest|PathDecisionCommit|DE_FULL_HIT|DE_PARTIAL_HIT|pending_raw|Proxy/RPC|finished_recving" docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
```

Expected: every canonical concept appears in its defining and downstream sections.

- [ ] **Step 3: Verify Markdown structure and whitespace.**

Run:

```bash
git diff --check -- docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
awk '/^```/{f++} /^## [0-9]+\./{h++} END{print "fence_lines=" f, "numbered_h2=" h; if (f % 2 || h != 26) exit 1}' docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
```

Expected: `git diff --check` is silent, fence count is even, and `numbered_h2=26`.

- [ ] **Step 4: Review the final diff to ensure only the intended design and plan files changed during this task.**

Run:

```bash
git status --short
git diff -- docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md docs/superpowers/plans/2026-07-24-dual-path-option-a-design-refresh.md
```

Expected: unrelated pre-existing workspace changes remain untouched.

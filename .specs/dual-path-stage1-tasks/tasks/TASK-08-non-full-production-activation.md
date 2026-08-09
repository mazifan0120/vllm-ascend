# Task-08 Detailed Spec — Non-full Production Activation

## 1. Status, authority, and implementation starting point

This document is the complete implementation contract for Task-08 in the
[`DualPath Stage 1 Task Catalog`](../TASKS.md). It is intentionally
self-contained: an implementation Agent may read this file without reading
the Task-01 through Task-07 detailed specs.

The source-audit baseline used while writing this contract is:

```text
vllm-ascend@aa32e8542f8d76f28b51f6e03fc3944a8fed406e
```

At that baseline, Task-01 through Task-06 behavior is present, while Task-07
is represented by its approved detailed spec. Task-08 implementation must
begin only from a merge state that satisfies
[`TASK-07-bidirectional-split-runtime.md`](TASK-07-bidirectional-split-runtime.md).
The implementing Agent must inspect the live dependency merge rather than
assuming that its file layout is byte-identical to the audit baseline.

Where this document conflicts with an earlier Task spec, this document wins
for the final Stage 1 production composition. In particular, it intentionally
revises:

- Task-02 eligibility and decision-record facts;
- Task-03 `PathDecision` wire schema and queued value;
- Task-04 PE delivery timing and control-only failure metadata;
- Task-05 production `PE_READ` delivery timing;
- Task-06 snapshot retention and the Store commit precondition; and
- Task-07's production activation boundary.

It does not change the Task-06 rule that Store-full completes locally on DE
without Proxy or PE involvement, and it does not change Task-07's Worker
transfer implementation.

## 2. Outcome and final Stage 1 merge-state contract

Task-08 connects every Store-non-full production request to exactly one
previously tested route:

```text
Path.PE_READ
or
Path.DE_READ
```

The final route selection is eligibility-before-policy:

```text
L_PE >= K_DE
-> PE_READ is the only eligible path
-> PathPolicy is not called

L_PE < K_DE
-> PE_READ and DE_READ are both eligible
-> PathPolicy.choose() selects exactly once
```

A production `DE_READ` therefore always has a non-empty Reverse interval.
Task-07's empty-Reverse runtime remains a defensive injected capability but
is not reachable through the production Task-08 selector or wire protocol.

After acceptance:

- PE chooses and freezes the path during its first Scheduler lookup;
- PE sends no `PathDecision` until its final allocation is bound;
- one protocol-v2 `PathDecision` atomically carries the result and the
  route-required `ReversePlan` when the path is `DE_READ`;
- `PE_READ` continues through the existing PE AscendStore-or-compute path and
  inherited Forward runtime;
- `DE_READ` commits the frozen Decode Store snapshot only after the valid
  Decision is accepted, then runs Store, Reverse, PE compute, and Forward;
- Store miss or unavailable Store remains eligible for `DE_READ` when Decode
  HBM extends beyond PE HBM;
- no route starts both Decode Store and PE Store;
- no accepted route is reconsidered, refreshed, or changed after commit;
- all pre-data activation failures and all data-plane failures terminate
  through exact invalid-block ownership plus one local terminal; and
- HBM-complete, Store-full, ordinary Layerwise, and prior `PE_READ` behavior
  remain compatible.

## 3. Terminology and frozen facts

For one logical prompt request:

```text
P    = original prompt token count
R    = max(P - 1, 0), Decode-ready prefix
T    = parent Layerwise transfer target
       P for ordinary Attention
       R for Attention-Mamba hybrid
L_DE = contiguous Decode HBM prefix at DE admission
K_DE = usable, aligned Decode Store prefix, clamped to R
L_PE = contiguous Prefill HBM prefix at PE admission
K_PE = PE AscendStore prefix, when PE_READ later queries it
```

`K_DE` is frozen in `DecodeKVSnapshot`. It is not a live reference to current
Store coverage. A later PE Store lookup may observe a different `K_PE`, but
that fact belongs only to PE local `PE_READ` preparation and never expands the
already-frozen Decode Store interval.

The Stage 1 invariants are:

```text
0 <= L_DE <= K_DE <= R <= T
0 <= L_PE <= T
```

The routes use these logical ranges:

```text
PE_READ:
    PE Store/compute preparation: existing PE connector semantics
    Forward:                     [L_DE, T)

DE_READ:
    Decode Store:               [L_DE, K_DE), possibly empty
    Reverse:                    [L_PE, K_DE), always non-empty in production
    PE compute:                 [K_DE, T)
    Forward:                    [K_DE, T)
```

Complete physical blocks may contain boundary tokens outside a logical
interval. Logical endpoints remain authoritative for validation, accounting,
completion, and invalid-block selection; Worker execution uses the frozen
complete-block mappings produced by Task-07.

## 4. Final route matrix

The production matrix is normative:

| Condition | Route | Required behavior |
|---|---|---|
| `L_DE >= R` | HBM-complete | No Store probe, Decision, Proxy request, PE request, or P2P |
| `K_DE == R` | DE-local Store-full | Task-06 local Store commit and completion; no PE work |
| `K_DE < R` and `L_PE >= K_DE` | Forced `PE_READ` | Policy is not called; PE may use AscendStore and then Forward `[L_DE,T)` |
| `L_PE < K_DE < R`, policy selects `PE_READ` | `PE_READ` | No Decode Store commit or Reverse; PE prepares and Forwards `[L_DE,T)` |
| `L_DE < K_DE < R`, policy selects `DE_READ` | Partial split | Store `[L_DE,K_DE)`, Reverse `[L_PE,K_DE)`, compute/Forward from `K_DE` |
| `K_DE == L_DE`, `L_PE < L_DE`, policy selects `DE_READ` | Miss split | Store `SKIPPED`, Reverse Decode HBM `[L_PE,L_DE)`, compute/Forward from `L_DE` |

If `K_DE == L_DE == 0`, the eligibility gate forces `PE_READ` because
`L_PE >= K_DE` for every valid `L_PE`. No zero-length Reverse or Store task is
created.

## 5. Required deltas to earlier Tasks

An implementation Agent reading only this spec must apply every delta in this
section.

### 5.1 Task-01 delta — retained allocation wrapper

`DecodeKVSnapshot.final_block_ids` remains the immutable authority for all
plan and failure calculations. Task-08 additionally retains the original
Scheduler allocation wrapper solely so a later committed `DE_READ` can call
the existing KVPool Scheduler API:

```python
@dataclass(frozen=True)
class DecodeKVSnapshot:
    transfer_tokens: int
    local_tokens: int
    external_tokens: int
    store_load_spec: LoadSpec | None
    final_block_ids: BlockTable
    allocated_blocks: KVCacheBlocks = field(compare=False, repr=False)
```

`allocated_blocks`:

- is not serialized or sent to another process;
- is not used for dataclass equality or duplicate admission checks;
- does not own allocation or block release;
- remains valid only while the request owns the waiting allocation;
- must match `final_block_ids` again immediately before Store commit; and
- is released when the snapshot is removed on request terminal or shutdown.

### 5.2 Task-02 delta — eligibility before policy

`PathPolicy` remains lightweight and unchanged:

```python
class PathPolicy(Protocol):
    def choose(self, request: PathDecisionRequest) -> Path: ...
```

`PathDecisionDecider` additionally receives the current `L_PE` and owns the
eligibility gate before calling the configured policy. The exact method name
may follow local style, but its semantic input is:

```python
decide(
    request: PathDecisionRequest,
    prefill_local_tokens: int,
) -> PathDecisionResult
```

Its retained record becomes equivalent to:

```python
@dataclass(frozen=True)
class _DecisionRecord:
    request: PathDecisionRequest
    prefill_local_tokens: int
    result: PathDecisionResult | None
```

Rules:

1. Validate `0 <= L_PE <= T` against the inherited effective PE request and
   hybrid truncation rules. The stricter `L_PE < K_DE < R <= T` condition is
   required only when the selected path is `DE_READ`.
2. If `L_PE >= request.decode_store_tokens`, return `Path.PE_READ` without
   calling `PathPolicy.choose()`.
3. Otherwise call `PathPolicy.choose(request)` exactly once.
4. An identical replay returns the retained result and does not advance the
   Round Robin.
5. The same request key with different Decision Request facts or a different
   frozen `L_PE` is a conflict; retain the first record and fail locally.
6. A policy exception or invalid return records local failure, sends no
   Result, and lets DE converge through the Decision timeout.

There is no `PathCapabilities`, eligible-path collection, metrics provider,
or policy-side Store probe.

### 5.3 Task-03 delta — protocol v2 and complete queued Decision

Task-08 changes the exact `PathDecision` payload, so the shared protocol
constant becomes:

```python
DUAL_PATH_PROTOCOL_VERSION = 2
```

The same version gates both:

- `DualPathDecisionMetadata` carried DE -> Proxy -> PE; and
- the direct PE -> DE `PathDecision`.

Mixed protocol-v1/v2 deployments are unsupported. Both engines must be
coordinated on the Task-08 implementation. Store-full local completion does
not use this protocol.

The message is:

```python
@dataclass(frozen=True)
class PathDecision:
    protocol_version: int
    result: PathDecisionResult
    reverse_plan: ReversePlan | None
```

Exact path rules:

```text
PE_READ -> reverse_plan is None
DE_READ -> reverse_plan is present and non-empty
```

No standalone `prefill_local_tokens` wire field is added. For `DE_READ`,
`ReversePlan.token_start` already carries `L_PE`; for `PE_READ`, DE has no
data-plane use for `L_PE`.

The Decode Coordinator retains and queues the complete message, not only its
result:

```python
_accepted_decisions: dict[DualPathRequestKey, PathDecision]
_received_decisions: SimpleQueue[PathDecision]

def take_received_decisions(self) -> list[PathDecision]: ...
```

The receiver thread remains transport-only: receive, decode, validate the
protocol/key/duplicate shape, ACK, and enqueue. It must not mutate Scheduler
state or start Store/P2P work.

`ReversePlan` gains exact `to_dict()`/`from_dict()` support with deep-freezing
and the Task-07 validation contract. Reuse the Task-07 entity; do not create a
second wire-only split-plan type.

### 5.4 Task-04 delta — post-allocation delivery and generic local failure

PE no longer submits a `PathDecision` from its initial Scheduler lookup. It
only freezes the result there. Submission occurs once after PE
`update_state_after_alloc()` has successfully installed every required local
plan/binding.

The Task-04 Decision deadline on DE is unchanged. It still begins immediately
before asynchronous Proxy submission and covers Proxy transport, PE queueing,
PE allocation, and direct delivery.

Task-04's timeout-only metadata is generalized to an engine-local control
failure record equivalent to:

```python
class DualPathControlFailureReason(str, Enum):
    DECISION_TIMEOUT = "DECISION_TIMEOUT"
    ACTIVATION_FAILED = "ACTIVATION_FAILED"


@dataclass(frozen=True)
class DualPathControlFailureMetadata:
    request_id: str
    invalid_block_ids: tuple[int, ...]
    reason: DualPathControlFailureReason
```

`DualPathConnectorMetadata` carries `control_failures`, replacing the
timeout-specific list in the final composition. The Worker behavior remains
control-only:

```text
install no request map
start no Store/P2P/DMA operation
publish invalid_block_ids + finished_recving together
```

This is not a network `PathDecisionError`. PE still sends no error message.

### 5.5 Task-05 delta — production PE_READ timing only

Task-05's `ForwardPlan`, `ForwardReceiveBinding`, inherited send runtime,
early-terminal preservation, and terminal semantics remain unchanged.

Task-08 changes only production assembly:

- PE constructs/retains the `PE_READ` Forward plan after allocation;
- PE sends the no-Reverse `PathDecision` only after that succeeds; and
- DE consumes the Decision into the existing `PE_READ`
  `ForwardReceiveBinding` path.

### 5.6 Task-06 delta — one full/partial Store commit API

The same `KVPoolAdapter.commit_after_alloc()` supports Store-full and partial
Store coverage. Do not add `commit_partial_after_alloc()`.

The final precondition is:

```text
0 <= load_spec.vllm_cached_tokens
  < load_spec.kvpool_cached_tokens
  <= R
```

The delegated delta is always:

```python
store_delta = (
    load_spec.kvpool_cached_tokens
    - load_spec.vllm_cached_tokens
)
```

Task-06 Store-full still commits immediately during the initial DE
`update_state_after_alloc()` and creates no Decision state or PE request.
Task-08 partial Store commits later, only after a valid `DE_READ` Decision.

### 5.7 Task-07 delta — production activation

Task-07's Worker runtime, `ReversePlan`, `ReverseReceiveBinding`, split
tracker, runtime gates, mapping-first installation, early terminal retention,
failure provenance, and parent helper reuse remain unchanged.

Task-08 supplies their first production Scheduler producer. A real
`Path.DE_READ` now emits:

```text
PE metadata:
    ReverseReceiveBinding
    ForwardPlan retained for later PE execution

PE -> DE PathDecision:
    PathDecisionResult(Path.DE_READ)
    ReversePlan

DE metadata:
    ForwardReceiveBinding(path=DE_READ)
    ReversePlan
    optional Decode Store metadata
```

Task-07 empty-Reverse injection remains valid, but Task-08 must reject a
production `PathDecision(Path.DE_READ, reverse_plan=None)`.

## 6. End-to-end production sequence

```mermaid
sequenceDiagram
    autonumber

    participant DECore as "DE vLLM Core"
    participant DES as "DE DualPath Scheduler"
    participant Proxy
    participant PECore as "PE vLLM Core"
    participant PES as "PE DualPath Scheduler"
    participant Channel as "PE to DE Decision Channel"
    participant DEW as "DE DualPath Worker"
    participant Store as "DE KVPool Worker"
    participant PEW as "PE DualPath Worker"

    DECore->>DES: get_num_new_matched_tokens(L_DE)
    DES->>DES: probe Store and freeze K_DE
    DES-->>DECore: T - L_DE, True
    DECore->>DECore: allocate final DE slots
    DECore->>DES: update_state_after_alloc(DE blocks)
    DES->>DES: bind DecodeKVSnapshot and retained blocks
    DES->>Proxy: asynchronous Decision Request plus DE metadata
    DES-->>DECore: return
    DECore->>DECore: WAITING_FOR_REMOTE_KVS

    Proxy->>PECore: dispatch real Prefill request
    PECore->>PES: get_num_new_matched_tokens(L_PE)

    alt "L_PE >= K_DE"
        PES->>PES: force and freeze PE_READ; do not call policy
        PES-->>PECore: 0, False
        PECore->>PECore: optional PE AscendStore lookup/accounting
    else "L_PE < K_DE"
        PES->>PES: call PathPolicy exactly once
        alt "policy selects PE_READ"
            PES-->>PECore: 0, False
            PECore->>PECore: optional PE AscendStore lookup/accounting
        else "policy selects DE_READ"
            PES-->>PECore: K_DE - L_PE, True
        end
    end

    PECore->>PECore: allocate final PE slots
    PECore->>PES: update_state_after_alloc(PE blocks)

    alt "PE_READ"
        PES->>PES: install ForwardPlan [L_DE,T)
        PES->>Channel: PathDecision(PE_READ, no ReversePlan)
    else "DE_READ"
        PES->>PES: install ReverseReceiveBinding
        PES->>PES: install ForwardPlan [K_DE,T)
        PES->>PES: freeze ReversePlan [L_PE,K_DE)
        PES->>Channel: PathDecision(DE_READ, ReversePlan)
    end

    Channel->>DES: direct protocol-v2 delivery and ACK
    DES->>DES: next build_connector_meta takes complete Decision

    alt "PE_READ"
        DES->>DEW: ForwardReceiveBinding [L_DE,T)
        PEW->>DEW: inherited layerwise Forward
    else "DE_READ, partial Store"
        DES->>DES: validate snapshot and commit [L_DE,K_DE)
        DES->>DEW: Store metadata plus ReversePlan plus Forward binding
        DEW->>Store: async Store load
        Store-->>DEW: STORE_DONE
        DEW->>PEW: request-level Reverse [L_PE,K_DE)
        PEW-->>PECore: Reverse DONE / finished_recving
        PECore->>PECore: compute [K_DE,T)
        PEW->>DEW: inherited layerwise Forward [K_DE,T)
    else "DE_READ, Store miss"
        DES->>DEW: Store SKIPPED plus ReversePlan plus Forward binding
        DEW->>PEW: Reverse Decode HBM [L_PE,L_DE)
        PEW-->>PECore: Reverse DONE / finished_recving
        PECore->>PECore: compute [L_DE,T)
        PEW->>DEW: inherited layerwise Forward [L_DE,T)
    end

    DEW-->>DECore: finished_recving or exact invalid blocks plus terminal
```

## 7. PE Scheduler accounting and MultiConnector composition

For a valid DualPath Prefill request, the PE Scheduler retains the incoming
`num_computed_tokens` as `L_PE` and applies Section 5.2 before returning to
`AscendMultiConnector`.

The return contract is exact:

```text
PE_READ:
    return (0, False)

DE_READ:
    assert L_PE < K_DE
    return (K_DE - L_PE, True)
```

Consequences:

- `PE_READ` contributes no first-positive match, so the later PE
  AscendStore connector may load a longer PE prefix or all connectors may
  return zero;
- `DE_READ` contributes a strictly positive Reverse budget and becomes the
  first-positive winner, so the PE AscendStore sibling's result cannot affect
  accounting for this request; and
- there is no zero-token DualPath winner and no new MultiConnector
  terminal/sibling-stop seam.

If PE AscendStore later observes `K_PE > L_PE` on `PE_READ`, upstream
accounting naturally advances PE readiness to `K_PE`. This does not refresh
`K_DE`, alter the Decode Store interval, or create a mixed route.

`AscendMultiConnector.update_state_after_alloc()` already forwards real
blocks to a Layerwise child even when another connector is the accounting
winner. Task-08 relies on that existing behavior so DualPath can install the
`PE_READ` Forward plan after PE AscendStore wins. Do not modify
`AscendMultiConnector` for Task-08.

## 8. PE post-allocation plan construction

PE `update_state_after_alloc()` is the only production Decision-send point.
It must be non-blocking and perform no Store, P2P, or Worker operation.

### 8.1 Shared validation

Before installing either route, validate and freeze:

- protocol version 2;
- Decision Request key and retained result key equality;
- effective prompt/transfer target `T`, including inherited hybrid
  truncation;
- PE and DE block-size group counts and positive sizes;
- inherited remote engine/host/port/topology fields;
- exact advertised DE destination table and sufficient coverage;
- PE final source/destination block coverage; and
- no conflicting existing plan, binding, request key, or wire identity.

Identical duplicate allocation binding is idempotent. A conflict preserves
the first owner and fails activation locally.

### 8.2 PE_READ installation

For `PE_READ`:

```text
ForwardPlan.token_start = L_DE
ForwardPlan.token_end   = T
PathDecision.reverse_plan = None
```

Reuse Task-05 `SendReqInfo` adaptation and inherited per-layer Forward. PE
Store coverage and PE compute frontier do not change the Forward logical
start; DE still lacks every token after `L_DE`.

### 8.3 DE_READ installation

For `DE_READ`, production validation additionally requires:

```text
L_PE < K_DE < R <= T
```

PE constructs all three local/cross-engine facts from one allocation:

```text
ReverseReceiveBinding.destination = PE final block table
ReverseReceiveBinding.range       = [L_PE,K_DE)

ForwardPlan.source                = PE final block table
ForwardPlan.destination           = advertised DE final block table
ForwardPlan.range                 = [K_DE,T)

ReversePlan.source                = advertised DE final block table
ReversePlan.destination           = PE final block table
ReversePlan.range                 = [L_PE,K_DE)
```

`ReversePlan` also freezes the inherited PE peer endpoint, block-size, and
topology facts required by the DE sender.

### 8.4 Install-before-submit ordering

PE must complete this local order:

```text
validate all route facts
-> retain result and immutable plans
-> stage local ReverseReceiveBinding when required
-> stage local ForwardPlan
-> construct the complete PathDecision
-> submit it asynchronously exactly once
```

If any earlier step fails, send no Decision. A `DE_READ` PE request that was
placed in `WAITING_FOR_REMOTE_KVS` must fail its local Reverse destination
through engine-local control failure rather than wait forever. DE independently
converges through its Decision timeout. A `PE_READ` local plan failure sends
no Decision; DE times out and no Forward is authorized.

Task-08 adds no binding-ready ACK. Task-07 early-terminal retention continues
to cover cross-process arrival before Worker binding.

## 9. Protocol-v2 delivery and Decode acceptance

The existing bounded direct-channel behavior remains:

- one asynchronous PE sender Future;
- fresh REQ socket per retry attempt;
- send and receive timeout protection;
- at most three attempts with the existing fixed spacing;
- exact `ACK` validation;
- one DE ROUTER receiver thread; and
- no end-to-end resend, exponential backoff, or Decision Result reply.

The receiver validates message-local facts before accepting:

- exact top-level fields and version 2;
- pending request key and Decode incarnation;
- `PE_READ` has no Reverse plan;
- `DE_READ` has a Reverse plan whose key matches the Result;
- immutable Reverse fields pass standalone validation; and
- an identical duplicate is idempotent while a conflicting duplicate is
  rejected.

Snapshot-relative validation remains on the DE Scheduler thread because the
receiver does not own Scheduler state.

## 10. Decode Scheduler activation

`DualPathConnectorScheduler.build_connector_meta()` is the Scheduler-owned
safe point for consuming complete Decisions. Use the actual Coordinator
method; do not introduce a busy loop, callback into Scheduler state, or an
informal second processing thread.

The order for each Decision is normative:

```text
take complete PathDecision
-> find PENDING Decode state and snapshot
-> validate Result and optional ReversePlan against frozen state
-> prepare every Scheduler-owned metadata record
-> commit partial Store when required
-> mark COMMITTED only after activation succeeds
```

### 10.1 PE_READ activation

For `PE_READ`:

- reject a non-`None` Reverse plan;
- do not commit the detached Decode LoadSpec;
- create the existing `ForwardReceiveBinding(path=PE_READ)` for
  `[L_DE,T)`; and
- mark the Decision state committed only after the binding is retained.

### 10.2 DE_READ activation

For `DE_READ`:

1. Require a non-empty Reverse plan.
2. Validate `plan.token_end == K_DE` and `plan.token_start < K_DE`.
3. Validate source DE block IDs equal the snapshot's frozen final DE table.
4. Validate destination PE table, group counts, endpoint, block sizes, and
   topology facts.
5. Validate the retained `KVCacheBlocks.get_block_ids()` still freezes to
   `snapshot.final_block_ids`.
6. Create `ForwardReceiveBinding(path=DE_READ)` for `[K_DE,T)`.
7. If the snapshot owns a non-empty LoadSpec, call the unified
   `KVPoolAdapter.commit_after_alloc()` exactly once.
8. If the snapshot has no LoadSpec, represent Store as `SKIPPED` and perform
   no KVPool call.
9. Emit the Reverse plan, Forward binding, and optional Store metadata in the
   same Connector metadata lifecycle.
10. Mark committed only after all Scheduler-local steps succeed.

The Decode Scheduler must process Decisions before it asks
`KVPoolAdapter.build_connector_meta()` for the current output. That ordering
allows a newly committed partial Store load to appear in the same metadata
object without another control-only delay.

## 11. Partial KVPool commit contract

The adapter inserts a copy of the detached `LoadSpec`; the immutable snapshot
object remains unchanged. Before delegation validate:

- Decode role is consumer or both;
- `consumer_is_to_load=True` is configured;
- the private Store Scheduler uses `use_layerwise=False`;
- request ownership is not already committed;
- the block wrapper still matches `final_block_ids`; and
- `0 <= L_DE < K_DE <= R`.

Then delegate:

```python
pool.update_state_after_alloc(
    request,
    snapshot.allocated_blocks,
    K_DE - L_DE,
)
```

Although the final block table covers through `T`, existing non-layerwise
KVPool metadata derives its async load target from
`load_spec.kvpool_cached_tokens`. The Worker therefore loads only the
complete physical blocks intersecting `[L_DE,K_DE)` and must not write the
Forward-owned `[K_DE,T)` suffix.

On synchronous delegated failure, roll back every adapter-created
`load_specs`, unfinished, loading, and request-tracker ownership record before
reporting activation failure. Never fall back to `PE_READ`.

## 12. Worker metadata installation and runtime gates

Task-08 adds no new Worker transfer primitive. It composes Task-06 Store,
Task-07 Reverse, and Task-05 Forward metadata.

Within one Worker metadata consumption call, retain the Task-07 mapping-first
order:

```text
install inbound wire-to-local mappings
-> reconcile pending raw DONE/FAILED
-> install ForwardReceiveBinding
-> install ReverseReceiveBinding where local
-> register ReversePlan and split tracker
-> install optional Store metadata
-> evaluate Store/Reverse gates
-> delegate ordinary parent metadata
```

Production split initialization is:

```text
partial Store DE_READ:
    Store=PENDING, Reverse=PENDING, Forward=PENDING

Store-miss DE_READ:
    Store=SKIPPED, Reverse=PENDING, Forward=PENDING
```

Reverse submission is request-level:

- partial Store submits all registered Reverse layer tasks exactly once after
  STORE_DONE;
- Store miss submits all Reverse layer tasks after mappings and tracker are
  installed;
- no per-layer Reverse-ready message or ACK is added; and
- PE model execution remains blocked until the final request-level Reverse
  terminal.

Forward retains inherited compute-one-layer/send-one-layer behavior after PE
Core resumes.

## 13. Completion and exact failure behavior

### 13.1 Success predicates

`PE_READ` success remains:

```text
Forward DONE -> DE finished_recving
```

Production `DE_READ` success is:

```text
(Store is DONE or SKIPPED)
and Reverse is DONE
and Forward is DONE
-> exactly one DE finished_recving
```

Store DONE alone, Reverse DONE alone, or Forward DONE before another required
phase must not complete the Decode request. Forward DONE logically implies PE
received Reverse, but the explicit local Reverse terminal is still required
for source-lifetime and cleanup ownership.

### 13.2 Data-plane failure provenance

Task-07 exact destination ownership remains:

| Failure | Engine | Invalid destination range |
|---|---|---|
| Store FAILED | DE | Store `[L_DE,K_DE)` |
| Reverse FAILED | PE | Reverse `[L_PE,K_DE)` |
| Reverse FAILED | DE | never-to-arrive Forward `[K_DE,T)` |
| Forward FAILED | DE | Forward `[K_DE,T)` |

Every request-level failure publishes exact invalid block IDs and one local
terminal together. FAILED wins over late or duplicate DONE.

### 13.3 Pre-data control failure

Before any route metadata is authorized, DE fails activation on:

- Decision/snapshot key conflict;
- wrong path/Reverse-plan shape;
- range, mapping, endpoint, topology, or protocol mismatch;
- retained allocation wrapper mismatch;
- duplicate conflicting production plan; or
- synchronous partial Store commit failure.

The state transitions to an activation-failed terminal, unregisters its key,
emits exactly one `DualPathControlFailureMetadata`, and authorizes no Store,
Reverse, or Forward work. Since no external destination is guaranteed ready,
DE invalidates every allocated external destination block `[L_DE,T)`.

Decision delivery exhaustion, missing PE response, PE local policy failure,
and PE local plan construction failure still converge through the existing DE
Decision deadline and `DECISION_TIMEOUT` reason.

No `PathDecisionError`, fallback, second Decision Request, or route refresh is
introduced.

## 14. Duplicate, retry, and no-redecision contract

The following are permanent invariants:

1. Allocation failure before PE binding may repeat Scheduler lookup, but the
   retained eligibility facts and result are reused.
2. Deterministic singleton `PE_READ` does not consume a Round Robin turn.
3. An ambiguous request consumes at most one Round Robin turn.
4. PE submits at most one asynchronous delivery Future per request key.
5. An identical allocation bind, plan, binding, or Decision is idempotent.
6. A conflicting duplicate preserves the first owner and fails locally.
7. DE accepts at most one committed Decision per key.
8. A late Decision after timeout or activation failure is stale.
9. Store coverage is never re-probed on DE after admission.
10. Store or P2P failure never falls back to the other path.

The path is frozen before PE allocation and delivered after PE allocation;
those are two lifecycle moments for one Decision, not two Decisions.

## 15. Cleanup and shutdown

### 15.1 Decode cleanup

`request_finished()` and `request_finished_all_groups()` must idempotently:

- remove lookup cache and `DecodeKVSnapshot`, releasing the retained block
  wrapper reference;
- remove Decision state and unregister its key;
- remove staged Forward/Reverse/Store metadata;
- remove split tracker and direction-specific terminals;
- release KVPool adapter ownership and request trackers;
- remove pending control-failure publication; and
- preserve ordinary parent cleanup.

### 15.2 Prefill cleanup

PE cleanup must idempotently remove:

- retained request key, `L_PE`, Decision Request, and result;
- local Forward plan and send adapter;
- Reverse receive binding;
- submitted delivery Future after completion;
- decider record only after no active local request or Future retains the key;
- local control-failure state; and
- early raw terminal/tombstone state through Task-07 ownership rules.

### 15.3 Shutdown

Shutdown stops new admissions and submissions, closes the Coordinator, waits
or cancels owned send Futures according to the existing contract, closes
KVPool adapters/workers, clears all retained block wrappers and plans, and
then preserves parent shutdown order. It does not force-cancel already
submitted DMA and adds no cross-engine abort.

## 16. Observability

Task-08 adds structured logs, not a metrics-provider subsystem. Logs should
make the following facts reconstructable per request key without dumping full
token or block contents:

- `L_DE`, `K_DE`, `L_PE`, `R`, and `T`;
- eligibility outcome: singleton `PE_READ` or policy invocation;
- selected path and whether Store is partial or skipped;
- logical Store, Reverse, compute, and Forward ranges;
- protocol version and delivery attempt terminal;
- control failure reason or data-plane failure source; and
- final success/failure predicate.

Do not add a new environment variable, adaptive policy, metrics provider,
Value Function, or LinkMonitor.

## 17. Focused CPU test contract

Tests should remain under
`tests/ut/distributed/kv_transfer/dual_path/` unless a closest parent parity
test belongs with the existing Layerwise suite.

### 17.1 Eligibility and policy

1. `L_PE == K_DE` forces `PE_READ` without calling policy.
2. `L_PE > K_DE` forces `PE_READ` without calling policy.
3. `L_PE < K_DE` calls policy exactly once and accepts either enum path.
4. Singleton requests do not advance seeded Round Robin state.
5. Ambiguous requests alternate according to the existing seeded policy.
6. Identical replay reuses one result and one policy turn.
7. Conflicting `L_PE` or Decision Request facts retain the first record and
   send nothing.
8. Policy exception or invalid return sends no Decision and DE times out.
9. HBM-complete and Store-full requests never reach the PE decider.

### 17.2 MultiConnector accounting

10. Forced/policy `PE_READ` returns `(0,False)` from DualPath.
11. PE AscendStore may become the winner after `PE_READ`; DualPath still
    receives real blocks and installs Forward.
12. If PE AscendStore also returns zero, no connector is recorded as a
    token-accounting winner and Forward still installs.
13. `DE_READ` returns exactly `(K_DE-L_PE,True)` and wins first-positive
    accounting regardless of the Store sibling's result. The Store sibling is
    still queried by the current upstream `MultiConnector`, but cannot replace
    DualPath as the accounting winner.
14. No Task-08 change to `AscendMultiConnector` is required.

### 17.3 Protocol and delivery timing

15. Protocol-v2 Decision Request and Result round-trip exact fields.
16. Protocol-v1 or mixed-version payload is rejected.
17. `PE_READ` serializes only a `None` Reverse plan.
18. `DE_READ` requires one valid serialized Reverse plan.
19. No sender Future exists after PE lookup alone.
20. Successful PE allocation installs local state before creating one Future.
21. Allocation retry before bind sends nothing and does not re-decide.
22. The DE queue returns the complete `PathDecision`, not only its Result.
23. Identical direct-channel duplicate is accepted once; conflict is rejected.

### 17.4 Plan construction and exact mappings

24. `PE_READ` freezes Forward `[L_DE,T)` with final PE/DE block tables.
25. Partial `DE_READ` freezes Store `[L_DE,K_DE)`, Reverse `[L_PE,K_DE)`, and
    Forward `[K_DE,T)` with intentionally different PE/DE block IDs.
26. Miss `DE_READ` creates no Store metadata and freezes Reverse
    `[L_PE,L_DE)` plus Forward `[L_DE,T)`.
27. Production `DE_READ` with absent/empty Reverse plan is rejected.
28. Group-count, alignment, coverage, endpoint, topology, request-key, and
    wire-ID mismatches fail activation.
29. Hybrid `T` and inherited request truncation are applied exactly once.

### 17.5 KVPool commit and snapshot ownership

30. Snapshot retains final immutable IDs and the non-comparing allocation
    wrapper.
31. Retained wrapper mismatch before commit fails without KVPool mutation.
32. Unified commit accepts Store-full and partial specs.
33. Partial commit delegates exact delta `K_DE-L_DE` and final blocks.
34. Worker metadata targets only `K_DE`, despite allocation through `T`.
35. Store miss calls no commit and enters `SKIPPED`.
36. `PE_READ` never commits the Decode LoadSpec.
37. Duplicate commit is rejected by adapter ownership.
38. Delegated commit failure rolls back private KVPool state and triggers one
    control failure.

### 17.6 Runtime success and gates

39. Partial Store completes before Reverse submission.
40. Store miss submits Reverse after mappings/tracker installation.
41. PE model execution begins only after final Reverse DONE.
42. PE then uses inherited per-layer Forward.
43. DE does not complete on Store, Reverse, or Forward alone.
44. DE completes exactly once after the full predicate.
45. `PE_READ` Forward continues to complete DE through Task-05 semantics.
46. Task-06 Store-full still creates no split tracker or PE activity.

### 17.7 Failure, race, and cleanup

47. Decision timeout emits one control failure and invalidates all external DE
    destinations.
48. Snapshot/plan mismatch and commit setup failure use activation failure,
    not timeout wording.
49. Store, Reverse, and Forward failures retain Task-07 exact destination
    provenance.
50. Early Reverse/Forward DONE and FAILED reconcile after binding.
51. FAILED wins over late DONE.
52. Late Decision after terminal cannot reactivate the route.
53. Request finish releases retained block wrappers, plans, bindings, adapter
    state, Futures, tombstones, and Coordinator ownership idempotently.
54. Shutdown leaves no private lookup endpoint, thread, Future, or request
    record.
55. Ordinary Layerwise, HBM-complete, Store-full, and existing PE_READ tests
    remain healthy.

## 18. NPU production acceptance

Production activation requires a supported Ascend environment. The success
matrix is:

| Case | Required route and result |
|---|---|
| HBM-complete | Local completion path; no Store/Proxy/PE/P2P |
| Decode Store-full | Task-06 local load and final-token recomputation |
| `L_PE >= K_DE` with PE Store hit | Forced `PE_READ`; PE Store/compute then Forward `[L_DE,T)` |
| Partial Store, policy `PE_READ` | No Decode Store/Reverse; PE prepare then Forward `[L_DE,T)` |
| Store miss, policy `PE_READ` | PE compute then Forward `[L_DE,T)` |
| Partial Store, policy `DE_READ` | Exact Store, Reverse, compute, and Forward ranges |
| Store miss, policy `DE_READ` | Store `SKIPPED`, Reverse Decode HBM, compute, and Forward |

For every routed case verify:

- PE and DE use their actual, intentionally distinct physical block tables;
- protocol-v2 is sent only after PE allocation;
- no mixed Decode-Store plus PE-Store route occurs;
- ordinary Attention uses the correct `T=P` Forward target;
- supported hybrid truncation uses `T=R` exactly once;
- inherited topology/reshard handling remains compatible;
- Decode transitions `WAITING_FOR_REMOTE_KVS -> WAITING` on success;
- generation resumes and recomputes the required final prompt token; and
- each terminal and invalid-block set belongs to the correct local request.

The Store-disappears-after-probe case must execute on NPU and prove failure
without fallback or hang. Complex duplicate, race, malformed-wire, and forced
Reverse/Forward failure matrices remain mandatory CPU fault-injection tests
even when NPU success cases pass.

## 19. Supported configuration boundary

Task-08 production support is intentionally limited to the already-approved
Stage 1 configuration:

```text
Decode KV cache groups: exactly one
Decode kv_load_failure_policy: fail
Decode Store consumer_is_to_load: True
Decode Store use_layerwise: False
Store/P2P block and topology facts: compatible with existing validators
PE/DE deployment: coordinated protocol-v2 upgrade
```

The single-group guard remains because the control-only invalid-block failure
relay is single-group. Task-08 must not claim multi-group Decode support merely
because lower-level lookup or parent Layerwise components support additional
shapes.

## 20. Expected implementation touchpoints

Task-08 is expected to remain concentrated in:

```text
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/kvpool_adapter.py
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py
tests/ut/distributed/kv_transfer/dual_path/
```

No Task-08 change is expected in:

```text
Proxy/server request dispatch
AscendMultiConnector
KVPoolScheduler or KVPoolWorker core implementation
MooncakeLayerwise parent transfer semantics beyond approved Task-07 helpers
upstream vLLM Scheduler state transitions
```

Any need to modify those surfaces is a design deviation requiring review
against this spec before implementation continues.

## 21. Explicitly out of scope

Task-08 does not implement or promise:

- adaptive routing, Value Function, LinkMonitor, or metrics-provider input;
- a production route with both Decode Store and PE Store;
- Store coverage refresh or `max(K_PE,K_DE)` re-probe on Decode;
- production empty-Reverse `DE_READ`;
- protocol-v1/v2 compatibility or rolling mixed-version upgrade;
- `PathDecisionError` or a Result reply;
- post-commit redecision, fallback, or second Prefill dispatch;
- a new data-plane watchdog, exponential backoff, or terminal resend protocol;
- user abort or cross-engine cancellation;
- relay or multi-hop transfer;
- multi-KV-cache-group Decode activation; or
- a new environment variable beyond the existing Decision timeout.

## 22. Acceptance checklist

Task-08 is accepted only when all of the following are true:

1. `L_PE >= K_DE` deterministically selects `PE_READ` without advancing the
   policy; only `L_PE < K_DE` invokes `PathPolicy`.
2. Production `DE_READ` always has a non-empty Reverse, while Store miss keeps
   an empty Store interval and a Decode-HBM Reverse.
3. Protocol version 2 atomically carries one Result plus an optional
   `ReversePlan`; no independent `prefill_local_tokens` field or second plan
   message exists.
4. PE sends the Decision only after final allocation and local plan/binding
   installation.
5. `PE_READ` composes with PE AscendStore through existing first-positive
   accounting; `DE_READ` secures the first-positive accounting win regardless
   of that sibling's result.
6. `DecodeKVSnapshot` retains immutable IDs plus the allocation wrapper, and
   the unified Store commit supports exact full/partial deltas.
7. A real partial/miss `DE_READ` activates Task-07 Store/Reverse/Forward
   runtime with exact mappings and gates.
8. DE publishes success only after its complete source-aware predicate.
9. Decision timeout, activation failure, Store failure, Reverse failure, and
   Forward failure terminate without fallback, redecision, or hang.
10. Cleanup and shutdown release every Scheduler, Coordinator, adapter,
    Worker, plan, mapping, Future, retained wrapper, and terminal record.
11. Required CPU tests and NPU production matrix pass on the supported
    configuration.
12. HBM-complete, DE-local Store-full, ordinary Layerwise, and prior PE_READ
    regressions remain healthy.

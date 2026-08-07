# Task-07 Detailed Spec — Bidirectional Split Runtime

## 1. Status, authority, and implementation starting point

This document is the complete implementation contract for Task-07 in the
[`DualPath Stage 1 Task Catalog`](../TASKS.md). It is intentionally
self-contained: an implementation Agent may read this file without reading
the Task-01 through Task-06 detailed specs.

The implementation starting point is:

```text
vllm-ascend@aa32e8542f8d76f28b51f6e03fc3944a8fed406e
```

At that baseline:

- Task-05 provides an executable, test-forced `Path.PE_READ` Forward path;
- Task-06 provides deterministic DE-local Store-full completion;
- PE Workers own the inherited Forward send runtime;
- DE Workers own the inherited Forward receive runtime;
- the parent runtime starts only the send or receive side selected by the
  external producer/consumer configuration;
- `MooncakeLayerwiseConnectorWorker.wait_for_layer_load()` is a no-op;
- the inherited sender transfers Forward after every computed layer but emits
  only one request-level DONE/FAILED terminal after the final layer; and
- no production non-full `Path.DE_READ` result authorizes Store, Reverse, or
  Forward work.

Where this document conflicts with an earlier Task spec, this document wins
for the composed post-Task-07 Worker runtime. It does not revise Task-01
admission, Task-02 policy semantics, Task-03 transport reliability, Task-04
decision ownership, Task-05 `PE_READ`, or Task-06 Store-full routing except for
the explicit backward-compatible metadata/runtime deltas in this document.

## 2. Outcome and merge-state contract

Task-07 builds the bidirectional Worker runtime required by a Store-non-full
split `Path.DE_READ` route:

```text
optional DE Store load
-> request-level Reverse from DE to PE
-> PE compute after the complete Reverse terminal
-> inherited layerwise Forward from PE to DE
-> source-aware completion or failure
```

The route is exercised only by a test-built, immutable metadata plan. Task-07
must not connect a real `PathDecisionResult(Path.DE_READ)` to this runtime.

After acceptance:

- the PE Worker owns Forward-send and Reverse-receive capabilities;
- the DE Worker owns Reverse-send and Forward-receive capabilities;
- one existing TransferEngine and one KV-buffer registration back both
  capabilities on each Worker;
- a split request never publishes DE `finished_recving` on Store completion
  alone;
- Reverse begins only after the optional Store phase is satisfied;
- PE begins model execution only after the complete request-level Reverse
  terminal;
- PE then preserves the inherited compute-one-layer/send-one-layer Forward
  behavior;
- DONE/FAILED retains Store, Reverse, or Forward provenance until the owning
  split tracker consumes it;
- injected split success, failure, race, duplicate, empty-Store, and
  empty-Reverse cases are executable; and
- production non-full policy activation remains disabled until Task-08.

## 3. Terminology and exact logical ranges

For one Store-non-full request:

```text
P    = original prompt token count
R    = max(P - 1, 0), Decode-ready prefix
T    = parent Layerwise transfer target
       P for ordinary Attention
       R for Attention-Mamba hybrid
L_DE = contiguous Decode HBM prefix
K_DE = usable, aligned Decode Store prefix after clamp to R
L_PE = contiguous Prefill HBM prefix
```

The split route uses these logical ranges:

```text
DE Store load:  [L_DE, K_DE)
DE -> PE Reverse: [L_PE, K_DE)
PE compute:     [max(L_PE, K_DE), T)
PE -> DE Forward: [K_DE, T)
```

Store non-full guarantees `K_DE < R <= T`, so Forward is non-empty. Store and
Reverse may be empty independently:

```text
K_DE == L_DE  -> Store is SKIPPED
L_PE >= K_DE  -> Reverse is SKIPPED
```

An empty interval is represented by the absence of work metadata. Task-07
must not construct a zero-length `ReversePlan` or submit a zero-length Store
or P2P operation.

### 3.1 Example with an empty Store interval

```text
L_DE = 64, K_DE = 64, L_PE = 16, T = 128

Store:   [64, 64)   SKIPPED
Reverse: [16, 64)   uses the existing DE HBM prefix
Compute: [64, 128)
Forward: [64, 128)
```

Store miss, Store unavailable, or Store coverage that contributes no aligned
block beyond `L_DE` all use this same shape.

### 3.2 Complete-block execution

Stage 1 transfers complete physical blocks. Logical ranges remain part of the
plan for validation, accounting, observability, and invalid-block selection,
but the Worker must use the frozen source/destination block tables supplied by
the plan. It must not query a live Scheduler block table during execution or
failure handling.

## 4. End-to-end injected sequence

```mermaid
sequenceDiagram
    autonumber

    participant Harness as "Injected Task-07 harness"
    participant DEW as "DE DualPath Worker"
    participant Store as "DE KVPool Worker"
    participant DESend as "Inherited DE send runtime"
    participant PERecv as "Inherited PE receive runtime"
    participant PECore as "PE execution boundary"
    participant PESend as "Inherited PE send runtime"
    participant DERecv as "Inherited DE receive runtime"

    Harness->>PERecv: install ReverseReceiveBinding first
    Harness->>DEW: install ForwardReceiveBinding(path=DE_READ), ReversePlan, optional Store metadata
    DEW->>DEW: create split tracker before local work

    alt "Store interval non-empty"
        DEW->>Store: start async Store load [L_DE, K_DE)
        Store-->>DEW: STORE_DONE
    else "Store interval empty"
        DEW->>DEW: Store = SKIPPED
    end

    DEW->>DESend: enqueue all Reverse layers in registered order
    loop "registered layers"
        DESend->>PERecv: write Reverse blocks for current layer
    end
    DESend-->>DEW: local Reverse DONE/FAILED
    DESend-->>PERecv: existing final DONE_SENDING_MSG/FAILED_SENDING_MSG

    alt "Reverse DONE"
        PERecv-->>PECore: finished_recving
        PECore->>PECore: begin model execution only now
        loop "each computed layer"
            PECore->>PESend: inherited save_kv_layer()
            PESend->>DERecv: Forward current layer [K_DE, T)
        end
        PESend-->>DERecv: existing final DONE/FAILED
        DERecv->>DEW: Forward terminal with binding provenance
        DEW->>DEW: evaluate Store + Reverse + Forward predicate
    else "Reverse FAILED"
        PERecv-->>PECore: invalid Reverse destination blocks + finished_recving
        DEW->>DEW: fail split request locally; Forward will not arrive
    end

    alt "all required phases succeeded"
        DEW-->>Harness: finished_recving(decode_request_id)
    else "any required phase failed"
        DEW-->>Harness: exact invalid_block_ids + finished_recving
    end
```

"Enqueue all Reverse layers" is a request-level phase. The inherited send
thread may still execute one layer task at a time because its address,
quantization, and resharding logic is layer-oriented. PE observes no
intermediate Reverse readiness and does not compute until the final terminal.

Task-07 adds no per-layer Reverse READY message and leaves
`wait_for_layer_load()` unchanged.

## 5. Approved reuse design

The following alternatives were considered:

1. Configure every Worker as both producer and consumer. This changes parent
   Scheduler/Worker branches and external role semantics and is rejected.
2. Copy parent thread construction and `save_kv_layer()` into DualPath. This
   avoids parent edits but duplicates a long, topology-sensitive runtime and
   is rejected.
3. Extract only the parent runtime-start and single-layer enqueue seams, then
   let DualPath start the missing direction and reuse the inherited transfer
   logic. This is the approved design.

Only the two extractions in Sections 6 and 7 are authorized. Any additional
parent helper or behavior change is a design deviation requiring review.

## 6. Parent runtime-start extraction

The parent currently performs buffer registration, layer metadata creation,
and inline thread construction in one `register_kv_caches()` method. Task-07
must preserve buffer/metadata construction and extract idempotent protected
thread starters equivalent to:

```python
def _ensure_send_layer_runtime(self) -> None: ...

def _ensure_receive_layer_runtime(self) -> None: ...
```

Exact names may follow local style, but the contract is normative:

- each helper constructs and starts only its corresponding existing
  `KVCacheSendingLayerThread` or `KVCacheRecvingLayerThread`;
- it uses the already-created TransferEngine, layer metadata, registered
  buffers, ports, topology, reshard buffers, and callbacks;
- it is idempotent when the thread already exists;
- it does not call `TransferEngine.register_buffer()`;
- it does not change external producer/consumer configuration; and
- the parent calls the same helper it would have constructed inline.

After `super().register_kv_caches(kv_caches)`:

```text
PE DualPath Worker: ensure receive runtime for Reverse
DE DualPath Worker: ensure send runtime for Reverse
```

The resulting ownership is:

```text
PE Worker = Forward send + Reverse receive
DE Worker = Reverse send + Forward receive
```

Ordinary parent roles remain one-directional. DualPath must not create a
second TransferEngine, duplicate receive port, or second memory registration.

## 7. Parent single-layer enqueue extraction

DE has no ModelRunner prompt execution from which to receive Reverse
`save_kv_layer()` callbacks. After Store completes it must submit all
registered layers itself without copying the parent's quantization,
resharding, event, metadata lookup, or `SendTask` logic.

Task-07 therefore extracts one protected helper equivalent to:

```python
def _enqueue_kv_layer_send(
    self,
    *,
    layer_index: int,
    layer_name: str,
    kv_layer: list[torch.Tensor],
    ready_event: torch.npu.Event,
    metadata: MooncakeLayerwiseConnectorMetadata,
) -> None: ...
```

The exact parameter shape may use an existing parent object where that avoids
duplication, but it must support both call sites:

```text
Forward:
save_kv_layer()
-> derive the existing compute/reshape-ready event
-> call the protected enqueue helper once

Reverse:
Store DONE/SKIPPED
-> establish a Store/HBM-ready event
-> iterate registered layer order
-> call the same protected enqueue helper for every layer
```

The helper owns the unchanged parent behavior for:

- layer/group lookup;
- page-cache load into reshard/quant buffers when required;
- TP/PD resharding;
- KV and C8 quantization;
- remote static layer metadata lookup;
- request-to-session grouping;
- `SendTask` construction and queue insertion; and
- final-layer DONE/FAILED callback behavior.

Task-07 may retain read-only references to the `kv_caches` dictionary supplied
to `register_kv_caches()` and a deterministic registered layer order. It must
not copy KV tensors or register them again. The layer order comes from the
parent's registered layer index/name mapping, not an incidental later dict
iteration.

Focused parent regression tests must prove that ordinary Forward creates the
same send tasks, waits on the same events, uses the same topology transforms,
and emits the same terminal after this extraction.

## 8. Immutable metadata contracts

Task-07 does not introduce a production `SplitPlan` object. "Split plan" is a
conceptual composition of existing Store metadata, one Reverse plan/binding,
and the existing Forward plan/binding.

### 8.1 `ForwardReceiveBinding.path`

The existing binding becomes explicit about the owning route:

```python
@dataclass(frozen=True)
class ForwardReceiveBinding:
    request_key: DualPathRequestKey
    path: Path
    wire_request_id: str
    decode_request_id: str
    destination_block_ids: BlockTable
    token_start: int
    token_end: int
```

Task-05 emits `path=Path.PE_READ`. An injected Task-07 split binding uses
`path=Path.DE_READ`.

The field selects completion semantics only:

```text
PE_READ Forward DONE -> immediate DE finished_recving
DE_READ Forward DONE -> update split tracker and evaluate all phases
```

The field must be required; a default that silently treats missing data as
`PE_READ` is not allowed.

### 8.2 `ReversePlan`

`ReversePlan` is the immutable DE send authorization:

```python
@dataclass(frozen=True)
class ReversePlan:
    request_key: DualPathRequestKey
    wire_request_id: str
    token_start: int
    token_end: int
    source_block_ids: BlockTable
    destination_block_ids: BlockTable
    # Existing Mooncake peer facts needed to adapt the plan into ReqMeta:
    remote_engine_id: str
    remote_host: str
    remote_port: int
    remote_block_sizes: tuple[int, ...]
    remote_tp_size: int
    remote_pcp_size: int
    remote_dcp_size: int
```

The inherited peer fields are facts, not a new endpoint/protocol entity. The
plan does not contain layer addresses, a TransferEngine object, live tensors,
mutable request state, or a Store handle. Static remote layer addresses and TE
port continue to come from the inherited `GET_META` handshake.

The plan freezes logical `[L_PE, K_DE)` and complete source/destination block
tables. Worker adaptation may select the intersecting complete blocks once,
but must not mutate the plan or obtain replacement block IDs later.

Validation requires:

- non-empty strings and positive integer topology fields;
- `0 <= token_start < token_end`;
- a non-empty, deeply frozen block table on both sides;
- group counts compatible with local and remote block-size groups;
- sufficient source and destination coverage through `token_end`;
- block-compatible range alignment for the supported topology; and
- a request key/wire ID that does not conflict with an installed plan.

An empty Reverse interval creates no `ReversePlan`.

### 8.3 `ReverseReceiveBinding`

The PE receive-side ownership contract is:

```python
@dataclass(frozen=True)
class ReverseReceiveBinding:
    request_key: DualPathRequestKey
    wire_request_id: str
    prefill_request_id: str
    destination_block_ids: BlockTable
    token_start: int
    token_end: int
```

It freezes PE destination ownership for `[L_PE, K_DE)`, supplies the
wire-to-local request mapping, and identifies the blocks to invalidate on
Reverse failure. It contains no sender endpoint or Store state.

An empty Reverse interval creates no binding and the PE Reverse gate is
already satisfied.

### 8.4 `DualPathConnectorMetadata`

Task-07 extends the current metadata without placing Reverse in the parent's
ordinary request map:

```python
class DualPathConnectorMetadata(MooncakeLayerwiseConnectorMetadata):
    decision_timeouts: list[DecisionTimeoutMetadata]
    forward_receive_bindings: list[ForwardReceiveBinding]
    decode_store_metadata: AscendConnectorMetadata | None
    reverse_plans: list[ReversePlan]
    reverse_receive_bindings: list[ReverseReceiveBinding]
```

For the injected split harness:

```text
DE metadata:
    ForwardReceiveBinding(path=DE_READ)
    optional ReversePlan
    optional decode_store_metadata

PE metadata:
    optional ReverseReceiveBinding
    existing Forward metadata when PE later executes
```

Task-07 does not add a Scheduler injection method. Tests construct these
metadata objects directly.

## 9. Explicit Task-07/Task-08 handoff boundary

PE destination block IDs exist only after real PE allocation. The current
inherited `GET_META` response returns static layer addresses and TE port; it
does not return request-specific PE block IDs. Task-07 must not pretend that
the current handshake creates a production Reverse plan.

Task-07 owns only:

- immutable Reverse contracts;
- metadata validation and Worker consumption;
- the executable bidirectional runtime; and
- injected tests.

It must not:

- create a Reverse plan from a real `PathDecisionResult`;
- change `PathDecision` serialization;
- extend Coordinator delivery with PE block IDs;
- add Proxy fields;
- extend `GET_META` with a request-specific plan;
- add a Scheduler test-injection provider; or
- activate a production non-full `Path.DE_READ` result.

Task-08 must later:

```text
observe real Path.DE_READ
-> wait for/freeze final PE allocation
-> create ReversePlan and ReverseReceiveBinding
-> deliver the DE-owned plan and install the PE-owned binding
-> generate the matching Store/Forward metadata
-> activate the already-tested runtime
```

Task-08 chooses the production plan-delivery mechanism. That choice is not an
implementation gap that Task-07 may fill implicitly.

## 10. Worker registration and local plan installation

`DualPathConnectorWorker.register_kv_caches()` must:

1. delegate parent registration exactly once;
2. retain read-only tensor references and registered layer order;
3. ensure the PE receive runtime or DE send runtime as specified in Section 6;
4. preserve Task-06's one DE Store-worker registration; and
5. perform no request-level Store or P2P work.

For each metadata step, local installation is mapping-first.

### 10.1 DE installation order

```text
1. validate ForwardReceiveBinding(path=DE_READ)
2. install wire_request_id -> decode_request_id
3. create the split tracker
4. validate/install the optional ReversePlan
5. identify optional Store metadata ownership
6. delegate Store start when required
7. submit Reverse immediately only when Store is SKIPPED
8. delegate remaining ordinary parent metadata once
```

No Store or Reverse operation may start before the binding and tracker exist.

### 10.2 PE installation order

```text
1. validate ReverseReceiveBinding
2. install wire_request_id -> prefill_request_id
3. register terminal ownership
4. delegate remaining ordinary parent metadata once
```

Installing a binding does not itself write KV or publish completion.

## 11. DE split execution tracker

Each injected `Path.DE_READ` Forward binding owns one internal DE Worker
record keyed by Decode-local request ID. It is not a public protocol type and
does not cross processes.

The required state is equivalent to:

```text
Store phase   = SKIPPED | PENDING | DONE | FAILED
Reverse phase = SKIPPED | PENDING | DONE | FAILED
Forward phase = PENDING | DONE | FAILED
Store destination blocks = frozen exact slice, when Store is PENDING
terminal_published = bool
```

Initialization is derived from actual metadata:

```text
Store metadata for request present -> Store=PENDING
otherwise                          -> Store=SKIPPED

ReversePlan present                -> Reverse=PENDING
otherwise                          -> Reverse=SKIPPED
```

The only success predicate is:

```python
store_satisfied = store in {SKIPPED, DONE}
reverse_satisfied = reverse in {SKIPPED, DONE}

decode_ready = (
    store_satisfied
    and reverse_satisfied
    and forward is DONE
)
```

Any FAILED phase is terminal failure and wins over every DONE event.

### 11.1 Store terminal handling

The current KVPool Worker returns completed request IDs from `get_finished()`
and moves any per-request failed block IDs into its separately drained invalid-
block set. It does not return an already typed STORE_DONE/STORE_FAILED event.
Task-07 must compose that type without changing the KVPool thread protocol:

```text
1. call KVPoolWorkerAdapter.get_finished()
2. immediately drain KVPoolWorkerAdapter.get_block_ids_with_load_errors()
3. add every drained block to the DualPath Worker's composed invalid set
4. for each finished split Store request, intersect the drained IDs with that
   tracker's frozen Store destination slice
5. non-empty intersection -> STORE_FAILED
   empty intersection     -> STORE_DONE
```

Allocated destination blocks are request-owned, so the intersection
attributes concurrent Store failures without treating another request's
invalid block as this request's failure. An unexpected drained ID that matches
no active split tracker is still preserved for ordinary Task-06/Core error
reporting; it is never discarded.

For a split request, the resulting Store terminal is consumed internally:

- STORE_DONE changes Store to DONE and submits Reverse exactly once when a
  plan exists;
- STORE_FAILED changes Store to FAILED, preserves Store-owned invalid blocks,
  and publishes one failed DE terminal;
- Store completion alone never appears in the outer `done_recving` set.

Task-06 Store-full requests have no split tracker. Their existing Store DONE
or FAILED continues directly to the outer completion path. Draining Store
errors earlier inside `get_finished()` must remain externally invisible: the
same IDs are re-published through the composed DualPath invalid-block set.

### 11.2 Reverse submission

Reverse is authorized only when Store is DONE or SKIPPED. Submission:

1. adapts the frozen plan into existing parent request metadata;
2. establishes a readiness event after Store/HBM data is available;
3. iterates the registered layer order exactly once;
4. invokes the protected single-layer enqueue helper for every layer; and
5. relies on the existing final-layer callback for one request-level terminal.

Duplicate Store DONE, metadata replay, or polling must not enqueue Reverse a
second time.

## 12. Reverse sender-local and PE receive terminals

The existing sending thread already calls `callback_func` with
`trans_flag=True/False`. On the DE role, the DualPath Worker overrides the
existing callback behavior to record the local Reverse terminal under a lock,
then delegates the unchanged parent DONE/FAILED notification to PE.

Conceptually:

```python
if role == "decode":
    record_local_reverse_done_or_failed(req_id, trans_flag)
super().send_done_send_signal(req_id, req_meta, group_idx, trans_flag)
```

No new wire message is introduced. The remote PE continues to receive the
existing final `DONE_SENDING_MSG` or `FAILED_SENDING_MSG` with its existing
three-attempt ACK behavior.

The local DE terminal updates the split tracker:

- Reverse DONE changes Reverse to DONE but cannot complete DE before Forward;
- Reverse FAILED fails DE immediately because PE cannot produce Forward.

The PE terminal resolves `ReverseReceiveBinding`:

- DONE publishes PE `finished_recving`, after which PE may execute the model;
- FAILED marks Reverse destination blocks invalid and publishes PE
  `finished_recving` for failure handling.

PE does not receive or act on per-layer Reverse readiness.

### 12.1 Reliability boundary

Task-07 inherits the parent data-terminal reliability:

- TransferEngine failure produces `trans_flag=False` and is locally visible
  on DE before the best-effort remote FAILED notification;
- remote terminal notification uses the existing bounded retry and ACK;
- Task-07 adds no end-to-end P2P watchdog, new timeout variable, re-decision,
  fallback, or terminal replay service; and
- if a successful write's final terminal is permanently lost after inherited
  retries, the remote request may remain waiting, matching the acknowledged
  parent limitation.

## 13. Forward execution and DE completion

After PE Reverse DONE, the PE request begins normal model execution. Task-07
must not add a new Forward loop. The existing path remains:

```text
compute layer
-> save_kv_layer()
-> protected parent enqueue helper
-> Forward current layer [K_DE, T)
```

The injected Forward metadata must use `local_transferred_tokens=K_DE` so the
inherited sender transfers the complete Forward suffix, including any
PE-cached portion before the actual compute start when `L_PE > K_DE`.

On DE:

- a `Path.PE_READ` Forward binding retains Task-05 immediate completion;
- a `Path.DE_READ` Forward terminal updates the split tracker;
- Forward DONE publishes DE completion only when the full predicate in
  Section 11 is true; and
- Forward FAILED fails the split request regardless of earlier Store/Reverse
  success.

## 14. Mapping-first and early terminal reconciliation

Task-07 preserves Task-05's raw-terminal race handling and adds its PE Reverse
mirror:

```python
_reverse_receive_bindings
_pending_reverse_done
_pending_reverse_failed
_consumed_reverse_terminals
```

A generic terminal-manager class is not required. Small shared private
functions may remove mechanical duplication, but Forward and Reverse
ownership must remain explicit.

Rules:

1. An identical duplicate binding/plan is idempotent.
2. A conflicting duplicate is rejected while preserving the first owner.
3. DONE/FAILED received before a binding is retained in the direction's
   pending set.
4. The next `get_finished()` after binding reconciles pending and raw events.
5. FAILED wins when DONE and FAILED conflict.
6. A consumed terminal leaves a direction-specific tombstone until
   `finished_req_ids` releases it.
7. A late duplicate cannot bind to another local request.
8. Ordinary parent raw events that do not belong to a DualPath binding retain
   parent behavior.

Mapping-first is a strong local Worker order. It is not a new cross-engine
binding-ready handshake; early-event retention still covers unavoidable
cross-process races.

## 15. Exact failure and invalid-block provenance

Invalid blocks are destination slots that are partially written, missing, or
known never to become ready. They are not limited to physically corrupted
memory.

For each failure source:

| Failure | Engine | Invalid destination range |
|---|---|---|
| Store FAILED | DE | Store `[L_DE, K_DE)` |
| Reverse FAILED | PE | Reverse `[L_PE, K_DE)` |
| Reverse FAILED | DE | never-to-arrive Forward `[K_DE, T)` |
| Forward FAILED | DE | Forward `[K_DE, T)` |

The source-side blocks remain valid:

- Reverse failure does not invalidate DE source blocks;
- Forward failure does not invalidate PE source blocks; and
- one failed stage does not invalidate the complete request block table.

The Worker selects all complete destination blocks intersecting the logical
range from the frozen binding/plan. If the transfer layer cannot report a
finer partial-success boundary, the complete planned destination slice is
invalid.

Example for block size 16:

```text
L_DE=32, K_DE=64, L_PE=16, T=128

DE table:
[0,16)=D10 [16,32)=D11 [32,48)=D20 [48,64)=D21
[64,80)=D30 [80,96)=D31 [96,112)=D40 [112,128)=D41

PE table:
[0,16)=P70 [16,32)=P71 [32,48)=P80 [48,64)=P81
[64,80)=P90 [80,96)=P91 [96,112)=P92 [112,128)=P93
```

Then:

```text
Store FAILED   -> DE invalid {D20, D21}
Reverse FAILED -> PE invalid {P71, P80, P81}
                  DE invalid {D30, D31, D40, D41}
Forward FAILED -> DE invalid {D30, D31, D40, D41}
```

Failure publication is always:

```text
exact invalid_block_ids + exactly one finished_recving
```

so the existing failure policy can terminate the request. A late DONE cannot
change FAILED back to success.

## 16. Cleanup and shutdown

Task-07 does not add user abort or cross-engine cancellation.

Normal local cleanup is still mandatory:

- `finished_req_ids` releases installed Forward/Reverse bindings, plans,
  split trackers, pending terminals, and consumed-terminal tombstones owned by
  those local request IDs;
- repeated cleanup is idempotent;
- Task-05 `PE_READ` and Task-06 Store-full cleanup remains unchanged;
- shutdown prevents new Task-07 submissions, clears local plan/tracker/terminal
  state, closes the existing Task-06 Store adapter, and delegates every
  available parent cleanup behavior exactly once; and
- no cleanup path promises to stop DMA already submitted on another engine.

## 17. Focused CPU test contract

Tests should live under
`tests/ut/distributed/kv_transfer/dual_path/` unless a focused parent parity
test belongs with the parent Layerwise connector.

### 17.1 Contracts and immutability

1. `ForwardReceiveBinding.path` accepts only `Path.PE_READ` or
   `Path.DE_READ` and is required.
2. `ReversePlan` deep-freezes source/destination tables and peer facts.
3. `ReverseReceiveBinding` deep-freezes PE destination ownership.
4. Invalid ranges, booleans-as-integers, missing groups, insufficient
   coverage, invalid topology values, and conflicting identities are rejected.
5. Empty Reverse is represented by absence, not a zero-length object.

### 17.2 Parent extraction and registration parity

6. Ordinary producer registration starts only the send runtime.
7. Ordinary consumer registration starts only the receive runtime.
8. PE DualPath owns exactly one send and one receive runtime.
9. DE DualPath owns exactly one receive and one send runtime.
10. Repeated ensure calls create no duplicate thread, port, engine, or buffer
    registration.
11. Parent Forward before/after extraction produces equivalent layer order,
    `SendTask`, wait event, topology transform, and final callback.
12. Reverse iterates the retained registered layer order and uses the same
    enqueue helper.

### 17.3 Successful split lifecycle

13. Non-empty Store does not submit Reverse before STORE_DONE.
14. STORE_DONE submits every Reverse layer exactly once.
15. Empty Store submits Reverse immediately after local binding/tracker
    installation.
16. Reverse DONE does not complete DE before Forward.
17. PE publishes no completion before final Reverse DONE.
18. After Reverse DONE, the harness executes PE compute and inherited
    layerwise Forward.
19. DE publishes completion only after Store/Reverse/Forward satisfy the
    exact predicate.
20. Empty Reverse creates no P2P task and lets the PE Reverse gate start
    satisfied.
21. Store and Reverse both empty still waits for required Forward.

### 17.4 Failure provenance

22. Store failure starts no Reverse and invalidates only Store destination
    blocks on DE.
23. Reverse transfer failure records a local DE terminal, invalidates PE
    Reverse destinations, invalidates the never-to-arrive DE Forward suffix,
    and publishes no success.
24. Forward failure invalidates only the DE Forward destination suffix.
25. FAILED wins over duplicate/late DONE for every source.
26. Each failure publishes exactly one local terminal.

### 17.5 Race, duplicate, and cleanup

27. Reverse DONE before PE binding is retained and reconciled later.
28. Reverse FAILED before PE binding is retained and reconciled later.
29. Existing Forward early-terminal tests pass for both binding paths.
30. Identical duplicate plans/bindings are idempotent; conflicts preserve the
    first owner and fail.
31. Duplicate Store DONE cannot enqueue Reverse twice.
32. `finished_req_ids` and shutdown remove all Task-07 local state
    idempotently.
33. Unknown/ordinary parent terminals are never attributed to a DualPath
    request.

### 17.6 Regression boundaries

34. Task-05 `PE_READ` Forward DONE still completes DE immediately.
35. Task-06 Store-full still publishes completion directly and creates no
    split tracker, Reverse, Proxy, or PE work.
36. HBM-complete and ordinary Layerwise requests retain prior behavior.
37. A real production `Path.DE_READ` result still creates no Task-07 plan or
    Worker operation.
38. No Task-07 Scheduler hook blocks on Store, ZMQ, NPU, or TransferEngine
    work.

## 18. NPU injected acceptance

Before Task-07 merge, a supported Ascend environment must run an injected
two-sided Worker smoke test using real KV buffers and TransferEngine. It does
not use the production `PathPolicy` or plan-delivery path.

Required cases:

| Case | Store | Reverse | Forward |
|---|---|---|---|
| Partial Store | non-empty `[L_DE,K_DE)` | non-empty `[L_PE,K_DE)` | `[K_DE,T)` |
| Empty Store | SKIPPED | non-empty existing-HBM suffix | `[K_DE,T)` |

For each case verify:

- PE and DE use intentionally different physical block IDs;
- Store data lands in the frozen DE destinations;
- Reverse copies the exact logical/physical mapping into PE;
- PE does not begin model execution before final Reverse DONE;
- PE then computes and uses inherited per-layer Forward;
- DE receives the exact Forward suffix;
- only the final request-level Reverse and Forward terminals are visible;
- DE publishes completion exactly once; and
- no real policy result, Coordinator plan delivery, or Proxy mutation is
  involved.

CPU failure/race tests remain mandatory even when the NPU smoke test passes.
Task-08 owns the complete production route, topology, Hybrid, miss,
unavailable, and policy matrix.

## 19. Expected implementation touchpoints

Task-07 is expected to remain concentrated in:

```text
vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py
tests/ut/distributed/kv_transfer/dual_path/
tests/ut/kv_offload/test_mooncake_layerwise_connector.py  # if closest parity home
```

The parent file may contain only the two approved behavior-preserving
extractions. Task-07 is not expected to modify:

```text
path_decision.py
path_decision_channel.py
Proxy/server code
AscendMultiConnector selection
vLLM Scheduler state transitions
```

Any additional production control-plane touchpoint is a Task-08 concern and a
Task-07 design deviation.

## 20. Explicitly out of scope

Task-07 does not implement or promise:

- production non-full `Path.DE_READ` activation;
- real Scheduler creation or cross-engine delivery of Reverse plans;
- a `SplitPlan`, `ReverseHandle`, or per-layer readiness object;
- per-layer Reverse READY/ACK messages;
- Reverse/PE-compute overlap;
- a true all-layers-in-one-call bulk Reverse path;
- modifications to the Decision or Proxy wire schemas;
- request-specific PE block IDs in inherited `GET_META`;
- a new Store lookup, alignment, or key-generation algorithm;
- post-commit fallback or re-decision;
- a new data-plane watchdog or terminal-recovery protocol;
- user abort or cross-engine cancellation;
- multi-group Decode expansion beyond the inherited Stage 1 guard; or
- adaptive policy, metrics, relay, or topology selection.

## 21. Acceptance checklist

Task-07 is accepted only when all of the following are true:

1. PE and DE DualPath Workers each own the required send and receive
   capabilities through one inherited TransferEngine and one registration.
2. Parent runtime construction and single-layer send logic are extracted only
   through the two approved protected seams with ordinary behavior parity.
3. `ForwardReceiveBinding.path` selects immediate `PE_READ` completion or
   split `DE_READ` aggregation without a new route enum.
4. Immutable `ReversePlan` and `ReverseReceiveBinding` freeze exact logical
   ranges and physical ownership; no production `SplitPlan` exists.
5. Optional Store completion gates one request-level Reverse phase; PE waits
   for final Reverse DONE before computing.
6. PE retains inherited compute-one-layer/send-one-layer Forward semantics.
7. DE success requires Store and Reverse DONE/SKIPPED plus Forward DONE.
8. Store, Reverse, and Forward failures retain provenance and invalidate only
   the exact destination ranges defined in Section 15.
9. Mapping-first binding, early raw terminal retention, FAILED precedence,
   duplicate idempotence, tombstones, and cleanup are covered.
10. Task-05 `PE_READ`, Task-06 Store-full, and ordinary Layerwise regressions
    pass.
11. Focused CPU tests and the injected NPU smoke cases pass.
12. No real `Path.DE_READ` result can execute the new runtime; Task-08 remains
    the sole production activation gate.

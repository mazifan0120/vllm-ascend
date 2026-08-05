# Task-01 Detailed Spec — Decode Admission to `WAITING_FOR_REMOTE_KVS`

## 1. Status and authority

This document defines the implementation contract for Task-01 in the
[`DualPath Stage 1 Task Catalog`](../TASKS.md). It refines Task-01 without
changing the dependency graph or activating a production data path.

The source baseline is:

- `vllm-ascend@7c3983fce22b9c06aac05c57139973e08e53aa2a`
- sibling `vllm@0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`

Task-01 is complete only when a real initial Decode scheduling pass allocates
the final Decode blocks and leaves the request in
`WAITING_FOR_REMOTE_KVS`. The request is intentionally not promoted again in
this Task.

## 2. Outcome

For a DualPath Decode request, Task-01 must implement this call chain:

```mermaid
sequenceDiagram
    autonumber

    participant Core as "vLLM Scheduler"
    participant HBM as "KVCacheManager"
    participant DP as "DualPathConnectorScheduler"
    participant KVA as "KVPoolAdapter"
    participant KVS as "KVPoolScheduler"

    Core->>HBM: get_computed_blocks(request)
    HBM-->>Core: L_DE and local blocks
    Core->>DP: get_num_new_matched_tokens(request, L_DE)

    alt "L_DE >= R"
        DP-->>Core: (0, False)
        Note over DP,KVS: "No KVPool lookup and no Task-01 state"
    else "L_DE < R"
        DP->>KVA: lookup(request, L_DE)
        KVA->>KVS: get_num_new_matched_tokens(request, L_DE)
        KVS-->>KVA: "Store delta and async flag"
        KVA->>KVA: "pop detached LoadSpec"
        KVA-->>DP: "LoadSpec or None"
        DP->>DP: "cache (E_DE, detached LoadSpec)"
        DP-->>Core: (E_DE, True)

        Core->>HBM: "allocate_slots(..., external=E_DE, delay_cache_blocks=True)"
        alt "allocation fails"
            HBM-->>Core: None
            Note over DP: "Keep minimal lookup result for retry"
        else "allocation succeeds"
            HBM-->>Core: final blocks
            Core->>DP: "update_state_after_alloc(request, final blocks, E_DE)"
            DP->>DP: "consume lookup result; create DecodeKVSnapshot"
            Core->>Core: "status = WAITING_FOR_REMOTE_KVS"
        end
    end
```

No branch in this sequence submits to Proxy, starts Store load, builds a P2P
transfer, publishes `finished_recving`, or returns the request to `WAITING`.

## 3. Request selection

The new admission path applies only when both conditions are true:

1. `dual_path_cfg.role == "decode"`.
2. `request.kv_transfer_params` exists and
   `request.kv_transfer_params.get("do_remote_prefill") is True`.

Every other request delegates to the existing
`MooncakeLayerwiseConnectorScheduler` implementation. In particular:

- a Prefill-role DualPath connector preserves parent Layerwise behavior;
- a Decode-role request without `do_remote_prefill=True` preserves parent
  behavior;
- Task-01 does not reinterpret `do_remote_decode` or ordinary Layerwise send
  state.

This discriminator is required for behavior parity. Merely selecting
`DualPathConnector` must not route every request into the incomplete Task-01
admission path.

## 4. Token accounting

For an initial request:

```text
P    = request.num_tokens at initial scheduling
R    = max(P - 1, 0)
L_DE = num_computed_tokens passed by vLLM after local HBM lookup
E_DE = R - L_DE, when L_DE < R
K_DE = usable KVPool prefix recorded by a detached LoadSpec
```

The rules are:

| Condition | Return to vLLM Core | KVPool lookup | Task-01 state |
|---|---:|---|---|
| `L_DE >= R` | `(0, False)` | No | Clear stale unbound lookup state; create nothing |
| `L_DE < R` | `(E_DE, True)` | Yes, or reuse identical unbound result | Cache lookup result until allocation |

`E_DE` is the number of final Decode tokens that are not already present in
HBM. It is deliberately independent of the Store hit:

```text
Store full:    K_DE = R                 but Core still receives R - L_DE
Store partial: L_DE < K_DE < R          and Core still receives R - L_DE
Store miss:    no useful detached spec  and Core still receives R - L_DE
```

This makes vLLM allocate the final Decode destination blocks before any route
decision. Returning `K_DE - L_DE` would under-allocate partial and miss cases
and is forbidden.

The existing KVPool implementation remains authoritative for key generation,
cache-transfer alignment, `discard_partial_chunks`, multi-group intersection,
and the full-prompt last-token adjustment. Task-01 adds only a defensive target
clamp: any usable `K_DE` retained by DualPath must satisfy
`L_DE < K_DE <= R`.

## 5. Components and ownership

### 5.1 `KVPoolAdapter`

Add a private Decode Scheduler adapter under the DualPath package. Its public
surface within the package is:

```python
class KVPoolAdapter:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None: ...

    def lookup(
        self,
        request: Request,
        local_tokens: int,
    ) -> LoadSpec | None: ...

    def close(self) -> None: ...
```

Construction requirements:

- Own one dedicated `KVPoolScheduler` instance.
- Construct it with `use_layerwise=False` regardless of the Mooncake
  Layerwise P2P implementation inherited by DualPath.
- Use the first KV cache group's `page_size_bytes`, matching
  `AscendStoreConnector` construction.
- Do not share this scheduler or its `load_specs` with another connector.

`lookup()` must:

1. Call the existing
   `KVPoolScheduler.get_num_new_matched_tokens(request, local_tokens)`.
2. Ignore the returned async flag; DualPath owns the Core-facing async result.
3. Immediately remove `request.request_id` from the dedicated
   `KVPoolScheduler.load_specs` and take exclusive ownership of that raw
   object.
4. If a spec exists, verify the original Store delta equals
   `spec.kvpool_cached_tokens - spec.vllm_cached_tokens` before applying the
   target clamp.
5. Verify `spec.vllm_cached_tokens == local_tokens`.
6. Normalize the candidate with
   `usable_store_tokens = min(spec.kvpool_cached_tokens,
   max(request.num_tokens - 1, 0))`. If the value changes, return a
   `dataclasses.replace()` copy that preserves `can_load` and `token_len`;
   never mutate the raw object in place.
7. Return `None` for miss, `usable_store_tokens <= local_tokens`, or lookup
   error.
8. Leave no `LoadSpec` for the request inside the owned `KVPoolScheduler` on
   success, miss, or exception.

The detached `LoadSpec` is a candidate fact. `can_load` does not authorize I/O
in Task-01 and must not be changed to `True` by this adapter.

`close()` is idempotent and closes the lazily created `LookupKeyClient`, if
present. It does not start or stop Store transfer threads because none are
owned by the Scheduler adapter.

### 5.2 Pre-allocation lookup cache

`DualPathConnectorScheduler` owns:

```python
_lookup_results: dict[str, tuple[int, LoadSpec | None]]
```

The tuple is exactly:

```text
(external_tokens, detached_store_load_spec)
```

It exists only because vLLM separates lookup and allocation into two
callbacks. It is not a handle, has no `commit()` or `abort()` method, and owns
no active load or transfer.

The map is Scheduler-thread-confined. It is written by
`get_num_new_matched_tokens()` and consumed by
`update_state_after_alloc()`.

### 5.3 `DecodeKVSnapshot`

Create the complete record only after final allocation succeeds:

```python
@dataclass(frozen=True)
class DecodeKVSnapshot:
    target_tokens: int
    external_tokens: int
    store_load_spec: LoadSpec | None
    final_block_ids: tuple[tuple[int, ...], ...]

    @property
    def local_tokens(self) -> int:
        return self.target_tokens - self.external_tokens

    @property
    def store_tokens(self) -> int:
        if self.store_load_spec is None:
            return self.local_tokens
        return self.store_load_spec.kvpool_cached_tokens
```

`DualPathConnectorScheduler` owns admissions in:

```python
_decode_kv_snapshots: dict[str, DecodeKVSnapshot]
```

The field choices are intentional:

- `local_tokens` is derived rather than duplicated with
  `LoadSpec.vllm_cached_tokens`.
- `store_tokens` is derived rather than duplicated with
  `LoadSpec.kvpool_cached_tokens`.
- `target_tokens` has no equivalent in `LoadSpec` and is retained.
- `final_block_ids` is mandatory because the record does not exist before
  allocation.

`LoadSpec` is mutable in the existing AscendStore implementation, but the
detached instance is exclusively owned by this admission after binding.
Task-01 does not mutate it. A later Task must explicitly move or copy it when
authorizing a Store load.

### 5.4 Worker lookup service

The Decode Worker side owns a lookup-only adapter with this internal surface:

```python
class KVPoolWorkerAdapter:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None: ...

    def close(self) -> None: ...
```

Requirements:

- It is constructed only for `dual_path_cfg.role == "decode"`.
- It reuses `KVPoolWorker(..., use_layerwise=False, ...)` and the existing
  `LookupKeyServer`.
- Only the worker rank that satisfies the existing AscendStore lookup-server
  ownership rule binds the lookup endpoint.
- It does not call `register_kv_caches()`, start receive/send threads, build
  Store load metadata, or expose Store completion.
- `close()` is idempotent and terminates the lookup server cleanly.

The existing `LookupKeyServer.close()` must be made lifecycle-safe for this
first explicit owner: stop accepting requests, unblock its receive loop,
join its daemon thread, close the socket with zero linger, and terminate its
ZMQ context. This is the only Task-01 behavior change permitted in the shared
AscendStore connector file; the lookup protocol and response format remain
unchanged.

## 6. Configuration contract

Deployment explicitly sets:

```yaml
consumer_is_to_load: true
```

Task-01 must add the following existing KVPool settings to the DualPath
extra-config allowlist and pass them through unchanged:

| Key | Validation | Purpose in Task-01 |
|---|---|---|
| `consumer_is_to_load` | boolean | Enables consumer-side KVPool lookup |
| `backend` | non-empty string | Selects the existing KVPool backend |
| `lookup_rpc_port` | non-negative integer | Selects the lookup IPC endpoint |
| `mooncake_rpc_port` | non-negative integer | Existing deprecated endpoint fallback |
| `discard_partial_chunks` | boolean | Preserves existing lookup alignment policy |

These remain pass-through KVPool settings rather than new fields owned by
`DualPathConfig`. In particular, `consumer_is_to_load` is neither forced nor
given a new default by the adapter.

For a Decode deployment that expects Store probing,
`consumer_is_to_load` must be `True`. If it is absent or `False`, the existing
KVPool policy returns no Store hit; Task-01 still admits the request using
`E_DE` and records `store_load_spec=None`.

`use_layerwise` is not introduced as a DualPath Store option. Task-01's KVPool
lookup is always non-layerwise. This does not change the inherited Mooncake
Layerwise Forward/Reverse runtime.

No new environment variable is introduced. `load_async` is not admitted as a
Task-01 KVPool setting because DualPath, rather than the owned
`KVPoolScheduler`, supplies the Core-facing asynchronous admission result.
Other Store/backend configuration continues to use the current AscendStore
mechanisms.

## 7. Scheduler callback behavior

### 7.1 `get_num_new_matched_tokens()`

For a selected DualPath Decode request:

1. Calculate `R = max(request.num_tokens - 1, 0)`.
2. Validate `0 <= L_DE <= R` for the initial admission path.
3. If `L_DE >= R`, remove any stale `_lookup_results[request_id]` and return
   `(0, False)` without calling `KVPoolAdapter`.
4. Calculate `E_DE = R - L_DE`.
5. If an unbound entry already exists with the same `E_DE`, reuse it and do
   not repeat the KVPool lookup.
6. If an unbound entry exists with a different `E_DE`, discard it, perform a
   fresh lookup, and replace it.
7. Store `(E_DE, detached_spec_or_none)` and return `(E_DE, True)`.

An existing `DecodeKVSnapshot` for the same request ID makes a new initial
lookup invalid. Identical duplicate binding is handled in the allocation
callback; re-probing an already admitted request is a lifecycle error.

### 7.2 Allocation failure and retry

When `KVCacheManager.allocate_slots()` returns `None`, vLLM does not call
`update_state_after_alloc()`. Therefore:

- `_lookup_results[request_id]` remains owned by the Scheduler;
- `_decode_kv_snapshots` has no entry for the request;
- the next scheduling attempt with the same `E_DE` reuses the detached result;
- no Store lookup, LoadSpec creation, or side effect is duplicated.

This retained value is a lookup snapshot, not an active Store resource, so no
abort action is required between retries.

### 7.3 `update_state_after_alloc()`

For a selected DualPath Decode request:

1. Freeze `blocks.get_block_ids()` as
   `tuple(tuple(group) for group in block_ids_by_group)`.
2. If an admission already exists, accept the call only when target,
   `num_external_tokens`, and frozen blocks are identical; return without
   changing state. A conflicting duplicate raises `RuntimeError`.
3. Pop the request's entry from `_lookup_results`; absence is a lifecycle
   error.
4. Validate `num_external_tokens == cached_external_tokens == R - L_DE`.
5. If a detached spec exists, validate
   `spec.vllm_cached_tokens == L_DE` and
   `L_DE < spec.kvpool_cached_tokens <= R`.
6. Construct `DecodeKVSnapshot` and insert it into `_decode_kv_snapshots`.
7. Do not call the parent `update_state_after_alloc()` for this request.

Step 7 prevents the parent from populating `_reqs_need_recv`, mutating
`do_remote_prefill`, submitting its current metaserver request, or creating
Worker P2P metadata. Non-selected requests still delegate to the parent.

After this callback returns, the existing vLLM Scheduler code sets the request
to `WAITING_FOR_REMOTE_KVS` because Task-01 returned a positive external token
count with `load_kv_async=True`.

## 8. Metadata and side-effect boundary

Task-01 introduces no new `KVConnectorMetadata` type. It continues to use the
parent `MooncakeLayerwiseConnectorMetadata`.

For a Task-01 admission:

- `build_connector_meta()` contains no entry for the admitted request;
- a zero-token engine step may still execute the normal connector lifecycle,
  but the Worker sees no Store-load or P2P task for the request;
- `start_load_kv()`, Layerwise wait/save hooks, and `get_finished()` retain
  parent behavior for ordinary requests only;
- Task-01 never emits the admitted request in `finished_recving`.

The `DecodeKVSnapshot` is Scheduler-only state. It is not serialized to Worker
metadata until a later Task defines a committed data plan.

## 9. Cleanup and terminal behavior

The Scheduler owns all Task-01 request records and removes both
`_lookup_results[request_id]` and `_decode_kv_snapshots[request_id]` on:

- `request_finished()`;
- `request_finished_all_groups()`;
- cancellation/abort as observed through either finish callback;
- connector shutdown.

Cleanup is idempotent. Task-01 never delays block freeing and preserves the
parent return value from both finish callbacks.

Shutdown order is:

1. stop accepting new Task-01 work;
2. clear unbound lookup results and admissions;
3. close the Scheduler `KVPoolAdapter` or Worker `KVPoolWorkerAdapter` owned by
   that connector role;
4. invoke parent/base shutdown behavior.

There is no drain phase in Task-01 because no Store load or P2P transfer can
have been submitted.

## 10. Error policy

Task-01 distinguishes programming/lifecycle errors from Store availability:

| Condition | Required behavior |
|---|---|
| Store lookup returns no useful prefix | Treat as `store_load_spec=None`; admit with `E_DE` |
| Worker Store lookup catches backend error and returns zero | Treat as miss; admit with `E_DE` |
| Scheduler adapter lookup raises | Log once, remove any owned `load_specs` entry, treat as miss, admit with `E_DE` |
| Detached spec token invariants fail | Raise `RuntimeError`; do not create admission |
| Missing lookup result at first bind | Raise `RuntimeError`; do not create admission |
| Conflicting duplicate bind | Raise `RuntimeError`; preserve the original admission |
| Allocation failure | Retain minimal lookup result for retry; create no admission |

Task-01 intentionally does not preserve a separate Store-error bit. Until a
later route decision needs typed availability, Store error and Store miss have
the same safe meaning: no usable Decode Store candidate.

## 11. Focused test contract

### 11.1 `KVPoolAdapter` unit tests

Tests must cover:

- full hit: returned detached spec has `K_DE == R`;
- partial hit: returned detached spec has `L_DE < K_DE < R`;
- miss and hit not extending HBM: return `None`;
- full-prompt lookup is reduced to the Decode-ready `R` boundary;
- alignment and `discard_partial_chunks` behavior is inherited unchanged;
- multi-group lookup uses the existing minimum/common usable prefix;
- adapter ignores the KVPool async flag;
- Store delta validation;
- lookup exception becomes `None`;
- every outcome leaves the dedicated `load_specs` without the request ID;
- `close()` before and after lazy client creation is idempotent.

Existing `KVPoolScheduler` tests remain unchanged except where a reusable
fixture is imported; Task-01 must not rewrite its lookup algorithm.

### 11.2 Decode Scheduler unit tests

Tests must cover:

- HBM complete returns `(0, False)`, performs zero adapter calls, and creates
  no state;
- full, partial, and miss all return the same `E_DE = R - L_DE` for identical
  `R` and `L_DE`;
- the first incomplete lookup creates only `_lookup_results`;
- successful allocation consumes `_lookup_results` and creates exactly one
  `DecodeKVSnapshot` with frozen final block IDs;
- `local_tokens` and `store_tokens` derivation;
- identical duplicate lookup performs one underlying Store lookup;
- changed `E_DE` replaces the unbound lookup result;
- allocation-failure retry reuses the unbound result;
- identical duplicate bind is idempotent;
- conflicting duplicate bind fails without replacing the first admission;
- selected Decode requests do not populate parent `_reqs_need_recv` and do not
  submit the parent metaserver call;
- ordinary Decode and Prefill Layerwise requests still delegate to the parent;
- both finish callbacks and shutdown remove all Task-01 state.

### 11.3 Worker lookup service tests

Tests must cover:

- Decode role creates the non-layerwise lookup service on the owning worker
  rank;
- non-owning ranks and Prefill role do not bind a duplicate endpoint;
- request/response lookup uses the existing protocol;
- the service creates no KV receive/send thread and no connector metadata;
- repeated `close()` terminates the lookup thread and releases the endpoint.

### 11.4 Scheduler integration acceptance

A CPU-focused integration test must use the real vLLM Scheduler call order,
with Store and allocation dependencies constrained or faked at their existing
seams. For `L_DE < R`, one `schedule()` call must prove:

1. the connector returned `(R - L_DE, True)`;
2. `allocate_slots()` received `num_external_computed_tokens=R-L_DE` and
   `delay_cache_blocks=True`;
3. final Decode block IDs exist and equal the snapshot's frozen IDs;
4. request status is `WAITING_FOR_REMOTE_KVS`;
5. `request.num_computed_tokens == R` as set by vLLM for the pending async
   receive;
6. no model tokens were scheduled for the request in that step;
7. connector metadata contains no Store or P2P work for the request;
8. no `finished_recving` completion is published.

A paired HBM-complete case must prove normal local scheduling and last-token
recomputation without KVPool lookup or Task-01 state.

## 12. File-level change boundary

Task-01 is expected to touch only:

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
  for Decode callback overrides, snapshot ownership, cleanup, and adapter
  composition;
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/kvpool_adapter.py`
  for Scheduler and Worker KVPool adapters;
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`
  to accept and validate the existing lookup-only KVPool pass-through keys;
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
  only for lifecycle-safe `LookupKeyServer.close()`;
- focused tests under
  `tests/ut/distributed/kv_transfer/dual_path/`;
- an existing vLLM Scheduler test seam or a focused vLLM-Ascend Scheduler test
  file for the admission acceptance case.

Task-01 must not modify `KVPoolScheduler.get_num_new_matched_tokens()`,
`KVPoolWorker.lookup_scheduler()`, parent Mooncake Layerwise scheduling, Proxy,
Coordinator, or Worker transfer code.

## 13. Review gates

Task-01 is acceptable only if all of the following are true:

- the adapter reuses current KVPool lookup behavior rather than copying it;
- no shared `KVPoolScheduler.load_specs` entry survives adapter lookup;
- no handle, public `StoreCoverage`, or pre-allocation admission object exists;
- partial and miss cases allocate `R - L_DE`, not `K_DE - L_DE`;
- `DecodeKVSnapshot` exists only after real block allocation;
- the request reaches and remains in `WAITING_FOR_REMOTE_KVS`;
- no Proxy, Store load, P2P transfer, or completion path is accidentally
  activated;
- ordinary Layerwise behavior-parity tests remain green;
- focused cleanup tests prove no Scheduler record, lookup client, lookup
  endpoint, or background lookup thread leaks at terminal cleanup.

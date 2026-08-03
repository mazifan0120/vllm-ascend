# PR-00 DualPathConnector Foundation Detailed Spec

## 1. Status and authority

- Parent PR contract: [../PR-00-foundation.md](../PR-00-foundation.md)
- Series tracker: [../../TRACKING.md](../../TRACKING.md)
- Architecture baseline:
  [../../2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md](../../2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md)
- Scope: PR-00 only
- Implementation status: the existing foundation commit is a baseline to clean
  up and verify, not accepted merge evidence

The parent PR contract owns the stable PR boundary. This document owns the
implementation-level contract. `TRACKING.md` owns commands, environment details,
results, and external evidence links.

## 2. Objective

PR-00 introduces a selectable `DualPathConnector` that is a behavior-preserving
alias of `MooncakeLayerwiseConnector` for ordinary Layerwise workloads. It creates
safe Scheduler and Worker subclass seams for later PRs without adding a DualPath
decision or data path.

The foundation is complete only when the alias runs successfully on NPU and its
observable request behavior matches the parent connector.

## 3. Terminology and naming

The user-facing foundation role is:

```python
Literal["prefill", "decode"]
```

`pe` and `de` are not accepted configuration aliases. Protocol formulas and
later path results may continue to use PE/DE abbreviations, such as `L_PE`,
`L_DE`, `DE_LOCAL_FULL_HIT`, and `DE_PARTIAL_HIT`; those are protocol terms, not
foundation role values.

PR-00 is described as `foundation` or `behavior-preserving alias`. It is not
described as `shadow mode`: no decision computation, observer, shadow metric, or
request-local shadow result exists in this PR.

## 4. Merge-state contract

Immediately after merge:

1. The connector registry resolves `DualPathConnector`.
2. Scheduler construction creates `DualPathConnectorScheduler` only.
3. Worker construction creates `DualPathConnectorWorker` only.
4. Ordinary Layerwise requests use the inherited accounting, allocation,
   metadata, transfer, completion, invalid-block, cleanup, and failure paths.
5. The parent Scheduler or Worker and all of its runtime resources initialize
   exactly once.
6. The connector emits no additional network request, Store operation, P2P
   operation, thread, background task, or completion event.
7. Switching the same workload configuration back to
   `MooncakeLayerwiseConnector` changes no request semantics.

## 5. Strict non-goals

PR-00 does not define or retain placeholders for:

- Store probe, Store load, coverage, token accounting, or local-full admission;
- `PathDecisionRequest`, `PathDecisionCommit`, decision Future, decision inbox,
  callback endpoint, or Coordinator;
- round-robin selection, Value Function, LinkMonitor, adaptive policy, or shadow
  observation;
- `_req_path` or any request-local path side table;
- `DE_PARTIAL_HIT`, positive first-winner accounting, or MultiConnector changes;
- DualPath transfer plans, metadata subclasses, Reverse, or changed Forward
  semantics;
- raw-completion reconciliation, terminal tombstones, execution timeout, or
  delayed-free state;
- future Stage 1 topology restrictions that ordinary Layerwise execution does
  not need.

Future types, configuration fields, callbacks, and state are added only by the
first later PR with a real producer, consumer, and complete tests.

## 6. Code ownership and file surface

| File | PR-00 responsibility |
|---|---|
| `vllm_ascend/distributed/kv_transfer/__init__.py` | Lazy connector registration only |
| `dual_path/__init__.py` | Package description and lazy import boundary only |
| `dual_path/config.py` | Foundation role parsing and fail-fast validation only |
| `dual_path/connector.py` | Facade and two behavior-preserving subclasses |
| `test_dual_path_connector.py` | Config, construction, inheritance, and CPU parity tests |
| `test_foundation_parity.py` | Real NPU parent-versus-alias smoke test |

The shortened `dual_path/` paths in this table are relative to
`vllm_ascend/distributed/kv_transfer/kv_p2p/`. PR-00 does not modify Store,
Proxy, MultiConnector, or `mooncake_layerwise_connector.py` implementation code.

## 7. Foundation configuration contract

### 7.1 Parsed configuration

```python
@dataclass(frozen=True)
class DualPathConfig:
    role: Literal["prefill", "decode"]
```

`DualPathConfig` contains no strategy, monitor, relay, planner, topology, Store,
timeout, endpoint, or shadow field.

### 7.2 Accepted extra-config keys

The foundation recognizes `role` and preserves the existing extra-config keys
consumed by the inherited Mooncake implementation:

```text
role
tls_config
prefill
decode
```

The `prefill` and `decode` mapping keys in this list are inherited Mooncake
parallel/runtime configuration. They are distinct from the string stored in the
top-level `role` field.

All other keys fail fast in PR-00. In particular, the implementation rejects
the removed foundation fields `path_strategy`, `relay`, `path_planner`,
`monitor`, `topology`, `enable_value_function_shadow`, and
`enable_link_monitor_shadow`.

If the parent Mooncake connector later adds a required extra-config key, the
compatibility guard must fail until this allow-list and its parity tests are
reviewed together.

### 7.3 Role and capability validation

| `role` | Required KV capability | Allowed `kv_role` examples |
|---|---|---|
| `prefill` | producer | `kv_producer`, `kv_both` |
| `decode` | consumer | `kv_consumer`, `kv_both` |

Missing role, `pe`, `de`, unknown role values, and role/capability mismatches
raise `ValueError` during construction. PR-00 does not add PP, DP, TP, cache
layout, model, dtype, or cross-process topology validation.

## 8. Connector construction contract

### 8.1 Facade initialization

`DualPathConnector` inherits `MooncakeLayerwiseConnector`, but it does not call
the parent facade `__init__` because that method hard-constructs the parent
Scheduler or Worker. It calls `KVConnectorBase_V1.__init__` exactly once, creates
the same facade state, and constructs the corresponding DualPath subclass.

The complete copied facade state is:

```text
_is_kv_producer
engine_id
_connector_metadata
connector_scheduler
connector_worker
```

No other parent setup may be copied without updating this contract and the
drift guard. `_connector_metadata` remains an unmodified
`MooncakeLayerwiseConnectorMetadata` instance.

### 8.2 Scheduler and Worker initialization

```python
class DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler):
    dual_path_cfg: DualPathConfig


class DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker):
    dual_path_cfg: DualPathConfig
```

Each subclass calls its parent `__init__` exactly once and then stores the frozen
configuration. The Scheduler has no `_req_path`. The Worker starts no
LinkMonitor, extra thread, endpoint, runtime, or background task.

### 8.3 Initialization failure rule

If direct base initialization cannot be proven equivalent through construction
parity and the NPU smoke test, implementation stops for design review. PR-00 may
not compensate by copying additional parent implementation or by constructing a
parent object and replacing its Scheduler or Worker after initialization.

## 9. Inheritance contract

Only `__init__` may be overridden in the three DualPath classes. The following
facade lifecycle methods remain the exact inherited parent method objects:

```text
get_num_new_matched_tokens
update_state_after_alloc
build_connector_meta
request_finished
request_finished_all_groups
register_kv_caches
get_finished
get_block_ids_with_load_errors
start_load_kv
wait_for_layer_load
save_kv_layer
wait_for_save
```

The Scheduler and Worker subclasses likewise add no request lifecycle override.
A method-ownership guard and parent-signature snapshot fail on upstream drift.

## 10. Error behavior

PR-00 adds errors only for invalid foundation configuration and unsupported
connector construction roles. Parent Scheduler, Worker, Mooncake runtime,
network, transfer, and completion errors propagate with the same type and timing
as the parent connector. The alias does not wrap, downgrade, retry, fall back, or
switch connectors after a parent error.

## 11. CPU verification contract

### 11.1 Configuration tests

Tests cover:

- valid `prefill` producer and `decode` consumer;
- both foundation roles with `kv_both`;
- missing or unknown role;
- rejected legacy `pe` and `de` values;
- role/capability mismatch;
- preserved inherited `tls_config`, `prefill`, and `decode` mappings;
- rejected removed DualPath foundation fields;
- frozen configuration state.

### 11.2 Construction tests

Spy-based tests prove:

- `KVConnectorBase_V1.__init__` runs once;
- the selected parent Scheduler or Worker `__init__` runs once;
- the opposite process-side object is absent;
- facade fields match the equivalent parent construction;
- metadata type and initial contents match;
- no request side table, monitor, extra runtime, or extra thread exists;
- runtime, Transfer Engine, and KV-buffer registration events occur once.

Mocking the complete parent Scheduler or Worker constructor to a no-op is
insufficient merge evidence; such a helper may be used only by tests that do not
claim construction parity.

### 11.3 Method and behavior parity

Equivalent parent and DualPath instances receive the same inputs. Tests compare
return values, structured metadata, state transitions, and collaborator call
traces for:

| Scenario | Required parity evidence |
|---|---|
| No remote transfer | matched tokens, empty metadata, zero extra calls |
| Decode remote prefill | matched tokens, `load_async`, block IDs, metaserver payload |
| Prefill remote decode | send metadata, layer order, chunk state, terminal callback |
| Attention/Mamba hybrid | prompt truncation and remote block trimming |
| Allocation | request state, blocks, and external-token arguments |
| Worker load | receive registration, load task, and thread calls |
| Worker save | layer order, send task, and final-chunk behavior |
| Completion | finished sets and invalid blocks |
| Cleanup | pending state and request removal |
| Failure | exception type, propagation, and invalid-block result |

The tests do not treat matching method names or a single call count as behavior
parity when structured output or observable state exists.

### 11.4 Focused commands

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
pytest -sv tests/ut/kv_offload/test_mooncake_layerwise_connector.py
bash format.sh ci
```

## 12. Mandatory NPU parity smoke

The NPU smoke test uses the same model, prompt, Mooncake endpoints, process
topology, and parallel settings in two runs:

1. baseline with `MooncakeLayerwiseConnector`;
2. candidate with `DualPathConnector` and `role=prefill` or `role=decode` on the
   corresponding Engine.

The test must exercise a real remote-prefill transfer and verify:

- identical output tokens;
- identical matched-token and request-terminal outcomes;
- no Store operation, decision RPC, round-robin selection, or Reverse transfer;
- one Transfer Engine and one KV-buffer registration sequence per Worker;
- no extra send/receive thread relative to the baseline role;
- no invalid blocks, leaked request state, or abnormal process exit.

The test lives at:

```text
tests/e2e/nightly/multi_node/dual_path/test_foundation_parity.py
```

The exact command, image, model, hardware topology, baseline revision, and
PASS/FAIL result are recorded in `TRACKING.md`. CPU results cannot substitute for
this gate.

## 13. Acceptance checklist

- [ ] Registry lookup resolves the connector lazily.
- [ ] The only DualPath-owned foundation config field is `role`.
- [ ] User-facing role values are exactly `prefill` and `decode`.
- [ ] Removed foundation and shadow fields fail fast.
- [ ] `_is_kv_producer` and all other copied facade state match the parent.
- [ ] Base, Scheduler/Worker, runtime, Transfer Engine, threads, and buffers are
      initialized exactly once.
- [ ] No request lifecycle method is overridden.
- [ ] CPU configuration, construction, behavior, error, and regression tests
      pass.
- [ ] The mandatory NPU parity smoke passes.
- [ ] No Store, decision, round-robin, Reverse, or new Forward behavior is
      reachable.
- [ ] `bash format.sh ci` passes.

## 14. Rollback and review stop conditions

Reverting PR-00 removes the registry entry, package, configuration, subclasses,
and tests without changing the parent connector.

Implementation returns to design review if any of the following occurs:

- a parent implementation file must change;
- direct base initialization requires copied state not listed in Section 8.1;
- behavior parity requires a lifecycle override;
- a second runtime, Transfer Engine, buffer registration, or thread appears;
- the NPU smoke differs in output, completion, invalid blocks, or resource
  lifetime;
- a later-PR configuration field is required merely to make the foundation run.

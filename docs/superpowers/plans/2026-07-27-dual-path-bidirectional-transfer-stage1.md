# DualPath Stage 1 Bidirectional Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Stage 1 Forward/Reverse data-transfer slice of
`DualPathConnector` with one inherited Mooncake layerwise runtime, frozen
block-pair plans, Store-gated Reverse, model-driven Forward, and race-safe
request completion.

**Architecture:** Preserve the public `MooncakeLayerwiseConnector` contract
while extracting behavior-preserving protected helpers from its Worker.
`DualPathConnectorWorker` enables both inherited threads, composes a bulk
`KVPoolWorker` only on DE, and executes immutable directional plans produced by
the Scheduler. A receive-side reconciler converts raw wire terminals to
Engine-local request IDs, retaining early completions and terminal tombstones.

**Tech Stack:** Python 3.10+, vLLM v1 KV connector interfaces,
`MooncakeLayerwiseConnectorWorker`, `KVCacheSendingLayerThread`,
`KVCacheRecvingLayerThread`, AscendStore `KVPoolWorker`, `msgspec`, ZMQ,
PyTorch/NPU, `pytest`, and `unittest.mock`.

## Global Constraints

- The design baseline is
  `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md`.
- This plan implements the data-plane and Worker/Scheduler metadata slice; the
  Proxy callback and path-decision transport are separate work.
- `PathDecisionCommit` is an immutable input. A Worker never recomputes or
  changes the decision.
- `DE_LOCAL_FULL_HIT` creates no candidate, no `PathDecisionCommit`, no PE
  request, and no Forward/Reverse plan.
- Partial admission guarantees `L_PE < K_DE < R`; both Reverse and Forward
  contain at least one real block pair.
- PE waits for request-level Reverse DONE before model execution. Stage 1 does
  not overlap Reverse with PE compute.
- Forward starts only from the existing per-layer `save_kv_layer()` callback
  and sends only newly completed aligned blocks.
- One `DualPathConnectorWorker` owns one Mooncake send thread and one Mooncake
  receive thread. It does not construct a second Mooncake Worker or mutate
  `kv_transfer_config` at runtime.
- Modify `MooncakeLayerwiseConnectorWorker` only through behavior-preserving
  protected helper extraction. Do not change the
  `MooncakeLayerwiseConnector` facade or ordinary Layerwise behavior.
- Scheduler-frozen `BlockPair` values are authoritative. Worker code must not
  re-zip independent block arrays, reselect blocks, or narrow by token count.
- Install every inbound binding in a Worker batch before evaluating any
  outbound gate.
- Receive thread output remains raw wire IDs. Only the Worker publishes
  Engine-local request IDs to `finished_recving`.
- Request-level DONE and FAILED are mutually exclusive and published at most
  once per direction.
- A terminal request submits no new Store or transport work. Blocks remain
  retained until already submitted work succeeds, fails, or times out and is
  quiesced.
- New functionality requires unit tests for happy paths, race conditions,
  duplicate/conflicting terminals, failure, cancellation, and timeout.
- Commits use Conventional Commits and include `Signed-off-by` via
  `git commit -s`.

## Planned File Structure

| File | Responsibility |
|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` | Extract parent thread and metadata helpers without changing ordinary behavior |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py` | Immutable identities, coverage, block pairs, direction plans, bindings, Worker plans, and validation |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py` | Direction gates, Forward frontier, plan-to-parent conversion, raw completion reconciliation, and request runtime state |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py` | DE-only composition of the existing bulk `KVPoolWorker` |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py` | Scheduler queueing and facade/Worker hook integration |
| `tests/ut/kv_offload/test_mooncake_layerwise_connector.py` | Parent behavior-regression coverage |
| `tests/ut/distributed/kv_transfer/dual_path/test_metadata.py` | Plan and block-pair validation |
| `tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py` | Gates, ordering, frontier, pending completion, tombstones, and drain state |
| `tests/ut/distributed/kv_transfer/dual_path/test_store_adapter.py` | Store bulk-load composition and local-full behavior |
| `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py` | End-to-end Scheduler/Worker hook integration |
| `tests/e2e/nightly/multi_node/dual_path/test_dual_path_transfer.py` | NPU multi-process Forward/Reverse validation |

---

### Task 1: Extract Behavior-Preserving Mooncake Worker Helpers

**Files:**

- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:1127`
- Modify:
  `tests/ut/kv_offload/test_mooncake_layerwise_connector.py:1051`

**Interfaces:**

- Consumes: Current `MooncakeLayerwiseConnectorWorker.register_kv_caches()`,
  `start_load_kv()`, `KVCacheSendingLayerThread`, and
  `KVCacheRecvingLayerThread`.
- Produces:
  `_needs_send_thread() -> bool`,
  `_needs_receive_thread() -> bool`,
  `_ensure_send_thread_started() -> None`,
  `_ensure_receive_thread_started() -> None`,
  `_bind_receive_metadata(MooncakeLayerwiseConnectorMetadata) -> None`, and
  `_prepare_send_metadata(MooncakeLayerwiseConnectorMetadata) -> None`, and
  `_enqueue_send_layer(layer_name, kv_layer, attn_metadata, metadata) -> None`.

- [ ] **Step 1: Write parent regression tests before changing the Worker**

Add tests that instantiate an uninitialized Worker with `__new__`, patch the
two thread constructors, and assert the ordinary roles keep their current
thread and metadata-dispatch behavior:

```python
def _make_worker_without_runtime(
    self,
    *,
    is_kv_producer: bool,
    is_kv_consumer: bool,
) -> MooncakeLayerwiseConnectorWorker:
    worker = MooncakeLayerwiseConnectorWorker.__new__(
        MooncakeLayerwiseConnectorWorker
    )
    worker.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=is_kv_producer,
            is_kv_consumer=is_kv_consumer,
        )
    )
    return worker


def test_parent_producer_starts_only_send_thread(self):
    worker = self._make_worker_without_runtime(
        is_kv_producer=True,
        is_kv_consumer=False,
    )
    worker._ensure_send_thread_started = MagicMock()
    worker._ensure_receive_thread_started = MagicMock()

    worker._start_required_layer_threads()

    worker._ensure_send_thread_started.assert_called_once_with()
    worker._ensure_receive_thread_started.assert_not_called()


def test_parent_consumer_binds_receive_metadata_only(self):
    worker = self._make_worker_without_runtime(
        is_kv_producer=False,
        is_kv_consumer=True,
    )
    metadata = MooncakeLayerwiseConnectorMetadata()
    worker._bind_receive_metadata = MagicMock()
    worker._prepare_send_metadata = MagicMock()

    worker.start_load_kv(metadata)

    worker._bind_receive_metadata.assert_called_once_with(metadata)
    worker._prepare_send_metadata.assert_not_called()


def test_parent_save_delegates_to_layer_enqueue(self):
    worker = self._make_worker_without_runtime(
        is_kv_producer=True,
        is_kv_consumer=False,
    )
    worker._enqueue_send_layer = MagicMock()
    metadata = MooncakeLayerwiseConnectorMetadata()
    metadata.requests["request"] = MagicMock()

    worker.save_kv_layer("layer.0", [], None, metadata)

    worker._enqueue_send_layer.assert_called_once_with(
        "layer.0",
        [],
        None,
        metadata,
    )
```

- [ ] **Step 2: Run the tests and verify they fail because the helpers do not
  exist**

Run:

```bash
pytest -sv \
  tests/ut/kv_offload/test_mooncake_layerwise_connector.py \
  -k "parent_producer_starts_only_send_thread or parent_consumer_binds_receive_metadata_only"
```

Expected: FAIL with missing helper attributes.

- [ ] **Step 3: Extract the role predicates and thread dispatch**

Add these methods to `MooncakeLayerwiseConnectorWorker`:

```python
def _needs_send_thread(self) -> bool:
    return self.vllm_config.kv_transfer_config.is_kv_producer

def _needs_receive_thread(self) -> bool:
    return self.vllm_config.kv_transfer_config.is_kv_consumer

def _start_required_layer_threads(self) -> None:
    if self._needs_send_thread():
        self._ensure_send_thread_started()
    if self._needs_receive_thread():
        self._ensure_receive_thread_started()
```

At the end of `register_kv_caches()`, save the already constructed
`MooncakeAgentMetadata` on `self._local_agent_metadata`, replace the two
inline role branches with `_start_required_layer_threads()`, and move each
existing constructor block unchanged into its corresponding
`_ensure_*_thread_started()` method. Each helper starts a thread only when its
field is `None` and waits on the existing `ready_event`.

- [ ] **Step 4: Extract receive binding and send preparation without changing
  parent dispatch**

Move the current consumer body of `start_load_kv()` into
`_bind_receive_metadata()` and the current producer body into
`_prepare_send_metadata()`. Keep the parent's consumer-first `if/elif`:

```python
def start_load_kv(
    self,
    metadata: MooncakeLayerwiseConnectorMetadata,
) -> None:
    self.current_layer = 0
    if self.vllm_config.kv_transfer_config.is_kv_consumer:
        self._bind_receive_metadata(metadata)
    elif self.vllm_config.kv_transfer_config.is_kv_producer:
        self._prepare_send_metadata(metadata)
```

This preserves current `kv_both` dispatch in the parent. DualPath will call
both protected helpers explicitly in Task 7.

- [ ] **Step 5: Extract the per-layer enqueue body**

Move the current body of `MooncakeLayerwiseConnectorWorker.save_kv_layer()`
after its producer/requests guard into `_enqueue_send_layer()`. Keep the
public parent method as a role-preserving wrapper:

```python
def save_kv_layer(
    self,
    layer_name: str,
    kv_layer: list[torch.Tensor],
    attn_metadata: AttentionMetadata | None,
    connector_metadata: MooncakeLayerwiseConnectorMetadata,
    **kwargs: Any,
) -> None:
    if (
        self.vllm_config.kv_transfer_config.is_kv_producer
        and connector_metadata.requests
    ):
        self._enqueue_send_layer(
            layer_name,
            kv_layer,
            attn_metadata,
            connector_metadata,
        )
```

The extracted helper keeps the current event handling, quantization,
resharding, `current_layer`, `SendTask`, queue, callback, and final-layer
semantics byte-for-byte except for indentation.

- [ ] **Step 6: Run the complete parent test module**

Run:

```bash
pytest -sv tests/ut/kv_offload/test_mooncake_layerwise_connector.py
```

Expected: PASS, including existing callback, mapping, address-planning, and
failure tests.

- [ ] **Step 7: Commit the refactor**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py \
  tests/ut/kv_offload/test_mooncake_layerwise_connector.py
git commit -s -m "refactor(kv-transfer): extract layerwise worker helpers"
```

---

### Task 2: Add Immutable Direction Plans and Validation

**Files:**

- Create:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- Create:
  `tests/ut/distributed/kv_transfer/dual_path/test_metadata.py`
- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/__init__.py`

**Interfaces:**

- Consumes: `MooncakeLayerwiseConnectorMetadata` and AscendStore
  `AscendConnectorMetadata`.
- Produces: `DualPathRequestKey`, `StoreCoverage`,
  `DualPathTransferIds`, `PathDecisionCommit`, `BlockPair`,
  `LayerwiseDirectionPlan`, `InboundRequestBinding`,
  `DualPathWorkerPlan`, `LocalFullHitWorkerPlan`,
  `RequestCompletionFacts`, `DualPathConnectorMetadata`,
  `freeze_block_pairs()`, and `validate_direction_plan()`.

- [ ] **Step 1: Write validation tests**

```python
def make_direction_plan(
    *,
    block_pairs: tuple[tuple[BlockPair, ...], ...],
) -> LayerwiseDirectionPlan:
    return LayerwiseDirectionPlan(
        direction=TransferDirection.FORWARD,
        wire_external_id="forward-wire",
        local_request_id="local-request-123456789",
        token_start=16,
        token_end=32,
        block_pairs=block_pairs,
        remote_block_size=(16,),
        remote_engine_id="remote-engine",
        remote_host="127.0.0.1",
        remote_port=5000,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
    )


def test_direction_plan_requires_non_empty_pairs():
    plan = make_direction_plan(block_pairs=((),))
    with pytest.raises(ValueError, match="at least one block pair"):
        validate_direction_plan(plan, expected_group_count=1)


def test_direction_plan_rejects_duplicate_destination():
    plan = make_direction_plan(
        block_pairs=(
            (
                BlockPair(local_block_id=1, remote_block_id=9),
                BlockPair(local_block_id=2, remote_block_id=9),
            ),
        )
    )
    with pytest.raises(ValueError, match="duplicate remote block"):
        validate_direction_plan(plan, expected_group_count=1)


def test_freeze_block_pairs_rejects_length_mismatch():
    with pytest.raises(ValueError, match="block count mismatch"):
        freeze_block_pairs(
            local_block_ids=((1, 2),),
            remote_block_ids=((10,),),
        )


def test_metadata_preserves_parent_schema():
    metadata = DualPathConnectorMetadata()
    assert metadata.requests == {}
    assert metadata.send_task.send_request == {}
    assert metadata.plans == ()
```

- [ ] **Step 2: Run the new test module and verify import failure**

Run:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_metadata.py
```

Expected: FAIL because `dual_path.metadata` does not exist.

- [ ] **Step 3: Define the immutable transport schema**

Implement the accepted dataclasses and enums in `metadata.py`. Use a normal
subclass initializer for parent metadata because the parent class is not a
dataclass:

```python
class DualPathConnectorMetadata(MooncakeLayerwiseConnectorMetadata):
    def __init__(
        self,
        plans: tuple[DualPathWorkerPlan | LocalFullHitWorkerPlan, ...] = (),
    ) -> None:
        super().__init__()
        self.plans = plans
```

Keep `store_metadata` typed as `AscendConnectorMetadata | None`, and keep the
Scheduler-side probe handle out of every Worker plan.

- [ ] **Step 4: Implement plan validation**

```python
def freeze_block_pairs(
    *,
    local_block_ids: tuple[tuple[int, ...], ...],
    remote_block_ids: tuple[tuple[int, ...], ...],
) -> tuple[tuple[BlockPair, ...], ...]:
    if len(local_block_ids) != len(remote_block_ids):
        raise ValueError("KV cache group count mismatch")
    frozen: list[tuple[BlockPair, ...]] = []
    for local_group, remote_group in zip(
        local_block_ids,
        remote_block_ids,
    ):
        if len(local_group) != len(remote_group):
            raise ValueError("source/destination block count mismatch")
        frozen.append(
            tuple(
                BlockPair(
                    local_block_id=local_id,
                    remote_block_id=remote_id,
                )
                for local_id, remote_id in zip(local_group, remote_group)
            )
        )
    return tuple(frozen)


def validate_direction_plan(
    plan: LayerwiseDirectionPlan,
    expected_group_count: int,
) -> None:
    if plan.token_start >= plan.token_end:
        raise ValueError("direction token range must be non-empty")
    if len(plan.block_pairs) != expected_group_count:
        raise ValueError("direction KV cache group count mismatch")
    if len(plan.remote_block_size) != expected_group_count:
        raise ValueError("remote block-size group count mismatch")

    pair_count = 0
    for group_pairs in plan.block_pairs:
        pair_count += len(group_pairs)
        remote_ids = [pair.remote_block_id for pair in group_pairs]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError("direction contains duplicate remote block")
    if pair_count == 0:
        raise ValueError("direction must contain at least one block pair")
```

Validate `PathDecisionCommit` in `__post_init__`: `use_dual_path=True` requires
`DE_PARTIAL_HIT`; `False` requires `dual_path_kind is None`.

- [ ] **Step 5: Run the metadata tests**

Run:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_metadata.py
```

Expected: PASS.

- [ ] **Step 6: Commit the schema**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/__init__.py \
  tests/ut/distributed/kv_transfer/dual_path/test_metadata.py
git commit -s -m "feat(kv-transfer): add DualPath direction plans"
```

---

### Task 3: Implement Race-Safe Raw Completion Reconciliation

**Files:**

- Create:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- Create:
  `tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py`

**Interfaces:**

- Consumes: `InboundRequestBinding` and `TransferDirection`.
- Produces: `RawTerminalKind`, `PendingRawCompletion`,
  `TerminalWireRecord`, `CompletionConflictError`, and
  `RawCompletionReconciler`.

- [ ] **Step 1: Write early, duplicate, conflict, and orphan tests**

```python
def test_raw_done_before_binding_is_published_after_binding():
    reconciler = RawCompletionReconciler()
    reconciler.ingest(
        TransferDirection.REVERSE,
        done={"reverse-wire"},
        failed=set(),
        now=10.0,
    )
    assert reconciler.publish_ready(now=10.1) == (set(), set())

    reconciler.install_binding(
        InboundRequestBinding(
            direction=TransferDirection.REVERSE,
            wire_external_id="reverse-wire",
            request_key=DualPathRequestKey(
                de_engine_incarnation="de-boot-1",
                de_engine_local_request_id="de-local",
            ),
            engine_local_request_id="pe-local",
        )
    )

    assert reconciler.publish_ready(now=10.2) == ({"pe-local"}, set())


def test_conflicting_terminal_raises_protocol_error():
    reconciler = RawCompletionReconciler()
    reconciler.ingest(
        TransferDirection.FORWARD,
        done={"forward-wire"},
        failed=set(),
        now=10.0,
    )
    with pytest.raises(CompletionConflictError):
        reconciler.ingest(
            TransferDirection.FORWARD,
            done=set(),
            failed={"forward-wire"},
            now=10.1,
        )


def test_orphan_expires_after_execution_timeout_plus_retry_slack():
    reconciler = RawCompletionReconciler()
    reconciler.ingest(
        TransferDirection.FORWARD,
        done={"orphan-wire"},
        failed=set(),
        now=5.0,
    )
    assert reconciler.expire_orphans(now=14.9, ttl=10.0) == set()
    assert reconciler.expire_orphans(now=15.0, ttl=10.0) == {"orphan-wire"}
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
pytest -sv \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py \
  -k "raw_done or conflicting_terminal or orphan"
```

Expected: FAIL because the reconciler is not defined.

- [ ] **Step 3: Implement pending and tombstone state**

Use `dict[wire_id, PendingRawCompletion]` rather than bare sets:

```python
@dataclass(frozen=True)
class PendingRawCompletion:
    direction: TransferDirection
    kind: RawTerminalKind
    first_seen_monotonic: float


@dataclass(frozen=True)
class TerminalWireRecord:
    direction: TransferDirection
    kind: RawTerminalKind
    engine_local_request_id: str
    published_at_monotonic: float
```

`RawCompletionReconciler` owns `active_inbound_bindings`,
`pending_raw_completions`, and `terminal_wire_tombstones`. `ingest()` is
idempotent for the same terminal and raises
`CompletionConflictError` for DONE/FAILED conflict.

- [ ] **Step 4: Implement Engine-local publication and cleanup**

```python
def publish_ready(self, now: float) -> tuple[set[str], set[str]]:
    done: set[str] = set()
    failed: set[str] = set()
    for wire_id, pending in tuple(
        self.pending_raw_completions.items()
    ):
        binding = self.active_inbound_bindings.get(wire_id)
        if binding is None:
            continue
        target = done if pending.kind is RawTerminalKind.DONE else failed
        target.add(binding.engine_local_request_id)
        self.terminal_wire_tombstones[wire_id] = TerminalWireRecord(
            direction=pending.direction,
            kind=pending.kind,
            engine_local_request_id=binding.engine_local_request_id,
            published_at_monotonic=now,
        )
        self.pending_raw_completions.pop(wire_id)
        self.active_inbound_bindings.pop(wire_id)
    return done, failed
```

Implement:

- `expire_orphans(now, ttl) -> set[str]`;
- `cleanup_tombstones(cleaned_local_ids, now, retry_window) -> set[str]`;
- a duplicate counter and conflict counter for later metrics.

- [ ] **Step 5: Run the full reconciler tests**

Run:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
```

Expected: PASS.

- [ ] **Step 6: Commit the reconciler**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
git commit -s -m "feat(kv-transfer): retain early DualPath completions"
```

---

### Task 4: Add Direction Gates, Batch Ordering, and Forward Frontier

**Files:**

- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- Modify:
  `tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py`

**Interfaces:**

- Consumes: `DualPathConnectorMetadata`, immutable directional plans, and
  `RawCompletionReconciler`.
- Produces: `can_start_reverse()`, `ForwardFrontier`,
  `RequestTransferState`, `DualPathLayerwiseRuntime.mark_store_done()`, and
  `DualPathLayerwiseRuntime.invalid_blocks_for()`.

- [ ] **Step 1: Write a four-request mapping-first test**

```python
def make_runtime(events: list[str]) -> DualPathLayerwiseRuntime:
    return DualPathLayerwiseRuntime(
        enqueue_reverse=lambda plan: events.append(
            f"reverse:{plan.local_request_id}"
        ),
        enqueue_forward=lambda plan, interval: events.append(
            f"forward:{plan.local_request_id}:{interval}"
        ),
    )


def make_partial_worker_plan(
    *,
    request_key: DualPathRequestKey,
    reverse: LayerwiseDirectionPlan,
    binding: InboundRequestBinding,
) -> DualPathWorkerPlan:
    forward = LayerwiseDirectionPlan(
        direction=TransferDirection.FORWARD,
        wire_external_id=binding.wire_external_id,
        local_request_id=request_key.de_engine_local_request_id,
        token_start=16,
        token_end=32,
        block_pairs=((BlockPair(200, 300),),),
        remote_block_size=(16,),
        remote_engine_id="de",
        remote_host="127.0.0.1",
        remote_port=6000,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
    )
    return DualPathWorkerPlan(
        ids=DualPathTransferIds(
            request_key=request_key,
            pe_engine_local_request_id="pe-local",
            forward_wire_external_id=binding.wire_external_id,
            reverse_wire_external_id=reverse.wire_external_id,
        ),
        decision=PathDecisionCommit(
            request_key=request_key,
            use_dual_path=True,
            dual_path_kind=DualPathKind.DE_PARTIAL_HIT,
        ),
        store_coverage=StoreCoverage(
            raw_hit_tokens=16,
            aligned_hit_tokens=16,
            local_tokens=0,
            ready_prefix_tokens=16,
            decode_ready_tokens=16,
            store_load_tokens=16,
            prefill_tail_tokens=16,
            is_full_hit=False,
        ),
        store_metadata=AscendConnectorMetadata(
            set(),
            set(),
            loading_req_ids={
                request_key.de_engine_local_request_id,
            },
        ),
        reverse=reverse,
        forward=forward,
        inbound_bindings=(binding,),
    )


def make_four_partial_plans() -> DualPathConnectorMetadata:
    plans: list[DualPathWorkerPlan] = []
    for index in range(1, 5):
        key = DualPathRequestKey(
            de_engine_incarnation="de-boot-1",
            de_engine_local_request_id=f"R{index}",
        )
        binding = InboundRequestBinding(
            direction=TransferDirection.FORWARD,
            wire_external_id=f"FWD_wire_{index}",
            request_key=key,
            engine_local_request_id=f"R{index}",
        )
        reverse = LayerwiseDirectionPlan(
            direction=TransferDirection.REVERSE,
            wire_external_id=f"REV_wire_{index}",
            local_request_id=f"R{index}",
            token_start=0,
            token_end=16,
            block_pairs=((BlockPair(index, index + 100),),),
            remote_block_size=(16,),
            remote_engine_id="pe",
            remote_host="127.0.0.1",
            remote_port=5000,
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
        )
        plan = make_partial_worker_plan(
            request_key=key,
            reverse=reverse,
            binding=binding,
        )
        plans.append(plan)
    return DualPathConnectorMetadata(plans=tuple(plans))


def test_batch_installs_all_forward_bindings_before_reverse_gate():
    events: list[str] = []
    runtime = make_runtime(events)
    runtime.reconciler.install_binding = MagicMock(
        side_effect=lambda binding: events.append(
            f"bind:{binding.wire_external_id}"
        )
    )
    metadata = make_four_partial_plans()

    runtime.bind_batch(metadata)
    runtime.mark_store_done(
        DualPathRequestKey("de-boot-1", "R1")
    )
    runtime.mark_store_done(
        DualPathRequestKey("de-boot-1", "R3")
    )

    assert events[:4] == [
        "bind:FWD_wire_1",
        "bind:FWD_wire_2",
        "bind:FWD_wire_3",
        "bind:FWD_wire_4",
    ]
    assert events[4:] == ["reverse:R1", "reverse:R3"]
```

Add tests proving that commit without Store DONE does not start Reverse and
that repeated Store DONE starts it once.

- [ ] **Step 2: Write Forward frontier tests**

```python
def test_forward_frontier_returns_only_new_aligned_range():
    frontier = ForwardFrontier(
        token_start=64,
        token_end=128,
        transferred_until=64,
    )

    assert frontier.advance(computed_until=79, block_size=16) is None
    assert frontier.advance(computed_until=96, block_size=16) == (64, 96)
    assert frontier.advance(computed_until=96, block_size=16) is None
    assert frontier.advance(computed_until=128, block_size=16) == (96, 128)
    assert frontier.finished
```

- [ ] **Step 3: Implement the explicit Reverse gate**

```python
def can_start_reverse(
    *,
    committed: bool,
    plan_bound: bool,
    store_done: bool,
    reverse_enqueued: bool,
    terminal: bool,
) -> bool:
    return (
        committed
        and plan_bound
        and store_done
        and not reverse_enqueued
        and not terminal
    )
```

- [ ] **Step 4: Implement mapping-first `bind_batch()`**

`DualPathLayerwiseRuntime.bind_batch()` performs two scans. The first installs
all `inbound_bindings` and calls `reconciler.publish_ready()`. The second
registers request states and invokes `try_start_reverse()` for eligible
partial plans. It never starts Forward. `mark_store_done(request_key)` sets
the completion fact and calls the same `try_start_reverse()` gate.

Expose injected callables:

```python
@dataclass
class RequestTransferState:
    request_key: DualPathRequestKey
    plan: DualPathWorkerPlan | LocalFullHitWorkerPlan
    facts: RequestCompletionFacts = field(
        default_factory=RequestCompletionFacts
    )
    reverse_enqueued: bool = False
    forward_frontier: ForwardFrontier | None = None


class DualPathLayerwiseRuntime:
    def __init__(
        self,
        enqueue_reverse: Callable[[LayerwiseDirectionPlan], None],
        enqueue_forward: Callable[
            [LayerwiseDirectionPlan, tuple[int, int]], None
        ],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enqueue_reverse = enqueue_reverse
        self.enqueue_forward = enqueue_forward
        self.monotonic = monotonic
        self.reconciler = RawCompletionReconciler()
        self.requests: dict[DualPathRequestKey, RequestTransferState] = {}
```

- [ ] **Step 5: Implement monotonic Forward frontier**

`ForwardFrontier.advance()` rounds `computed_until` down to the common
physical block size, clamps to `token_end`, returns only the new interval, and
sets `finished` when `transferred_until == token_end`. Reject a regressing
`computed_until`.

Implement `invalid_blocks_for(local_request_ids)` by reading only destination
block IDs already present in the frozen receive-side plans. It must not query
block tables or derive a new token range.

- [ ] **Step 6: Run the runtime tests**

Run:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
```

Expected: PASS.

- [ ] **Step 7: Commit gates and frontier**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
git commit -s -m "feat(kv-transfer): gate DualPath direction dispatch"
```

---

### Task 5: Compose the Existing Bulk KVPoolWorker on DE

**Files:**

- Create:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py`
- Create:
  `tests/ut/distributed/kv_transfer/dual_path/test_store_adapter.py`

**Interfaces:**

- Consumes: `KVPoolWorker(vllm_config, use_layerwise=False,
  kv_cache_config=kv_cache_config)`, `AscendConnectorMetadata`,
  `DualPathWorkerPlan`, and `LocalFullHitWorkerPlan`.
- Produces: `StorePollResult` and `DualPathStoreWorkerAdapter`.

- [ ] **Step 1: Write bulk-load and local-full tests**

```python
def make_local_full_plan() -> LocalFullHitWorkerPlan:
    metadata = AscendConnectorMetadata(
        set(),
        set(),
        loading_req_ids={"de-request"},
    )
    return LocalFullHitWorkerPlan(
        request_key=DualPathRequestKey(
            de_engine_incarnation="de-boot-1",
            de_engine_local_request_id="de-request",
        ),
        store_coverage=StoreCoverage(
            raw_hit_tokens=128,
            aligned_hit_tokens=128,
            local_tokens=64,
            ready_prefix_tokens=128,
            decode_ready_tokens=128,
            store_load_tokens=64,
            prefill_tail_tokens=0,
            is_full_hit=True,
        ),
        store_metadata=metadata,
    )


def test_start_uses_existing_bulk_worker_plan():
    pool_worker = MagicMock()
    adapter = DualPathStoreWorkerAdapter(pool_worker)
    plan = make_local_full_plan()

    adapter.start(plan)

    pool_worker.start_load_kv.assert_called_once_with(plan.store_metadata)


def test_local_full_completion_does_not_start_transport():
    pool_worker = MagicMock()
    pool_worker.get_finished.return_value = (set(), {"de-request"})
    pool_worker.get_block_ids_with_load_errors.return_value = set()
    adapter = DualPathStoreWorkerAdapter(pool_worker)
    plan = make_local_full_plan()
    adapter.start(plan)

    result = adapter.poll(finished_req_ids=set())

    assert result.done_request_ids == {"de-request"}
    assert result.failed_request_ids == set()
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_store_adapter.py
```

Expected: FAIL because `store_adapter.py` does not exist.

- [ ] **Step 3: Implement the thin Worker adapter**

```python
@dataclass(frozen=True)
class StorePollResult:
    done_request_ids: set[str]
    failed_request_ids: set[str]
    invalid_block_ids: set[int]


class DualPathStoreWorkerAdapter:
    def __init__(self, worker: KVPoolWorker) -> None:
        self.worker = worker
        self.active_metadata: dict[str, AscendConnectorMetadata] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        self.worker.register_kv_caches(kv_caches)

    def start(
        self,
        plan: DualPathWorkerPlan | LocalFullHitWorkerPlan,
    ) -> None:
        metadata = plan.store_metadata
        if metadata is None:
            return
        request_key = (
            plan.request_key
            if isinstance(plan, LocalFullHitWorkerPlan)
            else plan.ids.request_key
        )
        local_id = request_key.de_engine_local_request_id
        if local_id in self.active_metadata:
            return
        self.active_metadata[local_id] = metadata
        self.worker.start_load_kv(metadata)
```

`poll()` calls the existing
`KVPoolWorker.get_finished(finished_req_ids, metadata)` for each active
metadata object. Treat any block ID returned by
`get_block_ids_with_load_errors()` as failure for the owning plan, and remove
active metadata only after DONE or failure.

- [ ] **Step 4: Prove Reverse is released only after Store DONE**

Add an integration test with `DualPathLayerwiseRuntime`: call `bind_batch()`,
poll Store with no completion, then with the request ID complete. Assert the
injected `enqueue_reverse` callable runs exactly once after the second poll.

- [ ] **Step 5: Run Store and runtime tests**

Run:

```bash
pytest -sv \
  tests/ut/distributed/kv_transfer/dual_path/test_store_adapter.py \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
```

Expected: PASS.

- [ ] **Step 6: Commit the Store adapter**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/store_adapter.py \
  tests/ut/distributed/kv_transfer/dual_path/test_store_adapter.py \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
git commit -s -m "feat(kv-transfer): compose DE Store bulk loads"
```

---

### Task 6: Convert Frozen Plans to Parent Layerwise Tasks

**Files:**

- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- Modify:
  `tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py`

**Interfaces:**

- Consumes: `LayerwiseDirectionPlan`, parent `ReqMeta`, parent `SendTask`, and
  registered `layer_metadata`.
- Produces: `to_parent_request_id()`, `direction_plan_to_req_meta()`, and
  per-layer `SendTask` construction using frozen pairs.

- [ ] **Step 1: Write exact pair-preservation tests**

```python
def make_direction_plan(
    *,
    block_pairs: tuple[tuple[BlockPair, ...], ...],
) -> LayerwiseDirectionPlan:
    return LayerwiseDirectionPlan(
        direction=TransferDirection.FORWARD,
        wire_external_id="forward-wire",
        local_request_id="local-request-123456789",
        token_start=16,
        token_end=48,
        block_pairs=block_pairs,
        remote_block_size=(16,),
        remote_engine_id="remote-engine",
        remote_host="127.0.0.1",
        remote_port=5000,
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
    )


def test_plan_conversion_preserves_frozen_pair_order():
    plan = make_direction_plan(
        block_pairs=(
            (
                BlockPair(local_block_id=8, remote_block_id=30),
                BlockPair(local_block_id=3, remote_block_id=11),
            ),
        )
    )

    request_id, req_meta = direction_plan_to_req_meta(plan)

    assert request_id.startswith(plan.wire_external_id)
    assert req_meta.local_block_ids == [[8, 3]]
    assert req_meta.remote_block_ids == [[30, 11]]
```

Add a test that `get_external_request_id(to_parent_request_id(plan))` equals
`plan.wire_external_id`; this prevents direction identity from being lost by
the parent's fixed nine-character EngineCore suffix removal.

Add a Reverse test whose registered layer names are deliberately different
from lexical order. Assert submission follows `worker.index_to_name` order
from `register_kv_caches()`, and that the last registered layer alone is
eligible to emit the request-level terminal.

- [ ] **Step 2: Implement plan conversion**

```python
def to_parent_request_id(plan: LayerwiseDirectionPlan) -> str:
    suffix = plan.local_request_id[-9:]
    if len(suffix) != 9:
        raise ValueError("Engine-local request ID must contain a 9-byte suffix")
    return f"{plan.wire_external_id}{suffix}"
```

Build the parent `ReqMeta` directly from each group’s ordered
`BlockPair.local_block_id` and `remote_block_id`. Copy endpoint and topology
fields from the plan. Do not sort pairs or call `zip()` on independently
computed arrays.

Implement Reverse dispatch by building one direction-only parent metadata
object, calling `_prepare_send_metadata()` once, then iterating
`for layer_index in range(worker.total_layers)` and each
`worker.index_to_name[layer_index]` entry in registered order. For every layer
call
`_enqueue_send_layer(layer_name, worker.kv_caches[layer_name], None,
direction_metadata)`. Store `kv_caches` on the DualPath Worker during its
single `register_kv_caches()` call. This reuses the parent's event,
quantization, resharding, and final-layer terminal path without a model
callback.

- [ ] **Step 3: Add interval-to-pair selection for Forward chunks**

```python
def select_interval_pairs(
    plan: LayerwiseDirectionPlan,
    interval: tuple[int, int],
    group_block_sizes: tuple[int, ...],
) -> tuple[tuple[BlockPair, ...], ...]:
    start, end = interval
    selected: list[tuple[BlockPair, ...]] = []
    for group_pairs, block_size in zip(
        plan.block_pairs,
        group_block_sizes,
    ):
        first = (start - plan.token_start) // block_size
        last = (end - plan.token_start) // block_size
        selected.append(group_pairs[first:last])
    return tuple(selected)
```

Validate both interval boundaries are aligned for every participating group
and every selected group stays within the frozen plan.

- [ ] **Step 4: Test final terminal semantics**

Use a mocked `KVCacheSendingLayerThread`. Submit two Forward chunks over all
registered layers and assert:

- no request terminal before the final layer of the final chunk;
- final success emits one DONE;
- a failed layer stops unsent work and emits one FAILED;
- DONE and FAILED cannot both be emitted.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
pytest -sv \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py \
  -k "pair or parent_request_id or forward or terminal"
```

Expected: PASS.

- [ ] **Step 6: Commit plan conversion**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py
git commit -s -m "feat(kv-transfer): execute frozen DualPath block pairs"
```

---

### Task 7: Integrate the Shared Bidirectional Worker and Scheduler Metadata

**Files:**

- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py:51`
- Modify:
  `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py:189`

**Interfaces:**

- Consumes: Parent protected helpers, `DualPathConnectorMetadata`,
  `DualPathLayerwiseRuntime`, `DualPathStoreWorkerAdapter`, and committed
  Worker plans from the path-decision integration.
- Produces:
  `DualPathConnectorScheduler.queue_worker_plan()`,
  `DualPathConnectorWorker.start_load_kv()`,
  `save_kv_layer()`, `get_finished()`, and DE-only Store composition.

- [ ] **Step 1: Write construction and shared-runtime tests**

```python
def test_dual_path_worker_requests_both_parent_threads():
    worker = DualPathConnectorWorker.__new__(DualPathConnectorWorker)
    assert worker._needs_send_thread()
    assert worker._needs_receive_thread()


def test_register_kv_caches_registers_once_and_starts_one_thread_each():
    worker = DualPathConnectorWorker.__new__(DualPathConnectorWorker)
    worker.dual_path_cfg = SimpleNamespace(role="de")
    worker.store_adapter = MagicMock()
    kv_caches = {"layer.0": MagicMock()}
    with patch.object(
        MooncakeLayerwiseConnectorWorker,
        "register_kv_caches",
    ) as parent_register:
        worker.register_kv_caches(kv_caches)

    parent_register.assert_called_once_with(kv_caches)
    worker.store_adapter.register_kv_caches.assert_called_once_with(
        kv_caches
    )
```

- [ ] **Step 2: Add Scheduler plan queueing**

```python
class DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler):
    def queue_worker_plan(
        self,
        plan: DualPathWorkerPlan | LocalFullHitWorkerPlan,
    ) -> None:
        request_key = (
            plan.request_key
            if isinstance(plan, LocalFullHitWorkerPlan)
            else plan.ids.request_key
        )
        existing = self._pending_worker_plans.get(request_key)
        if existing is not None and existing != plan:
            raise CompletionConflictError(
                f"conflicting worker plan for {request_key}"
            )
        self._pending_worker_plans[request_key] = plan

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> DualPathConnectorMetadata:
        parent = super().build_connector_meta(scheduler_output)
        plans = tuple(self._pending_worker_plans.values())
        self._pending_worker_plans.clear()
        metadata = DualPathConnectorMetadata(plans=plans)
        metadata.requests = parent.requests
        metadata.send_task = parent.send_task
        return metadata
```

The path-decision code is the only caller of `queue_worker_plan()`. Repeated
identical plans are idempotent; a different plan for the same request key
raises a protocol error.

- [ ] **Step 3: Enable both inherited threads in the DualPath Worker**

Override:

```python
def _needs_send_thread(self) -> bool:
    return True

def _needs_receive_thread(self) -> bool:
    return True
```

Call `super().register_kv_caches(kv_caches)` once. On DE, also call
`store_adapter.register_kv_caches(kv_caches)` once.

Construct the DE adapter in `DualPathConnectorWorker.__init__`:

```python
self.store_adapter = (
    DualPathStoreWorkerAdapter(
        KVPoolWorker(
            vllm_config,
            use_layerwise=False,
            kv_cache_config=kv_cache_config,
        )
    )
    if dual_path_cfg.role == "de"
    else None
)
```

In `DualPathConnector.__init__`, initialize
`self._connector_metadata = DualPathConnectorMetadata()` while preserving the
existing facade and Scheduler/Worker selection.

- [ ] **Step 4: Implement inbound-first `start_load_kv()`**

`DualPathConnectorWorker.start_load_kv()` must:

1. require `DualPathConnectorMetadata`;
2. call `runtime.bind_batch(metadata)` before any outbound work;
3. call `_bind_receive_metadata()` for inbound plans;
4. call `_prepare_send_metadata()` only for direction plans that pass an
   explicit gate;
5. start DE Store metadata through `store_adapter`;
6. avoid Store and P2P entirely for a plan already terminal.

Add the four-request mixed-batch test at the Worker hook, not only at the pure
runtime class.

- [ ] **Step 5: Drive Forward from `save_kv_layer()` only**

On PE, `save_kv_layer()` asks the runtime for each request’s newly aligned
Forward interval, converts only that interval to a parent `SendTask`, and then
uses the inherited layer send queue. `start_load_kv()` never enqueues Forward.

- [ ] **Step 6: Poll Store and raw receive terminals in `get_finished()`**

Poll the receive thread whenever it exists, regardless of the static
producer/consumer role. Ingest raw DONE/FAILED into the reconciler, poll DE
Store, re-evaluate Reverse gates, and publish only Engine-local IDs:

```python
done_local, failed_local = self.runtime.reconciler.publish_ready(
    now=time.monotonic()
)
self._invalid_block_ids.update(
    self.runtime.invalid_blocks_for(failed_local)
)
return set(), done_local | failed_local
```

Failure IDs must be represented in the runtime’s completion facts before
publication so Scheduler-side timeout/error handling can distinguish failure
from success through invalid blocks and request state.

- [ ] **Step 7: Run connector integration tests**

Run:

```bash
pytest -sv \
  tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py \
  tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py \
  tests/ut/distributed/kv_transfer/dual_path/test_store_adapter.py
```

Expected: PASS.

- [ ] **Step 8: Commit Worker integration**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py \
  tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
git commit -s -m "feat(kv-transfer): run DualPath bidirectional worker"
```

---

### Task 8: Add Terminal, Cancellation, Timeout, and MultiConnector Coverage

**Files:**

- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py`
- Modify:
  `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- Modify:
  `tests/ut/distributed/kv_transfer/dual_path/test_layerwise_transfer.py`
- Modify:
  `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`
- Modify:
  `tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py`

**Interfaces:**

- Consumes: Framework `finished_recving`, invalid block IDs,
  `AscendMultiConnector` broadcast Worker hooks, and existing delayed-free
  behavior.
- Produces: Drain-first terminal cleanup, execution timeout, tombstone cleanup,
  and integration regression coverage.

- [ ] **Step 1: Write drain-first cancellation tests**

```python
def test_cancel_stops_new_work_but_retains_plan_until_quiesced():
    runtime = DualPathLayerwiseRuntime(
        enqueue_reverse=MagicMock(),
        enqueue_forward=MagicMock(),
    )
    state = MagicMock(spec=RequestTransferState)
    state.request_key = DualPathRequestKey(
        de_engine_incarnation="de-boot-1",
        de_engine_local_request_id="de-request",
    )
    state.terminal = False
    state.transport_quiesced = False
    state.retained_blocks = True
    runtime.requests[state.request_key] = state

    runtime.cancel(state.request_key)

    assert state.terminal
    assert not runtime.can_submit_new_work(state.request_key)
    assert runtime.has_retained_blocks(state.request_key)

    runtime.mark_transport_quiesced(state.request_key)

    assert not runtime.has_retained_blocks(state.request_key)
```

Add tests for request execution timeout, late terminal absorption by a
tombstone, tombstone cleanup only after request cleanup plus retry window, and
orphan raw cleanup at `path_execution_timeout + transport_retry_slack`.

- [ ] **Step 2: Implement terminal state transitions**

Add `terminal`, `transport_quiesced`, `cleanup_seen`, `retained_blocks`,
`deadline_monotonic`, `forward_started_monotonic`,
`reverse_started_monotonic`, `forward_terminal_monotonic`, and
`reverse_terminal_monotonic` to `RequestTransferState`. Implement:

- `fail_request(request_key, error_code)`;
- `cancel(request_key)`;
- `mark_transport_quiesced(request_key)`;
- `expire_requests(now)`;
- `cleanup_request(request_key, now)`;
- `can_submit_new_work(request_key) -> bool`;
- `has_retained_blocks(request_key) -> bool`.

Every transition is idempotent. A terminal state makes
`can_submit_new_work()` false immediately but retains the plan until
`transport_quiesced` is true.

- [ ] **Step 3: Expose the accepted observability snapshot**

Add `snapshot_observability(now)` to the runtime and assert it returns these
exact fields:

```python
{
    "pending_raw_completion_count": 0,
    "oldest_pending_raw_seconds": 0.0,
    "orphan_raw_expired_total": 0,
    "active_inbound_mapping_count": 0,
    "terminal_wire_tombstone_count": 0,
    "duplicate_terminal_total": 0,
    "conflicting_terminal_total": 0,
    "forward_failure_total": 0,
    "reverse_failure_total": 0,
}
```

Record Forward and Reverse start/terminal timestamps in
`RequestTransferState`, and emit per-direction request latency only at
request-level terminal. Do not add per-layer remote completion messages.

- [ ] **Step 4: Test first-positive and broadcast integration**

Patch an `AscendMultiConnector` with an earlier Store connector and
`DualPathConnector`. Assert:

- first positive load selection remains configuration-order based;
- `update_state_after_alloc()` still gives real blocks to the Layerwise
  subclass;
- Worker `bind_connector_metadata()`, `start_load_kv()`,
  `save_kv_layer()`, and `get_finished()` reach DualPath;
- no request is counted complete until its path-specific completion predicate
  is true.

- [ ] **Step 5: Test path-specific completion predicates**

Cover:

```text
DE_LOCAL_FULL_HIT: Store DONE
DE_PARTIAL_HIT on PE: Reverse DONE
DE_PARTIAL_HIT on DE: Store DONE and Forward DONE
DualPath not adopted on DE: Forward DONE
```

Include one TP-rank failure test using the framework Worker-output
aggregation path; do not add a second Scheduler barrier.

- [ ] **Step 6: Run all host-side KV transfer tests**

Run:

```bash
pytest -sv \
  tests/ut/distributed/kv_transfer/dual_path \
  tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py \
  tests/ut/kv_offload/test_mooncake_layerwise_connector.py
```

Expected: PASS.

- [ ] **Step 7: Commit lifecycle integration**

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/layerwise_transfer.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py \
  tests/ut/distributed/kv_transfer/dual_path \
  tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py
git commit -s -m "fix(kv-transfer): drain DualPath terminal work safely"
```

---

### Task 9: Add NPU E2E and Final Verification

**Files:**

- Create:
  `tests/e2e/nightly/multi_node/dual_path/test_dual_path_transfer.py`
- Modify:
  `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md`
  only if implementation names differ from the accepted schema.

**Interfaces:**

- Consumes: Completed Stage 1 data-plane implementation and the existing
  disaggregated PE/DE launch harness.
- Produces: Real-NPU evidence for local full, partial Reverse/Forward,
  rejection Forward, race handling, failure, cancellation, and cleanup.

- [ ] **Step 1: Add a parametrized NPU scenario matrix**

The test module must launch PE and DE processes with fixed coverage fixtures
and assert these cases:

```python
SCENARIOS = (
    "de_local_full_hit",
    "de_partial_hit",
    "dual_path_rejected",
    "early_reverse_done",
    "forward_failure",
    "tp_rank_failure",
    "cancel_during_forward",
)
```

For each scenario, capture output tokens, selected path, Store/Reverse/Forward
token ranges, terminal counts, invalid blocks, and process exit status.

- [ ] **Step 2: Add race and retention assertions**

For `early_reverse_done`, delay the PE Worker metadata bind until the DE has
sent Reverse DONE and assert the request still resumes. For cancellation and
failure, assert source/destination blocks are not reused before transport
quiescence and are released afterward.

- [ ] **Step 3: Run host-side formatting and tests**

Run:

```bash
bash format.sh ci
pytest -sv \
  tests/ut/distributed/kv_transfer/dual_path \
  tests/ut/distributed/kv_transfer/test_kv_transfer_failures.py \
  tests/ut/kv_offload/test_mooncake_layerwise_connector.py
```

Expected: format check succeeds and all tests pass.

- [ ] **Step 4: Run the NPU E2E suite on the supported topology**

Run:

```bash
pytest -sv \
  tests/e2e/nightly/multi_node/dual_path/test_dual_path_transfer.py
```

Expected: all seven scenarios pass on the topology admitted by the detailed
design.

- [ ] **Step 5: Verify design invariants in source**

Run:

```bash
rg -n \
  "pending_raw_completions|terminal_wire_tombstones|can_start_reverse|ForwardFrontier|InboundRequestBinding|BlockPair" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path
git diff --check
git status --short
```

Expected: every accepted data-plane invariant has an implementation symbol,
`git diff --check` is silent, and unrelated user files remain untouched.

- [ ] **Step 6: Commit E2E coverage**

```bash
git add \
  tests/e2e/nightly/multi_node/dual_path/test_dual_path_transfer.py \
  docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
git commit -s -m "test(kv-transfer): cover DualPath bidirectional transfer"
```

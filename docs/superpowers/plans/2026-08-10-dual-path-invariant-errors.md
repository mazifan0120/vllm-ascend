# DualPath Invariant Errors and Protocol Field Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove DualPath protocol-version fields and treat Prefill decision metadata and local decision failures as propagating internal-invariant violations instead of recoverable Scheduler failures.

**Architecture:** First remove `protocol_version` atomically from both control messages and every producer, consumer, log, and fixture. Then make the Prefill Scheduler commit request state only after a successful direct decision. Finally reduce `PathDecisionDecider` to successful records while preserving successful replay and same-key fact-conflict protection.

**Tech Stack:** Python 3.12, dataclasses, msgspec MessagePack, ZeroMQ, pytest.

## Global Constraints

- Decode and Prefill use compatible DualPath code from the same deployment.
- Do not add protocol negotiation, compatibility aliases, fallback behavior, or new configuration.
- Malformed decision metadata and local decision failures propagate to the caller.
- Keep `_pe_invalid_request_ids` for Forward-plan, activation, delivery, and other post-decision runtime failures.
- Keep malformed MessagePack rejection in the asynchronous decision-result receiver.
- Keep successful decision replay and same-key fact-conflict protection.
- Do not modify AscendStore or Mooncake Layerwise behavior.
- Preserve unrelated untracked files under `.kimi-code/` and `docs/superpowers/`.

## File Structure

- `path_decision_channel.py` owns the field-less wire schema and receiver validation.
- `scheduler.py` owns message production/consumption and Prefill decision state commits.
- `path_decision.py` owns successful decision idempotency and conflict detection.
- Channel tests pin the valid dictionary/MessagePack schema and receiver behavior.
- Scheduler tests pin decision flow and state ownership.
- Decider tests pin successful records, replay, discard, and conflicts.

---

### Task 1: Remove the protocol field from both control messages

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py:31-165,438-454`
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py:43-50,334-353,738-790,949-965,1145-1175`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward_integration.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_de_local_store_full.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_forward_receive_binding.py`

**Interfaces:**
- Produces: `DualPathDecisionMetadata(decision_request, decode_control_endpoint)`.
- Produces: `PathDecision(result, reverse_plan)`.
- Preserves: `encode_path_decision(PathDecision) -> bytes` and `decode_path_decision(bytes) -> PathDecision`.
- Preserves: malformed MessagePack produces `PathDecisionValidationError` and receives no ACK.

- [ ] **Step 1: Change valid wire-schema tests and fixtures first**

Update constructors and expected dictionaries to omit the field:

```python
metadata = DualPathDecisionMetadata(
    decision_request=request,
    decode_control_endpoint=endpoint,
)
assert set(metadata.to_dict()) == {
    "decision_request",
    "decode_control_endpoint",
}

decision = PathDecision(
    result=PathDecisionResult(
        request_key=request.request_key,
        path=PathKind.PE_READ,
    ),
    reverse_plan=None,
)
assert set(decision.to_dict()) == {"result", "reverse_plan"}
assert decode_path_decision(encode_path_decision(decision)) == decision
```

Remove `DUAL_PATH_PROTOCOL_VERSION` imports and `protocol_version=` arguments
from all files listed above. Remove these version-only tests:

```text
test_path_decision_from_dict_rejects_bad_version
test_protocol_v1_or_mixed_payload_rejected
test_unsupported_protocol_version_gets_no_ack
test_version_mismatch_fails_locally_and_sends_nothing
```

Remove these malformed data-model contract tests; malformed transport handling
remains covered separately by the MessagePack receiver tests:

```text
test_decision_metadata_rejects_forged_store_full_before_send
test_dual_path_decision_metadata_rejects_non_exact_keys
test_path_decision_from_dict_rejects_non_exact_keys
```

Rename `test_protocol_v2_decision_request_and_result_round_trip` to
`test_decision_request_and_result_round_trip` and assert the field-less keys.

- [ ] **Step 2: Run focused tests and verify the old schema fails**

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py
```

Expected: FAIL because the dataclasses still require `protocol_version` or
still serialize that extra field.

- [ ] **Step 3: Implement the field-less message definitions**

Delete `DUAL_PATH_PROTOCOL_VERSION`, `_require_protocol_version()`, and an
otherwise-unused `Final` import. Replace the message shapes with:

```python
@dataclass(frozen=True)
class DualPathDecisionMetadata:
    decision_request: PathDecisionRequest
    decode_control_endpoint: DecodeControlEndpoint

    def __post_init__(self) -> None:
        if not isinstance(self.decision_request, PathDecisionRequest):
            raise PathDecisionValidationError(
                "decision_request must be a PathDecisionRequest"
            )
        if not isinstance(self.decode_control_endpoint, DecodeControlEndpoint):
            raise PathDecisionValidationError(
                "decode_control_endpoint must be a DecodeControlEndpoint"
            )

    def to_dict(self) -> _JsonObject:
        return {
            "decision_request": self.decision_request.to_dict(),
            "decode_control_endpoint": self.decode_control_endpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: _JsonValue) -> DualPathDecisionMetadata:
        data = _require_exact_payload(
            payload,
            frozenset({"decision_request", "decode_control_endpoint"}),
        )
        return cls(
            decision_request=PathDecisionRequest.from_dict(data["decision_request"]),
            decode_control_endpoint=DecodeControlEndpoint.from_dict(
                data["decode_control_endpoint"]
            ),
        )


@dataclass(frozen=True)
class PathDecision:
    result: PathDecisionResult
    reverse_plan: ReversePlan | None

    def to_dict(self) -> _JsonObject:
        return {
            "result": self.result.to_dict(),
            "reverse_plan": (
                None if self.reverse_plan is None else self.reverse_plan.to_dict()
            ),
        }
```

Keep the current `PathDecision` type checks, minus protocol validation. Make
`PathDecision.from_dict()` require exactly `{"result", "reverse_plan"}`.

- [ ] **Step 4: Remove receiver and Scheduler protocol usage**

In `PathDecisionResultReceiver._handle_frames()`, keep malformed payload
rejection and delete only the unsupported-version comparison. Construct messages
without the field:

```python
decision = PathDecision(result=result, reverse_plan=reverse_plan)

decision_metadata = DualPathDecisionMetadata(
    decision_request=decision_request,
    decode_control_endpoint=coordinator.decode_control_endpoint,
)
```

Keep Scheduler metadata parse recovery until Task 2, but delete its version
comparison. Remove `protocol=%s` and the corresponding argument from delivery
and activation logs without reordering the remaining fields.

- [ ] **Step 5: Run wire, Scheduler, and fixture tests**

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py \
  tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py \
  tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward.py \
  tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward_integration.py \
  tests/ut/distributed/kv_transfer/dual_path/test_de_local_store_full.py \
  tests/ut/distributed/kv_transfer/dual_path/test_forward_receive_binding.py
```

Expected: PASS.

- [ ] **Step 6: Verify symbol removal and commit**

```bash
rg -n "DUAL_PATH_PROTOCOL_VERSION|protocol_version" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path \
  tests/ut/distributed/kv_transfer/dual_path
```

Expected: no matches.

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py \
  tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py \
  tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward.py \
  tests/ut/distributed/kv_transfer/dual_path/test_pe_read_forward_integration.py \
  tests/ut/distributed/kv_transfer/dual_path/test_de_local_store_full.py \
  tests/ut/distributed/kv_transfer/dual_path/test_forward_receive_binding.py
git commit -s -m "refactor(dual_path): remove control protocol field"
```

---

### Task 2: Let Prefill decision invariant violations propagate

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py:314-399`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py:543-880,1240-1280,1540-1580`

**Interfaces:**
- Consumes: field-less `DualPathDecisionMetadata.from_dict()` from Task 1.
- Produces: Scheduler-owned Prefill state only after `decide()` succeeds.
- Preserves: `_pe_invalid_request_ids` handling outside metadata parse and local decision.

- [ ] **Step 1: Remove recovery-oriented Scheduler tests**

Delete these unsupported recovery tests:

```text
test_parent_accounting_runs_exactly_once_for_malformed_dual_path
test_forged_full_payload_is_invalidated_without_decide_or_submit
test_malformed_nested_metadata_marks_invalid_and_sends_nothing
test_policy_exception_or_invalid_return_records_failure_sends_nothing
test_local_failures_never_enter_parent_forward_queue
test_update_state_after_alloc_suppresses_send_queue_for_malformed_dual_path
```

In `test_pe_finish_clears_invalid_request_marker` and
`test_shutdown_is_idempotent_and_leaves_no_owned_state`, replace the malformed
metadata setup with the runtime-invalid precondition those tests actually need:

```python
scheduler._pe_invalid_request_ids.add(request.request_id)
```

Keep the two same-key conflict tests but expect direct propagation and prove the
first successful metadata was not overwritten:

```python
with pytest.raises(
    PathDecisionValidationError,
    match="request key is already associated with different token facts",
):
    scheduler.get_num_new_matched_tokens(conflicting, 0)

assert scheduler._pe_invalid_request_ids == set()
assert (
    scheduler._pe_decision_metadata[original.request_id]
    .decision_request.decode_store_tokens
    == 24
)
```

- [ ] **Step 2: Run conflict tests and verify the old Scheduler catches them**

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py \
  -k "conflicting_facts or conflicting_frozen_prefill_prefix"
```

Expected: FAIL because the current Scheduler catches the exception, marks the
request invalid, and returns `parent_result`.

- [ ] **Step 3: Deserialize and decide before committing Scheduler state**

Use this order in `_decide_prefill_path_for_admission()`:

```python
metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
decision_request = metadata.decision_request
effective_prefill_tokens = _expected_prefill_token_end(
    decision_request,
    self.need_truncate,
)
if not 0 <= prefill_local_tokens <= effective_prefill_tokens:
    logger.error(
        "DualPath Prefill local tokens are invalid for request %s: "
        "expected 0 <= %s <= %s",
        request_id,
        prefill_local_tokens,
        effective_prefill_tokens,
    )
    return parent_result

assert self._path_decider is not None
result = self._path_decider.decide(decision_request, prefill_local_tokens)

request_key = decision_request.request_key
self._pe_request_keys[request_id] = request_key
self._pe_decision_metadata[request_id] = metadata
self._pe_prefill_local_tokens.setdefault(request_id, prefill_local_tokens)
self._pe_path_results.setdefault(request_id, result)
```

Delete the metadata-invalid and local-decision-failed logs, catches, invalid-set
writes, and `parent_result` recovery returns. Do not change the separate local
token-range branch.

- [ ] **Step 4: Run and commit the Scheduler change**

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py
```

Expected: PASS.

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path/test_scheduler_decision_control.py
git commit -s -m "refactor(dual_path): propagate decision invariant errors"
```

---

### Task 3: Retain only successful local decisions

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py:164-220`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py:256-399`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_rejection_evidence.py`

**Interfaces:**
- Produces: `_DecisionRecord.result: PathDecisionResult` with no failure state.
- Preserves: `decide(...) -> PathDecisionResult`, `discard(...) -> None`, successful replay, and fact-conflict detection.

- [ ] **Step 1: Remove tests for failed-decision retention**

Delete `ExplodingPolicy` and `InvalidResultPolicy` if they become unused. Delete:

```text
test_policy_result_not_a_path_raises_and_retains_failure
test_policy_exception_is_chained_retained_and_invoked_once
test_discard_removes_failure_record
test_retained_failure_replay_raises_without_policy
```

Keep the eligibility, successful replay, conflict, successful discard, and
round-robin tests.

- [ ] **Step 2: Run the valid decision tests as the refactor baseline**

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_rejection_evidence.py \
  -k "eligibility or replay or conflicting or discard or store_full"
```

Expected: PASS before the refactor. No new test is added for failures the design
explicitly does not support.

- [ ] **Step 3: Simplify the Decider to successful records**

```python
@dataclass(frozen=True)
class _DecisionRecord:
    request: PathDecisionRequest
    prefill_local_tokens: int
    result: PathDecisionResult


class PathDecisionDecider:
    def decide(
        self,
        request: PathDecisionRequest,
        prefill_local_tokens: int,
    ) -> PathDecisionResult:
        if not isinstance(request, PathDecisionRequest):
            raise PathDecisionValidationError(
                "decision input must be a PathDecisionRequest"
            )

        existing = self._decision_records.get(request.request_key)
        if existing is not None:
            if (
                request != existing.request
                or prefill_local_tokens != existing.prefill_local_tokens
            ):
                raise PathDecisionValidationError(
                    "request key is already associated with different token facts "
                    "or prefill_local_tokens"
                )
            return existing.result

        path = (
            PathKind.PE_READ
            if prefill_local_tokens >= request.decode_store_tokens
            else self._policy.choose(request)
        )
        result = PathDecisionResult(request_key=request.request_key, path=path)
        self._decision_records[request.request_key] = _DecisionRecord(
            request=request,
            prefill_local_tokens=prefill_local_tokens,
            result=result,
        )
        return result
```

Delete `_record_failure()` and `decision previously failed locally`. A policy
exception propagates unchanged; an invalid return reaches `PathDecisionResult`
validation without creating a retained record.

- [ ] **Step 4: Run and commit the Decider change**

```bash
.venv/bin/python -m pytest -q \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision_rejection_evidence.py
```

Expected: PASS.

```bash
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py \
  tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py
git commit -s -m "refactor(dual_path): retain successful decisions only"
```

---

### Task 4: Verify the complete DualPath change

**Files:**
- Verify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/`
- Verify: `tests/ut/distributed/kv_transfer/dual_path/`

**Interfaces:**
- Produces: a verified three-commit implementation series.

- [ ] **Step 1: Confirm removed symbols are absent**

```bash
rg -n \
  "DUAL_PATH_PROTOCOL_VERSION|protocol_version|decision previously failed locally|_record_failure|DualPath Prefill decision metadata is invalid|DualPath Prefill decision failed locally" \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path \
  tests/ut/distributed/kv_transfer/dual_path
```

Expected: no matches.

- [ ] **Step 2: Run every DualPath unit test**

```bash
.venv/bin/python -m pytest -q tests/ut/distributed/kv_transfer/dual_path
```

Expected: PASS with zero failures; record pass and skip counts.

- [ ] **Step 3: Run Ruff on changed Python scope**

```bash
.venv/bin/python -m ruff check \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path

.venv/bin/python -m ruff format --check \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py \
  tests/ut/distributed/kv_transfer/dual_path
```

Expected: both commands exit successfully. If Ruff is unavailable, report that
instead of claiming static verification passed.

- [ ] **Step 4: Inspect commit and worktree scope**

```bash
git log -4 --oneline --decorate
git status --short
git diff HEAD~3..HEAD --stat
git diff HEAD~3..HEAD --check
```

Expected: three implementation commits follow the design commit; only listed
DualPath source/tests changed; unrelated untracked files remain untouched; the
diff check reports no whitespace errors.

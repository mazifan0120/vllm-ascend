# DualPath invariant errors and protocol field removal

## Goal

Simplify the prefill path-decision flow by treating decision metadata and
local path selection as internal invariants instead of recoverable runtime
failures. Decode and Prefill are assumed to run the same DualPath build, so
the control messages do not carry an explicit protocol version.

## Assumptions

- Decode always emits a structurally valid `DualPathDecisionMetadata` payload.
- `PathDecisionDecider.decide()` is called with valid, consistent request facts.
- The configured path policy returns a `PathKind` and does not raise.
- Decode and Prefill use compatible DualPath code from the same deployment.

Violating these assumptions is a programming or deployment error. The error
propagates to the caller; DualPath does not log, latch, invalidate, retry, or
fall back to another connector for these cases.

## Wire format

Remove `protocol_version` from both control-plane messages:

```text
Decode -> Prefill: DualPathDecisionMetadata
  decision_request
  decode_control_endpoint

Prefill -> Decode: PathDecision
  result
  reverse_plan
```

Delete `DUAL_PATH_PROTOCOL_VERSION`, protocol type validation, producer-side
population, receiver-side comparisons, and protocol fields in logs. This is an
intentional wire-incompatible change. Compatibility negotiation is unnecessary
under the same-build deployment assumption.

Basic object construction and deserialization validation remains in place so
valid messages still have typed `PathDecisionRequest`, `DecodeControlEndpoint`,
`PathDecisionResult`, and `ReversePlan` values. Tests only pin valid schema and
round-trip behavior; they do not enumerate malformed payload recovery behavior
or exact malformed-payload error messages.

## Prefill Scheduler behavior

`_decide_prefill_path_for_admission()` directly deserializes metadata and calls
the decider:

```python
metadata = DualPathDecisionMetadata.from_dict(params["dual_path"])
result = self._path_decider.decide(
    metadata.decision_request,
    prefill_local_tokens,
)
```

Remove the following recovery branches:

- metadata `PathDecisionValidationError` logging and invalidation;
- protocol mismatch logging and invalidation;
- local `decide()` exception logging and invalidation.

Commit `_pe_request_keys`, `_pe_decision_metadata`,
`_pe_prefill_local_tokens`, and `_pe_path_results` only after `decide()`
returns successfully. This avoids leaving partial Scheduler state if an
invariant is violated and a higher layer happens to catch the propagated
exception.

No fallback to `MooncakeLayerwiseConnectorScheduler` is added. These failures
abort the call instead of returning `parent_result`.

## PathDecisionDecider behavior

The decider retains only successful decisions:

- `_DecisionRecord.result` is always a `PathDecisionResult`;
- policy exceptions propagate unchanged;
- `_record_failure()` and the `result=None` failure latch are removed;
- the `decision previously failed locally` branch is removed;
- an invalid policy return is rejected by `PathDecisionResult` construction.

Successful identical replay continues to reuse the original result without
advancing the policy. The same-key fact-conflict check remains: it protects a
successfully committed decision from being reused with different token facts
and is part of normal idempotency, not recovery from a failed decision.

## Failures that remain fail-closed

`_pe_invalid_request_ids` remains for runtime failures that can occur after a
valid decision, including Forward-plan validation, DE_READ activation,
delivery completion, and related control-plane activation failures. This change
does not weaken those lifecycle protections.

Malformed MessagePack handling in the asynchronous decision-result receiver is
also outside this change. It protects the receiver thread at a transport
boundary; only its protocol-version comparison is removed.

## Tests

Update tests to reflect the field-less wire schema:

- valid `DualPathDecisionMetadata` and `PathDecision` construction;
- dictionary and MessagePack round trips without `protocol_version`;
- Decode metadata production and Prefill decision production;
- receiver acceptance without a version gate;
- successful replay and same-key fact-conflict behavior.

Remove tests dedicated to:

- Scheduler recovery from malformed decision metadata;
- protocol-version mismatch and unsupported-version rejection;
- policy-exception or invalid-return failure latching;
- retrying a previously failed local decision;
- exact malformed data-model error behavior.

Run the focused DualPath unit-test suite after the implementation. No NPU
hardware is required because the changed paths are Scheduler and control-plane
logic.

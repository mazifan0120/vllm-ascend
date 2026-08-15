# PR-03 Unified Path Decision Control Plane

> **Superseded lifecycle clauses (2026-08-16):** This planned PR record keeps
> its historical timeout design for traceability. Timeout output and deadline
> requirements below are superseded by the
> [ABORT and watchdog-removal decision](../../../docs/superpowers/specs/2026-08-16-dual-path-abort-notification-and-watchdog-removal.md).
> The implemented control plane accepts explicit Decision and request-terminal
> ABORT outcomes and has no timer-driven fallback.

- Series position: 3 of 7
- Spec status: `PLANNED`
- Depends on: PR-01
- Blocks: PR-04
- Implementation tasks: `DP-CTL-01` through `DP-CTL-06`
- User-visible behavior: none; active DualPath configuration remains fail-closed
- Activation after merge: all non-HBM-complete coverage classes can complete a
  PE-owned decision round trip, but no decision authorizes data I/O

## Goal

Complete one non-blocking Decode-to-Prefill decision rendezvous for Store full,
partial, and miss candidates without authorizing Store or P2P data-plane I/O.

## Merge-state contract

After this PR merges, an integration-only `dual_path_v1` request with
`L_DE < R` can send `StoreCoverage` through the existing Decode-first Proxy
flow, create a real PE request, enter the PE Scheduler `DualPathConnector`,
receive one `PathDecisionCommit`, and deliver it to the DE Scheduler inbox.
Ordinary Layerwise requests retain their current Proxy flow.

The PE decision operates on `PathKind.PE_READ` and `PathKind.DE_READ`.
Eligibility is computed before policy. When both paths are eligible, the PE
round-robin policy alternates; a single eligible path is selected without
advancing the counter. Full coverage is a valid decision input and never an
automatic local bypass.

Production activation remains fail-closed in this PR. A returned commit is
observable in the control-plane integration harness but cannot become positive
first-winner accounting or authorize Store, Reverse, or Forward I/O.

## In scope

- Typed request key, coverage proposal, `PathKind`, decision commit, and error
  schemas.
- Eligible-path calculation and PE-owned round-robin policy.
- Proxy `dual_path_v1` opt-in and decision Future registration before PE
  dispatch.
- `/v1/path-decision` validation and commit-once resolution.
- PE Scheduler decision hook and PE/DE `PathDecisionCoordinator` asynchronous
  HTTP ownership.
- DE decision inbox with Scheduler-thread drain.
- Duplicate-identical commit idempotency and conflicting-commit rejection.
- Decision timeout and PE-request-before-commit failure.
- Ordinary Proxy behavior and mixed ordinary/DualPath concurrency regression.
- A closed-loop test with a fake or constrained responder proving both schema
  endpoints have exercised producers and consumers.

## Out of scope

- Positive Scheduler-visible `DE_READ` accounting.
- Store load, Reverse, or new Forward execution.
- Worker ZMQ decision messages.
- Whole-application middleware.
- Post-commit fallback or decision retry protocol.
- Production configuration that can activate a returned decision.
- Releasing Proxy Prefill load at decision-return time; release remains tied to
  final PE model-request completion.

## Expected code surface

- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/control_plane.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`
- `examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`
- `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_proxy.py`
- Focused control-plane tests under the DualPath UT directory

## Required tests

- Decision Future is registered before PE dispatch.
- Proxy overwrites untrusted callback URLs with its configured callback.
- Request and commit schemas reject invalid keys, coverage, and path values.
- Full coverage is serialized and reaches the PE Scheduler; it does not bypass
  the decision rendezvous.
- Round-robin alternates only when both `PE_READ` and `DE_READ` are eligible.
- Miss or an unavailable DE read selects `PE_READ` without advancing the
  round-robin counter.
- Same commit twice is idempotent; a different second commit is an error.
- PE HTTP failure before commit resolves a typed decision error.
- Timeout produces `DECISION_TIMEOUT` without blocking Scheduler hot paths.
- DE executor writes only to the inbox; Scheduler thread mutates commit state.
- Final PE response, not decision response, releases Prefill load accounting.
- Requests without `dual_path_v1` follow the existing `/v1/metaserver` path.
- Mixed ordinary and DualPath requests cannot resolve each other's Future.
- No test can observe Store/P2P I/O from any decision result.
- Active configuration fails fast until PR-04 provides the first complete data
  consumer.

Focused commands:

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_dual_path_proxy.py
pytest -sv tests/ut/distributed/kv_transfer/dual_path/ -k decision
bash format.sh ci
```

## Acceptance gates

- The control round trip is complete and non-blocking.
- Every protocol type has an exercised producer and consumer.
- Ordinary Layerwise Proxy behavior is unchanged.
- Incomplete data-plane work cannot be activated by configuration.
- If review size becomes excessive, the split rule in
  `../PR-SPEC-CONTRACT.md` is applied before opening the PR.

## Rollback contract

Reverting this PR removes the `dual_path_v1` rendezvous and returns all requests
to the inherited Decode-first Proxy flow. PR-01 coverage/probe facts remain
available but cannot leave Decode or authorize I/O.

## Review focus

- Are Scheduler responsibilities separated from Coordinator HTTP ownership?
- Is every non-HBM-complete coverage class decided by the PE Scheduler rather
  than locally on Decode?
- Is eligibility evaluated before round-robin state changes?
- Is Future registration ordered before dispatch on every path?
- Can callbacks be abused as arbitrary outbound HTTP targets?
- Is there any hidden data-plane side effect before PR-04 activation?

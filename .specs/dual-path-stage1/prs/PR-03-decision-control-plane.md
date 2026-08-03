# PR-03 Cross-Engine Decision Control Plane

- Series position: 3 of 6
- Spec status: `PLANNED`
- Depends on: PR-01
- Blocks: PR-05
- Implementation tasks: `DP-05` through `DP-12`
- User-visible behavior: protocol opt-in only; no active remote partial path
- Activation after merge: remote candidates complete a decision round trip,
  while Scheduler-visible partial selection remains rejected or shadow-only

## Goal

Complete the non-blocking DE-to-PE decision rendezvous without authorizing
partial-hit Store or P2P data-plane I/O.

## Merge-state contract

After this PR merges, an explicitly marked `dual_path_v1` remote-required
request can send a typed candidate through the existing Decode-first Proxy flow,
receive one PE commit through `/v1/path-decision`, and deliver it to the DE
Scheduler inbox. Ordinary Layerwise requests retain their current Proxy flow.
Until PR-06, a would-be `DE_PARTIAL_HIT` result cannot become a positive
first-winner accounting result and cannot authorize Store, Reverse, or Forward
I/O.

## In scope

- Active Stage 1 config cleanup and static-policy parsing.
- Typed request key, decision request, commit, and error schemas.
- Proxy `dual_path_v1` opt-in and decision Future registration before PE
  dispatch.
- `/v1/path-decision` validation and commit-once resolution.
- PE and DE `PathDecisionCoordinator` asynchronous HTTP ownership.
- DE decision inbox with Scheduler-thread drain.
- Duplicate-identical commit idempotency and conflicting-commit rejection.
- Decision timeout and PE-request-before-commit failure.
- Ordinary Proxy behavior and mixed ordinary/DualPath concurrency regression.
- A closed-loop test with a fake or constrained responder proving both schema
  endpoints have exercised producers and consumers.

## Out of scope

- Positive Scheduler-visible partial accounting.
- Reverse, Forward, or DE partial Store load.
- Worker ZMQ decision messages.
- Whole-application middleware.
- Post-commit fallback or decision retry protocol.
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
- Request and commit schemas reject invalid keys and invalid kind/use pairs.
- Same commit twice is idempotent; a different second commit is an error.
- PE HTTP failure before commit resolves a typed decision error.
- Timeout produces `DECISION_TIMEOUT` without blocking Scheduler hot paths.
- DE executor writes only to the inbox; Scheduler thread mutates commit state.
- Final PE response, not decision response, releases Prefill load accounting.
- Requests without `dual_path_v1` follow the existing `/v1/metaserver` path.
- Mixed ordinary and DualPath requests cannot resolve each other's Future.
- No test can observe partial Store/P2P I/O from a decision result.

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
- Incomplete partial-hit data-plane work cannot be activated by configuration.
- If review size becomes excessive, the split rule in
  `../PR-SPEC-CONTRACT.md` is applied before opening the PR.

## Rollback contract

Reverting this PR removes the `dual_path_v1` rendezvous and returns all
remote-required requests to the inherited Decode-first Proxy flow. PR-01 local
full hit remains functional because it never enters the rendezvous.

## Review focus

- Are Scheduler responsibilities separated from Coordinator HTTP ownership?
- Is Future registration ordered before dispatch on every path?
- Can callbacks be abused as arbitrary outbound HTTP targets?
- Is there any hidden data-plane side effect before PR-06 activation?

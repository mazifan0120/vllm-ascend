# Task-02 Detailed Spec — PE-Owned Decision Protocol and Policy

## 1. Status and authority

This document defines the implementation contract for Task-02 in the
[`DualPath Stage 1 Task Catalog`](../TASKS.md). It replaces the historical
`StoreCoverage`, `PathKind`, candidate, capability, and Proxy-Future decision
model for this Task.

The source baseline is:

- `vllm-ascend@24af3ee65b8ba45afa69a0ec0822b6120c0e9d25`
- sibling `vllm@0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`

Task-02 is transport-independent pure logic. It does not add Scheduler hooks,
open a network endpoint, contact Proxy, start Store or P2P I/O, or activate a
production route.

## 2. Outcome

Task-02 defines the minimal language and replaceable policy surface for one
path decision owned by the Prefill-side `DualPathConnectorScheduler`:

```text
PathDecisionRequest
    -> full Decode Store hit: Path.DE_READ
    -> otherwise: PathPolicy.choose(request)
    -> PathDecisionCommit or PathDecisionError
```

The only paths are:

```python
class Path(str, Enum):
    PE_READ = "PE_READ"
    DE_READ = "DE_READ"
```

`PathKind` is not part of the design.

## 3. Request identity

```python
@dataclass(frozen=True)
class DualPathRequestKey:
    decode_engine_instance_id: str
    decode_request_id: str
```

`decode_engine_instance_id` identifies one running Decode Engine instance. A
restart must create a different value so a delayed decision from an earlier
instance cannot bind to a new request.

`decode_request_id` is the request ID local to that Decode Engine instance.
The pair is the stable identity used by the request, decision result, direct
control channel, duplicate detection, and terminal cleanup.

Task-02 has one logical decision per request key. It does not define
`candidate_id`, `decision_attempt_id`, or a retry-generated identity.

## 4. Decision request

```python
@dataclass(frozen=True)
class PathDecisionRequest:
    request_key: DualPathRequestKey
    target_tokens: int
    decode_local_tokens: int
    decode_store_tokens: int
```

The fields are derived from the Task-01 `DecodeKVSnapshot`:

```text
target_tokens       = snapshot.target_tokens
decode_local_tokens = snapshot.local_tokens
decode_store_tokens = snapshot.store_tokens
```

Validation is strict:

```text
0 <= decode_local_tokens <= decode_store_tokens <= target_tokens
decode_local_tokens < target_tokens
```

The second invariant ensures an HBM-complete request never enters the decision
protocol. Task-01 returns `(0, False)` for that request and does not submit a
decision request.

The wire request deliberately excludes:

- `DecodeKVSnapshot` itself;
- `LoadSpec`, `LoadSpec.can_load`, or final Decode block IDs;
- a public `StoreCoverage` object or coverage enum;
- Prefill-local token counts;
- control endpoint, retry, ACK, or transport configuration.

`LoadSpec.can_load` is a post-allocation Store-I/O authorization bit. It is not
a Store capability and must not participate in path selection.

## 5. Store state derived from token facts

Store state is a derived fact, not a serialized type:

```text
decode_store_tokens == target_tokens
    -> full

decode_local_tokens < decode_store_tokens < target_tokens
    -> partial

decode_store_tokens == decode_local_tokens
    -> miss or unavailable
```

Miss and unavailable intentionally have the same decision input. Their local
diagnostic provenance remains owned by Task-01 and is not required by the PE
policy.

## 6. Fixed full-hit rule

A full Decode Store hit always selects `Path.DE_READ`:

```python
if request.decode_store_tokens == request.target_tokens:
    path = Path.DE_READ
else:
    path = policy.choose(request)
```

This rule is outside the replaceable policy. Therefore:

- full requests do not consume or change policy state;
- no later policy may redirect a full request to `Path.PE_READ`;
- the request still reaches the PE `DualPathConnectorScheduler`, because PE is
  the sole producer of the decision commit;
- later data-plane Tasks make the PE request a no-work request after the
  direct `DE_READ` commit is delivered.

## 7. Replaceable policy surface

Task-02 defines one lightweight protocol:

```python
class PathPolicy(Protocol):
    def choose(self, request: PathDecisionRequest) -> Path:
        ...
```

The policy is invoked only for non-full requests. Partial, miss, and
unavailable inputs all use the same policy call. A Store-miss `DE_READ` is a
valid non-full selection: its Store interval is empty, while later Tasks may
use the Decode HBM prefix, Reverse, PE computation, and Forward through the
same split-plan semantics.

Task-02 does not define:

- `PathPolicyContext` or `PathCapabilities`;
- an eligible-path collection;
- a metrics provider, metrics schema, or listener;
- a policy registry, dynamic loader, or `**kwargs` extension surface.

Future policies extend the implementation by satisfying `PathPolicy`. The
interface is intentionally limited to the stable request facts accepted in
this Task.

## 8. Round-robin policy

The initial policy implementation is:

```python
class RoundRobinPathPolicy:
    def __init__(self, rng: random.Random | None = None) -> None:
        self._next = (rng or random.Random()).choice(
            (Path.PE_READ, Path.DE_READ)
        )

    def choose(self, request: PathDecisionRequest) -> Path:
        selected = self._next
        self._next = (
            Path.DE_READ
            if selected is Path.PE_READ
            else Path.PE_READ
        )
        return selected
```

Rules:

1. The first non-full request chooses a random starting path.
2. Every subsequent unique non-full request alternates from the previous
   policy result.
3. Full requests bypass `choose()` and do not advance the policy.
4. The policy instance is owned by the PE `DualPathConnectorScheduler`.
5. The policy is Scheduler-thread-confined; Task-02 adds no lock.
6. Unit tests inject a seeded `random.Random` and never rely on a
   nondeterministic production seed.

## 9. Decision results

```python
@dataclass(frozen=True)
class PathDecisionCommit:
    request_key: DualPathRequestKey
    path: Path


@dataclass(frozen=True)
class PathDecisionError:
    request_key: DualPathRequestKey
    error_code: str
    message: str
```

```python
PathDecisionResult = PathDecisionCommit | PathDecisionError
```

Task-02 errors cover invalid protocol input, a conflicting request with the
same key, or a policy result that cannot be validated as `Path`. Transport
timeout, stale receiver, ACK, and delivery exhaustion belong to Task-03.

## 10. Duplicate and policy-state contract

`RoundRobinPathPolicy.choose()` is intentionally not a request registry. The
PE `DualPathConnectorScheduler` integration in Task-04 must retain:

```python
_path_decisions: dict[
    DualPathRequestKey,
    tuple[PathDecisionRequest, PathDecisionCommit],
]
```

The caller contract is:

```text
new key
    -> apply the full-hit rule or invoke choose() once
    -> retain request and commit

same key + identical request
    -> return the retained commit
    -> do not invoke choose()

same key + different request facts
    -> PathDecisionError(CONFLICTING_REQUEST)
    -> do not invoke choose()
```

Task-02 defines and unit-tests this semantic contract with frozen structural
equality. Task-04 owns the real request map, terminal cleanup, and proof that a
duplicate PE scheduling event does not call the policy twice.

## 11. Serialization contract

Every protocol type must have a strict, lossless serialization round trip.
Unknown enum values, missing required fields, extra fields in strict mode,
empty identity fields, boolean values supplied for integer fields, and invalid
token ordering are rejected before policy invocation.

Transport envelopes are excluded. In particular, the following Task-03 fields
must not be added to `PathDecisionRequest`:

```text
decode_control_endpoint
protocol framing/version
ACK/NACK status
retry counters or deadlines
```

## 12. Required unit tests

Focused CPU tests must cover:

1. request-key equality and invalid empty fields;
2. valid partial, full, and miss request construction;
3. every invalid token ordering;
4. HBM-complete request rejection;
5. full request always returns `Path.DE_READ` without invoking policy;
6. partial, miss, and unavailable-equivalent inputs invoke policy;
7. seeded random initial `Path.PE_READ` and `Path.DE_READ` cases;
8. alternation across unique non-full requests;
9. full requests do not advance round-robin state;
10. identical duplicate semantics do not call `choose()` twice;
11. conflicting duplicate facts produce a typed error;
12. request, commit, and error serialization round trips;
13. invalid path and malformed/untrusted payload rejection;
14. a second test policy can satisfy `PathPolicy` without changing the caller.

No Task-02 test may instantiate a real Scheduler, open a socket, call Proxy,
construct Store Worker metadata, or observe Store/P2P I/O.

## 13. Expected code surface

The detailed implementation plan should keep the surface focused under the
DualPath package, for example:

```text
vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision.py
tests/ut/distributed/kv_transfer/dual_path/test_path_decision.py
```

Protocol types may live in a small metadata module if required by existing
package structure, but Task-02 must not create transport or Scheduler files.

## 14. Out of scope

- ZMQ endpoints, direct Commit delivery, ACK, retry, timeout, and shutdown.
- Proxy changes or `/v1/path-decision`.
- PE or DE Scheduler hooks and request-local runtime maps.
- Store load, Reverse, Forward, Worker plans, or completion predicates.
- Metrics collection or a metrics-driven policy implementation.
- Policy configuration, registry, hot reload, or adaptive feedback.
- A second decision attempt or post-commit fallback.

## 15. Acceptance endpoint

Task-02 is accepted when pure unit tests prove:

- the minimal request and result schemas are strict and round-trip safely;
- full requests deterministically select `Path.DE_READ` outside policy;
- every non-full request is delegated to the replaceable `PathPolicy`;
- `RoundRobinPathPolicy` starts randomly and then alternates;
- the duplicate contract prevents a logical request from consuming policy
  state twice;
- no Scheduler, Proxy, direct-channel, Store, or P2P side effect exists.

After merge, no production route is activated. Task-03 provides direct
delivery, and Task-04 connects the policy to real PE/DE
`DualPathConnectorScheduler` lifecycle hooks.

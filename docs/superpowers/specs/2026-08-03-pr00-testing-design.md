# PR-00 DualPathConnector Foundation — Testing Design

**Date:** 2026-08-03
**Status:** Draft — pending user review
**PR:** PR-00 DualPathConnector Foundation (`.specs/dual-path-stage1/prs/PR-00-foundation.md`)
**Branch:** `dev/dualpath` @ `d72b067a`
**Authoritative spec:** `.specs/dual-path-stage1/prs/PR-00-foundation/DETAILED-SPEC.md`

## Purpose

Make PR-00 merge-ready by passing all 4 acceptance gates defined in
`PR-00-foundation.md`:

| Gate | Type | Command |
|------|------|---------|
| 1 | DualPath foundation UT | `pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py` |
| 2 | Mooncake Layerwise parent regression | `pytest -sv tests/ut/kv_offload/test_mooncake_layerwise_connector.py` |
| 3 | Lint and format | `bash format.sh ci` |
| 4 | NPU parity smoke | `pytest -sv tests/e2e/nightly/multi_node/dual_path/test_foundation_parity.py` |

Per `TRACKING.md`, all 4 gates are currently `NOT RUN`. The smoke config
shipped with the repo targets GLM-4.7-W8A8C8 at TP=8/DP=2 on 2 nodes × 16
NPU (32 cards), which exceeds available hardware. This design amends the
smoke topology while preserving the spec's parity semantics.

## Environment

| Resource | Available | Required |
|----------|-----------|----------|
| Host | 8 × Ascend 910B (32 GB each) | 2 cards (1 prefill + 1 decode) |
| User | `llq` (NPU denied) + `root` (NPU works) | root for NPU access |
| Python | 3.9.9 system | 3.10–3.12 (per vllm-ascend README) |
| CANN | 8.5.0 installed | Dockerfile pins 9.0.1 in image |
| vllm | not installed | v0.23.0 (matches Dockerfile) |
| K8s | TBD — verify before NPU phase | required for kubectl deploy |
| Docker | TBD — verify before image build | required |

## Architecture

```text
┌─ Host (8×910B 32GB, root) ──────────────────────────────────────┐
│                                                                  │
│  Phase 1 — CPU verification (Python 3.10 venv + vllm)            │
│    Gate 1: DualPath UT (49 tests, spec §11)                      │
│    Gate 2: Mooncake Layerwise parent regression                  │
│    Gate 3: bash format.sh ci (ruff + markdownlint)               │
│                                                                  │
│  Phase 2 — NPU parity smoke (K8s + self-built image)             │
│    Gate 4: baseline vs candidate parity                          │
│    ┌────────────┐        ┌────────────┐                         │
│    │ baseline   │  vs    │ candidate  │                         │
│    │ MooncakeLW │        │ DualPath   │                         │
│    │ Qwen3-8B   │        │ Qwen3-8B   │                         │
│    │ PD split   │        │ PD split   │                         │
│    └────────────┘        └────────────┘                         │
│    assertions: token equality + marker presence/absence          │
│                + lifecycle count parity                          │
│                                                                  │
│  Deployment: raw K8s manifests (3 pods per run):                 │
│    prefill pod + decode pod + proxy pod                          │
│    each pod runs vllm serve with explicit kv-transfer-config     │
└──────────────────────────────────────────────────────────────────┘
```

The two phases are independent: Phase 1 runs on CPU without NPU
hardware. Phase 2 runs only after Phase 1 passes, so code defects
surface cheaply before consuming NPU time.

### Why raw K8s manifests instead of InferNex Helm

`DualPathConfig.from_extra_config` (config.py:93-98) requires an
explicit `role` field (`"prefill"` or `"decode"`) in
`kv_connector_extra_config`. It does NOT derive `role` from `kv_role`.

InferNex's chart helper (`_helpers.tpl:442-572`) auto-fills `kv_role`
per node type (kv_producer for prefill, kv_consumer for decode) but
does NOT inject DualPath's `role` field. Both prefill and decode pods
share one `connectorConfig` entry, so:

- Setting `role: "prefill"` breaks decode (role/capability mismatch).
- Omitting `role` fails fast (ValueError: role required).
- Setting `role` per-node requires chart modifications.

Raw K8s manifests avoid this entirely: each pod's `vllm serve` command
includes the exact `--kv-transfer-config` JSON with `role` set
explicitly for that pod. No code changes to vllm-ascend or InferNex.

## Phase 1 — CPU Gates (1-3)

### 1.1 Environment setup

Create an isolated venv to avoid disturbing the system CANN install:

```bash
# Install python3.10 if missing (via system package manager or pyenv)
python3.10 -m venv ~/dualpath-venv
source ~/dualpath-venv/bin/activate
pip install --upgrade pip
pip install vllm==0.23.0 ruff markdownlint-cli2
cd /home/llq/dualpath/vllm-ascend
pip install -e .
```

Fallback if `python3.10` is unavailable via apt: install miniconda,
then `conda create -n dualpath python=3.10`.

### 1.2 Gate 1 — DualPath foundation UT

```bash
pytest -sv tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
```

**Scope (49 tests per spec §11):**

- Connector registry resolves `DualPathConnector`.
- `role=prefill` and `role=decode` build the correct Scheduler or Worker
  subclass and validate KV capability.
- Legacy `pe`/`de` roles, capability mismatches, and removed DualPath
  foundation fields fail fast.
- Parent runtime, Transfer Engine, threads, and KV buffers initialize
  exactly once (drift guard on facade state).
- Ten behavior-parity scenarios comparing Scheduler/Worker structured
  outputs and collaborator call traces.
- Section 9 inheritance and drift guards.

**CPU feasibility confirmed:** the test file (lines 27-40) stubs
`mooncake.engine`, `torch_npu`, and `uvloop` via `MagicMock` before
importing the connector modules. No NPU hardware is required.

### 1.3 Gate 2 — Mooncake Layerwise parent regression

```bash
pytest -sv tests/ut/kv_offload/test_mooncake_layerwise_connector.py
```

Verifies that the DualPath foundation commits did not break the parent
`MooncakeLayerwiseConnector` behavior. The TRACKING.md notes this file
is untouched by the PR-00 cleanup diff, so failures here indicate an
environmental issue, not a PR-00 regression.

### 1.4 Gate 3 — Lint and format

```bash
bash format.sh ci
```

Checks `ruff` (Python) and `markdownlint` (`.md` files). If
`markdownlint` modifies files, re-add and commit them before opening
the PR.

### 1.5 Phase 1 completion criteria

- All three commands return exit code 0.
- Results appended to `TRACKING.md` evidence log.
- PR-00 status advances from `IN_PROGRESS` to `LOCAL_READY` for the
  CPU portion.

## Phase 2 — NPU Parity Smoke (Gate 4)

### 2.1 Spec amendment

The shipped smoke config (`GLM-4.7-W8A8C8-DualPath-parity.yaml`)
specifies a topology that exceeds available hardware. This amendment
preserves the spec's parity semantics at a smaller scale.

| Parameter | Original (shipped) | Amended |
|-----------|-------------------|---------|
| Model | GLM-4.7-W8A8C8 | **Qwen/Qwen3-8B** |
| Topology | 2 physical nodes × 16 NPU | **K8s: prefill pod + decode pod + proxy pod** |
| Cards | 32 total (TP=8, DP=2) | **2 total (TP=1, DP=1)** |
| Deployment | bare-metal multi-node pytest | **raw K8s manifests + custom image** |
| Test runner | `test_foundation_parity.py` (multi_node framework) | **standalone parity script** |

**Justification:** PR-00 is a behavior-preserving alias. The parity
proof requires:

1. Token-identical output for identical prompts (temperature=0).
2. DualPath scheduler/worker init markers present.
3. Forbidden markers (AscendStore, PathDecision, round-robin,
   DE_PARTIAL_HIT, DE_LOCAL_FULL_HIT) absent.
4. Lifecycle counts (registration, admission, transfer completion,
   recv thread) match between baseline and candidate.

None of these depend on model size or card count. Qwen3-8B at TP=1
fits in a single 910B card (32 GB) with ample headroom. The amended
config exercises the real `MooncakeLayerwiseConnector` →
`DualPathConnector` drop-in replacement path end-to-end.

### 2.2 Image build

```bash
cd /home/llq/dualpath/vllm-ascend
git checkout dev/dualpath   # already on this branch
docker build \
  -t <REGISTRY>/dualpath-vllm-ascend:pr0 \
  --build-arg SOC_VERSION=ascend910b1 \
  -f Dockerfile .
docker push <REGISTRY>/dualpath-vllm-ascend:pr0
```

The Dockerfile (verified) builds on `quay.io/ascend/cann:9.0.1-910b-ubuntu22.04-py3.12`,
installs vllm v0.23.0, then `pip install -e` the checked-out vllm-ascend.
The resulting image contains `DualPathConnector` registered in the
connector factory.

`<REGISTRY>` is a placeholder — see Open Questions below.

### 2.3 K8s manifests

Each run deploys 3 pods: prefill, decode, and proxy. Each pod's
`vllm serve` command includes an explicit `--kv-transfer-config` JSON
string, so `role` (required by `DualPathConfig`) and `kv_role` are set
per-pod rather than templated.

**`baseline-prefill-pod.yaml`** (MooncakeLayerwiseConnector, producer):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: baseline-prefill
  labels:
    app: parity-baseline
    openfuyao.com/pdRole: prefill
spec:
  restartPolicy: Never
  containers:
    - name: vllm
      image: <REGISTRY>/dualpath-vllm-ascend:pr0
      command: ["/bin/bash", "-c"]
      args:
        - |
          source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
          vllm serve "Qwen/Qwen3-8B" \
            --host 0.0.0.0 --port 8000 \
            --tensor-parallel-size 1 \
            --seed 1024 --enforce-eager \
            --max-num-seqs 8 --max-model-len 8192 \
            --max-num-batched-tokens 8192 \
            --trust-remote-code --no-enable-prefix-caching \
            --gpu-memory-utilization 0.8 \
            --kv-transfer-config \
            '{"kv_connector": "MooncakeLayerwiseConnector",
              "kv_role": "kv_producer",
              "kv_port": "30000",
              "kv_connector_extra_config": {
                "prefill": {"dp_size": 1, "tp_size": 1},
                "decode":  {"dp_size": 1, "tp_size": 1}
              }
            }'
      resources:
        limits:
          huawei.com/Ascend910: 1
      ports:
        - containerPort: 8000
```

**`baseline-decode-pod.yaml`** (MooncakeLayerwiseConnector, consumer):

```yaml
# Same as prefill except:
#   name: baseline-decode
#   openfuyao.com/pdRole: decode
#   --max-num-seqs 4   (smaller to leave VRAM headroom)
#   "kv_role": "kv_consumer"
#   "kv_port": "30200"
```

**`candidate-prefill-pod.yaml`** (DualPathConnector, producer):

```yaml
# Same as baseline-prefill except:
#   name: candidate-prefill
#   labels.app: parity-candidate
#   kv_connector: "DualPathConnector"
#   kv_connector_extra_config adds: "role": "prefill"
#   Full --kv-transfer-config:
#     '{"kv_connector": "DualPathConnector",
#       "kv_role": "kv_producer",
#       "kv_port": "30000",
#       "kv_connector_extra_config": {
#         "role": "prefill",
#         "prefill": {"dp_size": 1, "tp_size": 1},
#         "decode":  {"dp_size": 1, "tp_size": 1}
#       }
#     }'
```

**`candidate-decode-pod.yaml`** (DualPathConnector, consumer):

```yaml
# Same as candidate-prefill except:
#   name: candidate-decode
#   openfuyao.com/pdRole: decode
#   --max-num-seqs 4
#   "kv_role": "kv_consumer"
#   "kv_port": "30200"
#   "role": "decode"
```

**`proxy-pod.yaml`** (shared by both runs; uses the proxy-server image
from InferNex or a standalone proxy script):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: parity-proxy
  labels:
    app: parity-proxy
spec:
  restartPolicy: Never
  containers:
    - name: proxy
      image: <REGISTRY>/dualpath-vllm-ascend:pr0
      command: ["/bin/bash", "-c"]
      args:
        - |
          source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
          python3 /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
            --host 0.0.0.0 --port 8000 \
            --prefiller-label "openfuyao.com/pdRole=prefill" \
            --decoder-label "openfuyao.com/pdRole=decode" \
            --prefiller-container vllm --decoder-container vllm \
            --prefiller-port-name http --decoder-port-name http \
            --namespace default \
            --port 8000 \
            --discovery-interval 10
      ports:
        - containerPort: 8000
```

> The proxy script
> (`examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py`)
> is shipped inside the vllm-ascend repo and discovers prefill/decode
> pods via the `openfuyao.com/pdRole` label. Both baseline and
> candidate runs reuse the same proxy manifest; only the prefill and
> decode pod labels change between runs (`parity-baseline` vs
> `parity-candidate`), but the `pdRole` label stays constant so the
> proxy can find them.

### Differences between baseline and candidate runs

The **only** differences between the two runs are:

1. `kv_connector`: `"MooncakeLayerwiseConnector"` → `"DualPathConnector"`
2. `kv_connector_extra_config`: adds `"role": "<prefill|decode>"` per pod
3. Pod names/labels: `baseline-*` → `candidate-*`

Everything else — model, TP size, DP size, max-model-len, seed,
prompts, proxy script — is identical.

### 2.4 Parity verification procedure

The existing `test_foundation_parity.py` uses
`tests/e2e/nightly/multi_node/internal_dp/scripts/multi_node_config.py`,
which resolves separate physical node IPs via `resolve_cluster_ips`.
It cannot run on a single K8s cluster without modification. Instead, a
standalone script orchestrates the two runs and performs assertions.

**Procedure:**

```text
1. kubectl apply -f proxy-pod.yaml
   kubectl apply -f baseline-prefill-pod.yaml
   kubectl apply -f baseline-decode-pod.yaml
   → wait for all 3 pods Ready (kubectl wait --for=condition=Ready)

2. Send 6 fixed prompts (temperature=0, seed=1024, max_tokens=64)
   to the proxy pod's ClusterIP:8000 via httpx
   → record content + completion_tokens per prompt

3. kubectl logs baseline-prefill > baseline-prefill.log
   kubectl logs baseline-decode  > baseline-decode.log

4. kubectl delete pod baseline-prefill baseline-decode
   → wait for cleanup (settle 5s); proxy pod stays running

5. kubectl apply -f candidate-prefill-pod.yaml
   kubectl apply -f candidate-decode-pod.yaml
   → wait for pods Ready

6. Send the same 6 fixed prompts to the proxy
   → record content + completion_tokens per prompt

7. kubectl logs candidate-prefill > candidate-prefill.log
   kubectl logs candidate-decode  > candidate-decode.log

8. Run parity_verify.py
```

**Fixed prompts** (from spec §12, `FIXED_PROMPTS`):

```python
PROMPTS = [
    "What is two plus three?",
    "Name the largest planet in our solar system.",
    "Give one synonym for quick.",
    "What color is a clear daytime sky?",
    "Complete the sequence: 2, 4, 6,",
    "Name the author of Hamlet.",
]
SEED = 1024
MAX_TOKENS = 64
```

### 2.5 Parity assertions

`parity_verify.py` performs 4 assertion classes:

| # | Assertion | Baseline | Candidate | Pass criterion |
|---|-----------|----------|-----------|----------------|
| A | Token equality | 6 prompt contents | 6 prompt contents | per-prompt string equal |
| B | Positive markers | "Initializing Mooncake Scheduler" | "Initializing DualPath Scheduler" | each > 0 in respective logs |
| C | Negative markers absent | n/a | AscendStore, StoreConnector, PathDecision, round-robin, round_robin, Reverse transfer, DE_PARTIAL_HIT, DE_LOCAL_FULL_HIT | none present in candidate logs |
| D | Lifecycle count parity | scheduler_init, registration, admission, transfer_completion, recv_thread counts | same markers | counts equal (except worker_init: parent logs twice, subclass once — only assert > 0) |

Marker sources (verified against `mooncake_layerwise_connector.py`
line numbers cited in `test_foundation_parity.py:65-77`):

| Marker | Source line | Node |
|--------|-------------|------|
| "Initializing Mooncake Scheduler" | :798 | prefill + decode |
| "Initializing DualPath Scheduler" | connector.py:68 | prefill + decode |
| "Initializing Mooncake work" | :1135, :1171 (×2) | prefill + decode |
| "Initializing DualPath Worker" | connector.py:92 | prefill + decode |
| "KVCacheRecvingLayerThread listening on" | :607 | decode only |
| "Send request:" | :953 | decode only |
| "Number of completed KV cache recv requests" | :1425 | decode only |
| "num_blocks: " | registration | prefill + decode |

### 2.6 Resource requirements

| Resource | Required | Available |
|----------|----------|-----------|
| NPU cards | 2 (1 prefill + 1 decode) | 8 ✅ |
| VRAM per card | ~16 GB (Qwen3-8B TP=1) | 32 GB ✅ |
| K8s + npu-operator + LWS | required | verify before Phase 2 |
| Docker | required | verify before image build |
| Image registry (push/pull) | required | verify — see Open Questions |

### 2.7 Phase 2 completion criteria

- All 4 assertion classes pass.
- Baseline and candidate logs captured and saved as evidence
  artifacts (not committed to repo; referenced by path in TRACKING.md).
- Results appended to `TRACKING.md` evidence log with exact commands.
- PR-00 status advances to `IN_REVIEW` (all 4 gates passed).

## Spec Amendment Documentation

After Gate 4 passes, update two files:

### `TRACKING.md` evidence log

```text
YYYY-MM-DD | PR-00 | pytest DualPath UT | PASS/FAIL | <result>
YYYY-MM-DD | PR-00 | pytest mooncake_layerwise regression | PASS/FAIL | <result>
YYYY-MM-DD | PR-00 | bash format.sh ci | PASS/FAIL | <result>
YYYY-MM-DD | PR-00 | NPU parity smoke (Qwen3-8B, K8s, TP=1) | PASS/FAIL | <result>
```

### `TRACKING.md` spec amendment note

```text
Spec amendment YYYY-MM-DD:
- Model: GLM-4.7-W8A8C8 → Qwen3-8B (parity proof does not require large model)
- Topology: 2 nodes × 16 NPU → K8s prefill pod + decode pod (1 card each)
- Deployment: bare-metal multi-node → raw K8s manifests + custom vllm-ascend image
- Test runner: test_foundation_parity.py (multi_node framework) → standalone
  parity_verify.py (K8s pod orchestration via kubectl)
- Justification: PR-00 is behavior-preserving alias; parity semantics
  (token equality + marker presence/absence + lifecycle counts) are
  independent of model size and card count. The 4 assertion classes
  in §2.5 are lifted directly from spec §12.
```

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Python 3.10 not installable on host | Low | Blocks Phase 1 | Fallback to miniconda |
| vllm v0.23.0 install conflicts | Medium | Blocks Phase 1 | Copy deps from built Docker image |
| K8s / npu-operator not deployed | Medium | Blocks Phase 2 | Install per upstream docs before Phase 2 |
| Docker not available | Low | Blocks image build | Install docker-ce |
| Image registry inaccessible | Medium | Blocks Phase 2 | Use `docker save` + `docker load` on host |
| Mooncake transfer engine needs Redis metadata server | Low | Blocks PD transfer | Run a Redis sidecar or standalone pod; Qwen3-8B PD parity does not require Mooncake store, only the transfer engine |
| Proxy script can't discover prefill/decode pods | Low | Blocks requests | Verify `openfuyao.com/pdRole` labels match the proxy's `--prefiller-label`/`--decoder-label` args |
| `role` field set wrong per pod | Medium | Wrong subclass built / ValueError | Each pod manifest sets `role` explicitly in `--kv-transfer-config`; verify before `kubectl apply` |
| Qwen3-8B needs trust-remote-code | Low | Model fails to load | `--trust-remote-code` already in `vllm serve` args |
| Pods can't reach each other (network policy / DNS) | Medium | PD transfer fails | Use ClusterIP services or verify pod-to-pod networking; K8s default allows pod-to-pod |

## Open Questions

These must be answered before Phase 2 execution but do not block the
spec or the implementation plan:

1. **Image registry**: what `<REGISTRY>` should be used? Options:
   - Local registry on the host (`registry:2` container)
   - Internal corporate registry
   - `cr.openfuyao.cn` (if push access available)

2. **K8s cluster state**: is K8s already deployed on this host with
   npu-operator and LWS? Run `kubectl get pods -A` to verify.

3. **Model download**: does the host have network access to
   HuggingFace or ModelScope? If air-gapped, the model must be
   pre-downloaded and mounted via a hostPath volume.

## Out of Scope

- Performance benchmarks (PR-00 adds no new data path).
- MultiConnector integration testing (PR-00 spec explicitly excludes
  MultiConnector behavior changes).
- PD-Orchestrator or elastic-scaler interaction.
- Hermes-router routing through DualPath.
- cache-indexer L3 KV-aware integration.
- PR-01 through PR-06 feature behavior.

## Rollback

If Phase 1 fails: fix code defects on `dev/dualpath`, re-run gates 1-3.
No external artifacts created.

If Phase 2 fails:

- `kubectl delete pod` baseline and candidate prefill/decode pods.
- `kubectl delete pod parity-proxy` if no longer needed.
- Remove pushed image from registry (if applicable).
- No code changes to vllm-ascend repo (image was built from existing
  commit).
- Diagnosis of failure captured in TRACKING.md before retry.

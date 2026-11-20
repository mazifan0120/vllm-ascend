# Layerwise KV offloading runtime

This folder is a standalone companion to the feature report. It uses Bash,
Docker and Python 3.9+ standard library. No deployment repository, predecessor
container, private benchmark image, external design PDF or orchestration runtime
is needed.

You must obtain the **feature inference image tar** separately from its provider,
and prepare the complete GLM-5.2-w4a8c8 checkpoint on each A3 node. The ordinary
upstream image does not contain this feature. Model weights and the large image
archive are not included here.

The measured image content ID is:

```
sha256:3de4ba4a2801ea65b97a7d32b02bf837abb7a5744e0b82605efa293cec5ada44
```

After loading it, assign the local alias used in `cluster.example.json`:

```bash
docker load -i glm52-layerwise-kv-image.tar
docker tag sha256:3de4ba4a2801ea65b97a7d32b02bf837abb7a5744e0b82605efa293cec5ada44 glm52-layerwise-kv:20261119
cp cluster.example.json cluster.json
# Fill the four communication IPv4 addresses and the NIC name.
python3 render.py --config cluster.json --profile performance --output launch-performance
```

Copy each rendered node directory to its corresponding host. On all four hosts:

```bash
MODEL_DIR=/absolute/path/to/GLM-5.2-w4a8c8 bash run-node.sh create
```

This uses the entire node. It refuses to proceed if `npu-smi` reports existing
NPU processes. It does not stop unrelated work or replace an existing container.
The host driver mount list reflects the measured A3 installation; adapt it if
host paths differ. Only private container shm is remounted for huge pages.

Start Master on p0, check its metrics endpoint, start engines on all nodes,
wait for all role APIs to return 200, then start the proxy on p0:

```bash
bash run-node.sh master    # p0 only
bash run-node.sh engines   # every node
bash run-node.sh proxy     # p0 only, after every engine API is healthy
```

Use `/healthcheck` on the proxy to verify 2 P / 8 D (performance) or 4 P / 16 D
(capacity). Logs remain in the rendered node directory's `logs/`. Stop requests,
check running/waiting are zero, and use `bash run-node.sh stop` on every host to
stop only this labelled deployment. No container or log removal is automatic.

For the independent capacity profile, render with `--profile capacity` to a new
output directory and repeat on idle nodes. The profiles share devices and some
API ports, so they cannot run concurrently.

`tools/http_bench.py` reads prepared prompt JSONL and performs closed-loop SSE
requests without extra Python packages. It checks completion, token counts and
reports TTFT, TPOT and throughput. It does **not** generate token-calibrated
prompts, warm a prefix working set, control P routing, clear local cache or prove
external cache hits. Follow the report protocol to perform those steps.

`tools/smoke.py` runs short/multi-chunk/prefix semantic cases. Use
`--skip-log-audit` for client-only execution, then collect four-node logs and run
`--audit-results` separately to verify Ready/Closed. `tools/long_probe.py`
validates a real near-1M request against the service tokenizer.

Validation status: the supplied engine configurations and feature image have
been exercised on the measured cluster. The portable host wrapper has local
render/shell checks; the reference HTTP client has synthetic CPU completion and
error checks. Neither newly packaged wrapper nor reference client has been used
to repeat the full experiment on another cluster. Historical measurements in
the report are unchanged.

## Build the report bundle from this source repository

From the repository root:

```bash
python3 examples/disaggregated_prefill_v1/layerwise_kv_offload/package_report.py
```

The tracked report is `docs/source/user_guide/feature_guide/layerwise_kv_offload.md`.
Figures and compact measurement evidence are under `docs/source/assets/layerwise_kv_offload/`.
The ZIP and checksum are generated under `build/reports/`; `--output PATH` overrides this.
All report/runtime/evidence inputs are tracked in this repository; model weights and
the large inference image tar are supplied separately.

> 用途：按 OrangeLLM/vllm-ascend issue #69 的测试口径，对「DCP2 基线」与「layerwise 特性档」做四档同口径复测（0%/90% × C32/C64），给出可复核的数字、差异幅度与过程中的坑。
> 关联：[[23 DCP2 与 layerwise 四组对照 — 0命中与90命中 全过程与数据]]
> 关联：[[22 DCP 与 layerwise 对照 — 交接文档]]
> 关联：[[21 DCP 与 layerwise 对照 — 数据与归因]]
> 日期：2026-11-20
> 口径：55,000 in / 700 out；每档 112 条正式请求 = **112 个独立前缀组 × 每组只出 1 条**；命中来源用 P 侧 counter delta 判定；每档跑 1 次，无置信区间。

# issue #69 口径复测 — DCP2 基线 vs layerwise 特性四档

## 0 一句话结论

在 issue 的口径下，**layerwise 四档全部胜出**：0 命中吞吐 +59~60%、TTFT −36~37%、TPOT −40%；90 命中吞吐 +31~52%、TTFT −14~31%、TPOT −27~34%，且**高命中 C64 的优势明显收窄**——与 issue 报告"高命中 C64 差距收窄"的规律一致。

| 档位 | DCP2 req/s | LW req/s | DCP2 TTFT 均值 | LW TTFT 均值 | DCP2 TPOT | LW TPOT | DCP2 输出吞吐 | LW 输出吞吐 |
|---|---|---|---|---|---|---|---|---|
| 0% C32 | 0.276 | **0.439** | 74.88 s | **47.89 s** | 36.7 ms | **22.1 ms** | 192.9 | **307.4** |
| 0% C64 | 0.275 | **0.439** | 145.21 s | **90.85 s** | 39.5 ms | **23.5 ms** | 192.8 | **307.3** |
| 90% C32 | 0.699 | **1.062** | 5.59 s | **3.88 s** | 49.7 ms | **33.0 ms** | 489.4 | **743.6** |
| 90% C64 | 0.982 | **1.290** | 12.26 s | **10.49 s** | 59.7 ms | **43.8 ms** | 687.2 | **902.9** |

## 1 为什么要换口径（旧口径错在哪）

以前的"90% 命中"用的是 GSP 数据集：**2 个前缀组 × 64 条请求**。后果是：

| | 旧口径（GSP） | issue 口径 |
|---|---|---|
| 前缀组数 | 2 组 | **112 组** |
| 每组被复用次数 | 64 次 | **1 次** |
| 工作集 | 2 × 48K ≈ 96 K tokens | 112 × 49,152 = **5.5 M tokens** |
| 占单 P HBM KV 容量 | **约 3%** | **1.75 倍** |
| 实际命中来源 | **本地 HBM APC** | **外部 Store** |

旧口径下，第一条请求就把前缀写进 P 的 HBM，后面 63 条全部本地命中——测的不是"外部缓存/跨层恢复"，而是普通 APC。**判据也错了**：当时看的是 `External prefix cache hit rate`，该指标在 0 命中也显示 100%（KV 本来就常驻 host DRAM）。

正确判据是 P 侧 counter：

```text
外部命中率 = Δ vllm:external_prefix_cache_hits_total / Δ vllm:prompt_tokens_total
本地命中   = Δ vllm:prefix_cache_hits_total  必须为 0
```

## 2 issue 的测试方法（复刻要点）

1. **负载**：55,000 in / 700 out；C32、C64；每档 **112 条正式请求**；seed 20261106；`ignore_eos=true`；关 profiler。
2. **"90%" 的定义**：每条请求约九成**输入 token** 能从外部缓存恢复，不是九成请求命中。块对齐（128）后实测 49,152/55,000 ≈ 89~90%。
3. **工作集必须超过单 P 的 HBM KV**：单 P 容量 × 1.0 不足以保证冷；用 112 组 × 49,536 = 5.55 M tokens（约为单 P 容量的 1.75 倍）。
4. **每组只出 1 条正式请求** —— 结构上保证"本地不可能二次命中"，这比 reset 更根本。
5. **预热两波**（DCP 侧）：同一批 112 组各预热一次、共 224 条，让两个 P 都遍历完整工作集；随后**对两个 P 显式 `POST /reset_prefix_cache`**（保留外部 Store），再进正式阶段。
6. **0 档**：每请求内容全新（每轮不同 salt），完全不预热。
7. **统计口径**：TTFT 由客户端测（含代理、排队、P 处理、P→D、D 首 token）；TPOT = 逐请求 `(E2E − TTFT)/(实际输出 − 1)` 再平均；输出吞吐 = 正式阶段输出 tokens / 时长。

### 2.1 一次踩坑：共用前缀会让 90 档退化成 HBM 命中

第一次跑 90 档时把 112 条请求设计成**共用一个公共前缀**，结果：

| | 目标 | 实测（错误版） |
|---|---|---|
| 外部命中 | ~90% | **5.2%** |
| 本地命中 | 0% | **83.8%** |

原因正是"第一条正式请求把公共前缀写进本地 HBM，后 111 条全本地命中"。改成**每组独立前缀**后立刻变成 88.9%，本地命中 0。

> 这两组错误数据保留在 `logs/issue-method/90hit-c32`、`90hit-c64`，仅作审计，**不要引用**。

## 3 两套栈的部署参数

### 3.1 DCP2（基线）

| 项 | P（.237/.31） | D（.230/.226） |
|---|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc1-a3` | 同 |
| TP / DP | 16 / 2 | **2 / 16** |
| DCP / PCP | 16 / 1 | **2 / 1** |
| MTP | 1（eager） | 5（draft eager + target FULL） |
| batched / seqs | 8192 / 64 | 192 / **32** |
| max_model_len | 1,048,576 | **524,288** |
| gpu 利用 | 0.90 | **0.92** |
| HCCL_BUFFSIZE | 400 | **800** |
| 图捕获 | eager | 默认推导，**最大 192** |
| 其他 | 本地 APC 开 | `sparse_li_c8`、`recompute_scheduler` |

入口 `http://172.168.4.237:8498`，Master 50165/metrics 52165，Store 段 40 GiB，指纹 `vllm-0.30.0-tp2-dp16-ep-fdde765e`。

### 3.2 layerwise（issue 特性档）

| 项 | P（.237/.31） | D（.230/.226） |
|---|---|---|
| 镜像 | `swr.../vllm-ascend-kvoffloading:v1119`（= mc-rewrite-43e124eb5，ID `3de4ba4a2801`） | 同 |
| TP / DP | 16 / 2 | **4 / 8** |
| DCP / PCP | 1 / 1 | **1 / 1**（sparse offload 不支持 CP>1） |
| MTP | 1（eager） | 5（draft eager + target FULL） |
| batched / seqs | **16384** / 64 | 192 / **8** |
| max_model_len | **1,048,576**（issue 写 131072，按需改 1M） | 1,048,576 |
| gpu 利用 | 0.90 | 0.92 |
| 图捕获 | eager | `mode 3` + `[1,2,4,6,12,18,24,30,36,42,48]` |
| 共享槽 / prefetch / GET | **5 / 3 / 0（整层）** | — |
| 连接器 | MultiConnector：MooncakeLayerwiseConnector + AscendStoreConnector(use_layerwise) | MooncakeLayerwiseConnector + `layerwise_host_offload` |
| 其他 | **APC 关闭** | `dram_size_per_dp_GB=100`、`topk_buffer_size=4096`、`lru_max_threads=16`、`AsyncLayerwiseDecodeScheduler` |

入口 `http://172.168.4.237:8496`（`--prefill-restore-only`），Master 50163/metrics 52163，Store 段 40 GiB，指纹 `vllm-0.30.0-tp4-dp8-ep-549644c5`。

## 4 四档完整数据

112 条/档，全部 **112/112 成功**。

| 档位 | 栈 | req/s | 时长 | TTFT 均值 | P50 | P99 | TPOT | TPOT P99 | ITL | 输出吞吐 | 外部命中 | 本地命中 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0% C32 | DCP2 | 0.276 | 406.3 s | 74.88 | 79.64 | 107.74 | 36.7 | 55.0 | 151.2 | 192.9 | 0.13% | 0% |
| 0% C32 | LW | 0.439 | 255.0 s | 47.89 | 51.10 | 68.28 | 22.1 | — | 88.2 | 307.4 | 0.00% | 0% |
| 0% C64 | DCP2 | 0.275 | 406.7 s | 145.21 | 181.99 | 216.28 | 39.5 | 55.8 | 152.0 | 192.8 | 0.00% | 0% |
| 0% C64 | LW | 0.439 | 255.1 s | 90.85 | 113.52 | 134.87 | 23.5 | — | 91.7 | 307.3 | 0.00% | 0% |
| 90% C32 | DCP2 | 0.699 | 160.2 s | 5.59 | 3.99 | 15.89 | 49.7 | 71.4 | 184.3 | 489.4 | 89.09% | 0% |
| 90% C32 | LW | 1.062 | 105.4 s | 3.88 | 2.10 | 13.96 | 33.0 | — | 124.8 | 743.6 | 88.91% | 0% |
| 90% C64 | DCP2 | 0.982 | 114.1 s | 12.26 | 8.42 | 31.25 | 59.7 | 90.8 | 219.1 | 687.2 | 88.66% | 0% |
| 90% C64 | LW | 1.290 | 86.8 s | 10.49 | 6.46 | 28.14 | 43.8 | — | 162.3 | 902.9 | 89.52% | 0% |

### 4.1 命中证据（P 侧 token 计数，正式阶段前后 delta）

| 档位 | prompt tokens | local_compute | local_cache_hit | external_kv_transfer |
|---|---|---|---|---|
| DCP2 0% C32 | 6,165,425 | 6,157,233 | 0 | 8,192（0.13%，噪声） |
| DCP2 0% C64 | 6,161,564 | 6,161,564 | 0 | 0 |
| DCP2 90% C32 | 6,179,497 | 674,473 | 0 | **5,505,024** |
| DCP2 90% C64 | 6,209,173 | 704,149 | 0 | **5,505,024** |
| LW 0% C32 | 6,165,425 | 6,165,425 | 0 | 0 |
| LW 0% C64 | 6,161,564 | 6,161,564 | 0 | 0 |
| LW 90% C32 | 6,215,639 | 689,623 | 0 | **5,526,016** |
| LW 90% C64 | 6,169,746 | 646,546 | 0 | **5,523,200** |

90 档外部恢复量 ≈ 112 × 49,280，两套栈一致；**本地命中全程为 0**，说明测到的确实是外部 Store 路径。

## 5 layerwise 相对 DCP2 的变化

| 档位 | req/s | TTFT 均值 | TTFT P99 | TPOT | 输出吞吐 |
|---|---|---|---|---|---|
| 0% C32 | **+59.1%** | **−36.0%** | −36.6% | **−39.8%** | **+59.4%** |
| 0% C64 | **+59.6%** | **−37.4%** | −37.6% | **−40.5%** | **+59.4%** |
| 90% C32 | **+51.9%** | −30.6% | −12.1% | −33.6% | +51.9% |
| 90% C64 | **+31.4%** | −14.4% | −10.0% | −26.6% | +31.4% |

16 个格子无一反转。

## 6 与 issue 报告结论的一致性

| | issue 报告（特性 vs DCP） | 本次实测（LW vs DCP2） |
|---|---|---|
| TTFT 均值 | −4.0% ~ −46.6% | −14.4% ~ −37.4% |
| TPOT | −26.7% ~ −39.6% | **−26.6% ~ −40.5%** |
| 输出吞吐 | +25.2% ~ +79.4% | +31.4% ~ +59.6% |
| 差距收窄点 | 高命中 C64 TTFT 几乎追平 | 高命中 C64 TTFT 差距从 −30.6% 收窄到 −14.4% |

TPOT 的改善区间几乎完全重合，说明这不是测量巧合。

## 7 四点观察

1. **0 命中下两套栈都被 P 卡死**：C32 与 C64 吞吐完全相同（DCP2 0.276/0.275；LW 0.439/0.439），加并发只把 TTFT 翻倍（纯排队）。差异来自 P 本身——LW 用 16384 batched + 5 个共享槽，prefill 快约 60%（6.17 M tokens / 255 s vs 406 s）。
2. **命中链路完全不同但都验到了**：DCP2 是"外部 Store → P 直接命中、本地 APC 为 0"；LW 是"外部 Store → P 整层恢复 → 逐层推给 D"，P 侧 APC 本来就关着。
3. **预热签名**：LW 预热第一波 211 s、第二波 38 s（5.6×）；DCP2 是 340 s / 21.9 s（15.5×）。第二波变快即证明恢复链路生效。
4. **TPOT 两边都随并发退化，LW 起点更低**：LW 22.1 → 33.0 → 43.8 ms；DCP2 36.7 → 49.7 → 59.7 ms。

## 8 复现过程中踩到的坑（复现价值最高）

| 坑 | 现象 | 处理 |
|---|---|---|
| **DCP2 侧：DCP=2 与 TP=4 互斥** | `Assertion failed, DCP for SFA with replicated indexer ... dcp_size(2) is pcp_size(1) or tp_size(4)*pcp_size(1)(4)` | 源码 `platform.py` 里 `supported_dcp_sizes = {pcp, tp×pcp}`。要 DCP=2 就必须 TP=2 → DP=16 |
| **DCP2 侧：MoE 通信窗口不够** | `HCCL_BUFFSIZE_EP is too SMALL, maxBs=96, epWorldSize=32` | TP2 时每 rank 派发 192/2=96 token（TP4 只有 48），`HCCL_BUFFSIZE` 400 → **800** |
| **DCP2 侧：1M 上下文 KV 放不下** | `To serve at least one request with max seq len (1048576), (49.94 GiB KV cache is needed...)` | TP2/DCP2 每 rank 承担一半 KV，1M 需 49.94 GiB → D maxlen 降 **524,288** |
| **v0.30 配置 schema** | `ValidationError: Unexpected keyword argument: fuse_muls_add` | `fuse_muls_add` 下沉到 `ascend_compilation_config`、`recompute_scheduler_enable` 下沉到 `scheduler_config` |
| **layerwise 侧：代理端口配错** | 首轮 97/112 失败，proxy 日志 `Decode request ... failed on 172.168.4.226:8420: All connection attempts failed` | .226 的 D 实例在 **8424–8427**，代理却配成 8420–8423 → 改成 `8420 8421 8422 8423 8424 8425 8426 8427` |
| **layerwise 侧：镜像 tag 差异** | 三个节点本地没有 `v1119` tag，docker 自动 pull（manifest digest 与本地 ID 不同） | 核对后四节点 config ID 均为 `3de4ba4a2801`，一致，可继续 |
| **命中判定踩坑** | 112 条共用前缀 → 本地命中 83.8%、外部仅 5.2% | 改为 **112 个独立前缀组** |

## 9 局限

1. 两套栈**不是 issue 的原始配对**：基线是 DCP2（P dcp16 / D dcp2、TP2×DP16），特性侧按 issue 特性档但 **P max_model_len 用了 1M**（issue 写 131072）。
2. 每档只跑 **1 次 112 条**，无置信区间；同环境历史噪声 ±5–7%，**±10% 以内的差异（如 90% C64 的 TTFT）不宜当结论**。
3. thinking 未显式关闭，两套一致，内部可比；与 issue "关 thinking" 有差异。
4. 0% C32 DCP2 的 8,192 token 外部命中（0.13%）是随机内容偶发重合，不影响结论。
5. 图捕获在 DCP2 侧最大 192、LW 侧最大 48，二者都覆盖了各自 `max_num_seqs × (MTP+1)`，未踩 cliff。

## 10 数据与复现

### 10.1 数据位置

```text
/mnt/sfs_turbo/yy/bench-perf/logs/
├── issue-method/               # DCP2 基线：0hit-c32, 0hit-c64, 90hitB-c32, 90hitB-c64
│   ├── stats.json
│   └── 90hit-c32, 90hit-c64    # 旧错误版（共用前缀），仅留档
├── issue-method-lw/            # layerwise：lw-0hit-c32/64, lw-90hit-c32/64
└── cmp-dcp2-vs-lw.json         # 两栈对照汇总
```

每档目录内含 `summary.json`（指标）、`requests.jsonl`（逐请求）、`prompts_meta.json`（token 校准与前缀模式）。

### 10.2 部署目录

| 栈 | 目录 | 容器 | 镜像 |
|---|---|---|---|
| DCP2 | `/mnt/sfs_turbo/yy/dcp2-issue2/` | `dcp2-issue` | `quay.io/ascend/vllm-ascend:v0.30.0rc1-a3` |
| layerwise | `/mnt/sfs_turbo/yy/lw-issue/` | `lw-issue` | `swr.../vllm-ascend-kvoffloading:v1119` |

### 10.3 客户端

`/mnt/sfs_turbo/yy/issue-bench/issue_bench.py`，关键参数：

```bash
# DCP2
python3 issue_bench.py --tag 0hit-c32 --hit 0 --concurrency 32 --salt 0 \
  --proxy http://172.168.4.237:8498 \
  --p-engines http://172.168.4.237:9081,http://172.168.4.31:9081 \
  --d-engines <16 个 9900-9907> --max-tokens 700

# layerwise（端口换成 8496 / 8410 / 8420-8427）
python3 issue_bench.py --tag lw-90hit-c32 --hit 90 --concurrency 32 --salt 2 \
  --proxy http://172.168.4.237:8496 \
  --p-engines http://172.168.4.237:8410,http://172.168.4.31:8410 \
  --d-engines <8 个 8420-8427> --max-tokens 700
```

## 11 待办

- [ ] 每档再跑 2 次（新 salt）补置信区间
- [ ] 补 issue 的原始配对：DCP 侧改 TP4×DP8 + DCP4、P maxlen 131072，专门回答"issue 的 DCP 基线 vs 特性"
- [ ] 复测 issue 报告里的 P queue / prefill 分解（`num_requests_waiting_by_reason{reason="capacity"}` 已在采集范围内）
- [ ] 若需对外结论，补 thinking 关闭 + 单变量化（同一镜像跑两套参数）

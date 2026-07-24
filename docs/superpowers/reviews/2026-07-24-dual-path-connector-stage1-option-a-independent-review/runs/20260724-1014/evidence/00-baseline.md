# 00 — 评审基线（Baseline）

记录时间：2026-07-24 10:14（本地）。本文件记录本次 run 的全部输入基线。

## 1. Snapshot 完整性

| 文件 | 期望 SHA-256（manifest / GOAL-PROMPT §2.3） | 实际 SHA-256（评审开始时 `shasum -a 256`） | 一致 |
|---|---|---|---|
| `inputs/option-a-detailed-design.snapshot.md` | `9a7be6369d468bd845cb2cf181dd0ac0d3d794eaa7a3fa58ca7be7aa5f852808` | `9a7be6369d468bd845cb2cf181dd0ac0d3d794eaa7a3fa58ca7be7aa5f852808` | 是 |
| `inputs/option-a-existing-review.snapshot.md` | `82980ecdb1d0ab3c976c3ba1c720705065a54539e920cf1b95f548bfb998d3d5` | `82980ecdb1d0ab3c976c3ba1c720705065a54539e920cf1b95f548bfb998d3d5` | 是 |

- 行数：design snapshot 2752 行；existing review snapshot 177 行；manifest 43 行（`wc -l`，与 manifest 记录一致）。
- 评审结束时会重新校验一次 hash，结果记入 `05-coverage-and-self-check.md`。

## 2. Snapshot 结构（H2，共 26 个编号章节）

1. 文档目标 (L10)
2. 目标与非目标 (L26)
3. 已确认的硬约束 (L62)
4. 现有组件逻辑与设计影响 (L158)
5. 统一术语和 token accounting (L247)
6. 请求 ID 与传输身份 (L339)
7. 路径模型 (L391)
8. 路径决策协议 (L476)
9. 公共类设计 (L818)
10. 传输计划和事件接口 (L1362)
11. TransferFence 与逐层依赖 (L1575)
12. BlockOwnershipLedger (L1670)
13. 完成语义 (L1745)
14. 三条统一数据通路 (L1935)
15. 方案 A″：外层 first-winner 编排 (L2039)
16. 配置设计 (L2201)
17. Stage 1 拓扑限制 (L2336)
18. 初始化与握手 (L2366)
19. 失败处理 (L2411)
20. 并发与幂等 (L2461)
21. 代码布局 (L2476)
22. 方案 A″ 实施门槛 (L2515)
23. 测试设计 (L2529)
24. 可观测性 (L2655)
25. 验收标准 (L2693)
26. 开发顺序 (L2732)

（H3 结构已用 `grep '^#{1,3} '` 全量提取并用于分章阅读；设计 snapshot 已被评审者完整读取：L1–1000、L1001–2000、L2001–2752 三段连续读取，无跳过。）

## 3. 输入隔离声明

- 本次评审**从未读取**以下两个持续变化的原文件，且全程不得读取：
  - `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md`
  - `docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-review.md`
- existing review snapshot 在独立 Finding Ledger 冻结前**未读取**（冻结点见 `03-independent-findings.md` 末尾）。
- 所有设计章节、行号引用均以 `inputs/option-a-detailed-design.snapshot.md` 为准。

## 4. vllm-ascend checkout

- 绝对路径：`/Users/leqi/Documents/Code/vllm-ascend`
- branch：`dev/dualpath`
- HEAD：`0ec11a4703b987d3103f8859f184c492f22bde88`
- 版本：`v0.19.1rc1-991-g0ec11a470`（`git describe`）
- `.codegraph/` 存在（`codegraph.db` + WAL；daemon 日志/pid/socket 文件在，索引可用性以实际调用为准）。
- `git status --short`（评审开始时，用户未提交修改，全程保留、不还原）：

```text
 D docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-a-b-design.md
 M docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
 M docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-review.md
?? .kimi-code/
?? docs/superpowers/reviews/
?? docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-detailed-design.md
?? docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-b-detailed-design.md
```

注意：工作区中被 `M` 的两个文件正是被冻结 snapshot 覆盖的原文件；评审一律以 snapshot 为基线，不读取这些工作区版本。

## 5. Upstream 只读参考

- 搜索根：`/Users/leqi/Documents/Code`；候选仓库存在。
- 实际采用：`/Users/leqi/Documents/Code/vllm`
- branch：`main`
- HEAD：`8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`
- 版本：`v0.23.1rc0-1050-g8df14cfc8`（`git describe`）
- `.codegraph/` 存在。
- 版本匹配判断：vllm-ascend `docs/source/conf.py:77,85` 记录配套 upstream 为 **v0.23.0**（`vllm_version`/`pip_vllm_version`）。本地 checkout 为 v0.23.1rc0 之后 1050 个提交，**不是精确的 v0.23.0 tag，但同属 0.23 minor 系列**。结论标记为「参考版本存在小幅偏差」：upstream 接口结论适用于 0.23 系列契约，但不保证与 vllm-ascend CI 实际使用的精确 commit 完全一致；涉及 upstream 行为的关键判断均在 evidence 中标注该偏差。
- upstream 仓库全程只读：不修改、不切分支、不安装依赖、不提交。

## 6. 源码读取范围（随 Phase 推进更新）

初始声明，最终状态见 `05-coverage-and-self-check.md`：

- 计划读取（vllm-ascend 当前 checkout）：
  - `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/`（connector.py、config.py 等 foundation）
  - `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`
  - `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
  - `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`（ascend_store_connector.py、pool_scheduler.py、pool_worker.py、lookup key server）
  - `tests/ut/distributed/kv_transfer/dual_path/`
  - worker/metadata 聚合链路相关文件（model_runner_v1.py 等）
- 计划读取（upstream，只读契约）：`KVConnectorBase_V1`、`MultiConnector`、Scheduler async KV load/失败策略、`KVOutputAggregator`、`KVConnectorWorkerMetadata`。
- 不读取：两个禁止的原 spec 文件；existing review snapshot（冻结前）；网络版本代码。
- 无法验证：本机为 macOS 开发环境，无 NPU、未安装 vllm/vllm-ascend 运行时，**不做任何运行验证**；全部结论为静态源码证据，按 GOAL-PROMPT §4.7 标签分级。

## 7. 环境与方法约束

- 运行环境：macOS，`bash`；无 NPU 硬件；`pip show vllm` 无结果（未安装）。
- 所有评审产物只写入 `runs/20260724-1014/`；scratch 证据位于 `evidence/phase1-scratch/`、`evidence/phase2-scratch/`、`evidence/phase3-scratch/`（均属本 run 内）。
- 不实现修复、不修改设计、不运行 E2E、不安装依赖、不提交 Git。

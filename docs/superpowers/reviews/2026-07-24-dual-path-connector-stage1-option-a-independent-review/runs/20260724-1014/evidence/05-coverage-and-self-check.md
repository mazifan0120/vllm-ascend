# 05 — 覆盖度与自检（Phase 5）

执行时间：评审收尾。全部检查命令的真实输出已核对，结论如下。

## 1. 章节覆盖（设计 snapshot 26 个编号章节）

判定依据：`independent-review.md` §17 覆盖表（与本表一致）。设计 snapshot 已完整读取（L1–1000、L1001–2000、L2001–2752 三段连续读取 + 全量 H2/H3 结构提取，见 `00-baseline.md` §2）。

| § | 章节 | 已完整读取 | 已抽样验证 | 已形成 Finding | 无问题及依据 / 无法验证及原因 |
|---|---|---|---|---|---|
| 1 | 文档目标 | 是 | 元信息 | — | 无问题（元信息） |
| 2 | 目标与非目标 | 是 | 与 §3/§22 交叉核对 | IR-001、IR-020 | 失败语义目标 vs 不修改边界的张力已定位 |
| 3 | 已确认硬约束 | 是 | §3.1/§3.2/§3.3 与源码核对 | IR-020 | §3.4/§3.5 无问题（与 POS-4/6 一致） |
| 4 | 现有组件逻辑 | 是 | §4.1–§4.5 全部源码核对 | IR-007（§4.4） | §4.1/§4.2/§4.3 属实（F1/F8/F10/POS-12） |
| 5 | 术语与 token accounting | 是 | 专项 9.1 逐变量核对 | IR-007、IR-018 | 其余变量闭合（report §8.1） |
| 6 | 请求 ID 与传输身份 | 是 | 与 MLC wire 机制核对 | IR-027、IR-036 | R-005 一致（正确 YAGNI 回避） |
| 7 | 路径模型 | 是 | 谓词与框架门禁核对 | IR-028 | POS-5 |
| 8 | 路径决策协议 | 是 | 与 U:scheduler 调用点核对 | IR-003/004/006/008/009/023/024/035 | — |
| 9 | 公共类设计 | 是 | 构造/注册/父类接口核对 | IR-021/029/039/040 | POS-12 |
| 10 | 传输计划和事件接口 | 是 | 与 MLC ReqMeta 字段核对 | IR-003/034/036、RF-001 | — |
| 11 | TransferFence | 是 | 屏障机制核对 | RF-001 | 抽象判定：必须保留 |
| 12 | BlockOwnershipLedger | 是 | 与框架 invalid 语义核对 | IR-013、RF-001 | 判定：保留但简化 |
| 13 | 完成语义 | 是 | 与 U:scheduler.py:1579-1586/2517-2586 核对 | IR-005、IR-038 | POS-8/9/11 |
| 14 | 三条数据通路 | 是 | 三路径 T0–T17 时序重建 | IR-001/002/007、RF-001 | happy path 闭合 |
| 15 | A″ 编排 | 是 | 与 AMC:32-41/87-102 核对 | IR-011/012/032 | POS-1/2 |
| 16 | 配置设计 | 是 | 与 config.py/pool_scheduler 核对 | IR-012/018/030/031/033/040 | — |
| 17 | 拓扑限制 | 是 | 与 MLC 端口推导核对 | — | 无问题（合理 fail-fast） |
| 18 | 初始化与握手 | 是 | 与 MLC:1869-1918 核对 | IR-019 | — |
| 19 | 失败处理 | 是 | 与 U:scheduler 失败链核对 | IR-002/004/010/013/016/017 | — |
| 20 | 并发与幂等 | 是 | 与框架调用线程模型核对 | IR-006、IR-035 | POS-10 |
| 21 | 代码布局 | 是 | 与现有 dual_path/ 目录核对 | — | 无问题 |
| 22 | 实施门槛 | 是 | 逐条与 finding 对照 | IR-012 | 门槛本身表述无问题，但缺 sibling 配置断言 |
| 23 | 测试设计 | 是 | 与 finding 测试缺口对照 | IR-022、RF-001 | — |
| 24 | 可观测性 | 是 | 字段完备性检查 | — | 无实质问题（shadow_* 隔离正确）；建议补 2 项指标（非阻塞） |
| 25 | 验收标准 | 是 | 可执行性检查 | IR-041、RF-001 | — |
| 26 | 开发顺序 | 是 | 依赖顺序检查 | — | 无问题 |

结论：26/26 章节均有覆盖记录，无仅凭标题判定的章节。

## 2. Evidence coverage

| 检查项 | 结果 |
|---|---|
| 每个 P0/P1 是否有具体触发时序和源码证据 | 是：IR-001～IR-007 全部含 T0…Tn 时序 + file:line（Ledger 对应条目） |
| 每个源码引用是否对应本次记录的 checkout | 是：vllm-ascend 引用均相对 `dev/dualpath @ 0ec11a47` 工作树（该目录源码部分未含用户未提交修改，`git status` 仅 docs 变更）；upstream 引用相对 `8df14cfc` 并已标注版本偏差 |
| 是否把设计描述误当成当前代码事实 | 否：全 Ledger 按五档证据标签区分；「设计推导」与「当前源码已确认」严格分离 |
| 是否把版本不匹配的 upstream 结论说成已确认 | 否：统一标注「参考版本存在小幅偏差」，并识别 F18 三个具体漂移提交 |
| 是否区分源码可达风险和运行时已验证问题 | 是：本机无任何运行时验证，相关条目均标 `[源码可达，尚未运行验证]` |
| token accounting / block ownership / completion provenance 专项 | 完成：report §8（8.1 变量表 + 8.2 range ownership 矩阵）；provenance 见 phase3-scratch/03 维度 9（九问核对）与 IR-036 |
| 三路径与异常场景 | 完成：`02-path-timelines.md` + 3 份 phase2-scratch（43 个场景段） |
| R-001～R-005 独立复核 | 完成：`04-existing-review-cross-check.md`（冻结后执行） |

## 3. Prompt isolation

| 检查项 | 命令/方法 | 结果 |
|---|---|---|
| 两份 snapshot SHA-256 未改变 | 评审结束 `shasum -a 256 inputs/*.md` | 均与 manifest 一致（design `9a7be636…`、review `82980ecd…`）；manifest 自身 `85a6bd3a…` |
| 未读取持续变化的原设计/原 review | 全程纪律（所有子代理 prompt 均含禁读条款） | 确认未读取 |
| existing review snapshot 只在冻结点后读取 | 冻结指纹 `492a5a25…365cf1` 于 Phase 4 前计算；读取发生在冻结后 | 确认 |
| `inputs/`、`README.md`、`GOAL-PROMPT.md` 未被修改 | git status 与 baseline 逐行一致 | 确认（无新增 M/D） |
| vllm-ascend 源码和测试未被修改 | `git status --short` 与 `00-baseline.md` §4 快照逐行一致（仅评审前已存在的 docs 变更 + untracked docs/reviews） | 确认 |
| upstream 仓库未被修改 | `git status --short`（vllm）仅评审前已存在的 3 个 untracked 项；HEAD 仍 `8df14cfc` | 确认 |
| 其他已有 run 未被修改 | 本 run（20260724-1014）为首个 run，无其他 run | 不适用/确认 |
| 全部持久化输出位于本 run | `find runs/20260724-1014`：7 个规定文件中的 6 个 + 本文件 + `independent-review.md` + 13 个 scratch（phase1/2/3-scratch 均属本 run 内） | 确认；无 run 外输出 |
| 冻结 Ledger 未被回改 | 重新计算冻结点前内容 SHA-256 = `492a5a25bb2a4217e30eabe1f0a2544daa69f434af5fb0b2980ca3c240365cf1`，与冻结指纹一致 | 确认 |

## 4. 文档质量

| 检查项 | 结果 |
|---|---|
| `TBD`/`TODO`/空占位 | `grep -rn "TBD\|TODO\|FIXME\|占位"` 命中 4 处，全部为**对被评审代码占位状态的描述**（如「配置占位」「纯占位」指 dual_path 现状），非评审产物本身的未完成标记；无空占位章节 |
| 重复或冲突的 Finding | 合并规则已在 Ledger 声明（C-PR-1+AB-3、C-PE-5+C-PR-4、C-FH-8+C-PR-3、C-FH-4+C-PR-5、C-PE-2+C-PR-2、C-PE-4+C-PR-9、CP-1+AB-10）；Phase 4 复核未发现需推翻的 IR 事实判断 |
| 没有解释「为什么」的结论 | 每条 IR/RF 含「为什么不正确」字段；每条 POS 含成立前提与源码契约 |
| 没有最小修正方案的阻塞项 | IR-001～IR-007 均含最小修正建议与验证方法 |
| 行号/符号引用准确性 | 关键锚点经主评审者原文复核（AMC:32-41、MLC:1000-1087/1395-1438/1603-1702、design snapshot 全文）；子代理报告的关键行号在 Phase 3 经第二代理复核（见各 phase3-scratch 复核节） |
| Markdown fence/表格/链接 | fence 平衡检查：全部 19 个 md 文件通过（无奇数 fence）；表格为 GFM 管道语法 |
| `git diff --check` | `git diff --check -- docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review` exit 0（未执行任何 `git add`/commit） |
| 行尾空白 | `grep -rn " $"` 无命中 |
| markdownlint | 本机未安装 markdownlint；以上述 git diff --check + fence/空白扫描作为适当静态检查替代 |
| 不必要的大段源码复制 | 无（均以 file:line 引用） |

## 5. 未验证边界（沿用最终报告 §18）

无运行验证（无 NPU）；upstream v0.23.1rc0-1050 vs 配套 v0.23.0 小幅偏差；`m_store.get` 超时语义、Store DMA→P2P 设备级可见性、Mooncake TE session 失效检测为外部接口待确认；IR-003 置信度中高，其余 P0/P1 高置信度。

# DualPathConnector Stage 1 Option A″ 独立评审 Goal Prompt

请使用 Goal 机制持续完成本次评审，不要只在当前回合输出概览。

## 1. 创建 Goal

先创建以下 Goal，不要设置 `token_budget`：

> 完成一份针对 DualPathConnector Stage 1 方案 A″ 的独立、源码可追溯、
> 覆盖完整时序与兼容性的架构评审报告，判断该方案是否具备进入实现阶段
> 的条件，并给出可执行的最小设计修正意见。

不要因为单轮结束、上下文压缩、已经发现若干问题或某个外部依赖暂时
不可用，就提前将 Goal 标记为 complete。

## 2. 输入与只读边界

### 2.1 主要设计

```text
docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review/inputs/option-a-detailed-design.snapshot.md
```

### 2.2 已有评审

```text
docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review/inputs/option-a-existing-review.snapshot.md
```

已有评审只能在独立 findings 冻结后读取，不是第一阶段输入，也不能作为
源码证据。

### 2.3 文档快照完整性

这两份 snapshot 是本次评审唯一允许读取的设计与已有 review 输入。

评审开始时读取：

```text
docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review/inputs/SNAPSHOT-MANIFEST.md
```

校验两份 snapshot 的 SHA-256 必须分别为：

```text
9a7be6369d468bd845cb2cf181dd0ac0d3d794eaa7a3fa58ca7be7aa5f852808  option-a-detailed-design.snapshot.md
82980ecdb1d0ab3c976c3ba1c720705065a54539e920cf1b95f548bfb998d3d5  option-a-existing-review.snapshot.md
```

评审期间禁止读取、比较或引用以下持续变化的原文件：

```text
docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-review.md
```

所有设计章节、行号和文档事实都必须来自 snapshot。即使原文件后来发生
修改，也不得用原文件刷新、纠正或补充本次评审基线。

如果任一 hash 不匹配：

- 不得回读原文件；
- 不得自行重新复制快照；
- 在本次 run 的 baseline 中记录输入完整性错误；
- 停止内容评审并向用户报告需要重新冻结输入。

### 2.4 当前 vllm-ascend

以 Agent 启动时 `/Users/leqi/Documents/Code/vllm-ascend` 的实际 checkout
为准。必须记录 branch、HEAD、版本信息和工作区状态。

当前工作区可能包含用户尚未提交的修改。保留这些修改，不得还原、覆盖、
格式化或移动用户文件。

### 2.5 Upstream 只读参考

本地 upstream 代码的搜索根目录是：

```text
/Users/leqi/Documents/Code
```

需要验证 upstream vLLM 接口时：

1. 先在该目录下定位候选仓库，通常为
   `/Users/leqi/Documents/Code/vllm`；
2. 记录实际采用的仓库绝对路径、branch、HEAD 和版本；
3. 根据当前 vllm-ascend 的依赖、版本文件或关联提交，判断 upstream
   checkout 是否匹配；
4. 如果不匹配，明确标记为“参考版本存在偏差”，不得把结论描述为当前
   组合已经确认兼容；
5. upstream 仓库只能读取，不得修改文件、切换分支、安装依赖或提交代码；
6. 不得用网络上的最新版代码替代本地实际 checkout；
7. 如果没有可匹配的 upstream，记录验证边界并继续完成其余评审，不得
   凭记忆补全接口行为。

## 3. 输出隔离

本次评审的固定根目录是：

```text
docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review
```

开始评审时，在该目录下创建一个新的、唯一的：

```text
runs/<YYYYMMDD-HHMM>/
```

如果相同时间戳目录已经存在，添加递增后缀，例如 `-02`。不得覆盖、清空
或复用已有 run。

全部持久化输出只能写入本次 run：

```text
runs/<YYYYMMDD-HHMM>/
├── independent-review.md
└── evidence/
    ├── 00-baseline.md
    ├── 01-current-control-flow.md
    ├── 02-path-timelines.md
    ├── 03-independent-findings.md
    ├── 04-existing-review-cross-check.md
    └── 05-coverage-and-self-check.md
```

其中：

- `independent-review.md`：完整、可独立阅读的最终评审报告；
- `00-baseline.md`：仓库、版本、工作区和证据范围；
- `01-current-control-flow.md`：当前框架的真实调用链；
- `02-path-timelines.md`：三条路径及异常场景的时序；
- `03-independent-findings.md`：冻结后的独立 Finding Ledger；
- `04-existing-review-cross-check.md`：R-001～R-005 对照；
- `05-coverage-and-self-check.md`：章节覆盖、证据缺口和最终自检。

禁止把本次评审的草稿、报告、证据、临时 Markdown 或修改建议写到该 run
以外的任何位置。系统临时目录中的非持久化命令输出不算评审产物，但不得
把它当作最终证据文件。

禁止修改：

- `inputs/` 下的 snapshots 和 manifest；
- 本目录的 `README.md` 和 `GOAL-PROMPT.md`；
- `docs/superpowers/specs/` 下的原设计和原 review；
- Option B 文档；
- 双方案索引；
- vllm-ascend 源码、配置和测试；
- `/Users/leqi/Documents/Code` 下的 upstream 仓库；
- 本目录下已有的其他 run。

本任务只做评审：不实现修复、不修改设计、不运行 NPU E2E、不安装依赖、
不提交 Git。

## 4. 基本评审原则

1. 详细设计是待评审方案，不是当前实现事实。
2. existing review snapshot 中 R-001～R-005 是冻结后待独立复核的输入，
   不是权威结论。
3. 必须先追踪当前 checkout 的真实代码，再判断设计是否成立。
4. 对代码优先使用 CodeGraph，从具体符号开始追踪；文档可以直接读取。
5. 如果一个仓库没有 `.codegraph/`，停止对该仓库调用 CodeGraph，改用
   `rg` 和精确文件读取。
6. 如果 CodeGraph 提示文件尚未重新索引，只对提示中的具体文件进行原始
   文件复核，不要重复 grep 已确认的其他结果。
7. 任何重要判断必须标记为以下一种：
   - `[当前源码已确认]`
   - `[源码可达，尚未运行验证]`
   - `[设计推导]`
   - `[外部接口待确认]`
   - `[事实冲突]`
8. 不得把“尚未实现”本身当作设计缺陷。
9. 不得因为未来可能需要某能力，就默认 Stage 1 必须加入该能力。
10. 不得只给“建议考虑”“可能有风险”这类不可执行意见。
11. 不得用设计文档中的描述反向证明设计本身正确。
12. 发现问题时必须说明为什么不正确、在什么前提下触发、具体时序、
    影响范围，以及最小修正方式。
13. 如果设计是正确的，也要说明为什么正确以及依赖哪些前提。
14. 严格区分：
    - 当前源码行为；
    - 源码可达但未经运行验证的风险；
    - 方案 A″ 的设计承诺；
    - Stage 2 或未来能力；
    - 尚未回答的设计问题。

## 5. 执行计划

创建 Goal 后，建立以下计划，并保持最多一个步骤为 in progress：

1. 建立代码、文档和版本基线；
2. 重建当前框架真实控制流；
3. 审查三条路径的完整时序；
4. 审查架构、兼容性、过度设计和失败语义；
5. 冻结独立 Finding Ledger；
6. 冻结后独立复核 existing review snapshot 中的 R-001～R-005，仅做
   遗漏检查和结论对照，不得改写已冻结的 IR Findings；
7. 生成最终评审报告；
8. 完成覆盖度和 Markdown 自检。

每完成一个阶段，更新对应 evidence 文件和计划状态。不要等到最后才记录
证据，也不要把中间推理散落到工作区其他位置。

## 6. Phase 0：建立基线

在 `evidence/00-baseline.md` 中记录：

- snapshot manifest 和两份 snapshot 的实际 SHA-256；
- hash 是否与第 2.3 节完全一致；
- snapshot 的行数和 H2/H3 结构；
- 明确声明没有读取持续变化的原设计和原 review；
- vllm-ascend 的绝对路径、branch、完整 HEAD；
- `git status --short`；
- 实际采用的 upstream 仓库、branch、HEAD 和版本；
- vllm-ascend 与 upstream 的版本匹配判断；
- 本次实际读取、未读取和无法验证的源码范围；
- 是否存在 `.codegraph/` 以及索引状态。

所有源码引用都以本次记录的 checkout 为准。历史文档、记忆、网络版本和
其他分支只能作为线索，不能替代当前代码证据。

## 7. Phase 1：重建当前框架控制流

从 Decode 收到带远程 Prefill 语义的请求开始，追踪到请求成功、失败或
取消后的资源释放。

至少覆盖：

1. Scheduler 查询缓存和 Connector；
2. matched-token accounting；
3. first-positive winner 选择；
4. KV block 分配；
5. `update_state_after_alloc`；
6. Scheduler metadata 到 Worker metadata；
7. MultiConnector/AscendMultiConnector 的 Worker hook 广播；
8. Store lookup 和 bulk load；
9. Reverse Layerwise；
10. PE model compute；
11. Forward Layerwise；
12. `get_finished()`、invalid blocks 和多 rank 聚合；
13. `WAITING_FOR_REMOTE_KVS` 等状态转换；
14. 请求成功、失败、取消和资源释放。

重点验证这些组件，不得只引用设计中的源码锚点：

- DualPathConnector foundation、config 和现有 UT；
- AscendMultiConnector；
- MooncakeLayerwiseConnector Scheduler/Worker；
- AscendStoreConnector；
- KVPoolScheduler、KVPoolWorker、LookupKeyServer；
- upstream MultiConnector、KVConnectorBase_V1；
- upstream Scheduler 的 async KV load、完成处理和失败策略；
- KVOutputAggregator 或当前版本中的等价多 rank 聚合逻辑。

对每个关键接口记录：

- 谁调用它；
- 输入和返回值；
- 是否有副作用；
- 是否在 Scheduler、Worker、后台线程或 transport callback 中运行；
- 调用顺序是框架保证、当前实现偶然顺序，还是设计假设；
- 对请求状态、block、metadata 和完成集合的影响。

结果写入 `evidence/01-current-control-flow.md`。

## 8. Phase 2：三条路径独立时序审查

分别为以下路径写出 T0、T1、T2……的完整时序：

- `PE_READ`
- `DE_FULL_HIT`
- `DE_PARTIAL_READ`

每条路径都必须回答：

- 谁做 probe？
- 谁决定路径？
- 哪一刻算 committed？
- 每个 Engine 向 Scheduler 声明多少 external tokens？
- 谁分配正式 blocks？
- 每一段 token range 由谁写、谁读？
- 哪些操作可能并发？
- 哪个事件使请求离开等待状态？
- 谁发布 `finished_recving` / `finished_sending`？
- 多 rank 如何汇聚成功和失败？
- 取消、超时和部分失败后谁负责 drain、quarantine 和资源释放？

至少覆盖以下场景：

- Store miss、partial hit、full hit；
- `L_DE=0` 和 `L_DE>0`；
- `L_PE<K_DE`、`L_PE=K_DE`、`L_PE>K_DE`；
- winner 与非 winner Layerwise child 同时拿到真实 blocks；
- terminal event 早于 request-ID mapping；
- control message 重试、重复、乱序、迟到；
- 一个 TP rank 成功、另一个 rank 失败或超时；
- Sender 成功但 Receiver 失败；
- 在 probe、allocation、commit、Store load、Reverse、Forward 各阶段取消；
- Engine restart 后 local request ID 复用；
- shutdown/drain 时仍有 in-flight 请求。

不要把“可能并发写”描述为已经确认的物理同时写，除非当前代码能证明。
精确区分：无协调的 ownership/order 风险、可达竞态和已验证冲突。

结果写入 `evidence/02-path-timelines.md`。

## 9. Phase 3：强制评审维度

逐项审查，不得只挑最显眼的问题：

1. Stage 1 目标、非目标和硬边界是否互相一致；
2. PE/DE Connector 配置是否能真实表达设计拓扑；
3. first-winner 是否只承担 accounting，还是被错误地当成数据面编排器；
4. Scheduler、Connector、Worker、Store、P2P 的职责边界；
5. token accounting 是否闭合；
6. token range 是否无遗漏、无重叠、无多 writer；
7. 正式 KV blocks 的分配、有效性与 ownership 生命周期；
8. decision、commit、data-start 之间是否存在竞态；
9. finished event 是否有明确的 source、direction、rank 和 request
   provenance；
10. raw terminal retention、wire ID、tombstone 是否足以处理迟到事件；
11. retry、duplicate、cancel、timeout 是否幂等；
12. 多 rank 聚合是否可能提前成功或永久等待；
13. 失败后 invalid blocks、Scheduler 状态和物理传输清理是否一致；
14. 与现有 AscendMultiConnector 广播语义是否兼容；
15. 与 MooncakeLayerwise 父类初始化、线程、metadata、完成接口是否兼容；
16. 与 AscendStore lookup/load/failure 语义是否兼容；
17. 是否违反“不修改既有 Connector 和 upstream vLLM”的硬边界；
18. 配置、初始化、握手、shutdown 和版本兼容性；
19. NPU 性能风险：同步、CPU-NPU 搬运、线程、额外拷贝和控制面阻塞；
20. 测试、可观测性和验收标准是否足以证明正确性。

### 9.1 Token accounting 专项

对设计中的所有 token 变量建立定义、单位、开闭区间和 owner 表。

至少验证：

- prompt tokens 与 decode-ready tokens 的最后一个 token 语义；
- `L_DE`、`L_PE`、`K_DE`、`K_PE`、`R`；
- 每个 Engine 的 external token 增量；
- Store 对齐或丢弃 partial chunks 后的 coverage；
- Scheduler accounting 与实际写入区间是否一致；
- partial hit 是否被错误表达成 sibling Connector 自动组合。

### 9.2 Block ownership 专项

对每条路径建立表格：

```text
token range | physical blocks | allocator | writer | reader |
validity transition | release condition
```

验证：

- 是否存在无人填充区间；
- 是否存在多个未协调 writer；
- Store 和 Forward 是否写入相同正式 blocks 的不同范围；
- Reverse 读取前 Store DMA 是否已完成；
- 请求失败后 blocks 是否可能过早复用；
- invalid blocks 是否在请求成功事件之后才被发现。

### 9.3 Completion provenance 专项

验证所有完成或失败事件能否唯一回答：

- 哪个请求；
- 哪个 Engine；
- Forward 还是 Reverse；
- 哪个 TP rank；
- Store、P2P 还是 control transport；
- 哪个已经提交的 plan/decision；
- 是否允许重复；
- 何时可以转成公开 `finished_*`；
- 何时可以释放 blocks。

## 10. 过度设计与复用检查

逐一检查：

- PathDecisionCoordinator；
- DualPath control transport；
- SharedMooncakeTransferRuntime；
- TransferPlan / Command / Event；
- TransferFence；
- BlockOwnershipLedger；
- Store adapter；
- raw terminal retention；
- wire identity 和 tombstone。

每个抽象必须回答：

1. 它保护了哪个 Stage 1 correctness invariant？
2. 当前已有模块能否直接提供该能力？
3. 如果删除它，哪个具体时序会出错？
4. 它是 Stage 1 必需，还是仅为 Stage 2、可观测性或代码整洁服务？
5. 是否引入新的状态源、生命周期 owner 或双重记账？
6. 是否存在更小且兼容现有接口的替代方案？

分类为：

- 必须保留；
- 必须保留但应简化；
- 可以复用已有模块替代；
- Stage 1 YAGNI；
- 当前证据不足。

不要因为类或状态数量多就直接判定过度设计。只有当抽象不保护 Stage 1
invariant、重复已有 owner，或引入的协调成本大于其解决的问题时，才能
形成过度设计 finding。

## 11. 独立 Finding Ledger

已有 review 在本阶段仍然禁止读取。

每条独立意见使用唯一编号：

```text
IR-001、IR-002、IR-003……
```

固定格式：

```markdown
### IR-001 标题

- 严重级别：P0 / P1 / P2 / P3
- 类型：正确性 / 时序 / 生命周期 / 兼容性 / 过度设计 /
  性能 / 可测试性 / 文档一致性
- 证据等级：
- 设计位置：精确到章节和当前行号
- 源码位置：文件、符号和当前行号
- 触发前提：
- 具体时序：T0、T1、T2……
- 为什么不正确：
- 可能影响：
- 与已有模块的兼容性：
- 是否属于过度设计：
- 最小修正建议：
- 修正后需要增加的测试：
- 置信度：高 / 中 / 低
```

严重级别：

- P0：方案在现有架构下无法安全实现，或会造成确定性错误/数据损坏；
- P1：进入实现前必须修复的正确性、竞态、生命周期或接口问题；
- P2：应在设计冻结前处理的兼容性、维护性、性能或测试缺口；
- P3：表述、命名、局部简化等非阻塞问题。

P0/P1 必须具备具体时序和当前源码证据；否则只能标记为待确认风险或降低
严重级别。

除问题外，也记录重要的正向结论。正向结论必须说明成立原因、前提和对应
源码契约，不能只写“通过”。

初版完成后，将完整 ledger 写入：

```text
evidence/03-independent-findings.md
```

在文件末尾增加“独立评审冻结点”，列出冻结时存在的全部 IR ID、摘要、
严重级别和文件哈希或等价的内容指纹。

冻结前不得读取已有 review。

## 12. Phase 4：已有 review 对照

独立 Finding Ledger 冻结后，才读取：

```text
docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review/inputs/option-a-existing-review.snapshot.md
```

对 R-001～R-005 分别判断：

- 与独立发现一致；
- 补充了独立评审遗漏；
- 部分正确；
- 证据不足；
- 与当前源码冲突；
- 已被当前设计修改取代。

每项必须提供：

- 当前设计位置；
- 当前源码证据；
- 成立所需前提；
- 未覆盖的反例或时序；
- 是否还需要修改设计；
- 对应的 IR Finding，若不存在则明确写“独立阶段未发现”。

已有 review 只能用于：

- 检查独立评审是否遗漏；
- 验证或反驳已有结论；
- 形成 R-001～R-005 对照矩阵。

不得：

- 将已有 review 原文复制为独立 Finding；
- 用已有 review 作为源码证据；
- 因已有 review 标记“通过”而降低验证强度；
- 修改已有 review；
- 为了保持一致而删除独立发现。

对照已有 review 后才发现的问题，使用：

```text
RF-001、RF-002、RF-003……
```

并标记为“由已有 review 触发的二次验证发现”。RF Finding 仍须满足和 IR
Finding 相同的证据、时序和修正要求。

如果对照后发现 IR Finding 有事实错误：

- 不删除原 Finding；
- 记录原判断；
- 提供推翻它的新证据；
- 记录修正原因和修正后结论；
- 在最终报告中使用修正后的结论。

结果写入 `evidence/04-existing-review-cross-check.md`。

## 13. 最终评审报告

`independent-review.md` 必须可脱离 evidence 文件独立阅读，至少包含：

1. Executive Verdict：通过 / 有条件通过 / 不通过；
2. 评审基线和实际采用的 checkout；
3. 当前源码真实行为摘要；
4. 三条路径的端到端时序；
5. P0/P1 阻塞问题；
6. P2/P3 非阻塞问题；
7. 重要正向结论及成立前提；
8. token accounting 与 range ownership 矩阵；
9. 组件兼容性矩阵；
10. 过度设计与已有模块复用矩阵；
11. 失败、取消、重试和多 rank 场景矩阵；
12. R-001～R-005 复核矩阵；
13. review-triggered RF Findings；
14. 测试与可观测性缺口；
15. 按优先级排列的最小设计修改清单；
16. 仍需设计者回答的问题；
17. 设计文档 26 个编号章节的评审覆盖表；
18. 本次未验证事项与证据边界。

每条最终 Finding 必须保留：

- IR/RF ID；
- 严重级别；
- 设计位置；
- 源码证据；
- 触发时序；
- 为什么不正确；
- 影响；
- 最小修正；
- 验证方法。

最终结论以独立 IR Findings 为主。已有 review 对照只能作为验证层，不能
改变报告的独立评审属性。

## 14. 覆盖度和自检

在 `evidence/05-coverage-and-self-check.md` 中完成：

### 14.1 章节覆盖

对设计 snapshot 的 26 个编号章节逐项记录：

- 已完整读取；
- 已抽样验证；
- 已形成 Finding；
- 无问题及其依据；
- 无法验证及原因。

不能仅凭标题判断章节内容。

### 14.2 Evidence coverage

检查：

- 每个 P0/P1 是否有具体触发时序和源码证据；
- 每个源码引用是否对应本次记录的 checkout；
- 是否把设计描述误当成当前代码事实；
- 是否把版本不匹配的 upstream 结论说成已确认；
- 是否区分源码可达风险和运行时已验证问题；
- 是否覆盖 token accounting、block ownership、completion provenance；
- 是否覆盖三条路径和异常场景；
- 是否完成 R-001～R-005 的独立复核。

### 14.3 Prompt isolation

确认：

- 两份 snapshot 和 manifest 的 SHA-256 未改变；
- 没有读取持续变化的原设计和原 review；
- `inputs/`、`README.md` 和 `GOAL-PROMPT.md` 未被修改；
- vllm-ascend 源码和测试未被修改；
- upstream 仓库未被修改；
- 其他已有 run 未被修改；
- 本次全部持久化输出都位于当前 run。

### 14.4 文档质量

扫描并处理：

- `TBD`、`TODO` 和空占位；
- 重复或互相冲突的 Finding；
- 没有解释“为什么”的结论；
- 没有最小修正方案的阻塞项；
- 行号或符号引用不准确；
- Markdown fence、表格和链接错误；
- 不必要的大段源码复制。

对当前 run 下所有 Markdown 执行适当的静态检查，并至少执行：

```bash
git diff --check -- <本次 run 的路径>
```

该命令只用于检查，不得 `git add` 或 commit。

## 15. Goal 完成判据

只有同时满足以下条件，才能将 Goal 标记为 complete：

- 两份 snapshot 的 hash 在评审开始和结束时均与 manifest 一致；
- 设计 snapshot 已完整读取；
- existing review snapshot 只在冻结点后读取；
- 持续变化的原设计和原 review 从未被读取；
- 设计的 26 个编号章节均有覆盖记录；
- 当前框架控制流已经与源码核对；
- `PE_READ`、`DE_FULL_HIT`、`DE_PARTIAL_READ` 均完成端到端时序；
- R-001～R-005 均已独立复核；
- 所有 P0/P1 均有触发时序、影响和源码证据；
- token accounting、block ownership、completion provenance 均有专项结论；
- 已完成兼容性、过度设计、失败和多 rank 检查；
- 当前 run 下的七个规定文件全部存在且内容完整；
- 没有在当前 run 外产生或修改评审输出；
- Markdown 和 `git diff --check` 自检通过；
- 最终回复给出总体结论、P0/P1 摘要和当前 run 的绝对文件链接。

如果证据不足，记录为未验证边界并继续完成其他部分；不要凭推测下结论，
也不要因为单个外部依赖不可用就提前结束整个 Goal。

最终回复必须明确说明：

- Goal 是否完成；
- 实际 run 路径；
- 总体 verdict；
- P0/P1 数量和标题；
- 未验证边界；
- 未读取原设计和原 review；
- 未修改 snapshots、原文件、源码和其他 run。

# 03 — 独立 Finding Ledger（独立评审，冻结稿）

**独立性声明**：本 Ledger 全部内容在读取 existing review snapshot **之前**形成并冻结。证据来源：设计 snapshot（行号以其为准）、Phase 1 当前源码证据（`01-current-control-flow.md` + `phase1-scratch/`，关键锚点经主评审者原文复核）、Phase 2 三路径时序（`02-path-timelines.md` + `phase2-scratch/`）、Phase 3 维度评审（`phase3-scratch/`）。Phase 2 候选编号（C-PE-*/C-FH-*/C-PR-*）与 Phase 3 本地编号（AB-*/TB-*/CF-*/CP-*）在各条目中标注溯源。

**合并规则**：同根候选合并为一条 IR 并保留全部溯源编号；Phase 3 对候选的定级调整（升级/降级/修正）以 Phase 3 复核结论为准并注明。

统计：**P0 × 1，P1 × 6，P2 × 15，P3 × 19**，另有正向结论 12 条、过度设计判定 11 项。

---

## P0 — 阻塞级

### IR-001 PE_READ 的 PE Store 失败在当前代码下必然静默数据损坏，且 A″ 在 §3.1 硬边界内无修复通路

- 严重级别：P0
- 类型：正确性
- 证据等级：`[当前源码已确认]`（机制）；`[源码可达，尚未运行验证]`（端到端后果）
- 设计位置：§14.1 L1969「Store load 失败、PE 计算失败或 Forward 失败都使请求失败」；§15.4 L2131-2141「依赖现有 Scheduler failure gate 保证 Store 失败不会继续模型计算」；§3.1 L73-80（不修改 AscendStore 既有组件）
- 源码位置：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:457-465`（创建 `KVCacheStoreRecvingThread` 未注入 invalid_block_ids）、`pool_worker.py:1333-1337`（只 drain worker 私有集合）、`kv_transfer.py:908-910`（失败块进线程私有集合）、`kv_transfer.py:942`（失败也无条件 `set_finished_request`）；对照契约 `U:v1/base.py:385-389`
- 溯源：C-PE-1（Phase 3 agent-12 确认 P0 并扩展）
- 触发前提：A″ PE 拓扑中 PE Store load 由外层既有 `AscendStoreConnector` sibling 承担且 `load_async=True`（§9.7 L1334 派生 config 强制；sibling 自身配置亦然）；任一 Store DMA 失败。
- 具体时序：T0 PE sibling `get_num_new_matched_tokens` 命中 → T1 Scheduler 分配 blocks → T2 `update_state_after_alloc` 入 `loading_req_ids` → T3 Worker 提交 bulk `m_store.get` → T4 部分 DMA 失败 → T5 失败块进线程私有集合（无人读取）→ T6 请求仍被 `set_finished_request` 标为完成 → T7 sibling 发布 PE-local `finished_recving` → T8 PE Scheduler 解除 WAITING 并 `cache_blocks` → T9 PE 在**含失败块的脏 KV** 上 compute 并按层 Forward `[L_DE,R)` 给 DE → T10 DE `FORWARD_DONE` → T11 双 Engine「成功」，脏 KV 已进入两侧 prefix cache 污染后续请求。
- 为什么不正确：upstream 契约要求失败 blocks 不晚于 finished 上报（`U:v1/base.py:385-389`），当前 KVPoolWorker 的 load_async 失败上报链路断裂（F11）；设计承诺的失败语义因此不成立，且后果是静默脏数据而非可检测错误。
- 可能影响：PE_READ 全路径数据正确性；prefix cache 跨请求污染；违反 §25 验收「输出 token 正确性」。
- 与已有模块的兼容性：DE 侧内部 adapter 可通过控制 `register_kv_caches` 时机向 recv 线程注入失效集合（见 IR-002）；但 PE sibling 是**既有组件**，§3.1 禁止修改，A″ 自身代码（Multi 子类之外）无注入点。
- 是否属于过度设计：否。
- 最小修正建议：二选一并写入设计——(a) 将 §3.1 边界显式放宽一条：允许对 `KVCacheStoreRecvingThread` 做一行构造参数级修复（注入 invalid_block_ids），这是最小且语义正确的修法；(b) A″ 放弃 sibling 承担 PE Store，改由 DualPath 内部 Store adapter 统一承担 PE/DE Store（即把 PE 也纳入 adapter 路线），代价是放弃 §15.7「最大程度复用 sibling」卖点。不接受「不处理」。
- 修正后需要增加的测试：PE_READ Store 注入失败 UT + E2E（断言 invalid blocks 上报、请求 `FINISHED_ERROR`、无脏 KV 进 prefix cache）；锚定 `load_async=True` 配置。
- 置信度：高（机制证据完整；端到端触发未运行验证）。

---

## P1 — 进入实现前必须修复

### IR-002 DE Store adapter 强制 `load_async=True` 正踩 F11 失败上报断裂，且修复所须的注入窗口与归因规则未写入设计

- 严重级别：P1
- 类型：正确性 / 失败语义
- 证据等级：`[当前源码已确认]`（断裂机制）；`[设计推导]`（修复路线）
- 设计位置：§9.7 L1334「派生 Store config 强制 load_async=True」；§9.7 L1347「Store failed blocks 统一进入 DualPath 的 request-level invalidation」；§19.2 L2440-2459
- 源码位置：同 IR-001（pool_worker.py:457-465/1333-1337；kv_transfer.py:908-910,942）；修复窗口：recv 线程构造器原生支持注入（pool_worker.py:457-465 调用点）
- 溯源：C-FH-4 + C-PR-5 合并（Phase 3 双方确认 P1）
- 触发前提：DE_FULL_HIT / DE_PARTIAL_READ；DE 内部 Store adapter 按设计强制 `load_async=True`；Store load 部分失败。
- 具体时序：T0 commit 后 `commit_after_alloc` → T1 请求入 `loading_req_ids` → T2 bulk load 部分 DMA 失败 → T3 失败块进线程私有集合（`get_block_ids_with_load_errors` 读不到）→ T4 请求仍被标完成 → T5 DualPath 消费为内部 `STORE_DONE` → T6（FH）发布 DE `finished_recving`，脏 KV 进 prefix cache；（PR）继续 Reverse 把脏前缀发给 PE。
- 为什么不正确：与 IR-001 同根，但作用在 DualPath 自己的 adapter 路径上；§19 的请求级失效语义在 Store 失败时根本不会触发。
- 可能影响：DE 两条路径静默数据损坏。
- 与已有模块的兼容性：修复可达——adapter 在 `register_kv_caches` 后、首请求前向 recv 线程注入失效集合（无竞态窗口）；但须触 KVPool 私有成员，设计未声明其合法性（见 IR-020）。
- 是否属于过度设计：否。
- 最小修正建议：在 §9.7 写明 (1) adapter 注入失效块集合的确切时点与私有成员清单；(2) block→request 失败归因规则（按 `loading_req_ids` 与 LoadSpec block 区间归属）；(3) 注入后失败请求必须走 §13.2 的 invalid+延迟终态流程。
- 修正后需要增加的测试：DE_FULL_HIT/DE_PARTIAL_READ Store 部分 DMA 失败注入 UT（断言 invalid 上报且早于/不晚于终态、`FINISHED_ERROR`、无 `STORE_DONE` 误发布）。
- 置信度：高。

### IR-003 跨 Engine block manifest 双向传输通道缺失：Reverse 目标（PE）blocks 与 Forward 源（DE）blocks 的 remote mapping 均无生产者

- 严重级别：P1
- 类型：正确性 / 协议缺口
- 证据等级：`[设计推导]`（基于设计消息集与框架分配时序的推演）
- 设计位置：§8.1 L489-497（PathDecision 无 block manifest 字段）；§8.2 协议 11 步（L553-572）；§8.7 L790-795（envelope 仅 4 类消息）；§10.1 L1391-1412（`reverse_rank_mappings`/`forward_rank_mappings` 需 remote_block_ids）；§10.1 L1428-1431（LayerwiseEndpoint 须足以生成 `remote_block_ids`）
- 源码位置：`U:scheduler.py:774-779`（connector 查询发生在分配前）、`U:multi_connector.py:381-400`；MLC:1441+（`_get_kv_split_metadata` 消费 remote_block_ids）
- 溯源：C-PR-1 + AB-3 合并（Phase 3 扩展为双向缺口）
- 触发前提：DE_PARTIAL_READ（Reverse 方向）与 PE_READ/DE_PARTIAL_READ（Forward 方向）均需要。
- 具体时序（Reverse 向）：T0 DE 派发 proposal（只带 `de_block_manifest_digest`）→ T1 PE 在**自己的 blocks 分配之前**做出并发出 commit → T2 PE Scheduler 分配 PE 正式 blocks → T3 DE 需要按 `reverse_rank_mappings.remote_block_ids`=PE blocks 构造 SendTask → **该信息的任何消息都没有承载者**（envelope 只有 PROPOSAL/COMMIT/ACK/ERROR 四类；DECISION_ACK 只有 request_key+version）。Forward 向同理：PE 构造 Forward SendTask 需要 DE 完整 block manifest，proposal 只有 digest。
- 为什么不正确：设计规定「ReqMeta/SendTask builder 只能消费已冻结的 RankBlockMapping，不得在 Worker 侧重新推导」（§10.2 L1511-1512），但冻结 mapping 的原料没有传输通道；digest 只能校验不能还原内容。
- 可能影响：DE_PARTIAL_READ 不可实现（P1 缺口）；PE_READ 的 Forward mapping 同样无协议内来源。
- 与已有模块的兼容性：现有 MLC 靠 `kv_transfer_params` 携带 remote block 信息（do_remote_prefill/decode 链路），设计可复用该 envelope 扩展。
- 是否属于过度设计：否（是缺失而非多余）。
- 最小修正建议：在协议中增加一次「commit 后、data-start 前」的 block manifest 交换：DE→PE 的 frozen DE manifest 与 PE→DE 的 frozen PE manifest（可并入 `PathDecisionCommit`/`DecisionAck` 载荷或新增一类 envelope），并把它加入 data-start 前置条件（§8.2 步骤 11）。
- 修正后需要增加的测试：三条路径的 manifest 交换 UT（含 digest 不匹配拒绝）；DE_PARTIAL_READ E2E 断言 Reverse 目标块与 PE 实际分配一致。
- 置信度：中高（逻辑推演严密，但最终依赖实现时消息格式，存在设计者可澄清的空间）。

### IR-004 跨 Engine 失败/取消传播缺失：`CONTROL_ERROR` 有类型无流程，post-commit 单侧失败使对侧永久挂起

- 严重级别：P1
- 类型：时序 / 失败语义
- 证据等级：`[设计推导]`（基于消息集与状态机）；`[当前源码已确认]`（框架侧无兜底超时）
- 设计位置：§8.7 L790-795（message_type 枚举含 CONTROL_ERROR 但全文仅此一处，无任何处理流程）；§13（完成语义全部单 Engine 闭环）；§19 L2411-2459（失败处理全部单 Engine）；§15.4 L2136-2138「Forward 一旦已经启动，其成败由 DualPath 自己管理」
- 源码位置：`U:scheduler.py:2517-2534`（离开 WAITING 唯一条件是 finished_recving）；MLC:1400-1428（无超时机制）
- 溯源：C-PE-2 + C-PR-2 合并（Phase 3 双方确认 P1；CONTROL_ERROR 全文 grep 仅定义处命中）
- 触发前提：commit 之后任一 Engine 侧失败：PE Store 失败、PE compute 失败、Forward 失败、Reverse 失败、Receiver 失败而 Sender 成功、单侧 Engine 宕机。
- 具体时序（PE_READ 例）：T0 commit(PE_READ) → T1 PE 侧 Store 失败（即使 IR-001 修复后能检测）→ T2 PE 请求 `FINISHED_ERROR` → T3 **无任何消息通知 DE** → T4 DE 等 `FORWARD_DONE`，永不到达 → T5 DE 请求永久滞留 `WAITING_FOR_REMOTE_KVS`，正式 blocks 永不释放。DE_PARTIAL_READ 反向对称：Receiver(PE) 失败而 Sender(DE Reverse) 成功时 DE 永 hang。
- 为什么不正确：请求级失败语义（§2.1 L40-41）要求「已提交路径的任一必要传输失败，最终使请求进入 FINISHED_ERROR」，但「最终」在跨 Engine 场景没有机制保证；框架本身没有 connector 级超时兜底。
- 可能影响：blocks 泄漏、Engine 容量耗尽；双 Engine 状态不一致。
- 与已有模块的兼容性：control transport 已存在，补流程不需要新通道。
- 是否属于过度设计：否。
- 最小修正建议：为 `CONTROL_ERROR`（或新增 `PATH_ABORT`）定义完整流程：失败方 control 线程发送 → 对侧 `PathDecisionCoordinator.fail(request_key)` → 对侧走本地 §13.2 invalid+终态流程；同时为数据面等待增加请求级超时（transfer timeout → 本地失败，不依赖对侧消息）。
- 修正后需要增加的测试：post-commit 每个失败注入点（PE Store/Reverse/Forward/单侧宕机）× 三路径的 UT+E2E，断言双 Engine 均收敛到 `FINISHED_ERROR` 且 blocks 释放。
- 置信度：高。

### IR-005 pre-commit / 决策窗口内 abort 无终态发布者，且 §9.5 与 §13.4 的 finished_sending 前置条件互相矛盾

- 严重级别：P1
- 类型：生命周期 / 资源泄漏
- 证据等级：`[当前源码已确认]`（框架 delay_free 契约）；`[设计推导]`（设计缺口）
- 设计位置：§9.5 L1108-1111「对其中的 delayed-free ID，仅在本地 ownership=0 且无 unknown in-flight 后返回 finished_sending」；§13.4 L1829-1841；§8.2（窗口内无 abort 处理）
- 源码位置：`U:scheduler.py:2144-2147`（WAITING 请求默认 delay_free）、`U:scheduler.py:2162`（abort 也触发 request_finished 钩子）、`U:scheduler.py:2580-2586`（延迟释放依赖 connector 之后上报）；`pool_scheduler.py:896-901`（build_connector_meta 的 finished 清理点会抹掉迟到 commit 的 load 状态）
- 溯源：C-FH-1（Phase 3 确认 P1，矛盾条款经原文复核）
- 触发前提：DE 声明 `E_DE` 并入 WAITING → blocks 分配（delay_free 成立）→ decision commit 到达之前请求被 abort/取消。
- 具体时序：T0 abort → T1 框架调 `request_finished_all_groups` → T2 按 §13.4 保守返回 `delay_free=True` → T3 无 plan、无 Worker 侧状态 → 无人发布 `finished_sending`/`finished_recving` → T4 blocks 永久延迟释放；若迟到 commit 此时到达，load 状态又被 `build_connector_meta` 的 finished 清理抹掉，`self.requests`/probe handle 残留。
- 为什么不正确：框架契约是「delay_free 的 ID 必须由 connector 之后恰一次上报 finished 才释放」；设计两个条款互相矛盾：§9.5 要求「ownership=0 且无 in-flight 才返回 finished_sending」，§13.4 的保守 delay_free 却可能对「从未有过任何 owner 的请求」注册 delay_free，使 §9.5 的前置条件永真/永假不可判定。
- 可能影响：取消场景 blocks 泄漏；迟到 commit 复活已清理状态。
- 与已有模块的兼容性：修正只需设计条款，不触碰既有组件。
- 是否属于过度设计：否。
- 最小修正建议：(1) 统一规则：任何曾使框架 delay_free 成立的 request ID，DualPath 必须保证恰一次发布终态（成功或失败）后再清理状态——包括「从未启动 plan」的情形（直接发布，不等 ownership）；(2) 修正 §13.4 保守条件，delay_free 注册时同步登记「终态义务」；(3) 与 IR-027（决策面 tombstone）联动处理迟到 commit。
- 修正后需要增加的测试：决策窗口内 abort × 三路径 UT（断言 blocks 释放、状态无残留、迟到 commit 被拒/幂等）。
- 置信度：高。

### IR-006 `commit_after_alloc`/`abort_probe`/`get_committed` 的应用线程未指定：control 线程直写无锁 KVPoolScheduler 将与 scheduler 循环竞态

- 严重级别：P1
- 类型：时序 / 并发
- 证据等级：`[当前源码已确认]`（KVPoolScheduler 无锁、scheduler 循环内被调）；`[设计推导]`（竞态后果）
- 设计位置：§8.7 L746-784（control transport 独立 receive/poll 线程；未说明 handler 在哪消费）；§9.7 L1250-1257（`commit_after_alloc` 签名无线程约定）；§20 L2461-2475（只要求请求级串行化，未指定执行者）
- 源码位置：`pool_scheduler.py:478-565`（`get_num_new_matched_tokens` 在 scheduler 循环内读写 `load_specs`/`_loading_req_ids` 等无锁字典）、`pool_scheduler.py:617-618/896-915`
- 溯源：C-FH-2（Phase 3 确认 P1 并扩展到 abort_probe/get_committed）
- 触发前提：control transport receive 线程直接调用 adapter 的 `commit_after_alloc`/`abort_probe`（设计目前唯一合理的接线方式）。
- 具体时序：T0 scheduler 循环正在 `build_connector_meta` 迭代 `_loading_req_ids.copy()`/遍历 `load_specs` → T1 control 线程同时写入 `commit_after_alloc`（新增 load spec / 改 loading 集合）→ T2 字典迭代期写 → RuntimeError/EngineCore 崩溃；或 T2' `loading_req_ids` 快照错位 → Worker 永远收不到该请求的 metadata → `STORE_DONE` 永久丢失 → 请求滞留。
- 为什么不正确：KVPoolScheduler 的全部状态都假设单线程（scheduler 进程主循环）访问；设计引入了第二个写者但没有同步纪律。
- 可能影响：EngineCore 崩溃或请求级挂起。
- 与已有模块的兼容性：框架本身提供了解法——所有 scheduler 侧状态变更都应在 scheduler 循环内应用（如 `update_connector_output` 的模型）。
- 是否属于过度设计：否。
- 最小修正建议：在 §8.7/§20 写明线程模型：control 线程只做接收/校验/入队；`commit_after_alloc`/`abort_probe`/`fail` 等状态变更由 DualPathConnectorScheduler 在 scheduler 循环内（如下一次 `get_num_new_matched_tokens`/`build_connector_meta`/`update_connector_output` 边界）统一应用；`wait_for_commit` 删除或钉死调用者与阻塞上限。
- 修正后需要增加的测试：commit 与 build_connector_meta 并发压力 UT（注入交错，断言无字典竞态、无 metadata 丢失）。
- 置信度：高（竞态机制确定，触发概率依赖时序）。

### IR-007 `S_DE_raw`/`S_DE` 语义与 Store probe 实际返回值冲突：full hit 系统性退化为 partial，或照字面接线触发 assert 崩溃

- 严重级别：P1
- 类型：正确性 / token accounting
- 证据等级：`[当前源码已确认]`（probe 侧语义）；`[事实冲突]`（设计两处表述不一致）
- 设计位置：§5.2 L271-274（`S_DE_raw`「原始绝对前缀长度」、`S_DE=min(floor(S_DE_raw/G)*G, R)`）；§4.4 L214-228 与 §14.2 L1980-1981（「coverage >= prompt_len - 1」判定）
- 源码位置：`pool_scheduler.py:502-503`（lookup 侧已按 G floor）、`pool_scheduler.py:604`（assert）、`pool_scheduler.py:849-852`（full-hit 时 +1 slot 处理）
- 溯源：C-FH-3（Phase 3 确认 P1 并补充分情形数学）
- 触发前提：DE Store probe 返回值为「已 floor + full-hit 减 1」语义（当前实现），设计再对 `S_DE_raw` 二次 floor。
- 具体时序：设 `G=256`。T0 probe 返回 `S_DE_raw`（已 floor）；T1 设计再 floor 一次 → 情形 A：`P%G==0`，full hit 被二次 floor 削掉一个 chunk → `K_DE<R` → 系统性退化为 DE_PARTIAL_READ；情形 B：`P%G==1` 恰好正确；情形 C：其余余数下 full hit 结构性不可达；情形 D：若实现者改为不 floor 直接对照 probe 语义接线，则触发 `pool_scheduler.py:604` assert → EngineCore 崩溃。
- 为什么不正确：同一个 `S_DE_raw` 在设计与实现之间有两种单位语义；§4.4/§14.2 的判定表述与 §5.2 的数学不一致。
- 可能影响：DE_FULL_HIT 不可达或崩溃；路径选择系统性错误。
- 与已有模块的兼容性：修正为文档/语义对齐，不需改既有组件。
- 是否属于过度设计：否。
- 最小修正建议：钉死 `S_DE_raw` 的定义为「KVPoolScheduler lookup 的原始返回语义」（含已 floor 与 full-hit -1 约定），`S_DE` 只做 `min(..., R)` clamp 不再二次 floor；同步修正 §4.4/§14.2 判定表述；补充分情形真值表（`P%G` 各余数）。
- 修正后需要增加的测试：`P%G==0/1/其他` 三余数 × full/partial/miss 的 accounting UT。
- 置信度：高。

---

## P2 — 设计冻结前应处理

### IR-008 「只在 num_external_tokens>0 时冻结 block plan」与两段式分配冲突，PE 侧 Forward 源 mapping 无协议内来源

- 严重级别：P2
- 类型：生命周期 / 协议
- 证据等级：`[当前源码已确认]`（框架两段式调用）
- 设计位置：§8.5 L681-684
- 源码位置：`U:scheduler.py:822-827,942-954,969-974`（异步请求两次调用，第二次 =0；chunked prefill 下 blocks 逐 chunk 到达）
- 溯源：C-PE-3（Phase 3 确认 P2 并扩展到 DE_PARTIAL_READ 的 PE Forward 源 `[K_DE,R)`）
- 触发前提：PE_READ 的 PE 侧（Store miss 时 `H_PE=0`，第一次调用即为 0，永不冻结）与 chunked prefill（blocks 逐 chunk 到达）。
- 具体时序：T0 PE 第一次 `update_state_after_alloc(blocks, H_PE)` 只覆盖 Store 前缀 blocks → T1 tail `[K_PE,R)` blocks 在第二次调用（=0）才出现 → T2 按字面规则不冻结 → T3 Forward 源 mapping 缺 tail blocks。
- 为什么不正确：冻结规则的意图是「I/O 前必须有完整 manifest」，但字面条件使其在最需要 Forward 的场景失效。
- 可能影响：Forward mapping 不完整或依赖隐式渠道。
- 最小修正建议：改为「每次调用都合并进冻结中的 manifest，I/O 启动门槛 = manifest 覆盖已提交 plan 的全部 region」；明确第二次 =0 调用的语义是「补全 tail」。
- 修正后需要增加的测试：Store miss（H_PE=0）与 chunked prefill 场景的 manifest 完整性 UT。
- 置信度：高。

### IR-009 `decide_on_pe` 触发点未钉死：早于拿到真实 `L_PE` 则 DE partial 准入失效

- 严重级别：P2
- 类型：时序 / 正确性
- 证据等级：`[当前源码已确认]`（`L_PE` 来源）；`[设计推导]`
- 设计位置：§8.6 L697-703（`decide_on_pe(proposal, pe_local_tokens, ...)` 未说明何时被谁调用）；§7.3 L452-475
- 源码位置：`U:scheduler.py:774-779`（`num_computed_tokens` 作为第二参数传入 connector 查询）
- 溯源：C-PR-11（Phase 3 确认 P2；补充：DE_FULL_HIT control-only 不需要 `L_PE`，约束仅限经 PE 模型请求的决策）
- 触发前提：实现把 decide 挂在 proposal 到达时刻（control 线程）而非 PE 首次 scheduler pass。
- 具体时序：T0 proposal 到达（control 线程）→ T1 立即 decide（`L_PE` 未知/按 0）→ T2 `L_PE<K_DE` 准入误判 → T3 commit(DE_PARTIAL_READ) → T4 PE scheduler pass 得到真实 `L_PE>=K_DE` → T5 winner 旁落/PE 读未反转的脏前缀。
- 最小修正建议：写明 decide 只能在 PE Scheduler 首次以真实 `num_computed_tokens` 调用 connector 时执行；之前 proposal 只登记（register_candidate）。
- 修正后需要增加的测试：`L_PE` 各取值（</=/>`K_DE`）下的准入 UT。
- 置信度：高。

### IR-010 DECISION_TIMEOUT / 决策阶段失败无 Scheduler→Worker 传播路径，设计自声明的失败路径不可兑现

- 严重级别：P2
- 类型：失败语义
- 证据等级：`[设计推导]` + `[当前源码已确认]`（F13 delay_free 契约）
- 设计位置：§8.7 L810-811（重试耗尽进入 DECISION_TIMEOUT）；§19.1 L2416-2438
- 源码位置：`U:scheduler.py:2144-2147,2580-2586`；`pool_scheduler.py:896-901`
- 溯源：C-FH-8 + C-PR-3 合并（Phase 3 确认并上调 P2）
- 触发前提：重试耗尽、dispatch 失败等发生在 Scheduler 进程的决策失败；此时请求无 plan、无 Worker 状态。
- 具体时序：T0 决策失败 → T1 Scheduler 侧标记失败 → T2 无 Worker 终态发布者 → T3 delay_free blocks 无 finished 上报 → T4 泄漏。
- 最小修正建议：决策失败记录由 `build_connector_meta` drain 到 Worker，Worker 按冻结 manifest 合成 invalid+terminal；或 Scheduler 侧直接合成一条 metadata-only plan 驱动同一终态机。
- 修正后需要增加的测试：DECISION_TIMEOUT UT（断言 blocks 释放、请求 FINISHED_ERROR）。
- 置信度：高。

### IR-011 取消未启动 plan / DECIDING 阶段取消：终态发布义务人与 CANCEL 通道未规定

- 严重级别：P2
- 类型：生命周期
- 证据等级：`[当前源码已确认]`（F13）；`[设计推导]`
- 设计位置：§15.4 L2136-2138（finished ID「用于取消未启动 plan、清理 fence」）；§8.2（无 CANCEL 消息）
- 源码位置：`U:scheduler.py:2144-2147,2162,2580-2586`
- 溯源：C-PE-4 + C-PR-9 合并（Phase 3 确认；泄漏窗口精确界定为 dispatch 成功之前/之后两种形态）
- 具体时序：T0 DE 请求在 DECIDING/alloc 阶段被 abort → T1 已派发的 PE 模型请求无 CANCEL 通知继续存在 → T2 §15.4 要求「取消未启动 plan」但未要求仍发布一次 receive terminal → T3 delay_free blocks 泄漏。
- 最小修正建议：与 IR-005 统一框架：任何 delay_free 可能成立的 ID 必须恰一次发布终态；为已派发 PE 请求定义 CANCEL（可复用 IR-004 的 PATH_ABORT）。
- 修正后需要增加的测试：DECIDING/alloc/dispatch 后三时点取消 UT。
- 置信度：高。

### IR-012 A″ 未强制 PE sibling `use_layerwise=false`：误配时非 winner Store child 仍可发起 Store GET，与 Forward/Reverse 写竞争

- 严重级别：P2
- 类型：兼容性 / 配置
- 证据等级：`[当前源码已确认]`
- 设计位置：§16.2（PE 示例未约束 sibling 配置）；§22 L2521（门槛只说「不触发 Store GET」但未把配置固化为边界）
- 源码位置：`pool_scheduler.py:589-593`（layerwise 模式且曾命中时非 winner 也置 `can_load=True`）、`pool_scheduler.py:565/617`（use_layerwise 强制 load_async 关闭）
- 溯源：C-PE-5 + C-PR-4 合并（Phase 3 确认 P2 并扩展第二条静默损坏路径）
- 具体时序：T0 sibling 误配 `use_layerwise=true` → T1 DualPath 成 winner，sibling 收 `(..., 0)` → T2 sibling 仍置 can_load → T3 Worker Store GET 写 PE blocks `[L_PE,...)`，与 Reverse/Forward 写并发 → 数据竞争；且 layerwise 下 load_async 关闭引入另一条语义漂移。
- 最小修正建议：A″ config validator 强制 sibling `use_layerwise=false`（fail fast）；§22 门槛补上该配置断言。
- 修正后需要增加的测试：误配 validator UT + 竞争场景 UT。
- 置信度：高。

### IR-013 请求级失效未排除共享 prefix blocks：invalid 误伤无关请求并误逐出 prefix cache

- 严重级别：P2
- 类型：正确性 / 生命周期
- 证据等级：`[当前源码已确认]`（框架失效语义）
- 设计位置：§19.2 L2445-2448「收集请求相关所有 KV block IDs」；§12 L1726-1732
- 源码位置：`U:scheduler.py:2665,2736-2737`（invalid → 截断 + prefix cache 逐出）
- 溯源：C-PR-6（Phase 3 确认 P2 并扩展 PE 侧同构）
- 触发前提：`L_DE>0`（DE 本地已有 prefix 与其他请求共享 blocks）；请求失败。
- 具体时序：T0 失败 → T1 按「所有相关 blocks」上报 invalid，含共享段 `[0,L_DE)` → T2 框架逐出这些 blocks → T3 正在读它们的无关请求数据被拔、prefix cache 误逐出。
- 最小修正建议：invalid 集合按 plan region 过滤为「本请求外部加载/写入的区间」（`[L_DE,R)` 中由 Store/Forward 写入的部分），排除本地共享段；PE 侧 `[0,L_PE)` 同构。
- 修正后需要增加的测试：L_DE>0 失败 UT（断言共享 blocks 不在 invalid 集合）。
- 置信度：高。

### IR-014 Reverse/Forward 继承 MLC 发送线程失败归因 bug（残留循环变量），可能误杀+假阳性 DONE

- 严重级别：P2
- 类型：正确性 / 兼容性
- 证据等级：`[当前源码已确认]`
- 设计位置：§9.2（共享 runtime 复用 send/recv thread）
- 源码位置：MLC:507（`failed_reqs.add(req_id)` 用 :469 循环残留变量）
- 溯源：C-PR-7（Phase 3 确认 P2 并扩展第三形态：真失败未被标记 → 接收方把未写完数据当完成）
- 具体时序：T0 同一 SendTask 批内多请求 → T1 其中一个传输失败 → T2 失败归因到残留变量指向的请求 → T3a 误杀无辜请求；T3b 真失败请求未被标记 → 假阳性 DONE → 对端读未写完数据。
- 最小修正建议：Directional adapter 提交任务时对 batch 内请求做独立归因（不依赖父类批内变量），或在 §3.1 授权下做一行修复；设计需声明继承该 bug 的规避方案。
- 修正后需要增加的测试：多请求同 batch 单失败注入 UT。
- 置信度：中高（bug 存在确定；触发条件依赖批处理聚合）。

### IR-015 Store→Reverse 的数据可见性无 device-level fence

- 严重级别：P2
- 类型：时序 / 正确性
- 证据等级：`[源码可达，尚未运行验证]`；`[外部接口待确认]`（m_store.get 内部语义）
- 设计位置：§11.2 L1608-1616（STORE_DONE→Reverse 依赖）；§14.3 L2031
- 源码位置：`kv_transfer.py:902-942`（m_store.get host 同步返回逐 key 结果）
- 溯源：C-PR-8（Phase 3 确认 P2；host 侧闭合已证实，残留设备级可见性 + Reverse SendTask 的 wait_event 语义未规定）
- 具体时序：T0 Store DMA 写 HBM → T1 host 侧 get 返回 → T2 Reverse SendTask 读同一 HBM → 设备级是否保证 DMA 写对后续 P2P 读可见，设计无 fence 约定。
- 最小修正建议：写明 Reverse 提交前依赖的同步原语（m_store.get 的同步语义或显式 event wait）；在 E2E 注入乱序验证。
- 修正后需要增加的测试：Store→Reverse 紧随时序压力 E2E。
- 置信度：中。

### IR-016 Store probe 的 zmq REQ `recv()` 无超时，挂在 DE Scheduler 循环上

- 严重级别：P2
- 类型：性能 / 可用性
- 证据等级：`[当前源码已确认]`
- 设计位置：§9.7（probe 复用 KVPool lookup）；§19.1（lookup RPC 失败降级不覆盖 hang）
- 源码位置：`pool_scheduler.py:1107`（REQ recv 无超时）
- 溯源：C-FH-5（Phase 3 确认 P2；DE 侧为相对现状新增阻塞，PE 侧为继承）
- 具体时序：T0 probe RPC 发出 → T1 LookupKeyServer 慢/死 → T2 scheduler 循环阻塞 → T3 全 Engine 停摆。
- 最小修正建议：adapter 层为 probe 加超时+重试+降级 PE_READ（§19.1 已允许 lookup 失败转 PE_READ）；或把 probe 移出 scheduler 循环关键路径。
- 修正后需要增加的测试：LKS 无响应注入 UT（断言有界等待+降级）。
- 置信度：高。

### IR-017 `m_store.get` hang 无 watchdog：请求永久 WAITING、容量泄漏

- 严重级别：P2
- 类型：可用性
- 证据等级：`[源码可达，尚未运行验证]`；`[外部接口待确认]`（get 内部超时）
- 设计位置：§19（无 transport hang 分类）
- 源码位置：`pool_worker.py:840`；`kv_transfer.py:902-942`
- 溯源：C-FH-6（Phase 3 确认 P2）
- 最小修正建议：Store load 加请求级超时→按 STORE_LOAD_ERROR 走 §13.2 流程；与 IR-004 的 transfer timeout 合并设计。
- 修正后需要增加的测试：get 不返回注入 UT。
- 置信度：中。

### IR-018 `cache_transfer_granularity` 双源：DualPath 配置值与 KVPool 推导值可能不一致

- 严重级别：P2
- 类型：配置 / 正确性
- 证据等级：`[当前源码已确认]`
- 设计位置：§16.1 L2213（`cache_transfer_granularity: 256`）
- 源码位置：`pool_scheduler.py:119,404-413`（G=lcm(block_size, group family granularity) 推导）
- 溯源：AB-1
- 具体时序：T0 配置 G=256 但 KVPool 推导 G=512 → T1 probe 已按 512 floor → T2 设计按 256 计算 coverage → T3 accounting 与物理加载错位。
- 最小修正建议：单一来源——DualPath 从 KVPoolScheduler 读取推导值，配置项仅作可选覆盖且启动时校验一致。
- 修正后需要增加的测试：不一致配置 fail-fast UT。
- 置信度：高。

### IR-019 「Forward/Reverse handshake 成功」作为提交前准入无现有机制承载

- 严重级别：P2
- 类型：兼容性 / 协议
- 证据等级：`[当前源码已确认]`
- 设计位置：§7.3 L459（准入条件）；§18.2 L2402-2409
- 源码位置：MLC:1869-1918（握手为发送时 lazy 建立，无请求前握手状态查询）
- 溯源：AB-2
- 具体时序：T0 准入检查需要「handshake 均成功」→ T1 当前 MLC 无预先握手接口 → T2 实现只能在首次传输时才发现失败 → T3 准入形同虚设或需新增初始化期握手。
- 最小修正建议：把 handshake 收敛到 §18.2 初始化顺序（connector ready 前完成双向 handshake 并记录 capability），准入只查初始化结果。
- 修正后需要增加的测试：handshake 失败时 DE partial 不提交的 UT。
- 置信度：高。

### IR-020 「不修改既有组件」边界与失败语义目标的冲突需显式授权私有成员绕行

- 严重级别：P2
- 类型：文档一致性 / 架构边界
- 证据等级：`[当前源码已确认]`（绕行点清单）
- 设计位置：§3.1 L73-80；§2.1 L40-41
- 源码位置：pool_worker.py:457-465（注入点私有）、pool_scheduler.py:549-553（LoadSpec 私有）、MLC:507（归因 bug）
- 溯源：AB-8
- 为什么不正确：IR-002 的修复、probe 的 abort、IR-014 的规避都需触私有成员；设计未声明这类「读私有/注入」是否算修改边界，实现者将各行其是。
- 最小修正建议：在 §3.1 增加显式清单：允许的私有成员接触点（成员名、只读/注入、理由）；禁止项保持不变。
- 修正后需要增加的测试：边界清单的静态检查（lint 级）。
- 置信度：高。

### IR-021 PE 侧 Store adapter 与 sibling 双建 LookupKeyServer/backend：同 ipc path REP bind 冲突 + 双 buffer 注册

- 严重级别：P2
- 类型：兼容性 / 生命周期
- 证据等级：`[当前源码已确认]`
- 设计位置：§9.7 L1284-1297（Worker adapter 创建 KVPoolWorker + LookupKeyServer）、L1349「以上 adapter 用于方案 A″ 的 DE Store」
- 源码位置：`ascend_store_connector.py:119-120`（rank0 非 layerwise 必建 LKS）；`pool_scheduler.py:1115-1130`（ipc path 由 port+dp 派生，同源配置 → 同路径）
- 溯源：CP-1（合并 AB-10：build_store_vllm_config 的 PE 视图分支是死代码且是冲突来源）
- 具体时序：T0 PE 侧若同时创建 DualPath adapter（PE 视图）与 sibling AscendStoreConnector → T1 两者都以 rank0+同 port 派生同一 ipc path bind REP → T2 bind 冲突崩溃；或双 backend 双 buffer 注册。
- 最小修正建议：删除 adapter 的 PE 视图分支（Stage 1 YAGNI），设计显式声明「A″ 中 adapter 仅在 DE 创建」（L1349 已有此意但未落到构造约束）。
- 修正后需要增加的测试：PE 拓扑启动 UT（断言单 LKS）。
- 置信度：高。

### IR-022 测试设计四类系统性缺口：决策面失败、跨 Engine 传播、abort-in-window、sibling 配置 validator；F11 回归未锚定 load_async

- 严重级别：P2
- 类型：可测试性
- 证据等级：`[设计推导]`（对照本 Ledger 失败类 finding）
- 设计位置：§23 L2529-2654；§25 L2693-2731
- 源码位置：—（对照 IR-001/002/004/005/010/011/012）
- 溯源：CP-4
- 为什么不正确：§23 的数据面失败注入覆盖认真，但本 Ledger 的 P0/P1 大多落在未被任何测试项覆盖的决策面/跨 Engine/取消窗口；F11 的回归测试若不锚定 `load_async=True` 配置会测不到断裂路径。
- 最小修正建议：§23 增加上述四类测试项；§25 增加对应验收断言（与 IR-001/002/004/005 的「修正后测试」对齐）。
- 置信度：高。

---

## P3 — 非阻塞

### IR-023 Ack 门位置语义含糊（§8.2 步骤 11 与 A″ sibling 持有 PE Store 矛盾）
- 严重级别 P3｜类型：文档一致性｜证据等级：`[设计推导]`｜设计位置：§8.2 L571-572｜源码位置：—｜溯源：C-PE-6
- 触发/时序：PE_READ 中 PE Store 由 sibling 启动，不受 Commit+Ack 门约束。为什么不正确：「Commit+Ack 后才启动 Store load」未限定仅指 DE Store/Reverse/Forward。影响：实现歧义。最小修正：改写为「Ack 门只约束 DE 侧数据操作」。测试：无（文档）。置信度：高。

### IR-024 PE 侧 proposal 等待无超时
- P3｜类型：时序｜`[设计推导]`｜§8.2 L579-581｜源码：—｜溯源：C-PE-7
- 时序：proposal 因 envelope 异常永不到达时请求滞留 waiting（清理便宜，故 P3）。修正：candidate 登记加 TTL，超时按 lookup unknown 转 PE_READ 或清理。测试：滞留注入 UT。置信度：高。

### IR-025 数据面 shutdown 无有界 drain 语义
- P3｜类型：生命周期｜`[设计推导]`｜§8.7 L782-783 仅覆盖 control transport｜源码：MLC 线程 daemon 无 drain｜溯源：C-PE-8
- 修正：§20/§9.3 shutdown 序列写明 send_queue 语义与有界 drain。测试：shutdown drain UT。置信度：中。

### IR-026 control-only 被 PE 降级为 PE_READ 时补派发 PE 模型请求的路径未规定
- P3｜类型：协议｜`[设计推导]`｜§8.7 L813-816｜溯源：C-FH-7（降级条件几乎只剩 Store 运行期失效窄窗口）
- 修正：写明该窗口的升级流程或直接禁止 control-only 降级。测试：降级场景 UT。置信度：中。

### IR-027 决策面无 tombstone：迟到 commit 可复活已终态请求
- P3｜类型：生命周期｜`[设计推导]`｜§6 L378-379（tombstone 仅覆盖 wire ID）｜溯源：C-FH-9（当前被泄漏悖论掩盖，随 IR-005 修复升级）
- 修正：决策状态与 wire ID 对称的 retention；迟到 commit 命中 tombstone 拒绝。测试：迟到 commit UT。置信度：中。

### IR-028 §7.3「超过」措辞未含 `L_PE==K_DE` 等号
- P3｜类型：文档一致性｜`[设计推导]`｜§7.3 L468 vs §5.2 L306｜溯源：C-PR-10（语义已被严格小于三重兜住）
- 修正：措辞改「达到或超过」。置信度：高。

### IR-029 `bind_gpu_block_pool` 被设计写入 Worker 侧 ledger，但框架只在 Scheduler 侧调用
- P3｜类型：兼容性｜`[当前源码已确认]`｜§9.3 L997-998/§9.4 L1062-1063｜源码：`U:scheduler.py:282`｜溯源：AB-4
- 修正：删除 Worker 侧转交或注明仅 Scheduler。置信度：高。

### IR-030 DualPath config 静默吞未知键
- P3｜类型：配置｜`[当前源码已确认]`｜config.py（from_extra_config 未拒绝未知键）｜溯源：AB-5
- 修正：未知键 fail fast（防 §16 拼写错误静默失效）。测试：未知键 UT。置信度：高。

### IR-031 §16.1 公共配置示例非法（缺 role/store、PE 拓扑矛盾）
- P3｜类型：文档一致性｜`[设计推导]`｜§16.1 L2205-2228｜溯源：AB-6
- 修正：标注「片段」或补齐字段。置信度：高。

### IR-032 AMC「只允许一个 kv_transfer_params 产出者」约束未被设计引用
- P3｜类型：兼容性｜`[当前源码已确认]`｜AMC:93-96｜设计 §8.2（proposal 放 kv_transfer_params["dual_path"]）｜溯源：AB-7
- 触发：sibling 也产出 params 时 Multi 抛 RuntimeError。修正：设计声明 DualPath 是唯一 params 产出者并加 validator。置信度：高。

### IR-033 现有 `path_strategy` 默认值与静态-only 未钉死
- P3｜类型：配置｜`[当前源码已确认]`｜config.py（path_strategy.type 允许 static/adaptive 但无实现）｜设计 §7.2｜溯源：AB-9
- 修正：Stage 1 校验只允许 static。置信度：高。

### IR-034 Store 物理 +1 slot 写出格于「不越过 R」的 plan 模型
- P3｜类型：文档一致性｜`[当前源码已确认]`｜§10.1 L1417｜源码：pool_scheduler.py:849-852｜溯源：TB-1
- 修正：plan 模型注明 full-hit 末 token slot 例外并加校验。置信度：高。

### IR-035 幂等 key `request_key+decision_version` 在 pre-commit 首次冻结时 version 未定义
- P3｜类型：协议｜`[设计推导]`｜§8.5 L681-684｜溯源：TB-2
- 修正：规定 pre-commit 保留版本值（如 0/None 语义）。置信度：高。

### IR-036 事件缺 `plan digest`/`decision_version` 字段，与 §10.3 归属校验要求不一致；envelope 的 wire ID 冗余
- P3｜类型：协议｜`[设计推导]`｜§10.3 L1534-1573（校验要求 plan digest）vs L1535-1552（事件字段无 digest）｜溯源：CF-1
- 修正：事件补 `decision_version`/`plan_digest` 或放宽校验表述。置信度：高。

### IR-037 raw terminal retention window 无取值/配置/GC 责任
- P3｜类型：生命周期｜`[设计推导]`｜§10.3 L1564-1565｜溯源：CF-2
- 修正：给出默认值、配置项与 GC 触发点。置信度：中。

### IR-038 SchedulerReleaseLedger 的 `rank_statuses`/`update_from_worker_metadata` 无决策读者，属未用双重记账
- P3｜类型：过度设计｜`[设计推导]`｜§13.4 L1843-1900｜溯源：CF-3
- 为什么不正确：delay_free 决策只用 `plan_has_async_sender_or_reader` 静态位，Worker 状态回流无消费者。最小修正：Stage 1 删除状态副本（或写明诊断用途并标注 shadow）。是否属于过度设计：是（简化级）。置信度：高。

### IR-039 `_is_kv_producer` 未从父类构造复制
- P3｜类型：兼容性｜`[当前源码已确认]`｜DPC connector.py:120-127 vs MLC:701｜溯源：CP-2（当前全仓无人读，无害但属静默偏差）
- 修正：显式赋值或注释说明。置信度：高。

### IR-040 继承 init 从 `kv_port`（默认 14579）派生数据面端口，§16 未定义其语义
- P3｜类型：配置｜`[当前源码已确认]`｜MLC:809-812,1164-1169｜溯源：CP-3
- 修正：§16 写明 kv_port 与 forward/reverse base_port 的关系/覆盖顺序。置信度：高。

### IR-041 §25 验收无性能项
- P3｜类型：可测试性｜`[设计推导]`｜§25 L2693-2731｜溯源：CP-5（与 AGENTS.md 性能回归要求不齐）
- 修正：补决策延迟/传输吞吐验收阈值。置信度：中。

---

## 正向结论（成立原因 + 前提 + 源码契约）

- POS-1 first-winner 只做 accounting，未被当成数据面编排器：§15.2 L2081「winner 只决定 Scheduler accounting，不自动授权 Worker 副作用」+ §15.4 L2125「真实 blocks 被传给 Layerwise 子类不构成路径授权」与 AMC:32-41 的实际语义精确对齐；数据面授权链全部经已提交 PathDecision，不经 winner 状态。
- POS-2 PE_READ 的 decision 协议与 first-winner 查询顺序自洽：proposal 随 `kv_transfer_params` 原子到达、`L_PE` 在首个 schedule pass 作为调用参数可得（`U:scheduler.py:774-779`）、decision 不依赖 PE Store 结果（§7.2），无循环依赖；框架每 pass 重查使 `(None,False)` 重试语义成立。前提：decide 触发点按 IR-009 钉死。
- POS-3 DE_FULL_HIT 的末 token 语义与框架精确咬合：`E_DE=R-L_DE` 声明 + 乐观 `num_computed=R` 使 promote 时 `-1` 分支天然不触发（`U:scheduler.py:2521-2522`），DE 本地重算末 token 零框架修改。
- POS-4 路径无关 blocks 一次分配使 decision 改判零收敛：DE 无论何路径都需 `R-L_DE` external tokens（§7.3 L470-474），窗口内 blocks 处「已预留未发布」纯预留态，改判 PE_READ 时 blocks/accounting 无需回滚。
- POS-5 `REVERSE_DONE` 不进 DE 成功谓词的传递性论证成立：机制是框架 scheduler 门禁（PE 收齐 REVERSE_DONE 才离开 WAITING、才 compute、才 Forward），`FORWARD_DONE` 传递性蕴含 Reverse 完成；不依赖 no-op 的 `wait_for_layer_load`（MLC:1976-1977）。
- POS-6 无已证实物理并发写：Store 写 `[L_DE,K_DE)` 与 Forward 写 `[K_DE,R)` 区间不相交且由 STORE_DONE→Reverse→compute→Forward 串行定序；DE 本地段 `[0,L_DE)` 由 mask 跳过。前提：fence 按 §11/§12 实现。
- POS-7 pending raw terminal inbox 正确修复父类早到事件丢弃缺陷（MLC:1407 过滤 + get_and_clear 清空），且「暂存→mapping 建立后重新校验归属」的方向与框架单步内 start_load_kv 先于 get_finished 的顺序兼容。
- POS-8 §13.2 原子终态规则与框架 assert/释放通道精确兼容：invalid 先于 finished 消费（`U:scheduler.py:1579-1586`）、finished_recving 不变式 assert（:2576-2585）、finished_sending 即释放（:2583-2586）——设计的「invalid 不晚于失败终态」「同一 ID 只发布一次」「drain 后才发布 failed terminal」正是这些不变式的充分条件。
- POS-9 多 rank 聚合利用而非对抗框架：每 rank 恰发布一次 + 框架计数制全 rank barrier（`U:kv_connector/utils.py:78-90`）防止提前成功；不在 Connector 内另造 rank 集合是正确取舍。
- POS-10 control 幂等规则完整：相同 message_id 重试 + payload_hash 校验 + 相同 id 不同 hash 报协议错误（§8.7 L806-811）覆盖了 retry/duplicate 的主场景。
- POS-11 §13.4 的 delayed-free 闭环使用了框架真实存在的通道（`request_finished_all_groups`→delay_free→Worker 逐 rank finished_sending→聚合→释放），不要求 Scheduler 跨进程读 Worker 内存，方向正确（但内部状态冗余见 IR-038）。
- POS-12 kv_both 下一次 `register_kv_caches` 完成全部注册的复用判断属实（MLC:1359-1398 + register_buffer 单次守卫），SharedMooncakeTransferRuntime 的核心前提成立。

## 过度设计判定矩阵（六问结论，详见 phase3-scratch）

| 抽象 | 判定 | 关键理由 |
|---|---|---|
| PathDecisionCoordinator | 必须保留 | 唯一提交方 + CAS 语义保护 decision 不变式，无现有模块承载 |
| DualPath control transport | 必须保留但应简化 | 禁改 MLC 使 control 必须独立；删/钉死 `wait_for_commit`，CONTROL_ERROR 补流程（IR-004） |
| SharedMooncakeTransferRuntime | 必须保留但应简化 | 保护 register_buffer 单次/唯一线程对；可省第二 worker 实例（runtime_worker=self） |
| TransferPlan/Command/Event | 必须保留但应简化 | 逐层事件可降为计数；PATH_COMMITTED 移出传输事件枚举 |
| TransferFence（RankTransferFence） | 必须保留 | 六布尔值极简形态；跨 rank barrier 正确交给框架 |
| BlockOwnershipLedger | 必须保留但应简化 | Stage 1 只需每请求每操作 owner 计数 + plan-region invalid 过滤（IR-013）；逐 layer×block 粒度属 Stage 2 |
| Store adapter | 必须保留；PE 视图分支 Stage 1 YAGNI | probe/commit/abort 隔离 KVPool 非纯 probe 语义；PE 视图删除（IR-021） |
| raw terminal retention | 必须保留但应简化 | 修复 F8 所必需；需量化 retention + GC（IR-037） |
| SchedulerReleaseLedger | 必须保留但应简化 | 框架 delayed-free 闭环的合法载体；删未用状态副本（IR-038） |
| wire identity + tombstone | 必须保留；tombstone 应简化/对称化 | 迟到事件归属必需；决策面 tombstone 缺失（IR-027） |
| TransferFenceRegistry | 当前证据不足（倾向 YAGNI） | 全文仅类图两处出现、无接口定义；建议降级为 Worker 内部 dict |
| 不引入 transfer_epoch | 正确 YAGNI 回避（正向） | Stage 1 无 post-commit 重开语义，§6 L386-389 判断正确 |

## 独立评审冻结点

- 冻结时间：2026-07-24（本 run 内 Phase 3 完成后、读取 existing review snapshot 之前）。
- 冻结时 IR 清单（41 条）：
  - P0：IR-001（PE sibling Store 失败静默损坏）
  - P1：IR-002（DE adapter F11 断裂）、IR-003（block manifest 通道缺失）、IR-004（跨 Engine 失败传播缺失）、IR-005（pre-commit abort 无终态发布者）、IR-006（commit 应用线程竞态）、IR-007（S_DE_raw 语义冲突）
  - P2：IR-008～IR-022（冻结规则/决策触发点/决策失败传播/取消义务/sibling layerwise/共享块误伤/F15 继承/可见性 fence/probe 无超时/get 无 watchdog/G 双源/handshake 准入/私有成员授权/双建 LKS/测试缺口）
  - P3：IR-023～IR-041（Ack 门/proposal 超时/drain/control-only 降级/决策 tombstone/措辞/bind_gpu_block_pool/未知键/示例非法/params 唯一产出者/path_strategy/+1 slot/version 未定义/事件 digest/retention 取值/ReleaseLedger 冗余/_is_kv_producer/kv_port/性能验收）
- 内容指纹：见本文件末尾「指纹」行（对本冻结点之前全部内容的 SHA-256）。
- 冻结后纪律：本文件不再修改；若 Phase 4 对照发现事实错误，只在 `04-existing-review-cross-check.md` 与最终报告中记录修正，不回改本文件。

指纹（冻结点前全部内容的 SHA-256）：`492a5a25bb2a4217e30eabe1f0a2544daa69f434af5fb0b2980ca3c240365cf1`

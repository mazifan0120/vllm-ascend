# 02 — 三条路径独立时序审查（Phase 2 汇总）

基线同 `00-baseline.md` / `01-current-control-flow.md`。完整细节见 `evidence/phase2-scratch/01-pe-read.md`（T0–T16 + 14 场景 + 8 缺陷候选）、`02-de-full-hit.md`（T0–T17 + 12 场景 + 9 缺陷候选）、`03-de-partial-read.md`（T0–T17 + 17 场景 + 11 缺陷候选）。本文件是三条路径的主链时序与结论汇总；缺陷候选在 Phase 3 复核后统一进入 IR Ledger（编号以 `03-independent-findings.md` 为准）。

证据等级标签：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[设计推导]` / `[外部接口待确认]` / `[事实冲突]`。

---

## 1. PE_READ（主链 T0–T16，详见 scratch-01）

| 时刻 | 事件 | 执行者 | 证据等级 |
|---|---|---|---|
| T0 | DE Scheduler 以 `L_DE` 调 DualPath `get_num_new_matched_tokens`，得 `E_DE=max(R-L_DE,0)`；`E_DE==0` 直接 `(0,False)` 终止 | DE Scheduler 进程 | `[设计推导]`（接口契约 `[当前源码已确认]`） |
| T1 | DE Store probe（无副作用=不启动 HBM I/O，但创建 LookupKeyClient/LoadSpec 临时状态） | DE Scheduler 进程内 adapter | `[当前源码已确认]`（F10） |
| T2 | DE 记录 pending candidate，返回 `(E_DE, True)`；请求入 `WAITING_FOR_REMOTE_KVS`，`num_computed_tokens` 乐观置满 | DE Scheduler | `[当前源码已确认]`（U:scheduler.py:1004） |
| T3 | DE 分配正式 blocks（路径无关，一次分足 `R-L_DE`），`update_state_after_alloc`（异步请求两次，第二次 =0） | DE Scheduler | `[当前源码已确认]`（F4） |
| T4 | DualPath 冻结 DE block manifest，不启动 I/O；派发带 `CoverageProposal` 的截断 PE 模型请求（`R=P-1`） | DE Connector | `[设计推导]` |
| T5 | PE Scheduler 首个 pass：proposal 随 `kv_transfer_params` 原子到达，`L_PE` 作为调用参数可得；PE DualPath 在 `get_num_new_matched_tokens` 内同步完成 decide+commit（本地），**返回 0 让出 winner** | PE Scheduler 进程 | `[设计推导]`（时序自洽性已论证，见下） |
| T6 | PE Multi 顺序查询：DualPath→0，AscendStore→`H_PE>0` 成为 winner，唯一 accounting | PE Scheduler | `[当前源码已确认]`（F1/F3 机制） |
| T7 | PE 分配 blocks；非 winner DualPath（Layerwise 子类）也拿到真实 blocks+真实 tokens | PE Scheduler→Multi | `[当前源码已确认]`（AMC:36，F1） |
| T8 | `PathDecisionCommit` 经 control transport 发 DE，DE 持久化后 `DecisionAck`；DE 侧 `abort_probe(handle)` | PE/DE control 线程 | `[设计推导]` |
| T9 | PE sibling AscendStore bulk load `[L_PE,K_PE)` 到 PE 正式 blocks | PE Worker + Store 线程 | `[当前源码已确认]`（F11 机制） |
| T10 | Store 完成 → sibling 发布 PE-local `finished_recving`（Multi 并集、全 rank 聚合）→ PE 请求离开 WAITING | PE 框架 | `[当前源码已确认]` |
| T11 | PE compute `[K_PE,R)`，按层 Forward `[L_DE,R)`（含 PE 本地已有+Store 加载+新计算三段） | PE Worker send thread | `[设计推导]`（方向化 adapter 为新增） |
| T12 | DE 收齐全部层/rank 的 receiver terminal → `FORWARD_DONE` | DE Worker | `[设计推导]` |
| T13 | DE 发布 DE-local `finished_recving`（全 rank 聚合后）→ DE 离开 WAITING → 重算末 token → decode | DE 框架 | `[当前源码已确认]`（框架段） |

关键问题回答：
- 谁做 probe：DE（本路径 DE probe 后 abort）；PE Store lookup 由外层 sibling 完成。谁决定路径：PE（唯一提交方）。committed 时刻：T5 本地 commit + T8 Ack 持久化。
- external tokens：DE 声明 `E_DE`；PE 声明 `H_PE`（只含 Store 实际加载前缀）——两 Engine 各自闭合。
- 并发：PE Store load 与 DE 等待并发；Forward 各层流水。无多 writer：Forward 写 `[L_DE,R)`，DE 本地段 `[0,L_DE)` 不重写。
- first-winner 与 decision 协议的先后关系**自洽**（正向结论）：proposal 随请求原子到达、`L_PE` 首 pass 可得、decision 不依赖 PE Store 结果（§7.2），无循环依赖；框架每 pass 重查（U:scheduler.py:774-787，agent 直验），`(None,False)` 重试语义成立。
- 完成契约：正常路径两 Engine 各自闭合；**失败路径不闭合**（见候选 C-PE-1/2）。

缺陷候选（Phase 3 复核对象）：
- C-PE-1（P0 候选）：PE sibling `load_async` 下 Store 失败上报断裂（F11）——invalid 永不上报 + 请求无条件标完成 → 双 Engine 静默脏数据「成功」。§14.1 L1969 与 §15.4 L2133-2135「依赖现有 Scheduler failure gate」在当前代码下不成立。pool_worker.py:457-465 / kv_transfer.py:908-910,942。
- C-PE-2（P1 候选）：commit 后 PE→DE 无失败/取消通知（envelope 仅 4 类，L790-795）+ DE 等 `FORWARD_DONE` 无超时 → PE 侧 post-commit 失败使 DE 请求永久滞留、blocks 泄漏。
- C-PE-3（P2 候选）：§8.5「只在 num_external>0 冻结 block plan」（L681-684）与 PE 两段式分配冲突——tail `[K_PE,R)` blocks 在第二次调用（=0）才出现；Store miss（`H_PE=0`）时永不冻结。
- C-PE-4（P2 候选）：DECIDING/alloc 阶段 DE abort 的终态合成责任与 probe handle 回收未规定（F13 delay_free 泄漏）；已派发 PE 请求无 CANCEL 通道。
- C-PE-5（P2 候选）：A″ 未强制 PE sibling `use_layerwise=false`；误配时非 winner Store child 被置 `can_load=True`（pool_scheduler.py:589-593，F2）→ Store GET 与 Forward/Reverse 写竞争 PE blocks。
- C-PE-6（P3 候选）：§8.2 步骤 11「Commit+Ack 后才启动 Store load」（L571-572）与 A″ sibling 持有 PE Store 矛盾；Ack 等待点未声明。
- C-PE-7（P3 候选）：PE 侧 proposal 等待无超时（L579-581）。
- C-PE-8（P3 候选）：数据面 shutdown 无有界 drain（仅 control transport 有，L782-783）。

---

## 2. DE_FULL_HIT（主链 T0–T17，详见 scratch-02）

| 时刻 | 事件 | 执行者 | 证据等级 |
|---|---|---|---|
| T0–T3 | 同 PE_READ（probe、声明 `E_DE`、分配/冻结 blocks、入 WAITING） | DE | 同上 |
| T4 | 不创建 PE 模型请求；control-only `CoverageProposal` 经 ZMQ control transport 发 PE | DE control 线程 | `[设计推导]` |
| T5 | PE PathDecisionCoordinator 按静态规则 1（`K_DE>=R`）commit `DE_FULL_HIT`；不需要 `L_PE` | PE control 线程 | `[设计推导]`（正向：pe_coverage 可空与该规则兼容） |
| T6 | commit 回 DE，DE 持久化 + Ack | 双向 control | `[设计推导]` |
| T7 | `commit_after_alloc(handle, blocks, H_DE)` 冻结 LoadSpec，请求入 `loading_req_ids` | DE Scheduler 侧 adapter | `[设计推导]`（触发线程未定，见 C-FH-2） |
| T8–T10 | 每 step metadata 发射 → 各 rank bulk `m_store.get` 写 DE 正式 blocks `[L_DE,K_DE)` → done_recving 内部消费为 `STORE_DONE` | DE Worker + Store 线程 | `[当前源码已确认]`（机制）；失败段见 C-FH-4 |
| T11–T13 | 无 in-flight writer 后发布 DE-local `finished_recving` → 全 rank 聚合 → promote（`cache_blocks(R)`，`-1` 分支天然不触发）→ 重算末 token → decode | DE 框架 | `[当前源码已确认]`（U:scheduler.py:2521-2522 agent 直验） |

关键问题回答：
- 谁做 probe：DE；谁决定：PE control-only（不创建 PE 模型请求）；committed：T6。
- blocks：T3 一次分足、路径无关——decision 改判 PE_READ 时 blocks/accounting 零收敛成本（正向结论 Z2：窗口内 blocks 处「已预留未发布」纯预留态）。
- 成功谓词只有 `STORE_DONE`：T10 的 done_recving 链路在当前 KVPool 语义下存在，但**失败分支断裂**（F11）。
- 末 token 语义：`E_DE=R-L_DE` + 乐观 `num_computed=R` 与 upstream assert/promote 精确咬合，重算零框架修改（正向结论 Z1，本路径最干净一环）。
- 正向：第二次 `update_state_after_alloc(0)` 只可能在 promote 后到达，`can_load=False` 危险分支结构上不可达（Z3）；多 rank 聚合与 STORE_DONE→finished_recving 链路被当前框架原生承载（Z4/Z5）。

缺陷候选：
- C-FH-1（P1 候选）：冻结窗口内 abort 无终态发布者——迟到 commit 的 load 状态被 build_connector_meta finished 清理抹掉，blocks/`self.requests` 永久泄漏；§9.5 与 §13.4 的 finished_sending 前置条件互相矛盾。
- C-FH-2（P1 候选）：`commit_after_alloc` 触发线程未指定——control 线程直写无锁 `KVPoolScheduler` 与 scheduler 循环竞态：字典迭代期写 → EngineCore 崩溃；`loading_req_ids` 快照错位 → `STORE_DONE` 永久丢失。
- C-FH-3（P1 候选）：设计 `S_DE` 二次 floor（§5.2 L273）与 probe「已 floor + full-hit 减 1」语义冲突——full hit 系统性退化为 partial，或照字面接线触发 pool_scheduler.py:604 assert 崩溃；§4.4/§14.2 判定表述与 §5.2 数学不一致。
- C-FH-4（P1 候选）：F11 断裂不修复则部分 DMA 失败被当 `STORE_DONE`，脏 KV 经 `cache_blocks` 进 prefix cache 并污染后续请求；修复可达（recv 线程构造器原生支持注入）但依赖私有属性，设计未写明 block→request 归因。
- C-FH-5（P2 候选）：probe zmq REQ `recv()` 无超时挂在 DE 调度循环上（F10），LKS 慢/死则全引擎停摆；§19.1 的 RPC 失败降级不覆盖 hang。
- C-FH-6（P2 候选）：`m_store.get` hang 无 watchdog——请求永久 WAITING、容量泄漏。
- C-FH-7（P3 候选）：control-only 被 PE 降为 PE_READ 时补派发 PE 模型请求的路径未规定。
- C-FH-8（P3 候选）：`DECISION_TIMEOUT` 的 scheduler→worker 失败传播（无 plan 请求）与 PE 侧 candidate TTL/incarnation 失联清理未规定。
- C-FH-9（P3 候选）：决策态无 tombstone 保留期（§6 只覆盖 wire ID），迟到 commit 可复活已终态请求。

---

## 3. DE_PARTIAL_READ（主链 T0–T17，详见 scratch-03）

| 时刻 | 事件 | 执行者 | 证据等级 |
|---|---|---|---|
| T0–T4 | 同 PE_READ（probe、声明、分配、dispatch 截断 PE 请求 + partial coverage） | DE | 同上 |
| T5 | PE 首个 pass 拿到真实 `L_PE`，静态策略 + 硬准入（`L_DE<K_DE<R`、`L_PE<K_DE`）→ commit `DE_PARTIAL_READ` | PE Scheduler 进程 | `[设计推导]`（触发点见 C-PR-11） |
| T6 | PE Multi：DualPath 返回 `K_DE-L_PE>0` 成为 winner；AscendStore 仍被查询但不覆盖 | PE Scheduler | `[当前源码已确认]`（F1/F3 机制）；child0 短路语义见下 |
| T7 | commit+Ack 回 DE；`commit_after_alloc` 启动 DE Store load `[L_DE,K_DE)` | 双向 | `[设计推导]` |
| T8 | `STORE_DONE`（内部）→ DE 逐层 Reverse `[0,K_DE)`（PE 只写 `[L_PE,K_DE)`，manifest 带全量映射） | DE Worker send thread | `[设计推导]`（入口为新增 adapter，F5/F6） |
| T9 | PE 收齐所有层/rank 的 receiver terminal → `REVERSE_DONE` → 发布 PE-local `finished_recving`（Stage 1 不做跨层流水） | PE Worker→框架 | `[设计推导]`；与 upstream promote/prefix-cache 语义精确兼容（正向 P-1） |
| T10 | PE 逐层 compute `[K_DE,R)` + Forward 写回 DE `[K_DE,R)` | PE Worker | `[设计推导]` |
| T11 | DE `STORE_DONE && FORWARD_DONE` → 发布 DE-local `finished_recving` → decode | DE | `[设计推导]` |

关键问题回答：
- 成功谓词不含 `REVERSE_DONE` 的传递性论证**成立**（正向）：机制 = 框架 scheduler 门禁（PE 收齐 REVERSE_DONE 才离开 WAITING 才 compute 才 Forward），不依赖 no-op 的 `wait_for_layer_load`（F14）；`FORWARD_DONE` 传递性蕴含 Reverse 完成。
- 并发写：Store 写 `[L_DE,K_DE)` 与 Forward 写 `[K_DE,R)` block 区间不相交且由 STORE_DONE→Reverse→compute→Forward 串行定序，**无已证实物理并发写**（正向 P-2）。
- PE `(None,False)` 等待在 Multi 下无侧漏：child0 短路使 AscendStore 本轮不被调用，F3 残留前提不成立（S11 结论）。
- DE Reverse send task 无任何现有入口（F5/F6），必须靠新 `submit_layer_send` adapter 直接入队——设计 §9.2 的覆盖要求与此一致，但属全新代码。
- pending raw terminal inbox 正确修复父类 F8 早到事件丢弃缺陷（正向 P-3）。
- rank 分裂：invalid 无 quorum 先杀、finished 计数扣住成功（与 F12 对齐，正向）。

缺陷候选：
- C-PR-1（P1 候选）：Reverse 目标（PE）block manifest 无 PE→DE 传输通道——commit 在 PE 分配前发出且消息集只有 4 种类型，`reverse_rank_mappings.remote_block_ids` 无生产者（§8.1/§8.7/§10.1）。
- C-PR-2（P1 候选）：跨 Engine 失败/取消传播缺失——`CONTROL_ERROR` 有类型无流程；Sender(DE Reverse) 成功但 Receiver(PE) 失败时 DE 永 hang（§8.7 L794、§13/§19 均单 Engine 闭环）。
- C-PR-3（P2 候选）：决策阶段失败（DECISION_TIMEOUT/dispatch 失败）发生在 Scheduler 进程，无 Scheduler→Worker 失败注入路径；`wait_for_commit` 驱动者未指定。
- C-PR-4（P2 候选）：同 C-PE-5（sibling `use_layerwise=True` 误配 → can_load 竞争）。
- C-PR-5（P1 候选）：DE Store adapter 强制 `load_async=True` 正踩 F11 断裂 → 失败 load 被消费成 `STORE_DONE`，静默数据损坏。
- C-PR-6（P2 候选）：§19.2「收集所有相关块」未排除 DE 共享 prefix 块 `[0,L_DE)` → invalid 误伤无关请求 + prefix cache 误逐出（U:scheduler.py:2665/2736-2737）。
- C-PR-7（P2 候选）：F15 发送线程失败归因 bug（MLC:507）被 Reverse/Forward 复用继承 → 误杀+hang+跨 Engine 状态不一致。
- C-PR-8（P2 候选）：Store→Reverse 可见性仅依赖 host 顺序 + `m_store.get` 同步语义 `[外部接口待确认]`，无 device-level fence。
- C-PR-9（P2 候选）：§15.4 取消未启动 plan 未要求仍发布一次 receive terminal → delay_free blocks 泄漏（U:scheduler.py:2144-2147/2580-2586）。
- C-PR-10（P3 候选）：§7.3 L468「超过」未含 `L_PE==K_DE` 等号（§5.2 L306 严格小于已兜住，措辞瑕疵）。
- C-PR-11（P2 候选）：`decide_on_pe` 触发点未钉死；若早于拿到真实 `L_PE` 则准入失效 → winner 旁落 + PE 读未反转的脏前缀。

---

## 4. 跨路径共性观察

1. **三条路径的 happy path 均在「设计推导」层面闭合**；不闭合集中在失败/取消/超时语义（C-PE-2、C-FH-1、C-PR-2、C-PR-3）与两个当前代码缺陷的继承（F11 Store 失败断裂、F15 归因 bug）。
2. **F11（Store load_async 失败上报断裂）是全局性前提**：PE_READ（经 sibling）、DE_FULL_HIT、DE_PARTIAL_READ（经内部 adapter）全部依赖 Store 失败能被上报；设计 §19 的失败语义在当前 KVPoolWorker 代码下不成立，且「不修改 AscendStore 既有 Worker/backend」的硬边界（§3.1）使修复只能走 adapter 注入路线。
3. **decision 协议与框架的咬合点整体自洽**（PE_READ/DE_FULL_HIT 正向结论），但 DE_PARTIAL_READ 多出两个消息面缺口（C-PR-1 block manifest 通道、C-PR-2 失败传播）。
4. 无已证实的物理并发写；所有「多 writer」风险均为「无协调的 ownership/order 风险」级别，需靠 BlockOwnershipLedger + fence 落实，当前证据不构成已验证冲突。

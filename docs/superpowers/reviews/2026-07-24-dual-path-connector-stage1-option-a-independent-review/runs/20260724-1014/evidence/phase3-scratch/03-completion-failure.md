# Phase 3 维度评审 03：完成/失败事件语义（维度 9–13 + §9.3 专项 + 过度设计检查）

- 评审人：Phase 3 维度评审代理（completion/failure 分工）。
- 设计基线：`inputs/option-a-detailed-design.snapshot.md`（行号以此为准，下称「设计 §x Lnnn」）。
- 证据基线：`evidence/00-baseline.md`、`01-current-control-flow.md`（F1–F18）、`02-path-timelines.md`、phase1-scratch 03/04/05/06（均已读）；关键锚点由本人对当前源码逐一复核原文（下方标注「本人复核」）。
- 源码基线：vllm-ascend `dev/dualpath @ 0ec11a47`；upstream `vllm @ 8df14cfc`（v0.23.1rc0-1050，配套 v0.23.0，**涉及 upstream 的结论存在小幅版本偏差**）。
- 缩写：`MLC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`；`AMC` = `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`；`U:` = upstream 仓库内文件；`pool_scheduler.py`/`pool_worker.py`/`kv_transfer.py` 均指 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` 下文件。
- 证据等级：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[设计推导]` / `[外部接口待确认]` / `[事实冲突]`。
- 本人复核过的源码锚点（全部与 Phase 1/2 证据一致）：MLC:1400-1438（get_finished/invalid）、MLC:467-527（发送线程失败归因）、pool_worker.py:457-465 与 1333-1337（F11 断裂两端）、kv_transfer.py:820-844,902-942,321-337（recv 线程构造/失败集合/暂存语义）、pool_scheduler.py:896-915（finished 清理点）、U:scheduler.py:770-787,2138-2182,2559-2586（None 重试/abort-delay_free/终态消费）、U:kv_connector/utils.py:66-165（聚合器计数与 invalid 并集）。

---

## 维度结论

### 维度 9（含专项 9.3）：finished event 的 source / direction / rank / request provenance

**结论：事件身份模型在「单 Engine 可辨识性」上基本完备——专项 9.3 的九个追问中七个有明确、可落地的答案；但存在三处断裂：(a) 事件结构缺少 §10.3 归属校验所要求的 plan digest/decision_version 字段（设计内部不一致，新发现 CF-1）；(b) 跨 Engine 失败/取消传播缺失使「哪个已提交 plan 已失败」在 sender 侧可能永远不可知（C-PE-2/C-PR-2，P1）；(c) 事件信封把两个方向的 wire ID 都装进 `ids` 再另加 `channel_identity`，存在冗余与一致性校验义务（CF-1 合并记录）。**

逐问核对（依据：设计 §6 L339-389、§10.3 L1514-1573、§13 L1745-1933）：

1. **哪个请求**：`DualPathTransferEvent.ids.request_key`（DE incarnation + DE local ID，L346-350）；公开 `finished_*` 只含本 Engine 精确 local ID（L382「任何对外完成集合只能包含当前 Engine 的精确 local request ID」；L1817-1819 禁止 wire ID/proxy ID 进入 finished_*）。✓
2. **哪个 Engine**：`source_engine_incarnation` / `target_engine_incarnation`（L1541-1542）。✓
3. **Forward 还是 Reverse**：`direction`（store_to_engine/de_to_pe/pe_to_de，L1539）+ `operation`（STORE/REVERSE/FORWARD，L1528-1531）。两者语义重叠但一致；Forward/Reverse wire ID 从 request key 加 direction 确定性派生（L374-375）。✓
4. **哪个 TP rank**：`tp_rank` 字段（L1540）；L383-384 明确序列化字段名 `tp_rank`，不依赖 PP=DP=1 下与 global rank 的数值巧合。✓
5. **Store、P2P 还是 control transport**：Store/P2P 由 `operation`+`channel_identity` 区分；control transport 消息走独立的 `ControlEnvelope`（message_id/request_key/payload_hash，L787-800），与数据面事件是两套身份体系，不混淆。✓
6. **哪个已提交 plan**：事件 → `request_key` → `RankTransferFence.key` → `fence.decision`（L1588-1598）。**但**：§10.3 L1559-1561 要求 mapping 建立后「重新校验 …和 plan digest」，而 `DualPathTransferEvent`（L1535-1552）没有任何 plan digest 或 `decision_version` 字段——校验没有可比对的输入。属设计内部不一致（CF-1，P3）。
7. **是否允许重复**：完全相同重复成功事件幂等忽略（L1556）；同一 local ID 只发布一次 receive terminal（L1795）；每个公开 terminal 至多一次（L2472）；fence 有 `terminal_published` 守卫（L1597）。✓ 规则闭环。
8. **何时可转成公开 finished_***：§13.2 七条规则（L1787-1799：先 invalid、drain/quarantine 解除后才发 failed terminal、恰一次）；§13.3（框架全 rank 聚合并 Publish）；§13.4（finished_sending 仅限 delayed-free 释放，L1829-1841）；DE_PARTIAL_READ 的 PE 侧仅完整 REVERSE_DONE 后才发 PE-local finished_recving（L1618-1636）；Store done 不透传（L1341-1342）。✓ 定义完整且与框架消费语义兼容（U:scheduler.py:2574-2586，本人复核）。
9. **何时可释放 blocks**：§13.5 逻辑/物理生命周期分离（L1902-1933）+ §12 ownership 归零 + quarantine 解除。**缺口**：quarantine 解除条件全部是外部事件（「source process 已死亡、TE session 已失效、target fence 已确认」或明确 receiver terminal，L1571-1573），无任何本地超时/看门狗触发器——对端活着但传输静默停滞时，释放可永不发生（活性缺口，与维度 12/13 结论及 C-PE-2 同源）。

跨 Engine provenance 断裂（P1，链 C-PE-2/C-PR-2）：§10.3 L1567-1568「成功后收到失败=协议错误」「失败后收到成功=只用于 drain」都假定失败事实对双 Engine 可知；但 `ControlEnvelope` 的 `CONTROL_ERROR` 在全设计中只有类型定义、无任何发送/处理流程（grep snapshot 全文仅 L794 一处命中，本人复核），PE 侧 post-commit 失败无法通知 DE。此时 DE 侧 fence 永不满足 §13.1 成功谓词、也无失败事件可归属——provenance 链在「失败」这一分支上断裂。

### 维度 10：raw terminal retention、wire ID、tombstone 是否足以处理迟到事件

**结论：对「早到」场景机制充分且是对父类已确认缺陷的正确修复（正向）；对「迟到」场景，数据面有 tombstone 但保留期无取值/无配置/无 GC 责任方，决策面（迟到 commit/ack）完全无 tombstone——后两者是尚未回答的设计问题（CF-2，P3），决策面缺口已由 C-FH-9 独立立档（不在本人复核清单，维度结论引用）。**

依据：

- **早到（F8 修复，正向）**：父 Worker `get_finished` 用 `if s in self.request_map` 过滤且 `get_and_clear` 已清空，早到 DONE 被永久丢弃（MLC:1401-1416，本人复核），请求将滞留 `WAITING_FOR_REMOTE_KVS`。设计的 pending raw terminal inbox 按 `(wire_external_id, direction, tp_rank)` 暂存、mapping 建立后重新校验归属（L1557-1561；§8.2 L576-577；§9.5 L1104-1106），修复方向正确。Store 侧 done 事件另有天然暂存：`get_and_clear_finished_requests(req_ids)` 只清除交集、未命中保留在线程集合下步重取（kv_transfer.py:321-337，本人复核）。
- **wire ID 防 stale**：active/retired 永不重分配（L378-379）；事件需 wire ID+direction+incarnation+channel identity 四元组匹配（L380-381）；命中 tombstone 的迟到事件丢弃并记录、不得重新归属（L1562-1563）。✓ 规则完整。incarnation 隔离 restart（当前 engine_id 缺省 uuid4，MLC:702）使 request key 复用风险在主语义上闭合。
- **retention 三缺**（CF-2）：L379「直到 transport drain 或 **terminal retention window** 到期」——窗口时长无值、§16 配置（L2201-2251）无对应项、无 GC 责任线程/触发点；§10.3 L1564「**长期**无法匹配…进入协议错误或 quarantine」——「长期」无定义；tombstone 只覆盖 wire/数据面事件，§6 全节（L339-389）对决策面消息无对应物（C-FH-9 的迟到 commit 复活风险即源于此）。
- **长期无法匹配的去向**：协议错误或 quarantine、不得静默丢弃（L1564-1565）✓ 方向正确，但与上一条叠加后，「多久算长期」决定 inbox 内存上界与 quarantine 误杀率，必须给出取值。

### 维度 11：retry、duplicate、cancel、timeout 是否幂等

**结论：retry 与 duplicate 的幂等规则完整且与框架语义精确兼容（正向）；cancel 不闭合——pre-commit abort 无终态发布者（C-FH-1）、取消未启动 plan 未要求发布恰一次 receive terminal（C-PR-9）、probe handle 请求级回收未规定（C-PE-4），三条均直接违反框架「delay_free 的释放以 connector 事后上报 finished 为前提」的已确认契约；timeout 只覆盖决策面且其失败传播无通道（C-PR-3），数据面无请求级超时。**

依据：

- **control retry/duplicate（正向）**：重试复用同一 `message_id`，接收端按 `(message_id, payload_hash)` 幂等（L806-809）；相同 id 不同 hash=协议错误（L809）；重试耗尽且未 commit → `DECISION_TIMEOUT`、不自行切路（L810-811）；commit CAS + 同 version 幂等 + 不同 path/version 协议错误（L705-709，L2467-2469）。
- **框架重入幂等（正向）**：`update_state_after_alloc` 异步请求必调两次、第二次 `num_external_tokens=0`（U:base.py:488-512；U:scheduler.py:822-827→969-974，F4），设计按 `request_key+decision_version` 幂等且仅 tokens>0 冻结（L681-684）✓；PE `(None,False)` 重试由框架保证（U:scheduler.py:781-787，本人复核；Multi child0 短路使 sibling 本轮不被调用，无侧漏——F16/F3 在该拓扑下不可达）。
- **数据面操作幂等（正向）**：`start_store_load` 每 request_key 至多一次；每 `(request_key, rank, layer, direction)` 发送至多一次；每个公开 terminal 至多一次（L2470-2472）。
- **cancel（不闭合，P1/P2）**：框架事实——abort 也同步触发 `request_finished`/`request_finished_all_groups`（U:scheduler.py:2162，本人复核）；WAITING 请求 abort 默认 `delay_free_blocks=True`（U:scheduler.py:2144-2147，本人复核）；延迟 blocks 只在 connector 日后上报该 ID 的 finished_recving/sending 后释放（U:scheduler.py:2578-2586，本人复核；F13）。设计 §13.5/§19.2 的终态机全部以「子操作失败」为触发（L1902-1933、L2440-2452），对「从未 commit、从未启动 I/O」的 candidate 和「取消未启动 plan」两条路径没有终态义务人——详见候选复核 C-FH-1/C-PR-9/C-PE-4。
- **timeout（半闭合）**：决策面有 `request_timeout_ms × max_retries`（L810-811）；但 DECISION_TIMEOUT 判定在 Scheduler 进程，失败传播到 Worker（发布 invalid+terminal）无通道（C-PR-3）；数据面（Store DMA、Forward/Reverse 等待）全无请求级超时或心跳（§19.1 错误分类无对应项；RankTransferFence 无时间戳字段）。

### 维度 12：多 rank 聚合是否可能提前成功或永久等待

**结论：「提前成功」在「每 rank 每 ID 恰一次」规则 + fence `terminal_published` 守卫下被防住，该规则对框架计数语义是必要条件（正向）；「永久等待」在设计上未被排除——无请求级 transfer 超时 + 跨 Engine 失败传播缺失（C-PE-2/C-PR-2，P1）+ F15 归因错误可致真正失败方永不发布（C-PR-7），任一发生时该 Engine 的请求永久滞留。**

依据：

- **不会提前成功（正向）**：框架 `KVOutputAggregator` 对 finished_sending/recving 按 req_id 倒计数，`_expected_finished_count`（默认 world_size）减到 0 才对外发布（U:kv_connector/utils.py:78-90，本人复核；F12）。注意该计数是「次数」而非「不同 rank」——同一 rank 跨 step 重复发布会提前凑满，故设计 §13.3 L1816「每个 rank 对该 ID 只发布一次」+ fence `terminal_published`（L1597）是**必要**防护而非冗余；§9.6 `aggregate()` 按 `(request_key, tp_rank)` 合并、冲突即协议错误（L1157）可捕获同 step 内的重复/身份错乱（跨 step 重复由 fence 守卫兜底）。设计不启用 worker 动态改写 `expected_finished_count` 的机制（U:utils.py:103-114），且 §17 L2343 fail-fast 锁定 `world_size == TP size`——攻击面已关闭。
- **失败优先、成功被扣住（正向）**：invalid 并集无 quorum（U:utils.py:159，本人复核）且 Scheduler 先消费 invalid 再消费 finished（U:scheduler.py:1578-1586 vs 1838）——任一 rank 失败先杀请求（与 §13.3 L1814 一致），成功 rank 的 finished 被倒计数扣住不会误放成功；FINISHED_ERROR 后各 rank 发布一次 failed terminal 凑满计数 → else 分支 `_free_blocks`（U:scheduler.py:2580-2582），§19.2 步骤 7-8（L2450-2452）与此对齐。
- **永久等待（P1 缺口）**：(a) 跨 Engine 失败无通知（C-PE-2/C-PR-2）→ 对端 fence 永不满足谓词、无失败事件；(b) 无请求级 transfer 超时（S8/C-FH-6 同源）；(c) F15 发送线程失败归因 bug（MLC:507 用 :469 循环残留变量，本人复核）可使真正失败的请求不被标记 failed → 该 rank 永不发布 failed terminal → 倒计数悬挂（C-PR-7）；(d) 框架聚合器的 `remaining_count_dict` 无 GC（U:utils.py:84-90，本人复核），悬挂的半完成计数在 Engine 生命周期内残留（小内存泄漏，附带观察，非设计缺陷）。

### 维度 13：失败后 invalid blocks、Scheduler 状态和物理传输清理是否一致

**结论：设计的终态规则本身（invalid 不晚于 failed terminal、terminal 恰一次、逻辑/物理生命周期分离）与框架消费顺序和 assert 精确兼容（正向）；但一致性在三处不成立：(a) F11 断裂使 DE Store adapter 路径上 invalid 根本不会发布、失败被消费成 STORE_DONE（C-FH-4/C-PR-5，P1，本维度最严重项）；(b) pre-commit abort 场景无 plan 无 I/O，§9.5 与 §13.4 的 finished_sending 前置条件互相矛盾，释放义务人不存在（C-FH-1，P1）；(c) quarantine 解除无本地触发器，物理清理可永不完成（活性缺口，P2）。另有 C-PR-6（invalid 范围含共享 prefix 块误伤无关请求，P2，非本人复核候选，维度结论引用其证据）。**

依据：

- **规则与框架兼容（正向）**：§13.2「invalid 可更早发布但不晚于失败终态」（L1794）与 upstream 契约（U:base.py:384-389「失败 block 不得晚于该请求 finished 上报的同一 pass」）及「invalid 先于 finished 消费」（U:scheduler.py:1578-1586，本人复核）同向；「同一 ID 一次 receive terminal」「失败也经 finished_recving」（L1795-1799）正是 scheduler assert（U:scheduler.py:2576/2581/2585，本人复核）的充要条件；fail 策略下 FINISHED_ERROR + delay_free + terminal 到达后 `_free_blocks`（U:scheduler.py:1823-1825,2144-2147,2580-2582，本人复核）与 §13.5 物理生命周期（L1919-1928）一致。F9（父类 `request_finished_all_groups` 恒 `(False,None)`，MLC:1114-1124，已由 Phase 1 复核）使 §13.4 的 delayed-free 覆盖成为必需。
- **(a) F11 断裂（P1）**：本人逐行复核两端——`KVPoolWorker` 创建 `KVCacheStoreRecvingThread` 时未注入 invalid 集合/锁（pool_worker.py:457-465，仅 7 个位置参数），而线程构造器原生接受注入（kv_transfer.py:830-831）未注入则退化私有集合（:843-844）；失败块入私有集合（:908-910,924-926）无任何 drain 方；`KVPoolWorker.get_block_ids_with_load_errors` 只 drain worker 自有集合（pool_worker.py:1333-1337）；请求仍无条件 `set_finished_request`（kv_transfer.py:942）。⇒ DE Store adapter（强制 `load_async=True`，L1334）会把部分/全部失败的 load 消费成 `STORE_DONE`，脏 KV 经 promote 的 `cache_blocks` 进 prefix cache 污染后续请求。设计 §9.7 L1317/L1347 声明「统一进入 request-level invalidation」但未写明在不修改 KVPool（§3.1 L73-80）前提下的检测手段（注入共享集合或 drain 私有集合均触私有属性）与 block→request 归因规则。详见 C-FH-4 复核。
- **(b) pre-commit abort 矛盾（P1）**：C-FH-1 复核中逐条展开。
- **(c) quarantine 无本地解除触发器（P2）**：§10.3 L1571-1573 的解除条件全部是外部事件；§13.2 规则 3 又把 failed receive terminal 门在「quarantine 已解除后」（L1791-1793）——对端存活但传输静默停滞时，失败 terminal 永不发布、blocks 永不释放。与维度 9 第 9 问、维度 12 永久等待同源。
- **C-PR-6 引用（P2）**：§19.2 L2446「收集请求相关所有 KV block IDs」未排除 DE 共享 prefix 段 `[0,L_DE)`；框架会对落在 invalid 集合的 running/waiting 请求截断 `num_computed`（U:scheduler.py:2665）、fail 策略逐出 prefix cache（:2736-2737）——共享段纳入 invalid 会把单请求失败放大为无关请求失败。该候选由其他评审代理复核，此处仅登记其与维度 13 的直接关联。

---

## 过度设计检查

判定标尺：六问（①保护哪个 Stage 1 invariant；②已有模块能否提供；③删除后哪个时序出错；④Stage 1 必需还是 YAGNI；⑤是否引入新状态源或双重记账；⑥更小替代方案）。分类：必须保留 / 必须保留但应简化 / 可以复用已有模块替代 / Stage 1 YAGNI / 当前证据不足。

### A. raw terminal retention（pending raw terminal inbox + wire ID tombstone + retention window）

1. **保护的 invariant**：receiver terminal 不因其到达早于本地 completion mapping 而丢失；迟到事件不归属到已终止/已复用身份。这直接守护「请求必然离开 WAITING_FOR_REMOTE_KVS」与「finished_* 只含精确 local ID」两条 Stage 1 契约。
2. **已有模块能否提供**：父 Worker 不能——它正是缺陷源（早到 DONE 被 `request_map` 过滤永久丢弃，MLC:1407/1416，本人复核）；Store 侧有天然暂存（kv_transfer.py:321-337 交集语义）但只覆盖 Store done，不覆盖 P2P DONE/FAILED；`KVOutputAggregator` 无暂存语义（U:utils.py:78-90）。
3. **删除后哪个时序出错**：T13/T16（PE 收 Reverse、DE 收 Forward）中，metadata 经 ZMQ 广播（U:multiproc_executor.py:310-320）与已常驻的 recv thread 之间存在真实跨步窗口；terminal 早到即丢 → 该 rank 永不发布 → 聚合倒计数悬挂 → 请求永久滞留 WAITING（维度 12 的永久等待形态之一）。删除 tombstone：engine restart/incarnation 复用后迟到事件可归属到新请求，污染跨请求完成语义。
4. **Stage 1 必需还是 YAGNI**：**必需**（修复的是当前代码已确认缺陷 F8，不是假想需求）。
5. **新状态源/双重记账**：引入两个新状态（inbox、tombstone 表），但与 fence 记录的不是同一事实（inbox=未归属事件缓冲，fence=已归属进度；tombstone 与 active wire ID 互斥），生命周期衔接清晰，不构成双重记账。
6. **更小替代**：可复用 KVPool 的「完成集合按键保留直到命中」模式统一 Store/P2P 两类事件的暂存，少一个结构；retention window 必须给出取值/配置与 GC 责任方（当前缺失，CF-2）。
- **分类：必须保留但应简化**（统一暂存模式 + 补 retention 取值与 GC 责任）。

### B. SchedulerReleaseLedger（§13.4 L1843-1900）

1. **保护的 invariant**：请求结束（含 abort）时若 plan 仍可能持有 async sender/reader owner，blocks 不得被 Scheduler 立即释放复用——防止 in-flight 单边写落入已复用 block 污染后续请求。覆盖父类恒 `(False,None)` 的 F9（MLC:1102-1124）。
2. **已有模块能否提供**：框架提供 delay_free 通道（`request_finished_all_groups` → `finished_sending` → `_free_blocks`，U:scheduler.py:2162,2583-2586）但**不提供「是否该 delay」的判定**；Multi 的 `_extra_async_saves`（AMC:97-98）只解决多 child 发布去重；KVPool `_delayed_free_req_ids` 只管 Store save。判定逻辑无已有模块。
3. **删除后哪个时序出错**：T17（PE_READ/DE_PARTIAL_READ 请求结束时 Forward send 仍 in-flight）→ 父类行为下 blocks 立即进 free 流程 → 新请求复用 → in-flight 单边写污染新请求 KV（跨请求静默损坏）。
4. **Stage 1 必需还是 YAGNI**：**判定功能必需**；但实现形态有 YAGNI 成分——`request_finished_all_groups` 只读 `plan_has_async_sender_or_reader` 一个布尔（L1870-1877，且注释自承「Conservative: once a plan could own async I/O, always ask delay」），`SchedulerReleaseState.rank_statuses` 与 `update_from_worker_metadata()` 在 Scheduler 侧**没有任何决策读取者**。
5. **新状态源/双重记账**：**存在未使用的双重记账**——`rank_statuses` 把 Worker `BlockOwnershipLedger`/fence 的 `active_owner_count/unknown_inflight/send_drain_done` 在 Scheduler 进程再存一份，唯一潜在消费者是本 ledger 自己的判定，而判定不用它。释放安全性实际由 Worker 侧「ownership=0 且无 unknown in-flight 才发 finished_sending」（L1892-1894）保证，Scheduler 侧副本是死状态。
6. **更小替代**：ledger 退化为「plan registry + 一个布尔查询」，`request_finished_all_groups` 直接查 plan 标记保守返回 delay_free=True；`RankTransferStatus` 仍可作为纯诊断经 WorkerMetadata 回流（供 §24 可观测性与冲突检测），但不入 Scheduler 状态机。
- **分类：必须保留但应简化**（CF-3：删除/降级 `rank_statuses` 状态副本，`update_from_worker_metadata` 限定为诊断用途）。

### C. SharedMooncakeTransferRuntime 之外的完成语义类族

逐项判定（类清单取自 §9.1 类图 L822-862、§10.3、§11.1、§13.2、§9.6）：

- **RankTransferFence（§11.1 L1588-1598）**：① 保护「每 (request, rank) 三操作进度可判定 §13.1 成功谓词 + failed 态 + terminal_published 恰一次」。② 父类只有 done/failed 两集合（MLC:1400-1428），无逐请求三操作进度；KVPool loading_req_ids 不表达 Reverse/Forward。③ 删除后 finished_recving 发布条件与 drain 完成判定无法实现（T14/T16）。④ 必需。⑤ Worker 侧唯一进度源，无双重记账。**分类：必须保留。**
- **TransferFenceRegistry（§9.1 类图 L836/L850）**：①-④ 全文无任何接口/职责定义（grep snapshot 仅类图两处命中，本人复核）——它保护的 invariant 可由 Worker 内一个 `dict[TransferFenceKey, RankTransferFence]` 承载。⑤ 作为独立类会诱生「Registry 与 Worker 状态何者为准」的疑问。⑥ 更小替代：Worker 私有字典。**分类：当前证据不足（倾向 Stage 1 YAGNI，建议降级为 Worker 内部容器而非公共类）。**
- **AtomicRequestOutcome（§13.2 L1766-1780）**：① 承载「invalid 不晚于 failed terminal、terminal 恰一次」的单 pass 决策结果，purpose 三分（PE_REVERSE_TERMINAL/DE_RECEIVE_TERMINAL/DELAYED_FREE_RELEASE）对应三种公开动作。② 无已有模块。③ 删除后 §13.2 七条规则无载体，实现易散落成临时布尔。④ 必需。⑤ 纯值对象，非状态源。**分类：必须保留。**
- **DualPathTransferEvent（§10.3 L1535-1552）**：① 维度 9 provenance 的载体。② 父类事件只是 req_id 字符串，无 direction/rank/incarnation。④ 必需。⑤ 值对象；但字段冗余：`ids` 内含两个方向的 wire ID，事件又带 `channel_identity`——应按 direction 只带单一 wire ID（并入 CF-1）。**分类：必须保留但应简化。**
- **RankTransferStatus / DualPathConnectorWorkerMetadata（§9.6 L1132-1158）**：① Worker→Scheduler 的诊断回流 + Store metadata 委托聚合 + `(request_key, tp_rank)` 冲突检测（捕获重复发布/身份错乱）。② 框架提供聚合通道（U:utils.py:135-144）但无 DualPath 语义。③ 删除后 Store adapter 的 `update_connector_output` 委托与冲突检测消失。④ 通道与委托必需；状态字段的 Scheduler 侧决策消费为空（见 B）。⑤ `statuses` 与 Worker fence 构成双重记账（未用）。⑥ 更小替代：保留 metadata 结构与冲突检测，`active_owner_count/unknown_inflight/send_drain_done` 标记为诊断字段。**分类：必须保留但应简化。**
- **BlockOwnershipLedger（§12）**：属另一评审代理的过度设计分工范围，本文件不重复判定；仅记录接口依赖：§13.4 闭环（L1880-1897）要求 Worker 侧 ledger 输出 `active_owner_count/unknown_inflight/send_drain_done` 三值供 Worker 自己判定 finished_sending——该用途成立，与 B 的「Scheduler 侧副本冗余」不冲突。

---

## Phase 2 候选复核

### C-PE-2（P1）：commit 后 PE→DE 无失败/取消通知，DE 等 FORWARD_DONE 无超时 → DE 永久滞留

**复核：确认，严重级别 P1 维持；补充一条间接检测路径不可依赖的证据。**

- 缺口存在性（本人复核）：`ControlEnvelope.message_type` 仅 4 类（L790-795）；`CONTROL_ERROR` 在 snapshot 全文仅 L794 定义处出现，无发送/处理流程；§8.7 超时只覆盖 proposal/commit/ack（L810-811）；§14.1 DE 成功条件仅 FORWARD_DONE（L1968）；RankTransferFence 无时间戳/心跳字段（L1588-1598）。
- 框架侧（本人复核）：离开 `WAITING_FOR_REMOTE_KVS` 的唯一条件是聚合后 finished_recving（U:scheduler.py:2526-2541 及 2574-2579），无超时分支；delay_free blocks 释放依赖 finished 到达（U:scheduler.py:2144-2147,2580-2586）。
- 具体时序（T0..Tn）：T0 commit+Ack 完成 → T1 PE 侧 post-commit 失败（Store 失败/abort/崩溃/Forward send 静默失败）→ T2 无 CONTROL_ERROR 流程，DE 收不到任何事件 → T3 DE fence 的 forward_done 永不置位且无失败标记 → T4 DE 请求永久滞留 WAITING → T5 blocks 因 delay_free 永久持有、`_inflight_prefills` 容量泄漏。
- 补充：scratch 已指出 §10.3 L1571-1573 的 quarantine 解除条件「只解 quarantine 不产生请求终态」；本人进一步确认：即使「TE session 已失效」被确认，设计也没有任何组件负责把该事实翻译成请求级 `fail()`——监控/确认主体未定义，间接检测路径不可依赖。
- 为什么不属「尚未实现」：§19.1 失败分类自称完备（L2416-2438）且 CONTROL_ERROR 类型已定义，缺口是**设计未规定流程**而非实现未做。

### C-PE-4（P2）：DECIDING/alloc 阶段 DE abort 的终态合成责任与 probe handle 回收未规定；已派发 PE 请求无 CANCEL 通道

**复核：确认，并扩展——泄漏窗口可精确界定为「dispatch 成功之前」，dispatch 成功后 DE 侧可由数据面自然收敛；与 C-FH-1 同根因但收敛性不同，P1/P2 分级差异因此成立。**

- 框架事实（本人复核）：abort 触发 `request_finished`（U:scheduler.py:2162）、WAITING 默认 delay_free（:2144-2147）、释放依赖 finished 到达（:2580-2586）。
- 设计缺口（本人复核）：§13.5/§19.2 均以「失败」为触发（L1902-1933、L2440-2452）；§9.7 仅 shutdown 兜底 abort pending handles（L1359-1360）；envelope 无 CANCEL（L790-795）。
- 扩展分析（新证据/新推论）：
  - **窗口 (i) abort 在 dispatch 前（T1–T2）**：无 PE 请求、无 I/O、无 plan → 无任何终态发布者 → 泄漏（与 C-FH-1 同机制）。
  - **窗口 (ii) abort 在 dispatch 后、commit/Forward 完成前**：PE 请求无 CANCEL 可达而继续执行；若 PE 跑完（Store+compute+Forward），DE recv thread 收到 DONE → fence FORWARD_DONE → DE Worker 对该 DE-local ID 发布一次 finished_recving → 聚合 → scheduler else 分支（assert is_finished ✓）→ `_free_blocks` —— **DE 侧可经数据面自然收敛**；PE 侧浪费有界（max_tokens=1）。
  - **窗口 (iii) dispatch 后 PE 请求失败**（如 F11 修复后的 Store 失败 → PE FINISHED_ERROR → 无 Forward）→ 退化为 C-PE-2（DE 无通知滞留）。
  - **probe handle 面**：请求级 abort 无 handle 回收义务人；KVPool `load_specs` 在 abort 路径本就无清理（F10，pool_scheduler.py:690/742/845 之外无 pop；request_finished 1012-1037 不触碰）——双重残留。
- 结论：P2 合理（窗口 (i) 泄漏 + 窗口 (iii) 已另有 P1 立档）；scratch 原文未区分窗口 (ii) 的收敛性，此处补齐。

### C-FH-1（P1）：冻结窗口内 abort 无终态发布者；§9.5 与 §13.4 的 finished_sending 前置条件互相矛盾

**复核：确认，P1 维持；矛盾两条款与清理点均经本人原文复核。**

- 矛盾条款（本人复核）：§9.5 L1108-1111「对其中的 delayed-free ID，仅在本地 ownership=0 且无 unknown in-flight 后返回 finished_sending」——凭框架传入的 finished_req_ids 即可发布；§13.4 L1829-1835「公共 finished_sending 只能用于…`request_finished()` 或 `request_finished_all_groups()` 曾返回 delay_free=True」——本场景 `SchedulerReleaseLedger` 无 plan 返回 `(False,None)`（L1870-1872），前置条件 2 不满足。两条规则对本场景给出相反答案，实现者照哪条都有据。
- 清理点（本人复核）：`build_connector_meta` 对 `finished_req_ids` 逐条 discard `_loading_req_ids`、pop `_unfinished_requests`（pool_scheduler.py:896-901）——迟到 commit 应用后写入的 load 状态会在下一步被该清理抹掉，load metadata 永不发射（ReqMeta 由 `_process_async_load_request` 生成且 LoadSpec pop 只发一次，:839-885,951-957）→ 无 I/O → 无终态 → 泄漏。子情形 (b) 成立。
- 具体时序：T0 分配+WAITING（delay_cache 隔离）→ T1 abort（窗口 ≥ 一次 control 往返，必然非空）→ T2 框架 delay_free=True 持有 blocks → T3 connector 无 plan 返回 (False,None) → T4a commit 永不到达且 DECISION_TIMEOUT 传播未实现 → 无发布者；或 T4b 迟到 commit → `commit_after_alloc` 写 `_loading_req_ids` → T5 下一步 finished 清理抹掉 → load 永不发射 → 仍无发布者 → T6 blocks/`self.requests`/`_inflight_prefills` 永久泄漏（`has_finished_requests` 使引擎不退净，U:scheduler.py:2250-2260）。
- 附带泄漏：pending candidate、probe handle、`load_specs[req]`（F10）。
- 最小修正方向（与 scratch 一致，本人认可）：`request_finished(_all_groups)` 对 pending candidate 必须 `coordinator.fail(request_key, ABORTED)` + `abort_probe(handle)` + 规定由 Worker 对该 DE-local ID 发布恰一次终态（failed receive terminal 或 finished_sending 二选一写死）；统一 §9.5/§13.4 前置条件；commit 应用前查 coordinator 是否已 terminal/aborted。

### C-FH-4（P1）：F11 断裂不修复则部分 DMA 失败被当 STORE_DONE，脏 KV 进 prefix cache；修复依赖私有属性且未写明归因

**复核：确认，P1 维持；断裂两端与修复可达性均经本人逐行复核，并补充两点：单 KV group 限制恰好保证失败块入集合的分支可达；归因可行性的框架依据成立。**

- 断裂（本人复核）：pool_worker.py:457-465 创建 recv 线程未注入 invalid 集合/锁 → kv_transfer.py:843-844 退化私有集合 → 失败块入私有集合（:908-910,924-926）无 drain 方 → `get_block_ids_with_load_errors` 只 drain worker 自有（pool_worker.py:1333-1337）→ invalid 永不上报；请求无条件 `set_finished_request`（kv_transfer.py:942）。
- 补充 1：失败块入集合以 `len(req_meta.block_ids_by_group) == 1` 为条件（kv_transfer.py:908,924）；§17 单 KV group 限制（L2345）保证走该分支——断裂在 Stage 1 拓扑下**必然**生效（hybrid 的「只 log 完全静默」分支被拓扑限制排除）。
- 补充 2：修复可达性——线程构造器原生接受 `invalid_block_ids/lock` 注入（kv_transfer.py:830-831），adapter 拥有 `KVPoolWorker` 实例，属组合使用而非修改（不违反 §3.1）；block→request 归因可行：窗口内 blocks 请求独占（`delay_cache_blocks` 未入 cache + async load 不支持块共享，U:scheduler.py:2652-2653 注释），按 block_id ∩ 冻结 manifest 求交即可。
- 设计缺口确认：§9.7 L1317/L1347 只声明目标（「统一进入 request-level invalidation」），未写明注入/drain 私有集合的手段与其脆弱性（私有属性名、KVPool 重构即碎），未写明归因规则。后果时序：T11 部分失败 → T12 消费成 STORE_DONE → T13-15 promote + `cache_blocks(R)` → 脏 KV 进 prefix cache → 本请求输出错误 + 后续请求命中同一前缀被污染（静默、跨请求扩散）。

### C-PR-1（P1）：Reverse 目标（PE）block manifest 无 PE→DE 传输通道

**复核：确认，并扩展——缺口是双向的：`CoverageProposal` 只携带 `de_block_manifest_digest`（摘要），Forward 方向（PE→DE 写）所需的 DE 完整 block manifest 同样没有已定义的通道；扩展影响 PE_READ/DE_PARTIAL_READ 两条路径的 Forward mapping。**

- 原候选证据（本人复核）：`PathDecision` 无 block 字段（L489-497）；`PathDecisionCommit` 仅含 decision（L543-545）；消息仅 4 类（L790-795）；commit 在 T6 发出而 PE blocks 在 T7 才分配；`reverse_rank_mappings.remote_block_ids`（L1409；`RankBlockMapping` L1392-1397）无生产者。
- 扩展证据（本人复核）：`CoverageProposal` 字段为 `ids/de_coverage/capability_digest/de_block_manifest_digest/requested_policy`（L532-540）——只有 DE manifest 的**摘要**；而 Forward SendTask 由 PE 构造，`forward_rank_mappings.remote_block_ids` 需要 DE 的**完整** block ids，摘要只够等值校验不够构造 SendTask。§7.3 L473「PE commit 时可把已冻结的 DE block manifest 作为硬准入条件」隐含 PE 应持有 manifest 本体，进一步印证完整 manifest 需要通道而设计未定义。现有 MLC 的 D→P 回传机制（consumer 在 `update_state_after_alloc` 把 local_block_ids 经 metaserver POST 回传，MLC:957-972）在 DualPath 完全覆盖 `update_state_after_alloc`（L905-908）后无对应物；PE→DE 方向（Reverse 需要）则连现有机制都不存在。
- 影响面修正：scratch 将缺口定位在 DE_PARTIAL_READ 的 Reverse；扩展后——Reverse（PE→DE manifest）无任何现有机制可借用（P1 不变）；Forward（DE→PE 完整 manifest）有现有 metaserver 回传模式可复用但设计未声明（同等设计缺口，修复成本较低）。
- 最小修正方向（覆盖两个方向）：dispatch envelope 携带完整 DE block manifest（替代/补充 digest），commit 推迟到 PE `update_state_after_alloc` 后并携带 PE manifest，或新增幂等 `POST_ALLOC_MANIFEST` 控制消息（复用 message_id/payload_hash 规则），并钉死其为 DATA_START 前置条件。

### C-PR-2（P1）：跨 Engine 失败/取消传播缺失（CONTROL_ERROR 有类型无流程）

**复核：确认，P1 维持；与 C-PE-2 同根因、本候选覆盖反向（DE 失败/abort 使 PE 成孤儿）与「Sender 成功 Receiver 失败」形态。**

- 本人复核佐证：CONTROL_ERROR 全文仅 L794（grep 确认）；§13/§19 的完成/失败语义全部单 Engine 闭环（L1745-1933、L2411-2459）；S9 形态（PE 收齐 REVERSE_DONE 后本地失败 → PE FINISHED_ERROR，DE reverse 已 done → 永等 Forward → hang 至客户端 abort）在设计上无事件可达 DE；C2 形态（DECIDING 阶段 DE abort → PE 已 commit 等 Reverse → 孤儿）同理。
- 框架后果（本人复核）：delay_free  blocks 长期占用（U:scheduler.py:2144-2147）；`has_finished_requests` 阻止引擎退净（U:scheduler.py:2250-2260）。
- 边界待确认（与 scratch 一致）：proxy/metaserver 是否传播客户端取消到被派发请求 `[外部接口待确认]`——不影响缺口成立（即便 proxy 传播取消，DE 侧的等待也无超时）。
- 最小修正方向：定义 CONTROL_ERROR/cancel 流程（失败/abort 方 coordinator → 对端 coordinator，按 request_key 路由、幂等），对端按 §19.2 进入失效流程；声明 proxy 取消传播是依赖还是非目标。

### C-PR-9（P2）：取消未启动 plan 未要求仍发布一次 receive terminal → delay_free blocks 泄漏

**复核：确认，P2 维持；补充其与其他取消候选的统一修正框架。**

- 设计缺口（本人复核）：§15.4 步骤 5（L2136-2137）「finished PE-local ID 用于取消未启动 plan、清理 fence」——只字未提发布终态；对照 §13.2（L1787-1799）的发布规则无回指。
- 框架依赖（本人复核）：WAITING abort → delay_free（U:scheduler.py:2144-2147）→ 释放只在 finished_recving/sending 到达后发生（:2578-2586）；upstream 契约「即使失败，请求仍必须经 get_finished 上报」（U:base.py:385-389）。
- 具体时序：T0 PE 侧请求在 WAITING 期被 abort/失败（如 sibling Store 失败经 Multi 汇总——F11 修复后）→ T1 框架 delay_free 持有 PE blocks → T2 finished_req_ids 传到 DualPath Worker → T3 按 L2136-2137 取消未启动 plan、清理 fence，但不发布任何 terminal → T4 Scheduler 等不到释放事件 → blocks + `self.requests` 泄漏。
- 统一修正（覆盖 C-PR-9/C-FH-1/C-PE-4）：任何经 `finished_req_ids` 到达 Worker 的取消/失败 ID，只要框架可能已 delay_free（WAITING abort 默认如此），DualPath 必须对该 local ID **恰一次**发布终态（失败 receive terminal 或 finished_sending，按 §13.4 分工写死），再清理 fence/plan。

---

## 新发现

### CF-1：`DualPathTransferEvent` 缺少 §10.3 归属校验要求的 plan digest / decision_version 字段；事件信封 wire ID 冗余

- 标题：事件结构与归属校验规则不一致 + 信封冗余
- 严重级别：P3
- 类型：设计内部不一致（事件 schema vs 校验规则）
- 证据等级：`[事实冲突]`（设计文本间，本人复核）
- 设计位置：snapshot L1535-1552（事件字段）vs L1559-1561（「mapping 建立后：重新校验 …和 plan digest」）；冗余：L353-360（`TransferRequestIds` 含双方向 wire ID）+ L1543（`channel_identity`）
- 源码位置：—（纯设计层；对照现有实现 MLC:1407 的归属仅按 request_map，无 digest 概念）
- 触发前提：任何 raw terminal 早到进入 pending inbox、mapping 建立后执行归属重校验的时刻。
- 具体时序：T0 recv thread 收到 DONE/FAILED → T1 本地无 mapping → 按 `(wire_external_id, direction, tp_rank)` 暂存（L1557-1558）→ T2 plan/mapping 经 metadata 到达 → T3 按 L1559-1561 重新校验「request key、direction、incarnation、channel identity、**plan digest**」→ T4 事件结构中无 plan digest / decision_version 可比 → 校验要么退化为无 digest 版本（弱化防 stale 语义），要么实现者自行扩展事件字段（绕过评审）。
- 为什么不正确：校验规则要求的输入在数据结构里不存在；属设计内部矛盾。冗余面：每个事件携带 `ids`（含 forward+reverse 两个 wire ID）再加 `channel_identity`，事件的单一 wire 身份需按 direction 二选一推导，平增一致性校验义务与误接风险。
- 可能影响：归属校验弱化（迟到/错归属事件拦截率下降）或实现分叉。
- 与已有模块的兼容性：不影响框架通道；仅 DualPath 内部 schema。
- 是否属于过度设计：否（是欠设计/冗余并存）。
- 最小修正建议：`DualPathTransferEvent` 增加 `decision_version: int` 与 `plan_digest: str`（或在 `channel_identity` 中形式化嵌入 digest 并写清编码）；按 direction 只序列化单一 `wire_external_id`，删除事件级 `channel_identity` 或将其定义为该 wire ID 的别名。
- 修正后需要增加的测试：早到 terminal 归属时 digest 不匹配被拒；重复/迟到事件四元组+digest 全匹配才归属；单 wire ID 序列化字段回归（§23.1「ID 与传输身份」组）。
- 置信度：高（文本矛盾客观存在）；影响低，故 P3。

### CF-2：raw terminal retention window 无取值、无配置项、无 GC 责任方；「长期无法匹配」无时长定义

- 标题：tombstone/inbox 保留期是未回答的设计问题
- 严重级别：P3
- 类型：设计未回答问题（生命周期/GC）
- 证据等级：`[设计推导]`（文本穷举，本人 grep 复核）
- 设计位置：snapshot L378-379（「直到 transport drain 或 terminal retention window 到期」）、L1564（「长期无法匹配…进入协议错误或 quarantine」）；§16 配置全文（L2201-2251）无对应项
- 源码位置：—（纯设计层）
- 触发前提：请求终止后存在迟到事件，或 pending inbox 中有长期无法归属的事件。
- 具体时序：T0 请求 terminal → T1 wire ID 退休留 tombstone → T2 迟到事件到达命中 tombstone 被丢弃 ✓ → T3 tombstone 何时期满、谁清理、inbox 中无法匹配条目多久算「长期」→ 全部无定义 → 实现者各自取值 → 过短则迟到事件误进协议错误/quarantine（误杀），过长则 tombstone/inbox 无界增长（内存缓漏）。
- 为什么不正确：GC 三要素（时长、责任线程/触发点、配置项）全部缺失；维度 10 的「足以处理迟到事件」在该取值缺席时无法最终判定。
- 可能影响：误杀（quarantine 抖动）或内存缓漏；行为不可审计。
- 与已有模块的兼容性：无冲突；§16 配置体系可直接扩展。
- 是否属于过度设计：否（是必要机制的缺失参数）。
- 最小修正建议：§16 增加 `terminal_retention_ms`（默认值与 quarantine 判定解耦）+ 写明 GC 由 Worker 的周期 pass（如 `get_finished` 内）执行；「长期无法匹配」按同一时长界定。
- 修正后需要增加的测试：tombstone 期内迟到事件被丢弃、期后 tombstone 被回收；inbox 超期条目进协议错误路径（§23.1 事件组）。
- 置信度：高（缺失客观存在）；P3 因不阻塞 happy path。

### CF-3：`SchedulerReleaseState.rank_statuses` / `update_from_worker_metadata` 在 Scheduler 侧无决策消费者，构成未使用的双重记账

- 标题：SchedulerReleaseLedger 的状态副本冗余
- 严重级别：P3
- 类型：过度设计（状态冗余/双重记账）
- 证据等级：`[设计推导]`（文本穷举，本人复核 L1843-1900 全部使用点）
- 设计位置：snapshot L1843-1878（SchedulerReleaseState/Ledger）、L1880-1897（闭环描述）、L1132-1158（RankTransferStatus 字段来源）
- 源码位置：—（组件不存在；对照框架机制 U:scheduler.py:2583-2586 finished_sending→_free_blocks，本人复核）
- 触发前提：任何请求结束走 `request_finished_all_groups`。
- 具体时序：T0 Worker 每步上报 `RankTransferStatus(active_owner_count, unknown_inflight, send_drain_done)` → T1 经 WorkerMetadata.aggregate 回流 → T2 `SchedulerReleaseLedger.update_from_worker_metadata` 写入 `rank_statuses` → T3 `request_finished_all_groups` 判定只读 `plan_has_async_sender_or_reader` 与 `delayed_free_registered`（L1870-1877）→ T4 `rank_statuses` 永不参与任何判定 → 真正的释放安全由 Worker 侧「ownership=0 且无 unknown in-flight 才发 finished_sending」（L1892-1894）保证，Scheduler 副本是死状态。
- 为什么不正确：与 Worker `BlockOwnershipLedger`/fence 记录同一批事实的第二份拷贝，无读者；增加一致性问题面（两份状态何时一致、以谁为准）而无收益。
- 可能影响：实现与维护成本；误导后续开发者把判定建在过期副本上。
- 与已有模块的兼容性：删除不影响框架通道（WorkerMetadata 仍可回流作诊断）。
- 是否属于过度设计：**是**（详见「过度设计检查 B」）。
- 最小修正建议：ledger 退化为 plan registry + 布尔查询；`update_from_worker_metadata` 删除或降级为诊断日志更新；`RankTransferStatus` 三字段标注为诊断用途，不进入 Scheduler 状态机。
- 修正后需要增加的测试：`request_finished_all_groups` 保守 delay_free=True 的判定不依赖 WorkerMetadata 回流时序（先 finish 后回流/先回流后 finish 两序）；§23.1 Ownership 组现有条目（L2613-2616）改写。
- 置信度：中高（「无读者」基于 snapshot 文本穷举；若设计者本意是让 Scheduler 侧做更细判定，文本未体现）。

### 正向结论（成立原因 + 前提 + 源码契约）

- **PC-1：pending raw terminal inbox 是对父类已确认缺陷的正确修复**。成立原因：父 Worker 早到 DONE 被 `request_map` 过滤永久丢弃（MLC:1407/1416，本人复核）→ 请求滞留 WAITING；inbox 暂存 + mapping 后重归属 + 四元组校验 + tombstone（L1556-1563）闭合该窗口。前提：inbox 键控与 tombstone 按 CF-2 补全 GC。源码契约：Worker 每 step 必调 `get_finished`（U:mixin:102-104）；Store 侧天然暂存（kv_transfer.py:321-337，本人复核）。
- **PC-2：§13.2 原子终态规则与框架消费顺序/assert 精确兼容**。成立原因：invalid 先于 finished 消费（U:scheduler.py:1578-1586，本人复核）支撑「invalid 不晚于失败终态」；「同 ID 恰一次 receive terminal + 失败也经 finished_recving」是 scheduler assert（:2576/2581/2585）的充要条件；FINISHED_ERROR 后 delay_free + terminal 到达 `_free_blocks`（:2144-2147,2580-2582）与 §13.5 物理生命周期一致。前提：每 rank 每 ID 恰一次（fence `terminal_published` 守卫）；F11 修复（否则 invalid 根本不发布，见 C-FH-4）。
- **PC-3：复用 KVOutputAggregator 计数 barrier 做公开终态聚合是「利用」而非「对抗」框架**。成立原因：finished 倒计数需全期望 rank（U:utils.py:78-90）、invalid 并集无 quorum 且先消费（:159 + scheduler.py:1578-1586）——失败优先、成功被扣住不误放；§13.3 的「每 rank 恰一次」规则+fence 守卫恰好补上「计数按次数不按 rank」的必要条件。前提：不启用 worker 动态改写 expected_finished_count（:103-114，设计未使用）；`world_size == TP size` 由 §17 fail-fast 锁定。
- **PC-4：control transport 的 retry/duplicate/冲突语义设计完整**。成立原因：message_id+payload_hash 幂等（L806-809）、不同 hash 即协议错误（L809）、commit CAS + 版本幂等（L705-709, L2467-2469）、重试耗尽不自行切路（L810-811）——与 §3.5 的 commit 后不可切换约束自洽。前提：接收端幂等表与 coordinator 状态的 GC 有界（与 CF-2/C-FH-9 关联）。
- **PC-5：§13.4 delayed-free 闭环的框架通道真实存在且被正确识别**。成立原因：父类恒 `(False,None)`（F9，MLC:1102-1124）使覆盖必需；框架释放通道 `request_finished_all_groups → delay_free → finished_sending → _free_blocks`（U:scheduler.py:2162,2179-2181,2583-2586，本人复核）与 L1880-1897 闭环逐步对应；Worker 侧「不阻塞等待、每步 get_finished 轮询」的模型与现有 connector 家族一致。前提：CF-3 的简化不影响 Worker 侧判定。

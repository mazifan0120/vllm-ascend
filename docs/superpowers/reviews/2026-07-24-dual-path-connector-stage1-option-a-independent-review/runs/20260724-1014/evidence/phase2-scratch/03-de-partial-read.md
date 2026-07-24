# Phase 2 时序审查 03：DE_PARTIAL_READ（方案 A″）

- 审查路径：`DE_PARTIAL_READ`（Store 部分命中后 DE→PE 逐层 Reverse 前缀 + PE 计算 tail + PE→DE 逐层 Forward）。
- 设计基线：`inputs/option-a-detailed-design.snapshot.md`（行号以此为准，下称「设计 §x.y Lnnn」）。
- 源码基线：vllm-ascend `dev/dualpath @ 0ec11a47`；upstream `/Users/leqi/Documents/Code/vllm @ 8df14cfc`（v0.23.1rc0-1050，配套 v0.23.0 有小幅偏差，涉及 upstream 的结论均受此约束）。
- 缩写：`MLC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`；`AMC` = `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`；`U:` = upstream 仓库内文件；F1-F18 指 `evidence/01-current-control-flow.md` §16 事实清单。
- 证据等级：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[设计推导]` / `[外部接口待确认]` / `[事实冲突]`。顺序性质分三类：**框架保证**（upstream 控制流/assert 强制）、**当前实现偶然**（当前代码恰好如此，无契约保护）、**纯设计假设**（组件不存在：PathDecisionCoordinator、control transport、fence/ledger、方向化 adapter、pending inbox，见 phase1-01 §15）。

## 必答问题速查

- **谁做 probe**：DE Scheduler 进程内 `DualPathStoreSchedulerAdapter.probe()`（设计 §9.7 L1236-1245），底层复用 `KVPoolScheduler` lookup（非纯 probe，会建 LookupKeyClient/LoadSpec，pool_scheduler.py:515-553，F10）。PE 不做 Store probe 参与决策；外层 AscendStore sibling 的 lookup 仅是 first-winner 协议的副产物（设计 §15.6 L2175-2176）。
- **谁决定路径**：PE 是唯一提交方（设计 §8.2 L501），`PathDecisionCoordinator.decide_on_pe()`（设计 §8.6 L697-703）。
- **哪一刻算 committed**：PE coordinator `commit()` CAS `DECIDING→COMMITTED`（设计 §8.6 L705-709）；数据面启动还要等 `PathDecisionCommit + DecisionAck + frozen block plan` 三者齐备（设计 §8.2 L571-572）。
- **每个 Engine 声明多少 external tokens**：DE 声明 `E_DE = R - L_DE`（设计 §5.2 L281/L290）；PE 声明 `max(K_DE - L_PE, 0)`（设计 §5.2 L292、§15.3 L2097-2102）。两者都是 `load_async=True`。
- **谁分配正式 blocks**：各 Engine 自己的 Scheduler 走正常 `allocate_slots`（U:scheduler.py:942-954，`delay_cache_blocks=True`）；DE 在 decision 之前分配并冻结（设计 §7.3 L470-472）。
- **token range 写/读分工**：`[0,L_DE)` DE 本地共享 prefix cache（Reverse 只读）；`[L_DE,K_DE)` Store bulk load 写 DE 正式 blocks（Reverse 读）；Reverse 在 PE 只写 `[L_PE,K_DE)`（设计 §10.1 L1421，PE compute 读）；`[K_DE,R)` PE 计算（Forward 读）并写 DE 正式 blocks（DE decode 读）。
- **哪些操作可能并发**：见 S15——已提交路径上 Store 写与 Forward 写 token 区间不相交且时序串行，无物理并发写；真正并发的是「DE Store load（T10）‖ PE 等待」「DE 等待 commit ‖ PE 决策/分配」这类跨引擎等待，以及多 TP rank 之间的并行传输。
- **使请求离开等待状态的事件**：PE = 聚合后的 PE-local `finished_recving`（仅在完整 REVERSE_DONE 后发布，设计 §11.2 L1618-1636）；DE = 聚合后的 DE-local `finished_recving`（`STORE_DONE && FORWARD_DONE`，设计 §13.1 L1759-1760）。机制均为 U:scheduler.py:2526-2541 `_try_promote_blocked_waiting_request` `[当前源码已确认]`。
- **谁发布 finished_recving/finished_sending**：各 Engine 的 DualPath Worker 按 rank 各发布一次本 Engine 精确 local ID（设计 §13.3 L1810-1819）；`finished_sending` 只用于 delayed-free 释放（设计 §13.4 L1824-1841）。
- **多 rank 汇聚**：复用框架 `KVOutputAggregator`——finished 倒计数需全部期望 rank（world_size==TP size，U:kv_connector/utils.py:78-90）；invalid blocks 并集无 quorum（:159，F12）。
- **取消/超时/部分失败后的 drain/quarantine/释放**：Worker 侧 `BlockOwnershipLedger`（设计 §12），Scheduler 侧 `SchedulerReleaseLedger`（设计 §13.4），请求级失效流程 §19.2；quarantine 解除条件 §10.3 L1571-1573。跨引擎传播缺失见 D-2。

## T 时序主链（happy path：L_DE ≥ 0，L_PE < K_DE，无失败）

### T0 — 初始化（PE/DE 各自 Engine 启动期）

- 执行者：PE/DE 的 Scheduler 进程与全部 Worker 进程。
- DualPathConnector 构造不调父构造器，直接 `KVConnectorBase_V1.__init__`（connector.py:120，`[当前源码已确认]` foundation 现状，与设计 §9.3 L924-927 一致）。
- 每 Worker 进程一次 `register_kv_caches`（model_runner_v1.py:3824-3825 唯一调用点，`[当前源码已确认]`）：复用父函数一次完成 TE buffer 注册 + 一个 `KVCacheSendingLayerThread` + 一个 `KVCacheRecvingLayerThread`（MLC:1247-1398，kv_both 下两个 if 都命中，F7）。`global_te.register_buffer` 有单次守卫（mooncake_transfer_engine.py:31-40）。
- 派生 kv_both runtime 的本地 receive port 按角色选择：PE=Reverse receiver port，DE=Forward receiver port（设计 §18.2 L2388-2390）——每个 Engine 的 recv thread 只服务一个方向，端口空间天然独立（佐证设计 §7.3 L459 可满足）。
- control transport：PE ROUTER bind、DE DEALER connect（设计 §8.7 L753）；capability 交换与拓扑 fail-fast 校验（设计 §17 L2336-2351、§18.2）。
- 顺序性质：父机制 `[当前源码已确认]`；DualPath 专属部分 **纯设计假设**。

### T1 — DE Scheduler：本地 lookup + DualPath matched-token 查询

- 执行者/线程：DE Scheduler（EngineCore）进程，调度循环主线程。
- 请求进 waiting 循环且 `request.num_computed_tokens == 0`：先 `get_computed_blocks` 得本地命中 `L_DE`（U:scheduler.py:724,761-763，已复核原文），随后同分支内调用 `connector.get_num_new_matched_tokens(request, L_DE)`（U:scheduler.py:774-779，已复核原文——第二参数即本地命中数）。**框架保证顺序** `[当前源码已确认]`。
- DE DualPath：`E_DE = max(R - L_DE, 0)`；`E_DE == 0` 直接返回 `(0, False)` 终止 candidate（设计 §8.2 L555-556、§5.2 L301）。`[设计推导]`

### T2 — DE Store probe

- 执行者/线程：DE Scheduler 进程，`DualPathStoreSchedulerAdapter.probe(request, L_DE)`（设计 §9.7 L1236-1245）。
- 底层 `KVPoolScheduler.get_num_new_matched_tokens`：按 `G = lcm(lcm_block_size, family granularity)` 对齐 floor（pool_scheduler.py:404-416,502-503）；非 layerwise 走 zmq REQ 到 rank-0 worker 的 LookupKeyServer，**`recv()` 无超时，在调度循环内阻塞**（pool_scheduler.py:515-521,1107，`[当前源码已确认]` F10）。
- 产出 `StoreCoverage`：`S_DE → H_DE → K_DE = min(max(L_DE,S_DE),R)`（设计 §5.2 L273-276）。副作用：可能创建常驻 LookupKeyClient（pool_scheduler.py:515-516）与 `load_specs[req]`（:549-553）——设计要求 adapter 把临时状态绑定 handle、事后 commit/abort 二选一（设计 §9.7 L1351-1360）。`[当前源码已确认]` 副作用 + `[设计推导]` 收敛规则。

### T3 — DE 声明并分配：pending candidate + 正式 blocks

- DE DualPath 记录 pending candidate，返回 `(E_DE, True)`（设计 §8.2 L559）。
- Scheduler `allocate_slots(num_external_computed_tokens=E_DE, delay_cache_blocks=True)`（U:scheduler.py:942-954）：DE 正式 blocks = `[0,L_DE)` 共享 prefix cache 块 + `[L_DE,⌈R⌉)` 新分配私有块（async load 不支持 block 共享仅指 external 段，U:scheduler.py:2652-2653 注释）。**框架保证** `[当前源码已确认]`。
- `update_state_after_alloc(request, real_blocks, E_DE)` 第一次调用（U:scheduler.py:969-974）；第二次调用（promote 后）`num_external_tokens=0`，实现必须幂等且只在 tokens>0 时冻结（设计 §8.5 L681-684，现实依据 F4）。**框架保证调用次数语义** `[当前源码已确认]`。

### T4 — DE 冻结 manifest + 派发 PE 模型请求；DE 请求进入等待

- DE DualPath 冻结 DE block manifest，**不启动任何 Store/P2P I/O**（设计 §7.3 L470-474、§8.2 L561）。`[设计推导]`
- 派发带 `CoverageProposal` 的 PE 模型请求：envelope 在 `kv_transfer_params["dual_path"]`（设计 §8.2 L528-530、§8.7 L815-816），含 de_coverage/capability digest/**de_block_manifest_digest**/requested_policy（设计 §8.2 L534-540），PE 请求被截断到 `R = P-1`、`max_tokens=1`（设计 §8.3 L595-617）。传输通道复用 metaserver/proxy 异步 POST（当前机制：MLC:973-982 线程池 + :1089-1100 重试 3 次）`[当前源码已确认]` 通道 + `[设计推导]` envelope。
- DE 请求进 `WAITING_FOR_REMOTE_KVS`，`num_computed_tokens` 乐观置 `L_DE + E_DE = R`（U:scheduler.py:986-1004）。**框架保证** `[当前源码已确认]`。

### T5 — PE 接收请求；proposal 原子到达

- PE Engine `add_request` → `on_new_request`；proposal 随 `kv_transfer_params` 与请求同体到达——**PE 侧 DualPath `get_num_new_matched_tokens` 被调用时 proposal 必然已在 request 对象上**（机制与当前 MLC 把 do_remote_decode params 随派发送达一致，MLC:957-972，`[当前源码已确认]` 通道能力 + `[设计推导]` envelope 内容）。
- PE Scheduler waiting 循环：`num_computed_tokens==0` → 本地 lookup 得真实 `L_PE`（block 对齐），传入 connector（同 T1 机制）。
- AscendMultiConnector：无 preempted-priority child 时等价 upstream（AMC:43-63，F1 节）；child 0 = DualPath（配置顺序强制，设计 §15.2 L2066-2072）。

### T6 — PE 决策与 commit；成为 winner

- PE DualPath 读 proposal，`decide_on_pe(proposal, L_PE, pe_store_coverage=None)`：静态策略 + 硬准入（`L_DE < K_DE < R`、`L_PE < K_DE`、layout/TP/handshake/capability，设计 §7.3 L452-460）→ `commit()` CAS（设计 §8.6 L705-709）→ 返回 `(K_DE - L_PE, True)`（设计 §15.3 L2097-2102）。`[设计推导]`；**decide_on_pe 的触发点设计未钉死，必须用本次调用的真实 L_PE——见 D-11**。
- 若 decision 未就绪：返回 `(None, False)` → Multi 任一 child 返 None 立即整体短路（U:multi_connector.py:393-394，F16）→ 请求被 pop 回 skipped 下轮重查（U:scheduler.py:781-787）。child 0 短路意味着 AscendStore 本轮**不被调用**，无任何 lookup 副作用。场景分析见 S11。
- committed 的完整时点：PE 本地 commit + commit 发送 + DE 持久化 + DecisionAck（设计 §8.2 L568-572）。

### T7 — PE Multi 收尾：AscendStore 副产物 + PE 分配

- Multi 循环无 break，继续调 child 1 AscendStore：lookup 副作用照常（可能建 LoadSpec/LookupKeyClient，pool_scheduler.py:515-553），但 `to_return[0]==0` 守卫使结果不覆盖 winner（U:multi_connector.py:397-399）；`_requests_to_connector[req]=0`（:398）。`[当前源码已确认]`
- PE `allocate_slots` → `update_state_after_alloc`：winner DualPath 收 `(real blocks, K_DE-L_PE)`；非 winner AscendStore 收 `(empty blocks, 0)`（AMC:41）→ `num_external==0` 分支：`can_load = use_layerwise and kvpool_cached>0`（pool_scheduler.py:589-593，已复核原文）——**非 layerwise 配置下 can_load=False 无 GET；layerwise 配置下存在竞争窗口，见 D-4**。
- PE 请求进 `WAITING_FOR_REMOTE_KVS`，`num_computed_tokens` 乐观 = `L_PE + (K_DE-L_PE) = K_DE`。**框架保证** `[当前源码已确认]`。

### T8 — commit/ack 控制面往返；DE 发现 decision

- PE → DE：`PathDecisionCommit` 经 control transport（DEALER→ROUTER，同 message_id 幂等重试，设计 §8.7 L807-812）；DE coordinator 持久化后回 `DecisionAck`。**纯设计假设**（组件不存在）。
- ⚠ commit 在 T6 即发出，而 PE blocks 在 T7 才分配——`PathDecision`/`PathDecisionCommit` 数据类均不含 block 字段（设计 §8.1 L489-497、L543-545），消息类型仅 4 种（设计 §8.7 L790-795）：**PE→DE 的 reverse 目标 manifest 无传输通道，见 D-1**。
- DE 侧经 `get_committed()` 发现 decision（设计 §8.6 L717-721）——**驱动点（哪个线程/ hook 轮询、`wait_for_commit` 由谁调用）设计未指定，见 D-3**；随后 `commit_after_alloc(handle, blocks, H_DE)` 冻结 LoadSpec、请求进入真实 `loading_req_ids`（设计 §9.7 L1334-1336）。`[设计推导]`

### T9 — DE plan 下发与 Worker 注册

- DE Scheduler `build_connector_meta()`（每 step，U:scheduler.py:1166-1189）产出 `DualPathConnectorMetadata(plans, store_metadata)`（设计 §9.6 L1127-1130）→ 随 SchedulerOutput ZMQ 广播（U:multiproc_executor.py:310-320,377）→ Worker 主线程 `bind_connector_metadata`（mixin:89）。**框架保证通道** `[当前源码已确认]`；内容 **纯设计假设**。
- DE Worker `register_request(plan)` 幂等建 `RankTransferFence`（设计 §11.1 L1588-1598）与 ownership：`STORE_WRITER` acquire `[L_DE,K_DE)×全 layer`（设计 §12 表 L1738）。`[设计推导]`

### T10 — DE Store bulk load（整段，非逐层）

- DE Worker `start_store_load` → Store adapter → `KVPoolWorker` 异步路径：`KVCacheStoreRecvingThread._handle_request` 一次 bulk `m_store.get` 写 `[L_DE,K_DE)` 全部 layer 的 DE 正式 blocks（pool_worker.py:787-791、kv_transfer.py:846-902；非 layerwise 无 DMA 拆分）`[当前源码已确认]` 机制 + `[设计推导]` adapter。
- 完成：`set_finished_request`（kv_transfer.py:942）→ adapter 按 `store_metadata.loading_req_ids` 交集消费为**内部 STORE_DONE**（不透传公共 finished_recving，设计 §9.7 L1339-1342；交集语义天然暂存早到事件，kv_transfer.py:321-337）。fence：`store_done=True`。
- ⚠ 失败分支继承 F11 断裂：失败块进线程私有集合无人 drain、请求仍无条件置完成（pool_worker.py:457-465、kv_transfer.py:908-910,942）——**adapter 会把失败的 load 消费成 STORE_DONE，见 D-5**。
- Store→Reverse 的物理可见性：依赖 host 侧顺序（`m_store.get` 同步返回后才置完成）+ RDMA 读 HBM；**无 device-level fence 描述，`[外部接口待确认]`，见 D-8**。

### T11 — DE 启动逐层 Reverse（新入口，父类不会代劳）

- DE Worker 状态机观察 `STORE_DONE` → 逐层 `start_reverse_layer(ReverseLayerCommand)`（设计 §9.5 L1086-1090）→ `ReverseLayerwiseTransfer.submit_layer_send()` 消费冻结的 `RankBlockMapping` 构造 SendTask 放入共享 runtime 的 `send_thread.send_queue`（设计 §10.2 L1446-1454、L1501-1512）。**纯设计假设**——当前 kv_both 下父 `start_load_kv` 只走 consumer 分支（MLC:1606-1615，F6），producer mapping 计算整体跳过；`save_kv_layer` 只在 forward 中按层触发（MLC:1699/1836），而 DE 对 prefix 层无计算，**没有任何现有 hook 会把 reverse 任务送入 send thread**。adapter 还需自行复制父 producer 分支的 transfer mapping 逻辑（MLC:1615-1688）并满足接收端 `trans_count` 计数协议（MLC:589-601,1935）。
- 前置依赖：D-1（PE manifest 通道）必须先解决，否则 `reverse_rank_mappings.remote_block_ids` 无来源。

### T12 — DE send thread 执行 Reverse 写

- `KVCacheSendingLayerThread.run`（MLC:266-283，daemon）：逐 SendTask `wait_event.synchronize()`（MLC:481-492）→ 按 `session_id = host:te_rpc_port`（MLC:470）合并 → `batch_transfer_sync_write` 单边写 PE 正式 blocks `[L_PE,K_DE)` 对应 layer（MLC:497-499）`[当前源码已确认]` 线程机制 + `[设计推导]` reverse 任务。
- 末层且 chunk_finish → callback → `send_done_send_signal` 发 `DONE_SENDING_MSG` 到 PE recv thread（MLC:519-527,1920-1935）。失败路径继承 F15 归因 bug（MLC:507 用 :469 残留循环变量），见 D-7。
- wire identity：reverse 使用独立 direction wire ID 与 PE endpoint（设计 §9.2 L902-903、§6 L374-375）；握手走 PE recv thread 的 `GET_META_MSG`（MLC:603-653）——同一 ROUTER 多消息类型，无端口冲突。`[源码可达，尚未运行验证]`

### T13 — PE 接收 Reverse；REVERSE_DONE（rank-local）

- PE recv thread：DONE 按 `side_channel_path` 计数凑满 `trans_count` 置 done（MLC:589-601）。PE Worker 主线程 `get_finished`（每 step，mixin:102，**框架保证每步必调**）：raw terminal 若早于本地 mapping 则进 pending inbox 暂存、mapping 建立后重归属（设计 §8.2 L576-577、§10.3 L1557-1561、§9.5 L1104-1106）——修复父类直接丢弃的缺陷（MLC:1407 过滤 + get_and_clear 清空，F8）。
- 收齐全部 layer → rank-local `REVERSE_DONE`；ownership：`REVERSE_WRITER` release、`MODEL_COMPUTE` acquire（设计 §12 表 L1740-1741）。`[设计推导]`

### T14 — PE 发布 finished_recving；PE 请求离开等待

- PE Worker（每 rank 一次）发布 PE-local `finished_recving` → `KVOutputAggregator` 倒计数（world_size==TP size，U:kv_connector/utils.py:78-90）→ 全 rank 到齐才对外发布。`[当前源码已确认]` 聚合 + `[设计推导]` 发布条件。
- PE Scheduler `_update_from_kv_xfer_finished`：assert 不变式（U:scheduler.py:2576/2581）→ `finished_recving_kv_req_ids` → 下一步 `_try_promote_blocked_waiting_request` → `cache_blocks`（U:scheduler.py:2517-2541）→ PE 离开 `WAITING_FOR_REMOTE_KVS`。**框架保证**。
- Stage 1 刻意等完整 REVERSE_DONE 才发布（不做跨层流水，设计 §11.2 L1618-1624、§2.2 L52），理由：promote 后 blocks 进 prefix cache，逐层 `wait_for_layer_load` 只能保护本请求——论证成立（正向结论 P-1）。

### T15 — PE 计算 tail 并逐层 Forward

- PE 首次真实调度：非 winner AscendStore 的 `load_specs` 在 `_process_new_request` 被 pop（pool_scheduler.py:690），非 layerwise 下 `can_load=False` → ReqMeta 无 load 任务 → **无 Worker Store GET** `[当前源码已确认]`（layerwise 例外见 D-4；waiting 期 abort 则泄漏，F10）。
- 逐层 forward：非 MLA 走 upstream `@maybe_transfer_kv_layer` 装饰器（U:kv_transfer_utils.py:38-59）；ascend MLA/SFA/DSA 走 impl 内 helper，`wait_for_kv_layer_from_connector` 仅 has_prefill 时调用（mla_v1.py:1683-1684，F14）——此处 wait 为父类 no-op（MLC:1976-1977），按设计应立即通过（设计 §11.2 L1635）。
- 每层计算 `[K_DE,R)` 后 `save_kv_layer`（覆盖版，设计 §9.3 L939-950）→ Forward SendTask 只含 `[K_DE,R)`（不得覆盖 Store 前缀，设计 §10.1 L1426）→ PE send thread → 写 DE 正式 blocks `[K_DE,R)`。`[当前源码已确认]` hook 机制 + `[设计推导]` 区间裁剪。

### T16 — DE 收齐 Forward；DE 发布 finished_recving 并离开等待

- DE recv thread DONE 计数 → DE Worker pending inbox 归属 → `FORWARD_LAYER_DONE` → 全层 `FORWARD_DONE` → fence 满足 `store_done && forward_done`（设计 §13.1 L1759-1760）→ 每 rank 发布一次 DE-local `finished_recving` → 聚合 → DE Scheduler promote → `cache_blocks` → DE 离开等待。
- promote 后 DE 计算位置 `R` 的最后一个 prompt token：`num_computed=R=P-1 < num_tokens=P`，不触发 full-hit 末 token 调整（U:scheduler.py:2519-2522）——与 decode-ready 语义精确咬合（正向结论 P-7）。
- `REVERSE_DONE` 不在 DE 成功谓词中：传递性论证成立（详见 S16），但 REVERSE_DONE 仍须跟踪用于失败发现/quarantine（设计 §7.1 L419-424）。

### T17 — 请求结束与 delayed free（两 Engine 各自）

- `_free_request` → `_connector_finished`（U:scheduler.py:2162；abort 也触发，F13）→ `request_finished_all_groups` → `SchedulerReleaseLedger`：`plan_has_async_sender_or_reader` 保守返回 `delay_free=True`（设计 §13.4 L1865-1877）——覆盖父类恒 False（MLC:1102-1124，F9）。
- Worker 等 `ownership=0 && !unknown_inflight` → 每 rank 发布一次 `finished_sending` → 聚合 → Scheduler `_free_blocks`（U:scheduler.py:2583-2586）。**框架保证释放通道** `[当前源码已确认]`；ownership 闭环 **纯设计假设**。

## 场景矩阵

### S1. L_DE = 0（DE 无任何本地前缀）

- 小时序：T1 中 `L_DE=0` → `E_DE=R` → T3 DE blocks 全部新分配（无共享块混合）→ T10 Store 写 `[0,K_DE)` → T12 Reverse 源 `[0,K_DE)` 全部来自 Store → T15 Forward 写 `[K_DE,R)`。
- 与 L_DE>0 的差异：Reverse 源数据单一来源（全部 Store），可见性只依赖 D-8 的 host 顺序；失败时 invalid 范围不含共享块，§19.2 请求级失效天然安全（对照 S2/D-6）。
- 结论：**主链闭合**（以 D-1/D-5 修正为前提）。

### S2. L_DE > 0（DE 有本地共享前缀）

- 小时序：T3 DE blocks = 共享 `[0,L_DE)` + 私有 `[L_DE,R)`；T10 Store 只写 `[L_DE,K_DE)`（不碰共享区，设计 §10.1 L1418）；T12 Reverse 读 `[0,K_DE)`——读共享块安全；失败时若 invalid 含 `[0,L_DE)` 共享块 → 误伤（D-6）。
- 结论：**happy path 闭合；失败路径有缺陷 D-6**。

### S3. L_PE < K_DE（标准 DE_PARTIAL_READ）

- 小时序：T6 PE 返回 `(K_DE-L_PE>0, True)` 成 winner → T7 乐观 `num_computed=K_DE` → T12 Reverse 只写 `[L_PE,K_DE)`（设计 §10.1 L1421）→ T14 promote → T15 PE 算 `[K_DE,R)`。
- 结论：**闭合**。注意 Reverse 写区间与 PE 声明的 external tokens（`K_DE-L_PE`）精确一致；PE 共享块 `[0,L_PE)` 不被覆写（正向结论 P-4）。

### S4. L_PE = K_DE（准入边界）

- 小时序（若错误提交 DE_PARTIAL_READ）：T6 PE 返回 `max(K_DE-L_PE,0)=0` → 必须 `(0,False)` → winner 旁落 AscendStore → PE 请求不进 WAITING、立刻本地调度，attention 读 `[0,K_DE)`——**Reverse 尚未写入 → 读脏数据**；同时 DE 已 commit 等 Forward，Forward 区间会计错。
- 设计的防护：准入要求严格 `L_PE < K_DE`（设计 §5.2 L306、§7.3 L455），不满足则由 §7.2 rule 3 落 `PE_READ`；§7.3 L468 转换清单只写「超过」未含等号——文档瑕疵（D-10），但严格不等式已兜住等号情形。
- 残余风险：准入判定必须使用 T5 时刻的真实 L_PE；若实现把 `decide_on_pe` 放在 proposal 到达时（L_PE 未知按 0 处理）则防护失效——D-11。
- 结论：**设计闭合**（依赖 D-11 指出的实现约束 + D-10 措辞修正）。

### S5. L_PE > K_DE（§7.3 必须转 PE_READ）

- 小时序：T6 `decide_on_pe` 检测到 `L_PE > K_DE`（「PE 已有 prefix 超过 DE ready prefix」，设计 §7.3 L468）→ commit 前转 `PathDecision(PE_READ)` → DE `abort_probe(handle)`（设计 §8.5 L671-673）→ 走 PE_READ 通路（PE Store + PE compute + Forward `[L_DE,R)`，属另一评审路径）。
- DE 已分配 blocks 不受影响（路径无关分配，设计 §7.3 L470-472）；符合 §3.5「commit 前准入失败属正常选路」。
- 结论：**闭合**。

### S6. Reverse 某层失败（DE send `ret<0`）

- 小时序：T12 send thread `batch_transfer_sync_write` 失败 → `failed_reqs.add`（**F15：MLC:507 用 :469 残留循环变量，多请求同 SendTask 时归因可能错误**）→ 末层 callback `trans_flag=False` → `FAILED_SENDING_MSG` → PE recv `update_failed_task`（MLC:639-643）→ PE Worker 按 §19.2 请求级 invalid（全部相关块）→ 并集无 quorum 即发布（F12）→ PE fail 策略 → `FINISHED_ERROR` + delay_free → drain → PE terminal → 释放。DE 侧 sender callback 知失败 → fence failed → DE invalid → `FINISHED_ERROR`。
- 归因错误时（D-7）：被错怪请求误 invalid；真正失败请求在 DE 侧 hang（永等 Forward）。
- 结论：**PE 侧闭合；DE 侧依赖 D-7 修复 + D-2 跨引擎通知**。

### S7. Forward 某层失败

- 小时序：T15 PE send 失败 → FAILED → DE recv → DE 请求级 invalid → `FINISHED_ERROR`；PE sender 侧同步知失败 → PE invalid → `FINISHED_ERROR`。两侧 delay_free → drain → terminal → 释放。
- 结论：**闭合**（同 S6 的 D-7/D-2 前提）。

### S8. 一个 TP rank 成功、另一个失败/超时

- 小时序：失败 rank 的 invalid 并集无 quorum 立即发布（U:utils.py:159）→ 该 Engine 请求失败；成功 rank 的 finished 被 aggregator 倒计数扣住（不足 world_size 不发布，:78-90）→ 不会误放成功；各 rank fence 独立（设计 §11.1 L1600-1602），失败 rank 进 drain/quarantine，成功 rank 待 terminal 保留窗/tombstone（设计 §6 L378-379）。
- 超时：设计无请求级 transfer timeout；依赖 HCCL 层 `ASCEND_TRANSFER_TIMEOUT`（MLC:1131）表面化为传输失败；对端进程死亡无心跳 → hang（并入 D-2/D-3）。
- 结论：**聚合语义闭合**（设计 §13.3 与 F12 精确对齐）；**活性依赖传输层超时，请求级超时未指定**（P3 级缺口，见 D-3 备注）。

### S9. Sender（DE Reverse）成功但 Receiver（PE）失败

- 小时序：T13 后 PE 某 rank 收齐 DONE，但 PE 本地失败（如 ownership 校验/后续 compute 失败）→ PE invalid → PE `FINISHED_ERROR`；DE 对 PE 失败**毫无感知**：reverse 已 done，Forward 永不到来 → DE 请求 hang 在 `WAITING_FOR_REMOTE_KVS` 直至客户端 abort；abort 后 delay_free 持有 blocks（U:scheduler.py:2144-2147），terminal 有本地 fence 可发布 → 释放。若客户端不 abort → 长期占用 + engine 因 `has_finished_requests` 不退净（U:scheduler.py:2250-2260）。
- `ControlEnvelope` 有 `CONTROL_ERROR` 类型（设计 §8.7 L794）但全设计无任何使用流程（grep 全 snapshot 仅定义处命中）。
- 结论：**不闭合——D-2**。

### S10. STORE_DONE 早于 request-ID mapping（F8 场景）

- 小时序：T10 recv 线程 `set_finished_request` 早于 T9 plan 到达 Worker → 当前 Store 机制按 `loading_req_ids` 交集取完成（kv_transfer.py:321-337），未命中者**保留在线程集合中**下步重取——天然暂存 `[当前源码已确认]`；Reverse/Forward 的 raw terminal 由 pending inbox 暂存重归属（设计 §8.2 L576-577、§9.5 L1104-1106）——修复父类丢弃缺陷（MLC:1407，F8）。
- 结论：**闭合**（正向结论 P-2）。

### S11. PE Multi 中 DualPath 成为 winner 的时刻：proposal/decision 就绪性与 (None,False) 后果

- proposal 必然已到达：随 PE 模型请求的 `kv_transfer_params` 同体到达（T5），`add_request` 先于任何 schedule——**框架保证**。
- decision 未必已形成：设计 §15.3 L2095-2096 `decision is None → (None, False)`；§8.2 L579-581 允许等待重试。
- (None,False) 的具体后果（F16/F3 在此路径）：(a) child 0 返 None → Multi 立即短路，**AscendStore 本轮不被调用**（U:multi_connector.py:393-394）→ 无 LoadSpec/LookupKeyClient 副产物；(b) 请求 pop 回 skipped，下轮重查（U:scheduler.py:781-787）→ 代价是每轮一次调度延迟；(c) F3 的 winner 映射残留以「先前迭代已写 winner」为前提——本路径 DualPath 从未返过正数，前提不成立，**无残留**；(d) preemption 重查（U:scheduler.py:1204 清零 num_computed）时 decision 已持久化 → 稳定返回正数。
- 结论：**闭合**（正向结论 P-5）。小缺口：`decide_on_pe` 触发点未钉死（D-11），若迟迟不触发则 None 自旋延长。

### S12. F1：winner 与非 winner blocks + PE 侧 AscendStore 临时 lookup 状态清理

- DE_PARTIAL_READ 在 PE：DualPath 是 winner；AscendStore 非 winner 收 `(empty, 0)`（AMC:41）——F1 的真实 blocks 例外只适用于 Layerwise 子类（AMC:36），即 PE_READ 场景下非 winner DualPath 才收真实 blocks（§15.4「真实 blocks 不构成路径授权」处理，不在本路径）。
- AscendStore 临时状态：其 lookup 在 T7 已被调用（Multi 无 break），若 `S_PE > L_PE` 则建 LoadSpec；非 layerwise 下 T15 `_process_new_request` pop 且 `can_load=False` → 无 Worker GET `[当前源码已确认]`；waiting 期 abort → `load_specs` 无清理路径残留（F10，pool_scheduler.py:690/742/845 之外无 pop）。
- layerwise sibling：T7 `can_load=True`（pool_scheduler.py:589-593，已复核）→ T15 可能发起 layerwise Store GET 写 PE blocks，与 Reverse 已写 `[L_PE,K_DE)` 区间重叠竞争（下游链路 `[源码可达，尚未运行验证]`）——D-4。
- 结论：**默认（非 layerwise sibling）闭合**；layerwise sibling 配置无约束 → D-4；abort 泄漏为既有 F10，adapter/sibling 层均无清理。

### S13. 各阶段取消（probe/alloc/commit/Store load/Reverse/Forward）

- C1 PROBE~CANDIDATE_ALLOCATED（DE abort）：`request_finished` 钩子 abort 也触发（U:scheduler.py:2162，F13）→ adapter `abort_probe` 清 handle/LoadSpec（设计 §9.7 L1359-1360 要求 shutdown 兜底；abort 触发点映射 `[设计推导]`）→ DE 本地闭合。
- C2 DECIDING（DE abort）：PE 模型请求可能已在途中/已到达 → PE 决策、commit 发到无此请求的 DE → `STALE_EVENT` 丢弃（设计 §19.1 L2437）；PE 本地 commit 后声明 tokens 进 WAITING 等 Reverse —— **永不到来，无超时 → D-2/D-3**。
- C3 Store load 中（DE abort）：WAITING 默认 delay_free（U:scheduler.py:2144-2147）→ §13.2 drain 后发布 terminal → 释放；PE 孤儿同 C2。
- C4/C5 Reverse/Forward 中（任一侧 abort）：本地 fence 取消未启动子操作、drain 在途（设计 §13.5）；对端感知缺失 → D-2。
- C6 PE abort（proxy/客户端取消 PE 请求）：PE 本地闭合；DE 孤儿。
- 另：§15.4 L2136-2137「取消未启动 plan、清理 fence」未明确要求仍发布一次 receive terminal——delay_free 的释放依赖 terminal 到达（U:scheduler.py:2580-2586），不发布即泄漏 → D-9。
- 结论：**单引擎侧设计闭合（以 D-9 修正为前提）；跨引擎取消传播缺失 → D-2**。

### S14. shutdown drain

- 小时序：connector `shutdown`：先拒新请求 → 有界 drain（control transport 设计 §8.7 L782-783；数据面 §9.3 L1008-1009）→ 未知请求转 quarantine（设计 §20 L2474）→ quarantine 解除条件「source 死亡/session 失效/receiver terminal」（设计 §10.3 L1571-1573）。
- 现状组件无任何 drain（daemon 线程 + 基类 no-op shutdown，MLC 侧 scratch-03 §8；Store 侧 scratch-04 Q8）——设计要求为全新实现 **纯设计假设**；引擎退出时在途请求 terminal 永不发布，但进程消亡使框架语义 moot。
- 结论：**设计自洽；实现量为全新**。

### S15. DE Store 写 `[L_DE,K_DE)` 与 Forward 写 `[K_DE,R)` 是否可能物理并发写

- token 区间不相交（设计 §14.3 L2036-2037）；`K_DE` 经 `G = lcm(lcm_block_size, family granularity)` 对齐（pool_scheduler.py:404-416），`L_DE` 为 prefix cache 命中必 block 对齐 → 两区间 block 集合不相交（同一 block id 在不同 layer 是不同物理内存，但这里连 block id 区间都不相交）。
- 时序：Forward 启动晚于「REVERSE_DONE + PE compute」，而 Reverse 启动晚于「Store 整段完成」（设计 §14.3 L2031）→ 已提交路径上两写者时间也不重叠。
- 结论：**无物理并发写，逐层 fence 在此闭合**（正向结论 P-3）。注脚：Stage 1 pcp=dcp=1（设计 §17 L2347）时 G 退化为 block lcm；若 family granularity 引入非 block 对齐需再验证 `[源码可达，尚未运行验证]`。

### S16. REVERSE_DONE 不作为 DE 成功谓词的传递性论证

- 链：`FORWARD_LAYER_DONE(L)` ← PE 发送层 L ← PE 计算层 L ← PE 离开 WAITING ← PE-local finished_recving ← 完整 REVERSE_DONE（Stage 1 全 barrier，设计 §11.2 L1618-1636）。故 `FORWARD_DONE` 传递性蕴含 REVERSE 完成（设计 §7.1 L408-417）。
- 实现机制：**框架保证的 scheduler 门禁**（promote 只在聚合 finished_recving 后，U:scheduler.py:2526-2541）+ PE DualPath get_finished 的发布条件（新代码，纯设计假设）；**不依赖** `wait_for_layer_load`（父类 no-op，MLC:1976-1977，F14）。
- 前提：(a) 除 DualPath 外无其他 child 提前发布该 PE id——非 winner AscendStore 的 `loading_req_ids` 不含该请求（pool_scheduler.py:617-618 要求 num_external>0），不会发布；(b) 聚合需全 rank（F12）；(c) PE 不在 REVERSE_DONE 前因其他原因 promote——无此路径。
- 结论：**论证成立**（正向结论 P-1）。

### S17. DE 侧 Reverse send task 如何进入共享 runtime 的 send thread（特别追问）

- 现有代码无任何入口：kv_both `start_load_kv` 只走 consumer 分支（MLC:1606-1615，F6）；`save_kv_layer` 仅 forward 中按层触发（MLC:1699/1836），DE 对 Store 加载的 prefix 层无计算、无 hook；`build_connector_meta` kv_both 恒走 consumer 分支（MLC:1006-1022，F5）。
- 设计答案：新 `ReverseLayerwiseTransfer.submit_layer_send()` 直接构造 SendTask 入队（设计 §10.2 L1446-1454、L1501-1512；§9.2 L905-908 明确要求完全覆盖、不落回父类 dispatch）——**纯设计承诺，非已有能力**；须自带父 producer 分支被跳过的 mapping 计算（MLC:1615-1688）并满足 `trans_count`/`chunk_finish` 协议（MLC:589-601,1935），且继承 F15 与 `wait_event` 语义问题（D-7/D-8）。
- 结论：**机制可行但未闭合于当前代码**——关键前置 D-1（PE manifest 通道）+ D-7。

## 缺陷候选

### D-1：Reverse 目标（PE）block manifest 无 PE→DE 传输通道

- 严重级别建议：**P1**
- 类型：设计缺口（决策协议 / 数据面衔接）
- 证据等级：`[设计推导]` + `[事实冲突]`（消息集合与数据需求矛盾）
- 设计位置：§8.1 L489-497（`PathDecision` 无 block 字段）、L543-545（`PathDecisionCommit` 仅含 decision）；§8.7 L790-795（消息类型仅 COVERAGE_PROPOSAL/PATH_DECISION_COMMIT/DECISION_ACK/CONTROL_ERROR）；§10.1 L1392-1410（`reverse_rank_mappings` 需 remote=PE blocks）；§8.2 L553-577（协议顺序）
- 源码位置：对照现有通道 MLC:957-972（接收方 blocks 随 dispatch params 送达发送方，仅 D→P 方向存在）
- 触发前提：任何 DE_PARTIAL_READ。
- 具体时序：T6 PE commit（此刻 PE blocks 尚未分配——T7 才 `allocate_slots`）→ T8 commit/ack 往返 → T11 DE 构造 Reverse SendTask 需要 `reverse_rank_mappings.remote_block_ids` = PE 正式 blocks。PE→DE 方向在 PE 分配之后没有任何已定义消息可携带该 manifest：proposal 是 DE→PE；commit/ack 均不含 blocks；`kv_transfer_params` 反向回传（现有 MLC 的 do_remote_decode 回参机制）在 DualPath 完全覆盖 `update_state_after_alloc` 后无对应物。
- 为什么不正确：`DualPathTransferPlan.reverse_rank_mappings`（设计 §10.1 L1409）在 DATA_START 前必须冻结（§10.1 L1427「block IDs 在计划冻结后不可重映射」），但其关键输入在已定义的协议消息中不存在生产者。
- 可能影响：Reverse 无法构造 SendTask，整条 DE_PARTIAL_READ 不可实现；或实现被迫私下扩展协议（绕过评审）。
- 最小修正：二选一——(a) 把 commit 发送点推迟到 PE `update_state_after_alloc` 之后并在 commit 中携带 PE block manifest（需扩展 `PathDecisionCommit`）；(b) 新增幂等 `POST_ALLOC_MANIFEST` 控制消息（复用 message_id/payload_hash 幂等规则）。同时在 §8.2 协议顺序中钉死该消息的时点（DATA_START 前置条件）。
- 置信度：高（针对 snapshot 文本）。

### D-2：跨引擎失败/取消传播缺失（CONTROL_ERROR 有类型无流程）

- 严重级别建议：**P1**
- 类型：设计缺口（失败语义 / 活性）
- 证据等级：`[设计推导]`
- 设计位置：§8.7 L794（CONTROL_ERROR 仅定义）；§13 L1745-1934、§19 L2411-2459（完成/失败语义均单引擎）；§15.4 L2117-2141（只覆盖 PE_READ sibling 失败）；§13.5 L1902-1933
- 源码位置：U:scheduler.py:2144-2147（WAITING abort 默认 delay_free，释放依赖 terminal 到达）、2250-2260（未释放请求阻止 engine 退净）；MLC 无任何取消钩子（scratch-03 §8）
- 触发前提：任一引擎失败/abort，且对端正等待其动作（DE 等 Forward；PE 等 Reverse）；或对端进程死亡。
- 具体时序：见 S9/C2——PE 失败 → PE `FINISHED_ERROR`；DE reverse 已成功 → 永等 Forward → DE hang 至客户端 abort；反之 DE 在 DECIDING/Store 阶段 abort → PE 成为孤儿请求（无自己的客户端，proxy 是否代其 abort `[外部接口待确认]`）。
- 为什么不正确：完成语义（§13）与失效流程（§19.2）都闭环在单 Engine 内；DE_PARTIAL_READ 把一次逻辑请求劈成两个 Engine 的调度生命周期，任一单点失败都必须让对端进入失效流程，设计未提供该通道（CONTROL_ERROR 是唯一现成的载体但无发送/处理流程）。
- 可能影响：对端请求长期 hang；delay_free blocks 长期占用；engine 不退净；级联资源耗尽。
- 最小修正：定义 CONTROL_ERROR/cancel 通知流程（失败/abort 方 coordinator → 对端 coordinator，幂等、按 request_key 路由），对端收到后按 §19.2 走请求级失效；并声明 proxy/metaserver 层取消传播是依赖还是非目标。
- 置信度：高。

### D-3：决策阶段失败的框架通道与 `wait_for_commit` 驱动者未指定

- 严重级别建议：**P2**
- 类型：设计缺口（失败路径衔接 / 活性）
- 证据等级：`[设计推导]`
- 设计位置：§8.7 L776-780（`wait_for_commit` 无调用者）、L811（DECISION_TIMEOUT 不自行切路）；§19.1 L2419-2420；§8.2 L571-572
- 源码位置：框架只认 Worker 侧 invalid blocks 与 finished（mixin:102-105 → U:scheduler.py:1578-1586,2559-2586）；metaserver POST 失败仅重试 3 次（MLC:1089-1100）
- 触发前提：dispatch 失败（POST 重试耗尽）、commit 丢失/超时、协议错误。
- 具体时序：DE 请求已进 `WAITING_FOR_REMOTE_KVS`（T4）；失败在 **Scheduler 进程**的控制面被判定（DECISION_TIMEOUT/PROTOCOL_ERROR）；此刻无任何 Worker I/O 发生过 → 没有 Worker 会发布 invalid blocks 或 terminal → 请求滞留 WAITING，delay_free blocks 悬置。
- 为什么不正确：§19.2 的失效流程从「发布 invalid block IDs」开始，其生产者是 Worker；决策期失败发生在 Scheduler 进程，设计未描述 Scheduler→Worker 的失败注入路径（例如经 `build_connector_meta` 下发 fail-plan 命令）。
- 可能影响：决策期失败 = 请求 hang（与 D-2 不同：这里连失败方自己都闭环不了）。
- 最小修正：明确（a）`wait_for_commit`/超时的驱动线程（coordinator 后台线程或 build_connector_meta 轮询）；(b) 决策期失败经 metadata 向各 rank Worker 下发失效命令，由 Worker 按 §19.2 发布 invalid + 恰一次 terminal。
- 置信度：中高。备注：请求级 transfer timeout 整体缺失（S8），可一并在此修正。

### D-4：PE sibling AscendStore 为 layerwise 配置时，非 winner 仍可能发起 Store GET 与 Reverse 写竞争

- 严重级别建议：**P2**
- 类型：设计约束缺失（配置准入 / 数据竞争）
- 证据等级：`[当前源码已确认]`（can_load 置位）+ `[源码可达，尚未运行验证]`（layerwise 下游是否真发起 load）
- 设计位置：§15.4 L2123-2124（承诺「不能获得该请求的 Store load plan」但无机制）；§22 L2521（门槛未含 sibling 配置约束）；§9.7 L1204（use_layerwise=False 仅约束 DualPath 内部 adapter view）
- 源码位置：pool_scheduler.py:589-593（`num_external==0` 且 `use_layerwise and kvpool_cached>0` → `can_load=True`，已复核原文）；AMC:41（非 winner Store child 收 empty+0）
- 触发前提：PE 侧 sibling `AscendStoreConnector` 配置 `use_layerwise=True`，且 PE Store 命中 `S_PE > L_PE`（lookup 时已建 LoadSpec）。
- 具体时序：T7 `update_state_after_alloc(empty, 0)` → LoadSpec `can_load=True` → T15 promote 后 `_process_new_request` pop（pool_scheduler.py:690）生成带 load 任务的 ReqMeta → Worker layerwise recv 线程 Store GET 写 PE blocks（区间约 `[L_PE,S_PE)`）——与 T12-13 Reverse 已写/在写的 `[L_PE,K_DE)` 重叠；两个来源的 KV（Store 存档 vs DE 反转）可能逐位不同。
- 为什么不正确：§15.4 的不授权承诺依赖「hook 读已提交 decision」，但 AscendStore 是独立 sibling，不读 DualPath 的 decision；§3.1 禁止修改它，故唯一可行约束是配置准入，而设计未声明。
- 可能影响：Reverse 数据被覆盖/层间脏读 → 静默错误 KV（比请求失败更糟）。
- 最小修正：§22 实施门槛增加 validator：A″ 下 PE sibling AscendStore 必须 `use_layerwise=False`（并 fail fast）；或在文档中明确该配置组合非法。
- 置信度：中（can_load 置位已确认；layerwise load 任务是否真的对 can_load=True 的非 winner 请求执行，未逐行追踪到底）。

### D-5：DE Store adapter 继承 F11 异步失败上报断裂——失败 load 会被消费成 STORE_DONE

- 严重级别建议：**P1**
- 类型：复用组件缺陷继承（静默数据损坏）
- 证据等级：`[当前源码已确认]`
- 设计位置：§9.7 L1334-1347（派生 config 强制 `load_async=True`——正是断裂路径；要求「Store failed blocks 统一进入 DualPath request-level invalidation」）；§3.1 L73-80 / §3.3 L108-119（禁止修改既有组件）
- 源码位置：pool_worker.py:457-465（创建 `KVCacheStoreRecvingThread` 未注入 invalid 集合/锁）；kv_transfer.py:908-910,924-926（失败块进线程私有集合，无 drain 方）、:942（请求无条件 `set_finished_request`）；契约 U:base.py:385-389
- 触发前提：DE Store bulk load 部分或全部失败（m_store.get 返回非 0 或 None）。
- 具体时序：T10 `m_store.get` 失败 → 失败 block 进线程私有集合（无人读取）→ 请求仍被置完成 → adapter 按 loading_req_ids 交集消费为**内部 STORE_DONE** → T11-12 Reverse 把未加载/垃圾数据写进 PE → PE 用错 KV 计算 tail → T15 Forward 把基于错误前缀的结果写回 DE → 全链 terminal 正常发布。
- 为什么不正确：adapter 的消费语义（§9.7 L1339-1340「done_recving → STORE_DONE」）默认 done 即成功，但复用组件在 load_async 下「失败也 done 且 invalid 不可见」（F11）。设计提出了统一 invalidation 的目标，却没有给出在不修改既有组件前提下的失效检测机制。
- 可能影响：静默数据损坏（prefix KV 错误），且成功谓词 `STORE_DONE && FORWARD_DONE` 全部满足——最难发现的一类故障。
- 最小修正：adapter 自行 drain recv 线程的私有 invalid 集合（属性可达，kv_transfer.py:843-844），按 plan `store_target` 的 block ids 归因到请求；消费任何 done 前交叉校验该请求的 target blocks 是否出现在失败集合中；命中即按 §19.2 走请求级失效。将该机制写入 §9.7。
- 置信度：高（断裂本身已确认；绕行可行性中高）。

### D-6：失败 invalid 范围若含 DE 本地共享 prefix blocks `[0,L_DE)` 将误伤无关请求

- 严重级别建议：**P2**
- 类型：设计缺口（失效范围界定）
- 证据等级：`[设计推导]` + `[当前源码已确认]`（框架消费语义）
- 设计位置：§19.2 L2446（「收集请求相关所有 KV block IDs」）；§12 L1730（请求级失效）；§5.2 L267-268（L_DE 定义）
- 源码位置：U:scheduler.py:2691-2760（`_handle_invalid_blocks`：WAITING 请求 evict=False 但截断 `num_computed` 至首个失败块边界 :2665；扫 running 请求共享块 :2721-2725；fail 策略 prefix cache 逐出 :2736-2737；共享块处理注释 :2647-2654「仅 sync loading」）
- 触发前提：`L_DE > 0` 且 DE_PARTIAL_READ 任一必要子操作失败。
- 具体时序：T3 DE blocks 含共享 prefix 块 `[0,L_DE)`（其他 running 请求可能正引用）；失败时按 §19.2「所有相关块」发布 invalid → 框架对块 id 落在 invalid 集合的 running/waiting 请求截断/标失败；fail 策略还把块从 prefix cache 逐出。
- 为什么不正确：`[0,L_DE)` 的数据来自 DE 本地 cache，不是本次传输的产物，其「失效」没有依据；把它们纳入 invalid 会把单请求失败放大为无关请求失败/重算与 cache 抖动。
- 可能影响：无关请求 `FINISHED_ERROR` 或重算；prefix cache 误逐出。
- 最小修正：§19.2/§12 明确 invalid 集合只含本请求私有区间（`[L_DE, ⌈R⌉)` 对应块 + PE 侧 `[L_PE, ⌈K_DE⌉)` 对应块），共享命中段永不进入 invalid；`invalidate_request()` 按 OwnershipRegion 过滤 engine-local 共享段。
- 置信度：中高（框架行为已确认；设计文本未排除共享段）。

### D-7：F15 发送线程失败归因 bug 被 Reverse/Forward 复用继承

- 严重级别建议：**P2**
- 类型：复用组件缺陷继承（错误归因 → 误杀 + hang）
- 证据等级：`[当前源码已确认]`
- 设计位置：§9.2 L892-909（共享 runtime、不改父 Worker）；§3.1 L73-80；§10.3 L1567-1568（归因错误会把「成功后收到失败」放大为协议错误）
- 源码位置：MLC:507（`failed_reqs.add(req_id)` 的 `req_id` 是 :469 `for req_id, req_meta in send_task.send_request.items()` 循环残留变量，非失败 session 的请求）
- 触发前提：同一 SendTask 合并多个请求（按 session 分组，MLC:467-479）且某 session 写失败。
- 具体时序：S6——DE send thread 写 PE 失败 → 错误请求被记 failed → 错误请求 invalid/`FINISHED_ERROR`；真正失败的请求在 DE 侧永等 Forward（其 fence 无失败标记），PE 侧收到 FAILED 的请求正确失效 → 两 engine 失败集合不一致。
- 为什么不正确：失败归因是所有后续状态机（fence failed、invalid、quarantine）的输入；输入错了，§13/§19 的正确流程作用在错误的请求上。
- 可能影响：无辜请求被误杀；真正失败请求 hang；跨 engine 状态不一致。
- 最小修正：方向化 adapter 不依赖 send thread 的 `failed_reqs` 归因——按 session/wire ID 自行维护「SendTask ↔ 请求」映射（发送方知道每个 session 包含哪些 wire ID）；或以每 session 单请求的方式构造 Reverse/Forward SendTask 规避（牺牲合并收益）。可行性受 `batch_transfer_sync_write` 失败粒度影响 `[外部接口待确认]`。
- 置信度：高（bug 存在性）；中（规避方案细节）。

### D-8：Store→Reverse 数据可见性仅依赖 host 侧顺序 + `m_store.get` 同步语义

- 严重级别建议：**P2**（若 m_store.get 实为异步完成则升 P1）
- 类型：未验证假设（内存排序 / 外部接口语义）
- 证据等级：`[外部接口待确认]`
- 设计位置：§14.3 L2031（「Store load 整段完成后才开始 Reverse」）；§12 L1739（Reverse layer read acquire 时点）
- 源码位置：kv_transfer.py:846-902（`_handle_request` 一次 bulk get 后 `set_finished_request`）；MLC:481-492（send thread 传输前只 `wait_event.synchronize()`/`resharding_stream.synchronize()`）
- 触发前提：`m_store.get` 返回时底层 DMA 尚未物理完成（如在某个 stream 上异步），或 store DMA 与 Mooncake RDMA 读之间无 device 级排序保证。
- 具体时序：T10 置完成 → T11-12 send thread 的 `wait_event` 若是新 record 的事件会立即通过 → RDMA 读到的可能是旧数据。
- 为什么不正确：设计的「整段完成后」是 host 线程观察语义；RDMA 是绕过 CPU 的设备读，需要 device/HBM 级别的完成保证。
- 可能影响：Reverse 发送陈旧前缀 KV（静默错误）。
- 最小修正：设计补充可见性论证：确认 `m_store.get` 为 host-synchronous（同步返回即 HBM 写完成）——若是，写明该契约依赖；若否，在 STORE_DONE 前插入显式同步（event record/synchronize 于 store 完成 stream），并把它作为 Reverse acquire 的前置。
- 置信度：中（大概率同步，但必须核实）。

### D-9：取消未启动 plan 时未明确要求仍发布 receive terminal → 块泄漏

- 严重级别建议：**P2**
- 类型：设计缺口（取消语义 / 资源释放）
- 证据等级：`[设计推导]` + `[当前源码已确认]`（框架释放依赖）
- 设计位置：§15.4 L2136-2137（「取消未启动 plan、清理 fence」）；对照 §13.2 L1787-1799（terminal 发布规则）
- 源码位置：U:scheduler.py:2144-2147（WAITING abort → delay_free=True）、2580-2586（释放依赖 finished_recving 到达）；U:base.py:385-389（失败也必须经 get_finished 上报）
- 触发前提：请求在 WAITING 期被 abort/失败，且 DualPath plan 尚未启动任何子操作（无 invalid、无 terminal 可发）。
- 具体时序：C2/C3——fence 被清理但无任何 terminal 发布 → Scheduler 侧 delay_free 持有的 blocks 等不到释放事件 → 泄漏；`self.requests` 不删 → engine 不退净。
- 为什么不正确：§15.4 的取消动作清单漏掉了 upstream 契约要求的「即使失败/取消也必须恰好发布一次 receive terminal」；§13.2 的规则覆盖失败场景但 §15.4 的取消分支没有回指。
- 最小修正：§15.4 point 5 增补：取消未启动 plan 必须按 §13.2 发布一次（失败的）receive terminal 后再清理 fence。
- 置信度：中高。

### D-10：§7.3 转换清单「超过」未涵盖 `L_PE == K_DE`

- 严重级别建议：**P3**
- 类型：文档不一致（措辞）
- 证据等级：`[事实冲突]`（文本间）
- 设计位置：§7.3 L468（「PE 已有 prefix 超过 DE ready prefix」）vs §7.3 L455 与 §5.2 L306（严格 `L_PE < K_DE`）
- 触发前提：`L_PE == K_DE`。
- 具体时序/为什么不正确：等号情形准入失败（否则 S4 的 winner 旁落 + 脏读），但 L468 的「超过」字面不含等号，读者可能误判等号可走 partial。
- 可能影响：实现/评审误读。
- 最小修正：L468 改为「达到或超过」。
- 置信度：高（低 severity；§5.2 L306 + §7.2 rule 3 已兜住语义）。

### D-11：`decide_on_pe` 的触发点未钉死，若早于真实 L_PE 可得时刻则准入失效

- 严重级别建议：**P2**
- 类型：设计缺口（协议实现约束）
- 证据等级：`[设计推导]`
- 设计位置：§8.6 L697-703（`decide_on_pe(proposal, pe_local_tokens, ...)`「仅在 PE 调用」）；§8.2 L562-563（「PE 此时可得到真实 L_PE」——「此时」指派发后，未钉死为 scheduler 查询时刻）；§8.5 L662
- 源码位置：真实 L_PE 唯一可信来源是 waiting 循环中 `get_num_new_matched_tokens(request, num_new_local_computed_tokens)` 的第二参数（U:scheduler.py:774-779，已复核原文；本地 lookup :761-763 在其前）
- 触发前提：实现把 decide 放在 proposal 到达时（如 `on_new_request`/control 接收线程），此刻 L_PE 未知（按 0 处理）。
- 具体时序：真实 `L_PE ≥ K_DE`，但 decide 按 `L_PE=0` 通过准入并 commit DE_PARTIAL_READ → T6 实际调用时返回 `max(K_DE-L_PE,0)=0` → winner 旁落 → S4 的脏读/错账序列。
- 为什么不正确：准入条件 `L_PE < K_DE` 的有效性完全取决于 L_PE 的取值时刻；设计未禁止过早 decide。
- 最小修正：§8.2/§8.6 明确 decide_on_pe 只能在获得框架传入的真实 `num_computed_tokens` 之后执行（即首次 `get_num_new_matched_tokens` 调用内或之后），此前一律 `(None, False)`。
- 置信度：中（设计文本未明确；属于实现陷阱而非文本自相矛盾）。

## 正向结论（为什么正确、依赖前提、对应源码契约）

- **P-1（Stage 1 全 REVERSE_DONE barrier 设计正确）**：PE 收齐完整 REVERSE_DONE 才发布 finished_recving，与 upstream promote/prefix-cache 语义精确兼容——promote 后 blocks 进 prefix cache（U:scheduler.py:2517），必须全量就绪；成功谓词 `STORE_DONE && FORWARD_DONE` 的传递性论证成立（S16）。依赖前提：仅 DualPath 发布该 PE id（非 winner AscendStore 的 loading_req_ids 不含此请求，pool_scheduler.py:617-618）；聚合全 rank（U:kv_connector/utils.py:78-90）；不依赖 no-op 的 `wait_for_layer_load`（MLC:1976-1977，F14——「防御性检查」措辞准确）。
- **P-2（早到 terminal 处理正确）**：pending raw terminal inbox + mapping 后重归属（设计 §8.2 L576-577、§10.3 L1557-1561）正确修复父类丢弃缺陷（MLC:1407 过滤 + get_and_clear 清空，F8）；Store 侧更有天然暂存（按 loading_req_ids 交集取完成，未命中保留，kv_transfer.py:321-337）。依赖前提：inbox 必须按 (wire_id, direction, tp_rank) 键控并有 tombstone/保留窗（设计 §6 L378-384 已含）。
- **P-3（无物理并发写）**：Store 写 `[L_DE,K_DE)` 与 Forward 写 `[K_DE,R)` block 区间不相交且时序串行（S15）；Reverse 在 PE 只写 `[L_PE,K_DE)`，与 PE compute 写 `[K_DE,R)` 同样不相交且被 REVERSE_DONE barrier 串行化。对应源码契约：`K_DE` 经 granularity 对齐（pool_scheduler.py:404-416）、async load 不支持 block 共享（U:scheduler.py:2652-2653）。
- **P-4（Reverse 写区间选择正确）**：§10.1 L1421「只写缺失的 `[L_PE,K_DE)`、manifest 带全 `[0,K_DE)`」既避免覆写 PE 共享 prefix 块，又兼容现有 ReqMeta/SendTask 对连续区间的构造（对照 MLC:1048-1084 的 local_transed_tokens 切片语义）；与 §5.2 L308-309「Reverse prefix 是 `[0,K_DE)`」并读时后者指 DE 源读范围，两处不矛盾但需在文档中交叉引用以免误实现。
- **P-5（(None,False) 等待在 PE Multi 下无侧漏）**：child 0 返 None 使 AscendStore 本轮不被调用（U:multi_connector.py:393-394）→ 无临时 lookup 副产物；F3 残留前提（先写 winner 后 None）在本路径不成立；F16 语义退化为有界调度延迟（S11）。
- **P-6（框架顺序支点真实存在）**：单步内 bind→start_load_kv→forward→get_finished（mixin:89-105）；invalid 先于 finished 消费（U:scheduler.py:1578-1586 先于 1838）；abort 也触发 `request_finished`（U:scheduler.py:2162）；WAITING 默认 delay_free（:2144-2147）；finished 到达时不变式由 assert 强制（:2576/2581/2585）——设计的 §13.2 原子终态规则（invalid 不晚于失败 terminal、同 ID 恰一次）与这些 assert 方向一致，可落地。
- **P-7（token accounting 硬契约与乐观 num_computed 语义自洽）**：DE 声明 `E_DE` → 乐观 `num_computed=R` → promote 后 DE 算位置 R（`R=P-1<P` 不触发 full-hit 调整，U:scheduler.py:2519-2522）；PE 声明 `K_DE-L_PE` → 乐观 `=K_DE` → promote 后算 `[K_DE,R)`；两 Engine 各只见一次本 Engine 增量（设计 §15.3 L2115），无重叠无缺口。
- **P-8（端口/线程拓扑可满足）**：§18.2「PE=Reverse receiver port、DE=Forward receiver port」使每 Engine 的单一 recv thread 只服务一个方向，与现有 `KVCacheRecvingLayerThread` 多消息类型 ROUTER（MLC:603-653）兼容；kv_both 一次 `register_kv_caches` 完成 buffer+send+recv（MLC:1359-1398，F7）支撑共享 runtime 前提。

## 残留待确认（非缺陷，需运行/外部文档）

- `m_store.get` 完成语义（D-8 的裁决依据）。
- `batch_transfer_sync_write` 失败返回的粒度（D-7 规避方案依据）。
- proxy/metaserver 是否传播客户端取消到被派发请求（D-2 的边界）。
- layerwise 下非 winner `can_load=True` 是否真产生 Worker GET（D-4 的置信度升级依据）。

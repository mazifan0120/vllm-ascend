# Phase 2 时序审查 — 路径 DE_FULL_HIT

- 审查基线：设计 snapshot `inputs/option-a-detailed-design.snapshot.md`（行号以此为准）；Phase 1 证据 `evidence/01-current-control-flow.md` 及 `phase1-scratch/01`~`06`；补查源码 vllm-ascend @ 0ec11a47、upstream vllm @ 8df14cfc（v0.23.1rc0-1050，配套 v0.23.0 有小幅偏差，涉及 upstream 处均受此影响）。
- 证据标签：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / [设计推导] / `[外部接口待确认]` / [事实冲突]。顺序性质另注明：**框架保证** / **当前实现偶然顺序** / **纯设计假设**。
- 本路径定义（snapshot §14.2 L1971-2000、§7.1 L402-406）：DE 从 AscendStore 获得足够 prompt KV（`K_DE >= R`），本地重算最后一个 prompt token，直接 Decode；不创建 PE 模型请求、无 Reverse/Forward；唯一成功 barrier 是 `STORE_DONE`；决策经 control-only proposal 走独立 ZMQ control transport（§8.7 L731-817）。

## 主链结论速答（时序必答问题）

| 问题 | 答案 | 依据 |
|---|---|---|
| 谁做 probe | DE Scheduler 进程、调度循环线程内：`DualPathConnectorScheduler.get_num_new_matched_tokens` → `DualPathStoreSchedulerAdapter.probe` → `KVPoolScheduler.get_num_new_matched_tokens`（非 layerwise → zmq REQ RPC 到 DE worker rank-0 的 LookupKeyServer，**阻塞 recv 无超时**） | snapshot L555-559, L1236-1245；pool_scheduler.py:478-565, 1106-1108 `[当前源码已确认]` |
| 谁决定路径 | PE Scheduler 进程内 control transport 接收线程上的 `PathDecisionCoordinator`（静态规则 + capability digest + DE block manifest digest）。DE 只按自身 coverage **预测**路径并选择通道（K_DE>=R → control-only），最终判定权在 PE | snapshot L445-450, L553-572, L660-668 [设计推导] |
| 哪一刻算 committed | PE `coordinator.commit()` CAS `DECIDING→COMMITTED` 成功；数据面启动还要求 `PathDecisionCommit + DecisionAck + frozen block plan` 三者齐备 | snapshot L568-572, L705-709 [设计推导] |
| 每个 Engine 声明多少 external tokens | DE 声明 `E_DE = max(R - L_DE, 0)`（路径无关统一总量），`load_async = E_DE > 0`；DE_FULL_HIT 不创建 PE 模型请求，PE 不声明任何 token | snapshot L281-293, L2086-2106；§4.4 L214-228 |
| 谁分配正式 blocks | DE upstream Scheduler（`allocate_slots(num_external=E_DE, delay_cache_blocks=True)`），路径无关、commit 前完成；DualPath 只冻结 manifest | U:scheduler.py:942-954 `[当前源码已确认]`；snapshot L470-474, L560-561 |
| 各 token 区间谁写谁读 | `[0, L_DE)`：DE 本地 prefix cache（既有，Store 经 mask 跳过不写）；`[L_DE, K_DE)`：Store bulk DMA 写（各 TP rank 写本 rank shard），DE attention 读；最后一个 token `[R, P)`：DE 本地 forward 重算写 | snapshot L1418-1420；kv_transfer.py:858-862（mask_num 跳过本地段）`[当前源码已确认]`；§4.4 |
| 哪些操作可能并发 | ① 双 Engine 的 control transport 线程 vs 各自 scheduler 循环线程（决策/提交应用 vs build_connector_meta）；② 各 TP rank 的 Store DMA daemon 线程彼此并发、与模型 forward（其他请求）并发；③ abort 处理（EngineCore 主循环 execute_model 与 update_from_output 之间）与 control 线程的 commit 应用并发 | ①② 见 D2；③ U:core.py:510-512 `[当前源码已确认]` |
| 哪个事件使请求离开等待状态 | 聚合后的 `finished_recving`（全 TP rank 上报同一 DE-local ID）→ `finished_recving_kv_req_ids` → 下一 schedule 的 `_try_promote_blocked_waiting_request` | U:scheduler.py:2526-2541, 2574-2579 `[当前源码已确认，框架保证]` |
| 谁发布 finished_recving / finished_sending | DE 各 Worker rank：`DualPathConnector(Worker).get_finished` 在 `STORE_DONE` 且无 in-flight writer 后发布一次 DE-local ID 进公共 `finished_recving`；`finished_sending` 仅用于请求结束后的 delayed-free 释放（本路径计划含 async Store writer，`SchedulerReleaseLedger` 保守 delay_free） | snapshot L1098-1112, L1343, L1824-1899 [设计推导]；聚合机制 F12 `[当前源码已确认]` |
| 多 rank 如何汇聚成功/失败 | 框架 `KVOutputAggregator`：finished 需全部期望 rank 上报同一 ID（计数制，world_size==TP size）；invalid_block_ids 任一 rank 上报即并集发布、无 quorum | U:kv_connector/utils.py:78-90, 159 `[当前源码已确认]`（F12） |
| 取消/超时/部分失败后谁 drain/quarantine/释放 | Worker 侧 `BlockOwnershipLedger` drain/quarantine；invalid 先行、failed receive terminal 在 ownership 归零且无 unknown in-flight 后发布；Scheduler 在 fail 策略下 `FINISHED_ERROR` 并在 terminal 到达后 `_free_blocks`。**但 pre-commit 取消与 DECISION_TIMEOUT 的 worker 传播链路未闭合（见 D1/D8）** | snapshot L1764-1804, L1902-1933, L2440-2456；U:scheduler.py:1578-1586, 2144-2147, 2574-2586 |

---

## T 时序主链

执行者速记：DES=DE Scheduler 进程（循环线程）；DECT=DE Scheduler 进程内 control transport 线程（DEALER）；PECT=PE Scheduler 进程内 control transport 线程（ROUTER）；DEW[n]=DE 第 n 个 Worker 进程（model-runner 线程）；DERT[n]=DEW[n] 内 KVCacheStoreRecvingThread（daemon）；LKS=DEW[0] 内 LookupKeyServer（daemon）；FW=upstream 框架。

- **T0｜请求到达 DE**（FW，DE EngineCore 进程）。请求带远程 prefill 语义进入 waiting 队列。[当前源码已确认]
- **T1｜本地前缀查询**（DES，调度循环）。`num_computed_tokens==0` → `find_longest_cache_hit` 得 `L_DE`。[框架保证顺序；U:scheduler.py:724-763 `[当前源码已确认]`]
- **T2｜E_DE 计算**（DES→DualPathConnectorScheduler）。全 scheduler 唯一 connector 查询点：`get_num_new_matched_tokens(request, L_DE)`。`R=P-1`，`E_DE=R-L_DE`；`E_DE==0` → 返回 `(0,False)`，纯本地路径终止（此时 `L_DE>=R`，等价 full-hit 已由本地 cache 满足）。[U:scheduler.py:774-779 `[当前源码已确认]`；计算规则为设计推导 snapshot L281, L555-556]
- **T3｜Store probe**（DES 同一线程，同步）。adapter.probe → `KVPoolScheduler.get_num_new_matched_tokens`：`token_len=floor(P/G)*G`（`token_len<G` 直接 miss）→ 惰性建 `LookupKeyClient` → zmq REQ → LKS → `m_store.exists` → 命中 tokens；`hit==P` 时减 1；`hit>L_DE` 时创建 `LoadSpec(vllm_cached=L_DE, kvpool_cached=hit)`。**recv() 无超时，DES 循环被阻塞至 LKS 应答**。[当前源码已确认 pool_scheduler.py:502-527, 549-553, 1106-1108；LKS 存在性 ascend_store_connector.py:119-120]（设计声称 probe 非纯——属实且更强，F10）
- **T4｜pending candidate 登记**（DES）。DualPath 记录 `DualPathRequestKey{de_incarnation, de_local_req_id}` + `StoreCoverage`，返回 `(E_DE, True)`。[纯设计假设 snapshot L557-559]
- **T5｜正式 blocks 分配与冻结**（DES/FW）。`allocate_slots(num_external_computed_tokens=E_DE, delay_cache_blocks=True, reserved_blocks=_inflight_prefill_reserved_blocks())` → 覆盖 `[0,R)` 的全部正式 blocks 就位但**不进 prefix cache**；`update_state_after_alloc#1(request, 全部 blocks, E_DE)` → DualPath 冻结 block manifest（按 `request_key+decision_version` 幂等，仅 `num_external>0` 时冻结）；`status=WAITING_FOR_REMOTE_KVS`；`num_computed_tokens=R`（乐观值）；入 `_inflight_prefills`。**不启动任何 I/O**。[框架保证 U:scheduler.py:942-1006 `[当前源码已确认]`；冻结语义为设计推导 snapshot L560-561, L681-684]
- **T6｜control-only proposal 发送**（DES 或 DECT——**设计未指定发送线程**，见 D2）。DE 依据自身 coverage 预测 full-hit（K_DE>=R），经 DEALER 发 `CoverageProposal{ids, de_coverage, capability_digest, de_block_manifest_digest, requested_policy}`；不发 PE 模型请求。[纯设计假设 snapshot L562-568, L813-814]
- **T7｜PE 判定与提交**（PECT，PE Scheduler 进程 control 线程）。`register_candidate`（幂等）→ `decide_on_pe`：静态规则 1「已验证 coverage 满足 full hit → DE_FULL_HIT」+ 硬准入（lookup 成功、coverage 已对齐、`K_DE>=R`、capability 已确认）→ `commit()` CAS `DECIDING→COMMITTED` → 发 `PathDecisionCommit`。**PE 调度循环、PE Multi/first-winner、PE Worker 全程不参与；L_PE 不查询（见场景 S5）**。[纯设计假设 snapshot L438, L445-450, L660-668, L686-729]
- **T8｜commit 应用与 Store 计划冻结**（DECT——按 §8.5 顺序图的自然读法，见 D2）。DE 收 commit → coordinator 持久化 → 回 `DecisionAck` → `commit_after_alloc(handle, request, blocks, H_DE)` → `KVPoolScheduler`：`LoadSpec.can_load=True`、`_loading_req_ids.add(req)`（`num_external==kvpool_cached-vllm_cached` 断言，见 D3）。[设计推导 snapshot L667-673, L1250-1257；KVPool 机制 `[当前源码已确认]` pool_scheduler.py:604-618]
- **T9｜plan+store metadata 发射**（DES，下一个 schedule 的 `build_connector_meta`）。产出 `DualPathConnectorMetadata{plans, store_metadata=AscendConnectorMetadata}`；`_process_async_load_request` 弹 LoadSpec 生成 ReqMeta（`token_len=kvpool_cached`，`==P-1` 且非块对齐时 +1）；随 SchedulerOutput 经 ZMQ 广播到全部 DEW。ReqMeta 只发一次（LoadSpec pop）。[KVPool 机制 `[当前源码已确认]` pool_scheduler.py:839-885, 951-957；框架通道 U:multiproc_executor.py:310-320 `[当前源码已确认，框架保证]`；plan 内容为设计推导 snapshot L1123-1174]
- **T10｜Worker 绑定与启动 load**（DEW[n] model-runner 线程，forward 前）。`bind_connector_metadata` → `register_request(plan)`（fence、`STORE_WRITER` ownership acquire）→ adapter `start_load_kv` → `KVPoolWorker.start_load_kv` → `kv_recv_thread.add_request`。[顺序 bind→start_load 为框架保证 U:mixin:89-95；KVPool 机制 `[当前源码已确认]` pool_worker.py:745-878；plan/fence 为设计推导]
- **T11｜Store bulk DMA**（DERT[n] daemon，与其他 rank、与本进程后续 forward 并发）。一次 bulk `m_store.get`，Store→HBM 直接写正式 blocks；`mask_num=floor(L_DE/bs)*bs` 跳过 `[0,L_DE)`；部分失败块入**线程私有** `_invalid_block_ids`（F11 断裂点）；**无论成败** `set_finished_request(req_id)`。无 DMA 拆分、无超时。[当前源码已确认 kv_transfer.py:846-943；pool_worker.py:457-465]
- **T12｜rank-local STORE_DONE → 公共 finished_recving**（DEW[n] model-runner 线程，任一 forward 后 finally 或 no-forward 空步）。`DualPathConnector(Worker).get_finished(finished_req_ids)`（完全重写、不落父类）→ adapter 调 `KVPoolWorker.get_finished(finished_req_ids, store_metadata)` → `recv_thread.get_and_clear_finished_requests(meta.loading_req_ids)` → rank-local `done_recving` 被消费为内部 `STORE_DONE` → fence.store_done=True、释放 `STORE_WRITER` → 路径为 DE_FULL_HIT 且无 in-flight writer → 公共 `finished_recving` 发布**精确 DE-local ID 一次**。同 pass：`get_block_ids_with_load_errors` drain（修复后的）invalid 集合（见 D4）。[框架调用点 U:mixin:35-48, 102-105 `[当前源码已确认，框架保证]`；KVPool 链路 pool_worker.py:1551-1555 `[当前源码已确认]`；内部消费与发布规则为设计推导 snapshot L1332-1347, L1787-1799]
- **T13｜跨 rank 聚合**（FW，EngineCore 进程）。`KVOutputAggregator` 对 DE-local ID 倒计数，**全部 TP rank 上报同一 ID**才对外发布；invalid 并集无 quorum。[当前源码已确认 U:kv_connector/utils.py:78-90, 159]（F12）
- **T14｜Scheduler 消费终态**（DES）。`update_from_output`：invalid 先于 finished 处理；`finished_recving` → `assert req in self.requests` → 状态==WAITING → 入 `finished_recving_kv_req_ids`。[框架保证 U:scheduler.py:1578-1586, 2574-2579 `[当前源码已确认]`；不变式由 assert 强制]
- **T15｜promote**（DES，下一 schedule）。`_try_promote_blocked_waiting_request` → `_update_waiting_for_remote_kv`：`cache_blocks(request, R)`（blocks 此刻才进 prefix cache）；`num_computed_tokens==R==P-1≠P` → **不触发** `==num_tokens` 的 -1 分支；状态回 WAITING。[框架保证 U:scheduler.py:2492-2541 `[当前源码已确认]`；与 §4.4 完全咬合，见场景 S8]
- **T16｜末 token 重算**（DES→DEW[n]）。请求按普通路径调度（else 分支，`num_computed_tokens>0`）：`update_state_after_alloc#2(num_external=0)` → 幂等 no-op；`num_new_tokens=P-(P-1)=1` → forward 以 Store 加载的 `[0,R)` 为上下文重算最后一个 prompt token、采样首个 decode token；该 token 的 KV 写入其 block（若 T11 的 +1 边界已写过该 slot，此处覆盖写，顺序由 STORE_DONE barrier 保证）。[框架保证 U:scheduler.py:822-847 `[当前源码已确认]`；+1 边界 pool_scheduler.py:849-852 `[当前源码已确认]`]
- **T17｜Decode 与最终释放**（DES/DEW[n]）。Decode 正常进行；请求结束 → `_connector_finished` → `request_finished_all_groups` → `SchedulerReleaseLedger`：plan 含 async Store writer → 保守 `delay_free=True` → 各 Worker 在 ownership=0 且无 unknown in-flight 后发布 `finished_sending` → 全 rank 聚合 → `_free_blocks`。[设计推导 snapshot L1843-1897；框架机制 U:scheduler.py:2162, 2583-2586 `[当前源码已确认，框架保证]`]

主链评价：**happy path 在机制上闭合**——每一环都有当前代码载体（probe/blocking RPC、异步 load metadata 每步发射、loading_req_ids 跨步存续、get_finished 每步必调、聚合器计数、promote/末 token 重算），且 `num_computed_tokens=R` 与 upstream assert/promote 语义精确兼容（S8）。**闭合依赖三个纯设计假设成立**：commit 应用线程的串行化（D2）、F11 失败检测修复（D4）、pre-commit 取消的终态发布（D1）；三者任一不成立，主链在对应异常窗口断裂。

---

## 场景矩阵

### S1：`S_DE_raw` 对齐到 G 后 `K_DE>=R` 的判定时序
小时序：T3 probe 返回 hit → T4 DE 计算 `S_DE=min(floor(S_DE_raw/G)*G, R)`、`K_DE=min(max(L_DE,S_DE),R)` → T6 按 `K_DE>=R` 预测并选 control-only 通道 → T7 PE 复核同一硬准入（§7.3 L447-449）→ commit。
关键事实：probe 的 `S_DE_raw` 若来自 `KVPoolScheduler.get_num_new_matched_tokens`，查询长度已被 floor 到 G（pool_scheduler.py:502-503），且 `hit==P` 时被减 1（:526-527）——即 full hit 时 `S_DE_raw=R=P-1`；设计再 floor 一次会把 `R%G≠0` 的 full hit 打掉一个 chunk（如 P=1024,G=256：`S_DE_raw=1023→S_DE=768→K_DE=768<R`，退化为 DE_PARTIAL_READ）。仅当 probe 用**未减 1 的原始 lookup**（`client.lookup` 层级）时，`min(...,R)` clamp 才使 `P≡0或1 (mod G)` 的 full hit 正确触发。
结论：**判定时刻本身（probe 后、commit 前双侧各一次）设计上是良定的，但触发条件的数学与 §4.4 L226「Store 覆盖到 prompt_len-1 即满足」/§14.2 L1981「coverage >= prompt_len-1」的表述不一致，且与 probe 实现风味强耦合——不闭合，见 D3**。附注：物理 load 的 +1 边界（pool_scheduler.py:849-852）使实际写入可达 `[0,P)`，末 token slot 后在 T16 被重算覆盖，顺序安全（STORE_DONE barrier）。

### S2：`L_DE=0` 与 `L_DE>0`
`L_DE` 即 T1 本地 prefix cache 命中（框架在 connector 查询前完成，U:scheduler.py:760-779）。`L_DE=0`：probe 从 block 0 查询，`E_DE=R`。`L_DE>0`：probe 的 `query_start_block=L_DE//bs`（pool_scheduler.py:255）跳过本地块；load 侧 `mask_num=floor(L_DE/bs)*bs`（kv_transfer.py:858-862）使 Store 不重写 `[0,L_DE)`；`E_DE=R-L_DE`，`H_DE=S_DE-L_DE` 与 LoadSpec delta 同构。两侧 accounting 自洽；`L_DE>=R` ⇒ `E_DE=0` ⇒ T2 直接 `(0,False)` 终止，不创建 candidate（§5.2 L301-302）。
结论：**闭合**。注意 `L_DE` 天然块对齐（prefix cache 性质），与 Store 块粒度 mask 一致，无跨界写。

### S3：冻结窗口内请求被取消（commit 前 abort）
小时序：T5 后请求在 WAITING（blocks 由 `delay_cache_blocks` 隔离、`num_computed_tokens=R` 乐观）→ abort：EngineCore 在 execute_model 与 update_from_output 之间处理（U:core.py:510-512）→ `finish_requests(FINISHED_ABORTED)`：`delay_free_blocks = req_id not in finished_recving_kv_req_ids = True`（U:scheduler.py:2144-2147）→ `_connector_finished` 同步触发（:2162）→ DualPath `request_finished_all_groups`：`SchedulerReleaseLedger` **无 plan（plan 只在 commit 后注册）→ 返回 (False,None)**（snapshot L1865-1877）；框架 delay 仍持有 blocks → `finished_req_ids` 下步通知 worker。
此后三条分岔：(a) commit 永不到达且 DECISION_TIMEOUT 失败传播未实现（D8）→ **无任何发布者**，blocks 与 `self.requests` 条目永久滞留（F13 机制）；(b) 迟到的 commit 到达 → `commit_after_alloc` 写 `_loading_req_ids` → 但下一步 `build_connector_meta` 的 finished 清理（pool_scheduler.py:896-901）按 `finished_req_ids` discard `_loading_req_ids`、pop `_unfinished_requests` → **load metadata 永不发射 → 无 I/O → 无终态 → 同样泄漏**；(c) 若 worker 侧实现「任意 delayed-free ID 在 ownership=0 时发布 finished_sending」（snapshot L1109-1111 的语义边界不清），则可经 U:scheduler.py:2583-2586 释放——但 §13.4 L1829-1835 把 finished_sending 限定为「request_finished 曾返回 delay_free=True」，本场景 connector 返回的是 False，规则自相矛盾。
结论：**不闭合（D1）**。附带泄漏：pending candidate、probe handle、`load_specs[req]`（F10 已知无 abort 清理）。

### S4：control-only proposal 往返中的 timeout/retry/duplicate/乱序/迟到
- retry：同 `message_id` 重发，接收端按 `(message_id, payload_hash)` 幂等（L806-809）✓。
- duplicate commit：同 message_id 同 hash 幂等；不同 hash 协议错误（L809）✓；coordinator CAS 拒二次不同提交（L705-709, L2467-2469）✓。
- timeout：`request_timeout_ms × max_retries` 耗尽 → `DECISION_TIMEOUT` 失败、不自行切路（L810-811）。**断裂点在下游**：超时判定在 DECT（Scheduler 进程），而 invalid/终态发布在 DEW（Worker 进程）；设计未给 scheduler→worker 的失败传播元数据通道，§19.2 L2440-2452 的通用失败流程假设 plan/fence 已存在，本场景 DEW 可能连 plan 都没见过（T9 未发生）。**不闭合（D8）**。
- 乱序：本路径 PE→DE 仅 commit 一类消息、DE→PE 仅 ack/error，coordinator 状态机天然定序（commit 前无 candidate 即协议错误）✓ 基本闭合。
- 迟到 commit（DECISION_TIMEOUT 已失败之后）：`fail()` 冻结失败态（L723-728）可拒，但**决策态 tombstone 保留期未规定**（§6 L378-379 只规定 wire ID tombstone）；若状态已 GC，迟到 commit 会复活一条已失败/已释放的请求（D9）。
- 迟到 ack：PECT 侧 `acknowledge()` 无候选时的行为未规定（D9）。

### S5：control-only 模式下 PE 如何获得 `L_PE`（是否需要）
`decide_on_pe(proposal, pe_local_tokens, pe_store_coverage)`（L697-703）在 control-only 下没有 PE 模型请求，PE 调度循环不参与，**没有任何自然调用点提供 `L_PE`**；proposal 也不携带 block hashes（L532-540），PE 无法用本地 `kv_cache_manager` 反查。
结论：**Stage 1 静态规则下不需要**——规则 1「full hit → DE_FULL_HIT」（L438）只依赖 DE coverage，`PathDecision.pe_coverage` 允许为 None（L493）。设计闭合但**隐含限制未写明**：任何未来考虑 PE 本地命中以省 Store 流量的策略（如 PE 已有全前缀则改 PE_READ）在 control-only 通道下结构性不可达；建议设计显式声明「control-only 决策与 L_PE 无关」。

### S6：STORE_DONE 从 KVPoolWorker 内部 done_recving 到 DE 公共 finished_recving 的叠加链路
当前语义：`KVPoolWorker.get_finished` 返回 `(done_sending, done_recving)`，`done_recving = recv_thread.get_and_clear_finished_requests(meta.loading_req_ids)`（pool_worker.py:1551-1555）；`loading_req_ids` 在请求 finished/preempted 前跨步存续（:901, 913），故终态可在 load 完成后的任一空步被收集（mixin:35-48 保证空步也调 get_finished）。
设计叠加方式：**不叠加**——DE 侧是独立 `DualPathConnector`（§15.1 L2051-2055），其 `get_finished` 完全重写（L905-908），store 的 `done_recving` 只是 adapter 内部输入（L1339），公共集合只由 DualPath 自己的 fence 逻辑产生；父类 MLC 语义（F8：finished_sending 恒空、失败请求不进完成集、早到 terminal 丢弃）被整体替换而非继承。DE_PARTIAL_READ 不透传 store done（L1341-1342）与 DE_FULL_HIT 可透传（L1343）的分叉也在此层。
结论：**机制闭合**，前提：① adapter 每步把当前步的 `store_metadata` 传给 `KVPoolWorker.get_finished`；② `commit_after_alloc` 确实使 req 进入真实 `loading_req_ids`（L1335-1336），否则 done 被交集过滤、请求永久滞留——这正是 D2 竞态 (b) 的断点。

### S7：Store load 部分 DMA 失败（F11 断裂与 adapter 修补）
当前断裂：`KVPoolWorker` 创建 recv 线程时**未注入** invalid 集合（pool_worker.py:457-465），失败块入线程私有集合（kv_transfer.py:908-910, 924-926），`get_block_ids_with_load_errors` 只 drain worker 自有集合（pool_worker.py:1333-1337）→ **load_async 下 invalid 永不上报**，但 `set_finished_request` 无条件执行（kv_transfer.py:942）。
不修补的时序后果：T11 部分失败 → T12 adapter 仍消费到 `done_recving` → 当作 `STORE_DONE` → T13-15 promote + `cache_blocks(R)` → **脏 KV 进入 prefix cache 并服务后续请求**（静默正确性腐蚀，且污染面超出本请求）。
可修补性：recv 线程构造器**接受** `invalid_block_ids/lock` 参数（kv_transfer.py:830-831），adapter 拥有 `KVPoolWorker` 实例，可在构造后注入共享集合或直接 drain `kv_recv_thread._invalid_block_ids`；失败归因按 block_id 与冻结 manifest 求交即可（该窗口内 blocks 请求独占：`delay_cache_blocks` 未入 cache + async load 不支持块共享 U:scheduler.py:2652-2653）。代价：依赖私有属性，且必须在「不晚于 failed terminal」的 pass 内发布 invalid（§13.2 L1794 与 base.py:384-389 契约一致）。
结论：**设计方向正确（L1347「统一进入 request-level invalidation」）但修复手段依赖 KVPool 私有属性且未写明 block→request 归因要求——列为 D4 验收门槛**。多 group hybrid 的彻底静默（kv_transfer.py:911-918）由 §17 单 KV group 限制规避 ✓。

### S8：DE 本地重算末 token 的 `num_computed_tokens` 与 upstream assert 兼容性
- T5：`assert num_computed <= num_tokens`：R=P-1<P ✓（U:scheduler.py:800）。
- T14：`assert req_id in self.requests` ✓（请求在 WAITING，:2576）；`:2581` 的 finished 分支不触发 ✓。
- T15：`-1` 调整的条件是 `num_computed_tokens == num_tokens`（:2521-2522）；DualPath 声明 R=P-1，条件不成立——**不需要也不发生二次调整**，`num_computed_tokens=R` 直接正确。
- T16：`num_new_tokens = P - R = 1`（:847），调度 1 token 重算 ✓。
- 对比：若某 connector 声明全部 P tokens，框架会代做 -1。设计的 `R=P-1` 声明与 §4.4「Scheduler 保留最后一个 token」是**精确同构**的两种表达，互不冲突。
结论：**闭合，且是设计上最干净的一环**（依赖前提：E_DE 声明严格等于 R-L_DE，不得声明 P-L_DE）。

### S9：engine restart 后 wire ID / request key 复用
`DualPathRequestKey` 含 `de_engine_incarnation`（L347-350）；当前 MLC `engine_id` 缺省 uuid4、进程级（MLC:702；upstream kv_transfer.py:92-94）→ restart 天然换 incarnation；wire 事件四元组（wire ID/direction/incarnation/channel）匹配 + tombstone（L378-381）防 stale 归属 ✓。request ID 复用被 incarnation 隔离 ✓；aggregator 随进程重建 ✓。
残余开口：① PE 侧按旧 incarnation 登记的 candidate 在 DE 死亡后无过期机制（并入 D8）；② DEW[0] 重启后 LookupKeyServer 重新 bind 同一 ipc path，旧 socket 文件/`close()` 无调用方（pool_scheduler.py:1111-1112 无生产调用）——bind 是否受残留文件影响 `[外部接口待确认]`；失败后果是 probe 报错→按 §19.1 降 PE_READ，属降级非致命。
结论：**主语义闭合**（依赖 engine_id 确实跨 restart 唯一），残余项并入 D8/D9。

### S10（特别追问）：decision 最终是 PE_READ 时，已冻结 blocks 与 accounting 如何收敛
窗口内 blocks 状态：已分配给本请求、`delay_cache_blocks=True` 未进 prefix cache、不被其他请求命中、无 writer/reader（I/O 未启动）——**纯「已预留未发布」态**。
PE_READ commit 后：DE 侧 `abort_probe(handle)` 清理 LoadSpec/client 临时态（L1247-1248, L1355）；**blocks 无需收敛**——它们是路径无关的最终目标（L470-474, L1965），PE_READ 的 Forward target `[L_DE,R)` 落在同一组 blocks 上；accounting 也无需收敛——`E_DE` 路径无关、乐观 `num_computed=R` 在两条路径下语义一致，请求继续等 `FORWARD_DONE` 而非 `STORE_DONE`。
结论：**设计闭合（统一总量 + 路径无关 blocks 是 A″ 最成立的一笔）**。实现负担：`abort_probe` 要能从外部清理 `KVPoolScheduler.load_specs`（当前唯一清理点是三个 `_process_*` pop，F10）——不改 KVPool 类就只能触私有字典，脆弱但可达。
反向（decision=DE_FULL_HIT 时 PE 侧无 probe handle，L1356）✓ 无此类收敛。

### S11：多 rank 部分失败汇聚
rank k Store DMA 失败（经 D4 修复后可检出）→ rank k 该 pass 发布 invalid（并集、无 quorum、**先于** finished 被 Scheduler 消费，U:scheduler.py:1578-1586）→ fail 策略截断 + `FINISHED_ERROR`（:1823-1835）→ 请求已 finished、blocks 因 WAITING+delay_free 持有。其余 rank 的 `done_recving` 与 rank k 的 failed receive terminal 各自上报同一 DE-local ID → 计数凑满（utils.py:78-90，聚合器不区分成败）→ 对外发布 → `:2581` assert is_finished ✓ → `_free_blocks`。
结论：**闭合**，前提：每个 rank 对该 ID **恰好发布一次** receive terminal（含失败终态，§13.3 L1810-1816），且 failed terminal 不早于 drain/ownership 归零（§13.2 L1789-1793）——Store bulk get 是 recv 线程内同步调用，返回即无 in-flight，failed terminal 可紧随 invalid 发布，时序可行。

### S12：Store DMA hang（`m_store.get` 不返回）
recv 线程永久阻塞于 kv_transfer.py:902 → 无 `STORE_DONE`、无 invalid → 请求永久滞留 WAITING，blocks 被 `_inflight_prefills`+delay 语义持有。当前代码无超时；设计的 `DECISION_TIMEOUT`（L810-811）只覆盖决策阶段，§19 无 Store I/O watchdog。
结论：**不闭合（D6）**。

---

## 缺陷候选

### D1：pre-commit（冻结窗口）abort 无终态发布者，blocks 永久泄漏
- 严重级别建议：**P1**｜类型：生命周期/资源泄漏（时序缺口）｜证据等级：[设计推导] + 框架机制 `[当前源码已确认]`
- 设计位置：§8.2 L499-593（四阶段协议无 abort 语义）；§13.4 L1824-1841（finished_sending 限定条件）；§13.5 L1902-1933（只覆盖 RUNNING→terminal）；§19 L2411-2459（只覆盖失败）
- 源码位置：`U:vllm/v1/core/sched/scheduler.py:2144-2147`（WAITING abort → delay_free=True）、`:2162`（abort 同步触发 request_finished）、`:2574-2586`（释放依赖 connector 终态）、`U:vllm/v1/engine/core.py:510-512`（abort 处理时点）；phase1 F13
- 触发前提：T5（分配+WAITING）之后、commit 应用之前，客户端 abort 或 Engine 关停；该窗口 ≥ 一次 control 网络往返，是必然非空窗口。
- 具体时序：见 S3。abort → 框架 delay_free 持有 blocks → connector 无 plan 返回 (False,None) → (a) 无 commit/无超时传播：无人发布终态；(b) 迟到 commit：`commit_after_alloc` 写入的状态被下一步 `build_connector_meta` 的 finished 清理（pool_scheduler.py:896-901）抹掉 → load 永不发射 → 仍无终态。
- 为什么不正确：框架契约是「delay_free 的释放以 connector 事后上报 finished_recving/finished_sending 为前提」（U:scheduler.py:2580-2586；base.py:547-562）；设计只规定了 post-commit plan 的 drain 终态，对「从未 commit、从未启动 I/O」的 candidate 没有任何终态发布义务人。§9.5 L1109-1111「delayed-free ID 在 ownership=0 时发布 finished_sending」若按字面可实现兜底，但 §13.4 L1829-1835 又把 finished_sending 限定在「request_finished 曾返回 delay_free=True」——而本场景 `SchedulerReleaseLedger` 返回 False（L1870-1872），两条规则互相排斥，实现者无论照哪条都有据可依。
- 可能影响：每次窗口内 abort 泄漏一组正式 KV blocks + 一条 `self.requests` 条目 + `_inflight_prefills` 计数；`has_finished_requests` 使引擎永不退净（U:scheduler.py:2250-2260）；长期运行 block pool 耗尽。另泄漏 pending candidate、`load_specs[req]`（F10）。
- 最小修正建议：① 设计明确「`request_finished(_all_groups)` 对 pending candidate：coordinator.fail(request_key, ABORTED)、abort_probe(handle)、并规定由 Worker 在 ownership=0（本场景恒真）时对该 DE-local ID 发布一次 failed receive terminal 或 finished_sending（二选一，写死）」；② 统一 §9.5 与 §13.4 的 finished_sending 前置条件；③ commit 应用前先查 coordinator 是否已 terminal/aborted，是则回 CONTROL_ERROR 并拒绝 `commit_after_alloc`（同时消除 S3-b 的无效 I/O）。
- 置信度：高（机制链全部由当前源码佐证；仅「设计未规定」属文本判断）。

### D2：commit 应用（`commit_after_alloc`）执行线程未指定，与 scheduler 循环存在数据竞态
- 严重级别建议：**P1**｜类型：并发/时序（线程模型缺口）｜证据等级：[设计推导] + KVPool 无锁事实 `[当前源码已确认]`
- 设计位置：§8.5 L639-684（顺序图把 commit→ack→commit_after_alloc 画在控制消息流内）；§8.7 L755-759, L812（handler 在接收线程）；§20 L2461-2474（只有泛化要求）
- 源码位置：`pool_scheduler.py:578`（`_unfinished_requests` 写）、`:604-618`（can_load/`_loading_req_ids` 写）、`:845`（load_specs pop）、`:910-957`（build_connector_meta 迭代/快照）；全文件无 `threading.Lock`（grep 证实）
- 触发前提：DECT 在 DE scheduler 循环执行 `build_connector_meta` / `update_state_after_alloc` 期间应用 commit（两线程同进程并发是设计的必然形态——决策协议全程跑在 control 线程，T6-T8）。
- 具体时序：竞态 (a)：循环在 :952 `for ... in self._unfinished_requests.items()` 迭代时，control 线程 :578 写入同名字典 → `RuntimeError: dictionary changed size during iteration`（EngineCore 崩溃）。竞态 (b)：control 线程在 :913 `loading_req_ids.copy()` 快照之后、:954 ReqMeta 生成之前完成 `_loading_req_ids.add` → 本步 metadata 的 `loading_req_ids` 不含该 req → Worker load 完成后 `get_and_clear_finished_requests(meta.loading_req_ids)` 交集为空 → `STORE_DONE` 永久丢失 → 请求滞留 WAITING 直至 abort（且 abort 又落入 D1）。竞态 (c)：control 线程读 pending candidate（T4 由循环线程写）无同步。
- 为什么不正确：`KVPoolScheduler` 全部方法假定单线程（调度循环）调用；设计把 `commit_after_alloc`（内部必然触达这些方法）放在协议消息处理流里，却未指定它必须回填到调度循环线程执行，§20 的「串行化或等价原子」没有落到线程归属上。
- 可能影响：EngineCore 崩溃（a）；请求永久滞留 + 后续 abort 泄漏（b）；candidate 状态撕裂（c）。均为低频高害。
- 最小修正建议：设计写死线程模型——control 接收线程只写 `PathDecisionCoordinator`（自身加锁）并投递「已提交」队列；`commit_after_alloc`/`abort_probe`/plan 发射一律在 scheduler 循环上下文（如 `build_connector_meta` 开头 drain 提交队列）执行；PE 侧同理。
- 置信度：高（无锁事实与调用点已核实；仅「设计意图是否本就如此」无法从文本判定，按自然读法立档）。

### D3：`S_DE` 二次 floor 与 probe 实际语义不兼容——full-hit 判定退化或 `commit_after_alloc` 触发 assert 崩溃
- 严重级别建议：**P1**｜类型：语义/契约不匹配（token accounting）｜证据等级：`[当前源码已确认]`（KVPool 行为）+ [设计推导]（两种实现读法）
- 设计位置：§5.2 L273-276（`S_DE`/`H_DE` 公式）、L304（`K_DE>=R` 判定）；§4.4 L214-228 与 §14.2 L1980-1981（「coverage>=prompt_len-1」表述）；§9.7 L1250-1257（commit_after_alloc 签名）
- 源码位置：`pool_scheduler.py:502-503`（查询长度预 floor）、`:526-527`（`hit==P` 减 1）、`:604-614`（`num_external == kvpool_cached - vllm_cached` assert）、`:849-852`（`==P-1` 非对齐时 +1）
- 触发前提：读法一（probe 复用 `KVPoolScheduler.get_num_new_matched_tokens`，即 §9.7「允许 KVPoolScheduler 创建 client/load-spec」的自然实现）：full hit 且 `(P-1) % G != 0`（G=256 下绝大多数 prompt 长度）。
- 具体时序：读法一：T3 返回 `S_DE_raw = R`（如 P=1024,G=256 → 1023）→ T4 `S_DE = floor(1023/256)*256 = 768` → `K_DE=768 < R` → full hit 系统性退化为 DE_PARTIAL_READ（功能不错但 DE_FULL_HIT 近乎不可达，与 §4.4/§14.2 的判定表述矛盾）；若 adapter 仍按设计把 `store_load_tokens=H_DE=768-L_DE` 传给 `commit_after_alloc` 进而调 `pool_scheduler.update_state_after_alloc`，:604 assert 要求等于 `1023-L_DE` → **AssertionError，scheduler 崩溃**。读法二（probe 用 `client.lookup` 原始层）：`S_DE_raw=1024`，`min(...,R)` clamp 得 `K_DE=1023=R`，判定正确；但此时 probe 不产生 LoadSpec，`commit_after_alloc` 须自建 `LoadSpec(kvpool_cached=K_DE)` 才能过 assert——设计未写。
- 为什么不正确：设计的 token 数学假定 `S_DE_raw` 是「原始绝对前缀」，而可复用的 probe 出口给的是「已 floor 且 full-hit 减 1」的值；两套语义在 `S_DE` 公式里叠加后，判定条件与 §4.4/§14.2 的文字契约不符，且与 KVPool 冻结 LoadSpec 时的 assert 直接冲突。
- 可能影响：DE_FULL_HIT 路径名存实亡（全量退化为 partial，增加本可避免的 PE 往返），或实现照字面接线后 full hit 即崩溃。
- 最小修正建议：设计钉死 probe 语义——要么明确「adapter 使用原始 lookup，LoadSpec 由 commit_after_alloc 以 `vllm_cached=L_DE, kvpool_cached=K_DE` 自建」；要么修正 `S_DE` 公式为「不再二次 floor，仅 `min(S_DE_raw, R)`」并说明 `S_DE_raw` 已含 store 侧对齐与 -1；同时修订 §4.4/§14.2 的判定表述与 §5.2 对齐（补一句：`+1` 边界 load 覆盖末 token slot、由 T16 重算覆盖写，顺序安全）。
- 置信度：高（KVPool 侧行为逐行核实；设计文本存在两种读法本身即为缺陷证据）。

### D4：F11 失败上报断裂的修复依赖 KVPool 私有属性；不修复则脏 KV 进 prefix cache
- 严重级别建议：**P1**｜类型：失败语义/正确性｜证据等级：`[当前源码已确认]`（断裂）+ [设计推导]（修复路径可行性）
- 设计位置：§9.7 L1317（adapter get_block_ids_with_load_errors）、L1347（「统一进入 request-level invalidation」）；§13.2 L1789-1794（invalid 不晚于终态）；§3.1 L73-80（不得修改 KVPool）
- 源码位置：`pool_worker.py:457-465`（recv 线程未注入 invalid 集合）、`:1333-1337`（只 drain worker 自有集合）；`kv_transfer.py:830-831`（线程构造器其实接受注入）、`:843-844`（私有集合）、`:908-910, 924-926`（失败块入私有集合）、`:942`（无条件 set_finished）；phase1 F11
- 触发前提：DE_FULL_HIT 的 Store bulk get 部分 chunk 返回非 0（磁盘/后端抖动），单 KV group（§17 限定，hybrid 分支更糟但被排除）。
- 具体时序：T11 部分失败 → 失败块入线程私有集合、无人 drain；`set_finished_request` 照常 → T12 adapter 消费到 `done_recving` → 误判 `STORE_DONE` → T13-15 promote + `cache_blocks(request, R)` → 含脏数据的 blocks 进入 prefix cache → 本请求输出错误 + **后续请求命中同一前缀被污染**。
- 为什么不正确：upstream 契约要求「失败 block 不得晚于该请求的 finished 上报」（base.py:384-389），当前 KVPool 在 load_async 下结构性违反；设计把修复责任放在 adapter，但 §3.1 禁止改 KVPool，于是唯一可达路径是注入/读取 `kv_recv_thread._invalid_block_ids` 等私有成员——这一手段与其脆弱性（私有名、未来 KVPool 重构即碎）在设计中完全未被提及，也没有写清「按 block_id ∩ 冻结 manifest 归因到 request」这一必要步骤（失败集合只有 block 粒度、无 request 键）。
- 可能影响：静默正确性腐蚀且污染面跨请求扩散；属本路径最严重的单点正确性风险。
- 最小修正建议：① 设计显式授权 adapter 在构造 `KVPoolWorker` 后注入共享 `invalid_block_ids+lock`（线程构造器 kv_transfer.py:830-831 原生支持，属组合使用而非修改）或同步 drain 线程私有集合；② 写明 block→request 归因规则（窗口内 blocks 请求独占：`delay_cache_blocks` + async 无块共享，U:scheduler.py:2652-2653）；③ 把「部分 DMA 失败 → invalid 先于 terminal → FINISHED_ERROR」列为方案 A″ 实施门槛（§22）级的验收项并配失败注入 UT（§23.1 已有条目，需标注覆盖 load_async）。
- 置信度：高（断裂逐行核实；修复路径可达性已核实构造器签名）。

### D5：Store probe 的阻塞 RPC 无超时，挂在 DE 调度循环上
- 严重级别建议：**P2**｜类型：可用性（head-of-line blocking）｜证据等级：`[当前源码已确认]`
- 设计位置：§8.2 L583-586（只承认 client/load-spec 状态，未提阻塞）；§9.7 L1236-1245；§19.1 L2433-2435（只覆盖「RPC 失败返回」，不覆盖 hang）
- 源码位置：`pool_scheduler.py:515-521`（循环内 RPC）、`:1106-1108`（REQ `recv()` 无超时）；LKS 位于 DEW[0] daemon（ascend_store_connector.py:119-120, 309-314）
- 触发前提：每个 DualPath candidate 的 T3；DEW[0] 繁忙（其 LookupKeyServer 与 Store backend 争抢）或死亡。
- 具体时序：DES 循环在 get_num_new_matched_tokens 内同步发 REQ 并 `recv()` 阻塞 → 整个 DE 调度停摆（所有请求，不止本请求），直到 LKS 应答或进程级干预。
- 为什么不正确：probe 被设计为准入热路径上的同步原语，却继承了「无超时阻塞单线程调度器」的现状；§19.1 的「lookup RPC 失败可降 PE_READ」对 hang 不适用（没有失败返回）。
- 可能影响：LKS 慢/死 → DE 引擎级吞吐归零；与 D6 叠加（load 也 hang）时故障面扩大。
- 最小修正建议：设计要求 adapter probe 带超时（`RCVTIMEO` + 失败后按 miss/unknown 走 PE_READ 候选），并把「probe 最坏耗时」写入 §24 指标（decision latency 分解）。
- 置信度：高。

### D6：Store bulk DMA 无 hang 检测，请求可永久滞留 WAITING
- 严重级别建议：**P2**｜类型：可用性/失败检测缺口｜证据等级：`[当前源码已确认]`（无超时）+ [设计推导]（设计无 watchdog）
- 设计位置：§19.1 L2416-2438（错误分类无 STORE_IO_TIMEOUT）；§13.2 L1789-1793（terminal 延后规则反而依赖「transport 明确停止」，hang 时永不满足）
- 源码位置：`kv_transfer.py:902`（`m_store.get` 同步调用、无超时包装）；recv 线程 daemon（:292 附近，scratch-04）
- 触发前提：Store backend hang（不是返回错误）。
- 具体时序：T11 永不返回 → 无 STORE_DONE、无 invalid、无 terminal → 请求永久 WAITING；blocks 被持有；`_inflight_prefills` 不释放，侵蚀后续 async load 准入容量（U:scheduler.py:934-940）。
- 为什么不正确：设计的失败分类假设所有失败都「可检测」，DMA hang 不在其列；quarantine 语义（L1728-1729, L1571-1573）也只针对「无法证明完成的 transport」的人工介入，没有自动触发器。
- 可能影响：单请求级永久滞留 + 容量泄漏；需外部（客户端超时 abort）才可能收敛——而 abort 后又落入 D1 的终态发布问题（I/O 仍 hang，ownership 永不为 0，finished_sending 也发不出）→ 只能等进程重启。
- 最小修正建议：设计增加 Store load watchdog（超时 → STORE_LOAD_ERROR → quarantine + 按失败流程上报 invalid+terminal；即便底层 DMA 线程无法真正取消，也须把请求级状态收敛，blocks 进 quarantine 而非正常释放）。
- 置信度：中高（hang 语义按接口行为推断，`m_store.get` 内部是否自带超时 `[外部接口待确认]`）。

### D7：control-only proposal 被 PE 降为 PE_READ 时，PE 模型请求的「补派发」路径未规定
- 严重级别建议：**P3**｜类型：协议完备性缺口｜证据等级：[设计推导]
- 设计位置：§8.2 L562-568（派发只发生在 proposal 阶段）；§7.3 L462-468（提交前可转 PE_READ 的条件）；§8.5 L660-673
- 源码位置：—（纯设计层）；相关：`proxy_dispatch_id` 语义 L371
- 触发前提：DE 按自身 coverage 预测 full-hit 走 control-only；PE 因 handshake/topology/capability 不相容（§7.3 允许）提交 PE_READ。
- 具体时序：T6 control-only proposal → T7 PE commit(PE_READ) → T8 DE 收到 → **此后谁、在何时、用什么 envelope 补建 PE 模型请求？** §8.2 step 6 的 candidate dispatch 时机早已错过；§8.5 无 retro-dispatch 分支。
- 为什么不正确：DE 的通道选择是对 PE 决策的预测，预测可与 PE 实际 commit 不一致；设计只覆盖了「预测正确」的分支。
- 可能影响：低频（capability 在 init fail-fast 后仅余运行时 handshake 失败）但出现时请求无 PE 计算方、只能等 DECISION_TIMEOUT 后按失败收场——把「本可正常 PE_READ 服务的请求」变成失败请求。
- 最小修正建议：设计补一条「commit(PE_READ) 到达且无 PE 模型请求时，DE 在 scheduler 循环上下文补派发（复用同一 `DualPathRequestKey`/envelope，decision_version 不变）」的规则，或更简单地取消预测——full-hit 候选也一律先走 control-only 但允许 commit 携带「需要 PE 模型请求」标志。
- 置信度：中（触发条件真实存在于 L462-468；但发生概率低，且能力校验前置可能使该分支实际不可达——若设计者意图如此，应写死「capability 不相容在 init fail-fast，control-only 永不降级」）。

### D8：DECISION_TIMEOUT / PE 侧 candidate 过期等「决策面失败」的跨进程传播与状态回收未规定
- 严重级别建议：**P3**｜类型：协议完备性/状态回收｜证据等级：[设计推导]
- 设计位置：§8.7 L810-811（DECISION_TIMEOUT 只有一句话）；§19.1 L2419；§8.6 L686-729（无 expire/GC 接口）
- 源码位置：失败传播所需的框架通道本身存在（每步 metadata + `get_block_ids_with_load_errors`，U:mixin:102-105），缺的是设计接线。
- 触发前提：(i) PE 不可达/不决策导致重试耗尽；(ii) PE 侧 candidate 登记后 DE（incarnation）死亡。
- 具体时序：(i) DECT 判定 DECISION_TIMEOUT → 需要 DEW 发布 invalid+failed terminal（请求在 WAITING，D1/S4），但 DEW 可能从未见过 plan（T9 未发生），§19.2 流程以 plan/fence 存在为前提——传播内容（无 plan 的纯失败指令）未定义。(ii) PECT 的 candidate 永驻（无 TTL/无 incarnation 失联检测）。
- 为什么不正确：决策面与数据面的失败模型不对称——数据面失败有完整 §19.2 流程，决策面失败只有一个错误码。
- 可能影响：(i) 与 D1 叠加形成永久滞留；(ii) PE coordinator 内存缓漏 + 迟到消息归属混乱（与 D9 相关）。
- 最小修正建议：① 定义「decision-failed」控制/元数据记录随 `build_connector_meta` 下发，Worker 收到后对无 plan 请求直接按 manifest 发布 invalid+terminal；② coordinator 增加 candidate TTL 与 incarnation 失联清理。
- 置信度：中。

### D9：决策态 tombstone / 迟到控制消息的归属保留期未规定
- 严重级别建议：**P3**｜类型：协议完备性（与 §6 wire tombstone 不对称）｜证据等级：[设计推导]
- 设计位置：§6 L378-381（只有 wire ID tombstone）；§8.6 L705-728；§10.3 L1554-1573（事件侧迟到规则完整，决策侧没有对应物）
- 触发前提：请求 terminal 且 coordinator 状态被回收后，迟到 commit/ack/CONTROL_ERROR 到达。
- 具体时序：GC 后迟到 commit → `register_candidate` 幂等重建一条「新」candidate → commit CAS 成功 → `commit_after_alloc` 为已释放请求重启 I/O（若 blocks 已被复用则为跨请求写风险；当前被 D1 的泄漏悖论「保护」——状态若泄漏则永不 GC——两个缺陷互相掩盖）。
- 为什么不正确：数据面事件有四元组匹配+tombstone 的完整防 stale 设计，决策面消息只有 message_id 幂等，缺少请求级终态保留。
- 可能影响：与 D1/D8 的修复方式耦合；若 D1 按「terminal 后删状态」修复，本项即升级为实质风险。
- 最小修正建议：coordinator 状态采用与 wire ID 相同的 terminal retention window 语义；迟到消息命中 tombstone 一律回 CONTROL_ERROR。
- 置信度：中（依赖状态回收策略，设计未写故按缺口立档）。

---

## 正向结论（设计正确之处及依赖前提）

- **Z1（accounting 与 upstream 精确咬合）**：`E_DE=R-L_DE` 统一声明 + 乐观 `num_computed_tokens=R` + promote 时 `==num_tokens` 的 -1 分支不触发（U:scheduler.py:800, 1004, 2521-2522）——DE 重算末 token 的语义不依赖任何框架修改。依赖前提：声明严格为 `R-L_DE` 而非 `P-L_DE`。证据：S8，`[当前源码已确认]`。
- **Z2（路径无关 blocks 一次分配）**：commit 前分配、`delay_cache_blocks` 隔离、decision=PE_READ 时零收敛成本（S10）。依赖前提：`update_state_after_alloc` 两次调用按 `num_external>0` 幂等冻结（F4 已核实框架行为，设计 L681-684 要求正确）。
- **Z3（第二次 update_state_after_alloc(0) 无危险窗口）**：该调用只可能在 promote 后发生（U:scheduler.py:822-827 路径），而 promote 以 STORE_DONE 为前提，STORE_DONE 以 ReqMeta 发射（LoadSpec pop，pool_scheduler.py:845）为前提——`can_load=False` 分支（:589-593）在危险窗口内不可达。`[源码可达，尚未运行验证]` 的推演，链条每一环均为 `[当前源码已确认]`。
- **Z4（完成/失败的多 rank 汇聚语义与框架天然匹配）**：finished 全 rank 计数、invalid 并集无 quorum、invalid 先于 finished 消费、terminal 状态 assert（U:scheduler.py:1578-1586, 2574-2586；utils.py:78-90, 159）——§13.2/§13.3 的规则是「利用」而非「对抗」框架。前提：每 rank 每 ID 恰一次 receive terminal（含失败）。`[当前源码已确认]`。
- **Z5（STORE_DONE→公共 finished_recving 链路机械闭合）**：loading_req_ids 跨步存续（pool_scheduler.py:901）、metadata 每步发射、get_finished 含空步每步必调（U:mixin:35-48, 102-105）、adapter 内部消费 done_recving 与公共集合分离（L1339-1343）。前提：D2 竞态 (b) 被排除（loading_req_ids 快照必须先于或同步于 ReqMeta 发射包含该 req）。
- **Z6（incarnation 隔离 restart 身份）**：request key 含 incarnation + wire 事件四元组匹配 + tombstone，与当前 uuid4 engine_id 缺省（MLC:702）组合后，restart 复用风险在主语义上闭合（S9）。
- **Z7（control-only 不需要 L_PE）**：静态规则 1 使 full-hit 判定与 PE 本地状态无关（L438, L493 pe_coverage 可空），避免了对 PE 调度循环的反向依赖——这是 control-only 通道成立的关键，但建议设计显式写明该结论及其对未来策略的限制（S5）。

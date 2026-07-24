# Phase 3 维度评审 04：兼容性（Multi / Mooncake 父类 / AscendStore）+ NPU 性能 + 测试验收

- 评审人：Phase 3 维度评审代理（compat/perf/test 组）。
- 负责维度：§9-14（AscendMultiConnector 广播语义）、§9-15（MooncakeLayerwise 父类兼容）、§9-16（AscendStore 语义兼容）、§9-19（NPU 性能风险）、§9-20（测试/可观测性/验收）；§10 过度设计检查两项（SharedMooncakeTransferRuntime、Store adapter）。
- 负责复核的 Phase 2 候选：C-PE-1、C-PE-5、C-PE-8、C-FH-2、C-FH-5、C-FH-6、C-PR-4、C-PR-5、C-PR-7。
- 基线：设计 snapshot `inputs/option-a-detailed-design.snapshot.md`（行号以此为准）；vllm-ascend `dev/dualpath @ 0ec11a47`；upstream `vllm @ 8df14cfc`（配套 v0.23.0 有小幅偏差）。
- 证据等级：`[当前源码已确认]`（本代理已逐行复核原文）/ `[源码可达，尚未运行验证]` / `[设计推导]` / `[外部接口待确认]` / `[事实冲突]`。本机无 NPU/运行时，全部结论为静态证据。
- 行号缩写：`MLC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`；`AMC` = `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`；`DPC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`；`pool_scheduler.py` / `pool_worker.py` / `kv_transfer.py` / `ascend_store_connector.py` 均指 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/` 下文件；`U:` = upstream vllm。

---

## 维度结论

### 维度 14：与 AscendMultiConnector 广播语义的兼容性

**结论：广播语义本身与设计假设精确一致，正常路径兼容；两个条件性缺口（sibling `use_layerwise` 未固定 → C-PE-5/C-PR-4；F18 版本漂移为 watch 项）。设计 §4.1/§15.2/§15.4 对当前代码事实的描述经复核全部属实，无夸大。**

逐项核对（本代理已复核原文）：

1. **§4.1 first-positive + Layerwise 例外（F1/F2）——属实，兼容。**
   - `AMC:32-41`：winner 与 `isinstance(c, MooncakeLayerwiseConnector)` 的非 winner child 收 `(真实 blocks, 真实 tokens)`（AMC:36）；其余非 winner 收 `(empty_blocks, 0)`（AMC:41，`blocks.new_empty()`）。`[当前源码已确认]`
   - upstream first-winner 守卫 `U:multi_connector.py:397-399`（`to_return[0]==0 and toks>0`）：第一个正数 child 成为唯一 winner，后续 child 仍被调用但不覆盖——设计 §4.1 L162-168 与 §15.2 L2074-2081 的描述逐字成立。`[当前源码已确认]`
   - §4.1 L177-179 主动承认「非 winner AscendStore child 仍可能产生 Scheduler-side lookup/LoadSpec 临时状态」，与 F10 一致——表述诚实，没有把 sibling 假设成纯函数。
2. **None 短路（F3/F16）在 A″ 拓扑下无残留风险。** DualPath 为 child0（§15.2 L2068-2072 强制配置顺序），child0 返回 `(None, False)` 时 Multi 立即整体短路（`U:multi_connector.py:393-394`），sibling 该轮根本不被调用，无 LoadSpec/LookupKeyClient 副产物；「先写 winner 后 None」的残留形态要求 DualPath 先返回过正数，而 preemption 重查时 decision 已持久化、稳定返回正数。结论：兼容（正向，见 PT-1）。
3. **Worker hook 广播（§15.4 L2117-2126）——属实，兼容机制存在。** `start_load_kv`/`wait_for_layer_load`/`save_kv_layer`/`wait_for_save`/`register_kv_caches` 均无条件顺序广播（`U:multi_connector.py:289-309, 245-247`）；唯一例外 `get_handshake_metadata`（第一个非 None，:452），设计不依赖它。decision 经 `DualPathConnectorMetadata.plans` 随框架 metadata 通道（`bind_connector_metadata` 按 zip 逐位分发，`U:multi_connector.py:260-266`）到达 Worker，Worker 侧无需跨进程查 coordinator——通道闭合。
4. **完成集合合并（§15.4 L2128-2141）——兼容且恰好够用。** `get_finished` 并集（`U:multi_connector.py:311-336`）：PE_READ 时 sibling 独占 PE-local `finished_recving`（DualPath 不发布该 ID）；DE_PARTIAL_READ 时 DualPath 独占（sibling 的 `_loading_req_ids` 入口要求 `num_external>0`，`pool_scheduler.py:617-618`，非 winner 收 0 → 不发布）——两侧互不重复发布，upstream assert（`U:scheduler.py:2576/2581`）不触发。`finished_sending` 的 `_extra_async_saves` 计数门控（`AMC:97-98` / `U:multi_connector.py:325-334`）恰好承载「DualPath Forward drain + sibling Store save drain 都完成才释放延迟 blocks」——A″ 不建跨 sibling 原子完成器（L2128-2141）在正常路径不损失正确性（正向 PT-1）。失败路径不闭合见 C-PE-1（P0，复核确认）。
5. **`request_finished_all_groups`（AMC:78-102）——兼容。** 全 HMA 路径逐 child 调用、或语义 delay_free、params 唯一产出者（冲突 RuntimeError）。DualPath 与 sibling 均返回 `(bool, None)` 形态（`pool_scheduler.py:1039-1068` 返回 params=None），无 params 冲突。前提：DualPath 的 `request_finished_all_groups` 不得产出 `kv_transfer_params`。
6. **F18（upstream v0.23.0 后 3 提交）——当前兼容，列为 watch 项。** AMC 覆写保持 v0.23.0 语义 + Mooncake 例外；设计 §4.1/§15.2 依赖的「非 winner Layerwise child 收真实 blocks」是 AMC 自身代码（AMC:36），不随 upstream 漂移；漂移面在 AMC 未覆写的部分（params dict-merge、`has_pending_push_work`），设计均未依赖。升级 upstream 时需重核 AMC——建议设计 §4.5 或 §22 加一条版本锚定说明（非缺陷，P3 级提示）。

### 维度 15：与 MooncakeLayerwise 父类的兼容性

**结论：五个重点项（不调父 `__init__`、共享 runtime 单次注册、方向化 dispatch 覆盖、`get_finished` 重写、`request_finished_all_groups` 重写）在当前源码下均有事实依据且路径可行；两个 P3 级偏差（`_is_kv_producer` 未复制 = CP-2；继承 init 消费未定义的 `kv_port` = CP-3）；F15 归因 bug 被继承 = C-PR-7（确认 P2，扩展假阳性 DONE 形态）。**

1. **不调父 `__init__` 的构造（§9.3 L922-927）——可行，已被当前 foundation 验证。** 父构造器（MLC:697-712）只做 5 件事：`KVConnectorBase_V1.__init__`、`_is_kv_producer`、`engine_id`、`_connector_metadata`、创建 Scheduler/Worker。DPC:120-142 当前已按同一模式运行（直接 `KVConnectorBase_V1.__init__`，自建 DualPath 子类），注册链路完整（phase1-01 §15）。**偏差**：DPC:121-123 未复制 `_is_kv_producer`（MLC:701）——当前全仓无读取方（phase1 证据），属静默偏差而非现行 bug → CP-2（P3）。另注意 facade 的 `start_load_kv`/`save_kv_layer`/`wait_for_layer_load` 内 `assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)`（MLC:768/774/782）——DualPath 换用 `DualPathConnectorMetadata` 后必须覆盖这三个 facade（设计 §9.3 L929-950 已列覆盖，闭合）。
2. **共享 runtime `register_kv_caches` 单次（F7）——属实，且「只调一次」是硬约束而非偏好。** MLC:1359-1398：`kv_both` 下两个 if（:1359 producer、:1386 consumer）都命中，一次调用完成 buffer 注册 + 一个 send thread + 一个 recv thread；`global_te.register_buffer` 有 `is_register_buffer` 单次守卫（`mooncake_transfer_engine.py:31-40`），**第二次调用静默跳过**——若实现错误地建第二个 runtime/worker 各自注册，第二个的 buffer 根本不注册，RDMA 写未注册内存。设计 §9.2 L892-898 与 §10.2 L1478-1482 的约束与此精确对齐。框架调用点每 Worker 进程恰一次（`model_runner_v1.py:3824-3825`）；Multi 下广播到各 child（`U:multi_connector.py:245-247`），sibling 注册的是 Store backend buffer（`pool_worker.py:742`），与 TE 不同资源，无冲突。`[当前源码已确认]`
3. **方向化 adapter 覆盖 dispatch（F5/F6）——必要且充分。** F5：MLC:1006 `build_connector_meta` 在 `kv_both`（`is_kv_consumer` 为真）下恒走 consumer 分支，`_reqs_need_send_layerwise` 只进不出（已复核 MLC:1006-1022）；F6：MLC:1606-1615 `start_load_kv` if/elif 使 `kv_both` 恒走 consumer 分支，producer transfer mapping 整体跳过（已复核）。设计 §9.2 L905-908 要求完全覆盖 `update_state_after_alloc()`/`build_connector_meta()`/`start_load_kv()`/`get_finished()`——这正是最小必要覆盖集。补充：facade `get_finished`（MLC:756-759）当前**丢弃** `finished_req_ids` 参数（调 `worker.get_finished()` 无参版），设计 §9.5 L1108-1109 要求 facade 转交该参数——覆盖清单应显式包含 facade 层（§9.3 L952-959 的签名已隐含，建议写明）。
4. **`get_finished` 重写（F8）——必要，重写方向正确。** MLC:1400-1428（已复核）：返回 `(set(), done_recving)`，`finished_sending` 恒空；失败请求只把 `meta.local_block_ids` 汇入 `_invalid_block_ids`（:1416-1419）；**terminal 早于 mapping 到达时事件被永久丢弃**（:1407 `if s in self.request_map` 过滤 + `get_and_clear` 已清空）。设计 §9.5 L1098-1112（pending raw terminal inbox + 失败也发布 receive terminal）同时修复三点，且与 upstream「失败 block 不得晚于 finished 上报」契约（`U:base.py:384-389`）和 assert（`U:scheduler.py:2576/2581/2585`）方向一致。
5. **`request_finished_all_groups` 重写（F9）——必要。** MLC:1102-1112/1114-1124 两个方法均恒 `(False, None)`（已复核原文）；框架实际走 `request_finished_all_groups`（MLC 继承 `SupportsHMA`，`U:scheduler.py:2461-2469` 分发）。设计 §13.4 L1865-1877 `SchedulerReleaseLedger` 的保守 delay_free 策略覆盖之，机制与框架 `_free_blocks` 等待 `finished_sending` 的通道（`U:scheduler.py:2583-2586`）咬合。
6. **继承 init 的隐性消费（新观察，CP-3 相关）。** `MooncakeLayerwiseConnectorScheduler.__init__`（MLC:793-834，已复核）：consumer 下断言 `pcp==1`（:803-806，Stage 1 拓扑满足）、从 `kv_transfer_config.kv_port` 派生 `side_channel_port`（:809-812）、创建 `ThreadPoolExecutor(32)` + httpx client（:820-834）。Worker `__init__`（MLC:1130-1201，已复核）：进程级副作用 `os.environ["ASCEND_TRANSFER_TIMEOUT"]`（:1131，幂等）、`global_te` 单例初始化（:1172）、同样从 `kv_port` 派生端口（:1164-1169，recv 线程实际监听 `side_channel_port + tp_rank`，MLC:605-607）。`kv_port` 默认 14579（`U:config/kv_transfer.py:56`）；设计 §16.1/§16.2 配置模式只定义 `forward.base_port`/`reverse.base_port`/`control.pe_port`，**未写派生 runtime config 的 `kv_port` 字段如何设置**——§18.2 L2388-2390 的「本地 receive port 按角色选择」没有落到具体配置字段 → CP-3（P3，配置缺口）。
7. **F14/F15。** `wait_for_layer_load` 为 `pass`（MLC:1976-1977，已复核）——设计 §11.2 用 Scheduler 门禁而非该 hook，措辞「防御性检查」准确。F15 归因 bug（MLC:507，已复核：`:469` 循环残留变量）被共享 send thread 继承 → C-PR-7（确认并扩展，见复核节）。

### 维度 16：与 AscendStore lookup/load/failure 语义的兼容性

**结论：probe/commit/abort 生命周期与 LoadSpec 机制可对接（commit 唯一公共入口 = `pool_scheduler.update_state_after_alloc`，abort 必须触私有 `load_specs`——可达但脆弱）；`load_async=True` 强制与 `_loading_req_ids` 唯一入口咬合；F11 断裂的 adapter 修补路线对 DE 真实可达（本代理给出具体注入窗口），但对 PE sibling 在 §3.1 边界内不可达（C-PE-1 维持 P0）；F17 在 DE adapter 配置下不可达（正向 PT-5）；新发现 PE 侧 adapter/LookupKeyServer 双建冲突（CP-1，P2）。**

1. **probe → LoadSpec 生命周期（F10）——设计认知准确，可达性已核实。**
   - `KVPoolScheduler.get_num_new_matched_tokens`（pool_scheduler.py:478-565，已复核 505-565 原文）：非 layerwise 走 zmq REQ 到 rank-0 worker 的 LKS（:515-521）；`hit==num_tokens` 减 1（:526-527）；命中且需分配时创建 `LoadSpec(vllm_cached, kvpool_cached)`（:549-553）。设计 §8.2 L583-586「会创建 client/load-spec 状态，不能直接当作纯 probe」属实。
   - `commit_after_alloc` 的唯一公共通路是 `pool_scheduler.update_state_after_alloc(request, blocks, H_DE)`：`num_external>0` 分支 assert `num_external == kvpool_cached - vllm_cached`（:604-614，已复核）、`can_load=True`（:616）、`_loading_req_ids.add`（:617-618——进入 `AscendConnectorMetadata.loading_req_ids` 的唯一入口，设计 L1334-1336 的目标可达）。**约束**：`H_DE` 必须与 LoadSpec delta 逐 token 相等，这把 adapter 的 token 口径钉死在 probe 的语义上（已 floor + full-hit 减 1）——与 C-FH-3 的二次 floor 问题直接相关（不属本组复核，但维度 16 记录该耦合）。
   - `abort_probe` 无公共出口：`load_specs` 清理点仅三个 `_process_*` pop（:690/742/845），waiting 期 abort 无清理（F10）。adapter 只能 `pool_scheduler.load_specs.pop(req_id, None)` 触私有字典——**不改 KVPool 前提下可达，但属私有成员依赖**，设计 L1247-1248 未写明该机制与脆弱性。
   - `LookupKeyClient` 首次 lookup 惰性创建后常驻（:515-516），`close()` 无生产调用方（:1111-1112）——probe 的隐性常驻资源，设计未提及（轻微，记录备查）。
2. **`load_async` 强制（§9.7 L1334）——与现有机制咬合。** 派生 config `load_async=True` + 非 layerwise → `:617-618` 入口可用；worker 侧 `KVCacheStoreRecvingThread` 仅 `load_async=True` 时创建（pool_worker.py:455-467）。`use_layerwise=True` 会强制 `load_async` 关闭（:565）——§9.7 L1204 固定 `use_layerwise=False` 与此一致（反向印证 sibling 也必须 false，见 C-PE-5 扩展）。
3. **F11 失败断裂与 adapter 修补路线——DE 可达，PE 不可达（关键分界）。** 已复核：pool_worker.py:457-465（recv 线程创建只传 7 个位置参数，**未注入** invalid 集合/锁）；kv_transfer.py:830-831（线程构造器原生接受 `invalid_block_ids`/`invalid_block_ids_lock` 可选参数）；:908-910/:924-926（失败块入线程私有集合）；:942（无条件 `set_finished_request`）；pool_worker.py:1333-1337（`get_block_ids_with_load_errors` 只 drain worker 自有集合）。
   - **DE adapter 修补窗口（具体化，正向 PT-4）**：transfer 线程在 `KVPoolWorker.register_kv_caches` 内启动（pool_worker.py:743 → `_start_kv_transfer_threads`，:350-468 幂等）；adapter 自己持有 `KVPoolWorker` 实例并控制 `register_kv_caches` 调用时机（设计 L1299-1302）——可在该调用返回后、首个 `start_load_kv` 之前，把 `worker.kv_recv_thread._invalid_block_ids`/`._invalid_block_ids_lock` 替换为共享对象（或在 adapter 的 `get_block_ids_with_load_errors` 里同步 drain 线程私有集合）。此刻线程虽已 start 但阻塞在 `request_queue.get()` 上、无任何请求在途，替换无竞态。代价：依赖两个私有属性名；block→request 归因需按「失败 block ∩ plan `store_target` block ids」求交（窗口内 blocks 请求独占：`delay_cache_blocks` + async load 不支持块共享，`U:scheduler.py:2652-2653`）。设计 L1347 只写了目标，未写机制与归因规则——C-PR-5/C-FH-4 的修正建议成立。
   - **PE sibling 修补路线在 §3.1 内不存在**：sibling 是既有 `AscendStoreConnector`，DualPath 不持有其 `KVPoolWorker`，§3.1 L73-80 又禁止修改/monkey-patch——adapter 注入路线对 PE_READ 不可达。C-PE-1 的修复只能二选一：放松 §3.1 允许对 pool_worker.py:457-465 做一处构造参数修复（本质是修既有契约 bug，不是为 DualPath 改行为），或接受端到端静默损坏（不可接受）。维持 P0。
4. **F17（`request_finished_all_groups` 不对称）——DE adapter 配置下不可达（正向 PT-5）。** pool_scheduler.py:1045-1047（已复核原文）：`kv_consumer && !consumer_is_to_put` → 提前返回 `(False, None)`。§16.3 L2327 要求派生 Store config 关闭 consumer put/save，故 DE adapter（consumer + `consumer_is_to_load=True` + 关 put）恒走早退分支，`:1053` tracker-None 不对称分支不可达。PE sibling（producer）的 F17 是既有边界（abort-before-first-step 时 tracker None → 可能 delay_free 而 send 线程永不报完成，`[源码可达，尚未运行验证]`）——非 A″ 引入，但应并入 C-PE-4 的 abort 终态处理一并覆盖。
5. **§4.3 bulk load 语义——属实。** 非 layerwise 每请求一次 bulk `m_store.get`（pool_worker.py:840 同步路径 / kv_transfer.py:902 异步路径），本层无 DMA 拆分（拆分只在 layerwise `_batch_copy_with_limits`，kv_transfer.py:397-448）；`discard_partial_chunks` 默认 True（pool_scheduler.py:124-126）；load 目标为模型正式 blocks。设计 L205-209 描述准确（「可以在 backend 内部拆成多个 DMA」对当前代码是 aspirational，当前为单次同步调用，记录备查）。
6. **LookupKeyServer 创建条件（§9.7 L1295-1296）——与 ascend_store_connector.py:119-120 精确一致**（worker 角色 + 非 layerwise + `parallel_config.rank==0`）。**但设计未声明 PE 不建 adapter** → 同一 PE rank-0 进程内 sibling 的 LKS 与 DualPath adapter 的 LKS bind 同一 ipc path（path 由 `lookup_rpc_port` 派生，pool_scheduler.py:1115-1130；LKS 构造即 bind，ascend_store_connector.py:291-297），且 `m_store` 按 KVPoolWorker 实例创建（pool_worker.py:249-260）、`register_buffer` 每实例一次（:742）——双 backend client + 双 buffer 注册 + REP bind 冲突 → CP-1（P2）。

### 维度 19：NPU 性能风险

**结论：设计未引入新的 `.item()`/CPU-NPU 搬运/额外 HBM 拷贝（§3.4 禁 staging 与现有代码一致）；控制面独立线程方向正确（§8.7），但存在四个已确认的控制面阻塞/串行化风险点：① probe 的 zmq REQ `recv()` 无超时挂在 DE 调度循环（F10，C-FH-5 确认）；② `wait_for_commit` 调用者未指定，若落在 scheduler 循环即 head-of-line 阻塞（与 C-PR-3 合并）；③ commit 应用线程未指定 → 不仅是正确性竞态（C-FH-2）也是调度循环停顿源；④ 非 layerwise bulk load 是单 recv daemon 线程内的单次同步 `m_store.get`，大命中量会串行化同 rank 其他请求的 Store load。均不构成 P0，但 ①②③ 必须在实现前写死线程模型。**

逐项（对照 AGENTS.md「Tensor item() 操作」「Memory and Performance」规范）：

1. **item()/同步**：设计全文无新增 `.item()`；device 同步点均为继承：send thread `wait_event.synchronize()`（MLC:490，含 ADXL hang 注释 :483-489）、resharding stream（仅 pd_head_ratio>1/quant，Stage 1 普通 Attention 下不触发）。Stage 1 的 `REVERSE_DONE` 全 barrier（§11.2 L1618-1624）是 Scheduler 级门禁，不是 device sync，无 CPU-NPU transfer。合规。
2. **CPU-NPU 搬运与额外拷贝**：无新增。Store bulk load 直写正式 KV blocks（pool_worker.py:840/kv_transfer.py:902 + `token_database` 地址映射）；Forward/Reverse 单边写远端正式 blocks（MLC:497-499）；§3.4 L121-132 禁 Relay Staging/二次搬运与现状一致。
3. **线程模型**：
   - 新增：control transport 每 Engine Scheduler 进程 1 个 receive/poll 线程（§8.7 L755-759）——把网络等待移出调度循环，方向正确；DE 侧 rank-0 worker 的 LKS daemon（继承模式）。线程总量可控。
   - **风险①（确认）**：probe 在 `get_num_new_matched_tokens` 内同步发 REQ 并 `recv()` 无超时（pool_scheduler.py:1106-1108，已复核）——每 DE candidate 一次；LKS 慢/死则整个 DE 调度循环停摆（所有请求）。这是 DualPath 相对现状 Mooncake-only 部署**新增**到 DE 循环的每请求一次阻塞 RPC。C-FH-5 维持 P2。
   - **风险②**：`wait_for_commit(request_key, timeout_ms)`（§8.7 L776-780）全设计无调用者。若实现把它放在 scheduler 循环（等待 PE commit），等价于把决策网络 RT 注入调度循环——必须写死「control 线程持有等待，scheduler 循环只读已持久化 decision」（与 C-PR-3/C-FH-2 的修正合并）。
   - **风险③**：commit 应用若在 control 线程直写无锁 `KVPoolScheduler`（全文件无 `threading.Lock`，mutation points :578/:604-618/:845/:896-915/:951-957 已复核）——除 C-FH-2 的正确性竞态外，若实现者为避险改用「control 线程同步请求 scheduler 循环执行并等待结果」，又引入跨线程停顿。线程模型必须在 §8.7/§20 写死：control 线程只写带锁 coordinator + 投递已提交队列；一切 `KVPoolScheduler`/`load_specs` 触碰（`commit_after_alloc`/`abort_probe`/plan 发射）在 scheduler 循环上下文（如 `build_connector_meta` 开头 drain 队列）执行。
   - **风险④（串行化）**：异步 bulk load 在单 `KVCacheStoreRecvingThread` daemon 内逐请求单次同步 `m_store.get`（kv_transfer.py:846-902），无 DMA 拆分——一个大命中请求（长前缀 × 全层）独占 recv 线程期间，同 rank 后续请求的 Store load 全部排队。继承现状，DE_FULL_HIT/DE_PARTIAL_READ 把它放到关键路径（Reverse 必须等整段 `STORE_DONE`，§14.3 L2031）。建议 §24 增加 store-load 时长/队列深度指标，§25 增加 bulk load 上界或拆分策略说明（见 CP-5）。
4. **决策协议固有延迟**（非违规，记录）：DE 路径在 `Commit+Ack+frozen block plan` 三齐（L571-572）后才启动 Store load，commit→load-start ≥ 1 次 control RT + 1 个 schedule step（metadata 每步发射，`U:scheduler.py:1166-1189`）。属方案固有，应写入 §24 decision latency 分解。
5. **PE_READ sibling 的继承阻塞**：PE 调度循环每 PE_READ 请求一次 sibling lookup RPC（同一无超时 recv，pool_scheduler.py:1107）——继承现状，§3.1 内不可改；写入已知限制与指标（C-FH-5 复核意见同样适用）。

### 维度 20：测试、可观测性与验收标准

**结论：§23 对三条数据通路的 happy path、逐类数据面失败注入、迟到/重复事件、ownership/drain 的覆盖是认真的；但对照本评审已发现问题，测试设计在四个类别上存在系统性缺口（决策面失败传播、跨 Engine 失败/取消、abort-in-window、配置 validator 的 sibling `use_layerwise` 项），且 F11 的回归测试未被显式锚定到 load_async 路径——按现状，测试不足以证明 Stage 1 的核心承诺「提交后失败进入 FINISHED_ERROR 且无效块不晚于终态」。验收标准功能项大体可执行，性能项整体缺席（CP-4 P2 / CP-5 P3）。**

1. **已覆盖（正面）**：失败注入（§23.1 L2594-2604：Store/Reverse/Forward 失败、任一 TP rank、invalid 不晚于 terminal、失败也进 finished_recving、FINISHED_ERROR、drain 前不释放；§23.3 L2638-2641 E2E 注入 4 类）；迟到/重复事件（L2561-2563 tombstone/幂等，L2642 E2E #9）；多 rank 聚合（L2597/L2641 #8）；A″ 专属 first-winner/accounting（§23.2 L2618-2628）；completion 谓词（L2586-2590）。
2. **缺口（对照已发现问题，逐条映射 → CP-4）**：
   - **F11 回归未锚定 load_async**：L2594「Store load 失败」与 L2598「invalid blocks 不晚于 terminal」若实现为同步路径（`load_async=False`）测试即可通过，而真实配置强制 `load_async=True`（L1334）——当前代码在该路径下 invalid 永不上报（F11）。必须显式写「load_async=True + 部分 DMA 失败 → `KVConnectorOutput.invalid_block_ids` 到达 Scheduler → 不 promote → FINISHED_ERROR」，否则 C-PE-1/C-PR-5 修复无验收锚点。L2603 的「load_async=True 时 Store loading ID 能产生内部 STORE_DONE」只测了成功半支。
   - **决策面失败无终态测试**：L2572「重试耗尽进入 DECISION_TIMEOUT」只到错误码；无「DECISION_TIMEOUT → 请求 FINISHED_ERROR / blocks 释放」用例（对应 C-PR-3/C-FH-8，Scheduler→Worker 失败注入通道本身也是设计缺口）。
   - **跨 Engine 失败/取消传播无测试**（C-PE-2/C-PR-2）：PE post-commit 失败 → DE 终态，全 §23 无用例。若该缺口被接受为 Stage 1 已知限制，§25 应显式列出而非隐含。
   - **abort-in-window 无 UT**：§23.3 #10「请求取消与 shutdown drain」是 E2E 级且未分解到窗口（DECIDING/alloc 后 commit 前，C-PE-4/C-FH-1/C-PR-9）；§23.1 无对应 UT。
   - **线程竞态无可测约束**：C-FH-2 的修正（commit 应用只在 scheduler 循环上下文）应先落成设计约束，再加守卫测试（如 `commit_after_alloc`/`abort_probe` 入口断言当前线程为调度循环线程）。
   - **sibling `use_layerwise=false` validator 测试缺失**：§23.2 L2620 只测「Connector 顺序错误」，未覆盖 C-PE-5/C-PR-4 的配置项（§22 L2521 有行为门槛但无配置固定项）。
   - **token 数学三档未指定**：C-FH-3 相关——L2546-2553 的 granularity/full-hit 用例应显式覆盖 P≡0 (mod G)、P≡1 (mod G)、一般情形，且经真实 `KVPoolScheduler` probe 语义（hit==num_tokens 减 1，pool_scheduler.py:526-527），否则 mock probe 会漏掉二次 floor。
3. **验收标准（§25 L2693-2730）可执行性**：「invalid blocks 与 terminal 顺序」（L2720）、「failed finished_recving 不早于 writer/reader terminal」（L2721）、「多 TP rank 只发布一次」（L2725）、「WorkerMetadata 闭环」（L2726）、「无泄漏」（L2727 + §23.3 per-E2E 检查 L2645-2653）均可经 E2E 注入 + 断言执行；「三通路通过」（L2707-2711）依赖 §23.3 检查清单，可执行。**缺席**：全清单无任何性能/时延阈值（probe 上界、decision RT、bulk load 串行化影响、控制面 RT），§26 L2748-2749 有「性能与显存审计」任务但 §25 无对应验收项——与 AGENTS.md「Performance-critical code: Include benchmarks and performance regression tests」不齐 → CP-5（P3）。
4. **可观测性（§24 L2655-2691）**：请求级字段与指标族设计良好（含 quarantine 驻留、stale/duplicate 计数）。缺口：probe RPC 延迟/超时计数（C-FH-5 的直接观测）、control transport 线程活性与队列深度、pending raw terminal inbox 大小/滞留时长、跨 Engine orphan（对端失败/取消通知）计数、store adapter 私有 invalid 集合 drain 计数（F11 修补后的直接健康指标）、sibling 临时 LoadSpec 残留计数（F10）。

---

## 过度设计检查

### SharedMooncakeTransferRuntime（§9.2 L872-909、§10.2 L1467-1512）

六问：

1. **保护哪个 Stage 1 invariant**：(i) `global_te.register_buffer` 进程内单次（守卫：mooncake_transfer_engine.py:31-40，第二次静默跳过——两个 runtime 各自注册会使第二个的 buffer 根本不注册，RDMA 写未注册内存）；(ii) 唯一 send thread + 唯一 recv thread + 唯一 handshake 端口空间（否则 MLC:605-607 同一 `side_channel_port+tp_rank` 公式下第二个 recv 线程 bind 冲突）；(iii) Forward/Reverse 共享同一物理 TE 链路但 wire identity/端口按方向分离（§9.2 L902-903）。
2. **已有模块能否提供**：物理层（buffer 注册 + 一对 send/recv 线程）由 `MooncakeLayerwiseConnectorWorker` 在 `kv_both` 下一次 `register_kv_caches` 即可提供（F7，MLC:1359-1398）——这部分**已有模块完全可提供**；但方向化语义不能提供：`kv_both` 下父 `build_connector_meta`/`start_load_kv` 恒走 consumer 分支（F5/F6），必须新代码绕过。
3. **删除后哪个时序出错**：若删除共享 runtime、Forward/Reverse 各建 Worker 实例：第二次 `register_kv_caches` 的 `register_buffer` 被守卫静默跳过 → Reverse 方向 buffer 未注册 → 传输失败；两对线程 → 端口推导冲突。即「某种形式的单 runtime」不可删除。
4. **Stage 1 必需还是 YAGNI**：单 runtime 约束是 Stage 1 必需（Reverse 是 DE_PARTIAL_READ 主链）。但「新建一个独立 `MooncakeLayerwiseConnectorWorker` 实例包进 runtime」（§10.2 L1472-1476 的 `runtime_worker` 参数形态）不是唯一实现——可简化。
5. **是否引入新状态源或双重记账**：不引入新传输状态源（线程/buffer 仍单份）；若按字面另建 worker 实例，会复制一份 worker 级死状态（SizedDict 远端 metadata 缓存、`request_map`、可能的 `resharding_stream`）——不双记账但冗余；`os.environ`/`global_te` 副作用幂等，无害。
6. **更小替代方案**：`SharedMooncakeTransferRuntime` 直接复用 `DualPathConnectorWorker` 自身（`runtime_worker=self`，或以派生 `kv_both` config 初始化其 `super().__init__` 使 self 即 runtime），省掉第二实例；方向化 adapter（`ForwardLayerwiseTransfer`/`ReverseLayerwiseTransfer`）是真正必需的新增，保留。

**分类：必须保留但应简化。**（保留「单 runtime + 方向化 adapter」；简化掉「第二个 worker 实例」。不因类多判过度设计：DirectionalLayerwiseTransfer 的 Protocol 拆分对应真实的双向语义差异。）

### DualPath Store adapter（§9.7 L1190-1360）

六问：

1. **保护哪个 Stage 1 invariant**：(i) §3.3 不嵌套完整 `AscendStoreConnector`；(ii) probe 临时状态（LookupKeyClient/LoadSpec，F10）与 commit 后 load plan 的生命周期分离——`commit`/`abort` 恰好一次（L1359-1360）；(iii) `load_async=True` 与 `_loading_req_ids` 唯一入口（pool_scheduler.py:617-618）正确接入；(iv) Store 失败统一进入 request-level invalidation（L1347）。
2. **已有模块能否提供**：`KVPoolScheduler`/`KVPoolWorker` 提供 lookup/load/完成机制，但其接口把「lookup 副作用 → LoadSpec 冻结 → metadata 发射 → 完成映射」钉死在 vLLM 调用时序（`get_num_new_matched_tokens` → `update_state_after_alloc` → `build_connector_meta`）上，与 DualPath 的 probe→(commit|abort) 两阶段生命周期不匹配；abort 无公共出口（`load_specs` 清理仅三个 `_process_*` pop，:690/742/845）。**无已有模块可直接提供该生命周期翻译**。
3. **删除后哪个时序出错**：裸持 `KVPoolScheduler`：PE_READ 分支 probe 产生的 LoadSpec 无清理 → `_process_async_load_request`（:839-885）可能在后续步骤把它当真实 load 发射 → 未授权 Store GET（与 §15.2 L2080-2081 直接冲突）；commit 若绕过 adapter 直调 `update_state_after_alloc`，:604-614 assert 的 token 口径与线程归属（C-FH-2）无人管控。
4. **Stage 1 必需还是 YAGNI**：DE Store load 是 DE_FULL_HIT/DE_PARTIAL_READ 主链——adapter 本体必需。**但 `engine_role` 的 `"pe"` 视图（L1206「PE Store view 使用 producer load 语义」）在 A″ 无使用者**（PE Store 归 sibling，L1349；§15.3 中 `pe_store_coverage` 恒 None）——PE 视图分支是 Stage 1 YAGNI，且其存在与 CP-1 的双建冲突直接相关。
5. **是否引入新状态源或双重记账**：probe handle 表（handle_id → LoadSpec/client 引用）是必要新状态，但与 `KVPoolScheduler.load_specs` 形成同一 LoadSpec 的两个引用——所有权仍在 KVPool，非双记账；要求 handle 生命周期与 `load_specs` 清理严格对齐（L1359-1360 已规定恰好一次 + shutdown 兜底）。边缘：`abort_probe` 需触私有 `load_specs` 字典——私有成员依赖，非状态源问题。
6. **更小替代方案**：无（§3.3 禁嵌套完整 connector、§3.1 禁改 KVPool，adapter 是唯一可达形态）。简化点：删除 `build_store_vllm_config` 的 pe 分支与 PE 侧 adapter 创建（与 CP-1 修正合并）。

**分类：必须保留；其中 PE 视图分支 Stage 1 YAGNI（建议删除而非保留——它同时是 CP-1 冲突的来源）。**

---

## Phase 2 候选复核

### C-PE-1（P0 候选：PE sibling load_async 失败上报断裂 → 双 Engine 静默脏数据成功）——**确认 P0，并扩展修复路径分析**

- 本代理逐行复核：pool_worker.py:457-465（recv 线程未注入）、kv_transfer.py:830-831（构造器支持注入）、:908-910/:924-926（私有集合）、:942（无条件 set_finished）、pool_worker.py:1333-1337（只 drain worker 自有集合）。断裂链完整成立，`[当前源码已确认]`。
- 与设计的冲突点复核：§14.1 L1969「Store load 失败…使请求失败」、§15.4 L2133-2135「依赖现有 Scheduler failure gate」、§16.3 L2333-2334「不改变 sibling 既有逻辑」、§22 L2524「不修改既有 Connector」——四者联合使该路径在当前代码下不成立，且 **A″ 自身代码内无修复通路**：DE 侧 adapter 注入路线（见维度 16-3，PT-4 的具体窗口）对 PE sibling 不适用，因为 DualPath 不持有 sibling 的 `KVPoolWorker`，§3.1 又禁止修改/monkey-patch。
- 扩展结论：修复只有两条路——(a) 放松 §3.1，允许对 pool_worker.py:457-465 做一处构造参数级修复（注入 worker 的 invalid 集合/锁；性质是修既有契约 bug 而非为 DualPath 改行为，blast radius 一行调用点）；(b) 接受静默损坏（不可接受）。「强制 sibling `load_async=false`」不可行：同步 bulk get 阻塞 model-runner 线程（pool_worker.py:840）且破坏 §15.4 的 WAITING 语义。维持 P0；任何 Stage 1 门槛（§22）必须把该修复列为前置。

### C-PE-5（P2 候选：sibling `use_layerwise=True` 误配 → can_load 竞争）——**确认 P2，并扩展第二条损坏路径**

- 已复核 pool_scheduler.py:589-593：`num_external==0` 且 `use_layerwise and kvpool_cached>0` → `can_load=True`，与设计 winner 场景（sibling 收 `(empty, 0)`，AMC:41）叠加后非 winner 仍可发起 Worker Store GET。
- 扩展（新论据）：`use_layerwise=True` 还会强制 `load_async` 关闭（pool_scheduler.py:565 第二元素恒 False、:617）——即使不触发 can_load 竞争，PE_READ 的 PE 请求也不再进入 `WAITING_FOR_REMOTE_KVS`，§15.4 L2132 的「Store 成功后 PE 请求才离开等待」整体失效，PE 可能在 Store 未加载的前缀上计算——第二条独立静默损坏路径。
- 修正不变且应加强：config validator 强制 sibling `use_layerwise=false`（写入 §16.2 配置约束 + §22 门槛 + §23.2 测试，见 CP-4）。维持 P2。与 C-PR-4 同根，合并修复。

### C-PE-8（P3 候选：数据面 shutdown 无有界 drain）——**确认 P3，补充队列语义**

- 继承现状属实：MLC send/recv 线程 daemon、无 drain、connector 无 shutdown 覆写（phase1-03 §8）；设计仅 control transport 有界 drain（L782-783），§9.3 L1008-1009 的「drain/quarantine」无对应现有机制。
- 补充：`pd_head_ratio==1` 时 `send_queue` 无界（MLC:251-257，phase1-03 §4）——drain 语义还需定义队列排空/放弃策略与 in-flight `DONE/FAILED` 信号的归属截止点；recv 线程 zmq ROUTER 无关闭流程。进程退出兜底使危害有限。维持 P3。

### C-FH-2（P1 候选：`commit_after_alloc` 执行线程未指定 → 与 scheduler 循环竞态）——**确认 P1，扩展影响面**

- 已复核 `pool_scheduler.py` 全部 mutation points（:578 `_unfinished_requests` 写、:604-618 can_load/`_loading_req_ids` 写、:845 `load_specs` pop、:896-915 finished 清理与 metadata 快照、:951-957 ReqMeta 生成），全文件无锁。设计 §8.5 L667-673 顺序图把 commit→ack→`commit_after_alloc` 画在控制消息流内、§8.7 L755-759 handler 在接收线程——按自然读法竞态成立。
- 扩展：同一缺口还覆盖 `abort_probe`（触 `load_specs`）与 Scheduler 循环对 `coordinator.get_committed()` 的读取——修正应是统一的线程模型条款（写入 §8.7/§20）：control 线程只写带锁 coordinator + 投递已提交队列；一切 KVPoolScheduler 触碰在 scheduler 循环上下文执行（如 `build_connector_meta` 开头 drain 队列），并加线程身份守卫测试（CP-4）。竞态 (a)（字典迭代期写 → RuntimeError → EngineCore 崩溃）与 (b)（`loading_req_ids` 快照错位 → `STORE_DONE` 永久丢失）的时序分析成立。维持 P1。

### C-FH-5（P2 候选：probe zmq REQ `recv()` 无超时挂在 DE 调度循环）——**确认 P2，扩展部署对比**

- 已复核 pool_scheduler.py:1106-1108（`send_multipart` + `recv()` 无超时）、:515-521（循环内 RPC）、LKS 位于 rank-0 worker daemon（ascend_store_connector.py:119-120, 302-325）。
- 扩展：现状 Mooncake-only 部署的 DE 调度循环无此调用——这是 DualPath 新增到 DE 循环的每 candidate 一次阻塞 RPC；PE sibling 的同类阻塞（PE_READ 每请求一次）是继承现状，§3.1 内不可改。修正：adapter probe 加 `RCVTIMEO` + 超时按 miss/unknown 降 PE_READ 候选（§19.1 L2433-2435 的降级语义可覆盖「超时返回」但不能覆盖 hang，需写明）；sibling 侧写入已知限制与 §24 指标。维持 P2。

### C-FH-6（P2 候选：`m_store.get` hang 无 watchdog）——**确认 P2**

- 已复核 kv_transfer.py:902（recv daemon 内单次同步调用，无超时包装）；`m_store.get` 内部是否自带超时 `[外部接口待确认]`。设计 §19 错误分类无 Store I/O 超时码；§13.2 的 terminal 延后规则在 hang 时永不满足，与 C-FH-1 的 abort 链叠加后只能等进程重启——该耦合分析成立。修正（watchdog + quarantine 收敛）合理。维持 P2。

### C-PR-4（P2 候选：同 C-PE-5）——**确认，与 C-PE-5 同根合并**

- 同 C-PE-5 复核。补充本路径特有表述：DE_PARTIAL_READ 下竞争区间是 Store GET（约 `[L_PE,S_PE)`）与 Reverse 已写/在写的 `[L_PE,K_DE)` 重叠，两来源 KV 可能逐位不同 → 比请求失败更糟的静默错 KV。can_load 置位 `[当前源码已确认]`；layerwise 下游是否真对该请求执行 load `[源码可达，尚未运行验证]`——不影响「必须配置准入」的结论。维持 P2。

### C-PR-5（P1 候选：DE Store adapter 强制 `load_async=True` 正踩 F11 断裂）——**确认 P1，扩展具体修补窗口**

- 机制复核同 C-PE-1。本候选的特殊性成立：设计 L1334 强制 `load_async=True`（loading_req_ids 唯一入口 :617-618 所需），即 DE 路径**必然**走在断裂路径上，不是可选配置。
- 扩展（修补可达性升级）：adapter 持有 `KVPoolWorker` 实例且控制 `register_kv_caches` 调用时机（设计 L1299-1302；线程在其中启动，pool_worker.py:743）——调用返回后、首个 `start_load_kv` 前替换 `kv_recv_thread._invalid_block_ids`/`._invalid_block_ids_lock` 为共享对象无竞态（线程阻塞在 `request_queue.get()`）。block→request 归因：失败集合 ∩ plan `store_target` block ids（窗口内 blocks 请求独占）。该机制必须写入 §9.7（L1317/L1347 当前只有目标没有机制）。维持 P1。

### C-PR-7（P2 候选：F15 归因 bug 被 Reverse/Forward 继承）——**确认 P2，扩展第三形态（假阳性 DONE）**

- 已复核 MLC:507：`self.failed_reqs.add(req_id)` 的 `req_id` 是 :469 `for req_id, req_meta in send_task.send_request.items()` 循环残留变量；:519-527 在末层 `chunk_finish` 时按 `req_id in self.failed_reqs` 决定 callback `trans_flag`。
- 扩展（严重性细化）：多请求合并同一 SendTask 是同层并发请求的常态（`save_kv_layer` 每层的 SendTask 含 `metadata.requests` 全部请求，MLC:1810-1819 区域）——失败触发时除候选所述「误杀无辜 + 真失败 hang」两形态外，存在第三形态：**真失败 session 的请求未被标记 → :526 走 `trans_flag=True` → 向接收方发 `DONE_SENDING_MSG` → 接收方把未完整写入的数据当完成接收**（跨 Engine 静默数据损坏 + 双 Engine 状态不一致）。
- 定级理由：触发以传输失败为前提；规避可达（方向化 adapter 自维护 session↔wire-ID 映射，或以每 session 单请求构造 SendTask 牺牲合并收益）；且 §10.3 L1567-1568 的「成功后收到失败=协议错误」规则会把部分误归因放大为可检测错误而非纯静默。维持 P2，但建议 Ledger 记录假阳性 DONE 形态供最终定级参考；修正建议与 C-PR-7 原建议一致，另补「接收侧 DONE 应与 fence 的 layer 计数交叉校验」。

---

## 新发现

### CP-1：PE 侧 Store adapter 与 sibling 的 LookupKeyServer/Store backend 双建冲突（设计未声明 adapter 仅 DE 创建）

- 严重级别：**P2**
- 类型：设计缺口（初始化/资源冲突；兼过度设计来源）
- 证据等级：`[当前源码已确认]`（机制逐行核实）+ `[设计推导]`（设计是否意图 PE 不建 adapter 属文本解读）
- 设计位置：§9.1 L852-861（类图：`DualPathConnectorScheduler *-- DualPathStoreSchedulerAdapter`、`DualPathConnectorWorker *-- DualPathStoreWorkerAdapter`，无角色条件）；§9.3 L991-995（facade `register_kv_caches`「把 KV caches 绑定给 Store adapter」，无角色条件）；§9.7 L1284-1297（worker adapter 创建 `KVPoolWorker` + rank-0 建 `LookupKeyServer`）、L1206（PE Store view 语义）；对照 §9.7 L1349（「以上 adapter 用于方案 A″ 的 DE Store」）
- 源码位置：`ascend_store_connector.py:119-120`（sibling 在 PE rank-0 worker 已建 LKS）、`:291-297`（LKS 构造即 bind REP socket）；`pool_scheduler.py:1115-1130`（ipc path 由 `lookup_rpc_port` 派生，两份 store config 同值即同 path）；`pool_worker.py:249-260`（`m_store` 按 KVPoolWorker 实例创建）、`:742`（每实例 `m_store.register_buffer`）
- 触发前提：A″ PE 部署（Multi 下 sibling AscendStoreConnector 与 DualPathConnector 同进程），DualPath 在 PE 侧也创建 store adapters。
- 具体时序：T0 PE Worker 进程启动 → sibling `AscendStoreConnector.__init__` 建 `KVPoolWorker#1` + `LKS#1` bind `ipc://.../lookup_rpc_port_P_dp_rank0` → DualPath Worker adapter 建 `KVPoolWorker#2`（第二个 backend client）+ `LKS#2` bind 同一 ipc path → bind 冲突或 ipc socket 文件被抢（第一个 server 失联）→ sibling probe RPC 路由不确定；同时 store backend buffer 双倍注册。Scheduler 进程侧同理双 `KVPoolScheduler`（状态无害但冗余）。
- 为什么不正确：同一进程同一 ipc path 两个 REP bind 行为未定义/冲突；双 backend client 与双 buffer 注册违反单资源假设；且 PE 视图在 A″ 无业务使用者（L1349 明示 PE Store 归 sibling；§15.3 的 `pe_store_coverage` 恒 None）。
- 可能影响：PE 启动期 zmq bind 异常（EngineCore 起不来），或 LKS 路由混乱 → probe 失败/错误命中 → 选路错误。
- 与已有模块的兼容性：与 sibling `AscendStoreConnector` 直接冲突。
- 是否属于过度设计：是——PE 视图分支 Stage 1 YAGNI（见过度设计检查）。
- 最小修正建议：设计写明 store adapters 仅 `engine_role=="de"` 创建（PE 侧为 None），facade `register_kv_caches`/`bind_gpu_block_pool`/metadata 的 store 部分按 role 跳过；`build_store_vllm_config` 删除 pe 分支；§9.1 类图的组合关系加角色条件注记。
- 修正后需要增加的测试：PE 启动集成测试（sibling + DualPath 同进程初始化成功、rank-0 仅一个 LKS bind）；config validator UT 拒绝 pe 角色创建 store adapter。
- 置信度：高（机制证据确凿；唯「设计者是否本就打算 PE 不建」无法从文本判定——L1349 暗示 DE-only 但类图/facade 未落实，按缺口立档）。

### CP-2：`DualPathConnector` 未复制父类 `_is_kv_producer` 属性（静默偏差）

- 严重级别：**P3**
- 类型：父类契约偏差（潜伏）
- 证据等级：`[当前源码已确认]`
- 设计位置：§9.3 L922-927（不调父构造器的说明，未列属性复制清单）
- 源码位置：`DPC:120-127`（复制了 `engine_id`/`_connector_metadata`，未复制 `_is_kv_producer`）vs `MLC:701`（父构造器设置 `self._is_kv_producer`）
- 触发前提：任何代码读取 `connector._is_kv_producer`（当前全仓无读取方，phase1-01 §15 已核）。
- 具体时序：T0 读取 → `AttributeError`。
- 为什么不正确：「不调父 `__init__`」模式要求显式复制父构造器的全部对外可见状态；漏一个属性即留潜伏缺口，未来框架/工具升级可能踩中。
- 可能影响：当前无；未来 `AttributeError` 崩溃。
- 与已有模块的兼容性：无现行冲突。
- 是否属于过度设计：否（欠设计）。
- 最小修正建议：`__init__` 中复制一行 `self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer`；§9.3 文档列出「父构造器状态复制清单」。
- 修正后需要增加的测试：扩展现有父签名哨兵 UT（`tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`）断言该属性存在且与 config 一致。
- 置信度：高（事实小、修复小）。

### CP-3：继承 init 从 `kv_transfer_config.kv_port` 派生数据面端口，设计配置未定义 `kv_port` 语义

- 严重级别：**P3**
- 类型：配置缺口（隐性配置消费）
- 证据等级：`[当前源码已确认]`（派生逻辑）+ `[设计推导]`（配置文本缺口）
- 设计位置：§16.1 L2203-2251 / §16.2 L2253-2321（配置模式无 `kv_port` 字段）；§18.2 L2388-2390（「派生 kv_both runtime config，本地 receive port 按角色选择」——未写明落在哪个配置字段）
- 源码位置：`MLC:809-812`（scheduler 从 `kv_port` 派生 `side_channel_port`）、`MLC:1164-1169`（worker 同）、`MLC:605-607`（recv 线程实际监听 `side_channel_port + tp_rank`）；`U:config/kv_transfer.py:56`（`kv_port` 默认 14579）
- 触发前提：实现者按 §16 配置（只设 `forward.base_port`/`reverse.base_port`/`control.pe_port`）构造派生 runtime config，未设置 `kv_port`。
- 具体时序：T0 初始化 → 派生 runtime worker 的 `side_channel_port` 由默认 14579 派生 → recv 线程监听错误端口 → handshake 无法到达或对端连到非预期端口空间；top-level `DualPathConnectorScheduler` 的 `super().__init__` 同样消费 top-level `kv_port`（派生值在覆盖 dispatch 后多半不被使用，但静默错误配置难排查）。
- 为什么不正确：继承的初始化器消费一个设计配置模式里不存在的字段，端口空间隔离（§16.1 L2247-2249 的硬性校验对象）因此缺一角。
- 可能影响：方向端口冲突或 handshake 失败；排障困难（默认值 14579 静默生效）。
- 与已有模块的兼容性：与 §16.1 的端口不重叠校验目标冲突。
- 是否属于过度设计：否（欠设计）。
- 最小修正建议：§16 写明派生 runtime config 的 `kv_port` = 角色对应方向端口（PE=`reverse.base_port`，DE=`forward.base_port`）；top-level `kv_port` 在 DualPath 下或弃用（fail-fast 提示）或强制显式且不与三方向端口空间重叠；validator 覆盖。
- 修正后需要增加的测试：端口派生 UT（PE recv=reverse port、DE recv=forward port、互不重叠校验）。
- 置信度：中高（若实现者自明派生 config 必须带 `kv_port` 则无碍；文本未写即按缺口立档）。

### CP-4：测试设计对本评审已发现缺口存在四类系统性未覆盖 + F11 回归未锚定 load_async

- 严重级别：**P2**
- 类型：测试设计缺口（验收能力）
- 证据等级：`[当前源码已确认]`（被测机制）+ `[设计推导]`（测试文本缺口）
- 设计位置：§23.1 L2531-2616（UT 清单）、§23.2 L2618-2628（A″ 专属）、§23.3 L2630-2653（E2E）、§25 L2693-2730（验收）
- 源码位置：F11 断裂点（pool_worker.py:457-465; kv_transfer.py:908-910,942）；决策面（§8.7 L810-811）；abort 框架依赖（`U:scheduler.py:2144-2147, 2162, 2580-2586`）；sibling 配置（pool_scheduler.py:565, 589-593）
- 触发前提：按 §23 现状编写测试。
- 具体时序：T0 测试通过（同步路径 Store 失败 UT 绿）→ T1 真实配置 `load_async=True`（L1334）→ T2 部分 DMA 失败 → invalid 永不上报（F11）→ 请求照常完成——测试体系无任何红灯。同理：DECISION_TIMEOUT 后无终态用例、PE post-commit 失败无跨 Engine 用例、DECIDING 窗口 abort 无用例、sibling `use_layerwise=True` 无 validator 用例。
- 为什么不正确：Stage 1 核心承诺是「提交后失败进入 `FINISHED_ERROR` 且 invalid 不晚于终态」（§2.1 L40-41、§25 L2718-2721），而直接验证该承诺的四个失败类别（F11 回归、决策面传播、跨 Engine 传播、abort-in-window）均无对应用例。
- 可能影响：关键修复（C-PE-1/C-PR-5/C-FH-2 等）无验收锚点，回归不可见。
- 与已有模块的兼容性：—（测试维度）。
- 是否属于过度设计：否（欠设计）。
- 最小修正建议：§23.1 增补——(i) load_async=True + 部分 DMA 失败的 invalid 全链路断言（到达 Scheduler、不 promote、FINISHED_ERROR）；(ii) DECISION_TIMEOUT → 请求级终态与 blocks 释放；(iii) DECIDING/alloc 窗口 abort 的终态合成与释放；(iv) `commit_after_alloc`/`abort_probe` 线程身份守卫；§23.2 增补——sibling `use_layerwise=false` validator、P≡0/1 (mod G) 全 hit 判定（经真实 probe 语义）；§23.3 #10 分解取消窗口；跨 Engine 失败传播或入测或在 §25 列为显式已知限制。
- 修正后需要增加的测试：即上述条目本身。
- 置信度：高（缺口与设计文本逐条对照）。

### CP-5：验收标准无性能项（与 AGENTS.md perf 回归要求不齐）

- 严重级别：**P3**
- 类型：验收标准缺口
- 证据等级：`[设计推导]`
- 设计位置：§25 L2693-2730（无性能阈值）；§26 L2748-2749（有「性能与显存审计」任务无标准）；对照 §24 L2679-2688（有指标定义但无验收绑定）
- 源码位置：被测性能点——probe 阻塞 RPC（pool_scheduler.py:1106-1108）、单线程串行 bulk load（kv_transfer.py:846-902）、决策 RT（L571-572 三齐门）
- 触发前提：Stage 1 验收评审。
- 具体时序：T0 全部功能项通过 → T1 probe 平均/尾部耗时、单 bulk load 对 recv 线程的阻塞时长、commit→data-start 延迟无任何上界断言 → 性能退化（如 LKS 慢、大命中串行化）在验收中不可见。
- 为什么不正确：probe 与决策协议在请求关键路径上（每 candidate 一次阻塞 RPC + ≥1 RT + 1 step），AGENTS.md 要求 perf-critical 代码含 benchmark/回归测试。
- 可能影响：性能回归无门禁。
- 与已有模块的兼容性：—。
- 是否属于过度设计：否（欠设计）。
- 最小修正建议：§25 增加最低性能验收：probe p99 上界、decision commit RT 上界、单 bulk load 时长/队列深度上界或拆分策略说明、控制面线程活性检查；绑定 §24 已定义指标。
- 修正后需要增加的测试：NPU 基准用例（benchmarks/ 下）覆盖上述三项阈值。
- 置信度：中。

### 正向结论（成立原因 + 前提 + 源码契约）

- **PT-1：AMC 广播语义与设计 §4.1/§15.2/§15.4 精确一致，正常路径 sibling 生命周期可由框架现有机制承载。** 成立原因：first-winner 守卫（`U:multi_connector.py:397-399`）、AMC 分叉（`AMC:32-41`）、hook 无条件广播（:289-309）、`get_finished` 并集 + `_extra_async_saves` 门控（`AMC:97-98`）、delay_free 或语义（`U:multi_connector.py:479-518`）逐条核实；None 短路的 F3 残留形态在「DualPath=child0」拓扑下不可达。前提：sibling `use_layerwise=false`（C-PE-5 修正）+ AMC 覆写语义不随 upstream 升级漂移（F18 watch）。
- **PT-2：不调父 `__init__` 的构造模式可行且已被当前 foundation 验证。** 父构造器仅 5 项职责（MLC:697-712），DPC:120-142 已按同模式运行且注册链路完整；`KVConnectorBase_V1.__init__` 直接调用与父路径等价。前提：复制清单补 `_is_kv_producer`（CP-2）。
- **PT-3：`kv_both` 单次 `register_kv_caches` 完成 buffer + 单 send + 单 recv（MLC:1359-1398），共享 runtime 前提成立**；`register_buffer` 单次守卫（mooncake_transfer_engine.py:31-40）使「只调一次」成为硬约束，设计 §9.2/§10.2 的约束方向正确。
- **PT-4：DE adapter 的 F11 修补窗口具体存在。** transfer 线程在 `KVPoolWorker.register_kv_caches` 内启动（pool_worker.py:743），adapter 控制该调用时机（设计 L1299-1302），可在首个请求前无竞态注入共享 invalid 集合/锁（构造器原生支持，kv_transfer.py:830-831）。前提：接受私有属性依赖 + 写明 block→request 归因。
- **PT-5：F17（`request_finished_all_groups` tracker-None 不对称）在 DE adapter 配置下不可达。** `kv_consumer && !consumer_is_to_put` 提前返回（pool_scheduler.py:1045-1047，已复核），§16.3 L2327 关闭 consumer put/save 与之咬合。PE sibling 侧的 F17 是既有边界，非 A″ 引入（并入 C-PE-4 覆盖）。
- **PT-6：性能面合规项。** 设计未引入新 `.item()`/CPU-NPU 搬运/额外 HBM 拷贝（§3.4 与现状一致）；Store load 强制异步在 daemon 线程（kv_transfer.py:902）；控制面独立 receive/poll 线程方向正确（§8.7 L755-759）；Stage 1 的 REVERSE_DONE 全 barrier 是 Scheduler 级门禁而非 device sync。

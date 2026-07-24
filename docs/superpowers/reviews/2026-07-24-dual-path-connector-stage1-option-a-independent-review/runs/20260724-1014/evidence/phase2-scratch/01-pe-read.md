# Phase 2 时序审查：PE_READ（方案 A″ / outer_first_winner）

- 审查路径：`PE_READ`。设计 snapshot：`inputs/option-a-detailed-design.snapshot.md`（行号以此为准）。
- Phase 1 证据：`evidence/01-current-control-flow.md`（F1/F2/F3/F10/F11/F13 等）与 `evidence/phase1-scratch/02,04,05`，均直接引用；个别关键点由本人复核原文（另行标注）。
- 基线：vllm-ascend `dev/dualpath @ 0ec11a47`；upstream `vllm @ 8df14cfc`（v0.23.1rc0-1050，配套 v0.23.0 有小幅偏差）。
- 证据等级：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[设计推导]` / `[外部接口待确认]` / `[事实冲突]`。
- 顺序性质三分：**框架保证**（upstream 控制流/assert 强制）、**当前实现偶然**（当前代码碰巧如此、无契约）、**设计假设**（snapshot 承诺、当前代码不存在）。
- 符号：`R = P-1`（decode-ready 前缀）；`L_DE/L_PE` 本地已有；`S_DE/S_PE` Store 对齐命中；`K_PE = min(max(L_PE,S_PE),R)`；`H_PE = max(S_PE-L_PE,0)`；`E_DE = max(R-L_DE,0)`（snapshot L259-282）。

## 时序必答问题速查

| 问题 | 答案（详见 T 链与场景矩阵） |
|---|---|
| 谁做 probe | DE：DualPathStoreSchedulerAdapter 的**独立 probe 接口**（禁止复用 `KVPoolScheduler.get_num_new_matched_tokens`，snapshot L583-586）；PE：外层既有 `AscendStoreConnector` sibling 自己的 lookup（非纯 probe，F10 的副作用被设计接受，L177-179） |
| 谁决定路径 | **PE 是唯一提交方**：`PathDecisionCoordinator.decide_on_pe()`（L501, L697-703），在 PE DualPath 首次 `get_num_new_matched_tokens` 内同步完成（见场景 S13） |
| 哪一刻算 committed | PE Scheduler 线程内 CAS `DECIDING→COMMITTED`（L705-709）成功即本地 committed；`PATH_DECISION_COMMIT` 经 control transport 发 DE，DE 持久化后回 `DecisionAck`（L568）；**数据面启动要求 Commit+Ack+frozen block plan 三者齐**（L571-572，但 PE Store sibling 不受此门，见缺陷 D5） |
| 每个 Engine 声明多少 external tokens | DE（单 connector）：`E_DE = R - L_DE`，与最终路径无关（L288, L470-472）；PE（Multi）：DualPath 在 PE_READ 候选下返回 `0`，由 AscendStore sibling 声明 `H_PE`（L291, L2078-2079, L2103-2105） |
| 谁分配正式 blocks | 各 Engine 自己的 vLLM Scheduler（kv_cache_manager），`delay_cache_blocks=load_async`；DE blocks 是路径无关的最终目标 blocks（L1965） |
| 各 token 区间谁写谁读 | PE blocks：`[L_PE,K_PE)` 由 sibling Store recv 线程写、`[K_PE,R)` 由 PE 模型计算写；DE blocks：`[L_DE,R)` 由 Forward（PE send 线程单边写、DE recv 线程落盘）写；DE forward 读 `[0,R)` 计算最后一个 token；Forward 读 PE blocks `[L_DE,R)` |
| 哪些操作可能并发 | DE 等 commit（控制面）∥ PE Store load（数据面）；PE 逐层 compute ∥ 已算层的 Forward 发送（逐层流水，L1636）；DE recv ∥ PE send；Multi 内 child hook 是**同线程顺序**广播而非并发（U:multi_connector.py:289-309） |
| 哪个事件使请求离开等待 | PE：聚合后的 `finished_recving`（sibling AscendStore 的 done_recving）→ promote（U:scheduler.py:2534-2541）；DE：聚合后的 `finished_recving`（DualPath 的 FORWARD_DONE terminal）→ promote |
| 谁发布 finished_recving / finished_sending | PE：finished_recving 由 sibling AscendStore 每 rank 发布、Multi 并集、KVOutputAggregator 全 rank 计数门控；finished_sending 由两 child 各自 drain 后经 `_extra_async_saves` 计数门控发布。DE：两者都由 DualPath 按 §13.2/§13.4 规则发布（当前继承父类行为不满足，必须重写，F8/F9） |
| 多 rank 汇聚成功/失败 | 框架 `KVOutputAggregator`：finished 需全部期望 rank（world_size==TP size，§17 L2340-2343）；invalid 并集无 quorum，任一 rank 失败即失败（F12，与 L1814 一致） |
| 取消/超时/部分失败的 drain、quarantine、释放 | 设计：§13.2（invalid 不晚于失败终态、terminal 一次）、§13.5/§19.2（drain→quarantine→ownership 归零→failed finished_recving→释放）；框架前提：abort 也触发 `request_finished`、WAITING 默认 delay_free、connector 不报 finished 即泄漏（F13）。**缺口见缺陷 D2/D4** |

## T 时序主链

主链取最典型形态：DE 有请求需远程 prefill；`E_DE > 0`；静态策略/准入裁定 `PE_READ`；PE Store partial hit（`H_PE > 0`）；单 KV group、PP=DP=1、TP=n（§17 L2338-2348）。变体见场景矩阵。

- **T0｜请求到达 DE**｜执行者：框架/API 层→DE EngineCore｜进程：DE EngineCore 主线程｜请求带远程 prefill 语义的 `kv_transfer_params` 进入 waiting。`[当前源码已确认]`（现有 MLC 链路即如此，01-current-control-flow §0）｜顺序：框架保证。
- **T1｜DE 本地查询 + E_DE 计算 + Store probe**｜执行者：DE Scheduler → DE DualPathConnectorScheduler｜同线程｜框架先做本地 prefix cache 查询得 `L_DE`（仅 `num_computed_tokens==0` 分支，U:scheduler.py:724,774-779）→ DualPath 计算 `E_DE = R - L_DE`；`E_DE==0` → 返回 `(0,False)`，candidate 流程终止（L301-302, L555-556）｜否则经**独立 probe 接口**做 DE Store probe 得 `StoreCoverage` + probe handle，记录 pending candidate，返回 `(E_DE, True)`（L557-559）。`[框架调用点当前源码已确认；probe/candidate 为设计推导]`｜顺序：框架保证查询位置；probe 同步发生在该调用内是设计假设。注意 DE 不等 decision 即返回 matched tokens（L580-581），这是四阶段协议解耦的核心。
- **T2｜DE 分配正式 blocks + 冻结 manifest + 派发 PE 请求**｜执行者：DE Scheduler → DE DualPath｜同线程｜`allocate_slots(num_external=E_DE, delay_cache_blocks=True)`（U:scheduler.py:942-954）→ `update_state_after_alloc(request, 全部已分配 blocks, E_DE)`（U:scheduler.py:969-974，本人复核原文）→ DualPath 冻结 DE block manifest（路径无关最终目标 blocks，L1965），**不启动 Store/P2P I/O**（L561-562）→ 经 proxy/metaserver 派发**截断到 R、max_tokens=1** 的 PE 模型请求，`CoverageProposal` 序列化在 `kv_transfer_params["dual_path"]` envelope（L528-530, L562-564, §8.3 L601-615）→ 请求置 `WAITING_FOR_REMOTE_KVS`，`num_computed_tokens` 乐观置为 `L_DE + E_DE = R`（U:scheduler.py:986-1005）→ DE control transport 开始 `wait_for_commit`（超时+重试，L776-784, L810-811）。`[分配/WAITING 当前源码已确认；冻结/派发/wait_for_commit 设计推导；metaserver 派发路径外部接口待确认（现 MLC 在 update_state_after_alloc 内 POST metaserver，MLC:928-998，可复用但 DualPath 侧实现不存在）]`。
- **T3｜PE 接收模型请求**｜执行者：proxy → PE EngineCore｜PE 侧生成本地 request ID；`pe_engine_local_request_id` 与 `DualPathRequestKey` 的映射建立（L345-370）；`on_new_request` 钩子可解析 envelope（U:base.py:529-535, U:scheduler.py:2088-2089）。`[框架钩子当前源码已确认；envelope 解析设计推导]`。
- **T4｜PE 首次 connector 查询：decide + commit + 返回 0**｜执行者：PE Scheduler → AscendMultiConnector → child0 DualPath｜PE EngineCore 主线程｜框架先做本地 prefix 查询得 `L_PE`，作为调用参数传入（U:scheduler.py:774-779）。DualPath 读 envelope 中 proposal：不完整 → 返回 `(None, False)`，Multi 整体短路（U:multi_connector.py:393-394），请求放回 skipped 队列**下一轮重查**（U:scheduler.py:781-787，本人复核）；完整 → 同步执行 `decide_on_pe(proposal, pe_local_tokens=L_PE, pe_store_coverage=None)`：静态规则（L436-441）+ 准入（L443-468）裁定 `PE_READ` → `commit()` CAS 本地落 `COMMITTED_PE`（L626, L705-709）→ 经 control transport 向 DE `send_commit`（L766-769）→ 按 §15.3 返回 `(0, False)`（L2103-2105）。**返回 0 时 decision 已本地 commit，但 Ack 未必要到**（Ack 异步）。`[调用点/None 重试/短路当前源码已确认；decide/commit 设计推导]`｜关键：decision 不需要 PE Store 结果（§7.2 规则只用 DE coverage + L_PE + 准入），因此 first-winner 顺序（DualPath 先于 Store 查询，L2068-2072）与 decision 协议自洽。
- **T5｜PE Store lookup（sibling 成为 winner）**｜执行者：AscendMultiConnector → child1 AscendStoreConnector｜同线程｜同一 Multi pass 内继续调用：`KVPoolScheduler.get_num_new_matched_tokens` 非 layerwise 走 zmq REQ 到 rank-0 worker 的 LookupKeyServer（pool_scheduler.py:515-521），得 `S_PE_raw`→对齐 `S_PE`；`H_PE>0` → 创建 `LoadSpec`（:549-553）→ 返回 `(H_PE, load_async=True)`（:565）；Multi first-winner：DualPath 返回 0，故 AscendStore 成为 winner，写 `_requests_to_connector[req]=1`（U:multi_connector.py:397-399）。`[当前源码已确认]`｜注意：lookup 的 REQ `recv()` **无超时**（pool_scheduler.py:1106-1108），同步阻塞 PE 调度循环——继承现状，PE_READ 每请求一次（F10）。
- **T6｜PE 分配 + update_state_after_alloc 分叉**｜执行者：PE Scheduler → AscendMultiConnector｜同线程｜`allocate_slots(num_external=H_PE, delay_cache_blocks=True)` 只覆盖 `[0,K_PE)`（load_async 不分配新计算量，U:scheduler.py:834-837,942-954）→ `update_state_after_alloc(request, 全部 blocks, H_PE)`（U:scheduler.py:969-974）→ AMC 分叉（AMC:32-41）：winner AscendStore 收 `(真实 blocks, H_PE)` → `can_load=True` + `_loading_req_ids.add`（pool_scheduler.py:616-618）；**非 winner DualPath 因 `isinstance(MooncakeLayerwiseConnector)` 例外同样收 `(真实 blocks, H_PE)`**（AMC:36，F1）——此 `H_PE` 是 sibling 的 Store tokens，不是 DualPath 自己声明的量；DualPath 只可据此记录 `[0,K_PE)` 源 block 视图，不得视为路径授权（L2125）。请求 → `WAITING_FOR_REMOTE_KVS`，`num_computed_tokens` 乐观 = `K_PE`（U:scheduler.py:986-1005）。`[当前源码已确认 + 设计推导]`｜顺序：框架保证。
- **T7｜DE 收 commit、持久化、Ack、收敛 probe**｜执行者：DE control transport 接收线程 → DE DualPath Scheduler｜DE EngineCore（独立 poll 线程 + 调度线程）｜收 `PATH_DECISION_COMMIT` → coordinator 校验 version 并持久化（L812）→ 回 `DECISION_ACK`（L771-774）→ 按 probe 收敛表对 DE probe handle 执行 `abort_probe()`（L1355）→ 构建 Forward-receive plan，经下一步 `build_connector_meta` 随 SchedulerOutput 广播到 DE Worker（L1028-1032；框架通道 U:scheduler.py:1166-1168 → U:multiproc_executor.py:310-320）。`[框架 metadata 通道当前源码已确认；协议动作设计推导]`｜与 T8-T10 **并发**（不同进程）。
- **T8｜PE Worker 绑定与 start_load_kv 广播**｜执行者：框架 mixin → Multi → 两 child｜PE 每 rank Worker 进程 model-runner 线程｜`bind_connector_metadata`（zip 分发，U:multi_connector.py:260-266）→ `start_load_kv` **无条件顺序广播**（U:multi_connector.py:289-291）：sibling AscendStore 见 `can_load` 且 req ∈ `loading_req_ids` → `kv_recv_thread.add_request` 异步 bulk get（pool_worker.py:787-791）；DualPath 读已提交 decision=`PE_READ` → **不启动 Store、不启动 recv**，仅登记 plan/fence（L2122, §9.5 L1074-1078）。`[广播与 sibling 行为当前源码已确认；DualPath 门控设计推导]`｜顺序：框架保证（mixin:89-95）。
- **T9｜PE Store load 执行**｜执行者：sibling 的 `KVCacheStoreRecvingThread`｜PE Worker 进程 daemon 线程｜每请求一次 bulk `m_store.get`（kv_transfer.py:846-902），写入 PE 正式 blocks `[L_PE,K_PE)`；完成 → `set_finished_request(req_id)`（kv_transfer.py:942）。失败分支见场景 S8（**F11 断裂**）。`[当前源码已确认]`｜与 T7、T12 之前各步并发。
- **T10｜PE finished_recving 聚合与 promote**｜执行者：PE Worker（model-runner 线程）→ Multi → KVOutputAggregator → PE Scheduler｜每步 `get_finished` 无条件执行（mixin:102-104）：sibling 返回 `done_recving ∩ loading_req_ids`（pool_worker.py:1551-1555），Multi 并集（U:multi_connector.py:321），聚合器倒计数至全 TP rank（U:utils.py:78-90）→ Scheduler：`finished_recving` → `_try_promote_blocked_waiting_request` → `cache_blocks(K_PE)`，状态回 WAITING（U:scheduler.py:2517-2541）。`[当前源码已确认，框架保证]`。
- **T11｜PE tail 调度与第二次 update_state_after_alloc**｜执行者：PE Scheduler｜同线程｜promote 后请求经 waiting→running：`num_computed>0` 走 else 分支（U:scheduler.py:822-827），`allocate_slots` 补 `[K_PE,R)` tail blocks → **第二次** `update_state_after_alloc(request, 全量 blocks, 0)`（U:scheduler.py:969-974；F4 双调用契约 U:base.py:495-504）。DualPath 须按 `request_key+decision_version` 幂等（L681-684）——**但 §8.5「只在 num_external_tokens>0 时冻结」与 tail block 获取存在张力，见缺陷 D3**。`[框架当前源码已确认]`。
- **T12｜PE 逐层 compute + 提交 Forward**｜执行者：PE Worker model-runner 线程（compute）→ DualPath save hook → 共享 runtime send 线程｜forward 计算 `[K_PE,R)`；每层 `save_kv_layer`（upstream 装饰器包 unified attention，F14 的 MLA 例外在 Stage 1 普通 Attention 限制下不适用，L2346）→ DualPath 按 committed plan 提交 `ForwardLayerCommand(layer, region=[L_DE,R))`（L946-949, L1457-1465）：本地段 `[L_DE,L_PE)` 读 PE 已缓存块、Store 段 `[max(L_DE,L_PE),K_PE)` 读 T9 已加载块、计算段 `[K_PE,R)` 读本层刚算的 KV。时序安全依据：Store 是**整段 bulk load**（L1668），promote（T10）之前 forward 不会开始（框架 WAITING 门禁），故任一层的 save 钩子触发时三段源数据全部就绪。`[装饰器/调度门当前源码已确认；整 region 提交语义设计推导]`。
- **T13｜Forward 传输与逐层 terminal**｜执行者：PE send 线程 →（Mooncake 单边写）→ DE recv 线程 → DE Worker fence｜PE/DE Worker 进程 daemon 线程｜逐层写 DE 正式 blocks `[L_DE,R)`；receiver terminal 是写入完成的权威证明（L1569）；逐层 `FORWARD_LAYER_DONE` 更新 `RankTransferFence`（L1588-1598）；raw terminal 早于 mapping 时进 pending inbox 暂存重新归属（L576-577, L1556-1561；父类丢弃缺陷 F8 的重写动机）。`[Mooncake 机制当前源码已确认；fence/inbox 设计推导]`｜与 T12 后续层流水并发。
- **T14｜DE FORWARD_DONE → finished_recving → promote**｜执行者：DE Worker（get_finished，model-runner 线程）→ KVOutputAggregator → DE Scheduler｜PE_READ 成功谓词 = 仅 `FORWARD_DONE`（L1755-1756, L1968）；每 rank 对精确 DE local ID 发布一次（L1808-1819）→ 全 rank 计数门控 → DE Scheduler promote：`cache_blocks(R)`（U:scheduler.py:2517-2541）。`[聚合/promote 当前源码已确认；DualPath 终态生成设计推导——当前继承的父类 get_finished 语义（F8）必须重写]`。
- **T15｜DE 末 token 计算；PE 请求收尾**｜DE：`num_computed=R=P-1`，本地重算最后一个 prompt token 并采样（§4.4 L216-227），进入正常 decode。PE：模型请求 `max_tokens=1`，产 1 token 即 finish → `_connector_finished` → Multi `request_finished_all_groups`：DualPath 若仍有 in-flight Forward send 须 `delay_free=True`（§13.4 L1865-1877，覆盖父类恒 False，F9）；sibling 作为 kv_producer 的 Store save（put）亦可能 delay_free（pool_scheduler.py:1031-1037）→ 多 async-save child 由 `_extra_async_saves` 计数门控（AMC:97-98），全部 drain 后每 rank 发 `finished_sending` → 聚合 → PE Scheduler 释放延迟 blocks（U:scheduler.py:2583-2586）。`[框架/兄弟侧当前源码已确认；DualPath delay_free 设计推导]`。
- **T16｜DE 请求终结与资源释放**｜执行者：DE Scheduler/Worker｜请求正常结束 → `request_finished_all_groups`：有 sender/reader owner 则 delay_free，ownership 归零且无 unknown in-flight 后发一次 `finished_sending`（L1928-1933）；wire ID 退休留 tombstone（L378-379）。`[框架钩子当前源码已确认；ledger 设计推导]`。

主链闭合性：**正常路径（无失败、无取消）下时序闭环**——两个 Engine 的 accounting、等待解除、终态发布、延迟释放各有框架锚点；Forward 区间 `[L_DE,R)` 的三段源在框架 WAITING 门禁 + bulk Store load 前提下时序安全。失败/取消路径存在缺口，见缺陷 D1/D2/D4。

## 场景矩阵

### S1　L_DE = 0（DE 无任何本地前缀）
`E_DE = R`；T1 probe、T2 分配全量 external blocks；Forward 区间 `[0,R)`。DE 本地段为空，三段构成退化为「Store 段 + 计算段」。结论：**闭合**。注意 DE blocks 全部 delay_cache，promote 前不进 prefix cache（U:kv_cache_manager 语义，scratch-05 §2.2），与「分配先于 decision 但不启动 I/O」一致。

### S2　L_DE > 0（DE 有部分本地前缀）
`E_DE = R - L_DE`；DE 本地 `[0,L_DE)` 可能被多请求共享（prefix cache）；Forward 必须只写 `[L_DE,R)`，不得覆盖共享前缀——block manifest 边界由 T2 冻结保证。`L_DE ≥ R`（`E_DE=0`）→ T1 直接 `(0,False)`，纯本地 decode，candidate 不创建（L301-302）。结论：**闭合**。

### S3　L_PE 与 K_PE 各关系下 Forward `[L_DE,R)` 的构成
| 关系 | PE 本地段 | Store 段 | 计算段 | 触发/门 |
|---|---|---|---|---|
| `L_DE < L_PE`（典型，§7.3 L468 的转 PE_READ 条件） | `[L_DE,L_PE)` 读 prefix cache | `[L_PE,K_PE)` | `[K_PE,R)` | 本地段 commit+Ack+binding 后即可发；Store 段被 T10 promote 门挡；计算段随层钩子 |
| `L_DE ≥ L_PE` | 空 | `[L_DE,K_PE)` | `[K_PE,R)` | 同上 |
| `S_PE ≤ L_PE`（Store 无新增，`H_PE=0`） | `[L_DE,L_PE)` | 空 | `[L_PE,R)` | 见 S6：无 winner，无 WAITING 阶段 |
| `K_PE = R`（PE Store full hit） | `[L_DE,L_PE)` | `[L_PE,R)` | promote 后重算 `[R-1,R)`（U:scheduler.py:2519-2522；lookup 侧 hit==num_tokens 减 1，pool_scheduler.py:526-527） | Store 段全量门挡 |

结论：**构成闭合**——设计 L311-316, L1966-1967 明确三段且禁止以 `[K_PE,R)` 替代；时序安全依赖「Store 为整段 bulk load + 框架 WAITING 门禁」两个前提（均成立）。唯一未写明的是**整 region 提交点**：`save_kv_layer` 文档（L946-949）暗示每层钩子提交整个 `[L_DE,R)` 的该层切片，但本地段/Store 段并非当次 forward 计算的 token，block ids 来源（T6 记录 + attn_metadata）未在协议中显式化——见缺陷 D3。

### S4　PE Multi 中 DualPath 返回 0 后 AscendStore 成为 winner 的完整 accounting
T4 DualPath `(0,False)` → T5 sibling `(H_PE,True)` → winner=sibling，`_requests_to_connector[req]=1` → Multi 返回 `(H_PE,True)` → 一次且仅一次 Scheduler accounting（U:multi_connector.py:397-399；`to_return[0]==0` 守卫防覆盖）→ T6 分叉：sibling 得 `(real,H_PE)` 入 `_loading_req_ids`；DualPath 经 Mooncake 例外得 `(real,H_PE)` 但无 accounting 效果 → T10/T11 promote 后第二次 `update_state_after_alloc(...,0)`：sibling 幂等（pool_scheduler.py:589-602），DualPath 幂等（L681-684）→ 请求结束 winner 映射 pop（AMC:100）。结论：**闭合**（正向结论 P-1/P-4）。两个附带事实：sibling 的 LoadSpec 在 winner 路径会被 `_process_async_load_request` 正常消费（pool_scheduler.py:845）；Multi 的 None 短路不会残留 winner 映射——DualPath 是 child0，返回 None 时 sibling 该轮根本未被调用（U:multi_connector.py:393-394），F3 的残留形态在此拓扑不可达。

### S5　非 winner DualPath（Layerwise 子类）拿到真实 blocks 时发生什么
发生：T6 中 DualPath 收 `(真实 blocks, H_PE)`（AMC:36，F1）。设计约束：真实 blocks ≠ 路径授权（L2125）；DualPath 须完全覆盖父类 `update_state_after_alloc`（L905-908），不落回父类 consumer-first dispatch——否则父类逻辑会按 `num_external>0` 把请求入 `_reqs_need_recv`（MLC:928-998 语义），产生错误的接收计划。**不应发生**：启动 Store/Reverse/recv、把 `H_PE` 当作自己的 coverage。结论：**可闭合但脆弱**——正确性完全依赖 DualPath 覆写的纪律性与 committed decision 查询；这正是 §23.2「Layerwise real blocks 不会绕过 decision gate」测试（L2624）必须覆盖的。另注意版本偏差 F18：若 upstream 升级到「非 winner 收真实 blocks+0」语义（U:multi_connector.py:410-412），AMC 覆写行为漂移需重新核对。

### S6　PE Store lookup miss（`H_PE = 0`）
T5 sibling 返回 `(0,False)`（need_to_allocate≤0 不建 LoadSpec，pool_scheduler.py:545-547）→ Multi 整体 `(0,False)`，**无 winner** → 请求作为普通本地请求同步调度：`allocate_slots` 全量 prompt → `update_state_after_alloc(request, blocks, 0)`（本人复核 U:scheduler.py:969-974 无条件调用）→ AMC：无 winner（chosen=-1），sibling 收 `(empty,0)`，DualPath 收 `(real,0)`（AMC:36-41）。此时：decision=PE_READ **已 commit**，DE 在等 `[L_DE,R)` 的 Forward；PE 将在本地逐 chunk 计算 `[L_PE,R)`。问题：(a) 按 §8.5 字面规则（L681-684）`num_external=0` 不冻结 block plan，PE 侧 Forward 源 mapping 无协议内来源；(b) 无 WAITING 阶段，T10 门不存在，但 Store 段为空所以不需要门；(c) Forward 只能由 save 钩子 + attn_metadata 驱动。机制上可行（计算段覆盖全部缺失区间），但设计未规定此形态。结论：**设计未闭合**（并入缺陷 D3）。DE 侧不受影响（仍等 FORWARD_DONE）。

### S7　PE Store lookup partial / full
partial（`H_PE>0, K_PE<R`）：主链 T5-T15，闭合。full（`S_PE ≥ R`）：`K_PE=R`，promote 时 `num_computed=num_tokens-1` 重算末 token（PE 请求已截断到 R，U:scheduler.py:2519-2522），tail 仅 1 token；lookup 侧 `hit==num_tokens` 减 1（pool_scheduler.py:526-527）与之自洽。结论：**闭合**。

### S8　Store load 失败（F11 对 PE_READ 的影响）——请求卡住还是 FINISHED_ERROR？
**都不是：当前代码下请求静默"成功"**。[当前源码已确认] sibling 在 `load_async=True`（§15.4 L2132 的 WAITING 语义所必需）下：recv 线程未被注入 worker 的 `_invalid_block_ids`（pool_worker.py:457-465），失败块进线程私有集合（kv_transfer.py:908-910,924-926），无任何 drain 方；`get_block_ids_with_load_errors` 只 drain worker 自己的集合（pool_worker.py:1333-1337）→ **`KVConnectorOutput.invalid_block_ids` 永远为空**；同时请求被无条件 `set_finished_request`（kv_transfer.py:942）→ T10 照常 promote → PE 在脏前缀上计算 tail → T12/T13 把脏 KV Forward 到 DE → 双引擎正常完成。§14.1「Store load 失败使请求失败」（L1969）、§15.4 步骤 3「Multi 汇总 invalid blocks…按 fail 结束 PE 请求」（L2133-2134）、§19.2 失效流程在此全部失效。若改配 sibling `load_async=false`：同步 bulk get 在 model-runner 线程执行，失败块可入 worker 集合（pool_worker.py:846-847,861-862）→ invalid 上报 → fail 策略 FINISHED_ERROR [当前源码已确认]——但同步阻塞引擎步骤，且与 §15.4 的 WAITING 叙述矛盾。结论：**设计不闭合**，见缺陷 D1（本评审最重发现）。

### S9　Store load 成功后 Forward 失败
T9/T10 成功，PE promote 并计算；T13 中某层失败：DE 侧检测（recv 线程失败事件/对端错误）→ DualPath 按 §19.2：冻结 failed → 收集并发布 invalid blocks → DE Scheduler（fail 策略）截断 + `FINISHED_ERROR`（U:scheduler.py:1578-1586,1823-1825）→ drain/quarantine → ownership 归零 → 发布 failed `finished_recving`（L1794-1799, L2440-2452）→ 释放 blocks（U:scheduler.py:2580-2582）。PE 侧：请求已正常结束（max_tokens=1），send 侧失败仅影响 drain/telemetry（L1569-1570）；延迟 blocks 经 `finished_sending` 释放。结论：**设计闭合**——前提：DualPath 完整实现 §13.2/§19.2 终态机（当前父类 F8/F9 语义必须重写）且失败**能被检测到**；对端静默死亡无事件时检测无着落，见缺陷 D2。

### S10　winner 与非 winner child 的 hook 广播下，DualPath 如何知道不启动 Store
机制：Worker hook 无条件广播（U:multi_connector.py:289-309，F1 节）；DualPath 每个 hook 先读已提交 `PathDecision`（L2121）——Scheduler 进程内的 coordinator 状态经 `DualPathConnectorMetadata.plans`（含 PathDecision，L1028-1032）随框架通道到 Worker，Worker 侧不需要跨进程查 coordinator；PE_READ 下 DualPath 的 Store adapter 无 plan（PE Store 由 sibling 承担，L1349），`store_metadata=None`（L1328-1330），无 `StoreLoadCommand` 即无 `start_store_load`。sibling 反向同理：DE 路径时 sibling 收 `(empty,0)`，非 layerwise 下 `can_load=False`、不进 `_loading_req_ids`（pool_scheduler.py:589-618，F2），Worker 无 GET。结论：**闭合**（依赖：AMC 对非 Mooncake 非 winner 发 empty blocks 的 v0.23.0 语义不漂移，F18；sibling 非 layerwise，见缺陷 D7）。

### S11　probe / alloc / commit / Forward 各阶段取消
- probe 阶段（T1-T2 之间 DE abort）：请求尚在 waiting 未入 WAITING_FOR_REMOTE_KVS；abort → `finish_requests` → `_connector_finished`（U:scheduler.py:2162，F13）；pending candidate 与 probe handle 须清理——设计仅有 shutdown 时 abort 所有 pending handles（L1359-1360），**请求级 abort 的 handle 回收未规定**（小泄漏面，并入 D4）。
- alloc 后 / DECIDING（T2 后 DE abort）：请求在 WAITING_FOR_REMOTE_KVS，delay_free 默认 True（U:scheduler.py:2144-2147）→ blocks 释放依赖 connector 日后发布 `finished_recving`；**设计未规定 abort 如何合成该终态**（§13.5/§19.2 只覆盖失败路径）→ 若未处理即泄漏（F13）。同时 PE 请求已派发且无 CANCEL 控制消息（envelope 仅 4 类，L790-795）→ PE 空跑 + Store 白加载（有界浪费：max_tokens=1）。结论：**不闭合**，见缺陷 D4。
- commit 后 / Forward 前（DE abort）：PE 因 Ack 未达或 plan 取消而不启动 Forward（L571-572 门 + L2136-2137 清理）；DE 侧同样缺 abort 终态合成。不闭合（D4）。
- Forward 中（DE abort）：in-flight 写入落 DE blocks → quarantine，drain 后 failed terminal → 释放（§19.2 L2444-2452 机制可复用，但触发源是 abort 而非子操作失败，设计未显式接通）。部分闭合。
- PE 侧取消（PE 请求被 abort/shutdown）：WAITING 中 → sibling recv 线程仍会完成并 `set_finished` → 释放闭环 [当前源码已确认 F13 链路]；但 **DE 侧无人通知**，DE 等 FORWARD_DONE 无超时 → 滞留，见缺陷 D2。

### S12　shutdown 时 in-flight Forward
引擎 shutdown → `finish_requests(None, FINISHED_ABORTED)`（U:core.py:1355-1357）→ 各请求走 S11 取消路径；connector `shutdown()`：先拒新请求、drain/quarantine、关 Store/双 P2P（L1008-1009）；control transport 有界 drain（L782-783）；unknown in-flight 仅在「源进程死亡/TE session 失效/target fence 确认」或明确 terminal 后解除 quarantine（L1571-1573）。缺口：**数据面（send/recv 线程）无有界 drain 语义**——继承的 Mooncake 线程是 daemon 无 drain（scratch-01 §11），shutdown 窗口内 in-flight 层的归属依赖进程退出兜底。进程退出场景危害有限。结论：**基本闭合**，残余见缺陷 D8。

### S13　特别追问 1：PE 侧 DualPath 返回 0 时 PathDecision 是否已 commit？decision 协议与 first-winner 查询先后是否自洽？
- 前提校正：「PE Scheduler 只在 `num_computed_tokens==0` 时调用一次 connector」不准确——是**每个 schedule pass 都调用**，直到返回非 None 且请求离开 waiting（U:scheduler.py:774-787，None → skipped → 下轮重查；preemption 清零后会重查，U:scheduler.py:1204）。`[当前源码已确认，本人复核]`
- 时序：proposal 随请求 envelope 原子到达（L528-530）；`L_PE` 仅在 schedule pass 的本地 prefix 查询后作为调用参数可得；因此 decide 只能发生在首次（或 proposal 补齐后的首个）`get_num_new_matched_tokens` 内；§15.3 的签名（L2085-2106）要求 decision 作为输入——**返回 0 当且仅当 decision==PE_READ 已本地 commit**；返回 `(None,False)` 仅在 proposal 未齐时。commit→DE 的发送与 Ack 是异步后续。
- 自洽性：**自洽**。decision 不依赖 PE Store 结果（§7.2 规则输入为 DE coverage + L_PE + 准入），故「DualPath 先被查询、返回 0、Store 后查询」的 first-winner 顺序（L2068-2072）不构成循环依赖；`decide_on_pe` 的 `pe_store_coverage` 参数在 A″ 恒为 None（L1349 sibling 承担）。唯一未写明：decide+commit 同步发生在该调用内这一放置本身是隐含的（§8.5 图 L661-662 仅示「obtain L_PE and decide」），且 proposal 等待无超时（见 D6）。
- 结论：**协议自洽（正向结论 P-1）**，建议设计显式声明同步放置点与 None 重试预算。

### S14　特别追问 2：PE 侧 finished_recving 由 sibling 发布、Forward 完成由 DE 侧 DualPath 发布，两 Engine 完成契约是否各自闭合？
- PE：`sibling done_recving → Multi 并集 → KVOutputAggregator 全 rank → promote`（T9-T10），请求离开 WAITING 后正常计算、正常结束；结束时双 child 的 delay_free 由 `_extra_async_saves` 计数门控（T15）。**正常路径闭合**。**失败路径不闭合**：F11 使 sibling 永不报 invalid（S8/D1）。
- DE：`FORWARD_DONE → 每 rank finished_recving → 聚合 → promote`（T13-T14）；失败路径依赖 DualPath 自建的 §13.2 终态机（invalid 不晚于 failed terminal、terminal 仅一次，匹配框架 assert U:scheduler.py:2576/2581/2585）；结束时 delay_free/`finished_sending` 按 §13.4。**正常路径与可检测失败路径闭合（设计推导，实现不存在）**；**对端静默失败无检测与通知**（D2）。
- 两契约互相独立（各自 Engine-local ID，L1817-1819），无跨 Engine 原子完成器——这是 A″ 的明示取舍（L2128-2141），其代价正是 D1/D2 两个缺口。

## 缺陷候选

### D1　PE sibling（AscendStoreConnector）load_async 失败上报断裂使 PE_READ「Store 失败即失败」契约不成立，双引擎静默脏数据成功
- 严重级别建议：**P0**｜类型：失败语义/时序断裂（当前代码缺陷被设计路径原样继承）｜证据等级：**[当前源码已确认]**
- 设计位置：§14.1 要求 L1969；§15.4 步骤 3 L2133-2135；§15.5 时序 L2158；§16.3「不改变 sibling 既有逻辑」L2333-2334；§22「不修改既有 Connector」L2524；§9.7 sibling 承担 PE Store L1344-1349
- 源码位置：`pool_worker.py:457-465`（recv 线程未注入 invalid 集合/锁）；`kv_transfer.py:908-910,924-926`（失败块入线程私有集合）；`kv_transfer.py:942`（无条件 `set_finished_request`）；`pool_worker.py:1333-1337`（只 drain worker 自有集合）；F11（01-current-control-flow §8/§16）
- 触发前提：A″ PE sibling 配置 `load_async=True` 且非 layerwise（§15.4 的 WAITING 语义、T5 winner accounting 所必需）；Store `m_store.get` 部分或全部失败
- 具体时序：T9 失败 → invalid 永不上报 + 请求无条件标完成 → T10 照常 promote（Scheduler 认为 `K_PE` 已加载）→ T12 在脏前缀上计算 → T13 Forward 把 `[L_DE,R)` 脏 KV 写 DE → T14 双引擎正常完成
- 为什么不正确：违反 §14.1 硬性要求与 §19.2 失效流程；违反 upstream「失败 block 不得晚于 finished 上报」契约（U:base.py:383-392）；结果是**端到端静默数据损坏**，比 hang 更严重
- 可能影响：PE_READ 路径任何 Store 故障 → 错误推理结果，无任何信号
- 最小修正建议：修复 sibling 失败上报——构造 `KVCacheStoreRecvingThread` 时注入 worker 的 `_invalid_block_ids` 与锁（对齐同步路径 pool_worker.py:846-847）；若坚持「不改既有 Connector」，则 A″ 必须强制 sibling `load_async=false` 并改写 §15.4 的 WAITING 叙述（代价：同步 bulk get 阻塞 model-runner 线程）。两者须在 §16.2 配置约束与 §22 门槛中固化一条
- 置信度：高（源码直接证据 + 设计文本直接冲突）

### D2　commit 之后 PE→DE 无请求级失败/取消通知，DE 等 FORWARD_DONE 无超时，PE 侧任何 post-commit 失败使 DE 请求永久滞留
- 严重级别建议：**P1**｜类型：协议缺口/时序（缺失检测与通知边）｜证据等级：**[设计推导]**（缺口基于 snapshot 文本穷举）+ 框架无 WAITING 超时 **[当前源码已确认]**
- 设计位置：§8.7 envelope 仅 `COVERAGE_PROPOSAL/PATH_DECISION_COMMIT/DECISION_ACK/CONTROL_ERROR` 四类（L790-795），`CONTROL_ERROR` 语义未定义为请求级失败通知；§8.7 超时只覆盖 proposal/commit/ack（L810-811）；§14.1 DE 成功条件仅 FORWARD_DONE（L1968）；§13/§19 无 transfer 等待超时
- 源码位置：U:scheduler.py:2517-2541（离开 WAITING 唯一条件是 finished_recving，无超时分支）；当前 MLC 同样无 receive 超时（scratch-01 §11，继承性背景）
- 触发前提：T7 commit+Ack 完成后，PE 侧发生：Store load 失败（D1 修复后 PE 请求 FINISHED_ERROR）、PE 请求 abort/shutdown、PE 进程崩溃、Forward send 静默失败（无事件）
- 具体时序：T12/T13 永不发生且无失败事件 → DE 请求永远停在 `WAITING_FOR_REMOTE_KVS` → blocks 因 delay_free 永不释放（F13）→ 用户请求 hang
- 为什么不正确：§19 失败分类自称完备但缺少「已 commit 请求的跨 Engine 失败传播」边；DECISION_TIMEOUT 只保护 commit 前
- 可能影响：任何 PE 侧 post-commit 故障 → DE 请求泄漏 + KV blocks 泄漏
- 最小修正建议：(a) 定义 `CONTROL_ERROR`（或新增 CANCEL/FAIL 消息）承载 `request_key + error_code`，PE 在请求终态（失败/abort）且存在已 commit plan 时发送；(b) DE 侧为 FORWARD 等待加超时/心跳（fence 时间戳 + 超时进 §19.2 失败流程）
- 置信度：高（缺口存在性）；中（「完全无任何兜底」——PE 进程崩溃可能经 TE session 失效被间检测，L1571-1573，但该机制只解 quarantine 不产生请求终态）

### D3　§8.5「只在 num_external_tokens>0 时冻结真实 block plan」与 PE 侧两段式 block 分配冲突，PE_READ Forward 源 `[K_PE,R)`（及 miss 时全部）block mapping 无协议内来源
- 严重级别建议：**P2**｜类型：协议自相矛盾/会计规则适用范围未声明｜证据等级：**[当前源码已确认]**（框架调用语义）+ **[设计推导]**（冲突解读）
- 设计位置：§8.5 L681-684；§8.2 步骤 5/10-11（L561-562, L569-572）；§9.3 `save_kv_layer` L946-949；§15.5 时序缺 block 绑定环节（L2156-2159）
- 源码位置：U:scheduler.py:942-954（load_async 首次只分配 local+external=`K_PE`）；U:scheduler.py:822-827 + 969-974（第二次调用 `num_external=0`、blocks 覆盖 `[0,R)`，本人复核）；F4（双调用契约）
- 触发前提：PE_READ 且 `H_PE>0`（tail blocks 只在第二次调用出现）；或 PE Store miss（S6：所有调用 `num_external=0`，字面规则下永不冻结）
- 具体时序：T6 第一次调用（real blocks, H_PE>0）→ 按字面规则冻结，但 blocks 只覆盖 `[0,K_PE)`；T11 第二次调用（real blocks, 0）→ 规则要求幂等不冻结 → Forward 源 `[K_PE,R)` 的 block ids 无冻结来源；miss 时更无第一次冻结
- 为什么不正确：§8.5 规则是按 DE 侧「一次分配全覆盖」写的（DE `E_DE` 统一总量 L470-472），未区分 Engine 侧；PE 侧 block 信息天然分段到达
- 可能影响：若实现严格遵循字面规则 → Forward tail 段无 mapping → 绑定失败 → 请求 `FINISHED_ERROR`（L473-474）或 plan 不完整导致 hang；实现者各自补救则行为不一
- 最小修正建议：把冻结规则改写为分侧语义：「Store target/DE manifest 仅在 `num_external>0` 的首次调用冻结；PE Forward source 允许在后续调用（含 `num_external=0`）与 `save_kv_layer` 的 `attn_metadata` block table 增量扩展，按 `request_key+decision_version` 幂等」；或要求 `build_connector_meta` 每步携带最新 block slice
- 置信度：高（文本与框架语义冲突客观存在）；实际可修复性高，故 P2

### D4　DECIDING/alloc 阶段 DE 请求 abort：终态合成责任与 probe handle 回收未规定，delay_free blocks 可泄漏；已派发 PE 请求无取消通道
- 严重级别建议：**P2**｜类型：取消语义缺口｜证据等级：框架行为 **[当前源码已确认]**（F13，U:scheduler.py:2162,2144-2147,2580-2586）；设计缺口 **[设计推导]**
- 设计位置：§13.5 L1902-1933 与 §19.2 L2440-2452 仅覆盖失败路径；§9.7 仅 shutdown 时 abort pending handles（L1359-1360）；§8.7 无 CANCEL 消息（L790-795）；§20 仅 shutdown quarantine（L2474）
- 触发前提：DE 请求在 T1 之后、FORWARD_DONE 之前被 abort（客户端取消/超时）
- 具体时序：abort → `finish_requests` → WAITING 默认 delay_free → `_connector_finished` 调 `request_finished_all_groups` → 此后**只有 connector 发布该 ID 的 finished_recving，blocks 才释放**（U:scheduler.py:2580-2582）；设计未规定 abort 触发的 failed terminal 合成时机与 drain 范围；同时 PE 模型请求继续空跑（Store load + 1 token 计算），无 CANCEL 可达
- 为什么不正确：F13 明示「connector 永不上报 finished 即泄漏」；§13.2 的终态机以「失败」为触发，abort 不是子操作失败
- 可能影响：取消场景 KV blocks 与 connector 状态泄漏；PE 侧有界资源浪费
- 最小修正建议：显式规定 abort → `coordinator.fail(request_key, ABORTED)` 复用 §19.2 全流程（禁新操作→drain/quarantine→ownership 归零→failed finished_recving）；pending probe handle 在 `request_finished*` 回收；评估是否向 PE 发 CANCEL（与 D2 合并设计）
- 置信度：中高

### D5　§8.2 步骤 11「Commit+Ack+frozen block plan 三者齐才启动 Store load」与 A″ sibling 持有 PE Store 生命周期矛盾；Ack 等待点未声明
- 严重级别建议：**P3**｜类型：设计内部不一致（门控适用范围未声明）｜证据等级：**[设计推导]**
- 设计位置：§8.2 L571-572；§9.7 L1349（PE Store 由 sibling 承担）；§14.1 L1963；§15.4 L2131-2138
- 触发前提：PE_READ 正常路径
- 具体时序：T8 sibling 在框架 metadata 到达即启动 Store load——它不知 decision/Ack 存在；若 Ack 永不到达（DE 在 T4 后死亡），PE 已完成 Store load 甚至 compute，仅 Forward 被（未声明位置的）Ack 门挡住
- 为什么不正确：§8.2 的门被写成普适（「Store load、Reverse 或 Forward」），A″ 的 sibling 委托使其对 PE Store 不可执行；浪费有界（max_tokens=1）但语义含糊
- 最小修正建议：§8.2 显式声明：A″ PE_READ 的 PE Store load 按框架既有语义启动、不受 Commit+Ack 门控；Ack 门仅约束 Forward/Reverse/DE Store，并指明 Ack 等待的实现位置（如 plan binding 条件）
- 置信度：高（文本冲突）；影响低（无正确性风险，仅浪费与语义含糊）

### D6　PE 侧 proposal 等待无超时：envelope 缺失/版本不兼容时请求永久滞留 waiting
- 严重级别建议：**P3**｜类型：协议健壮性缺口｜证据等级：**[设计推导]** + 框架重试语义 **[当前源码已确认]**（U:scheduler.py:781-787）
- 设计位置：§8.2 L579-581（允许 `(None,False)` 重试，无预算）；§19.1 无对应错误码挂载点
- 具体时序：T4 每 pass 返回 None → Multi 短路 → 请求反复 skipped，永不调度也永不失败
- 最小修正建议：proposal-wait 加预算（次数或时长），耗尽按 `DECISION_PROTOCOL_ERROR` fail 请求（经 invalid+failed terminal 通道）
- 置信度：中（触发需 envelope 异常，属协议错误类；但后果是永久滞留）

### D7　A″ 未强制 PE sibling `use_layerwise=false`：误配时 DualPath winner 场景非 winner Store child 仍可 `can_load=True` 发起 Worker Store GET，与 Reverse 写并发竞争同一 PE blocks
- 严重级别建议：**P2**｜类型：配置校验缺口（跨路径数据竞争风险，从 PE_READ 共享拓扑观察到）｜证据等级：**[当前源码已确认]**（F2：pool_scheduler.py:589-593 layerwise 下 `num_external=0` 仍置 `can_load=True`）
- 设计位置：§15.2 L2080-2081（lookup 阶段不得启动 Store load）；§22 门槛 L2521（要求验证不触发 GET）但未列配置固定项；§16.2 未固定 sibling `use_layerwise`
- 具体时序（DE_PARTIAL_READ 形态）：DualPath 返回正数成 winner → sibling 收 `(empty,0)` → layerwise 下 `can_load=True` → Worker `process_layer_data` 发起逐层 load（pool_worker.py:754-755,1288-1315）→ 与 Reverse 写 `[L_PE,K_DE)` 物理并发写同 block 区间
- 为什么不正确：§15.2 的保证只在非 layerwise 成立（F2：非 layerwise `_loading_req_ids` 入口要求 `num_external>0`）
- 最小修正建议：A″ config validator（L2291-2292, L2523）强制 PE sibling `use_layerwise=false`（与 §16.2 DE 派生 store 配置 L2317 一致）
- 置信度：中高（机制证据确凿；触发需误配，属配置契约类）

### D8　数据面 shutdown 无有界 drain：in-flight Forward 在 shutdown 窗口的归属依赖进程退出兜底
- 严重级别建议：**P3**｜类型：生命周期语义未规定｜证据等级：**[设计推导]** + 继承现状 **[当前源码已确认]**（Mooncake send/recv 线程 daemon、无 drain，scratch-01 §11）
- 设计位置：§9.3 shutdown L1008-1009；§8.7 仅 control transport 有界 drain（L782-783）；§20 L2474
- 最小修正建议：为 send/recv 线程定义有界 drain 或明确的「shutdown 即失败」语义，并与 §10.3 quarantine 解除条件（L1571-1573）对齐
- 置信度：中；影响低（进程退出兜底）

### 继承性负担（非新设计缺陷，记录备查）
- **B1**：sibling lookup 的 zmq REQ `recv()` 无超时（pool_scheduler.py:1106-1108），T5 同步阻塞 PE 调度循环；LookupKeyClient/LoadSpec 在 abort 路径泄漏（F10）。设计要求 DE 侧 probe 独立隔离（L583-586），但 PE 侧 sibling lookup 原样继承。
- **B2**：MLC 发送线程失败归因 bug（MLC:507 用残留循环变量，F15）——Forward 复用共享 runtime send 线程（§9.2 L892-901）将继承该 bug，可能影响 §19 失败归类与 §13.2 终态的正确归属。
- **B3**：F18 版本偏差——upstream v0.23.0 后 MultiConnector 三处语义变更（非 winner 真实 blocks、params merge、has_pending_push_work），AMC 覆写停留在 v0.23.0；S5/S10 的结论依赖 AMC 当前语义不漂移。

## 正向结论（为什么正确 + 依赖前提 + 源码契约）

- **P-1　decision 协议与 first-winner 顺序自洽**（场景 S13）：proposal 随请求 envelope 原子到达；`L_PE` 在首次 schedule pass 以调用参数可得；decision 不依赖 PE Store 结果（§7.2 L436-441）；`(None,False)` 重试由框架保证（U:scheduler.py:781-787；U:base.py:472-474）；返回 0 ⇒ 已本地 commit（§15.3 L2103-2106）。依赖前提：DualPath 在该调用内同步执行 decide+commit（设计隐含，建议显式化，D5/D6 为周边缺口）。
- **P-2　Forward 区间 `[L_DE,R)` 三段构成时序安全**（场景 S3）：Store 为整段 bulk load（L1668）且框架 WAITING 门禁保证 promote 前无 forward（U:scheduler.py:834-837,986-1006,2534-2541）→ 任一层 save 钩子触发时本地段/Store 段/计算段源数据全部就绪；DE 侧 `E_DE` 统一声明使目标 blocks 路径无关、一次分配（L470-472, L1965）+ `delay_cache_blocks` 保证 promote 前不进 prefix cache。
- **P-3　多 rank 汇聚与终态不变式可直接复用框架**：finished 全 rank 计数门控、invalid 并集无 quorum（U:utils.py:78-90,159，F12）与 §13.3 L1806-1819 一致；invalid 先于 finished 消费（U:scheduler.py:1578-1586）支撑「invalid 不晚于失败终态」；`finished_recving` 不变式由 assert 强制（U:scheduler.py:2576/2581/2585），§13.2「同一 ID 一次 terminal、失败也经 finished_recving」正是其充要条件。
- **P-4　A″ accounting 只计一次且 winner 语义清晰**（场景 S4/S5）：first-winner 守卫（U:multi_connector.py:397-399）+ AMC 分叉（AMC:32-41）+ 双调用幂等契约（U:base.py:495-504，F4）使 DualPath 返回 0 → sibling winner 的链路只产生一次 Scheduler accounting；None 短路残留 winner 映射（F3）在「DualPath 为 child0」拓扑下不可达。
- **P-5　PE 请求结束时双 child 的异步持有可由框架表达**：`_extra_async_saves` 计数门控（AMC:97-98；U:multi_connector.py:503-504,325-334）保证 DualPath Forward drain 与 sibling Store save drain 都完成后才发布 `finished_sending`、释放延迟 blocks（U:scheduler.py:2583-2586）——A″ 不建跨 sibling 原子完成器（L2128-2141）在正常路径不损失正确性。
- **P-6　worker 侧「不启动 Store」的判定通道闭合**（场景 S10）：hook 广播是事实（U:multi_connector.py:289-309），但 decision 经 `DualPathConnectorMetadata.plans` 随框架 metadata 通道到 Worker（L1028-1032；U:scheduler.py:1166-1168），无需跨进程查 coordinator；sibling 非 layerwise 时 `(empty,0)` 保证无 Worker Store GET（pool_scheduler.py:589-618，F2）。

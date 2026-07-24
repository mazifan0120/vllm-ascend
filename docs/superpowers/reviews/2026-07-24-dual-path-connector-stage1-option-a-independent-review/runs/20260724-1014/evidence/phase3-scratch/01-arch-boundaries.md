# Phase 3 维度评审 01：架构边界（目标/配置/first-winner/职责/不修改边界/生命周期兼容）

- 评审代理：Phase 3 架构边界维度代理。负责维度：§9-1（目标/非目标/硬边界一致性）、§9-2（配置表达拓扑）、§9-3（first-winner 是否只做 accounting）、§9-4（Scheduler/Connector/Worker/Store/P2P 职责边界）、§9-17（不修改既有 Connector/upstream 硬边界）、§9-18（配置/初始化/握手/shutdown/版本兼容）；§10 过度设计检查（PathDecisionCoordinator、DualPath control transport、wire identity 与 tombstone）；复核候选 C-PE-6、C-PE-7、C-FH-7、C-FH-8、C-FH-9、C-PR-3、C-PR-10。
- 设计行号 = `inputs/option-a-detailed-design.snapshot.md`；源码基线 vllm-ascend `dev/dualpath @ 0ec11a47`、upstream `vllm @ 8df14cfc`（v0.23.1rc0-1050，配套 v0.23.0 有小幅偏差）。
- 行号约定：`MLC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`；`AMC` = `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`；`U:` = upstream 仓库；`DPC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`；F1-F18 指 `evidence/01-current-control-flow.md` §16。
- 证据等级：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[设计推导]` / `[外部接口待确认]` / `[事实冲突]`。本机无 NPU/运行时，全部静态阅读。

---

## 维度结论

### 维度 1：Stage 1 目标、非目标和硬边界是否互相一致（§2/§3/§22）

**结论：文本层面逐项自洽；存在三处「目标 vs 边界」张力，其中一处（失败语义 vs 不修改边界）是实质缺陷（→ AB-8），两处为门槛未固化。**

逐项核对（一致项）：

- §2.2 L51「不做 Store 逐层读取」↔ §4.3 L211-212（禁止 layerwise Store 路径用于反向链路）↔ §9.7 L1204（派生 config 固定 use_layerwise=False）↔ §16.2 L2317（DE store.use_layerwise: false）——一致。
- §2.2 L53/L121-132（禁 Relay Staging/中转 HBM）↔ §3.4 KV 内存边界 ↔ §10.1 L1426（正向不覆盖 Store 前缀、无二次搬运）——一致。
- §2.2 L55（不修改上游/既有 Connector）↔ §3.1 L73-80 ↔ §22 L2524 ↔ §25 L2699——一致。
- §2.2 L48-49（无运行时 Value Function/LinkMonitor 切路）↔ §7.2 L428（仅静态策略）↔ L59-60（shadow 不得影响决策）↔ §24 L2690-2691（shadow_* 指标隔离）——一致。
- §3.5 两阶段提交（L136-148）↔ §7.3 L462-468（提交前转 PE_READ 清单）↔ §8.4 状态机（L621-637）↔ §20 L2467-2469（CAS/幂等/协议错误）——一致。
- `kv_load_failure_policy: fail` 要求（L152-156、L2244）可落地：该字段是 upstream `KVTransferConfig` 顶层字段、默认即 `"fail"`（`U:vllm/config/kv_transfer.py:69`）[当前源码已确认]；DualPath 在 init 读取并 fail-fast 属 dual_path 内新代码，不越界。当前 foundation 未实现该联动（DPC 无此字段解析，`phase1-scratch/01` §4），属「尚未实现」而非缺陷。
- §22 门槛与 §15/§16 的拓扑、配置契约逐项对应（L2519-2525），「任一失败不得进入实现」（L2527）是过程门槛，与 §2 无冲突。

张力项（不一致/未固化）：

1. **（实质，→ AB-8，P2）** §2.1 L40-41「任一必要传输失败 → FINISHED_ERROR」与 §3.1 L73-80「不修改 AscendStore 既有 Worker/backend」在 F11 下不能同时成立：KVPoolWorker 的 load_async 失败上报当前是断裂的（`pool_worker.py:457-465` 未注入 invalid 集合；`kv_transfer.py:908-910,942` 失败块进私有集合且请求无条件置完成）[当前源码已确认]。§3.3 L112-119 把组合前提写成「既有组件提供的**完整功能**」，而异步失败上报不是完整功能。F15（MLC:507 归因 bug）同型。设计全文未声明绕行手段。
2. **（门槛未固化，交叉 C-PE-5/C-PR-4）** §22 L2521「DualPath winner 场景下 AscendStore 临时 lookup 状态不会触发 Worker Store GET」仅在 PE sibling 非 layerwise 时成立（F2：`pool_scheduler.py:589-593` layerwise 下 `(empty,0)` 仍置 can_load；非 layerwise 下 `:617-618` 要求 num_external>0）[当前源码已确认]。§2/§3/§16 均未把 sibling `use_layerwise=false` 钉为硬边界。
3. **（目标达成度提示，不判缺陷）** §2.1 L42「完整定义……数据面、状态机和完成契约」与 Phase 2 已立档的消息面缺口（C-PR-1 通道、C-PR-2 跨引擎失败传播）存在差距；目标本身合理，差距记入维度 4 与候选复核。

### 维度 2：PE/DE Connector 配置是否能真实表达设计拓扑（§16 全部 + §17；当前 config.py 缺口与注册机制）

**结论：拓扑可以被真实表达，注册机制完整承载且无需新增公开名；配置面有 1 个 P2 双源冲突（cache_transfer_granularity，→ AB-1）和 4 个 P3 缺口（AB-5/AB-6/AB-9 + sibling 配置未固定）。**

已验证的承载能力（正向，`[当前源码已确认]`）：

- PE 侧 `MultiConnector -> [DualPathConnector, AscendStoreConnector]`：由 upstream MultiConnector 的 `connectors` 列表承载（`U:multi_connector.py:217-236`）；每个 child 用 `KVTransferConfig(**ktc, engine_id=...)` 独立构造，**kv_role 不继承外层**（`:222-227`）——设计 §16.2 L2288 的声称属实。
- 注册链路：`DualPathConnector` 已注册（`vllm_ascend/distributed/kv_transfer/__init__.py:57-61`）；`"MultiConnector"` 被 pop 后以 `AscendMultiConnector` 同名替换（`:21-27`）——§16.2 L2286-2287 声称属实。§21 L2512-2513「复用现有注册、不新增第二个公开名」与现状一致。
- 角色交叉校验：§16.2 的 `role: pe/de` + `kv_both` 与现有 `_validate_role`（`DPC:195-210`）兼容（kv_both 同时满足 producer/consumer 要求）。
- DE 侧 `store:` 嵌套配置（L2316-2321）经派生扁平 config 注入 KVPool 的路线**构造器级可行**：`KVPoolScheduler(vllm_config, use_layerwise, kv_cache_config, page_size_bytes)`（`pool_scheduler.py:48-54`）与 `KVPoolWorker(vllm_config, use_layerwise, kv_cache_config)`（`pool_worker.py:81-86`）均可独立构造；二者读取扁平 extra_config——`consumer_is_to_load`（`pool_scheduler.py:87-88`）、`load_async`（`pool_worker.py:133`，默认 False）、`consumer_is_to_put`（`pool_worker.py:136`）真实存在；`use_layerwise=True` 会强制关闭 load_async（`pool_scheduler.py:565/617`），故设计 L1204 同时固定两者是正确的。backend 在 KVPoolWorker 构造器内初始化（`pool_worker.py:105`），属组合而非修改，符合 §3.3。
- PE 侧不需要 store adapter（L1349），不会出现同进程双 backend。

缺口：

1. **（P2，→ AB-1）** `cache_transfer_granularity: 256`（§16.1 L2213）是配置值，而 KVPool 的 G 是**内部推导值**（`pool_scheduler.py:119` `self.cache_transfer_granularity = self._infer_cache_transfer_granularity()`，`:404-413` = lcm(block sizes, cache family granularity)；`pool_worker.py:158` 同）——全 KVPool 无任何读取该配置项的代码（grep 全目录仅见推导与使用）。§5.2 的 token 数学（L272-276 的 `S_DE/S_PE`）若用配置 G 而 Store 用推导 G，覆盖区间与 accounting 错位，还可能在 `pool_scheduler.py:604` 的 assert 处崩溃。§16.2 两个具体示例反而都不含该键。设计未说 G 的运行时来源。
2. **（P3，→ AB-5）** 当前 `DualPathConfig.from_extra_config`（`DPC:147-192`）只读已知键、**静默忽略未知键**——按 §16 schema 写的配置在今天会被静默丢弃；§16 未规定 unknown-key fail-fast。
3. **（P3，→ AB-6）** §16.1「公共配置约束」示例不是任一 Engine 的合法配置：缺必填 `role`（`DPC:154-159`）、缺 DE 必需的 `store:`、且顶层 `kv_connector: DualPathConnector` 与 §15.1 的 PE Multi 拓扑矛盾。
4. **（P3，→ AB-9）** 现有 `path_strategy.type` 默认 `"value_function"`（`DPC:100,213-234`），与 §7.2 L428「Stage 1 仅静态策略」未在 §16 配置校验层钉死；§16 全文未提 path_strategy 与 `partial_read_policy` 的关系。
5. **（交叉 C-PE-5，确认其 P2）** PE sibling 的 `use_layerwise=false` 与 `load_async` 未在 §16.2/§22 固定（见维度 1 张力 2）；§16.2 L2290「Store backend 必需参数沿用现有 AscendStore 配置」过于放手。
6. **（记录在案的运维风险，P3，不立新发现）** 配置顺序即行为契约（L2197），DualPath child 看不到 siblings、无法在代码内校验顺序（L2291-2292 设计已承认），§22 L2523 交给外部 validator——这是边界内不可消除的缺口；误配（Store 在 DualPath 前）的后果是**静默语义漂移**：Store 先返回正数成为 winner，DE 路径永不激活、partial 退化为纯 Store load，无报错。证据：`U:multi_connector.py:397-399` first-wins 守卫 [当前源码已确认]。
7. **（P3，并入 AB-6/AB-1 修正）** 端口「空间不重叠」（L2248-2249）未写明每个空间宽度为 `tp_size` 个连续端口（recv 监听 `side_channel_port + tp_rank`，MLC:605-607, 1164-1169）[当前源码已确认]；control 端口亦需避开两段宽度。

### 维度 3：first-winner 是否只承担 accounting，还是被错误地当成数据面编排器（§15 全部）

**结论：设计把 first-winner 严格限定为 Scheduler accounting，未发现被当成数据面编排器的证据；机制与框架事实自洽，残余风险均已明示。**

依据：

- §15.2 L2081「winner 只决定 Scheduler accounting，不自动授权 Worker 副作用」——显式声明，且与 upstream 语义一致（`U:multi_connector.py:196-204` 注释：async load 只允许单一 winner）[当前源码已确认]。
- §15.4 L2119-2126 是对框架事实的准确描述而非编排企图：Worker hook 无条件广播（`U:multi_connector.py:289-309`，F1 节核实），所以每个 DualPath hook 先读已提交 `PathDecision`；L2125「真实 blocks 被传给 Layerwise 子类不构成路径授权」正确防御了 F1（`AMC:36` 的 isinstance 例外）[当前源码已确认]。
- 数据面授权链不经过 winner 状态：committed decision → `build_connector_meta` 的 plans（L1028-1032）→ Worker fence/ownership。PE_READ 的 sibling Store load 由框架自己的 accounting 驱动（winner 的 `_loading_req_ids` 要求 `num_external>0`，`pool_scheduler.py:617-618`），不是 DualPath 借 winner 编排 sibling [当前源码已确认]。
- accounting 数学（§15.3 L2085-2106）与乐观 `num_computed_tokens` 语义咬合（phase2-scratch-03 P-7 已论证）；「两个 Engine 各只见一次本 Engine 增量」（L2115）成立。
- 外层 first-winner 只存在于 PE（§15.1 L2043-2055：DE 是顶层单 Connector）；文档标题「外层 MultiConnector first-winner 编排」（L4）对 DE 不适用——表述范围问题，不构成编排误用。
- 过宽措辞一处（P3 级，不立新发现）：L2076-2077「`DE_FULL_HIT`/`DE_PARTIAL_READ` 候选满足硬准入：DualPath 返回唯一正数，赢得 accounting」——`DE_FULL_HIT` 不创建 PE 模型请求（L293, L562-568），PE 侧无 `get_num_new_matched_tokens` 调用、无 accounting 事件；该表述只对 `DE_PARTIAL_READ` 的 PE 侧成立。
- §15.8 L2191-2199 自身列出的风险（广播误读为双 winner、跨 child 生命周期、顺序契约）与 Phase 1/2 证据一致，无隐瞒。

### 维度 4：Scheduler、Connector、Worker、Store、P2P 的职责边界（§9 类图与 §8 协议）

**结论：分工与框架进程模型对齐、两处反越界纪律（L1188、L1880-1900）值得肯定；发现 1 个 P1 通道缺口的扩展（→ AB-3，扩展 C-PR-1）、1 个 P3 边界措辞矛盾（→ AB-4）、1 个 P3 死分支（→ AB-10）。**

正向（边界划得对的地方）：

- 决策/accounting 在 Scheduler 进程（coordinator + control transport；L804「control endpoint 只存在于 PE/DE Scheduler 进程，不按 TP rank 起多份」）；fence/ownership/终态发布在 Worker 进程——与框架事实对齐：connector 按 role 分两半实例化（MLC:697-712；foundation `connector.py:129-144` 同构）[当前源码已确认]。
- 下行通道：plans 经 `build_connector_meta` → 框架 ZMQ 广播（`U:multiproc_executor.py:310-320`）；上行通道：`get_finished`/`get_block_ids_with_load_errors`/`build_connector_worker_meta` 每步必调（`U:kv_connector_model_runner_mixin.py:89-112`）→ `KVOutputAggregator`（`U:kv_connector/utils.py:70-175`）。§9.6 L1163-1186 的闭环描述与框架一致 [当前源码已确认]。
- L1188「不得新增不会被框架调用的 `DualPathConnectorScheduler.get_finished()`」与 L1880-1900「Scheduler 不能直接读取 Worker 的 BlockOwnershipLedger」+ 闭环链——两条纪律都是针对真实越界诱惑的正确禁令。
- Store 边界：Worker adapter「只能消费 metadata，不得从 BlockRegion 反向猜测 ReqMeta/LoadSpec/block hashes」（L1328-1330）——防止 Worker 侧状态双源。Scheduler/Worker 两个 adapter 分别包 `KVPoolScheduler`/`KVPoolWorker`，与进程模型一致。
- P2P 边界：`ForwardLayerwiseTransfer`/`ReverseLayerwiseTransfer` 只是共享 runtime send/recv thread 的方向化 adapter（L900-901），不创建第二个 Worker（L909）——唯一自洽读法是 `SharedMooncakeTransferRuntime(runtime_worker=self)`（DualPathConnectorWorker IS-A MLC worker，且顶层 kv_both），与 L892-895 一致；F7（kv_both 下一次 `register_kv_caches` 完成全部注册，MLC:1359-1398）支撑共享前提 [当前源码已确认]。
- Store 与 P2P 的 buffer 注册不冲突：`KVPoolWorker.register_kv_caches`（`pool_worker.py:650-680`）只记录 block 元数据/基地址供 Store DMA 使用，**不调用** `global_te.register_buffer`，与 MLC 的单次守卫（`mooncake_transfer_engine.py:31-40`，F7）无竞争 [当前源码已确认]。

缺口（详见新发现）：

- **AB-3（P1，扩展 C-PR-1）**：`RankBlockMapping.remote_endpoint`（L1396-1397）需要 PE 侧 per-rank host/port 等接收端信息，但现有 endpoint 发现通道只有 D→P 方向（consumer endpoint 经 metaserver POST + `kv_transfer_params` 回传，MLC:957-972；发送方按需 GET_META_MSG 握手，MLC:1869-1918）；§8.7 的 4 种消息（L790-795）与 proposal/commit 字段（L532-545）均无 endpoint 载体；§18.2 step 10「注册 rank-local channel」（L2399）未定义对端发现。与 C-PR-1 同根因（PE→DE 无任何消息通道）、同一修复。
- **AB-4（P3）**：`bind_gpu_block_pool` 框架只在 **Scheduler 侧**调用（`U:scheduler.py:282`）；facade L997-998「Scheduler 侧转交给 Store scheduler adapter」正确，但 §9.4 L1062-1063「把 block pool 绑定到 Store adapter 和 ownership ledger」把 Worker 侧的 `BlockOwnershipLedger`（类图 L858 组合在 Worker 内）写进了 Scheduler 方法职责——跨进程对象不可达，措辞矛盾。
- **AB-10（P3）**：§9.7 L1206「PE Store view 使用 producer load 语义」与 L1349「以上 adapter 用于方案 A″ 的 DE Store；PE Store 继续由外层 sibling 承担」矛盾——A″ 下 `build_store_vllm_config` 的 `engine_role="pe"` 分支是死分支。

### 维度 17：是否违反『不修改既有 Connector 和 upstream vLLM』的硬边界（结合 F5/F6/F8/F9/F11/F15）

**结论：未发现任何显式的修改要求——F5/F6/F8/F9 全部经「继承 + 覆盖」合法解决，multi/upstream 依赖项均为现状行为。但 F11/F15 暴露一个必须显式化的边界张力（→ AB-8，P2）：设计没有声明「读取/注入 KVPool 私有成员」「绕过 MLC 归因 bug」是否属于 §3.1 允许的组合使用，导致失败语义目标在字面边界下不可实现。**

逐项核对：

- **F5**（`MLC:1006-1022` kv_both 下 `build_connector_meta` 恒走 consumer 分支、send 元数据不产出）[当前源码已确认] → §9.2 L905-908 要求**完全覆盖** `build_connector_meta()`。子类覆盖，不修改父类源码，合法。
- **F6**（`MLC:1606-1615` kv_both 下 `start_load_kv` 只走 consumer）[当前源码已确认] → 覆盖 Worker `start_load_kv`，方向化 adapter 直接向共享 runtime 的 send/recv thread 提交任务（L1501-1512）。合法；复用的是 DualPathConnectorWorker 自身继承的 runtime，不嵌套第二个 Worker（L909）。
- **F8**（`MLC:1400-1438`：finished_sending 恒空、失败请求不进完成集、早到 terminal 被 `:1407` 丢弃）[当前源码已确认] → 覆盖 `get_finished` + pending raw terminal inbox（L1104-1106, L1556-1561）。合法。
- **F9**（`MLC:1102-1124` request_finished(_all_groups) 恒 `(False,None)`）[当前源码已确认] → 覆盖 `request_finished_all_groups` + 新增 `SchedulerReleaseLedger`（L1843-1877）。合法。
- **F11**（`pool_worker.py:457-465` 创建 recv 线程未注入 invalid 集合/锁；`kv_transfer.py:908-910,924-926` 失败块进线程私有集合；`:942` 无条件 `set_finished_request`）[当前源码已确认] → §19 失败语义与 L1347「Store failed blocks 统一进入 request-level invalidation」都要求**检测** Store 失败。唯一不修改源码的可达路径：recv 线程构造器原生接受注入（`kv_transfer.py:830-831`），但线程在 `KVPoolWorker.__init__` 内部创建（`pool_worker.py:457-465`），adapter 无法经构造注入，只能事后读取/替换 `kv_recv_thread._invalid_block_ids` 等**私有属性**并自行按 block∩manifest 归因。设计全文（§3.1/§3.3/§9.7/§19）未提这一手段及其脆弱性。→ AB-8。
- **F15**（`MLC:507` `failed_reqs.add(req_id)` 用 `:469` 循环残留变量）[当前源码已确认] → Reverse/Forward 复用共享 send thread 必然继承该归因 bug；设计未声明任何 adapter 侧规避（如自维护 session→request 映射、或每 session 单请求 SendTask）。不构成越界，但属未声明的继承风险（C-PR-7 已立档，本维度交叉确认其边界含义：绕行只能在 dual_path 内做，且可行性受 `batch_transfer_sync_write` 失败粒度影响 `[外部接口待确认]`）。
- **Multi**：§15 依赖 `AMC:36`（Layerwise 例外）、`AMC:41`（非 winner Store child 收 empty+0）、`AMC:97-98`（`_extra_async_saves` 计数）、first-wins（`U:multi_connector.py:397-399`）——全部现状行为，无修改要求 [当前源码已确认]。
- **upstream**：§8.3 截断函数在 DualPath 自己的 dispatch/on-new-request 边界改 Request 数据字段（L601-615），与 MLC 既有 `_truncate_request_for_hybrid_prefill`（MLC:859-879）同模式，不改 upstream 代码；`KVConnectorWorkerMetadata.aggregate`（`U:base.py:150-168`）、`KVOutputAggregator`、`bind_gpu_block_pool`（`U:scheduler.py:282`）均为既有接口读取。无 monkey patch 要求（全文 grep 无）。
- 边界附带效果记录：顺序校验无法代码化（维度 2 缺口 6）是「不改 AMC」的直接代价，设计已诚实声明（L2291-2292）。

### 维度 18：配置、初始化、握手、shutdown 和版本兼容性（§17/§18 + F18）

**结论：初始化/shutdown 的职责与边界可落地；握手章节有 1 个 P2 机制缺口（→ AB-2）；版本兼容有 1 个 P3 聚合语义陷阱（→ AB-7）和版本锚定缺失（建议补 §4.5）。**

- **§17 fail-fast**：全部校验项（PP=DP=1、PE/DE TP 相同、completion world_size==tp_size、单 KV group、普通 Attention、PCP/DCP 关闭、模型/block size/dtype/layout 一致，L2338-2348）均可从 `vllm_config`/`kv_cache_config` 在 Connector init 读取——新代码不越界；「初始化最前面」（L2350）与现有构造顺序（`connector.py:120-144` 先解析配置再角色分支）兼容。聚合器 expected count = `get_finished_count() or world_size`（`U:kv_connector/utils.py:66-90`），PP=DP=1 时 == TP size，与 L2343 的前提一致 [当前源码已确认]。
- **§18.2 初始化顺序**：混排了 Scheduler 进程步骤（step 6 control transport）与 Worker 进程步骤（step 3-5 runtime/adapter、step 9 buffer 注册），且「Connector ready」（step 11）无框架对应物——实现者需自行拆半；P3 级表述问题，修正并入 AB-2。
- **握手（→ AB-2，P2）**：§7.3 L458 把「Forward 和 Reverse handshake 均成功」列为**提交前**准入；§18.2 L2402-2409 要求 ready 前「分别完成 handshake」。但现有 MLC 握手是**发送时按需 lazy 握手**：`update_decoder_info`（MLC:1869-1918）在首个 SendTask 时才向对端 recv 端口发 GET_META_MSG（接收循环 MLC:603-653）；不存在 init 期全 rank 对握手机制，且 L574-577 明确不增加请求级 `RankArmAck`。后果二选一：准入门落空（decision 时无法知道握手成败，partial 提交后首层 Reverse 失败 → 按请求失败收场而非提交前回退 PE_READ），或必须新写 init 期握手协议（对端发现 + 无传输时 handshake 的语义均未定义，工作量未评估）。机制上 Reverse 握手复用现有 ROUTER 多消息循环可行 `[源码可达，尚未运行验证]`，缺的是设计与触发点。
- **端口**：§18.2 step 2「PE 使用 Reverse receiver port，DE 使用 Forward receiver port」（L2388-2390）可用「派生 kv_both runtime config 的 kv_port」实现——现有推导公式 scheduler `kv_port + dp_rank*tp_size`（MLC:809-812）与 worker `kv_port + dp_rank*pcp*tp + pcp_rank*tp`（MLC:1164-1169）在 §17（dp=pcp=1）下一致，两 half 用同一派生 kv_port 即可保持通告与实际监听一致 [当前源码已确认]。当前 foundation 直接把原 config 传给父 worker init（`connector.py:87-100`），派生逻辑待写（属尚未实现）。
- **shutdown**：facade `shutdown()`（L1008-1009：拒新→drain/quarantine→关 Store/双 P2P）与 control transport 有界 drain（L782-783）为全新代码；现状组件无 drain（daemon 线程 + 基类 no-op，`phase1-scratch/03` §8）[当前源码已确认]；Multi 的 shutdown 广播存在（`U:multi_connector.py:273-284`），PE sibling 随之关闭。数据面有界 drain 缺口见 C-PE-8（非本代理候选，交叉引用，确认其 P3 判断与本维度一致）。边界无越界。
- **版本兼容（F18）**：设计实际依赖的 AMC 语义全部是 **v0.23.0 时刻的 Ascend 私有覆写**，不是 upstream 契约：`AMC:36/41`（upstream main 已改为非 winner 收 real blocks+0，commit 2285cfca4）、`AMC:93-96`（params 唯一产出者；main 已改 dict-merge，77654d080）、`AMC:97-98`（`_extra_async_saves`）；另有 `has_pending_push_work`（88ed63621）为 main 新增、Ascend 未覆写（对设计无影响）[当前源码已确认，F18]。设计 §4.5 L244-245 只说「配套 vLLM 只作为接口契约读取」，未列出对 AMC 私有语义的依赖清单与 upstream 升级复核项 → 建议补入 §4.5/§22。
- **AB-7（P3）**：PE 侧 Multi 下，`request_finished*` 的 kv_transfer_params 在 AMC 全 HMA 覆写中是「唯一产出者，第二个即 RuntimeError」（`AMC:93-96`）[当前源码已确认]；设计 §9.3 L975-980/§9.4 L1042-1054 未约束 DualPath 的 params 返回值 → 必须钉死「A″ 下 DualPath 恒返回 `(bool, None)`」，否则与 sibling（Store save 可能产出 params）冲突。
- **incarnation 兼容**：`de_engine_incarnation`（L347-350）与现有 uuid4 engine_id 缺省（MLC:702；`U:kv_transfer.py:92-94`）+ TP 内同步（`U:kv_transfer_state.py:87-94`）组合后 restart 天然换 incarnation（phase2-scratch-02 Z6）[当前源码已确认]。

---

## 过度设计检查（§10 六问）

### 1. PathDecisionCoordinator —— 分类：**必须保留**

1. **保护哪个 Stage 1 invariant**：§3.5 的单提交语义（`DECIDING→COMMITTED` 只一次、同 version 幂等、不同 path/version 二次提交为协议错误，L2467-2469）+ §15.4 的 decision gate（每个副作用 hook 先读已提交 decision，L2121）+ commit/ack 落盘后再回复（L812）。
2. **已有模块能否提供**：不能。当前分支无任何决策状态组件（`phase1-scratch/01` §6 grep 零命中）；MLC/Multi/Store 均无 per-request 决策 CAS。
3. **删除后哪个时序出错**：T5/T6（scratch-01/03）的 commit 无 CAS——控制消息重试（同 message_id 幂等之外的迟到不同 message_id 提交）与 decide 重入会双提交；§15.4 的 hook 无权威 decision 源，decision gate 落空；`get_committed()`（L717-721）是 Scheduler/Worker hook 的只读查询点，删除后 plan 冻结无依据。
4. **Stage 1 必需还是 YAGNI**：必需——四阶段决策协议是方案 A″ 的核心，单提交是硬约束（§3.5）。
5. **是否引入新状态源或双重记账**：它是决策的**唯一**状态源。两处表面重复均非双重记账：(a) fence/plan 中的 `decision` 字段（L1591, L1402）是提交后的不可变冻结副本；(b) PE/DE 各有一份 coordinator 是跨进程复制 + `decision_version` 校验（L711-715 的 ack 语义），DE 侧副本是本 Engine 数据面启动的本地门闩——两进程无法共享内存，复制不可避免。
6. **更小替代方案**：并入 `DualPathConnectorScheduler` 作为私有 dict+方法可行，但该接口（register/decide/commit/ack/query/fail）内聚且被 control handler、scheduler hook、worker plan 三方使用，独立成类不算过度。无需简化；唯一建议：`fail()` 的 `DualPathErrorCode` 与终态发布的衔接随 C-FH-8/C-PR-3 一并钉死。

### 2. DualPath control transport —— 分类：**必须保留但应简化**

1. **保护哪个 invariant**：在「禁止修改 MLC receive loop」的硬边界（L733-734）下完成 proposal/commit/ack 的可靠交付；`DE_FULL_HIT` 不创建 PE 模型请求时仍能完成 control-only commit（L813-814，Stage 1 目标路径 L35-36 的使能条件）；decision 面与数据面端口空间隔离（L805）。
2. **已有模块能否提供**：不能。(a) Mooncake zmq side channel 属 MLC 接收线程、per-TP-rank、worker 侧，且禁改；(b) metaserver 是单向 HTTP 派发通道（MLC:973-982 线程池 POST、`:1089-1100` 重试 3 次即弃），无请求/响应语义，无法承载 commit/ack；(c) `kv_transfer_params` envelope 单向且依附于请求派发——DE_FULL_HIT 没有 PE 模型请求可依附 [当前源码已确认]。
3. **删除后哪个时序出错**：DE_FULL_HIT 的 T4-T7（scratch-02）完全无通道——proposal 发不出、commit/ack 回不来，该路径不可实现；PE_READ/DE_PARTIAL 的 commit/ack 可靠性（超时重试、幂等）也无载体。
4. **Stage 1 必需还是 YAGNI**：必需。
5. **新状态源/双重记账**：无——传输层不记账，状态在 coordinator。代价是引入第二条运维平面（control 端口、PE 先 bind DE 后 connect 的启动顺序，L806）与跨引擎配置一致性要求（protocol_version，L2251）。
6. **更小替代方案/简化点**：(a) `wait_for_commit`（L776-780）无任何指定调用者（C-PR-3 确认），应删除该阻塞 API 或钉死驱动者（建议 scheduler 循环在 `build_connector_meta` 开头 drain，免新线程）；(b) `CONTROL_ERROR`（L794）有类型无流程（全 snapshot 仅定义处命中）——要么按 C-PR-2 补上失败传播流程，要么 Stage 1 从枚举中删除，避免「看起来有兜底」；(c) ROUTER/DEALER + `(message_id, payload_hash)` 幂等（L807-809）已是最小可靠交付，无需再简。

### 3. Wire identity 与 tombstone（§6）—— 分类：**wire identity 必须保留；tombstone 必须保留但应简化**

1. **保护哪个 invariant**：wire 事件四元组（wire ID、direction、engine incarnation、channel identity）匹配才可接收（L380-381）；active/retired wire ID 永不重分配（L378-379）；迟到/重复/不匹配事件不污染新请求（验收标准 L2653）；Forward/Reverse 端口/wire 空间独立（L459-460）。
2. **已有模块能否提供**：不能。现有 wire 标识只有 `session_id = host:te_rpc_port`（MLC:470）和可跨请求复用的 request id 字符串（`get_external_request_id` 剥 9 字符后缀，MLC:2087-2090）；无方向、无 incarnation、无 tombstone；父 Worker 对早到 terminal 直接丢弃（MLC:1407，F8）[当前源码已确认]。
3. **删除后哪个时序出错**：共享 runtime 下两方向共用进程内 send/recv 线程与端口推导空间，请求终止 + block 复用后，迟到的 DONE/FAILED 会按裸 request id 误归属新请求——这正是设计要修复的 F8 类误归属；方向混淆（DE 把 Reverse 完成当 Forward 完成）会使 §13.1 成功谓词误判。
4. **Stage 1 必需还是 YAGNI**：identity（请求级 wire ID + direction 派生 + incarnation）必需。tombstone 的「terminal retention window」是 drain 不可证明时的必要兜底，但窗口时长、配置项、与 drain 的先后关系未量化（L378-379 只说「drain 或 window 到期」）——必需但欠规格。
5. **新状态源/双重记账**：tombstone 集合是派生状态（随请求终止单写产生），非双重记账；与 fence 的关系是 fence 删、tombstone 留，职责互补。
6. **更小替代方案/简化点**：(a) 钉死保留策略——以「transport drain 确认」为主释放条件、retention window 为兜底，给出默认时长与配置项；(b) 决策侧 tombstone 缺失（C-FH-9 确认）应**对称复用**同一保留语义，而不是新造机制；(c) 明确不引入 `transfer_epoch`（L386-389）是正确的 YAGNI 回避，予以肯定——Stage 1 无 post-commit recovery，引入代次才是过度设计。

---

## Phase 2 候选复核

### C-PE-6（§8.2 步骤 11 的 Commit+Ack 门与 A″ sibling 持有 PE Store 矛盾；Ack 等待点未声明）—— **确认（P3）**

- 理由：`L571-572`「`PathDecisionCommit + DecisionAck + frozen block plan` 完成后，按已提交路径启动 Store load、Reverse 或 Forward」被写成普适门控，但 A″ 的 PE Store load 由 sibling AscendStoreConnector 按框架语义启动——metadata 到达即 `start_load_kv`（`U:multi_connector.py:289-291` 广播 + `pool_worker.py:787-791` 入 recv 线程），它不知道 decision/Ack 的存在 [当前源码已确认]。该门实际只能约束 **DE Store load、Reverse、Forward** 三类 DualPath 自有动作。§14.1 L1963 与 §15.4 L2131-2138 的正确读法与此一致，仅 L571-572 措辞普适 `[事实冲突]`（文本间）。
- 补充：Ack 等待点同样未声明——PE 侧 Forward 启动等 Ack 的实现位置（plan binding 条件？worker fence 前置？）无指定；scratch-01 D5 已指出。无正确性风险（Ack 永不到达时浪费有界：PE Store 白加载 + max_tokens=1 计算），属门控适用范围未声明。
- 最小修正：§8.2 显式声明「A″ PE_READ 的 PE Store load 按框架既有语义启动、不受 Commit+Ack 门控；Ack 门仅约束 DE Store/Reverse/Forward」，并写明 Ack 等待的实现位置（如 DE 侧 plan 激活条件、PE 侧 Forward 提交条件）。置信度：高。

### C-PE-7（PE 侧 proposal 等待无超时）—— **确认（P3），补充三点收窄**

- 理由成立：L579-581 允许 `(None, False)` 无限重试且无预算；请求每 pass 被 skipped（`U:scheduler.py:781-787`）[当前源码已确认]，永不调度也永不失败。
- 收窄 1（触发面）：model-request 路径下 proposal 随 `kv_transfer_params` envelope 与请求**原子到达**（L528-530），「尚未收到完整 proposal」在正常路径结构上不可达——该重试只是对抗畸形/部分派发的防御，不是常规等待。
- 收窄 2（清理成本）：该请求从未离开普通 WAITING（None 在分配前短路），未分配 blocks、无 delay_free、无需 receive terminal——预算耗尽后按 `DECISION_PROTOCOL_ERROR` 直接 `finish_requests(FINISHED_ERROR)` 即可闭环，修复便宜。
- 收窄 3（不对称残留）：DE 侧由 `DECISION_TIMEOUT`（L810-811）收敛（其自身传播问题归 C-FH-8/C-PR-3），永久滞留只残留在 PE 侧的 waiting 条目 + `has_finished_requests` 阻止退净（`U:scheduler.py:2250-2260`）。
- 维持 P3：触发需协议错误（非正常运行事件），后果是泄漏级而非损坏级。置信度：高。

### C-FH-7（control-only 被 PE 降为 PE_READ 时 retro-dispatch 未规定）—— **确认（P3），扩展触发条件分析**

- 理由成立：DE 按自身 coverage 预测 full-hit 走 control-only（L562-568），§7.3 允许提交前转 PE_READ（L462-468），但 §8.2 的 candidate dispatch 时机（step 6）已过，谁、何时、用什么 envelope 补建 PE 模型请求未规定。
- 扩展（哪些转换条件对 full-hit 候选真实可触发）：(a)「PE 已有 prefix 超过 DE ready prefix」（L468）结构上不可达——`K_DE >= R >= L_PE` 蕴含 `L_PE > K_DE` 不可能，`L_PE == K_DE` 由 §7.2 rule 3 兜住；(b) topology/layout/capability 不相容是 init fail-fast（§17/§18），ready 后不再变化；(c) 运行期 handshake 失败按现有 lazy 握手在 decision 时不可知（→ AB-2）；(d) 剩余真实触发面仅剩「probe 成功后、commit 前 Store 能力的运行期丧失」这一极窄窗口（§19.1 L2433-2435 明确允许此时降 PE_READ）。即该缺口真实存在但触发概率低，维持 P3 合理。
- 最小修正二选一：(a) 钉死「control-only 候选不降级——准入失败按协议错误/请求失败处理」，最简单且与 init fail-fast 体系一致；(b) 定义 retro-dispatch（复用同一 `DualPathRequestKey`/envelope、`decision_version` 不变，DE 在 scheduler 循环上下文补派发）——若选 (b) 必须与 C-PR-1/AB-3 的 PE→DE 通道扩展合并设计。置信度：中高。

### C-FH-8（DECISION_TIMEOUT 的 scheduler→worker 失败传播与 PE 侧 candidate TTL/失联清理未规定）—— **确认，严重级别由 P3 上调至 P2，建议与 C-PR-3 合并为一个发现**

- 上调理由：`DECISION_TIMEOUT` 是设计**自己声明**的失败路径（L810-811「重试耗尽且请求尚未 commit，进入 DECISION_TIMEOUT 失败」），其触发条件——PE 控制端不可达/PE 宕机/版本不兼容——是正常运维事件而非边角案例；但该承诺不可兑现：超时判定在 Scheduler 进程 control 线程，请求已在 `WAITING_FOR_REMOTE_KVS` 且框架默认 delay_free（`U:scheduler.py:2144-2147`）[当前源码已确认]；终态只能由 Worker 的 `get_finished` 发布（`U:kv_connector_model_runner_mixin.py:102-105` → 聚合器），而从未 commit 的请求没有任何 plan 到达过 Worker（T9 未发生）→ 无人发布 invalid/terminal → blocks 与 `self.requests` 条目永久泄漏（F13，`U:scheduler.py:2580-2586`）。「自己声明的路径走不通」比「未声明的缺口」严重一档，P2。
- PE 侧部分确认：`register_candidate`（L691-695）后无 TTL、无 incarnation 失联清理（§8.6 全接口无 expire/GC），DE 死亡则 PE candidate 永驻——同属决策面状态回收缺失，一并并入。
- 修正形态：见 C-PR-3（metadata 下发 decision-failed 记录 + Worker 侧合成 invalid/terminal）；PE 侧加 candidate TTL（与 wire tombstone 保留窗同语义，见 C-FH-9）。置信度：高（机制链全部 `[当前源码已确认]`，仅设计缺口的文本判断为 [设计推导]）。

### C-FH-9（决策态无 tombstone 保留期，迟到 commit 可复活已终态请求）—— **确认（P3，潜伏性），扩展对称缺口**

- 理由成立：§6 L378-379 的 tombstone 只覆盖 wire ID；§8.6 coordinator 接口无终态保留/GC 语义；§10.3 L1554-1573 对数据面事件有完整防 stale 设计，决策面消息只有 message_id 幂等——不对称 `[设计推导]`。
- 潜伏性确认（当前不可触发的原因）：在 C-FH-1/C-FH-8 修复之前，coordinator 状态永不 GC（泄漏悖论），迟到 commit 会命中 `fail()` 冻结态被拒（L723-728）——两个缺陷互相掩盖；一旦按候选修复引入 GC，`register_candidate` 幂等重建 + CAS 成功 → `commit_after_alloc` 为已释放请求重启 I/O，blocks 已复用时成跨请求写。故 P3 现在、随 GC 修复升级，定性准确。
- 扩展：PE 侧 `acknowledge()`（L711-715）对「无候选的迟到 ack」行为同样未规定（phase2-scratch-02 S4），应适用同一 tombstone 规则——迟到消息（commit/ack/CONTROL_ERROR）命中 tombstone 一律回 CONTROL_ERROR。
- 最小修正：coordinator 终态保留窗与 §6 wire-ID tombstone 同语义、同配置项；修正时机与 C-FH-8/C-PR-3 的 GC 引入绑定。置信度：中（依赖回收策略先存在）。

### C-PR-3（决策阶段失败无 Scheduler→Worker 注入路径；wait_for_commit 驱动者未指定）—— **确认（P2），与 C-FH-8 合并跟踪**

- 理由成立且证据链完整：(a) `wait_for_commit`（L776-780）全文无调用者 [设计推导]；(b) `DECISION_TIMEOUT`（L810-811）与 dispatch 失败（metaserver POST 重试 3 次即弃，MLC:1089-1100 [当前源码已确认]）的判定都在 Scheduler 进程；(c) §19.2 的失效流程从「Worker 发布 invalid block IDs」开始（L2440-2452），以 plan/fence 存在为前提，决策期失败的请求在 Worker 侧可能连 plan 都没有；(d) 框架释放依赖 connector 发布的终态（`U:scheduler.py:2580-2586`，F13）[当前源码已确认]。与 C-FH-8 是同一缺口的两个实例（PR 路径多一个 dispatch 失败触发源），合并为一个发现管理。
- 最小修正具体化：(a) 钉死驱动者——推荐 scheduler 循环线程在 `build_connector_meta` 开头 drain coordinator 的「已提交/已超时」队列（与 C-FH-2 的线程模型修正同形，避免新线程与无锁 `KVPoolScheduler` 竞态）；(b) 定义「decision-failed」记录随 metadata 下发各 rank Worker；Worker 对无 plan 的请求按 **update_state_after_alloc 时已冻结的 block manifest**（不依赖 plan）直接发布 invalid + 恰一次 failed receive terminal，复用 §13.2 规则。置信度：中高。

### C-PR-10（§7.3 L468「超过」未含 `L_PE == K_DE` 等号）—— **确认（P3）**

- 理由成立：L468「PE 已有 prefix **超过** DE ready prefix」字面不含等号 `[事实冲突]`（文本间）；但等号情形被三重兜住——§5.2 L306 严格 `L_PE < K_DE` 准入、§7.2 rule 3「其余情况：PE_READ」、§15.3 L2097-2102 等号返回 0 → 非 winner。无功能后果，纯措辞。
- 补充：phase2-scratch-03 S4 已验证等号误提交的实际后果链（返回 0 → winner 旁落 → 立即本地调度读未反转前缀），兜住它的正是严格不等式，所以修正只需文档对齐。
- 最小修正：L468 改「达到或超过」。置信度：高。

---

## 新发现

### AB-1：`cache_transfer_granularity` 配置项与 KVPool 内部推导的 G 双源冲突

- 标题：§16.1 配置 G 与 Store 推导 G 双源，§5.2 token 数学可能用错对齐粒度
- 严重级别：**P2**
- 类型：配置/会计一致性（双源真相）
- 证据等级：`[当前源码已确认]`（KVPool 推导）+ `[设计推导]`（冲突触发）
- 设计位置：§16.1 L2213（`cache_transfer_granularity: 256`）、L2246（仅要求「与 backend 相容」）；§5.2 L272-276（`G`、`S_DE/S_PE` 公式）；§16.2 两个示例均不含该键
- 源码位置：`pool_scheduler.py:119`（`self.cache_transfer_granularity = self._infer_cache_transfer_granularity()`）、`:404-413`（lcm 推导）、`:502-503`（lookup floor 用它）、`:604`（commit 时 `num_external == kvpool_cached - vllm_cached` assert）；`pool_worker.py:158`（worker 侧同推导）；grep 全 `kv_pool/` 无任何读取配置 G 的代码
- 触发前提：部署按 §16.1 配置 `cache_transfer_granularity: 256`，而推导 G 实际为其他值（如 block_size=128、无 family 放大时 G=128）
- 具体时序：T0 probe 返回的命中已被 Store 按推导 G=128 floor（`pool_scheduler.py:502-503`）→ T1 DualPath 按配置 G=256 再算 `S_DE = floor(S_raw/256)*256` → `H_DE/K_DE` 与 Store 实际 LoadSpec（128 对齐）错位 → `commit_after_alloc` 传入的 `store_load_tokens` 与 `kvpool_cached - vllm_cached` 不等 → 触发 `pool_scheduler.py:604` assert（scheduler 崩溃），或（若绕过 assert）Reverse/Forward 区间按错误边界切分 → 静默错位
- 为什么不正确：G 是 Store 的行为属性（由 block size/family 推导），设计把它写成独立配置项，形成第二个真相源；「与 backend 相容」（L2246）只是愿望，无机制保证相等
- 可能影响：配置与拓扑组合不匹配时 DE 路径系统性错账或崩溃；且 §16.1 有该键、§16.2 没有，实现者无从知道 G 该从哪来
- 与已有模块的兼容性：KVPool 的 G 可从 adapter 持有的 `KVPoolScheduler.cache_transfer_granularity` 属性直接读取（公开属性），无需修改任何既有组件
- 是否属于过度设计：否（是配置来源错误，不是多余抽象）
- 最小修正建议：删除配置项或降级为可选校验——运行时从 `KVPoolScheduler.cache_transfer_granularity` 读取 G；若保留配置项，init 时与推导值比较、不等即 fail-fast；§5.2 注明 G 的唯一来源
- 修正后需要增加的测试：G 不一致时 init fail-fast 的 UT；G 取自 adapter 属性的 UT；不同 block_size 下 `S_DE` 与 LoadSpec 对齐一致的 UT
- 置信度：高

### AB-2：「handshake 成功」准入条件与「ready 前完成 handshake」无现有机制支撑

- 标题：§7.3 提交前准入要求 handshake 成功、§18.2 要求 ready 前完成握手，但 MLC 握手是发送时 lazy 触发，机制与触发点均未定义
- 严重级别：**P2**
- 类型：设计承诺与可复用机制不匹配（初始化/握手）
- 证据等级：`[当前源码已确认]`（lazy 握手）+ `[设计推导]`（设计机制缺失）
- 设计位置：§7.3 L458（准入：「Forward 和 Reverse handshake 均成功」）；§18.2 L2402-2409（「分别完成 handshake」「在 Connector ready 前完成所有 rank 的 runtime 初始化、buffer 注册和 receive thread readiness」）；§8.2 L574-577（明确不增加请求级 `RankArmAck`）
- 源码位置：`MLC:1869-1918`（`update_decoder_info`：首个 SendTask 时才向对端 recv 端口发 GET_META_MSG 并缓存）；`MLC:603-653`（接收线程 ROUTER 循环处理 GET_META_MSG/DONE/FAILED）；`MLC:809-812,1164-1169`（端口推导）；init 期无任何全 rank 对握手代码
- 触发前提：任何 `DE_PARTIAL_READ` 候选的准入判定（需要「handshake 已成功」这一事实输入）
- 具体时序：T0 init——按 §18.2 应完成双方向 handshake，但没有代码知道对端是谁（endpoint 发现未定义，→ AB-3）也没有无传输语义的握手原语 → 两种实现分支：(a) 准入门落空：decision 时 handshake 未发生，L458 恒按「未验证」处理——若按失败处理则 DE_PARTIAL_READ 永不准入（路径名存实亡），若按跳过处理则准入形同虚设；(b) 提交后首层 Reverse 才做 lazy 握手，失败 → 已提交路径按请求失败收场（§19.2），丧失「提交前回退 PE_READ」的设计意图（§3.5 L143）
- 为什么不正确：设计把一个只能在「首次发送时」求值的事实提前为「提交前准入」和「init 完成条件」，且未给出提前求值的机制（对端发现 + 握手原语 + ready 判定）
- 可能影响：DE_PARTIAL_READ 的准入体系整体失效或路径不可用；实现者自由发挥则行为不一
- 与已有模块的兼容性：lazy 握手本身复用现有 ROUTER 循环可行 `[源码可达，尚未运行验证]`；缺的是提前触发它的设计
- 是否属于过度设计：否（是必要能力缺失规格；若改要求 init 期全 rank 对主动握手，反而是超出 Stage 1 需求的新机制——更小的替代是放宽准入表述）
- 最小修正建议：二选一——(a) 把 L458 改为「handshake 通道已建立（recv thread ready + 对端 endpoint 可解析）」，承认真正的握手在首次发送时完成、失败按 §19.2 处理，并删除 §18.2「分别完成 handshake」的 init 前置；(b) 若坚持准入语义，则设计 init 期握手：PE/DE 经 metaserver 互注册 recv endpoint（新 dual_path 代码）+ 每 rank 对一次 GET_META_MSG 预握手（复用 MLC:1869-1918 路径）+ ready 条件写明。同时把 §18.2 的步骤按 Scheduler/Worker 两进程拆半，并定义「Connector ready」的判定
- 修正后需要增加的测试：准入门输入的 UT（handshake 状态机的每个取值）；Reverse 首次发送握手失败 → §19.2 请求失败的 UT/E2E；（若选 b）init 期握手成功/失败/对端缺席的 UT
- 置信度：中高

### AB-3：Reverse 所需 PE 接收端 endpoint 无 PE→DE 传输通道（扩展 C-PR-1）

- 标题：`RankBlockMapping.remote_endpoint` 的 PE per-rank endpoint 信息无生产者——与 C-PR-1 同根因的通道缺口
- 严重级别：**P1**
- 类型：设计缺口（决策协议/数据面衔接）
- 证据等级：`[设计推导]` + `[事实冲突]`（消息集合与数据需求矛盾）+ 现有通道方向 `[当前源码已确认]`
- 设计位置：§10.1 L1377-1397（`LayerwiseEndpoint`/`RankBlockMapping.remote_endpoint`）；§8.1 L489-497（`PathDecision` 无 endpoint 字段）、L532-545（proposal/commit 均无）；§8.7 L790-795（消息仅 4 类）；§9.2 L902-903（只说「Reverse receiver 是 PE engine/port」，未说 DE 如何得知）；§18.2 L2399（「注册 rank-local channel」未定义）
- 源码位置：现有 endpoint 发现仅 D→P 方向：consumer 经 metaserver POST + `kv_transfer_params` 回传 host/port/tp（MLC:957-972）；发送方按需 GET_META_MSG（MLC:1869-1918）——没有任何 PE→DE 方向的既有通道
- 触发前提：任何 `DE_PARTIAL_READ`（Reverse 的 SendTask 构造需要 PE 的 host/port/te 信息）
- 具体时序：T6 PE commit（commit 消息无 endpoint 字段）→ T8 DE 侧构建 plan：`reverse_rank_mappings` 的 `remote_endpoint` 需要 PE 每 rank 的 host/port → 无任何已定义消息/注册机制提供 → T11 `submit_layer_send` 无法构造 ReqMeta（对照：Forward 方向的 remote endpoint 在现有 MLC 里由 consumer 回传 params 提供，MLC:957-972）
- 为什么不正确：`reverse_rank_mappings` 是 DATA_START 前必须冻结的 plan 部件（L1427），其关键输入在协议中无生产者；C-PR-1 已立档 block manifest 缺口，本发现确认 endpoint 元数据缺的是**同一条** PE→DE 通道
- 可能影响：Reverse 不可构造，DE_PARTIAL_READ 不可实现；或实现私下扩展协议绕过评审
- 与已有模块的兼容性：可通过新 dual_path 代码让 PE scheduler 向 metaserver 注册其 reverse-recv endpoint（与现有 consumer 注册模式同构），不修改既有组件
- 是否属于过度设计：否（必要数据无通道）
- 最小修正建议：与 C-PR-1 同一修复——扩展 `PathDecisionCommit` 携带 PE block manifest + per-rank endpoint（commit 推迟到 PE `update_state_after_alloc` 后发送），或新增幂等 `POST_ALLOC_MANIFEST` 消息同时携带两者；§8.2 协议顺序钉死其为 DATA_START 前置条件
- 修正后需要增加的测试：commit/manifest 消息的幂等与乱序 UT；DE 侧 plan 冻结时 endpoint+blocks 齐备性校验 UT；DE_PARTIAL_READ E2E
- 置信度：高（针对 snapshot 文本）

### AB-4：`bind_gpu_block_pool` 的 Scheduler 方法职责写入 Worker 侧对象，边界措辞矛盾

- 标题：§9.4 L1062-1063「绑定到 Store adapter 和 ownership ledger」跨进程不可达
- 严重级别：**P3**
- 类型：文档/接口边界矛盾
- 证据等级：`[当前源码已确认]`（框架调用侧）+ `[事实冲突]`（设计文本间）
- 设计位置：§9.3 L997-998（facade：「Scheduler 侧转交给 Store scheduler adapter」——正确）；§9.4 L1062-1063（Scheduler 方法：「把 block pool 绑定到 Store adapter 和 ownership ledger」）；类图 L858（`DualPathConnectorWorker *-- BlockOwnershipLedger`）
- 源码位置：`U:vllm/v1/core/sched/scheduler.py:282`（`bind_gpu_block_pool` 只在 Scheduler 进程调用）；Worker 进程无此调用
- 触发前提：实现者按 L1062-1063 在 Scheduler 方法里尝试把 block pool 绑给 Worker 进程的 ledger
- 具体时序：T0 init：Scheduler 进程调 `bind_gpu_block_pool` → 若照文档访问 ownership ledger → 对象不存在于本进程 → 实现者只能忽略该半句或自创跨进程通道
- 为什么不正确：ownership ledger 在 Worker 进程（类图），Scheduler 方法无法触达；且 ledger 以 block id 工作，本不需要 block pool 对象
- 可能影响：实现歧义；若有人为此引入 scheduler→worker 的额外通道则违反 L1880-1900 自己的纪律
- 与已有模块的兼容性：—
- 是否属于过度设计：否
- 最小修正建议：L1062-1063 删去「和 ownership ledger」；如需 Worker 侧感知 block pool，另行说明经 metadata 通道（或明确不需要）
- 修正后需要增加的测试：无（文档修正）；可在 §23 增加「Scheduler 侧不引用 Worker 对象」的静态检查
- 置信度：高

### AB-5：当前 DualPathConfig 静默忽略未知键；§16 未规定 unknown-key fail-fast

- 标题：按 §16 schema 部署在今天会被静默吞掉全部新键；扩展后的 config.py 必须拒绝未知键
- 严重级别：**P3**
- 类型：配置校验缺口
- 证据等级：`[当前源码已确认]`（现有解析行为）+ `[设计推导]`（设计未规定）
- 设计位置：§16 全部（未提 unknown-key 策略）；§22 L2523（validator 只覆盖顺序）
- 源码位置：`DPC:147-192`（`from_extra_config` 只 `extra.get()` 已知键，无 unknown 检查）；`DPC:1-15` docstring
- 触发前提：分阶段 rollout 期间，部署方按 §16 写 `orchestration_mode`/`forward`/`reverse`/`control`/`store` 等键，而代码仍是旧 config.py（或扩展后写错键名）
- 具体时序：T0 init：全部新键被静默忽略 → control transport 等以默认/缺失状态运行 → 运行期才以远端连接失败等形式暴露（或更糟，静默使用默认端口连错端点）
- 为什么不正确：设计的整个行为契约（路径策略、端口空间、control endpoint）都靠配置表达（L2197「配置顺序成为行为契约，运维误配风险较高」），静默吞键与 fail-fast 精神（L156, L2338, L2350）直接冲突
- 可能影响：误配晚暴露、排障成本高
- 与已有模块的兼容性：—
- 是否属于过度设计：否（一条校验）
- 最小修正建议：扩展 config.py 时对每层已知键集合做 unknown-key 拒绝（ValueError 列出未知键）；§16 增加该约束
- 修正后需要增加的测试：未知顶层键/嵌套键各一例的 fail-fast UT
- 置信度：高

### AB-6：§16.1「公共配置约束」示例不是任一 Engine 的合法配置

- 标题：§16.1 示例缺 `role`、缺 DE `store:`、且与 §15.1 PE Multi 拓扑矛盾
- 严重级别：**P3**
- 类型：文档/配置一致性
- 证据等级：`[事实冲突]`（文本间 + 与现有校验）
- 设计位置：§16.1 L2205-2228 vs §15.1 L2043-2055（PE 必须 Multi）、§16.2 L2295-2321（DE 完整示例含 `role: de` 与 `store:`）
- 源码位置：`DPC:154-159`（`role` 必填，缺省即 ValueError）
- 触发前提：实现/部署者把 §16.1 当作可用模板
- 具体时序：照抄 §16.1 → init 即 `ValueError`（缺 role）；补上 role=pe 后又与「PE 必须 Multi」冲突；补 role=de 则缺 `store:` 段，adapter 无配置来源
- 为什么不正确：文档自称「公共配置约束」但呈现为完整 yaml，实际三个示例（§16.1/§16.2-PE/§16.2-DE）字段集互不相同（`cache_transfer_granularity` 只在 §16.1，`store:` 只在 DE 示例，shadow flags 只在 §16.1）
- 可能影响：模板误用；字段归属混乱（哪个键属于哪个 Engine/哪一层无权威表述）
- 与已有模块的兼容性：—
- 是否属于过度设计：否
- 最小修正建议：把 §16.1 改为「字段约束清单」而非完整 yaml，或补全为合法示例并标注它对应哪个 Engine；给出 PE/DE/公共三栏的字段归属表（含 `cache_transfer_granularity` 按 AB-1 处理后的去向）
- 修正后需要增加的测试：§16.2 两个示例直接喂给扩展后的 `from_extra_config` 的 UT（ golden config 测试）
- 置信度：高

### AB-7：AMC 的 params「唯一产出者」语义下，DualPath 在 PE 侧 Multi 中必须恒返回 None params

- 标题：`request_finished*` 返回值未约束，与 sibling 同时产出 kv_transfer_params 将 RuntimeError
- 严重级别：**P3**
- 类型：版本语义陷阱（Multi 聚合契约）
- 证据等级：`[当前源码已确认]`（AMC 行为）+ `[设计推导]`（设计未约束）
- 设计位置：§9.3 L975-980、§9.4 L1042-1054（`request_finished(_all_groups)` 返回 `tuple[bool, dict[str, Any] | None]`，未约束第二元素）
- 源码位置：`AMC:78-102`（全 HMA 覆写）、`AMC:93-96`（「只允许一个 connector 产出 KV transfer params，第二个即 RuntimeError」）；upstream main 已改为 dict-merge（commit 77654d080，F18）——Ascend 覆写仍是 v0.23.0 语义
- 触发前提：PE 侧 Multi 下，DualPath 的 `request_finished_all_groups` 返回非 None params，且 sibling AscendStoreConnector（Store save/offload 语义）也产出 params
- 具体时序：T15 请求结束 → `_connector_finished` → AMC all_groups 循环 → 第二个非 None params → RuntimeError（EngineCore 崩溃）
- 为什么不正确：设计依赖 AMC 现状却未遵守其聚合约束
- 可能影响：请求收尾路径崩溃（低频高害）
- 与已有模块的兼容性：MLC 父类恒返回 `(False, None)`，DualPath 只要保持 params=None 即天然兼容
- 是否属于过度设计：否
- 最小修正建议：§9.3/§9.4 钉死「A″ 下 `request_finished*` 恒返回 `(bool, None)`；若未来需要 params，先解决 AMC 聚合语义（随 upstream 升级复核）
- 修正后需要增加的测试：PE 双 child 均触发 request_finished_all_groups 时不抛异常的 UT
- 置信度：中高

### AB-8：§3.1/§3.3 的「不修改 + 组合完整功能」前提在 F11/F15 下不成立，设计未声明绕行手段的合法性

- 标题：失败语义目标要求检测 Store 失败与正确归因，但 KVPool/MLC 的对应功能当前是坏的，绕行只能靠私有成员访问与 adapter 侧重归因——边界条款需要明确授权
- 严重级别：**P2**
- 类型：硬边界与目标冲突（元设计缺口）
- 证据等级：`[当前源码已确认]`（F11/F15 断裂）+ `[设计推导]`（边界解释）
- 设计位置：§2.1 L40-41（失败语义目标）；§3.1 L73-80（不修改/不 patch）；§3.3 L112-119（「组合调用既有组件提供的**完整功能**」）；§9.7 L1347（统一 invalidation 目标）；§19.2 L2440-2452
- 源码位置：F11——`pool_worker.py:457-465`（recv 线程创建未注入 invalid 集合/锁）、`kv_transfer.py:830-831`（线程构造器其实接受注入）、`:843-844`（私有集合）、`:908-910,924-926`、`:942`（无条件 set_finished）、`pool_worker.py:1333-1337`（只 drain 自有集合）；F15——`MLC:507` vs `:469`
- 触发前提：任一 Store load 部分失败（F11）；任一多请求 SendTask 的 session 写失败（F15）
- 具体时序：F11——Store DMA 部分失败 → invalid 永不上报 + 请求无条件置完成 → adapter 消费为 `STORE_DONE` → 脏 KV 经 Reverse/Forward 扩散、经 `cache_blocks` 进 prefix cache（即 C-PE-1/C-FH-4/C-PR-5 的时序）；F15——错误请求被记 failed → 误 invalid + 真失败请求 hang（C-PR-7 时序）
- 为什么不正确：§3.3 的组合前提是「完整功能」，而这两个功能点是破的；不修（越界改源码）与不管（目标落空）之外唯一的第三条路——adapter 读/替换 `kv_recv_thread._invalid_block_ids` 私有属性、按 block∩manifest 归因；adapter 自维护 session→request 映射规避 F15——在设计中完全未被提及，实现者无从知道这是否被边界允许
- 可能影响：若不澄清，实现二选一：改 KVPool/MLC（违反 §3.1，评审打回）或失败语义落空（P0 级静默数据损坏，C-PE-1）
- 与已有模块的兼容性：绕行不修改任何源码，属「组合使用」的灰色地带，需要显式授权 + 脆弱性声明（私有名、未来 KVPool 重构即碎）
- 是否属于过度设计：否
- 最小修正建议：§3.1 或 §9.7 增加一条显式授权：「允许 adapter 读取/注入所持有 KVPoolWorker 实例的 recv 线程 invalid 集合（构造器 kv_transfer.py:830-831 原生支持语义），并按冻结 manifest 做 block→request 归因；允许方向化 adapter 自维护 SendTask↔请求映射以规避 MLC:507 归因缺陷」；把「部分 DMA 失败 → invalid 先于 terminal → FINISHED_ERROR」列入 §22 门槛
- 修正后需要增加的测试：load_async 下 Store 部分失败注入 UT（验证 invalid 上报与归因）；多请求同 SendTask 失败归因 UT
- 置信度：高

### AB-9：现有 `path_strategy.type` 默认 `value_function` 与 §7.2 静态-only 未在配置层钉死

- 标题：§16 未规定 path_strategy 与 `partial_read_policy`/`orchestration_mode` 的关系，默认配置与 Stage 1 静态策略静默分歧
- 严重级别：**P3**
- 类型：配置契约缺口
- 证据等级：`[当前源码已确认]`（现有默认与校验）+ `[设计推导]`（设计沉默）
- 设计位置：§7.2 L428-441（仅静态策略）；§16 全部（未提 path_strategy）；§2.2 L59-60（shadow 约束）
- 源码位置：`DPC:100`（`PathStrategyCfg.type` 默认 `"value_function"`）、`DPC:86`（支持列表含 static/value_function/adaptive）、`DPC:213-234`（解析）、`phase1-scratch/01` §4（三者均无实现）
- 触发前提：部署不显式设置 `path_strategy`（§16 示例均不设置）→ 默认 `value_function`
- 具体时序：实现落地后，若决策代码以 `path_strategy.type` 选路而 value_function 无实现 → 行为未定义（静默退化或崩溃）；若以 `partial_read_policy` 选路 → path_strategy 成死配置
- 为什么不正确：同一语义（Stage 1 静态选路）有两个配置入口且互相未对齐
- 可能影响：实现分歧；shadow 开关（L2226-2227）与 path_strategy.value_function 的关系不明
- 与已有模块的兼容性：—
- 是否属于过度设计：否（是配置面未收敛；顺带：`orchestration_mode` 在 Stage 1 只有一个合法值，作为 fail-fast 守卫可接受，不算 YAGNI）
- 最小修正建议：§16 明确 Stage 1 要求 `path_strategy.type: static`（缺省即视为 static 或 fail-fast），value_function/adaptive 仅 shadow 用途并与 `enable_value_function_shadow` 对齐；扩展 config.py 时落实该校验
- 修正后需要增加的测试：type=value_function 且 shadow=false 时 fail-fast/告警 UT
- 置信度：中高

### AB-10：§9.7 `build_store_vllm_config` 的 PE 分支与「A″ PE 不建 store adapter」矛盾（死分支）

- 标题：L1206「PE Store view 使用 producer load 语义」与 L1349「adapter 用于 DE Store」冲突
- 严重级别：**P3**
- 类型：文档/接口一致性
- 证据等级：`[事实冲突]`（文本间）
- 设计位置：§9.7 L1196-1208（`engine_role: Literal["pe","de"]` 参数 + PE 分支语义）vs §9.7 L1349（「以上 adapter 用于方案 A″ 的 DE Store；PE Store 继续由外层既有 AscendStoreConnector sibling 承担」）；§15.1 L2064
- 源码位置：—（纯设计层）
- 触发前提：实现者按 L1206 为 PE 也构建 store adapter
- 具体时序：PE 侧多建一个 KVPoolWorker（backend 重复连接、LookupKeyServer 端口冲突风险——现有创建条件 ascend_store_connector.py:119-120 按 rank0+非 layerwise，两个实例同条件会撞 ipc path `[源码可达，尚未运行验证]`）→ 与 sibling 双写 Store 语义混乱
- 为什么不正确：同一节内自相矛盾；A″ 下 PE 分支是死代码路径
- 可能影响：实现歧义、资源重复
- 与已有模块的兼容性：—
- 是否属于过度设计：轻度（接口泛化超出 A″ 需要；若为方案 B 预留应注明）
- 最小修正建议：A″ 版本将 `engine_role` 限定为 `"de"` 或明确标注 PE 分支为方案 B 预留、A″ 不得实例化
- 修正后需要增加的测试：A″ 配置下 PE 侧不创建 store adapter 的构造断言 UT
- 置信度：高

### 正向结论（成立原因 + 前提 + 源码契约）

- **正-1（配置与注册完整承载设计拓扑）**：成立原因——Multi child 列表（`U:multi_connector.py:217-236`）+ child 独立 KVTransferConfig（`:222-227`，kv_role 不继承）+ `DualPathConnector`/`AscendMultiConnector` 注册替换链路（`vllm_ascend/distributed/kv_transfer/__init__.py:21-27,57-61`）+ 现有 role 交叉校验（`DPC:195-210`）。前提：child 顺序由部署保证（代码内不可校验，维度 2 缺口 6）。`[当前源码已确认]`
- **正-2（DE store adapter 的「组合而非修改」构造器级可行）**：成立原因——`KVPoolScheduler`/`KVPoolWorker` 均可脱离 Connector 独立构造（`pool_scheduler.py:48-54`、`pool_worker.py:81-86`），配置全部经扁平 extra_config 注入（`consumer_is_to_load` `pool_scheduler.py:87-88`、`load_async` `pool_worker.py:133`、`use_layerwise` 构造参数），`build_store_vllm_config` 的派生视图路线与这些读取点精确对齐；backend 在构造器内初始化（`pool_worker.py:105`），PE 不建 adapter（L1349）故无同进程双 backend。前提：派生 config 的其余字段（model_config 等）由浅拷贝保留（L1197-1208 已声明）。`[当前源码已确认]`
- **正-3（Store 与 P2P 的 buffer 注册互不冲突）**：成立原因——`KVPoolWorker.register_kv_caches`（`pool_worker.py:650-680`）只记录 block 元数据/基地址供 Store DMA，不触 `global_te.register_buffer`；MLC 侧单次守卫（`mooncake_transfer_engine.py:31-40`）只被共享 runtime 的一次 `register_kv_caches`（MLC:1359-1398，F7）使用。前提：每 Worker 进程 register 恰一次（`model_runner_v1.py:3824-3825`）。`[当前源码已确认]`
- **正-4（决策面传输必要性成立且规模已最小）**：成立原因——scheduler↔scheduler 的请求/响应消息在现有组件中无载体（Mooncake side channel 属 MLC 且禁改；metaserver 单向 POST 重试 3 次即弃 MLC:973-982,1089-1100；`kv_transfer_params` 单向且 DE_FULL_HIT 无 PE 请求可依附），ROUTER/DEALER + `(message_id, payload_hash)` 幂等是最小可靠交付。前提：control 端口配置与启动顺序（PE 先 bind）由部署保证（L805-806）。`[当前源码已确认]`（缺口分析为 [设计推导]）
- **正-5（incarnation 隔离 restart 身份，无需额外机制）**：成立原因——`DualPathRequestKey` 含 `de_engine_incarnation`（L347-350）；engine_id 缺省 uuid4 进程级（MLC:702；`U:kv_transfer.py:92-94`）+ TP 内同步（`U:kv_transfer_state.py:87-94`）→ restart 天然换 incarnation；配合 wire 四元组匹配 + tombstone（L378-381）防 stale 归属。前提：部署不显式固定 engine_id 为跨 restart 相同值。`[当前源码已确认]`

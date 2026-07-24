# Phase 3 维度评审 02：token accounting / token range / KV blocks 生命周期 / decision-commit-data-start 竞态

- 负责维度：§9 强制维度 5（token accounting 闭合，专项 9.1）、6（token range 无遗漏/重叠/多 writer，专项 9.2）、7（正式 KV blocks 分配/有效性/ownership 生命周期，专项 9.2）、8（decision/commit/data-start 竞态）。
- 过度设计检查：TransferPlan/Command/Event（§10）、TransferFence（§11）、BlockOwnershipLedger（§12）。
- Phase 2 候选复核：C-PE-3、C-FH-3、C-PR-6、C-PR-8、C-PR-11。
- 设计行号 = `inputs/option-a-detailed-design.snapshot.md`；源码基线 vllm-ascend `dev/dualpath @ 0ec11a47`、upstream `vllm @ 8df14cfc`（配套 v0.23.0 有小幅偏差）。`U:` = upstream，`MLC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`，`AMC` = `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`。
- 本文全部源码引用均已在本阶段逐行复核（非转引 Phase 1/2）。

---

## 维度结论

### 维度 5：token accounting 是否闭合（专项 9.1）

**结论：DE 侧与 PE 侧的 accounting 框架在三条路径上各自闭合（正向），但存在一个已确认的 P1 级公式缺陷（C-FH-3：`S_DE` 二次 floor 与 KVPool probe 实际语义冲突），以及 `S_DE_raw/S_PE_raw` 的 owner/语义未钉死这一根因。partial hit 未被错误表达成 sibling 自动组合（正向，有框架证据）。**

#### 5.1 Token 变量定义/单位/开闭区间/owner 表（专项 9.1 要求）

所有变量单位均为 token 数；区间为绝对 token 索引、左闭右开 `[start, end)`；下标 `x` ∈ {DE, PE}。

| 变量 | 定义（snapshot） | 区间/取值域 | 计算 owner | 消费 owner | 复核结论 |
|---|---|---|---|---|---|
| `P` | 原始 prompt token 数（L266） | `[0, ∞)`，= `request.num_tokens` | 框架 Request | 两侧 | 闭合。PE 侧被 §8.3 截断为 R（L601-615），此后 PE 请求的 `num_tokens=R`，两 Engine 的 `num_tokens` 不同——设计未给 PE 截断后的 `P'` 单独符号，是文档瑕疵但无语义歧义 |
| `R = max(P-1, 0)` | decode-ready 前缀（L267） | `[0, P)` | DE（公式） | 两侧所有判定 | 闭合，见 5.2 末 token 语义 |
| `L_DE`/`L_PE` | 本地可复用 token 数（L268-269） | `[0, R)`（候选流程内；`≥R` 时 `E_DE=0` 流程终止，L301-302）；block 对齐（prefix cache 性质） | 框架 `get_computed_blocks`，作为 `get_num_new_matched_tokens` 第 2 参数传入（U:scheduler.py:761-763, 774-779，已复核） | 各 Engine connector | 闭合。唯一可信来源是该调用参数（支撑 C-PR-11） |
| `S_DE_raw`/`S_PE_raw` | probe 返回的"原始绝对前缀长度"（L270-271） | 未定义域——**语义未钉死** | Store probe adapter（§9.7 L1236-1245） | `S_DE`/`S_PE` 公式 | **不闭合（C-FH-3 根因）**：可复用的 KVPool probe 出口给的是「已 floor（pool_scheduler.py:502-503）且 full-hit 减 1（:526-527）」的值，与"原始"二字矛盾 |
| `G` | `cache_transfer_granularity`（L272） | `lcm(lcm_block_size, family granularity)` | KVPoolScheduler 推断（pool_scheduler.py:404-416） | 对齐公式 | 闭合 |
| `S_DE`/`S_PE` | `min(floor(S_raw/G)*G, R)`（L273-274） | `[0, R]` | 设计公式 | `H/K` 公式 | **缺陷（C-FH-3）**：对已被 probe floor+减 1 的输入二次 floor，`P≡0 (mod G)` full hit 被打掉一个 chunk |
| `H_x = max(S_x - L_x, 0)` | Store 新增加载量（L275, L277） | `[0, R]` | 设计公式 | `commit_after_alloc`、accounting | 随 `S_x` 缺陷继承；partial 对齐命中时与 LoadSpec delta 同构（pool_scheduler.py:532, 604-614），闭合 |
| `K_x = min(max(L_x, S_x), R)` | Store load 后连续 ready prefix（L276, L278） | `[L_x, R]` | 设计公式 | 路径判定（L304-306）、Reverse/Forward 区间 | 闭合。`max` 隐含「Store 从 `L_x` 起连续加载」——由 load 侧 `mask_num = floor(L_x/bs)*bs` 跳过本地段保证（kv_transfer.py:858-862，已复核），`[0,L_x)∪[L_x,S_x)` 连续成立 |
| `T_DE = R - K_DE` / `T_PE = R - K_PE` | PE 必须计算的尾部（L279-280） | `[0, R]` | 设计公式 | PE compute 区间 | 闭合 |
| `E_DE = max(R - L_DE, 0)` | DE 向 Scheduler 声明的 external 总量（L281） | `[0, R]` | DE DualPath | DE Scheduler accounting | 闭合（见 5.3），是 A″ 最成立的一笔 |
| `F_PE_READ = E_DE` | PE_READ 的 Forward 逻辑 token 数（L282） | `[0, R]` | 设计公式 | §14.1 | 闭合但注意：该符号在 §15.3 代码（L2085-2106）中无人使用，Forward 区间直接写作 `[L_DE, R)`——冗余定义，非缺陷 |

`StoreCoverage`（L322-333）字段与上表一一对应（raw→`S_raw`、aligned→`S`、local→`L`、ready_prefix→`K`、decode_ready→`R`、store_load→`H`、prefill_tail→`T`），owner 按 Engine 分离（L335-337）✓；frozen dataclass，"只描述已验证已对齐 coverage，不代表 load 完成"（L336-337）的语义边界清晰 ✓。

#### 5.2 末 token 语义验证（§4.4 + §8.3 + §15.3）

- `R = P - 1` 与 upstream 的咬合**精确**（已复核）：connector 声明 `R` 使乐观 `num_computed_tokens = R = P-1 < num_tokens`（U:scheduler.py:1004, :800），promote 时 `== num_tokens` 的 -1 调整分支（U:scheduler.py:2521-2522）**结构性不触发**，DE 以 `num_new_tokens = P - R = 1`（:847）重算最后一个 prompt token——零框架修改。证据等级 `[当前源码已确认]`。
- PE 侧 §8.3 把 PE 模型请求截断到 R（L601-615），使 sibling Store 的 `hit == num_tokens` 减 1 约定（pool_scheduler.py:526-527）在 PE 侧同样成立（PE 的 full hit 退化为 `R-1`，PE 重算 `[R-1, R)` 一个 token，与框架 promote 语义自洽）。
- 判据「Store 覆盖到 `prompt_len - 1` 即满足 full-hit」（§4.4 L226、§14.2 L1981）与 §5.2 公式**不一致**——这是 C-FH-3 的文本面，见候选复核。

#### 5.3 每 Engine external 增量与实际写入区间一致性（§5.2 L284-297 + §15.3）

逐路径核对「Scheduler accounting 声明量 vs 实际物理写入区间」（写入方见维度 6 表）：

| Engine/路径 | 声明量 | 乐观 `num_computed` | 实际异步写入区间 | 一致性 |
|---|---|---|---|---|
| DE / 任意 | `E_DE = R - L_DE`（L288-290） | `R` | PE_READ: Forward `[L_DE,R)`；DE_FULL_HIT: Store `[L_DE,K_DE=R)`；DE_PARTIAL: Store `[L_DE,K_DE)` + Forward `[K_DE,R)` | ✓ 三条路径写入区间的并集恰好 `[L_DE,R)`，无缺口无重叠（`K_DE` 经 G 对齐为 block 边界，pool_scheduler.py:404-416；`L_DE` block 对齐） |
| PE / PE_READ | sibling 声明 `H_PE = hit - L_PE`（L291；pool_scheduler.py:532, 565） | `K_PE` | Store `[L_PE,K_PE)` | ✓（sibling 自身 accounting，原生闭合） |
| PE / DE_PARTIAL_READ | DualPath 声明 `max(K_DE - L_PE, 0)`（L292, L2097-2102） | `K_DE` | Reverse `[L_PE,K_DE)`（L1421） | ✓ 声明区间与 Reverse 写区间逐 token 一致 |
| PE / DE_FULL_HIT | 不创建 PE 模型请求（L293） | — | — | ✓ |

硬性契约「PE 不得返回 `R - L_PE`」（L295-297）**正确且承重**：若 PE 在 DE_PARTIAL 下声明 `R - L_PE`，框架会把乐观 `num_computed` 置满 R，promote 后只算 1 个 token，`[K_DE,R)` 尾部永不被计算——静默脏数据。该契约的存在正是对「框架无法表达两段 token 分别由两个 connector 异步提供」（01-current-control-flow §2）的正确回应。

#### 5.4 Store 对齐 / partial chunks 丢弃后 coverage

- `discard_partial_chunks=True`（默认，pool_scheduler.py:124-129）下：查询长度 floor 到 G（:502-503）、save 侧同 floor（:628-631）、Store 只持 G 对齐 chunk。被丢弃的尾部 partial chunk 由 `T_DE`/`T_PE`（PE 计算）覆盖 ✓，无 coverage 缺口。
- 推论（设计未写明，应补）：`P % G ∉ {0,1}` 时 full-hit 在 KVPool 查询语义下**结构性不可达**（查询长度 `floor(P/G)*G < R`，最多证明 partial hit），DE_FULL_HIT 仅当 `P%G==0`（查询 P、命中 P、减 1 得 R）或 `P%G==1`（查询 R、命中 R）时可触发。这与 §4.4/§14.2 的「coverage >= prompt_len-1 即 full hit」表述冲突，并入 C-FH-3 修正。

#### 5.5 partial hit 是否被错误表达成 sibling 自动组合——**否（正向结论）**

- DE_PARTIAL_READ 在 PE：DualPath（child 0）返回正数成为唯一 winner，AscendStore sibling 虽被调用但 `to_return[0]==0` 守卫使其结果不进 accounting（U:multi_connector.py:397-399，已复核）；非 winner 且非 Layerwise 的 sibling 收 `(empty, 0)`（AMC:41），非 layerwise 下 `can_load=False`、不进 `_loading_req_ids`（pool_scheduler.py:589-593, 617-618）→ 无 Worker GET、无第二段异步供给。`[当前源码已确认]`
- §5.2 L307「Store 只负责新增区间，不能重复参与 Scheduler accounting」与框架 first-winner 语义方向一致，未依赖任何不存在的组合能力。
- 前提：sibling 必须 `use_layerwise=false`——该配置约束缺失是 C-PR-4（非本文复核范围），不改变本结论。

**维度 5 小结**：accounting 骨架（变量表、per-Engine 声明、区间一致性、末 token 语义、防 sibling 组合）成立；唯一实质性缺陷是 `S_DE_raw` 语义未钉死导致的 C-FH-3（P1）。

---

### 维度 6：token range 无遗漏、无重叠、无多 writer（专项 9.2）

**结论：三条路径的 token range 表在 happy path 上闭合——声明区间、物理 blocks、写者、读者、有效性转换、释放条件均可枚举且无缺口/重叠/多 writer；失败路径的 invalid 范围界定有缺陷（C-PR-6，确认）；Store 物理 +1 slot 写出格于设计的 token 模型（新发现 TB-1，P3）。**

#### 6.1 PE_READ（设计 §14.1 L1937-1969）

| token range | physical blocks | allocator | writer | reader | validity transition | release condition |
|---|---|---|---|---|---|---|
| DE `[0,L_DE)` | DE prefix cache 共享块 | 框架（历史请求） | 历史 compute（本请求无 writer） | DE decode attention | 始终 valid | 共享块生命周期（引用计数），本请求不释放 |
| DE `[L_DE,R)` | DE 新分配私有块（`delay_cache_blocks=True`） | DE 首次 `allocate_slots(num_external=E_DE)`（U:scheduler.py:942-954） | Forward（PE send thread 单边写，MLC:497-499） | DE decode attention | 分配即 invalid/unpublished → `FORWARD_DONE` → promote `cache_blocks(R)` 变 valid（U:scheduler.py:2517-2541）；失败按 §19.2 invalid | 请求结束 + `request_finished_all_groups`；有 async owner 时 delay_free 至 ownership=0（§13.4） |
| DE `[R,R+1)` | DE promote 后补分配 | DE 第二次 `allocate_slots`（U:scheduler.py:822-847） | DE 本地重算 | DE attention | compute 后 valid | 正常请求生命周期 |
| PE `[0,L_PE)` | PE prefix cache 共享块 | 框架 | 历史 compute | Forward sender 读（当 `L_DE<L_PE`）+ PE attention | valid | 共享生命周期 |
| PE `[L_PE,K_PE)` | PE 新分配私有块（delay_cache） | PE 首次 `allocate_slots(num_external=H_PE)` | sibling Store bulk DMA（kv_transfer.py:902） | PE attention + Forward sender 读 | sibling `finished_recving` → promote `cache_blocks(K_PE)` | 正常 + sibling save drain 的 delay_free（`_extra_async_saves` 门控，AMC:97-98） |
| PE `[K_PE,R)` | PE promote 后补分配 | PE 第二次 `allocate_slots`（`=0` 调用，U:scheduler.py:969-974） | PE model compute | Forward sender 读 + PE attention | 随层 compute 后 valid | 正常 + DualPath Forward drain 的 delay_free |
| PE `[R,R+1)` | 同上 | 同上 | PE decode token（`max_tokens=1`） | — | valid | 正常 |

Forward 源区间 `[L_DE,R)` 的三段构成（本地段 `[L_DE,L_PE)`、Store 段 `[max(L_DE,L_PE),K_PE)`、计算段 `[K_PE,R)`）在任意 `L_DE/L_PE/K_PE` 关系下并集恒为 `[L_DE,R)`——**PE compute tail `[K_PE,R)` 是所有覆盖缺口的兜底**（含 `K_PE<L_DE` 的极端情形），无遗漏 ✓。时序安全：Store 整段 bulk load（L1668）+ 框架 WAITING 门禁（promote 前无 forward，U:scheduler.py:834-837, 986-1006）⇒ 任一 save 钩子触发时三段源全部就绪。`[当前源码已确认]`（框架段）+ `[设计推导]`（方向化 adapter）。

#### 6.2 DE_FULL_HIT（设计 §14.2 L1971-2000）

| token range | physical blocks | allocator | writer | reader | validity transition | release condition |
|---|---|---|---|---|---|---|
| DE `[0,L_DE)` | 共享 prefix 块 | 框架 | 历史 compute | DE attention；Store 经 `mask_num=floor(L_DE/bs)*bs` 跳过不写（kv_transfer.py:858-862，已复核） | valid | 共享生命周期；**失败时不得进 invalid（C-PR-6）** |
| DE `[L_DE,K_DE=R)` | DE 新分配私有块（delay_cache） | DE 首次 `allocate_slots(num_external=E_DE)` | Store bulk DMA（各 TP rank 写本 rank shard） | DE attention | `STORE_DONE` → DE-local `finished_recving` → promote `cache_blocks(R)` | 正常；`STORE_WRITER` owner 于 `STORE_DONE` 释放（§12 表 L1738） |
| DE `[R,R+1)` | 末 block 内 slot（已在首次分配的 ⌈R⌉ 块内） | （随首次分配） | **Store +1 slot 物理写**（pool_scheduler.py:849-852：`kvpool_cached==P-1` 且非块对齐时 `token_len+1`）→ 之后 DE 重算覆盖写 | DE attention | 两次写由 `STORE_DONE` barrier 定序（Store→promote→重算） | 正常；**该物理写出格于 §10.1「不越过 R」的 token 模型（TB-1）** |

#### 6.3 DE_PARTIAL_READ（设计 §14.3 L2001-2037）

| token range | physical blocks | allocator | writer | reader | validity transition | release condition |
|---|---|---|---|---|---|---|
| DE `[0,L_DE)` | 共享 prefix 块 | 框架 | 历史 compute | **Reverse sender 读** + DE attention | valid | 共享生命周期；**失败时不得进 invalid（C-PR-6）** |
| DE `[L_DE,K_DE)` | DE 新分配私有块 | DE 首次 `allocate_slots` | Store bulk DMA | Reverse sender 读 + DE attention | 内部 `STORE_DONE`（不透传公共终态，L1341-1342）；公共 valid 于最终 promote | 正常；Store owner 于 `STORE_DONE` 释放、Reverse reader owner 于 receiver terminal 释放（§12 表 L1739） |
| DE `[K_DE,R)` | DE 新分配私有块（与 Store target 同组 blocks，block 区间与 `[L_DE,K_DE)` 不相交：`K_DE` 为 G 对齐 block 边界） | DE 首次 `allocate_slots` | Forward（PE send 单边写） | DE attention | `FORWARD_DONE`（且 `STORE_DONE`）→ promote | 正常 |
| DE `[R,R+1)` | promote 后补分配 | DE 第二次 `allocate_slots` | DE 重算 | DE attention | valid | 正常 |
| PE `[0,L_PE)` | PE 共享 prefix 块 | 框架 | 历史 compute | PE attention | valid | 共享生命周期；失败时不得进 invalid（C-PR-6 的 PE 侧对应） |
| PE `[L_PE,K_DE)` | PE 新分配私有块 | PE 首次 `allocate_slots(num_external=K_DE-L_PE)` | **Reverse（DE send 单边写）** | PE attention | 完整 `REVERSE_DONE` → PE-local `finished_recving` → promote（§11.2 L1618-1636） | 正常 |
| PE `[K_DE,R)` | PE promote 后补分配 | PE 第二次 `allocate_slots`（`=0`） | PE model compute | Forward sender 读 + PE attention | 随层 compute 后 valid | 正常 + Forward drain delay_free |

#### 6.4 遗漏/重叠/多 writer 判定

- **无遗漏**：每路径声明区间 = 写入区间并集（5.3 表）；promote 后每侧仅剩 `[R,R+1)` 由本地重算覆盖 ✓。
- **无重叠**：各区间边界（`L_DE/L_PE` block 对齐、`K_DE/K_PE` G 对齐）使写者区间两两不相交；Forward 被明确禁止覆盖 Store 前缀（L1426）✓。
- **无多 writer（物理）**：每条路径每个区间恰一个 writer；唯一的双写点是 DE_FULL_HIT 的 slot R（Store +1 → 重算），由 `STORE_DONE` barrier 严格定序，不构成并发写 ✓。未协调 writer 风险仅剩 sibling layerwise 误配（C-PR-4，非本文范围）。
- **多 reader 无协调需求**：Reverse 读 `[0,K_DE)`（含共享段）与 DE 本地 forward 读其他请求的共享块均为只读，无 writer 冲突。

**维度 6 小结**：token range 表闭合；两个修正项——invalid 范围排除共享段（C-PR-6）、+1 slot 写入模（TB-1）。

---

### 维度 7：正式 KV blocks 的分配、有效性与 ownership 生命周期（专项 9.2）

**结论：happy path 生命周期闭合（分配时机、validity 转换、writer 定序、释放条件均有框架或设计锚点）；失败路径有两个已确认缺陷（C-PR-6 invalid 范围、以及 invalid 检测对 F11 修复的依赖——后者属 C-PR-5/C-FH-4 复核范围，本文只登记依赖关系）。逐问回答如下。**

1. **无人填充区间？** 无。三条路径的声明区间都有明确 writer（6.1-6.3 表）；决策悬置窗口内 blocks 处「已预留未发布」纯预留态（`delay_cache_blocks` 隔离，无 writer 是协议意图而非缺陷）；该窗口的异常收口（DECISION_TIMEOUT/abort）属 C-FH-1/C-FH-8/C-PR-3 范围。
2. **多个未协调 writer？** 无已证实实例（维度 6.4）。Store 与 Forward 写同一组 DE 正式 blocks 的**不同 token 范围**（`[L_DE,K_DE)` vs `[K_DE,R)`，L2036-2037），block 区间因 G 对齐而不相交，且由 `STORE_DONE → Reverse → REVERSE_DONE → compute → Forward` 链串行定序（§14.3 L2031-2035）✓。
3. **Reverse 读取前 Store DMA 是否已完成？** Host 顺序闭合：`m_store.get` 为 host-synchronous 调用，返回即评估逐 key 结果、随后才 `set_finished_request`（kv_transfer.py:902-942；两个 backend 均为同步 RPC 返回失败集：yuanrong_backend.py:169-199、mooncake_backend.py:233-266，均已复核）；adapter 消费 done 后置 `STORE_DONE` 才提交 Reverse。**设备级可见性（HBM 写对 Mooncake RDMA 读）无设计论证**——即 C-PR-8（确认，P2）。
4. **失败后 blocks 过早复用？** 设计机制闭合（`[设计推导]`，全新代码）：§13.2 L1789-1793 规定仍有 owner/unknown in-flight 时只发 invalid、不发 failed terminal，ownership 归零且 quarantine 解除后才发 `finished_recving`；框架仅在 terminal 到达后 `_free_blocks`（U:scheduler.py:2580-2586）；fail 策略下 prefix cache 逐出（:2736-2737）也发生在 invalid 消费之后——时序方向一致。前提：ownership/terminal 状态机全部为新实现（phase1 F8/F9 表明父类语义不可用）。
5. **invalid 是否可能在成功事件后才被发现？** 设计三层防护方向正确：§13.2 L1794「invalid 可早不可晚于失败终态」；§10.3 L1567「成功后收到失败 = 协议错误 + quarantine」；框架每步先消费 invalid 后消费 finished（U:scheduler.py:1578-1586）。**但防护依赖失败可被检测**：Store load_async 下 F11 断裂使 Store 失败结构性不可见（pool_worker.py:457-465; kv_transfer.py:908-910, 942）——修复责任在 C-PR-5（DE 内部 adapter）与 C-FH-4/C-PE-1（sibling），本维度登记为外部依赖；修复前「成功事件后发现 invalid」不是时序问题而是「失败永远 invisible」问题。
6. **分配与 ownership 的 keying**：blocks 由框架分配、路径无关（DE 一次分足 `[0,R)`，L470-474），`delay_cache_blocks` 保证 promote 前不进 prefix cache、不被他请求命中（U:scheduler.py:949, :2652-2653 注释）——该窗口内 blocks 请求独占，是失败归因可按「block ∩ 冻结 manifest → request」进行的结构性前提 ✓。

---

### 维度 8：decision、commit、data-start 之间的竞态（§8.2/§8.5 + F4）

**结论：四阶段解耦（PROBE → ALLOCATE_AND_DISPATCH_CANDIDATE → DECISION_COMMIT → DATA_START，L514-516）在 DE 侧与框架精确咬合（正向）；但 §8.5 的冻结规则是「按 DE 单侧分配形态写的」，与 PE 侧两段式分配冲突（C-PE-3 确认，且应扩展到 DE_PARTIAL_READ 的 Forward 源）；另有两个上下文缺口（Ack 门位置、control 线程应用）分别归 C-PE-6/C-FH-2，本文只登记不展开。**

1. **DE 侧无竞态（正向）**：DE 声明路径无关的统一量 `E_DE`（L518-521）→ 框架一次分足 `[0,R)`（load_async 不分配新计算量，U:scheduler.py:834-837, 942-954）→ 第一次 `update_state_after_alloc(real blocks, E_DE>0)` 即可完整冻结 manifest；第二次调用（promote 后，`=0`；U:scheduler.py:718-719, 822-827, 969-974，F4 已复核）为幂等 no-op。blocks 在窗口内被 delay_cache + inflight 预留钉住（:934-940, 949），commit 前后不会变化 ⇒ 「先分配冻结、后 commit、再 data-start」无窗口竞态。`[当前源码已确认]`（框架）+ `[设计推导]`（冻结语义）。
2. **PE 侧冻结规则与两段式分配冲突（C-PE-3，确认+扩展）**：PE 的 blocks 天然分两（多）段到达——load_async 首次只分 `[0,K_x)`，promote 后第二次 `=0` 调用才带全量（含 tail `[K_x,R)`）；Store miss（`H_PE=0`）时所有调用 `num_external=0`，且 chunked prefill（U:scheduler.py:867-869）下 blocks 逐 chunk 增量到达。§8.5 L681-684「只在 `num_external_tokens > 0` 时冻结真实 block plan」按字面执行 ⇒ PE_READ 的 Forward 源 `[K_PE,R)`、DE_PARTIAL_READ 的 Forward 源 `[K_DE,R)`（**扩展点**：Phase 2 只列了 PE_READ）以及 miss 时全部 Forward 源都没有协议内冻结来源。详见候选复核。
3. **commit 先于 PE 分配的次序**：PE 在 `get_num_new_matched_tokens` 内 decide+commit（T5/T6），PE blocks 在紧随其后的 `allocate_slots` 才分配——`PathDecision`/`PathDecisionCommit` 均不含 block 字段（L489-497, L543-545），DE_PARTIAL 的 Reverse target manifest 无 PE→DE 通道：即 C-PR-1（非本文复核范围，但作为 commit/data-start 衔接缺口登记上下文）。DE 侧 data-start 门「Commit + DecisionAck + frozen block plan」（L571-572）中 **Ack 等待点的实施位置未声明**（C-PE-6 范围）；DE 侧 commit 应用线程与 scheduler 循环的竞态（C-FH-2 范围）。这两个缺口与本文维度 8 的交集仅在于：它们都是「commit 已在控制面完成、但数据面启动条件尚未在正确线程/正确时刻闭合」的同类缺口。
4. **data-start 幂等 key 的 pre-commit 未定义（新发现 TB-2，P3）**：§8.5 要求以 `request_key + decision_version` 幂等，但首次冻结发生在 commit 之前，此时 `PathDecision.decision_version`（L495）尚不存在。
5. **decision 本身无双侧竞态**：PE 是唯一提交方（L501），禁止两侧各自重算（L592）；`DECIDING` 窗口内 DE 仅凭 coverage 预测通道、不作数据面动作（L561-562）✓；`commit()` CAS 一次提交、不同二次提交协议错误（L705-709, L2467-2469）✓。

---

## 过度设计检查

### A. TransferPlan / Command / Event（§10，L1362-1573）

1. **保护哪个 Stage 1 invariant**：(a) plan 冻结后 block 不可重映射（L1427）+ Worker 不得按 token 数重推 remote blocks（L1511-1512）——这是「分配先于 decision」协议（§7.3 L470-474）成立的载体；(b) command 携带 `ids/message_id/payload_hash` 实现「每个 (request_key, rank, layer, direction) 发送至多一次」（§20 L2471）；(c) event 四元组（wire ID/direction/incarnation/channel）匹配是终态归属与 §13.1 成功谓词（`STORE_DONE && FORWARD_DONE`）的唯一输入来源。
2. **已有模块能否提供**：不能。MLC 的 ReqMeta/SendTask 只覆盖单方向逐层发送；kv_both 下其 metadata 恒走 consumer 分支（F5）、`get_finished` 丢弃早到 terminal（F8）、且无跨操作（Store+Reverse+Forward）请求级谓词能力。KVPool 只感知 Store done。请求级三操作组合状态无任何现有载体。
3. **删除后哪个时序出错**：删 plan 冻结 → Reverse SendTask 无 remote block 来源（叠加 C-PR-1 后整条 DE_PARTIAL_READ 不可构造）；删 event → T14/T16 的发布条件（`REVERSE_DONE`/`FORWARD_DONE`）无求值输入，成功谓词不可实现；删 command 幂等键 → 重试/metadata 重放下同一层重复发送，单边写重复写同一远端区间（数据自洽但终态计数错乱，`trans_count` 协议破坏）。
4. **Stage 1 必需还是 YAGNI**：核心必需。可质疑的只是粒度——Stage 1 不做跨层流水（§2.2 L52），`REVERSE_LAYER_DONE/FORWARD_LAYER_DONE` 不驱动调度，只服务 §12 的逐层 owner 释放；用「每方向已完成层计数」即可等价。`PATH_COMMITTED` 是本地状态迁移而非传输事件，混在 `DualPathTransferEvent`（带 tp_rank/channel_identity 字段）里语义错位但无害。
5. **是否引入新状态源或双重记账**：event 是唯一状态源，fence（§11）与 ledger（§12）是同一事件流的两个物化视图；Store 的 done_recving 被 adapter 内部消费为 `STORE_DONE` 且**不透传**为公共 finished_recving（L1339-1343）——设计上明确避免了与 KVPool 的双重记账 ✓。前提：fence 与 ledger 必须在同一临界区消费同一事件（设计未写明，实现注意）。
6. **更小替代方案**：逐层事件退化为计数器、`PATH_COMMITTED` 移出传输事件枚举；plan/command/event 三元组本身不可删。

**分类：必须保留但应简化**（简化项：逐层事件→计数；PATH_COMMITTED 出枚举；不构成重新设计）。

### B. TransferFence（§11，L1575-1668）

1. **保护的 invariant**：rank-local「恰好一次」终态发布（§13.2 L1795）与三操作组合成功谓词求值（§13.1）；跨 rank barrier 明确交给框架 `KVOutputAggregator`（L1600-1602），不另造 rank 集合 ✓。
2. **已有模块能否提供**：不能。聚合器只做跨 rank 计数，不感知单 rank 上 Store/Reverse/Forward 的组合谓词；MLC/KVPool 各自的完成语义均不可用（F8/F11）。
3. **删除后哪个时序出错**：T14（PE REVERSE_DONE→finished_recving）与 T16（DE STORE_DONE&&FORWARD_DONE→finished_recving）无求值点 → 要么过早发布（脏 blocks 进 prefix cache，U:scheduler.py:2517 后不可挽回），要么 hang。
4. **Stage 1 必需还是 YAGNI**：必需，且数据结构已是 6 个布尔值的极简形态；§11.2 的 layer fence 主要是论证性文字（Stage 1 全 barrier 的理由），落实仍靠请求级 fence + 框架 WAITING 门禁，不引入额外机制 ✓。
5. **是否引入新状态源或双重记账**：fence 是事件的唯一消费点之一，本身不是独立状态源；`RankTransferStatus`（L1132-1143）是其 DTO（经 WorkerMetadata 上送诊断），方向单一，不构成双写。
6. **更小替代方案**：无实质更小方案（并入 Worker 状态机字段 vs 独立 dataclass 是风格差异）。

**分类：必须保留。**

### C. BlockOwnershipLedger（§12，L1670-1743）

1. **保护的 invariant**：「仍存在 writer/reader/unknown in-flight 时 block 不可释放」（L1727）——这是 §13.2 防 Scheduler 在 `FINISHED_ERROR` 后立即复用仍可能被写入 blocks 的关键约束（L1801-1804）的求值机制；以及 quarantine 的入/出（L1728-1729）。
2. **已有模块能否提供**：不能。框架只有请求级 delay_free 二元语义 + terminal 计数释放（U:scheduler.py:2144-2147, 2580-2586）；「何时可以发 finished_sending」的 Worker 侧判据（ownership=0 且无 unknown in-flight，L1893）无任何现有实现（父类 `request_finished*` 恒 False，F9）。
3. **删除后哪个时序出错**：failed terminal 提前发布 → Scheduler 释放并复用 in-flight writer 的 blocks → 跨请求物理写污染（§13.2 明示的场景）；delayed-free 请求（L1892-1896）永无释放触发 → blocks 泄漏。
4. **Stage 1 必需还是 YAGNI**：**机制必需、粒度超出 Stage 1 需要**。Stage 1 的写者少且严格定序（bulk Store → Reverse → compute → Forward）、失败按请求级失效（L1730）、不做局部复用（L1731）、不做跨层流水——每请求每操作类型的 owner 计数 + quarantine 标志 + 按 plan region 过滤的 invalid 集合即可支撑全部 Stage 1 语义（`RankTransferStatus` 上送的 `active_owner_count/unknown_inflight/send_drain_done` 恰好就是这个粒度）。`OwnershipRegion`（engine × group × layer × block × token region，L1676-1684）的细度只服务诊断（L2458）和 Stage 2 的局部复用。
5. **是否引入新状态源或双重记账**：它是 Worker 侧释放安全的唯一状态源，Scheduler 被明确禁止跨进程直读（L1880），经 `RankTransferStatus` 单向上送 ✓；但与 `SchedulerReleaseLedger`（§13.4）形成「Worker 精确 ledger + Scheduler 保守近似」双结构，两处规则文本已出现互相矛盾（C-FH-1 的 §9.5 vs §13.4 finished_sending 前置条件）——这是规格缺陷而非抽象本身过度。
6. **更小替代方案**：fence 上挂每操作 in-flight 计数 + unknown_inflight 标志 + plan-region invalid 过滤器；保留 ledger 接口与 quarantine 语义，Stage 1 内部实现降为计数器，细粒度 region 留到 Stage 2 局部复用时引入。

**分类：必须保留但应简化**（保留抽象与对外语义，Stage 1 实现粒度降至每请求每操作计数；细粒度 region 是 Stage 2 能力，按纪律不构成 Stage 1 缺陷，但当前规格写着细粒度实现属于可削减范围）。

---

## Phase 2 候选复核

### C-PE-3（P2：§8.5 冻结规则与 PE 两段式分配冲突）——**确认，并扩展**

- 复核结果：**确认**。独立复核框架语义：每 waiting 迭代初始化 `num_external_computed_tokens=0`（U:scheduler.py:718-719）；connector 查询只在 `num_computed==0` 分支发生（:774-789）；load_async 首次分配 `num_new_tokens=0`、只覆盖 local+external（:834-837, 942-954）；`update_state_after_alloc(request, get_blocks(request_id), num_external)` **无条件**每调度 pass 调用（:969-974）；promote 后走 else 分支（:822-827）→ 第二次调用 `=0` 且 blocks 为全量。故 PE 侧 tail blocks 只在 `=0` 调用出现、Store miss（`H_PE=0`，sibling 返 0，无 winner，请求同步调度）时**所有**调用均为 `=0`、且 chunked prefill（:867-869）下 blocks 逐 chunk 增量到达——字面规则下这三类都没有冻结来源，与 Phase 2 描述一致且更强（chunked prefill 情形为本文补充）。
- **扩展**：同一缺陷命中 **DE_PARTIAL_READ 的 PE Forward 源 `[K_DE,R)`**——PE 首次调用只冻结 `[0,K_DE)`，Forward 源 tail blocks 同样在 promote 后的 `=0` 调用才到达（Phase 2 只列了 PE_READ 形态）。
- 严重级别：维持 P2。理由不变（修复方向明确：把 L681-684 改写为分侧语义——「DE manifest/Store target 仅在 `num_external>0` 首次调用冻结；PE Forward source 允许后续调用（含 `=0`）按 `request_key` 幂等增量扩展，或以 `build_connector_meta` 每步携带最新 block slice」）。升级论据存在（照字面实现则 PE_READ 主路径必然绑定失败），但与 Phase 2 一致按「规格自相矛盾、可修复性高」立档。
- 附带：`request_key + decision_version` 幂等 key 在 pre-commit 冻结时无 version 可用（TB-2）。

### C-FH-3（P1：`S_DE` 二次 floor 与 probe 语义冲突）——**确认，维持 P1，补充分情形数学**

- 复核结果：**确认**。逐行复核 pool_scheduler.py：查询长度 `floor(P/G)*G`（:502-503）、`hit == request.num_tokens` 时减 1（:526-527）、`update_state_after_alloc` 的 `num_external == kvpool_cached - vllm_cached` assert（:604-614）、物理 load 的 +1 边界（:849-852）。
- 分情形精确化（本文补充）：设 G 对齐。① `P%G==0` 且 Store 全量：probe 得 `S_DE_raw = P-1 = R`；设计再 floor → `S_DE = P-G` → `K_DE = P-G < R`，full hit 系统性退化为 partial；若 adapter 把 `H_DE = P-G-L_DE` 传给 pool 的 `update_state_after_alloc`，assert 要求 `P-1-L_DE` → **AssertionError，EngineCore 崩溃**。② `P%G==1`：`token_len = R`，命中 ≤R，不触发减 1，`S_DE_raw = R` 时 `S_DE = R` ✓ 正确。③ `P%G∉{0,1}`：查询长度 `<R`，full hit 结构性不可达（与 discard_partial_chunks 语义一致，但 §4.4 L226/§14.2 L1981 的「coverage >= prompt_len-1 即 full hit」表述对此情形不成立——文本缺陷面）。
- 修正方向（与 Phase 2 一致，补一条约束）：`S_DE = min(S_DE_raw, R)` 不再二次 floor（probe 出口已含 store 侧对齐与 -1）；`commit_after_alloc` 以 `vllm_cached=L_DE, kvpool_cached=S_DE_raw` 自建/冻结 LoadSpec 过 assert；修订 §4.4/§14.2 表述并写明 full-hit 可达性受 `P mod G` 与 `discard_partial_chunks` 约束。probe 不能用「未 floor 的 R 查询」绕过——KVPool 查询入口本身 floor（:502-503）。

### C-PR-6（P2：invalid 未排除 DE 共享 prefix 块 `[0,L_DE)`）——**确认，维持 P2，补 PE 侧对应**

- 复核结果：**确认**。逐行复核 `_handle_invalid_blocks`（U:scheduler.py:2691-2760）与其内层（:2640-2689）：WAITING 中的 async load 请求（`skipped_waiting`，evict=False）与 **running 请求**（evict=True）都被扫描；命中 invalid 集合的 running 请求被截断到首个失败块边界（:2665）、其失败块及下游块被收集逐出（:2672-2673）；fail 策略下逐出执行（:2736-2737）且所有受影响请求 ID 一并 fail（:2740-2748）。`[0,L_DE)` 是 prefix cache 共享块，其他 running 请求可能正引用——把单请求失败放大为无关请求 `FINISHED_ERROR` + prefix cache 误逐出，机制链完全成立。
- **扩展**：PE 侧共享段 `[0,L_PE)` 同构（PE 失败按 §19.2「收集请求相关所有 KV block IDs」L2446 同样未排除）；PE_READ 路径 DE 侧 Forward 失败的 invalid 收集亦同。
- 修正：§19.2/§12 明确 invalid 集合只含本请求私有区间（DE `[L_DE,⌈R⌉)`、PE `[L_PE,⌈K_DE⌉)` 对应块），`invalidate_request()` 按 plan `BlockRegion` 过滤 engine-local 共享段——该过滤用 plan 的 token range + block ids 即可表达，不依赖 ledger 的逐 layer 粒度（与过度设计检查 C 的简化方向兼容）。

### C-PR-8（P2：Store→Reverse 可见性仅依赖 host 顺序）——**确认，维持 P2，host 侧证据补强**

- 复核结果：**确认，且 host 侧比 Phase 2 描述的更闭合**。补强证据：`m_store.get` 在 recv 线程内同步调用、返回后立即评估逐 key 结果、最后才 `set_finished_request`（kv_transfer.py:902-942）；两个现役 backend 的 `get` 都是同步 RPC 并返回逐 key 失败集（yuanrong_backend.py:169-199 的 `mget_h2d`；mooncake_backend.py:233-266 的 `batch_get_into_multi_buffers`）——**host 语义上 `m_store.get` 返回即完成**，不是「调用即返回」的异步形态。`[当前源码已确认]`（调用与结果求值结构）。
- 仍然开放的只有两点：① 设备级——HBM 写对随后 Mooncake TransferEngine 的 RDMA/device 读是否立即可见，Python 接口之下，依赖 CANN/ADXL/H2D 拷贝的完成语义 `[外部接口待确认]`；② 现有 Forward 路径的 SendTask 带 `wait_event=reshape_cache_event`（MLC:1810-1811），send thread 传输前 `wait_event.synchronize()`（MLC:490）——**Reverse 的新 `submit_layer_send` adapter 给 SendTask 配什么 event，设计完全未规定**（§12 L1739 只有 host 级 acquire 时点）；若实现随手 record 一个空 stream 事件，fence 形同虚设。
- 维持 P2：升级为 P1 的条件（`m_store.get` 实为异步）在当前 backend 证据下不成立。修正：设计写明「Store bulk load 的 host-synchronous 契约依赖 + Reverse SendTask 的 wait_event 语义（如以 store 完成后的显式同步点 record）」，并把 backend 完成语义列入实施门槛验证项（§22）。

### C-PR-11（P2：`decide_on_pe` 触发点未钉死）——**确认，维持 P2，补一条边界**

- 复核结果：**确认**。真实 `L_PE` 的唯一可信来源是 waiting 循环里 `get_num_new_matched_tokens(request, num_new_local_computed_tokens)` 的第二参数（U:scheduler.py:774-779，本地 lookup :761-763 在其前）；`on_new_request`、control 接收线程均无 `L_PE`。设计 §8.6 L697-703 把 `pe_local_tokens` 作为入参但未禁止在拿到真实值前 decide；§8.2 L562-563「PE 此时可得到真实 L_PE」的「此时」指派发后任意时刻，不钉死为 scheduler 查询时刻。若按 `L_PE=0` 过早 commit DE_PARTIAL_READ：`L_PE ≥ K_DE` 的真实情形下首次调用返回 `max(K_DE-L_PE,0)=0` → winner 旁落 AscendStore → PE 不进 WAITING、立刻读尚未 Reverse 写入的 `[0,K_DE)` 脏前缀 + Forward 区间错账（scratch S4 序列）。
- **补充边界**：该约束只适用于经 PE 模型请求决策的路径（PE_READ/DE_PARTIAL_READ）；DE_FULL_HIT 的 control-only decide 不需要 `L_PE`（静态规则 1 只依赖 DE coverage，L438, L493）——修正文字应明确两个通道各自的 decide 时点，避免把 control-only 也错误地绑到 scheduler 查询上。
- 修正（与 Phase 2 一致）：§8.2/§8.6 写死「`decide_on_pe` 只能在框架传入真实 `num_computed_tokens` 的首次 `get_num_new_matched_tokens` 调用内（或之后）执行；此前一律 `(None, False)`」。

---

## 新发现

### TB-1：Store 物理 +1 slot 写出格于 §10.1「token 区间不越过 R」的 plan 模型

- 标题：Store 物理 +1 slot 写出格于 §10.1「token 区间不越过 R」的 plan 模型
- 严重级别：**P3**
- 类型：设计模型与继承行为不一致（文档/校验规则）
- 证据等级：`[当前源码已确认]`（KVPool 行为）+ `[设计推导]`（冲突解读）
- 设计位置：§10.1 L1417（「不越过 R」）、L1418（DE Store 只负责 `[L_DE,K_DE)`）；§5.2 L266-276
- 源码位置：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py:848-852`（`kvpool_cached == len(prompt)-1` 且非块对齐时 `num_tokens_to_compute += 1`）
- 触发前提：DE_FULL_HIT 且 `(P-1) % block_size != 0`（`kvpool_cached = P-1`）。
- 具体时序：T0 probe 得 `kvpool_cached=P-1` → T1 commit_after_alloc 冻结 LoadSpec → T2 `_process_async_load_request` 把 `token_len` 调整为 P → T3 bulk `m_store.get` 物理写入含 token index R（=P-1）的 slot → T4 `STORE_DONE` → T5 promote → T6 DE 重算 index R 覆盖写该 slot。
- 为什么不正确：plan 的 `store_target`/region 模型被限定为「不越过 R」，而继承的 KVPool 物理写入可达 `[0,P)`；若实现按 L1417 对 plan 与 ReqMeta 做一致性校验会误报，按 region 推导写区间则会少算一个 slot。物理安全本身无问题（slot 在首次分配的 ⌈R⌉ 块内；两次写由 `STORE_DONE` barrier 定序）。
- 可能影响：实现者按字面模型加校验 → 误拒绝合法 full-hit load；或文档读者误判存在越块写。
- 与已有模块的兼容性：+1 是 KVPool 既有语义，不可修改（§3.1），设计只能容纳。
- 是否属于过度设计：否。
- 最小修正建议：§10.1/§5.2 补一条「Store bulk load 的物理 token_len 可在末 slot 越出 R 一格（KVPool +1 边界），该 slot 随后由本地重算覆盖，region 校验按 max(token_len, R) 放行」（与 C-FH-3 修正一并落地）。
- 修正后需要增加的测试：UT——`(P-1)%block_size!=0` 的 DE_FULL_HIT 下 store_target 校验放行且物理写区间 = `[L_DE, P)`；E2E——full hit 末 token 重算后输出正确性（§23.3 已有条目 2，标注覆盖非块对齐 P）。
- 置信度：高（两侧均逐行核实；影响为文档/校验级）。

### TB-2：§8.5 幂等 key `request_key + decision_version` 在 pre-commit 冻结时 version 未定义

- 标题：§8.5 幂等 key `request_key + decision_version` 在 pre-commit 冻结时 version 未定义
- 严重级别：**P3**
- 类型：规格小缺口（键定义）
- 证据等级：`[设计推导]`
- 设计位置：§8.5 L681-684；§8.1 L489-497（`decision_version` 是 `PathDecision` 字段，commit 时才存在）
- 源码位置：U:scheduler.py:969-974（首次 `update_state_after_alloc` 必然发生在 commit 之前，由四阶段协议 L514-516 决定）
- 触发前提：任何路径的首次冻结（DE T3 / PE T7）。
- 具体时序：T0 `get_num_new_matched_tokens` → T1 分配 → T2 首次 `update_state_after_alloc` 需按幂等 key 冻结——此刻 `PathDecision` 尚未生成，`decision_version` 无值 → T3 commit 后第二次调用按 `(key, version)` 校验，与 T2 的冻结记录键不匹配。
- 为什么不正确：幂等键的两半在不同协议阶段才齐备，规格未定义 pre-commit 形态。
- 可能影响：实现者各自发明（version=0/None/单独 candidate key），行为不一；与 C-PE-3 的修复叠加时易出错。
- 与已有模块的兼容性：无冲突。
- 是否属于过度设计：否。
- 最小修正建议：在 §8.5 写明「commit 前冻结以 `request_key` + 保留版本值（如 `decision_version=0` 表示 pre-commit candidate）键控；commit 后用真实 version 升级同一记录」；并入 C-PE-3 的分侧冻结规则一起改。
- 修正后需要增加的测试：UT——同一请求两次 `update_state_after_alloc`（pre-commit 与 post-commit）幂等不双冻结、不错配 version。
- 置信度：高（文本层面）。

### 正向结论（成立原因 + 前提 + 源码契约）

- **TB-P1（三条路径写入区间恰好划分声明区间）**：DE 的 `E_DE` 在三条路径下分别由 Forward `[L_DE,R)` / Store `[L_DE,R)` / Store+Forward `[L_DE,K_DE)+[K_DE,R)` 精确覆盖，无缺口无重叠；PE 侧同理。前提：`K_x` G 对齐（pool_scheduler.py:404-416）、`L_x` block 对齐（prefix cache 性质）、Forward 禁止覆盖 Store 前缀（L1426）。源码契约：U:scheduler.py:834-837, 942-954, 1004。`[当前源码已确认 + 设计推导]`
- **TB-P2（partial hit 未被表达成 sibling 自动组合）**：first-winner 守卫（U:multi_connector.py:397-399）+ 非 winner 非 Layerwise child 收 `(empty,0)`（AMC:41）+ 非 layerwise 下 `can_load=False`（pool_scheduler.py:589-593, 617-618）⇒ DE_PARTIAL 在 PE 只有一次 accounting、一段异步供给。前提：sibling `use_layerwise=false`（配置约束缺失归 C-PR-4）。`[当前源码已确认]`
- **TB-P3（PE compute tail 是 Forward 源的全能兜底）**：任意 `L_DE/L_PE/K_PE` 关系下，PE 的本地段+Store 段+计算段并集恒为 `[L_DE,R)`（含 `K_PE<L_DE` 极端情形——计算段 `[K_PE,R)` 必覆盖）；时序由 Store 整段 bulk（L1668）+ 框架 WAITING 门禁（U:scheduler.py:986-1006, 2517-2541）保证。`[当前源码已确认 + 设计推导]`
- **TB-P4（末 token 语义与框架精确咬合）**：`E_DE = R - L_DE` 声明 + 乐观 `num_computed = R < num_tokens` ⇒ promote 的 -1 分支结构性不触发（U:scheduler.py:2521-2522），DE 以 `num_new_tokens=1`（:847）重算末 token，零框架修改。前提：声明严格为 `R - L_DE` 而非 `P - L_DE`（设计 L281 满足）。`[当前源码已确认]`
- **TB-P5（DE 侧 F4 双调用良性）**：DE 一次分足 `[0,R)`（路径无关）→ 首次调用即完整冻结；第二次 `=0` 调用只可能在 promote 后到达（U:scheduler.py:822-827 路径），幂等 no-op；窗口内 blocks 被 delay_cache + inflight 预留钉住（:934-940, 949）。`[当前源码已确认]`
- **TB-P6（「PE 不得返回 R - L_PE」硬契约正确且承重，L295-297）**：若 PE 在 DE_PARTIAL 声明整个 decode-ready prefix，乐观 `num_computed=R` 将使 PE 跳过 `[K_DE,R)` 本地计算（U:scheduler.py:1004 语义）→ 静默脏数据；设计 §15.3 的分路径返回值（L2085-2106）与该契约一致。`[当前源码已确认 + 设计推导]`

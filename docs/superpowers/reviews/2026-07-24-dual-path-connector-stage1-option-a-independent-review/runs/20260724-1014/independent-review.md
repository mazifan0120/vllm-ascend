# DualPathConnector Stage 1 方案 A″ 独立架构评审报告

- 评审 run：`docs/superpowers/reviews/2026-07-24-dual-path-connector-stage1-option-a-independent-review/runs/20260724-1014/`
- 评审对象（冻结基线）：`inputs/option-a-detailed-design.snapshot.md`（2752 行，SHA-256 `9a7be636…852808`）
- 已有评审（冻结后对照）：`inputs/option-a-existing-review.snapshot.md`（177 行，SHA-256 `82980ecd…998d3d5`）
- 评审性质：独立评审。Finding Ledger（41 条 IR）在读取已有评审**之前**冻结（指纹 `492a5a25…365cf1`）；已有评审仅作对照层，产生 1 条 RF Finding。
- 证据文件：`evidence/00-baseline.md`（基线）、`01-current-control-flow.md`（当前控制流 + F1–F18 事实表）、`02-path-timelines.md`（三路径时序）、`03-independent-findings.md`（冻结 Ledger）、`04-existing-review-cross-check.md`（R 对照）、`05-coverage-and-self-check.md`（覆盖与自检）。

---

## 1. Executive Verdict

**有条件通过（当前不具备直接进入实现阶段的条件）。**

方案主干成立：三条路径的 happy path 时序在设计推导层面闭合；token accounting 骨架与 upstream Scheduler 的 assert/promote/聚合语义精确咬合；first-winner 仅承担 accounting 的编排定位正确；不修改既有组件的边界在大多数能力点上可由子类覆盖合法实现（12 条正向结论，见 §7）。

但存在 **1 个 P0 + 6 个 P1**，全部有具体时序与源码证据：

- P0：PE_READ 的 Store 失败在当前 KVPoolWorker 代码下必然静默数据损坏，且 A″ 在自设硬边界内无修复通路（IR-001）。
- P1：DE Store adapter 同根失败断裂（IR-002）、跨 Engine block manifest 通道缺失（IR-003）、跨 Engine 失败/取消传播缺失（IR-004）、pre-commit abort 无终态发布者（IR-005）、commit 应用线程竞态（IR-006）、`S_DE_raw` 语义冲突（IR-007）。

判据：P0/P1 全部修正并落实到设计文本后，方案可进入实现；修正均为**最小设计修正**（协议补充、语义钉死、边界显式授权），无一要求推翻主干。另有 P2×15、P3×19、RF×1（P2），应在设计冻结前处理。此结论与设计自身 §22 实施门槛一致：门槛未满足前不得进入实现。

## 2. 评审基线和实际采用的 checkout

| 项 | 值 |
|---|---|
| vllm-ascend | `/Users/leqi/Documents/Code/vllm-ascend`，branch `dev/dualpath`，HEAD `0ec11a4703b987d3103f8859f184c492f22bde88`，`v0.19.1rc1-991-g0ec11a470` |
| 工作区状态 | 含用户未提交文档修改（评审全程保留，未还原/覆盖；详见 `evidence/00-baseline.md` §4） |
| upstream 参考 | `/Users/leqi/Documents/Code/vllm`，branch `main`，HEAD `8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`，`v0.23.1rc0-1050-g8df14cfc8` |
| 版本匹配 | vllm-ascend 配套 upstream 为 **v0.23.0**（`docs/source/conf.py:77,85`）；本地 checkout 同 minor 系列但非精确 tag——**参考版本存在小幅偏差**，涉及 upstream 的结论均按此标注（含 F18：v0.23.0 后 3 个提交已改变 MultiConnector 语义，Ascend 覆写停留在 v0.23.0） |
| 输入隔离 | 两份 snapshot hash 评审开始/结束均与 manifest 一致；持续变化的原设计/原 review **从未读取**；existing review snapshot 仅在 IR 冻结后读取 |
| 验证能力 | 本机 macOS、无 NPU、未安装运行时：**全部结论为静态源码证据**，未做任何运行验证 |

## 3. 当前源码真实行为摘要

（完整版：`evidence/01-current-control-flow.md`，含 F1–F18 事实表与 file:line。）

- **Scheduler 异步 KV 流程**：connector 查询仅发生在 `num_computed_tokens==0`（`U:scheduler.py:774-779`）；`update_state_after_alloc` 对异步请求**调两次、第二次 tokens=0**；`load_async` 请求乐观置满 `num_computed_tokens` 并挂 `WAITING_FOR_REMOTE_KVS`；离开等待的唯一触发是聚合后的 `finished_recving`，不变式由 assert 强制；invalid blocks 先于 finished 消费；`kv_load_failure_policy` 默认 `fail`，失败 → `FINISHED_ERROR` + prefix cache 逐出。
- **聚合**：`KVOutputAggregator` 计数制——finished 需全 rank 上报同一 ID；**invalid 只取并集无 quorum**。
- **AscendMultiConnector**：配置序 first-positive，winner 唯一；**Layerwise 子类即使非 winner 也收真实 blocks+真实 tokens**（AMC:36）；其余非 winner 收 empty blocks+0（AMC:41）；Worker hook 无条件广播到全部 child；finished 取并集。
- **MooncakeLayerwiseConnector（父类）**：kv_both 下一次 `register_kv_caches` 完成全部注册；但 kv_both 下 `build_connector_meta` 恒走 consumer 分支（send 元数据不产出）、`start_load_kv` 只走 consumer 分支；`get_finished` 返回 `(set(), done_recving)`——**finished_sending 恒空、失败请求只进 invalid、早到 terminal 直接丢弃**；`request_finished(_all_groups)` 恒 `(False,None)`；`wait_for_layer_load` 是 `pass`；发送线程存在失败归因 bug（MLC:507 残留循环变量）。
- **AscendStore/KVPool**：`get_num_new_matched_tokens` **不是纯 probe**（惰性建 LookupKeyClient、命中建 LoadSpec、abort 无清理、REQ recv 无超时）；非 layerwise bulk load 无 DMA 拆分；**`load_async` 下失败上报断裂**——invalid 进线程私有集合无人读取，请求却无条件标完成（pool_worker.py:457-465；kv_transfer.py:942）。
- **DualPath foundation**：仅有骨架（connector.py 144 行 + config.py 304 行 + 26 个 UT）；注册链路完整；PathDecisionCoordinator/control transport/Reverse/fence/ownership/Store adapter/WorkerMetadata **全部尚不存在**——设计是对骨架的补全，「尚未实现」本身不计为缺陷。

## 4. 三条路径的端到端时序

（完整 T0–T17 表：`evidence/02-path-timelines.md` + `phase2-scratch/`。）

**PE_READ**：DE probe → 声明 `E_DE=R-L_DE` → 分配/冻结路径无关 blocks → dispatch 截断 PE 请求（`R=P-1`）→ PE 首 pass 以真实 `L_PE` decide+commit(PE_READ) 并返回 0 让出 winner → sibling AscendStore 以 `H_PE` 成 winner（唯一 accounting）→ Commit+Ack → PE Store bulk load `[L_PE,K_PE)` → sibling 发布 PE finished_recving → PE compute `[K_PE,R)` 并逐层 Forward `[L_DE,R)` → DE `FORWARD_DONE` → DE finished_recving → decode。committed 时刻=PE commit + DE Ack；无多 writer；完成契约两 Engine 各自闭合（正常路径）。

**DE_FULL_HIT**：DE probe → 声明/分配（同上）→ control-only proposal（不建 PE 模型请求）→ PE 按 `K_DE>=R` commit（不需要 `L_PE`）→ Ack → `commit_after_alloc` → 各 rank bulk load `[L_DE,K_DE)` → `STORE_DONE` → DE finished_recving → promote（`-1` 分支天然不触发）→ 本地重算末 token。末 token 语义与框架精确咬合（正向 POS-3）。

**DE_PARTIAL_READ**：DE probe → 声明/分配 → dispatch → PE 准入（`L_DE<K_DE<R` 且 `L_PE<K_DE`）commit → PE Multi 中 DualPath 以 `K_DE-L_PE>0` 成 winner → Ack → DE Store load `[L_DE,K_DE)` → `STORE_DONE` → 逐层 Reverse（PE 只写 `[L_PE,K_DE)`）→ 全量 `REVERSE_DONE` → PE finished_recving → PE compute+Forward `[K_DE,R)` → DE `STORE_DONE&&FORWARD_DONE` → decode。`REVERSE_DONE` 不进 DE 成功谓词的传递性论证成立（正向 POS-5）；Store/Forward 写区间不相交且串行定序，无已证实物理并发写（POS-6）。

异常场景（Store miss/partial/full、L_DE=0/>0、L_PE⋛K_DE、取消/超时/rank 分裂/迟到事件/shutdown）逐场景小时序见 `02-path-timelines.md`；**失败/取消类场景的不闭合点即 IR-002/004/005/010/011**。

## 5. P0/P1 阻塞问题

> 完整字段（触发前提/时序/影响/兼容性/测试）见 `evidence/03-independent-findings.md`。

### IR-001（P0，正确性）PE_READ 的 PE Store 失败必然静默数据损坏，A″ 边界内无修复通路
- 设计位置：§14.1 L1969；§15.4 L2131-2141；§3.1 L73-80。源码：pool_worker.py:457-465/1333-1337；kv_transfer.py:908-910,942；契约 U:v1/base.py:385-389。
- 时序：Store DMA 部分失败 → 失败块进线程私有集合（无人读）→ 请求无条件标完成 → sibling 发布 PE finished_recving → PE 在脏 KV 上 compute 并 Forward `[L_DE,R)` → 双 Engine「成功」，脏 KV 进两侧 prefix cache。
- 为什么不正确：设计承诺「Store 失败使请求失败」依赖的 failure gate 在当前代码不存在；后果是静默脏数据而非可检测错误。
- 最小修正：二选一——(a) §3.1 显式放宽一条：允许对 `KVCacheStoreRecvingThread` 做一行构造参数级修复（注入 invalid_block_ids）；(b) PE Store 改由 DualPath 内部 adapter 统一承担（放弃 sibling 复用卖点）。
- 验证：PE_READ Store 注入失败 UT+E2E（断言 invalid 上报、FINISHED_ERROR、无脏 KV 进 cache），锚定 `load_async=True`。

### IR-002（P1，正确性）DE Store adapter 强制 load_async 正踩同一断裂，修复窗口与归因规则未写入设计
- 设计位置：§9.7 L1334/L1347；§19.2。源码：同 IR-001；修复窗口存在于 recv 线程构造器。
- 时序：commit 后 bulk load 部分失败 → 失败不可见 → 请求标完成 → 误消费为 `STORE_DONE` →（FH）脏 KV 进 cache /（PR）脏前缀 Reverse 给 PE。
- 最小修正：§9.7 写明注入时点（register_kv_caches 后、首请求前）、私有成员清单、block→request 归因规则、§13.2 失败流程接线。
- 验证：FH/PR 部分 DMA 失败注入 UT。

### IR-003（P1，协议缺口）跨 Engine block manifest 双向传输通道缺失
- 设计位置：§8.1 L489-497；§8.7 L790-795；§10.1 L1391-1412/L1428-1431。源码：U:scheduler.py:774-779（查询先于分配）。
- 时序：commit 在 PE 分配前发出 → DE 构造 Reverse SendTask 需 `remote_block_ids`=PE blocks → 四类 envelope 无一承载（digest 不能还原内容）；Forward 方向对 DE manifest 同构。
- 最小修正：commit 后、data-start 前增加一次双向 manifest 交换（扩展 COMMIT/ACK 载荷或新增 envelope），并入 data-start 前置条件。
- 验证：三路径 manifest 交换 UT + digest 不匹配拒绝。

### IR-004（P1，失败语义）跨 Engine 失败/取消传播缺失：CONTROL_ERROR 有类型无流程
- 设计位置：§8.7 L790-795（CONTROL_ERROR 全文仅此一处）；§13/§19 均单 Engine 闭环。源码：U:scheduler.py:2517-2534（无兜底超时）。
- 时序：commit 后 PE 侧任何失败 → 无消息通知 DE → DE 永等 `FORWARD_DONE` → 请求永久滞留、blocks 泄漏；Receiver 失败 Sender 成功同构。
- 最小修正：定义 PATH_ABORT/CONTROL_ERROR 完整流程（对侧 `coordinator.fail()` → 本地 §13.2 终态）+ 数据面请求级超时。
- 验证：post-commit 各失败注入点 × 三路径 UT+E2E，断言双 Engine 收敛。

### IR-005（P1，生命周期）pre-commit/决策窗口 abort 无终态发布者；§9.5 与 §13.4 前置条件互相矛盾
- 设计位置：§9.5 L1108-1111；§13.4 L1829-1841。源码：U:scheduler.py:2144-2147/2162/2580-2586；pool_scheduler.py:896-901。
- 时序：声明+分配后、commit 前 abort → 保守 delay_free 成立 → 无 plan/无 Worker 状态 → 无人发布 finished → blocks 永久延迟释放；迟到 commit 的 load 状态被 build_connector_meta 清理抹掉。
- 最小修正：统一规则「任何 delay_free 可能成立的 ID 必须恰一次发布终态再清理」（含从未启动 plan 的情形）+ delay_free 注册时登记终态义务。
- 验证：决策窗口 abort × 三路径 UT。

### IR-006（P1，并发）commit/abort/get_committed 应用线程未指定：control 线程直写无锁 KVPoolScheduler 与 scheduler 循环竞态
- 设计位置：§8.7 L746-784；§9.7 L1250-1257；§20。源码：pool_scheduler.py:478-565/617-618/896-915（无锁、scheduler 循环内访问）。
- 时序：scheduler 循环迭代 `_loading_req_ids`/`load_specs` 时 control 线程并发写 → 字典迭代期写崩溃；或快照错位 → `STORE_DONE` 永久丢失。
- 最小修正：写明线程模型——control 线程只接收/校验/入队，状态变更由 DualPathConnectorScheduler 在 scheduler 循环边界统一应用；`wait_for_commit` 删除或钉死。
- 验证：commit × build_connector_meta 并发压力 UT。

### IR-007（P1，token accounting）`S_DE_raw` 语义与 probe 实际返回冲突：full hit 退化或 assert 崩溃
- 设计位置：§5.2 L271-274 vs §4.4 L214-228/§14.2 L1980-1981。源码：pool_scheduler.py:502-503/604/849-852。
- 时序：`G=256` 示例——`P%G==0`：二次 floor 削掉一个 chunk → full hit 系统性退化为 partial；`P%G==1`：恰好正确；其余余数：full hit 结构性不可达；照字面不接 floor 则触发 :604 assert。
- 最小修正：钉死 `S_DE_raw`=「KVPool lookup 原始返回语义（已 floor + full-hit -1）」，`S_DE` 只做 `min(…,R)`；修正 §4.4/§14.2 表述；补 `P%G` 真值表。
- 验证：`P%G==0/1/其他` × full/partial/miss accounting UT。

## 6. P2/P3 非阻塞问题

（每条完整字段见冻结 Ledger；此处一行摘要。）

**P2（15 条）**：
- IR-008 「只在 tokens>0 冻结 block plan」与两段式分配冲突，Forward 源 mapping 缺 tail（§8.5 L681-684；U:scheduler.py:942-954）。
- IR-009 `decide_on_pe` 触发点未钉死，早于真实 `L_PE` 则准入失效（§8.6 L697-703）。
- IR-010 DECISION_TIMEOUT 无 scheduler→worker 传播，失败路径不可兑现（§8.7 L810-811）。
- IR-011 DECIDING 取消/未启动 plan 的 terminal 义务与 CANCEL 通道未规定（§15.4 L2136-2138）。
- IR-012 未强制 sibling `use_layerwise=false`，误配 → can_load 竞争/第二条静默损坏路径（§16.2；pool_scheduler.py:589-593）。
- IR-013 请求级失效未排除共享 prefix blocks，invalid 误伤无关请求（§19.2 L2445-2448；U:scheduler.py:2665/2736-2737）。
- IR-014 继承 MLC:507 失败归因 bug：误杀 + 假阳性 DONE（§9.2）。
- IR-015 Store→Reverse 无 device-level 可见性 fence（§11.2；kv_transfer.py:902-942）。
- IR-016 probe REQ recv 无超时挂 scheduler 循环（pool_scheduler.py:1107）。
- IR-017 `m_store.get` hang 无 watchdog（pool_worker.py:840）。
- IR-018 `cache_transfer_granularity` 双源（§16.1 L2213 vs pool_scheduler.py:404-413）。
- IR-019 「handshake 成功」准入无现有机制（§7.3 L459；MLC:1869-1918 lazy 握手）。
- IR-020 不修改边界 vs 失败语义：私有成员绕行点需显式授权清单（§3.1）。
- IR-021 PE 侧 adapter/sibling 双建 LookupKeyServer/backend，同 ipc path bind 冲突（§9.7 L1284-1297；删 PE 视图分支）。
- IR-022 测试四类系统性缺口：决策面失败/跨 Engine 传播/abort-in-window/sibling validator；F11 回归未锚定 load_async（§23/§25）。

**P3（19 条）**：IR-023 Ack 门语义限定；IR-024 proposal 等待无超时；IR-025 数据面 shutdown 无有界 drain；IR-026 control-only 降级补派发未规定；IR-027 决策面无 tombstone；IR-028 §7.3「超过」措辞；IR-029 bind_gpu_block_pool 写 Worker 侧；IR-030 config 吞未知键；IR-031 §16.1 示例非法；IR-032 AMC params 唯一产出者未引用；IR-033 path_strategy 未钉死 static；IR-034 Store +1 slot 出格 plan 模型；IR-035 pre-commit version 未定义；IR-036 事件缺 decision_version/plan digest；IR-037 retention window 无取值/GC；IR-038 SchedulerReleaseLedger 状态副本冗余；IR-039 `_is_kv_producer` 未复制；IR-040 kv_port 语义未定义；IR-041 §25 无性能验收项。

## 7. 重要正向结论及成立前提

（完整版：Ledger「正向结论」节。）

1. **POS-1 first-winner 定位正确**：winner 仅 accounting，数据面授权全经已提交 PathDecision——与 AMC:32-41 实际语义对齐。前提：§15.2/§15.4 的纪律在实现中保持。
2. **POS-2 decision 协议与 first-winner 查询顺序自洽**：proposal 原子到达 + `L_PE` 首 pass 可得 + decision 不依赖 PE Store 结果，无循环依赖；框架每 pass 重查支撑 `(None,False)` 重试。前提：IR-009 修正。
3. **POS-3 DE_FULL_HIT 末 token 语义与框架精确咬合**（U:scheduler.py:2521-2522），零框架修改。
4. **POS-4 路径无关 blocks 一次分配使 decision 改判零收敛**（§7.3 L470-474 + F4）。
5. **POS-5 `REVERSE_DONE` 不进 DE 成功谓词的传递性论证成立**：屏障机制是框架 scheduler 门禁而非 no-op 的 wait_for_layer_load。
6. **POS-6 无已证实物理并发写**：Store/Forward 区间不相交且串行定序。前提：fence 按 §11/§12 实现。
7. **POS-7 pending raw terminal inbox 正确修复父类早到事件丢弃缺陷**（MLC:1407）。
8. **POS-8 §13.2 终态规则是框架 assert/释放通道的充分条件**（U:scheduler.py:1579-1586/2576-2586）。
9. **POS-9 多 rank 聚合利用框架计数制 barrier，不另造 rank 集合**——取舍正确。
10. **POS-10 control 幂等规则完整**（message_id 复用 + payload_hash + 协议错误三分支）。
11. **POS-11 §13.4 delayed-free 使用框架真实通道**，不跨进程读 Worker 内存（内部冗余见 IR-038）。
12. **POS-12 kv_both 单次注册复用判断属实**（MLC:1359-1398 + register_buffer 守卫），共享 runtime 前提成立。

## 8. Token accounting 与 range ownership 矩阵

### 8.1 变量表（owner/区间均已逐行核对，详见 phase3-scratch/02）

| 变量 | 定义 | owner | 状态 |
|---|---|---|---|
| `P`、`R=max(P-1,0)` | prompt 长度 / decode-ready 前缀 | 两 Engine 各自 | 闭合（POS-3） |
| `L_DE`/`L_PE` | 各 Engine 本地已复用 tokens | 各 Scheduler | 闭合（不得互相复制，§5.3 L335-336 正确） |
| `S_DE_raw`/`S_PE_raw` | Store probe 原始返回 | KVPoolScheduler | **IR-007：语义未钉死（P1）** |
| `S_DE`/`S_PE`、`G` | 对齐命中 | DualPath vs KVPool 推导 | **IR-018：G 双源（P2）** |
| `K_DE`/`K_PE`、`H_DE`/`H_PE` | ready prefix / 新增加载 | 各 Engine | 闭合 |
| `E_DE=max(R-L_DE,0)` | DE 统一 external 声明 | DE Scheduler | 闭合（POS-4） |
| `F_PE_READ=E_DE` | PE 必须 Forward 的逻辑区间 | PE | 闭合（§5.2 L282） |
| 每 Engine 增量 | DE: `E_DE`；PE: `H_PE`（PE_READ）/ `max(K_DE-L_PE,0)`（PR） | 各 Scheduler | 闭合（§5.2 L284-297 硬契约正确：PE 若返回 `R-L_PE` 将跳过 tail 计算） |
| Store 对齐后 coverage | floor + discard_partial_chunks | KVPool | 闭合（IR-007 修复后） |
| partial hit 表达 | DE 统一声明 + PE 只声明 Reverse 写入段 | 两 Engine | 闭合——**未**被错误表达成 sibling 自动组合（§15.3 L2109-2115 正确） |

### 8.2 Range ownership（三路径汇总；完整逐路径表见 phase3-scratch/02）

| 路径 | token range | physical blocks | allocator | writer | reader | validity transition | release condition |
|---|---|---|---|---|---|---|---|
| 全部 | `[0,L_DE)` | DE 已有 blocks | DE 前缀复用 | 历史 writer | DE 模型 | 始终 valid（共享段） | 不参与请求级 invalid（**IR-013 修正后**） |
| PE_READ | `[L_DE,R)` | DE 新分配 | DE Scheduler | PE Forward（逐层） | DE 模型 | FORWARD_DONE 后 valid | 请求终态 + 无 owner |
| PE_READ（PE 侧） | `[L_PE,K_PE)` / `[K_PE,R)` | PE blocks | PE Scheduler | PE Store / PE compute | PE 模型→Forward 读 | sibling finished_recving / compute 完成 | 请求终态 |
| DE_FULL_HIT | `[L_DE,K_DE)`(= `[L_DE,R)`) | DE 新分配 | DE Scheduler | DE Store bulk | DE 模型 | STORE_DONE 后 valid | 请求终态 + drain |
| DE_PARTIAL | `[L_DE,K_DE)` | DE 新分配 | DE Scheduler | DE Store bulk | Reverse 读 | STORE_DONE 后可被 Reverse 读 | 请求终态 + drain |
| DE_PARTIAL | `[K_DE,R)` | DE 新分配（同组 blocks，区间不相交） | DE Scheduler | PE Forward | DE 模型 | FORWARD_DONE 后 valid | 请求终态 |
| DE_PARTIAL（PE 侧） | `[L_PE,K_DE)` | PE blocks | PE Scheduler | Reverse 写 | PE compute 读 | 全量 REVERSE_DONE 后 valid（Stage 1 屏障） | 请求终态 |

结论：无遗漏区间、无重叠 writer、无已证实物理并发写（POS-6）；例外项为 IR-013（共享段 invalid 过滤）与 IR-034（+1 slot 出格，P3）。

## 9. 组件兼容性矩阵

| 组件 | 判定 | 依据 |
|---|---|---|
| upstream Scheduler/聚合器 | 兼容（偏差已标注） | POS-2/3/8/9；F18 版本漂移需锚定（v0.23.0 后 3 个 MultiConnector 语义变更，Ascend 覆写停留在旧语义） |
| AscendMultiConnector | 正常路径兼容 | §4.1/§15.2/§15.4 对事实描述全部属实；IR-012（sibling 配置）、IR-032（params 唯一产出者） |
| MooncakeLayerwiseConnector 父类 | 覆盖路线合法可行 | F5/F6/F8/F9 均可由子类覆盖解决；继承 F15 bug（IR-014）、`_is_kv_producer` 偏差（IR-039）、kv_port 语义（IR-040） |
| AscendStoreConnector（PE sibling） | **P0 断裂** | IR-001；IR-021 双建冲突；IR-012 配置约束 |
| KVPoolScheduler/Worker（DE adapter） | 可组合但需修补 | IR-002（失败注入）、IR-006（线程模型）、IR-016/017（超时）、F10（非纯 probe → probe/commit/abort 设计方向正确） |
| LookupKeyServer | 兼容（条件） | 创建条件与设计 §9.7 声称精确一致（ascend_store_connector.py:119-120）；IR-021 冲突除外 |
| 既有 DualPath foundation | 兼容 | 构造方式、注册链路、config 骨架与设计一致；设计字段缺口属正常补全（IR-030/033） |

## 10. 过度设计与已有模块复用矩阵

| 抽象 | 判定 | 关键理由（六问详见 phase3-scratch） |
|---|---|---|
| PathDecisionCoordinator | 必须保留 | 唯一提交方+CAS 保护 decision 不变式，无现有模块承载 |
| DualPath control transport | 必须保留但应简化 | 禁改 MLC 使独立 control 必需；钉死/删除 `wait_for_commit`，补 CONTROL_ERROR 流程（IR-004） |
| SharedMooncakeTransferRuntime | 必须保留但应简化 | 保护 register_buffer 单次/唯一线程对；可省第二 worker 实例 |
| TransferPlan/Command/Event | 必须保留但应简化 | 逐层事件降 telemetry（与 RF-001 合流） |
| TransferFence | 必须保留 | 极简形态，跨 rank barrier 正确交给框架 |
| BlockOwnershipLedger | 必须保留但应简化 | Stage 1 只需 per-request/per-op owner 计数 + plan-region invalid 过滤 |
| Store adapter | 必须保留；PE 视图 YAGNI | 隔离 KVPool 非纯 probe 语义；删 PE 分支（IR-021） |
| raw terminal retention | 必须保留但应简化 | 修 F8 必需；量化 retention+GC（IR-037） |
| SchedulerReleaseLedger | 必须保留但应简化 | 删无读者的状态副本（IR-038） |
| wire identity + tombstone | 必须保留；tombstone 对称化 | 决策面 tombstone 缺失（IR-027） |
| TransferFenceRegistry | 证据不足（倾向 YAGNI） | 全文仅类图两处出现、无接口；建议降为 Worker 内部 dict |
| 不引入 transfer_epoch | 正确 YAGNI 回避 | 与 R-005 一致 |

## 11. 失败、取消、重试和多 rank 场景矩阵

| 场景 | 设计承诺 | 评审结论 |
|---|---|---|
| Store miss / lookup unknown | 转 PE_READ（非失败） | 闭合（§19.1 L2432） |
| Store load 失败 | 请求 FINISHED_ERROR | **不闭合**：IR-001（PE）/IR-002（DE） |
| Reverse/Forward 单层失败 | 请求级 FAILED | 单 Engine 侧闭合；继承 F15 归因 bug（IR-014）；跨 Engine 传播缺失（IR-004） |
| 一个 rank 成功一个失败/超时 | invalid 并集先杀、finished 计数扣住 | 闭合（与 F12 对齐）；永久等待未排除（无 transfer 超时，IR-004/017） |
| Sender 成功 Receiver 失败 | — | **不闭合**（IR-004） |
| probe/alloc/commit 前取消 | — | **不闭合**：IR-005/IR-011；probe handle 回收见 F10 泄漏 |
| Store load/Reverse/Forward 阶段取消 | drain/quarantine 后释放 | 机制方向正确（POS-8/11），缺跨 Engine CANCEL（IR-011） |
| control retry/dup/乱序/迟到 | message_id 幂等 + hash 校验 | 闭合（POS-10）；迟到 commit 缺决策面 tombstone（IR-027） |
| terminal 早于 mapping | 暂存重新归属 | 闭合（POS-7）；retention 量化缺失（IR-037） |
| Engine restart 后 ID 复用 | incarnation 区分 | 闭合（R-005 一致） |
| shutdown 仍有 in-flight | quarantine/drain | 方向正确；数据面 drain 未有界（IR-025）；quarantine 本地解除触发器缺失 |
| 多 rank 提前成功/永久等待 | 全 rank barrier | 提前成功被防住（POS-9）；永久等待未排除（IR-004/016/017） |

## 12. R-001～R-005 复核矩阵

（完整版：`evidence/04-existing-review-cross-check.md`。）

| 条目 | 判定 | 对应 IR/POS | 摘要 |
|---|---|---|---|
| R-001 共享 runtime | 与独立发现一致 | POS-12、IR-014 | 单次注册/绕行必要性经源码证实；补充 send thread 归因 bug 与环境变量副作用 |
| R-002 逐层事件收敛 | 补充了独立评审遗漏 | → RF-001 | §10.3/§11.2/§12/§14.3 的正确性纠缠经核对属实 |
| R-003 路径无关 blocks + Forward 区间 | 部分正确 | IR-005/010/011、POS-4 | 区间修正已落实且闭合；「超时/取消时释放 blocks」缺兑现机制 |
| R-004 raw terminal retention | 与独立发现一致 | POS-7、IR-027/037 | 方向与源码证据一致；补充量化/GC 缺口 |
| R-005 删除 transfer_epoch | 与独立发现一致 | 过度设计矩阵 | 正确 YAGNI 回避 |

## 13. Review-triggered RF Findings

### RF-001（P2，文档一致性/协议）设计稿事件定义未按 R-002 收敛：§10.3/§11.2/§12/§14.3 仍以逐层事件参与正确性
- 设计位置：§10.3 L1517-1526；§11.2 L1608-1616 vs L1618-1621；§12 L1734-1743；§14.3 L2019-2023/L2034-2035；§23/24/25 措辞。
- 源码证据：框架唯一公开屏障是请求级 finished_recving（U:scheduler.py:2517-2534）；wait_for_layer_load 为 pass（MLC:1976-1977）。
- 时序：实现 A 按 §11.2 链等 `REVERSE_LAYER_DONE(layer)` 放行 compute；实现 B 按 L1618 等全量 REVERSE_DONE——两份实现行为分叉且各自能引用原文；§12 按 layer done 释放 ownership 使 blocks 在请求终态前进入可释放判断。
- 最小修正：逐层事件统一标注 telemetry-only；§11.2 改请求级单屏障；§12 释放锚定请求级终态；§14.3 图加注或改全量屏障；§23/24/25 同步。
- 验证：「Forward 仅在完整 REVERSE_DONE 后启动」UT + ownership 释放时点 UT。

## 14. 测试与可观测性缺口

- IR-022：§23 缺决策面失败（DECISION_TIMEOUT/传播）、跨 Engine 失败传播、abort-in-window、sibling 配置 validator 四类；F11 回归必须锚定 `load_async=True` 配置，否则测不到断裂路径。
- RF-001：§23/24/25 的逐层事件表述需与 telemetry-only 收敛。
- IR-041：§25 无性能验收项（决策延迟、传输吞吐、控制面阻塞上限）。
- 可观测性（§24）本身良好：请求级字段与 `shadow_*` 隔离标注正确；建议补「decision 等待时长分布」与「暂存事件滞留量」以支撑 IR-024/037 的运营观测。
- 现有 UT（26 个）只覆盖 config/构造/注册，无法发现本报告任何运行时 finding——属正常阶段状态，不扣分，但 §23 的新增项应与 IR-001/002/004/005 的「修正后测试」一一对应落实。

## 15. 按优先级排列的最小设计修改清单

1. **IR-001（P0）**：§3.1 显式放宽一条（允许对 recv 线程做一行构造参数级修复注入 invalid_block_ids），或改为 adapter 统一承担 PE Store；二选一写入设计。
2. **IR-002（P1）**：§9.7 补注入时点、私有成员清单、block→request 归因、§13.2 接线。
3. **IR-004（P1）**：§8.7 定义 PATH_ABORT/CONTROL_ERROR 全流程 + 数据面请求级超时。
4. **IR-003（P1）**：§8.2 增加 commit 后双向 block manifest 交换（扩展 COMMIT/ACK 或新 envelope），并入 data-start 前置。
5. **IR-005（P1）**：统一「delay_free 可能成立 ⇒ 恰一次终态」规则，修正 §9.5/§13.4 矛盾。
6. **IR-006（P1）**：§8.7/§20 写明 control 线程只入队、scheduler 循环内应用的线程模型。
7. **IR-007（P1）**：钉死 `S_DE_raw` 语义，修 §4.4/§14.2/§5.2，补 P%G 真值表。
8. **IR-008/IR-009（P2）**：冻结规则改「合并式 manifest + I/O 门槛」；decide 触发点钉在 PE 首 pass。
9. **IR-010/IR-011（P2）**：决策失败的 scheduler→worker 通道；CANCEL 复用 PATH_ABORT；终态义务统一。
10. **IR-012/IR-021/IR-032（P2）**：config validator 强制 sibling `use_layerwise=false`、删 adapter PE 视图、声明 params 唯一产出者。
11. **IR-013（P2）**：invalid 按 plan region 过滤，排除共享段。
12. **IR-014～IR-019（P2）**：F15 规避声明、可见性 fence、probe/get 超时、G 单源、handshake 收敛到初始化。
13. **IR-020（P2）**：§3.1 增加允许的私有成员接触点清单。
14. **IR-022 + RF-001（P2）**：测试/事件定义补齐。
15. **P3 批次（IR-023～IR-041）**：随设计冻结一并处理（措辞、配置校验、量化默认值、删除冗余状态）。

## 16. 仍需设计者回答的问题

1. IR-001 路线取舍：(a) 边界放宽一行修复，还是 (b) PE Store 收归 adapter？两者对 §15.7 卖点影响相反。
2. IR-003 manifest 交换的载体偏好：扩展 COMMIT/ACK 载荷还是新增 envelope 类型？
3. IR-037 retention window 的默认值、配置项与 GC 责任归属（Worker 本地还是 fence 生命周期跟随）？
4. `m_store.get` 内部是否有超时/取消语义（影响 IR-017 的修法）——`[外部接口待确认]`。
5. Store DMA → Reverse P2P 的设备级可见性保证出自哪个组件契约（Mooncake TE / memfabric 文档）——`[外部接口待确认]`。
6. TransferFenceRegistry 是否保留为独立类（当前证据不足，倾向降为 Worker 内部 dict）？
7. upstream 版本锚定：以 v0.23.0 tag 还是跟随 main（F18 的 3 个语义变更何时纳入 Ascend 覆写）？

## 17. 设计文档 26 个编号章节的评审覆盖表

| § | 章节 | 覆盖 | 结论/关联 |
|---|---|---|---|
| 1 | 文档目标 | 已完整读取 | 无问题（元信息） |
| 2 | 目标与非目标 | 已完整读取 | IR-001/IR-020：失败语义目标与不修改边界的张力 |
| 3 | 已确认硬约束 | 已完整读取 | IR-020（私有成员授权）；§3.5 与 POS-4 一致 |
| 4 | 现有组件逻辑 | 已完整读取+逐条源码核对 | §4.1/§4.2/§4.3 属实（F1/F8/F10）；§4.4 表述→IR-007 |
| 5 | 术语与 token accounting | 已完整读取+专项 | IR-007；其余闭合（§8.1 矩阵） |
| 6 | 请求 ID 与传输身份 | 已完整读取 | IR-027/IR-036；R-005 一致 |
| 7 | 路径模型 | 已完整读取 | POS-5；IR-028（措辞） |
| 8 | 路径决策协议 | 已完整读取 | IR-003/004/006/008/009/023/024/035 |
| 9 | 公共类设计 | 已完整读取 | IR-021/029/039/040；§9.2 可行性 POS-12 |
| 10 | 传输计划和事件接口 | 已完整读取 | IR-003/034/036；RF-001 |
| 11 | TransferFence | 已完整读取 | 判定必须保留；RF-001（§11.2 收敛） |
| 12 | BlockOwnershipLedger | 已完整读取 | 判定保留但简化；IR-013；RF-001 |
| 13 | 完成语义 | 已完整读取 | POS-8/9/11；IR-005/038 |
| 14 | 三条数据通路 | 已完整读取+时序重建 | happy path 闭合；IR-001/002/007；RF-001（§14.3 图） |
| 15 | A″ 编排 | 已完整读取 | POS-1/2；IR-011/012/032 |
| 16 | 配置设计 | 已完整读取 | IR-012/018/030/031/033/040 |
| 17 | 拓扑限制 | 已完整读取 | 无问题（合理 fail-fast） |
| 18 | 初始化与握手 | 已完整读取 | IR-019 |
| 19 | 失败处理 | 已完整读取 | IR-002/004/010/013/016/017 |
| 20 | 并发与幂等 | 已完整读取 | POS-10；IR-006/035 |
| 21 | 代码布局 | 已完整读取 | 无问题 |
| 22 | 实施门槛 | 已完整读取 | IR-012：门槛未含 sibling 配置断言 |
| 23 | 测试设计 | 已完整读取 | IR-022；RF-001 |
| 24 | 可观测性 | 已完整读取 | 无实质问题（shadow_* 隔离正确）；建议补两项指标 |
| 25 | 验收标准 | 已完整读取 | IR-041（无性能项）；RF-001 |
| 26 | 开发顺序 | 已完整读取 | 无问题 |

## 18. 本次未验证事项与证据边界

- **无运行验证**：本机无 NPU/运行时，全部结论为静态源码证据；「源码可达，尚未运行验证」类风险（IR-014/015/017 等）需 E2E 证实。
- **upstream 版本小幅偏差**：本地 checkout v0.23.1rc0-1050 vs 配套 v0.23.0；已识别 3 个语义漂移提交（F18），不排除其他未识别漂移。
- **外部接口待确认**：`m_store.get` 超时/取消语义（IR-017）；Store DMA→P2P 设备级可见性（IR-015）；Mooncake TE session 失效检测能力（IR-004 的间接检测不可依赖）。
- **未读取**：持续变化的原设计/原 review（按规约禁止）；existing review snapshot 仅在冻结后读取一次。
- **未修改**：snapshots、manifest、README、GOAL-PROMPT、原文件、vllm-ascend 源码/测试、upstream 仓库、其他 run——全部评审产物位于本 run 内（含 phase1/2/3-scratch）。
- **IR-003 置信度中高**：消息面缺口为逻辑推演，存在设计者可澄清的空间；其余 P0/P1 均为高置信度。

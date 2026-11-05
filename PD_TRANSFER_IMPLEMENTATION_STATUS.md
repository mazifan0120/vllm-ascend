# P buffer reuse 与 layerwise receipt：实现现状及剩余工作

2026-11-05 最新核对：已实际 fetch 并在原 `p-main-buffer-reuse` 纳入 receipt
`fa5aec40b` 的 12 个新增提交。已接生产 owner RPC／scheduler 门禁、Host 目标注册
选择、manifest 逐 entry 计划、精确 receipt 与通知 P 前的全 TP receive admission。
SourceReady 已接 manifest writer／exact epoch；D 接收依赖、整个 model invocation 的 owner-use、
all-rank readiness／ReleaseAck 和晚到 proof 保留已接线并通过控制测试。
TP16×78 层的 32K／64K／128K 元数据容量回归通过，保持原 wire 限额。
真实新注册／远端 WRITE、D→P 取消、restore-only 调度及完整模型功能和性能尚未验证。
当前清单与验证边界见 [最新补齐审查](PD_TRANSFER_COMPLETION_REVIEW.md)。
下文保留 2026-11-04 的整理记录，不以旧的分工或接口测试代替完整端到端验收。

首次 review 建议先读 [双方实现与验证摘要](CODE_REVIEW_SUMMARY.md)，其中包含固定代码快照、提交分组和建议审查顺序。

核对日期：2026-11-04。按用户要求，暂不继续64K异常定位；本文件从两分支代码核对开始，后续 A1/A2 开发进度见下一节。

**P独立的共享缓冲区／Store核心已经实现；receipt提供了传输与回执骨架，完整的
P共享Main → D Host Main／HBM Indexer → 稀疏decode／写回链路尚未接通。**
不需要重新开发已有Host offload kernel，也不需要重写TargetProvider四方法。

## 后续实现进度：A1/A2

A1/A2 现统一在本仓 `p-main-buffer-reuse` 分支，部署工作区为
`src/vllm-ascend-reuse`。本分支快进纳入已测 receipt 基线 `6c233175d`，保留原 P 历史，
再加入 A1/A2 代码、测试和文档。原 `reuse-pd-integration` 工作区仅保留历史快照。
下面的分支对照表记录整理时的基线；最新实现与接线边界见
[owner 交接](TARGET_OWNER_RUNTIME_HANDOFF.md)、[验证报告](TARGET_OWNER_RUNTIME_VALIDATION.md)
及 [开发计划](PD_TRANSFER_IMPLEMENTATION_PLAN.md)。
当前已写入真实资源 pin、TargetRuntime、注册 catalog、延迟绑定 facade、纯数据 IPC
适配器和 manager/connector 初始化入口。40 项相关 CPU、139 项 P 回归、27 项 receipt 账本、TE/HostPool 回归及最终快照
TP1/TP2 原生目标资源验证、实际 scheduler 验证与原环境恢复审计通过；完整 PD 模型端到端仍未接通，不能据此声称功能上线可用。

## 1. 两个分支的关系和位置

| 对象 | 固定提交 | 本地位置／说明 |
| --- | --- | --- |
| 设计基线 | `4234a1f1815252e8b52c75d6956d0bdf9ae80cc0` | `dsa_offload_rebase_30` |
| 我方分支 | `06e16d717b95f7720c0b85457351a6c27d30576c` | 整理时的 P 基线；当前同名分支已包含 receipt 基线及 A1/A2 |
| receipt集成分支 | `6c233175d95909ec8df1175f85e5e212496fa6ec` | `src/vllm-ascend-receipt-review`，独立detached worktree |

上述路径相对于部署仓库deploy-glm52。两个远端HEAD已重新查询，未变化。
`06e16d717`是`6c233175d`的祖先：receipt分支**已经合入我方源码**，并非两套互不相干的实现。
它相对我方HEAD增加8个提交，差异为8个文件，主要是connector、receipt账本、测试与交接说明。
核对阶段没有合并或部署；随后按用户要求在原 P 分支纳入 receipt 基线和 A1/A2。

共同的Main布局、Store worker、源接口和TargetProvider实现逐文件比较一致。
共同契约的差异是store side改称target side，类型和语义未变。
旧`m1-prefill-reuse`分支不纳入本次实现清单。

## 2. 代码从哪里读

公共前缀为`vllm_ascend/distributed/kv_transfer/`，下表路径相对于该前缀。

| 模块 | 职责 | 所属状态 |
| --- | --- | --- |
| `kv_pool/ascend_store/layerwise_cache_layout.py` | Main共享槽、独立Indexer、物理页与预算规划 | 两分支共有 |
| `kv_pool/ascend_store/backend/mooncake_layerwise.py` | 显式opt-in、拓扑限制、统一cache plan | 两分支共有 |
| `kv_pool/ascend_store/pool_worker.py` | Store恢复／保存、attention依赖、源视图及restore-only入口 | 两分支共有 |
| `kv_pool/ascend_store/kv_transfer.py`、`attention_fence.py` | 覆盖前等待、空GET、计算完成与失败传播 | 两分支共有 |
| `kv_pool/ascend_store/metadata.py`、`pool_scheduler.py`、`backend/session_tracker.py` | 请求实例、恢复长度、partial快照版本与提交状态 | 两分支共有 |
| `kv_pool/ascend_store/reuse_source.py` | SourceView／SlotEpoch／五方法consumer契约及registry | 两分支共有 |
| `sparse_kv_offload/target_provider.py` | D目标四方法核心、DSA布局工厂、TargetRuntime协议 | 两分支共有 |
| `ascend_multi_connector.py` | P外部waiter／源consumer发现；集成树另加D provider注入 | receipt扩展共同文件 |
| `kv_p2p/mooncake_layerwise_contract.py` | RequestKey、TransferKey、TargetLease、ReceiveReceipt等纯数据契约 | 两分支共有，receipt措辞更新 |
| `kv_p2p/mooncake_layerwise_receipt.py` | P SlotLedger、D ReceiptLedger | receipt新增 |
| `kv_p2p/mooncake_layerwise_connector.py` | 消息、metadata下发、目标申请与release入口、逐层sender | receipt扩展 |
| `sparse_kv_offload/sparse_kv_offload_manager.py`、`mooncake_host_pool.py` | D Host池、稀疏访问和写回的基础运行时 | 基线已有 |
| `kv_p2p/mooncake_connector.py` | 已有blockwise DSA布局／注册及分阶段传输参考 | 基线已有，不等于layerwise已接通 |

## 3. 功能清单：不要把接口存在当成链路可用

“已实现”指相应代码存在；硬件证据另见第7节。“待接线”表示两端能力存在，但生产调用链没有闭合。

| 功能 | 当前状态 | 具体边界／缺口 |
| --- | --- | --- |
| P Main跨层共享、首尾等独立层、Indexer独立 | 已实现 | 使用同一物理planner分配和计算预算 |
| P写槽前等待计算、Store、外部reader | 已实现 | 外部waiter与空GET保护已接；不等于新sender五方法已接 |
| 完整前缀恢复、partial／private full快照 | 已实现 | Store实例和快照代次内部管理，不作为PD wire私有key |
| 源视图、open／ready／seal／wait／abort | 接口及Store接线已实现 | 真实sender尚未声明version 1或实现consumer委托 |
| 无model forward的restore-only | 显式driver已实现 | scheduler没有自动选择路径；当前receipt sender也没有消费该driver |
| D Host offload基础运行时 | 基线已有 | manager／Host池／算子不需重写；新layerwise路径尚未绑定 |
| TargetProvider四方法与布局工厂 | 核心及集成树owner factory已实现 | 新manager入口实际调用工厂；完整模型初始化仍需B/J1配置依赖 |
| TargetRuntime retain／make_visible／release | A1集成树已实现 | 真实pins、scheduler guard、stream依赖与静止release已测；模型调用范围仍需B接线 |
| scheduler与worker共享租约状态 | owner/facade和纯数据IPC适配已实现 | 真正父子进程测试通过；B仍需绑定生产transport及retention查询 |
| 注册后构造catalog并挂载provider | A2集成树已实现 | 延迟facade、真实注册catalog、manager/connector绑定入口已测；B须输出完整Host注册 |
| receipt消息、双账本、metadata规格下发 | 骨架已实现，需修正 | 已有具体协议问题，见第5节 |
| retention token签发 | 增加真实KVCacheManager核验与guard | 真实scheduler分配/追加/retire已测；生产跨进程查询、free/cancel/preempt hook仍待B4/J1 |
| 目标manifest语义传至P | 未完整实现 | P仍消费旧remote metadata，未消费TargetLease视图／writer plan；不要求新建同名类，但字段语义必须完整 |
| Host Main／device Indexer目标传输计划 | 待实现 | 新layerwise sender未绑定D Host布局，也未按内存类型独立组批 |
| Main唯一writer与Indexer各消费rank覆盖 | 待实现／待接线 | sender未应用writer plan；不能按group号或tensor序号推断组件 |
| partial有效字节范围、D required ranges差集 | 待补完整计划 | 旧sender主要按整block地址／长度传输，不能作为新契约的有效尾部传输证明 |
| D完整receipt后建立消费stream依赖 | 本地runtime已实现并原生验证 | B仍需提供真实远端完成依赖和worker use范围，修复receipt强制门禁 |
| 取消、抢占、远端DMA及本地写回静止后回收 | 未闭合 | release入口存在，但远端／跨rank静止汇总及scheduler安全回收仍缺 |
| 丢回执／依赖失败后的有界失败与退出 | 未闭合 | 无独立D receipt watchdog；P槽位300秒等待不等于D失败／安全回收保证 |
| 可部署的receipt＋D Host配置和端到端验收 | 尚未交付 | 现有a3-reuse是旧connector、INT8、普通D对照档 |

```mermaid
flowchart LR
    P["P Main共享槽 + Store：已有"] --> S["源视图／代次／ready：已有"]
    S -. "consumer待接" .-> T["layerwise sender + receipt骨架：需补齐"]
    T -. "目标视图、writer、分组件传输待接" .-> D["D TargetProvider核心：已有"]
    D -. "A1/A2本地能力已实现；B/J接线待闭合" .-> R["D Host稀疏decode／写回：基线已有"]
```

### P端还缺的关键衔接

receipt分支明确把五方法consumer列为后续轮次；当前仍从旧`save_kv_layer` hook发送。
MultiConnector发现没有version-1 consumer时会跳过新协议，因此“两个分支已合并”不等于“源契约已在真实网络sender运行”。
需要让sender使用真实SourceView和SlotEpoch登记整批读者，再发布任务；旧hook必须去重。
`wait_for_layer_reuse`应与新源代次共享同一状态，不能保留互相独立、可能提前放行的完成标志。
restore-only还需scheduler在D目标admission之后选择driver，并避免同一metadata再进入正常forward。

### D端还缺的关键适配

真实runtime应复用既有manager和TE分配结果：retain核验有效scheduler retention并持有owner引用；
make_visible证明远端Host／device写入完成并接到正确消费stream；release等计算／写回静止后同步解除引用。
跨进程facade必须指向同一租约状态，不能scheduler acquire后让另一个空状态provider直接prepare。

MultiConnector构造时就扫描provider，而完整catalog要等manager分配／TE注册之后才能构造。
需要可先安装、后绑定catalog的facade，或等价的分阶段初始化；只在注册后加一个属性并不能解决次序问题。

## 4. 剩余工作及分工

A指我方P复用／目标内存侧，B指PD connector侧。分工按mc.pdf的所有权划分，不重复移植Host runtime或分配第二套请求blocks。

| 编号 | 工作 | 主要负责 | 完成标准 |
| --- | --- | --- | --- |
| A1 | D owner-side TargetRuntime | A | 真实retain／visibility／同步release；不使用空回调 |
| A2 | 导出已分配D布局、catalog与owner facade | A，B提供注册输出 | 真实backing／region／层／group／容量／代次一一对应 |
| B1 | sender接入P五方法consumer | B，A配合 | 登记后入队、ready后读源、整代封口、源静止后释放；旧hook不重复发 |
| B2 | 目标发布和分组件数据面 | B | admission及worker就绪后P才能开始；真实目标身份、writer plan、Host／device分批、有效tail范围 |
| B3 | 修正receipt协议及普通模式兼容 | B | 第5节问题有真实方法回归；关闭功能保持旧路径 |
| B4 | scheduler retention、远端／跨rank静止及回收确认 | B，A提供本地runtime状态 | 正常结束、取消、抢占均不提前归还blocks；失败可保留并安全回收 |
| B5 | 可靠错误通道、有界等待及安全退出 | B，A配合Store失败保护 | 丢回执或原生传输卡住不能只留下日志；区分等待失败和DMA已静止 |
| J1 | scheduler-worker facade／IPC和初始化接线 | A+B | 锁、tensor、事件留在owner；纯数据lease/catalog跨进程，完成确认真实 |
| J2 | restore-only scheduler路径 | A+B | D目标已持有后逐层恢复并发送，全程无计算hook，失败不放行D |
| J3 | 同布局BF16 Main联调配置 | A+B | 相同TP、Main Host／Indexer HBM、明确拒绝不支持组合 |
| J4 | 内容、decode写回、取消及内存／性能验收 | A+B | 正常及单路径失败都具备内容对比和资源生命周期证据 |

## 5. 现有实现需要先修的地方

以下与“没有实现”分开列，均针对receipt固定提交。前7项此前已有CPU复现，本次在完整独立worktree重新运行确认。

| 项目 | 当前行为／风险 | 应补的行为 |
| --- | --- | --- |
| 普通模式门控 | D即使无provider、开关关闭仍传播instance；P进入receipt而D无账本 | 仅在明确启用且目标admission成功时参与新协议，普通路径保持兼容 |
| backing身份 | P源ticket及Indexer wire使用空backing，契约拒绝 | 绑定真实身份／合法ticket语义，不能伪造ID绕检查 |
| 覆盖和失败账本 | 按backing／consumer汇总，未按必需层计划检查；FAILED条目未阻止ready | 固定required层／组件／entry／writer／范围，失败保留为失败 |
| terminal状态 | 历史失败先从集合清掉，随后可能发success terminal | 发送完成前冻结请求判定；不依赖旧FAILED消息兜底 |
| visibility门禁 | 缺provider／lease仍可能跳过prepare并发布done | 不完整状态拒绝放行，prepare失败进入明确错误通道 |
| admission先后 | 先提交通知P的异步任务，再acquire target；未知instance回执被丢弃并ACK | 持有目标且D worker账本就绪后再发布，或提供等价可靠握手 |
| request清理 | 内部ID传给保存外部ID的forget入口，残留旧状态 | 生命周期路径统一身份，测试ID重用与旧回调 |
| 组件识别 | emitter按primary tensor个数标Main／Indexer，没有遍历同层所有组件 | 依据真实组件视图／spec，不按tensor数量猜测 |
| connector测试收集 | 类名以Tests结尾，默认pytest只收集Test开头的类 | 调整类名／配置并真正运行connector接线测试 |

本次新增组件复现：一个Main tensor、独立Indexer data/scale，两者同属group 0，sender实际传输任务含两个layer name，
但回执只发Main，没有Indexer。数据传完也不能据此证明Indexer账本就绪。
实际`register_kv_caches`保留不同组件名、合并同逻辑层，与此结构相符。

还需确认原始传输错误的源静止语义；不能把调用返回错误或线程仍存活直接当作DMA已终止。
前述CPU复现不等于已经在NPU端到端观察到所有错误放行；例如旧FAILED消息可能另行拦住请求，但不能替代receipt自身正确性。

## 6. 明确的首版范围

- P producer、V1、eager、PP=1、CP/DCP=1、无MTP/speculative；支持已核实的Main／Indexer cache spec和单manager group。
- 首次目标链路使用相同P/D TP、相同Main布局与BF16精度；P独立INT8 1M档不能直接与D BF16 Host Main拼接。
- v1一个backing一个region；大池分段注册不能靠伪造backing ID绕过，需要限制配置或共同扩展契约。
- 传输／分配代次当前固定0，完整的重试／抢占代次处理未交付；不能靠递增一个数就立即复用仍在途的内存。
- D侧跨层复用、不同TP、量化转换、图模式与MTP/DCP支持不属于本次补齐首版链路的默认要求。

这些限制应显式校验和记录；不将主动不支持的组合混入“忘记实现”清单。

## 7. 已有证据和验收欠账

| 项目 | 可使用的证据 | 不代表什么 |
| --- | --- | --- |
| P独立1M | 较早r05版本1048541-token冷prefill及首token、前缀检索通过 | 不代表最新两分支或receipt端到端1M通过 |
| P源接口 | r07受控读者／NPU／Store 64K，以及人工KV的恢复和阻塞验证 | 不代表真实网络sender已接入 |
| P当前代码 | 本次139项聚焦CPU测试通过 | 不覆盖整个上游测试或所有NPU路径 |
| receipt集成树的共同实现 | 本次相同139项聚焦CPU测试通过 | 不覆盖新connector网络／scheduler生命周期 |
| receipt纯账本 | 本次27项CPU测试通过 | 不能覆盖第5节真实方法接线缺陷 |
| 真实connector测试 | 仅做依赖占位后的收集检查：默认配置收集0项 | 没有执行这些测试体，不是NPU验收 |
| 目标Host Main＋HBM Indexer全链 | 尚无本次目标链路验收 | 普通D HBM的旧connector通过不能替代 |

64K间歇性错误答案／纯计算NaN仍是单独的未解决验收项，已按用户要求暂不继续定位。
此处不预判它属于复用或receipt问题，也不把后续重放通过作为修复结论。

本次只执行本地读取、独立worktree和CPU检查，没有集群命令或生产源码行为修改。
源码／测试证据位于部署仓库`artifacts/p-main-buffer-reuse/two-branch-inventory/`：

- `inventory.json`：固定SHA、分支关系、差异及测试日志索引。
- `protocol-gap-repro.py/json`：原有协议问题在完整receipt worktree的复现。
- `extra-gap-checks.py/json`：组件识别、源consumer及目标视图调用检查。
- `connector-collection.log`：只验证pytest收集规则，依赖占位不作为运行时测试。

两个交接文档仍是接口细节入口：[P／目标侧交接](PD_TRANSFER_HANDOFF.md)、
[receipt侧交接](https://github.com/mazifan0120/vllm-ascend/blob/6c233175d95909ec8df1175f85e5e212496fa6ec/RECEIPT_HANDOFF.md)。

## 8. 建议推进顺序

1. 先完成B3协议修正和测试收集，确保普通模式及失败路径不退化；A1/A2可同时开发。
2. 用真实NPU源、Host Main／device Indexer测试目标接B1/B2，验证数据、有效tail、慢reader及源槽保护。
3. 接A1/A2、J1和B4，完成真实D decode／写回、取消／抢占／重分配保护及有界失败。
4. 完成J2无计算restore-only，再交付J3配置并执行几十K目标模型联调。
5. J4完整短链路通过后，再推进更长上下文和性能优化；本轮不重启64K数值问题定位。

receipt树已经包含P分支，适合作为后续共同集成的起点；不要再次整批cherry-pick同一P实现。
两分支核对阶段未合并或部署远端；A1/A2 现统一交付到原 `p-main-buffer-reuse` 分支，详见文首更新。

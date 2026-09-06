# MooncakeConnectorV1 DSA multi-TP 传输设计（总体设计定稿）

- 日期：2026-09-05
- 基线分支：`dsa_offload_rebase_main_0912_with_connector`
- 基线提交：`026611dbbe52637aae55edfce7659351a74947bd`
- 状态：Q1–Q8 及复审 R1–R4 已收口，作为核心 connector multi-TP 的总体设计基线；用户已于本轮批准开始实施；不代表全部兼容配置的详细设计、代码或验收已经完成。

## 1. 目标与范围

让多个 Decode TP rank 并行拉取 Prefill Main KV，写入 Decode TP 组共享的同一份 Host KV Pool。必须支持 Prefill DCP > 1；Indexer 继续通过 D2D 写入各 Decode rank 的设备缓存。最新兼容目标为普通 MooncakeConnectorV1 的全部合法并行配置；此前 Decode DCP=1 是当前 offload 能力边界，不能据此静默缩小最新目标，相关差距见第 11 节。

保持现有 Host KV 物理布局以及 fused Host→Device 消费方式；按已确认的 Q3 A 保留每 rank 已有的本地 Host view，由 manager 统一提供，不再用 TP0 绝对地址覆盖 Mooncake 的本地 view。最终改动主体是 MooncakeConnectorV1，SFA connector 和旧分支只作为参考。

### 已确认的目标

- 保留同一份 TP 共享 Host KV Pool，保留所有 Decode TP rank 的共享映射和访问能力。
- TP0 继续作为 allocation owner，但不再独占 Main D2RH 传输。
- 所有 Decode TP rank 都具备本地 Transfer Engine registration 和 Main D2RH 能力，使用各自的 Transfer Engine 并行拉取到共享 Pool。
- 各 rank 写入互不重叠的 Main 数据，合起来完整覆盖请求，避免重复传输、重复写入整份 Host KV。
- 任务量足够时应利用全部 TP rank；小请求允许部分 rank 没有 Main 工作，不强制制造空 Transfer Engine 调用，但仍保留必要的逻辑空任务和完成通知。
- 并行策略、模型适配和 SFA 来源路由对齐普通 MooncakeConnectorV1 路径；共享 Host writer 分工是额外的目标策略，不另建一套拓扑路由。
- 并行兼容目标覆盖普通 MooncakeConnectorV1 的全部合法配置，不只覆盖某个部署组合；模型/cache dtype 适配同样沿用普通路径。
- 性能动机是大 KV 传输可能受限于单个 NPU 的传输硬件能力，目标是实际使用多个 NPU 的传输资源；目前没有测量结果或指定加速比。
- 当前验收硬件为 Ascend A3（910C）。用户不需要预先指定 cache dtype，由有效配置、cache spec 和实际 tensor 确认。
- 所有相关 rank 的接收任务全部成功后才允许 decode；失败、取消和目标 block 的 delayed-free 对齐普通非 DSA 路径的任务完成语义，底层超时限制见第 7 节。

上述目标及下表中的 Q1–Q8 选项均已确认，具体接口与兼容性工作包按此细化；本轮用户已批准按此基线编码。

### 本轮设计决策（Q1–Q8）

此处 Q 编号对应代码审查后的八项设计问题，不是此前关于性能、硬件和兼容范围的问答。

| 问题 | 状态 | 已确认内容 |
| --- | --- | --- |
| Q1：传输失败恢复 | 已确认 A | 对齐 vLLM 请求失败机制，接入 `invalid_block_ids` 与 `kv_load_failure_policy="fail"`；不以 Decode 本地重算兜底 |
| Q2：Transfer Engine 超时与排空 | 已确认：对齐普通路径 | 沿用非 DSA 的同步 Transfer Engine 返回、任务完成及释放语义；不改 Mooncake，不新增独立排空或资源隔离机制，不再作为设计阻塞项 |
| Q3：本地 Host 地址接口 | 已确认 A | 保留每 rank 已有的本地 tensor/view，由 manager 统一提供给 connector/fused；增加初始化布局校验，不把 view 构造职责迁入 Pool |
| Q4：Prefill 完成计数 | 已确认 A，按 R4 澄清复用边界 | 复用普通来源规划、逻辑任务通知边界、P 端计数与 worker 聚合；通知拓扑不变时可复用计数值；固定参与者与空通知覆盖 Main/Indexer，身份去重保留为轻量增强 |
| Q5：请求取消 | 已确认 A | 逻辑取消；跳过尚未开始的 Transfer Engine 操作，不删除逻辑任务及其通知义务，保留状态直到接收任务收尾，完成标准遵循 Q2 |
| Q6：公共 CP/DCP 规划复用 | 已确认 A | 最小公共提取加 DSA adapter，不复制独立 split，不一次性重构整个传输框架 |
| Q7：完整兼容交付 | 已确认 B | 先列完整兼容矩阵，再按能力差距分阶段实施；完整对齐的最终验收目标不变 |
| Q8：Main writer 分工 | 已确认 B | 按完整请求的全局逻辑 Host block ordinal 轮转，`owner = g % D_TP` |

Q2 已关闭，不再要求先解决底层超时排空问题。Q1/Q4/Q5 按普通路径的同步调用及软件任务完成语义设计；这不代表超时返回后底层内存访问必然停止。该公共路径的已知限制仅作记录，不纳入本次修复范围。自动远端恢复、Decode 本地重算、真正的 Transfer Engine 取消和自适应 writer 策略不因本轮确认而加入实施范围。

### 复审结论（R1–R4）

R 编号对应总体设计复审，不替代上面的 Q 编号。本轮不再保留需要用户选择的新核心方案。

| 问题 | 收口结论 |
| --- | --- |
| R1：upstream 多 cache group 失败处理 | 保持原样，不修 upstream、不加兼容补丁；记录已知限制，不以修复或升级为实施前置条件 |
| R2：无数据请求的通知义务 | 复用普通空接收任务与现有队列；无数据则跳过 Transfer Engine，但保留完成计数和 P 端通知，补齐 DSA scheduler/metadata 入口 |
| R3：rank 内多任务完成 | 复用现有最后提交标记与 pending 计数；不要求最后提交的任务最后完成，不新增独立 seal API 或任务状态机 |
| R4：多消费者与 P 端释放 | 普通 CP/DCP 路径已经处理多消费者；复用其规划、计数、tracker 和 worker 聚合，仅在通知拓扑改变时调整计数；身份去重不是并行新增的必然要求，但保留此前 Q4 已确认的轻量增强 |

本次收口不重设计 Host Pool 物理布局，不新增 D 端通知汇总者，不修改 Mooncake 库。完整并行配置与 offload/kernel 能力差距仍按 Q7 分阶段补详细设计和验证，不能据此缩小兼容目标。

本设计明确区分三种并行性：

1. 不同 Decode TP rank 分担 Main 写入：本次核心目标。
2. 一个 Decode rank 从多个 Prefill DCP endpoint 读取：正确覆盖远端分片所必需。
3. 一个 rank 内不同 endpoint 并发：可利用现有 executor，不要求另建线程池。

当前 DSA 基线对 P/D TP 的限制为 `P_TP >= D_TP` 且 `P_TP % D_TP == 0`。不再默认保留这一额外限制：应逐项核对普通路径的合法范围，消除 DSA adapter 独有的限制。Prefill PCP/PP、P/D block size 及 TP 比例均纳入兼容性盘点。底层 offload 不支持的配置必须明确列为能力差距，不能宣称已经对齐或通过简单拒绝将其排除出目标。

经代码复核，当前非压缩 SFA 的平台校验还要求开启 DCP 时 `P_DCP == P_TP`。因此 `P_TP=8/P_DCP=2` 不能作为当前 SFA 的合法运行配置；此类组合只能用于通用规划算法测试。本文针对 Prefill offload 关闭的 SFA 路径，不能直接推广到带 `compress_ratios` 的模型。

## 2. 当前结构与复用边界

配置名 `MooncakeConnectorV1` 注册到 `mooncake_connector.py` 中的 `MooncakeConnector` 类。

```text
MooncakeConnector：vLLM 接口与角色选择
├─ Scheduler 进程
│  ├─ MooncakeConnectorScheduler：普通路径及 DSA Prefill
│  └─ _MooncakeDsaDecodeScheduler：DSA Decode
└─ Worker 进程
   └─ MooncakeConnectorWorker：注册、路由规划、任务提交
      ├─ KVCacheSendingThread：P 端 handshake / done 消息服务
      └─ KVCacheRecvingThread：D 端 pull 执行
         ├─ 普通 D2D handler
         └─ DSA handler
```

普通路径：

```text
ReqMeta
  → _get_sfa_replicate_k_block_ids()
  → _get_kv_split_metadata()
  → _get_group_pulls_metadata()
  → add_request()
  → _transfer_kv_cache_all_groups()
  → batch_transfer_sync_read()
  → 本 rank 多任务聚合、必要的 reformat、完成上报
```

DSA 当前路径：

```text
DsaStepRequest
  → _dispatch_dsa_commands()
  → 一个 Prefill leader endpoint
  → add_dsa_request()
  → Indexer D2D
  → 仅 Host Pool owner 执行 Main D2RH
  → DsaLocalResult
  → scheduler 等待全部 Decode TP rank
```

普通路径与 DSA 共用 worker、Transfer Engine、handshake 缓存、socket、接收队列和 executor。当前 DSA 在 `start_load_kv()` 提前分支并返回，未复用普通 CP/DCP 拆分；字节地址生成和完成处理也分别实现。

接收线程已有按 peer 排队机制：同一 `(host, handshake_port)` 串行，不同 peer 可并发，executor 上限为 32 个线程。Main single-writer 是 rank ownership 限制，不是 executor 单线程限制。

另一个重要区别：vLLM cache group 与 Mooncake transfer group 并非一一对应。一个 cache group 可按物理 spec 拆成多个 transfer group；所有适配代码必须保持各自的 ID 空间。

## 3. 总体方案

把传输规划分成有明确输入输出的三个阶段：

```text
请求拓扑、component groups、prefix、源/目标 block IDs
  → 来源规划：哪个 Prefill shard/replica 提供哪些 block
  → writer 分配：哪个 Decode TP rank 写哪些 Main block
  → 地址生成：该 rank 的本地 Host 地址、远端地址、长度
```

来源规划复用普通 Mooncake 的 CP/DCP 逻辑。writer 分配属于共享 Host 接收语义。地址生成保留 Host 与 Device 目标的差异，复用底层读取和连续区间处理。

现有 `_get_kv_split_metadata()` 混合了来源映射、本地设备分布和 endpoint 选择，不能把每个 rank 的普通 D2D 输出直接作为共享 Host 写入计划，否则可能重复写入。Q6 已确认采用最小公共提取加 DSA adapter：共享来源 shard/replica、cache spec、transfer group 和 layer 映射语义，保留普通 D2D 的设备目标分布，由 DSA adapter 提供 Host 目标几何及第 5 节的轮转 ownership。

公共接口必须明确 cache-group ID、transfer-group ID、layer/component 身份、全局 token/page 位置、物理 page ID、manager block ID、子页 offset 和有效长度。已有 helper 中的 group-ID 假设必须先核对，不能原样套用到 offload 分组。缺失必需 component 应在提交前报错；packing ratio 按 token/page 几何确定，再校验 dtype、字节长度和 stride，不能仅由字节长度之比推断。

禁止通过临时修改共享 worker 的 `tp_rank`、group metadata 或其他可被并发任务读取的状态来模拟另一个 writer。若需要规划视图，应采用显式参数或独立、不可变的规划输入。

## 4. Host Pool：创建者与 writer 分离

继续使用一个 TP 组共享的连续 Host region，以及现有每层 K/Rope tensor 布局。

| 责任 | 目标归属 |
| --- | --- |
| 创建共享 allocation | 保留 TP0 / pool owner |
| 本地映射与 Transfer Engine registration | 所有 Decode TP rank |
| Main D2RH | 按请求 block ownership 分配的 writer |
| 现有 decode current-token writeback | 保持原有归属 |
| Host→Device | 现有 attention fused offload 路径 |

每个 writer 使用本进程映射的 Host tensor 地址注册到自己的 Transfer Engine。不能从“fused kernel 能访问 TP0 广播的 GVA”推导出“任意 Transfer Engine context 都能使用该地址”。

当前代码中，所有 rank 参与 `create_shared_segment()` 并取得映射，但 connector 仅在 `pool.is_owner` 时注册 Transfer Engine 并生成 Main layout。目标是保留前者、扩展后者到所有 rank。shared-segment 的 `host_register`（NPU 可访问性）与 `engine.register_memory()`（Transfer Engine 注册）是不同步骤。

本地 Host layout 接口需要表达 pool 大小、每层 component offset、shape、dtype、block stride 和容量。Q3 已确认 A：这些信息从每 rank 已有的本地 tensor/view 及 Pool 映射派生；布局描述用于校验和传输，不成为另一套 view 构造来源。

当前 manager 在 `register_kv_caches()` 中仅保留 TP0 的 Host tensors，随后广播 TP0 地址并为非 owner 恢复 tensor。对允许不同 VA 的 Mooncake 映射，这不能作为正确的本地地址来源。按 Q3 A 修正后，Transfer Engine、fused 输入及其他 Host 地址使用者都使用各自进程有效的映射；保留 kernel 算法不等于保留该地址广播。vLLM-Ascend 中 Mooncake Host backend 的适配还应与现有 MemFabric 地址语义隔离，不涉及修改 Mooncake 库。

registration/unregistration 按本 rank 管理；关闭时停止新任务提交，等待本 rank 已提交的接收任务收尾，再注销注册。最后释放共享 allocation 的时机需遵循 shared-segment 生命周期；不另加 Q2 范围外的底层排空机制。多进程注册同一物理 region 的支持情况必须在目标 Mooncake/NPU 环境验证。

多个进程成功注册并不证明已利用多个 NPU 的传输硬件。需核实每个 rank 的 device/context、注册 location 和实际 Transfer Engine 通路，确认没有全部回落到 owner 的同一通路。

### Q3：本地 Host view 与地址接口（已确认 A）

沿用现有分配、reshape 流程：`MooncakeHostPool.allocate_tensors()` 从本进程共享映射中切出 byte views，现有 cache spec 适配负责 dtype/shape，manager 保存各层的本地 Host tensors。保留的是同一物理 Pool 的本地 view，不是为每 rank 复制完整 KV。

```text
本 rank 的共享 Pool 映射
  → 现有分配 / reshape
  → manager 保存本地 Host views
      ├─ connector：派生 Main 目标地址、block stride 和长度
      ├─ fused attention：获取完整 Host KV 输入
      └─ 其他 Host 地址使用者：派生本地指针列表
```

接口与改造约束：

- 每个 rank 的 manager 在 Mooncake backend 下保存本进程的原始 Host tensors，保留其 dtype、shape、stride、device 属性及有效引用；不再用 TP0 指针通过 `from_blob` 恢复替代 view。现有恢复算子只包装指针，并不建立共享映射或翻译其他进程的地址。
- 提供与 fused 算法解耦的统一访问接口，例如拟议的 `get_local_host_kv_views(layer_name)`；connector 和现有 fused getter 均取自同一组 view。名称可在实现时细化，不另起一套 tensor 构造逻辑。
- Main 字节布局从该接口返回的 tensor 派生；`gvas_k_bases` / `gvas_v_bases` 等本地指针列表也保持同源，不能只修正 connector 而让其他消费者继续使用 owner 地址。
- 仅调整 Mooncake backend 的地址路径，保留 MemFabric 原有语义；TP0 allocation ownership 和 current-token writeback 归属不因 Q3 改变。P/D handshake 中必要的远端源地址交换不受影响。
- 初始化时按共享 Pool 域校验 layer/component 身份、相对 offset、shape、dtype、block stride、容量和地址范围；一致性检查应包含原始 segment 到 Pool 起点的对齐偏移，不能只比较相对各自已对齐 base 的 offset。允许各 rank 绝对 VA 不同，但同一 component 必须对应共享 segment 内相同字节区间。
- 保持现有连续布局和零拷贝 view；校验 fused 输入的 flatten/`contiguous()` 不会意外复制整份 Host KV。布局校验在初始化完成，不给每次请求增加跨 rank 布局同步。

不采用 Q3 B 的 Pool descriptor 驱动 view 构造方案。Pool 继续负责共享内存、分配和本地注册生命周期；模型/cache 语义沿用现有适配流程。允许增加轻量、派生的布局描述，但不据此迁移全部 view 构造职责。

## 5. Main block ownership

Q8 已确认选择 B：先建立完整、正确的源目标映射，再按全局逻辑 Host block 轮转分配 writer。设 `g` 是完整请求中从零开始、按目标 Main Host block 几何定义的逻辑 ordinal，Decode TP 数为 T：

```text
owner(g) = g % T
rank r receives external Host blocks satisfying g % T == r
```

`g` 不是 allocator 的物理 block ID、远端 shard-local ordinal，也不是过滤 prefix 后重新编号的位置。prefix block 不进入外部写入集合，但剩余 block 保留完整请求中的 `g`。例如 T=4、首个待接收 block 的 `g=5`，其 writer 是 rank 1，而不是 rank 0。

每个 rank 从所有相关 DCP endpoint 任务中过滤出自己负责的目标 block。不同 rank 只有在确有等价 Prefill replica 时才能选择不同来源；必须保持原始 DCP shard 身份，不能把 replica 选择变成 shard 替换。

一个 Main Host block 的全部层、K/Rope component 和远端子块由同一个 writer 负责。连续的 T 个待传 Host blocks 能覆盖全部 T 个 writer；N < T 时允许空 Main 分片，该 rank 仍可能需要 Indexer 读取以及固定参与者协议通知。非连续的待传逻辑集合可能出现负载偏斜，需纳入验证，不能仅由 block 总数推断均衡，也不自动切换到未确认的策略。

在 P-TP=P-DCP=8、D-TP=4、P/D Main block size 相同的合法 SFA 示例中，来源 shard 为 `g % 8`，writer 为 `g % 4`：D0 对应 P0/P4，D1 对应 P1/P5，D2 对应 P2/P6，D3 对应 P3/P7。该关系只描述 Main 来源，不能用于缩减各 rank 的 Indexer 内容或直接充当 Q4 的完整通知计划。

连续区间分工、完整 source-shard 分工和自适应来源亲和策略不再作为当前默认方案；后续更改需要单独讨论。

必要不变量：

- writer 目标集合两两不相交，其并集等于需要接收的 Main block 集合。
- 每个目标 component 有且只有一个正确来源，不因 transfer-group 拆分重复写入。
- prefix 已有 block 不进入外部 Main 写入集合。
- 尾部不满块只需保证有效 token 覆盖；padding 不得被当作有效 KV。
- 普通路径允许的 P/D block-size 差异属于本次完整对齐目标。一个 Host block 对应多个远端子块时，先按 Host block 的 `g` 定义 ownership，再由所有子块继承，不能分别分配造成跨 writer 写同一 block。

## 6. Prefill DCP 与 Indexer

参考旧分支 `mooncake-dsa-asymmetric-dcp` 的两个提交：

- `bcafaefad087cd3b7a28d0408a26ca9c7e0c7722`：适配 DSA 请求到普通 `ReqMeta`，复用 CP split、group-pull 和 endpoint 任务聚合。
- `5e7e286f9ad16341fcacad86235e14da45800ed7`：区分 replicated Indexer 的远端 physical page ID、Decode manager block ID 和请求内 page 位置。

复用其设计，不整批搬入旧实现。旧分支仍为 Main single-writer，且普通 Mooncake 在新基线上已有 transfer-group 等变化。

DSA metadata 需要保留足够的规划信息：远端 TP/PCP/DCP、Main block size、prompt block 数、prefix/external token 数、完整和待接收的本地 block IDs，以及远端 Main/Indexer component group 身份。不能假设 P/D cache group 顺序相同。

对于 DCP-replicated Indexer，每个 Decode rank 保留自己所需的完整 Indexer 内容。它不随 Main writer 分片缩减。远端 page ID 与本地 manager ID 保持分离，handshake 得到 packing ratio 后再转换：

```text
manager ordinal, page slot = divmod(full-request page position, packing ratio)
destination = local manager base + manager offset + page-slot offset
```

其中 manager offset 和 page-slot offset 必须按实际 tensor/handshake 语义计算，不能混用 byte length 和 tensor stride。prefix 场景仍使用完整请求中的 page 位置，不能把剩余 pages 从零重新编号。

## 7. 接收执行、完成与取消

一个 request 在一个 rank 上可拆成多个 endpoint pull。使用逻辑 endpoint task 携带各 component 的源目标 IDs、Indexer page 位置及 P 端释放通知信息，提交到现有 peer queues；这不要求另建任务执行框架。同一任务内的 Main/Indexer 或多次 Transfer Engine 调用共享一个通知边界，不能各自增加 P 端 done 计数。

D 端接收完成分两级；P 端通知及 worker 聚合另见 Q4：

```text
本 rank 全部 endpoint 接收任务完成/退出（沿用同步 Transfer Engine 返回语义）
  → 汇总该 rank 成功/失败
  → 全部 Decode TP rank 结果到齐
  → 成功：允许 request 进入 decode
  → 失败：按 vLLM fail 策略结束请求，不触发 Decode 本地重算
```

rank 内维护 pending task 数和失败状态，不允许第一个 endpoint 完成就上报整个 rank 成功。失败可提前通过 invalid block 上报，但 `finished_recving` 应等待规定的任务/rank 完成集合；它表示接收生命周期结束，不单独代表 KV 有效。复用 scheduler 已有的 rank-aware 聚合，不沿用当前 DSA 的本地重算恢复分支。R1 的 upstream 失败处理限制不因本设计而得到修复。

### R2：复用普通空接收任务

普通 scheduler 的 `update_state_after_alloc()` 在没有 external tokens 时仍可产生空接收请求；请求在分配前被拒绝时，`request_finished()` 也会补入空接收请求。worker 仍通过 `add_request()` 和现有 peer queues 执行：`_transfer_kv_cache_all_groups()` 在普通 blocks 与 replicated Indexer blocks 均为空时跳过 Transfer Engine，`_handle_request()` 的收尾仍完成本地计数并发送 P 端 done。

DSA 按相同机制补齐入口和语义：

- Main writer 过滤只改变数据范围，不删除已规划的通知义务；Main 为空但 Indexer 非空时仍执行 Indexer D2D。
- Main/Indexer 都为空时，保留逻辑任务用于完成计数与通知，不调用 Transfer Engine。整请求无数据或分配前被拒绝时也必须能传递相应通知元数据，不能依赖非空 `DsaStepRequest` 才能通知 P。
- 通知元数据与数据任务共用现有队列和收尾路径，不要求独立 `NotificationOnly` 执行通道。没有本地接收生命周期的拒绝请求不能被伪造为新接收请求并产生无效的 `finished_recving`。
- 空 block IDs 保留 cache-group 维度；普通零 external token 分支目前可能生成 `[]`，而无 CP 展开会按 group 下标访问，存在越界风险。复用空任务机制不等于原样复制该表示，也不代表所有普通空请求路径已端到端验证。

普通路径已有 `test_handle_empty_transfer_sends_done` 和 `test_request_finished_rejected_remote_prefill_enqueues_empty_recv`，可作为 DSA 测试的行为参照。

### R3：复用最后提交标记与 pending 计数

普通接收线程已有 `request_task_counts` 和 `finished_request_markers`。`_mark_request_task_submitted()` 先增加 pending；遇到 `all_task_done=True` 的最后一个规划任务时，在提交阶段记录 marker。`_mark_request_task_done()` 只有在 pending 归零且 marker 已存在时才判定本 rank 完成。

```text
本请求最后一个逻辑任务已提交 AND 本 rank pending == 0
  → 汇总一次本 rank 成功/失败结果
```

`all_task_done` 不是“这个任务最后执行完成”的保证。例如 A、B 依次提交，B 带最后标记但先完成，仍需等 A 完成才上报。乱序本身不是普通路径的缺陷；风险来自新适配在过滤、取消时丢弃任务，尤其是最后提交标记。

DSA 当前在 `_submit_request()` 中绕过普通计数，并在单 endpoint 回调后产生 rank 结果。改造时接入同一计数机制：先确定逻辑任务集合，过滤后保留空任务，逻辑取消时跳过未启动 Transfer Engine 但保留任务收尾；计数完成后只产生一次 `DsaLocalResult`，再交给现有 DSA scheduler 聚合，不能同时从两条路径提前上报整个请求完成。

保留上述提交约束即可，不新增独立 seal API、增量任务规划框架或另一套状态机。若以后引入动态追加任务或重试计划，应另行评估，不加入本次范围。

### Q1：请求失败策略（已确认 A）

接入本地 vLLM 已有的 `invalid_block_ids` 和 `kv_load_failure_policy="fail"`，不再把清零 `num_computed_tokens` 当作失败恢复。必须核对 invalid block 的 group-ID 空间及上报时序，使失败请求可被 scheduler 正确识别；配置 fail 本身不能替代 connector 的失败上报。

R1 已确认保持 upstream 原样：当前 vLLM scheduler 的 invalid-block 处理仍有单 cache group 假设，多 group 请求可能在 connector 结果回调前触发异常。本分支不修复该 upstream 路径、不添加 Ascend 兼容补丁，也不以等待 upstream 合入或升级作为实施前置条件。Q1 仍要求正确接入 connector 失败上报并去掉本地重算；不能宣称由此已实现所有多 group 场景的端到端 fail 闭环。验收中应单独记录这一继承限制，不通过缩减测试或错误重算掩盖它。

当前 Mooncake Host backend 禁止保留完整设备 Main cache，不能将 Decode 本地重算作为兜底。自动重试拉取、重新远端 prefill 和本地重算都不是本轮选定方案。逻辑失败与资源释放应分开：即使提前向上层报告失败，也必须保留 delayed-free 状态直到相关接收任务收尾，完成标准沿用 Q2，不额外要求独立的 DMA 停止证明。

### Q2：传输结果与排空（已确认：对齐普通非 DSA 路径）

沿用普通 MooncakeConnectorV1 的现有行为，不将底层超时治理作为 multi-TP D2RH 的前置条件：

- 继续使用 `batch_transfer_sync_read()`；负返回值或异常按失败处理，通过 invalid block / fail 策略上报。
- 已启动的同步调用返回并完成任务收尾后，参与本 rank 的完成聚合；区分失败上报与接收生命周期完成，保留 scheduler 的 delayed-free 机制。
- endpoint handler 收尾后，无论成功或失败都按通知计划发送 P 端 done；P 端计数释放、worker 聚合和超时兜底沿用普通路径。通知拓扑的复用或调整以及轻量身份去重按 Q4/R4 适配，不将多消费者等待描述为 DSA 独有的新问题。
- 不修改 Mooncake 库，不新增显式 transfer handle、底层取消、独立 drain 或资源隔离机制；也不以等待社区 PR 合入或升级版本获得额外排空保证作为实施门槛。

已知限制：同步 Transfer Engine 超时返回不等于 DMA/RDMA 已停止，普通路径并没有完整的物理排空保证。本设计继承这一限制，不宣称对齐后已解决底层 late write 问题，也不再为此保留待决策项。

### Q5：请求取消（已确认 A）

采用逻辑取消并等待接收任务收尾：跳过尚未开始的 Transfer Engine 操作，但不从逻辑任务集合中删除其计数、最后提交标记和通知义务；已启动的同步调用继续执行直到返回，本轮不依赖真正的底层取消能力。保留取消及 pending-task 记录，不能只删除 worker 的 active-command 而丢弃晚回调。未提交、排队中、执行中、完成但尚未上报四种状态都必须最终产生唯一终态，并按完成协议解除 scheduler 的 delayed-free。

相关接收任务收尾前，不允许因取消而提前释放或复用这些 block。正常完成、失败和取消共享同一套任务终态管理；这里的等待遵循 Q2 的软件完成语义，不新增底层物理排空保证。如后续增加重试，还需区分 attempt/generation，防止旧回调匹配新一轮传输。

### Q4：P 端完成通知（已确认 A）

R4 的结论是复用普通路径的来源规划、逻辑任务通知边界、P 端计数、tracker 和 worker 完成聚合，不因 D2RH 重建完成协议。以下区分当前普通行为与本设计的扩展。

普通路径已经处理的条件：

- 无 CP 的常规 MLA/SFA 路由主要通过 `_get_remote_ranks_for_req()` 分配不同来源副本：相同 P/D TP 对应 rank，不同 TP 的副本选择使用请求 ID 和不放回采样。全部 D TP 并行不等于全部读取同一 P endpoint；单消费者分支收到一次 done 即可报告本 worker 完成。hybrid 还有 group-specific 来源约束，不能将常规路由推论当作所有模型组合已验证的证明。
- CP/DCP 路径允许多个 D rank 读取同一 P endpoint。`get_remote_port_send_num()` 遍历全体 D 的来源映射，统计每个端口的逻辑任务数；它不是 Transfer Engine 调用次数，也不是固定的 `D_TP`。P 端 `port_send_num` 累计收到的 done，达到阈值后调用 tracker。
- 普通 SFA 将 replicated Indexer 附加到已有 endpoint task，任务内所有 component 收尾后只发一次 done。Indexer 不自动增加任务或通知数量，Main 为空也不能跳过仍需执行的 Indexer。
- `KVCacheTaskTracker.update_done_task_count()` 实际负责标记 worker 请求完成，不承担通知累加。之后 `KVOutputAggregator` 等待预期数量的 P worker 完成，再由 scheduler 释放 blocks；不能把一个 P worker 的完成等同于整请求立即释放。

```text
P 请求跟踪 / 保留源 KV
  → 本 endpoint 预期通知到齐（或沿用现有超时兜底）
  → KVCacheTaskTracker 标记本 worker 完成
  → get_finished → KVOutputAggregator 等待预期 worker 完成数
  → scheduler 释放源 KV
```

DSA 的计数复用条件与通知约束：

- 将 Main 目标从 device 改成 Host，或只过滤 Main writer 的数据范围，本身不改变通知拓扑。若来源与逻辑通知任务集合不变，且空任务仍通知，可以直接复用普通路径的计数值；不再保留“不能直接复用现有拓扑计数”的绝对表述。
- 增加来源 endpoint、改变消费者集合，或拆分、合并逻辑通知任务时，必须使预期计数与最终计划一致。不能把 TP0 single-writer 的简化通知原样用于任意 multi-TP/DCP，也不能统一将阈值改成 `D_TP`。
- 固定逻辑通知参与者必须覆盖 Main 与 Indexer。每个参与者对同一 endpoint 的全部 Main/Indexer 子任务收尾后只贡献一次完成；writer 过滤后无实际读取的参与者也保留空完成通知。若合并了普通路径中同一参与者到同一 endpoint 的多个通知任务，计数单位已经改变，必须相应调整阈值，不能沿用旧任务数。
- 普通路径存在两类空通知：有通知义务但无实际数据的参与者仍计入预期数；完全没有 reader 的 P endpoint 则沿用指定 D worker 对 `num == 0` 端口发送清理通知的机制。后者不能被误当成还需等待的普通 reader。
- 源端保留策略必须与来源计划一致，当前 DSA 无 CP 的固定 leader 保留规则需同步适配。P 端释放只等待对 P 源 KV 的读取收尾，不等待随后只消费 D Host Pool 的 fused Host→Device。

身份去重保留为此前 Q4 已确认的轻量增强，不是多 TP 或共享 Host 新增的必然要求，也不是普通路径已经提供的保证。普通路径按消息次数计数，ZMQ ACK 不等于参与者幂等；tracker 的已完成请求保护不能阻止完成前的重复 done 提前累加到阈值。

沿现有 done 消息和 P 端状态增加稳定的参与者身份、固定的预期通知信息及已完成集合。重复通知只 ACK、不重复计数；已结束请求不因晚到通知重建计数状态，新增记录随正常结束或现有超时清理。身份在当前请求生命周期内稳定，跨节点/PP 时不能仅用裸 TP rank；集合固定后不能增加未纳入协议的读取。不新增 D 端通知汇总者、独立终态服务或 Mooncake 库接口，也不扩展为自动重试/跨重启恢复协议。P 端超时兜底语义不重设计。

## 8. 地址生成与性能

Main 使用本地 Host layout 和远端 handshake 生成字节地址。Indexer 使用自己的 device layout/page mapping。普通 D2D 的设备 reformat 不应自动施加到 Main Host 数据上。

### 与普通 D2D 地址分层对齐

按职责区分内存注册、tensor 布局和请求字节地址：注册层允许一个 Pool region 覆盖多个 tensor；布局层仍保留各 layer/component view 的独立 base、block length、stride 和 scale；请求层再结合来源 block 映射、writer 分工及子页 offset 生成 Transfer Engine 地址列表。Q3 A 接入的是本地 view/layout 来源，不向 CP/DCP 规划中混入 Host 裸指针。

代码中 `group_pulls_list[shard_idx][remote_port_idx][group_pull_idx]` 是分片、endpoint、transfer group 的三层任务列表，不是三级地址表。`GroupPull` 的 `num_group_pulls` 描述组装该 group 所需的 TP pulls，不是 Main writer 数；manager group、transfer group、注册 region 也不要求一一对应。

Main 的一般目标地址为 `local_component_base + host_physical_block_id * host_block_stride + intra_block_byte_offset`。完整请求的逻辑 ordinal `g` 仅用于 Q8 writer 选择，不替代物理 block ID。P/D block/page 粒度不一致时，由 adapter 明确映射后再计算地址，不能只替换普通 D2D 的 base 而沿用不匹配的 scale 或设备拼接 offset。Indexer 继续使用设备目标布局及完整复制语义。

最终复用 `batch_transfer_sync_read(session_id, local_addresses, remote_addresses, lengths)`；普通代码的 `src_list` 实际是本地接收地址，`dst_list` 是远端读取地址，不能按变量名反向接入。公共提取保持 Q6 的最小范围，不要求新增完整传输框架。

### 性能验证

仅在源与目标都连续时合并传输段，不能跨越 padding、不同 endpoint 或不连续 stride。记录每 rank 的 bytes、任务数、合并前后 entry 数和耗时，以判断瓶颈。

已选的逻辑 block 轮转可在部分 TP/DCP 比例下减少 Main endpoint 扇出，但 Host 目标地址通常更分散；不同 block-size 比例也会改变来源关系。因此不能预先承诺 coalescing 比例或 TP 扩展的线性收益。共享 DRAM、NUMA、NIC/NPU 通路和注册成本都需要实测。

当前瓶颈是假设，不是测量结论。首先使用大 KV 单请求比较单 NPU writer 与多个 NPU writer，观察实际传输资源参与情况、有效带宽和总耗时；再进行并发请求测试检查总吞吐与回归。不能仅凭线程数或 Transfer Engine 实例数增加就宣称完成性能目标。

## 9. 预计改动边界

| 组件 | 预计职责变化 |
| --- | --- |
| `MooncakeConnector` | 保持接口转发；补充必要的配置/拓扑校验 |
| `_MooncakeDsaDecodeScheduler` | 完整 metadata、空接收/拒绝请求通知入口、prefix 范围、fail 上报与 delayed-free 生命周期；不修复 R1 upstream 缺陷 |
| `MooncakeConnectorScheduler` | 如必要，输出明确的远端 component group 身份 |
| `MooncakeConnectorWorker` | 最小公共 CP 规划提取、DSA adapter、逻辑 block 轮转与 endpoint 聚合 |
| `KVCacheRecvingThread` | DSA endpoint 执行与地址生成；复用空任务、最后提交标记、pending 计数和统一收尾；保持 Main/Indexer 通知边界 |
| `KVCacheSendingThread` / 源端保留逻辑 | 复用普通计数与 tracker，仅按通知拓扑变化调整阈值；轻量身份去重、与来源计划一致的保留；沿用 worker 聚合与释放语义 |
| `mooncake_dsa_metadata.py` | 请求级拓扑、block/page 身份及无数据请求的通知元数据 |
| `MooncakeHostPool` | 明确本 rank writer registration 及清理 |
| `SparseKVOffloadManager` | 按 Q3 A 保存每 rank 已有 Host views，统一消费接口、派生本地指针并校验布局；保留 fused 算法 |
| 测试 | 覆盖跨 TP/DCP、地址、完成、失败和取消 |

是否单独增加纯规划模块属于实现组织问题，应由公共代码提取范围和可测试性决定，不作为用户待决策事项；不需要重新建立 connector 或传输执行框架。

## 10. 实施顺序与验收

Q7 已确认 B：先列完整兼容矩阵，再按能力差距分阶段实施。矩阵至少覆盖模型/backend、runner、P/D TP/CP/PP、block size、Main/Indexer dtype 和单机/跨机共享域；逐项标记公共不合法、已有支持、connector 缺口或 offload/kernel 缺口。阶段性可用不等于完整目标已经达成。

1. 根据已确认的 Q3 A 细化本地 view 接口与布局校验，列出上述兼容矩阵，按 Q2 对齐公共失败/完成语义；据此补齐改造边界和阶段验收，不自动执行全分支 rebase。
2. 最小提取普通 CP/group-ID 来源语义，接入 DSA adapter、replicated Indexer 与 P/D block-size 映射；single-writer 仅可作为内部来源正确性对照，不是交付目标。
3. 接入各 rank 本地 Host registration 和 Q8 逻辑 block 轮转；同时完成 Q1 fail 上报、R2 空任务、R3 现有计数接入、Q4/R4 通知复用与轻量去重、Q5 取消任务收尾。connector 生命周期适配不能留到功能交付后，R1 upstream 修复仍不在范围内。
4. 按矩阵分别完成 Decode CP/PP、runner、cache 格式等尚有缺口的工作包；具体接口/模型执行方案需在对应阶段先讨论，不因分阶段实施而取消这些目标。
5. 每阶段执行对应 CPU UT、集成测试和所需 NPU 验证；最终以完整兼容矩阵和多 NPU 性能结果验收。

至少覆盖：

- 当前 SFA 合法的 P-TP/P-DCP/D-TP：`8/8/4`、`16/16/8`，以及 DCP=1 回归。`8/2/4`、`16/4/8` 可作为通用规划算法测试，但当前 SFA 配置校验应拒绝它们。
- 无 prefix、部分 prefix、非连续物理 block IDs、尾块、N < T、空 endpoint 分片；prefix 过滤后保留全局逻辑 ordinal、轮转覆盖及负载分布。
- 整请求无外部数据、分配前拒绝、Main 为空但 Indexer 非空、两者均为空；保留 group 维度与通知义务，不产生无效的接收完成上报。
- Main 全层/component 恰好一次覆盖，各 Decode rank Indexer page 覆盖正确。
- P/D block size 不同、一个 Host block 跨多个远端子块/shard，所有子块继承同一 writer。
- 最后提交的任务先完成、提交间隙 pending 暂时归零但最后标记未到、失败/逻辑取消后保留最后任务与收尾；每 rank 只上报一次结果。
- 某 endpoint 失败时 connector 正确上报 invalid blocks 且不触发本地重算、重复结果、取消时同步调用仍在执行及 delayed-free 收尾；晚回调不能因提前删除状态而丢失。多 group 端到端失败若受 R1 upstream 缺陷影响，明确记录继承限制，不将其标记为已修复或通过。
- 通知拓扑不变时计数与普通路径一致、拓扑改变时阈值同步调整、Main/Indexer 共用通知边界、固定参与者空通知、零 reader 端口清理、重复通知去重，以及 P 端保留/worker 聚合/超时清理的一致性；不将底层超时后的物理排空作为新增验收项。
- 每 rank 本地 registration、布局不一致、注册失败和 unregister 生命周期。
- Q3 A：不同 rank 使用不同本地 VA 时，connector/fused/本地指针列表仍指向相同共享内容；覆盖 component offset、起始对齐和越界校验，保留原始 device/dtype 属性，验证 fused flatten 不引入完整 KV 拷贝，并回归 MemFabric 地址路径。
- 普通 D2D 路径回归。

NPU 验收需比较 single-writer 和 multi-writer 的有效 KV 内容及 decode 输出，并测量总接收时间、每 rank bytes/耗时、有效带宽和 Host 内存用量。CPU 测试依赖已补齐；当前验证环境没有 NPU，不能把 CPU helper 测试视为硬件链路验证。

## 11. 已确认的验收方向与工程差距

此前关于兼容范围、性能动机、硬件与 dtype 的问题已得到回答，不再列为用户待决策问题；这些历史问题不沿用本轮 Q1–Q8 编号：

| 项目 | 已确认方向 |
| --- | --- |
| 并行兼容 | 对齐普通 MooncakeConnectorV1 的全部合法并行配置 |
| 性能目标 | 大 KV 使用多个 NPU 的传输硬件能力，避免单 NPU writer 限制；尚无测量 |
| 模型/cache dtype | 对齐普通 Mooncake 适配机制，实际 dtype 由代码和运行配置确认 |
| 验收硬件 | Ascend A3（910C） |

不将 single-writer 调试开关作为待决策需求：当前没有新增此选项的需求，旧基线可用于对照测试。多 writer 是最终验收目标。

### 必须盘点的能力差距

兼容目标不等于当前能力。需要区分模型/硬件/runner 的公共合法性限制、DSA connector 独有的限制，以及 Host offload/fused kernel 自身限制。

- 当前 offload 禁止 Decode CP>1、PP>1。普通 connector 在相关合法场景下若支持这些配置，满足完整对齐会涉及 offload 之外的模型执行与共享 Pool 分组设计；应先给出具体差距和改造范围，再讨论实施，不把它们自动移出目标。
- 当前 manager 要求 BF16 Main，拒绝 compressed 模型；普通路径的其他 cache 格式不能仅靠复用字节传输就让 fused kernel 正确消费。Indexer dtype、量化 scales、打包和 stride 需要按实际 spec/tensor 适配。
- `P_DCP == P_TP` 等当前 SFA 公共平台限制仍有效。“全部合法配置”不包含被相同模型/硬件平台拒绝的配置。
- Mooncake 版本/commit 和具体测试模型尚未记录，由后续环境盘点补齐；用户无需猜测 dtype。

### 工程验证项（由实现者调查，不要求用户决定正确性）

- 核实 Mooncake shared-segment 本地映射、每进程 Transfer Engine 注册与关闭语义，并在目标环境验证。
- 跟踪 vLLM scheduler delayed-free、失败与取消路径，验证接收任务收尾前不提前回收、不丢终态；R1 upstream 多 group 失败处理与 Q2 底层 Transfer Engine 超时后的 late write 风险均记录为继承限制，不加入本次修复范围。
- 核对公共 CP split 的 ID 空间、模型/spec 适配及 P 端释放计数，确定最小公共提取范围。
- 根据已选 Q8 逻辑 block 轮转策略验证覆盖、均衡、endpoint 扇出和多 NPU 性能；是否需要独立纯规划模块由最小公共提取范围决定，不重新开放已确认的 writer 选择。
- 生成普通路径与 DSA offload 的并行/模型/cache 格式兼容矩阵，逐项区分已有支持、adapter 差距和需要扩展底层 offload 的差距。

### 设计决策状态

Q1–Q8 与复审 R1–R4 已收口，核心 connector multi-TP 总体设计可以作为后续实施基线，不再保留需要用户拍板的新核心方案。Q3 采用第 4 节的 A 方案；Q4 保留固定逻辑通知参与者及轻量身份去重，R4 只澄清普通完成协议和计数值的复用条件，没有取消去重要求。

接口命名、最小公共提取、任务计数接入及正确性/性能测试属于后续实现与验证。Decode CP/PP、其他 cache 格式、runner 和 offload/kernel 能力差距则按 Q7 在对应阶段补详细设计；总体设计定稿不表示这些配置的具体方案或完整兼容目标已经完成，也不重新开放本轮已确认的基本选择。

此前设计定稿阶段未批准编码；本轮用户已明确批准开始实施。性能动机仍待 A3 实测，文档中的目标行为不能当作当前代码已实现或测试通过的证据。

## 12. 初始未提交草稿的历史说明

此前提前开始的工作树改动包括 `register_local_writer()`、纯 block 分片/地址合并 helper 和测试，以及尚未接入的 endpoint 类型、metadata 字段。它们不是已获确认的实现，也没有完成 multi-TP/DCP 主链路。旧 helper 的连续区间分片不是本轮已确认的 Q8 轮转策略，后续获准编码时需要调整，不能以其现有测试作为 Q8 已实现的证据。

用户随后明确授权重置除本文外的未提交改动；上述草稿已撤销，本轮重新实现进度见第 13 节。

## 13. 实施记录（2026-09-05）

### 工作树基线

分支和 HEAD 与文首一致。开始时三个 tracked 文件有草稿修改：
`mooncake_connector.py`、`mooncake_dsa_metadata.py`、`mooncake_host_pool.py`；
未跟踪文件为本文、`mooncake_dsa_transfer.py` 和对应 UT。
生产接收路径仍为固定 leader / owner 单 writer，草稿 endpoint 类型和 metadata 尚未接入。
初始要求保留已有修改；随后按用户明确授权重置代码/测试草稿并保留本文。未 rebase、未提交、未推送。

### Q7 兼容矩阵与阶段范围

以下矩阵记录实施前的能力差距；本轮已实现项以随后状态表为准。
维度组合还必须通过相同模型的公共平台校验。
“已有”仅指代码路径存在，不代表本轮 A3 验收通过。

| 维度 / 配置 | 普通路径 / 公共约束 | DSA 差距与改造范围 | 阶段 / 验证 |
| --- | --- | --- | --- |
| 非压缩 SFA + Mooncake + BF16 Main | 已有 SFA component 路由 | 核心目标，共享 Host adapter | 本轮基础；A3 待验证 |
| MemFabric Host backend | 已有自身地址语义 | Q3 不改变其 owner GVA 恢复路径 | CPU 回归 + 硬件回归 |
| 其他 MLA/GQA/hybrid、压缩模型 | 普通按 spec/group 适配；合法性依模型 | offload 要求 index_topk、拒绝 compress_ratios；涉及模型与 fused 消费扩展 | 先详细设计，不自动扩大 |
| runner V1 | 已有 offload 路径 | connector 接入与回归 | 核心阶段 |
| runner V2 | 公共支持范围依模型；PCP+DCP 同开受限 | ascend_config 明确拒绝 offload V2；涉及 runner buffer/attention 生命周期 | 后续先详细设计 |
| P-TP/P-DCP/D-TP = 8/8/4、16/16/8 | SFA 合法候选；P offload 关闭 | 固定 leader 无法完整读取 DCP；须复用来源规划与通知计数 | 核心必需；CPU + A3 |
| P-DCP=1、P-TP>=D-TP 且整除 | 当前 DSA 已有单 writer | 全 ordinal 轮转、每 rank registration | 核心必需；回归 |
| P-TP<D-TP 或其他 TP 比例 | 普通 worker 同样拒绝 P-TP<D-TP；其余依普通副本路由规则 | 已移除 DSA 额外整除限制并接入来源 adapter；保留公共限制 | CPU 路由回归；A3 待验证 |
| P-TP/P-DCP=8/2、16/4 | 当前非压缩 SFA 公共校验拒绝，要求 DCP=TP | 仅通用算法测试，不作为合法 SFA 部署 | CPU 规划测试 |
| Prefill PCP>1 / PP>1 | 普通已有 CP split / layer 范围路径，需按模型确认组合合法性 | metadata、endpoint host/port、层身份与通知参与者未完整表达 | connector 规划阶段逐项验证 |
| Decode PCP/DCP>1 | 普通存在 CP 几何约束 | offload 明确拒绝；需要 Host Pool 共享域、逻辑 block 与 attention 消费分布设计 | 后续先报告详细接口方案 |
| Decode PP>1 | 普通有 PP 层范围路径 | offload 明确拒绝；Pool 名称目前只有 DP，需 stage 隔离、层归属、完成聚合设计 | 后续先详细设计 |
| P/D block size 相同 | 普通支持 | Q8 使用完整请求 ordinal，prefix 不重编号 | 核心阶段 |
| P/D block size 不同 | 普通要求互为整倍数；Decode CP 还有比例约束 | Host block 与远端 kernel page 映射、子页偏移不能只按 byte ratio 推导 | adapter 阶段 |
| BF16 Main + BF16/C8 Indexer | manager Main 要求 BF16；Indexer device resident | 完整 Indexer page/scale/component 校验 | 核心阶段；实际 tensor + A3 |
| 其他 Main dtype / packing / scales | 普通按 cache spec/tensor | fused BF16 消费及量化布局能力差距，不能靠 Transfer Engine 字节复制解决 | 后续 kernel 工作包 |
| P/D 跨机，D TP 同一共享域 | endpoint 支持远端 host；Host Pool 由 TP group 创建 | 本地 VA/offset 一致性、每 rank Transfer Engine context 与 NIC 通路需验证 | 本轮布局；A3 待验证 |
| D TP 跨机共享域 | 不能假设普通 device 通路意味着 shared_segment 跨机共享可用 | Mooncake 版本/segment 后端能力未确认；需共享域与 offload 架构核查 | 不宣称支持、不修改 Mooncake |

依据：`ascend_config.py::_validate_preconditions`、`platform.py` SFA/MRV2 校验、
`mooncake_connector.py::_get_local_remote_cp_params/_get_kv_split_metadata`、
`SparseKVOffloadManager.register_kv_caches` 与 `allocate_mooncake_host_region`。
Mooncake 版本、目标模型与跨进程 registration 能力仍待目标环境核对。

### 重置后核心实现状态

用户本轮明确要求撤销除设计文档外的全部未提交改动。已恢复全部 tracked
代码/测试到 HEAD，并删除三个未跟踪的代码/测试草稿；操作后 git status 仅剩本文。
以下是从该基线重新实现的状态，取代上轮基础改动/49 项测试的阶段记录。
未 rebase、未提交、未推送；安装在 .venv 中的测试依赖保留。

| 工作包 | 本轮代码状态 | 验证边界 |
| --- | --- | --- |
| Q3 本地映射 | manager 保留各 rank 原始 views；connector/fused/指针列表同源；仅 Mooncake 绕开 owner 地址恢复 | CPU layout/manager 测试；真实不同 VA 共享物理内容待 A3 |
| Q3 布局/注册 | 校验 segment 对齐偏移、component/shape/dtype/stride/容量/重叠；每 rank 注册；shutdown 等队列和同步调用收尾后注销 | CPU 注册失败/重试/关闭；多进程注册能力待 A3 |
| Q6 来源规划 | 从普通 CP split 提取 `_get_cp_source_ports` 共用 endpoint 规划和计数；无 CP 复用 `_get_remote_ranks_for_req`；P 保留源端普通副本选择 | 普通 CP/hybrid 回归；不复制普通 block split |
| Q8 Main writer | 实际 handler 已移除 owner 门控，所有 rank 按完整请求 Host block ordinal 轮转，prefix 和远端子页不重编号 | 8/8/4、16/16/8、DCP=1，互斥覆盖/尾块/小请求/不等 block size CPU 测试 |
| Indexer | 每 rank 每 PP stage 选择已有 endpoint 接收完整 Indexer；按 token/page 几何计算 packing，再验证 dtype/bytes | BF16/int8 字节映射测试；真实 C8 scales 内容与模型输出待 A3 |
| component 身份 | P 显式输出 Main/Indexer cache-group ID 和 block size；DSA 独立 metadata 下标，远端按 layer name 对齐 | group 顺序、metadata 下标冲突和 PP 通知规划测试 |
| R2 空任务 | 无 external token/分配前拒绝均发通知任务；不调用读取、不产生无效 finished_recving；固定参与者和零 reader 端口保留 | CPU scheduler/handler 测试 |
| R3 完成聚合 | 复用 request_task_counts/finished_request_markers，最后提交标记与执行顺序解耦 | 最后任务先完成、提交间隙 pending 归零测试 |
| Q1 失败 | 使用 group 0 的 allocator ID 上报 invalid blocks，要求 fail policy，删除本地重算 | connector 失败测试；R1 多 group upstream 限制未修 |
| Q5 取消 | 共享逻辑取消 event，保留 active command 直到所有任务收尾，跳过尚未启动读取 | 在途同步读取取消/乱序完成测试；不承诺物理 DMA 排空 |
| Q4 去重 | 原 done 消息扩展 participant identity；tracker 内固定预期数与已完成集合；重复/晚到消息不重复计数，结束/超时清理 | duplicate/late/timeout/计数变化测试；未扩展重试协议 |
| 观测 | debug 日志包含每 rank 任务数、分阶段 bytes/耗时及合并前后 entry 数 | 性能数据待 A3 |

P/D 均需运行包含新 DSA metadata 的版本。缺失 dtype/component 握手信息按失败处理，
不猜测字节兼容性。block/page 地址规划用有效 token 范围，不将 padding 当有效数据。
普通 scheduler、Mooncake 库和 fused kernel 算法不修改。

### CPU 验证环境与结果

- 运行入口：`VLLM_VERSION=0.28.1 .venv/bin/python -m pytest ...`。
  本地 upstream 未生成 `_version.py`，沿用已有版本 override 机制；这不是版本验收证明。
- .venv：Python 3.13、torch 2.10 CPU；安装 xgrammar 0.2.3、pre-commit 4.0.1、ruff 0.14.0。
  保留 Transformers 5.14.1，避免新版 XGrammar 对 Transformers <5 的约束冲突。
- 最终核心测试：远端 DSA metadata/scheduler/handler/字节规划、Host Pool 和
  manager 六个测试文件 **148 passed**，包含最终传输容量越界校验。
- 远端扩大回归：`tests/ut/kv_offload` 与
  `tests/ut/distributed/kv_transfer/sparse_kv_offload`，使用
  `--continue-on-collection-errors`，结果 **467 passed、1 failed、1 collection error**。
  失败为下面的 `shared_by` API 不兼容；收集错误为 native offloading 导入 upstream
  已移除的 `is_kv_cache_tensor_packed`。未绕过测试或修改这些范围外 API。
- 普通 connector/hybrid：解除 loopback socket 的 sandbox 限制后 128 passed，1 failed。
  失败为既有 `test_m3_index_spec_is_preserved_and_splits_transfer_group` 读取本地 upstream
  已移除的 `KVCacheTensor.shared_by`，发生在调用本次改动之前；未修改 upstream 或隐藏失败。
- CPU Host Pool 测试显式 mock shared_segment API；manager staging 测试显式用 CPU allocation
  代替 pinning，只验证布局/时序，不代表实际 Mooncake 或 NPU 可用。
- 本次全部修改文件的 `pre-commit run --hook-stage manual --files ...` 在远端通过，
  包括 Ruff、codespell、typos、markdownlint、Gitleaks 与仓库自定义检查；
  日志为 `logs/lint-changed-final.log`。
- 必需的 `bash format.sh ci` 已在远端 `lint-snapshot` 独立副本执行：
  除 Ruff 外全部 hook 通过。Ruff 报告未修改的基线文件
  `tests/ut/ops/test_sparse_kv_offload.py:185` 的 F841（side_stream_marker）和
  `vllm_ascend/worker/model_runner_v1.py:3946` 的 F821（torch_npu）；
  全仓格式化还改写了基线文件，未带回本次工作树。日志为 `logs/lint-all.log`。
  因此不宣称全仓检查通过；本次修改文件检查已通过。
- Git diff whitespace 检查通过；同步 dry-run 确认远端已测源码与本地一致。
- 用户授权的远端为 `corporal@192.168.100.10`，全部源码、uv Python、venv、依赖缓存
  和日志位于 `~/Works/huawei/mayi`；CPU 64 GiB，没有 `npu-smi`。
  upstream 源码 HEAD 为 `ee3c00bbf47e0ef7e975705cc980b06ee5576bb0`；
  测试入口为该目录的 `run-tests.sh`，日志为 `logs/dsa-core-tests.log` 和
  `logs/kv-offload-tests-final.log`。远端使用 Python 3.13.15、torch 2.10 CPU，
  Ascend 源码基线与本地 HEAD 一致。

### 仍未验收的目标

- A3 的真实 P/D 请求、Host 内容与 decode 输出比对、不同本地 VA、registration/context、
  single-writer 基线对照、多 NPU 带宽/总耗时/并发吞吐和 Host 内存用量均未执行。
- Decode CP/PP、runner V2、非 BF16 Main/压缩模型等 offload/kernel 工作包仍需先细化方案。
  原有能力校验保留，完整兼容目标未撤销。本轮完成的是核心 connector 实现，不是全部矩阵验收。
- V1 PCP>1 被当前平台公共校验拒绝。Prefill V2 PCP 组合的实际 Indexer 复制几何仍需对应
  runner 环境验证；不能把通用 CP planner 测试当作该组合已完成支持。
- 普通 worker 当前同样拒绝 P_TP < D_TP；本轮移除的是 DSA 额外整除限制，未宣称更改公共范围。
- R1 的多 cache group scheduler 失败闭环、Q2 超时返回后的 late write 均为继承限制。
  无 A3 硬件验收，不宣称链路或性能通过。

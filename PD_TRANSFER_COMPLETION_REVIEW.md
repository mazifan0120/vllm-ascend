# PD transfer 最新分支核对与补齐进度

核对日期：2026-11-05。工作分支始终为 `p-main-buffer-reuse`，没有切换分支或 push。

**远端 `feat/layerwise-receipt` 尚未补齐完整 PD 链路，不能直接据此开启服务。**
本次已纳入远端增量并修复一批入口、元数据和失败路径；模型端到端功能及性能尚未验证。
早期 A1/A2 的原生资源验证仍然有效，但不等于完整 PD 验收。

## 第五阶段：D 消费与回收接线、元数据容量（2026-11-05）

本阶段仍未切换模型部署。新增生产接线已通过控制测试，真实远端 WRITE 与模型验收仍待运行。

- 精确逐 entry 结果及全部 P terminal 成功、静止后，receiver 才调用 worker 本地
  `confirm_native_received()`，记录实际 D consumption stream 的 NPU event 并建立 Host
  可见性依赖，再走 `prepare_decode()`。这个入口不向 scheduler owner RPC 暴露；
  不能用 request end 代替原生完成。sync WRITE 的真实 Host／NPU 可见性仍需设备验证。
- 实际 `NPUModelRunner.execute_model()` 用 `use_receipt_targets` 包住整个调用，覆盖
  异常和返回 None 的分支。MultiConnector 逐 owner 进入 runtime.use，退出时记录
  compute／manager writeback stream 尾部事件，不能在 Python 返回时直接丢引用。
  普通模式只经过空上下文。必须覆盖整个 runner 调用，是因为逐层 hook 不能涵盖
  forward 外的预处理和异步写回；这里采用现有 runner 的显式入口，没有全局 monkey patch。
  实际 NPU 开销及兼容性尚需模型回归，不据控制测试宣称性能无影响。
- `ReceiptScheduler` 保存单次 finished_recving 通知，全部 TP owner ready 后才交给基类；
  后续空闲调度继续轮询，不丢掉较早完成 rank 的通知。回收必须有所有 owner 的
  remote_quiesced、真实 GPU 处理 fence 和 owner stream 静止，之后才签发 ReleaseAck
  并 retire 原分配。未 admission／未暴露地址的目标可由关闭 lookup gate 证明没有写者。
- exact target 使用带 EngineCore 后缀的内部请求 ID，完成接收后保留 ledger、block map
  和失败归属直至 owner retirement；同一外部 ID 的两个不同生命周期不会互相覆盖。
  watchdog／显式失败永久取消 decode，但继续接收迟到的静止 proof。真实 vLLM
  `failed_recving` 按请求报告失败，不用不同 Main／Indexer pool 可能重叠的数字 block ID。
  已进入 RUNNING 的 retained 请求也会被迟到失败命中；未知 DMA 继续保留资源。
- 地址 lookup 在 worker 本地 gate 下检查 lease／generation／terminal／abort／retirement；
  closed 目标的首次 lookup 被拒绝，P 在尚未提交 native WRITE 时按安全取消处理。
  **这尚未关闭 P 已缓存地址后的全部未来写者，不等于 D→P 远端取消协议完成。**
- owner codec v2 仅引用不可变 primitive tuple，保留类型、深度和所有既有限额；
  HTTP target lease v2 的 view block map 引用 allocation group。两种 decoder 都接受 v1。
  TP16×78 层、32K／64K／128K prompt 的合法 manifest 通过 owner 与 HTTP 往返；
  未扩大 8 MiB／500k node 限额。旧版 reader 不识别 v2，P／D／scheduler／worker 须一致升级。

启用此路径的 D 引擎必须选择
`vllm_ascend.core.receipt_scheduler.ReceiptScheduler`、`kv_load_failure_policy: fail`，
并在 D connector extra config 配置 `layerwise_receipt_mode: true`、
`target_owner_rpc_base_port` 和 `target_retention_rpc_port`（端口须空闲）；
P 仍由请求 manifest 触发。现有普通部署配置不自动改成 receipt 模式。

待验证／待完成：新 bounded 注册路径和真实 RoCE WRITE→Host／NPU read；D→P
关闭未来写者并取得静止证明；P restore-only 的真实 Store coverage／调度选择；
全模型短请求→32K／64K、持续 decode／写回、并发取消／抢占／重分配／失败及普通兼容性；
匹配配置的性能和关键路径分析。当前跨 owner release 部分成功后 RPC 失败仍保持内存
保守持有，但正在运行的抢占候选可能需要隔离／恢复；不能宣称请求级故障恢复已完整。

本阶段组合 CPU 回归 284 项／76 子用例通过（1 项运行时限定 skip），P 聚焦
141 项通过。原 P 容器独立进程内 749 个候选生产 Python 源文件内存加载、
15 项真实 connector 方法及 7 条真实 vLLM scheduler 控制路径通过，最终源码摘要匹配。
这些测试没有远端 DMA 或模型计算；Host 注册选择另由实际 CPU storage／recording TE 检查。
变更文件 manual hooks 全通过；隔离副本全仓 CI 因未改动基线 lint／拼写失败，
自动格式化的 7 个基线文件与交付无交集，没有复制回源码。四个原容器身份、image、
owner、脚本及各 10 个已部署源码摘要匹配，健康 200、running／waiting 均为零。

证据：`artifacts/p-main-buffer-reuse/pd-transfer-decode-wiring-20261105/`。

## 第四阶段：SourceReady 直接发送与 exact epoch（2026-11-05）

- 新增 `ManifestSourceWriter`，`prepare_layerwise_step()` 先登记 manifest 请求，
  保留 manager block ID，不再套用旧 sender 的 kernel block／port 转换。
  Store `SourceReady` 直接生成全部请求的 read job，无需等待 model save hook。
  使用已有 NPU send thread 执行真实 `batch_transfer_sync_write` 调用；Host／device
  分批，Main P0 唯一写者、Indexer Pi→Di，地址通过原始 owner lease 的 peer lookup 取得。
  **本轮验证使用 CPU 实存储 copy engine 和真实依赖入口，尚未提交原生远端 WRITE。**
- source consumer 现在登记整个 batch 的 exact epoch ticket 后才发布任何任务；
  等待必须满足 ready、sealed、所有读者完成且无错误，不再委托 layer-number 等待。
  普通 sender 的读者也有完成回调；晚于 seal 的 legacy hook 使用预留 ticket。
  已完成 epoch 按物理 slot 保留高水位，旧 Indexer occupant 重复等待不会追逐新 Main。
- manifest 请求从旧 `save_kv_layer` 发送循环剔除，避免双发；混合 batch 仍保留普通任务。
  必须显式接入 Store source registry，否则收到 manifest 时失败，不静默等 watchdog。
  实际有效尾部取 Store source 与 D prompt 的交集，私有 Store instance 不进入 PD wire。
- 原生完成先回调 source reader；peer 回执由独立线程发送，等待 D ACK 不占用源槽。
  逐 entry 结果按 receiver 批量发送，并复用该线程独占的 REQ socket；terminal 携带
  明确 generation 和 source_quiesced。D 的实际 batch handler 已接到精确账本。
  原生负返回／异常一律当作 DMA 未决，保留 source；不能据此发静止成功。
  某请求的 peer 发送耗尽重试后，不对它后续每层重复等待同一超时，D 仍保持不放行。
- P worker 收到 finished request 后，完整已登记计划仍排空在途读者，随后清理重型状态；
  未覆盖完整计划的结束按 CANCELLED 封口，未执行的 job 不提交 DMA，也不污染安全源槽。
  清理后保留目标代际到写者的精简记录，拒绝相同／不同 P 内部 ID 重放已封口目标。
  这不是 D→P 远端取消协议：后者以及 D proof／retirement 清理仍待完成。

本阶段组合 CPU 回归 182 项／50 子用例通过（含旧 receipt），1 项运行时限定 skip；
P 聚焦 141 项通过。覆盖实际 CPU memmove 字节、跨页尾部、原生 fence 顺序、
ACK 阻塞与源释放分离、失败静止保留、封口、部分取消、重型状态清理及真实 ZMQ 连接复用。
原 P 容器独立进程中，748 个候选生产 Python 源文件内存导入通过，9 项真实准备／
receiver／early-admission 方法、既有 hook／sender fence／CPU 注册选择和四条实际
vLLM scheduler 控制路径通过；没有替换服务源码、启动模型或真正远端 DMA。
证据：`artifacts/p-main-buffer-reuse/pd-transfer-source-writer-20261105/`。

下一步先补 D 实际接收依赖、forward／writeback owner-use、全 rank readiness 与
ReleaseAck、超时后晚到 proof 的保留／安全清理，再接远端取消及真实 P restore-only。
之后必须验证新注册路径与真实 RoCE WRITE→Host／NPU read，及完整模型功能和性能。
当前完整模型功能／性能仍为 NOT_RUN；不能把上述 CPU copy／控制测试当作验收。

## 第三阶段：实际目标计划与 receive admission（2026-11-05）

仍在原分支开发，本阶段没有重启模型或替换运行服务的源码。

- `mooncake_layerwise_plan.py` 由完整 manifest 为每个 layer／component／entry
  分配稳定 operation ID；Main 只由 P0 写一次，共享完成消息送全部 D owner，Indexer
  由对应 P rank 写入对应 D rank 的 data／scale。复用现有 DSA 页映射器，支持 P/D
  独立的物理页、stride、分组、block ID 和未对齐有效尾部。
- `ManifestWritePlan` 在请求 admission 时预编译每个 producer／layer 与 receiver
  的 entry 列表，避免未来 SourceReady 路径逐层重扫整个 TP manifest。
  当前尚未由生产 source driver 调用 `plan_layer_writes()` 提交原生传输。
- worker 本地 `resolve_target_regions()` 在原始 lease 有效期内解析真实 MR 基址，
  经 D 侧既有 peer listener 返回绑定 request／token／allocation／registration 代际
  的地址。地址不进入 scheduler owner IPC，后者继续只接受白名单 catalog／lease 值。
- D 注册复用现有 bounded DSA packer，支持同一真实 backing 的多段不重叠 MR，
  不再把整个 Host pool 当作一个无限大的注册区。组件 view 仍须完整落在一个实际 MR
  内，单 entry 跨 MR 的超大几何明确拒绝；本轮没有完成真实新注册路径的 NPU 验证。
- `ExactReceiptLedger` 按 admitted entry 的完整 key 和有效 token 区间记账，
  不用 layer×backing 的笛卡尔积。SUCCESS 且 source_quiesced 才计覆盖；全部写者
  明确封口且所有已知原生操作静止才提供 remote_quiesced。失败／取消可满足安全回收，
  但不会满足 decode readiness；冲突、未知 entry、错误代际或未决 DMA 均不能放行。
- scheduler 在通知 P 前对全部 TP owner 执行同步 `admit_target()`，逐个安装真实
  原始 lease、精确账本、内部／外部 request 映射和失败 block 列表后才确认。
  connector metadata 随后重复到达是幂等的，不重置已收到的完成结果。
  部分 admission 失败保留目标，不通知 P；acquire 后校验失败也保留可追踪 lease。
- 真实 receiver 处理器已切入 manifest 精确账本；terminal 必须附带明确的 DMA
  静止 bool，并匹配 request generation。旧 plain／legacy 协议路径保留。
  新账本已有 all-rank 状态查询，但尚未生成生产 ReleaseAck；缺证明仍保持分配。

本阶段组合 CPU 回归 119 项／46 子用例通过，另 1 项运行时限定测试 skip；
P 聚焦 141 项通过。组合覆盖真实双 owner loopback RPC admission、CPU 实际存储和
memmove 的页／尾块／padding 检查、碎片注册、未知／冲突／乱序消息等。
原 P 容器的新独立 Python 进程通过 747 个候选生产 Python 源文件内存加载、
3 项真实 receiver／early-admission 方法、3 项 sender fence、1 项 CPU 存储注册选择、
6 项 hook 及 4 条实际 vLLM scheduler 控制路径。
上述进程未提交真实远端 DMA 或模型计算；完整模型功能及性能仍为 NOT_RUN。
证据目录：`artifacts/p-main-buffer-reuse/pd-transfer-data-plane-20261105/`。

接下来必须接通 SourceReady 直接提交这些计划与 exact epoch reader、真实原生完成
及 peer 回执、D Host／device 可见性与完整 forward／writeback use、全 rank readiness
和远端静止聚合，再完成取消／抢占及 P restore-only。完成后才启动短请求和 32K／64K
同配置无共享／共享对照，不能用本阶段控制测试代替完整验收。

## 第二阶段：生产初始化和 scheduler 门禁（2026-11-05）

本轮继续在同一分支开发，新增以下真实调用入口；**尚未启动完整 PD 模型**。

- `ReceiptScheduler` 在基类构造完成后将同一个串行 KVCacheManager 适配器重新绑定给
  connector，启动 scheduler 反向 retention server，并连接全部 TP owner。
  它覆盖正常结束／取消后的延迟 free、常规抢占、强制 prefix reset 及异步接收失败。
  标准 GPU step fence 必须先满足；不额外增加 block ref_cnt。
- `target_owner_factory.py` 接入 connector 的 scheduler／worker 初始化，worker 在真实
  注册并绑定 owner 后才启动自己的 endpoint。生产路径不再使用模块全局 rendezvous。
  参数为 `target_owner_rpc_base_port`、`target_retention_rpc_port` 和可选
  `target_owner_rpc_timeout_s`；每 rank 一个端口，retention 端口不得与其重叠。
- `MultiRankTargetProvider` 保留各 rank 原始 lease；`TargetManifest` 只选择一次共享
  Host Main 注册，并包含所有消费 rank 的 Indexer。P HTTP 元数据携带完整 manifest，
  D worker 选择自己的原始 lease，不把聚合 lease 当成本地 owner 的分配。
  校验 Main backing／generation／偏移一致、entry／分组／页大小／stride／容量和有效范围。
- MultiConnector 明确转发 worker owner 初始化。直接使用 LayerwiseConnector 也会自行
  挂载稳定 provider；重复挂载同一对象不重置状态。
- D 注册代码改用 manager 已有 Host Main 与 HBM Indexer 视图，Host pool 单独注册且不含
  原始 segment 的对齐余量；保留每个 entry 的页比例，支持 Main／Indexer 属于不同 group。
  这是代码接线与 CPU 注册范围回归，**真实新路径的 TE／HostRegister 仍待原生验证**。
- `request_finished` 只报告需要延迟回收；实际 release 留给 scheduler 在 GPU fence 后执行。
  还必须提供匹配 token 的真实远端静止证明，不能因请求结束就制造 safe ReleaseAck。
  证明的真实数据路径尚待下一阶段接线；缺证明时保持分配，不会自行释放。

本轮证据在部署仓 `artifacts/p-main-buffer-reuse/pd-transfer-production-wiring-20261105/`：
P 聚焦 141 项通过；组合 67 项、33 个子用例通过，另 1 项真实依赖限定用例明确 skip。
包含实际 loopback socket 的双 rank factory 往返／反向 retention、TP16×78 层描述符
编解码、部分 acquire／release 失败保留、共享锁阻止 RPC 看到分配中间态等。
真实运行时测试执行了实际 vLLM Scheduler／KVCacheManager 的四条控制路径，
使用 CPU 未提交 DMA 的证明 fixture；它不构成真实远端取消或模型写回验证。

model runner 的调整只发生在 KV 初始化：通过 wrapper 的显式入口传入已存在的 manager、
消费 stream 和 TE，再注册实际目标；不增加逐 token／逐层 forward 调用。
原入口只识别外层单 connector，无法覆盖 MultiConnector，这是改动的必要性。
scheduler subclass 则用于 connector 的 finished hook 无法覆盖的 free／preempt 路径，
并复用上游 GPU fence 与原分配器；没有改模型实现或通过额外引用计数触发 CoW。
这些初始化／控制路径回归已执行，性能影响仍需实际模型与相同部署参数对照确认。

下一步优先完成 manifest 驱动的有效范围传输计划、worker-ready、SourceReady 直接发送、
真实 receive 可见性／forward-use／writeback 与取消／静止证明，再执行模型部署与性能测试。
下面的“本次已修复／完整特性仍缺什么”保留首批 `9cbdde548` 时的审查上下文，
其中 IPC、初始化和 scheduler 门禁的代码进展以上述本节为准；模型验收状态仍未完成。

## 固定审查对象

| 对象 | 提交／证据 |
| --- | --- |
| 我方起点 | `717f6e6e0d3e327e292c3b3d0d95da116b5319fe` |
| 远端 receipt | `fa5aec40b0cbb4662ec4c5dae68114a850642337`，已实际 fetch |
| 共同祖先 | `e817452b2` |
| 远端新增 | 12 个提交、25 个变更文件；包含 source consumer、pipe transport、retention 和 watchdog 等 |
| 运行时 API | 从已恢复的原 P 容器只读取回实际 vLLM 源码，版本 `0.30.0+empty` |

完整增量、逐文件摘要和执行日志保存在部署仓的
`artifacts/p-main-buffer-reuse/pd-transfer-completion-20261105/`。
不能只按 `RECEIPT_HANDOFF.md` 中的描述判断真实接线。

## 本次已修复

1. `ReqMeta` 在默认字段后声明必填字段，实际 dataclass 构造会抛 `TypeError`。
   已抽出纯数据 metadata 模块，保留旧参数顺序，新增字段放在末尾。
2. D 原先将 `WriterPlan` 对象直接放入 HTTP JSON，且没有传完整 lease。
   已新增严格的 JSON lease 编解码，携带 region、view、allocation、writer plan 和身份；
   P 入口校验 request instance。sender 消费目标 view 地址仍待接线。
3. 真实 vLLM scheduler 使用 `self.connector`，并会主动调用
   `bind_kv_cache_manager()`；原先自定义 scheduler 的 `.kv_connector` 路径没有生效。
   已改用 connector 的标准 hook，并删除错误接线。
4. D 有完整目标容量不代表 P 的 Store 有完整前缀。已删除 D 根据容量自动标记
   restore-only 的逻辑；P 真实覆盖判定和无 forward 路由仍待实现。
5. P source 的有效末尾原先取公开完整 block 的保存边界，遗漏 partial tail。
   已改用实际 `target_token_len`，restore driver 也按实际有效长度验证。
6. source consumer 原先覆盖发送任务的 `wait_event`，可能丢失量化／重排依赖。
   现在单独携带 Store source fence，在读 source 前等待，并保留原发送路径的 fence。
7. scheduler 原先在 retention retirement 失败后仍允许 free，且提前删除 lease。
   现在 retirement 失败或 authority 缺失均延迟释放，保留 lease 供核验／重试。
   远端静止证明和真实取消／抢占门禁仍待完成。
8. 新增只传白名单值的 loopback RPC：独立进程无需继承 pipe FD，worker 可在 scheduler
   建立前构造反向 retention client；不使用 pickle、动态类型导入、新 env 或全局 rendezvous。
   超时会将 client 标为结果未决，拒绝自动重发 acquire/release。
   **该 transport 尚未接入生产 factory。**
9. 保留普通 Store／无 Store 的旧协商路径；CPU 测试对必须真实 vLLM 的用例明确 skip，
   修正 fork fixture 错将子进程自行签发的 token 当作父 scheduler 权威 token 的问题。

## 完整特性仍缺什么

| 项目 | 实际缺口与完成标准 |
| --- | --- |
| 生产 IPC 与 TP16 | Pipe helper 没有生产调用者；旧默认路径依赖进程内 rendezvous，且错误读取 multiprocessing 默认值。需接实例级双向通道和全部 rank 的 catalog／lease，不能只连接 rank 0 |
| MultiConnector 初始化 | model runner 仅探测外层单 connector 属性，包装层不会透传。需转发 owner 初始化，在真正 Host 注册后构造 catalog |
| Host Main 注册 | 当前 layerwise worker 仍注册旧 HBM cache；需复用既有 Host pool／DSA 注册逻辑，注册真实 Host Main 和 HBM Indexer，不另分配生产 KV |
| 完整传输计划 | sender 仍使用旧 layer metadata。需按实际目标 view／entry／consumer rank 生成 Main 唯一 writer、Indexer 各 rank data／scale、有效尾部、独立 Host／device 批次 |
| 精确 receipt | 必需结果应来自实际计划，不能将所有层与所有 Indexer backing 做笛卡尔积。每项含真实 backing／generation／entry／consumer／range，完整覆盖后才可发布 |
| worker ready | scheduler 发布后 worker 还要等下一轮 metadata 安装 ledger；提前到达的结果会被丢弃。必须先确认所有 worker 安装完成再通知 P |
| SourceReady 与 epoch | consumer 仍等旧 `save_kv_layer` 组装任务，restore driver 不调用该 hook。需 SourceReady 直接生成／登记真实读者，exact epoch 封口与静止后才放行，避免重复发送和记录增长 |
| D 可见性及使用范围 | `publish_received()` 和 owner `use()` 尚未接到真实 receive／forward／写回。需实际远端完成及本地 stream 依赖，覆盖完整消费和异步写回 |
| 安全回收 | 请求结束不能替代全 rank DMA 静止证明；上游 preempt 仍直接 free。需远端取消、未决任务、本地使用结束、scheduler 串行 retirement／延迟回收，失败保留分配 |
| restore-only | 仅由 P 实际已提交的完整 Store 覆盖选中，且真正绕开 model forward；混合 batch 不能整体跳过普通请求 |
| 有界失败与兼容 | 覆盖 admission 前失败、丢回执、未决 RPC、尾部、并发、重分配及普通模式；watchdog 超时不构成可回收证明 |
| 部署及验收 | BF16 Main、D Host Main／HBM Indexer、eager、DCP1／无 MTP；实测连续 decode／写回、池命中／restore-only、取消／抢占及同配置性能 |

## 验证范围与执行顺序

已完成 CPU 验证：P 聚焦回归 141 项，receipt 账本 37 项；metadata、wire、owner、
独立进程双向 RPC、source consumer 与 retirement 失败路径组合为 61 项、24 子用例，
另有 1 项需要完整运行时的测试明确跳过。证据以对应日志和后续最终快照复跑为准。
RPC 子进程测试只证明值传递、反向 retain／validate／release 和未决失败处理，
没有提交模型计算或远端 DMA，不作为 NPU 可见性证明。

原 P 容器内的独立 Python 进程还通过了 739 个候选 Python 源文件的内存加载检查：
真实导入 connector、MultiConnector、owner RPC 和 model runner，并验证 receipt／普通
两种标准绑定 hook。真实 sender 方法的 3 项 CPU fence 用例通过：两个 fence 都在提交前，
任一 fence 失败均不提交。没有替换容器已安装源码，没有启动模型或真正远端 DMA。
首次检查脚本未按 Python 源文件规则处理 UTF-8 BOM，修正脚本后复测通过；保留首次日志，
不将其误记为产品源码故障。

本次变更的全部 manual hooks 通过。全仓 `bash format.sh ci` 在隔离副本执行，
仍因未改动的 `ascend_config.py`、`sfa_kv_offload.py` 的既有 lint／拼写问题失败；
隔离副本的自动格式化文件与本次交付无交集，未回写到工作分支。没有宣称全仓 CI 通过。
检查后原四容器的 ID／image／owner、各 10 文件源码摘要一致，健康 200，运行／等待请求均为 0。

下一步按真实依赖补齐：scheduler free／preempt 权威接线与多 rank owner 通道、
Host 注册和目标计划；再接精确 receipt／worker ready／source 任务和 D 消费范围；
随后补 P restore-only、取消和异常回收，形成可审查部署包。

Mooncake 五部署档 32K／64K 测试已经结束并恢复原 P r05／D r03。
开始模型实验前仍需实时核验原容器 ID、owner、源码、健康和空闲。
先做短请求功能，再测 32K／64K 同配置无共享与共享性能，依据实际 trace 优化。
不运行本轮用户暂缓的 1M 回归；不把旧 INT8 档数据当作新 BF16 PD 链路性能。

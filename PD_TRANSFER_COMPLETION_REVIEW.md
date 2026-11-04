# PD transfer 最新分支核对与补齐进度

核对日期：2026-11-05。工作分支始终为 `p-main-buffer-reuse`，没有切换分支或 push。

**远端 `feat/layerwise-receipt` 尚未补齐完整 PD 链路，不能直接据此开启服务。**
本次已纳入远端增量并修复一批入口、元数据和失败路径；模型端到端功能及性能尚未验证。
早期 A1/A2 的原生资源验证仍然有效，但不等于完整 PD 验收。

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

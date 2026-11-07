# P侧共享KV缓冲区与PD传输交接

> 2026-11-04 两分支整理入口：[实现现状与剩余工作](PD_TRANSFER_IMPLEMENTATION_STATUS.md)。
> 本分支现包含 receipt 基线及 A1/A2 owner 实现，见 [owner 交接](TARGET_OWNER_RUNTIME_HANDOFF.md)。
> sender、生产 IPC 与完整生命周期仍待接线，详见清单。
> 下文保留各阶段交接记录，不将历史验证扩大为最新集成树的端到端验收。

本文面向负责 Mooncake layerwise P/D transfer 的开发者，说明已实现的 P 侧源接口、接入顺序和验收边界。
交接版本为 `47f98315724770372462399d390068cebb7c3c16`，分支 `p-main-buffer-reuse`，日期为 2026-11-03。

P 侧 Main 跨层复用、Store 保存与恢复已经实现；新的源接口已通过受控读者的真实模型 64K 验证。
**真实 P/D 网络 sender 尚未接入，本文不是 P/D 端到端完成声明。**

## 1. 版本与支持范围

| 项目 | 版本或约束 |
| --- | --- |
| 固定上游基线 | `4234a1f1815252e8b52c75d6956d0bdf9ae80cc0`，`dsa_offload_rebase_30` |
| Main-only reuse 功能提交 | `9824ec91c` |
| 格式整理提交 | `f9b995c68` |
| 本次源接口提交 | `47f983157`，包含空 GET 接线修复及测试 |
| 目标组合 | GLM-5.2 / A3；硬件模型验证为 TP16 × DP2 |
| 执行限制 | P producer、V1、eager、PP=1、CP/DCP=1、无 speculative/MTP |
| 缓存限制 | 支持的 Main/Indexer spec、单 KV manager group；Indexer 保持独立分配 |

共享槽数是 Main 的共享槽数，独立层另占槽。例如 B=2、首尾独立时，Main 共用 4 个物理槽。
不支持的布局必须显式拒绝，不能假定本接口已经支持任意压缩、混合缓存或原生跨层 Indexer 别名。

已有 P 独立 1M 验收对应此前 r05 版本，不包括本次新增源接口。
当前集群已恢复该 r05 服务；带测试读者的 r06/r07 容器已停止保留。

## 2. 从哪些代码开始读

下列链接相对于本仓库根目录，无需依赖部署仓库文档才能定位接口。

| 文件 | 主要职责 |
| --- | --- |
| [reuse_source.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/reuse_source.py) | 不可变源视图、槽位代次、请求范围、五个 consumer 方法及 registry |
| [ascend_multi_connector.py](vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py) | provider/sink 发现、v1 自动安装、provider 优先 hooks 与退出顺序 |
| [ascend_store_connector.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py) | worker 接口转发、metadata 绑定、显式 restore-only 入口 |
| [pool_worker.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py) | 实际视图生成、ready 发布、恢复编排、计算完成及清理 |
| [kv_transfer.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py) | 接收线程覆盖前屏障，range 和空 GET 分支，失败传播 |
| [layerwise_cache_layout.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/layerwise_cache_layout.py) | Main-only 布局规划；槽位映射的唯一规划来源 |
| [attention_fence.py](vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/attention_fence.py) | attention-start 与独立 attention-done 的设备事件语义 |
| [test_reuse_source.py](tests/ut/distributed/ascend_store/test_reuse_source.py) | 源契约、真实 receiver 分发、自动接线和 restore-only 的 CPU 测试 |

本次未实现另一套生产 sender，也未要求改走现有 SFA/MemFabric connector。
新 connector 继续使用 Mooncake；具体传输 API、wire 协议和 D 端接收汇总由传输开发者实现。

## 3. P侧现在提供什么

### 3.1 源视图与身份

| 类型 | 字段与使用方式 |
| --- | --- |
| `SourceView` | `layer/component/group/slot/address/length_bytes/shape/strides_bytes/dtype/manager_block_tokens/page_tokens/entry` |
| `SlotEpoch` | `worker/slot/generation/layer`；标识本 worker 内某物理槽的一代占用 |
| `LayerSource` | 一层的 `views` 和 `epochs`；按 `view.slot == epoch.slot` 关联，不按列表位置 zip |
| `RequestSource` | `request_id/request_instance/start_token/end_token/block_ids_by_group`；范围为左闭右开 |

约定如下：

- `component` 当前是字符串 `MAIN` 或 `INDEXER`；Indexer 不存在的层没有相应 view。
- 同一组件可有多个非空 tensor，例如 Indexer data/scale；`entry` 是过滤空 tensor 后的组件内序号。
  各 entry 的含义需结合受支持 cache spec 冻结，不能跨模型猜测。
- `address` 是当前 P 进程的本地 NPU 地址；`length_bytes` 是可访问物理跨度，不是本请求要发送的字节数。
- 根据有效 token 范围、实际 block IDs、物理 page 和字节 stride 构造地址；不能把整个 span 或 partial 尾部 padding 都作为有效 payload。
- `slot` 是本地不透明标识。Indexer slot 可能含 storage identity，不是远端注册区 ID。
- `worker + slot + generation` 用于隔离重启和旧回调。不要只按逻辑层或可重用的外部 request ID 记状态。
- 当前每 step 发布 `[0, token_len_chunk)` 的有效源范围；其中包含历史前缀，并不表示每次都要全部发给 D。
- `request_instance` 是 P 侧身份。B 需将其与 D manifest 的目标身份显式绑定，不默认双方独立生成的 UUID 相同。
- Slot generation、Store snapshot generation、P/D transfer generation 是三种版本，不能互相替代。

这些对象是 worker-local 契约。事件、tensor 和本地引用不能直接序列化到 wire。
Store 私有快照 key 不属于 P/D 协议。

### 3.2 五个consumer方法

| 方法 | 要求 |
| --- | --- |
| `open_source(source)` | 在本代首次 GET/scatter 之前打开登记窗口；OPEN 不能被当作空闲 |
| `source_ready(source, requests, ready_event)` | 为完整 batch 的所有发送项先登记读者，再使异步任务可运行 |
| `seal_source(source)` | 表示本代不会再增加读者；零请求、零 payload 也必须封口 |
| `wait_source(source)` | 只有 SEALED、全部必要源读取已静止且成功才返回；失败必须抛异常 |
| `abort_source(source, error)` | 标记失败、停止新提交并保留仍在途的引用；不能用它冒充源已静止 |

前四个登记／通知动作中的 `open/source_ready/seal` 以及 `abort` 必须是线程安全的本地状态操作。
它们不能同步等网络、等设备完成或等 D 目标；真正的等待放在发送线程和 `wait_source`。
回调可能来自 Store receiver 或模型线程；等待时不持有完成回调需要的锁。

同一 `LayerSource` 的重复 ready 已在 registry 内去重。传输子项、重复完成、迟到旧代、多个组件和多个请求的计数及幂等仍由 consumer 负责。
v1 的 `wait_source` 等待该层全部相关读者，不应假定已经提供 Main/Indexer 独立提前释放的接口。

## 4. 如何接入MultiConnector

### 4.1 connector外壳条件

`AscendMultiConnector` 选出的 provider 必须满足：

1. `is_producer` 为真，且已有 `connector_worker`。
2. `supports_layerwise_buffer_reuse` 为真。
3. 提供可调用的 `wait_for_layer_reuse(layer_idx)`。
4. 若启用新契约，在 **connector 外壳**声明 `reuse_source_contract_version = 1`。
5. 外壳实现五个 consumer 方法，可直接委托给自己的 worker。

只在 worker 上声明 version 不会触发当前自动安装。D consumer 不应被误判为 P provider。

v1 自动安装目前要求恰好一个传输 consumer，以及一个已接好 external waiter 的 Store sink。
Store 的 `set_reuse_source_consumer()` 必须在 KV 注册和传输线程启动前调用，不能中途替换。
未声明 version 的旧 connector 不会自动启用新协议；当前没有完成更高版本的协商。

保留 `wait_for_layer_reuse()` 兼容入口，使它和五个新方法使用同一份槽位／代次状态。
不要再引入一份可能提前置位的独立“发送完成”状态。

### 4.2 正常执行时序

```text
Store receiver 准备进入层 M
  等旧占用层的 Store PUT 源读取结束
  等旧占用层 attention-done
  调用 external waiter -> provider.wait_for_layer_reuse(M)
  registry 等旧源 wait_source 成功
  registry 推进代次 -> open_source(M)
  GET 历史 KV（可以没有 GET，屏障和 open 仍执行）
  通知 layer-load 完成，模型才能继续写 KV

本层 KV-ready
  记录本地 NPU ready_event
  source_ready(M, 完整 batch, ready_event)
    consumer 先登记全部读者，再发布异步发送任务
  seal_source(M)
  发送线程等 ready_event 后才读源
  attention、Store PUT、P/D 读取可按依赖重叠
  下一次覆盖前等待计算、Store 和外部读者全部结束
```

已有 `on_kv_cache_written/save_kv_layer` 路径采取 provider 优先调用，之后才调用 Store sibling。
如果新 connector 保留旧 hook 发送路径，必须与 `source_ready` 去重，不能两边各发一次。
建议用新 ready 契约统一发布发送计划，旧 hooks 仅作必要适配。

不能以 Store 整对象 commit 作为同 step 槽位释放条件：对象可能等最后一层才提交，B=1 会形成循环依赖。
`KV-ready`、`attention-done`、`source-read-done`、`snapshot-committed`、`D request ready` 分别管理。

## 5. 传输侧还要完成什么

### 5.1 源读取状态机

至少维护 OPEN、SEALED、读者 pending 和失败状态。每个发送项绑定请求实例、slot epoch、组件和有效范围。
完整计划先 reserve，再 enqueue；一个请求或组件完成不能释放仍有其他读者的源。

发送入队或 RPC 提交成功不代表源读取结束。必须核对使用的 Mooncake API 在返回／callback 时对源生命周期的真实保证。
初版不依赖 staging 提前释放；若以后增加 staging，需证明源拷贝完成、staging 自身不被提前复用，并另记其内存成本。

### 5.2 D目标、范围与完成

- D 在相关 P 执行前分配并保留目标，完成注册／兼容性校验，再发布目标 manifest；资源不足先背压或失败。
- Main Host 与 Indexer HBM 分批传输，不在同一批混合两种目标介质。
- Main 按共享 backing 身份和 writer plan 确定唯一写者；不能用各进程虚拟地址相等来判断共享。
- Indexer 按每个消费 rank 的目标覆盖，不直接套用 Main writer 过滤。
- 根据 D 的 required ranges 和已完成范围决定本次 payload，避免每个 chunk 重传完整历史。
  P prefix 命中而 D 无缓存时，历史范围必须完整传入。
- partial 只传有效范围；初版不并发改写相同目标字节，也不以晚到旧代结果补新代覆盖。
- Main/Indexer 分别累计结果；无 Main payload 的 rank 仍可能发送 Indexer，也仍需发送 terminal。
- D 只在 required ranges、必要 rank terminal 全部成功且数据可见时开始 decode。

P目前没有提供完整的 D TargetManifest、TargetLease 或 wire encoder。
真实 D Host allocation/runtime 使用既有 manager/scheduler 能力，在共同集成阶段绑定视图和 block retention。
无需等待完整模型路径才能开始：独立阶段可以用测试分配的 Host Main／HBM Indexer 目标。

### 5.3 失败与取消

失败结果和 `source_quiesced` 必须分开。失败可立即让等待者抛异常，但源仍在读取时不能回收或覆盖。
后来确认静止只解除资源引用，不把失败改成成功；旧代回调只能清理旧代引用。
D 目标同样保留至远端写入和本地 runtime 访问均静止，不能只因请求取消就归还 blocks。

P提供 `abort_reuse_sources(error)` 协调入口。现实现失败后 latch error，禁止原地继续写入，并保留分配。
正常退出采用 drain；同步传输卡住的底层超时／取消／进程终止保证仍需共同落实，不能以线程存活检查代替。

## 6. 无计算hook的restore-only路径

当前有显式 `AscendStoreConnector.restore_kv_for_transfer(metadata)` 入口，逐层恢复并发布 ready，无需 model forward。
它要求有效 `request_instance`，且每请求 `restore_end_token == token_len_chunk`，不允许混入尚需计算的新 token。

调用方必须先完成 D 目标 admission；同一份 metadata 不得再走正常 forward，也不能重复 prepare/bind。
**scheduler 尚未自动选择这条路径。** 需要双方确定请求调度位置、完整恢复范围及失败返回通道。

“无 GET”已有正常模型验证；“无计算 hook”目前通过的是人工 KV 的真实 NPU/Store 驱动测试，
尚不是完整模型 scheduler 的端到端 restore-only 验收。不要用 prefix 命中但仍有新 token 计算的请求替代该项。

## 7. 建议开发和验收顺序

1. 先实现 consumer 状态机和外壳委托，用模拟 Store 验证未封口、部分完成、迟到／重复回调及失败。
2. 使用真实 NPU 共享源、Host Main／HBM Indexer 测试目标，验证 Mooncake 数据内容和源读取完成保证。
3. 对接实际 Store sink，先覆盖首 chunk 空 GET、普通多 chunk、partial 和不同组件 stride。
4. 接入真实 D 目标与 readiness，覆盖全 prefix、restore-only、多请求、慢发送和取消。
5. 再验证真实模型连续 decode、D Host 写回及内存／性能；普通无复用场景也做回归。

必须有确定性阻塞测试：

- OPEN 未封口、零 payload 未封口、只完成部分 ticket，均不能覆盖。
- attention 与 Store 已结束、外部读者仍未结束时，GET 写入和模型写入都必须被挡住；空 GET 同样受保护。
- 旧 epoch 迟到回调不得释放新 epoch；相同 request ID 重建需新实例。
- GET/PUT/enqueue/传输失败不能制造成功 ready 或提前放行 D。
- Main 或 Indexer 单路径失败、缺一个 rank、最后层空 payload 时，D 不得错误开始。
- 取消后立即分配新请求时，旧源／目标在途访问不能污染新请求。

## 8. 已有验证与证据边界

| 验证 | 结果 |
| --- | --- |
| 聚焦 CPU UT | 122项通过，包括本次接口及空GET真实分发回归 |
| 模型冷 prefill | P0实际8190／65532 tokens，检索正确；64K约43.7秒 |
| 同前缀检索 | P0实际65533 tokens，BETA正确；本地prefix hits 65408 |
| 第二个P引擎 | P1实际32762 tokens，新请求检索正确 |
| 真实源接口调用 | 32 rank，17472次open/read配对，51520次样本比较无reader错误，两节点均覆盖78层 |
| 人工KV恢复 | 65521 tokens、6层、B=1/2/3，Main/Indexer全量有效内容正确，持有reader阻止覆盖，GET失败阻止继续复用 |
| 全量pytest | 尝试在e2e收集阶段因缺少`huggingface_hub`中止；未报告全量通过 |
| 真实新P/D sender及连续decode | 未验收 |

r06首次8K实测暴露空 GET 未调用新代次回调，已在交接提交中修复；r07为修复后验证。
模型测试读者采用额外 NPU→CPU 双读取比较，它会影响性能；模型样本按整page检查，不作为partial有效字节范围的证明。
人工 KV 测试使用生产 receiver 和 restore-only driver，但 metadata／地址 builder／无模型层回调由测试构造。
这两类硬件证据不能替代新 connector 的真实网络和完整模型 restore-only 验收。

源码仓库内单文件测试入口（需配置本仓库正常UT依赖）：

```bash
python -m pytest tests/ut/distributed/ascend_store/test_reuse_source.py
```

部署工作区还提供隔离CPU环境的聚焦入口；在本项目当前目录结构下，从本源码仓库执行：

```bash
python3 ../../tools/run_prefill_reuse_ut.py
```

后一入口及硬件脚本不随本源码仓库携带。部署仓库 `deploy-glm52` 中可取得：

- `docs/p-main-buffer-reuse-pd-source-hardware-validation.md`：完整实测说明。
- `artifacts/p-main-buffer-reuse/pd-source-runtime/`：请求、压缩reader日志、扫描结果及恢复审计。
- `artifacts/p-main-buffer-reuse/handoff/pd-source-hardware-review.tar.gz`：冻结的硬件证据和测试脚本包。
- `artifacts/p-main-buffer-reuse/handoff/p-main-buffer-reuse-latest.bundle`：截至交接提交的增量历史，需要已有固定基线。

不要在模型正在占用的设备上直接运行人工KV裸测试脚本；使用其容器身份和设备占用门禁入口。

## 9. 性能接入注意事项

当前先保证安全：step开始和正常shutdown会 drain 外部源；v1按整层源等待，存在进一步细化空间。
P重复恢复历史、Store传输批量、实际KV-ready位置及TP同步都需要时间线测量，不能只凭有异步线程就宣称完全掩盖。

sender首先避免不必要的全前缀重传、小批调用和额外全设备同步。
不要通过跳过ready、封口、失败检查或覆盖保护换取性能，也不要为同step发送等待Store整对象commit。
生产性能应去掉测试读者，在相同模型、chunk、并行配置、容量和缓存命中条件下比较。

## 10. 双方共同合入前的检查

- 确认本地P请求实例与D目标身份、物理视图／注册区绑定及协议版本的映射。
- 确认五个方法的线程、安全终结和失败语义，以及旧hook与新ready的去重。
- 确认目标admission和scheduler restore-only接入位置；不把当前显式入口当成自动支持。
- 共同审查 `ascend_multi_connector.py` 的改动，避免覆盖双方修改或留下两套状态。
- 提供正常和单路径失败的最小时序、内容对比、取消回收证据及启动参数。

接口名称可在共同评审后调整；调整必须同步调用方、适配器和契约测试。
本文没有授权推送、合并或切换集群服务。

## 11. TargetProvider首版 2026年11月4日

本节记录交接提交之后的TargetProvider第一阶段实现。本轮没有修改集群；实现范围与尚未完成的runtime接线见下文。

### 11.1 已实现

- [target_provider.py](vllm_ascend/distributed/kv_transfer/sparse_kv_offload/target_provider.py) 提供 `RegisteredTargetProvider` 四方法和三个属性。
- `provider_from_registered_dsa_layouts()` 消费已有 `(Indexer layouts, Main layouts)`、已注册 region、本进程 base address、显式 layer/group 映射。
  它只描述既有分配，不申请 tensor、Host pool 或第二套请求 blocks；local base address 不进入可序列化 lease。
- 注册范围、物理 page footprint／stride、完整 prompt 容量、Main Host／Indexer device、BF16 Main、模型要求的层覆盖和 writer plan 均有校验。
- `acquire_target` 要求匹配 request identity 和非空 scheduler retention token；同一身份的完全相同申请幂等，活动 blocks 不交给另一租约。
- `prepare_decode` 验证 receipt 的请求、目标分配代次及三项完整性，再调用 runtime 可见性操作；本类不自行设置 `visible=True`。
- `release_target` 仅在 runtime 同步释放成功后解除本地 block 占用；失败保留占用并允许安全重试。
- runtime callback 在状态锁外执行；对外数据使用值语义快照，避免可变 writer plan 篡改已注册身份。
- 成功释放只保留紧凑身份指纹，拒绝旧身份再次申请，迟到重复释放不影响新请求。指纹墓碑的长期回收策略仍需结合安全代次推进制定。

共享契约原样导入自 `YYangY/vllm-ascend` 的固定提交 `46f045aa0acbe7d26903ff364ea13f652ac29d08`：

- [mooncake_layerwise_contract.py](vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_contract.py)
- SHA256：`af939c9f2d1e7449d8b23315164f500fb75cd024b3439eb41527351f3d4c5a08`

这是对方维护的接口，未修改类型／枚举来适应本地实现，也没有合并她的 sender／MultiConnector 分支。

### 11.2 必须接上的真实runtime

provider构造必须传入 `TargetRuntime`，当前只有协议，没有会默认成功的生产实现：

| 操作 | 必须由真实分配所有者落实 |
| --- | --- |
| `retain(lease)` | 核验 scheduler retention 有效并持有已有物理分配；失败必须原子回滚自身部分引用 |
| `make_visible(lease, receipt)` | 为真实Host／device写入落实可见性，并将依赖接入正确的消费stream |
| `release(lease)` | 阻止新的旧请求访问，确认计算／写回引用已静止，同步解除自身引用；失败仍保留租约 |

这些操作不能用 `lambda: None` 配置到生产环境。测试使用受控替身只能证明调用和状态机，不能证明NPU可见性。
目前未自动将 provider 暴露到 AscendStoreConnector，因此导入本模块不会悄悄激活 receipt 模式。

下一步需在D初始化及请求生命周期中建立 owner-side runtime 和 scheduler facade：

1. worker在既有manager与TE完成分配／注册后生成真实catalog；保留实际owner引用、注册代次和backing身份。
2. scheduler基于自己的BlockAllocation及有效retention申请目标，不能直接访问worker内存对象。
3. 通过真实进程间通道传播纯数据lease／catalog及安全释放确认；锁、tensor、NPU事件留在各自所有者进程。
4. `prepare_decode`在正确worker／消费stream落地；远端写入完成／Host可见性证据由传输侧提供，不使用默认完成事件。
5. 两侧共享相同RequestKey、allocation／registration generation；remote DMA静止和scheduler归还blocks仍由B共同保证。

### 11.3 当前集成阻塞和范围

- 首版交接时对方 `_maybe_publish_target()` 未填 `retention_token`；receipt集成树已由 `bc51f47` 补齐字段签发及scheduler记录。
  该基线现已纳入本分支，A1/A2 已补真实 owner 核验和本地生命周期；生产跨进程查询与取消／抢占的 block 保留仍需 B/J 接线，不能只以非空字段作为完成依据。
- 契约 `TargetLease` 拒绝同一 `(backing_id, backing_generation)` 出现多个region。
  大Host池可能按TE上限分段注册，本实现会显式拒绝不能表示的catalog；不能伪造多个backing ID掩盖同一物理池。
- D基线 sparse offload明确要求BF16 Main；当前P独立1M的INT8 Main配置不能直接作为其匹配端。
  首次共同联调使用双方已核实的相同布局／精度，不自动引入量化转换。
- 当前四方法已完成本地核心实现，但真实owner runtime、跨进程安装、D receipt放行、取消及写回回收均未验收。
- 不改变已有P源接口或基线D路径；不要据此开启生产 `layerwise_receipt_mode`。

### 11.4 测试

新增 [test_target_provider.py](tests/ut/distributed/ascend_store/test_target_provider.py)，17项测试。
连同既有P相关用例，聚焦入口共139项通过，日志在部署工作区
`artifacts/p-main-buffer-reuse/ut-20261104-110837-974015.log`。
自身provider／测试的Ruff检查通过；导入契约保留原样，其两个`str, Enum`继承触发当前本地Ruff的UP042建议，未做可能改变共享枚举行为的unsafe fix。

本轮没有运行D模型或真实P/D链路，不能沿用此前P独立1M或受控读者64K结果证明TargetProvider运行时正确。


## 12. Main reuse + P MTP1 的跨请求 Store 前缀（2026-11-07）

Main reuse 与 P MTP1 现在允许通过 Mooncake block-key 路径恢复跨请求前缀。
本地 HBM prefix caching 仍须关闭：共享 Main 槽不能作为跨请求物理驻留证明。
P 的多步 MTP 或其他 draft 方法暂时维持禁用外部前缀查询。

公共完整块使用 `mtp_prefix_v1:<block_index>:<successor_chain_hash>`。
MTP 块尾消耗下一输入 token，因此使用下一完整块的链式 hash 作为保守身份，
继承 vLLM 的 cache_salt 和 hash extras，不与 target-only 公共对象混用。
只共享 prompt 内有完整后继块的完整 KV；最后一个完整块、部分块和生成区间
继续保留请求私有身份，最终生成点仍重算。比纯 Main 前缀最多少命中一块。
新请求查询、保存与恢复必须使用同一个身份函数，查询要求所有对应 rank 可读。

同一请求跨 chunk 仍优先使用成功提交的快照。公共 PUT 争用时使用请求私有
完整块回退，其他 TP rank 通过相同 instance/index 推导该回退；不等待未提交
公共 writer。跨请求 GET 后对象若被淘汰则在 attention 前失败，不借用其他
请求的私有版本。清理只处理私有 key，公共前缀继续由 Store 管理。

控制机聚焦测试 40 项及 10 个子用例通过，覆盖 successor 身份隔离、最终块
重算、生成区间私有、跨请求和跨 rank 恢复、PUT 争用与 GET 淘汰。
完整 scheduler 测试中的 6 个失败在原始 HEAD 上同样复现，另 74 项通过；
涉及 mock 缺失的 parser 及既有 receive-registration 断言。
真实 P/D 命中、MTP 继续执行、并发行为和性能仍待新镜像环境验证，不能用
本节单测结果代替运行时验收。证据写入部署仓库
`artifacts/mtp-prefix-20261107/`。

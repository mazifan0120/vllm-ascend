# receipt 逐层链路交接（connector side → target side）

> 历史接口交接记录：当前接线、MTP/图支持与2026-11-08重构状态请以
> [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)最新日期条目及部署仓库runbook§11.78–79为准。
> 下文的初版测试计数和未实现清单不代表当前状态。

2026-11-05 集成核对：本文保留远端交接记录，部分“已实现”指代码入口存在，
完整生产调用仍未闭合。针对 `fa5aec40b` 的实际缺口、集成修复和验证范围见
[PD transfer 最新审查](PD_TRANSFER_COMPLETION_REVIEW.md)。完整模型 PD 尚未验收。

本文面向负责 D 侧目标内存（target runtime）适配的开发者（下称 target 侧），记录 receipt 链路的集成现状、TargetProvider 交付状态与剩余共同接线项。
交接代码版本为 `bc51f47`，分支 `feat/layerwise-receipt`，双仓同步：`mazifan0120/vllm-ascend`（共同仓）与 `YYangY/vllm-ascend`；基线 `4234a1f1`（与 `p-main-buffer-reuse` 相同）。
接口准绳为 `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_contract.py`：类型、TargetProvider 协议与校验函数均以该文件为准。合并树上的契约为 `4b616b8` 版（store side → target side 措辞更新，类型与语义无变化）；target 侧此前 SHA 钉定的 `46f045a` 请挪至该版。

## 1. 交接范围

已实现（connector 侧）：

- 共享契约、双账本、逐层回执协议、provider 注入与启动校验、测试替身与单元测试（§2）。
- receipt 规格的跨进程下发、请求级 release 入口、发布路径缺陷修复、scheduler block retention 签发（§2）。

已交付（target 侧，`p-main-buffer-reuse` handoff §11）：

- `RegisteredTargetProvider` 四方法核心（`sparse_kv_offload/target_provider.py`）+ `provider_from_registered_dsa_layouts()` 目录绑定工厂 + 17 项测试。
- 集成树上 connector 侧 27 项与 target 侧 17 项测试均已本机通过（`5af09db`）。

待共同接线（receipt 全链剩余缺口，§6）：

- TargetRuntime 真实实现（retain / make_visible / release）与 provider 挂载；D 侧注册约束；D receipt 放行 / 取消 / 写回回收。

未包含（后续轮次）：connector 侧发送线程对五方法源接口（`open_source` 等）的接入。本链路不经过该接口；P 侧自动安装发现无 version-1 consumer 时静默跳过，无需等待本侧。

## 2. 分支与提交结构

| 提交 | 内容 |
|---|---|
| `a5581b9` | connector 侧：契约文件；双账本（ReceiptLedger / SlotLedger）；逐层回执协议（LAYER_RESULT / RANK_TERMINAL，DONE 携带版本号分流）；provider 注入三支判定；FakeTargetProvider；27 项纯 stdlib 单元测试 |
| `e26962a` | connector 侧：receipt 规格经 connector metadata 跨进程下发（worker 侧建账本）；`request_instance` 写入 kv_transfer_params；请求级 release 入口（失败保留租约并延迟释放块）；修复 `backings` 未定义 |
| `f4f534e` / `46f045a` / `4b616b8` | connector 侧：本文档及措辞更新（store side → target side） |
| `bd683f7` / `06e16d7` | target 侧：`RegisteredTargetProvider` 核心 + 目录绑定工厂 + 17 项测试；`PD_TRANSFER_HANDOFF.md` §11 |
| `5af09db` | 集成合并：契约保留 connector 侧版本；`ascend_multi_connector.py` 两侧注入逻辑共存（`_configure_reuse_source_contract` 与 `_configure_target_provider`，判据不重叠） |
| `bc51f47` | connector 侧：scheduler block retention token 签发（闭合 target 侧 §11.3 阻塞项） |

关键文件：

| 文件 | 职责 |
|---|---|
| `kv_p2p/mooncake_layerwise_contract.py` | 类型、协议、校验（接口准绳） |
| `kv_p2p/mooncake_layerwise_receipt.py` | ReceiptLedger（D 侧，按七字段身份判就绪）；SlotLedger（P 侧，槽位代次栅栏） |
| `kv_p2p/mooncake_layerwise_connector.py` | 回执协议、发布与释放路径、retention 签发 |
| `kv_p2p/fake_target_provider.py` | TargetProvider 测试替身 |
| `sparse_kv_offload/target_provider.py` | RegisteredTargetProvider（target 侧核心，已合并） |
| `ascend_multi_connector.py` | 双注入逻辑：`_configure_target_provider`（connector 侧）与 `_configure_reuse_source_contract`（P 侧） |
| `tests/ut/kv_offload/test_layerwise_receipt.py` | connector 侧 27 项，`python tests/ut/kv_offload/test_layerwise_receipt.py` 本机可复跑 |
| `tests/ut/distributed/ascend_store/test_target_provider.py` | target 侧 17 项，`python -m unittest tests.ut.distributed.ascend_store.test_target_provider`（经 `_mock_deps` 垫片）本机可复跑 |

## 3. TargetProvider：核心已交付，余下 runtime 接线

### 3.1 属性

```python
contract_version = 1
visibility_contract = "stream-dependency"
supports_layerwise_buffer_reuse = True
```

### 3.2 方法语义（合并树上已实现并验证）

| 方法 | 语义 |
|---|---|
| `acquire_target(request, allocation) -> TargetLease` | 覆盖全 prompt；容量不足 raise；不返回部分目标；活动 blocks 不交第二份租约 |
| `get_target_views(lease) -> tuple[CacheView, ...]` | 平铺返回；消费方按 `view.component` 分组 |
| `prepare_decode(lease, receipt) -> None` | 校验身份与代次及三项完整性；真实依赖由 `runtime.make_visible` 落地；本类不自行设置 visible |
| `release_target(lease) -> None` | 正常返回 = runtime 同步释放完成；失败保留租约可安全重试 |

`TargetRuntime`（retain / make_visible / release，target 侧 handoff §11.2）的真实实现是剩余工作的核心：provider 构造必须传入该 runtime，不存在默认成功的实现。

### 3.3 数据约束

TargetLease 经 connector metadata 随调度跨进程序列化，必须为纯数据（可 pickle）；运行时对象（锁、句柄、本地基地址）不得放入。

### 3.4 安装位置与启动校验

挂在 D 节点的 connector 或 worker 上、暴露 `acquire_target` 即可（扫描范围含两级对象）。注入时执行 `assert_provider_capabilities`，三支判定：

- receipt 模式未开启且无 provider：跳过（普通模式，合法）；
- receipt 模式开启且无 provider：拒绝启动；
- 多于一个 provider：拒绝启动。

当前 provider 尚未自动暴露到 connector 对象（target 侧 §11.3），挂载是共同接线的一部分。

## 4. receipt 模式门控行为

- 启动：开启 `layerwise_receipt_mode` 后无 provider 即启动失败（拒绝半套上线）。
- 放行：`visible` 是 `prepare_decode` 的产出而非前提。数据齐备后链路为：data ready → receipt(visible=False) → `prepare_decode` → 正常返回后置 visible → decode 放行。无该调用则请求不会终止。
- 释放：`release_target` 返回后同地址可能被下一请求复用重写；引用未同步清空会产生在途脏读。

## 5. 行为问题确认（已由 target 侧 handoff §11 与代码回答）

1. `prepare_decode`：真实依赖在 `runtime.make_visible` 层落地，provider 层无空壳路径——真实性由 runtime 实现保证（共同接线验收项）；
2. `acquire_target`：任何不足即 `ContractError`，无部分成功路径——与本侧"异常 = 整请求失败"假设一致；
3. `release_target`：正常返回即 runtime 同步释放完成，失败保留租约。

## 6. 剩余共同接线（receipt 全链开机清单）

1. `TargetRuntime` 真实实现：retain（核验 scheduler retention 并持有既有分配）/ make_visible（真实 Host/device 可见性 + 消费 stream 依赖）/ release（确认计算/写回静止后同步解除）——挂 D 侧既有 manager 与 TE 注册输出；
2. provider 构造与挂载：worker 在 manager/TE 完成分配注册后生成 catalog，经 `provider_from_registered_dsa_layouts()` 构造，并暴露 `acquire_target` 到 D connector/worker（本侧 D2 开箱即扫）；
3. D 侧 TE 注册约束：一 backing 一 region（TargetLease v1 表达限制）；
4. 远端写入完成 / Host 可见性证据由 connector 侧提供：已备 API 语义记录（`batch_transfer_sync_write` 忙等轮询至 COMPLETED；源读取结束已证；远端可见 ⚠️ 待首机"写完立刻读"实证；备选 notify 路径 `batchTransferAsync` + `submitTransferWithNotify`）；
5. D receipt 放行 / 取消 / 写回回收路径；
6. 首次联调配置：BF16 Main，双方同布局同精度（不引入量化转换）。
7. scheduler 侧 retention 绑定：部署的 scheduler wrapper 启动时调用一次 `bind_scheduler_retention(kv_cache_manager)`，装配 target 侧的 `SchedulerRetentionRegistry`（`sparse_kv_offload/target_retention.py`）。batch-aware wrapper 已内建（`BatchJobAwareScheduler.__init__` 为参考实现）；其他 scheduler 需一行挂点。未绑定 authority 时 receipt 准入 fail-closed（拒绝发布），不退回伪造 token。

已闭合（升级）：`BlockAllocation.retention_token` 的签发走 scheduler 权威路径——`registry.issue()` 核对活动分配后写入 token，随 `lease.allocation` 下发；`registry.retire(allocation, ack)` 的调用点位于 ReleaseAck 汇总路径（§6-5），本侧 `release_target` 不直接归还 blocks。

## 7. 部署

`layerwise_receipt_mode: true` 仅配置于 D 节点 kv_transfer extra config；P 节点不配置（无 provider，配置即启动失败）；P 节点回执行为由请求自带的 `request_instance` 驱动。生产开启前须完成 §6 全部条目（target 侧 §11.3 同此口径）。

## 8. 验收边界与已知限制

- 已验证：connector 侧 27 项 + target 侧 17 项单元测试（合并树）；ruff 0.14.0；语法编译。
- 待真机验证：`make_visible` 的 stream 依赖真实性；`release_target` 同步性；远端写入完成 / Host 可见（§6-4 ⚠️ 项）。
- v1 限制：multi-backing 场景 wire 的 `backing_id` 发空，D 侧以租约内第一个 MAIN backing 解析；`transfer_generation` 恒 0；无独立看门狗（slot 等待以 300s 为上界）；TargetLease 单 backing 单 region。

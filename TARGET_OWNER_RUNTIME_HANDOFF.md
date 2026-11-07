# A1/A2：D target owner 运行时交接

2026-11-04。当前开发与交付分支为原 `p-main-buffer-reuse`，本地位置
`deploy-glm52/src/vllm-ascend-reuse`。A1/A2 依赖已纳入的 receipt 基线 `6c233175d`；
初始隔离工作区现仅作历史验证快照。本次只实现 A1/A2；B 的 sender、receipt
协议修正、远端 DMA 静止、scheduler 生命周期和完整模型联调仍按原分工推进。
未修改 `mc.pdf` 的共享契约，也没有创建第二套生产 KV 池或请求 blocks。

## 1. 已提供的代码

下列路径以 `vllm_ascend/distributed/kv_transfer/` 为前缀。

| 文件 | 责任 |
| --- | --- |
| `utils/allocation_lifetime.py` | 实例级 ownership pin；活动 pin 阻止显式 destroy，失败可重试 |
| `utils/mooncake_transfer_engine.py` | 成功注册快照、真实 region/代次、pin 与 unregister 互斥 |
| `sparse_kv_offload/mooncake_host_pool.py` | 原 Host 分配持有；TP owner 协商同一 backing ID，地址不跨进程比较 |
| `sparse_kv_offload/target_retention.py` | scheduler 权威分配登记和 guard；核对真实 block map，不修改 block ref_cnt |
| `sparse_kv_offload/target_runtime.py` | retain、Host-visible 等待、Indexer NPU stream 依赖、compute/writeback 静止后同步 release |
| `sparse_kv_offload/target_catalog.py` | 从既有 Main Host/Indexer device tensor 和 TE 注册导出 catalog；保留本地资源引用 |
| `sparse_kv_offload/target_owner.py` | 稳定延迟绑定 facade、生产 factory、纯数据 dispatcher/client |
| `sparse_kv_offload/sparse_kv_offload_manager.py` | 按真实 cache spec 识别组件、层/group/容量，注册后绑定 owner，close 前检查活动租约 |
| `kv_p2p/mooncake_layerwise_connector.py` | 显式 receipt D 模式创建稳定 facade，注册前配置依赖、注册后绑定 catalog |
| `ascend_multi_connector.py` | 发现显式 facade，按对象 identity 去重并注入 |

普通模式不创建 target facade；新增 collective 仅在显式构造 shared target catalog 时执行。
启用功能但未提供真实 owner/retention/stream 时会拒绝初始化。

## 2. 初始化顺序：B 必须补的调用

1. Scheduler 用真实 `KVCacheManager` 创建 `SchedulerRetentionRegistry`。
   `issue(internal_request_id, allocation)` 核对实际分配并签发 token；不能在 worker
   用镜像字典或任意字符串代替权威记录。
2. Worker 保持原 sparse manager 的分配流程。拿到真实 model consumption stream、
   到 scheduler 的 `RetentionAuthority` client 后，在 KV 注册前调用：

   ```python
   connector.configure_target_owner(
       manager,
       retention=retention_authority,
       consumption_stream=model_stream,
       timeout=receive_timeout_seconds,
       te=actual_global_te,
   )
   ```

   `actual_global_te.transfer_engine` 必须就是 connector worker 使用的 engine。
   不创建另一个 engine 或第二个 KV allocator。
3. B 的注册逻辑注册真正的 Host Main 与 device Indexer，随后 connector 的
   `register_kv_caches()` 调用 manager factory 绑定 owner。当前旧 layerwise 注册尚未
   导出完整 DSA Host 注册，因此完整 receipt 链路仍需 B2/J1 接线，不能直接打开开关上线。
   HostRegister 返回的 NPU VA 在本环境需要显式 `npu:<rank>` location；TE snapshot
   保存实际成功的 location。物理 Host 身份仍取自原 shared pool。
4. MultiConnector 在注册前可以持有同一个 `DeferredTargetProvider`；未绑定时调用报错，
   不假装 ready。Scheduler 的 `TargetProviderClient` 由 B 用真实 transport 通过
   `bind_target_owner_transport()` 接通。transport 必须执行 owner dispatcher 并有有界等待。
5. 每个 TP rank 在同一初始化阶段绑定 catalog；只在 owner rank 产生 shared Host UUID，
   经现有 TP CPU group 协商。所有 rank 的 layout 和 registration 成功后，B 才能完成
   admission/worker-ready 握手并通知 P。

`manager.bind_registered_target_owner()` 也可作为直接集成入口；它使用既有
`provider_from_registered_dsa_layouts()`，不是平行实现另一套 provider。
支持边界：eager、PP/CP/DCP=1、无 speculative/MTP，BF16 Main，明确的 Indexer 布局。

## 3. 请求生命周期与真实完成依赖

- `acquire_target(request, allocation)`：core provider 校验完整分配；runtime 向权威
  retention 取得 guard，并 pin 原 Host pool 和 TE 注册。失败不发布租约，回滚失败会
  quarantine 新 acquisition，直到 owner 本地 `retry_failed_retains()` 清理重试成功。
- B 收到真正的目标写完成时，在 **D owner 本地** 调用：

  ```python
  facade.publish_received(lease, remote_host_visible_fence, native_indexer_write_event)
  ```

  Host fence 的 `wait(timeout)` 只能在远端写对 Host 可见后返回 `True`，不能拿
  source-read-done 或预先 set 的 event 代替。Indexer 必须是实际记录过的 NPU event；
  Python event/tensor/owner 不能跨 scheduler-worker 消息传递。事件在租约存续期间
  不得重录并用于另一请求。
- `prepare_decode(lease, receipt)`：精确核对请求/目标代次及完整 receipt，等待 Host
  可见性，在真实 consumption stream 安装 Indexer `wait_event`。缺依赖、超时或失败不放行。
  provider 的 prepare 失败是终态，不能给同一 transfer generation 更换依赖后重试冒充成功。
- B 在真实 worker forward/使用目标的范围中包 `with facade.use(lease): ...`。
  该上下文绑定正确 NPU/stream，记录 consumption 和 manager 已订阅的 compute、
  fused-plan、writeback streams 完成事件。B 必须覆盖所有相关使用；运行时不会猜测模型调用。
- B 先停止该租约后续任务提交，再同步 `release_target(lease)`。仍有 use、prepare、
  并发 release 或未完成设备事件时拒绝/有界失败，保留状态及资源以供安全重试。
  最终事件静止后才释放 owner refs 和 retention guard；没有 device-global synchronize。
- **本地 release 不代表允许归还 blocks。** B 仍必须汇总所有 rank、所有远端 DMA
  和取消任务的静止，产生精确安全 `ReleaseAck`，再在 scheduler 串行分配路径执行
  `retire(allocation, ack)`，最后调用现有 free/preempt 路径。
  取消时不能用等待超时伪造 safe ack。

registry 的 guard 不调用 `BlockPool.touch()`，防止人为提高 ref_cnt 导致 decode COW。
它不是全局 free hook；B4 仍需把正常结束、取消、抢占路径接入上述门禁。

## 4. catalog 与进程边界

catalog 只含 contract version、rank/TP、region、entry/shape/stride、层/group、容量、
backing/registration identity、代次和 writer plan。请求 lease 额外含实际 block IDs、
retention token 和请求/transfer identity。

本地基址、tensor、pool/TE pins、NPU event、锁和 runtime 留在 owner。
`TargetOwnerCall/Reply` 与 dispatcher/client 的允许字段检查阻止活对象进入 IPC；
实际 transport 由 B 的 worker 通道提供。`publish_received()`、`use()` 不开放为远程 wire 操作。
registration/owner 不允许跨代重绑，销毁后必须重新建立 owner 和 facade。

Receipt v1 的一 backing 多注册 region 不能被表达，catalog 显式拒绝；不通过伪造
backing ID 规避。大分配若必须拆 MR，应先扩展共同契约与 B 数据面，再启用。

## 5. 验证与未覆盖范围

当前 CPU 证据在部署仓库 `artifacts/p-main-buffer-reuse/target-owner-runtime/`：

- 40 项 lifetime/runtime/provider 测试通过，包含真实 CPU storage 几何和真正父子进程 Pipe。
- 原 P 聚焦回归 139 项、receipt 账本 27 项通过。
- TE/HostPool 回归 16 项及 4 个子用例通过；旧测试中 7 个失败在冻结基线同样复现，
  已将其断言更新为当前 legacy mmap/HostRegister 和 wildcard fallback 行为。

这些 CPU 用例使用测试传输/backend 和 CPU completion driver，不能替代 NPU 证据。
最终源码快照的 TP1/TP2 原生验证通过，包含实际 connector A2 方法注入/注册顺序、
真实 vLLM scheduler 分配/追加/retire、storage 加固、原生 event 和延迟 writeback。
原生内容检查使用同一 Host backing 的 CPU 映射；Indexer 的慢完成逻辑另有 CPU 用例。
完整模型的初始化/数据面/forward 接线不在该合成验证的通过范围内。原环境恢复审计通过：四引擎健康/空闲，原 ID、镜像、脚本和源码 manifest 匹配，r14 停止。
验证范围和恢复记录见 [验证报告](TARGET_OWNER_RUNTIME_VALIDATION.md)；原始日志仍保留在部署仓库 artifacts。
尚未验证完整模型的 PD 写入、稀疏 decode/写回、远端取消/DMA 回收；64K 数值定位保持停止。

恢复审计曾误用 `max_tokens=16` 的普通 P 聊天请求，触发未安装本轮代码的旧 r05
Store prefix 错误并使 P 引擎退出。失败日志保留；原 P 容器按原 ID/镜像/脚本恢复后，
P0/P1 的 `max_tokens=1` P-only 检查均返回 `4`。这不代表旧 P 多 token decode 已通过。

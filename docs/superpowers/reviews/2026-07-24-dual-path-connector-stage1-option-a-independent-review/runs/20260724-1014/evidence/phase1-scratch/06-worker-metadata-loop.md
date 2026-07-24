# Phase 1 证据 06：Worker 进程侧 connector 调用链（metadata loop）

> 范围：upstream `/Users/leqi/Documents/Code/vllm`（HEAD 8df14cfc，v0.23.1rc0-1050，**参考版本存在小幅偏差**：vllm-ascend 官方配套 v0.23.0）与 vllm-ascend `/Users/leqi/Documents/Code/vllm-ascend`（HEAD 0ec11a47，branch dev/dualpath）。
> 本文件只记录当前源码事实，不评审设计。除标注外，结论均为 `[当前源码已确认]`。
> 行号：upstream 文件以 `u:` 前缀、ascend 文件以 `a:` 前缀标注于各节标题；正文直接写 `文件:行号`。

---

## 0. 进程模型与 connector 实例位置

- Scheduler 运行在 **EngineCore 进程**；`EngineCore.step()` 中 `scheduler_output = self.scheduler.schedule(...)` → `self.model_executor.execute_model(scheduler_output, non_block=True)` → `model_output = future.result()` 或 `self.model_executor.sample_tokens(grammar_output)` → `self.scheduler.update_from_output(scheduler_output, model_output)`。证据：`vllm/v1/engine/core.py:499-515`。
- Worker 是独立进程（MultiprocExecutor）。每个 **Worker 进程**持有全局 connector 单例 `_KV_CONNECTOR_AGENT`（module-level global），由 `ensure_kv_transfer_initialized(vllm_config, kv_cache_config)` 以 `KVConnectorRole.WORKER` 创建。证据：`vllm/distributed/kv_transfer/kv_transfer_state.py:25-28, 72-94`。
- ascend 侧初始化调用点：`NPUWorker.initialize_from_config` → `ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)`（`vllm_ascend/worker/worker.py:894-896`；upstream 对应 `vllm/v1/worker/gpu_worker.py:729`）。Worker 进程关闭时 `NPUWorker.shutdown` → `ensure_kv_transfer_shutdown()`（`vllm_ascend/worker/worker.py:376-378`）。
- Scheduler 进程侧另有独立 connector 实例（role=SCHEDULER），`EngineCore.__init__` 用它初始化输出聚合器：`self.model_executor.init_kv_output_aggregator(self.scheduler.connector)`（`vllm/v1/engine/core.py:164` → `vllm/v1/executor/abstract.py:280-282`）。

## 1. `bind_connector_metadata()`：调用点与 metadata 传输路径

### 1.1 Scheduler → Worker 传输路径 `[当前源码已确认]`

1. Scheduler 构建输出：`SchedulerOutput(finished_req_ids=self.finished_req_ids, ...)`（`vllm/v1/core/sched/scheduler.py:1142-1160`），随后 `meta = self._build_kv_connector_meta(self.connector, scheduler_output)`（即 `connector.build_connector_meta(scheduler_output)`，`scheduler.py:1186-1189`），赋到 `scheduler_output.kv_connector_metadata`（`scheduler.py:1166-1168`）。
2. 跨进程传输：`MultiprocExecutor.execute_model` → `collective_rpc("execute_model", args=(scheduler_output,), unique_reply_rank=self.output_rank, kv_output_aggregator=self.kv_output_aggregator)`（`vllm/v1/executor/multiproc_executor.py:310-320`）→ `self.rpc_broadcast_mq.enqueue((send_method, args, kwargs, output_rank))`（`multiproc_executor.py:377`；MessageQueue 创建于 `:151`）。即 **`SchedulerOutput`（含 `kv_connector_metadata`）整体经 ZMQ MessageQueue 广播到所有 worker**，无单独 metadata 通道。
3. Worker 侧接收：`WorkerProc.worker_busy_loop` `method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue(...)`（`multiproc_executor.py:986-990`）→ 调用 `NPUWorker.execute_model(scheduler_output)`（`vllm_ascend/worker/worker.py:599-634`）→ `self.model_runner.execute_model(...)`。
4. ascend 特有：`NPUModelRunner.execute_model` 在若干条件下会 `replace()`/`deepcopy()` `scheduler_output`（ngram_gpu：`vllm_ascend/worker/model_runner_v1.py:1922-1934`；async scheduling + spec 或 PCP+MM：`:1944-1956`；PP 非首 rank：`:1958-1962`）。这些复制发生在 bind 之前，不改变 `kv_connector_metadata` 字段语义，但意味着 worker 侧后续读到的可能是副本。`[当前源码已确认]`

### 1.2 Worker 进程内 bind 调用点 `[当前源码已确认]`

- 唯一调用点：`KVConnectorModelRunnerMixin._get_kv_connector_output`（contextmanager），`vllm/v1/worker/kv_connector_model_runner_mixin.py:78-89`：

  ```python
  kv_connector = get_kv_transfer_group()
  assert scheduler_output.kv_connector_metadata is not None
  kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)   # :89
  kv_connector.start_load_kv(get_forward_context())                              # :95
  ```

- 进入该 contextmanager 的两条路径：
  - `maybe_get_kv_connector_output(scheduler_output, defer_finalize=...)`（`kv_connector_model_runner_mixin.py:51-61`），在 `NPUModelRunner.execute_model` 的 forward `with` 栈中进入：`vllm_ascend/worker/model_runner_v1.py:2241-2263`。注意 `with` 顺序：`set_ascend_forward_context(...)` 先 enter（`:2243`），`maybe_get_kv_connector_output` 后 enter（`:2258`）；退出时逆序——**先执行 connector 的 wait_for_save/get_finished/clear，再退出 ascend forward context**（forward context 在 get_finished 期间仍 active）。
  - `kv_connector_no_forward`（`kv_connector_model_runner_mixin.py:36-48`）：空 step（无 token 被调度）时，包 `set_forward_context(None, vllm_config)` 后进入同一 contextmanager（`wait_for_save=False`）。ascend 调用点：`model_runner_v1.py:2022` 与 `:2037`（后者是 ascend 额外加的 `tokens` 为空/全 0 分支，见 §6）。
- base 实现：`bind_connector_metadata` 仅 `self._connector_metadata = connector_metadata`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:211-221`）；`clear_connector_metadata` 置 None（`base.py:223-229`）；`has_connector_metadata()` 判读（`:243-249`）。MultiConnector 覆写：把 `MultiKVConnectorMetadata` 按序分发给各子 connector 并合并 `extra_async_saves`（`vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:260-266`）。
- 运行位置：Worker 进程**主线程**（execute_model RPC 处理线程），非后台线程。`[当前源码已确认]`
- 时序性质：在 `execute_model` 的 forward 之前、由框架代码顺序保证（先 bind 后 start_load_kv，再 yield 给 model forward）。`[当前源码已确认]`

## 2. `get_finished(finished_req_ids)`：调用点、入参语义、返回值去向

### 2.1 调用点 `[当前源码已确认]`

- `kv_connector_model_runner_mixin.py:102-104`（`_get_kv_connector_output` 的 `finally` 块）：

  ```python
  output.finished_sending, output.finished_recving = (
      kv_connector.get_finished(scheduler_output.finished_req_ids))
  ```

- **每个 step 都会调用**（包括 `defer_finalize=True` 的 spec-decode 场景——`finally` 中仅 `wait_for_save`（`:99-100`）与 `clear_connector_metadata`（`:111-112`）受 defer 抑制，`get_finished`/`get_block_ids_with_load_errors`/stats/events/`build_connector_worker_meta` 都无条件执行）；空 step 经 `kv_connector_no_forward` 也会调用（`wait_for_save=False` 但 get_finished 照常）。
- 运行位置：Worker 进程主线程，forward 结束后、采样结果返回前（spec decode 时在 target model forward 结束处，draft model 之前）。`[当前源码已确认]`

### 2.2 `finished_req_ids` 入参包含什么 `[当前源码已确认]`

- 来源：`scheduler_output.finished_req_ids = self.finished_req_ids`（`scheduler.py:1155`），注释（`scheduler.py:1151-1154`）："It contains the request IDs that are finished in between the previous and the current steps."
- 填充点：`Scheduler._free_request` 无条件 `self.finished_req_ids.add(request_id)`（`scheduler.py:2175`）。`_free_request` 被两条路径调用：
  - 正常完成（`update_from_output` 处理 stopped 请求，`scheduler.py:1766`）；
  - 中止/取消（`finish_requests` → `scheduler.py:2152`）。
- **包含 delayed-free 等待中的请求**：`_free_request` 中 `delay_free_blocks` 只推迟 `_free_blocks`（`scheduler.py:2179-2181`），不影响加入 `finished_req_ids`。所以集合 = "上一 step 与本 step 之间所有结束（含 abort、含延迟释放等待中）的请求"。
- 每 step 重置：`_update_after_schedule` 末尾 `self.finished_req_ids = set()`（`scheduler.py:1262`，注释说明故意 rebind 而非 `clear()`，避免影响已发出的 scheduler_output）。
- 结果：worker 侧 connector 经 `get_finished(finished_req_ids)` 收到的是"调度器视角已结束请求"的全集快照（滞后约一个 step；async scheduling 下滞后由同一机制天然覆盖）。connector 契约（`base.py:357-373`）：返回的 finished saves/sends id 必须属于"本次或之前某次调用提供的集合"。

### 2.3 返回值如何进入 KVConnectorOutput 并回到 Scheduler `[当前源码已确认]`

1. Worker 进程内：直接写进 contextmanager yield 出去的 `KVConnectorOutput`（字段定义 `vllm/v1/outputs.py:196-221`），该对象最终挂到 `ModelRunnerOutput.kv_connector_output`（`outputs.py:262`；ascend 挂接点 `model_runner_v1.py:2289-2300/2342/2374/2482`，见 §6）。
2. 跨 worker 聚合：`KVOutputAggregator.aggregate(outputs, output_rank)`（`vllm/distributed/kv_transfer/kv_connector/utils.py:70-175`）。聚合规则：对 finished_sending/recving 做**计数**——每个 req_id 需 `_expected_finished_count` 个 worker 都报告后才进入聚合集合（`utils.py:78-90`）；`_expected_finished_count` 初始为 `connector.get_finished_count() or world_size`（`utils.py:66-68`；`MultiConnector.get_finished_count` 当前恒返回 None → 用 world_size，`multi_connector.py:355-358`；ascend 唯一覆写是 `ucm_connector.py:284`），且可被 worker 输出中的 `expected_finished_count>0` 动态更新（`utils.py:103-114`）。聚合结果写回 `outputs[output_rank].kv_connector_output`（`utils.py:162-173`）。
3. 聚合触发位置：executor 收集全部 worker 响应时——`MultiprocExecutor.collective_rpc` 在 `kv_output_aggregator` 非 None 时 `output_rank=None`（收所有 response_mq）并以 `partial(kv_output_aggregator.aggregate, output_rank=unique_reply_rank or 0)` 作为结果处理函数（`multiproc_executor.py:343-371`）。即**聚合在 EngineCore 进程、收到全部 worker 回复后同步发生**；`execute_model` 与 `sample_tokens` 两个 RPC 各自独立聚合一次（`multiproc_executor.py:310-332`）。
4. Scheduler 消费：`update_from_output` 取出 `kv_connector_output`（`scheduler.py:1562`）→ `_update_from_kv_xfer_finished`（`scheduler.py:1838-1839` → `:2559-2586`）：
   - `finished_recving`：若请求处于 `WAITING_FOR_REMOTE_KVS` → 加入 `finished_recving_kv_req_ids`（下一步可调度，`scheduler.py:2574-2579`）；否则（已 finished）直接 `_free_blocks`（`:2580-2582`）。
   - `finished_sending`：`self._free_blocks(self.requests[req_id])`（`:2583-2586`）——**delayed free 的释放点**。
   - 前置还调用 `self.connector.update_connector_output(kv_connector_output)`（`:2570-2571`）。

## 3. `build_connector_worker_meta()`：调用点与回流/聚合

- 调用点：`kv_connector_model_runner_mixin.py:109` —— `output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()`，与 get_finished 同处 `finally` 块，**每 step 调用**（含 defer_finalize 场景）。base 默认返回 None（`base.py:429-437`）；MultiConnector 收集各子 connector 的非 None meta 包成 `MultiKVConnectorWorkerMetadata`（`multi_connector.py:360-370`）。
- 回流路径：随 `ModelRunnerOutput.kv_connector_output` 经 response MessageQueue 回 EngineCore 进程（同 §1.1 的反向）。
- 聚合时机：`KVOutputAggregator.aggregate` 内，跨 worker 调 `KVConnectorWorkerMetadata.aggregate(other)`（`utils.py:135-144`；抽象定义 `base.py:150-168`，语义注释明确："all metadata objects returned by workers will be aggregated using the `aggregate` method ... before being passed to the Scheduler KVConnector"）。实现例：`SimpleCPUOffloadWorkerMetadata.aggregate`（`vllm/v1/simple_kv_offload/metadata.py:53-55`）、`MultiKVConnectorWorkerMetadata.aggregate`（逐子项聚合，`multi_connector.py:55-66`）。
- Scheduler 侧消费：不直接在 scheduler 主循环读取，而是经 `connector.update_connector_output(kv_connector_output)`（`scheduler.py:2570-2571`）由 connector 自取；MultiConnector 在 `update_connector_output` 中把 per-child meta 临时替换后逐个分发（`multi_connector.py:431-450`）。消费实例：`vllm/v1/simple_kv_offload/manager.py:672-681`（"Store completions arrive via kv_connector_worker_meta"）。
- 运行位置归纳：构建在 Worker 主线程（forward 末尾）；聚合在 EngineCore 进程（executor RPC 收包线程上下文）；消费在 EngineCore 进程 scheduler 线程。`[当前源码已确认]`

## 4. `get_block_ids_with_load_errors()`：调用点与 invalid blocks 回流

- 调用点：`kv_connector_model_runner_mixin.py:105` —— `output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()`，每 step 在 forward 末尾调用（Worker 主线程）。base 默认返回空集（`base.py:375-393`），契约注释：异步 load 的失败 block 最迟必须在"该请求被 get_finished 报告完成的同一 pass"之前/之中上报。
- 跨 worker 聚合：**并集**（`utils.py:97, 159, 171`），无计数门槛。
- Scheduler 消费：`update_from_output` 开头 `if kv_connector_output and kv_connector_output.invalid_block_ids:` → `_handle_invalid_blocks(invalid_block_ids, num_scheduled_tokens)`（`scheduler.py:1578-1586` → `:2691-2723`）→ `_update_requests_with_invalid_blocks`（`:2588` 起）：对 running/waiting 请求把失效 block 标记 invalid、回退 `num_computed_tokens` 触发重算、evict 对应 block。`[当前源码已确认]`
- ascend 实现：
  - `MooncakeConnector.get_block_ids_with_load_errors` → worker 侧 `kv_recv_thread.get_and_clear_invalid_block_ids()`（仅 kv_consumer；`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py:1511-1514, 2437-2440`）。
  - `MooncakeLayerwiseConnector`：worker `get_finished` 内对 `failed_recving` 请求把 `meta.local_block_ids` 并入 `self._invalid_block_ids`（`mooncake_layerwise_connector.py:1411-1419`），`get_block_ids_with_load_errors` 返回并清空（`:1430-1434`）。
  - 注意：失败的 recv 仍通过 `done_recving/failed_recving` 让请求完成上报（layerwise `:1420-1428` 把 failed 与 done 合并清理），符合 base 契约"即使失败也必须经 get_finished 报告"。

## 5. forward 过程中 `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` 的位置与顺序

### 5.1 框架层（upstream，ascend 原样继承）`[当前源码已确认]`

顺序（Worker 主线程，单 step）：

1. `bind_connector_metadata`（`kv_connector_model_runner_mixin.py:89`）
2. `start_load_kv(get_forward_context())`（`:95`）——forward 前一次性调用，非按层/非按 request 循环；"Background KV cache transfers happen here"（注释 `:91-94`）。
3. 模型 forward：每层 attention 触发一次 `wait_for_layer_load(layer_name)`（层计算前）与一次 `save_kv_layer(layer_name, kv_cache, attn_metadata)`（层计算后）——**按层调用**（对整个 batch，不按 request 拆分）。触发机制因路径而异，见 5.2/5.3。
4. forward 退出（contextmanager `finally`）：`wait_for_save()`（`:99-100`，`wait_for_save and not defer_finalize` 时）→ `get_finished` → `get_block_ids_with_load_errors` → stats/events/worker_meta → `clear_connector_metadata()`（`:111-112`，非 defer 时）。
5. spec decode（`defer_finalize=True`，upstream `gpu_model_runner.py:4354`；ascend `model_runner_v1.py:2240,2260-2262`）：`wait_for_save` 与 `clear_connector_metadata` 推迟到 draft model 之后的 `finalize_kv_connector()`（`kv_connector_model_runner_mixin.py:64-72`；ascend 调用点 `model_runner_v1.py:2470-2474`，在 `sample_tokens` 的 `draft_token` 记录块内；upstream 对应 `gpu_model_runner.py:4687`）。即 defer 期间 draft model forward 仍在 metadata bound 状态下运行。

### 5.2 按层钩子：upstream 机制 `[当前源码已确认]`

- 装饰器 `@maybe_transfer_kv_layer`（`vllm/model_executor/layers/attention/kv_transfer_utils.py:15-61`）包在 `unified_attention_with_output`（`vllm/model_executor/layers/attention/attention.py:811-818`）与 `unified_mla_attention_with_output`（`mla_attention.py:1080-1082`）上。wrapper 逻辑（`kv_transfer_utils.py:38-59`）：无 connector 或无 metadata 直接放行（`:39-48`）；否则 `:51 connector.wait_for_layer_load(layer_name)` → 执行层计算 → `:57 connector.save_kv_layer(layer_name, kv_cache, attn_metadata)`。
- 适用条件：`Attention/MLAAttention.forward` 走 custom-op 分支，即 `use_direct_call = not current_platform.opaque_attention_op()` 为 False（`attention.py:438`；MLA 直接调用分支 `mla_attention.py:568-603` 无此钩子）。

### 5.3 按层钩子：ascend 实际路径（与 upstream 的关键差异）`[当前源码已确认]`

- `NPUPlatform.opaque_attention_op()` 返回 **True**（`vllm_ascend/platform.py:877-878`）→ upstream `Attention`（非 MLA 模型）在 ascend 上走 5.2 的装饰器 custom-op 路径，`wait/save` 由 upstream 装饰器触发；`AscendAttentionImpl` 自身**不**调 wait/save，仅在写完 KV cache 后调 `notify_kv_cache_written()`（`vllm_ascend/attention/attention_v1.py:1453, 1983`）。
- **ascend MLA/SFA/DSA 路径绕过装饰器**：模型用 `AscendMultiHeadLatentAttention`（`vllm_ascend/ops/mla.py:67-176`），其 forward 调 ascend 自注册的 `torch.ops.vllm.mla_forward`（`ops/mla.py:174, 207-213`，**未**包 `maybe_transfer_kv_layer`），直接调 `self.mla_attn.impl.forward(...)`（`ops/mla.py:192-194`）。因此 wait/save 由 impl 内部的 ascend helper 触发：
  - `wait_for_kv_layer_from_connector(layer_name)`（`vllm_ascend/attention/utils.py:428-439`）：`AscendMLAImpl._mla_preprocess` 中 **仅 `has_prefill` 时**调用（`vllm_ascend/attention/mla_v1.py:1683-1684`）；dsa_v1.py:1702、sfa_v1.py:1611/1643/1672、dsa_cp.py:1200 同模式。→ **decode-only step 在 MLA 路径上不产生 `wait_for_layer_load` 调用**。
  - `maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))`（`utils.py:442-456`）：`AscendMLAImpl.forward` 末尾（`mla_v1.py:1807`）；dsa_v1.py:1736、sfa_v1.py:1934、dsa_cp.py:1271 同模式。
  - `notify_kv_cache_written(layer_name)`（`utils.py:459-476`）：ascend 独有钩子，KV cache 写完后无条件调用，connector 实现了 `on_kv_cache_written` 才生效；调用点 mla_v1.py:1695、dsa_v1.py:1950/2089/2383、attention_v1.py:1453/1983、attention_cp.py:862、mla_cp.py:457、sfa_v1.py:1873。
  - GDN（线性注意力）：`vllm_ascend/ops/gdn.py:137` 与 `_310p/ops/fla/gdn_310.py:429` 调 `maybe_save_kv_layer_to_connector("", [])`——**空 layer_name、空 cache**。
- 语义差异提示：ascend MLA 的 wait 在 `_mla_preprocess` 中（q/kv 投影之后、attention kernel 之前），而非 upstream 装饰器的"层计算入口处"；且以 `has_prefill` 为条件。`[当前源码已确认]`

### 5.4 ascend connector 的 wait_for_save 特例 `[当前源码已确认]`

- AscendStore pool connector：`pool_worker.wait_for_save` 末尾 `self.kv_send_thread.request_queue.join()`（`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py:1390-1394`）——在 Worker 主线程**同步阻塞**直到发送线程队列清空（注释说明动机：保证 store 对后续相同 prompt 可见）。
- `CPUOffloadingConnector.save_kv_layer/wait_for_save` 是 no-op（`vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py:104-110`）；其 save 由专属后台线程 `_save_listener`（`:373-396`）执行，任务在 `bind_connector_metadata` 时按 metadata 中 `finished_req_ids` 入队（`:277-288`）——**save 触发点是 bind 而非按层钩子**。

## 6. vllm-ascend model_runner / worker 对这条链路的定制点（vs upstream）

`NPUModelRunner(GPUModelRunner)`（`vllm_ascend/worker/model_runner_v1.py:267`），mixin 方法全部继承未覆写。差异点 `[当前源码已确认]`：

1. **额外的空-batch 分支**：`model_runner_v1.py:2033-2037`（`total_num_scheduled_tokens<=0 or not tokens or sum(tokens)==0` → `kv_connector_no_forward`）。upstream `gpu_model_runner.py:4149-4165` 只有 `not num_scheduled_tokens` 一个分支。ascend 注释未说明动机。（参考版本存在小幅偏差）
2. **PP 非末 rank 的输出挂接**：ascend 同时设置 `hidden_states.kv_connector_output = kv_connector_output` 与 `self.kv_connector_output`（`model_runner_v1.py:2289-2290`）；upstream 只设置后者（`gpu_model_runner.py:4399-4403`）。配套地，`NPUWorker.execute_model` 从返回的 `IntermediateTensors.kv_connector_output` 读取并做透传（`vllm_ascend/worker/worker.py:652-662`：无 finished 集合时回 `EMPTY_MODEL_RUNNER_OUTPUT`，否则 copy EMPTY 并挂上 kv_connector_output）。
3. **`sample_tokens` 透传逻辑自写**：`model_runner_v1.py:2354-2375`（`kv_connector_output` 为 None → return None；`is_empty()` → `EMPTY_MODEL_RUNNER_OUTPUT`；否则 copy EMPTY 挂载）。upstream 用 `ModelRunnerOutput.with_kv_conn_output_only(...)` 一行（`gpu_model_runner.py:4486-4494`；该 helper 定义 `vllm/v1/outputs.py:284-294`，None/empty 都返回 EMPTY）。差异：execute_model_state 为 None 且 kv_connector_output 为 None 时 ascend 返回 `None`，upstream 返回 EMPTY。`KVConnectorOutput.is_empty()` 判定含 worker_meta/stats/events/invalid_block_ids（`outputs.py:213-221`）。
4. **pooling 分支**：`model_runner_v1.py:2295-2302` 把 `kv_connector_output` 传给 `self._pool(...)` 并显式 `output.kv_connector_output = kv_connector_output`；upstream `_pool(...)` 内处理（`gpu_model_runner.py:3397,3434`）。
5. **handle_preemptions 前置调用**：ascend `model_runner_v1.py:1964-1969` 与 upstream `gpu_model_runner.py:4128-4131` 一致（在 `_update_states` 之前，注释说明为防 preemption store 与新分配清零冲突）——**非差异**，但它是 worker 链路的一部分：每 step 在 bind 之前先调 `get_kv_transfer_group().handle_preemptions(kv_connector_metadata)`。
6. **`finalize_kv_connector` 时机**：ascend 在 `sample_tokens` 的 `draft_token` 记录块内、仅 `speculative_config is not None` 时调用（`model_runner_v1.py:2470-2474`）；upstream 同语义（`gpu_model_runner.py:4687`）。非差异。
7. **forward context**：用 `set_ascend_forward_context`（`model_runner_v1.py:2243-2257`）替代 `set_forward_context`；mixin 内 `start_load_kv(get_forward_context())` 依赖同一全局 forward context，`[源码可达，尚未运行验证]` ascend 版本保持兼容（从 §5.3 钩子依赖 `get_forward_context()` 且现有 connector 工作可推断）。
8. **connector 注册**：`register_kv_caches` 在 `model_runner_v1.py:3824-3825`（initialize_kv_cache 路径）。Mooncake 相关注释/对齐逻辑见 `:3953-3988`（block 对齐、注册 buffer 合并，KV cache 布局层面，不改变本链路调用关系）。
9. **scheduler 侧 patch（条件启用）**：`vllm_ascend/patch/platform/patch_balance_schedule.py`（`--additional-config enable_balance_scheduling`，注册说明 `vllm_ascend/patch/__init__.py:70-79`）重写 `Scheduler.schedule`，其中 connector 调用保持同构：`get_num_new_matched_tokens`（`:348-349`）、`update_state_after_alloc`（`:473-474`）、`build_connector_meta`（`:601-603`）。默认未启用时不影响 upstream 路径。`[当前源码已确认（启用条件为配置）]`
10. **connector 实现替换**：ascend 自带 connector 家族（`vllm_ascend/distributed/kv_transfer/`）：`kv_p2p/mooncake_connector.py`、`kv_p2p/mooncake_layerwise_connector.py`、`kv_p2p/mooncake_hybrid_connector.py`、`kv_pool/ascend_store/`（AscendStoreConnector+pool_worker）、`kv_pool/cpu_offload/`、`kv_pool/recompute_cpu_offload/`、`kv_pool/ucm_connector.py`。均按 scheduler/worker 双角色拆分实现 `KVConnectorBase_V1`，worker 侧传输在**后台守护线程**执行（如 mooncake 的 `kv_send_thread`/`kv_recv_thread`，`mooncake_connector.py:2404-2413`；cpu_offload 的 `_save_listener`，`cpu_offload_connector.py:245-246`），`get_finished` 只做非阻塞轮询（`mooncake_connector.py:2415-2435`；`cpu_offload_connector.py:332-339` 用 `get_nowait`）。
11. **cpu_offload 的 get_finished 内含 TP 集合通信**：tp_rank0 收其他 rank 的 done 集合做计数（`cpu_offload_connector.py:344-366`）——即 `get_finished` 里发生 `tp_group.recv_object/send_object`，运行在 Worker 主线程的 step 末尾。
12. **mooncake_layerwise 的 finished_sending 恒空**：worker `get_finished` 返回 `(set(), done_recving)`（`mooncake_layerwise_connector.py:1428`）——该 connector worker 侧从不上报 finished_sending（layerwise 为 consumer/decode 侧语义）；`request_map`/`_recving_metadata` 在完成或失败时清理（`:1420-1423`），并合并 `virtual_request`（`:1408-1409`）。
13. **MultiConnector 无 ascend 修改**：upstream `multi_connector.py` 原样使用；ascend 仅在 `patch_mamba_config.py:21-26` 读取其 extra_config 判断子 connector 是否含 AscendStoreConnector。

## 7. 请求完成/取消时 worker 侧状态清理；delayed free 的 finished_sending 如何被等待

- **worker 侧无 per-request 取消钩子**：取消/完成都只经 `get_finished(finished_req_ids)` 入参通知（§2.2：abort 与正常完成同集合、delayed-free 中的请求也在其中）。`[当前源码已确认]`
- **每 step 的 metadata 清理**：`clear_connector_metadata()`（`kv_connector_model_runner_mixin.py:111-112`）把 `_connector_metadata` 置 None——metadata 生命周期恰为一个 step；spec decode 时推迟到 draft 后的 `finalize_kv_connector`（`:64-72`）。
- **connector 内部 per-request 状态**：各实现自行清理，时机在 `get_finished` 轮询到完成时：cpu_offload `del self.requests[id]`（`cpu_offload_connector.py:340-341`）；mooncake_layerwise 清 `request_map`/`_recving_metadata`（`:1420-1423`）；MultiConnector 的 `_extra_async_saves` 在子 connector 报 finished_sending 时递减归零删除（`multi_connector.py:325-334`，scheduler→worker 经 metadata 传播的 `extra_async_saves` 在 bind 时合并，`:262-263`）。
- **delayed free 的等待方式**：**worker 侧不阻塞等待**。传输在 connector 后台线程推进，每个 step 末尾 `get_finished` 非阻塞轮询一次；完成的 req_id 经聚合（需全部 worker 报告，§2.3）回到 scheduler，由 `_update_from_kv_xfer_finished` 对 `finished_sending` 调 `_free_blocks` 释放延迟持有的块（`scheduler.py:2583-2586`）。延迟期间块由 scheduler 侧持有（`scheduler.py:2179-2181` 跳过 `_free_blocks`）。契约见 `base.py:366-372`。
- 例外性的同步点：`wait_for_save` 在 forward 末尾调用且可阻塞（ascend_store 的 `request_queue.join()`，`pool_worker.py:1390-1394`）——这是"save 完成"的等待，不是 finished_sending（请求级完成通知）的等待。
- connector 整体析构：`NPUWorker.shutdown` → `ensure_kv_transfer_shutdown()`（`vllm_ascend/worker/worker.py:376-378`）；`MultiConnector.shutdown` 逐个关停子 connector（`multi_connector.py:273-284`）。

## 8. 端到端时序汇总（单 step，Worker 视角）`[当前源码已确认]`

```
EngineCore 进程: schedule() ─ build_connector_meta ─ SchedulerOutput(+kv_connector_metadata)
      │  ZMQ broadcast (execute_model RPC)                [core.py:499-500; multiproc_executor.py:310-320,377]
Worker 进程主线程:
  NPUWorker.execute_model                       [worker.py:599]
   └─ NPUModelRunner.execute_model              [model_runner_v1.py:1897]
       ├─ handle_preemptions(metadata)          [:1964-1969]（bind 之前）
       ├─ _update_states / _prepare_inputs / _build_attention_metadata ...
       └─ with set_ascend_forward_context(...), maybe_get_kv_connector_output(...):
            ├─ bind_connector_metadata          [kv_connector_model_runner_mixin.py:89]
            ├─ start_load_kv                    [:95]
            ├─ model forward：逐层 wait_for_layer_load / save_kv_layer
            │    （非 MLA：upstream 装饰器；MLA/SFA/DSA：ascend impl 内 helper，wait 仅 has_prefill）
            ├─ (exit) wait_for_save             [:99-100]（defer 时移到 sample_tokens 的 finalize_kv_connector）
            ├─ (exit) get_finished(finished_req_ids)  [:102-104]
            ├─ (exit) get_block_ids_with_load_errors  [:105]
            ├─ (exit) stats / kv_cache_events / build_connector_worker_meta  [:107-109]
            └─ (exit) clear_connector_metadata  [:111-112]（defer 时同样推迟）
       └─ sample_tokens：draft 后 finalize_kv_connector  [:2470-2474]；产出 ModelRunnerOutput.kv_connector_output
      │  ZMQ response（所有 worker）→ KVOutputAggregator.aggregate  [multiproc_executor.py:343-371; utils.py:70-175]
EngineCore 进程: update_from_output
      ├─ invalid_block_ids → _handle_invalid_blocks       [scheduler.py:1578-1586]
      └─ _update_from_kv_xfer_finished：update_connector_output / finished_recving→可调度 / finished_sending→_free_blocks  [scheduler.py:1838-1839,2559-2586]
```

## 9. 标注为「无法确定/待确认」的残余点

- ascend 各 connector 后台线程内部的完成判定细节（如 mooncake `KVCacheSendingThread` 的 done 集合产生条件）未逐一展开——属于 connector 内部实现，本文件只到 `get_finished` 边界。`[源码可达，尚未运行验证]`
- upstream 参考版本为 v0.23.1rc0-1050，vllm-ascend 配套 v0.23.0；upstream 相关行号与细节以同 minor 系列参考计（已在上文相应处标注"参考版本存在小幅偏差"）。

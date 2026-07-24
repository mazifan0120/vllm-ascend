# Phase 1 证据 05：upstream Scheduler ↔ KVConnector 控制流

- 仓库：`/Users/leqi/Documents/Code/vllm`，branch `main`，HEAD `8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`，版本 v0.23.1rc0-1050。
- **版本偏差声明**：vllm-ascend 官方配套 upstream 为 v0.23.0，本证据基于 v0.23.1rc0-1050（同 minor 系列，存在小幅偏差）。所有涉及 upstream 的结论均受此偏差影响。
- 收集方式：只读源码（Read/Grep），未运行任何代码。除特别标注外，结论为 `[当前源码已确认]`；运行期先后顺序若由 assert/控制流强制则注明"框架保证"，否则注明"当前实现偶然顺序"。
- 路径速记：下文 `base.py` = `vllm/distributed/kv_transfer/kv_connector/v1/base.py`；`scheduler.py` = `vllm/v1/core/sched/scheduler.py`；`utils.py` = `vllm/distributed/kv_transfer/kv_connector/utils.py`；`multi_connector.py` = `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`；`core.py` = `vllm/v1/engine/core.py`；`mixin` = `vllm/v1/worker/kv_connector_model_runner_mixin.py`。
- 注意：`vllm/distributed/kv_transfer/kv_connector/base.py`（任务描述所给路径）仅 10 行，是 `KVConnectorBase = KVConnectorBase_V1` 的 re-export（该文件 5-8 行）。真正的接口定义在 `v1/base.py`。

---

## 1. KVConnectorBase_V1 接口契约（`v1/base.py`，共 708 行）

### 1.1 `get_num_new_matched_tokens(request, num_computed_tokens) -> tuple[int | None, bool]`
- 定义：base.py:453-486（abstractmethod，Scheduler 侧）。
- 返回值含义（base.py:468-477 docstring）：
  - 第一个元素：可从外部 KV 加载的、超出 `num_computed_tokens` 的 token 数；**`None` = connector 暂时无法确定，scheduler 应稍后再次查询该请求**（base.py:472-474）。
  - 第二个元素：`True` 表示外部 KV 将在 scheduler step 之间异步加载；**第一个元素为 0 时必须为 False**（base.py:475-477）。
- 类级 docstring（base.py:10-12）："Might be called multiple times for a given request and should be side-effect free" —— 契约要求**无副作用**。
- docstring 附加约束（base.py:479-485）：只应考虑调用时刻实际可用的最长 prefix；因连通性/驱逐等原因无法加载的 token 不得计入。

### 1.2 `update_state_after_alloc(request, blocks, num_external_tokens)`
- 定义：base.py:488-512（abstractmethod，Scheduler 侧）。
- 调用次数契约（base.py:495-499）：若 `get_num_new_matched_tokens` 曾返回 async（True），**同一请求可能被调用两次**——第一次在异步加载目标 block 分配后，第二次在传输完成、追加 block 分配后。
- 关键语义（base.py:501-504）：**应根据 `num_external_tokens` 而非 `blocks` 是否为空来决定是否加载**——`num_external_tokens == 0` 时 `blocks` 也可能非空（例如 MultiConnector 未被选中的子 connector 仍收到真实 blocks）。
- 第二次调用时 `num_external_tokens` 的值：scheduler.py:822-827（WAITING_FOR_REMOTE_KVS 返回后走 else 分支，`num_external_computed_tokens` 保持 scheduler.py:718 初始化的 0）→ scheduler.py:969-974 以 `num_external_tokens=0` 调用。`[当前源码已确认]`

### 1.3 `request_finished(request, block_ids) -> tuple[bool, dict | None]` 与 `request_finished_all_groups`
- `request_finished`：base.py:547-566。契约："Called exactly once when a request has finished, before its blocks are freed"（base.py:553-554）。返回 `True` = connector 接管 block 的异步释放，直到该 req_id 从 `get_finished()` 返回（base.py:556-562）；第二个返回值为可选 KVTransferParams，随引擎输出返回。
- `request_finished_all_groups`：`SupportsHMA` ABC 上的 abstractmethod，base.py:92-114。"Called exactly once when a request has finished **for all kv cache groups**, before its blocks are freed for each group"（base.py:99-100）。仅 HMA connector 支持（base.py:102 注释）。
- 调度侧调用点：`_free_request` → `_connector_finished`（scheduler.py:2162 → 2434-2469）。`_connector_finished` 先 `remove_skipped_blocks`（scheduler.py:2448-2454），再用 `get_block_ids_for_computed_tokens(num_computed_tokens=request.num_computed_tokens)` 取 block_ids（scheduler.py:2456-2459）；非 HMA 走 `request_finished(request, block_ids[0])`（scheduler.py:2461-2467），HMA 走 `request_finished_all_groups(request, block_ids)`（scheduler.py:2469）。运行进程：Scheduler（EngineCore）进程主循环。调用顺序：框架保证"exactly once"——`_free_request` 仅在请求进入 finished 状态后调用一次（scheduler.py:2159 assert）。

### 1.4 `get_finished(finished_req_ids) -> tuple[set[str] | None, set[str] | None]`
- 定义：base.py:357-373（Worker 侧，非 abstract，默认返回 `(None, None)`）。
- 返回：已完成异步传输的请求 id，元组为 **(finished_sending/saving, finished_recving/loading)**（base.py:366-370）。
- 契约约束（base.py:370-372）：返回的 finished saves/sends ids **必须属于历次调用传入的 `finished_req_ids` 集合之一**（本次或之前某次）。
- 调用点（Worker 进程，model runner 执行路径，非 transport callback）：
  - v1 GPU/TPU runner：mixin `_get_kv_connector_output` 的 `finally` 块，每次 execute_model 都执行（mixin:98-112，调用在 102-104）；无调度 token 的空步也走 `kv_connector_no_forward`（mixin:35-48），由 gpu_model_runner.py:4162-4165 在 `total_num_scheduled_tokens == 0` 时调用。
  - v2 runner：`vllm/v1/worker/gpu/kv_connector.py:77-96`（`post_forward`）与 98-105（`no_forward`）。
- `finished_req_ids` 参数内容：`scheduler_output.finished_req_ids`（mixin:103），即"上一步到本步之间 scheduler 侧 finish 的请求 id 集合"（scheduler.py:1151-1155 注释；output.py:211-214；累积点 `_free_request` scheduler.py:2175；每步 schedule 末尾重置 scheduler.py:1262）。
- 同处还调用 `get_block_ids_with_load_errors`（mixin:105）、`get_kv_connector_stats`（107）、`get_kv_connector_kv_cache_events`（108）、`build_connector_worker_meta`（109）。

### 1.5 `get_block_ids_with_load_errors() -> set[int]`
- 定义：base.py:375-393（Worker 侧，默认返回空集）。
- 契约（base.py:383-392 Notes）：同步/异步加载均适用；**异步加载：失败 block 可在直到（含）该请求被 `get_finished()` 返回的那个 forward pass 的任意 pass 上报；即使失败，请求仍必须经 `get_finished()` 上报，且失败 block id 不得晚于同一 pass 出现**。同步加载：在检测到的 pass 上报。
- Scheduler 后续动作：见 §2.4。

### 1.6 聚合链路：`build_connector_worker_meta` / `update_connector_output` / `bind_connector_metadata`
- `build_connector_worker_meta()`：base.py:429-437，Worker 侧每步调用（mixin:109），返回 `KVConnectorWorkerMetadata | None`。
- 聚合：所有 worker 的 metadata 在 scheduler 进程由 `KVOutputAggregator.aggregate` 用 `KVConnectorWorkerMetadata.aggregate` 两两合并（utils.py:135-144；契约 base.py:150-168，见 §4）。
- `update_connector_output(connector_output)`：base.py:537-545，Scheduler 侧，由 `_update_from_kv_xfer_finished` 在处理 finished 集合**之前**调用（scheduler.py:2570-2571，调用点 scheduler.py:1838-1839）。输入是**已跨 worker 聚合后的** `KVConnectorOutput`。
- `bind_connector_metadata(meta)`：base.py:211-221，Worker 侧，model runner 每次 forward 前调用（mixin:88-89；v2: gpu/kv_connector.py:65-68 `pre_forward`，先 `handle_preemptions` 再 bind 再 `start_load_kv`）；forward 后 `clear_connector_metadata`（mixin:111-112）。
- 反向（Scheduler→Worker）元数据：`build_connector_meta(scheduler_output)`（base.py:514-527 abstract），在 `schedule()` 末尾构造（scheduler.py:1166-1168 经 `_build_kv_connector_meta` 1186-1189），挂到 `SchedulerOutput.kv_connector_metadata`（output.py:234-235）。注意 base.py:522："calling this function will reset the state of the connector"。

### 1.7 其他相关钩子
- `on_new_request(request)`：base.py:529-535（默认 no-op），由 `Scheduler.add_request` 调用（scheduler.py:2088-2089），Scheduler 进程。
- `has_pending_push_work()`：base.py:577-587（默认 False）；`Scheduler.has_requests` 用它让引擎在 push 传输未完成时保持运转（scheduler.py:2262-2273）。
- `get_finished_count()`：base.py:630-640（默认 None），用于覆盖 KVOutputAggregator 的 expected count（见 §3）。
- `handle_preemptions(metadata)`：base.py:285-290（默认 no-op），Worker 侧 forward 前调用（gpu_model_runner.py:4128-4131）。

---

## 2. Scheduler async KV load 全流程（`vllm/v1/core/sched/scheduler.py`，共 2760 行）

引擎主循环顺序（core.py:488-517 `step()`）：`scheduler.schedule()`（499）→ `executor.execute_model(non_block=True)`（500）→ `_process_aborts_queue()`（510-512，注释明确"Before processing the model output, process any aborts that happened during the model execution"）→ `scheduler.update_from_output()`（513-515）。`[当前源码已确认，框架保证顺序]`

### 2.1 `get_num_new_matched_tokens` 调用点
- 全 scheduler.py 唯一调用点：scheduler.py:774-779，位于 waiting 队列调度循环内，且**仅当 `request.num_computed_tokens == 0`**（scheduler.py:724 的 if 分支内）。
- 返回 `None`：请求被 pop 并放回 step_skipped_waiting，本步跳过、后续重试（scheduler.py:781-787）。
- 重复调用的可能路径：preemption 将 `num_computed_tokens` 清零（scheduler.py:1204 `_preempt_request`），被抢占请求重新调度时会再次进入该调用点。`[当前源码已确认]`

### 2.2 `WAITING_FOR_REMOTE_KVS` 进入与离开
- 进入（scheduler.py:834-1006）：
  1. `load_kv_async=True` 时 `num_new_tokens = 0`（834-837，不分配新计算量）；
  2. `allocate_slots(..., num_external_computed_tokens=N, delay_cache_blocks=load_kv_async, reserved_blocks=...)`（942-954）；`delay_cache_blocks` 使 blocks 暂不进入 prefix cache（kv_cache_manager.py:474-477）；
  3. `connector.update_state_after_alloc(request, blocks, N)`（969-974，第一次调用）；
  4. `request.status = WAITING_FOR_REMOTE_KVS`，放回 skipped 队列（986-990）；
  5. **`request.num_computed_tokens` 被设为 local+external 之和，尽管 KV 尚未加载**（991-1004；注释说明失败时由 `_update_requests_with_invalid_blocks` 重设）；
  6. 加入 `_inflight_prefills`（1005），用于后续 async load 的容量准入（934-940）。
- 离开（scheduler.py:2526-2541 `_try_promote_blocked_waiting_request`，由调度循环 691-701 触发）：
  - 条件：`request.request_id in self.finished_recving_kv_req_ids`（2534），否则返回 False 继续等待；
  - 动作：`_update_waiting_for_remote_kv(request)`（2536 → 2492-2524）：正常路径 `cache_blocks(request, num_computed_tokens)`（2517），全量命中时 `num_computed_tokens = num_tokens - 1`（2519-2522，重算最后一个 token 以便采样）；失败路径见 §2.4；
  - 之后 `finished_recving_kv_req_ids.remove(...)`（2524）；状态置为 PREEMPTED（曾被抢占）或 WAITING（2537-2540）。
- 状态枚举：`RequestStatus.WAITING_FOR_REMOTE_KVS`（vllm/v1/request.py:343）。

### 2.3 finished_recving / finished_sending 的 scheduler 侧消费
- 入口：`update_from_output` 尾部 `_update_from_kv_xfer_finished(kv_connector_output)`（scheduler.py:1837-1839 → 2559-2586）。运行进程：Scheduler 进程主循环（非 callback）。
- 先 `connector.update_connector_output(...)`（2570-2571）。
- `finished_recving`（2574-2582）：逐 id `assert req_id in self.requests`（2576）；若状态 == WAITING_FOR_REMOTE_KVS → 加入 `finished_recving_kv_req_ids`（2578-2579，下一步 promote）；否则 `assert RequestStatus.is_finished(req.status)`（2581）并立即 `_free_blocks`（2582，含 `del self.requests[req_id]`，2185-2188）。**即：finished_recving 到达时请求必须处于 WAITING_FOR_REMOTE_KVS 或已 finished，否则 assert 崩溃——框架以 assert 强制该不变式。**
- `finished_sending`（2583-2586）：同样 assert 存在性后立即 `_free_blocks`。这对应 `request_finished` 返回 True 的延迟释放。
- `has_finished_requests`（scheduler.py:2250-2260）：delay_free 的请求已从队列移除但仍留在 `self.requests`，靠 `len(self.requests) > num_in_queues` 保持引擎不退净。

### 2.4 KV load 失败处理（invalid blocks / FINISHED_ERROR）
- 策略配置：`kv_load_failure_policy: Literal["recompute", "fail"] = "fail"`（vllm/config/kv_transfer.py:69-72，**默认 fail**）；`self.recompute_kv_load_failures = policy == "recompute"`（scheduler.py:144-145）。
- 入口：`update_from_output` 开头，`kv_connector_output.invalid_block_ids` 非空即调 `_handle_invalid_blocks`（scheduler.py:1578-1586）——**先于** per-request 输出循环（1614 起）和 `_update_from_kv_xfer_finished`（1838）。
- `_handle_invalid_blocks`（2691-2760）：
  - async load 请求：从 `self.skipped_waiting` 中筛 `status == WAITING_FOR_REMOTE_KVS`（2703-2707），`evict_blocks=False`（2708-2715）；
  - sync load 请求：扫 `self.running`（2721-2725），`evict_blocks=True`；
  - `_update_requests_with_invalid_blocks`（2588-2689）：把受影响请求的 `num_computed_tokens` 截断到**第一个失败 block 的边界**（2665）；`evict_blocks=True` 时收集失败 block 及其全部下游 block（2671-2673）；共享 block 只由第一个请求重算（2647-2654，注释明确"Currently this only applies to sync loading; **Async loading does not yet support block sharing**"）；
  - **fail 策略**（should_fail=True，2700）：返回全部失败 req_id（2739-2748）→ update_from_output 主循环跳过这些请求（1619-1621）→ `finish_requests(failed_ids, FINISHED_ERROR)`（1823-1825）并为每个失败请求生成带 `finish_reason=error` 的 EngineCoreOutput（1826-1835）。FINISHED_ERROR → FinishReason.ERROR（vllm/v1/request.py:353, 377）。
  - **recompute 策略**：失败 async 请求 id 记入 `failed_recving_kv_req_ids`（2758）；sync 受影响 id 返回以跳过本步输出（2760）。
  - **prefix cache 失效**：仅当 `sync_blocks_to_evict` 非空**且非 recompute 策略**时 `kv_cache_manager.evict_blocks(...)`（2736-2737；evict_blocks 定义 kv_cache_manager.py:534-536，按 block id 从 prefix cache 逐出）。recompute 策略下不逐出（注释：block 将被重算并被共享请求复用，2733-2735）。
  - **失败请求的 block 释放时机**（fail 策略）：经 `finish_requests` → `_free_request(delay_free_blocks=...)`；对仍处于 WAITING_FOR_REMOTE_KVS 的请求，`delay_free_blocks = req_id not in finished_recving_kv_req_ids`（2144-2147），几乎总是 True → blocks 暂不释放、`self.requests` 保留；待 connector 按契约（§1.5）经 `get_finished()` 上报 finished_recving 后，由 `_update_from_kv_xfer_finished` 的 else 分支 `_free_blocks`（2580-2582）。`[当前源码已确认，框架保证顺序]`
  - recompute 策略下 async 失败请求的后续：finished_recving 到达 → promote → `_update_waiting_for_remote_kv` 检测 `failed_recving_kv_req_ids`（2502）：有有效 token 则 `cache_blocks(request, num_computed_tokens)`（2505-2507，此时 num_computed_tokens 已被截断），否则 `kv_cache_manager.free(request)` 释放全部已分配 block（2508-2511）；随后请求回到 WAITING/PREEMPTED 重新调度、重算失败段。

### 2.5 其余与 block 释放时机相关的机制
- `defer_block_free`：async scheduling/PP（`max_concurrent_batches > 1`）且为 kv_consumer 时置 True（scheduler.py:147-153）；释放经 `deferred_frees` fence 延迟到对应 step 的 output 处理完之后（2197-2236，drain 点 1567-1569）。

---

## 3. KVOutputAggregator（`vllm/distributed/kv_transfer/kv_connector/utils.py:55-175`）

- 构造：`KVOutputAggregator.from_connector(connector, world_size)` → `cls(connector.get_finished_count() or world_size)`（utils.py:66-68）。`world_size = parallel_config.world_size`（vllm/v1/executor/abstract.py:280-284 `init_kv_output_aggregator`）；初始化于 EngineCore `__init__`（core.py:163-164，仅当 scheduler.connector 非 None）。
- 聚合规则（`aggregate`，utils.py:70-175）：
  - **finished_sending / finished_recving：按 req_id 倒计数**，初始计数 = `_expected_finished_count`，每收到一个 worker 上报减 1，**减到 0 才对外发布该 id**（utils.py:78-90）——即要求所有期望的 rank 都上报同一 ID。`[当前源码已确认]`
  - `_expected_finished_count` 可被 worker 动态更新：`kv_output.expected_finished_count > 0` 且不同则替换（utils.py:103-114；字段语义 vllm/v1/outputs.py:206-211，注释称供 Nixl 类 handshake connector 用）。
  - **invalid_block_ids：纯并集 `|=`，不做 quorum/计数门控**（utils.py:159）——任一 worker 上报即随本步输出发布。`[当前源码已确认；与 finished 集合的计数语义不同，值得评审注意]`
  - `kv_connector_stats`、`kv_connector_worker_meta`：取首个非 None 为累加器，后续逐一 `.aggregate()`（utils.py:123-144）。
  - 输出：以 `output_rank`（默认 0）worker 的 ModelRunnerOutput 为载体，替换其 `kv_connector_output`（utils.py:161-173）。
- 运行位置：Scheduler/EngineCore 进程。multiproc：`collective_rpc` 内对全部 worker 回复应用 `partial(aggregator.aggregate, output_rank=...)`（multiproc_executor.py:364-368，execute_model 传入 319）；ray：`ray_executor.py:459-465`。`[当前源码已确认]`

---

## 4. KVConnectorWorkerMetadata.aggregate 契约（base.py:150-168）

- 契约原文（base.py:152-158）：每个 worker 可输出自己的 metadata；**同一 engine step 内所有 worker 返回的 metadata 对象会先经 `aggregate` 方法聚合，再交给 Scheduler 侧 connector**。
- 谁调用：`KVOutputAggregator.aggregate`（utils.py:135-144）——两个非 None 的 metadata 对象间调用 `aggregate(other)`（utils.py:139-144）。
- 何时：每个 engine step 收集齐 worker 输出后，在 Scheduler 进程、进入 `scheduler.update_from_output` 之前。`[当前源码已确认]`
- 冲突处理：基类契约**未定义**任何冲突语义（abstractmethod 只有一句 docstring，base.py:161-168）；具体语义由子类决定。MultiConnector 的实现 `MultiKVConnectorWorkerMetadata.aggregate`（multi_connector.py:55-68）：按子 connector 位次逐元素合并——一侧为 None 取另一侧，两侧均非 None 递归调子 connector 的 `aggregate`；`assert isinstance(other, MultiKVConnectorWorkerMetadata)` 且长度相等（56-58）。
- Scheduler 侧消费：`connector.update_connector_output(kv_connector_output)`（scheduler.py:2570-2571）；MultiConnector 会把聚合后的 meta 拆包、临时替换 `connector_output.kv_connector_worker_meta` 为各子 connector 的分片再分发（multi_connector.py:431-450，finally 恢复整体 meta）。

---

## 5. upstream MultiConnector 语义（multi_connector.py，共 667 行）及与 Ascend 版差异

### 5.1 upstream 语义
- 类 docstring（multi_connector.py:128-136）：**Load 用按配置顺序第一个声明有可用 token 的 connector；Save 到全部 connector**。
- Scheduler 侧：
  - `get_num_new_matched_tokens`（381-400）：逐个调子 connector；**任一返回 None 立即返回 `(None, False)`**（393-394）；第一个 `toks > 0` 的 connector 记入 `_requests_to_connector[req_id] = i`（397-399）。
  - `update_state_after_alloc`（402-412）：被选中的 connector 收到 `(request, blocks, num_external_tokens)`；**未选中的也收到真实 blocks 但 `num_external_tokens=0`**（410-412——这正是 base.py:501-504 docstring 警告的来源）。
  - `request_finished` / `request_finished_all_groups`（479-537）：全部子 connector 都调用；统计 async_save 数量，>1 时 `_extra_async_saves[req_id] = async_saves - 1`（503-504）；多个非 None txfer_params 做 key 合并、**key 冲突抛 RuntimeError**（492-499）；弹出 `_requests_to_connector`（506）。
  - `build_connector_meta`（418-429）：打包各子 connector meta 为 `MultiKVConnectorMetadata`，并附带 `_extra_async_saves` 后清空（426-428）。
- Worker 侧：
  - `get_finished`（311-336）：recving 取各子 connector 并集；**sending 需等 `_extra_async_saves` 倒计时耗尽才上报**（325-334）——保证多个 connector 异步保存同一请求时只对外发布一次。
  - `get_block_ids_with_load_errors`（338-342）：各子 connector 并集。
  - `bind_connector_metadata`（260-266）：分发子 meta，并 `update(_extra_async_saves)`。
  - `get_finished_count`（355-358）：返回 None（注释：目前无 connector 返回非 None）。
- 进程/顺序：Scheduler 侧方法在 Scheduler 进程调度循环内同步调用；Worker 侧方法在 Worker 进程 execute_model 路径内同步调用（§1.4）。调用顺序为框架保证（schedule → execute_model → update_from_output）。

### 5.2 与 Ascend 版差异（AscendMultiConnector）
- vllm-ascend **注册表级替换**：启动时将 `"MultiConnector"` 从 `KVConnectorFactory._registry` 弹出并注册 `AscendMultiConnector`（vllm-ascend 仓 `vllm_ascend/distributed/kv_transfer/__init__.py:23-26`），类定义 `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py:19`（继承 `MultiConnector, SupportsHMA`）。
- 差异点（`[当前源码已确认]`，行号为 vllm-ascend 仓 ascend_multi_connector.py）：
  1. `update_state_after_alloc`（32-41）：未选中的 connector 收到的是 **empty blocks（`blocks.new_empty()`）+ 0**——与 upstream 的"真实 blocks + 0"不同；例外：`MooncakeLayerwiseConnector` 实例即使未被选中也收到真实 blocks + num_external_tokens（36-38）。
  2. `get_num_new_matched_tokens`（43-63）：先扫描带 `has_preempted_request` 钩子的子 connector 并给 preempted 请求优先权，命中即返回；否则回落 `super()`。
  3. `request_finished_all_groups`（78-102）：任一第二个非 None txfer_params 直接 `RuntimeError("Only one connector can produce KV transfer params")`（94-95）——upstream 是 merge + key-clash 检查。
  4. 新增 `update_state_before_preempt` 扇出钩子（65-76），upstream MultiConnector 无此方法（注：此为 connector 自定义钩子，非 base 接口）。
  5. `_all_support_hma` 改为按子 connector 实例 `supports_hma(c)` 计算（27），upstream 用基于配置的 `all_children_support_hma`（multi_connector.py:153-167, 188）。

---

## 6. 请求取消 / abort 时 connector 相关清理路径

- 入口：`EngineCore.abort_requests(request_ids)` → `scheduler.finish_requests(ids, FINISHED_ABORTED)`（core.py:418-424）；shutdown 路径 `finish_requests(None, FINISHED_ABORTED)`（core.py:1355-1357, 1699-1701）。处理时机：abort 在 `execute_model` 与 `update_from_output` 之间统一处理（core.py:510-512 `_process_aborts_queue`）。`[当前源码已确认，框架保证顺序]`
- `finish_requests`（scheduler.py:2093-2154）：
  - 从 running/waiting/skipped_waiting 队列移除（2119-2139）；
  - **对 WAITING_FOR_REMOTE_KVS 请求**：`delay_free_blocks = (req_id not in finished_recving_kv_req_ids)`（2144-2147），并从 `finished_recving_kv_req_ids` / `failed_recving_kv_req_ids` 中 discard（2148-2149）；
  - 置状态后 `_free_request(request, delay_free_blocks=delay_free_blocks)`（2151-2152）。
- `_free_request`（2156-2183）：无论是否 delay，都先 `_connector_finished(request)` 调 connector 的 `request_finished`/`request_finished_all_groups`（2162；即 abort 也会触发该钩子——core.py:413-416 注释亦表明这是刻意设计："Immediately abort so that the connector's request_finished hook runs to free any pre-admission KV-transfer resources"）；`finished_req_ids.add(request_id)`（2175，下一步随 SchedulerOutput 通知 worker）；`delay_free_blocks |= connector_delay_free_blocks`（2179），两者皆 False 才 `_free_blocks`（2180-2181）。
- 延迟释放的落地：worker 后续上报 finished_recving/finished_sending → `_update_from_kv_xfer_finished`：若请求已 finished（非 WAITING_FOR_REMOTE_KVS）则 assert 后 `_free_blocks`（scheduler.py:2578-2586），`_free_blocks` 内 `del self.requests[req_id]`（2185-2188）。`[当前源码已确认；依赖 §1.5 契约——失败/aborted 的加载请求仍必须经 get_finished 上报，否则 self.requests 泄漏、blocks 永不释放]`
- Worker 侧清理：scheduler 的 `finished_req_ids` 经 SchedulerOutput 传到 worker，作为 `get_finished(finished_req_ids)` 入参（mixin:102-104），connector 据此释放 per-request 缓存状态（base.py:361-364 docstring："The scheduler process (via the Executors) will use this output to track which workers are done"）。
- 引擎保活：delay_free 请求经 `has_finished_requests`（scheduler.py:2250-2260）防止引擎提前静默；push 模式经 `has_pending_push_work`（2262-2273）。

---

## 7. 与常见假设不符 / 需评审注意的点（均为源码事实，非设计评价）

1. **invalid blocks 不做跨 rank 聚合计数**：KVOutputAggregator 对 finished_sending/recving 用"所有期望 rank 都上报才发布"的倒计数（utils.py:78-90），但对 `invalid_block_ids` 是**任一 worker 上报即并集发布**（utils.py:159）。TP>1 时单 rank 上报失败 block 就会影响所有共享该 block 的请求。
2. **`num_computed_tokens` 在异步加载完成前就被置为 full 值**（scheduler.py:1004），失败时靠 `_update_requests_with_invalid_blocks` 截断（2665）；任何在 promote 之前读取该字段的代码看到的都是"乐观值"。
3. **async load 不支持 block 共享**：`_update_requests_with_invalid_blocks` 内注释两处明确（scheduler.py:2652-2653, 2680-2681）。
4. **abort 也会同步触发 `request_finished` 钩子**（含 WAITING_FOR_REMOTE_KVS 中的请求），且此时传入的 block_ids 基于乐观的 `num_computed_tokens`（scheduler.py:2456-2459 + 1004）。
5. **finished_recving/sending 到达时请求状态不变式由 assert 强制**（scheduler.py:2576, 2581, 2585）：既不在 `self.requests` 也不是 finished/WAITING_FOR_REMOTE_KVS 的情况会直接 crash scheduler，而非降级处理。
6. `kv_load_failure_policy` 默认是 **fail**（config/kv_transfer.py:69），不是 recompute。
7. base.py 类 docstring 声称 `get_num_new_matched_tokens` "should be side-effect free"（base.py:10-12），但 upstream MultiConnector 的实现会在其中写 `_requests_to_connector`（multi_connector.py:398）——框架自身实现即与该文档表述有出入 `[事实冲突：文档 vs 实现]`。
8. `update_state_after_alloc` 的第二次调用以 `num_external_tokens=0` 进行（§1.2），connector 若误用"blocks 非空即加载"会重复加载——base.py:501-504 专门就此发出警告。

# Phase 1 证据 04 — AscendStoreConnector / KVPoolScheduler / KVPoolWorker 控制流

- 基线：vllm-ascend `dev/dualpath` @ `0ec11a4`（v0.19.1rc1-991）；upstream vllm `main` @ `8df14cf`（v0.23.1rc0-1050，官方配套 v0.23.0，**涉及 upstream 的结论存在小幅版本偏差，下文标注 "upstream±"**）。
- 范围文件（vllm-ascend 根相对路径，下文省略前缀 `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`）：
  - `ascend_store_connector.py`（328 行，全读）
  - `pool_scheduler.py`（1130 行，全读）
  - `pool_worker.py`（1943 行，关键区段全读）
  - `kv_transfer.py`（1570 行，线程基类/Sending/Recving 线程全读）
  - `config_data.py`（1121 行，metadata 相关类全读）
- 标注：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[外部接口待确认]` / `[事实冲突]`。

## 进程/线程模型总览 `[当前源码已确认]`

- **Scheduler 进程**：`KVPoolScheduler` 全部方法同步运行在调度循环内；非 layerwise 时内含 `LookupKeyClient`（zmq REQ，**阻塞式 recv、无超时**，`pool_scheduler.py:1106-1108`）。
- **Worker 进程（每个 rank 一个）**：`KVPoolWorker.start_load_kv / wait_for_save / get_finished` 运行在 model-runner 主线程；传输在 daemon 线程：
  - 非 layerwise：`KVCacheStoreSendingThread`（`kv_transfer.py:628`）与 `KVCacheStoreRecvingThread`（`kv_transfer.py:820`，仅 `load_async=True` 时创建，`pool_worker.py:455-467`），均 `daemon=True`（`kv_transfer.py:292`）。
  - layerwise：`KVCacheStore{KeyLayer,Layer}{Sending,Recving}Thread`（`pool_worker.py:354-436`）。
  - 非 layerwise 且 `parallel_config.rank == 0` 的 worker 进程额外跑 `LookupKeyServer` daemon 线程（`ascend_store_connector.py:119-120, 324-325`）。
- 调用顺序：`start_load_kv` 在 forward 前由框架调用，`get_finished` 每步执行后由 `kv_connector_model_runner_mixin._get_kv_connector_output` 调用（upstream `vllm/v1/worker/kv_connector_model_runner_mixin.py:78-109`）——框架保证的顺序（upstream±）。

## Q1. Scheduler 侧 `get_num_new_matched_tokens()` 完整流程 `[当前源码已确认]`

入口：`ascend_store_connector.py:126-128` → `KVPoolScheduler.get_num_new_matched_tokens`（`pool_scheduler.py:478-565`）。

1. `kv_consumer` 且 `consumer_is_to_load=False` → 直接 `(0, False)`（495-496）。
2. 三条 lookup 路径：
   - **use_gva_layerwise**（layerwise+memcache）：本地 `self.store_scheduler.batch_get_key_info`，`_get_layerwise_gva_hit_tokens`（498-500, 301-368）。`store_scheduler` 在 `__init__` 由 `backend_class.create_scheduler_client(parallel_config)` 创建（149-157）。
   - **use_layerwise（非 memcache）**：本地 `self.store_scheduler.batch_is_exist`，`_get_store_lookup_hit_tokens(include_layers=True)`（510-513, 245-286）。
   - **非 layerwise**：**跨进程同步 RPC**。首次进入时惰性创建 `self.client = LookupKeyClient(self.vllm_config)`（515-516），`client.lookup(token_len, request.block_hashes, self.kv_cache_group_ids)`（517-521）→ zmq REQ 发到 worker rank-0 的 `LookupKeyServer`。
3. 命中 == `request.num_tokens` 时减 1（526-527）；`need_to_allocate = hit - num_computed_tokens`（529-532）。
4. `need_to_allocate <= 0` 且非 `force_layerwise_load` → `(0, False)`，**不创建 LoadSpec**（545-547）。
5. 否则创建 `self.load_specs[request.request_id] = LoadSpec(vllm_cached_tokens, kvpool_cached_tokens, can_load=force_layerwise_load)`（549-553）。
6. 返回 `(need_to_allocate, self.load_async and not self.use_layerwise)`（565）——第二元素即 upstream 的 `load_async` 标志；layerwise 下恒为 False。

**设计声称「会创建 client/load-spec 状态，不能当纯 probe」——验证结果：成立，且更精确**：

- `LookupKeyClient`：第一次「通过 granularity 门槛的」非 layerwise lookup 时创建（515-516），之后常驻 scheduler 进程；`close()`（1111-1112）在生产代码中**无任何调用方**（grep 全仓库，仅定义处）。socket 永不关闭，靠进程退出清理。
- `LoadSpec`：仅在「命中 >0 且（need_to_allocate>0 或 force_layerwise_load）」时创建（549-553）；纯 miss（523-524）与「全部本地已缓存」（546-547）不创建。
- LoadSpec 的清理点仅三处：`_process_new_request` pop（690）、`_process_preempted_cached_request` pop（742）、`_process_async_load_request` pop（845）。**请求在 waiting 阶段被 abort 时无清理路径**：`request_finished`（1012-1037）与 `build_connector_meta` 的 finished 清理循环（896-901）均不触碰 `load_specs` → 残留的 `load_specs[req_id]` 永久滞留（泄漏量级=每 abort 一个已命中请求一条小 dataclass）。
- 附带：GVA 路径的 hit 检查会顺手创建 `RequestTracker`（312）。

## Q2. `cache_transfer_granularity` 与 `discard_partial_chunks` `[当前源码已确认]`

- `_infer_cache_transfer_granularity`（`pool_scheduler.py:404-413`）：`math.lcm(lcm_block_size, 各 group 的 get_cache_family_granularity(...))`；`lcm_block_size` 是 `grouped_block_size` 的 lcm（118），`grouped_block_size = original_block_size * (pcp_size*dcp_size)`（106-108）。`_floor_to_cache_transfer_granularity` 为整除下取整（415-416）。worker 侧有同构实现（`pool_worker.py:542-552`，`pool_worker.py:158`）。
- `_discard_partial_chunks`：`pool_scheduler.py:124-126` 从 extra_config 读，**默认 True**；127-130 在 `use_layerwise` 时用同一默认值重读一次（冗余代码，无语义差异）。
- 精确语义（开启时）：
  - lookup 查询长度先 floor 到 granularity（502-503）；`token_len < granularity` 直接 `(0, False)`（507-508）。
  - save 侧 `_get_last_chunk_tokens_num` 同样 floor（628-631）。
  - `ReqMeta.from_request_tracker` 中 `num_tokens_to_save = target_token_len // granularity * granularity`，`chunk_boundary = cdiv(num_saved+1, granularity)*granularity`（`config_data.py:896-905`），未过边界且无 partial block 时 `skip_save=True`（923-924）。
  - worker 同步 load 时 `kvpool_cached_tokens % granularity != 0` 且差 1 的边界情形 token_len +1（`pool_worker.py:769-775`）。

## Q3. `update_state_after_alloc()` 与 LoadSpec 冻结 `[当前源码已确认]`

`pool_scheduler.py:567-626`：

1. 无条件登记 `_unfinished_requests[req_id] = (request, local_block_ids)` 与 `_unfinished_request_ids.add`（578-579）。
2. 无 load_spec → return（580-587）。
3. `num_external_tokens == 0` → `can_load = use_layerwise and kvpool_cached_tokens > 0`（589-602），return。
4. `num_external_tokens > 0` → assert 与 `kvpool_cached - vllm_cached` 相等（604-614），`can_load = True`（616）；**`load_async and not use_layerwise` 时 `_loading_req_ids.add(req_id)`（617-618）——这是请求进入 `AscendConnectorMetadata.loading_req_ids` 的唯一入口**。
5. metadata 构建：`AscendConnectorMetadata(self._unfinished_request_ids, preempted_req_ids, self._loading_req_ids.copy(), self._delayed_free_req_ids.copy())`（910-915）。注意第一个参数**按引用传递**（非 copy），结构见 `config_data.py:976-992`（`requests: list[ReqMeta]` + 四个 id 集合）。
6. `_loading_req_ids` 的清理：仅在 `build_connector_meta` 中对 finished（901）/preempted（907）discard；`update_finished_recving`（1077-1079）本应按完成集合清理但**生产代码无调用方**（仅 `tests/ut/distributed/ascend_store/test_pool_scheduler.py:839-848` 调用）。实际不会重复上报：worker 侧 `get_and_clear_finished_requests` 上报后即从线程集合清除（`kv_transfer.py:321-337`）。
7. 异步 load 请求的 ReqMeta 由 `_process_async_load_request`（839-885）在 `build_connector_meta` 中对「已分配但未调度」的请求生成（951-957）；`load_specs.pop`（845）保证带 load_spec 的 ReqMeta 只发一次。

## Q4. `consumer_is_to_load` 与 `use_layerwise` 实际语义 `[当前源码已确认]`

- `consumer_is_to_load`：extra_config，默认 False（`pool_scheduler.py:87-89`）。**唯一用途**：495 行——`kv_consumer` 角色下不做外部命中查询（D 侧默认不加载 store 数据，除非显式打开）。
- `use_layerwise`：extra_config（`ascend_store_connector.py:86`）。影响面：
  - `requires_piecewise_for_cudagraph` 返回 True（74-80）；
  - 不创建 `LookupKeyServer`（119-120）；
  - lookup 走本地 store_scheduler 而非 zmq RPC（510-513）；
  - 返回值第二元素恒 False，即**layerwise 下 load_async 被强制关闭**（565, 617）；
  - `wait_for_save` 跳过（connector 240-241），save 改走逐层 `save_kv_layer`（224-233）；
  - `request_finished{,_all_groups}` 立即释放块（注释：layerwise 无 sending event，delay-free 会泄漏；1024-1026, 1048-1051）；
  - worker 侧改用逐层传输线程（`pool_worker.py:354-436`）。

## Q5. KVPoolWorker：`start_load_kv` / `get_finished` / 失败语义

### start_load_kv（`pool_worker.py:745-878`）`[当前源码已确认]`

- layerwise：`process_layer_data(metadata.requests)`（754-755 → 1272-1286）：按物理层×group 建 save/load 任务，GVA 分配在 worker 侧（注释 642-647 说明 memcache 要求 batch_alloc 与 batch_copy 同进程）；逐层 load 经 `LayerLoadTask` 入 recv 线程（1288-1315），`wait_for_layer_load` 按层等 `threading.Event`，超时 10s 仅 log（1317-1331）。
- 非 layerwise、逐请求（757-877）：
  - 无 load_spec 或 `can_load=False` → 跳过（759-765）。
  - 构造 key/addr/size（`token_database.process_tokens_with_block_ids` + `prepare_value`，797-825），按 `tp_rank % len` 循环移位打散后端压力（828-831）。
  - **一次 bulk `self.m_store.get(key_list_c, addr_list_c, size_list_c)`（840）**。`load_async=False` 时这是 model-runner 线程上的**同步阻塞调用**；本层**无 DMA 拆分**（拆分只存在于 layerwise `_batch_copy_with_limits`，`kv_transfer.py:397-448`，由 extra_config `layerwise_max_transfer_blocks/bytes` 控制，`pool_worker.py:160-161`）。
  - `load_async=True` → `self.kv_recv_thread.add_request(request)`（787-791），异步由 `KVCacheStoreRecvingThread._handle_request` 做同样的一次 bulk get（`kv_transfer.py:846-902`）。

### get_finished 签名与语义（`pool_worker.py:1534-1563`）`[当前源码已确认]`

- 签名：`get_finished(finished_req_ids: set[str], meta: AscendConnectorMetadata) -> tuple[set[str], set[str]]`，返回 `(done_sending, done_recving)`，与 upstream 契约一致（upstream± `base.py:357-373`）。
- 调用方：`ascend_store_connector.py:245-255`；注意 249-253 在 `_current_step_has_real_forward` 时先 `ensure_store_initialized()`（`pool_worker.py:1565-1568`，调 backend 的 `ensure_initialized`，惰性初始化后端）。
- done_sending：非 layerwise 时 `send_thread.get_and_clear_finished_requests(meta.delayed_free_req_ids)`（1545-1547）；layerwise 恒空集（1541-1543）。
- done_recving：`load_async` 时 `recv_thread.get_and_clear_finished_requests(meta.loading_req_ids)`（1551-1555）——**Store 加载完成 → finished_recving 的链路**：recv 线程 `_handle_request` 末尾 `set_finished_request(req_id)`（`kv_transfer.py:942`）→ 本处按 scheduler 快照 `loading_req_ids` 取交集并清除 → connector 返回 → upstream mixin 填入 `KVConnectorOutput.finished_recving`（`kv_connector_model_runner_mixin.py:102-103`）→ scheduler `_update_from_kv_xfer_finished`（upstream `scheduler.py:1839`，upstream±）。
- 预处理：对 preempted 请求清理两线程的 finished 集合（1537-1540, 1553）。

### Store 失败时产生什么 `[当前源码已确认]`

- `m_store.get` 返回码列表有非 0，或整体返回 None（视为全失败）：`record_failed_blocks`（`kv_transfer.py:1556-1570`）收集失败 block_id 并 `logger.error`。
- **单 group**：加入 `_invalid_block_ids`（同步路径 `pool_worker.py:846-847, 861-862`；异步路径 `kv_transfer.py:908-910, 924-926`）。`KVPoolWorker.get_block_ids_with_load_errors`（1333-1337）copy+clear 返回 → connector（`ascend_store_connector.py:257-260`）→ upstream mixin 每步 `get_finished` 之后立即取（`kv_connector_model_runner_mixin.py:105`）→ `KVConnectorOutput.invalid_block_ids` → scheduler `_handle_invalid_blocks`（upstream `scheduler.py:1578-1586, 2691-2760`）：默认 `recompute` 策略（`scheduler.py:130,145`）重算受影响块；`fail` 策略 → 请求 FINISHED_ERROR（1823-1835）（upstream±）。
- **多 group（hybrid）**：**只 `logger.error`，明确跳过 invalid-block 回退**（注释原话 "Skip invalid-block fallback to avoid scheduler crash"，`pool_worker.py:848-855, 863-870`；异步 `kv_transfer.py:911-918, 927-934`）→ hybrid 下加载失败对 scheduler **完全不可见**，块内为脏数据但请求照常标记完成。
- **异步路径失效上报存在断裂（重要）**：`KVPoolWorker` 创建 `KVCacheStoreRecvingThread` 时**没有注入** `invalid_block_ids`/`lock`（`pool_worker.py:457-465`，仅传 7 个位置参数），线程于是使用自己私有的集合（`kv_transfer.py:843-844`）；而 `KVPoolWorker.get_block_ids_with_load_errors` 只 drain worker 自己的 `_invalid_block_ids`（1333-1337）。⇒ **load_async 模式下加载失败的 block 永远不会进入 `KVConnectorOutput.invalid_block_ids`**，但请求仍被无条件 `set_finished_request`（`kv_transfer.py:942`）并经 `get_finished` 上报为完成。这与 upstream `get_block_ids_with_load_errors` 文档契约（"failed block IDs must appear here no later than that same pass"，`base.py:384-389`）不符。
- save（put）路径失败：`self.m_store.put(keys, addrs, sizes)` 的返回值**未检查**（`kv_transfer.py:805`），save 失败静默；completion 在 finally 中照常 `mark_completed_events`（810-812）。

## Q6. LookupKeyServer `[当前源码已确认]`

- **创建条件**（`ascend_store_connector.py:106-120`）：worker 角色分支内，`not self.use_layerwise and vllm_config.parallel_config.rank == 0`（119-120）——设计声称「非 layerwise 且 process rank==0 时创建」**精确成立**；rank 是全局 parallel rank，TP/PP 下只有 rank-0 worker 起服务。
- 位置：`LookupKeyServer` 就在 `ascend_store_connector.py:283-328`（不依赖外部文件）。
- socket：zmq REP bind 于 `get_zmq_rpc_path_lookup`（`pool_scheduler.py:1115-1130`）：`ipc://{envs.VLLM_RPC_BASE_PATH}/lookup_rpc_port_{port}_dp_rank{dp_rank}`；port 取 extra_config `lookup_rpc_port`，回退 `mooncake_rpc_port`（deprecated warning），默认 0；按 dp_rank 区分。
- 请求格式（302-322）：multipart frames = `[token_len(4B big-endian)] [msgpack(kv_group_ids)] [msgpack(hash_hex_str)...]`；处理为 `pool_worker.lookup_scheduler(token_len, hashes, kv_group_ids, use_layerwise=False)`（309-314，**use_layerwise 硬编码 False**，与创建条件自洽）；响应 = 4B big-endian 命中 token 数。
- `lookup_scheduler`（`pool_worker.py:1757-1874`）：把 key 按 `group_tp_size × pp_size` 扩展出所有 rank 变体（1803-1818），`m_store.exists`（1820），按 granularity 对齐的连续/断续命中位置（1841-1848, 1900-1930），跨 group 取最大公共命中（1867, 1876-1886）；任何异常 → `logger.error` 并返回 0（1860-1866，miss 处理）。运行在 rank-0 worker 进程的 server daemon 线程上，用的是该 worker 的 `m_store` client。
- 生命周期：`close()`（327-328）只关 socket、**生产代码无调用方**；`self.running` 无出处被置 False；`AscendStoreConnector` 未覆写 `shutdown`（upstream 基类为 no-op，`base.py:395-401`）→ 线程/socket 随进程退出回收（daemon）。
- 附带：`KVPoolWorker.lookup`（1570-1649，单 rank exists 版本）生产无调用方，仅单测使用（`tests/ut/distributed/ascend_store/test_pool_worker.py:338 等`）。

## Q7. `page_size_bytes` 与 `bind_gpu_block_pool` `[当前源码已确认]`

- Scheduler 侧来源：`kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes`（`ascend_store_connector.py:108`），传入 `KVPoolScheduler`（109-111），存 `self.page_size_bytes` 并 info 日志（`pool_scheduler.py:144-145`）。**在 pool_scheduler.py 内除赋值/日志外无任何使用**。
- Worker 侧：`register_kv_caches` 中 `self.page_size_bytes = sum(self.block_len)`（`pool_worker.py:729`）；`block_len = cache[0].numel() * element_size * block_size_scale`（575-584），即「单层单块字节数 × K/V 份数」。用途：`_build_group_layer_builders` 的 group page size 回退（329-348）与 GVA layerwise 收发线程构造参数（371, 410）；`pool_scheduler.py:177-180` 注释要求 `keys_per_block_hash` 与 worker 保持同步（影响 GVA 分配大小）。
- `bind_gpu_block_pool`：`ascend_store_connector.py:274-276` → `pool_scheduler.py:1070-1071` 存 `BlockPool` 引用。唯二用途：mamba hybrid 下 `touch_sending_mamba_blocks` touch 待发送块（983-986）与 `update_connector_output` 在所有 worker（`_expected_worker_count = world_size`，142）报齐 event 后 `free_blocks`（996-1010）。upstream 调用点：scheduler 初始化 `scheduler.py:282`（upstream±）。

## Q8. request_finished / request_finished_all_groups / shutdown / save 路径

### request_finished（`pool_scheduler.py:1012-1037`）`[当前源码已确认]`

- `kv_consumer && !consumer_is_to_put` → `(False, None)`（1021-1023）；`use_layerwise` → `(False, None)`（1024-1026）；tracker 缺失或 `num_saved_tokens <= 0` → `(False, None)`（1027-1030）。
- 否则 `delay_free_blocks = len(block_ids) > 0`，为真则 `_delayed_free_req_ids.add`（1031-1037）→ 块延迟到 `get_finished` 的 done_sending 上报后才由 upstream 释放。

### request_finished_all_groups（1039-1068）`[当前源码已确认]`

- HMA 路径（connector 声明 `SupportsHMA`，`ascend_store_connector.py:73`；upstream 分发 `scheduler.py:2461-2469`，SupportsHMA 走 all_groups）。
- 同样 consumer/layerwise 早退（1045-1051）；SW 裁剪（1056）后按非空 group 决定 delay_free（1057-1068）。
- **与 request_finished 的不对称**：`if tracker is not None and tracker.num_saved_tokens <= 0`（1053）——tracker 为 None 时**不早退**，会继续按 block_ids 决定 delay_free；而单 group 版 tracker 为 None 直接 `(False, None)`（1028）。对「从未进过 build_connector_meta 就结束」的请求，all_groups 可能对其块做 delay-free 但发送线程永远不会上报该 req_id。

### shutdown `[当前源码已确认]`

- `AscendStoreConnector` 未覆写 `shutdown`（基类 no-op，`base.py:395-401`）。`LookupKeyServer.close` / `LookupKeyClient.close` 无生产调用方；传输线程与 lookup 线程均 daemon；无任何显式资源回收路径，依赖进程退出。

### Store save（put）路径存在性与配置控制 `[当前源码已确认]`

- 存在。非 layerwise：`KVCacheStoreSendingThread` 仅当 `kv_role in ["kv_producer","kv_both"] or consumer_is_to_put` 时创建（`pool_worker.py:438-454`）。
- 触发链：forward 退出时框架调 `connector.wait_for_save`（`ascend_store_connector.py:235-243`）——`kv_consumer && !consumer_is_to_put` 跳过；`use_layerwise` 跳过（layerwise 走逐层 `save_kv_layer`，224-233，kv_consumer 同样跳过 230-232）→ `KVPoolWorker.wait_for_save`（1364-1394）记录 NPU event、逐请求入队、`request_queue.join()` 屏障（1391-1394，注释明确为让后续相同 prompt 的 lookup 可见）。
- Scheduler 侧 `force_skip_save = kv_consumer && !consumer_is_to_put`（894）使 ReqMeta `can_save=False`；`save_decode_cache`（默认 False，94-96）门控 decode 增量保存（794-796）；chunked-prefill 增量总是保存（791-792 注释）。
- 发送线程行为：exists 去重（`kv_transfer.py:725-729`）、`current_event.synchronize()` 后 `m_store.put`（803-805）、完成后计数归零 `set_finished_request`（813-816）→ done_sending 释放延迟块。

## 补充：对请求状态/完成集合影响一览 `[当前源码已确认]`

| 状态/集合 | 写入点 | 清理点 |
|---|---|---|
| `load_specs[req_id]` | `get_num_new_matched_tokens` 549-553 | 三个 _process_* pop（690/742/845）；**abort 无清理** |
| `_unfinished_requests/_unfinished_request_ids` | `update_state_after_alloc` 578-579 | `build_connector_meta` finished 896-899 / preempted 903-908 |
| `_loading_req_ids` | `update_state_after_alloc` 617-618（load_async 且非 layerwise） | finished/preempted（901/907）；`update_finished_recving` 无生产调用方 |
| `_delayed_free_req_ids` | `request_finished{,_all_groups}` 1033/1060 | done_sending 经 upstream 释放；`update_finished_sending` 无生产调用方 |
| `_invalid_block_ids`（worker） | 同步 load 失败 846-847/861-862 | `get_block_ids_with_load_errors` 1333-1337 每步 drain |
| `_invalid_block_ids`（recv 线程私有） | 异步 load 失败 `kv_transfer.py:908-910/924-926` | **无任何 drain 方**（见 Q5 断裂） |
| `sending_events/sending_blocks` | `touch_sending_mamba_blocks` 985-986 | `update_connector_output` 凑齐 world_size 后 free（996-1010） |

# Phase 1 证据 03：MooncakeLayerwiseConnector 控制流重建

- 主文件：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`（2090 行，下称 `MLC`）
- 仓库：vllm-ascend @ `0ec11a47`（branch `dev/dualpath`）
- upstream 参考：`/Users/leqi/Documents/Code/vllm` @ `8df14cfc`（v0.23.1rc0-1050）；vllm-ascend 官方配套 v0.23.0，涉及 upstream 的结论均「参考版本存在小幅偏差」
- 本文件只记录当前源码事实，不评审设计。标注约定：`[当前源码已确认]` / `[源码可达，尚未运行验证]` / `[外部接口待确认]` / `[事实冲突]`。凡未标注 upstream 路径的行号均指 `MLC` 内行号。

## 0. 文件结构与符号清单

| 符号 | 行号 | 说明 |
|---|---|---|
| `LayerMetadata` | 87-92 | tensor_group_idx / kv_caches_base_addr / block_len / block_size_scale |
| `MooncakeAgentMetadata` | 95-97 | msgspec Struct：te_rpc_port + layer_metadata，握手时发给对端 |
| `ReqMeta` | 100-122 | 每请求传输元数据（含 chunk_finish/prompt_len/trans_count/local_computed_tokens 等） |
| `SendTask` | 125-144 | 逐层发送任务：send_request dict + wait_event + k/v(/quant) buffer + group_* 重排信息 |
| `SendReqInfo` | 155-181 | scheduler 侧按步累积的 send 状态（block ids、transferred/computed tokens） |
| `KVCacheSendingLayerThread` | 204-527 | daemon 发送线程 |
| `KVCacheRecvingLayerThread` | 530-653 | daemon 接收/握手线程（zmq ROUTER） |
| `MooncakeLayerwiseConnectorMetadata` | 656-694 | requests: dict[str, ReqMeta] + send_task: SendTask |
| `MooncakeLayerwiseConnector` | 697-787 | `KVConnectorBase_V1, SupportsHMA`，按 KVConnectorRole 分发 |
| `MooncakeLayerwiseConnectorScheduler` | 790-1124 | scheduler 侧实现 |
| `MooncakeLayerwiseConnectorWorker` | 1127-1977 | worker 侧实现 |
| `get_external_request_id` | 2087-2090 | `request_id[:-9]`，剥掉 EngineCore 9 字符后缀 |
| `GET_META_MSG` | `mooncake_connector.py:81` | `b"get_meta_msg"`；`DONE_SENDING_MSG`/`FAILED_SENDING_MSG` 定义于 MLC:83-84 |

role 判定来自 upstream：`KVProducer = Literal["kv_producer","kv_both"]`、`KVConsumer = Literal["kv_consumer","kv_both"]`（upstream `vllm/config/kv_transfer.py:11-12`），`is_kv_producer`/`is_kv_consumer` 为 property（同文件 113-118 行）。**kv_both 下两者同时为 True** `[当前源码已确认]`（参考版本存在小幅偏差）。

## 1. Scheduler 侧

调用点（upstream `vllm/v1/core/sched/scheduler.py`，参考版本存在小幅偏差）：
- `get_num_new_matched_tokens`：scheduler.py:775-779（waiting 循环，新请求准入时）`[当前源码已确认]`
- `update_state_after_alloc`：scheduler.py:969-974（分配块之后立即调用）`[当前源码已确认]`
- `load_kv_async=True` 时请求置 `WAITING_FOR_REMOTE_KVS` 并跳过本轮：scheduler.py:986-990 `[当前源码已确认]`

### 1.1 `get_num_new_matched_tokens()`（MLC 893-926）

**不看 role，只看 `request.kv_transfer_params`** `[当前源码已确认]`：

- `params.get("do_remote_prefill")`（consumer 拉取语义）：断言 `num_computed_tokens % min(block_size) == 0`（918 行），返回 `(count, count > 0)`，`count = max(_hybrid_prefill_token_count(len(prompt_token_ids)) - num_computed_tokens, 0)`（916-920）。即返回**全部待拉取 prompt token 数**（hybrid 时减 1），且 async=True → 请求进 `WAITING_FOR_REMOTE_KVS`。
- `params.get("do_remote_decode")`（producer 推送语义）：先调 `_truncate_request_for_hybrid_prefill(request)`（仅 hybrid 生效，见 §6），返回 `(0, False)`（922-923）。producer 从不声明外部命中。
- 其他：`(0, False)`（926）。
- kv_both：行为同样完全由 params 驱动；`do_remote_prefill` 分支先于 `do_remote_decode` 判断，同一请求两标记同真时走 prefill 分支 `[当前源码已确认]`。

### 1.2 `update_state_after_alloc()`（MLC 928-998）

- `do_remote_prefill` 分支（936-982）：
  - `local_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else []`（938）；`remote_block_ids = _trim_hybrid_remote_block_ids(...)`（939）。
  - 请求入 `self._reqs_need_recv[request_id] = (request, [], local_block_ids)`（945-949）。
  - `params["do_remote_prefill"] = False`（951）——一次性消费标记。
  - 构造回传 P 侧的 `kv_transfer_params`（`do_remote_decode=True`、remote_block_ids/host/port/tp/pcp/dcp/remote_cached_tokens 等，957-972）；非 `do_virtual` 时经 `ThreadPoolExecutor(32)` 异步 POST metaserver（820, 973-982，重试 3 次见 `_access_metaserver` 1089-1100）。
- `do_remote_decode` 分支（985-998）：入 `self._reqs_need_send_layerwise[request_id] = SendReqInfo(local_block_ids, local_transferred_tokens=remote_cached_tokens, local_computed_tokens=0, request=request)`。
- 两分支互不排斥（顺序 if，非 elif）；副作用：请求状态字典变更 + 后台线程池 HTTP 调用 `[当前源码已确认]`。

### 1.3 `build_connector_meta()`（MLC 1000-1087）

产出 `MooncakeLayerwiseConnectorMetadata{requests, send_task}`（1004；结构见 656-659）。

- **`if self.vllm_config.kv_transfer_config.is_kv_consumer:`（1006）**：把 `_reqs_need_recv` 全部 `add_new_req`（token_ids=[]，chunk_finish=False 默认）后 `clear()`（1008-1021）。
- **`else:`（1022，即纯 producer）**：遍历 `scheduler_output.scheduled_cached_reqs/scheduled_new_reqs`：
  - 新分配块并入 SendReqInfo（1026-1028）；
  - 更新 transferred/computed tokens（computed = 已计算 + 本轮调度 − spec_decode；1033-1046）；
  - `add_transfer_task` → `meta.add_new_req(..., chunk_finish=..., prompt_len=len(request.all_token_ids), remote_cache_tokens, local_computed_tokens, local_transed_tokens)`（1048-1084）；
  - `chunk_finish = local_computed_tokens >= len(request.all_token_ids)`（1082）；为真则从 `_reqs_need_send_layerwise` pop（1085-1086）。
- **kv_both 关键事实**：`is_kv_consumer` 为 True → 恒走 consumer 分支，`else`（producer）分支不可达；`_reqs_need_send_layerwise` 中的条目永远不会被消费/弹出（观察到的代码事实，kv_both 下 producer 推送链路在 scheduler 侧断裂）`[当前源码已确认]`。

### 1.4 `request_finished()` / `request_finished_all_groups()`

- `request_finished`（1102-1112）：无条件 `return False, None`。
- `request_finished_all_groups`（1114-1124）：无条件 `return False, None`。
- 设计声称「父类 `request_finished_all_groups` 恒定返回 False」：
  - 若指 `SupportsHMA.request_finished_all_groups`——那是**抽象方法**，`raise NotImplementedError`（upstream `vllm/distributed/kv_transfer/kv_connector/v1/base.py:92-114`），并非"返回 False" `[事实冲突]`（参考版本存在小幅偏差）；
  - 若指本 connector（DualPath 方案的直接父类）的 scheduler 实现——**属实**，恒 `(False, None)`（1114-1124）`[当前源码已确认]`。
  - 另：`KVConnectorBase_V1.request_finished` 默认实现也是 `return False, None`（base.py:547-566）。
- 实际被调度器调用的是哪一个：upstream `Scheduler._connector_finished`（scheduler.py:2434-2469）——connector 是 `SupportsHMA` 实例时调 `request_finished_all_groups`；MLC 继承 `SupportsHMA`（697）→ **实际走 `request_finished_all_groups`，恒返回不延迟释放，块立即进入 free 流程**（`_free_request` scheduler.py:2162, 2179-2181）`[当前源码已确认]`（参考版本存在小幅偏差）。

## 2. Worker 侧 `register_kv_caches()`（MLC 1247-1398）

调用方：`NPUModelRunner.initialize_kv_cache` 末尾 `get_kv_transfer_group().register_kv_caches(kv_caches)`（`vllm_ascend/worker/model_runner_v1.py:3824-3825`），**每 worker 进程恰好一次**；v2 runner 无第二调用点 `[当前源码已确认]`。

一次调用内完成（顺序即代码顺序）：

1. layer→group 映射（1249-1253）；扫描 spec 检测 attn+mamba 混合（`use_attn_mamba_hybrid`，1255-1270）。
2. 逐层构建 `LayerMetadata`（tensor_group_idx/base_addr/block_len/block_size_scale，1292-1313），并更新 `kernel_block_size_scale`（1306）；`(pd_head_ratio>1 且 FullAttention/SlidingWindow) 或 quant` 的 group 记入 `attn_resharding_group_idx` 并选定 kv_buffer 来源层（1283-1291）。
3. 内存注册（一次调用内只注册一轮）：
   - hybrid：`RegisterRegions(ptrs=每 tensor 最小对齐地址, lengths=tensor.size)`（1315-1324, 1326-1327）；
   - 非 hybrid：`collect_storage_merged_register_regions(kv_caches)`（1329-1331；storage 合并实现见 `utils/utils.py:363-425`）；
   - `validate_register_region_count`（1333；超 `MAX_HCCL_REGISTER_REGIONS` 抛错，`utils/utils.py:428-445`）；
   - `global_te.register_buffer(ptrs, lengths)`（1334）。**注意：`GlobalTE.register_buffer` 有 `is_register_buffer` 单次守卫，进程内第二次调用直接 return 不注册**（`utils/mooncake_transfer_engine.py:31-40`）`[当前源码已确认]`。
4. `use_kv_buffer`（pd_head_ratio>1 或 enable_kv_quant/c8）时 `create_kv_buffer`：额外 2 次 `engine.register_memory`（k_buffer/v_buffer，各 2MiB 对齐，不经 global_te 守卫，1217-1245, 1336-1337）。
5. `index_to_name` 建立（不支持同层多 attn_module，断言 1346-1348；mtp 层排最后，1349-1350）；`total_layers` 校正（1351-1352）。
6. 构造 `MooncakeAgentMetadata(te_rpc_port, layer_metadata)`（1355-1358）。
7. **`if is_kv_producer:`（1359-1384）**：创建并 `start()` `KVCacheSendingLayerThread`（daemon，`ready_event.wait()` 阻塞至线程就绪），`callback_func=self.send_done_send_signal`。
8. **`if is_kv_consumer:`（1386-1398）**：创建并 `start()` `KVCacheRecvingLayerThread`（同样 ready_event 同步）。

**kv_both：两个 if 均命中，一次 `register_kv_caches` 调用内顺序完成 buffer 注册 + 发送线程 + 接收线程创建** `[当前源码已确认]`——设计声称"kv_both 下一次调用即可完成全部注册"对本 connector 成立；但 `is_register_buffer` 守卫意味着任何第二次 register_buffer 路径会被静默跳过（对本 connector 的正常单调用流程无影响）。

## 3. `start_load_kv()` dispatch（MLC 1603-1688）

调用时机：worker 进程、forward 之前——upstream `KVConnectorModelRunnerMixin._get_kv_connector_output`（`vllm/v1/worker/kv_connector_model_runner_mixin.py:89-95`：先 `bind_connector_metadata` 再 `start_load_kv`）；Ascend 侧经 `maybe_get_kv_connector_output` 包裹 forward（`model_runner_v1.py:2258-2263`）`[当前源码已确认]`（参考版本存在小幅偏差）。

- 首行 `self.current_layer = 0`（1605）。
- **`if is_kv_consumer:`（1606-1614）**：`do_virtual` 请求入 `virtual_request` 并 skip；否则 `request_map[external_req_id] = req_id`、`_recving_metadata[req_id] = meta`。
- **`elif is_kv_producer:`（1615-1688）**：`_align_remote_block_ids`（1570-1589）→ 逐 group 计算 transfer_mappings（attention 走 `_get_kv_split_metadata` 1441-1550，mamba 走 `_get_kv_split_metadata_for_mamba` 1552-1568）→ `assert len(transfer_mappings) <= 1`（1644）→ 展开 kernel block ids（`_get_kernel_block_ids` 1591-1601）→ 重写 metadata.requests；pd_head_ratio≠1 或 quant 时构建 `send_task.group_*`（重排 block 表/长度 tensor，1657-1688）。
- **设计声称「kv_both 下走 consumer 优先分支、跳过 P 侧 producer 映射」：属实**——if/elif 结构（1606/1615）使 kv_both 恒走 consumer 分支，producer 的 transfer_mappings 计算整体不执行 `[当前源码已确认]`。
- 补充：kv_both 下 `save_kv_layer` 的门是 `is_kv_producer`（1699，kv_both 为 True），即 save 路径仍会执行；但送入的 `metadata.requests` 只会是 build_connector_meta consumer 分支产出的 recv 型请求（见 §1.3）。

## 4. `save_kv_layer()` / `wait_for_layer_load()`

- `save_kv_layer`（1690-1837）：Worker 进程、forward 期间逐层调用。调用链：attention impl → `maybe_save_kv_layer_to_connector`（`vllm_ascend/attention/utils.py:442-456`）→ `connector.save_kv_layer`；调用点例：`dsa_v1.py:1736`、`dsa_cp.py:1271`、fla `gdn_310.py:429`（以 `("", [])` 形式）`[当前源码已确认]`。
  - 门：`is_kv_producer and connector_metadata.requests.keys()`（1699）；`current_layer >= total_layers` 时只自增并返回（1700-1702）。
  - 取/建 `reshape_cache_event`（1706-1720）；pd_head_ratio≠1/quant 时在 `resharding_stream` 上 `npu_paged_cache_load` 聚合 + 可选 `kv_alltoall_and_rearrange`/量化（1727-1806）。
  - 构造 `SendTask`（1810-1819），逐请求 `update_decoder_info` 补齐对端 metadata（1820-1834，失败仅 warning 并跳过该请求），`self.kv_send_layer_thread.send_queue.put(layer_send_task)`（1836），`current_layer += 1`（1837）。
  - 同步性：`send_queue.put` 在 pd_head_ratio≠1 时 maxsize=1（或 mamba hybrid 的 len(specs)），满则阻塞 forward 线程；pd_head_ratio==1 时 maxsize=0 无界（251-257）`[当前源码已确认]`。
- `wait_for_layer_load`（1976-1977）：**`pass`，纯 no-op**。调用链：`wait_for_kv_layer_from_connector`（`attention/utils.py:428-439`）→ `connector.wait_for_layer_load`；调用点 `mla_v1.py:1684`、`sfa_v1.py:1611/1643/1672`、`dsa_v1.py:1702`、`dsa_cp.py:1200`（均在 attention 计算前）`[当前源码已确认]`。即本 connector 的层间 load 等待在 Ascend attention 路径上被调用但无任何效果——consumer 的 KV 拉取完成完全依赖 scheduler 侧 `WAITING_FOR_REMOTE_KVS` 门禁，而非 worker 侧逐层等待。

## 5. `get_finished()` / invalid blocks / 失败语义

Worker 侧组装（upstream `kv_connector_model_runner_mixin.py:96-112`）：forward 后 finally 中 `get_finished(scheduler_output.finished_req_ids)` → `output.finished_sending/finished_recving`；`get_block_ids_with_load_errors()` → `output.invalid_block_ids`（102-105）`[当前源码已确认]`（参考版本存在小幅偏差）。

`MooncakeLayerwiseConnectorWorker.get_finished()`（1400-1428）：

- `done_recving`：仅 consumer 时从 `kv_recv_layer_thread.get_and_clear_done_requests()` 取（1401-1406）；经 `request_map` 映射回内部 req_id（1407），并入 `virtual_request`（1408-1409）。
- `failed_recving`：`get_and_clear_failed_requests()`（1411-1416）；**失败请求的全部 local_block_ids 并入 `self._invalid_block_ids`**（1417-1419）。
- 清理：`done_recving ∪ failed_recving` 的请求清掉 `request_map`/`_recving_metadata`（1420-1423）。
- **返回 `(set(), done_recving)`——`finished_sending` 恒为空集**（1428）；失败请求**不进** `done_recving`，即「父 Worker 只上报正常完成请求」**属实** `[当前源码已确认]`。
- `get_block_ids_with_load_errors`（1430-1438）：返回并清空 `_invalid_block_ids`。这是失败接收的唯一暴露接口 `[当前源码已确认]`。

失败信号链：发送线程 `batch_transfer_sync_write` 返回 ret<0（497-507）→ 末层 chunk_finish 时 `callback_func(..., trans_flag=False)`（519-527）→ `send_done_send_signal` 发 `FAILED_SENDING_MSG`（1920-1935）→ consumer 接收线程 `update_failed_task`（639-643, 577-587）→ 上述 failed 路径 `[当前源码已确认]`。

scheduler 侧消费（upstream scheduler.py，参考版本存在小幅偏差）：
- `finished_recving`：`_update_from_kv_xfer_finished`（2559-2586）——`WAITING_FOR_REMOTE_KVS` → 入 `finished_recving_kv_req_ids`（下一步 `_try_promote_blocked_waiting_request` → `_update_waiting_for_remote_kv` 缓存块并转 WAITING，2492-2541）；否则要求状态已 finished 并 `_free_blocks`（2578-2582）。
- `invalid_block_ids`：`_handle_invalid_blocks`（1579-1586, 2691-2760）——截断受影响请求的 `num_computed_tokens`；async load 失败请求入 `failed_recving_kv_req_ids`（2758），后续在 `_update_waiting_for_remote_kv` 走失败分支（2502-2513）；按 failure_policy 决定重算或 `FINISHED_ERROR`（2739-2748, 1825-1835）`[当前源码已确认]`。

**terminal 早于 mapping 到达时的行为：事件被丢弃，不暂存** `[当前源码已确认]`：
- 接收线程把 DONE 计入 `done_requests` 并 ACK（555-564, 632-638）；`get_finished` 在 1407 行用 `if s in self.request_map` 过滤——`start_load_kv` 尚未注册映射（1613）的 DONE 事件在此被**永久丢弃**（done 集已被 `get_and_clear` 清空）。
- 顺序性质：单步内 `start_load_kv` 一定先于 `get_finished`（mixin 95 vs 102-104）`[当前源码已确认]`；但跨步（DONE 在携带该请求 metadata 的 step 之前到达）无任何防护——**属当前实现的偶然顺序，非框架保证**。后果：该请求永远收不到 finished_recving，滞留 `WAITING_FOR_REMOTE_KVS`（直至 abort/超时）。
- 另一处相关顺序：`update_done_task` 按 `side_channel_path` 计数，凑满 `trans_count`（P 侧发来，`send_done_send_signal` 1935 行取 `req_meta.trans_count[group_idx]`）才置 done（589-601）。

## 6. 最后一个 prompt token 截断

**截断逻辑全部在本 connector scheduler 内，且仅以 attn+mamba 混合 cache 为条件** `[当前源码已确认]`：

- 条件：`self.need_truncate = self._has_attn_mamba_hybrid_cache(kv_cache_config)`（819；判定 845-852：spec 中同时存在 `AttentionSpec` 与 `MambaSpec`）。
- `_hybrid_prefill_token_count`（854-857）：`need_truncate and num_prompt_tokens > 1` 时返回 `num_prompt_tokens - 1`——consumer 拉取计数减 1（919 行使用）。
- `_truncate_request_for_hybrid_prefill`（859-879）：pop 最后一个 prompt token（或截 prompt_embeds）、`_all_token_ids.pop()`、`num_prompt_tokens -= 1`、`max_tokens = 1`、`params["_p_side_truncated"] = True` 防重入；调用点仅在 `get_num_new_matched_tokens` 的 `do_remote_decode` 分支（922-923）。
- `_trim_hybrid_remote_block_ids`（881-891）：`prompt_len % block_size == 1` 时去掉末尾 block；调用点 `update_state_after_alloc`（939）。
- 设计声称「父类只对 hybrid cache 特例截断」**属实**（条件即 `_has_attn_mamba_hybrid_cache`，上述三处）`[当前源码已确认]`。
- upstream 框架内无同类"末 token 截断"；另有通用逻辑 `scheduler.py:2519-2522`（full prompt hit 时 `num_computed_tokens -= 1` 以重算末 token 供采样），与 hybrid 无关 `[当前源码已确认]`（参考版本存在小幅偏差）。

## 7. transfer engine / 端口 / engine_id / 线程主循环

- **TransferEngine**：`global_te`（`utils/mooncake_transfer_engine.py:43`）进程级单例，`get_transfer_engine` 双重检查锁懒建并 `initialize(hostname, "P2PHANDSHAKE", "ascend", "")`（11-29）；worker `__init__` 获取并取 `te_rpc_port = engine.get_rpc_port()`（1172-1173）。`TransferEngine`/`batch_transfer_sync_write`/`register_memory`/`get_rpc_port` 为 mooncake 外部库接口 `[外部接口待确认]`（语义按名单边 RDMA 写理解，源码不可达于本仓库）。
- **engine_id**：`kv_transfer_config.engine_id`（702），缺省 `uuid4`（upstream `kv_transfer.py:30, 92-94`）；worker 初始化时 `_sync_engine_id_across_tp` 在 TP 内同步（upstream `kv_transfer_state.py:87-94`）`[当前源码已确认]`（参考版本存在小幅偏差）。
- **端口推导**：
  - scheduler：`side_channel_port = kv_port + dp_rank * tp_size`（809-812）；
  - worker：`side_channel_port = kv_port + dp_rank * pcp_size * tp_size + pcp_rank * tp_size`（1164-1168），`handshake_port = side_channel_port + tp_rank`（1169）；
  - 接收线程监听 `side_channel_port + tp_rank`（605-607）；P 侧 DONE 信号发往映射给出的 `(remote_host, remote_port)`（`get_local_remote_block_port_mappings`，`utils/utils.py`）。
- **wire identity**：`session_id = f"{remote_host}:{remote_te_rpc_port}"`（470, 1906）；远端 metadata 经 zmq REQ `GET_META_MSG` 握手拉取并缓存于 `remote_layer_metadata/remote_te_port`（`SizedDict` 上限 16000，184-201, 1869-1918）；pd_head_ratio>1 时握手后立即发 128B 试写建链（1904-1915）。
- **发送线程主循环**（`KVCacheSendingLayerThread`，204-527；daemon 228）：
  - `run()`：`send_queue.get()` 阻塞，逐 `SendTask` 处理（266-273）；异常仅 log 不退出（275-283）。
  - `_transfer_kv_cache`（447-527）：可选把 resharding/quant 数据拷入 k_buffer/v_buffer（452-465）；按 session 合并多请求 TransferMeta（467-479）；等待 reshape/缓存写完事件（`wait_event.synchronize()` 或 `resharding_stream.synchronize()`，481-492，含 ADXL hang 的 CANN 版本注释 483-489）；逐 session `engine.batch_transfer_sync_write`（497-499）；末层（`layer_idx == total_layers - 1`）且 `chunk_finish` 的请求触发 callback（519-527）。
  - 注意（代码事实）：ret<0 时 `self.failed_reqs.add(req_id)`（507）中的 `req_id` 是 469 行 `for req_id, req_meta in send_task.send_request.items()` 循环的**残留变量**，并非当前失败 session 的 req——多请求同 task 时失败归因可能不准 `[当前源码已确认]`。
- **接收线程主循环**（`KVCacheRecvingLayerThread`，530-653；daemon 541）：zmq ROUTER 收三类消息——`GET_META_MSG`→回 `MooncakeAgentMetadata`；`DONE_SENDING_MSG`→`update_done_task`+ACK；`FAILED_SENDING_MSG`→`update_failed_task`+ACK（603-653）。DONE 判定：同一 req 的不同 `side_channel_path` 凑满 `trans_count`（589-601）。

## 8. shutdown / drain / 请求取消

- connector **未重写 `shutdown`**；基类默认 `return None`（upstream base.py:395-401）`[当前源码已确认]`（参考版本存在小幅偏差）。worker 退出路径：`vllm_ascend/worker/worker.py:376-378` → `ensure_kv_transfer_shutdown()` → `_KV_CONNECTOR_AGENT.shutdown()`（upstream `kv_transfer_state.py:97-101`）→ 对本 connector 为 no-op。**无 drain**：send_queue 积压任务、在途 DONE/FAILED 信号、metaserver 线程池（scheduler 侧 820）均无显式等待/清理；两个传输线程为 daemon，随进程退出（228, 541）`[当前源码已确认]`。
- 请求取消/abort：
  - connector 自身无任何取消钩子；`_reqs_need_recv`/`_recving_metadata`/`request_map`/`_reqs_need_send_layerwise` 中的条目只在 get_finished 的 done/failed 路径清理（1420-1423, 1085-1086）`[当前源码已确认]`。
  - upstream 对 `WAITING_FOR_REMOTE_KVS` 请求的 abort：`finish_requests` 置 `delay_free_blocks=True`（recv 未完成时）并跳过立即 `_free_blocks`（scheduler.py:2143-2152, 2179-2181）；之后该 req_id 若出现在 `finished_recving` 且状态已 finished → `_free_blocks`（2574-2582）`[当前源码已确认]`（参考版本存在小幅偏差）。即框架依赖"终端事件迟早到达"来释放块；结合 §5 的早到事件丢弃问题，若 DONE 事件丢失，该释放路径不会再触发。
  - `handle_preemptions`：基类默认 no-op（base.py:285-290），MLC 未重写；调用点 `model_runner_v1.py:1969` `[当前源码已确认]`。

## 9. 其它已核实的横切事实

- `MooncakeLayerwiseConnector.__init__`（697-712）：SCHEDULER/WORKER 两 role 分别只实例化对应半侧；`_connector_metadata` 初始为空 `MooncakeLayerwiseConnectorMetadata()`（703）。
- scheduler `__init__` 断言：consumer（含 kv_both）下 `prefill_context_parallel_size == 1`（803-806）`[当前源码已确认]`。
- `wait_for_save`（785-787）：`pass`（mixin 在每步 finally 调用，实际无效果）。
- `get_finished` 的 connector 外壳（756-759）忽略传入的 `finished_req_ids` 参数。
- worker `__init__` 设置 `ASCEND_TRANSFER_TIMEOUT` 环境变量（1131；取值逻辑 `utils/utils.py:55-61`，默认由 HCCL_RDMA_TIMEOUT/RETRY_CNT 推导）。
- `SizedDict`（184-201）max 16000 条，LRU 弹出最旧——远端 metadata 缓存有上限。
- 本文件结论均未做运行验证；静态阅读基于 HEAD `0ec11a47` 工作区源码。

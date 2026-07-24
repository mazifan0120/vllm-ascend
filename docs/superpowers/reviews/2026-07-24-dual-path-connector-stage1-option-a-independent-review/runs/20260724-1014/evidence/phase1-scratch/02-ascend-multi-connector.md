# Phase 1 证据：AscendMultiConnector 与 upstream MultiConnector 控制流重建

- 收集时间：2026-07-24
- vllm-ascend：`/Users/leqi/Documents/Code/vllm-ascend`，branch `dev/dualpath`，HEAD `0ec11a47`（v0.19.1rc1-991）
- upstream vllm：`/Users/leqi/Documents/Code/vllm`，branch `main`，HEAD `8df14cfc`（describe: `v0.23.1rc0-1050-g8df14cfc8`）
- **版本偏差警告**：vllm-ascend 官方配套 vllm v0.23.0，本地 upstream 检出为 v0.23.1rc0-1050。`multi_connector.py` 在 v0.23.0 之后有 3 个相关提交（见 §8），凡引用 upstream 行号均以本地检出为准，与 v0.23.0 的差异单独标注。

标记约定：`[当前源码已确认]`（直接读到代码）/ `[源码可达，尚未运行验证]` / `[外部接口待确认]` / `[事实冲突]`。

涉及文件：

- A = `vllm-ascend/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`（全文 102 行，已通读）
- B = `vllm/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py`（全文 667 行，已通读）
- C = `vllm/vllm/v1/core/sched/scheduler.py`（仅调用点）
- D = `vllm-ascend/vllm_ascend/distributed/kv_transfer/__init__.py`（注册）
- E = `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`（仅相关方法）

---

## 1. Scheduler 侧 `get_num_new_matched_tokens()` 精确逻辑

### 1.1 框架调用点

- C:774-787：waiting 循环中 `ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)`。`ext_tokens is None` 时请求被弹出并放回 `step_skipped_waiting`，本轮跳过（C:781-787）`[当前源码已确认]`。
- 每轮 schedule 对 waiting 请求都会重新调用；返回值不落盘，winner 选择状态只存在 connector 的 `self._requests_to_connector` dict 里（B:198）。

### 1.2 AscendMultiConnector 覆写（A:43-63）

```
for i, connector in enumerate(self._connectors):
    has_preempted_request = getattr(connector, "has_preempted_request", None)
    if has_preempted_request is None or not has_preempted_request(request.request_id):
        continue
    tokens, load_async = connector.get_num_new_matched_tokens(request, num_computed_tokens)
    if tokens is None:  return None, False
    if tokens > 0:      self._requests_to_connector[request.request_id] = i; return tokens, load_async
    break
return super().get_num_new_matched_tokens(request, num_computed_tokens)
```

事实 `[当前源码已确认]`：

- 这是一段"recompute-offload 优先"前置扫描（注释 A:48-50）。`has_preempted_request` 是 duck-typing；全仓库仅 `RecomputeCPUOffloadConnectorV1` 实现（`vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/recompute_cpu_offload_connector.py:233`，委托 `manager.py:176`）。若 child 中无 RecomputeCPUOffload，整段循环为 no-op，直接走 upstream。
- 命中优先路径的 child（第一个对该 request 报 preempted 的）：`None` → 立即返回 `(None, False)`；`>0` → 写入 winner 映射并直接 return（**不再调用其余 child**）；`==0` → `break` 后落入 upstream 全量扫描（upstream 可能另选 winner）。
- 优先路径与 upstream 路径**不会叠加调用同一 child 两次以上**：优先路径 return 时其余 child 本轮根本未被调用；break/落空时才由 super() 对所有 child 各调一次。

### 1.3 upstream `MultiConnector.get_num_new_matched_tokens`（B:381-400）

事实 `[当前源码已确认]`：

- 按 `_connectors` 配置顺序**逐个调用所有 child**，即使已选出 winner 也继续调用后续 child（循环无 break；后续调用仅其返回值被忽略，child 内部 lookup 副作用仍发生）。
- `None` 处理：任一 child 在**任何位置**返回 `None` → 立即 `return (None, False)`（B:393-394）。注意：若 winner 已在先前迭代写入 `_requests_to_connector[request.request_id]`（B:398），该映射**不会被清除**，但本次返回值被 None 覆盖，scheduler 本轮跳过该请求；下轮重查时映射会被重写。
- `0` 处理：`to_return = (0, False)` 初始化；winner 判定条件 `to_return[0] == 0 and toks > 0`（B:397）——**第一个返回正数的 child 成为 winner**（first-wins，按配置顺序）；之后返回正数的 child **不会覆盖** winner 结果（`to_return[0]` 已非 0）。全部返回 0 → 返回 `(0, False)`，`_requests_to_connector` 无记录。
- `load_async` 取自 winner 的返回值（B:399）。

## 2. `update_state_after_alloc()`

### 2.1 框架调用点

- C:969-973：`self.connector.update_state_after_alloc(request, self.kv_cache_manager.get_blocks(request_id), num_external_computed_tokens)`——传入的是该请求**当前全部已分配 blocks**（不止 external 部分）与 winner 报告的 external token 数 `[当前源码已确认]`。

### 2.2 AscendMultiConnector 覆写（A:32-41）——设计声称验证

```python
chosen_connector = self._requests_to_connector.get(request.request_id, -1)
empty_blocks = blocks.new_empty()
for i, c in enumerate(self._connectors):
    if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
        c.update_state_after_alloc(request, blocks, num_external_tokens)   # 真实 blocks + 真实 tokens
    else:
        c.update_state_after_alloc(request, empty_blocks, 0)               # 空 blocks + 0
```

- **设计声称「即使 Layerwise Connector 不是 token winner 也会获得真实 KV blocks」→ 属实** `[当前源码已确认]`：A:36-38，`isinstance(c, MooncakeLayerwiseConnector)` 分支无条件转发真实 `blocks` 与真实 `num_external_tokens`。`KVCacheBlocks.new_empty()` 在配套 v0.23.0 已存在（v0.23.0 tag 的 `kv_cache_manager.py:103`）`[当前源码已确认]`。
- **非 winner 的 AscendStore child：收到 `empty_blocks` + `0`**（A:41）`[当前源码已确认]`。AscendStore scheduler 侧对 `num_external_tokens == 0` 的处理（`kv_pool/ascend_store/pool_scheduler.py:567-597`）：仍把请求登记进 `_unfinished_requests`（block ids 为空，:578-579）；若该请求有 load_spec（即 store 自己 lookup 时曾命中）且 `use_layerwise and kvpool_cached_tokens > 0`，**仍会把 `load_specs[req].can_load` 置 True**（:589-593）——非 winner store 在 layerwise 模式下并非完全 no-op `[当前源码已确认]`（其下游是否真发起 load 未追踪，`[源码可达，尚未运行验证]`）。
- 无 winner（chosen = -1）时：所有非 Mooncake child 走 else 分支（empty+0）；Mooncake child 仍收到真实 blocks + 真实 `num_external_tokens`（此情形通常为 0）。
- MooncakeLayerwise scheduler 侧 `update_state_after_alloc`（E:928-949+）：仅在 `request.kv_transfer_params["do_remote_prefill"]` 为真时有副作用（加入 `_reqs_need_recv`，E:945）；`local_block_ids = blocks.get_block_ids() if num_external_tokens > 0 else []`（E:938）。**边缘情形**：Mooncake 非 winner 但该请求带 `do_remote_prefill` 参数时，它会用真实 blocks + **winner 的** num_external_tokens 入队 recv `[当前源码已确认]`（是否实际发生取决于部署形态，`[源码可达，尚未运行验证]`）。

### 2.3 upstream 基类版本（B:402-412）

- 本地检出（v0.23.1rc0-1050）：**非 winner child 收到真实 blocks + 0**（B:410-412，注释 "Other connectors still receive the request's real blocks"）。
- **版本偏差**：该行为来自 v0.23.0 之后的提交 `2285cfca4`（#46865）。配套 v0.23.0 的 upstream 是"非 winner 收 empty blocks + 0"。AscendMultiConnector 的覆写 = **v0.23.0 行为 + Mooncake 例外**，即相对当前 upstream main 反而更保守 `[当前源码已确认]`（git diff v0.23.0..HEAD 逐行核对）。

## 3. Worker 侧 hook 广播（全部在 upstream B 中，Ascend 未覆写任何 worker hook）

均为**顺序 for 循环广播到所有 child，无条件过滤** `[当前源码已确认]`：

| Hook | 位置 | 语义 |
|---|---|---|
| `start_load_kv` | B:289-291 | 全部 child 顺序调用 |
| `wait_for_layer_load` | B:293-295 | 全部 |
| `save_kv_layer` | B:297-305 | 全部（save 语义=写所有 connector，类 docstring B:132-136） |
| `wait_for_save` | B:307-309 | 全部 |
| `get_finished` | B:311-336 | 全部调用；合并见 §5 |
| `register_kv_caches` | B:245-247 | 全部 |
| `register_cross_layers_kv_cache` | B:238-243 | 全部 |
| `bind_gpu_block_pool` | B:249-251 | 全部 |
| `bind_connector_metadata` | B:260-266 | 按 zip 把 `MultiKVConnectorMetadata.metadata` 元组逐位分发给对应 child；同时把 `extra_async_saves` merge 进 `self._extra_async_saves`（B:262-263）；最后调 super() 保证自身 `has_connector_metadata()` 为真 |
| `clear_connector_metadata` | B:268-271 | 全部 |
| `build_connector_worker_meta` | B:360-370 | 收集各 child 返回值成 tuple，前部 None 补齐；全 None 则返回 None |
| `get_block_ids_with_load_errors` | B:338-342 | 全部，block id 集合并集 |
| `set_host_xfer_buffer_ops` | B:344-347 | 全部 |
| `handle_preemptions` | B:349-353 | 按 zip 分发 per-child metadata |
| `set_xfer_handshake_metadata(_pp_aware)` | B:463-477 | 全部 |
| `shutdown` | B:273-284 | 全部，聚合首个异常最后抛出 |
| `take_events` | B:539-541 | 全部 yield from |
| `update_connector_output` | B:431-450 | 全部；调用每个 child 前把 `connector_output.kv_connector_worker_meta` 换成该 child 的分片，finally 恢复 |
| `get_handshake_metadata` | B:452-461 | **例外：非广播**——返回第一个非 None child 的结果 |
| `has_pending_push_work` | B:543-544 | 全部，`any()`（注意：此方法是 v0.23.0 之后新增，见 §8） |

Worker 侧框架调用点（输出回传）：`vllm/v1/worker/kv_connector_model_runner_mixin.py:102-105`——`get_finished()` 与 `get_block_ids_with_load_errors()` 在 execute 收尾调用，结果放入 `ModelRunnerOutput.finished_sending / finished_recving / invalid_block_ids` `[当前源码已确认]`。

## 4. 对 MooncakeLayerwiseConnector（及其子类）的全部特殊处理

1. **A:36** `update_state_after_alloc` 的 `isinstance(c, MooncakeLayerwiseConnector)` 例外——唯一一处 connector 类型特判 `[当前源码已确认]`。`DualPathConnector` 继承 `MooncakeLayerwiseConnector`（`kv_p2p/dual_path/connector.py:103`），其 docstring（:20-25）明确依赖该 isinstance 覆盖子类来获得真实 blocks。
2. A:52-53、A:65-76 的 `has_preempted_request` / `update_state_before_preempt` 是 duck-typing hook，**不是** Mooncake 特判：实现者是 `RecomputeCPUOffloadConnectorV1`（`recompute_cpu_offload_connector.py:180-235`）`[当前源码已确认]`。`update_state_before_preempt` 由 Ascend 的 `RecomputeScheduler` 在抢占路径调用（`vllm_ascend/core/recompute_scheduler.py:303-316`，同样 getattr duck-typing；该 scheduler 仅在 `recompute_scheduler_enable` 时启用，`vllm_ascend/platform.py:679-699`）。A:65-76 的广播语义：所有实现该 hook 的 child 都被调用，`offloaded` 取逻辑或。
3. upstream B 中对 Mooncake **零**特判 `[当前源码已确认]`。
4. 其它文件命中均非功能性特判：`vllm_ascend/ascend_config.py:371`（报错文案建议改用 MooncakeLayerwiseConnector）、`kv_transfer/utils/utils.py:254,305`（debug 日志）。

## 5. metadata / finished 集合 / invalid blocks 的聚合

- **`build_connector_meta()`（B:418-429）**：`MultiKVConnectorMetadata(metadata=tuple(c.build_connector_meta(so) for c in ...))`——按 `_connectors` 顺序组成**元组**，逐位对应，不做交叉合并；若 `self._extra_async_saves` 非空则挂到 metadata 上并**清空本地 dict**（B:426-428，随 metadata 传到 worker 侧，经 B:262-263 恢复）`[当前源码已确认]`。
- **`get_finished()`（B:311-336）**：**并集**语义。
  - `finished_recving`：所有 child 的 recving 直接取并集（B:321）。
  - `finished_sending`：并集但带 `_extra_async_saves` 计数门控——某 req 被 N(>1) 个 child 异步 save 时，`request_finished*` 已记录 `extra = N-1`；每收到一个 child 的 sending 完成信号就递减，**计数值未耗尽前不把 req_id 放入 finished_sending**（B:325-334）。即 sending 完成 = 所有 async-save child 都报告完成（等效交集时序，通过计数实现）。
  - 两集合皆空时返回 `(None, None)`（B:336 `or None`）。
- **invalid blocks / `get_block_ids_with_load_errors()`（B:338-342）**：所有 child 返回集合的**并集**（`|=`），不按 child 拆分、不去重来源。

## 6. 配置名 "MultiConnector" 如何解析到 AscendMultiConnector

- 插件入口：`setup.py:543-546`，`vllm.general_plugins` entry point `ascend_kv_connector = vllm_ascend:register_connector` `[当前源码已确认]`。
- `vllm_ascend/__init__.py:44-51` `register_connector()`：先 `_ensure_global_patch()`，再调 `vllm_ascend.distributed.kv_transfer.register_connector()`。
- **D:21-27**：若 upstream 已注册 `"MultiConnector"` 则先 `_registry.pop("MultiConnector")`，再以**同名** `"MultiConnector"` 注册到 `vllm_ascend...ascend_multi_connector:AscendMultiConnector`。即用户配置 `kv_connector="MultiConnector"` 时工厂解析到 Ascend 实现；upstream 类被替换而非共存 `[当前源码已确认]`。
- child 列表解析（upstream B:212-236 `_get_connector_classes_and_configs`）：从 `kv_transfer_config.kv_connector_extra_config["connectors"]` 读取 child 配置数组，逐个 `copy.copy(vllm_config)` + 构造 per-child `KVTransferConfig`（engine_id 可覆盖），类名经 `KVConnectorFactory.get_connector_class` 解析；`__init__`（B:179-185）按此顺序实例化 `self._connectors`。
- 相关注册（D:29-93）：`AscendStoreConnector`/`MooncakeConnectorStoreV1`（同一类两个名字）、`MooncakeLayerwiseConnector`、`DualPathConnector`（D:57-61）、`RecomputeCPUOffloadConnector` 等。

## 7. `request_finished` / `request_finished_all_groups` 在 Multi 下的语义

框架入口（C:2461-2469 `_connector_finished`）：connector 是 `SupportsHMA` → 调 `request_finished_all_groups(request, block_ids)`；否则调 `request_finished(request, block_ids[0])`。AscendMultiConnector 声明 `SupportsHMA`（A:19），故正常路径走 all_groups `[当前源码已确认]`。

- **delay_free 语义：任一 child 返回 async_save=True 即 delay**（并集/或语义），不是全部。upstream `request_finished`（B:510-518 → `_aggregate_request_finished` B:479-508）：`async_saves` 计数，`return async_saves > 0, ...`；`async_saves > 1` 时记 `_extra_async_saves[req] = async_saves - 1`（B:503-504）；最后 `self._requests_to_connector.pop(request.request_id, None)`（B:506）——winner 映射在请求结束时清除。
- **kv_transfer_params**：本地 upstream 检出为**跨 child 合并 dict、键冲突才报错**（B:492-502）；**这是 v0.23.0 之后的行为**（提交 `77654d080` #46777）。配套 v0.23.0 是"第二个产出 params 的 child 直接 RuntimeError"。
- **Ascend 覆写 `request_finished_all_groups`（A:78-102）**：
  - 非全 HMA（`_all_support_hma=False`）：assert 单 group 后委托 `super().request_finished(request, block_ids[0])`（A:83-85），即走 upstream 聚合。
  - 全 HMA：自写循环调各 child 的 `request_finished_all_groups`；`async_saves > 1` → `_extra_async_saves`（A:97-98）；`self._requests_to_connector.pop`（A:100）；返回 `async_saves > 0, kv_txfer_params`。
  - **与 upstream main 的差异**：params 处理是"**只允许一个 connector 产出 KV transfer params，第二个即 RuntimeError**"（A:93-96）——等同 v0.23.0 upstream，未采纳 main 的 dict-merge `[当前源码已确认]`。
  - Ascend **没有**覆写 `request_finished`，单 group 非 HMA 路径继承 upstream。
- delay_free 的下游效果：C:2143-2163（`finish_requests`/`_free_request`），`connector_delay_free_blocks` 为真时 block 延迟释放，直到 worker 侧 `finished_sending` 回报（§5 的计数门控保证所有 async-save child 都完成）`[当前源码已确认]`。

## 8. 版本偏差清单（v0.23.0 → 本地检出，仅 `multi_connector.py`）

`git log v0.23.0..HEAD -- multi_connector.py` 共 3 个提交 `[当前源码已确认]`：

1. `2285cfca4` #46865：非 winner child 从 (empty blocks, 0) 改为 (real blocks, 0) —— 影响 §2.3 结论；Ascend 覆写仍保持 v0.23.0 语义 + Mooncake 例外。
2. `77654d080` #46777：`_aggregate_request_finished` params 从"唯一产出者"改为"dict 合并 + 键冲突报错" —— 影响 §7；Ascend 的 all_groups 覆写仍是旧语义。
3. `88ed63621` #35264：新增 `has_pending_push_work()`（B:543-544，`any()` 广播；Ascend 未覆写，直接继承）。

## 9. 其它值得评审注意的事实

- B:196-204 注释明确：异步 load 只允许单一 connector（winner），异步 save 允许多个（靠 `_extra_async_saves` 计数）。
- `_requests_to_connector` 生命周期：写入于 `get_num_new_matched_tokens`（A:59 / B:398），清除于 `request_finished*`（A:100 / B:506）。**未发现**在 `None` 短路（B:393-394）或请求被 abort 而未经 `_connector_finished` 时的清理路径——若请求在 WAITING 阶段被 abort，映射可能残留 `[源码可达，尚未运行验证]`（abort 路径是否必走 `_free_request`/`_connector_finished` 未在本范围核实）。
- `prefer_cross_layer_blocks`（B:206-210）：全部 child 都为 True 才为 True。
- upstream 类 docstring（B:128-136）自述契约："Load KV from the first connector that advertises available tokens... Save to all connectors."
- `RecomputeScheduler`（`vllm_ascend/core/recompute_scheduler.py:95,1122`）是对 upstream `Scheduler.schedule()` 的整体覆写，其 waiting 循环中 `get_num_new_matched_tokens`/`update_state_after_alloc` 的调用点是否逐行对齐 upstream C:776/970 未逐行核对 `[源码可达，尚未运行验证]`；仅在 `recompute_scheduler_enable=True`（限 PD D 节点）时启用（`platform.py:679-699`）。

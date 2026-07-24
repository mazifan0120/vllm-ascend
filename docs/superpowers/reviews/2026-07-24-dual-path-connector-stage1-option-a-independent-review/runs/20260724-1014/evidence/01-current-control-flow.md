# 01 — 当前框架真实控制流（Phase 1 证据汇总）

基线：`vllm-ascend @ dev/dualpath 0ec11a47`；upstream `vllm @ main 8df14cfc`（v0.23.1rc0-1050，配套基线 v0.23.0，**参考版本存在小幅偏差**）。
证据来源：6 份 phase1-scratch 详细报告（`evidence/phase1-scratch/01`～`06`，含完整 file:line），主评审者对关键锚点做了原文抽查复核（见各节「已复核」标记）。全部结论为静态阅读 `[当前源码已确认]`，本机无 NPU/运行时，未做运行验证。

行号约定：无特殊说明时
- `MLC` = `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
- `AMC` = `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`
- `U:` 前缀 = upstream `/Users/leqi/Documents/Code/vllm` 内文件

## 0. 端到端骨架（当前代码，不含设计承诺）

Decode Engine 收到带远程 Prefill 语义的请求（`kv_transfer_params` 含 `do_remote_prefill` / `do_remote_decode`）后的真实链路：

1. `U:vllm/v1/core/sched/scheduler.py:774-779`：Scheduler 仅在 `num_computed_tokens==0` 时调用 connector 的 `get_num_new_matched_tokens()`。
2. Connector 返回 `(tokens, load_async)`；`tokens is None` 表示本轮无法回答、下轮重查（`U:.../v1/base.py:453-486`）。
3. Scheduler 分配 KV blocks 后调 `update_state_after_alloc()`；**异步请求会被调用两次，第二次 `num_external_tokens=0`**（`U:scheduler.py:822-827 → 969-974`，base.py:495-504 明确警告按 num_external_tokens 而非 blocks 判空）。
4. `load_async=True` 的请求进入 `WAITING_FOR_REMOTE_KVS`，`num_computed_tokens` 被乐观置满（`U:scheduler.py:1004`）。
5. 每 step，`build_connector_meta()` 产出的 metadata 随 `SchedulerOutput` 经 ZMQ 广播到 Worker（`U:multiproc_executor.py:310-320,377`）。
6. Worker 侧 `U:vllm/v1/worker/kv_connector_model_runner_mixin.py:89-112`：`_get_kv_connector_output` contextmanager 内依次 `bind_connector_metadata()`（:89）→ `start_load_kv()`（:95）→ forward → finally：`wait_for_save()` → `get_finished(scheduler_output.finished_req_ids)`（:102）→ `get_block_ids_with_load_errors()` → `build_connector_worker_meta()` → `clear_connector_metadata()`。**get_finished/invalid/worker_meta 每 step 无条件执行**（含空 step）。
7. 各 rank 的 `KVConnectorOutput` 在 EngineCore 进程由 `KVOutputAggregator` 聚合（`U:kv_connector/utils.py:70-175`）：finished 集合需**全部期望 rank 上报同一 ID 才对外发布**（计数制，默认 `parallel_config.world_size`，:66-90）；**invalid_block_ids 只取并集、无 quorum**（:159）；worker_meta 用 `aggregate()` 合并（:135-144）。
8. Scheduler `update_from_output`：**invalid blocks 先于 finished 消费**（`U:scheduler.py:1579-1586`）；`finished_recving` 到达使请求离开 `WAITING_FOR_REMOTE_KVS` 并 `cache_blocks`（:2517-2534），相关不变式由 **assert 强制**（:2576/2581/2585，违反即 crash）；`finished_sending` 立即 `_free_blocks` 释放延迟块（:2583-2586）。
9. `kv_load_failure_policy` 默认 `fail`（`U:config/kv_transfer.py:69`）；fail 策略下 invalid blocks → 截断 `num_computed_tokens`（:2665）→ `finish_requests(FINISHED_ERROR)`（:1823-1825）→ prefix cache 逐出仅 fail 策略（:2736-2737）。
10. 请求结束/abort：`_free_request` → `_connector_finished`（`U:scheduler.py:2162 → 2434-2469`）；HMA 走 `request_finished_all_groups`；`WAITING_FOR_REMOTE_KVS` 请求默认 delay_free（:2144-2147）；**abort 也触发 request_finished 钩子**（:2162，刻意设计）——connector 若永不上报对应 finished，blocks 将泄漏。

## 1. Scheduler 查询缓存和 Connector

- 调用点唯一：`U:scheduler.py:774-779`，且仅当 `num_computed_tokens == 0`（本地 prefix cache 查询在此之前完成；external connector 查询只发生一次）。`[当前源码已确认]`
- `get_num_new_matched_tokens` 契约（`U:v1/base.py:453-486`）：返回 `(int|None, bool)`；`None` = 本轮无法回答需重查；`bool` = 是否异步加载；0 token 时 bool 必须为 False。类 docstring（base.py:10-12）声称无副作用——**[事实冲突]** upstream `MultiConnector.get_num_new_matched_tokens` 恰恰在其中写 `self._requests_to_connector`（`U:multi_connector.py:398`），且 `None` 短路时该映射不清除。
- MLC 的 Scheduler 实现（MLC:893-926）**不看 kv_role，只看 kv_transfer_params**：`do_remote_prefill` → 返回 `(prompt剩余, async=True)`；`do_remote_decode` → 做 hybrid 截断副作用后返回 `(0, False)`。`[当前源码已确认]`

## 2. matched-token accounting（当前）

- 单 Connector 时 Scheduler 直接使用返回值；`load_async=True` 时 `num_computed_tokens` 乐观置满、请求挂入 `WAITING_FOR_REMOTE_KVS`（`U:scheduler.py:1004`）。`[当前源码已确认]`
- 当前框架**不存在**「一个请求的两段 token 分别由两个 child connector 异步提供」的 accounting 表达：winner 唯一，external tokens 只有一个整数。partial hit 若要靠两个 sibling 自动组合，当前 first-winner 语义无法表达（这点直接支撑设计 §5.2 的硬性契约动机）。

## 3. first-positive winner 选择（AscendMultiConnector）

- upstream `MultiConnector.get_num_new_matched_tokens`（`U:multi_connector.py:381-400`）：按配置顺序调用**所有** child；第一个返回正数者写入 `_requests_to_connector[req_id]=i` 并成为 winner；后续 child 仍被调用但 `to_return[0]==0` 守卫使结果不被覆盖；任一 child 返回 `None` 立即整体短路 `(None, False)`。
- Ascend 覆写（AMC:43-63）：仅在 upstream 逻辑前加 recompute-offload 优先扫描（duck-typed `has_preempted_request`，仅 `RecomputeCPUOffloadConnectorV1` 实现）；无此 child 时等价 upstream。`[当前源码已确认，已复核]`
- winner 映射在请求结束时 pop（AMC:100 / `U:multi_connector.py:506`）；WAITING 阶段 abort 时 `_requests_to_connector` 的清理路径未发现（**待确认风险**：残留映射影响同 ID 复用——但 request ID 复用本身受 Engine 侧限制）。

## 4. KV block 分配

- Scheduler 对 `load_async` 请求先分配 blocks 再进入等待；async load 不支持 block 共享（`U:scheduler.py:2652-2653` 注释）。`[当前源码已确认]`
- 分配结果通过 `update_state_after_alloc(request, blocks, num_external_tokens)` 传给 connector。

## 5. `update_state_after_alloc`（关键分叉点）

- upstream 契约：异步请求**调用两次**，第二次 `num_external_tokens=0`（`U:base.py:488-512`、`U:scheduler.py:822-827→969-974`）。`[当前源码已确认]`
- AscendMultiConnector 分叉（AMC:32-41，**已复核原文**）：
  - winner child：`(request, 真实 blocks, 真实 num_external_tokens)`；
  - **`isinstance(c, MooncakeLayerwiseConnector)` 的非 winner child：同样收到真实 blocks 和真实 num_external_tokens**（AMC:36）——设计 §4.1 的「Layerwise 例外」属实；
  - 其余非 winner child（如 AscendStoreConnector）：`(request, empty_blocks, 0)`（AMC:41）。
- 但注意：非 winner AscendStore child 在 `update_state_after_alloc(..., 0)` 下并非完全无副作用——`pool_scheduler.py:589-593` 在 layerwise 模式且曾命中时仍可能把 `load_specs[req].can_load` 置 True；非 layerwise 模式下 `_loading_req_ids` 入口要求 `num_external>0`（pool_scheduler.py:617-618），故不会进入 loading 集合。`[当前源码已确认]`
- MLC 的实现（MLC:928-998）：prefill 侧入 `_reqs_need_recv` + 清标记 + 线程池 POST metaserver；decode 侧入 `_reqs_need_send_layerwise`。

## 6. Scheduler metadata → Worker metadata

- 每 step `build_connector_meta()` 产出 `KVConnectorMetadata`，随 `SchedulerOutput` 经 ZMQ MessageQueue 广播；Worker 在 busy loop dequeue（`U:multiproc_executor.py:310-320,377,986-990`）。无独立 metadata 通道。`[当前源码已确认]`
- Multi 下 `build_connector_meta` 按 child 顺序组成 tuple 并转运 `_extra_async_saves`（`U:multi_connector.py:418-429`）；Worker 侧 `bind_connector_metadata` 按 zip 分发回各 child（`U:multi_connector.py:260`）。
- **MLC 在 `kv_both` 下的缺陷（已复核原文）**：`build_connector_meta` 是 `if is_kv_consumer ... else ...`（MLC:1006-1022），`kv_both` 时 `is_kv_consumer` 为真、**恒走 consumer 分支，producer 的 send 元数据永不产出**，`_reqs_need_send_layerwise` 只进不出（条目泄漏）。这正是设计 §9.2 要求 DualPath 完全覆盖 `build_connector_meta()` 的现实依据。`[当前源码已确认]`

## 7. Worker hook 广播（MultiConnector/AscendMultiConnector）

- 以下 hook **无条件顺序广播到所有 child**（`U:multi_connector.py`）：`start_load_kv`(:289)、`wait_for_layer_load`(:293)、`save_kv_layer`(:297)、`wait_for_save`(:307)、`register_kv_caches`(:245)、`bind_connector_metadata`(:260 按 zip)；唯一例外 `get_handshake_metadata`(:452，第一个非 None)。`[当前源码已确认]`
- 即：Worker 侧不存在「非 winner 不被调用」的过滤。设计 §15.4 的「广播约束」是对当前事实的准确描述：每个 child 的 hook 必须自行按已提交路径决定是否行动。
- `get_finished` 合并：recving=并集；sending=并集但经 `_extra_async_saves` 计数门控（`U:multi_connector.py:325-334`，等效「所有 async-save child 都完成才发布」）；invalid blocks=并集（:338-342）。

## 8. Store lookup 和 bulk load（AscendStore / KVPool）

- `KVPoolScheduler.get_num_new_matched_tokens`（pool_scheduler.py:478-565）三条 lookup 路径：GVA 本地 `batch_get_key_info`(:498-500)、layerwise 本地 `batch_is_exist`(:510-513)、非 layerwise 走 zmq RPC 到 rank-0 worker 的 LookupKeyServer(:515-521)。`[当前源码已确认]`
- **它不是纯 probe（设计 §8.2 声称属实且更强）**：
  1. 首次 lookup 惰性创建常驻 `LookupKeyClient`(:515-516)，`close()` 无生产调用方；
  2. 命中>0 且需分配时创建 `LoadSpec`(:549-553)，清理点只有三个 `_process_*` pop(690/742/845)；**waiting 阶段 abort 无清理路径，`load_specs` 残留泄漏**（`request_finished` 1012-1037 与 `build_connector_meta` 896-901 都不碰它）；
  3. REQ socket `recv()` **无超时**(:1107)，在 scheduler 循环内阻塞。
- 对齐：`cache_transfer_granularity = lcm(lcm_block_size, 各 group family granularity)`(:404-413)；`discard_partial_chunks` 默认 True(:124-126)，lookup/save 双侧 floor(:502-503, config_data.py:896-905)。
- `load_async=True` 且非 layerwise 且 `num_external>0` 时，请求进入 `_loading_req_ids`(617-618) → `AscendConnectorMetadata.loading_req_ids`(910-915)；清理仅在 finished/preempted(901/907)。
- Worker bulk load：非 layerwise 每请求一次 bulk `m_store.get`（pool_worker.py:840）；同步路径在 model-runner 线程阻塞，**无 DMA 拆分**（拆分只在 layerwise `_batch_copy_with_limits`，kv_transfer.py:397-448）。
- **异步失败上报断裂（重要当前缺陷）**：`KVPoolWorker` 创建 `KVCacheStoreRecvingThread` 时未注入 invalid_block_ids（pool_worker.py:457-465）；失败块进线程私有集合（kv_transfer.py:843-844,908-910），而 `get_block_ids_with_load_errors`(1333-1337) 只 drain worker 自己的集合——**load_async 下 Store 加载失败永不上报 invalid blocks，请求却无条件 `set_finished_request`(kv_transfer.py:942)**，违反 upstream base.py:385-389「失败 block 不得晚于 finished 上报」契约。`[当前源码已确认]`
- LookupKeyServer 创建条件与设计 §9.7 声称精确一致：worker 角色 + `not use_layerwise` + `parallel_config.rank==0`（ascend_store_connector.py:119-120）。
- `use_layerwise=True` 会强制 `load_async` 关闭（pool_scheduler.py:565/617）。
- `request_finished_all_groups` 与单 group 版不对称（1053 vs 1028：tracker 为 None 时不早退），可能对从未保存过的请求 delay-free 而发送线程永不报完成。`[源码可达，尚未运行验证]`

## 9. Reverse Layerwise（当前：不存在）

- 当前代码**不存在任何 DE→PE 反向逐层传输**：全仓 grep 无 reverse 传输实现（scratch-01、scratch-03 双重确认）。MLC 的 send/recv thread 语义固定为「P 发 D 收」。
- 反向链路是设计的全新能力；其载体是 MLC 的 `KVCacheSendingLayerThread`/`KVCacheRecvingLayerThread` 与 `global_te` TransferEngine 单例（mooncake_transfer_engine.py:31-40，`register_buffer` 有 `is_register_buffer` 单次守卫，第二次调用静默跳过）。`[当前源码已确认]`

## 10. PE model compute

- forward 中按层钩子：upstream 用 `@maybe_transfer_kv_layer` 装饰器（`U:kv_transfer_utils.py:38-59`）包 unified_attention/unified_mla_attention custom op；ascend `opaque_attention_op()=True`（platform.py:877），非 MLA 模型走该装饰器。`[当前源码已确认]`
- **ascend MLA/SFA/DSA 绕过 upstream 装饰器**：走自注册 `torch.ops.vllm.mla_forward`（ops/mla.py:174,207-213，未装饰）；`wait_for_kv_layer_from_connector` **仅 has_prefill 时调用**（mla_v1.py:1683-1684）——decode-only step 在 MLA 路径上没有 `wait_for_layer_load`；wait 位置在 `_mla_preprocess`（投影之后、kernel 之前），不是层入口。
- MLC 的 `wait_for_layer_load` 当前是 **`pass` 空操作**（MLC:1976-1977）；consumer 的等待完全靠 Scheduler `WAITING_FOR_REMOTE_KVS` 门禁。`[当前源码已确认，已复核附近代码]`

## 11. Forward Layerwise（当前 MLC P→D 链路）

- Worker `register_kv_caches`（MLC:1247-1398）每进程恰好被调一次（model_runner_v1.py:3825）；`kv_both` 下两个 if（1359/1386）都命中，**一次调用完成 buffer 注册 + 一个 send thread + 一个 recv thread**（设计 §9.2 声称属实，已复核 1359-1398 附近）。`global_te.register_buffer` 有单次守卫，重复调用静默跳过。
- `start_load_kv`（MLC:1603-1688，**已复核原文**）：`if is_kv_consumer ... elif is_kv_producer`——**kv_both 走 consumer 分支，producer 的 transfer mapping 计算整体跳过**（设计 §9.2 声称属实）；但 `save_kv_layer` 的门是 `is_kv_producer`（MLC:1699），kv_both 下仍会执行发送。
- wire identity：`session_id = host:te_rpc_port`（MLC:470）；端口按 kv_port + dp/pcp/tp 推导（MLC:809-812, 1164-1169）；engine_id 缺省 uuid4。
- send thread 主循环消费逐层 SendTask（ReqMeta/SendTask 结构）；单边写直接写远端正式 blocks。
- **发送线程失败归因 bug（当前缺陷）**：MLC:507 `failed_reqs.add(req_id)` 用的是 :469 循环残留变量，多请求同 SendTask 时失败归因可能错误。`[当前源码已确认]`
- 无 shutdown/drain（基类 no-op，线程 daemon）；取消无 connector 钩子，块释放依赖终端事件到达。

## 12. `get_finished()`、invalid blocks、多 rank 聚合（当前各组件）

- MLC Worker `get_finished`（MLC:1400-1428，**已复核原文**）：
  - **返回 `(set(), done_recving)`：finished_sending 恒空**；
  - 失败接收请求**不进**完成集合，只把 `meta.local_block_ids` 汇入 `_invalid_block_ids`（:1416-1419），经 `get_block_ids_with_load_errors`(:1430-1438) 暴露——设计 §4.2「父 Worker 只上报正常完成请求」属实；
  - **terminal 早于 mapping 到达时事件被永久丢弃**（:1407 `if s in self.request_map` 过滤 + `get_and_clear` 已清空），不暂存——设计 §8.2「Worker 必须暂存并在 mapping 建立后重新归属」正是针对此缺陷；
  - 单步内 `start_load_kv` 先于 `get_finished`（mixin:95 vs :102），但跨步早到无防护，请求会滞留 `WAITING_FOR_REMOTE_KVS`。
- KVPool（Store）侧：见 §8 的失败上报断裂。
- 多 rank 聚合：框架 `KVOutputAggregator`（`U:kv_connector/utils.py:55-175`）在 Scheduler/EngineCore 进程运行；expected count = `get_finished_count() or world_size`；finished 需全 rank；invalid 并集无 quorum。PP=DP=1 时 world_size==TP size，与设计 §13.3 的前提一致。`[当前源码已确认]`
- `KVConnectorWorkerMetadata.aggregate`：基类无冲突语义（`U:base.py:150-168`），由 KVOutputAggregator 逐步调用；MultiConnector 按子位次递归合并（`U:multi_connector.py:55-68`）。

## 13. `WAITING_FOR_REMOTE_KVS` 状态转换（当前）

- 进入：`load_async=True` 且 matched tokens > 0，`num_computed_tokens` 乐观置满（`U:scheduler.py:1004`）。
- 离开：请求 ID 出现在聚合后的 `finished_recving_kv_req_ids`（`U:scheduler.py:2517-2534`）→ `cache_blocks` 加入 prefix cache。
- 失败路径：invalid blocks 先到（`U:scheduler.py:1579-1586`）→ fail 策略截断 + `FINISHED_ERROR`。
- 不变式由 assert 强制（:2576/2581/2585）：finished_recving 到达时请求必须仍处于 WAITING 且 blocks 已分配，违反即 Engine crash。设计 §13.2「同一 ID 只能发布一次 receive terminal」「invalid 不晚于 failed terminal」正是要避免触发这些 assert。

## 14. 请求成功、失败、取消与资源释放（当前）

- 成功：finished_recving → 离开 WAITING → 正常 decode → 结束时 `_connector_finished`；HMA 下 `request_finished_all_groups` 决定 delay_free。
- MLC 的 `request_finished` / `request_finished_all_groups` **均恒 `(False, None)`**（MLC:1102-1112/1114-1124，已复核附近代码）——设计 §9.4「不能沿用父类恒定返回 False」属实。注意：`SupportsHMA.request_finished_all_groups` 在 upstream 基类是抽象方法（`U:base.py:92-114`），MLC 实现了它所以 Scheduler 实际调用此入口。
- Multi 下 delay_free：任一 child async_save 即 delay_free（`U:multi_connector.py:508` / AMC:87-102，已复核）；多个 async-save child 用 `_extra_async_saves` 倒计时（AMC:97-98）。
- 失败：`kv_load_failure_policy=fail`（默认）下 invalid blocks → FINISHED_ERROR → prefix cache 逐出。
- 取消/abort：abort 在 execute_model 与 update_from_output 之间处理（`U:core.py:510-512`）；**abort 也触发 `request_finished` 钩子**（`U:scheduler.py:2162`）；WAITING 请求默认 delay_free（:2144-2147）；延迟释放依赖 connector 之后上报 finished（:2580-2586）——connector 不上报即泄漏。
- KVPool 的 `load_specs`/`LookupKeyClient` 在 abort 路径无清理（见 §8）。

## 15. DualPath foundation 现状（设计 §4.5 锚点核对）

- `dual_path/` 目录仅 `__init__.py`（15 行 docstring）、`config.py`（304 行）、`connector.py`（144 行）。
- `DualPathConnectorScheduler(MooncakeLayerwiseConnectorScheduler)`（connector.py:51）、`DualPathConnectorWorker(MooncakeLayerwiseConnectorWorker)`（:79）、`DualPathConnector(MooncakeLayerwiseConnector, SupportsHMA)`（:103）。
- 构造器：**不调** `MooncakeLayerwiseConnector.__init__`，直接 `KVConnectorBase_V1.__init__`（connector.py:120），按 role 创建 DualPath Scheduler/Worker（:129-144）——与设计 §9.3 描述的构造方式一致。子类未复制父类的 `self._is_kv_producer`（父类 MLC:701；当前全仓无人读该属性，无害但属静默偏差）。父 worker init 有进程级副作用 `os.environ["ASCEND_TRANSFER_TIMEOUT"]`（MLC:1131），被子类继承。
- 注册链路完整：`setup.py:546` entry point → `vllm_ascend/__init__.py:44` → `distributed/kv_transfer/__init__.py:57-61` 注册 "DualPathConnector"；`"MultiConnector"` 被 pop 后同名替换为 `AscendMultiConnector`（:21-27，替换非共存）。
- 现有 config 字段仅：role/path_strategy/relay/path_planner/monitor/topology；**不存在** orchestration_mode、partial_read_policy、forward/reverse 端口、store 嵌套配置（全仓 grep 零命中）。`kv_load_failure_policy` 是 upstream `KVTransferConfig` 字段（默认 "fail"），vllm-ascend 平台层仅校验 hybrid（platform.py:892-899），与 DualPath 无联动。
- `_req_path` 侧表（connector.py:71）无人写无人读，纯占位。
- UT：26 个 test（16 config 校验 + 5 构造（父类 init 被 patch）+ 3 父签名哨兵 + 2 注册）；无任何运行时行为覆盖；本机未运行。
- PathDecisionCoordinator、control transport、Reverse、BlockOwnershipLedger、TransferFence、Store adapter、专属 WorkerMetadata：**全部不存在**（grep 零命中）。finished_recving/sending/invalid 行为 100% 继承父类。

## 16. 与设计强相关的当前事实清单（供 Phase 2/3 引用）

| # | 事实 | 位置 | 对设计的影响 |
|---|---|---|---|
| F1 | 非 winner Layerwise child 也收真实 blocks+真实 tokens | AMC:36 | 支撑 §4.1/§15.4「真实 blocks 不构成路径授权」 |
| F2 | 非 winner Store child 收 empty blocks+0，但 layerwise 模式下仍可能置 can_load | AMC:41; pool_scheduler.py:589-593 | §15.2「不产生 Worker Store GET」需验证 can_load 路径 |
| F3 | Multi 任一 child 返回 None 整体短路，winner 映射残留 | U:multi_connector.py:393-398 | §8.2 PE 返回 (None,False) 等待重试的副作用 |
| F4 | update_state_after_alloc 异步请求调两次，第二次 tokens=0 | U:scheduler.py:822-827,969-974 | §8.5 幂等要求的现实依据 |
| F5 | kv_both 下 MLC build_connector_meta 恒走 consumer 分支，send 元数据不产出且条目泄漏 | MLC:1006-1022 | §9.2 必须完全覆盖的依据 |
| F6 | kv_both 下 start_load_kv 只走 consumer 分支；save_kv_layer 门是 is_kv_producer | MLC:1606-1615,1699 | §9.2 方向化 adapter 的依据 |
| F7 | kv_both 下一次 register_kv_caches 完成全部注册；TE register_buffer 有单次守卫 | MLC:1359-1398; mooncake_transfer_engine.py:31-40 | §9.2 共享 runtime 可行性的依据 |
| F8 | MLC get_finished 返回 (set(), done_recving)；失败请求只进 invalid；早到 terminal 被丢弃 | MLC:1400-1438 | §4.2/§8.2/§13.2 的动机；DualPath 必须重写 |
| F9 | MLC request_finished(_all_groups) 恒 (False,None) | MLC:1102-1124 | §13.4 delayed-free 必须自行实现 |
| F10 | Store get_num_new_matched_tokens 非纯 probe：建 LookupKeyClient/LoadSpec，abort 无清理，REQ recv 无超时 | pool_scheduler.py:515-553,1107 | §9.7 probe/abort 设计的依据与负担 |
| F11 | Store load_async 失败上报断裂：invalid 永不上报但请求无条件 finished | pool_worker.py:457-465; kv_transfer.py:942 | §19 失败语义在 DE Store adapter 复用时必须修补 |
| F12 | finished 需全 rank 才发布；invalid 并集无 quorum | U:kv_connector/utils.py:78-90,159 | §13.3 多 rank 聚合前提 |
| F13 | abort 也触发 request_finished；WAITING 默认 delay_free；connector 不上报 finished 即泄漏 | U:scheduler.py:2162,2144-2147,2580-2586 | §13.5 取消语义的前提 |
| F14 | ascend MLA 路径 wait_for_layer_load 仅 has_prefill 时调用；MLC wait_for_layer_load 为 pass | mla_v1.py:1683-1684; MLC:1976-1977 | §11.2 防御性检查的实际效力 |
| F15 | MLC 发送线程失败归因 bug（残留循环变量） | MLC:507 vs :469 | Reverse 复用 send thread 时继承该 bug |
| F16 | upstream None 短路语义 = 下轮重查；Multi 下短路不清除已写映射 | U:base.py:472-474; multi_connector.py:393-398 | §8.2 PE (None,False) 重试语义 |
| F17 | Store request_finished_all_groups 可能对从未保存的请求 delay-free | pool_scheduler.py:1053 vs 1028 | §13.4 聚合时的边界 |
| F18 | upstream v0.23.0 后 3 个提交改变 MultiConnector 语义（非 winner 真实 blocks+0、params dict-merge、has_pending_push_work）；Ascend 覆写停留在 v0.23.0 语义 | U commits 2285cfca4/77654d080/88ed63621 | 版本偏差风险，升级 upstream 时 AMC 行为会漂移 |

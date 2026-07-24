# Phase 1 证据 01：vllm-ascend 现有 DualPath foundation

- 收集时间：2026-07-24
- 仓库：`/Users/leqi/Documents/Code/vllm-ascend`，branch `dev/dualpath`，HEAD `0ec11a4703b987d3103f8859f184c492f22bde88`（v0.19.1rc1-991）
- 参考 upstream：`/Users/leqi/Documents/Code/vllm`@`8df14cfc`（v0.23.1rc0-1050，与 vllm-ascend 官方配套 v0.23.0 同 minor 系列但有小幅偏差，凡引用 upstream 处均已标注）
- 范围：`vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/`（全部 3 个文件）+ `tests/ut/distributed/kv_transfer/dual_path/`（1 个文件）+ 为回答注册/构造器问题而追踪到的相关调用点
- 运行验证状态：本机（macOS，无 pytest / 无 vllm 运行时）**无法执行任何测试**，所有行为结论均为静态源码阅读，标注 `[源码可达，尚未运行验证]`；纯存在性/结构结论标注 `[当前源码已确认]`。

---

## 1. 目录实际内容

`vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/` 下**只有 3 个文件** `[当前源码已确认]`：

| 文件 | 内容 |
|---|---|
| `__init__.py`（15 行） | 纯 docstring，无任何 import / 符号导出。明确说明「故意不 eager import connector」（`__init__.py:12-14`） |
| `config.py`（304 行） | 配置解析/校验，见 §4 |
| `connector.py`（144 行） | 3 个类，见 §2 |

测试目录 `tests/ut/distributed/kv_transfer/dual_path/` 下**只有 1 个文件**：`test_dual_path_connector.py`（286 行，26 个 test 方法）`[当前源码已确认]`。

### 1.1 config.py 中的符号清单（全部无父类，均为 `@dataclass` 或模块级函数/常量）

- 常量：`PATH_STRATEGY_FEATURES`（6 个 value-function 特征名，`config.py:29-36`）、`DEFAULT_WEIGHTS`（`config.py:41-48`）、`DEFAULT_PARAMS`（`config.py:52-57`）、`DEFAULT_DECISION_WINDOW_MS=50`（`config.py:59`）、`DEFAULT_MONITOR_WINDOW_MS=200` / `DEFAULT_MONITOR_EWMA_ALPHA=0.3` / `DEFAULT_MONITOR_SOURCES`（`config.py:61-68`）、`DEFAULT_PATH_PLANNER_K_PATHS=4`（`config.py:70`）、`DEFAULT_PATH_PLANNER_COST_WEIGHTS`（`config.py:71-77`）、`DEFAULT_RELAY_ZMQ={"rep_port":5555,"pub_port":5556}`（`config.py:79`）、`DEFAULT_TOPOLOGY_SOURCE_PRIORITY=("ubutils","sysfs","udev","static")`（`config.py:81`）、`SUPPORTED_CONTROL_PLANES=("zmq",)`（`config.py:85`）、`SUPPORTED_PATH_STRATEGY_TYPES=("static","value_function","adaptive")`（`config.py:86`）
- dataclass：`ValueFunctionCfg`（`config.py:89`）、`PathStrategyCfg`（`config.py:98`）、`RelayCfg`（`config.py:104`）、`PathPlannerCfg`（`config.py:111`）、`MonitorCfg`（`config.py:121`）、`TopologyCfg`（`config.py:130`）、`DualPathConfig`（`config.py:136`，含 classmethod `from_extra_config`，`config.py:147`）
- 模块函数：`_validate_role`（`config.py:195`）、`_parse_path_strategy`（`config.py:213`）、`_validate_weights`（`config.py:237`）、`_parse_relay`（`config.py:251`）、`_parse_path_planner`（`config.py:261`）、`_parse_monitor`（`config.py:273`）、`_parse_topology`（`config.py:284`）

### 1.2 connector.py 中的类与继承 `[当前源码已确认]`

| 类 | 继承 | 位置 |
|---|---|---|
| `DualPathConnectorScheduler` | `MooncakeLayerwiseConnectorScheduler` | `connector.py:51` |
| `DualPathConnectorWorker` | `MooncakeLayerwiseConnectorWorker` | `connector.py:79` |
| `DualPathConnector` | `MooncakeLayerwiseConnector, SupportsHMA` | `connector.py:103` |

父类定义位置：`MooncakeLayerwiseConnector(KVConnectorBase_V1, SupportsHMA)`（`mooncake_layerwise_connector.py:697`）、`MooncakeLayerwiseConnectorScheduler`（`:790`，无父类）、`MooncakeLayerwiseConnectorWorker`（`:1127`，无父类）、`MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata)`（`:656`）。

## 2. DualPathConnector 构造器行为 `[当前源码已确认]`

`DualPathConnector.__init__`（`connector.py:112-144`）：

- **不调用** `MooncakeLayerwiseConnector.__init__`；在 `connector.py:120` 直接调用 `KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)`。注释（`connector.py:118-119`）说明原因：父类 `__init__` 会硬编码实例化父类自己的 scheduler/worker。
- 随后自行复制父类的小部分 setup：`engine_id`（`connector.py:122`）、`_connector_metadata = MooncakeLayerwiseConnectorMetadata()`（`connector.py:123`）。
- 解析配置：`self.dual_path_cfg = DualPathConfig.from_extra_config(kv_connector_extra_config, ktc)`（`connector.py:124-127`）。**注意：配置解析在角色分支之前，Scheduler 进程与 Worker 进程都会执行全部 fail-fast 校验**（含 `topology.static_yaml` 文件存在性检查，`config.py:294`）——即该文件必须在两类进程的部署机上都可访问 `[当前源码已确认]`。
- 角色分支（`connector.py:129-144`）：SCHEDULER → 建 `DualPathConnectorScheduler(vllm_config, kv_cache_config, str(engine_id), dual_path_cfg)` 且 `connector_worker=None`；WORKER → 反之；其他 → `ValueError("Unsupported KVConnectorRole")`。

**与父类 setup 的一处偏差**：父类 `__init__` 还设置 `self._is_kv_producer`（`mooncake_layerwise_connector.py:701`），DualPathConnector **没有**复制这一行。全仓 grep 显示 `_is_kv_producer` 仅在赋值处出现、**无任何读取点**，故当前无实际影响，但若父类未来消费该属性将出现静默偏差 `[当前源码已确认]`。

子类构造器：

- `DualPathConnectorScheduler.__init__`（`connector.py:61-76`）：先 `super().__init__`（父 scheduler 的 init 见 `mooncake_layerwise_connector.py:793-834`：side_channel host/port、`_reqs_need_recv`、`_reqs_need_send_layerwise`、`ThreadPoolExecutor(32)`、TLS/httpx metaserver client 等），再挂 `dual_path_cfg` 和 `self._req_path: dict[str, str] = {}`（`connector.py:71`）。
  - **`_req_path` 当前无人写、无人读**：全仓 grep 仅命中初始化处与 UT 的空断言（`test_dual_path_connector.py:214`）`[当前源码已确认]`。docstring 称其为「shadow wiring now; execution later」的 per-request 决策侧表（`connector.py:55-58, 70`）。
- `DualPathConnectorWorker.__init__`（`connector.py:87-100`）：先 `super().__init__`（父 worker init 见 `mooncake_layerwise_connector.py:1130-1201`：设置 `ASCEND_TRANSFER_TIMEOUT` 环境变量、TransferEngine、zmq Poller、msgspec、线程占位等重量级初始化），再挂 `dual_path_cfg`。docstring 声称「The LinkMonitor starts here」（`connector.py:83`）——**当前代码中没有任何 LinkMonitor 启动逻辑**，仅为注释性承诺 `[当前源码已确认]`。

进程归属：connector 本体由 vLLM 框架分别在 Scheduler 进程（role=SCHEDULER）与 Worker 进程（role=WORKER）各构造一次；Stage 1 自身不创建任何额外线程/进程，后台线程全部来自父类 worker（`KVCacheSendingLayerThread`/`KVCacheRecvingLayerThread`，`mooncake_layerwise_connector.py:204,530`）`[源码可达，尚未运行验证]`。

## 3. 注册路径 `[当前源码已确认]`

完整链条：

1. `setup.py:543-551`：entry point `"vllm.general_plugins": ["ascend_kv_connector = vllm_ascend:register_connector", ...]`
2. `vllm_ascend/__init__.py:44-51` `register_connector()`：先 `_ensure_global_patch()`，再调 `vllm_ascend.distributed.kv_transfer.register_connector()`
3. `vllm_ascend/distributed/kv_transfer/__init__.py:57-61`：
   ```python
   KVConnectorFactory.register_connector(
       "DualPathConnector",
       "vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.connector",
       "DualPathConnector",
   )
   ```
   注册进 upstream vllm 的 `KVConnectorFactory`（import 于 `:18`）。同函数内还注册/覆盖 MultiConnector→AscendMultiConnector、MooncakeConnectorV1、AscendStoreConnector、MooncakeLayerwiseConnector、UCMConnector 等。

结论：`DualPathConnector` 名字**已注册**，可作为顶层 `kv_connector` 名使用；UT（`test_dual_path_connector.py:255-282`）也以 mock factory 的方式验证了注册调用与模块路径可解析。是否可作为 `AscendMultiConnector` 的 sub-connector 使用——`__init__.py:5` 与 `connector.py:106-107` 的 docstring 均如此声称；upstream `MultiConnector` 通过 `kv_connector_extra_config` 逐个子 connector 调 factory 创建，路径相同 `[源码可达，尚未运行验证]`。

与 `AscendMultiConnector` 的协同（block 转发）：`ascend_multi_connector.py:32-41` `update_state_after_alloc` 中 `if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector)`（`:36`）——因 `DualPathConnector` IS-A `MooncakeLayerwiseConnector`，即使不是 first-wins 选中的 connector 也会收到**真实 blocks** 而非 empty blocks。这就是 `connector.py:20-25` docstring「Block forwarding is free」的依据 `[当前源码已确认]`。

## 4. 配置字段与 fail-fast 校验 `[当前源码已确认]`

`DualPathConfig`（`config.py:136-145`）顶层字段：`role`（必填，Literal["pe","de"]）、`path_strategy`、`relay`、`path_planner`、`monitor`、`topology`。全部来自 connector 自己的 `kv_connector_extra_config`（`config.py:1-15` docstring 明确：不引入环境变量）。

| 任务点名的字段 | 现状 |
|---|---|
| `orchestration_mode` | **不存在**（全仓 grep 无命中，dual_path 内外均无） |
| `partial_read_policy` | **不存在**（同上） |
| control transport | 仅有 `relay.control_plane`（默认 `"zmq"`，`config.py:106`；仅支持 `"zmq"`，`config.py:85,174-178`）+ `relay.zmq={rep_port:5555, pub_port:5556}`（`config.py:79`）+ `relay.urpc` dict（解析但禁用）。**没有任何 control transport 实现**，只是配置占位 |
| forward/reverse 端口 | **不存在**「forward/reverse」语义的端口；只有上述 rep/pub 两个 zmq 端口。父类的数据面端口来自 `kv_port` 派生（`mooncake_layerwise_connector.py:809-812, 1164-1169`） |
| store 嵌套配置 | **不存在**；dual_path 内无任何对 AscendStoreConnector / store 的引用 |
| `kv_load_failure_policy` | **不在 DualPathConfig 内**。它是 upstream `KVTransferConfig` 的顶层字段（upstream `vllm/config/kv_transfer.py:69`，`Literal["recompute","fail"]`，默认 `"fail"`；参考版本存在小幅偏差）。vllm-ascend 侧唯一校验在平台层 `platform.py:892-899` `_validate_kv_load_failure_policy`（`platform.py:677` 调用）：仅当 `"recompute"` 时断言非 hybrid model。**该校验对所有 connector 生效，与 DualPath 无专属关联，DualPathConfig 对它既无解析也无交叉校验** |
| 拓扑 fail-fast | `topology.static_yaml` 仅做**文件存在性**检查（`config.py:289-298`，`FileNotFoundError`）；docstring 自述「Full two-layer schema + sufficiency validation lands later」（`config.py:290-293`）。`source_priority` 原样收下、无合法值校验（`config.py:284-287`） |

其余 fail-fast（全部在 `from_extra_config` 路径上，Scheduler/Worker 进程构造时同步执行）：

- `role` 缺失/非法 → `ValueError`（`config.py:154-159`）
- role↔kv_role 交叉校验：pe 要求 `is_kv_producer`、de 要求 `is_kv_consumer`，kv_both 两者皆满足（`config.py:195-210`）
- 未知 weight 名 → `ValueError`（`config.py:237-243`）；全零权重仅 `logger.warning`（`config.py:244-248`）
- `decision_window_ms <= 0` → `ValueError`（`config.py:223-228`）
- `path_planner.enabled=True` → `NotImplementedError`（`config.py:169-173`）
- `relay.control_plane` 非 zmq → `NotImplementedError`（`config.py:174-178`）
- `path_strategy.type` 非法 → `ValueError`（`config.py:179-183`）
- 注意 `path_strategy.type` 的校验在 `path_planner`/`relay` 检查**之后**，且 "static"/"adaptive" 虽在支持列表中但**无任何对应实现**（仅 value_function 的 weights/params 被解析存储）`[当前源码已确认]`

对请求状态/KV block/metadata/完成集合的影响：Stage 1 的 DualPathConnector **不 override 任何运行时方法**（UT `test_dual_path_connector.py:228-232` 断言 `get_num_new_matched_tokens` 就是父类方法本身）。finished_recving/finished_sending 由父 worker `get_finished`（`mooncake_layerwise_connector.py:1400-1428`）产生，load-error invalid blocks 由 `get_block_ids_with_load_errors`（`:1430-1438`，返回并清空 `_invalid_block_ids`，`:1214`）产生，行为与 MooncakeLayerwiseConnector 完全一致 `[当前源码已确认]`。

## 5. UT 覆盖情况 `[当前源码已确认]`

`tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`，26 个 test，4 个 TestCase：

- `TestDualPathConfig`（16 个，`:77-186`）：role 合法/缺失/非法/kv_role 不匹配、kv_both、默认 weights/params、未知 weight、全零告警、`decision_window_ms`、path_planner 拒绝、urpc 拒绝、static_yaml 缺失/存在、monitor sources 覆盖合并。
- `TestDualPathConnectorConstruction`（5 个，`:189-232`）：继承关系、SCHEDULER/WORKER 两种构造、非法 role、`get_num_new_matched_tokens` 未被 override。**构造测试 patch 掉了父类 scheduler/worker 的 `__init__`（`:195-198`），因此父类重初始化路径（TransferEngine、httpx、zmq）完全未被触及**。
- `TestParentSignatureSnapshot`（3 个，`:235-252`）：父 scheduler/worker `__init__` 签名快照（`["self","vllm_config","kv_cache_config","engine_id"]`）、父 connector `__init__` 源码中仍硬编码父类 scheduler/worker 名（防 upstream drift 的哨兵）。
- `TestDualPathRegistration`（2 个，`:255-282`）：mock factory 验证注册调用、注册模块路径可 import 且解析到同类。

**未覆盖**：

- 无任何 e2e / 集成测试；`tests/e2e/` 下无 dual_path 相关文件（grep 无命中）`[当前源码已确认]`
- 构造后任何运行时行为（metadata 流转、`update_state_after_alloc` 与 AscendMultiConnector 的交互、finished 集合）均无测试
- `_req_path` 只断言初始为空，无写入/读取路径测试（本身也暂无实现）
- 未验证 `KVConnectorBase_V1.__init__` 被调用后的框架副作用（如 SupportsHMA 协议方法），也未覆盖「父类设置 `_is_kv_producer` 而子类不设置」的偏差
- 本机无 pytest/vllm 环境，**26 个测试的实际通过状态未验证** `[源码可达，尚未运行验证]`

## 6. 缺失能力清单 `[当前源码已确认]`（均为全仓 grep + 逐文件阅读结论）

| 能力 | 状态 | 证据 |
|---|---|---|
| 路径决策（PathDecisionCoordinator / PathStrategy 实现） | **完全不存在**。只有 `PathStrategyCfg` 配置 dataclass 与 `PATH_STRATEGY_FEATURES` 常量；无任何决策类/函数；`_req_path` 侧表空转 | `config.py:29-48,98-101`；`connector.py:71`；grep `PathDecisionCoordinator` 零命中 |
| control transport（独立控制面通道） | **完全不存在**。只有 `RelayCfg` 配置占位（zmq rep/pub 端口默认值）；无 socket/thread/协议代码。现存的 zmq side channel 属于父类 layerwise 握手机制，非 dual-path 控制面 | `config.py:79,104-108`；`mooncake_layerwise_connector.py:1196` |
| Reverse 传输（DE→PE 反向数据面） | **完全不存在**。无 forward/reverse 语义端口、无反向路径代码 | grep `reverse_port|forward_port` 零命中 |
| BlockOwnershipLedger | **完全不存在** | grep 零命中 |
| TransferFence | **完全不存在** | grep 零命中 |
| Store adapter（与 AscendStoreConnector 的桥接） | **完全不存在**。dual_path 目录内无对 ascend_store 的任何 import/引用；`AscendMultiConnector` 仅以 sibling 方式并存 | `ascend_multi_connector.py:10` 只 import MooncakeLayerwiseConnector |
| WorkerMetadata（专属） | **不存在**。dual_path 无 `KVConnectorWorkerMetadata` 子类、无 `build_connector_worker_meta` override；沿用父类 `MooncakeLayerwiseConnectorMetadata`（`connector.py:123`）。（对照：ascend_store 有 `AscendStoreKVConnectorWorkerMetadata`，`config_data.py:1102`） | grep `WorkerMetadata` 在 dual_path 零命中 |
| LinkMonitor | **完全不存在**，仅 docstring 提及（`connector.py:6,83`；`__init__.py:9`） | grep `LinkMonitor` 在 vllm_ascend/ 仅命中注释 |
| Topology 实现 | **完全不存在**。只有 `TopologyCfg` 配置 + static_yaml 文件存在性检查；无 `class *Topology` 定义、无 ubutils/sysfs/udev 采集代码 | `config.py:130-133,284-304` |
| shadow mode 决策 | **未实现**。docstring 声称「later / shadow wiring」（`connector.py:6-10,55-58`），代码中无任何 shadow 计算或打点 | grep `shadow_mode` 零命中 |
| `kv_load_failure_policy` 与 DualPath 的联动 | **未实现**（见 §4）；recompute 语义当前由平台层/upstream 调度器通用逻辑承载，DualPath 无感知 | `platform.py:892-899`；upstream `vllm/config/kv_transfer.py:69`（参考版本存在小幅偏差） |

已存在且可复用的相关件：`DualPathConfig` 校验框架、`DualPathConnector*` 三个继承骨架、`_req_path` 空侧表、注册链路、UT 骨架（含父签名哨兵）。父类提供的现成机制：layerwise 发送/接收后台线程、zmq handshake side channel、`get_finished`/`get_block_ids_with_load_errors` 完成集合与 invalid-block 上报（`mooncake_layerwise_connector.py:1400-1438`）。

## 附：与常见假设不符/需注意的点

1. `DualPathConnector.__init__` 不复制父类的 `self._is_kv_producer`（`mooncake_layerwise_connector.py:701` vs `connector.py:120-127`）；当前全仓无人读取该属性，属于「当前无害的静默偏差」。
2. `DualPathConfig.from_extra_config` 的 fail-fast 在 **Worker 进程同样执行**——`topology.static_yaml` 的文件存在性检查会在每个 worker 进程跑，部署上该文件需对两类进程可见。
3. `path_strategy.type` 允许 `"static"`/`"adaptive"` 通过校验，但二者没有任何实现，配置后静默等效于默认 value_function（且 value_function 本身也无实现）。
4. `_req_path` 存在于 scheduler 实例而非 `ReqMeta`，docstring 称是为了躲过父类对 req_meta 的 `copy.deepcopy`（`connector.py:55-58`）——该深拷贝断言本身未在本阶段独立验证。
5. 父 worker `__init__` 有进程级副作用：`os.environ["ASCEND_TRANSFER_TIMEOUT"]=...`（`mooncake_layerwise_connector.py:1131`），DualPathConnectorWorker 经 super() 继承该副作用。

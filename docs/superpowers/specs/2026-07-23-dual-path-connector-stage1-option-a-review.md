# DualPathConnector Stage 1 方案 A″ 评审意见

> 评审对象：`2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md`
>
> 状态：评审进行中
>
> 说明：本文记录已经共同确认的评审结论。设计稿正文的对应修改在评审完成后统一处理。

## R-001 共享 Mooncake runtime

**结论：通过。**

方案 A″ 可以让 Forward 和 Reverse 共用同一个 engine-local
`MooncakeLayerwiseConnectorWorker` runtime。

这里的“共享”以每个 Engine 的每个 TP Worker rank 为边界：

- 每个 `DualPathConnectorWorker` 创建一个
  `SharedMooncakeTransferRuntime`；
- runtime 内部复用一个以 `kv_both` 初始化的
  `MooncakeLayerwiseConnectorWorker`；
- 正式 KV buffers 只注册一次；
- 只启动一个 send thread 和一个 recv thread；
- PE 的 send thread 承担 Forward，recv thread 承担 Reverse；
- DE 的 send thread 承担 Reverse，recv thread 承担 Forward；
- `ForwardLayerwiseTransfer` 和 `ReverseLayerwiseTransfer` 是同一 runtime
  上的方向化 adapter，不创建第二个 Mooncake Worker。

该共享不表示 PE 与 DE 跨进程共享同一个 runtime，也不表示两个方向复用
相同的 wire identity、receiver endpoint、request metadata 或 fence。
Forward 和 Reverse 的协议身份及生命周期仍须保持独立。

实施约束：

1. `register_kv_caches()` 只能调用一次；
2. DualPath 必须绕开父 Worker `start_load_kv()` 的 consumer-first
   dispatch；
3. `SharedMooncakeTransferRuntime` 是 Mooncake runtime 启动、drain 和
   shutdown 的唯一 owner；
4. Store adapter 和 Scheduler control transport 不属于该共享 runtime。

## R-002 Layerwise 任务与完成事件语义

**结论：通过，但设计稿需要同步收敛事件定义。**

Stage 1 保持以下语义：

> 数据面逐层提交传输任务，正确性和资源生命周期只依赖请求级最终完成事件。

正式事件契约：

- `ReverseLayerCommand`：逐层 Reverse 传输任务；
- `ForwardLayerCommand`：逐层 Forward 传输任务；
- `REVERSE_DONE`：请求的全部 Reverse layer 已完成，是 Reverse 的唯一
  成功终态；
- `FORWARD_DONE`：请求的全部 Forward layer 已完成，是 Forward 的唯一
  成功终态；
- 某一 layer 传输失败时，立即冻结为请求级 `FAILED`；失败事件可以携带
  `layer_name` 用于诊断，但不存在对应 layer 的成功终态；
- PE 只有在请求级 `REVERSE_DONE` 后才能发布 `finished_recving` 并开始
  模型计算；
- `DE_PARTIAL_READ` 仍以 `STORE_DONE && FORWARD_DONE` 作为 DE 成功谓词；
- block ownership 保守持有到对应请求级终态或明确失败并完成
  drain/quarantine 后释放。

Stage 1 不要求：

- 每层 receiver ACK；
- 可参与正确性判断的 `REVERSE_LAYER_DONE`；
- 可参与正确性判断的 `FORWARD_LAYER_DONE`；
- Reverse 与 PE compute 的跨层流水重叠。

若保留逐层耗时、提交数或失败层位置，应作为 telemetry，不得参与：

- Scheduler accounting；
- `finished_recving` / `finished_sending`；
- 请求成功谓词；
- block validity；
- ownership release。

设计稿后续需要同步检查并修正：

1. 第 10.3 节事件枚举中的 `REVERSE_LAYER_DONE` 和
   `FORWARD_LAYER_DONE`；
2. 第 11.2 节以 layer done 表达的依赖链；
3. 第 12 节按 layer done 释放 ownership 的规则；
4. 第 14.3 节看起来允许 Reverse、PE compute、Forward 跨层交叠的时序；
5. 第 23、24、25 节相关测试、可观测性和验收表述。

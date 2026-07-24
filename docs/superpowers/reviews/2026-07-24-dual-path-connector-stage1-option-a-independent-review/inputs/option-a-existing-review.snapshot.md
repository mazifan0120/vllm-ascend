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

## R-003 DE 正式 blocks 的路径无关分配与 PE_READ Forward 区间

**结论：DE 提前分配正式 blocks 的设计通过；PE_READ Forward 区间错误已修正。**

DE 在 `PathDecision` 前按 `E_DE = max(R - L_DE, 0)` 分配的 blocks
不是 DE Read 专属 blocks，而是 Decode 最终模型读取的路径无关目标
blocks。三条路径分别通过以下方式填充：

- `PE_READ`：PE Forward 写入；
- `DE_FULL_HIT`：DE Store 写入；
- `DE_PARTIAL_READ`：DE Store 写入前缀，PE Forward 写入尾部。

未提交路径不能启动 Store/P2P I/O。decision 超时、失败或请求取消时，
必须 abort pending probe，并在没有 in-flight owner 后释放这些 blocks。

原设计把 `PE_READ` 的 PE compute 区间和 Forward 区间都写成
`[K_PE, R)`，会使 DE 缺失的 `[L_DE, K_PE)` 没有数据来源。现已统一为：

```text
PE Store load: [L_PE, K_PE)
PE compute:    [K_PE, R)
PE Forward:    [L_DE, R)
DE target:     [L_DE, R)
```

因此 `PE_READ` 的 Forward 同时包含 PE 本地已有、PE Store 加载及 PE
新计算但 DE 尚未拥有的 KV。

## R-004 删除请求级 RankArmAck，保留本地 raw terminal

**结论：通过。**

当前 Mooncake Layerwise 是基于初始化时完成的 KV buffer 注册和常驻
receive thread 的单边写模型，不要求每个请求重新 post/arm receive
buffer。因此不为 Stage 1 增加请求级 `RankArmAck`、跨 Scheduler/Worker
barrier 或 `ARM_ONLY -> DATA_START` 两阶段 Worker step。

原 `RankArmAck` 想规避的实际风险是：raw DONE/FAILED 已到达 Receiver
Worker，但 `wire_external_id -> engine-local request ID` mapping 尚未建立。
若直接调用父 Worker 的 clear-and-filter 逻辑，无法映射的终态可能被永久
丢弃。

DualPath 自定义 `get_finished()` 必须采用本地 raw terminal retention：

```text
raw DONE/FAILED 到达
  -> 按 wire ID + direction + TP rank 暂存
  -> mapping 未建立时不丢弃、不发布公共终态
  -> mapping 建立后校验 fence/identity/plan digest
  -> 归属到精确 engine-local request ID
  -> 发布请求级 finished_recving 或失败终态
```

长期无法匹配、身份冲突或超过保留期限的事件进入协议错误或 quarantine。
该机制只解决终态事件早于 ID mapping 的乱序问题，不承担数据面 buffer
注册、block 分配、路径选择或传输同步。

## R-005 删除 transfer epoch，复用 DE request identity

**结论：通过。**

Stage 1 不存在同一个请求在失败、取消或 commit 后重新开启第二个 transfer
attempt 的语义；控制消息超时重试复用相同 `message_id`，运行期失败则终止
请求。因此 `transfer_epoch` 没有生成、递增或退休的合法状态转换。

现有 Mooncake raw DONE/FAILED 只携带 external request ID，不携带 epoch。
如果 wire ID 被复用，本地保存 epoch 也无法判断迟到事件属于哪一代；如果
wire ID 永不复用，epoch 又是重复信息。因此 Stage 1 删除
`transfer_epoch`，用不可复用的方向化 wire identity 解决迟到事件归属：

```text
DualPathRequestKey =
  (de_engine_incarnation, de_engine_local_request_id)

forward_wire_external_id = derive(request_key, "forward")
reverse_wire_external_id = derive(request_key, "reverse")
```

`DualPathRequestKey` 是对已有 DE request identity 的复用，不新增独立
`route_request_id`。PE/DE control、fence 和诊断以该复合 key 关联；公开
`finished_*` 仍分别使用各 Engine 的精确 local request ID。

同一次 control retry 或 Scheduler 重调度复用原 key 和 wire IDs；新请求
执行使用新的 DE request identity。请求终止后保留 retired wire ID
tombstone，直到 transport drain 或 terminal retention window 到期。
Forward/Reverse wire IDs 永不重新分配。未来只有在支持 post-commit
recovery 且同一逻辑请求确实需要多个 transfer attempt 时，才新增显式、
不可复用的 `transfer_attempt_id`。

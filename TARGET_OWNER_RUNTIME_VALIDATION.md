# Target owner A1/A2 实现与验证记录

2026-11-04。用户目标：按 mc.pdf 的所有权划分，实现 D owner-side TargetRuntime，
导出既有 Host Main / device Indexer 分配及注册 catalog，并提供稳定 owner facade。
64K 数值问题保持暂停。

## 代码位置及范围

- 当前交付：`src/vllm-ascend-reuse` / 原 `p-main-buffer-reuse` 分支；保留 P 历史，纳入已测 receipt 基线 `6c233175d` 后提交 A1/A2。
- 原生验证最初在 `src/vllm-ascend-pd-integration` 完成；迁回后核验生产源码与该通过快照逐文件一致。
- 原 receipt 评审 worktree 保持固定提交，固定评审树不改动；未执行远端 push。
- 早期 S1 草稿单独归档在 `artifacts/p-main-buffer-reuse/target-owner-runtime/deferred-s1/`，
  当前生产代码 diff 限于 A1/A2 及其资源持有和初始化接线。
- 完整接口说明：集成仓 [owner 交接](TARGET_OWNER_RUNTIME_HANDOFF.md)。

## 已实现

A1 使用原池和 TE 的真实 ownership pin，取得 scheduler 权威 guard；prepare 必须等待
实际 Host-visible fence 并在指定 consumption stream 安装已记录的原生 Indexer event。
`use()` 记录计算、fused-plan 和 writeback stream 的完成事件，release 有界等待静止后才
释放本地引用。失败保留资源；部分 acquisition 回滚失败会 quarantine，支持 owner 本地显式重试。
没有增加 KV 数据副本、请求 blocks、全局 device synchronize 或 block ref_cnt。

A2 从 manager 的真实 cache spec、Main Host view、Indexer tensor/storage 和成功 TE
注册生成 catalog。保留原 storage，拒绝后续指针/形状/stride/dtype 被替换；校验容量、
实际注册边界和组件。Host backing UUID 由实际 TP owner 协商，跨进程地址不参与身份比较。
旧 registration/owner 不允许重绑。Receipt v1 的一 backing 多 MR 仍显式拒绝。

worker 在分配和注册前即可暴露同一个 deferred facade；成功注册后由 manager 构造真实
provider，未绑定时调用失败。scheduler/client 和 owner/dispatcher 只传纯数据，tensor、
事件、锁、基址、pins 保留在 owner。client 的实际生产 transport 由 B 接入。

## 验证

证据目录：`artifacts/p-main-buffer-reuse/target-owner-runtime/`。

| 验证 | 结果及边界 |
| --- | --- |
| lifetime/runtime/provider CPU | 40 项通过，包括真正父子进程 Pipe、失败回滚、旧代次、显式 close/unregister 保护及布局替换 |
| 原 P 聚焦回归 | 139 项通过 |
| receipt 纯账本 | 27 项通过，不代表既有 B 协议问题已修复 |
| TE/HostPool 回归 | 16 项及 4 子用例通过；修正基线已有的 stale 测试断言，冻结基线也有同样 7 个失败证据 |
| 真实 vLLM scheduler | 实际分配 prompt、decode 追加、ref_cnt 保持、活动 guard 拒绝 retire、确认后 free 通过 |
| TP1/TP2 原生目标资源 | 最终源码快照 TP1 和 TP2 两 rank 均通过，包含实际 connector A2 接线方法及 storage 加固 |

原生验证使用真实 Mooncake shared segment、HostRegister、TE/HIXL RoCE 注册、BF16
Main/device Indexer、真实 NPU stream/event。使用小规模合成 KV，覆盖内容及异步使用释放保护。
Host 可见性 fence 在测试中对应本地真实写完成，不能当作已验证远端 PD Host-visible 回执。

运行脚本已经进入源码仓 `tests/e2e/kv_transfer/target_owner_runtime_npu.py`：

```bash
python3 tests/e2e/kv_transfer/target_owner_runtime_npu.py --hostname <本机TE地址> --tp 1
python3 tests/e2e/kv_transfer/target_owner_runtime_npu.py --hostname <本机TE地址> --tp 2
```

需要先空出所用设备，并按部署环境配置正确可见设备和 RoCE。该手动验证脚本不负责停服务。
本轮控制脚本只操作已捕获 ID/镜像/owner 的自有开发容器，记录源码摘要，不写主机配置。

## 实验中修正的测试条件

- HostRegister 的 NPU VA 用 wildcard 注册会被当前 native backend 误识别为 Host 并失败；
  使用基线已有的显式 `npu:<rank>` registration location 后成功，未更换物理池。
- 通用 `.cpu()` 读取该 HostRegister device alias 出现原生崩溃；验证脚本改由同一 shm 的
  真实 CPU VA 读取。NPU 原写入和 writeback 保持实际发生，未用复制出来的 KV 代替目标。
  本轮没有继续定位通用 memcpy 的底层算子问题。
- 新 vLLM TP 初始化要求 `set_current_vllm_config`，补齐测试上下文后 TP2 通过。
- 实际 `Request` 构造已去掉 `eos_token_id`，修正脚本后真实 scheduler 验证通过。
- 上述失败/重试日志全部保留，不归因为 64K 数值问题或完整 PD 服务故障。

## 仍需 B/J 联调的部分

A1/A2 提供 owner 能力和调用入口；生产模型调用链仍需 B 补齐：完整 Host/Indexer 注册、
目标 admission/worker-ready、真实远端 Host-visible 和 device 完成依赖、forward `use()`
范围、scheduler-worker/retention IPC、sender 五方法及 receipt 协议修正。
正常/取消/抢占必须先证明远端及跨 rank DMA 静止，再给安全 ReleaseAck、retire 和 free；
本地 release 不能代替这个许可。restore-only 调度、完整稀疏模型 decode/写回和长上下文
性能也未在本轮宣称完成。

## 环境恢复

原 P r05 在原生资源验证期间保持运行；原 D r03 双节点在验证期间暂停，测试容器为独立 r14。
原 D 已恢复并通过健康检查。第一次恢复审计错误地给原 P r05 发送了普通
`max_tokens=16` 聊天请求，未沿用已有 P-only 的 `max_tokens=1` 检查，触发
`Main reuse could not open the complete committed Store prefix`，P0 退出，P1 因 DP
通信断开退出。该失败发生于未安装本轮 A1/A2 代码的原 r05，不作为新特性回归通过。
未继续定位或修复这条旧 P 多步恢复路径；保留失败日志并重启原 ID 的 P r05 双节点。
最终恢复审计通过：四引擎健康且空闲；原 ID/镜像/owner/启动脚本匹配，每节点 10 个
原源码文件摘要匹配；P0/P1 的 368/363-token 冷短请求各生成一个 token `4`，耗时
1.81/0.52 秒。D 成功请求计数仍为 0，proxy 保持原未运行状态，r14 停止并标记测试退役。
证据为 `restoration-audit.json`；失败记录未删除，未宣称旧 P 多 token decode 通过。

## 归回原分支后的提交检查

当前交付目录为 `src/vllm-ascend-reuse`，分支 `p-main-buffer-reuse`；保留原 P 历史，
快进纳入已验证的 receipt 基线后加入 A1/A2。迁回的生产源码及 NPU 验证脚本共 733 项
摘要与原 TP2 通过快照一致。CPU 40/139/27 项及 TE/HostPool 16 项、4 子用例复跑通过。
本次变更的全部 manual pre-commit 检查通过，包括 Markdown、拼写、导入规则和 secret scan。
合法的 transfer-engine 缩写加入拼写词表；仅四个可信值序列化/拒绝活对象测试加入 pickle 导入例外，生产导入限制不变。

已在隔离快照执行 `bash format.sh ci`。全仓仍有继承的 Ruff 错误和格式差异，例如
`vllm_ascend/ascend_config.py` 的长行和 `vllm_ascend/attention/sfa_kv_offload.py` 的
未使用变量；全量 typos 还报告旧配置注释和 receipt 测试注释的拼写，进程退出码含 -9。
这些文件与 receipt 基线逐字节一致，不属于本次改动。全量格式检查不能记为通过；
其自动修改仅发生在隔离验证快照，没有带入交付目录。
证据在部署仓库 `artifacts/p-main-buffer-reuse/branch-consolidation/`。

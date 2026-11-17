# P buffer reuse 与 receipt集成开发计划

2026-11-05：用户已将范围扩展为补齐并验证完整 PD 功能与性能；Mooncake 五档性能测试
已完成、原环境已恢复。现在继续在原 `p-main-buffer-reuse` 纳入最新 receipt 分支并开发，
不再以早期 A1/A2 接口交付作为整体完成标准。
最新缺口、已有修复和下一步依赖顺序见 [完整链路审查](PD_TRANSFER_COMPLETION_REVIEW.md)。

2026-11-04，用户授权直接开始实现；上一阶段验证与下一阶段独立模块开发允许并行。
64K数值问题暂不定位，不以它解释或跳过新链路的正确性要求。

## 工作区及边界

- 集成起点：`6c233175d95909ec8df1175f85e5e212496fa6ec`，已包含P分支`06e16d717`。
- 当前开发和交付位置：部署仓库 `src/vllm-ascend-reuse`，原分支 `p-main-buffer-reuse`。
- 原分支保留 `06e16d717` 历史，快进纳入已验证的 receipt 基线 `6c233175d`，再提交 A1/A2；不改写历史，不自动 push 或部署服务。
- 复用现有Host池、稀疏offload manager和算子；不另分配生产KV池或请求blocks。
- 第一条目标链路：同TP、V1 eager、PP/CP/DCP=1、无MTP，BF16 Main及已核实的Indexer布局。
- 需要的自有开发容器操作沿用身份、空闲检查和原服务恢复门禁；不做host/device reset、主机配置修改或Master清理。

## 当前执行范围

用户后续将本轮目标明确为 mc.pdf 对应的 **A1/A2**：真实 D owner runtime，及从
既有分配/TE 注册导出的 catalog 和稳定 facade。实现最初在隔离工作区验证，现已按用户要求统一回 `src/vllm-ascend-reuse` 的
`p-main-buffer-reuse`；原隔离目录仅作历史快照，固定 receipt 评审树不改动。
S1 早期协议修改已单独归档到
`artifacts/p-main-buffer-reuse/target-owner-runtime/deferred-s1/`，不混入本轮 A1/A2。
B3、sender、完整 scheduler IPC/free hook 和端到端集成仍是后续工作，以下全阶段表保留作路线图。

当前 A1/A2 进展：40 项新增/相关 CPU UT、139 项 P 回归、27 项 receipt 账本通过；
TE/HostPool 16 项及 4 子用例通过。最终快照 TP1/TP2 原生小规模目标资源测试通过，真实 vLLM scheduler 验证及
最终环境恢复审计完成；旧 P 多 token 审计失败单独记录，正确单 token P-only 检查通过。具体接口见 [owner 交接](TARGET_OWNER_RUNTIME_HANDOFF.md)，验证范围和恢复记录见 [验证报告](TARGET_OWNER_RUNTIME_VALIDATION.md)。

## 阶段与验收

| 阶段 | 开发内容 | 本阶段验收 | 可并行的下一步 |
| --- | --- | --- | --- |
| S0 基线与测试入口 | 固定集成树，修正connector测试收集，建立真实方法CPU回归 | 139项P聚焦、27项账本及缺陷复现基线留档 | S1缺陷修复 |
| S1 协议与普通模式 | 关闭模式不带receipt身份；完整层／组件／writer范围计划；失败terminal；visibility强制门禁；身份和清理；目标admission顺序 | 缺层、单路径失败、旧代、普通模式、空payload、错误provider不得误放行；已有回归不退化 | S2 owner runtime及catalog独立模块 |
| S2 D owner运行时 | 真正的retain／make_visible／release，绑定既有分配；注册catalog及可延迟就绪facade；scheduler retention一致性 | owner引用和block retention分离；失败保留，release静止；跨进程纯数据与owner状态测试 | S3 sender的纯计划器和源consumer |
| S3 P sender和目标数据面 | 五方法consumer接线；源epoch／ready统一；真实目标描述；Main唯一writer和Indexer各rank；Host／device分批及有效tail | 实际地址范围／writer／stride校验；慢reader、B=1/2/3、空GET、partial、失败和迟到回调内容测试 | S4生命周期接线及配置 |
| S4 调度和退出 | scheduler-worker通道、admission就绪握手、自动restore-only、取消／抢占、跨rank/DMA及写回静止回收、有界错误 | 取消后新请求不被旧写入污染；无计算恢复可发送；丢回执有界失败；失败不伪造DMA静止 | S5部署包和独立硬件准备 |
| S5 短上下文完整联调 | 同精度配置、隔离开发容器、真实Host Main／HBM Indexer模型链路 | 几K至几十K、连续decode／写回、prefix、并发、取消、普通模式回归及最终恢复审计 | 只准备长上下文实验，不依赖未通过结果上线 |
| S6 长上下文和性能 | 在短链路通过后逐步加长，测显存、TTFT、传输与屏障等待 | 逐级内容和资源回收正确；单独记录数值问题及性能 | 按实测决定优化，不提前保证1M端到端 |

## 允许overlap的规则

1. 每个验证使用固定源码快照／文件摘要和独立日志，不能一边修改被测实现一边沿用其通过结果。
2. CPU检查在后台运行时，主开发继续下一阶段不依赖结果的模块；失败先回到对应阶段修复。
3. NPU／网络测试不并发占用同一设备或改写同一目标，下一阶段只做本地开发和独立资源准备。
4. 验证结果只覆盖其版本；跨阶段接线后重新执行受影响回归。没有真实完成依赖时不发布ready或归还blocks。
5. 计划文档、实现清单和部署runbook随阶段进展更新；不把接口／mock通过标作NPU或端到端通过。

## 先冻结的接口语义

- P物理槽epoch、Store snapshot generation、PD transfer generation与D allocation/registration generation分别管理。
- required layers／components／目标范围从目标计划产生，不能从收到的成功消息反推。
- 源ticket使用真实源代次，目标结果使用真实目标身份；不以空backing或假ID绕过共享契约。
- `prepare_decode`成功才可visible；失败、取消与source/DMA静止是不同状态。
- provider在owner进程持有状态；scheduler/worker facade传纯数据并等待真实确认，不复制含锁或tensor的provider。
- 在target admission和D worker准备完成前不通知P开始；跨进程握手和错误返回不能只靠日志。

## 进展

- [x] 两分支代码和剩余清单已整理，原服务恢复审计已通过。
- [ ] S0：固定开发分支及可运行的回归入口。
- [ ] S1：协议和普通模式修正。
- [x] S2 / A1/A2：owner runtime、catalog/facade及初始化入口实现完成，CPU/TP1/TP2原生验证与恢复审计通过；生产B/J接线另列S3/S4。
- [ ] S3：源consumer和目标传输。
- [ ] S4：scheduler生命周期、restore-only及退出。
- [ ] S5：短上下文真实目标链路验收。
- [ ] S6：长上下文和性能验收。

本计划是执行顺序，不是已完成声明；具体产物和失败证据按阶段追加。

# 04 — 已有评审对照（Phase 4，冻结后执行）

**执行声明**：本文件在独立 Finding Ledger 冻结（`03-independent-findings.md`，指纹 `492a5a25…365cf1`）之后编写，existing review snapshot 在冻结前未被读取。对照仅用于遗漏检查与结论验证，未改写任何已冻结 IR；对照触发的二次验证发现编号为 RF-xxx。

输入：`inputs/option-a-existing-review.snapshot.md`（177 行，SHA-256 与 manifest 一致），含 R-001～R-005 五条「已共同确认」结论。

## 对照矩阵

| 条目 | 判定 | 对应 IR/POS | 一句话依据 |
|---|---|---|---|
| R-001 共享 Mooncake runtime | 与独立发现一致 | POS-12；IR-014；过度设计矩阵 | kv_both 单次注册/唯一线程对经源码证实（MLC:1359-1398 + register_buffer 守卫）；consumer-first 绕行必要（F5/F6） |
| R-002 逐层事件收敛 | **补充了独立评审遗漏** | → RF-001 | 设计 §10.3/§11.2/§12/§14.3 确实仍以逐层事件参与正确性，独立评审只在过度设计矩阵中提出弱化版简化建议 |
| R-003 路径无关 blocks + Forward 区间 | 部分正确 | IR-005、IR-011；POS-4 | 区间修正已在 snapshot 落实且 accounting 闭合；但「超时/失败/取消时 abort probe 并释放 blocks」的兑现机制不足 |
| R-004 删除 RankArmAck、本地 raw terminal | 与独立发现一致 | POS-7；IR-027、IR-037 | 早到事件丢弃缺陷（F8）与 retention 修复方向均经源码证实；独立评审补充了 retention 量化与决策面 tombstone 缺口 |
| R-005 删除 transfer epoch | 与独立发现一致 | 过度设计矩阵（正确 YAGNI 回避） | Stage 1 无 post-commit 重开语义，wire ID 不可复用使 epoch 成重复信息 |

---

## R-001 共享 Mooncake runtime —— 与独立发现一致

- 当前设计位置：§9.2 L872-910、§10.2 L1467-1509、§18.2 L2397-2400。
- 当前源码证据：`MLC:1359-1398`（kv_both 一次 `register_kv_caches` 建立 send+recv 线程与 buffer metadata）、`mooncake_transfer_engine.py:31-40`（`register_buffer` 单次守卫，二次调用静默跳过——「只能调一次」是硬约束而非选择）、`MLC:1606-1615`（kv_both 下 `start_load_kv` 只走 consumer 分支，绕行确有必要）、`MLC:1006-1022`（kv_both 下 `build_connector_meta` 恒走 consumer 分支，send 元数据不产出——DualPath 必须完全覆盖，设计 §9.2 L905-908 已要求）。
- 成立所需前提：方向化 adapter 完全覆盖 dispatch（§9.2 L905-908）；Forward/Reverse wire identity 独立（§6）。
- 未覆盖的反例或时序：R-001 未提及共享 send thread 继承 MLC:507 失败归因 bug（IR-014），多请求同 SendTask 时失败归因错误会同时污染两个方向；也未提及父 worker init 的进程级副作用 `os.environ["ASCEND_TRANSFER_TIMEOUT"]`（MLC:1131）被共享 runtime 继承。
- 是否还需要修改设计：不需要结构性修改；建议把 IR-014 的归因规避写入实施约束。
- 对应 IR Finding：POS-12（正向）、IR-014。

## R-002 Layerwise 任务与完成事件语义 —— 补充了独立评审遗漏（→ RF-001）

- 当前设计位置：§10.3 L1517-1526（事件枚举含 `REVERSE_LAYER_DONE`/`FORWARD_LAYER_DONE`）、§11.2 L1608-1616（依赖链以 layer done 表达）、§12 L1734-1743（ownership 按 layer done 释放）、§14.3 L2019-2023（时序图逐层交叠）、§23/24/25 相关表述。
- 当前源码证据：与 R-002 结论相容——Stage 1 设计自身已承诺「必须等待完整 REVERSE_DONE 后才发布 finished_recving」（§11.2 L1618-1621），即正确性屏障在请求级；逐层事件在 Stage 1 没有不可替代的正确性消费者。
- 独立评审核对结果：R-002 列出的 5 处设计位置经逐一对照 snapshot 属实：§10.3 枚举、§11.2 链、§12 释放表、§14.3 图确实仍以逐层事件参与正确性表达；§14.3 L2034「PE 每层只能在 REVERSE_LAYER_DONE(layer) 后计算」与 §11.2 L1618 的全量屏障并存，读者无法判断哪个是真实契约。
- 独立评审的相应覆盖：仅在过度设计矩阵中提出「逐层事件可降为计数、PATH_COMMITTED 移出枚举」的简化建议，**未定位到 §11.2/§12/§14.3 的正确性纠缠**——属独立评审遗漏，R-002 补充成立。
- 判定：补充了独立评审遗漏 → 形成 RF-001。
- 是否还需要修改设计：是（文档级收敛，见 RF-001）。
- 对应 IR Finding：独立阶段未发现（仅过度设计矩阵弱相关）。

## R-003 DE 路径无关分配与 PE_READ Forward 区间 —— 部分正确

- 当前设计位置：§5.2 L299-316（Forward 区间 `[L_DE,R)` 已修正并在边界条件中强调）、§7.3 L470-474（路径无关统一分配）、§14.1 L1965-1967。
- 当前源码证据：区间三段构成（本地/Store/计算）与 DE target `[L_DE,R)` 的 accounting 闭合经 Phase 3 token 专项逐行复核（phase3-scratch/02）；`E_DE=max(R-L_DE,0)` 统一声明与 upstream 两段式分配兼容（F4、POS-4）。
- 成立所需前提：Forward mapping 必须覆盖 `[L_DE,R)` 全段——依赖 IR-003（manifest 通道）与 IR-008（tail blocks 冻结）修复。
- 未覆盖的反例或时序：R-003 末段「decision 超时、失败或请求取消时，必须 abort pending probe，并在没有 in-flight owner 后释放这些 blocks」是**要求**而非机制——独立评审发现当前设计缺少兑现机制：pre-commit abort 无终态发布者（IR-005）、DECIDING 取消无 CANCEL 通道与 terminal 义务（IR-011）、决策失败无 scheduler→worker 传播（IR-010）。在 `U:scheduler.py:2144-2147/2580-2586` 的 delay_free 契约下，这些缺口导致 blocks 实际无法「在没有 in-flight owner 后释放」。
- 判定：部分正确——区间与分配结论成立且已在设计中落实；释放要求方向正确但机制不足。
- 是否还需要修改设计：是（以 IR-005/IR-010/IR-011 的修正为准）。
- 对应 IR Finding：IR-005、IR-010、IR-011；POS-4（正向）。

## R-004 删除 RankArmAck、保留本地 raw terminal —— 与独立发现一致

- 当前设计位置：§8.2 L574-577（不增加请求级 RankArmAck + 暂存要求）、§10.3 L1554-1573（暂存/归属/tombstone 规则）。
- 当前源码证据：父 Worker 早到事件丢弃（`MLC:1407` request_map 过滤 + `get_and_clear` 已清空）证实 RankArmAck 想规避的风险真实存在；「初始化时注册 + 常驻 recv thread 单边写」模型属实（MLC:1247-1398）。
- 成立所需前提：暂存必须在 mapping 建立后重新校验 wire ID/direction/incarnation/channel/plan digest（设计 §10.3 L1556-1561 已写明）。
- 未覆盖的反例或时序：独立评审补充两点边界——retention window 无取值/配置/GC 责任（IR-037）；「超过保留期限进协议错误或 quarantine」缺少本地解除触发器（phase3-scratch/03 维度 10）。
- 是否还需要修改设计：小幅（量化 retention + GC）。
- 对应 IR Finding：POS-7（正向）、IR-037；关联 IR-027（决策面对称 tombstone）。

## R-005 删除 transfer epoch、复用 DE request identity —— 与独立发现一致

- 当前设计位置：§6 L339-389（含 L386-389 的 transfer_epoch 回避论证）。
- 当前源码证据：MLC wire 事件只携带 external request ID（MLC:470, 1400-1428 的 request_map 机制）；wire ID 永不复用 + tombstone 的方案与 §10.3 事件校验字段一致。
- 成立所需前提：engine incarnation 真实唯一（engine_id 缺省 uuid4，MLC 引擎初始化）；incarnation 冲突时 tombstone 机制兜底。
- 未覆盖的反例或时序：Engine restart 后 local request ID 复用场景由 incarnation 区分，设计闭合；但决策面（非 wire 面）的迟到 commit 无 tombstone（IR-027）。
- 是否还需要修改设计：否。
- 对应 IR Finding：过度设计矩阵「不引入 transfer_epoch = 正确 YAGNI 回避」；IR-027。

---

## RF Findings（由已有 review 触发的二次验证发现）

### RF-001 设计稿事件定义未按 R-002 收敛：§10.3/§11.2/§12/§14.3 仍以逐层事件参与正确性

- 严重级别：P2
- 类型：文档一致性 / 协议
- 证据等级：`[设计推导]`（设计内部矛盾）；`[当前源码已确认]`（框架侧屏障在请求级）
- 设计位置：§10.3 L1517-1526；§11.2 L1608-1616 与 L1618-1621 并存；§12 L1734-1743；§14.3 L2019-2023、L2034-2035；§23 L2589-2590、§24、§25 L2722
- 源码位置：`U:scheduler.py:2517-2534`（解除 WAITING 的唯一触发是请求级 finished_recving）；MLC:1976-1977（wait_for_layer_load 为 pass，层屏障无框架载体）
- 触发前提：实现者按 §11.2 依赖链/§14.3 图实现逐层正确性逻辑。
- 具体时序：T0 实现 A 按 §11.2 链等待 `REVERSE_LAYER_DONE(layer)` 再放行 PE compute 该层 → T1 实现 B 按 §11.2 L1618 等全量 REVERSE_DONE → T2 两份实现行为分叉且各自能引用设计原文；同时 §12 按 layer done 释放 ownership 使「layer 完成但请求未终态」时 blocks 提前进入可释放判断。
- 为什么不正确：设计自身已选择请求级屏障（L1618）作为 Stage 1 契约，逐层正确性入口是未清理的旧语义残留；R-002 的收敛要求（telemetry-only）与 snapshot 文本直接冲突。
- 可能影响：实现分叉；ownership 提前释放风险（与 IR-013 的 region 过滤叠加时尤其）。
- 与已有模块的兼容性：收敛后更贴近框架（请求级 finished_recving 是唯一公开屏障）。
- 是否属于过度设计：是（逐层正确性语义属可删除残留；逐层事件作为 telemetry 保留）。
- 最小修正建议：按 R-002 五点逐处修改：§10.3 枚举将 `REVERSE_LAYER_DONE`/`FORWARD_LAYER_DONE` 标注 telemetry-only（或移出正确性事件集）；§11.2 依赖链改为请求级 `REVERSE_DONE` 单屏障；§12 ownership 释放锚定请求级终态/明确失败；§14.3 时序图改为全量 Reverse 屏障后再 compute/Forward（或显式标注「逐层展示仅为示意，不构成流水」）；§23/24/25 同步措辞。
- 修正后需要增加的测试：断言「Forward 仅在完整 REVERSE_DONE 后启动」的 UT；ownership 释放时点 UT。
- 置信度：高。

## 独立 Finding 事实错误检查

对照 R-001～R-005 后逐条复核 41 条 IR 的事实基础：**未发现需要推翻的 IR 事实判断**。R-003 与 IR-005/IR-011 是「要求 vs 机制」关系而非冲突；R-002 与过度设计矩阵方向一致且更精确，已以 RF-001 吸收；R-001/R-004/R-005 与 POS-7/POS-12/矩阵判定一致。已冻结 Ledger 无需修正记录。

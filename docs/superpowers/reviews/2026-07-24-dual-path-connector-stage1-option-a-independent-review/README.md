# DualPathConnector Stage 1 Option A″ 独立评审工作区

本目录用于保存 Option A″ 的独立评审 Prompt，以及由评审 Agent 生成的全部
评审产物。

## 使用方式

将 [`GOAL-PROMPT.md`](./GOAL-PROMPT.md) 的完整内容复制给评审 Agent。

Prompt 会要求 Agent：

1. 创建并持续执行一个 Goal；
2. 只使用 `inputs/` 中冻结的设计快照和当前源码进行独立评审；
3. 冻结独立 findings 后，再读取已有 review 快照做交叉验证；
4. 把全部评审文件写入本目录下独立的 `runs/<时间戳>/`；
5. 不读取评审期间继续变化的原设计和原 review；
6. 不修改快照、原文件、vllm-ascend 源码或 upstream 源码。

## 输入文件

以下快照是本次评审唯一允许读取的文档输入：

- `inputs/option-a-detailed-design.snapshot.md`
- `inputs/option-a-existing-review.snapshot.md`
- 当前 vllm-ascend checkout；
- `/Users/leqi/Documents/Code` 下与当前版本匹配的 upstream checkout。

两份快照复制自 2026-07-24 的当前工作树，包含当时尚未提交的文档修改。
来源、行数和 SHA-256 记录在
[`inputs/SNAPSHOT-MANIFEST.md`](./inputs/SNAPSHOT-MANIFEST.md)。

评审开始后，即使 `docs/superpowers/specs/` 下的原文件继续变化，也不得
重新读取原文件或用其刷新快照。

## 输出结构

每次评审必须新建一个 run，禁止覆盖已有 run：

```text
runs/<YYYYMMDD-HHMM>/
├── independent-review.md
└── evidence/
    ├── 00-baseline.md
    ├── 01-current-control-flow.md
    ├── 02-path-timelines.md
    ├── 03-independent-findings.md
    ├── 04-existing-review-cross-check.md
    └── 05-coverage-and-self-check.md
```

`independent-review.md` 是供设计者阅读的完整评审报告；`evidence/` 保存支撑
结论的源码基线、时序、finding ledger、已有 review 对照和最终自检记录。

`inputs/`、`README.md` 和 `GOAL-PROMPT.md` 均为只读。除新建
`runs/<时间戳>/` 外，本目录其他内容不得被评审 Agent 修改。本目录之外
不应产生任何本次评审输出。

# Input Snapshot Manifest

快照时间：2026-07-24（Asia/Shanghai）

来源仓库：

```text
/Users/leqi/Documents/Code/vllm-ascend
branch: dev/dualpath
HEAD: 0ec11a4703b987d3103f8859f184c492f22bde88
```

两份快照均复制自当时的工作树内容，包含尚未提交的文档修改；它们不等同于
上述 HEAD 中的 committed 版本。

## Design snapshot

```text
snapshot:
  inputs/option-a-detailed-design.snapshot.md
source:
  docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-detailed-design.md
lines: 2752
sha256: 9a7be6369d468bd845cb2cf181dd0ac0d3d794eaa7a3fa58ca7be7aa5f852808
```

## Existing review snapshot

```text
snapshot:
  inputs/option-a-existing-review.snapshot.md
source:
  docs/superpowers/specs/2026-07-23-dual-path-connector-stage1-option-a-review.md
lines: 177
sha256: 82980ecdb1d0ab3c976c3ba1c720705065a54539e920cf1b95f548bfb998d3d5
```

## Isolation contract

- 本次独立评审只读取以上两份 snapshot。
- `source` 路径仅用于记录快照来源，不得在评审期间读取或比较。
- snapshot 和本 manifest 均为只读。
- 如果 SHA-256 不匹配，不得从 source 路径自行刷新快照。

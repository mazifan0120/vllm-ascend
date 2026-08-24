# DualPath KV 加载失败清理（wedge 根治）+ Store 读取分批 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 DualPath connector 在 store 加载失败后的 KV 块永久泄漏（EngineCore livelock/wedge），并为非 layerwise 的 mooncake get 路径加分批以根除触发器。

**Architecture:** 三个相互独立但同源的修复——Fix A（decode worker 为“从未提交的 reverse”合成失败上报，使 completion 关闭、块释放）；Fix B（控制通道增加 decode→prefill 方向的失败通知，Prefill 使用独立的接收身份/registry，Decode 通过统一 admission fail-close helper 覆盖 activation 与 worker 失败；Prefill 只在 exact attempt terminal proof 或 frozen admission activation ceiling 建立远端安全性后关闭未 dispatch 的 REVERSE_RECEIVE，已 dispatch 的 attempt 则桥接到真实 worker terminal 和 all-worker barrier 后释放镜像块）；Fix D（`KVCacheStoreRecvingThread` 的整批 `m_store.get` 按 raw/usable staging 双预算分批，参照上游 vLLM 0.23 `_split_disk_offload_load_batches` 并计入 align/padding）。所有关闭/释放动作复用现有 completion 生命周期通道，不新增并行释放机制。

**Tech Stack:** Python 3.12, vllm-ascend (vllm_ascend), pytest (tests/ut, CPU 可跑，conftest 有 mooncake/torch_npu stub), ZMQ (控制通道), msgspec。

## Global Constraints

- Fix D 不新增环境变量、connector extra-config 或运行时开关；backend 报告有限 staging capacity 时始终自动分批。
- usable budget 固定为 raw budget 的 90%，通过代码内不可变命名常量表达；不向用户暴露调参接口。
- cleanup 故障注入只存在于外部 E2E 测试资产，不能进入产品源码、正式 manifest 或最终提交。

## 2026-08-24 Critical lifecycle correction

实现后复审确认：Task 4/8 原文把“尚无 worker report”误当成“binding 尚未 dispatch”。真实事故时序中，Prefill scheduler 已把 `REVERSE_RECEIVE` binding 下发给 worker，但 Decode Store 随后在 Reverse 从未提交前失败。此时 Prefill 不能由 scheduler 直接关闭 completion（worker 可能拥有数据面写入），也不能等待不存在的 wire terminal。

本节及 `.specs/2026-08-24-dualpath-prefill-reverse-terminal-delta.md` 是该窗口的权威修订，覆盖下文 Task 4/7/8 中所有与之冲突的伪代码和测试 oracle：

- `dispatched` 守卫保留；scheduler 只立即关闭尚未 dispatch 的 completion。
- Decode 只在能够证明 exact Reverse attempt 已 `TERMINALIZED` 时，随 ABORT 传播 `(request_key, reverse_attempt_id)` 终端证明。`TERMINALIZED` 表示每个 Decode worker 对该 attempt 要么从未提交，要么已在 wire terminal 获得 Prefill ACK 后停止写入。
- Prefill 对已 dispatch 的 exact attempt 不直接关闭；它向所有 Prefill workers 下发 exact `ReverseReceiveFailureTerminal`。每个 worker 复用 `_consume_reverse_receive_binding(..., succeeded=False)` 贡献一次 failure report；scheduler 收齐现有 all-worker barrier 后才执行 `_run_completion_close_action` 和 `finished_recving` 注入。
- 普通 ABORT 若既不携带 exact terminal proof，也不携带 admission ceiling，记录 exact admission invalid；仅当它仍是 current exact admission 且 current waiting attempt 属于该 admission 时，生成 invalid-block control failure。它不修改 completion state、不关闭 completion、不 stage worker terminal，也不注入 `finished_recving`。
- Prefill control-channel registration 不是一次性 notice latch：同一 exact key 在 scheduler 注销前可依次接收 proofless ABORT、不同 attempt 的 terminalized ABORT。receiver 只按完整 `PathAbortNotice` 去重 sender retry；scheduler-driven unregister 和 coordinator close 才清 registry 及该 key 的 notice dedupe。
- 所有 Decision delivery failure 都是 ambiguous，包括最终返回 `UNKNOWN_REQUEST`、`STALE_CLOSED` 或 `PROTOCOL_ERROR` 的 `PathDecisionRejectedError`。一次较早的 send 可能已被 receiver 接受但 ACK 丢失，registry 随后 transition/unregister，retry 才得到 typed rejection；最终 reply 不能证明历史上从未接受。在协议提供 durable、attempt-specific `NEVER_ACCEPTED` 或 terminal receipt 前，rejected、timeout、ACK loss、cancel、generic exception 与普通 `PathDecisionDeliveryError` 都只能使 admission invalid 并生成 control failure，不得修改/关闭 completion、stage worker terminal 或注入 `finished_recving`。这是 fail-closed 的 availability limitation，本修复不扩展协议。
- `DualPathControlFailureMetadata(request_id, ...)` 继续只表达请求失败/无效块，不能用作 Reverse completion 身份；worker terminal 必须同时校验 admission、attempt、completion id 与 wire id。
- 当前 `test_prefill_peer_abort_after_binding_dispatch_waits_for_zero_report_barrier` 中手工注入 worker report 的 oracle 作废，必须替换为 scheduler→worker→worker metadata→scheduler 的真实闭环。

## 2026-08-24 Admission activation ceiling correction

实现终审又确认一个 exact-proof-only 的可用性窗口：Decode 只能证明 attempt 0 已停止时，Prefill 可能已为同一 exact admission refresh 到 attempt 1；禁止 request-id fallback 是正确的，但 Decode 进入 `ACTIVATION_FAILED` 后不会再启动 attempt 1，后者也就永远收不到 wire terminal。

`.specs/2026-08-24-dualpath-admission-activation-ceiling-design.md` 是该窗口的权威设计，并覆盖下文 Task 7/8 与 `.specs/2026-08-24-dualpath-prefill-reverse-terminal-delta.md` 中所有 exact-proof-only 的旧 oracle：

- `PathAbortNotice` 增加可选的 frozen admission proof。字段缺失表示没有 upper-bound 证据；字段存在且值为 explicit `null` 表示没有任何 Reverse attempt 可能启动；整数 `N` 表示 attempt `<= N` 可能已启动、attempt `> N` 确定从未启动。
- Decode 在 Reverse plan 对 worker 可见前推进 exact-admission publication watermark；publication final gate/watermark/metadata append 与 admission terminal transition/unregister/freeze 必须处于同一 scheduler 串行化域。已通过初始 validation 但在 freeze 后才到达 final gate 的 refresh 必须被拒绝。冻结后同 admission 不允许继续 activation/refresh，late ABORT context 必须保留同一份 ceiling。
- exact terminal proof 只授权它命名的 attempt；ceiling 只额外授权 `> N` 的 attempt（explicit `null` 授权所有 attempt）。两者可同时出现，且不得 fallback 到 request id 或 current attempt。
- Prefill 收到任意合法 ABORT 都先 fail 整个 exact admission。只要它仍是 current exact admission 且入口快照的 current waiting attempt 属于该 key，就生成 current plan 的 `control_failures`/`invalid_block_ids`；该动作不再要求 exact proof 命中 current attempt。
- completion close 仍保留 dispatch 边界：获得 exact/ceiling authority 后，未 dispatch 才允许 scheduler close；已 dispatch 必须 stage exact worker terminal 并等待真实 all-worker barrier。反过来，本地 `dispatched=False` 不是远端未启动的证据；attempt `<= N` 且无 exact proof 时仍保持 open。
- Prefill 首次观测到的显式 ceiling 进入 exact-key ledger；该 ledger 跨 request-state release 保留，与 abort-key registration 一起在最后一个 exact completion 关闭后退休。同一 admission 的后续显式 ceiling 必须完全相同；冲突 ceiling 按 protocol error fail closed，不能提供 ceiling-derived close authority。同包若有独立合法的 exact proof，仍只允许关闭它命名的 exact attempt。

---

## 背景（事故链，已定位）

32k 多轮压测中，decode 侧 DE_READ 整请求 store 读取 30720 token × 144KiB = 4.22GiB > mooncake `local_buffer_size` 4GB（只够 227/240 块），13 个 key 确定性失败 → 请求被 `failure_policy=fail` 杀掉 → **失败收尾永久 pinned 该请求的 257 个 KV 块**（两条 delay_free 路径叠加）→ 后续满长度请求永远分配不到块 → 调度器活锁（stats 被 idle 逻辑降为 DEBUG，看似"静默 25 分钟"），prefill 侧经 REVERSE_RECEIVE completion 镜像泄漏同样饿死。

关键事实：

- 泄漏点 1（decode）：reverse 发送只在 store DONE 后提交（`worker.py:611-614`）；store FAILED → reverse 从未提交 → activation 时打开的 REVERSE_SEND completion（`scheduler.py:1688-1694`）永不关闭 → `_delay_free_for_connector`（`scheduler.py:1950-1958`）永久 True。
- 泄漏点 2（prefill）：镜像 REVERSE_RECEIVE completion 同样永不关闭，且**当前控制通道是单向的**（prefill→decode 发 decision/abort；decode 的 coordinator 无 executor，无法回发），prefill 永远无法得知 decode 侧失败。
- 触发器：`KVCacheStoreRecvingThread._handle_request`（`kv_transfer.py:846-943`）对整个请求发一次不分批的 `m_store.get`。staging buffer 硬约束来自 mooncake 对象语义 API（`batch_get_into_multi_buffers`），单 key（=单 block 全层，Qwen3-8B 约 18MiB）不可再分但远小于 buffer。
- 上游对齐边界：vLLM 0.23 用 raw budget 判断单 key 是否绝对超限，用 usable budget（raw × ratio）约束多 key 批次，并按 4096B 对齐后额外计入 8192B padding。本文保留这一双预算/估算模型；唯一有意识的偏离是：单 key 超过 raw budget 时只把该 key 记为失败、继续处理其余 key，而不是像上游 disk-offload 路径那样放弃整个 GET。这里已有 per-key block failure 记账，局部失败能减少无关块重取。
- 现成的关闭通道：worker `_record_completion_report(completion_id, succeeded=...)`（`worker.py:341-359`）→ scheduler `_aggregate_worker_completion_reports`（`scheduler.py:2026-2075`）→ `_run_completion_close_action`（`:2078+`）→ finished_sending/recving 注入 → 上游释放延迟块。completion 失败时 decode 分支已会构建 `_decode_control_failures`（`:2047`），prefill 分支已有 `_prefill_control_failures` + `_recovery_invalid_block_ids` 清理链（`:2060-2073`）——**Fix B 主要是把"失败"这个消息跨节点送达，后续清理链全是现成的**。
- `TransferCompletionTracker.tally_reports`（`completion_tracker.py:149-173`）按 worker barrier 计数关闭；TP>1 的已 dispatch completion 必须由所有真实 Prefill workers 各贡献一次 terminal report 后关闭，不能用 scheduler forced closure 绕过 barrier。
- 控制投递是有界的：`_deliver_decision` 最多尝试 `_MAX_DELIVERY_ATTEMPTS = 3` 次；未收到合法 ACK 时 Future 以 `PathDecisionDeliveryError` 终止，不会无限重试。一次 retry 的 terminal registry reply 不携带 earlier-send acceptance history，因此 `PathDecisionRejectedError` 与无 reply 的 exhaustion/timeout 一样，均不是 Reverse terminal proof。

## 文件地图

| 文件 | 职责 | 改动 |
|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py` | Backend ABC | Fix D: 新增 `staging_buffer_bytes()` 默认 None |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py` | mooncake 后端 | Fix D: 对普通对象 API 返回 `config.local_buffer_size`，address-direct 返回 None |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | 传输线程 | Fix D: 纯函数 `_plan_get_batches` + recv 线程分批 get |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py` | decode/prefill worker | Fix A: `_fail_unsubmitted_reverse` |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py` | completion 生命周期 | Fix B-1: `force_fail_completion` |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py` | 控制通道/协议 | Fix B-2: `PathDecision.prefill_control_endpoint`；Fix B-3: coordinator prefill 接收端 + decode executor |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py` | connector 配置 | Fix B-3: Prefill-only `prefill_control_port` 严格解析/校验 |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py` | 调度侧状态机 | Fix B-4/5: endpoint activation 捕获、统一 Decode fail-close、Prefill 收 abort+关 completion |
| `tests/ut/distributed/kv_transfer/dual_path/test_*.py` | dual_path UT（conftest 有现成工厂） | 各 Fix 的回归测试 |
| `tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py` | 新建 | Fix D UT；复用该目录 `_mock_deps.py` |

部署注意：pod 启动时从 `/mnt/vllm-ascend-src` untar 本仓库，**代码改动 redeploy 即生效，不需要重建镜像**。提交规范：Conventional Commits + `git commit -s`（仓库 AGENTS.md）；执行时每次 commit 前向用户确认。

---

### Task 1: Fix D-1 — Backend raw staging 能力

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py`（`get` 抽象方法后）
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py`（`get` 实现后，~:273）
- Test: `tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py`（新建，Task 2 复用）

- [ ] **Step 1: 写失败测试（新文件；复用 AscendStore 现有 stub）**

```python
# SPDX-License-Identifier: Apache-2.0
"""Chunked-get batching for the non-layerwise store receive thread."""

from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend  # noqa: E402


def test_backend_staging_buffer_bytes_default_none():
    assert Backend.staging_buffer_bytes(MagicMock()) is None


def test_mooncake_backend_reports_local_buffer_size():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    backend = object.__new__(MooncakeBackend)
    backend.config = MagicMock(local_buffer_size=8 * 1024**3)
    backend._use_fabric_mem = False
    backend._contribute_memory = True
    assert backend.staging_buffer_bytes() == 8 * 1024**3


def test_mooncake_address_direct_backend_has_no_staging_constraint():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    backend = object.__new__(MooncakeBackend)
    backend.config = MagicMock(local_buffer_size=8 * 1024**3)
    backend._use_fabric_mem = True
    backend._contribute_memory = True
    assert backend.staging_buffer_bytes() is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py -v 2>&1 | tail -8`
Expected: FAIL `AttributeError: ... 'Backend' ... has no attribute 'staging_buffer_bytes'`

- [ ] **Step 3: 实现**

`backend.py`，紧接 `get` 抽象方法之后加：

```python
    def staging_buffer_bytes(self) -> int | None:
        """Largest total byte volume one get batch may occupy in this backend's
        client-side staging buffer, or None when the backend has no staging
        constraint (e.g. address-direct backends)."""
        return None
```

`mooncake_backend.py`，`get` 方法之后加：

```python
    def staging_buffer_bytes(self) -> int | None:
        if self._use_fabric_mem or not self._contribute_memory:
            return None
        return int(self.config.local_buffer_size)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py -v`
Expected: all passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py
git commit -s -m "feat(kv_pool): expose store get staging capacity"
```

---

### Task 2: Fix D-2 — raw/usable 双预算 recv 分批 get

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`（模块级纯函数 + `KVCacheStoreRecvingThread._handle_request` :900-935 段）
- Modify: `tests/ut/distributed/ascend_store/test_kv_transfer.py`（现有 `FakeStore` 补 `staging_buffer_bytes()`；`_handle_request` 请求级集成测试）
- Test: `tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py`

**Interfaces:**
- Consumes: `Backend.staging_buffer_bytes() -> int | None`。
- Produces: `_estimate_get_staging_bytes(key_sizes) -> int`、`_plan_get_batches(..., usable_budget_bytes, raw_budget_bytes) -> tuple[list[list[int]], set[int]]`。

设计：raw budget 是 backend 报告的 staging 硬上限；usable budget 固定为 `max(1, int(raw * _GET_STAGING_USABLE_FRACTION))`，其中命名常量为上游默认值 0.9，只限制多 key 批次，不提供配置入口。`_estimate_get_staging_bytes` 对每个 key 分别按上游规则计算 `align_up(sum(key_sizes), 4096) + 8192`，一个 batch 的预算占用是这些 per-key estimates 之和。超过 usable 但不超过 raw 的单 key 单独成批；超过 raw 的 key 只记该 key 失败、其余继续，这是相对上游 whole-GET abort 的有意识偏离。backend 返回有限 staging budget 时自动分批；返回 None 的 address-direct/unbounded backend 保持整批一次 get。

**范围限定（预检确认）**：Fix D 只改 `KVCacheStoreRecvingThread._handle_request`（本次事故路径：DualPath Decode non-layerwise 异步 load）。layerwise 按层线程（`kv_transfer.py:1197`）单层 key 远小于 buffer，`pool_worker.py` 的同步 load 路径不在事故链内——两者均不改动、不扩展。现有 `test_kv_transfer.py` 的 `FakeStore`（`:45`）不是 `Backend` 子类，而修订后的 `_handle_request` 会调用 `staging_buffer_bytes()`，必须补该方法，否则现有用例 AttributeError。

- [ ] **Step 1: 写失败测试（追加到 Task 1 的测试文件）**

```python
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (  # noqa: E402
    KVCacheStoreRecvingThread,
    _estimate_get_staging_bytes,
    _plan_get_batches,
)


def test_plan_get_batches_packs_keys_under_usable_budget():
    sizes = [[1 * 1024**2], [2 * 1024**2], [3 * 1024**2], [4 * 1024**2], [512 * 1024]]
    usable = sum(_estimate_get_staging_bytes(key_sizes) for key_sizes in sizes[:3])
    batches, oversized = _plan_get_batches(
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=8 * 1024**2,
    )
    assert batches == [[0, 1, 2], [3, 4]]
    assert oversized == set()


def test_plan_get_batches_runs_key_between_usable_and_raw_alone():
    sizes = [[1 * 1024**2], [5 * 1024**2], [1 * 1024**2]]
    batches, oversized = _plan_get_batches(
        sizes,
        usable_budget_bytes=4 * 1024**2,
        raw_budget_bytes=6 * 1024**2,
    )
    assert batches == [[0], [1], [2]]
    assert oversized == set()


def test_plan_get_batches_marks_only_key_over_raw_and_continues():
    sizes = [[1 * 1024**2], [7 * 1024**2], [1 * 1024**2]]
    batches, oversized = _plan_get_batches(
        sizes,
        usable_budget_bytes=4 * 1024**2,
        raw_budget_bytes=6 * 1024**2,
    )
    assert batches == [[0, 2]]
    assert oversized == {1}


def test_estimate_get_staging_bytes_includes_alignment_and_padding():
    assert _estimate_get_staging_bytes([4097]) == 16 * 1024


def _make_thread(backend):
    thread = object.__new__(KVCacheStoreRecvingThread)
    thread.m_store = backend
    return thread


def test_chunked_store_get_merges_per_key_results():
    backend = MagicMock()
    backend.get.side_effect = [[0, 0, 0], None, [0]]
    thread = _make_thread(backend)
    keys = ["k0", "k1", "k2", "k3", "k4"]
    sizes = [[1 * 1024**2], [2 * 1024**2], [3 * 1024**2], [7 * 1024**2], [512 * 1024]]
    usable = sum(_estimate_get_staging_bytes(key_sizes) for key_sizes in sizes[:3])
    raw = _estimate_get_staging_bytes(sizes[3])
    ret = thread._chunked_store_get(
        keys,
        [[0]] * 5,
        sizes,
        usable_budget_bytes=usable,
        raw_budget_bytes=raw,
    )
    # 批次 [0,1,2] 全成功、单独批次 [3] 的 None→失败、[4] 成功。
    assert ret == [0, 0, 0, 1, 0]
    assert backend.get.call_count == 3


def test_chunked_store_get_oversized_key_failed_without_issuing():
    backend = MagicMock()
    backend.get.return_value = [0, 0]
    thread = _make_thread(backend)
    sizes = [[1 * 1024**2], [7 * 1024**2], [1 * 1024**2]]
    ret = thread._chunked_store_get(
        ["k0", "big", "k2"],
        [[0]] * 3,
        sizes,
        usable_budget_bytes=4 * 1024**2,
        raw_budget_bytes=6 * 1024**2,
    )
    assert ret == [0, 1, 0]
    assert "big" not in backend.get.call_args.args[0]


def test_staging_get_budgets_uses_fixed_reserve_fraction():
    backend = MagicMock()
    backend.staging_buffer_bytes.return_value = 10_000
    assert _make_thread(backend)._staging_get_budgets() == (10_000, 9_000)


def test_staging_get_budgets_none_for_unbounded_backend():
    backend = MagicMock()
    backend.staging_buffer_bytes.return_value = None
    assert _make_thread(backend)._staging_get_budgets() is None


def test_staging_get_budgets_rejects_non_positive_backend_budget():
    backend = MagicMock()
    backend.staging_buffer_bytes.return_value = 0
    with pytest.raises(ValueError, match="raw staging budget must be positive"):
        _make_thread(backend)._staging_get_budgets()
```

`tests/ut/distributed/ascend_store/test_kv_transfer.py`（现有文件）两处追加：

- 现有 `FakeStore` 增加 `staging_bytes: int | None = None` 构造参数与 `staging_buffer_bytes()` 方法（None = 无约束，既有用例保持整批语义不动）。
- 追加 `test_handle_request_chunks_finite_staging_and_records_failed_blocks`：FakeStore 带有限 staging 预算 + 计数 `get`，按该文件既有 thread/req_meta 构造方式组装，让 `_plan_get_batches` 产出 ≥2 批且其中一批失败；断言 `get` 被调用多次、失败 key 经 `record_failed_blocks` 进入 `_invalid_block_ids` 记账。不引入新测试框架/helper；`staging_bytes=None` 的整批语义由既有 recv-thread 用例继续覆盖。

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py -v 2>&1 | tail -8`
Expected: FAIL `ImportError: cannot import name '_estimate_get_staging_bytes'`

Run: `.venv/bin/python -m pytest tests/ut/distributed/ascend_store/test_kv_transfer.py::TestKVCacheStoreRecvingThread::test_handle_request_chunks_finite_staging_and_records_failed_blocks -v`
Expected: FAIL because the current whole-request `get` path issues one call instead of the required staging-budget-sized sub-batches.

- [ ] **Step 3: 实现**

`kv_transfer.py` 模块级（放在 `record_failed_blocks` 之前）：

```python
_GET_STAGING_ALIGNMENT_BYTES = 4096
_GET_STAGING_PADDING_BYTES = 2 * _GET_STAGING_ALIGNMENT_BYTES
# Match the upstream staging reserve without adding a user-facing tuning knob.
_GET_STAGING_USABLE_FRACTION = 0.9


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_get_staging_bytes(key_sizes: list[int]) -> int:
    return _align_up(sum(key_sizes), _GET_STAGING_ALIGNMENT_BYTES) + _GET_STAGING_PADDING_BYTES


def _plan_get_batches(
    sizes_per_key: list[list[int]],
    *,
    usable_budget_bytes: int,
    raw_budget_bytes: int,
) -> tuple[list[list[int]], set[int]]:
    """Plan key-subset gets under usable and raw staging budgets."""
    if usable_budget_bytes <= 0 or raw_budget_bytes <= 0:
        raise ValueError("staging budgets must be positive")
    if usable_budget_bytes > raw_budget_bytes:
        raise ValueError("usable staging budget must not exceed raw budget")
    batches: list[list[int]] = []
    oversized: set[int] = set()
    current: list[int] = []
    current_bytes = 0
    for index, key_sizes in enumerate(sizes_per_key):
        key_bytes = _estimate_get_staging_bytes(key_sizes)
        if key_bytes > raw_budget_bytes:
            oversized.add(index)
            continue
        if key_bytes > usable_budget_bytes:
            if current:
                batches.append(current)
                current, current_bytes = [], 0
            batches.append([index])
            continue
        if current and current_bytes + key_bytes > usable_budget_bytes:
            batches.append(current)
            current, current_bytes = [], 0
        current.append(index)
        current_bytes += key_bytes
    if current:
        batches.append(current)
    return batches, oversized
```

`KVCacheStoreRecvingThread` 加两个方法：

```python
    def _staging_get_budgets(self) -> tuple[int, int] | None:
        raw_budget_bytes = self.m_store.staging_buffer_bytes()
        if raw_budget_bytes is None:
            return None
        if raw_budget_bytes <= 0:
            raise ValueError("backend raw staging budget must be positive")
        usable_budget_bytes = max(
            1,
            int(raw_budget_bytes * _GET_STAGING_USABLE_FRACTION),
        )
        return raw_budget_bytes, usable_budget_bytes

    def _chunked_store_get(
        self,
        key_list: list[str],
        addr_list: list,
        size_list: list[list[int]],
        *,
        usable_budget_bytes: int,
        raw_budget_bytes: int,
    ) -> list[int]:
        """Run one whole-request get as staging-budget-sized key subsets."""
        batches, oversized = _plan_get_batches(
            size_list,
            usable_budget_bytes=usable_budget_bytes,
            raw_budget_bytes=raw_budget_bytes,
        )
        results: dict[int, int] = {index: 1 for index in oversized}
        if oversized:
            logger.warning(
                "KV pool async recv skipping %d single keys larger than raw staging budget %d",
                len(oversized),
                raw_budget_bytes,
            )
        for batch in batches:
            sub_keys = [key_list[i] for i in batch]
            sub_addrs = [addr_list[i] for i in batch]
            sub_sizes = [size_list[i] for i in batch]
            sub_ret = self.m_store.get(sub_keys, sub_addrs, sub_sizes)
            if sub_ret is None:
                for index in batch:
                    results[index] = 1
            else:
                for index, code in zip(batch, sub_ret, strict=True):
                    results[index] = code
        if len(batches) > 1 or oversized:
            logger.info(
                "KV pool async recv chunked get: keys=%d batches=%d oversized=%d "
                "usable_budget=%d raw_budget=%d",
                len(key_list),
                len(batches),
                len(oversized),
                usable_budget_bytes,
                raw_budget_bytes,
            )
        return [results[i] for i in range(len(key_list))]
```

`_handle_request` 中把整批 `get` 改为：

```python
        budgets = self._staging_get_budgets()
        if budgets is None:
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        else:
            raw_budget_bytes, usable_budget_bytes = budgets
            ret = self._chunked_store_get(
                key_list_c,
                addr_list_c,
                size_list_c,
                usable_budget_bytes=usable_budget_bytes,
                raw_budget_bytes=raw_budget_bytes,
            )
```

后续既有 `record_failed_blocks` 逻辑不变；oversized key 和失败子批都通过同一 per-key 返回码进入现有失败记账。

- [ ] **Step 4: 跑测试确认通过 + 不回归**

Run: `.venv/bin/python -m pytest tests/ut/distributed/ascend_store/ -v`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py tests/ut/distributed/ascend_store/test_kv_transfer.py tests/ut/distributed/ascend_store/test_kv_transfer_chunked_get.py
git commit -s -m "fix(kv_pool): chunk store gets with raw and usable staging budgets"
```

---

### Task 3: Fix A — worker 为未提交的 reverse 合成失败上报

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py`（`_consume_store_completions` :584-618 + 新方法）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_split_failure_reporting.py`（追加；直接复用真实 split worker/metadata helper）

- [ ] **Step 0: 复核 `test_split_lifecycle.py` 的 `_make_worker`、`_make_split_metadata(include_reverse=True)`、`_make_reverse_plan(completion_id)`；不要手工 `object.__new__(_SplitTracker)`，避免漏掉 slots 字段**

- [ ] **Step 1: 写失败测试**

```python
def test_store_failure_before_reverse_submission_synthesizes_failure_report():
    worker = _make_worker()
    metadata = _make_split_metadata(include_reverse=True)
    metadata.reverse_plans[0] = _make_reverse_plan(reverse_send_completion_id=77)
    worker._kvpool_worker_adapter.get_finished.return_value = (set(), {DECODE_REQUEST_ID})
    worker._kvpool_worker_adapter.get_block_ids_with_load_errors.return_value = {20}

    worker.start_load_kv(metadata)
    worker.get_finished(set(), metadata)

    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    assert tracker.reverse_phase.value == "FAILED"
    assert tracker.reverse_submitted_attempt is None
    assert worker.build_connector_worker_meta().failure_reports == {77: 1}


def test_store_failure_after_reverse_submission_keeps_wire_terminal_path():
    # reverse 已提交：失败上报属于数据面 send_done_send_signal，不得合成。
    worker = _make_worker()
    metadata = _make_split_metadata(include_reverse=True)
    metadata.reverse_plans[0] = _make_reverse_plan(reverse_send_completion_id=88)
    worker.start_load_kv(metadata)
    tracker = worker._split_trackers[DECODE_REQUEST_ID]
    tracker.store_load_failed = True
    tracker.reverse_submitted_attempt = REVERSE_ATTEMPT_KEY

    worker._consume_store_completions({DECODE_REQUEST_ID}, set())

    assert tracker.reverse_phase.value == "PENDING"
    assert worker.build_connector_worker_meta() is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/ -k store_failure -v 2>&1 | tail -5`
Expected: FAIL（reverse_phase 仍 PENDING / 无 failure report）

- [ ] **Step 3: 实现**

`worker.py` `_consume_store_completions` 的 FAILED 分支（现 `tracker.store_phase = _SplitPhase.FAILED` 之后）插入调用：

```python
            if tracker.store_load_failed:
                tracker.store_phase = _SplitPhase.FAILED
                self._invalid_block_ids.update(tracker.store_destination_slice)
                self._fail_unsubmitted_reverse(request_id, tracker)
                logger.warning(
                    "dual_path data_terminal key=%s failure_source=STORE status=FAILED",
                    request_id,
                )
                self._report_split_terminal(request_id, tracker, reported_store_terminals)
```

新方法（放在 `_consume_store_completions` 之后）：

```python
    def _fail_unsubmitted_reverse(self, request_id: str, tracker: "_SplitTracker") -> None:
        """Fail-close the reverse-send completion when the store load failed
        before the reverse was ever submitted.

        The scheduler opens the REVERSE_SEND completion at activation and the
        reverse is submitted only after a DONE store read, so a FAILED store
        leaves the completion open forever: request_finished then delay-frees
        the request's blocks and the engine starves (livelock). Every worker
        runs this path, so the synthesized reports fill the all-worker barrier
        exactly like data-plane reports."""
        if tracker.reverse_phase is not _SplitPhase.PENDING:
            return
        if tracker.reverse_plan is None or tracker.reverse_plan.reverse_send_completion_id is None:
            return
        if tracker.reverse_submitted_attempt is not None:
            return
        tracker.reverse_phase = _SplitPhase.FAILED
        self._record_completion_report(tracker.reverse_plan.reverse_send_completion_id, succeeded=False)
```

- [ ] **Step 4: 跑测试确认通过 + dual_path 全套不回归**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/ 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py tests/ut/distributed/kv_transfer/dual_path/test_split_failure_reporting.py
git commit -s -m "fix(dual_path): fail-close reverse-send completion when store load fails before submission"
```

---

### Task 4: Fix B-1 — completion tracker 强制失败关闭

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py`（`fail_completion` :175 后）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py`

- [ ] **Step 1: 写失败测试**

```python
def test_force_fail_completion_closes_exact_unstarted_record():
    completion_tracker = _completion_tracker()
    tracker = completion_tracker.TransferCompletionTracker()
    attempt_key = ReverseAttemptKey(DualPathRequestKey("decode", "request", 0), 2)
    completion = tracker.open_completion(
        completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=3,
        reverse_attempt_key=attempt_key,
    )
    assert tracker.force_fail_completion(
        completion.completion_id,
        expected_kind=completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_attempt_key=attempt_key,
    ) is True
    assert completion.closed and completion.failed


def test_force_fail_completion_latches_but_does_not_close_dispatched_worker_barrier():
    completion_tracker = _completion_tracker()
    tracker = completion_tracker.TransferCompletionTracker()
    attempt_key = ReverseAttemptKey(DualPathRequestKey("decode", "request", 0), 2)
    completion = tracker.open_completion(
        completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=2,
        reverse_attempt_key=attempt_key,
    )
    assert tracker.mark_dispatched(
        completion.completion_id,
        expected_kind=completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_attempt_key=attempt_key,
    ) is True
    assert tracker.force_fail_completion(
        completion.completion_id,
        expected_kind=completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_attempt_key=attempt_key,
    ) is False
    assert completion.failed and not completion.closed


def test_force_fail_completion_rejects_identity_mismatch():
    completion_tracker = _completion_tracker()
    tracker = completion_tracker.TransferCompletionTracker()
    attempt_key = ReverseAttemptKey(DualPathRequestKey("decode", "request", 0), 2)
    completion = tracker.open_completion(
        completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=1,
        reverse_attempt_key=attempt_key,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        tracker.force_fail_completion(
            completion.completion_id,
            expected_kind=completion_tracker.CompletionKind.REVERSE_SEND,
            expected_attempt_key=attempt_key,
        )


def test_has_open_completion_for_exact_request_key():
    completion_tracker = _completion_tracker()
    tracker = completion_tracker.TransferCompletionTracker()
    request_key = DualPathRequestKey("decode", "request", 0)
    attempt_0 = ReverseAttemptKey(request_key, 0)
    attempt_1 = ReverseAttemptKey(request_key, 1)
    records = [
        tracker.open_completion(
            completion_tracker.CompletionKind.REVERSE_RECEIVE,
            expected_worker_count=1,
            reverse_attempt_key=attempt,
        )
        for attempt in (attempt_0, attempt_1)
    ]
    tracker.open_completion(
        completion_tracker.CompletionKind.REVERSE_RECEIVE,
        expected_worker_count=1,
        reverse_attempt_key=ReverseAttemptKey(
            DualPathRequestKey("decode", "foreign", 1),
            0,
        ),
    )
    assert tracker.has_open_completion(completion_tracker.CompletionKind.REVERSE_RECEIVE, request_key)
    for record, attempt in zip(records, (attempt_0, attempt_1), strict=True):
        tracker.force_fail_completion(
            record.completion_id,
            expected_kind=completion_tracker.CompletionKind.REVERSE_RECEIVE,
            expected_attempt_key=attempt,
        )
    assert not tracker.has_open_completion(completion_tracker.CompletionKind.REVERSE_RECEIVE, request_key)
    assert tracker.open_count() == 1
```

- [ ] **Step 2: 跑测试确认失败**（`AttributeError: force_fail_completion`）

- [ ] **Step 3: 实现**

```python
    def force_fail_completion(
        self,
        completion_id: int,
        *,
        expected_kind: CompletionKind,
        expected_attempt_key: ReverseAttemptKey,
    ) -> bool:
        """Fail an exact completion, closing only before worker dispatch.

        Used when the peer reports the failure out-of-band (control channel):
        before dispatch, no worker terminal can fill the barrier. Once the
        exact binding has been dispatched, preserve the all-worker barrier
        even when no worker report has arrived; workers own closure."""
        record = self._records.get(completion_id)
        if record is None or record.closed:
            return False
        if record.completion_kind is not expected_kind or record.reverse_attempt_key != expected_attempt_key:
            raise RuntimeError("completion identity mismatch while force-failing attempt")
        record.failed = True
        if record.dispatched or record.completed_worker_count != 0:
            return False
        record.closed = True
        return True

    def has_open_completion(
        self,
        completion_kind: CompletionKind,
        request_key: DualPathRequestKey,
    ) -> bool:
        return any(
            not record.closed
            and record.completion_kind is completion_kind
            and record.reverse_attempt_key is not None
            and record.reverse_attempt_key.request_key == request_key
            for record in self._records.values()
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py -v 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py
git commit -s -m "feat(dual_path): add force-fail close for completions with no in-flight workers"
```

---

### Task 5: Fix B-2 — PathDecision 携带 prefill 控制端点

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py`（`PathDecision` :115-140）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py` 或 `test_path_decision_channel.py`（按现有 round-trip 测试风格追加）

- [ ] **Step 1: 写失败测试**

```python
def test_path_decision_round_trip_with_prefill_endpoint():
    decision = PathDecision(
        result=PathDecisionResult(request_key=_request().request_key, path=PathKind.PE_READ),
        reverse_plan=None,
        prefill_control_endpoint=DecodeControlEndpoint(host="192.0.2.88", port=24999),
    )
    decoded = PathDecision.from_dict(decision.to_dict())
    assert decoded.prefill_control_endpoint == DecodeControlEndpoint(host="192.0.2.88", port=24999)


def test_path_decision_round_trip_without_prefill_endpoint_defaults_none():
    decision = PathDecision(
        result=PathDecisionResult(request_key=_request().request_key, path=PathKind.PE_READ),
        reverse_plan=None,
    )
    decoded = PathDecision.from_dict(decision.to_dict())
    assert decoded.prefill_control_endpoint is None


def test_path_decision_wire_rejects_unknown_keys():
    # require_exact_payload 更新后仍拒绝未登记 key
    payload = PathDecision(
        result=PathDecisionResult(request_key=_request().request_key, path=PathKind.PE_READ),
        reverse_plan=None,
    ).to_dict()
    payload["surprise"] = 1
    with pytest.raises(PathDecisionValidationError):
        PathDecision.from_dict(payload)
```

- [ ] **Step 2: 跑测试确认失败**（`TypeError: unexpected keyword argument` / 键校验失败）

- [ ] **Step 3: 实现**

```python
@dataclass(frozen=True)
class PathDecision:
    result: PathDecisionResult
    reverse_plan: ReversePlan | None
    # Optional reverse-direction control endpoint; the Decode side sends
    # terminal failure notices (ABORT) here when an activated request dies
    # before its reverse transfer could start.
    prefill_control_endpoint: DecodeControlEndpoint | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, PathDecisionResult):
            raise PathDecisionValidationError("result must be a PathDecisionResult")
        if self.reverse_plan is not None and not isinstance(self.reverse_plan, ReversePlan):
            raise PathDecisionValidationError("reverse_plan must be a ReversePlan or None")
        if self.prefill_control_endpoint is not None and not isinstance(
            self.prefill_control_endpoint, DecodeControlEndpoint
        ):
            raise PathDecisionValidationError("prefill_control_endpoint must be a DecodeControlEndpoint or None")

    def to_dict(self) -> JsonObject:
        return {
            "result": self.result.to_dict(),
            "reverse_plan": None if self.reverse_plan is None else self.reverse_plan.to_dict(),
            "prefill_control_endpoint": (
                None if self.prefill_control_endpoint is None else self.prefill_control_endpoint.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: JsonValue) -> PathDecision:
        data = require_exact_payload(payload, frozenset({"result", "reverse_plan", "prefill_control_endpoint"}))
        reverse_plan_payload = data["reverse_plan"]
        endpoint_payload = data["prefill_control_endpoint"]
        return cls(
            result=PathDecisionResult.from_dict(data["result"]),
            reverse_plan=None if reverse_plan_payload is None else ReversePlan.from_dict(reverse_plan_payload),
            prefill_control_endpoint=(
                None if endpoint_payload is None else DecodeControlEndpoint.from_dict(endpoint_payload)
            ),
        )
```

注意：P/D 两侧同构建部署，严格键校验同步更新无兼容问题；但 `decode_control_endpoint` 所在 `DualPathDecisionMetadata` 不动。同步更新该文件现有 msgpack expected bytes、`set(decision.to_dict())` 等精确 wire 断言，新增字段即使为 None 也必须出现在 wire payload 中。

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py -v 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "feat(dual_path): carry optional prefill control endpoint on path decisions"
```

---

### Task 6: Fix B-3 — coordinator：prefill 接收端 + decode 发送 executor

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py`（coordinator 构造、role-aware frame handler、close）
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py`（Prefill-only `prefill_control_port`）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py`
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py`

设计：不能直接复用 Decode receiver 的 registry 语义。现有 `_handle_decision_frame`/`_handle_abort_frame` 都读取 `decode_engine_instance_id` 和 `_pending_keys`，Prefill coordinator 没有这些状态。receiver 必须先按 role 分流：Decode 继续维护 decision admission registry；Prefill 只允许 ABORT，并用独立的 `_prefill_abort_keys` 做 exact-key admission。该 registration 覆盖 scheduler 所管理 admission 的完整生命周期，收到 notice 不自动注销；同 key 的 proofless→terminalized、attempt 0→attempt 1 必须分别入队，只有完整 `PathAbortNotice` 相同的 sender retry 被去重。scheduler-driven unregister 同时清该 key 的 notice dedupe，使未来重新 register 可再次接收相同 notice；`close()` 清空 registry、dedupe 和 queue。未知/已注销的 ABORT 仍 ACK（幂等丢弃，避免发送端无谓重试），但不得入队；投到 Prefill receiver 的 DECISION 明确回复 `PROTOCOL_ERROR`。`for_decode` 增加发送 executor，`for_prefill(control_endpoint=...)` 增加接收线程；两侧 `close()` 统一关闭自己实际创建的 receiver/executor 并清空各自 registry/queue。

- [ ] **Step 0: 读 `_handle_decision_frame`/`_handle_abort_frame` 与 config 测试，确认本 Task 不让 Prefill 访问 `decode_engine_instance_id`，也不复用 `_pending_keys`**

- [ ] **Step 1: 写失败测试（真实 127.0.0.1 ZMQ 回路，参照该文件既有 socket 测试）**

```python
def test_prefill_coordinator_receives_abort_on_control_endpoint():
    prefill_ep = _free_control_endpoint()
    prefill = PathDecisionCoordinator.for_prefill(control_endpoint=prefill_ep)
    sender = PathDecisionCoordinator.for_prefill()
    key = _request().request_key
    try:
        prefill.register_prefill_abort_key(key)
        notice = PathAbortNotice(request_key=key, reason=PathAbortReason.ACTIVATION_FAILED)
        sender.submit_abort(prefill_ep, notice).result(timeout=5)
        assert prefill.take_received_aborts() == [notice]
    finally:
        sender.close()
        prefill.close()


def test_prefill_coordinator_acks_but_drops_unknown_or_unregistered_abort():
    prefill_ep = _free_control_endpoint()
    prefill = PathDecisionCoordinator.for_prefill(control_endpoint=prefill_ep)
    sender = PathDecisionCoordinator.for_prefill()
    try:
        key = DualPathRequestKey("decode-engine-1:0:boot-1", "stale", 9)
        sender.submit_abort(
            prefill_ep,
            PathAbortNotice(request_key=key, reason=PathAbortReason.ACTIVATION_FAILED),
        ).result(timeout=5)
        assert prefill.take_received_aborts() == []
    finally:
        sender.close()
        prefill.close()
```

在 `TestDualPathConfig` 中追加：

```python
    def test_prefill_control_port_is_role_strict(self):
        self.assertEqual(
            DualPathConfig.from_extra_config(
                {"role": "prefill", "prefill_control_port": 7200},
                make_kv_transfer_config("kv_producer"),
            ),
            DualPathConfig(role="prefill", prefill_control_port=7200),
        )
        with self.assertRaisesRegex(ValueError, "only consumed by role='prefill'"):
            DualPathConfig.from_extra_config(
                {"role": "decode", "dual_path_control_port": 7100, "prefill_control_port": 7200},
                make_kv_transfer_config("kv_consumer"),
            )
        with self.assertRaisesRegex(ValueError, "only consumed by role='decode'"):
            DualPathConfig.from_extra_config(
                {"role": "prefill", "dual_path_control_port": 7100},
                make_kv_transfer_config("kv_producer"),
            )
```

同一测试文件再按已有低层 socket helper 补两条，不新增占位 helper：

- `test_prefill_receiver_rejects_decision_frames_as_protocol_error`：发合法 DECISION frame，断言 reply 为 `PROTOCOL_ERROR`，两条接收 queue 都为空。
- `test_decode_coordinator_can_deliver_abort_to_registered_prefill_receiver`：真实启动 Decode coordinator 与另一个已注册 exact key 的 Prefill receiver；用 Decode 自己的 executor 投递并等待 Future 成功，不能用 `cancel()` 掩盖三次投递后抛出的 `PathDecisionDeliveryError`。
- config suite 同步覆盖 `prefill_control_port` 的 bool/0/65536/非 int 拒绝、`ALLOWED_EXTRA_CONFIG_KEYS` 精确集合，以及两个 dataclass 字段的 `vars(config)` 断言。

- [ ] **Step 2: 跑测试确认失败**（`for_prefill(control_endpoint=...)`/Prefill registry 不存在，Decode `submit_abort` 无 executor）

- [ ] **Step 3: 实现**

`__init__` 新增角色无关 bind endpoint 与 Prefill 专属 registry；不要把 Prefill endpoint 填进 `_decode_control_endpoint`：

```python
        self._bind_endpoint: DecodeControlEndpoint | None = None
        self._prefill_abort_keys: set[DualPathRequestKey] = set()
```

`for_decode` 将 `control_endpoint` 同时赋给 `_decode_control_endpoint`（对外 metadata）和 `_bind_endpoint`（receiver 绑定），并创建发送 executor。`for_prefill` 保留现有发送 executor；若传 `control_endpoint`，只赋给 `_bind_endpoint` 并启动 receiver。将接收线程启动抽成 `_start_receiver(name)`，复用 ready/error/close 逻辑，避免两份生命周期代码：

```python
        coordinator._bind_endpoint = control_endpoint
        coordinator._executor = ThreadPoolExecutor(
            max_workers=_PATH_DECISION_SEND_WORKERS,
            thread_name_prefix="path-abort-sender",
        )

    def register_prefill_abort_key(self, key: DualPathRequestKey) -> None:
        if self._role != "prefill":
            raise RuntimeError("only a Prefill coordinator can register abort keys")
        with self._lifecycle_lock, self._registry_lock:
            if self._closed:
                raise RuntimeError("path decision coordinator is closed")
            self._prefill_abort_keys.add(key)

    def unregister_prefill_abort_key(self, key: DualPathRequestKey) -> None:
        if self._role != "prefill":
            raise RuntimeError("only a Prefill coordinator can unregister abort keys")
        with self._registry_lock:
            self._prefill_abort_keys.discard(key)
```

`_receive_decisions` 改为 assert/bind `_bind_endpoint`。frame handler 必须先做 role 分离：

- `_handle_decision_frame`: `self._role != "decode"` 时不解析/触碰 Decode registry，直接 reply `PROTOCOL_ERROR`；Decode 分支保持现有 incarnation、shape、pending/attempt 校验。
- `_handle_abort_frame`: 先解析 notice；Decode 分支保持现有 `_pending_keys`/`_accepted_decisions` 语义；Prefill 分支只检查 `key in _prefill_abort_keys`，命中才入 `_received_aborts`，未命中 ACK 后丢弃。Prefill 分支不得读取 `decode_engine_instance_id`。

`close()` 按“资源是否存在”而不是硬编码 role 关闭资源；随后按 role 清 registry，并 drain 两条 queue：

```python
        if self._receiver_thread is not None:
            assert self._context is not None
            self._context.term()
            self._receiver_thread.join()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        with self._registry_lock:
            self._pending_keys.clear()
            self._accepted_decisions.clear()
            self._closed_through_attempt_ids.clear()
            self._prefill_abort_keys.clear()
        self._drain_received_decisions()
        self._drain_received_aborts()
```

`config.py`：

- `ALLOWED_EXTRA_CONFIG_KEYS` 增加 `prefill_control_port`；`DualPathConfig` 增加 `prefill_control_port: int | None = None`。
- `dual_path_control_port` 仍为 Decode 必填、Prefill 禁止；`prefill_control_port` 为 Prefill 可选、Decode 禁止；两者都严格拒绝 bool/非 int/不在 1..65535。
- 更新所有 `vars(config)` 精确断言和 role/config 单测；不自动从 `dual_path_control_port` 推导 Prefill 端口，避免同一字段跨 role 产生双重含义。

- [ ] **Step 4: 跑测试确认通过 + 通道全套不回归**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py -v 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py tests/ut/distributed/kv_transfer/dual_path/test_dual_path_connector.py
git commit -s -m "feat(dual_path): support reverse-direction abort delivery on the control channel"
```

---

### Task 7: Fix B-4 — decode scheduler：存端点 + 失败时回发 ABORT

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`（`DecodePathDecisionState`、`_activate_received_decision`、worker failure 聚合、统一 helper）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_decode_scheduler.py` 或 `test_completion_and_delivery_recovery.py`（复用现有 Decode scheduler 工厂与 fake coordinator）

- [ ] **Step 0: 读 conftest 的 Decode scheduler 工厂与现有 activation/attempt-refresh 测试；扩展现有 fake coordinator 捕获 `submit_abort(endpoint, notice)`，不要假设存在新 fixture，也不要在 scheduler UT 发真实网络请求**

- [ ] **Step 1: 写失败测试**

按现有 helper 构造真实 state/snapshot/completion，新增以下 oracle（不写不存在的 fixture/helper 占位符）：

1. `test_decode_activation_validation_failure_sends_abort_without_persisting_untrusted_endpoint`
   - 注册 exact request key，decision 携带 `prefill_control_endpoint`，故意让 reverse plan 校验失败；
   - state 为 `ACTIVATION_FAILED`、Decode registry 注销、metadata 含本地 control failure；
   - fake coordinator 捕获一条 `PathAbortReason.ACTIVATION_FAILED`，并携带该失败 DE_READ decision 自身的 exact attempt `TERMINALIZED` proof；state 不持久化这次失败 decision 的 endpoint。
2. `test_decode_worker_failure_uses_persisted_prefill_endpoint`
   - 成功 activation 后断言 endpoint 才写入 state；用 `expected_worker_count=1` 的 failure report 关闭 REVERSE_SEND；
   - `_decode_control_failures` 有 `REVERSE_JOB_FAILED`；只有 all-worker barrier 关闭后，才向持久化 endpoint 发携带 `completion.reverse_attempt_key` 的 terminalized ABORT。
3. `test_decode_failure_still_sends_abort_when_control_failure_build_raises`
   - monkeypatch `_build_decode_control_failure` 抛 `RuntimeError`；
   - helper 不传播异常，但 `finally` 仍提交 ABORT，state/registry 仍 fail-close。
4. `test_decode_failure_without_endpoint_is_local_only`
   - endpoint=None 时本地状态与 control failure 正常，`submit_abort` 未调用。
   - PE_READ 或调用点没有 exact safe Reverse attempt 时，即使发送 ABORT 也不得携带 exact-attempt terminal proof；admission 终态建立后仍携带 frozen admission ceiling，无 Reverse plan 曾发布时通常为 explicit `null`。
5. `test_failed_attempt_refresh_does_not_force_close_older_inflight_attempt`
   - 先成功 COMMIT attempt N 并保持其 REVERSE_SEND completion open；再送更高 attempt N+1，令校验失败；
   - state fail-close，并用失败 decision 直接传入的 attempt N+1 发 terminalized ABORT；N 的 completion 仍 open、`_delay_free_for_connector` 仍为 True；
   - N 的完整 worker barrier 到达后才由 `_run_completion_close_action` 关闭，随后 finished_sending 注入并解除 delay-free。
6. `test_attempt_refresh_endpoint_mismatch_fails_closed_to_original_endpoint`
   - 已提交 state 的 endpoint=A，refresh 携带 B；拒绝 refresh，向 A（不是 B）发携带 attempt N+1 proof 的 ABORT，且不改 state endpoint。
7. `test_decode_late_worker_failure_after_request_finish_still_notifies_prefill`
   - 成功 activation（endpoint 已持久化、REVERSE_SEND open）后先 `request_finished` 释放 state/snapshot，再聚合该 completion 的 failure report；
   - 延迟上下文命中：每个失败关闭的 REVERSE_SEND completion 都向持久化 endpoint 发自身 exact attempt proof；endpoint 保留到该 request key 的最后一个 open send completion 关闭，不构建 control failure（snapshot 已删，本地块走正常释放）；未命中延迟上下文时保持现有 error 日志、不抛错。

- [ ] **Step 2: 跑测试确认失败**（无 abort 提交记录）

- [ ] **Step 3: 实现**

`DecodePathDecisionState` 加字段（slots dataclass，带默认值合法）：

```python
@dataclass(slots=True)
class DecodePathDecisionState:
    decision_request: PathDecisionRequest
    request: Request
    status: _DecodeDecisionStatus
    prefill_control_endpoint: DecodeControlEndpoint | None = None
```

`_register_pending_decode_decision` 创建 state 时**不要**传 endpoint：此时本地只有 admission request，wire `PathDecision` 尚未到达。endpoint 只在 `_activate_received_decision` 对 exact request key 完成校验并成功 activation 后写入 state。

在 `_activate_received_decision` 完成 exact-key/status 检查并取到 `snapshot = self._decode_kv_snapshots[request_id]` 后：

```python
        candidate_endpoint = decision.prefill_control_endpoint
        if (
            state.prefill_control_endpoint is not None
            and candidate_endpoint != state.prefill_control_endpoint
        ):
            # refresh 不得换 peer 身份；fail-close 时只信任已提交的 A。
            failure = self._fail_decode_admission(
                state,
                snapshot,
                local_reason=DualPathControlFailureReason.ACTIVATION_FAILED,
                peer_endpoint=state.prefill_control_endpoint,
            )
            if failure is not None:
                metadata.control_failures.append(failure)
            return
```

原 activation `try` 成功后、创建 completion/提交 worker metadata 之前再执行：

```python
        state.prefill_control_endpoint = candidate_endpoint
```

原 activation `except Exception` 与 `_aggregate_worker_completion_reports` 的 Decode failure 分支都改调同一个 helper；不得复制“置状态/注销/构建 failure/发 ABORT”四段逻辑：

```python
    def _fail_decode_admission(
        self,
        state: DecodePathDecisionState,
        snapshot: DecodeKVSnapshot,
        *,
        local_reason: DualPathControlFailureReason,
        peer_endpoint: DecodeControlEndpoint | None = None,
        reverse_attempt_id: int | None = None,
    ) -> DualPathControlFailureMetadata | None:
        """Make one exact Decode admission terminal and notify its Prefill peer.

        This helper intentionally does not close existing REVERSE_SEND records:
        an older in-flight attempt still owns the all-worker barrier and block
        release. It only closes admission/control state.
        """
        state.status = _DecodeDecisionStatus.ACTIVATION_FAILED
        self._path_decision_coordinator.unregister(state.request_key)
        endpoint = peer_endpoint or state.prefill_control_endpoint
        try:
            return self._build_decode_control_failure(
                state.request_key.decode_request_id,
                snapshot,
                local_reason,
            )
        except RuntimeError as error:
            logger.error(
                "DualPath could not build Decode control failure for request %s: %s",
                state.request_key.decode_request_id,
                error,
            )
            return None
        finally:
            if endpoint is not None:
                self._send_abort_notice(
                    state.request_key,
                    endpoint,
                    PathAbortReason.ACTIVATION_FAILED,
                    reverse_attempt_key=(
                        None
                        if reverse_attempt_id is None
                        else ReverseAttemptKey(state.request_key, reverse_attempt_id)
                    ),
                )
```

activation 异常调用时显式传 `peer_endpoint=state.prefill_control_endpoint or candidate_endpoint`，并仅对 DE_READ 从该 decision 直接传 `reverse_attempt_id=result.reverse_attempt_id`；不得从 request-id 当前状态恢复 attempt。失败的 candidate 不写入 state；worker failure 用已持久化 endpoint 和关闭 completion 自身的 `reverse_attempt_key`。helper 返回 failure 后，activation 路径 append 到当前 `metadata.control_failures`，worker 路径写入 `_decode_control_failures[request_id]`。即使 `_build_decode_control_failure` 失败，`finally` 仍发 peer ABORT。PE_READ 或没有 exact safe attempt 的失败不携带 exact-attempt proof，但在 admission 终态建立后仍携带 frozen admission ceiling。

attempt refresh 失败时同样调用 helper，但**不**调用 `force_fail_completion`、`discard_unstarted` 或删除 `_reverse_send_completion_ids` 中旧 attempt；旧 attempt 报告到齐后，现有 `_run_completion_close_action`/`_has_open_reverse_send_completion` 负责最终释放。这一条是状态机约束，不是仅测试备注。

**延迟失败上下文（请求先 finish、worker failure 后到，预检确认的真实缺口）**：`_release_scheduler_request_state`（`scheduler.py:1810-1813`）无条件 pop `_decode_decision_states`/`_decode_kv_snapshots`，且刻意让 open completion 跨 request cleanup 存活（":1888 注释"）。若请求先生成完成/被杀而 store load 仍在途，Fix A 合成的 failure report 后到时 Decode failure 分支只剩 `state is None → logger.error`，无法取 endpoint 发 ABORT，Prefill 仍 wedge。补一组按 exact request key 保存的延迟上下文：

```python
        # __init__：
        self._decode_late_abort_endpoints: dict[DualPathRequestKey, DecodeControlEndpoint] = {}
```

- 写入：`_release_scheduler_request_state` 的 decode 段，pop state 后：

```python
            if (
                state.prefill_control_endpoint is not None
                and self._has_open_reverse_send_completion(state.request_key)
            ):
                self._decode_late_abort_endpoints[state.request_key] = state.prefill_control_endpoint
```

- 消费：`_aggregate_worker_completion_reports` Decode failure 分支只在 `tally_reports()` 关闭失败的 REVERSE_SEND completion 后运行；在 `state is None or snapshot is None` 时，用 `completion.reverse_attempt_key.request_key` 查 `_decode_late_abort_endpoints`，命中则发送携带该 exact completion attempt 的 terminalized ABORT。不要在这里按 request key 立即 pop，否则同 key 的其他失败 attempt 无法传播各自证明。不构建 control failure——snapshot 已删，请求块走正常释放。
- 退休：`_run_completion_close_action` 的 REVERSE_SEND 分支在 `_has_open_reverse_send_completion(request_key)` 为 False（最后一个 attempt 关闭）时 pop 该 key 的延迟上下文；`shutdown()` 清空。

- [ ] **Step 4: 跑测试确认通过 + dual_path 全套不回归**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/ 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "fix(dual_path): notify prefill over the control channel when an activated decode request fails"
```

---

### Task 8: Fix B-5 — prefill scheduler：收 abort + 关闭 REVERSE_RECEIVE 释放块

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`（Prefill coordinator 装配、decision 端点、exact-key abort registry、收 abort、finished_recving 注入）
- Modify: `tests/ut/distributed/kv_transfer/dual_path/conftest.py`（`pe_scheduler_factory` 透传 `prefill_control_port`）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_de_read_recovery.py` 或 `test_abort_scheduler.py`（用 `pe_scheduler_factory`）

设计：`_prefill_request_keys` 是 active-admission map，request finish 时即使处于 delay-free 也会被移除，不能作为反向 ABORT 的唯一寻址表。新增 `_prefill_abort_request_ids: dict[DualPathRequestKey, str]`，其寿命覆盖“活跃请求 + request 已结束但 exact REVERSE_RECEIVE completion 仍 open”的窗口；只有请求已释放且该 exact key 不再有 open completion 时，才同时从 scheduler map 和 coordinator `_prefill_abort_keys` 注销。

- [ ] **Step 1: 写失败测试**

先扩展 conftest 的 `pe_scheduler_factory`（`:317`）：`make(...)` 增加 `prefill_control_port: int | None = None` 透传给 `DualPathConfig(role="prefill", prefill_control_port=...)`；现有 `PathDecisionCoordinator`/`get_ip` patch 已能覆盖装配路径，coordinator MagicMock 的 `register_prefill_abort_key`/`unregister_prefill_abort_key` 可直接断言，`take_received_aborts` 记得设 list 返回值（`method_calls` 顺序可证 register 先于 submit）。再按现有 DE_READ activation helper 增加以下 oracle：

1. `test_prefill_admission_registers_abort_key_before_decision_submit`：配置 `prefill_control_port` 后，`coordinator.register_prefill_abort_key(key)` 发生在 `submit(endpoint, decision)` 之前；decision 携带 scheduler 实际绑定的 endpoint。
2. `test_prefill_received_terminalized_abort_fail_closes_unstarted_reverse_receive`：ABORT 携带 exact terminalized attempt，且 completion 明确尚未 dispatch；处理后 record 关闭并走 `_run_completion_close_action`，request 从 `_pending_finished_recving` 移出、加入 `_scheduler_side_finished_recving`，key 标记 invalid。
3. `test_prefill_received_abort_without_terminal_proof_invalidates_without_closing_completion`：普通 ABORT 记录 exact admission invalid；仅在 current exact admission 的 current waiting attempt 属于该 admission 时生成 invalid-block control failure。completion 保持 open 且 `failed=False`，不生成 worker terminal；即使此时 worker report 数为 0，也不能把它解释为尚未 dispatch。
4. `test_prefill_peer_abort_does_not_bypass_started_worker_barrier` 与 real-chain 测试：binding dispatch 后收到 exact proof，只 latch failed 并向 worker 下发 `ReverseReceiveFailureTerminal`；TP=2 使用两个真实 Prefill workers 消费同一个 scheduler-produced terminal，各自经 worker metadata round-trip 回报，第一个 report 后 completion 仍 open，第二个后才由 all-worker 聚合关闭。
5. `test_duplicate_terminalized_abort_does_not_redispatch_before_worker_report`：第一次 metadata drain 后到 worker report 返回前，重复 ABORT 不得再次 stage/deliver terminal；exact completion 关闭时退休 dedupe 状态。
6. `test_terminalized_abort_closes_stale_attempt_without_aliasing_current_attempt`：exact proof 按 `(request_key, attempt)` 全局查找 named open completion，只授权 stale attempt；不得从 `_waiting_reverse_attempt_ids[request_id]` 恢复 current attempt。若同一 ABORT 另带 admission ceiling，则 ceiling 可独立授权 provably-never-started 的 current/newer attempt；即使不带 ceiling，current exact admission 仍生成 invalid-block control failure，但 completion 保持 open 且不得注入 `finished_recving`。
7. `test_prefill_abort_after_request_finished_still_finds_delayed_completion`：先让 request finish 并清掉 `_prefill_request_keys`，但 exact completion 仍 open；断言 `_prefill_abort_request_ids`/coordinator registry 仍保留，随后 exact terminalized ABORT 能关闭并释放。
8. `test_prefill_abort_registry_retires_only_after_request_release_and_last_completion_close`：请求仍 active 时一次 attempt 关闭不注销（允许后续 refresh）；请求已 release 且最后一个 exact completion 关闭后才注销 scheduler/coordinator 两处。
9. admission/attempt isolation：unknown、old admission 与 same-ID replacement、attempt 0 与 attempt 1 均不得 fallback 到相同 `request_id` 的 active/current state。
10. `test_scheduler_side_finished_recving_injects_without_worker_metadata`：即使本轮 `kv_connector_worker_meta is None`，`update_connector_output` 仍注入并清空 scheduler-side set。
11. `test_prefill_stale_attempt_completion_close_retires_abort_registry`：refresh 场景关闭 stale（非当前 waiting）的 REVERSE_RECEIVE completion；exact key 仍有 open completion 或 active admission 时不 retire，最后一个关闭且 admission 已释放后才 retire。

- [ ] **Step 2: 跑测试确认失败**（completion 仍 open / `_pending_finished_recving` 未释放）

- [ ] **Step 3: 实现**

scheduler `__init__` 加：

```python
        self._prefill_control_endpoint: DecodeControlEndpoint | None = None
        self._prefill_abort_request_ids: dict[DualPathRequestKey, str] = {}
        self._scheduler_side_finished_recving: set[str] = set()
        self._prefill_pending_reverse_receive_failure_terminals: dict[
            ReverseAttemptKey, ReverseReceiveFailureTerminal
        ] = {}
        self._prefill_staged_or_delivered_reverse_terminals: set[ReverseAttemptKey] = set()
```

Prefill 装配 coordinator 时，`prefill_control_port is not None` 才创建 endpoint；本项目已要求 DualPath DP==1，因此直接使用配置端口，不套 Decode 的 DP 派生函数：

```python
            if dual_path_cfg.prefill_control_port is not None:
                self._prefill_control_endpoint = DecodeControlEndpoint(
                    host=get_ip(),
                    port=dual_path_cfg.prefill_control_port,
                )
            self._path_decision_coordinator = PathDecisionCoordinator.for_prefill(
                control_endpoint=self._prefill_control_endpoint,
            )
```

在 `_decide_prefill_path_for_admission` 把 exact key 安装进 `_prefill_request_keys` 后，若 endpoint 启用，则同步写 `_prefill_abort_request_ids[key] = request_id` 并调用 `register_prefill_abort_key(key)`。这一步必须早于任何 `submit`。`_deliver_prefill_decision` 构造 wire decision 时加：

```python
        decision = PathDecision(
            result=decision_result,
            reverse_plan=reverse_plan,
            prefill_control_endpoint=self._prefill_control_endpoint,
        )
```

`_discard_undelivered_prefill_decision` 与 `submit()` 在创建 Future 前同步抛错的 rollback 路径可立即注销 exact key（Decode 不可能已接受）。已创建 delivery Future 后即使三次投递以 `PathDecisionDeliveryError` 结束也不能立即假定对端未处理：ACK 可能丢失，registry 随正常 request/completion 生命周期退休。

`build_connector_meta` prefill 分支，在 `self._reconcile_prefill_deliveries()` 之前插入：

```python
            for notice in self._path_decision_coordinator.take_received_aborts():
                self._handle_received_peer_abort(notice)
```

新方法（放在 `_handle_received_abort` 旁，互为镜像）：

```python
    def _handle_received_peer_abort(self, notice: PathAbortNotice) -> None:
        """Fail-close one exact Prefill Reverse receive after a peer ABORT."""
        request_id = self._prefill_abort_request_ids.get(notice.request_key)
        if request_id is None:
            return
        logger.warning(
            "dual_path peer_abort key=%s reason=%s: fail-closing reverse receive",
            notice.request_key.decode_request_id,
            notice.reason.value,
        )
        self._prefill_invalid_request_keys.add(notice.request_key)
        active_exact_admission = self._prefill_request_keys.get(request_id) == notice.request_key
        current_waiting_attempt = self._waiting_reverse_attempt_ids.get(request_id)
        reverse_terminal = notice.reverse_terminal
        terminal_attempt = None
        current_attempt_belongs_to_admission = (
            current_waiting_attempt is not None
            and current_waiting_attempt.request_key == notice.request_key
        )
        if reverse_terminal is not None:
            terminal_attempt = ReverseAttemptKey(
                notice.request_key,
                reverse_terminal.reverse_attempt_id,
            )
            completion = self._completion_tracker.find_open_completion(
                CompletionKind.REVERSE_RECEIVE, terminal_attempt
            )
            if completion is not None and self._completion_tracker.force_fail_completion(
                completion.completion_id,
                expected_kind=CompletionKind.REVERSE_RECEIVE,
                expected_attempt_key=terminal_attempt,
            ):
                _, recving = self._run_completion_close_action(completion)
                self._scheduler_side_finished_recving.update(recving)
            elif (
                completion is not None
                and not completion.closed
                and terminal_attempt not in self._prefill_staged_or_delivered_reverse_terminals
            ):
                self._prefill_pending_reverse_receive_failure_terminals[terminal_attempt] = (
                    ReverseReceiveFailureTerminal(
                        request_key=terminal_attempt.request_key,
                        reverse_attempt_id=terminal_attempt.reverse_attempt_id,
                        reverse_receive_completion_id=completion.completion_id,
                        wire_request_id=reverse_wire_id(terminal_attempt),
                    )
                )
                self._prefill_staged_or_delivered_reverse_terminals.add(terminal_attempt)
        invalid_block_ids = (
            self._recovery_invalid_block_ids(request_id)
            if active_exact_admission and current_attempt_belongs_to_admission
            else ()
        )
        if invalid_block_ids:
            self._prefill_control_failures[request_id] = DualPathControlFailureMetadata(
                request_id=request_id,
                invalid_block_ids=invalid_block_ids,
                reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
            )
```

普通 ABORT（`reverse_terminal is None` 且没有 admission ceiling）只记录 exact admission invalid；它不调用 `force_fail_completion`、不修改 completion state、不 close、不 stage worker terminal，也不注入 `finished_recving`。只要该 key 仍是 current exact admission 且 handler 入口快照到的 current waiting attempt 属于该 admission，就从 request-id keyed current plan 构建 invalid-block control failure；该 admission-wide 动作与 exact proof 是否命中 current attempt 无关。completion terminalization 则由两类互相独立的证据授权：exact proof 只命中它命名的 attempt；frozen admission ceiling 命中 provably-never-started 的 newer attempts。获得任一 authority 后，命中 undispatched completion 时 scheduler 才允许立即关闭；命中 dispatched completion 时 `force_fail_completion` 只 latch failed，scheduler 将 exact `ReverseReceiveFailureTerminal` 放入下一轮 metadata，由真实 worker 回报后关闭 barrier。`build_connector_meta()` drain pending payload 后，`_prefill_staged_or_delivered_reverse_terminals` 继续保留到 exact completion 关闭，防止重复 ABORT 再投递。读取 current request-id keyed plan 只依赖入口快照证明它属于同一 exact admission；已释放 admission 或同 request-id 的 replacement 不得读取/污染 current plan。完整 ceiling 语义、wire shape 与 late-context 要求见 `.specs/2026-08-24-dualpath-admission-activation-ceiling-design.md`。

新增 `_maybe_retire_prefill_abort_key(request_id, request_key)`：仅当 `_prefill_request_keys.get(request_id) != request_key`（active admission 已释放）且 `TransferCompletionTracker.has_open_completion(CompletionKind.REVERSE_RECEIVE, request_key)` 为 False 时，才 pop `_prefill_abort_request_ids` 并调用 coordinator `unregister_prefill_abort_key`。调用点：

- `_discard_undelivered_prefill_decision`（这里可直接 retire）；
- `_release_scheduler_request_state` 完成 exact released-key 清理后——含 `discard_closed_completions(REVERSE_RECEIVE, released_key)` 调用之后（这条路径也会移除已关闭记录）；
- `_close_reverse_receive_completion` 的**两个**返回分支：进入 waiting-attempt 查找前，先从 `completion.reverse_attempt_key` 退休 pending terminal 与 staged-or-delivered dedupe，再取 `completion_request_key`/`abort_request_id`。stale-attempt 分支（waiting `request_id is None` → discard → `return set()`）与当前 waiting-attempt 分支都在 discard/关闭后调用 `_maybe_retire_prefill_abort_key`；任一 exact-key 值缺失时只跳过退休，不得 fallback 到相同 request ID。

`_release_scheduler_request_state` 必须从 finishing metadata 解析出的 exact key 退休；metadata malformed 时继续 fail closed 地保留，不得用同 request-id 的 active key 兜底。shutdown 在 coordinator.close 前无需逐 key 注销，但要清 `_prefill_abort_request_ids`、`_scheduler_side_finished_recving`、pending failure terminals 与 staged-or-delivered dedupe。

`update_connector_output`（:1986+）在聚合 worker metadata 之后把 scheduler 侧注入桥接进 connector_output：

```python
        if self._scheduler_side_finished_recving:
            if connector_output.finished_recving is None:
                connector_output.finished_recving = set()
            connector_output.finished_recving.update(self._scheduler_side_finished_recving)
            self._scheduler_side_finished_recving.clear()
```

（`_close_reverse_receive_completion` 现成：返回 `{request_id}`、清 waiting/pending 并 discard record；修改时保留“只允许当前 exact attempt 产生 finished_recving”的既有约束。）

- [ ] **Step 4: 跑测试确认通过 + dual_path 全套不回归**

Run: `.venv/bin/python -m pytest tests/ut/distributed/kv_transfer/dual_path/ 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "fix(dual_path): fail-close prefill reverse-receive completions on peer abort"
```

---

### Task 9: 全量回归 + 静态检查

- [ ] **Step 1: 全量 UT**

Run: `.venv/bin/python -m pytest tests/ut/ 2>&1 | tail -3`
Expected: 全 passed；记录实际 collected/passed/skipped 数，不沿用历史计数作 oracle。

- [ ] **Step 2: lint + format**

Run: `ruff check vllm_ascend/ && ruff format --check vllm_ascend/`
Expected: 无新告警（只动过的文件范围内）

- [ ] **Step 3: 仓库 AGENTS.md 合规自查**：Fix D 未新增 env/config；0.9 使用带上游保留空间语义的不可变命名常量；无新全局可变状态；测试覆盖新增失败模式。

---

### Task 10: 集群集成验证（回归证明 + 性能轮解锁）

**这是 wedge 的端到端回归证明，必须在真实 NPU 集群跑。**

- [ ] **Step 0: 固定外部测试资产**——当前 checkout 不含 `deploy/`、`dualpath-npu-test/`。执行前记录二者的绝对路径、git remote、branch 与 commit SHA；确认 manifest 中 Decode 只配置 `dual_path_control_port`，Prefill 只配置 `prefill_control_port: 30910`。缺少任一外部 revision 时，本 Task 不能宣称已复现或 E2E 通过。

- [ ] **Step 1: E2E-A（4GB 自动分批，验证 Fix D）**——保持现有 `local_buffer_size="4GB"`，不增加任何 Fix D env/extra-config，redeploy 后跑 32k 多轮场景。有限 staging backend 必须自动使用代码内固定 90% usable budget。

  验收：
  - 同一大请求出现多个 chunked get 批次，日志同时打印 raw/usable budget；
  - 不出现 staging-budget 导致的 `Failed to load blocks`；
  - 场景完成，Engine stats 持续刷新，成功请求数与无故障 baseline 一致；
  - 这项只证明触发器被分批消除，不拿它证明 cleanup 链。

- [ ] **Step 2: E2E-B（测试资产 fail-once，确定性验证 Fix A/B cleanup）**——在外部 E2E 仓库新增、单独审计一个 `sitecustomize.py` 测试资产，只对 Decode 测试 pod 生效：保存原始 `KVCacheStoreRecvingThread._chunked_store_get`，用进程内 lock 保证第一次非空调用直接返回 `[1] * len(key_list)`，随后所有调用转发原方法。通过测试 pod 的临时 `PYTHONPATH` 注入，使用并发 1、关闭 warmup 的短轮场景，确保唯一一次故障落在目标请求。该文件及其启用方式不得进入本仓库产品提交或正式 manifest。

  外部测试资产 `dualpath-npu-test/fault_injection/sitecustomize.py` 的内容固定为：

  ```python
  """Test-only one-shot Store failure. Never package with vllm-ascend."""

  import threading

  from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
      KVCacheStoreRecvingThread,
  )

  _original_chunked_store_get = KVCacheStoreRecvingThread._chunked_store_get
  _failure_lock = threading.Lock()
  _failures_remaining = 1


  def _fail_first_chunked_store_get(
      self,
      key_list,
      addr_list,
      size_list,
      *,
      usable_budget_bytes,
      raw_budget_bytes,
  ):
      global _failures_remaining
      with _failure_lock:
          if key_list and _failures_remaining:
              _failures_remaining -= 1
              return [1] * len(key_list)
      return _original_chunked_store_get(
          self,
          key_list,
          addr_list,
          size_list,
          usable_budget_bytes=usable_budget_bytes,
          raw_budget_bytes=raw_budget_bytes,
      )


  KVCacheStoreRecvingThread._chunked_store_get = _fail_first_chunked_store_get
  ```

  验收：
  - Decode 确定性出现一次 Store per-key failure、Fix A 合成的 all-worker failure report，以及 Decode admission fail-close；
  - Decode→Prefill ABORT Future 必须在最多 3 次投递内成功；若 terminal 为 `PathDecisionDeliveryError`，记录为本场景失败（通知可能丢失，不能宣称 cleanup E2E 通过），且不得描述为无限重试；
  - Prefill 日志出现 exact-key `peer_abort`，unstarted completion 立即 force-fail；如已有 worker report，则等 all-worker barrier 后关闭；
  - 失败请求之后的请求继续完成，Engine stats 不冻结，Decode 与 Prefill KV cache usage 都回落；
  - registry/record 清理日志证明 exact key 已退休，无相同 request-id fallback。

- [ ] **Step 3: 移除测试注入并恢复产品部署**——删除测试 pod 的临时 `PYTHONPATH`/overlay，按未修改的产品 commit SHA 重新 deploy；保存外部测试资产 SHA、临时 manifest diff、注入命中一次的证据，以及恢复后 pod 环境，证明正式部署不存在 fault injector。

- [ ] **Step 4: E2E-C（8GB 性能轮）**——恢复正式 8GB buffer，跑完整 `mooncake-multiturn` 32k 性能轮，与固定 revision 的 baseline 比较吞吐、TTFT/TPOT、失败数和 cache usage。cleanup E2E-B 与性能 E2E-C 分开出结论，不能用一个结果替代另一个。

- [ ] **Step 5: 收尾**——在外部 perf/test 仓库的既定记录位置保存三轮命令、配置、revision 与日志；若验证通过，再整理 issue/PR 材料。修改外部 `SKILL.md` 或上报上游属于独立授权动作，不在执行本 spec 时自动进行。

---

## Self-Review 记录

- **Spec 覆盖**：Fix A → Task 3（已独立核对 `_SplitTracker`、`_record_completion_report` 与 all-worker barrier）；Fix B → Task 4-8（role-aware receiver 在 Task 6，统一 Decode admission fail-close 与 refresh 残留窗口在 Task 7，延迟生命周期 Prefill registry 在 Task 8）；Fix D → Task 1-2（align/padding + raw/usable 双预算 + 有意识的 per-key 偏离）。
- **类型一致性**：`_plan_get_batches(..., usable_budget_bytes, raw_budget_bytes)`、`force_fail_completion(..., expected_kind, expected_attempt_key)`、`has_open_completion(kind, request_key)`、`prefill_control_endpoint`、`_prefill_abort_request_ids`、`_scheduler_side_finished_recving` 在生产代码与测试 oracle 间一致。
- **Finding 1 措辞修正**：控制投递由 `_MAX_DELIVERY_ATTEMPTS = 3` 限制；耗尽后 Future 抛 `PathDecisionDeliveryError`。风险是通知最终丢失，不是无限循环。
- **生命周期约束**：endpoint 不在 pending state 创建时写入；activation 成功后才持久化。activation 异常可使用当前 wire candidate 通知但不污染 state；refresh endpoint mismatch 只信任已持久化 endpoint。统一 helper 的 `finally` 覆盖 control-failure metadata 构建异常，但不越权关闭旧 in-flight attempt。
- **预检补充（执行前确认）**：(1) Decode 请求先 finish 时 `_release_scheduler_request_state` 提前删 state/snapshot，同轮稍后的 worker failure report 找不到 state、无法发 ABORT——Task 7 增加 `_decode_late_abort_endpoints` 延迟上下文（写入/消费/退休三点）；(2) `_close_reverse_receive_completion` 的 stale-attempt 提前返回分支会漏掉 abort registry 退休——Task 8 明确两个返回分支都调 `_maybe_retire_prefill_abort_key`，并扩展 `pe_scheduler_factory` 支持 `prefill_control_port`；(3) 现有 `FakeStore` 非 `Backend` 子类需补 `staging_buffer_bytes()`，并补 `_handle_request` 请求级集成测试；Fix D 明确限定 DualPath Decode non-layerwise 异步 load，不扩展 layerwise 与 `pool_worker.py` 同步路径。
- **配置面保持最小**：Fix D 不新增 ratio 或 chunking 开关；有限 staging backend 始终自动分批，90% usable fraction 是内部不可变策略。
- **E2E 不再矛盾**：4GB 自动分批验证 Fix D 成功链；外部测试资产 fail-once 验证 Fix A/B cleanup；8GB 单独做性能轮。故障注入不进入产品代码或正式配置。
- **执行适配要求**：Task 3 复用已存在的 `_make_worker`/`_make_split_metadata`/`_make_reverse_plan`；Task 4 测试按现有 `TransferCompletionTracker()` 风格直接构造，不引用不存在的 factory；Task 7/8 先读现有 scheduler factory/fake coordinator 再把上述 oracle 映射到真实 helper 名称。
- **外部证据边界**：当前 checkout 没有 `deploy/` 或 `dualpath-npu-test/`；Task 10 只有在外部路径/revision 固定并真实运行后才可标记完成。
- **不做的事（YAGNI）**：不改上游 vLLM 的 delay_free 时序；不做 completion 超时看门狗；不改 memcache/yuanrong 后端行为（`staging_buffer_bytes` 默认 None 保持现状）；不增加 Fix D 用户配置或生产 fault-injection seam。

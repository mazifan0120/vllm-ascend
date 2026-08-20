# DualPath KV 加载失败清理（wedge 根治）+ Store 读取分批 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 DualPath connector 在 store 加载失败后的 KV 块永久泄漏（EngineCore livelock/wedge），并为非 layerwise 的 mooncake get 路径加分批以根除触发器。

**Architecture:** 三个相互独立但同源的修复——Fix A（decode worker 为"从未提交的 reverse"合成失败上报，使 completion 关闭、块释放）；Fix B（控制通道增加 decode→prefill 方向的失败通知，prefill 收到后强制关闭 REVERSE_RECEIVE completion 并释放镜像块）；Fix D（`KVCacheStoreRecvingThread` 的整批 `m_store.get` 按 staging buffer 容量分批，参照上游 vLLM 0.23 `_split_disk_offload_load_batches`）。所有关闭/释放动作复用现有 completion 生命周期通道，不新增并行机制。

**Tech Stack:** Python 3.12, vllm-ascend (vllm_ascend), pytest (tests/ut, CPU 可跑，conftest 有 mooncake/torch_npu stub), ZMQ (控制通道), msgspec。

---

## 背景（事故链，已定位）

32k 多轮压测中，decode 侧 DE_READ 整请求 store 读取 30720 token × 144KiB = 4.22GiB > mooncake `local_buffer_size` 4GB（只够 227/240 块），13 个 key 确定性失败 → 请求被 `failure_policy=fail` 杀掉 → **失败收尾永久 pinned 该请求的 257 个 KV 块**（两条 delay_free 路径叠加）→ 后续满长度请求永远分配不到块 → 调度器活锁（stats 被 idle 逻辑降为 DEBUG，看似"静默 25 分钟"），prefill 侧经 REVERSE_RECEIVE completion 镜像泄漏同样饿死。

关键事实：

- 泄漏点 1（decode）：reverse 发送只在 store DONE 后提交（`worker.py:611-614`）；store FAILED → reverse 从未提交 → activation 时打开的 REVERSE_SEND completion（`scheduler.py:1688-1694`）永不关闭 → `_delay_free_for_connector`（`scheduler.py:1950-1958`）永久 True。
- 泄漏点 2（prefill）：镜像 REVERSE_RECEIVE completion 同样永不关闭，且**当前控制通道是单向的**（prefill→decode 发 decision/abort；decode 的 coordinator 无 executor，无法回发），prefill 永远无法得知 decode 侧失败。
- 触发器：`KVCacheStoreRecvingThread._handle_request`（`kv_transfer.py:846-943`）对整个请求发一次不分批的 `m_store.get`。staging buffer 硬约束来自 mooncake 对象语义 API（`batch_get_into_multi_buffers`），单 key（=单 block 全层，Qwen3-8B 约 18MiB）不可再分但远小于 buffer。
- 现成的关闭通道：worker `_record_completion_report(completion_id, succeeded=...)`（`worker.py:341-359`）→ scheduler `_aggregate_worker_completion_reports`（`scheduler.py:2026-2075`）→ `_run_completion_close_action`（`:2078+`）→ finished_sending/recving 注入 → 上游释放延迟块。completion 失败时 decode 分支已会构建 `_decode_control_failures`（`:2047`），prefill 分支已有 `_prefill_control_failures` + `_recovery_invalid_block_ids` 清理链（`:2060-2073`）——**Fix B 主要是把"失败"这个消息跨节点送达，后续清理链全是现成的**。
- `TransferCompletionTracker.tally_reports`（`completion_tracker.py:149-173`）按 worker barrier 计数关闭；`fail_completion`（`:175`）只记一个失败终端，TP>1 时不够，需要 Fix B-1 的强制关闭。

## 文件地图

| 文件 | 职责 | 改动 |
|---|---|---|
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py` | Backend ABC | Fix D: 新增 `staging_buffer_bytes()` 默认 None |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py` | mooncake 后端 | Fix D: override 返回 `config.local_buffer_size` |
| `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py` | 传输线程 | Fix D: 纯函数 `_plan_get_batches` + recv 线程分批 get |
| `vllm_ascend/envs.py` | 环境变量注册 | Fix D: 新增比例 env（AGENTS.md 强制） |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py` | decode/prefill worker | Fix A: `_fail_unsubmitted_reverse` |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py` | completion 生命周期 | Fix B-1: `force_fail_completion` |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py` | 控制通道/协议 | Fix B-2: `PathDecision.prefill_control_endpoint`；Fix B-3: coordinator prefill 接收端 + decode executor |
| `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py` | 调度侧状态机 | Fix B-4/5: decode 存端点+发 abort；prefill 收 abort+关 completion |
| `tests/ut/distributed/kv_transfer/dual_path/test_*.py` | dual_path UT（conftest 有现成工厂） | 各 Fix 的回归测试 |
| `tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py` | 新建 | Fix D UT |

部署注意：pod 启动时从 `/mnt/vllm-ascend-src` untar 本仓库，**代码改动 redeploy 即生效，不需要重建镜像**。提交规范：Conventional Commits + `git commit -s`（仓库 AGENTS.md）；执行时每次 commit 前向用户确认。

---

### Task 1: Fix D-1 — Backend staging buffer 能力查询 + env 比例

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py`（`get` 抽象方法后）
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py`（`get` 实现后，~:273）
- Modify: `vllm_ascend/envs.py`
- Test: `tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py`（新建，Task 2 复用）

- [ ] **Step 1: 写失败测试（新文件，stub 模式照抄 dual_path conftest 的 mooncake/torch_npu stub 前言）**

```python
# SPDX-License-Identifier: Apache-2.0
"""Chunked-get batching for the non-layerwise store receive thread."""

import os
import sys
import types
from unittest.mock import MagicMock

import importlib.util

_fake_engine = types.ModuleType("mooncake.engine")
_fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("mooncake.engine", _fake_engine)

_fake_torch_npu = types.ModuleType("torch_npu")
_fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
_fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", _fake_torch_npu)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.backend import Backend  # noqa: E402


def test_backend_staging_buffer_bytes_default_none():
    assert Backend.staging_buffer_bytes(MagicMock()) is None


def test_mooncake_backend_reports_local_buffer_size():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend

    backend = object.__new__(MooncakeBackend)
    backend.config = MagicMock(local_buffer_size=8 * 1024**3)
    assert backend.staging_buffer_bytes() == 8 * 1024**3
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /home/llq/dualpath/vllm-ascend && .venv-ut/bin/pytest tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py -v 2>&1 | tail -5`（.venv-ut 在 workspace 根，用 `/home/llq/dualpath/.venv-ut/bin/pytest`）
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
        return int(self.config.local_buffer_size)
```

`envs.py` 的 `env_variables` dict 中按现有格式加（带注释说明用途/默认值/取值范围）：

```python
    # Fraction of the mooncake client staging buffer (local_buffer_size) that a
    # single chunked store-get sub-batch may fill. Valid range: (0, 1].
    "VLLM_ASCEND_MOONCAKE_GET_STAGING_USABLE_RATIO": lambda: float(os.getenv("VLLM_ASCEND_MOONCAKE_GET_STAGING_USABLE_RATIO", "0.9")),
```

- [ ] **Step 4: 跑测试确认通过**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/backend.py vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py vllm_ascend/envs.py tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py
git commit -s -m "feat(kv_pool): expose backend staging buffer size for chunked store gets"
```

---

### Task 2: Fix D-2 — recv 线程分批 get

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`（模块级纯函数 + `KVCacheStoreRecvingThread._handle_request` :900-935 段）
- Test: `tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py`

设计：分批规划做成模块级纯函数 `_plan_get_batches(sizes_per_key, budget) -> tuple[list[list[int]], set[int]]`（返回 key 下标的批次列表 + 超预算单 key 下标集），线程里按批次循环 `m_store.get`，子批返回 None 视为该批全失败（per-key 记账 `record_failed_blocks` 已现成）。`staging_buffer_bytes()` 为 None（memcache/yuanrong）时保持现状整批一次 get。

- [ ] **Step 1: 写失败测试（追加到 Task 1 的测试文件）**

```python
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (  # noqa: E402
    KVCacheStoreRecvingThread,
    _plan_get_batches,
)


def test_plan_get_batches_packs_keys_under_budget():
    sizes = [[100], [200], [300], [400], [50]]
    batches, oversized = _plan_get_batches(sizes, budget=600)
    assert batches == [[0, 1, 2], [3, 4]]
    assert oversized == set()


def test_plan_get_batches_marks_single_key_overflow():
    sizes = [[100], [700], [200]]
    batches, oversized = _plan_get_batches(sizes, budget=600)
    assert batches == [[0, 2]]
    assert oversized == {1}


def test_plan_get_batches_none_budget_single_batch():
    batches, oversized = _plan_get_batches([[10], [20]], None)
    assert batches == [[0, 1]]
    assert oversized == set()


class _RecordingBackend(MagicMock):
    pass


def _make_thread(backend):
    thread = object.__new__(KVCacheStoreRecvingThread)
    thread.m_store = backend
    return thread


def test_chunked_store_get_merges_per_key_results():
    backend = MagicMock()
    backend.get.side_effect = [[0, 0], None, [0]]
    thread = _make_thread(backend)
    keys = ["k0", "k1", "k2", "k3", "k4"]
    sizes = [[100], [200], [300], [400], [50]]
    ret = thread._chunked_store_get(keys, [[0]] * 5, sizes, budget=600)
    # 批次 [0,1,2] 全成功、[3] 所在批 None→失败、[4] 成功；无超预算单 key
    assert ret == [0, 0, 0, 1, 0]
    assert backend.get.call_count == 3


def test_chunked_store_get_oversized_key_failed_without_issuing():
    backend = MagicMock()
    backend.get.return_value = [0, 0]
    thread = _make_thread(backend)
    ret = thread._chunked_store_get(["k0", "big", "k2"], [[0]] * 3, [[100], [700], [200]], budget=600)
    assert ret == [0, 1, 0]
    issued_keys = backend.get.call_args[0][0]
    assert "big" not in issued_keys
```

- [ ] **Step 2: 跑测试确认失败**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py -v 2>&1 | tail -5`
Expected: FAIL `ImportError: cannot import name '_plan_get_batches'`

- [ ] **Step 3: 实现**

`kv_transfer.py` 模块级（放在 `record_failed_blocks` 之前）：

```python
def _plan_get_batches(
    sizes_per_key: list[list[int]],
    budget_bytes: int | None,
) -> tuple[list[list[int]], set[int]]:
    """Plan key-subset get batches that each fit the staging budget.

    Returns (batches of key indices, indices of single keys that alone exceed
    the budget). A None/non-positive budget disables batching (single batch).
    A key larger than the whole budget can never be staged (the mooncake get
    API has no intra-key offset read), so it is reported as failed without
    issuing; per-key failure accounting already exists downstream.
    """
    if budget_bytes is None or budget_bytes <= 0:
        return [list(range(len(sizes_per_key)))] if sizes_per_key else [], set()
    batches: list[list[int]] = []
    oversized: set[int] = set()
    current: list[int] = []
    current_bytes = 0
    for index, key_sizes in enumerate(sizes_per_key):
        key_bytes = sum(key_sizes)
        if key_bytes > budget_bytes:
            oversized.add(index)
            continue
        if current and current_bytes + key_bytes > budget_bytes:
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
    def _staging_get_budget_bytes(self) -> int | None:
        staging = self.m_store.staging_buffer_bytes()
        if staging is None or staging <= 0:
            return None
        from vllm_ascend import envs
        return int(staging * envs.VLLM_ASCEND_MOONCAKE_GET_STAGING_USABLE_RATIO)

    def _chunked_store_get(
        self,
        key_list: list[str],
        addr_list: list,
        size_list: list[list[int]],
        budget_bytes: int,
    ) -> list[int]:
        """Run one whole-request get as staging-budget-sized key-subset batches.

        Return codes follow the backend contract: 0 = success, non-zero = failed
        (a sub-batch returning None marks every key in it failed, matching the
        whole-batch None handling)."""
        batches, oversized = _plan_get_batches(size_list, budget_bytes)
        results: dict[int, int] = {index: 1 for index in oversized}
        if oversized:
            logger.warning(
                "KV pool async recv skipping %d single keys larger than staging budget %d",
                len(oversized),
                budget_bytes,
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
                for index, code in zip(batch, sub_ret):
                    results[index] = 0 if code is None else code
        if len(batches) > 1 or oversized:
            logger.info(
                "KV pool async recv chunked get: keys=%d batches=%d oversized=%d budget=%d",
                len(key_list),
                len(batches),
                len(oversized),
                budget_bytes,
            )
        return [results[i] for i in range(len(key_list))]
```

`_handle_request` 中把

```python
        ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
```

改为

```python
        budget = self._staging_get_budget_bytes()
        if budget is None:
            ret = self.m_store.get(key_list_c, addr_list_c, size_list_c)
        else:
            ret = self._chunked_store_get(key_list_c, addr_list_c, size_list_c, budget)
```

（后续 `if ret is not None and any(r != 0 for r in ret)` 等既有失败记账逻辑不变。）

- [ ] **Step 4: 跑测试确认通过 + 不回归**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/ascend_store/ -v`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py tests/ut/distributed/kv_transfer/ascend_store/test_kv_transfer_chunked_get.py
git commit -s -m "fix(kv_pool): chunk non-layerwise store gets to fit the mooncake staging buffer"
```

---

### Task 3: Fix A — worker 为未提交的 reverse 合成失败上报

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py`（`_consume_store_completions` :584-618 + 新方法）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_reverse_attempt_identity.py`（追加；该文件已覆盖 reverse attempt 语义，或新建 `test_store_failure_reverse_cleanup.py`，二选一，按现有文件风格定）

- [ ] **Step 0: 读 `tests/ut/distributed/kv_transfer/dual_path/conftest.py` 的 `init_dual_path_worker_state` 与 `worker.py` 的 `_SplitTracker` 定义（`grep -n "class _SplitTracker" -A 25 worker.py`），确认 tracker 字段名与 `_SplitPhase` 取值**

- [ ] **Step 1: 写失败测试**

```python
def test_store_failure_before_reverse_submission_synthesizes_failure_report():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker import (
        DualPathConnectorWorker,
        _SplitPhase,
        _SplitTracker,
    )

    worker = init_dual_path_worker_state(object.__new__(DualPathConnectorWorker), role="decode")
    tracker = object.__new__(_SplitTracker)
    tracker.store_phase = _SplitPhase.PENDING
    tracker.reverse_phase = _SplitPhase.PENDING
    tracker.forward_phase = _SplitPhase.PENDING
    tracker.store_destination_slice = {11, 12}
    tracker.store_load_failed = True
    tracker.reverse_submitted_attempt = None
    tracker.reverse_plan = SimpleNamespace(
        reverse_send_completion_id=77,
        request_key=SimpleNamespace(decode_request_id="req-1"),
        reverse_attempt_id=3,
        wire_request_id="wire-1",
        token_start=0,
        token_end=128,
    )
    tracker.terminal_reported = False
    tracker.core_request_finished = False
    worker._split_trackers["req-1"] = tracker

    worker._consume_store_completions({"req-1"}, set())

    assert tracker.reverse_phase is _SplitPhase.FAILED
    assert worker._pending_failure_reports.get(77) == 1
    assert 77 not in worker._pending_completion_reports


def test_store_failure_after_reverse_submission_keeps_wire_terminal_path():
    # reverse 已提交：失败上报属于数据面 send_done_send_signal，不得合成
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.path_decision import ReverseAttemptKey
    from vllm_ascend.distributed.kv_transfer.kv_p2p.dual_path.worker import (
        DualPathConnectorWorker,
        _SplitPhase,
        _SplitTracker,
    )

    worker = init_dual_path_worker_state(object.__new__(DualPathConnectorWorker), role="decode")
    attempt_key = ReverseAttemptKey(SimpleNamespace(decode_request_id="req-2"), 3)
    tracker = object.__new__(_SplitTracker)
    tracker.store_phase = _SplitPhase.PENDING
    tracker.reverse_phase = _SplitPhase.PENDING
    tracker.forward_phase = _SplitPhase.PENDING
    tracker.store_destination_slice = {21}
    tracker.store_load_failed = True
    tracker.reverse_submitted_attempt = attempt_key
    tracker.reverse_plan = SimpleNamespace(
        reverse_send_completion_id=88,
        request_key=attempt_key.request_key,
        reverse_attempt_id=3,
        wire_request_id="wire-2",
        token_start=0,
        token_end=128,
    )
    tracker.terminal_reported = False
    tracker.core_request_finished = False
    worker._split_trackers["req-2"] = tracker

    worker._consume_store_completions({"req-2"}, set())

    assert tracker.reverse_phase is _SplitPhase.PENDING
    assert 88 not in worker._pending_failure_reports
    assert 88 not in worker._pending_completion_reports
```

- [ ] **Step 2: 跑测试确认失败**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/ -k store_failure -v 2>&1 | tail -5`
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

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/ 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/worker.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "fix(dual_path): fail-close reverse-send completion when store load fails before submission"
```

---

### Task 4: Fix B-1 — completion tracker 强制失败关闭

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/completion_tracker.py`（`fail_completion` :175 后）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py`

- [ ] **Step 1: 写失败测试**

```python
def test_force_fail_completion_closes_open_record_without_worker_reports(tracker_factory):
    tracker, completion = tracker_factory(expected_worker_count=3)  # 按现有测试工厂形式适配
    assert tracker.force_fail_completion(completion.completion_id) is True
    record = tracker.get(completion.completion_id)
    assert record.closed and record.failed


def test_force_fail_completion_is_noop_for_closed_or_unknown(tracker_factory):
    tracker, completion = tracker_factory(expected_worker_count=1)
    tracker.fail_completion(completion.completion_id)  # barrier 满，已关闭
    assert tracker.force_fail_completion(completion.completion_id) is False
    assert tracker.force_fail_completion(999999) is False
```

- [ ] **Step 2: 跑测试确认失败**（`AttributeError: force_fail_completion`）

- [ ] **Step 3: 实现**

```python
    def force_fail_completion(self, completion_id: int) -> bool:
        """Force-close an open completion as failed without worker reports.

        Used when the peer reports the failure out-of-band (control channel):
        the data-plane transfer was never started, so no worker terminal will
        ever arrive to fill the barrier."""
        record = self._records.get(completion_id)
        if record is None or record.closed:
            return False
        record.failed = True
        record.closed = True
        return True
```

- [ ] **Step 4: 跑测试确认通过**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/test_completion_tracker.py -v 2>&1 | tail -3`
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
        result=_make_result(),          # 复用该文件现有的 result 构造器
        reverse_plan=None,
        prefill_control_endpoint=DecodeControlEndpoint(host="192.0.2.88", port=24999),
    )
    decoded = PathDecision.from_dict(decision.to_dict())
    assert decoded.prefill_control_endpoint == DecodeControlEndpoint(host="192.0.2.88", port=24999)


def test_path_decision_round_trip_without_prefill_endpoint_defaults_none():
    decision = PathDecision(result=_make_result(), reverse_plan=None)
    decoded = PathDecision.from_dict(decision.to_dict())
    assert decoded.prefill_control_endpoint is None


def test_path_decision_wire_rejects_unknown_keys():
    # require_exact_payload 更新后仍拒绝未登记 key
    payload = PathDecision(result=_make_result(), reverse_plan=None).to_dict()
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

注意：P/D 两侧同构建部署，严格键校验同步更新无兼容问题；但 `decode_control_endpoint` 所在 `DualPathDecisionMetadata` 不动。

- [ ] **Step 4: 跑测试确认通过**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel_compliance.py -v 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "feat(dual_path): carry optional prefill control endpoint on path decisions"
```

---

### Task 6: Fix B-3 — coordinator：prefill 接收端 + decode 发送 executor

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py`（`for_decode` :344-374、`for_prefill` :376-398、`close` :501+；接收循环 `_receive_decisions` 已同时处理 DECISION/ABORT 帧，无需改）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py`

设计：`for_prefill` 增加可选 `control_endpoint`——给了就和 decode 一样起 `_receive_decisions` 接收线程（收到 ABORT 入 `_received_aborts`；DECISION 帧因无 pending key 被现有 `_handle_decision_frame` 拒绝为 unexpected，行为安全）。`for_decode` 增加 `ThreadPoolExecutor` 使 decode 可调 `submit_abort`。两侧 `close()` 都负责关停自己创建的资源。

- [ ] **Step 0: 读 `_receive_decisions`（:521+）与 `_handle_abort_frame`（:600+），确认接收循环对 role 无假设（只用 `_decode_control_endpoint` 绑定地址）——若有，把绑定地址来源泛化为 `self._bind_endpoint`，decode/prefill 各自赋值**

- [ ] **Step 1: 写失败测试（真实 127.0.0.1 ZMQ 回路，参照该文件既有 socket 测试）**

```python
def test_prefill_coordinator_receives_abort_on_control_endpoint():
    prefill_ep = DecodeControlEndpoint(host="127.0.0.1", port=_free_port())
    prefill = PathDecisionCoordinator.for_prefill(control_endpoint=prefill_ep)
    try:
        decode_side = PathDecisionCoordinator.for_prefill()  # 纯发送端
        notice = PathAbortNotice(request_key=_make_key("req-9"), reason=PathAbortReason.ACTIVATION_FAILED)
        decode_side.submit_abort(prefill_ep, notice).result(timeout=5)
        deadline = time.time() + 5
        received = []
        while time.time() < deadline and not received:
            received = prefill.take_received_aborts()
            time.sleep(0.05)
        assert [n.request_key for n in received] == [notice.request_key]
    finally:
        prefill.close()


def test_decode_coordinator_can_submit_abort():
    decode = PathDecisionCoordinator.for_decode(
        engine_id="eng", data_parallel_rank=0,
        control_endpoint=DecodeControlEndpoint(host="127.0.0.1", port=_free_port()),
    )
    try:
        # 仅验证本地提交路径存在（executor 可用），投递失败允许重试后取消
        fut = decode.submit_abort(DecodeControlEndpoint(host="127.0.0.1", port=_free_port()),
                                  PathAbortNotice(request_key=_make_key("req-x"), reason=PathAbortReason.ACTIVATION_FAILED))
        assert fut is not None
        fut.cancel()
    finally:
        decode.close()
```

- [ ] **Step 2: 跑测试确认失败**（`TypeError: for_prefill() got an unexpected keyword argument 'control_endpoint'` / decode `submit_abort` 断言 executor None）

- [ ] **Step 3: 实现**

`for_decode` 末尾（return 前）加：

```python
        coordinator._executor = ThreadPoolExecutor(
            max_workers=_PATH_DECISION_SEND_WORKERS,
            thread_name_prefix="path-abort-sender",
        )
```

`for_prefill` 签名与函数体改为：

```python
    @classmethod
    def for_prefill(
        cls,
        *,
        socket_opener: _SocketOpener | None = None,
        sleep: _Sleep | None = None,
        send_timeout_ms: int = _SEND_TIMEOUT_MS,
        poll_timeout_ms: int = _POLL_TIMEOUT_MS,
        retry_spacing_s: float = _RETRY_SPACING_S,
        control_endpoint: DecodeControlEndpoint | None = None,
    ) -> PathDecisionCoordinator:
        coordinator = cls()
        coordinator._role = "prefill"
        coordinator._socket_opener = socket_opener or _zmq_req_opener
        coordinator._sleep = sleep or time.sleep
        coordinator._send_timeout_ms = send_timeout_ms
        coordinator._poll_timeout_ms = poll_timeout_ms
        coordinator._retry_spacing_s = retry_spacing_s
        coordinator._executor = ThreadPoolExecutor(
            max_workers=_PATH_DECISION_SEND_WORKERS,
            thread_name_prefix="path-decision-sender",
        )
        if control_endpoint is not None:
            # Reverse-direction receiver: the Decode side posts terminal
            # failure notices (ABORT frames) here. DECISION frames landing on
            # this socket are rejected by the pending-key check like any other
            # unexpected message.
            coordinator._decode_control_endpoint = control_endpoint
            coordinator._context = zmq.Context()
            ready_event = threading.Event()
            coordinator._receiver_thread = threading.Thread(
                target=coordinator._receive_decisions,
                args=(ready_event,),
                name="path-prefill-abort-receiver-0",
                daemon=True,
            )
            coordinator._receiver_thread.start()
            if not ready_event.wait(timeout=_RECEIVER_READY_TIMEOUT_S):
                coordinator.close()
                raise RuntimeError("prefill abort receiver did not become ready")
            if coordinator._receiver_error is not None:
                error = coordinator._receiver_error
                coordinator.close()
                raise error
        return coordinator
```

`close()` 的 prefill 分支补上接收端关停（有 `_receiver_thread` 时 `context.term()` + join + drain），decode 分支补 executor 关停：

```python
        if self._receiver_thread is not None:
            assert self._context is not None
            self._context.term()
            self._receiver_thread.join()
            self._drain_received_decisions()
            self._drain_received_aborts()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
```

（decode 分支原有的 pending/accepted 清理保持不变。）

- [ ] **Step 4: 跑测试确认通过 + 通道全套不回归**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/test_channel_registry.py -v 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/path_decision_channel.py tests/ut/distributed/kv_transfer/dual_path/test_path_decision_channel.py
git commit -s -m "feat(dual_path): support reverse-direction abort delivery on the control channel"
```

---

### Task 7: Fix B-4 — decode scheduler：存端点 + 失败时回发 ABORT

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`（`DecodePathDecisionState` :111-119、状态创建处 :1514 附近、`_aggregate_worker_completion_reports` decode 失败分支 :2044-2058；prefill 侧 connector 装配处在 Task 8）
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py`（decode coordinator 装配处，若 `for_decode` 调用点在此）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_decode_scheduler.py` 或 `test_completion_and_delivery_recovery.py`（用 `decode_scheduler_factory` + `decode_control_seams` 夹具）

- [ ] **Step 0: `grep -n "for_decode\|for_prefill" vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/*.py` 找到 coordinator 装配点；读 conftest 的 `decode_scheduler_factory`/`decode_control_seams`，确认 submit_abort 的可观测缝（fake coordinator 或 endpoint 捕获）**

- [ ] **Step 1: 写失败测试**

```python
def test_decode_completion_failure_sends_abort_to_prefill_endpoint(decode_scheduler_factory, decode_control_seams):
    scheduler = decode_scheduler_factory()
    state = _activate_de_read_request(scheduler, prefill_endpoint=DecodeControlEndpoint(host="192.0.2.88", port=24999))
    # 模拟 worker 对该请求 reverse_send completion 的 failure report（Fix A 的产物）
    metadata = DualPathWorkerMetadata(completion_reports={}, failure_reports={state.open_send_completion_id: 1})
    scheduler._aggregate_worker_completion_reports(metadata)
    notices = decode_control_seams.submitted_aborts  # 夹具捕获的 (endpoint, notice)
    assert (DecodeControlEndpoint(host="192.0.2.88", port=24999), state.request_key) in [
        (ep, n.request_key) for ep, n in notices
    ]
    assert notices[0][1].reason is PathAbortReason.ACTIVATION_FAILED


def test_decode_completion_failure_without_endpoint_only_local_cleanup(decode_scheduler_factory, decode_control_seams):
    # 未带端点的旧 decision：只走本地 control failure，不发 abort，不抛错
    scheduler = decode_scheduler_factory()
    state = _activate_de_read_request(scheduler, prefill_endpoint=None)
    metadata = DualPathWorkerMetadata(completion_reports={}, failure_reports={state.open_send_completion_id: 1})
    scheduler._aggregate_worker_completion_reports(metadata)
    assert decode_control_seams.submitted_aborts == []
    assert state.status is _DecodeDecisionStatus.ACTIVATION_FAILED
    assert state.request_key.decode_request_id in scheduler._decode_control_failures
```

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

状态创建处（:1514 附近 `DecodePathDecisionState(...)`）补传：

```python
            prefill_control_endpoint=decision.prefill_control_endpoint,
```

（`_activate_received_decision(decision, metadata)` 的 `decision` 即 Task 5 的 `PathDecision`，端点随 wire 到达。）

`_aggregate_worker_completion_reports` decode 失败分支，在 `self._decode_control_failures[failed_request_id] = ...` 的 try 块之后追加：

```python
                            if state.prefill_control_endpoint is not None:
                                self._send_abort_notice(
                                    state.request_key,
                                    state.prefill_control_endpoint,
                                    PathAbortReason.ACTIVATION_FAILED,
                                )
```

（`_send_abort_notice` :342 现成，走 Task 6 给 decode 配的 executor；失败只记日志不阻塞，安全。）

prefill 侧 `for_prefill(...)` 装配点补 `control_endpoint=`：端点从 `DualPathConfig` 新增可选字段 `prefill_control_port`（默认 None=不开接收，向后兼容）+ 本机 IP 推导（参照 `derive_decode_control_port` :44 的现有模式）。同时 prefill 发 decision 处把端点填进 `PathDecision(prefill_control_endpoint=...)`。manifest 之后配置 `prefill_control_port: 30910` 一类端口即可启用。

- [ ] **Step 4: 跑测试确认通过 + dual_path 全套不回归**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/ 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/connector.py vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/config.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "fix(dual_path): notify prefill over the control channel when an activated decode request fails"
```

---

### Task 8: Fix B-5 — prefill scheduler：收 abort + 关闭 REVERSE_RECEIVE 释放块

**Files:**
- Modify: `vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py`（`build_connector_meta` prefill 分支 :1797-1815、新方法；`update_connector_output` :1986+ 注入桥接）
- Test: `tests/ut/distributed/kv_transfer/dual_path/test_de_read_recovery.py` 或 `test_abort_scheduler.py`（用 `pe_scheduler_factory`）

- [ ] **Step 1: 写失败测试**

```python
def test_prefill_received_abort_fail_closes_reverse_receive_and_releases(pe_scheduler_factory):
    scheduler = pe_scheduler_factory()
    request_id, attempt_key = _install_waiting_reverse_receive(scheduler)  # 夹具/辅助：请求在 _waiting_reverse_attempt_ids + _pending_finished_recving 中，tracker 有 open completion
    notice = PathAbortNotice(request_key=attempt_key.request_key, reason=PathAbortReason.ACTIVATION_FAILED)
    scheduler._handle_received_peer_abort(notice)

    completion = scheduler._completion_tracker.find_open_completion(CompletionKind.REVERSE_RECEIVE, attempt_key)
    assert completion is None  # 已关闭并 discard
    assert request_id not in scheduler._pending_finished_recving
    assert request_id in scheduler._scheduler_side_finished_recving  # 待注入 finished_recving
    assert attempt_key.request_key in scheduler._prefill_invalid_request_keys


def test_prefill_received_abort_for_unknown_key_is_noop(pe_scheduler_factory):
    scheduler = pe_scheduler_factory()
    scheduler._handle_received_peer_abort(PathAbortNotice(request_key=_make_key("ghost"), reason=PathAbortReason.ACTIVATION_FAILED))
    assert scheduler._completion_tracker.open_count() == 0
```

- [ ] **Step 2: 跑测试确认失败**（completion 仍 open / `_pending_finished_recving` 未释放）

- [ ] **Step 3: 实现**

scheduler `__init__` 加：`self._scheduler_side_finished_recving: set[str] = set()`（shutdown 清理清单也加上）。

`build_connector_meta` prefill 分支，在 `self._reconcile_prefill_deliveries()` 之前插入：

```python
            for notice in self._path_decision_coordinator.take_received_aborts():
                self._handle_received_peer_abort(notice)
```

新方法（放在 `_handle_received_abort` 旁，互为镜像）：

```python
    def _handle_received_peer_abort(self, notice: PathAbortNotice) -> None:
        """The Decode side reports a terminal failure for an activated request.

        The reverse transfer never started, so no data-plane terminal will
        close the REVERSE_RECEIVE completion: fail-close it here so the
        request's delayed blocks release and the engine cannot starve."""
        request_id = next(
            (rid for rid, key in self._prefill_request_keys.items() if key == notice.request_key),
            None,
        )
        if request_id is None:
            return
        logger.warning(
            "dual_path peer_abort key=%s reason=%s: fail-closing reverse receive",
            notice.request_key.decode_request_id,
            notice.reason.value,
        )
        self._prefill_invalid_request_keys.add(notice.request_key)
        waiting_attempt = self._waiting_reverse_attempt_ids.get(request_id)
        if waiting_attempt is not None and waiting_attempt.request_key == notice.request_key:
            completion = self._completion_tracker.find_open_completion(
                CompletionKind.REVERSE_RECEIVE, waiting_attempt
            )
            if completion is not None and self._completion_tracker.force_fail_completion(completion.completion_id):
                _, recving = self._run_completion_close_action(completion)
                self._scheduler_side_finished_recving.update(recving)
        invalid_block_ids = self._recovery_invalid_block_ids(request_id)
        if invalid_block_ids:
            self._prefill_control_failures[request_id] = DualPathControlFailureMetadata(
                request_id=request_id,
                invalid_block_ids=invalid_block_ids,
                reason=DualPathControlFailureReason.REVERSE_JOB_FAILED,
            )
```

`update_connector_output`（:1986+）在聚合 worker metadata 之后把 scheduler 侧注入桥接进 connector_output：

```python
        if self._scheduler_side_finished_recving:
            if connector_output.finished_recving is None:
                connector_output.finished_recving = set()
            connector_output.finished_recving.update(self._scheduler_side_finished_recving)
            self._scheduler_side_finished_recving.clear()
```

（`_close_reverse_receive_completion` 现成：返回 {request_id}、清 `_waiting_reverse_attempt_ids`、discard `_pending_finished_recving`——上游收到 finished_recving 即释放延迟块。）

- [ ] **Step 4: 跑测试确认通过 + dual_path 全套不回归**

Run: `/home/llq/dualpath/.venv-ut/bin/pytest tests/ut/distributed/kv_transfer/dual_path/ 2>&1 | tail -3`
Expected: 全 passed

- [ ] **Step 5: Commit（先向用户确认）**

```bash
git add vllm_ascend/distributed/kv_transfer/kv_p2p/dual_path/scheduler.py tests/ut/distributed/kv_transfer/dual_path/
git commit -s -m "fix(dual_path): fail-close prefill reverse-receive completions on peer abort"
```

---

### Task 9: 全量回归 + 静态检查

- [ ] **Step 1: 全量 UT**

Run: `cd /home/llq/dualpath/vllm-ascend && /home/llq/dualpath/.venv-ut/bin/pytest tests/ut/ 2>&1 | tail -3`
Expected: 全 passed（基线 37 个 dual_path 相关 + 新增）

- [ ] **Step 2: lint + format**

Run: `ruff check vllm_ascend/ && ruff format --check vllm_ascend/`
Expected: 无新告警（只动过的文件范围内）

- [ ] **Step 3: 仓库 AGENTS.md 合规自查**：env 变量已注册 envs.py 并带注释；无新全局可变状态；无 magic number（0.9 比例走 env）；测试覆盖新增失败模式。

---

### Task 10: 集群集成验证（回归证明 + 性能轮解锁）

**这是 wedge 的端到端回归证明，必须在真实 NPU 集群跑。**

- [ ] **Step 1: 触发器回归**——把 `deploy/manifests/dualpath-crossnode.yaml` 的 `local_buffer_size` 临时改回 `"4GB"`（触发原事故条件），并在 decode extra_config 加 `"prefill_control_port": 30910`（prefill 侧同样配置），`python3 -m deploy up --profile dualpath-crossnode`

- [ ] **Step 2: 跑 32k 多轮场景**：`python3 dualpath-npu-test/dualpath_perf.py --run-cross --scenarios mooncake-multiturn`（后台，10800s）

- [ ] **Step 3: 验收标准**：
  - decode 日志出现单请求 `Failed to load blocks` + `status=FAILED`，**之后 `Engine 000` stats 持续刷新（不再静默）**，后续请求继续完成；
  - prefill 日志出现 `dual_path peer_abort ... fail-closing reverse receive`；
  - 压测跑完，Failed Requests 仅为触发失败的个别请求（预期 ≤ 并发内被波及的请求数），其余 190+ 请求成功；
  - `GPU KV cache usage` 在失败后回落（不再冻结 59.2%）。

- [ ] **Step 4: 恢复 8GB buffer（性能轮配置），重新 deploy，跑正式 32k 轮 dualpath 侧**（性能数据轮，与 baseline 对比）。

- [ ] **Step 5: 收尾**——SKILL.md（perf-npu-test）记录修复与验证结论；若验证通过，整理 issue/PR 材料上报上游 vllm-ascend（两次 wedge 现场日志在 `dualpath-npu-test/artifacts/perf/transfer_logs_round9/*_hang_20260820.log`）。

---

## Self-Review 记录

- **Spec 覆盖**：Fix A → Task 3；Fix B → Task 4-8（B 的"控制协议扩展"落在 Task 5-6，decode 侧 Task 7，prefill 侧 Task 8）；Fix D → Task 1-2。触发器回归 + 性能轮解锁 → Task 10。
- **类型一致性**：`_plan_get_batches`/`_chunked_store_get`/`staging_buffer_bytes`/`_fail_unsubmitted_reverse`/`force_fail_completion`/`prefill_control_endpoint`/`_handle_received_peer_abort`/`_scheduler_side_finished_recving` 在各 Task 间签名一致。
- **已知适配点（执行时先验证再落码）**：Task 3/7/8 的测试使用了 conftest 现有夹具（`decode_scheduler_factory`/`pe_scheduler_factory`/`decode_control_seams`/`init_dual_path_worker_state`），各 Task 的 Step 0 已安排先读夹具再定稿测试体；`_receive_decisions` 对 role 的假设在 Task 6 Step 0 安排了泛化检查。
- **不做的事（YAGNI）**：不改上游 vLLM 的 delay_free 时序（Fix A 落地后块可经 finished_sending 正常释放，上游问题另案上报）；不做 completion 超时看门狗（用户未选 Fix C，且 A+B 已确定性关闭失败路径）；不改 memcache/yuanrong 后端行为（`staging_buffer_bytes` 默认 None 保持现状）。
